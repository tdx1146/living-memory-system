#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LMS 运维监控（T1.11 / P1-9、P1-10）——三级健康检查 + 指标采集。

仅用标准库（socket / urllib / json / os / argparse），不依赖项目 venv、
不加载 torch，任何 python3 (>=3.8) 可运行；由 cron / systemd timer 驱动。

级别（对应总体方案 §3.5，按本阶段任务裁剪）：
  L1 存活  30s：TCP 连通 :8190 + GET /health 返回 200 且 status=ok
  L2 就绪  60s：L1 + POST /recall 返回 200 + 快照目录可写
  L3 深度  5min：L2 + 各会话 GET /status 正常 + latest_{sid}.pt 新鲜度(<1h)
                 + 进程数（api/mcp/glue/孤儿）+ 内存水位（avail/swap/API RSS）

输出：
  logs/lms_metrics.jsonl  追加式 JSONL 指标（ts/level/ok/latency_ms/
                          turn_count/进程数/内存/快照新鲜度/快照数/明细）
  logs/lms_alerts.jsonl   告警行（WARN/CRIT），同时打印到 stdout/stderr
                          （推送通道留待后续阶段，不在此实现）
退出码：0=全部通过  2=存在 WARN  3=存在 CRIT（便于 cron MAILTO 捕获）

用法示例：
  python3 scripts/lms_ops_monitor.py                       # 三级全跑一次
  python3 scripts/lms_ops_monitor.py --level L1            # 只跑 L1（存活）
  python3 scripts/lms_ops_monitor.py --level L2 --loop --interval 60
"""

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime

# ---------------------------------------------------------------------------
# 常量与默认值
# ---------------------------------------------------------------------------
DEFAULT_FRESHNESS_WARN_S = 3600   # latest_{sid}.pt 新鲜度告警阈值（1h）
DEFAULT_RECALL_QUERY = "__lms_ops_probe__"  # 最小非空探针查询
PROBE_FILENAME = ".lms_ops_probe"  # 快照目录可写探针文件（写完即删）

SEV_CRIT = "CRIT"
SEV_WARN = "WARN"


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")


# ---------------------------------------------------------------------------
# 底层探测工具（stdlib）
# ---------------------------------------------------------------------------
def tcp_probe(host: str, port: int, timeout: float = 5.0) -> bool:
    """TCP 连通性探测。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def http_request(url: str, payload: dict | None = None, timeout: float = 10.0):
    """GET（payload=None）或 POST（payload=dict）JSON 请求。

    返回 (http_status, latency_ms, json_obj)；网络异常返回 (None, ms, None)。
    """
    t0 = time.time()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8") if payload is not None else None,
        headers={"Content-Type": "application/json"} if payload is not None else {},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            latency_ms = round((time.time() - t0) * 1000, 1)
            body = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, latency_ms, json.loads(body)
            except ValueError:
                return resp.status, latency_ms, None
    except urllib.error.HTTPError as e:
        latency_ms = round((time.time() - t0) * 1000, 1)
        return e.code, latency_ms, None
    except OSError:
        return None, round((time.time() - t0) * 1000, 1), None


def read_meminfo() -> dict:
    """解析 /proc/meminfo → {total_kb, avail_kb, swap_total_kb, swap_free_kb}。"""
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, val = line.partition(":")
                out[key.strip()] = int(val.strip().split()[0])  # kB
    except OSError:
        pass
    return out


def scan_processes() -> list:
    """扫描 /proc 全部进程 → [{pid, ppid, cmdline, rss_kb}]。

    无 ps 依赖；解析失败（僵尸/权限）跳过。
    """
    procs = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                raw = f.read()
            cmdline = raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()
            with open(f"/proc/{pid}/stat") as f:
                stat = f.read()
            # stat 格式：pid (comm) state ppid ...（comm 可能含空格，从右括号切）
            rest = stat.split(") ", 1)[1].split()
            ppid = int(rest[1]) if len(rest) > 1 else 0
            rss_kb = 0
            try:
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            rss_kb = int(line.split()[1])
                            break
            except OSError:
                pass
            procs.append({"pid": pid, "ppid": ppid, "cmdline": cmdline, "rss_kb": rss_kb})
        except (OSError, ValueError, IndexError):
            continue
    return procs


def classify_processes(procs: list) -> dict:
    """按 cmdline 关键字分类 LMS 相关进程。

    返回 {api: [...], mcp: [...], http_mcp: [...], glue: [...], orphans: [...]}
    孤儿 = LMS 相关进程且 PPID==1（systemd 接管后 API 不再是孤儿；MCP/glue
    若仍由 setsid 裸跑则会被标记——正是需要回收的对象）。
    """
    api, mcp, http_mcp, glue = [], [], [], []
    for p in procs:
        cmd = p["cmdline"]
        if "api.run" in cmd:
            api.append(p)
        elif "mcp_memory_server.py" in cmd:
            mcp.append(p)
        elif "lms_http_mcp.py" in cmd:
            http_mcp.append(p)
        elif "glue_server.py" in cmd:
            glue.append(p)
    orphans = [p for p in api + mcp + http_mcp + glue if p["ppid"] == 1]
    return {"api": api, "mcp": mcp, "http_mcp": http_mcp,
            "glue": glue, "orphans": orphans}


def snapshot_state(snap_dir: str, sessions: list) -> dict:
    """快照目录状态：可写性 + 各会话 latest_{sid}.pt 新鲜度 + 快照计数。

    返回 {writable, freshness_s: {sid: age|None}, missing: [sid],
          snapshot_count, per_session_counts: {sid: n}}
    """
    state = {"writable": False, "freshness_s": {}, "missing": [],
             "snapshot_count": 0, "per_session_counts": {}}
    # 可写性探针（写完即删，不留痕迹）
    if os.path.isdir(snap_dir):
        probe = os.path.join(snap_dir, PROBE_FILENAME)
        try:
            with open(probe, "w") as f:
                f.write("probe")
            os.unlink(probe)
            state["writable"] = True
        except OSError:
            pass
    for sid in sessions:
        latest = os.path.join(snap_dir, sid, f"latest_{sid}.pt")
        if os.path.isfile(latest):
            state["freshness_s"][sid] = max(0.0, time.time() - os.path.getmtime(latest))
        else:
            state["missing"].append(sid)
            state["freshness_s"][sid] = None
        # 会话目录内历史快照计数（snapshot_{sid}_*.pt）
        sdir = os.path.join(snap_dir, sid)
        if os.path.isdir(sdir):
            n = len([n for n in os.listdir(sdir)
                     if n.startswith(f"snapshot_{sid}_") and n.endswith(".pt")])
            state["per_session_counts"][sid] = n
            state["snapshot_count"] += n
    return state


# ---------------------------------------------------------------------------
# 三级检查
# ---------------------------------------------------------------------------
class Monitor:
    def __init__(self, args):
        self.args = args
        self.base_url = args.base_url.rstrip("/")
        parsed = urllib.parse.urlsplit(self.base_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 8190
        self.alerts = []          # 本次运行产生的告警
        self.metrics_lines = []   # 本次运行产生的指标行

    # -- 告警/指标落盘 ------------------------------------------------
    def alert(self, level: str, severity: str, check: str, message: str) -> None:
        rec = {"ts": now_iso(), "level": level, "severity": severity,
               "check": check, "message": message}
        self.alerts.append(rec)
        with open(self.args.alerts_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # 同时打印：stdout 留痕（cron 邮件可捕获），stderr 供管道消费
        print(f"[lms-monitor][{severity}][{level}] {check}: {message}")
        print(f"[lms-monitor][{severity}][{level}] {check}: {message}", file=sys.stderr)

    def metric(self, level: str, ok: bool, latency_ms: float, fields: dict) -> None:
        rec = {"ts": now_iso(), "level": level, "ok": ok,
               "latency_ms": round(latency_ms, 1), **fields}
        self.metrics_lines.append(rec)
        with open(self.args.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # -- L1 存活 ------------------------------------------------------
    def check_l1(self) -> dict:
        t0 = time.time()
        checks = {}
        checks["tcp"] = tcp_probe(self.host, self.port, timeout=5)
        status, lat, body = http_request(f"{self.base_url}/health", timeout=10)
        checks["health_http"] = status
        checks["health_status_ok"] = bool(body and body.get("status") == "ok")
        ok = checks["tcp"] and checks["health_http"] == 200 and checks["health_status_ok"]
        if not ok:
            detail = (
                f"tcp={checks['tcp']} http={status} body={body!r}"
                if not checks["tcp"] else
                f"http={status} status_field={body and body.get('status')}")
            self.alert("L1", SEV_CRIT, "存活", f"API 不可用：{detail}")
        fields = {"turn_count": None, "process_count": None,
                  "api_count": None, "mcp_count": None, "mcp_orphan_count": None,
                  "glue_count": None, "avail_mb": None, "swap_used_mb": None,
                  "api_rss_mb": None, "snapshot_freshness_s": None,
                  "snapshot_count": None, "checks": checks}
        self.metric("L1", ok, (time.time() - t0) * 1000, fields)
        return {"ok": ok, "latency_ms": (time.time() - t0) * 1000, "checks": checks}

    # -- L2 就绪 ------------------------------------------------------
    def check_l2(self, l1: dict) -> dict:
        t0 = time.time()
        checks = dict(l1.get("checks", {}))
        # /recall 探针：端点设计上空 query 返回 400（fail-closed，正确行为），
        # 因此用最小非空查询验证就绪（走编码+检索，不 process_turn、不调 LLM）。
        status, lat, body = http_request(
            f"{self.base_url}/recall",
            payload={"session_id": "default", "query": DEFAULT_RECALL_QUERY, "k": 1},
            timeout=15)
        checks["recall_http"] = status
        checks["recall_body_ok"] = bool(body and "results" in body)
        # 快照目录可写
        checks["snapdir_writable"] = self._snap_state_cache.get("writable", False)

        problems = []
        if status != 200 or not checks["recall_body_ok"]:
            problems.append(("L2", SEV_CRIT, "就绪-recall",
                             f"/recall http={status} body_keys={body and list(body)}"))
        if not checks["snapdir_writable"]:
            problems.append(("L2", SEV_WARN, "就绪-快照目录",
                             f"{self.args.snapshot_dir} 不可写（检查目录存在与权限）"))
        for lv, sev, check, msg in problems:
            self.alert(lv, sev, check, msg)

        ok = l1["ok"] and status == 200 and checks["recall_body_ok"]
        fields = {"turn_count": None, "process_count": None,
                  "api_count": None, "mcp_count": None, "mcp_orphan_count": None,
                  "glue_count": None, "avail_mb": None, "swap_used_mb": None,
                  "api_rss_mb": None, "snapshot_freshness_s": None,
                  "snapshot_count": None, "checks": checks}
        self.metric("L2", ok, (time.time() - t0) * 1000, fields)
        return {"ok": ok, "latency_ms": (time.time() - t0) * 1000, "checks": checks}

    # -- L3 深度 ------------------------------------------------------
    def check_l3(self, l2: dict) -> dict:
        t0 = time.time()
        checks = dict(l2.get("checks", {}))

        # ① 会话枚举 + 各会话 /status
        sessions = []
        status, lat, body = http_request(f"{self.base_url}/sessions", timeout=10)
        checks["sessions_http"] = status
        if status == 200 and body:
            sessions = body.get("sessions", [])
        checks["sessions"] = sessions

        turn_total = 0
        session_status_ok = True
        bad_sessions = []
        for sid in sessions:
            s_status, s_lat, s_body = http_request(
                f"{self.base_url}/status/{urllib.parse.quote(sid)}", timeout=10)
            ok_sid = s_status == 200 and bool(s_body) and "status" in s_body
            if not ok_sid:
                session_status_ok = False
                bad_sessions.append(sid)
            else:
                turn_total += int(s_body["status"].get("turn_count", 0) or 0)
        checks["session_status_ok"] = session_status_ok
        checks["bad_sessions"] = bad_sessions
        if bad_sessions:
            self.alert("L3", SEV_CRIT, "深度-会话状态",
                       f"以下会话 /status 异常: {bad_sessions}")

        # ② 快照新鲜度（latest_{sid}.pt mtime < 1h）
        snap = self._snap_state_cache
        max_age = 0.0
        stale = []
        for sid, age in snap["freshness_s"].items():
            if age is None:
                stale.append(f"{sid}(无latest文件)")
            elif age > self.args.freshness_warn:
                stale.append(f"{sid}({int(age)}s)")
            else:
                max_age = max(max_age, age)
        checks["snapshot_missing"] = snap["missing"]
        checks["snapshot_stale"] = stale
        checks["snapshot_count"] = snap["snapshot_count"]
        if stale:
            self.alert("L3", SEV_WARN, "深度-快照新鲜度",
                       f"快照过期或缺失: {', '.join(stale)}（阈值 {self.args.freshness_warn}s）")

        # ③ 进程数 + 孤儿
        procs = classify_processes(scan_processes())
        api_count = len(procs["api"])
        mcp_count = len(procs["mcp"]) + len(procs["http_mcp"])
        glue_count = len(procs["glue"])
        orphan_count = len(procs["orphans"])
        process_count = api_count + mcp_count + glue_count
        checks["api_count"] = api_count
        checks["mcp_count"] = mcp_count
        checks["glue_count"] = glue_count
        checks["orphan_count"] = orphan_count
        if orphan_count:
            self.alert("L3", SEV_WARN, "深度-孤儿进程",
                       f"检测到 {orphan_count} 个 PPID=1 的 LMS 孤儿进程"
                       "（systemd 接管后 API 不应为孤儿；MCP 孤儿待回收 cron）")

        # ④ 内存水位
        mem = read_meminfo()
        total_kb = mem.get("MemTotal", 0)
        avail_kb = mem.get("MemAvailable", 0)
        swap_used_kb = mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)
        api_rss_kb = sum(p["rss_kb"] for p in procs["api"])
        avail_mb = round(avail_kb / 1024) if avail_kb else None
        swap_used_mb = round(swap_used_kb / 1024) if swap_used_kb else None
        api_rss_mb = round(api_rss_kb / 1024) if api_rss_kb else None
        checks["avail_mb"] = avail_mb
        checks["swap_used_mb"] = swap_used_mb
        checks["api_rss_mb"] = api_rss_mb
        if total_kb and avail_kb and avail_mb is not None:
            if avail_mb < 200:
                self.alert("L3", SEV_CRIT, "深度-内存",
                           f"可用内存 {avail_mb}MB < 200MB（OOM 高危）")
            elif avail_mb < 500:
                self.alert("L3", SEV_WARN, "深度-内存",
                           f"可用内存 {avail_mb}MB < 500MB")

        ok = l2["ok"] and session_status_ok
        fields = {"turn_count": turn_total, "process_count": process_count,
                  "api_count": api_count, "mcp_count": mcp_count,
                  "mcp_orphan_count": orphan_count, "glue_count": glue_count,
                  "avail_mb": avail_mb, "swap_used_mb": swap_used_mb,
                  "api_rss_mb": api_rss_mb,
                  "snapshot_freshness_s": round(max_age, 1) if max_age else 0,
                  "snapshot_count": snap["snapshot_count"],
                  "checks": checks}
        self.metric("L3", ok, (time.time() - t0) * 1000, fields)
        return {"ok": ok, "latency_ms": (time.time() - t0) * 1000, "checks": checks}

    # -- 编排 --------------------------------------------------------
    def run_once(self) -> int:
        # 快照目录状态在 L2/L3 间复用（一次探针）
        self._snap_state_cache = snapshot_state(
            self.args.snapshot_dir, self._known_sessions())
        results = {}
        if self.args.level in ("L1", "all"):
            results["L1"] = self.check_l1()
        if self.args.level in ("L2", "all"):
            results["L2"] = self.check_l2(results.get("L1", {"ok": True, "checks": {}}))
        if self.args.level in ("L3", "all"):
            results["L3"] = self.check_l3(results.get("L2", {"ok": True, "checks": {}}))

        worst = 0
        for lv, r in results.items():
            if not r["ok"]:
                worst = max(worst, 3)
        if any(a["severity"] == SEV_WARN for a in self.alerts):
            worst = max(worst, 2)
        return worst

    def _known_sessions(self) -> list:
        """快照新鲜度优先用 /sessions 列表；不可达时回退到快照目录枚举。"""
        try:
            _, _, body = http_request(f"{self.base_url}/sessions", timeout=10)
            if body and body.get("sessions"):
                return body["sessions"]
        except Exception:
            pass
        snap_dir = self.args.snapshot_dir
        if os.path.isdir(snap_dir):
            return [d for d in os.listdir(snap_dir)
                    if os.path.isdir(os.path.join(snap_dir, d))]
        return []


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="LMS 三级健康检查 + 指标采集（stdlib only）")
    p.add_argument("--level", choices=["L1", "L2", "L3", "all"], default="all",
                   help="检查级别（默认 all：三级顺序执行）")
    p.add_argument("--base-url", default=None,
                   help="API 地址（默认取 LMS_API_HOST/LMS_API_PORT 或 "
                        "http://127.0.0.1:8190）")
    p.add_argument("--project-root", default=None,
                   help="LMS 项目根（默认脚本所在目录的上级，用于定位 logs/）")
    p.add_argument("--snapshot-dir", default=None,
                   help="快照目录（默认 LMS_SNAPSHOT_DIR 或 <项目根>/snapshots）")
    p.add_argument("--metrics-file", default=None,
                   help="指标文件（默认 <项目根>/logs/lms_metrics.jsonl）")
    p.add_argument("--alerts-file", default=None,
                   help="告警文件（默认 <项目根>/logs/lms_alerts.jsonl）")
    p.add_argument("--freshness-warn", type=int, default=DEFAULT_FRESHNESS_WARN_S,
                   help=f"快照新鲜度告警阈值秒（默认 {DEFAULT_FRESHNESS_WARN_S}）")
    p.add_argument("--loop", action="store_true",
                   help="持续运行（默认单次，cron 驱动即可）")
    p.add_argument("--interval", type=float, default=60.0,
                   help="--loop 模式下两次检查间隔秒（默认 60）")
    return p


def main() -> int:
    args = build_parser().parse_args()

    # 路径解析：脚本位于 <项目根>/scripts/
    project_root = args.project_root or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    args.project_root = project_root
    args.base_url = args.base_url or (
        f"http://{os.environ.get('LMS_API_HOST', '127.0.0.1')}:"
        f"{os.environ.get('LMS_API_PORT', '8190')}")
    args.snapshot_dir = args.snapshot_dir or os.environ.get(
        "LMS_SNAPSHOT_DIR") or os.path.join(project_root, "snapshots")
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    args.metrics_file = args.metrics_file or os.path.join(log_dir, "lms_metrics.jsonl")
    args.alerts_file = args.alerts_file or os.path.join(log_dir, "lms_alerts.jsonl")

    mon = Monitor(args)
    if not args.loop:
        return mon.run_once()

    # 持续模式：捕获 KeyboardInterrupt 干净退出
    try:
        while True:
            mon.run_once()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())

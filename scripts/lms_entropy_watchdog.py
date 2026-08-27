#!/usr/bin/env python3
"""lms_entropy_watchdog.py — 熵看门狗（2026-08-22，dandan 指示：自动换新鸡蛋，别硬整焦蛋）

监测 LMS 场的健康：若进入"烧焦"态（熵>0.9995 或 σ 顶格>0.95，连续 N 次），
**自动**回滚到"最后一次健康快照"（自动换蛋）并重启 LMS。全程日志 + 事件记录。

设计要点：
  - 与 lms_ops_monitor.py 互补：那个只报警，这个直接处置（自愈）。
  - "最后一次健康快照"：每次观测到熵<0.995 时，记录当时的 latest_main.pt 路径。
    烧焦时回滚到它（健康蛋）。若从未记录过健康快照 → 用内置兜底快照（如有）。
  - fail-open：LMS 不可达（curl 失败）→ 只记日志不动手（可能正在重启/维护）。
  - 处置后冷却：一次处置后 30 分钟内不再处置（防抖）。
  - 自愈后验证：重启后等 20s，若仍烧焦 → 记警（交给人工，不再自动循环）。

用法（cron 每 5 分钟）：
  # [C11] stdout 建议重定向 /dev/null：log() 已只写日志文件（不再 print），
  # 避免 cron 与 log() 双写同一文件造成每条两行（crontab 属运维侧）
  */5 * * * * .venv/bin/python scripts/lms_entropy_watchdog.py > /dev/null 2>&1
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

LMS_URL = "http://127.0.0.1:8190"
CONTROL_URL = "http://127.0.0.1:8191"
STATE_FILE = "/vol2/1000/AI专用/living-memory-system-cloud/runtime/entropy_watchdog_state.json"
LOG_FILE = "/vol2/1000/AI专用/living-memory-system-cloud/logs/lms_entropy_watchdog.log"
SNAP_DIR = "/vol2/1000/AI专用/living-memory-system-cloud/snapshots/main"
ENV_DUMP = "/tmp/lms-orig-env.txt"

ENTROPY_BURNT = 0.9995   # 熵烧焦线（真正焦蛋才换蛋）
SIGMA_SATURATED = 0.95   # σ 顶格线（饱和 → 轻量 σ 重置，不换蛋）
ENTROPY_HEALTHY = 0.995  # 低于此线记"健康"
BURNT_STREAK_NEED = 2    # 连续 N 次烧焦才处置（防抖动）
DISPOSE_COOLDOWN_S = 1800  # 处置后冷却
# [C11] σ 重置冷却 600s = 2× cron 周期（300s）：原 300s 恰等于 cron 周期 → 实际无冷却
SIGMA_RESET_COOLDOWN_S = 600

# [C2] 选蛋标准（lms-field-restore 技能）：健康蛋 ≠ 近当前态（每 34s 覆盖的 latest）
EGG_MIN_AGE_S = 2 * 3600        # [C2] 蛋龄下限 2h：近当前态不是蛋，是刚孵化的鸡
EGG_J_BAND = (5.0, 9.0)         # [C2] J 范数健康带（健康≈7；烧焦≈12）
EGG_ISOLATION_S = 2 * 3600      # [C2] 换蛋后隔离期：2h 内不记录 last_healthy
# [C2] 灰池修复完成时刻（2026-08-22 12:28:52，epoch 1787372932）；早于它的蛋不用。
# 注：方案稿字面值 1724322532 系 2024-08-22 的笔误，与注释日期矛盾；按注释日期取正确 epoch，
# 否则灰池修复前的旧蛋（pre-clean 08-10/08-17 等）会通过过滤被当作健康蛋。
GRAY_FIX_MTIME = 1787372932.0


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    # [C11] 不再 print：cron 重定向 stdout 到同一日志文件会造成每条两行；
    # 如需 stdout 留痕，改由 crontab 重定向到 /dev/null（运维侧动作，见方案 C11）


def http_get(path, timeout=6):
    try:
        with urllib.request.urlopen(LMS_URL + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        return {"error": str(e)}


def read_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"streak": 0, "last_healthy": "", "last_dispose": 0}


def write_state(st):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(st, f)
    except Exception:
        pass


def latest_snapshot():
    try:
        cands = [p for p in os.listdir(SNAP_DIR) if p.endswith(".pt")]
        if not cands:
            return ""
        return max(cands, key=lambda p: os.path.getmtime(
            os.path.join(SNAP_DIR, p)))
    except Exception:
        return ""


def _probe_snapshot_jnorm(snap_path):
    """[C2] 子进程 torch.load 探测 J 范数（fail-open 返回 None，不阻塞主流程）。"""
    script = ('import torch,warnings; warnings.filterwarnings("ignore");'
              f'd=torch.load({os.path.join(SNAP_DIR, snap_path)!r},'
              ' map_location="cpu", weights_only=False);'
              'J=d["attractor"]["J"];'
              'print(round(float(torch.norm(J, p="fro").item()),2))')
    try:
        env = dict(os.environ)
        if os.path.exists(ENV_DUMP):
            for line in open(ENV_DUMP):
                line = line.rstrip("\n")
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k] = v
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd="/vol2/1000/AI专用/living-memory-system-cloud",
            env=env, capture_output=True, text=True, timeout=60)
        return float(proc.stdout.strip().splitlines()[-1])
    except Exception:
        return None


def _pick_healthy_egg(now):
    """[C2] 候选蛋筛选：蛋龄≥2h + mtime≥灰池修复 + J∈[5,9]（近当前态/毒蛋一律排除）。

    按 mtime 从新到旧探测，第一个合格即返回（通常 1 次 torch.load 即止）；
    无合格候选返回 None（保守：宁可无蛋也不回滚到近当前态）。"""
    try:
        cands = sorted(
            (p for p in os.listdir(SNAP_DIR) if p.endswith(".pt")),
            key=lambda p: os.path.getmtime(os.path.join(SNAP_DIR, p)),
            reverse=True)
    except Exception:
        return None
    for p in cands:
        full = os.path.join(SNAP_DIR, p)
        try:
            mtime = os.path.getmtime(full)
        except OSError:
            continue
        age = now - mtime
        if age < EGG_MIN_AGE_S or mtime < GRAY_FIX_MTIME:
            continue
        j = _probe_snapshot_jnorm(p)
        if j is None:
            continue
        if EGG_J_BAND[0] <= j <= EGG_J_BAND[1]:
            return p
    return None


def reset_sigma(state, reason):
    """[C3] σ 饱和处置 —— 走控制面 :8191 /control/reset-sigma（dandan 2026-08-23 批准）。

    原理：数据面新增 POST /reset-sigma/{session_id} 端点在 **live 进程内**
    执行 attractor.reset_state()（σ 归零、保留 J 与记忆、不重启不换蛋），
    由数据面唯一写者落盘。此前独立进程加载磁盘副本重置对 live 无效且
    与快照写者并发写同一 .pt（审计 P0-8）；本实现消除无效处置与撕裂风险。
    """
    token = os.environ.get("LMS_CONTROL_TOKEN", "").strip()
    if not token:
        # cron 环境不加载 .env → 兜底解析 .env 文件（审计 P1：产线 token 缺失
        # 导致自动重置死路）。只读，不污染进程环境。
        try:
            env_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", ".env")
            for line in open(env_path, encoding="utf-8"):
                line = line.strip()
                if line.startswith("LMS_CONTROL_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        except Exception:
            token = ""
    if not token:
        log(f"♻️ σ 饱和（{reason}）——LMS_CONTROL_TOKEN 未配置，无法调控制面，跳过（交人工）")
        return False
    try:
        req = urllib.request.Request(
            CONTROL_URL + "/control/reset-sigma",
            data=json.dumps({"session_id": "main"}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Control-Token": token,
            },
            method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            body = json.loads(r.read().decode("utf-8", errors="replace"))
        ok = bool(body.get("reset")) or bool(body.get("status") == "ok")
        log(f"♻️ σ 饱和重置（{reason}）→ {'成功' if ok else '响应异常'} {str(body)[:120]}")
        return ok
    except Exception as e:
        log(f"❌ σ 饱和重置失败（{reason}）: {e}")
        return False


def dispose_burnt(state, reason):
    """自动换蛋：停服 → 恢复健康快照 → 设 bias → 存档 → 重启（重启后验证）。"""
    log(f"⚠️ 触发自动换蛋（{reason}）")
    now = time.time()
    snap = state.get("last_healthy", "")
    if not snap or not os.path.exists(os.path.join(SNAP_DIR, snap)):
        log("❌ 无健康快照可用（从未记录），跳过自动处置，交人工")
        return False
    # [C2] 毒蛋字段弃用校验：last_healthy 不满足蛋龄/灰池修复时刻 → 不换蛋
    # （旧实现回滚近当前态 latest_main.pt → 恢复后 ~35 分钟再度饱和，换蛋循环丢记忆）
    try:
        mtime = os.path.getmtime(os.path.join(SNAP_DIR, snap))
    except OSError:
        log("❌ last_healthy 读取失败，跳过处置交人工")
        return False
    if mtime < GRAY_FIX_MTIME or now - mtime < EGG_MIN_AGE_S:
        log(f"❌ last_healthy 不满足蛋龄/灰池修复校验（mtime {mtime}），毒蛋字段弃用，跳过处置交人工")
        return False
    try:
        # 停服
        subprocess.run(["pkill", "-f", "api.run"], capture_output=True)
        time.sleep(5)
        # 恢复健康快照 + 设 bias + 存档
        script = f'''
import torch, warnings; warnings.filterwarnings("ignore")
from api.session_manager import SessionManager
from api.config import get_api_config
sm = SessionManager(default_config_factory=get_api_config)
loop = sm.get_or_create("main")
loop.load_state("{SNAP_DIR}/{snap}")
net = loop.attractor
import os
bias = float(os.environ.get("LMS_BIAS_SCALE", "0.9") or 0.9)
net.bias = torch.full((256,), bias)
hb = getattr(net, "homeostatic_bias", None)
if hb is not None: hb.bias = bias; hb.init_bias = bias
loop.save_session_state()
print("restored:", "{SNAP_DIR}/{snap}")
'''
        env = dict(os.environ)
        if os.path.exists(ENV_DUMP):
            for line in open(ENV_DUMP):
                line = line.rstrip("\n")
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k] = v
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd="/vol2/1000/AI专用/living-memory-system-cloud",
            env=env, capture_output=True, text=True, timeout=120)
        log(f"  恢复输出: {proc.stdout.strip()[-100:]} {proc.stderr.strip()[-100:]}")
        # [C5] 恢复脚本 returncode 检查：rc!=0 → 假成功换蛋（30 分钟盲区）；一律不重启
        if proc.returncode != 0:
            log(f"❌ 恢复脚本失败（rc={proc.returncode}），不重启，交人工: {proc.stderr.strip()[-200:]}")
            return False
        # 重启
        subprocess.Popen(
            [sys.executable, "-m", "api.run", "--host", "127.0.0.1",
             "--port", "8190"],
            cwd="/vol2/1000/AI专用/living-memory-system-cloud",
            env=env, stdout=open("/tmp/lms-api.log", "a"),
            stderr=subprocess.STDOUT, start_new_session=True)
        # [C5] 兑现 docstring"等 20s 验证"承诺：重启后复读 /status /landscape，
        # 仍烧焦/饱和 → 返回 False（main() 不置 last_dispose，下轮可重试/升级人工）
        time.sleep(20)
        st2 = http_get("/status/main")
        ls2 = http_get("/landscape/main")
        if "error" in st2 or "error" in ls2:
            log("❌ 重启后状态不可读（fail-open 复核失败）——不置 last_dispose（下轮重试/升级人工）")
            return False
        ent2 = float(st2.get("status", {}).get("entropy_ratio", 0) or 0)
        top2 = ls2.get("landscape", {}).get("activation", {}).get("top_activated", [])
        sig2 = max([n.get("sigma", 0) for n in top2], default=0)
        if ent2 > ENTROPY_BURNT or sig2 > SIGMA_SATURATED:
            log(f"❌ 重启后仍烧焦/饱和（熵 {ent2:.4f} / σmax {sig2:.3f}）——不置 last_dispose（下轮重试/升级人工）")
            return False
        log(f"✅ 换蛋成功且验证通过（熵 {ent2:.4f} / σmax {sig2:.3f}），LMS 已重启")
        return True
    except Exception as e:
        log(f"❌ 自动换蛋失败: {e}")
        return False


def main():
    state = read_state()
    now = time.time()
    # 处置冷却检查
    if now - state.get("last_dispose", 0) < DISPOSE_COOLDOWN_S:
        return 0

    status = http_get("/status/main")
    landscape = http_get("/landscape/main")
    if "error" in status or "error" in landscape:
        log(f"LMS 不可达（fail-open 不动手）: {status.get('error','')}")
        return 0

    entropy = float(status.get("status", {}).get("entropy_ratio", 0) or 0)
    ls = landscape.get("landscape", {})
    act = ls.get("activation", {})
    top = act.get("top_activated", [])
    sigma_max = max([n.get("sigma", 0) for n in top], default=0)

    # [C4] 判定顺序重构（2026-08-23 审计）：先判烧焦/饱和，再判健康——
    # 原实现"σ<0.9 或 熵<0.995"先于烧焦判定：σ 饱和但熵<0.995 的 onset 阶段会把
    # 饱和快照记为 last_healthy（毒蛋）且 return，σ 重置分支永不触发。
    # 两级化（2026-08-23 dandan 指出 σ 饱和≠焦蛋）：
    #   - 熵烧焦（entropy > ENTROPY_BURNT）→ 重量换蛋（恢复健康蛋快照重启）
    #   - σ 饱和（σmax > SIGMA_SATURATED）→ 轻量 σ 重置（C3 获批前只告警不写盘）
    sigma_sat = sigma_max > SIGMA_SATURATED
    ent_burnt = entropy > ENTROPY_BURNT

    if ent_burnt:
        state["streak"] = state.get("streak", 0) + 1
        write_state(state)
        log(f"⚠️ 烧焦信号（熵 {entropy:.4f} / σmax {sigma_max:.3f}），连续 {state['streak']}/{BURNT_STREAK_NEED}")
        if state["streak"] >= BURNT_STREAK_NEED:
            if dispose_burnt(state, f"熵 {entropy:.4f} / σmax {sigma_max:.3f}"):
                state["streak"] = 0
                state["last_dispose"] = now
                write_state(state)
        return 0

    if sigma_sat:
        # σ 饱和但熵未烧焦 → 轻量 σ 重置（不重启不换蛋；C3 获批前只告警不写盘）
        cooldown = state.get("last_sigma_reset", 0)
        if now - cooldown < SIGMA_RESET_COOLDOWN_S:
            return 0
        state["streak"] = 0
        if reset_sigma(state, f"σmax {sigma_max:.3f} / 熵 {entropy:.4f}"):
            state["last_sigma_reset"] = now
            write_state(state)
        return 0

    # 未烧焦未饱和 → 健康记录：
    # [C4] σmax<0.9 **且** 熵<ENTROPY_HEALTHY 同时成立才记录健康蛋；σmax>0.9 时绝不写
    # last_healthy（防毒蛋）；中间带（0.995≤熵≤0.9995 且 σ<0.9）不健康也不烧焦 → 只清 streak。
    # [C2] 健康蛋按选蛋标准记录（蛋龄≥2h + 灰池修复后 + J∈[5,9]）；换蛋后隔离期内跳过记录。
    if sigma_max < 0.9 and entropy < ENTROPY_HEALTHY:
        if now - state.get("last_dispose", 0) >= EGG_ISOLATION_S:
            snap = _pick_healthy_egg(now)
            if snap:
                state["last_healthy"] = snap
                log(f"✅ 已记录健康蛋: {snap[:40]} (σmax {sigma_max:.3f} / 熵 {entropy:.4f})")
        state["streak"] = 0
        write_state(state)
    else:
        state["streak"] = 0
        write_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())

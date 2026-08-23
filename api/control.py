#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMS 活体记忆系统 - 自我配置端口（Control Plane，:8191）

背景（阶段 2 / T2.1 子集 + §3.4，总体方案）：
    数据面 api/server.py（:8190）负责记忆读写与快照（唯一写者）；
    本模块是独立的管理面 FastAPI app，监听 127.0.0.1:8191，提供：
      - 认证：X-Control-Token 请求头（兼容 Authorization: Bearer），
        token 从 .env 读 LMS_CONTROL_TOKEN，校验用 hmac.compare_digest；
      - 缺失 token → fail-open 只读模式：读端点可用，写端点一律 503；
      - 写操作全量审计 → logs/control-audit.jsonl（append-only）；
      - 热配置仅内存生效（白名单键），不落盘、不重启数据面；
      - 接入方注册 → data/control/access.jsonl（token 只存 sha256）。

工程级约束（与 api/server.py 零冲突）：
    * 本模块【不 import】api.server / api.session_manager / torch / runtime.*，
      独立 app；对数据面的全部调用走 HTTP（urllib stdlib），超时钳制；
    * fail-open：:8190 不可达 / embed 探针失败等外部依赖问题只降级不崩溃；
    * 审计与注册文件均 0600 权限，追加写 + 线程锁，失败仅告警不阻断。

启动（见 scripts/run_control.py）：
    .venv/bin/python scripts/run_control.py --port 8191
"""

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 项目根与 .env 加载（独立于 api/server.py，零 import 冲突）
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """极简 .env 加载器：KEY=VALUE 逐行解析，已存在的环境变量优先（不覆盖）。

    与 MCP/API 启动的 `set -a && . ./.env` 语义一致，但让
    `python scripts/run_control.py` 无需手动 source 也能读到配置。
    """
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if not key:
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        if key not in os.environ:
            os.environ[key] = val


_load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api.control")

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
LMS_API_BASE = os.environ.get("LMS_CTRL_API_BASE", "http://127.0.0.1:8190").rstrip("/")
CONTROL_TOKEN = os.environ.get("LMS_CONTROL_TOKEN", "").strip()
CTRL_HOST = os.environ.get("LMS_CTRL_HOST", "127.0.0.1")
CTRL_PORT = int(os.environ.get("LMS_CTRL_PORT", "8191"))

SNAPSHOT_DIR = Path(os.environ.get("LMS_SNAPSHOT_DIR", str(PROJECT_ROOT / "snapshots")))
ACCESS_FILE = PROJECT_ROOT / "data" / "control" / "access.jsonl"
AUDIT_FILE = PROJECT_ROOT / "logs" / "control-audit.jsonl"

# 只读模式：token 缺失 → 写端点 503（fail-open 只读）
READ_ONLY_MODE = not bool(CONTROL_TOKEN)

# ---------------------------------------------------------------------------
# 热配置白名单（§3.4：DREAM_* / LMS_FEED_RATE_LIMIT / LMS_SELF_REF_* /
# LMS_EMBEDDER / LMS_CLOUD_EMBED_* / LMS_API_HOST|PORT；只允许有限键）
# ---------------------------------------------------------------------------
CONFIG_WHITELIST: Dict[str, str] = {
    # 做梦调度
    "DREAM_SESSION_ALLOW": "str",            # 允许做梦的会话白名单（逗号分隔）
    "DREAM_IDLE_THRESHOLD": "float",
    "DREAM_STEPS": "int",
    "DREAM_CHECK_INTERVAL": "float",
    "DREAM_FULL_CYCLE": "bool",
    # 数据面限流
    "LMS_FEED_RATE_LIMIT": "int",
    # 自指回路
    "LMS_SELF_REF_ENABLED": "bool",
    "LMS_SELF_REF_LLM_DISTILL_ENABLED": "bool",
    "LMS_SELF_REF_LLM_DISTILL_INTERVAL": "int",
    # 嵌入
    "LMS_EMBEDDER": "str",
    "LMS_CLOUD_EMBED_URL": "str",
    "LMS_CLOUD_EMBED_MODEL": "str",
    "LMS_CLOUD_EMBED_DIM": "int",
    "LMS_CLOUD_EMBED_FALLBACK_URL": "str",
    # 数据面监听
    "LMS_API_HOST": "str",
    "LMS_API_PORT": "int",
}

# 密钥类键：任何视图/清单中一律打码，绝不回显
_SECRET_KEY_RE = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|API_KEY)", re.IGNORECASE)

# 内存热覆盖（进程内生效；不落盘——落盘需显式 T2.1 /admin/config/persist）
_overrides: Dict[str, str] = {}
_overrides_lock = threading.Lock()

# ---------------------------------------------------------------------------
# 审计与注册文件（append-only，0600，线程锁）
# ---------------------------------------------------------------------------
_audit_lock = threading.Lock()


def _ensure_file(path: Path) -> None:
    """确保文件存在且权限 0600（首次创建时即收紧权限）。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            fd = os.open(str(path), os.O_CREAT | os.O_WRONLY, 0o600)
            os.close(fd)
        os.chmod(str(path), 0o600)
    except Exception as e:  # fail-open：权限设置失败不阻断服务
        logger.warning(f"文件权限设置失败 {path}: {e}")


def _append_jsonl(path: Path, entry: dict) -> None:
    """线程安全地追加一条 JSONL。失败仅告警（审计不阻断主流程）。"""
    try:
        _ensure_file(path)
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with _audit_lock, open(path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
    except Exception as e:
        logger.warning(f"JSONL 追加失败 {path}: {e}")


def _audit(action: str, client: str, status: str = "ok",
           extra: Optional[dict] = None, latency_ms: Optional[float] = None) -> None:
    """写操作审计：{ts, action, client, status, latency_ms, ...} → control-audit.jsonl。"""
    entry: Dict[str, Any] = {
        "ts": datetime.now().isoformat(),
        "action": action,
        "client": client,
        "status": status,
        "latency_ms": latency_ms,
    }
    if extra:
        entry.update(extra)
    _append_jsonl(AUDIT_FILE, entry)


# ---------------------------------------------------------------------------
# 认证（hmac.compare_digest 防时序攻击）
# ---------------------------------------------------------------------------
def _bearer_from_header(auth_header: str) -> str:
    """从 Authorization 头提取 Bearer token（§3.7 统一协议兼容）。"""
    if auth_header and auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return ""


def _client_id_of(request: Request) -> str:
    """审计用客户端标识：优先 X-Client-Id 头，否则 anon。"""
    cid = request.headers.get("X-Client-Id", "").strip()
    return cid if cid else "anon"


def _check_auth(request: Request) -> None:
    """只读端点鉴权：token 未配置 → 放行（fail-open 只读）；配置了则必须匹配。

    [B9] 消费方校验：CONTROL_TOKEN 或注册下发的 client token（sha256 比对
    access.jsonl 注册表）二选一通过——旧实现"只发不收"：register 下发的
    token 无任何消费方校验，_check_auth 只比对 CONTROL_TOKEN。
    """
    if READ_ONLY_MODE:
        return
    supplied = (
        request.headers.get("X-Control-Token", "").strip()
        or _bearer_from_header(request.headers.get("Authorization", ""))
    )
    if not supplied:
        _audit("auth_denied", _client_id_of(request), status="denied")
        raise HTTPException(
            status_code=401,
            detail="无效或缺失的控制令牌（X-Control-Token / Authorization: Bearer）",
        )
    if hmac.compare_digest(supplied, CONTROL_TOKEN):
        return
    # [B9] 注册表 token 校验（sha256 比对；命中 → 放行并审计区分来源）
    if _registered_token_valid(supplied):
        _audit("auth_ok_registered", _client_id_of(request))
        return
    _audit("auth_denied", _client_id_of(request), status="denied")
    raise HTTPException(
        status_code=401,
        detail="无效或缺失的控制令牌（X-Control-Token / Authorization: Bearer）",
    )


def _registered_token_valid(token: str) -> bool:
    """[B9] 注册下发的 client token 校验：sha256(token) 命中注册表 token_hash 即通过。

    注册流程（control_register）只存 token 的 sha256 哈希（防泄露即用）；
    消费方鉴权时把提交 token 哈希后与注册表逐条比对（hmac.compare_digest
    防时序攻击）。注册表为低频小文件，直接逐次读取可接受。
    """
    if not token:
        return False
    h = hashlib.sha256(token.encode("utf-8")).hexdigest()
    for r in _load_registrations():
        rh = r.get("token_hash")
        if rh and isinstance(rh, str) and hmac.compare_digest(rh, h):
            return True
    return False


def _require_write(request: Request) -> None:
    """写端点鉴权：token 缺失 → 503 只读模式（任务规格：禁用写端点只留只读）。"""
    if READ_ONLY_MODE:
        raise HTTPException(
            status_code=503,
            detail=(
                "控制面处于只读模式：LMS_CONTROL_TOKEN 未配置，写端点已禁用。"
                "请在 .env 中配置 LMS_CONTROL_TOKEN 后重启控制面。"
            ),
        )
    _check_auth(request)


# ---------------------------------------------------------------------------
# 数据面 :8190 HTTP 调用（stdlib urllib，超时钳制，fail-open）
# ---------------------------------------------------------------------------
def _http_json(method: str, path: str, payload: Optional[dict] = None,
               timeout: float = 5.0) -> Tuple[int, Any]:
    """对数据面 :8190 发起 JSON 调用。返回 (status, body)；网络错误 → (0, {detail})。

    绝不抛异常：外部依赖故障只降级，不拖垮控制面。
    """
    url = LMS_API_BASE + path
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8", errors="replace"))
        except Exception:
            return e.code, {"detail": f"HTTP {e.code}"}
    except Exception as e:
        return 0, {"detail": f"{type(e).__name__}: {e}"}


async def _ahttp_json(method: str, path: str, payload: Optional[dict] = None,
                      timeout: float = 5.0) -> Tuple[int, Any]:
    """异步包装：urllib 阻塞调用交给线程池（asyncio.to_thread），不卡事件循环。"""
    import asyncio
    return await asyncio.to_thread(_http_json, method, path, payload, timeout)


# ---------------------------------------------------------------------------
# 本机状态采样（ps / 快照文件，不依赖 :8190）
# ---------------------------------------------------------------------------
_PROC_PATTERNS = ("api.run", "api.server", "api.control", "mcp_memory_server",
                  "lms_http_mcp", "uvicorn")


def _process_info() -> dict:
    """采样本机 LMS 相关进程（api/mcp/control）：pid/rss/cmd 清单 + 汇总。"""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,rss=,args="],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception as e:
        return {"error": f"ps 不可用: {e}", "count": 0, "rss_total_mb": 0.0,
                "processes": []}
    procs = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, rss_kb, cmd = parts[0], parts[1], parts[2]
        if any(p in cmd for p in _PROC_PATTERNS):
            procs.append({
                "pid": int(pid),
                "rss_mb": round(int(rss_kb) / 1024, 1),
                "cmd": cmd[:120],
            })
    return {
        "count": len(procs),
        "rss_total_mb": round(sum(p["rss_mb"] for p in procs), 1),
        "processes": procs,
    }


_SID_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]")


def _snapshot_freshness(session_id: str) -> Tuple[Optional[str], Optional[int]]:
    """返回 (最新快照 iso 时间, 距今秒数)；无快照 → (None, None)。

    候选（新命名规范优先）：snapshots/{sid}/latest_{sid}.pt →
    snapshots/{sid}/latest.pt → snapshots/latest_{sid}.pt → snapshots/latest.pt
    """
    sid = _SID_SAFE_RE.sub("_", session_id)
    candidates = [
        SNAPSHOT_DIR / sid / f"latest_{sid}.pt",
        SNAPSHOT_DIR / sid / "latest.pt",
        SNAPSHOT_DIR / f"latest_{sid}.pt",
        SNAPSHOT_DIR / "latest.pt",
    ]
    best_mtime: Optional[float] = None
    for p in candidates:
        try:
            if p.is_file():
                m = p.stat().st_mtime
                if best_mtime is None or m > best_mtime:
                    best_mtime = m
        except Exception:
            continue
    if best_mtime is None:
        return None, None
    age = max(0, int(time.time() - best_mtime))
    return datetime.fromtimestamp(best_mtime).isoformat(), age


async def _gather_sessions() -> List[dict]:
    """聚合各会话状态：:8190 /sessions + /status/{sid} + 本地快照新鲜度。"""
    _, sbody = await _ahttp_json("GET", "/sessions", timeout=3.0)
    sids = sbody.get("sessions", []) if isinstance(sbody, dict) else []
    if not sids:
        return []
    import asyncio
    tasks = [
        _ahttp_json("GET", f"/status/{urllib.parse.quote(sid)}", timeout=3.0)
        for sid in sids
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out: List[dict] = []
    for sid, r in zip(sids, results):
        st: Dict[str, Any] = {"session_id": sid}
        if isinstance(r, tuple) and r[0] == 200 and isinstance(r[1], dict):
            status = r[1].get("status", {}) or {}
            st.update({
                "turn_count": status.get("turn_count"),
                "entropy": status.get("last_entropy"),
                "surprise": status.get("last_surprise"),
                "entropy_ratio": status.get("entropy_ratio"),
                "episodic_buffer_size": status.get("episodic_buffer_size"),
                "llm_enabled": status.get("llm_enabled"),
            })
        else:
            st["error"] = str(r)
        snap_time, snap_age = _snapshot_freshness(sid)
        st["snapshot_time"] = snap_time
        st["snapshot_age_s"] = snap_age
        out.append(st)
    return out


# ---------------------------------------------------------------------------
# 配置视图（脱敏）
# ---------------------------------------------------------------------------
def _redact(key: str, value: Any) -> Any:
    """脱敏：密钥类键永不回显值，只回显是否已配置。"""
    if _SECRET_KEY_RE.search(key):
        return {"configured": bool(value)}
    return value


def _effective_config() -> Dict[str, Any]:
    """生效配置 = .env/环境变量基底 + 内存热覆盖（overrides 优先）。"""
    cfg: Dict[str, Any] = {}
    for key in CONFIG_WHITELIST:
        val = _overrides.get(key) if key in _overrides else os.environ.get(key, "")
        cfg[key] = _redact(key, val)
    # 额外展示（密钥类，只显配置状态）
    for key in ("DEEPSEEK_API_KEY", "LMS_LLM_API_KEY"):
        cfg[key] = _redact(key, os.environ.get(key, ""))
    return cfg


def _validate_value(key: str, value: str) -> Optional[str]:
    """按白名单类型校验；非法返回错误信息，合法返回 None。"""
    kind = CONFIG_WHITELIST.get(key)
    if kind is None:
        return f"键不在热配置白名单内: {key}"
    if isinstance(value, str):
        value = value.strip()
    try:
        if kind == "int":
            int(str(value))
        elif kind == "float":
            float(str(value))
        elif kind == "bool":
            v = str(value).strip().lower()
            if v not in ("true", "false", "1", "0", "yes", "no"):
                return f"布尔值须为 true/false: {key}={value}"
    except ValueError:
        return f"值类型不符（期望 {kind}）: {key}={value}"
    return None


# ---------------------------------------------------------------------------
# Pydantic 请求模型
# ---------------------------------------------------------------------------
class ConfigUpdateRequest(BaseModel):
    key: Optional[str] = Field(None, description="单键热更新（与 updates 二选一）")
    value: Optional[str] = Field(None, description="单键值")
    updates: Optional[Dict[str, str]] = Field(
        None, description="批量热更新 {键: 值}（仅白名单键）")


class SnapshotRequest(BaseModel):
    session_ids: Optional[List[str]] = Field(
        None, description="要快照的会话列表；缺省/空数组 = 全部会话")


class RegisterRequest(BaseModel):
    client_id: str = Field(..., description="接入方唯一标识（如 openclaw-main / codex-cli）")
    purpose: str = Field("", description="接入用途说明")
    platform: str = Field("generic", description="平台：openclaw / codex / workbody / generic")
    role: str = Field("agent", description="预留角色：monitor / agent / admin（阶段 2 T2.1 角色鉴权启用）")


# ---------------------------------------------------------------------------
# FastAPI 应用（独立 app，与 api/server.py 的 app 无任何共享状态）
# ---------------------------------------------------------------------------
app = FastAPI(
    title="LMS Control Plane",
    description=(
        "活体记忆系统自我配置端口（管理面）。"
        "数据面为 :8190（唯一快照写者）；本端口只做管理/聚合/审计，"
        "不复制大脑、不改数据面代码。认证：X-Control-Token。"
    ),
    version="0.1.0",
)


@app.get("/control/health")
async def control_health():
    """服务 + 各会话状态总览（turn/熵/惊讶度/快照时间/进程数）。

    公开端点（存活探针，不要求 token）：token 未配置时全员可用；
    配置后仍公开，便于监控探活（敏感信息请走 /control/sessions）。
    """
    t0 = time.time()
    api_status, api_body = await _ahttp_json("GET", "/health", timeout=3.0)
    sessions = await _gather_sessions()
    procs = _process_info()
    return {
        "status": "ok",
        "service": "lms-control-plane",
        "version": app.version,
        "timestamp": datetime.now().isoformat(),
        "read_only_mode": READ_ONLY_MODE,
        "control_token": "missing" if READ_ONLY_MODE else "configured",
        "api": {
            "reachable": api_status == 200,
            "status": api_body.get("status") if isinstance(api_body, dict) else None,
            "active_sessions": (
                api_body.get("active_sessions") if isinstance(api_body, dict) else None
            ),
            "error": None if api_status == 200 else api_body.get("detail"),
        },
        "sessions": sessions,
        "processes": {
            "count": procs.get("count", 0),
            "rss_total_mb": procs.get("rss_total_mb", 0.0),
        },
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }


@app.get("/control/sessions")
async def control_sessions(request: Request):
    """会话列表 + 详情（turn/熵/惊讶度/快照新鲜度）。"""
    _check_auth(request)
    sessions = await _gather_sessions()
    return {"count": len(sessions), "sessions": sessions}


@app.get("/control/config")
async def control_config(request: Request):
    """只读配置视图（脱敏：token/密钥只回显 configured 状态）。"""
    _check_auth(request)
    return {
        "source": "env(.env) + 内存热覆盖",
        "read_only_mode": READ_ONLY_MODE,
        "overrides_persisted": False,
        "effective": _effective_config(),
        "overrides": dict(_overrides),
        "control": {
            "token_configured": not READ_ONLY_MODE,
            "api_base": LMS_API_BASE,
            "bind": f"{CTRL_HOST}:{CTRL_PORT}",
            "audit_file": str(AUDIT_FILE),
            "access_file": str(ACCESS_FILE),
        },
    }


@app.post("/control/config")
async def control_config_update(request: Request, req: ConfigUpdateRequest):
    """热更新白名单配置（只改内存，不落盘、不重启数据面）。

    生效范围说明：控制面维护 overrides 并展示于 /control/config；
    数据面 :8190 的运行时行为变更需要 T2.1 的 reload-config 端点
    （或重启数据面读 .env）——本端点当前负责"白名单校验 + 记账 + 审计"。
    """
    t0 = time.time()
    _require_write(request)
    client = _client_id_of(request)

    updates: Dict[str, str] = {}
    if req.updates is not None:
        updates = dict(req.updates)
    elif req.key is not None:
        if req.value is None:
            _audit("config_update", client, status="rejected",
                   extra={"key": req.key, "reason": "value 缺失"},
                   latency_ms=(time.time() - t0) * 1000)
            raise HTTPException(status_code=400, detail="value 不能为空")
        updates[req.key] = req.value
    if not updates:
        _audit("config_update", client, status="rejected",
               extra={"reason": "空更新"}, latency_ms=(time.time() - t0) * 1000)
        raise HTTPException(status_code=400, detail="未提供任何键值（key/value 或 updates）")

    # 白名单 + 类型校验（逐键，全部合法才整体生效——防半套配置）
    errors: Dict[str, str] = {}
    for key, value in updates.items():
        err = _validate_value(key, value)
        if err:
            errors[key] = err
    if errors:
        _audit("config_update", client, status="rejected",
               extra={"errors": errors}, latency_ms=(time.time() - t0) * 1000)
        raise HTTPException(status_code=400, detail={"invalid_keys": errors})

    with _overrides_lock:
        for key, value in updates.items():
            _overrides[key] = value.strip() if isinstance(value, str) else str(value)

    # [B8] 热配置假生效修复：控制面 :8191 与数据面 :8190 是独立进程，数据面直读
    # os.environ 且无 reload-config 端点 → 本更新仅记账展示，对数据面零生效。
    # 明示 data_plane_effect=none + 审计标记 data_plane_applied=False + 显式
    # 告警，不制造"已应用"假象（真生效需 [需批准] 新增 reload 转发端点或重启数据面）。
    _audit("config_update", client, extra={"keys": sorted(updates.keys()),
                                           "data_plane_applied": False},
           latency_ms=(time.time() - t0) * 1000)
    logger.warning(
        f"热配置更新（client={client}）: {sorted(updates.keys())} "
        "——仅控制面记账，对数据面 :8190 零生效（数据面为独立进程直读 env，"
        "无 reload 端点；需重启数据面或待 [需批准] reload 转发端点）")
    return {
        "status": "ok",
        "applied": sorted(updates.keys()),
        "overrides": dict(_overrides),
        # [B8] 诚实标注：跨进程无法直写数据面 env，热配置对数据面零生效
        "data_plane_effect": "none",
        "note": ("控制面与数据面 :8190 为独立进程且数据面无 reload 端点："
                 "本更新仅记账展示，对数据面零生效，需重启数据面"
                 "（或待 [需批准] 新增 reload 转发端点）"),
    }


@app.post("/control/snapshot")
async def control_snapshot(request: Request, req: Optional[SnapshotRequest] = None):
    """触发全会话快照（透传 :8190 /snapshot/{sid}，由唯一写者落盘）。"""
    t0 = time.time()
    _require_write(request)
    client = _client_id_of(request)

    _, sbody = await _ahttp_json("GET", "/sessions", timeout=3.0)
    all_sids = sbody.get("sessions", []) if isinstance(sbody, dict) else []
    sids = (req.session_ids if req and req.session_ids else None) or all_sids
    if not sids:
        _audit("snapshot", client, status="rejected", extra={"reason": "无会话"},
               latency_ms=(time.time() - t0) * 1000)
        raise HTTPException(status_code=404, detail="没有可快照的会话")

    import asyncio
    tasks = [
        _ahttp_json("POST", f"/snapshot/{urllib.parse.quote(sid)}", {}, timeout=15.0)
        for sid in sids
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    per_session = []
    ok = 0
    for sid, r in zip(sids, results):
        if isinstance(r, tuple) and r[0] == 200:
            ok += 1
            per_session.append({"session_id": sid, "saved": True,
                                "path": (r[1].get("path") if isinstance(r[1], dict) else None)})
        else:
            per_session.append({"session_id": sid, "saved": False,
                                "error": str(r)})

    _audit("snapshot", client, extra={"sessions": sids, "saved": ok, "failed": len(sids) - ok},
           latency_ms=(time.time() - t0) * 1000)
    return {
        "status": "ok" if ok == len(sids) else "partial",
        "triggered": len(sids),
        "saved": ok,
        "failed": len(sids) - ok,
        "results": per_session,
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }


class ResetSigmaRequest(BaseModel):
    session_id: str = Field("main", description="要重置 σ 的会话；缺省 main")
    reanchor: bool = Field(
        True, description="是否同时重锚 bias（默认 True：σ 归零 + bias←LMS_BIAS_SCALE，"
                          "防重置后 30s 级回弹——纯重置请传 False）")
    # 兼容旧客户端 sid 别名（与数据面 /reset-sigma/{sid} 同策略，防静默丢弃）
    sid: Optional[str] = Field(None, description="session_id 兼容别名（旧客户端）")


@app.post("/control/reset-sigma")
async def control_reset_sigma(request: Request,
                              req: Optional[ResetSigmaRequest] = None):
    """轻量 σ 重置（C3，2026-08-23 dandan 批准新增端点）。

    透传数据面 :8190 POST /reset-sigma/{session_id}——在 **live 进程内**
    执行 attractor.reset_state()（σ 归零、保留 J 与记忆、不重启不换蛋），
    由数据面唯一写者落盘（无并发撕裂）。此前 watchdog 在独立进程加载
    磁盘副本重置，对 live 内存无效且与快照写者并发写同一 .pt（审计
    P0-8）；本端点修复该无效处置，watchdog 改走本端点。

    鉴权：走 _require_write（LMS_CONTROL_TOKEN 或注册 client token）。
    """
    t0 = time.time()
    _require_write(request)
    client = _client_id_of(request)

    sid = "main"
    if req is not None:
        if req.sid is not None and req.session_id == "main":
            # 旧客户端 sid 别名（session_id 显式指定时以它为准）
            sid = req.sid
        elif req.session_id != "main":
            sid = req.session_id

    rc, body = await _ahttp_json(
        "POST", f"/reset-sigma/{urllib.parse.quote(sid)}",
        {"reanchor": req.reanchor if req else True}, timeout=15.0)
    if rc != 200:
        _audit("reset_sigma", client, status="failed",
               extra={"session_id": sid, "rc": rc, "error": str(body)[:200]},
               latency_ms=(time.time() - t0) * 1000)
        raise HTTPException(status_code=rc if rc else 502,
                            detail=f"数据面 σ 重置失败: {body}")
    _audit("reset_sigma", client, extra={"session_id": sid, **body},
           latency_ms=(time.time() - t0) * 1000)
    return {"status": "ok", "control": True, **body}


@app.post("/control/register")
async def control_register(request: Request, req: RegisterRequest):
    """接入方注册：写入 access.jsonl（client_id/用途/时间），返回一次性 client token。

    token 只存 sha256 哈希（防泄露即用）；明文仅本次响应返回一次。
    已存在的 client_id → 409（需显式吊销/改名，防静默覆盖）。
    """
    t0 = time.time()
    _require_write(request)
    client = _client_id_of(request)

    cid = req.client_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{2,63}", cid):
        _audit("register", client, status="rejected",
               extra={"reason": "client_id 格式非法"},
               latency_ms=(time.time() - t0) * 1000)
        raise HTTPException(
            status_code=400,
            detail="client_id 须为 3-64 位字母/数字/._-，且以字母或数字开头",
        )

    # 幂等防重：同 client_id 已注册 → 409
    existing = _load_registrations()
    if any(r.get("client_id") == cid for r in existing):
        _audit("register", client, status="rejected",
               extra={"client_id": cid, "reason": "已存在"},
               latency_ms=(time.time() - t0) * 1000)
        raise HTTPException(
            status_code=409,
            detail=f"client_id '{cid}' 已注册（防静默覆盖；如需轮换请先吊销）",
        )

    platform = re.sub(r"[^A-Za-z0-9._-]", "_", req.platform.strip() or "generic")
    role = re.sub(r"[^A-Za-z0-9._-]", "_", req.role.strip() or "agent")
    token = f"lms_{platform}_{role}_{secrets.token_hex(16)}"
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

    entry = {
        "client_id": cid,
        "purpose": req.purpose.strip()[:200],
        "platform": platform,
        "role": role,
        "created_at": datetime.now().isoformat(),
        "token_hash": token_hash,          # 只存哈希，明文不落盘
        "token_prefix": token[:12] + "…",  # 展示用前缀
    }
    _append_jsonl(ACCESS_FILE, entry)
    _audit("register", client, extra={"client_id": cid, "platform": platform},
           latency_ms=(time.time() - t0) * 1000)
    logger.info(f"接入方注册（client={client}）: {cid} ({platform}/{role})")
    return {
        "status": "ok",
        "client_id": cid,
        "client_token": token,   # 明文仅此一次
        "note": "token 已哈希存储，请立即保存；丢失需重新注册",
    }


@app.get("/control/access")
async def control_access(request: Request):
    """已注册接入方列表（token 脱敏：只显哈希 + 前缀）。"""
    _check_auth(request)
    regs = _load_registrations()
    redacted = []
    for r in regs:
        r2 = dict(r)
        r2["token_hash"] = (r2.get("token_hash") or "")[:16] + "…(sha256)"
        redacted.append(r2)
    return {"count": len(redacted), "registrations": redacted}


def _load_registrations() -> List[dict]:
    """读取 access.jsonl（读失败 → 空列表，fail-open）。"""
    if not ACCESS_FILE.is_file():
        return []
    try:
        out = []
        with open(ACCESS_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out
    except Exception as e:
        logger.warning(f"读取注册表失败: {e}")
        return []


@app.post("/control/diagnose")
async def control_diagnose(request: Request):
    """轻量诊断：会话状态 / 快照新鲜度 / embed 连通性 / 进程清单 → JSON。"""
    t0 = time.time()
    _check_auth(request)

    # 1) 数据面存活 + 只读探针（[B23] /recall 会 get_or_create("main") 制造幻影
    # 会话——只读诊断不应有建会话副作用；改打 /health + /landscape/main，
    # 两者均纯只读：/landscape 用 sm.get 而非 get_or_create，不创建会话）
    api_status, api_body = await _ahttp_json("GET", "/health", timeout=3.0)
    ls_t0 = time.time()
    ls_status, ls_body = await _ahttp_json("GET", "/landscape/main", timeout=6.0)
    ls_ms = round((time.time() - ls_t0) * 1000, 1)

    # 2) 各会话状态 + 快照新鲜度
    sessions = await _gather_sessions()
    stale = [
        {"session_id": s["session_id"], "snapshot_age_s": s.get("snapshot_age_s")}
        for s in sessions
        if s.get("snapshot_age_s") is not None and s["snapshot_age_s"] > 1800
    ]

    # 3) embed 连通性探针（直连 embed 服务，5s 超时；未配置则跳过）
    embed: Dict[str, Any] = {"configured_type": os.environ.get("LMS_EMBEDDER", "unknown")}
    embed_url = os.environ.get("LMS_CLOUD_EMBED_URL", "").strip()
    if embed_url:
        probe_t0 = time.time()
        try:
            # 注意：LMS_CLOUD_EMBED_URL 本身已是完整端点（含 /v1/embeddings），
            # 直接探测，勿再拼接路径（否则 404/405）。
            req = urllib.request.Request(
                embed_url,
                data=json.dumps({
                    "model": os.environ.get("LMS_CLOUD_EMBED_MODEL", "bge-m3"),
                    "input": ["lms-control-plane 连通性探针"],
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                resp.read()
            embed.update({"reachable": True,
                          "probe_ms": round((time.time() - probe_t0) * 1000, 1)})
        except Exception as e:
            embed.update({"reachable": False,
                          "probe_ms": round((time.time() - probe_t0) * 1000, 1),
                          "error": f"{type(e).__name__}: {e}"})
    else:
        embed.update({"reachable": None, "note": "LMS_CLOUD_EMBED_URL 未配置，跳过探针"})

    # 4) 进程清单 + 审计/注册文件统计
    procs = _process_info()
    audit_lines = 0
    if AUDIT_FILE.is_file():
        try:
            audit_lines = sum(1 for _ in open(AUDIT_FILE, encoding="utf-8"))
        except Exception:
            pass

    return {
        "ts": datetime.now().isoformat(),
        "latency_ms": round((time.time() - t0) * 1000, 1),
        "api": {
            "reachable": api_status == 200,
            "health": api_body.get("status") if isinstance(api_body, dict) else None,
            # [B23] recall 探针 → landscape 探针：原 POST /recall 探针会
            # get_or_create("main") 制造幻影会话（只读诊断不应有建会话副作用），
            # 改打 /landscape/main（sm.get 而非 get_or_create，纯只读）。
            # [F2] 收尾修复（审计发现 B23 计划偏差：响应字段被"替换"而非"追加"）：
            # 仓库内无 recall_probe 消费方（grep 证实），但外部控制面 :8191
            # 消费方若解析旧字段会断——保留同名字段作向后兼容占位并标注废弃，
            # 同时**追加** landscape_probe（原实现语义不变）。若外部消费方
            # 确需真值，恢复 POST /recall 探针前须先解决 get_or_create 副作用。
            "recall_probe": {
                "status": 410,
                "ok": False,
                "deprecated": True,
                "note": "已废弃（B23）：原 /recall 探针有建幻影会话副作用，"
                        "改用 landscape_probe（纯只读）",
            },
            "landscape_probe": {
                "status": ls_status,
                "duration_ms": ls_ms,
                "ok": ls_status == 200,
            },
        },
        "sessions": sessions,
        "snapshot_freshness": {
            "stale_count": len(stale),
            "stale_sessions": stale,
        },
        "embed": embed,
        "processes": procs,
        "audit": {"file": str(AUDIT_FILE), "lines": audit_lines},
        "registrations": len(_load_registrations()),
        "overrides": dict(_overrides),
        "read_only_mode": READ_ONLY_MODE,
    }


@app.on_event("startup")
async def _control_startup() -> None:
    """启动日志：模式 + 配置摘要（不打印任何密钥）。"""
    logger.info(
        f"LMS Control Plane 启动 | bind={CTRL_HOST}:{CTRL_PORT} | "
        f"api_base={LMS_API_BASE} | "
        f"mode={'READ-ONLY（LMS_CONTROL_TOKEN 未配置，写端点禁用）' if READ_ONLY_MODE else 'FULL（token 已配置）'}"
    )
    _ensure_file(AUDIT_FILE)

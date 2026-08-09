"""
活体记忆系统 - 审计日志（T2.6）
==============================

追加式 JSONL 审计：``audit(event, **fields)``。

设计约束（与总体方案 T2.6 一致）:
  - 线程安全：模块级 ``threading.Lock`` 串行化写入；
  - 按天轮转：``logs/audit-YYYYMMDD.jsonl``（文件名含日期，跨天自动切换句柄）；
  - 权限：文件以 0600 创建；
  - fail-open：任何失败仅 WARNING 日志，绝不影响主流程（纯观测，零侵入）；
  - 只追加不覆盖；每行一条事件 ``{"ts": 秒, "event": ..., ...fields}``。

事件类型（约定，见 api/server.py 与 api/session_manager.py 的接入点）:
  - startup / shutdown              服务启停
  - scheduler_start / scheduler_stop 做梦调度器启停
  - config_effective                配置生效
  - session_created / session_deleted  会话创建/删除
  - snapshot_saved                  快照保存（含被覆盖的 latest 旧 sha256）
  - snapshot_loaded / state_restored 快照加载（自动恢复）/ 手动回退
  - critical_error                  关键错误（LLM 失败、快照保存失败等）

环境变量:
    LMS_AUDIT_DIR: 审计日志目录（默认 <项目根>/logs）。
"""

import os
import json
import time
import logging
import threading
from datetime import datetime

logger = logging.getLogger("persistence.audit")

_lock = threading.Lock()
# 当前打开的（日期, 文件句柄）；跨天自动关闭旧句柄
_open_date: str = ""
_open_handle = None  # type: ignore[assignment]


def _audit_dir() -> str:
    """审计日志目录：env LMS_AUDIT_DIR 优先，默认 <项目根>/logs。"""
    d = os.environ.get("LMS_AUDIT_DIR", "").strip()
    if d:
        return d
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def _today() -> str:
    """当天日期字符串（YYYYMMDD）；测试可通过 monkeypatch 模拟跨天轮转。"""
    return datetime.now().strftime("%Y%m%d")


def _get_handle():
    """按天获取（并缓存）审计文件句柄；跨天自动切换。"""
    global _open_date, _open_handle
    today = _today()
    if _open_handle is not None and _open_date == today:
        return _open_handle
    # 日期变化：关闭旧句柄，打开新文件（按天轮转）
    if _open_handle is not None:
        try:
            _open_handle.close()
        except Exception:  # pylint: disable=broad-except
            pass
        _open_handle = None
    d = _audit_dir()
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:  # pylint: disable=broad-except
        pass
    path = os.path.join(d, "audit-{}.jsonl".format(today))
    # os.open 直接以 0600 创建（O_APPEND 追加；已有文件不改变权限）
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    _open_handle = os.fdopen(fd, "a", encoding="utf-8")
    _open_date = today
    return _open_handle


def audit(event: str, **fields) -> None:
    """写入一条审计事件（线程安全、按天轮转、fail-open）。

    参数:
        event: 事件类型（见模块 docstring 约定）。
        **fields: 事件字段。非 JSON 可序列化值（如 torch.Tensor）经
            ``default=str`` 降级为字符串，保证写入永不抛异常。
    """
    if not event or not event.strip():
        return
    record = {"ts": time.time(), "event": event}
    record.update(fields)
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _lock:
        try:
            f = _get_handle()
            f.write(line)
            f.flush()
        except Exception as e:  # pylint: disable=broad-except
            # fail-open：审计失败绝不影响主流程，仅 WARNING
            logger.warning("审计日志写入失败（忽略）: %s", e)

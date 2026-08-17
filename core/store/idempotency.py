# -*- coding: utf-8 -*-
"""core/store · 幂等键机制（M1 第一段）

规格依据：`四妹-LMS核心重写规格v2-20260817.md` §1.5 / §2.2（幂等防伪）。
根因：**60s 幂等竞态教训——"客户端超时" ≠ "服务端未写入"**。

机制要点：
  1. 写请求带幂等键（客户端自持键优先；未携带时服务端登记时生成并回传，
     客户端重试必须带**同一**键——禁止盲目重发）；
  2. 服务端按幂等键查重：已处理 → 直接返回原结果，**不重复写入**；
  3. **写成功才登记**（旧实现写前登记，失败也占用窗口 → 假 dedup_hit；
     本实现写失败不登记，重试同键可重新写入——修复 P0 竞态的另一半）；
  4. 同键并发（重试与首写同时在途）→ 同键锁串行化，只写一次；
  5. 60s 窗口（``LMS_STORE_IDEMPOTENCY_WINDOW``）——竞态防护不是永久去重，
     窗口外同键视为新写（旧结果自然过期）；
  6. 可选追加式 journal（``LMS_STORE_IDEMPOTENCY_JOURNAL``）：窗口内重启后
     重放，重试不双写（写侧默认保守：宁可拦错不轻信）。

本模块**不 import 任何 LMS 运行时模块**（纯 stdlib，依赖注入）——
可被 python 直接 import/运行；与血管（api 层）解耦。

参数 env 化（§5.3，M8 贯穿；全部 ``LMS_STORE_IDEMPOTENCY_*`` 前缀）：
  - ``LMS_STORE_IDEMPOTENCY_ENABLED``  幂等总开关（默认 1；0 = 每次直写不查重）
  - ``LMS_STORE_IDEMPOTENCY_WINDOW``   幂等窗口秒数（默认 60）
  - ``LMS_STORE_IDEMPOTENCY_JOURNAL``  追加式 journal 路径（默认空 = 纯内存）
  - ``LMS_STORE_IDEMPOTENCY_KEY_MAXLEN`` 幂等键长度上限（默认 128，防滥用）
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("core.store.idempotency")

# ---------------------------------------------------------------------------
# env 参数（运行时读取——灰度"随时可关"语义：改动即时生效，无需重启）
# ---------------------------------------------------------------------------

_ENV_ENABLED = "LMS_STORE_IDEMPOTENCY_ENABLED"
_ENV_WINDOW = "LMS_STORE_IDEMPOTENCY_WINDOW"
_ENV_JOURNAL = "LMS_STORE_IDEMPOTENCY_JOURNAL"
_ENV_KEY_MAXLEN = "LMS_STORE_IDEMPOTENCY_KEY_MAXLEN"

_DEFAULT_WINDOW = 60.0
_DEFAULT_KEY_MAXLEN = 128
#: 幂等键合法字符（防御：键是 API 输入，禁止换行/控制符污染 journal 行）
_KEY_ALLOWED_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")


def idempotency_enabled() -> bool:
    """幂等总开关（默认开：所有写操作幂等是硬要求，非验证链开关）。"""
    return os.environ.get(_ENV_ENABLED, "1") == "1"


def idempotency_window() -> float:
    """幂等窗口秒数（默认 60；60s 竞态防护窗口）。"""
    try:
        return max(1.0, float(os.environ.get(_ENV_WINDOW, "60") or 60))
    except (TypeError, ValueError):
        return _DEFAULT_WINDOW


def idempotency_journal_path() -> str:
    """journal 路径（空 = 纯内存注册表；非空 = 窗口内重启可重放）。"""
    return (os.environ.get(_ENV_JOURNAL, "") or "").strip()


def idempotency_key_maxlen() -> int:
    """幂等键长度上限（默认 128，防滥用）。"""
    try:
        return max(16, int(
            os.environ.get(_ENV_KEY_MAXLEN, "128") or 128))
    except (TypeError, ValueError):
        return _DEFAULT_KEY_MAXLEN


# ---------------------------------------------------------------------------
# 幂等键生成 / 规范化
# ---------------------------------------------------------------------------

def generate_idempotency_key() -> str:
    """服务端生成幂等键（客户端未携带时由登记动作生成，随响应回传）。

    用 UUID hex（无横线）——随机、全局唯一、且属于合法字符集。
    """
    return uuid.uuid4().hex


def fingerprint_key(*parts: Any) -> str:
    """请求指纹幂等键：``sha256(parts)``。

    用途：客户端未携带幂等键时，用请求语义指纹（session/text/llm_output/
    source/gray/semantics）作为确定性键——同 payload 重试自然命中同一键，
    兼容现有客户端（不发键）的 60s 去重行为（与旧 ``_store_dedup_key``
    同工程惯例，设计附录 B 标注）。显式键存在时**优先于**指纹（见 ingest）。
    """
    canonical = "\x00".join(
        str(p) if p is not None else "" for p in parts)
    return "fp_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_key(key: str) -> str:
    """规范化并校验幂等键（非法 → ValueError，api 层映射 422）。"""
    if not isinstance(key, str):
        raise ValueError(f"幂等键必须是字符串: {key!r}")
    k = key.strip()
    if not k:
        raise ValueError("幂等键不能为空")
    maxlen = idempotency_key_maxlen()
    if len(k) > maxlen:
        raise ValueError(
            f"幂等键超长（>{maxlen} 字符，LMS_STORE_IDEMPOTENCY_KEY_MAXLEN）")
    bad = [c for c in k if c not in _KEY_ALLOWED_CHARS]
    if bad:
        raise ValueError(
            f"幂等键含非法字符 {bad[:5]}（仅允许字母数字 _-.）")
    return k


# ---------------------------------------------------------------------------
# 幂等记录 / 注册表
# ---------------------------------------------------------------------------

@dataclass
class IdempotencyRecord:
    """一条已完成的写操作记录（窗口内缓存，供重试直接返回）。"""

    key: str                       # 规范化后的幂等键
    result: Dict[str, Any]         # 写结果（JSON 可序列化；重试原样返回）
    created_at: float              # 登记时间戳（墙钟）
    replayed: bool = False         # True = 从 journal 重放恢复（重启后命中）

    def expired(self, window: float) -> bool:
        """窗口外视为过期（竞态防护不是永久去重）。"""
        return (time.time() - self.created_at) > window


class IdempotencyRegistry:
    """线程安全的幂等键注册表（写侧默认保守：宁可拦错不轻信）。

    语义保证（claim，§5.2）：
      - **幂等**：同键重发不双写；已处理键直接返回原结果；
      - **写成功才登记**：writer 抛异常 → 不登记 → 重试同键可重写
        （修复旧实现"失败也占用窗口 → 假 dedup_hit"）；
      - **同键并发串行化**：同键请求（首写 + 超时重试同时到达）由同键锁
        互斥，后到者等待首写完成并直接命中缓存，全程只写一次；
      - 不同键互不阻塞（每键独立锁，非全局大锁——写操作可慢，不互相拖累）。

    可选 journal：登记时 append 一行 JSON；启动时重放窗口内记录
    （``replayed=True``），服务重启后窗口内重试仍不双写。
    """

    def __init__(self, window: Optional[float] = None,
                 journal_path: Optional[str] = None,
                 enabled: Optional[bool] = None) -> None:
        self._window = idempotency_window() if window is None else float(window)
        self._journal_path = (
            idempotency_journal_path() if journal_path is None
            else journal_path)
        self._enabled = idempotency_enabled() if enabled is None else bool(enabled)
        # 注册表本体 + 每键锁表（RLock 保护两表结构）
        self._lock = threading.RLock()
        self._records: Dict[str, IdempotencyRecord] = {}
        self._key_locks: Dict[str, Dict[str, Any]] = {}
        # 观测计数（G 模式：不静默——journal 写失败必须可见）
        self.journal_write_failures: int = 0
        self.replayed_records: int = 0
        if self._journal_path:
            self._replay_journal()

    # -- 查询 ---------------------------------------------------------------

    def lookup(self, key: str) -> Optional[IdempotencyRecord]:
        """按键查重（窗口内已处理 → 返回记录；未处理/过期 → None）。

        只读、无副作用（claim：供 §5.2 测试断言）。过期记录惰性清除。
        """
        if not self._enabled:
            return None
        with self._lock:
            rec = self._records.get(key)
            if rec is None:
                return None
            if rec.expired(self._window):
                self._records.pop(key, None)
                return None
            return rec

    # -- 登记 ---------------------------------------------------------------

    def register(self, key: str, result: Dict[str, Any]) -> IdempotencyRecord:
        """登记一次**已成功**的写结果（写失败绝不调用本方法——幂等语义）。

        同键重复登记（理论上同键锁下不会发生）时保留**先**登记的结果。
        """
        if not self._enabled:
            return IdempotencyRecord(key=key, result=result,
                                     created_at=time.time())
        with self._lock:
            existing = self._records.get(key)
            if existing is not None and not existing.expired(self._window):
                return existing
            rec = IdempotencyRecord(key=key, result=result,
                                    created_at=time.time())
            self._records[key] = rec
            self._append_journal(rec)
            return rec

    # -- 同键执行（查重 → 写 → 登记 原子化） ---------------------------------

    @contextmanager
    def key_lock(self, key: str):
        """同键互斥（重试与首写同时在途时串行化，防双写）。"""
        lk = self._acquire_key_lock(key)
        try:
            with lk["lock"]:
                yield
        finally:
            self._release_key_lock(key)

    def run_idempotent(
            self, key: str, writer: Callable[[], Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], bool, IdempotencyRecord]:
        """幂等执行一次写操作：查重 → （未处理）写 → 登记。

        Args:
            key: 规范化幂等键（同键 = 同一逻辑操作）。
            writer: 实际执行写的回调（返回 JSON 可序列化结果 dict；
                抛异常 = 写失败，不登记，重试可重写）。

        Returns:
            ``(result, dedup_hit, record)``：
              - ``dedup_hit=False``：本次执行了写并登记（首写）；
              - ``dedup_hit=True`` ：窗口内同键已处理，**未执行写**，
                原样返回首次结果（重试/竞态命中）。

        Raises:
            透传 writer 异常（写失败不登记；api 层映射 503/500）。
        """
        if not self._enabled:
            # 幂等总开关关闭：每次直写（逃生舱，观测用）
            return writer(), False, IdempotencyRecord(
                key=key, result={}, created_at=time.time())

        with self.key_lock(key):
            rec = self.lookup(key)
            if rec is not None:
                # 已处理（含 journal 重放记录）→ 直接返回原结果，不重复写
                logger.info(
                    "[store] 幂等命中 key=%s%s（不重复写入）",
                    key, " [journal 重放]" if rec.replayed else "")
                return rec.result, True, rec
            result = writer()
            rec = self.register(key, result)
            logger.info("[store] 幂等登记 key=%s（写成功）", key)
            return result, False, rec

    # -- journal（可选持久化，窗口内重启可重放） ------------------------------

    def _append_journal(self, rec: IdempotencyRecord) -> None:
        """登记结果追加到 journal（一行 JSON；G 模式：失败告警不静默）。"""
        if not self._journal_path:
            return
        line = json.dumps({
            "key": rec.key,
            "result": rec.result,
            "created_at": rec.created_at,
        }, ensure_ascii=False)
        try:
            with open(self._journal_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError as e:
            self.journal_write_failures += 1
            logger.error(
                "[store] 幂等 journal 写入失败（%s）：%s——窗口内重启后"
                "同键重试可能双写，请检查 %s", self._journal_path, e,
                _ENV_JOURNAL)

    def _replay_journal(self) -> None:
        """启动时重放窗口内记录（重启后重试不双写；写侧默认保守）。"""
        try:
            with open(self._journal_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return
        except OSError as e:
            logger.error("[store] 幂等 journal 读取失败（%s）：%s",
                         self._journal_path, e)
            return
        now = time.time()
        n_replayed = 0
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
                rec = IdempotencyRecord(
                    key=obj["key"],
                    result=obj.get("result", {}),
                    created_at=float(obj.get("created_at", 0.0)),
                    replayed=True,
                )
            except (ValueError, KeyError, TypeError) as e:
                logger.warning("[store] 幂等 journal 行解析失败，跳过：%s", e)
                continue
            if rec.expired(self._window) or now - rec.created_at > self._window:
                continue  # 窗口外记录不重放（竞态防护不是永久去重）
            if rec.key not in self._records:
                self._records[rec.key] = rec
                n_replayed += 1
        self.replayed_records = n_replayed
        if n_replayed:
            logger.info("[store] 幂等 journal 重放 %d 条窗口内记录", n_replayed)

    # -- 每键锁管理 ----------------------------------------------------------

    def _acquire_key_lock(self, key: str) -> Dict[str, Any]:
        with self._lock:
            lk = self._key_locks.get(key)
            if lk is None:
                lk = {"lock": threading.Lock(), "refs": 0}
                self._key_locks[key] = lk
            lk["refs"] += 1
            return lk

    def _release_key_lock(self, key: str) -> None:
        with self._lock:
            lk = self._key_locks.get(key)
            if lk is not None:
                lk["refs"] -= 1
                if lk["refs"] <= 0:
                    self._key_locks.pop(key, None)

    # -- 观测 ---------------------------------------------------------------

    def size(self) -> int:
        """当前窗口内登记数（观测/测试用）。"""
        with self._lock:
            return len(self._records)

    def clear(self) -> None:
        """清空注册表（测试用；生产不调用——窗口自动过期）。"""
        with self._lock:
            self._records.clear()

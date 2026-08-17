# -*- coding: utf-8 -*-
"""api/wiring —— M1 接线段：/store /feed 端点 → core/store 写侧统一入口

规格依据：`四妹-LMS核心重写规格v2-20260817.md` §1.5 / §4.1 / §5.2。
总纲：**核心重建，血管不换**——端点路径、请求参数名、返回结构逐字节同构，
本模块只换 /store /feed 的内部实现（改调 core/store）。

本模块职责（api 层接线，不含业务语义——语义唯一权威在 core/store）：
  1. 请求映射：``build_store_request`` / ``build_feed_request``
     → core/store 的 ``WriteRequest``（端点字段逐字段搬运，超集兼容）；
  2. writer 工厂：``make_store_writer`` / ``make_feed_writer``
     → 实际写回调（内部走 loop.process_turn + 条目 meta 后处理；
     做梦协调 + embed 熔断降级语义与旧端点逐字节等价）；
  3. 响应构建：``build_store_response`` / ``build_feed_response``
     → 与现有 StoreResponse / FeedResponse **字段集合逐字节同构**的 dict；
  4. 错误映射：``http_error_for`` / ``StoreDegradedError``
     → 写侧异常 → 端点 HTTP 语义（422 / 503 / degraded 响应体）；
  5. 幂等注册表：``store_ingest`` / ``feed_ingest``（core/store.ingest
     薄封装，注册表可注入——测试隔离用）。

设计约束：
  - **不 import fastapi / torch / 生产 api.*（extract_core 等）**——
    模块级只 import stdlib + core.store；生产依赖（extract_core /
    value_filter / segment_reply / loop / scheduler）一律在 writer
    内部**惰性导入或注入**。因此本模块可被轻量单测直接 import/运行
    （rewrite-M1/tests/test_wiring_m1.py，纯 stdlib，fake loop）。
  - core/store 的语义权威（source/gray 解析、幂等、feed/store 分离）
    本模块只透传，不重复实现（单一权威，防默认值三处各写各的——§4.5）。

决策登记（2026-08-17，全部带理由，详见 README-接线方案.md）：
  D-1  端点不新增 idempotency_key 请求字段（§4.1 请求参数冻结）；未携带
       键 → core/store 请求指纹兜底（sha256 语义，含 session/source/gray/
       feed_channel——同 payload 重试自然命中同一键，兼容现客户端）。
  D-2  dedup 命中响应返回**首次写结果**（core/store claim"已处理键原样
       返回原结果"），不再合成空值响应（旧实现 dedup 命中返回
       stored=True/core_chars=0/surprise=0 的合成体）。
  D-3  /feed 的 ``FeedRequest.source`` 保持**观测字段**（与现状一致：
       现有 /feed 只把 source 用于日志；落库 source 由 loop 硬编码
       'external'）。端点不透传 req.source：透传 'event_bus' 会改变落库
       source，而 L1 检索 source_filter='external' 将排除非 external 来源
       （feed 语义要求可召回——坑 6 口径）。resolved_source='external'
       与 loop 落库同值，语义权威与落库一致。
  D-4  /feed 幂等化（新保证，规格 §1.5"所有写操作幂等"）：同 payload 60s
       内重发不重复塑形——"客户端超时≠未写入"同款保证。
  D-5  做梦协调（acquire/release）移入 writer：dedup 命中路径零写入、
       零协调（命中不 acquire——与旧实现 dedup 提前 return 语义一致）。
  D-6  响应结构逐字节同构：StoreResponse 10 字段 / FeedResponse 7 字段；
       幂等键区段（idempotency_key/replayed）只进 WriteResult（内部），
       不进端点响应（§4.1 响应结构冻结）。

claim 登记（§5.2 machine-readable，D 模式：测试即机器验证）：
  - endpoint_contract_frozen ：端点返回字段集合与 StoreResponse/
    FeedResponse 逐字节同构 —— verified_by test_wiring_m1.py::
    test_store_response_isomorphic / test_feed_response_isomorphic；
  - store_goes_through_core_store：/store 写侧经 core/store 统一入口
    （幂等/语义分离生效）—— verified_by test_store_path_uses_core_store；
  - idempotent_key_effective：同 payload 重发不双写、dedup_hit 置位、
    writer 只调一次 —— verified_by test_store_idempotent_resend；
  - feed_semantics_enforced：feed 永不灰化 + source 可召回（观测字段
    不透传，D-3）—— verified_by test_feed_semantics_kept；
  - degraded_maps_503：embed 熔断降级 → 503 + degraded 响应体（P3 语义
    保留）—— verified_by test_store_degraded_maps_503。
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

from core.store import (  # noqa: E402  （rewrite-M1 或生产仓库的 core/store）
    FeedChannel,
    IdempotencyRegistry,
    SemanticViolation,
    WriteRequest,
    WriteResult,
    WriteSemantics,
    ingest as core_ingest,
)

logger = logging.getLogger("api.wiring")

# ---------------------------------------------------------------------------
# 端点响应字段集合（单一权威：端点响应结构冻结的机器可验常量，§4.1）
# ---------------------------------------------------------------------------

#: /store 端点响应字段（与 StoreResponse 逐字节同构——血管不换）
STORE_RESPONSE_FIELDS: Tuple[str, ...] = (
    "session_id", "turn_count", "stored", "dedup_hit", "value_filtered",
    "core_chars", "gray", "surprise", "info_value", "reason",
)
#: /feed 端点响应字段（与 FeedResponse 逐字节同构——血管不换）
FEED_RESPONSE_FIELDS: Tuple[str, ...] = (
    "status", "entropy", "surprise", "free_energy", "mse", "turn_count",
    "session_id",
)

# ---------------------------------------------------------------------------
# 写侧异常（端点层映射；与旧端点 HTTP 语义逐字节等价）
# ---------------------------------------------------------------------------


class WriteBusyError(Exception):
    """做梦协调冲突：写侧被做梦占用 → 503"系统正在做梦"。

    与旧端点 `scheduler.acquire_conversation() 失败 → 503` 语义等价。
    """

    def __init__(self, session_id: str) -> None:
        super().__init__(f"会话 '{session_id}' 正在做梦（记忆巩固中）")
        self.session_id = session_id


class StoreDegradedError(Exception):
    """写侧 embed 熔断降级 → 503 + degraded 响应体（P3 语义保留）。

    reason: "embed_unavailable"（process_turn 抛异常）/
            "embed_circuit_open"（降级但未落库——size 未增长）。
    """

    def __init__(self, reason: str, turn: int) -> None:
        super().__init__(f"store 写侧 embed 降级（{reason}）")
        self.reason = reason
        self.turn = turn


def http_error_for(exc: Exception) -> Optional[Tuple[int, str, Optional[Dict]]]:
    """写侧异常 → ``(status, detail, headers)`` 映射（端点层用）。

    返回 None = 非预期异常（端点按默认 500 处理，G 模式：不静默）。
    ``StoreDegradedError`` 不走本映射（响应体为 degraded JSON，端点
    单独构造——与旧端点 JSONResponse 语义一致）。
    """
    if isinstance(exc, SemanticViolation):
        return (422, str(exc), None)
    if isinstance(exc, WriteBusyError):
        return (503, "系统正在做梦（记忆巩固中），请稍后重试。", None)
    return None


# ---------------------------------------------------------------------------
# 幂等注册表（进程内单例；测试可重置/注入隔离实例）
# ---------------------------------------------------------------------------

_default_registry: Optional[IdempotencyRegistry] = None
_registry_lock = threading.Lock()


def get_default_registry() -> IdempotencyRegistry:
    """写侧默认幂等注册表（懒创建；/store 与 /feed 共用——键按语义
    fingerprint 命名空间隔离，无碰撞）。"""
    global _default_registry
    if _default_registry is None:
        with _registry_lock:
            if _default_registry is None:
                _default_registry = IdempotencyRegistry()
    return _default_registry


def reset_default_registry() -> None:
    """清空默认注册表（测试隔离用：下次调用重建）。"""
    global _default_registry
    with _registry_lock:
        _default_registry = None


def store_ingest(write_req: WriteRequest, writer: Callable,
                 registry: Optional[IdempotencyRegistry] = None) -> WriteResult:
    """/store 端点写侧统一入口：core/store.ingest 薄封装（注册表可注入）。"""
    return core_ingest(write_req, writer,
                       registry=registry or get_default_registry())


def feed_ingest(write_req: WriteRequest, writer: Callable,
                registry: Optional[IdempotencyRegistry] = None) -> WriteResult:
    """/feed 端点写侧统一入口：core/store.ingest 薄封装（注册表可注入）。"""
    return core_ingest(write_req, writer,
                       registry=registry or get_default_registry())


# ---------------------------------------------------------------------------
# 请求映射（端点字段逐字段搬运 → core/store WriteRequest）
# ---------------------------------------------------------------------------


def build_store_request(req: Any) -> WriteRequest:
    """/store 端点请求 → WriteRequest（语义=STORE）。

    搬运字段：user_input→text、session_id、llm_output。
    ``source``：StoreRequest 无该字段（§4.1 冻结）→ None（语义层推导
    'external' / 灰度时 'store_gray'）。
    ``idempotency_key``：端点无该字段（D-1）→ None（请求指纹兜底）。
    ``gray``：None → 语义层读 env ``LMS_STORE_GRAY``（灰度随时可关）。
    """
    return WriteRequest(
        semantics=WriteSemantics.STORE,
        text=req.user_input,
        session_id=req.session_id,
        source=None,               # StoreRequest 无 source 字段（血管冻结）
        llm_output=req.llm_output,
        idempotency_key=None,      # D-1：请求指纹兜底
        gray=None,                 # None → env LMS_STORE_GRAY（语义层权威）
        metadata={"endpoint": "/store"},
    )


def build_feed_request(req: Any) -> WriteRequest:
    """/feed 端点请求 → WriteRequest（语义=FEED）。

    搬运字段：text、session_id、llm_output。
    ``source``：**不透传**（D-3 决策登记）——FeedRequest.source 保持
    观测字段（metadata.source_hint），落库 source 由语义层推导
    'external'（与 loop 硬编码 source='external' 同值，L1 可召回）。
    ``gray``：强制 False（feed 永不灰化——语义硬约束，resolve 再校验）。
    """
    return WriteRequest(
        semantics=WriteSemantics.FEED,
        text=req.text,
        session_id=req.session_id,
        source=None,               # D-3：不透传（观测字段）
        llm_output=req.llm_output,
        idempotency_key=None,      # D-1：请求指纹兜底（D-4 feed 幂等化）
        gray=False,                # feed 永不灰化（语义硬约束）
        feed_channel=FeedChannel.PRIMARY,
        metadata={
            "endpoint": "/feed",
            "source_hint": getattr(req, "source", None),  # 仅观测
            "sender": getattr(req, "sender", ""),          # 仅观测
        },
    )


# ---------------------------------------------------------------------------
# writer 工厂（实际写回调；生产依赖惰性导入 / 测试注入）
# ---------------------------------------------------------------------------


def _lazy_extract_core(llm_output: str) -> str:
    """生产默认：api/extract_core.extract_core（≤300 字两段式提取）。

    保留为兼容层（方案 D：不替换）——core/store 只管写侧编排，不做
    文本提取（职责分离，见 README-接线方案.md §2）。
    """
    from api.extract_core import extract_core  # 惰性：生产 api 包
    return extract_core(llm_output)


def _lazy_segment_reply(text: str) -> list:
    """生产默认：api/segment_reply.segment_reply（分段标记，观测用）。"""
    from api.segment_reply import segment_reply  # 惰性：生产 api 包
    return segment_reply(text)


def _lazy_compute_info_value(**kwargs) -> float:
    """生产默认：api/value_filter.compute_info_value（价值分数）。"""
    from api.value_filter import compute_info_value  # 惰性：生产 api 包
    return compute_info_value(**kwargs)


def _lazy_value_filtered(info_value: float) -> bool:
    """生产默认：api/value_filter.value_filtered（价值过滤标记）。"""
    from api.value_filter import value_filtered  # 惰性：生产 api 包
    return value_filtered(info_value)


def make_store_writer(
    loop: Any,
    scheduler: Any = None,
    session_id: Optional[str] = None,
    extract_core_fn: Optional[Callable[[str], str]] = None,
    segment_reply_fn: Optional[Callable[[str], list]] = None,
    compute_info_value_fn: Optional[Callable[..., float]] = None,
    value_filtered_fn: Optional[Callable[[float], bool]] = None,
) -> Callable[[WriteRequest], Dict[str, Any]]:
    """/store 写回调（核心 ≤300 字塑形 + 条目 meta 后处理）。

    与旧端点内部流程**逐字节等价**（§7.1 流程 3-6）：
      1. 提取核心 ≤300 字（extract_core）＋分段标记（segment_reply，观测）；
      2. 做梦协调（acquire/release；失败 → ``WriteBusyError`` → 503）——
         D-5：dedup 命中不触发 writer，故不协调；
      3. process_turn(user_input, 核心)（embed 熔断：抛异常 →
         ``StoreDegradedError('embed_unavailable')``；降级未落库 →
         ``StoreDegradedError('embed_circuit_open')``）——P3 语义保留；
      4. 新条目 meta：core/ts/gray/source（= 语义层 resolved_source，
         写侧唯一权威）／info_value（fail-open）；
      5. 返回与旧端点字段同构的 dict（stored/turn_count/value_filtered/
         core_chars/gray/surprise/info_value/reason + source 观测）。

    参数均可注入（轻量单测传 stub；生产走默认惰性导入）。
    """
    extract_core_fn = extract_core_fn or _lazy_extract_core
    segment_reply_fn = segment_reply_fn or _lazy_segment_reply
    compute_info_value_fn = compute_info_value_fn or _lazy_compute_info_value
    value_filtered_fn = value_filtered_fn or _lazy_value_filtered

    def _write(req: WriteRequest) -> Dict[str, Any]:
        core = extract_core_fn(req.llm_output)
        try:
            segments = segment_reply_fn(core or req.llm_output)
        except Exception:  # pylint: disable=broad-except
            segments = []
        if segments and logger.isEnabledFor(logging.DEBUG):
            logger.debug("[store] 分段标记 %d 段（观测）", len(segments))

        epi_before = loop.memory.episodic_size()
        try:
            loop.process_turn(req.text, core)
        except Exception as e:  # pylint: disable=broad-except
            # 旧端点：process_turn 抛异常 → 503 embed_unavailable
            raise StoreDegradedError(
                "embed_unavailable", turn=getattr(loop, "turn_count", 0)) from e

        # 旧端点：降级但未落库（size 未增长 + last_turn_degraded）→
        # 503 embed_circuit_open（不落僵尸：无向量条目=检索不可达）
        if (loop.memory.episodic_size() == epi_before
                and getattr(loop, "last_turn_degraded", False)):
            raise StoreDegradedError(
                "embed_circuit_open", turn=getattr(loop, "turn_count", 0))

        # 后处理新条目 meta（S1-11 字段：core/info_value/ts/gray/source）
        entries = list(loop.memory.iter_episodic())
        new_entry = entries[-1] if entries else None
        stored = False
        reason = None
        info_value = 0.0
        gray = bool(req.resolved_gray)
        if new_entry is not None and loop.memory.episodic_size() > epi_before:
            new_entry.core = core or None
            new_entry.ts = time.time()
            if gray:
                new_entry.gray = True
            # 写侧唯一权威：语义层 resolved_source（store 非灰='external'，
            # 灰度='store_gray'——与旧端点逐值等价）
            new_entry.source = req.resolved_source
            try:
                surprises = [float(getattr(e, "surprise", 0.0) or 0.0)
                             for e in entries]
                refs = [float(getattr(e, "reference_count", 0) or 0)
                        for e in entries]
                info_value = float(compute_info_value_fn(
                    surprise=float(getattr(new_entry, "surprise", 0.0) or 0.0),
                    reference_count=int(
                        getattr(new_entry, "reference_count", 0) or 0),
                    surprise_max=max(surprises) if surprises else None,
                    ref_count_max=max(refs) if refs else None,
                    text=core or req.llm_output,
                ) or 0.0)
                new_entry.info_value = info_value
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("[store] info_value 计算失败（fail-open）: %s", e)
            stored = True
        else:
            stored = False
            reason = ("filtered_garbage" if not getattr(
                loop, "last_turn_degraded", False) else "embed_circuit_open")

        vf = True
        try:
            vf = bool(value_filtered_fn(info_value))
        except Exception:  # pylint: disable=broad-except
            pass

        status = loop.get_status()
        return {
            "stored": stored,
            "turn_count": int(getattr(loop, "turn_count", 0) or 0),
            "value_filtered": vf,
            "core_chars": len(core or ""),
            "gray": bool(gray and stored),
            "surprise": float(status.get("last_surprise", 0.0) or 0.0),
            "info_value": round(float(info_value or 0.0), 4),
            "reason": reason,
            # 观测：语义层解析口径（写侧唯一权威；不进端点响应）
            "source": req.resolved_source,
        }

    def writer(req: WriteRequest) -> Dict[str, Any]:
        sid = session_id or req.session_id
        if scheduler is None:
            # 无调度器（轻量测试 / 未接做梦）：直接写
            return _write(req)
        # D-5：做梦协调（与旧端点 register→acquire→touch 顺序逐字节一致）
        scheduler.register_session(sid)
        acquired = scheduler.acquire_conversation(sid)
        scheduler.touch(sid)
        if not acquired:
            raise WriteBusyError(sid)
        try:
            return _write(req)
        finally:
            scheduler.release_conversation(sid)

    return writer


def make_feed_writer(
    loop: Any,
    scheduler: Any = None,
    session_id: Optional[str] = None,
) -> Callable[[WriteRequest], Dict[str, Any]]:
    """/feed 写回调（塑形但不产 LLM 回复）。

    与旧端点逐字节等价：process_turn(text, llm_output) 后返回状态
    摘要（FeedResponse 字段）；**不**后处理条目（D-3：落库 source 由
    loop 硬编码 'external'，与语义层 resolved_source 同值）。
    做梦协调（acquire/release）同 /store（D-5）。
    """
    def _write(req: WriteRequest) -> Dict[str, Any]:
        loop.process_turn(req.text, llm_output=req.llm_output)
        status = loop.get_status()
        return {
            "status": "ok",
            "entropy": float(status.get("last_entropy", 0.0) or 0.0),
            "surprise": float(status.get("last_surprise", 0.0) or 0.0),
            "free_energy": float(status.get("last_free_energy", 0.0) or 0.0),
            "mse": float(status.get("last_mse", 0.0) or 0.0),
            "turn_count": int(status.get("turn_count", 0) or 0),
            "session_id": req.session_id,
            # 观测：语义层解析口径（feed 永不灰化 + source 可召回）
            "gray": bool(req.resolved_gray),
            "source": req.resolved_source,
        }

    def writer(req: WriteRequest) -> Dict[str, Any]:
        sid = session_id or req.session_id
        if scheduler is None:
            return _write(req)
        scheduler.register_session(sid)
        acquired = scheduler.acquire_conversation(sid)
        scheduler.touch(sid)
        if not acquired:
            raise WriteBusyError(sid)
        try:
            return _write(req)
        finally:
            scheduler.release_conversation(sid)

    return writer


# ---------------------------------------------------------------------------
# 响应构建（与现有 StoreResponse / FeedResponse 字段集合逐字节同构）
# ---------------------------------------------------------------------------


def build_store_response(result: WriteResult, loop: Any,
                         req: Any) -> Dict[str, Any]:
    """WriteResult → /store 端点响应 dict（10 字段，与 StoreResponse 同构）。

    dedup 命中（D-2）：返回首次写结果（result.data 缓存）——core/store
    claim"已处理键原样返回原结果"；turn_count 用首次结果（内容写入于
    该轮），非当前轮次。
    """
    data = result.data
    return {
        "session_id": req.session_id,
        "turn_count": int(data.get("turn_count",
                                   getattr(loop, "turn_count", 0)) or 0),
        "stored": bool(data.get("stored", False)),
        "dedup_hit": bool(result.dedup_hit),
        "value_filtered": bool(data.get("value_filtered", True)),
        "core_chars": int(data.get("core_chars", 0) or 0),
        "gray": bool(data.get("gray", False)),
        "surprise": float(data.get("surprise", 0.0) or 0.0),
        "info_value": float(data.get("info_value", 0.0) or 0.0),
        "reason": data.get("reason"),
    }


def build_feed_response(result: WriteResult, req: Any) -> Dict[str, Any]:
    """WriteResult → /feed 端点响应 dict（7 字段，与 FeedResponse 同构）。"""
    data = result.data
    return {
        "status": str(data.get("status", "ok")),
        "entropy": float(data.get("entropy", 0.0) or 0.0),
        "surprise": float(data.get("surprise", 0.0) or 0.0),
        "free_energy": float(data.get("free_energy", 0.0) or 0.0),
        "mse": float(data.get("mse", 0.0) or 0.0),
        "turn_count": int(data.get("turn_count", 0) or 0),
        "session_id": req.session_id,
    }


# ---------------------------------------------------------------------------
# claim 登记（§5.2 machine-readable；与 tests/test_wiring_m1.py 一一对应）
# ---------------------------------------------------------------------------

MODULE_CLAIMS: Dict[str, Any] = {
    "module": "api/wiring",
    "milestone": "M1-2",
    "rewrite_spec": "四妹-LMS核心重写规格v2-20260817.md §1.5 / §4.1 / §5.2",
    "claims": {
        "endpoint_contract_frozen": {
            "statement": "端点返回字段集合与 StoreResponse（10 字段）/ "
                         "FeedResponse（7 字段）逐字节同构——血管不换",
            "verified_by": "tests/test_wiring_m1.py::"
                           "test_store_response_isomorphic / "
                           "test_feed_response_isomorphic",
        },
        "store_goes_through_core_store": {
            "statement": "/store 写侧经 core/store 统一入口（语义分离 + "
                         "幂等键机制生效，非旧 _store_dedup 路径）",
            "verified_by": "tests/test_wiring_m1.py::"
                           "test_store_path_uses_core_store",
        },
        "idempotent_key_effective": {
            "statement": "同 payload 重发不双写：writer 只调一次，二次 "
                         "dedup_hit=True 且原样返回首次结果（D-2）",
            "verified_by": "tests/test_wiring_m1.py::"
                           "test_store_idempotent_resend",
        },
        "feed_semantics_enforced": {
            "statement": "feed 永不灰化 + source 可召回；FeedRequest.source "
                         "保持观测字段不透传（D-3，L1 source_filter 契约）",
            "verified_by": "tests/test_wiring_m1.py::test_feed_semantics_kept",
        },
        "degraded_maps_503": {
            "statement": "写侧 embed 熔断降级（embed_unavailable / "
                         "embed_circuit_open）→ StoreDegradedError → 端点 "
                         "503 + degraded 响应体 + Retry-After（P3 语义保留）",
            "verified_by": "tests/test_wiring_m1.py::"
                           "test_store_degraded_maps_503",
        },
        "write_busy_maps_503": {
            "statement": "做梦协调冲突 → WriteBusyError → 端点 503"
                         "（与旧端点 acquire 失败语义逐字节等价）",
            "verified_by": "tests/test_wiring_m1.py::test_busy_maps_503",
        },
    },
}

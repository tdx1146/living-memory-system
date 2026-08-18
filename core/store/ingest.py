# -*- coding: utf-8 -*-
"""core/store · 写侧统一入口（M1 第一段：ingest/feed/store 语义分离）

规格依据：`四妹-LMS核心重写规格v2-20260817.md` §1.5（store 提取层）。
本模块是**写侧唯一入口**（api 层的 /store /feed 端点、runtime/loop.py 的
写侧路径，最终都经此进入；血管端点签名不变，只换内部实现——核心重建，血管不换）。

职责：
  1. **语义分离**：``ingest()`` 统一入口按 ``WriteSemantics`` 分派
     （``feed()`` / ``store()`` 两个语义化便捷入口，禁止绕过语义层直写）；
  2. **幂等键机制**：查重 → 写 → 登记（写成功才登记）；重试同键不双写；
     客户端超时 ≠ 未写入（60s 竞态防护，见 idempotency.py）；
  3. **注入时怀疑入口**（§1.5：内建在 store 层，不在 api 层）：
     本段只建**钩子**（``register_doubt_hook``），M3 接入三时相怀疑的
     注入时相（高 surprise / rebuttal 命中 → 标 suspect 进验证链）；
  4. **写侧默认保守**（坑 3）：语义违规（feed 灰化 / 不可召回 source）→
     直接拒绝（``SemanticViolation`` → api 层 422），不落库。

依赖注入设计（本模块不 import runtime/loop、core/hippocampus.memory——
纯 stdlib，可独立 import/运行）：
  - ``writer``：调用方注入的实际写回调（api 层闭包，内部走
    ``loop.process_turn`` + 条目 meta 后处理）。签名：``(resolved_req) -> dict``，
    返回 JSON 可序列化结果（字段与现有 StoreResponse 同构：stored /
    turn_count / surprise / info_value / reason / core_chars / value_filtered /
    gray 等，血管不换）。
  - 落库语义（source/gray）由本层解析到 ``req.resolved_source`` /
    ``req.resolved_gray`` 后交给 writer，保证口径唯一权威（semantics.py）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .idempotency import (
    IdempotencyRecord,
    IdempotencyRegistry,
    fingerprint_key,
    generate_idempotency_key,
    normalize_key,
)
from .process_core import (
    PROCESS_CORE_FIELDS,
    attach_entry_fields,
    build_text_snapshot,
    empty_process_core,
    extract_process_core_for_request,
    init_evolution,
    make_transition,
    record_transition,
)
from .semantics import (
    FeedChannel,
    SemanticViolation,
    WriteSemantics,
    resolve_source,
    store_gray_default,
    validate_semantics,
)

logger = logging.getLogger("core.store.ingest")

# ---------------------------------------------------------------------------
# 请求 / 结果模型（与现有 API 端点签名兼容：字段为超集，血管不换）
# ---------------------------------------------------------------------------

#: writer 回调：接收已解析请求，返回 JSON 可序列化结果 dict
Writer = Callable[["WriteRequest"], Dict[str, Any]]
#: 注入时怀疑钩子：M3 接入（fail-open，绝不阻断写侧）
DoubtHook = Callable[["WriteRequest", "WriteResult"], None]


@dataclass
class WriteRequest:
    """写侧统一入口请求（语义分离的载体）。

    字段设计为现有端点请求的超集（api 层映射时逐字段搬运）：
      - ``semantics``：feed / store 语义（写侧唯一权威分派依据）；
      - ``idempotency_key``：客户端自持幂等键（可选）。未携带时服务端
        登记时生成/回传（见 WriteResult.idempotency_key），重试必须带同一键；
      - ``gray``：store 灰度（None → 读 env ``LMS_STORE_GRAY`` 默认）；
        feed 语义下强制 False（语义违规直接拒绝）；
      - ``source``：条目来源（None → 按语义推导默认值；feed 禁止不可召回源）；
      - ``process_core``：P1-1 调用方预提取的过程核心（可选）。None → 本层
        按"提取过程核心优先"从 text/llm_output/metadata 启发式提取。
    """

    semantics: WriteSemantics
    text: str
    session_id: str = "main"
    source: Optional[str] = None
    llm_output: str = ""
    idempotency_key: Optional[str] = None
    gray: Optional[bool] = None
    feed_channel: FeedChannel = FeedChannel.PRIMARY
    #: 观测透传字段（sender 等，仅日志/观测用，不参与幂等语义）
    metadata: Dict[str, Any] = field(default_factory=dict)
    #: P1-1 调用方预提取的过程核心（可选；None → ingest 时本层提取）
    process_core: Optional[Dict[str, Any]] = None

    # -- 本层解析结果（writer 必须采用；只读，勿手动覆盖） -------------------
    resolved_source: Optional[str] = None
    resolved_gray: bool = False
    #: P1-1 过程核心解析结果（提取过程核心优先；见 process_core.py）
    resolved_process_core: Optional[Dict[str, Any]] = None
    resolved_text_snapshot: Optional[str] = None
    resolved_evolution: Optional[Dict[str, Any]] = None

    def resolve(self) -> "WriteRequest":
        """语义解析：校验 + 推导 source/gray（幂等；可重复调用）。

        解析失败抛 ``SemanticViolation``（api 层映射 422）。解析成功
        后 ``resolved_source`` / ``resolved_gray`` 即写侧唯一权威口径。
        """
        if not self.text or not self.text.strip():
            raise SemanticViolation("text 不能为空")
        gray = bool(self.gray) if self.gray is not None \
            else (False if self.semantics is WriteSemantics.FEED
                  else store_gray_default())
        validate_semantics(self.semantics, self.source, gray)
        self.resolved_source = resolve_source(
            self.semantics, self.source, gray)
        self.resolved_gray = gray
        return self


@dataclass
class WriteResult:
    """写侧统一入口结果（字段与现有端点响应同构 + 幂等键区段）。

    血管兼容：现有 StoreResponse/FeedResponse 字段全部保留
    （session_id / turn_count / stored / dedup_hit / value_filtered /
    core_chars / gray / surprise / info_value / reason）；新增
    ``idempotency_key`` 与 ``replayed`` 为**独立追加区段**（§4.2 语义：
    只允许追加在既有结构之后，不得改动既有字段语义）。

    P1-1 追加区段（§3.2，独立追加不改既有字段）：``process_core`` /
    ``text_snapshot`` / ``evolution``——提取过程核心优先的产物，供
    writer（api 层落条目）与观测消费。
    """

    ok: bool = False
    stored: bool = False
    dedup_hit: bool = False
    idempotency_key: str = ""
    semantics: WriteSemantics = WriteSemantics.STORE
    session_id: str = "main"
    turn_count: int = 0
    value_filtered: bool = True
    core_chars: int = 0
    gray: bool = False
    surprise: float = 0.0
    info_value: float = 0.0
    reason: Optional[str] = None
    replayed: bool = False
    #: P1-1 过程核心（提取过程核心优先产物；写侧解析结果，见 process_core.py）
    process_core: Optional[Dict[str, Any]] = None
    #: P1-1 派生视图（从过程核心重建的当前结论快照）
    text_snapshot: Optional[str] = None
    #: P1-1 演化史（append-only 状态转移）
    evolution: Optional[Dict[str, Any]] = None
    #: writer 原始结果 dict（api 层可直接搬运响应字段，血管不换）
    data: Dict[str, Any] = field(default_factory=dict)

    def to_response(self) -> Dict[str, Any]:
        """转端点响应 dict（与现有响应同构；追加区段依次排在末尾）。"""
        return {
            "session_id": self.session_id,
            "turn_count": self.turn_count,
            "stored": self.stored,
            "dedup_hit": self.dedup_hit,
            "value_filtered": self.value_filtered,
            "core_chars": self.core_chars,
            "gray": self.gray,
            "surprise": self.surprise,
            "info_value": self.info_value,
            "reason": self.reason,
            # -- 幂等键区段（独立追加，不改既有字段语义） --
            "idempotency_key": self.idempotency_key,
            "replayed": self.replayed,
            # -- P1-1 过程核心区段（独立追加，不改既有字段语义） --
            "process_core": self.process_core,
            "text_snapshot": self.text_snapshot,
            "evolution": self.evolution,
        }


# ---------------------------------------------------------------------------
# 模块级单例 / 钩子注册（注入时怀疑入口——M3 接入点）
# ---------------------------------------------------------------------------

#: 默认注册表（api 层可注入自定义注册表；进程内单例即可满足 60s 竞态防护）
_default_registry: Optional[IdempotencyRegistry] = None
_default_registry_lock = __import__("threading").Lock()

#: 注入时怀疑钩子列表（M3 三时相怀疑的注入时相在此挂载；fail-open）
_doubt_hooks: List[DoubtHook] = []
_doubt_hooks_lock = __import__("threading").Lock()


def get_default_registry() -> IdempotencyRegistry:
    """进程内默认幂等注册表（懒创建；env 参数运行时生效）。"""
    global _default_registry
    with _default_registry_lock:
        if _default_registry is None:
            _default_registry = IdempotencyRegistry()
        return _default_registry


def register_doubt_hook(hook: DoubtHook) -> None:
    """注册注入时怀疑钩子（M3 接入点；本段只建入口不建逻辑）。

    hook 签名：``(resolved_request, write_result) -> None``。
    仅对**实际写入**的请求触发（dedup_hit 命中不重复触发）。
    """
    with _doubt_hooks_lock:
        _doubt_hooks.append(hook)


def _run_doubt_hooks(req: "WriteRequest", res: "WriteResult") -> None:
    """触发注入时怀疑钩子（fail-open：怀疑逻辑异常绝不阻断写侧——G 模式
    以日志可见，不以静默吞掉）。"""
    if not _doubt_hooks:
        return
    with _doubt_hooks_lock:
        hooks = list(_doubt_hooks)
    for hook in hooks:
        try:
            hook(req, res)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("[store] 注入时怀疑钩子异常（fail-open）：%s", e)


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

def _resolve_idempotency_key(req: WriteRequest) -> str:
    """幂等键解析（客户端自持键优先；未携带 → 服务端生成并回传）。

    重试语义：客户端重试**必须**携带首次响应回传的同一幂等键
    （禁止盲目重发——盲目重发 = 换键 = 服务端视为新写）。
    未携带键的请求以请求指纹（sha256 语义）兜底——兼容现有不发键客户端
    的同 payload 60s 去重行为（旧 ``_store_dedup_key`` 同工程惯例）。
    """
    if req.idempotency_key is not None:
        return normalize_key(req.idempotency_key)
    # 指纹兜底：semantics + session + text + llm_output + source + gray
    # （gray 已解析；feed_channel 参与指纹：不同通道 = 不同逻辑写）
    return fingerprint_key(
        req.semantics.value, req.session_id, req.text, req.llm_output,
        req.resolved_source, req.resolved_gray, req.feed_channel.value)


def _build_result(req: WriteRequest, data: Dict[str, Any],
                  key: str, dedup_hit: bool = False,
                  record: Optional[IdempotencyRecord] = None) -> WriteResult:
    """把 writer 结果包装为 WriteResult（字段搬运，血管不换）。"""
    return WriteResult(
        ok=bool(data.get("stored", False) or data.get("ok", False)),
        stored=bool(data.get("stored", False)),
        dedup_hit=dedup_hit,
        idempotency_key=key,
        semantics=req.semantics,
        session_id=req.session_id,
        turn_count=int(data.get("turn_count", 0) or 0),
        value_filtered=bool(data.get("value_filtered", True)),
        core_chars=int(data.get("core_chars", 0) or 0),
        gray=bool(data.get("gray", req.resolved_gray)),
        surprise=float(data.get("surprise", 0.0) or 0.0),
        info_value=float(data.get("info_value", 0.0) or 0.0),
        reason=data.get("reason"),
        replayed=bool(record.replayed) if record else False,
        # P1-1：过程核心区段（写侧解析结果；dedup_hit 路径同样携带）
        process_core=req.resolved_process_core,
        text_snapshot=req.resolved_text_snapshot,
        evolution=req.resolved_evolution,
        data=dict(data),
    )


def _resolve_process_core(req: WriteRequest) -> None:
    """P1-1：提取过程核心优先（§3.2）——写侧统一入口内建，fail-open。

    优先级：调用方预提取（``req.process_core``）> 本层从 text/llm_output/
    metadata 启发式提取（process_core.extract_process_core_for_request）。
    解析结果写入 ``req.resolved_*`` 只读字段，供 writer（api 层落条目）与
    WriteResult（观测）共同消费；演化史以 ``created`` 转移为起点
    （append-only 演化史的可审计开端）。全程 fail-open：提取/解析异常 →
    空过程核心 + 文本回退快照，绝不阻断写侧（G 模式以日志可见）。
    """
    try:
        if isinstance(req.process_core, dict):
            pc = {f: list(req.process_core.get(f, []) or [])
                  for f in PROCESS_CORE_FIELDS}
        else:
            pc = extract_process_core_for_request(req)
        req.resolved_process_core = pc
        req.resolved_text_snapshot = build_text_snapshot(
            pc, text=req.text, core=req.llm_output)
        ev = init_evolution(updated_by="ingest")
        record_transition(ev, make_transition("created", detail="ingest"))
        req.resolved_evolution = ev
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("过程核心解析失败（fail-open→空结构）：%s", e)
        req.resolved_process_core = empty_process_core()
        req.resolved_text_snapshot = build_text_snapshot(
            None, text=req.text, core=req.llm_output)
        req.resolved_evolution = init_evolution(updated_by="ingest")


def _attach_process_core_to_entry(req: WriteRequest, data: Dict[str, Any]) -> None:
    """把 P1-1 新字段附加到 writer 报告的新条目（严格增量，fail-open）。

    writer 结果 dict 若携带 ``entry`` 键（待集成点：api 层 writer 把新建
    条目放进结果），本层把 process_core/text_snapshot/evolution 附加到
    该条目并**弹出**该键——entry 对象含 torch 张量，非 JSON 可序列化，
    不允许进入幂等记录/journal/响应（防 journal 序列化炸裂）。附加失败
    仅告警（条目仍在写侧缓冲区，只是缺过程字段），绝不阻断写侧。
    """
    if not isinstance(data, dict) or "entry" not in data:
        return
    entry = data.pop("entry")
    try:
        attach_entry_fields(
            entry,
            process_core=req.resolved_process_core,
            text_snapshot=req.resolved_text_snapshot,
            evolution=req.resolved_evolution,
        )
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(
            "过程核心附加条目失败（fail-open，条目已写但缺过程字段）：%s", e)


def ingest(req: WriteRequest, writer: Writer,
           registry: Optional[IdempotencyRegistry] = None) -> WriteResult:
    """写侧统一入口：语义校验 → 过程核心解析 → 幂等查重 → 写 → 登记 → 怀疑钩子。

    Args:
        req: 写请求（semantics 决定 feed/store 语义分离）。
        writer: 实际写回调（api 层注入；内部走 loop.process_turn + 条目
            meta 后处理；返回 JSON 可序列化结果 dict）。writer 结果若带
            ``entry`` 键，本层把 P1-1 过程字段附加到该条目（待集成点）。
        registry: 幂等注册表（默认进程内单例；测试注入隔离实例）。

    Returns:
        WriteResult（dedup_hit=True 时原样返回首次结果，不重复写）。
        P1-1：结果携带 process_core / text_snapshot / evolution（§3.2）。

    Raises:
        SemanticViolation: 语义违规（api 层映射 422）。
        透传 writer 异常（写失败不登记——重试同键可重写，api 层映射 503/500）。

    Claim（§5.2 machine-readable，见 claims.json）：
      - 幂等：同键重发不双写；已处理键原样返回原结果；
      - 无副作用：dedup_hit 命中路径零写入、零状态变更；
      - 写成功才登记：writer 异常 → 不登记；
      - P1-1 提取过程核心优先：过程核心解析/附加全程 fail-open，绝不阻断
        写侧；旧条目读取走 getattr 默认值，迁移数据不崩溃。
    """
    req.resolve()  # 语义校验 + source/gray 口径解析（违规抛 SemanticViolation）
    registry = registry or get_default_registry()
    key = _resolve_idempotency_key(req)
    # P1-1：提取过程核心优先（fail-open；结果进 req.resolved_* 与 WriteResult）
    _resolve_process_core(req)

    def do_write() -> Dict[str, Any]:
        data = writer(req)
        _attach_process_core_to_entry(req, data)
        return data

    result, dedup_hit, record = registry.run_idempotent(key, do_write)
    res = _build_result(req, result, key, dedup_hit=dedup_hit,
                        record=record)
    if not dedup_hit and res.stored:
        # 仅对实际写入触发注入时怀疑（M3 接入三时相怀疑的注入时相）
        _run_doubt_hooks(req, res)
    return res


def feed(text: str, *, session_id: str = "main", source: Optional[str] = None,
         llm_output: str = "", idempotency_key: Optional[str] = None,
         channel: FeedChannel = FeedChannel.PRIMARY,
         writer: Optional[Writer] = None,
         registry: Optional[IdempotencyRegistry] = None,
         metadata: Optional[Dict[str, Any]] = None) -> WriteResult:
    """feed 语义写入口（知识补存双通道：可召回层——坑 6 口径）。

    语义约束（semantics.py 强制）：永不灰化；source 必须可召回；
    补存类知识一律走本入口，**禁止**再走 store_gray 导致探针必 miss。
    """
    req = WriteRequest(
        semantics=WriteSemantics.FEED,
        text=text,
        session_id=session_id,
        source=source,
        llm_output=llm_output,
        idempotency_key=idempotency_key,
        gray=False,  # feed 永不灰化（语义硬约束，resolve 再次校验）
        feed_channel=channel,
        metadata=metadata or {},
    )
    return ingest(req, writer or _default_writer, registry=registry)


def store(text: str, *, session_id: str = "main", source: Optional[str] = None,
          llm_output: str = "", idempotency_key: Optional[str] = None,
          gray: Optional[bool] = None,
          writer: Optional[Writer] = None,
          registry: Optional[IdempotencyRegistry] = None,
          metadata: Optional[Dict[str, Any]] = None) -> WriteResult:
    """store 语义写入口（普通存储；gray=True → store_gray 不可 L1 召回）。

    ``gray=None`` 时读 env ``LMS_STORE_GRAY``（默认 0）——灰度"随时可关"。
    """
    req = WriteRequest(
        semantics=WriteSemantics.STORE,
        text=text,
        session_id=session_id,
        source=source,
        llm_output=llm_output,
        idempotency_key=idempotency_key,
        gray=gray,
        metadata=metadata or {},
    )
    return ingest(req, writer or _default_writer, registry=registry)


#: 默认 writer（未注入时落一个空结果——保证模块可独立 import/运行；
#: 生产/api 层必须注入真实 writer，见 README 接线说明）
def _default_writer(req: WriteRequest) -> Dict[str, Any]:
    return {
        "stored": False,
        "turn_count": 0,
        "reason": "no_writer_injected",
    }

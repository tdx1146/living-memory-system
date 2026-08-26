# -*- coding: utf-8 -*-
"""core/store · 过程核心提取层（P1-1：存储主体是过程，结论是派生视图）

规格依据：
  - `四妹-更新版目的性审计-20260818.md` §三（P1-1 裁决全文，重点 §3.2 可执行
    方案与 §3.3 边界）；
  - `四妹-LMS核心重写规格v2-20260817.md` §1.5（store 提取层）。

裁决（§3.1）：**存储主体 = 过程（惊讶/熵/怀疑/转向的轨迹）；结论 = 派生视图
（从过程可重建，随时演化）**。本模块是对话侧（/store 提取层）的落点——对话侧
补上思考侧（thoughts.jsonl）同款的"过程记录"：

  1. **提取优先级反转**（§3.2）：由"提取核心 ≤300 字（结论优先）"改为
     **"提取过程核心优先"**——惊讶来源（什么输入→什么惊讶→什么修正动作）、
     怀疑轨迹、转向、悬案、置信度曲线、关联 thought id；``text_snapshot``
     降级为派生投影（``build_text_snapshot``）。
  2. **条目新字段**（§3.2 schema）：``process_core``（六段均为列表，允许空）、
     ``text_snapshot``（派生视图；向后兼容映射旧 ``text`` 字段）、
     ``evolution``（``{"history": [append-only 状态转移], "updated_by":
     "ingest|consolidation"}``）。
  3. **M7 旧条目必须仍可读**：所有新字段访问走 ``getattr`` 默认值——旧条目
     无 ``process_core`` → 空结构；无 ``text_snapshot`` → 回退 ``text``；
     无 ``evolution`` → 初始化空演化史。不允许迁移数据读取崩溃。
  4. **严格增量、向后兼容**：不删除、不破坏现有字段（text/confidence/source/
     gray/doubt_state 等写侧既有语义保持）；附加字段只增不改。
  5. **对话侧与思考侧同构**（§3.2）：``surprise_source`` 支持链接 thought id
     （存在即有字段，不强制——空列表即无链接）。

§3.3 诚实的边界：过程核心是**结构化的过程摘要（轨迹，不是日志）**——对话全文
+ 每轮状态全存 = 存储爆炸；原始日志已有 jsonl 事件流，过程核心是它的
"记忆形态投影"。``text_snapshot`` 仍存在（检索时需要快速理解），但地位是投影
不是本体。

设计约束（M1 core/store 同款）：**纯 stdlib**，不 import 任何 LMS 运行时模块
（torch / fastapi / core.hippocampus.*）——可被 python 直接 import/运行；
对条目的读写通过 ``getattr/setattr``（dict 亦兼容），不依赖具体条目类。
提取全程 fail-open：任何异常 → 空过程核心 + 文本回退，绝不阻断写侧（G 模式
以日志可见，不以静默吞掉）。
"""

from __future__ import annotations

import copy
import logging
import re
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("core.store.process_core")

# --------------------------------------------------------------------------- #
#  过程核心 schema（§3.2：六段均为列表，允许空）
# --------------------------------------------------------------------------- #

#: process_core 的六个字段（顺序即 schema 顺序；均为 list，允许空）
PROCESS_CORE_FIELDS: List[str] = [
    "surprise_trace",     # "<输入→惊讶值→修正动作>" 轨迹
    "doubt_events",       # "suspect→confirm|superseded 时间线"（摄入时记 suspect 开端，
    #                        后续状态转移由 evolution.history append-only 承接）
    "turns",              # 立场/结论转向点
    "open_tails",         # unresolved 悬案尾巴
    "confidence_curve",   # 置信度演化（数值列表）
    "surprise_source",    # 关联 thought id（"thought:<id>"，存在即有字段，不强制）
]

#: evolution.updated_by 合法写者（与规格 §3.2 一致：ingest | consolidation）
EVOLUTION_WRITERS = ("ingest", "consolidation")

#: text_snapshot 派生视图长度上限（与既有 extract_core ≤300 字口径同源）
DEFAULT_SNAPSHOT_MAX_CHARS = 300

#: 单条轨迹/事件的摘要上限（结构化过程摘要，不是日志——§3.3）
DEFAULT_TRACE_MAX_CHARS = 80


def empty_process_core() -> Dict[str, List[Any]]:
    """空过程核心（六段均为空列表——旧条目/无信号输入的兜底结构）。"""
    return {f: [] for f in PROCESS_CORE_FIELDS}


def init_evolution(updated_by: str = "ingest") -> Dict[str, Any]:
    """初始化演化史（``{"history": [], "updated_by": "ingest"}``）。

    ``updated_by`` 只允许 ingest / consolidation（写侧时相——与
    rebuttal_consistency 铁律同款：检索绝不写）。非法写者回退 "ingest"。
    """
    return {"history": [], "updated_by": (
        updated_by if updated_by in EVOLUTION_WRITERS else "ingest")}


def make_transition(state: str, *, at: Optional[float] = None,
                    detail: Optional[str] = None) -> Dict[str, Any]:
    """构造一条状态转移记录（append-only 的单个元素）。

    Args:
        state: 转移后的状态（如 "created" / "suspect" / "confirm" /
            "superseded" / "reconsolidated"）。
        at: 转移时间戳（默认 now）。
        detail: 可选补充（谁触发的、被谁反驳等，纯文本观测）。
    """
    tr: Dict[str, Any] = {"state": state, "at": float(at if at is not None else time.time())}
    if detail:
        tr["detail"] = str(detail)
    return tr


def record_transition(evolution: Dict[str, Any], transition: Dict[str, Any],
                      *, updated_by: Optional[str] = None) -> Dict[str, Any]:
    """向演化史追加一条状态转移（**append-only：绝不覆盖历史**）。

    演进的是同一个 ``evolution`` dict（调用方持有的引用被原地追加——条目
    持有该 dict 时即为持久形态）；返回同一 dict 便于链式/断言。
    """
    ev = evolution if isinstance(evolution, dict) else init_evolution()
    history = ev.setdefault("history", [])
    if not isinstance(history, list):
        ev["history"] = []
        history = ev["history"]
    history.append(dict(transition))
    if updated_by and updated_by in EVOLUTION_WRITERS:
        # 非法写者（如 retrieval——只读铁律同款）被拒绝：保持原写者不覆盖
        ev["updated_by"] = updated_by
    return ev


# --------------------------------------------------------------------------- #
#  启发式提取（提取优先级反转：过程核心优先）
# --------------------------------------------------------------------------- #
# 工程口径（§3.3）：过程核心是**结构化过程摘要**——对输入文本做确定性标记
# 提取（惊讶/怀疑/转向/悬案/置信/thought 链接），不是全文日志。标记词典为
# 中文 + 英文双覆盖；每类各取命中句摘要（去重、保序、截断）。

_SENT_SPLIT_RE = re.compile(r"[。！？!?\n；;]+")

_SURPRISE_MARKERS = (
    "惊讶", "意外", "没想到", "出乎意料", "surprise", "surprised", "unexpected",
)
_CORRECTION_MARKERS = (
    "修正", "修正了", "纠正", "纠正了", "推翻", "推翻了", "更正", "之前以为",
    "之前认为", "以前以为", "以前认为", "重新认识", "更正为", "改为",
    "corrected", "revised",
)
_DOUBT_MARKERS = (
    "怀疑", "怀疑了", "存疑", "可疑", "suspect", "conflict", "superseded",
    "证伪", "被反驳", "待验证", "verification", "rebuttal", "[doubt", "待核实",
)
_TURN_MARKERS = (
    "转向", "反转", "改变主意", "改变了主意", "改主意", "改变想法", "改口",
    "立场", "翻案", "推翻", "推翻了", "现在认为", "changed my mind",
    "turnaround", "180度", "推翻了我",
)
_OPEN_TAIL_MARKERS = (
    "悬案", "未解", "待查", "待定", "还不确定", "unresolved", "遗留问题",
    "未解决", "有待", "待考证", "待确认", "尚未明确",
)
#: thought 链接识别（"thought:abc" / "thought-123" / "thought_456" /
#: "thought abc" 均归一为 "thought:<id>"；id 允许字母数字与 -_，≥2 位）
_THOUGHT_LINK_RE = re.compile(
    r"thought\s*[:：_\-]?\s*([a-zA-Z0-9][a-zA-Z0-9_-]*)", re.IGNORECASE)
#: 置信度数值识别（"置信度 0.72" / "confidence 0.72" / "置信度: 0.65"）
_CONFIDENCE_RE = re.compile(
    r"(?:置信度|confidence)\s*[:：]?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def _snippets(text: str, markers: tuple, max_chars: int) -> List[str]:
    """按标记词典提取命中句摘要（去重、保序、截断）。

    判定按**句**做（不是整段文本）：句子包含任一标记（大小写不敏感）
    才命中——防"段内一处标记 → 全段句子误判"。
    """
    if not text:
        return []
    out: List[str] = []
    seen = set()
    for sent in _SENT_SPLIT_RE.split(text):
        s = sent.strip()
        if not s or len(s) < 2:
            continue
        low_s = s.lower()
        if not any(m in low_s for m in markers):
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s if len(s) <= max_chars else s[:max_chars] + "…")
    return out


def _normalize_confidence(value: Any) -> Optional[float]:
    """数值归一（含字符串 "0.72"）；非法 → None。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")):  # NaN/inf 防御
        return None
    return round(max(0.0, min(1.0, v)), 4)


def _extract_confidence_curve(metadata: Dict[str, Any],
                              confidence: Optional[float]) -> List[float]:
    """置信度曲线：显式 metadata["confidence_curve"] > 单点 confidence。"""
    curve = metadata.get("confidence_curve")
    if isinstance(curve, (list, tuple)) and curve:
        vals = [c for c in (_normalize_confidence(x) for x in curve)
                if c is not None]
        if vals:
            return vals
    c = _normalize_confidence(confidence)
    if c is not None:
        return [c]
    return []


def _extract_surprise_source(metadata: Dict[str, Any], text: str) -> List[str]:
    """surprise_source：metadata 显式列表 > 文本 thought 链接识别。

    归一为 ``thought:<id>``（§3.2 links 同款形态）；存在即有字段，
    无链接 → 空列表（不强制）。
    """
    out: List[str] = []
    seen = set()
    raw = metadata.get("surprise_source")
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if not isinstance(item, str):
                continue
            item = item.strip()
            if not item:
                continue
            norm = item if item.startswith("thought:") else "thought:" + item
            if norm not in seen:
                seen.add(norm)
                out.append(norm)
    if text:
        for m in _THOUGHT_LINK_RE.finditer(text):
            norm_id = m.group(1)
            if len(norm_id) < 2:  # 防 "thoughts"→"s" 这类单字符误匹配
                continue
            norm = "thought:" + norm_id
            if norm not in seen:
                seen.add(norm)
                out.append(norm)
    return out


def extract_process_core(*, text: str = "", llm_output: str = "",
                         metadata: Optional[Dict[str, Any]] = None,
                         surprise: Optional[float] = None,
                         confidence: Optional[float] = None,
                         turn: Optional[int] = None,
                         max_trace_chars: int = DEFAULT_TRACE_MAX_CHARS
                         ) -> Dict[str, List[Any]]:
    """提取过程核心（**过程核心优先**，§3.2）。

    输入（写请求的既有字段，不新增必填）：
      - ``text``       ：user_input（对话侧输入原文）；
      - ``llm_output`` ：LLM 回复（结论文本，作为惊讶/修正动作的上下文）；
      - ``metadata``   ：观测透传（可携带 ``confidence_curve`` /
        ``surprise_source`` / ``surprise`` / ``confidence``，均可选）；
      - ``surprise``   ：本轮惊讶值（可选；有值 → surprise_trace 带数值）；
      - ``confidence`` ：条目当前置信度（可选；单点起点进 confidence_curve）；
      - ``turn``       ：轮次（可选；参与 trace 标注，审计用）。

    返回：``process_core`` dict——六段均为列表，**允许空**（无信号输入返回
    空结构，不臆造）。全部为纯函数确定性提取，异常不抛出（fail-open：
    调用方兜底空结构）。

    ``surprise_trace`` 条目格式（§3.2："<输入→惊讶值→修正动作>"）：
      - 有惊讶值：``输入「<snippet>」→ 惊讶值=<v> → 修正:<action|无>``
      - 无惊讶值：``输入「<snippet>」→ 惊讶 → 修正:<action|无>``
      ``turn`` 存在时追加 `` @turn=<t>`` 标注。
    """
    metadata = metadata if isinstance(metadata, dict) else {}
    if surprise is None:
        surprise = _normalize_confidence(metadata.get("surprise"))
    if confidence is None:
        confidence = _normalize_confidence(metadata.get("confidence"))

    joined = "%s\n%s" % (text or "", llm_output or "")

    # -- surprise_trace：惊讶来源 + 修正动作 -------------------------------- #
    surprise_snippets = _snippets(joined, _SURPRISE_MARKERS, max_trace_chars)
    surprise_trace: List[str] = []
    for snip in surprise_snippets:
        has_correction = any(
            m in snip for m in _CORRECTION_MARKERS) or any(
            m in snip.lower() for m in _CORRECTION_MARKERS)
        if surprise is not None:
            entry = "输入「%s」→ 惊讶值=%.2f → 修正:%s" % (
                snip, surprise, "有" if has_correction else "无")
        else:
            entry = "输入「%s」→ 惊讶 → 修正:%s" % (
                snip, "有" if has_correction else "无")
        if turn is not None:
            entry += " @turn=%s" % turn
        surprise_trace.append(entry)
    # 有惊讶值但文本无显式惊讶标记：仍记一条最小轨迹（熵/惊讶进入存储结构）
    if not surprise_trace and surprise is not None:
        prefix = (text or llm_output or "").strip()
        snippet = (prefix[:max_trace_chars] + "…") if len(prefix) > max_trace_chars else prefix
        snippet = snippet or "（无文本）"
        entry = "输入「%s」→ 惊讶值=%.2f → 修正:无" % (snippet, surprise)
        if turn is not None:
            entry += " @turn=%s" % turn
        surprise_trace.append(entry)

    # -- doubt_events：怀疑轨迹（摄入时记 suspect 开端；后续转移在演化史） --- #
    doubt_events = ["suspect: %s" % s for s in _snippets(
        joined, _DOUBT_MARKERS, max_trace_chars)]

    # -- turns：立场/结论转向点 --------------------------------------------- #
    turns = ["转向: %s" % s for s in _snippets(
        joined, _TURN_MARKERS, max_trace_chars)]

    # -- open_tails：unresolved 悬案尾巴 ------------------------------------- #
    open_tails = ["悬案: %s" % s for s in _snippets(
        joined, _OPEN_TAIL_MARKERS, max_trace_chars)]

    # -- confidence_curve：数值列表 ------------------------------------------ #
    confidence_curve = _extract_confidence_curve(metadata, confidence)
    if not confidence_curve:
        # 文本显式置信度（"置信度 0.72"）兜底
        confs = [_normalize_confidence(m.group(1))
                 for m in _CONFIDENCE_RE.finditer(joined)]
        confidence_curve = [c for c in confs if c is not None]

    # -- surprise_source：关联 thought id（存在即有字段，不强制） -------------- #
    surprise_source = _extract_surprise_source(metadata, joined)

    return {
        "surprise_trace": surprise_trace,
        "doubt_events": doubt_events,
        "turns": turns,
        "open_tails": open_tails,
        "confidence_curve": confidence_curve,
        "surprise_source": surprise_source,
    }


def _truncate(text: str, max_chars: int) -> str:
    text = text if isinstance(text, str) else str(text or "")
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "…"


def _core_has_signals(pc: Dict[str, Any]) -> bool:
    """过程核心是否含任何信号（全空 → 视同无过程，快照回退旧文本）。"""
    return any(bool(pc.get(f)) for f in PROCESS_CORE_FIELDS)


def build_text_snapshot(process_core: Optional[Dict[str, Any]] = None,
                        *, text: str = "", core: str = "",
                        max_chars: int = DEFAULT_SNAPSHOT_MAX_CHARS) -> str:
    """派生视图：text_snapshot = 从过程核心重建的当前结论快照。

    §3.2/§3.3：快照是**投影不是本体**——可压缩、可重建、随演化更新。
    优先级：
      1. 过程核心有信号 → 由轨迹摘要派生（转向/怀疑/悬案/惊讶各取最近
         至多 2 条，压缩拼接，≤max_chars）；
      2. 无过程信号 → 回退 ``core``（既有 ≤300 字提取核心）；
      3. 再回退 ``text``（旧条目 text 字段——M7 向后兼容映射）。
    永不返回 None/抛异常（fail-open → 空串）。
    """
    try:
        pc = process_core if isinstance(process_core, dict) else None
        if pc is not None and _core_has_signals(pc):
            parts: List[str] = []
            # 各段条目自带标签前缀（"转向: "/"suspect: "/"悬案: "/"输入「…」"），
            # 快照只拼接不重复加标签（投影 = 过程轨迹的压缩视图）
            if pc.get("turns"):
                parts.append("；".join(pc["turns"][-2:]))
            if pc.get("doubt_events"):
                parts.append("；".join(pc["doubt_events"][-2:]))
            if pc.get("open_tails"):
                parts.append("；".join(pc["open_tails"][-2:]))
            if pc.get("surprise_trace"):
                parts.append("；".join(pc["surprise_trace"][-2:]))
            if parts:
                return _truncate(" | ".join(parts), max_chars)
        base = core or text or ""
        return _truncate(base, max_chars)
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("text_snapshot 派生失败（fail-open→文本回退）：%s", e)
        return _truncate(core or text or "", max_chars)


# --------------------------------------------------------------------------- #
#  条目附加 / 读取（M7 旧条目必须仍可读：一律 getattr 默认值）
# --------------------------------------------------------------------------- #

def _get(entry: Any, key: str, default: Any = None) -> Any:
    if isinstance(entry, dict):
        return entry.get(key, default)
    return getattr(entry, key, default)


def _set(entry: Any, key: str, value: Any) -> None:
    if isinstance(entry, dict):
        entry[key] = value
    else:
        setattr(entry, key, value)


def attach_entry_fields(entry: Any, *,
                        process_core: Optional[Dict[str, Any]] = None,
                        text_snapshot: Optional[str] = None,
                        evolution: Optional[Dict[str, Any]] = None,
                        updated_by: str = "ingest") -> Any:
    """把 P1-1 新字段附加到条目（**严格增量：只增不改，不动既有字段**）。

    兼容 EpisodicEntry 实例与 dict 两类条目形态。过程核心深拷贝落位
    （防跨条目共享可变列表）；未显式给快照时由过程核心派生（回退条目
    既有 ``text``/``core``）。返回 entry（链式/断言用）。
    """
    pc = empty_process_core()
    if isinstance(process_core, dict):
        pc = {f: list(process_core.get(f, []) or []) for f in PROCESS_CORE_FIELDS}
    _set(entry, "process_core", copy.deepcopy(pc))
    if text_snapshot is None:
        text_snapshot = build_text_snapshot(
            pc, text=_get(entry, "text", "") or "",
            core=_get(entry, "core", "") or "")
    _set(entry, "text_snapshot", str(text_snapshot or ""))
    _set(entry, "evolution", (
        evolution if isinstance(evolution, dict)
        else init_evolution(updated_by=updated_by)))
    return entry


def has_process_core(entry: Any) -> bool:
    """条目是否已携带 P1-1 过程核心字段（新条目 True / 旧条目 False）。"""
    return _get(entry, "process_core", None) is not None


def get_process_core(entry: Any) -> Dict[str, List[Any]]:
    """读取过程核心（getattr 兜底：旧条目无字段 → 空结构，不崩溃）。"""
    pc = _get(entry, "process_core", None)
    if not isinstance(pc, dict):
        return empty_process_core()
    return {f: list(pc.get(f, []) or []) for f in PROCESS_CORE_FIELDS}


def get_text_snapshot(entry: Any) -> str:
    """读取 text_snapshot（getattr 兜底：旧条目无字段 → 回退旧 text）。"""
    ts = _get(entry, "text_snapshot", None)
    if isinstance(ts, str) and ts:
        return ts
    # M7 迁移兼容：旧条目无 text_snapshot → 回退 text（旧字段语义保持）
    return _get(entry, "text", "") or ""


def get_evolution(entry: Any) -> Dict[str, Any]:
    """读取演化史（getattr 兜底：旧条目无字段 → 初始化空演化史，不崩溃）。

    返回的是**拷贝**（history 列表复制）——读侧不持有内部引用，防误改。
    """
    ev = _get(entry, "evolution", None)
    if isinstance(ev, dict) and isinstance(ev.get("history"), list):
        return {"history": list(ev["history"]),
                "updated_by": ev.get("updated_by", "ingest")}
    return init_evolution()


def append_transition(entry: Any, transition: Dict[str, Any], *,
                      updated_by: Optional[str] = None) -> Dict[str, Any]:
    """向条目演化史追加一条状态转移（append-only：不覆盖历史）。

    旧条目（无 evolution 字段）首次调用时自动初始化——**迁移数据首次
    演化写入不崩溃**（M7 兼容）。
    """
    ev = get_evolution(entry)
    record_transition(ev, transition, updated_by=updated_by)
    _set(entry, "evolution", ev)
    return ev


def extract_process_core_for_request(req: Any) -> Dict[str, List[Any]]:
    """从写请求提取过程核心（ingest 写路径接入点）。

    ``req`` 提供 ``text`` / ``llm_output`` / ``metadata``（与 WriteRequest
    同构即可，不依赖具体类——纯 stdlib）。fail-open：异常 → 空结构。
    """
    try:
        return extract_process_core(
            text=getattr(req, "text", "") or "",
            llm_output=getattr(req, "llm_output", "") or "",
            metadata=getattr(req, "metadata", None) or {},
        )
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("过程核心提取失败（fail-open→空结构）：%s", e)
        return empty_process_core()

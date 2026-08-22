"""结构化怀疑摄入（体验层 D，设计 v1.1 §6.6 / §8.1）—— fail-open。
⛔ 红线：_strip_artifact_prefix（2026-08-22 自喂断环）剥离"用户:/助手:"包装前缀——
   移除会导致 E3 路径B 产物无限自喂、gap_registry 嵌套 [doubt] 主题堆积。

数据源链路：doubt_ingest（/feed 结构化摄入）→ gap_registry（登记）→
/status doubt.gaps + 回魂怀疑灯。

前缀协议（纯工程接口，溯源 §4.2）：
    [doubt] conflict: <被反驳的旧记忆文本摘要>     → 证伪（rebuttal +1，labile）
    [doubt] 证伪: <同 conflict>                    → 别名（E3 拍板 2：证伪收编）
    [doubt] reactivate: <同 conflict>              → 别名（E3 拍板 2：重激活收编）
    [doubt] fok: <未决问题>                        → A 类缺口登记
    [doubt] lowconf: <低置信条目文本摘要>           → B 类缺口登记
    [doubt] event: <事件描述>                      → 仅登记（诊断）

E3（2026-08-20，dandan 拍板 2）：'证伪'/'reactivate' 别名在解析时归一为
'conflict' 语义（证伪收编进 conflict）；conflict 证伪命中补入队（根因 3
修复：labile 带证据进再巩固候选队列，独立受队列开关治理）。

fail-open：无前缀/解析失败 = 普通塑形，逐字节不变（红线）。
"""

from __future__ import annotations

import re
from typing import Optional

from core.doubt.confidence_field import is_low_confidence
from core.doubt.reconsolidation import mark_labile

# [doubt] 前缀协议（大小写不敏感；内容支持跨行，取前 300 字）
_DOUBT_PREFIX_RE = re.compile(
    r"^\s*\[doubt\]\s*(\w+)\s*[:：]\s*(.+)$", re.I | re.S)
_KNOWN_KINDS = {'conflict', 'fok', 'lowconf', 'event'}
# E3（dandan 拍板 2，2026-08-20：证伪收编）：'证伪'/'reactivate' 别名 →
# conflict 语义（解析时归一为 canonical kind，下游分支零改动）。
_KIND_ALIASES = {'证伪': 'conflict', 'reactivate': 'conflict'}
# 结构化摄入不再作为普通对话入库（系统事件不是对话，同 8/10 垃圾过滤哲学）
_DOUBT_EVENT_RE = re.compile(r"^\s*\[doubt\]", re.I)

# 2026-08-22 自喂断环（E3 路径B 产物污染修复）：路径B 重激活线索是
# episodic 条目文本（带"用户: "包装前缀），process_turn 摄入时前缀导致
# ^[doubt] 锚定失败 → 被当普通对话入库 → 又被选择器选中重激活 → 反馈环。
# 剥离包装前缀后 [doubt] 事件被正确识别为系统事件（不入库、正常登记）。
_ARTIFACT_PREFIX_RE = re.compile(r"^(?:用户|助手)\s*[:：]\s*", re.I)


def _strip_artifact_prefix(text: str) -> str:
    """剥离 E3 路径B 产物自带的"用户: /助手: "包装前缀（自喂断环）。"""
    if not text:
        return text
    return _ARTIFACT_PREFIX_RE.sub("", text, count=1)


def is_doubt_event(text: str) -> bool:
    """判断文本是否为结构化怀疑事件（带 [doubt] 前缀）。

    [收尾/存量失败] 修复 is_doubt_event 误判（审计 F3 存量项，286085b 起失败）：
    2026-08-22 自喂断环修复引入 ``_strip_artifact_prefix`` 后，任何带
    "用户:/助手:" 包装且以 [doubt] 开头的文本都被判为怀疑事件——真实用户
    提问 "用户: [doubt] 是什么意思？" 被当系统事件丢弃（不入库，文本丢失）。
    修复：包装前缀剥离后**必须是合法协议事件**（parse_doubt_event 非空，
    即带 kind+冒号）才判为怀疑事件；裸 [doubt] 前缀（锚定行首）判定不变，
    正文提及不误伤。E3 路径 B 产物（"用户: [doubt] conflict: …" 合法协议）
    仍被正确识别，08-22 断环语义保持。
    """
    if not text:
        return False
    # 锚定行首：裸 [doubt] 前缀（含未知 kind 的协议行）→ 系统事件（同旧语义）
    if _DOUBT_EVENT_RE.search(text):
        return True
    # E3 路径 B 产物：剥离 "用户:/助手:" 包装后须是**合法协议事件**
    # （kind+冒号），"用户: [doubt] 是什么意思？" 这类真实提问不满足
    # _DOUBT_PREFIX_RE（无 kind+冒号）→ 判为普通对话（正确入库）。
    stripped = _strip_artifact_prefix(text)
    return stripped != text and parse_doubt_event(stripped) is not None


def parse_doubt_event(text: str) -> Optional[dict]:
    """解析 [doubt] 前缀协议 → {kind, content}；解析失败返回 None（fail-open）。

    E3（拍板 2）：'证伪'/'reactivate' 别名在解析时归一为 'conflict'——
    下游按 conflict 分支处理（证伪收编进 conflict 语义）。
    """
    if not text:
        return None
    m = _DOUBT_PREFIX_RE.match(_strip_artifact_prefix(text).strip())
    if not m:
        return None
    kind = m.group(1).strip().lower()
    kind = _KIND_ALIASES.get(kind, kind)
    if kind not in _KNOWN_KINDS:
        return None
    content = m.group(2).strip()[:300]
    if not content:
        return None
    return {'kind': kind, 'content': content}


def ingest(loop, text: str, target_entry=None) -> Optional[dict]:
    """结构化摄入（process_turn /feed 路径调用，fail-open）。

    参数:
        loop: LivingMemoryLoop（提供 gap_registry / memory）。
        text: 当前轮文本。
        target_entry: E3 重激活专用（可选）——conflict 事件的目标条目已由
            选择器定位时显式传入，跳过文本重叠匹配（旧调用方不传 → 行为
            逐位不变）。纯增量参数，fail-open（异常回退重叠匹配）。

    返回:
        摄入事件 dict（含 kind/content/action），非怀疑事件返回 None。
    """
    ev = parse_doubt_event(text)
    if ev is None:
        return None
    kind, content = ev['kind'], ev['content']
    action = 'registered'
    hit_entry = None  # 阶段 3：conflict 证伪命中的条目（loop 校准钩子用）
    try:
        registry = getattr(loop, 'gap_registry', None)
        if kind == 'conflict':
            # 证伪：在 episodic 中找内容重叠的旧记忆 → rebuttal +1 + labile
            try:
                hit = target_entry if target_entry is not None \
                    else _find_overlapping_entry(loop, content)
            except Exception:  # pylint: disable=broad-except
                hit = _find_overlapping_entry(loop, content)
            if hit is not None:
                mark_labile(hit, violated_by=content)
                action = 'rebutted'
                hit_entry = hit
                detail = 'conflict 事件（已标记证伪条目）'
                # E3（根因 3 修复）：conflict 证伪命中 → 补入队（写侧时相
                # INJECTION——队列契约"入队只允许写侧时相"）。labile 条目
                # 带证据进再巩固候选队列，梦期/巩固期消化。独立受队列自身
                # 开关（LMS_DOUBT_RECONSOLIDATION_ENABLED）治理；fail-open。
                _enqueue_conflict_hit(loop, hit, content)
            else:
                detail = 'conflict 事件（未找到重叠条目）'
            if registry is not None:
                registry.register_fok_unresolved(topic=content, detail=detail)
        elif kind == 'fok':
            if registry is not None:
                registry.register_fok_unresolved(topic=content)
        elif kind == 'lowconf':
            hit = _find_overlapping_entry(loop, content)
            if registry is not None and hit is not None:
                registry.register_low_confidence(hit)
            elif registry is not None:
                registry.register_fok_unresolved(
                    topic=content, detail='lowconf 未找到重叠条目')
        elif kind == 'event':
            if registry is not None:
                registry.register_fok_unresolved(
                    topic=content, detail='doubt event 诊断登记')
    except Exception:
        # fail-open：结构化摄入异常绝不阻断主循环
        action = 'failed'
    # 阶段 3：entry 仅对 conflict-证伪 返回（其余为 None）——loop 据此
    # 收集 conformal 校准集 + 标记负性证据（对称性约束）。纯增量字段，
    # 旧调用方无感。
    return {**ev, 'action': action, 'entry': hit_entry}


def _enqueue_conflict_hit(loop, hit, content: str) -> bool:
    """E3（根因 3 修复）：conflict 证伪命中 → 补入队（写侧时相 INJECTION）。

    labile 标记后把候选带证据登记进再巩固候选队列（R3 C1 接线缺口修复：
    labile 不进队列 → 队列恒空 → rewritten 恒 0）。fail-open：任何异常
    返回 False，绝不阻断证伪标记与主循环。独立受队列自身开关治理
    （LMS_DOUBT_RECONSOLIDATION_ENABLED，默认 1=开——队列是 R3 机制本体）。
    """
    try:
        q = getattr(loop, 'reconsolidation_queue', None)
        if q is None or not q.enabled:
            return False
        from core.doubt.state_machine import DoubtPhase
        return bool(q.enqueue(
            hit, reason="doubt_conflict_labile", score=None,
            detail=str(content)[:300], phase=DoubtPhase.INJECTION.value))
    except Exception:  # pylint: disable=broad-except
        return False


def _find_overlapping_entry(loop, content: str):
    """在 episodic 缓冲区找内容与 content 重叠（子串/包含）的条目。

    工程匹配（无 LLM）：content 出现在条目文本中，或条目文本出现在
    content 中。找不到返回 None（fail-open）。
    """
    try:
        memory = getattr(loop, 'memory', None)
        if memory is None:
            return None
        needle = content[:120]
        for entry in memory.iter_episodic():
            text = getattr(entry, 'text', '') or ''
            if needle and (needle in text or text[:120] in needle):
                return entry
    except Exception:
        return None
    return None

"""结构化怀疑摄入（体验层 D，设计 v1.1 §6.6 / §8.1）—— fail-open。

数据源链路：doubt_ingest（/feed 结构化摄入）→ gap_registry（登记）→
/status doubt.gaps + 回魂怀疑灯。

前缀协议（纯工程接口，溯源 §4.2）：
    [doubt] conflict: <被反驳的旧记忆文本摘要>     → 证伪（rebuttal +1，labile）
    [doubt] fok: <未决问题>                        → A 类缺口登记
    [doubt] lowconf: <低置信条目文本摘要>           → B 类缺口登记
    [doubt] event: <事件描述>                      → 仅登记（诊断）

fail-open：无前缀/解析失败 = 普通塑形，逐字节不变（红线）。
"""

from __future__ import annotations

import re
from typing import Optional

from core.doubt.confidence_field import is_low_confidence
from core.doubt.reconsolidation import mark_labile

# [doubt] 前缀协议（大小写不敏感；内容支持跨行，取前 300 字）
_DOUBT_PREFIX_RE = re.compile(
    r"^\s*\[doubt\]\s*(\w+)\s*:\s*(.+)$", re.I | re.S)
_KNOWN_KINDS = {'conflict', 'fok', 'lowconf', 'event'}
# 结构化摄入不再作为普通对话入库（系统事件不是对话，同 8/10 垃圾过滤哲学）
_DOUBT_EVENT_RE = re.compile(r"^\s*\[doubt\]", re.I)


def is_doubt_event(text: str) -> bool:
    """判断文本是否为结构化怀疑事件（带 [doubt] 前缀）。"""
    if not text:
        return False
    return bool(_DOUBT_EVENT_RE.search(text))


def parse_doubt_event(text: str) -> Optional[dict]:
    """解析 [doubt] 前缀协议 → {kind, content}；解析失败返回 None（fail-open）。"""
    if not text:
        return None
    m = _DOUBT_PREFIX_RE.match(text.strip())
    if not m:
        return None
    kind = m.group(1).strip().lower()
    if kind not in _KNOWN_KINDS:
        return None
    content = m.group(2).strip()[:300]
    if not content:
        return None
    return {'kind': kind, 'content': content}


def ingest(loop, text: str) -> Optional[dict]:
    """结构化摄入（process_turn /feed 路径调用，fail-open）。

    参数:
        loop: LivingMemoryLoop（提供 gap_registry / memory）。
        text: 当前轮文本。

    返回:
        摄入事件 dict（含 kind/content/action），非怀疑事件返回 None。
    """
    ev = parse_doubt_event(text)
    if ev is None:
        return None
    kind, content = ev['kind'], ev['content']
    action = 'registered'
    try:
        registry = getattr(loop, 'gap_registry', None)
        if kind == 'conflict':
            # 证伪：在 episodic 中找内容重叠的旧记忆 → rebuttal +1 + labile
            hit = _find_overlapping_entry(loop, content)
            if hit is not None:
                mark_labile(hit, violated_by=content)
                action = 'rebutted'
                detail = 'conflict 事件（已标记证伪条目）'
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
    return {**ev, 'action': action}


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

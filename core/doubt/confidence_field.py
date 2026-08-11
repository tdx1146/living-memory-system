"""置信度场（体验层 D1，设计 v1.1 §6.1）—— 纯函数。

公式（P1 §2.1 原样）：
    confidence = clamp01(1.0 × (1 − rebuttal_rate) × source_trust)
    rebuttal_rate = rebuttal_count / (reference_count + 1)
    rebuttal ≥ 2 → 强制 0.1（memory_trust 工程先例，doubt 侧既有资产）

置信度三路汇入（设计 v1.1 §6.1）：
  ① 摄入时初始化（source_trust 按来源定）
  ② 被去稳定化/证伪时 rebuttal_count+1、violated_by 记录（reconsolidation）
  ③ 召回时 record_reference（reference_count+1，正向佐证；memory.py 钩子）

专注化强调：置信度挂在**条目**上（监控-控制内生于提取，Koriat &
Goldsmith 1996 A1），作用域是"这条已关注记忆值不值得信"，与
purpose.precision（管"关注哪"，Pouget 2016：confidence≠precision）正交。

依据：Koriat & Goldsmith 1996 (A1)、Pouget 2016 (A3)、Johnson 1993 (A2)、
Sperber 2010 (D3)。
"""

from __future__ import annotations

import time
from typing import Optional

# 来源可信度（工程定值，与 memory_trust 语义对齐；无论文数值，验收中校准）
_SOURCE_TRUST = {
    'external': 1.0,    # 外部对话（主记忆来源）
    'self_ref': 0.8,    # 自指回路产物（二阶反思，可信度略低）
    'doubt': 0.5,       # 怀疑复核/改写条目（待进一步佐证）
    'event_bus': 0.6,   # 总线事件塑形（无直接对话证据）
}

# 低置信阈值（<0.3 → 进入复核通道 / 不参与放大回放）
LOW_CONFIDENCE_THRESHOLD = 0.3
# 强制降权门槛（rebuttal ≥ 2 → confidence 压到 0.1）
FORCE_DOWNGRADE_REBUTTALS = 2
FORCE_DOWNGRADE_CONFIDENCE = 0.1


def get_source_trust(source: str) -> float:
    """按来源返回可信度（未知来源回退 1.0，fail-open）。"""
    return _SOURCE_TRUST.get(source or 'external', 1.0)


def compute_confidence(
    rebuttal_count: int, reference_count: int,
    source_trust: float = 1.0,
) -> float:
    """置信度公式：clamp01((1 − rebuttal_rate) × source_trust)。

    参数:
        rebuttal_count: 被反驳次数。
        reference_count: 被引用次数（正向佐证）。
        source_trust: 来源可信度 [0,1]。

    返回:
        [0, 1] 的置信度。rebuttal ≥ 2 → 强制 0.1。
    """
    rebuttal_rate = rebuttal_count / (reference_count + 1)
    c = 1.0 * (1.0 - rebuttal_rate) * max(0.0, min(1.0, source_trust))
    if rebuttal_count >= FORCE_DOWNGRADE_REBUTTALS:
        c = min(c, FORCE_DOWNGRADE_CONFIDENCE)
    return max(0.0, min(1.0, c))


def refresh_confidence(entry) -> None:
    """重算条目置信度（三路汇入后调用；fail-open）。"""
    try:
        entry.confidence = compute_confidence(
            entry.rebuttal_count, entry.reference_count,
            getattr(entry, 'source_trust', 1.0))
    except Exception:
        pass


def is_low_confidence(entry, threshold: float = LOW_CONFIDENCE_THRESHOLD) -> bool:
    """低置信判定（confidence < 阈值；字段缺失时按非低置信处理，fail-open）。"""
    try:
        return float(getattr(entry, 'confidence', 1.0)) < threshold
    except (TypeError, ValueError):
        return False


def record_reference(entry, now: Optional[float] = None) -> None:
    """③ 正向佐证：召回引用计数 +1（memory.py record_reference 钩子调用）。

    同时更新 recall_count / last_recalled_at（遗忘曲线与反教条抽查数据源）。
    fail-open：任何异常静默（不阻塞检索路径）。
    """
    try:
        entry.reference_count = getattr(entry, 'reference_count', 0) + 1
        entry.recall_count = getattr(entry, 'recall_count', 0) + 1
        entry.last_recalled_at = now if now is not None else time.time()
        refresh_confidence(entry)
    except Exception:
        pass


def mark_rebutted(entry, violated_by: Optional[str] = None,
                  now: Optional[float] = None) -> None:
    """② 证伪：rebuttal_count +1、violated_by 记录、置信度重算。

    （去稳定化/结构化 doubt 摄入共用；fail-open。）
    """
    try:
        entry.rebuttal_count = getattr(entry, 'rebuttal_count', 0) + 1
        if violated_by:
            entry.violated_by = str(violated_by)[:200]
        refresh_confidence(entry)
    except Exception:
        pass


def get_rebuttal_rate(entry) -> float:
    """条目当前反驳率（反流畅回放权重用；fail-open 回退 0.0）。"""
    try:
        rc = int(getattr(entry, 'rebuttal_count', 0) or 0)
        refc = int(getattr(entry, 'reference_count', 0) or 0)
        return rc / (refc + 1)
    except (TypeError, ValueError):
        return 0.0

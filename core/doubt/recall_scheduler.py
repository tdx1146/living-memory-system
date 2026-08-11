"""召回唤起 salience（体验层 D5，设计 v1.1 §6.5）—— 纯函数。

salience(m) = α·cue_sim(m) + β·confidence(m) + γ·forgetting(m) + δ·event(m)

**工程假设标注（溯源 §4.2）**：四权重与 event 项无论文定标 → 默认保守
（α 主导，β/γ 弱、δ=0 先观察）。salience 是加权和而非 eq.23 的贝叶斯
归一化，落地用实验标定而非理论推导。

**专注化修订（对 P1 的两处修订之一，设计 v1.1 §6.5）**：top-k 低置信
配额**必须 relevance-gated**——只有 cue_sim ≥ 0.5 × 当前 top1 分数的
待复核条目才占"1 个配额席位"，否则宁缺毋滥。理由：无门控的配额会把
无关低置信条目塞进注入 = 怀疑本身变成跑偏源（违反拍板 2）。

依据：母本 eq.22/23（检索=先验×似然后验）、Pouget 2016、母本 §1 ghost
attractors（遗忘项）、event 项标工程假设。
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

from core.doubt.confidence_field import LOW_CONFIDENCE_THRESHOLD

# 默认权重（保守：α 主导，β/γ 弱、δ=0 先观察——溯源 §4.2 工程假设）
DEFAULT_WEIGHTS = {"alpha": 0.6, "beta": 0.2, "gamma": 0.2, "delta": 0.0}

# 遗忘曲线参数（FEP 遗忘曲线工程简化：strength 随 age 衰减，随 recall 恢复）
FORGET_DECAY = 0.02       # 每天衰减（exp 形式，简化线性）
FORGET_RECALL_BOOST = 0.1  # 每次召回恢复量


def salience(
    cue_sim: float,
    confidence: float,
    forgetting: float,
    event: float = 0.0,
    weights: Optional[dict] = None,
) -> float:
    """salience = α·cue_sim + β·confidence + γ·forgetting + δ·event。

    各分量均应在 [0,1]；返回加权和（未归一化，仅用于排序比较）。
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    return (
        w["alpha"] * max(0.0, min(1.0, cue_sim))
        + w["beta"] * max(0.0, min(1.0, confidence))
        + w["gamma"] * max(0.0, min(1.0, forgetting))
        + w["delta"] * max(0.0, min(1.0, event))
    )


def forgetting(entry, now: Optional[float] = None) -> float:
    """遗忘度（工程简化，溯源 §4.2 标注）：age 越大遗忘越深，召回越多越浅。

    forgetting = clamp01(1 − FORGET_DECAY·age_days + FORGET_RECALL_BOOST·recall_count)
    参数 λ/r 无论文定标——工程假设，验收中标定。
    """
    now = now if now is not None else time.time()
    try:
        age_days = 0.0
        last = getattr(entry, 'last_recalled_at', None)
        if last is not None:
            age_days = max(0.0, (now - float(last)) / 86400.0)
        recall_count = int(getattr(entry, 'recall_count', 0) or 0)
        v = 1.0 - FORGET_DECAY * age_days + FORGET_RECALL_BOOST * recall_count
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return 1.0


def select_with_low_confidence_quota(
    scored: Sequence[Tuple[float, object]],
    k: int,
    quota: int = 1,
    gate_ratio: float = 0.5,
    low_threshold: float = LOW_CONFIDENCE_THRESHOLD,
) -> List[Tuple[float, object]]:
    """top-k 检索结果上应用低置信复核配额（relevance-gated，只读）。

    在按 cue_sim 降序的 scored 列表中：
      1. 正常取 top-k；
      2. 若存在低置信（confidence < low_threshold）且 cue_sim ≥
         gate_ratio × top1 分数的待复核条目，则用其中 cue_sim 最高者
         替换末位（占 1 个配额席位）；否则宁缺毋滥（不加塞）。

    专注化：配额只给"当前话题相关的待复核条目"——怀疑不变成跑偏源。

    参数:
        scored: [(score, entry), ...]（score 即 cue_sim，已降序）。
        k: 目标条数。
        quota: 最多为低置信复核保留的席位（默认 1）。
        gate_ratio: 相关性门控比例（默认 0.5 × top1）。
        low_threshold: 低置信阈值（默认 0.3）。

    返回:
        新的条目列表（长度 ≤ k，顺序保持按 score 降序）。
    """
    if not scored or k <= 0:
        return list(scored)[:k]
    top_k = list(scored[:k])
    if len(scored) <= k:
        return top_k
    top1_score = float(scored[0][0])
    gate = gate_ratio * top1_score if top1_score > 0 else 0.0

    # 找可占席的低置信待复核条目（相关性门控）
    candidates = []
    for score, entry in scored:
        try:
            conf = float(getattr(entry, 'confidence', 1.0) or 1.0)
        except (TypeError, ValueError):
            conf = 1.0
        if conf < low_threshold and float(score) >= gate:
            candidates.append((float(score), entry))
    if not candidates:
        return top_k

    # 用相关性最高的低置信条目替换末位（最多 quota 席）
    candidates.sort(key=lambda x: x[0], reverse=True)
    for _ in range(min(quota, len(candidates))):
        replaced = False
        for i in range(len(top_k) - 1, -1, -1):
            if top_k[i] not in candidates and candidates[0] not in top_k:
                top_k[i] = candidates.pop(0)
                replaced = True
                break
        if not replaced:
            break
    top_k.sort(key=lambda x: x[0], reverse=True)
    return top_k

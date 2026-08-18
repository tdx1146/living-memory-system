# -*- coding: utf-8 -*-
"""PER 概率采样重放（提取层 v1.4 S1-6，拍板④/M3）

严格照 Prioritized Experience Replay（Schaul et al. 2015, arXiv:1511.05952）：

  论文公式（照搬）：
    P(i) = p_i^α / Σ_k p_k^α
    p_i = |δ_i| + ε
    α ∈ [0,1]，论文经验默认 α=0.6

  v1.4 映射（不发明，只映射）：
    |δ_i|（预测误差）→ replay_score(i) = surprise_norm(i) × info_value_norm(i)
        —— LMS surprise 就是 FEP 预测误差（process_turn 原生量），映射同构；
           info_value 按 §6.2 论文重做（0.7×surprise_norm + 0.3×recall_hit）
    ε → 1e-6（v1.4 标注：自选小常数，论文仅说"小常数"）
    α → 0.6（论文默认值，env 可配 LMS_DREAM_REPLAY_ALPHA）
    采样 → 无放回概率采样取 k 条（v1.4 标注：无放回为工程决策——论文
        1511.05952 为有放回概率抽样 sum-tree；表述修正为"公式照搬、
        采样细节为工程决策"）

候选集：episodic 全量（排除 gray 条目，source=='store_gray'，灰度三重冻结①）。

调用关系（v1.4 §附 #5 定案）：dream_replay = dream_engine 的采样子模块
（被 dream_cycle 调用，非替代关系）；本模块为**无状态纯函数**（输入
episodic 条目列表 + 参数 → 输出采样集＋打分，不含记忆/循环/锁）。

重放后优先级更新（论文：新优先级 = max(旧, 新误差)）：由 dream_engine
对被采中条目做加固（reference_count+1、last_reinforced_turn 刷新，
S1-13 写点）承担——下次打分时 effective_wear 重置 → score 自然更新。
"""

import os
import random
from typing import List, Tuple, Optional

# ε 保底（自选小常数；论文仅说"小常数"）
EPSILON = 1e-6
# 默认采样参数（env 可配）
DEFAULT_REPLAY_K = 20
DEFAULT_REPLAY_ALPHA = 0.6
# 灰度条目来源标记（三重冻结①：不参与重放）
GRAY_SOURCE = "store_gray"


def replay_k() -> int:
    """采样条数 k（LMS_DREAM_REPLAY_K，默认 20）。"""
    return max(1, int(os.environ.get("LMS_DREAM_REPLAY_K", "20") or 20))


def replay_alpha() -> float:
    """PER 指数 α（LMS_DREAM_REPLAY_ALPHA，默认 0.6=论文经验值）。"""
    try:
        a = float(os.environ.get("LMS_DREAM_REPLAY_ALPHA", "0.6") or 0.6)
    except (TypeError, ValueError):
        a = 0.6
    return max(0.0, min(1.0, a))


def _candidates(entries) -> List:
    """候选集 = episodic 全量排除 gray（灰度三重冻结①）。"""
    return [e for e in entries
            if getattr(e, "source", "external") != GRAY_SOURCE]


def _minmax_norm(vmax: float, value: float) -> float:
    """min-max 归一化（vmax 为参照总体的最大值；vmax≤0 → 0，防除零）。"""
    if vmax <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, value / vmax))


def replay_score(entry, surprise_max: float, info_max: float) -> float:
    """单条重放分数：p_i = surprise_norm × info_value_norm + ε。

    无时间项（去时间偏好）；低分条目永远有 ε 保底非零概率（真防饿死）。
    """
    surprise = float(getattr(entry, "surprise", 0.0) or 0.0)
    info_value = float(getattr(entry, "info_value", 0.0) or 0.0)
    s_norm = _minmax_norm(surprise_max, surprise)
    i_norm = _minmax_norm(info_max, info_value)
    return s_norm * i_norm + EPSILON


def _compute_scores(cands: List) -> Tuple[List[float], float, float]:
    """计算候选集统一归一化基准与逐条分数。

    返回:
        (scores, surprise_max, info_max)——min-max 基准取候选集全局最大，
        保证跨条目可比（逐条局部归一化会破坏相对排序）。
    """
    surprise_max = max(
        (float(getattr(e, "surprise", 0.0) or 0.0) for e in cands),
        default=0.0)
    info_max = max(
        (float(getattr(e, "info_value", 0.0) or 0.0) for e in cands),
        default=0.0)
    scores = [replay_score(e, surprise_max, info_max) for e in cands]
    return scores, surprise_max, info_max


def sample_replay_set(
    entries,
    k: Optional[int] = None,
    alpha: Optional[float] = None,
    rng: Optional[random.Random] = None,
) -> List[Tuple]:
    """PER 无放回概率采样 k 条（无状态纯函数，S1-6 边界）。

    参数:
        entries: episodic 条目可迭代对象（EpisodicEntry 列表）。
        k: 采样条数（None → LMS_DREAM_REPLAY_K，默认 20）。
        alpha: PER 指数（None → LMS_DREAM_REPLAY_ALPHA，默认 0.6）。
        rng: 随机源（测试注入；None → 模块 random，线程安全由调用方保证）。

    返回:
        [(entry, score, prob), ...]（按采样顺序，无放回；k=min(k, 候选数)）。
        候选集为空（无条目或全 gray）时返回空列表。

    随机性验证（防饿死反例，v1.4 判据）：低分条目 P(i)≥0.01 时，
    N=500 次做梦被采中 ≥1 次的概率 ≥ 1-(1-0.01)^500 ≈ 99.3%。
    """
    if k is None:
        k = replay_k()
    if alpha is None:
        alpha = replay_alpha()
    rng = rng or random
    cands = _candidates(entries)
    if not cands:
        return []
    k = max(0, min(int(k), len(cands)))
    if k == 0:
        return []

    scores, _sm, _im = _compute_scores(cands)
    # P(i) = p_i^α / Σ p_j^α（论文公式照搬）
    powered = [max(s, 0.0) ** alpha for s in scores]
    total = sum(powered)
    if total <= 1e-300:
        # 全零退化（理论上 ε 保底不会发生）：退化为均匀
        probs = [1.0 / len(powered)] * len(powered)
    else:
        probs = [p / total for p in powered]

    # 无放回概率采样（工程决策，标注见模块 docstring）
    remaining = list(range(len(cands)))
    result = []
    for _ in range(k):
        if not remaining:
            break
        # 用剩余条目的归一化概率采样一个索引
        rem_probs = [probs[i] for i in remaining]
        s = sum(rem_probs)
        if s <= 1e-300:
            idx = rng.choice(remaining)
        else:
            idx = rng.choices(
                remaining, weights=[p / s for p in rem_probs], k=1)[0]
        remaining.remove(idx)
        result.append((cands[idx], scores[idx], probs[idx]))
    return result


def sampled_ids(replay_set: List[Tuple]) -> List[int]:
    """采样集条目 id 列表（观测用；无 id 时用 turn 兜底）。"""
    ids = []
    for entry, _score, _prob in replay_set:
        eid = getattr(entry, "id", None)
        if eid is None:
            eid = getattr(entry, "turn", None)
        ids.append(eid)
    return ids

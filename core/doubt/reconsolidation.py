"""惊讶度双角色-角色2（体验层 D2，设计 v1.1 §6.2）—— 去稳定化。

角色 1（学习信号）：purpose/memory/做梦采样/元可塑性**全部不动**（红线）。
角色 2（去稳定化）：近 200 轮 surprise 窗口（G1 思路，LMS 内自有 deque），
surprise > mean + 2·std → 定位**本轮检索命中条目中与输入冲突最强/置信
最高者** → mark_labile(violated_by=…)。

专注化强调：被标记的是"被当前输入违反的**旧记忆**"（已关注方向里的
证伪内容，Schiller 2010 B2 / Sinclair & Barense 2018 B1），**不是**
"值得探索的新方向"。

论文钉子：母本 eq.19 无 PE→零更新，同构 Schiller 2010（溯源 §三 机制2）。
z>2 阈值与 200 轮窗口为工程假设（溯源 §4.2），复用 G1 窗口经验。
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple


def detect_destabilization(
    surprise: float, window: Sequence[float],
    z_threshold: float = 2.0, min_window: int = 20,
) -> Tuple[bool, Optional[float]]:
    """高惊讶检测：surprise > mean + 2·std → (True, z)。

    窗口不足 min_window 条或 σ≈0 时不判（fail-open 保守，同 salience_gate
    _z_flag 哲学）。返回 (是否去稳定化, z 值或 None)。
    """
    if len(window) < min_window:
        return False, None
    mean = sum(window) / len(window)
    std = (sum((v - mean) ** 2 for v in window) / len(window)) ** 0.5
    if std < 1e-8:
        return False, None
    z = (surprise - mean) / std
    return z > z_threshold, float(z)


def _entry_confidence(entry) -> float:
    """条目置信度（缺字段回退 1.0，fail-open）。"""
    try:
        return float(getattr(entry, 'confidence', 1.0) or 1.0)
    except (TypeError, ValueError):
        return 1.0


def find_violated_entry(
    scored: Sequence[Tuple[float, object]],
) -> Optional[object]:
    """定位本轮检索命中条目中"最被违反"的旧记忆。

    工程近似（溯源 §4.2 标注）：语义矛盾检测无 LLM 依赖，取命中条目中
    **置信度最高**者（最 established 的旧记忆被当前输入违反的破坏性最大），
    置信度相同时取 cue_sim 最高者（最被当前输入直接唤起的记忆）。

    参数:
        scored: [(score, EpisodicEntry), ...]（按相似度降序）。

    返回:
        命中的条目；空输入返回 None。
    """
    if not scored:
        return None
    return max(scored, key=lambda x: (_entry_confidence(x[1]), x[0]))[1]


def mark_labile(entry, violated_by: Optional[str] = None,
                now: Optional[float] = None) -> bool:
    """标记条目为 labile（去稳定化窗口打开）。

    副作用（三路汇入②）：rebuttal_count +1、violated_by 记录、置信度重算；
    M3-1（规格 v2 §2.3）：证伪同时写 rebuttal-consistency 原生字段
    （``updated_by='ingest'``——写侧时相合法写者；检索只读铁律由
    rebuttal_field 写者守卫 + guard.py 四不变双保）。
    fail-open：异常返回 False，不阻塞调用方。

    返回:
        True 标记成功；False 异常/无条目。
    """
    if entry is None:
        return False
    try:
        from core.doubt.rebuttal_field import record_rebuttal_native
        entry.labile = True
        entry.labile_since = now if now is not None else time.time()
        # 证伪标记（写侧时相）：平坦字段（mark_rebutted 既有行为逐字节
        # 保留）+ 结构化原生字段同步（rebuttals/consistency/updated_by）
        record_rebuttal_native(entry, violated_by=violated_by,
                               updated_by='ingest')
        return True
    except Exception:
        return False


def resolve_labile(
    entry, now: Optional[float] = None,
    window_seconds: float = 86400.0,
    reconsolidate_gain: float = 1.02, decay: float = 0.98,
) -> str:
    """labile 窗口裁决（做梦 doubt_review 阶段① 调用，纯函数）。

    三种结局（P1 §2.3 原样；显式裁决流程为"睡眠=整合/复核窗口"的工程
    翻译，溯源 §4.2 标注）：
      - 有证伪证据（violated_by 在场）→ 'rewritten'：返回改写指令
        （由调用方新增 source='doubt' supersedes 条目；本函数不落库）
      - 无证据且未超时 → 'kept'：confidence ×1.02 重巩固（乘子保守）
      - 超时（labile 窗口关闭，默认 24h）→ 'downgraded'：confidence ×0.98
        折损

    返回:
        'rewritten' | 'kept' | 'downgraded'（已对条目应用乘子/复位）。
    """
    if entry is None:
        return 'kept'
    now = now if now is not None else time.time()
    try:
        violated_by = getattr(entry, 'violated_by', None)
        since = getattr(entry, 'labile_since', None)
        if violated_by:
            # 改写：复位 labile（实际 supersedes 条目由调用方落库）
            entry.labile = False
            entry.labile_since = None
            return 'rewritten'
        if since is not None and (now - since) > window_seconds:
            # 超时无证据 → 折损（轻微，保守）
            entry.confidence = max(0.0, min(
                1.0, getattr(entry, 'confidence', 1.0) * decay))
            entry.labile = False
            entry.labile_since = None
            return 'downgraded'
        # 窗口内无证据 → 重巩固（轻微增益，保守）
        entry.confidence = max(0.0, min(
            1.0, getattr(entry, 'confidence', 1.0) * reconsolidate_gain))
        entry.labile = False
        entry.labile_since = None
        return 'kept'
    except Exception:
        return 'kept'

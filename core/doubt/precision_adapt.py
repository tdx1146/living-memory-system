"""precision 三层动态化（阶段 3：质疑自动校准）—— 核心模块。

设计依据（8/13 设计定稿 §四 质疑自动校准 + 8/13 课题 C 调研，全部论文
经核验）：dandan 拍板"要机制自动校准，不要固定值"——判定阈值（"这条
该被怀疑吗"）必须随经验分布动；机制参数（EMA 学习率/窗口/分位百分点/
混合权重）允许默认值并受治理开关保护。

三层 precision 结构（LMS 目的层扩展）：
  - 条目级：adaptive_confidence（基础公式 + Koriat 自一致性混合 +
    反流畅折扣）——存于 EpisodicEntry.confidence 等字段
  - 域级：compute_domain_precision（召回簇聚合：域反驳率/域置信度/
    波动性代理；suspicion = 域反驳率在全局分布中的分数秩）
  - 全局级：PrecisionAdaptState（HGF 波动性 → doubt_baseline 全局怀疑
    强度；conformal 分位怀疑线）

四机制（零固定阈值）：
  1. HGF 波动性调制（Mathys 2011 / Behrens 2007）：volatility = 惊讶
     序列**尾窗标准差**（环境波动性水平；Behrens 2007：波动高→旧证据
     权重低→更新快→怀疑升），EMA 平滑；doubt_baseline = volatility 在
     自身历史中的**分数秩**（Mann-Whitney U 风格，稳定序列 → 0.5 中性；
     波动↑ → rank↑ → 怀疑↑ → precision↓）。
     注：曾用"变化率（|Δs| EMA）"方案——交替模式被感知层 EMA 适应后
     vol 恒定 → rank 中性（模拟实证发现，已弃）；水平语义更贴合 Behrens。
  2. Koriat 自一致性（2012）：召回时计算——同主题簇（两两相似度 ≥
     簇内相似度分布 P50 自适应线）成员的平均相似度 = consistency；
     adaptive_confidence = (1−w)·基础置信度 + w·一致性。工程同构
     SelfCheckGPT（8/13 C2）；用向量余弦度量"印证"，非语义矛盾检测
     （无 LLM，工程近似标注）。
  3. conformal 分位阈值（2107.07511 / Youden 1950）：怀疑线 = 被反驳
     条目**反驳前置信度**分布的高分位（默认 P85）。冷启动（校准样本
     < 20）回退初始线 0.3（允许的初始值，样本足够后完全由分布决定）。
  4. 对称性约束（Sharot 2011 C2 / Hasher 1977 D2）：
     - 坏消息 PE 不被系统性低估：负性惊讶 ≥ 正性惊讶中位（更新不对称
       补偿）；负性证据比例（近 50 轮 rebuttal 占比）经分数秩调制
       doubt_baseline——反驳会推高全局怀疑，不因"已熟悉"被压掉。
     - 重复曝光降权：verdict_confidence 对 reference/recall_count 做
       log1p 边际递减（重复≠真）；memory.py consolidate 另有回放权重
       对抗项（_repetition_factor，本模块外）。

治理开关：LMS_PRECISION_ADAPT（默认 1=开；0=关 → 全部路径零参与，
行为与开关引入前完全一致）。风格对齐 8/10 先例（LMS_NORM_SURPRISE /
LMS_J_TARGET_NORM）。状态为纯进程内存（同 gap_registry 先例，重启
即失，快照不落盘——回滚干净）。

fail-open：本模块所有方法异常静默（不阻塞调用方主路径）。

ABC §S3「B 落地（precision 学习化）」新增（四妹-ABC操作规划-20260817.md
§S3）：`PrecisionLearnState` —— **π = 1/Var(surprise) 滑动窗口估计**
（EMA 低通），与 baseline（`PrecisionAdaptState`，全局怀疑强度）**并行**、
独立治理开关（LMS_PRECISION_LEARN，默认 0=关；关 → 零参与，行为与开关
引入前完全一致）。理论依据：Ofner 2021「precision 加权 = natural
gradient」；PredProp 2021 直接实现先例。loop learn 侧 `lr_multiplier`
跟随 π 的接线由管理者统一处理（预留接口点：loop.py:631 `attractor.learn`
的 effective_lr 乘 `lr_multiplier()`），本模块只提供估计器，不改快照字段
名/语义（P1-4 影响面兼容性要求）。

π̄ 估计（数学定义，详见 PrecisionLearnState docstring）:
    var_raw(t)  = (1/N)·Σ_{i∈尾窗} (s_i − mean(尾窗))²   # 尾窗样本方差
    var_ema(t)  = (1−α)·var_ema(t−1) + α·var_raw(t)      # EMA 低通（α=LMS_PRECISION_VAR_EMA）
    π̄(t)       = clamp(1/var_ema(t), MIN, MAX)
                  （var_ema 退化 = 0/非有限（如全等序列）→ MIN 保护——
                    源规格原文：「var 过小（如全等序列）→ 用 MIN 保护」）
    lr_multiplier = π̄（effective_lr = prev_lr × π̄；π̄=1 → 不变）

env 参数表（**LMS_PRECISION_* 为唯一权威**；显式构造参数 > env > 默认值）:
    LMS_PRECISION_LEARN       默认 0（主开关；1/true/yes/on 视为开）
    LMS_PRECISION_VAR_WINDOW  默认 200（尾窗长度）
    LMS_PRECISION_VAR_EMA     默认 0.02（方差低通学习率）
    LMS_PRECISION_VAR_MIN     默认 0.05（π 下钳）
    LMS_PRECISION_VAR_MAX     默认 5.0 （π 上钳）
"""

from __future__ import annotations

import math
import os
from collections import deque
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from core.doubt.confidence_field import (
    FORCE_DOWNGRADE_REBUTTALS,
    LOW_CONFIDENCE_THRESHOLD,
    compute_confidence,
)


# ================================================================== #
#  治理开关
# ================================================================== #

def precision_adapt_enabled(explicit: Optional[bool] = None) -> bool:
    """治理开关解析：显式参数 > 环境变量 LMS_PRECISION_ADAPT（默认 1=开）。

    布尔接受（不区分大小写）：1/true/yes/on 视为开，其余为关。
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get('LMS_PRECISION_ADAPT', '1')
    return raw.strip().lower() not in ('0', 'false', 'no', 'off')


def precision_learn_enabled(explicit: Optional[bool] = None) -> bool:
    """治理开关解析：显式参数 > 环境变量 LMS_PRECISION_LEARN（默认 0=关）。

    ABC §S3「B 落地（precision 学习化）」主开关：默认关（零参与，行为与
    开关引入前完全一致）；开 → π = 1/Var(surprise) 估计 + lr_multiplier
    跟随 π。布尔接受同 precision_adapt_enabled 先例（1/true/yes/on 为开）。
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get('LMS_PRECISION_LEARN', '0')
    return raw.strip().lower() not in ('0', 'false', 'no', 'off')


def _env_float(name: str, default: float) -> float:
    """env 浮点参数读取（缺失/非法 → 默认值，fail-open 风格）。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    """env 整数参数读取（缺失/非法 → 默认值；容忍 "200.0" 形态）。"""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(float(raw.strip()))
    except (TypeError, ValueError):
        return default


# ================================================================== #
#  纯函数：分位工具（零固定阈值的基础件）
# ================================================================== #

def percentile(sorted_vals: Sequence[float], p: float) -> float:
    """线性插值分位数（sorted_vals 已升序；p ∈ [0,1]）。

    空序列抛 ValueError（调用方负责样本充足性检查）。
    """
    if not sorted_vals:
        raise ValueError('percentile: 空序列')
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f))


def fractional_rank(values: Sequence[float], x: float) -> float:
    """x 在 values 中的分数秩（Mann-Whitney U 风格，平局折半）。

    - 稳定序列（全部相等）→ 0.5（中性，不误报"高度怀疑"）
    - 高于常态 → >0.5；低于常态 → <0.5
    - 空序列 → 0.5（中性，fail-open）

    这是"零固定阈值"的核心：怀疑强度/域怀疑 = 当前值在自身经验分布
    中的秩，随数据流自动漂移，无任何拍脑袋常数。
    """
    if not values:
        return 0.5
    below = sum(1 for v in values if v < x)
    equal = sum(1 for v in values if v == x)
    return (below + 0.5 * equal) / len(values)


# ================================================================== #
#  条目级：Koriat 自一致性（召回时计算，纯函数）
# ================================================================== #

def _entry_vector(entry):
    """条目语义向量（CPU float 1-D）；缺失/异常 → None（fail-open）。"""
    try:
        v = getattr(entry, 'semantic_vector', None)
        if v is None:
            return None
        return v.detach().cpu().float().reshape(-1)
    except Exception:
        return None


def _pairwise_cosines(cohort) -> Tuple[Optional[List[List[float]]], int]:
    """簇内两两余弦相似度矩阵（兼容混合维度条目：维度不匹配对跳过）。

    返回 (sims, n)；任一条目无向量 → (None, 0)（整体放弃，fail-open）。
    """
    normed = []
    for _, entry in cohort:
        v = _entry_vector(entry)
        if v is None:
            return None, 0
        norm = float(v.norm().item())
        if norm < 1e-8:
            return None, 0
        normed.append(v / norm)
    n = len(normed)
    sims = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if normed[i].shape != normed[j].shape:
                continue  # 维度不匹配对跳过（混合 384/64 维缓冲区）
            s = float(normed[i].dot(normed[j]).item())
            sims[i][j] = sims[j][i] = s
    return sims, n


def compute_consistency(scored: Sequence[Tuple[float, object]],
                        top_n: int = 10) -> Dict[int, float]:
    """Koriat 自一致性（召回时计算，纯函数）。

    以本次召回簇（top_n 条）为"抽样总体"（Koriat 2012：置信度 = 从
    表征总体抽样、答案在抽样间的一致性）：条目与"同主题簇"成员（两两
    相似度 ≥ 自适应线 = 簇内相似度分布 P50）的平均相似度即一致性。

    - 无同簇成员（孤立条目）→ 不写入该条目（None → 只用基础置信度，
      向后兼容）
    - 一致性高 = 同主题多条记忆互相印证 → 置信度上升；
      一致性低 = 簇内分歧 → 置信度下调（Koriat 自一致性模型语义）

    工程近似标注（溯源 §4.2 风格）：向量余弦度量"印证"（同主题条目
    互相印证程度），非语义矛盾检测（无 LLM）；自适应线用簇内分布 P50
    ——零固定阈值。

    参数:
        scored: [(score, entry), ...]（按与 query 相似度降序）。
        top_n: 参与抽样的簇大小上限（设计 B2：检索 ≤8-10 条）。

    返回:
        {id(entry): consistency}（仅对有同簇成员的条目）。
    """
    if not scored:
        return {}
    cohort = list(scored)[:top_n]
    if len(cohort) < 2:
        return {}
    sims, n = _pairwise_cosines(cohort)
    if sims is None or n < 2:
        return {}
    # 自适应线：簇内成对相似度（上三角）的 P50
    tri = [sims[i][j] for i in range(n) for j in range(i + 1, n)]
    if not tri:
        return {}
    try:
        thr = percentile(sorted(tri), 0.5)
    except ValueError:
        return {}
    out: Dict[int, float] = {}
    for i, (_, entry) in enumerate(cohort):
        members = [sims[i][j] for j in range(n)
                   if j != i and sims[i][j] >= thr]
        if members:
            out[id(entry)] = sum(members) / len(members)
    return out


# ================================================================== #
#  域级：召回簇聚合（纯函数）
# ================================================================== #

def compute_domain_precision(entries: Sequence[object]) -> dict:
    """域级置信度（召回簇聚合，纯函数）。

    域 = 本次召回簇（工程近似：当前主题域，如"小说/系统/生活"——LMS
    无主题标注，用检索邻域作为域的操作化定义）。

    聚合输出:
        - confidence: 域内条目基础置信度均值（域置信度）
        - rebuttal_rate: 域反驳率 = Σrebuttal / (Σreference + 1)
        - surprise_dispersion: 域内条目惊讶的变异系数（域波动性代理）
        - suspicion: 域怀疑强度 = 域反驳率在全局反驳率分布中的分数秩
          （调用方注入全局分布；本函数置 0.5 占位）

    依据：域层波动性（Behrens 2007 C4：波动高 → 该域所有条目怀疑基准
    上调）；来源维度（Johnson 1993 / Sperber 2010，8/11 转引）。
    """
    if not entries:
        return {'confidence': 0.5, 'rebuttal_rate': 0.0,
                'surprise_dispersion': 0.0, 'suspicion': 0.5}
    rebut = sum(int(getattr(e, 'rebuttal_count', 0) or 0) for e in entries)
    ref = sum(int(getattr(e, 'reference_count', 0) or 0) for e in entries)
    confs: List[float] = []
    surprises: List[float] = []
    for e in entries:
        try:
            confs.append(compute_confidence(
                getattr(e, 'rebuttal_count', 0),
                getattr(e, 'reference_count', 0),
                getattr(e, 'source_trust', 1.0)))
        except (TypeError, ValueError):
            pass
        try:
            surprises.append(float(getattr(e, 'surprise', 0.0) or 0.0))
        except (TypeError, ValueError):
            pass
    mean_conf = sum(confs) / len(confs) if confs else 0.5
    mean_s = sum(surprises) / len(surprises) if surprises else 0.0
    dispersion = 0.0
    if len(surprises) > 1 and mean_s > 1e-8:
        var = sum((s - mean_s) ** 2 for s in surprises) / len(surprises)
        dispersion = math.sqrt(var) / mean_s
    return {
        'confidence': round(max(0.0, min(1.0, mean_conf)), 4),
        'rebuttal_rate': round(rebut / (ref + 1), 4),
        'surprise_dispersion': round(dispersion, 4),
        'suspicion': 0.5,  # 占位：调用方用全局分布分数秩覆盖
    }


# ================================================================== #
#  全局级：HGF 波动性调制 + conformal 分位怀疑线（状态机）
# ================================================================== #

class PrecisionAdaptState:
    """全局怀疑强度（precision 基线）+ 校准集 + 观测窗口。

    纯进程内存状态（同 gap_registry 先例：重启即失、快照不落盘、
    回滚干净）。enabled=False 时所有方法为 no-op / 中性值（零参与，
    行为与开关引入前完全一致）。

    机制参数（可默认值；非判定阈值）:
        lr_volatility: 波动层 EMA 学习率（波动性水平平滑）。
        vol_window: 惊讶尾窗长度（环境波动性 = 窗内标准差）。
        window: 各经验分布窗口长度。
        min_samples: doubt_baseline 冷启动样本数（不足 → 中性 0.5）。
        suspect_quantile: conformal 怀疑线分位点（默认 P85）。
        min_calibration_samples: 校准集样本数（不足 → 回退初始线 0.3）。
        consistency_weight: 自一致性混合权重 w（adaptive_confidence）。
        fluency_decay: 反流畅折扣系数（verdict_confidence 的 log1p 项）。
        neg_weight: 负性证据比例对 doubt_baseline 的调制权重。
    """

    def __init__(self, enabled: Optional[bool] = None,
                 lr_volatility: float = 0.1,
                 vol_window: int = 50,
                 window: int = 200,
                 min_samples: int = 30,
                 suspect_quantile: float = 0.85,
                 min_calibration_samples: int = 20,
                 consistency_weight: float = 0.5,
                 fluency_decay: float = 0.2,
                 neg_weight: float = 0.3) -> None:
        # 治理开关：显式参数 > 环境变量 LMS_PRECISION_ADAPT（默认 1=开）
        self.enabled = precision_adapt_enabled(enabled)
        self.lr_volatility = lr_volatility
        self.vol_window = int(vol_window)
        self.window = int(window)
        self.min_samples = int(min_samples)
        self.suspect_quantile = float(suspect_quantile)
        self.min_calibration_samples = int(min_calibration_samples)
        self.consistency_weight = float(consistency_weight)
        self.fluency_decay = float(fluency_decay)
        self.neg_weight = float(neg_weight)

        # 波动性状态：EMA 平滑的尾窗标准差（Behrens 2007 水平语义）
        self.volatility: Optional[float] = None
        self._vol_ema: Optional[float] = None

        # 经验分布（观测 + 校准）
        self.surprise_history: Deque[float] = deque(maxlen=self.window)
        self.volatility_history: Deque[float] = deque(maxlen=self.window)
        self.baseline_history: Deque[float] = deque(maxlen=self.window)
        self.pos_surprises: Deque[float] = deque(maxlen=self.window)
        self.neg_surprises: Deque[float] = deque(maxlen=self.window)
        self._neg_flags: Deque[bool] = deque(maxlen=50)
        self.neg_ratio_history: Deque[float] = deque(maxlen=self.window)
        # conformal 校准集：被反驳条目**反驳前**置信度
        self.rebuttal_before_confidences: Deque[float] = deque(
            maxlen=self.window)
        # 召回观测：判定置信度窗口 + 域反驳率分布 + 怀疑率
        self.confidence_window: Deque[float] = deque(maxlen=self.window)
        self.domain_rebuttal_rates: Deque[float] = deque(maxlen=self.window)
        self.verdict_history: Deque[float] = deque(maxlen=self.window)
        # 一致性缓存（召回时计算，id(entry) → consistency）
        self.consistency_cache: Dict[int, float] = {}
        self.last_domain: Optional[dict] = None

    # ------------------------------------------------------------------ #
    #  冷启动
    # ------------------------------------------------------------------ #

    def is_cold(self) -> bool:
        """全局层冷启动：波动性经验不足（doubt_baseline 尚未进入分布态）。"""
        return len(self.volatility_history) < self.min_samples

    # ------------------------------------------------------------------ #
    #  逐试次更新：HGF 波动性 + 对称性约束（主入口）
    # ------------------------------------------------------------------ #

    def observe_surprise(self, surprise: float,
                         is_negative: bool = False) -> None:
        """每轮惊讶观测（process_turn 调用）。

        对称性约束（Sharot 2011 C2）：坏消息（反驳/纠正轮）的 PE 不得
        被系统性低估——负性惊讶低于正性中位时抬升到正性中位（更新
        不对称补偿，零固定阈值：比较基准是正性分布中位，非常数）。

        波动性估计（Behrens 2007 水平语义，工程简化）:
            vol_raw = std(惊讶尾窗 vol_window 条)   # 环境波动性水平
            vol = EMA(vol_raw)                       # 平滑
        """
        if not self.enabled:
            return
        surprise = float(surprise)
        # 对称性约束：坏消息 PE 不低于正性中位
        if is_negative and self.pos_surprises:
            try:
                pos_median = percentile(sorted(self.pos_surprises), 0.5)
                surprise = max(surprise, pos_median)
            except ValueError:
                pass
        self.surprise_history.append(surprise)
        (self.neg_surprises if is_negative
         else self.pos_surprises).append(surprise)

        # 波动性：尾窗标准差（水平语义；样本 < 2 时 0）
        win = list(self.surprise_history)[-self.vol_window:]
        if len(win) >= 2:
            mean_s = sum(win) / len(win)
            var = sum((v - mean_s) ** 2 for v in win) / len(win)
            vol_raw = math.sqrt(var)
        else:
            vol_raw = 0.0
        # EMA 平滑（防单轮抖动；lr_volatility 机制参数）
        if self._vol_ema is None:
            self._vol_ema = vol_raw
        else:
            self._vol_ema += self.lr_volatility * (vol_raw - self._vol_ema)
        self.volatility = self._vol_ema
        self.volatility_history.append(self.volatility)

        # 负性证据比例（近 50 轮 rebuttal 占比）→ 怀疑基线调制
        self._neg_flags.append(bool(is_negative))
        ratio = (sum(1 for f in self._neg_flags if f)
                 / len(self._neg_flags))
        self.neg_ratio_history.append(ratio)

        self.baseline_history.append(self.doubt_baseline())

    # ------------------------------------------------------------------ #
    #  全局怀疑强度（precision 基线）
    # ------------------------------------------------------------------ #

    def doubt_baseline(self) -> float:
        """全局怀疑强度 [0,1]（precision 基线，随环境波动性漂移）。

        baseline = (1−w_neg)·rank(volatility) + w_neg·rank(neg_ratio)

        - rank = 分数秩（自身历史分布内，零固定阈值）
        - 波动↑ → rank↑ → 怀疑↑（precision↓）；环境稳定 → 0.5 中性
        - 负性证据比例高（近期反驳多）→ 怀疑额外上调（对称性约束）
        - 冷启动（样本不足）→ 0.5 中性
        """
        if not self.enabled:
            return 0.5
        if len(self.volatility_history) < self.min_samples:
            return 0.5
        vol_rank = fractional_rank(
            list(self.volatility_history), self.volatility_history[-1])
        base = vol_rank
        if len(self.neg_ratio_history) >= self.min_samples:
            r = self.neg_ratio_history[-1]
            r_rank = fractional_rank(list(self.neg_ratio_history), r)
            base = (1.0 - self.neg_weight) * vol_rank + self.neg_weight * r_rank
        return max(0.0, min(1.0, base))

    def global_precision(self) -> float:
        """全局信任度 = 1 − doubt_baseline（波动↑ → precision↓ 的直接映射）。

        注意：这是"整体怀疑强度/信任基线"，与 purpose.sensory_precision
        （管"关注哪"，Pouget 2016：confidence≠precision）正交——本值
        是怀疑语义的 precision 基线，供质疑层/注入/观测使用，不参与
        attractor.infer 的感官 clamping（红线：purpose precision 不动）。
        """
        return 1.0 - self.doubt_baseline()

    # ------------------------------------------------------------------ #
    #  conformal 分位怀疑线（判定阈值，随经验分布动）
    # ------------------------------------------------------------------ #

    def doubt_threshold(self) -> float:
        """conformal 分位怀疑线："这条该被怀疑吗"的判定阈值。

        线 = 被反驳条目**反驳前置信度**分布的高分位（默认 P85）——
        新条目置信度低于此线，意味着它处于历史上 85% 最终被证伪条目的
        置信度区间 → 该被怀疑（Youden 1950：阈值随分布/代价移动）。

        - 冷启动（校准样本 < min_calibration_samples）→ 回退初始线
          LOW_CONFIDENCE_THRESHOLD（0.3，允许的初始值；样本足够后
          完全由经验分布决定——零固定阈值原则）
        - 校准集随反驳事件流漂移：反驳密集期（坏消息多）线自动上移，
          怀疑更积极；长期无反驳线自动下移，怀疑收敛
        """
        if not self.enabled:
            return LOW_CONFIDENCE_THRESHOLD
        calib = sorted(self.rebuttal_before_confidences)
        if len(calib) < self.min_calibration_samples:
            return LOW_CONFIDENCE_THRESHOLD
        try:
            return percentile(calib, self.suspect_quantile)
        except ValueError:
            return LOW_CONFIDENCE_THRESHOLD

    # ------------------------------------------------------------------ #
    #  条目级：自适应置信度 + 判定置信度 + 怀疑判定
    # ------------------------------------------------------------------ #

    def adaptive_confidence(self, entry, consistency: Optional[float] = None
                            ) -> float:
        """条目级自适应置信度 = 基础公式 × Koriat 自一致性混合。

        conf_adapt = (1−w)·conf_base + w·consistency（w=0.5 默认）

        - 硬证据不被一致性覆盖：rebuttal ≥ 2（强制 0.1 先例）→ 纯基础
          公式（一致性是软信号，不推翻硬证伪）
        - consistency 缺省（无同簇成员/冷启动）→ 纯基础公式（向后兼容）
        - 一致性高 → 置信度上升（场景②方向）；一致性低 → 置信度下降
          （Koriat 2012：分歧即怀疑）
        """
        conf = compute_confidence(
            getattr(entry, 'rebuttal_count', 0),
            getattr(entry, 'reference_count', 0),
            getattr(entry, 'source_trust', 1.0))
        if consistency is None:
            return conf
        if int(getattr(entry, 'rebuttal_count', 0) or 0) \
                >= FORCE_DOWNGRADE_REBUTTALS:
            return conf
        w = self.consistency_weight
        return max(0.0, min(
            1.0, (1.0 - w) * conf + w * max(0.0, min(1.0, consistency))))

    def verdict_confidence(self, entry,
                           consistency: Optional[float] = None) -> float:
        """判定用置信度 = 自适应置信度 × 反流畅折扣。

        反流畅性偏误对抗（Hasher 1977 / Fazio 2015）：同一记忆被重复
        召回（reference/recall_count 高）不应线性推高信任——log1p 边际
        递减（重复≠真；重复是曝光不是独立抽样，Koriat 的"抽样一致性"
        由 consistency 承担，二者正交）。
        """
        conf = self.adaptive_confidence(entry, consistency)
        if not self.enabled:
            return conf
        n = max(int(getattr(entry, 'reference_count', 0) or 0),
                int(getattr(entry, 'recall_count', 0) or 0))
        if n > 0 and self.fluency_decay > 0:
            conf = conf / (1.0 + self.fluency_decay * math.log1p(n))
        return max(0.0, min(1.0, conf))

    def should_doubt(self, entry,
                     consistency: Optional[float] = None) -> bool:
        """怀疑判定（conformal 分位线）：verdict_confidence < doubt_threshold。

        开关关 → False（零参与，旧路径负责）。
        """
        if not self.enabled:
            return False
        try:
            return (self.verdict_confidence(entry, consistency)
                    < self.doubt_threshold())
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    #  事件钩子：证伪（conformal 校准集 + 负性证据）
    # ------------------------------------------------------------------ #

    def record_rebuttal(self, entry) -> None:
        """证伪事件钩子（mark_labile/conflict 后调用）。

        把条目**反驳前置信度**收入 conformal 校准集（被反驳条目在
        事发前处于什么置信度区间——预测"什么样置信度的条目会被证伪"）。
        触发方负责 _pending_negative_evidence 标记（observe_surprise
        的 is_negative 由此而来）。
        """
        if not self.enabled or entry is None:
            return
        try:
            before = getattr(entry, 'confidence_before_rebuttal', None)
            if before is None:
                before = getattr(entry, 'confidence', 1.0)
            self.rebuttal_before_confidences.append(
                max(0.0, min(1.0, float(before))))
        except (TypeError, ValueError):
            pass

    # ------------------------------------------------------------------ #
    #  召回钩子：一致性缓存 + 域级统计 + 置信度窗口（只读观测）
    # ------------------------------------------------------------------ #

    def record_recall_cohort(self, scored) -> None:
        """召回簇钩子（recall 路径调用；零持久化，只更新进程内观测）。

        - 域级：compute_domain_precision 聚合 + 域反驳率进全局分布 +
          suspicion = 分数秩
        - 置信度窗口：判定置信度进窗口（观测怀疑率）
        - 怀疑率：窗口内低于怀疑线的条目占比（recent_suspect_ratio）
        """
        if not self.enabled or not scored:
            return
        try:
            entries = [e for _, e in scored]
            domain = compute_domain_precision(entries)
            self.domain_rebuttal_rates.append(domain['rebuttal_rate'])
            domain['suspicion'] = round(fractional_rank(
                list(self.domain_rebuttal_rates),
                domain['rebuttal_rate']), 4)
            self.last_domain = domain
            thr = self.doubt_threshold()
            flagged = 0
            for e in entries:
                c = self.verdict_confidence(
                    e, self.consistency_cache.get(id(e)))
                self.confidence_window.append(c)
                if c < thr:
                    flagged += 1
            self.verdict_history.append(flagged / len(entries))
        except Exception:
            pass

    def recent_suspect_ratio(self) -> float:
        """近期召回怀疑率（verdict_history 均值；空 → 0.0）。"""
        if not self.verdict_history:
            return 0.0
        return sum(self.verdict_history) / len(self.verdict_history)

    # ------------------------------------------------------------------ #
    #  观测
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict:
        """观测块（/status precision_adapt、/react reaction.doubt 共用）。

        全部为纯增量字段；开关关 → {'enabled': False}（调用方降级）。
        """
        if not self.enabled:
            return {'enabled': False}
        try:
            return {
                'enabled': True,
                'cold': self.is_cold(),
                'baseline': round(self.doubt_baseline(), 4),
                'global_precision': round(self.global_precision(), 4),
                'volatility': round(self.volatility or 0.0, 4),
                'threshold': round(self.doubt_threshold(), 4),
                'threshold_quantile': self.suspect_quantile,
                'window_n': len(self.surprise_history),
                'calibration_n': len(self.rebuttal_before_confidences),
                'recent_suspect_ratio': round(
                    self.recent_suspect_ratio(), 4),
                'domain_suspicion': round(
                    self.last_domain['suspicion'], 4)
                    if self.last_domain else None,
                'domain_rebuttal_rate': (
                    self.last_domain['rebuttal_rate']
                    if self.last_domain else None),
                'domain_confidence': (
                    self.last_domain['confidence']
                    if self.last_domain else None),
            }
        except Exception:
            return {'enabled': True}


# ================================================================== #
#  ABC §S3：π = 1/Var(surprise) 估计（precision 学习化，与 baseline 并行）
# ================================================================== #

class PrecisionLearnState:
    """π = 1/Var(surprise) 滑动窗口估计（ABC §S3「B 落地（precision 学习化）」）。

    与 PrecisionAdaptState（全局怀疑强度 baseline：管"该不该怀疑"）**并行**
    的独立估计器：本类管"学习步长该多大"——precision 加权 = natural
    gradient（Ofner 2021；PredProp 2021 直接实现先例），供 loop learn 侧
    `lr_multiplier()` 跟随 π。

    数学定义:
        var_raw(t)  = (1/N)·Σ_{i∈尾窗} (s_i − mean(尾窗))²
                      # 尾窗（LMS_PRECISION_VAR_WINDOW 默认 200）样本方差；
                      # 样本 < 2 时方差未定义（EMA 保持，冷启动填充不污染）
        var_ema(t)  = (1−α)·var_ema(t−1) + α·var_raw(t)
                      # EMA 低通（α = LMS_PRECISION_VAR_EMA 默认 0.02）
        π̄(t)       = clamp(1/var_ema(t), MIN, MAX)
                      # MIN/MAX = LMS_PRECISION_VAR_MIN/MAX（默认 0.05/5.0，
                      # 防除零/爆值）。退化方差（=0/非有限，如全等序列）→
                      # MIN 保护——源规格原文：「var 过小（如全等序列）→ 用
                      # MIN 保护」：1/0 无定义，保守低 precision，防自然梯度
                      # 步长放大失控；非退化小方差 → 高 π（钳 MAX）。

    lr_multiplier（自然梯度语义，方向）:
        effective_lr = prev_lr × π̄
            # π̄ = 1 → 倍率 = 1（行为不变）；π̄ 高（环境稳定/惊讶方差小）→
            # 倍率 > 1（预测误差可信，步长放大）；π̄ 低（环境剧烈波动）→
            # 倍率 < 1（步长缩小，防对噪声 PE 过响应）。
        开关关 / 样本不足（冷启动）→ 原样返回 prev_lr（零参与，行为与
        现状完全一致）。

    治理开关：LMS_PRECISION_LEARN（默认 0=关；关 → 零参与：observe 可被
    调用但不记录、estimate 返回 None、lr_multiplier 原样返回、snapshot
    标记 enabled=False）。env 参数表见模块顶部 docstring（LMS_PRECISION_*
    为唯一权威）。状态为纯进程内存（同 PrecisionAdaptState 先例：重启即失、
    快照不落盘、回滚干净）。纯 stdlib（无 torch）。fail-open：所有方法
    异常静默（不阻塞调用方）。
    """

    def __init__(self, enabled: Optional[bool] = None,
                 var_window: Optional[int] = None,
                 ema: Optional[float] = None,
                 pi_min: Optional[float] = None,
                 pi_max: Optional[float] = None,
                 min_samples: int = 5) -> None:
        # 治理开关：显式参数 > 环境变量 LMS_PRECISION_LEARN（默认 0=关）
        self.enabled = precision_learn_enabled(enabled)
        # 机制参数：显式参数 > env（LMS_PRECISION_VAR_*，唯一权威）> 默认值
        self.var_window = max(2, int(
            var_window if var_window is not None
            else _env_int('LMS_PRECISION_VAR_WINDOW', 200)))
        self.ema = min(1.0, max(1e-9, float(
            ema if ema is not None
            else _env_float('LMS_PRECISION_VAR_EMA', 0.02))))
        self.pi_min = float(pi_min if pi_min is not None
                            else _env_float('LMS_PRECISION_VAR_MIN', 0.05))
        self.pi_max = float(pi_max if pi_max is not None
                            else _env_float('LMS_PRECISION_VAR_MAX', 5.0))
        # 参数合法性（fail-safe：非法 env/参数回退默认或保证 min<=max，不抛）
        if not (self.pi_min > 0.0) or not (self.pi_max > 0.0):
            self.pi_min, self.pi_max = 0.05, 5.0
        if self.pi_min > self.pi_max:
            self.pi_min, self.pi_max = self.pi_max, self.pi_min
        self.min_samples = max(1, int(min_samples))

        # 状态：尾窗 surprise + EMA 平滑方差（纯进程内存）
        self._window: Deque[float] = deque(maxlen=self.var_window)
        self._var_ema: Optional[float] = None

    # ------------------------------------------------------------------ #
    #  输入接口
    # ------------------------------------------------------------------ #

    def observe(self, surprise: float) -> None:
        """每轮惊讶观测（loop 侧每轮喂 activation.surprise）。

        尾窗滑动窗口追加 → 重算尾窗样本方差 → EMA 低通。开关关 → no-op
        （零参与，不记录）。非数值/非有限输入（NaN/inf）静默丢弃
        （fail-open：不抛、不污染方差估计）。
        """
        if not self.enabled:
            return
        try:
            s = float(surprise)
        except (TypeError, ValueError):
            return  # fail-open：非数值输入静默丢弃
        if not math.isfinite(s):
            return  # NaN/inf 无信号：丢弃（不污染方差）
        self._window.append(s)
        if len(self._window) >= 2:
            win = list(self._window)
            mean_s = sum(win) / len(win)
            var_raw = sum((v - mean_s) ** 2 for v in win) / len(win)
            if self._var_ema is None:
                self._var_ema = var_raw
            else:
                self._var_ema += self.ema * (var_raw - self._var_ema)
        # 样本 < 2 时方差未定义：EMA 保持上一状态（冷启动填充期不污染）

    # ------------------------------------------------------------------ #
    #  估计
    # ------------------------------------------------------------------ #

    def estimate(self) -> Optional[float]:
        """当前 π̄ = 1/Var(surprise) 估计（钳制后）；开关关或样本不足 → None。"""
        if not self.enabled:
            return None
        try:
            if len(self._window) < self.min_samples or self._var_ema is None:
                return None  # 冷启动保护：样本不足不给估计
            return self._pi_from_var(self._var_ema)
        except Exception:
            return None  # fail-open

    def _pi_from_var(self, var: float) -> float:
        """π = 1/var 钳制；退化方差（=0/非有限，如全等序列）→ MIN 保护。"""
        if not (var > 0.0):  # var==0 / var<0 / NaN 一律走 MIN 保护
            return self.pi_min
        return max(self.pi_min, min(self.pi_max, 1.0 / var))

    # ------------------------------------------------------------------ #
    #  学习率倍率（loop learn 侧调用）
    # ------------------------------------------------------------------ #

    def lr_multiplier(self, prev_lr: float) -> float:
        """π 加权学习率倍率 = π̄（自然梯度：effective_lr = prev_lr × π̄）。

        开关关 / 样本不足（冷启动）/ 异常 → 原样返回 prev_lr（零参与，
        行为与现状完全一致）。方向：π̄ 高（惊讶方差小，环境稳定）→ 倍率
        > 1；π̄ 低（惊讶剧烈波动）→ 倍率 < 1（见类 docstring）。
        """
        try:
            pi = self.estimate()
            if pi is None:
                return prev_lr
            return prev_lr * pi
        except Exception:
            return prev_lr  # fail-open

    # ------------------------------------------------------------------ #
    #  观测
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict:
        """观测块：{enabled, var_window, ema, pi_estimate, variance, samples}。

        pi_estimate / variance 冷启动或开关关时为 None；开关关时 samples = 0
        （零参与，未记录任何观测）。
        """
        try:
            return {
                'enabled': self.enabled,
                'var_window': self.var_window,
                'ema': self.ema,
                'pi_estimate': self.estimate(),
                'variance': self._var_ema,
                'samples': len(self._window),
            }
        except Exception:
            return {'enabled': self.enabled, 'var_window': self.var_window,
                    'ema': self.ema, 'pi_estimate': None, 'variance': None,
                    'samples': 0}

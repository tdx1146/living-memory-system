"""
活体记忆系统 - 海马体核心：FEP 吸引子网络
==========================================
⛔ 红线：bias（LMS_BIAS_SCALE）与 J 尺度（LMS_J_TARGET_NORM）是"场沉睡/饱和"的决定性参数，
   历史反复治疗三次（40 饱和→9.5 沉寂→12 现状）。改前必读红线清单 + 对照实验。

这是整个系统的核心记忆引擎。基于自由能原理（Free Energy Principle, FEP）
实现记忆的"塑形"——不是存档，而是持续改变网络的吸引子景观。

核心机制:
  1. 推断（infer）: Langevin 激活动力学 + 感官 clamping，收敛到吸引子态
  2. 学习（learn）: FEP 学习规则 ΔJ = -η∂F/∂J，准确性(Hebbian) + 复杂性(正交化)
  3. 无反向传播，无全局 loss——规则从第一性原理推导

数学基础:
  - Langevin 函数: L(b) = coth(b) - 1/b，是 CB（Chandrasekhar-Brink）分布的激活函数
  - 自由能: F = 准确性项 + 复杂性项(KL散度)
  - 准确性梯度 ≈ Hebbian 相关 (σ_i * σ_j)，强化共激活连接
  - 复杂性梯度 = 正交化压力，驱散相似表示，形成可区分的吸引子

参考：架构文档 第五节 5.3、第七节《FEP学习规则实现要点》
"""

import math
import os
import logging
import statistics
import torch
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Union

from core.types import Activation, resolve_device

logger = logging.getLogger(__name__)


def langevin(b: torch.Tensor) -> torch.Tensor:
    """Langevin 函数：CB 分布的激活函数。

    L(b) = coth(b) - 1/b = 1/tanh(b) - 1/b

    性质:
      - b → +∞: L(b) → 1
      - b → -∞: L(b) → -1
      - b → 0:  L(b) → b/3（泰勒展开极限，需特殊处理避免数值溢出）

    使用 1/tanh(b) 代替 cosh(b)/sinh(b) 以避免大 b 时的数值溢出
    （tanh 在 ±1 处饱和，不会溢出）。

    参数:
        b: 任意形状的张量，对数几率（log-odds）。

    返回:
        与 b 同形状的张量，取值范围 (-1, 1)。
    """
    result = torch.empty_like(b)
    # 小 b 区域使用泰勒展开 L(b) ≈ b/3，避免 1/tanh(b) 和 1/b 的除零问题
    small_mask = b.abs() < 1e-4
    large_mask = ~small_mask

    # 大 b 区域：使用 1/tanh(b) - 1/b（避免 cosh/sinh 溢出）
    b_large = b[large_mask]
    result[large_mask] = 1.0 / torch.tanh(b_large) - 1.0 / b_large

    # 小 b 区域：泰勒展开 L(b) ≈ b/3
    result[small_mask] = b[small_mask] / 3.0

    return result


# ================================================================== #
#  allostatic J 滑动设定点（论文机制 A：Mehra 1970 + Sterling 2012 + Gama 2014）
#  M4（2026-08-18）：原生并入 attractor.py（原外挂 runtime/allostatic_j.py
#  已删除——本文件成为唯一实现与唯一写者，j_target_norm 由本机制滑动）。
# ================================================================== #

# 观测快照保留长度（j_history / events 上限）
_SNAPSHOT_HISTORY = 200


def allostatic_j_enabled(explicit: Optional[bool] = None) -> bool:
    """治理开关解析：显式参数 > 环境变量 LMS_J_ALLOSTATIC（默认 0=关）。

    布尔接受（不区分大小写）：1/true/yes/on 视为开，其余为关。
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get('LMS_J_ALLOSTATIC', '0')
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


@dataclass
class SigmaStats:
    """σ 激活状态统计（每轮从 activation.state 计算）。

    对应双快照复测判据里的两个越界信号：
        frac_gt0_9: |σ| > 0.9 的节点占比（高 → 饱和信号）
        act05:      |σ| > 0.05 的节点数（极低 → 崩塌信号）
        valid:      统计是否来自真实数据（False = NaN/空/异常，中性不重锚）
    """
    frac_gt0_9: float
    act05: int
    valid: bool = True


def compute_sigma_stats(state) -> SigmaStats:
    """从激活态 state（torch.Tensor 或任意 .abs() 可迭代）计算 σ 统计。

    - NaN 防护：state 含 NaN → valid=False（中性统计，不触发饱和/崩塌，
      不误重锚，fail-open）。
    - 空 state → valid=False。
    """
    try:
        s = state.detach().cpu().abs()
        n = int(s.numel())
        if n <= 0:
            return SigmaStats(frac_gt0_9=0.0, act05=0, valid=False)
        if bool(torch.isnan(s).any()):
            logger.warning("allostatic J: σ 含 NaN，返回中性统计（不重锚）")
            return SigmaStats(frac_gt0_9=0.0, act05=n, valid=False)
        frac_gt0_9 = float((s > 0.9).sum().item()) / n
        act05 = int((s > 0.05).sum().item())
        return SigmaStats(frac_gt0_9=frac_gt0_9, act05=act05)
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("allostatic J: σ 统计失败（fail-open 中性）: %s", e)
        return SigmaStats(frac_gt0_9=0.0, act05=0, valid=False)


class AllostaticJController:
    """allostatic J 滑动设定点控制器（attractor 原生持有）。

    纯进程内存状态（同 precision_adapt 先例：重启即失、快照不落盘、
    回滚干净）。enabled=False 时所有方法为 no-op（返回初始固定值，
    零参与，行为与开关引入前完全一致）。

    背景（2026-08-17 双快照干净复测实锤，见 memory/子AI任务-双快照干净复测-
    20260817.md）：B 级 J=7.0 固定钳制在生产持续学习下漂移失效——J 内容漂移 →
    norm 7 工作点从"动态区"滑入"崩塌边缘"（act05 ~2/256、σ_std 0.26-0.29、
    组间差翻负 mean −0.345/−0.227）。dandan 拍板：不再扫描固定 J（Guo 2017
    已证"单一全局温度在分布漂移下必然失效"），直接上论文机制 A——J 从固定
    常数 → allostatic 滑动设定点。M4 原生并入 attractor：J_target_norm(t)
    不再是 env 固定常数，而是随数据流在线重估的滑动设定点：

      1. surprise 序列（= Kalman innovation，Mehra 1970）running window
         统计重估：z = (s − μ)/σ 持续越带（±k）→ innovation 分布漂移证据
         （越上带 = 持续高惊讶 → 设定点降；越下带 = 持续低惊讶 → 设定点升）。
      2. σ 越界信号（直接对应论文"σ 饱和/崩塌触发重锚"）：
         - 饱和（frac_gt0.9 高）：J 过强、σ 钉在极端 → 设定点下降
           （钳制加强 → J 减弱 → σ 脱离极端）；
         - 崩塌（act05 极低）：J 结构崩塌、σ 死亡 → 设定点上升
           （钳制放松 → 允许 J 增长 → 景观重建结构）；
         - 动态（健康）：不动作，设定点稳定（动态稳）。
      3. 重锚按 step 增量移动、钳制在 [j_min, j_max]、带 persist 轮确认
         （防单轮抖动）；越界触发打日志 + 进 events 窗口（可观测）。

    参数（全部 env 化；参数落盘表见 api/config.py 头部与任务报告）：
        LMS_J_ALLOSTATIC             0/1     治理开关（默认 0=关）
        LMS_J_TARGET_NORM            40.0    初始设定点（复用现有变量；关=固定值不变）
        LMS_J_ALLOSTATIC_WINDOW      200     surprise 窗口长度
        LMS_J_ALLOSTATIC_K           2.0     重锚阈值 k（z 带 ±k）
        LMS_J_ALLOSTATIC_STEP        0.5     每次重锚步长
        LMS_J_ALLOSTATIC_PERSIST     5       越带持续轮数（防单轮抖动）
        LMS_J_ALLOSTATIC_MIN         3.0     设定点下限
        LMS_J_ALLOSTATIC_MAX         40.0    设定点上限
        LMS_J_ALLOSTATIC_MIN_SAMPLES 30      冷启动窗口样本数（不足不动作）
        LMS_J_ALLOSTATIC_SAT_FRAC    0.9     σ 饱和判定：|σ|>0.9 节点占比 ≥ 此值
        LMS_J_ALLOSTATIC_COL_ACT     5       σ 崩塌判定：|σ|>0.05 节点数 ≤ 此值
    """

    def __init__(self, enabled: Optional[bool] = None,
                 init_target: float = 40.0,
                 window: int = 200,
                 k: float = 2.0,
                 step: float = 0.5,
                 persist: int = 5,
                 j_min: float = 3.0,
                 j_max: float = 40.0,
                 min_samples: int = 30,
                 sat_frac: float = 0.9,
                 col_act: int = 5) -> None:
        self.enabled = allostatic_j_enabled(enabled)
        self.init_target = float(init_target)
        self.window = int(window)
        self.k = float(k)
        self.step = float(step)
        self.persist = int(persist)
        self.j_min = float(j_min)
        self.j_max = float(j_max)
        self.min_samples = int(min_samples)
        self.sat_frac = float(sat_frac)
        self.col_act = int(col_act)

        # 设定点（初始 = env 现值；不另设固定值）
        self.j_target: float = max(self.j_min, min(self.j_max, self.init_target))

        # surprise 窗口（innovation 序列）
        self.surprise_window: Deque[float] = deque(maxlen=self.window)
        # 越带连续计数（persist 确认用）
        self._above_streak: int = 0
        self._below_streak: int = 0
        # σ 越界信号最近观测（供日志/快照）
        self.last_frac_gt0_9: float = 0.0
        self.last_act05: int = 0
        # 可观测：重锚事件 + 设定点历史（灵魂指标②/③）
        self.events: Deque[dict] = deque(maxlen=20)
        self.j_history: Deque[float] = deque(maxlen=_SNAPSHOT_HISTORY)

    # ------------------------------------------------------------------ #
    #  主入口：每轮观测 → 更新设定点
    # ------------------------------------------------------------------ #

    def update(self, surprise: float, sigma_stats: Optional[SigmaStats] = None
               ) -> float:
        """每轮观测（surprise + σ 统计）→ 更新 J 滑动设定点。

        返回新的 j_target（未启用 → 初始固定值，零参与）。

        重锚决策（优先级：σ 硬信号 > surprise 带漂移）:
            饱和（frac_gt0.9 ≥ sat_frac）          → 设定点 −step（饱和降）
            崩塌（act05 ≤ col_act）                → 设定点 +step（崩塌升）
            持续越上带（z > +k 达 persist 轮）      → 设定点 −step（高惊讶降）
            持续越下带（z < −k 达 persist 轮）      → 设定点 +step（低惊讶升）
            否则                                    → 不变（动态稳）
        """
        if not self.enabled:
            return self.j_target

        # 观测登记（冷启动也登记，只是不动作）
        try:
            surprise = float(surprise)
            if not math.isfinite(surprise):  # NaN/inf
                logger.warning("allostatic J: surprise 非法（%r），跳过本轮观测",
                               surprise)
                return self.j_target
            self.surprise_window.append(surprise)
        except (TypeError, ValueError):
            logger.warning("allostatic J: surprise 非法，跳过本轮观测")
            return self.j_target

        if sigma_stats is not None and sigma_stats.valid:
            self.last_frac_gt0_9 = float(sigma_stats.frac_gt0_9)
            self.last_act05 = int(sigma_stats.act05)
            sat = self.last_frac_gt0_9 >= self.sat_frac
            col = self.last_act05 <= self.col_act
        else:
            # 无有效 σ 数据（NaN/空/异常）：σ 信号本轮不参与（fail-open），
            # 仅 surprise 带漂移逻辑生效；last_* 保留最近有效观测供日志/快照。
            sat = False
            col = False
        if len(self.surprise_window) < self.min_samples:
            return self.j_target  # 冷启动：保持初始设定点

        # --- surprise 统计带（Mehra 1970 innovation）---
        z = self._surprise_z(surprise)
        if z > self.k:
            self._above_streak += 1
            self._below_streak = 0
        elif z < -self.k:
            self._below_streak += 1
            self._above_streak = 0
        else:
            self._above_streak = 0
            self._below_streak = 0
        persist_above = self._above_streak >= self.persist
        persist_below = self._below_streak >= self.persist

        # --- 重锚决策 ---
        if sat:
            new_target, reason = self.j_target - self.step, 'saturation'
        elif col:
            new_target, reason = self.j_target + self.step, 'collapse'
        elif persist_above:
            new_target, reason = self.j_target - self.step, 'surprise-above-band'
        elif persist_below:
            new_target, reason = self.j_target + self.step, 'surprise-below-band'
        else:
            new_target, reason = self.j_target, 'stable'

        new_target = max(self.j_min, min(self.j_max, new_target))
        if abs(new_target - self.j_target) > 1e-9:
            logger.info(
                "allostatic J 重锚: %.4f → %.4f (%s, z=%.2f, "
                "frac_gt0.9=%.3f, act05=%d, surprise=%.4f)",
                self.j_target, new_target, reason, z,
                self.last_frac_gt0_9, self.last_act05, surprise)
            self.events.append({
                'ts_turn': None,  # 调用方可在快照里补 turn；此处留 None
                'reason': reason,
                'j_before': round(self.j_target, 4),
                'j_after': round(new_target, 4),
                'z': round(z, 3),
                'surprise': round(surprise, 4),
                'frac_gt0_9': round(self.last_frac_gt0_9, 4),
                'act05': self.last_act05,
            })
        self.j_target = new_target
        self.j_history.append(self.j_target)
        return self.j_target

    # ------------------------------------------------------------------ #
    #  surprise 统计（Mehra 1970 innovation 法）
    # ------------------------------------------------------------------ #

    def _surprise_z(self, surprise: float) -> float:
        """surprise 在窗口分布中的 z 分数（越带 = innovation 漂移证据）。

        - 窗口样本 < 2 或 σ≈0 → 0.0（无漂移信号，中性）
        - 冷启动由调用方（update）保证不进入本方法。
        """
        win = list(self.surprise_window)
        if len(win) < 2:
            return 0.0
        mean = sum(win) / len(win)
        var = sum((v - mean) ** 2 for v in win) / len(win)
        std = var ** 0.5
        if std < 1e-8:
            return 0.0
        return (surprise - mean) / std

    # ------------------------------------------------------------------ #
    #  观测
    # ------------------------------------------------------------------ #

    def snapshot(self, turn_count: Optional[int] = None) -> dict:
        """观测块（/status allostatic_j；灵魂指标②J 动态 / ③越界触发）。

        开关关 → {'enabled': False}（调用方降级，零参与）。
        """
        if not self.enabled:
            return {'enabled': False}
        try:
            win = list(self.surprise_window)
            mean_s = (sum(win) / len(win)) if win else None
            std_s = (statistics.pstdev(win) if len(win) >= 2 else None)
            events = list(self.events)
            if turn_count is not None:
                for ev in events:
                    ev['ts_turn'] = turn_count
            return {
                'enabled': True,
                'j_target': round(self.j_target, 4),
                'j_initial': round(self.init_target, 4),
                'j_min': self.j_min,
                'j_max': self.j_max,
                'j_history': [round(v, 4) for v in self.j_history],
                'window_n': len(self.surprise_window),
                'min_samples': self.min_samples,
                'cold': len(self.surprise_window) < self.min_samples,
                'surprise_mean': (round(mean_s, 4)
                                  if mean_s is not None else None),
                'surprise_std': (round(std_s, 4)
                                 if std_s is not None else None),
                'last_z': round(self._surprise_z(
                    self.surprise_window[-1]), 3)
                    if self.surprise_window else None,
                'frac_gt0_9': round(self.last_frac_gt0_9, 4),
                'act05': self.last_act05,
                'events': events,
                'params': {
                    'window': self.window,
                    'k': self.k,
                    'step': self.step,
                    'persist': self.persist,
                    'sat_frac': self.sat_frac,
                    'col_act': self.col_act,
                },
            }
        except Exception as e:  # pylint: disable=broad-except
            logger.debug("allostatic J snapshot 组装失败（fail-open）: %s", e)
            return {'enabled': True}


class HomeostaticBias:
    """HomeostaticBias 滑动设定点控制器（方案 B，2026-08-22，dandan 拍板）。

    依据：Turrigiano 稳态可塑性"滑动阈值/设定点"（PubMed 10322495）——bias
    （工作点/静息电位）缓慢滑向目标平均活跃度；FEP 语义下 bias=先验均值×精度。
    结构照抄亲哥哥 AllostaticJController（attractor.py:130-371）——纯进程内存、
    开关关=零参与逐字节不变、冷启动攒基线、persist 防抖、step 重锚、钳制区间。

    与 allostatic J 的分工（不打架）：
      - J（哥哥）：管预测结构强度（surprise 创新带，Mehra 1970）。
      - bias（弟弟）：管工作点/静息活跃度（mean|σ| 目标带，Turrigiano）。
      饱和/崩塌信号两者都会看到：方向一致无害；若反向，以 J（结构信号）优先。

    参数（全 env 化，开关 LMS_BIAS_ADAPT 默认 0=关）：
        LMS_BIAS_INIT                   初始 bias（默认 1.5 = 手调现状）
        LMS_BIAS_ADAPT_TARGET_LO/HI     mean|σ| 目标带（默认 0.30/0.60，文献带）
        LMS_BIAS_ADAPT_STEP             重锚步长（默认 0.05，比 J 的 0.5 慢一个量级）
        LMS_BIAS_ADAPT_PERSIST          越带持续轮数（默认 5，防单轮抖动）
        LMS_BIAS_ADAPT_MIN/MAX          设定点钳制（默认 0.5/2.5，文献非线性带）
        LMS_BIAS_ADAPT_MIN_SAMPLES      冷启动窗口样本数（默认 30，与 allostatic 同步热身）
        LMS_BIAS_ADAPT_SAT_FRAC / COL_ACT  σ 饱和/崩塌判定（复用 allostatic 口径）
    """

    def __init__(self, init_bias: float = 1.5, input_dim: int = 0,
                 target_lo: float = 0.30, target_hi: float = 0.60,
                 step: float = 0.05, persist: int = 5,
                 bias_min: float = 0.5, bias_max: float = 2.5,
                 min_samples: int = 30, sat_frac: float = 0.9,
                 col_act: int = 5) -> None:
        self.enabled = True  # 由 AttractorNetwork 构造时按 env 决定是否实例化
        self.input_dim = int(input_dim)
        self.init_bias = float(init_bias)
        self.target_lo = float(target_lo)
        self.target_hi = float(target_hi)
        self.step = float(step)
        self.persist = int(persist)
        self.bias_min = float(bias_min)
        self.bias_max = float(bias_max)
        self.min_samples = int(min_samples)
        self.sat_frac = float(sat_frac)
        self.col_act = int(col_act)

        # 设定点（初始 = env 现值，不另设固定值）
        self.bias: float = max(self.bias_min, min(self.bias_max, self.init_bias))

        # 活跃度窗口（mean|σ| 序列，Turrigiano 目标跟踪）
        self.activity_window: Deque[float] = deque(maxlen=200)
        # 越带连续计数（persist 确认用）
        self._below_streak: int = 0
        self._above_streak: int = 0
        # 最近观测（供日志/快照）
        self.last_mean_abs: float = 0.0
        self.last_frac_gt0_9: float = 0.0
        self.last_act05: int = 0
        # 可观测：重锚事件 + bias 历史
        self.events: Deque[dict] = deque(maxlen=20)
        self.bias_history: Deque[float] = deque(maxlen=200)

    def update(self, sigma_state) -> float:
        """每轮观测（σ 激活态）→ 更新 bias 滑动设定点。

        返回新的 bias。冷启动（窗口 < min_samples）保持初始值不动（"渐渐"）。
        重锚决策（优先级：σ 硬信号 > 活跃度带）:
            饱和（frac_gt0.9 ≥ sat_frac）          → bias −step（场太闹，冷静）
            崩塌（act05 ≤ col_act）                → bias +step（场死了，叫醒）
            活跃度持续低于 target_lo（达 persist）  → bias +step（场睡回去，唤醒）
            活跃度持续高于 target_hi（达 persist）  → bias −step（场过激，降温）
            否则                                    → 不变（动态稳）
        """
        # 观测登记（冷启动也登记，只是不动作）
        try:
            stats = compute_sigma_stats(sigma_state)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("HomeostaticBias: σ 统计失败（fail-open 中性）: %s", e)
            return self.bias
        if not stats.valid:
            return self.bias

        # 活跃度 = 非感官（潜变量）节点的 mean|σ|（感官节点被输入钳制驱动，
        # 反映的是输入不是工作点；input_dim=0 时退化观测全部节点）
        try:
            s = sigma_state.detach().cpu().abs()
            if self.input_dim and s.numel() > self.input_dim:
                mean_abs = float(s[self.input_dim:].mean())
            else:
                mean_abs = float(s.mean())
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("HomeostaticBias: 活跃度计算失败（fail-open）: %s", e)
            return self.bias
        if not math.isfinite(mean_abs):
            return self.bias

        self.activity_window.append(mean_abs)
        self.last_mean_abs = mean_abs
        self.last_frac_gt0_9 = float(stats.frac_gt0_9)
        self.last_act05 = int(stats.act05)

        if len(self.activity_window) < self.min_samples:
            return self.bias  # 冷启动：保持初始设定点（渐渐）

        sat = stats.frac_gt0_9 >= self.sat_frac
        col = stats.act05 <= self.col_act

        # 活跃度带越界计数（persist 防抖）
        if mean_abs < self.target_lo:
            self._below_streak += 1
            self._above_streak = 0
        elif mean_abs > self.target_hi:
            self._above_streak += 1
            self._below_streak = 0
        else:
            self._below_streak = 0
            self._above_streak = 0
        persist_below = self._below_streak >= self.persist
        persist_above = self._above_streak >= self.persist

        # 重锚决策（σ 硬信号优先）
        if sat:
            new_bias, reason = self.bias - self.step, 'saturation'
        elif col:
            new_bias, reason = self.bias + self.step, 'collapse'
        elif persist_below:
            new_bias, reason = self.bias + self.step, 'activity-below-band'
        elif persist_above:
            new_bias, reason = self.bias - self.step, 'activity-above-band'
        else:
            new_bias, reason = self.bias, 'stable'

        new_bias = max(self.bias_min, min(self.bias_max, new_bias))
        if abs(new_bias - self.bias) > 1e-9:
            logger.info(
                "HomeostaticBias 重锚: %.4f → %.4f (%s, mean|σ|=%.3f, "
                "frac_gt0.9=%.3f, act05=%d)",
                self.bias, new_bias, reason, mean_abs,
                self.last_frac_gt0_9, self.last_act05)
            self.events.append({
                'reason': reason,
                'bias_before': round(self.bias, 4),
                'bias_after': round(new_bias, 4),
                'mean_abs': round(mean_abs, 4),
            })
            self.bias = new_bias
        self.bias_history.append(self.bias)
        return self.bias

    def snapshot(self, turn_count: Optional[int] = None) -> dict:
        """观测块（/status homeostatic_bias）。"""
        try:
            win = list(self.activity_window)
            mean_a = (sum(win) / len(win)) if win else None
            events = list(self.events)
            return {
                'enabled': True,
                'bias': round(self.bias, 4),
                'bias_initial': round(self.init_bias, 4),
                'bias_min': self.bias_min,
                'bias_max': self.bias_max,
                'target_lo': self.target_lo,
                'target_hi': self.target_hi,
                'cold': len(win) < self.min_samples,
                'window_n': len(win),
                'min_samples': self.min_samples,
                'mean_abs': round(mean_a, 4) if mean_a is not None else None,
                'last_mean_abs': round(self.last_mean_abs, 4),
                'last_frac_gt0_9': round(self.last_frac_gt0_9, 4),
                'last_act05': self.last_act05,
                'step': self.step,
                'persist': self.persist,
                'bias_history': [round(float(x), 4) for x in self.bias_history],
                'events': events,
            }
        except Exception as e:  # pylint: disable=broad-except
            logger.debug("HomeostaticBias snapshot 失败（fail-open）: %s", e)
            return {'enabled': True}


class AttractorNetwork:
    """FEP 吸引子网络：核心记忆引擎。

    网络结构:
      - num_nodes 个节点，前 input_dim 个为感官节点（接收外部 clamping）
      - J: 耦合矩阵 [num_nodes, num_nodes]，对称，对角线为 0
      - bias: 偏置向量 [num_nodes]
      - sigma: 当前状态 [num_nodes]，取值 (-1, 1)

    推断时，感官节点被外部信号 clamping，其余节点通过 Langevin 动力学
    收敛到吸引子态。学习时，根据 FEP 规则更新 J 矩阵。

    关键: 无反向传播，无全局 loss。学习规则从自由能原理推导。
    """

    def __init__(self, num_nodes: int, input_dim: int,
                 seed: int = 42,
                 temperature: float = 0.05,
                 device: Union[str, torch.device] = "auto",
                 norm_surprise: Optional[bool] = None,
                 ewc: Optional["EwcPenalty"] = None) -> None:
        """初始化吸引子网络。

        参数:
            num_nodes: 网络节点总数（建议 256-1024）。
            input_dim: 感官输入维度。前 input_dim 个节点为感官节点。
            seed: 随机种子，保证可复现。
            temperature: Langevin 动力学温度（扩散项噪声强度）。
                T > 0 时相似但不同的输入有机会收敛到不同吸引子，
                使正交化学习规则能够发挥作用。设为 0 则退化为纯平均场推断。
            device: 计算设备（E-P2-1）。支持 "auto"/"cpu"/"cuda"/"cuda:0"
                或 torch.device。所有张量（J、bias、sigma）将创建在该设备上。
            ewc: EWC 持续学习保护（S4 C 条件触发，默认 None=不启用，向后
                兼容）。显式传入 core.continual.ewc.EwcPenalty 对象优先；
                为 None 时按环境变量 LMS_EWC_ENABLE（默认 0=关）决定——
                开则内部构造（对角 Fisher 于健康窗口 EMA 累积 + 缩放不变
                二次罚项，挂载 learn()，见 learn() docstring）。关 → 零参与，
                行为与开关引入前逐字节一致。
        """
        assert input_dim <= num_nodes, (
            f"input_dim({input_dim}) 不能大于 num_nodes({num_nodes})"
        )
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.seed = seed
        # E-P2-1: 统一设备管理
        self.device: torch.device = resolve_device(device)

        generator = torch.Generator()
        generator.manual_seed(seed)

        # 耦合矩阵 J：初始为小随机值，对称化，对角线为 0
        # 小初始化保证初始状态接近中性，学习逐步塑造结构
        # E-P2-1: 在 CPU 上用 CPU 生成器产生随机值（保证种子可复现性跨设备
        # 一致），再迁移到目标设备
        self.J: torch.Tensor = (
            torch.randn(num_nodes, num_nodes, generator=generator) * 0.01
        ).to(self.device)
        # 对称化（非序列模式：J_ij = J_ji）
        self.J = (self.J + self.J.T) / 2
        # 无自连接
        self.J.fill_diagonal_(0)

        # 偏置向量：初始为 0（中性）；LMS_BIAS_SCALE>0 时给节点一个静息
        # 电位，把 σ 从平凡不动点 σ≈0 唤醒（2026-08-22 场动力学修复第一步）。
        # 常量偏置 b 使 σ ≈ coth(b)−1/b：b=1.5 → σ≈0.44、b=2 → σ≈0.54
        # （离开 0 又未饱和 ±1 的活跃区）。默认 0 = 逐字节不变。
        _bias_scale = 0.0
        try:
            _bias_scale = float(os.environ.get("LMS_BIAS_SCALE", "0") or 0)
        except Exception:
            _bias_scale = 0.0
        # 方案 B（2026-08-22 dandan 拍板）：LMS_BIAS_ADAPT=1 → HomeostaticBias
        # 滑动设定点（Turrigiano 稳态可塑性）；=0/缺省 → 静态 bias（现状逐字节不变）。
        _bias_adapt = os.environ.get("LMS_BIAS_ADAPT", "0").strip().lower() \
            in ('1', 'true', 'yes', 'on')
        if _bias_adapt:
            _init_bias = _bias_scale if _bias_scale > 0 else 1.5
            try:
                self.homeostatic_bias = HomeostaticBias(
                    init_bias=_init_bias,
                    input_dim=input_dim,
                    target_lo=float(os.environ.get(
                        "LMS_BIAS_ADAPT_TARGET_LO", "0.30")),
                    target_hi=float(os.environ.get(
                        "LMS_BIAS_ADAPT_TARGET_HI", "0.60")),
                    step=float(os.environ.get("LMS_BIAS_ADAPT_STEP", "0.05")),
                    persist=int(os.environ.get("LMS_BIAS_ADAPT_PERSIST", "5")),
                    bias_min=float(os.environ.get("LMS_BIAS_ADAPT_MIN", "0.5")),
                    bias_max=float(os.environ.get("LMS_BIAS_ADAPT_MAX", "2.5")),
                    min_samples=int(os.environ.get(
                        "LMS_BIAS_ADAPT_MIN_SAMPLES", "30")),
                )
                self.bias: torch.Tensor = torch.full(
                    (num_nodes,), self.homeostatic_bias.bias,
                    device=self.device)
                logger.warning(
                    "LMS_BIAS_ADAPT=1 启用：bias 自适应滑动（HomeostaticBias，"
                    "init=%.3f, 带[%.2f,%.2f], step=%.2f）",
                    _init_bias, self.homeostatic_bias.target_lo,
                    self.homeostatic_bias.target_hi, self.homeostatic_bias.step)
            except Exception as e:  # pylint: disable=broad-except
                # 自适应构造失败 → 回退静态（fail-open，绝不阻断构造）
                logger.warning("HomeostaticBias 构造失败，回退静态 bias: %s", e)
                self.homeostatic_bias = None
                self.bias: torch.Tensor = torch.full(
                    (num_nodes,), _init_bias, device=self.device)
        else:
            self.homeostatic_bias = None
            if _bias_scale > 0:
                self.bias: torch.Tensor = torch.full(
                    (num_nodes,), _bias_scale, device=self.device)
                logger.warning(
                    "LMS_BIAS_SCALE=%.3f 启用：bias 非零（场唤醒实验）", _bias_scale)
            else:
                self.bias: torch.Tensor = torch.zeros(
                    num_nodes, device=self.device)

        # 当前状态 sigma：初始为 0（中性状态）
        self.sigma: torch.Tensor = torch.zeros(num_nodes, device=self.device)

        # 复杂性/正交化参数（可外部调整）
        self.complexity_weight: float = 0.01
        self.orth_weight: float = 0.5

        # Langevin 温度：扩散项的噪声强度（G7 修复：从构造参数获取）
        # 物理上 Langevin 动力学 = 漂移(确定性) + 扩散(随机性)
        # temperature > 0 时，相似但不同的输入有机会收敛到不同吸引子，
        # 使正交化学习规则能够发挥作用。设为 0 则退化为纯平均场推断。
        self.temperature: float = temperature

        # T2.8/P2-1：惊讶度归一化开关（LMS_NORM_SURPRISE=1 启用，默认 0）。
        # ⚠️ 已弃用（2026-08-10 惊讶度语义拆分，惊讶度修复-01-设计方案.md §3.3）：
        #     F/‖J‖ 归一化已整块移除，本开关不再参与任何计算。属性保留仅为
        #     防 config/api 审计引用崩（api/config.py、api/server.py 仍读取）。
        if norm_surprise is None:
            norm_surprise = os.environ.get("LMS_NORM_SURPRISE", "0") == "1"
        self.norm_surprise: bool = bool(norm_surprise)
        if self.norm_surprise:
            logger.warning(
                "LMS_NORM_SURPRISE=1 已弃用：F/‖J‖ 归一化自 2026-08-10 起 "
                "不再参与任何计算（no-op），可安全移除此环境变量")
        # 2026-08-10 设计回归：J 范数钳制目标（‖J‖_F 上限）
        self.j_target_norm: float = float(os.environ.get("LMS_J_TARGET_NORM", "40.0"))

        # ── M4（2026-08-18）：allostatic J 滑动设定点原生并入 ──
        # 治理开关 LMS_J_ALLOSTATIC（默认 0=关 → 固定 J 行为完全不变，回滚干净；
        # 关时本机制全部路径零参与）。开启后 j_target_norm 不再是 env 固定常数，
        # 而是由 surprise 序列 + σ 越界信号在线重估的滑动设定点（Mehra 1970
        # innovation 法 + Sterling 2012 allostasis；饱和降 / 崩塌升 / 动态稳）。
        # 控制器由本网络原生持有（唯一实现与唯一写者），状态为纯进程内存
        # （同 precision_adapt 先例：重启即失、快照不落盘——get_landscape/
        # set_landscape 只保存 J/bias/sigma，恢复后设定点回到 env 初值，回滚干净）。
        self.allostatic: Optional[AllostaticJController] = None
        if allostatic_j_enabled():
            self.allostatic = AllostaticJController(
                enabled=True,
                # 初始设定点 = 当前 env 值（LMS_J_TARGET_NORM），不另设固定值
                init_target=float(os.environ.get("LMS_J_TARGET_NORM", "40.0")),
                window=int(os.environ.get("LMS_J_ALLOSTATIC_WINDOW", "200")),
                k=float(os.environ.get("LMS_J_ALLOSTATIC_K", "2.0")),
                step=float(os.environ.get("LMS_J_ALLOSTATIC_STEP", "0.5")),
                persist=int(os.environ.get("LMS_J_ALLOSTATIC_PERSIST", "5")),
                j_min=float(os.environ.get("LMS_J_ALLOSTATIC_MIN", "3.0")),
                j_max=float(os.environ.get("LMS_J_ALLOSTATIC_MAX", "40.0")),
                min_samples=int(
                    os.environ.get("LMS_J_ALLOSTATIC_MIN_SAMPLES", "30")),
                sat_frac=float(
                    os.environ.get("LMS_J_ALLOSTATIC_SAT_FRAC", "0.9")),
                col_act=int(os.environ.get("LMS_J_ALLOSTATIC_COL_ACT", "5")),
            )
            # 初始设定点立即接管（learn 的范数钳制按动态值执行）
            self.j_target_norm = self.allostatic.j_target
            logger.info(
                "allostatic J 已启用（原生并入 attractor）：初始设定点 "
                "J_target=%.4f（window=%d, k=%.2f, step=%.2f, persist=%d, "
                "range=[%.2f, %.2f]）",
                self.allostatic.j_target, self.allostatic.window,
                self.allostatic.k, self.allostatic.step,
                self.allostatic.persist, self.allostatic.j_min,
                self.allostatic.j_max)
        else:
            logger.debug(
                "allostatic J 未启用（LMS_J_ALLOSTATIC=0）：j_target_norm "
                "保持 env 固定值 %.4f", self.j_target_norm)

        # ── 阶段 1 弥散态修复（2026-08-16，v1.2 §三 A 级）──
        # per-node precision 差异性重标定（env 开关 + 快照回滚）：
        #   弥散态根因 = J@σ 主导（J_norm 40 自 8/10 钳制）→ σ 输入不变
        #   （inter-input cosine ≈ 1.0）→ precision 塌缩（极差 1.1%）→
        #   surprise 退化为 mse 线性函数（方向响应丢失）。
        # A 级 = 每个 sensory 节点的 precision 按其激活模式（跨输入误差
        #   方差）单独重标定，恢复 node 间差异（非全局缩放），并把均值恢复
        #   到 8/10 有结构期水平（π̄≈1.5，当前 ≈0.34）——恢复感官钳制
        #   s·π 与 J@σ 的平衡（实验：J→5 + π×12~16 → surprise 组间差转正）。
        # 开关：LMS_PRECISION_RECALIB=1 启用（默认 0，零行为变化，可回滚）；
        #   LMS_PRECISION_RECALIB_MEAN=目标均值（默认 1.5）；
        #   LMS_PRECISION_RECALIB_STRENGTH=重标定强度（默认 1.0）。
        self.precision_recalib_enabled: bool = (
            os.environ.get("LMS_PRECISION_RECALIB", "0") == "1")
        self.precision_recalib_mean: float = float(
            os.environ.get("LMS_PRECISION_RECALIB_MEAN", "1.5"))
        self.precision_recalib_strength: float = float(
            os.environ.get("LMS_PRECISION_RECALIB_STRENGTH", "1.0"))
        if self.precision_recalib_enabled:
            logger.info(
                "per-node precision 差异性重标定已启用 "
                f"(mean_target={self.precision_recalib_mean}, "
                f"strength={self.precision_recalib_strength})——"
                "阶段1 弥散态修复 A 级，env 可回滚")

        # ── S4（2026-08-18）：EWC 持续学习保护（C 条件触发，ABC 操作规划
        # §S4，默认关）──
        # 治理开关 LMS_EWC_ENABLE（默认 0=关 → 零参与，行为与现状逐字节
        # 一致；learn() 的 EWC 块整体跳过）。显式传入 ewc 对象优先（如测试
        # 注入 enabled=False 的 EwcPenalty 验证零参与）；未传且 env 开 →
        # 内部构造（对角 Fisher 于健康窗口 EMA 累积 + 缩放不变二次罚项，
        # 详见 core/continual/ewc.py）。懒 import：本文件顶部依赖保持
        # 不变（只 torch 与 core.types），避免循环依赖；关路径不 import。
        # EWC 状态为纯进程内存（同 allostatic 先例：重启即失、快照不落盘、
        # 回滚干净）。
        self.ewc: Optional["EwcPenalty"] = None
        if ewc is not None:
            self.ewc = ewc
        elif os.environ.get("LMS_EWC_ENABLE", "0").strip().lower() in (
                "1", "true", "yes", "on"):
            from core.continual.ewc import EwcPenalty  # 懒 import（S4）
            self.ewc = EwcPenalty(shape=self.J.shape, device=self.device)
            logger.info(
                "EWC 保护已启用（LMS_EWC_ENABLE=1）: λ=%.4f, "
                "fisher_window=%d, ema=%.4f",
                self.ewc.lam, self.ewc.fisher_window, self.ewc.ema)
        else:
            logger.debug(
                "EWC 保护未启用（LMS_EWC_ENABLE=0）：learn() 零参与，"
                "行为与开关引入前一致")

    # ------------------------------------------------------------------ #
    #  推断
    # ------------------------------------------------------------------ #

    def _recalibrate_precision_per_node(
            self, precision: torch.Tensor,
            sensory_input: torch.Tensor) -> torch.Tensor:
        """per-node precision 差异性重标定（阶段 1 弥散态修复 A 级，v1.2 §三）。

        弥散态根因链：J@σ 主导（J_norm 40 自 8/10 钳制）→ σ 输入不变
        （inter-input cosine ≈ 1.0）→ precision 塌缩（极差 1.1%）→ surprise
        退化为 mse 线性函数（方向响应丢失）。本方法恢复感官钳制 s·π 与
        内部驱动 J@σ 的平衡：

          π_i = π̄_target × (1 + strength × z_i)

        其中 z_i 是每节点激活模式（跨输入误差 std）的 z-score——恢复
        node 间差异（非全局乘一个系数）；π̄_target 默认 1.5（8/10 有结构期
        均值，当前 ≈0.34）。每个节点按其历史方差/激活模式单独重标定。

        激活模式来源：本次输入的 per-dim 误差 (σ−s)² 历史由调用方在
        memory 层积累；本方法用 sensory_input 与 precision 当前值的
        局部结构做轻量估计（零额外状态，可回滚）。

        参数:
            precision: 原始 precision 向量 [input_dim]。
            sensory_input: 感官输入 [input_dim]（激活模式参考）。

        返回:
            重标定后的 precision 向量（克隆，不改原张量）。
        """
        # 局部激活模式：|sensory_input| 归一化后作为每节点权重基底
        # （不同输入簇激活不同维度——即使 σ 被压平，输入模式仍有结构）。
        act = sensory_input.abs()
        act_sum = act.sum()
        if act_sum < 1e-9:
            return precision.clone()
        # 激活占比（概率归一化）→ 每节点“重要性”权重
        w = act / act_sum  # [input_dim]，和为 1
        # 权重去中心化 z-score（保留 node 间差异的方向）
        w_mean = w.mean()
        w_std = w.std()
        if w_std < 1e-9:
            return precision.clone()
        z = (w - w_mean) / w_std

        # 重标定：均值恢复到目标，node 间差异按激活模式放大
        pi_bar = float(self.precision_recalib_mean)
        strength = float(self.precision_recalib_strength)
        new_pi = pi_bar * (1.0 + strength * z)
        # clamp 到 precision 合法范围（对齐 purpose 层 [0.1, 10]）
        new_pi = torch.clamp(new_pi, 0.1, 10.0)
        return new_pi.to(precision.device)

    def infer(self, sensory_input: torch.Tensor, precision: torch.Tensor,
              num_steps: int = 10,
              temperature_override: Optional[float] = None,
              initial_state: Optional[torch.Tensor] = None,
              update_internal_state: bool = True) -> Activation:
        """FEP 推断：给定感官输入和 precision，跑 K 步收敛到吸引子态。

        推断规则（从 FEP 推导）:
            b_q = bias + J @ sigma           # 对数几率（内部模型预测）
            sigma = langevin(b_q)             # Langevin 激活: coth(b) - 1/b
            sigma += sqrt(2*T) * noise        # Langevin 扩散项（随机性）
            感官节点被 sensory_input * precision clamping（外部证据注入）

        Langevin 动力学物理上包含漂移项（确定性激活）和扩散项（随机噪声）。
        扩散项的作用:
          - 打破对称性：相似但不同的输入有机会收敛到不同吸引子
          - 探索景观：避免陷入次优局部最小值
          - 使正交化学习规则生效：不同吸引子产生不同的 Hebbian 相关

        高 precision = 高信任 = 感官证据主导推断。
        低 precision = 低信任 = 内部模型主导推断。

        E-P2-1: 输入张量自动迁移到网络所在 device。
        E-P2-5: 新增 override 参数，避免调用方通过"保存→修改→恢复实例属性"
            的方式临时改变推断行为：

            - temperature_override: 临时使用的 Langevin 温度（不修改
              self.temperature）。为 None 时使用 self.temperature。
            - initial_state: 推断的起始 sigma（不修改 self.sigma）。为 None
              时从 self.sigma 出发。常用于做梦引擎的生成式回放种子。
            - update_internal_state: 是否将收敛后的 sigma 写回 self.sigma。
              默认 True（在线模式）。设为 False 时推断不污染内部状态
              （如做梦引擎需保持在线 sigma 不变）。

        参数:
            sensory_input: 感官向量，形状 [input_dim]。
            precision: 感官精度向量，形状 [input_dim]。
            num_steps: 推断迭代步数 K。
            temperature_override: 临时温度覆盖（E-P2-5）。
            initial_state: 推断起始状态（E-P2-5）。
            update_internal_state: 是否写回 self.sigma（E-P2-5）。

        返回:
            收敛后的 Activation（激活态 + 熵 + 惊讶度）。
        """
        # E-P2-1: 输入张量自动迁移到正确 device
        sensory_input = sensory_input.to(self.device)
        precision = precision.to(self.device)

        # 阶段 1 弥散态修复（A 级，env 开关）：per-node precision 差异性
        # 重标定——每个 sensory 节点按其激活模式单独重标定（恢复 node 间
        # 差异），并把均值恢复到 8/10 有结构期水平。默认关（零行为变化）。
        if self.precision_recalib_enabled:
            precision = self._recalibrate_precision_per_node(
                precision, sensory_input)

        # B7 修复：输入形状校验，防止晦涩的广播错误
        assert sensory_input.shape == (self.input_dim,), (
            f"sensory_input shape mismatch: {sensory_input.shape} "
            f"!= ({self.input_dim},)"
        )
        assert precision.shape == (self.input_dim,), (
            f"precision shape mismatch: {precision.shape} "
            f"!= ({self.input_dim},)"
        )

        # E-P2-5: 起始状态——提供 initial_state 时从它出发，否则从 self.sigma
        if initial_state is not None:
            sigma = initial_state.to(self.device).clone()
        else:
            sigma = self.sigma.clone()

        # E-P2-5: 临时温度覆盖（不修改实例属性）
        temperature = (temperature_override
                       if temperature_override is not None
                       else self.temperature)

        for _step in range(num_steps):
            # 对数几率：内部模型的预测
            b_q = self.bias + self.J @ sigma  # [num_nodes]

            # 感官 clamping：将外部证据注入感官节点
            # precision 权衡感官证据与内部预测的信任程度
            b_q[:self.input_dim] = (
                b_q[:self.input_dim] + sensory_input * precision
            )

            # Langevin 漂移项：确定性激活
            sigma = langevin(b_q)

            # Langevin 扩散项：小量随机扰动
            # 这不是数值噪声，而是 Langevin 方程的物理组成部分
            # 温度 T 控制噪声强度，T=0 时退化为平均场推断
            if temperature > 0:
                noise = torch.randn_like(sigma) * temperature
                sigma = sigma + noise
                # 保持激活值在 (-1, 1) 范围内
                sigma = torch.clamp(sigma, -0.999, 0.999)

        # E-P2-5: 仅在需要时写回内部状态（避免做梦等场景污染在线 sigma）
        if update_internal_state:
            self.sigma = sigma.clone()

        # 计算惊讶度（准确性项）、自由能、逐维惊讶度、MSE
        surprise = self._compute_surprise(sigma, sensory_input, precision)
        free_energy = self._compute_free_energy(sigma, sensory_input, precision)
        per_dim_surprise = self._compute_per_dim_surprise(
            sigma, sensory_input, precision)
        mse = float(torch.mean(
            (sigma[:self.input_dim] - sensory_input) ** 2).item())
        entropy = self._compute_entropy(sigma)

        return Activation(state=sigma, entropy=entropy, surprise=surprise,
                          free_energy=free_energy,
                          per_dim_surprise=per_dim_surprise,
                          mse=mse)

    # ------------------------------------------------------------------ #
    #  学习
    # ------------------------------------------------------------------ #

    def learn(self, activation: Activation, sensory_input: torch.Tensor,
              learning_rate: float = 0.01,
              orth_weight_override: Optional[float] = None,
              complexity_weight_override: Optional[float] = None) -> None:
        """FEP 学习：更新耦合矩阵 J。

        学习规则（从 FEP 推导）:
            ΔJ = -η * ∂F/∂J
            F = 准确性项 + 复杂性项(KL)

        准确性项的梯度:
            ∂F_accuracy/∂J_ij = -σ_i * σ_j  （共激活应增强连接以降低自由能）
            因此 -∂F_accuracy/∂J = +σ_i * σ_j = Hebbian 相关

        复杂性项的实现（双层正交化）:

            层 1 — sigma 正交化（方向性）:
                在 Hebbian 学习之前，从当前 sigma 中减去与
                已学结构（J @ sigma，即先验）重叠的投影分量。
                这样新模式只在"新颖方向"上学习，不与已有吸引子叠加。
                这是 KL(后验||先验) 的直接体现。

                sigma_orth = sigma - orth_weight * proj * (J @ sigma)

            层 2 — 饱和度压力（幅度性）:
                对 J 中已经大的连接施加额外学习阻力，防止发散。
                见 _complexity_gradient()。

        最终更新:
            ΔJ = Hebbian(sigma_orth) - 复杂性梯度
            J += η * (ΔJ + ΔJ^T) / 2   （对称化）
            J.fill_diagonal_(0)          （无自连接）

        无反向传播。无全局 loss。规则从第一性原理推导。

        E-P2-1: 输入张量（激活态）自动迁移到网络所在 device。
        E-P2-5: 新增 orth_weight_override / complexity_weight_override 参数，
            允许调用方临时覆盖正交化权重与复杂度权重，而无需"保存→修改→
            恢复"实例属性。覆盖值仅在本次调用内生效，不修改 self.orth_weight
            / self.complexity_weight。为 None 时使用实例属性（向后兼容）。

        S4（2026-08-18）：EWC 持续学习保护（C 条件触发，ABC 操作规划 §S4，
        默认关）——**只做加法**，开关关时本块整体跳过，行为与开关引入前
        逐字节一致。开关开（self.ewc is not None 且 enabled）时：

        1. Fisher 对角累积（健康窗口 EMA）——healthy 由 σ 统计判定
           （compute_sigma_stats：valid 且非崩塌 act05>col_act 且非饱和
           frac_gt0_9<sat_frac；col_act/sat_frac 复用 allostatic 语义，
           allostatic 关时用其默认 5/0.9）。崩塌/过渡态 → healthy=False →
           不更新 Fisher（关键约束：绝不在坏工作点计算 Fisher）。
           累积对象 grads = 对称化 ΔJ（= −∂F/∂J；平方后与 ∂F/∂J 同）。
           首个健康更新定格保护锚点 θ* = 当时的 J。
        2. EWC 罚项梯度（缩放不变方向罚）并入 ΔJ——**挂载点**：对称化更新
           J += η·sym(ΔJ) 之前，罚项梯度与 Hebbian/复杂性同乘 η
           （Kirkpatrick 2017 标准：罚项梯度与数据梯度同学习率）：

               ΔJ ← ΔJ − ∂P/∂J
               P = λ·ΣFᵢ(θ̂ᵢ − θ̂*ᵢ)²，θ̂ = J/‖J‖_F，θ̂* = J*/‖J*‖_F
               ∂P/∂J = (2λ/‖J‖)·[F⊙(θ̂−θ̂*) − ⟨F⊙(θ̂−θ̂*), θ̂⟩·θ̂]

           θ̂ 归一（scale-invariant）使罚项对 A 重锚（标量乘 J，只改 ‖J‖
           不改方向）不变——C 豁免 A 的重锚缩放方向，保护不变成阻碍。
           ∂P/∂J 径向分量为 0，EWC 力永不沿整体缩放方向施加。

        参数:
            activation: 推断得到的激活态。
            sensory_input: 产生该激活态的感官输入。
            learning_rate: 学习率 η。
            orth_weight_override: 临时正交化权重覆盖（E-P2-5）。
            complexity_weight_override: 临时复杂度权重覆盖（E-P2-5）。
        """
        # E-P2-1: 激活态自动迁移到正确 device
        sigma = activation.state.to(self.device)
        # sensory_input 在当前实现中未参与 J 更新计算，但保持接口一致性
        # 仍迁移到正确 device（向后兼容 + 防御性）
        sensory_input = sensory_input.to(self.device)

        # E-P2-5: 解析有效权重（override 优先，None 时回退实例属性）
        effective_orth = (orth_weight_override
                          if orth_weight_override is not None
                          else self.orth_weight)
        effective_cw = (complexity_weight_override
                        if complexity_weight_override is not None
                        else self.complexity_weight)

        # --- 层 1: sigma 正交化（改变学习方向） ---
        # 从当前模式中减去与已学结构（先验）重叠的部分
        # 只在"新颖"方向上做 Hebbian 学习
        if effective_orth > 0:
            recall = self.J @ sigma  # 网络对当前模式的回忆（先验）
            recall_energy = torch.dot(recall, recall)
            if recall_energy > 1e-10:
                # 当前 sigma 在回忆方向上的投影系数
                proj_coef = torch.dot(recall, sigma) / recall_energy
                # 减去投影（保留正交分量）
                sigma = sigma - effective_orth * proj_coef * recall
                # 保持在 Langevin 激活值的有效范围
                sigma = torch.clamp(sigma, -0.999, 0.999)

        # --- 准确性梯度：Hebbian 相关（在正交化 sigma 上） ---
        # 共激活的节点增强连接，降低自由能（形成吸引子）
        hebbian = torch.outer(sigma, sigma)  # [num_nodes, num_nodes]

        # --- 层 2: 复杂性梯度：饱和度压力 + 权重衰减 ---
        complexity_grad = self._complexity_gradient(
            sigma, orth_weight=effective_orth, complexity_weight=effective_cw)

        # --- 总更新: ΔJ = -∂F/∂J = Hebbian - 复杂性 ---
        # 注意符号: ∂F_accuracy/∂J = -Hebbian（负号因为增强连接降低自由能）
        #          ∂F_complexity/∂J = +complexity_grad（正号因为复杂性增加自由能）
        #          ΔJ = -(∂F_accuracy + ∂F_complexity) = Hebbian - complexity_grad
        delta_J = hebbian - complexity_grad

        # ── S4（2026-08-18）：EWC 持续学习保护（C 条件触发，只做加法）──
        # 开关关（self.ewc is None 或 enabled=False）→ 本块整体跳过，零参与，
        # 行为与开关引入前逐字节一致。开关开时：
        #   1. Fisher 对角累积（健康窗口 EMA）——healthy 由 σ 统计判定
        #      （valid 且非崩塌 act05>col_act 且非饱和 frac_gt0_9<sat_frac；
        #      col_act/sat_frac 复用 allostatic 语义，allostatic 关时用默认
        #      5/0.9）；崩塌/过渡态 → healthy=False → 不更新 Fisher（绝不在
        #      坏工作点计算）；grads = 对称化 ΔJ（= −∂F/∂J，平方后同）。
        #   2. EWC 罚项梯度（缩放不变方向罚）并入 ΔJ：
        #      ΔJ ← ΔJ − ∂P/∂J，P = λ·ΣFᵢ(θ̂ᵢ−θ̂*ᵢ)²，θ̂ = J/‖J‖_F
        #      （挂载点：对称化更新 J += η·sym(ΔJ) 之前；罚项梯度与
        #      Hebbian/复杂性同乘 η，Kirkpatrick 2017 标准）。θ̂ 归一使罚项
        #      对 A 重锚（标量乘 J，只改 ‖J‖ 不改方向）不变——C 豁免 A 的
        #      重锚缩放方向。公式与推导见本方法 docstring。
        ewc = self.ewc
        if ewc is not None and ewc.enabled:
            stats = compute_sigma_stats(activation.state)
            col_act = (self.allostatic.col_act
                       if self.allostatic is not None else 5)
            sat_frac = (self.allostatic.sat_frac
                        if self.allostatic is not None else 0.9)
            healthy = (stats.valid and stats.act05 > col_act
                       and stats.frac_gt0_9 < sat_frac)
            ewc.update(weights=self.J, grads=(delta_J + delta_J.T) / 2,
                       healthy=healthy)
            delta_J = delta_J - ewc.gradient(self.J)

        # 对称化（非序列模式）
        self.J = self.J + learning_rate * (delta_J + delta_J.T) / 2

        # 对角线置零（无自连接）
        self.J.fill_diagonal_(0)

        # 设计回归 2026-08-10（dandan 拍板：回到首版精神，治本而非开关）：
        # J 范数钳制（LMS_J_TARGET_NORM=40）为参数先验/复杂度上界，无条件生效
        # （惊讶度修复-01-设计方案.md §3.4）。J 永不发散，F 中复杂性项
        # 0.5·complexity_weight·‖J‖² = 0.5×0.01×1600 = O(8)，远小于准确性项
        # 常态 O(1~100)，不主导。与做梦 SHY（shy_target_norm=10）并存不冲突：
        # 在线钳制到 40 防发散，做梦 SHY 下调到 10 恢复容量。
        # 原理：J = J * min(1, target/‖J‖_F)。
        target = float(getattr(self, "j_target_norm", 40.0))
        j_norm = float(torch.norm(self.J, p="fro").item())
        if j_norm > target > 1e-8:
            self.J.mul_(target / j_norm)

    def _complexity_gradient(self, sigma: torch.Tensor,
                             orth_weight: Optional[float] = None,
                             complexity_weight: Optional[float] = None
                             ) -> torch.Tensor:
        """复杂性梯度：正交化压力 + 权重衰减。

        正交化压力（核心创新）:
            基于 J 矩阵的已有结构（饱和度）施加学习阻力。

            J 中已经大的连接代表被之前模式强化的方向。
            新模式在这些"已占用"方向上的 Hebbian 学习应该被抑制，
            迫使新模式使用网络中尚未被占用的节点子空间。

            这是一种"饱和度门控学习"：
                saturation = |J_ij| / max(|J|)     # 已学习程度
                orth_pressure = orth_weight * saturation * |Hebbian|

            结果：不同模式使用不同的连接子集，形成近似正交的吸引子。

            与 Langevin 扩散项配合使用：
            - 扩散项让相似输入产生不同的 sigma（打破对称性）
            - 正交化让不同 sigma 的学习不重叠（形成独立吸引子）

        权重衰减:
            complexity_weight * J，防止 J 发散，提供抗灾难性遗忘能力。
            保持 J 有界使得旧模式不会被新模式完全覆盖。

        E-P2-5: orth_weight / complexity_weight 参数允许调用方传入临时
            覆盖值（来自 learn() 的 override），为 None 时回退实例属性，
            保证向后兼容。

        参数:
            sigma: 当前激活态，形状 [num_nodes]。
            orth_weight: 正交化权重（覆盖 self.orth_weight）。
            complexity_weight: 复杂度权重（覆盖 self.complexity_weight）。

        返回:
            复杂性梯度矩阵，形状 [num_nodes, num_nodes]。
        """
        if orth_weight is None:
            orth_weight = self.orth_weight
        if complexity_weight is None:
            complexity_weight = self.complexity_weight

        # J 矩阵的饱和度：连接强度的归一化
        J_strength = self.J.abs()
        max_strength = J_strength.max()
        if max_strength > 1e-8:
            saturation = J_strength / max_strength  # 归一化到 [0, 1]
        else:
            saturation = torch.zeros_like(J_strength)

        # 当前 Hebbian 项的方向
        hebbian = torch.outer(sigma, sigma)

        # 正交化压力：在已饱和（已学习）的方向上施加阻力
        # saturation 高 = 这些连接已被之前的模式强化
        # 新模式在这些方向的学习被抑制，被迫寻找新的方向
        orth_pressure = orth_weight * saturation * hebbian.abs()

        # L2 权重衰减（防止 J 发散，抗灾难性遗忘）
        weight_decay = complexity_weight * self.J

        return orth_pressure + weight_decay

    # ------------------------------------------------------------------ #
    #  自由能与熵
    # ------------------------------------------------------------------ #

    def _compute_surprise(self, sigma: torch.Tensor,
                          sensory_input: torch.Tensor,
                          precision: torch.Tensor) -> float:
        """惊讶度 = 准确性项 = precision-weighted prediction error（恒 ≥ 0）。

        surprise = 0.5 · Σ_{i∈sensory} π_i · (σ_i − s_i)²
        = 主动推断中"precision-weighted prediction error"的标准二次型
          （Feldman & Friston 2010；Spisak & Friston v2 的 accuracy 分量——
          与论文 eq.(14) 逐节点 VFE 中的期望对数似然项构成**结构性对应
          （高斯似然类比）**：同为"期望对数似然"分量，但 LMS 用高斯二次型、
          论文用连续伯努利（CB）参数化，非字面同一公式，措辞以"类比"为准）。
        注意：只对感官节点（前 input_dim 个）求和；非感官节点无外部参照，
        不进入惊讶度。σ、s 均在 (-1,1) 附近、π ∈ [0.1, 10]，故恒 ≥ 0。
        """
        sensory_error = sigma[:self.input_dim] - sensory_input
        return float(0.5 * torch.sum(precision * sensory_error ** 2).item())

    def _compute_per_dim_surprise(self, sigma: torch.Tensor,
                                   sensory_input: torch.Tensor,
                                   precision: torch.Tensor) -> torch.Tensor:
        """逐维惊讶度 surprise_i = π_i·(σ_i−s_i)²，形状 [input_dim]。

        供目的层/注意力使用；detach + clone 避免与计算图耦合。
        """
        sensory_error = sigma[:self.input_dim] - sensory_input
        return (precision * sensory_error ** 2).detach().clone()

    def _compute_free_energy(self, sigma: torch.Tensor,
                             sensory_input: torch.Tensor,
                             precision: torch.Tensor) -> float:
        """计算自由能（未规范化变分能量，可负；严格 VFE ≥ 0 需 Bregman
        形式 = 后续项 §3.6；仅供学习目标与诊断）。

        F = 能量项 + 准确性项 + 复杂性项

        能量项（网络内部能量，负号因为共激活降低能量）:
            -0.5 * σ^T J σ - b^T σ
            高共激活 → 低能量 → 低自由能（状态更"自然"）

        准确性项（感官预测误差，与 surprise = _compute_surprise 数值一致）:
            0.5 * Σ precision_i * (σ_i - sensory_i)^2
            预测越准 → 误差越小 → 自由能越低

        复杂性项（模型复杂度正则化）:
            0.5 * complexity_weight * ||J||^2
            模型越简单 → 自由能越低

        注意：LMS 的 F 缺熵项 E_q[ln q] 与归一化常数，是**未规范化**变分
        能量，可负（深陷吸引子时能量项为负）——这是设计意图（"负值是能量，
        不是惊讶度"），不是缺陷。

        参数:
            sigma: 激活态，形状 [num_nodes]。
            sensory_input: 感官输入，形状 [input_dim]。
            precision: 精度向量，形状 [input_dim]。

        返回:
            自由能标量。越低表示状态越符合网络预期（仅学习目标/诊断）。
        """
        # 能量项：共激活越强，自由能越低
        corr_term = -0.5 * (sigma @ self.J @ sigma)
        bias_term = torch.dot(self.bias, sigma)

        # 准确性项：感官预测误差
        sensory_error = sigma[:self.input_dim] - sensory_input
        accuracy = 0.5 * torch.sum(precision * sensory_error ** 2)

        # 复杂性项：J 的 L2 正则化
        complexity = 0.5 * self.complexity_weight * torch.sum(self.J ** 2)

        free_energy = corr_term + bias_term + accuracy + complexity

        return float(free_energy)

    def _compute_entropy(self, sigma: torch.Tensor) -> float:
        """计算激活熵。

        熵衡量激活状态的分散程度:
          - 高熵：激活均匀分布（不确定状态）
          - 低熵：激活集中于少数节点（确定状态/明确吸引子）

        使用 |σ| 的信息熵近似（sigma 取值 (-1,1)，用绝对值度量激活强度）。

        参数:
            sigma: 激活态，形状 [num_nodes]。

        返回:
            熵标量。
        """
        abs_sigma = sigma.abs()
        # 归一化为概率分布
        total = abs_sigma.sum()
        if total < 1e-8:
            return 0.0
        p = abs_sigma / total
        entropy = -torch.sum(p * torch.log(p + 1e-8))
        return float(entropy)

    # ------------------------------------------------------------------ #
    #  景观快照
    # ------------------------------------------------------------------ #

    def get_landscape(self) -> dict:
        """返回当前吸引子景观状态（用于快照）。

        景观 = J矩阵 + bias + sigma 的完整状态。
        这就是"火种"——保存它即可在任何地方重新点燃同一身份。

        返回:
            包含 J、bias、sigma 的字典（均为张量副本）。
        """
        return {
            "J": self.J.clone(),
            "bias": self.bias.clone(),
            "sigma": self.sigma.clone(),
            "num_nodes": self.num_nodes,
            "input_dim": self.input_dim,
            # B 级（2026-08-19 四妹审核）：allostatic j_target 持久化
            # （只加不改字段；旧快照缺省回退 init_target）。
            # 用 snapshot(turn_count=None) 避免 side-effect（ts_turn 写入）。
            "allostatic": (
                self.allostatic.snapshot()
                if self.allostatic is not None else None),
        }

    def set_landscape(self, landscape: dict) -> None:
        """从快照恢复吸引子景观（重新点燃火种）。

        E-P2-1: 恢复的张量自动迁移到网络当前 device，保证快照跨设备
        恢复后张量设备一致。

        参数:
            landscape: get_landscape() 返回的字典。
        """
        self.J = landscape["J"].clone().to(self.device)
        self.bias = landscape["bias"].clone().to(self.device)
        self.sigma = landscape["sigma"].clone().to(self.device)
        self.num_nodes = landscape["num_nodes"]
        self.input_dim = landscape["input_dim"]
        # B 级（2026-08-19）：恢复 allostatic j_target（只加不改；
        # 缺省/非法值回退 init_target——重启后继续下探而非回 40）。
        if self.allostatic is not None:
            _as = landscape.get("allostatic") or {}
            _jt = _as.get("j_target")
            if isinstance(_jt, (int, float)) and _jt > 0:
                self.allostatic.j_target = max(
                    self.allostatic.j_min,
                    min(self.allostatic.j_max, float(_jt)))

    def reset_state(self) -> None:
        """重置内部状态 sigma 为零（不影响已学习的 J 矩阵）。

        E-P2-1: 零向量创建在网络当前 device 上。
        """
        self.sigma = torch.zeros(self.num_nodes, device=self.device)

    # ------------------------------------------------------------------ #
    #  allostatic J 原生（M4）：每轮观测 → 滑动设定点 → 写回 j_target_norm
    # ------------------------------------------------------------------ #

    def update_allostatic(self, surprise: float,
                          sigma_state: Optional[torch.Tensor] = None) -> float:
        """每轮观测（surprise + σ 激活态）→ 更新 allostatic J 滑动设定点。

        原生并入（M4）：本方法是 j_target_norm 的唯一写者（state_update 后
        emit 步）——计算 σ 统计 → 控制器重锚决策 → 写回 self.j_target_norm，
        learn() 的范数钳制按动态设定点执行。开关关（self.allostatic is None）
        → 零参与，返回当前固定值。fail-open：异常只告警不阻断主路径。

        参数:
            surprise: 本轮惊讶度（= Kalman innovation，Mehra 1970）。
            sigma_state: 本轮收敛激活态（torch.Tensor）。默认 None → 使用
                self.sigma（在线模式 infer 已写回时两者一致）。

        返回:
            更新后的 j_target_norm（设定点）。
        """
        if self.allostatic is None and self.homeostatic_bias is None:
            return self.j_target_norm
        try:
            if sigma_state is None:
                sigma_state = self.sigma
            stats = compute_sigma_stats(sigma_state)
            if self.allostatic is not None:
                self.j_target_norm = self.allostatic.update(surprise, stats)
            # 方案 B：HomeostaticBias 滑动设定点（每轮观测 σ 活跃度 → 重锚 bias）
            if self.homeostatic_bias is not None:
                new_bias = self.homeostatic_bias.update(sigma_state)
                if abs(new_bias - float(self.bias.mean())) > 1e-9:
                    self.bias = torch.full(
                        (self.num_nodes,), new_bias, device=self.device)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("allostatic/bias 更新失败（fail-open）: %s", e)
        return self.j_target_norm

    def allostatic_snapshot(self, turn_count: Optional[int] = None) -> dict:
        """观测块（/status allostatic_j；灵魂指标②J 动态 / ③越界触发）。

        开关关 → {'enabled': False}（调用方降级，零参与）。
        """
        if self.allostatic is None:
            return {'enabled': False}
        return self.allostatic.snapshot(turn_count=turn_count)

    def homeostatic_bias_snapshot(self) -> dict:
        """观测块（/status homeostatic_bias；方案 B bias 滑动曲线）。

        开关关（LMS_BIAS_ADAPT=0）→ {'enabled': False}（零参与）。
        """
        if getattr(self, 'homeostatic_bias', None) is None:
            return {'enabled': False}
        return self.homeostatic_bias.snapshot()

    def to(self, device: Union[str, torch.device]) -> 'AttractorNetwork':
        """将网络所有张量迁移到指定设备（E-P2-1）。

        迁移 J、bias、sigma 到目标设备，并更新 self.device。
        适用于运行时动态切换设备（如先在 CPU 上初始化，再迁移到 GPU）。

        参数:
            device: 目标设备（str / torch.device）。

        返回:
            self（支持链式调用）。
        """
        self.device = resolve_device(device)
        self.J = self.J.to(self.device)
        self.bias = self.bias.to(self.device)
        self.sigma = self.sigma.to(self.device)
        return self

"""allostatic J 滑动设定点（论文机制 A：Mehra 1970 + Sterling 2012 + Gama 2014）。

背景（2026-08-17 双快照干净复测实锤，见 memory/子AI任务-双快照干净复测-20260817.md）：
    B 级 J=7.0 固定钳制在生产持续学习下漂移失效——J 内容漂移 → norm 7 工作点
    从"动态区"滑入"崩塌边缘"（act05 ~2/256、σ_std 0.26-0.29、组间差翻负
    mean −0.345/−0.227）。dandan 拍板：不再扫描固定 J（Guo 2017 已证"单一
    全局温度在分布漂移下必然失效"），直接上论文机制 A——J 从固定常数 →
    allostatic 滑动设定点。

机制（论文机制 A，一次一变量，2026-08-17 实施）：
    J_target_norm(t) 不再是 env 固定常数，而是随数据流在线重估的滑动设定点：
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

治理开关：LMS_J_ALLOSTATIC（默认 0=关 → 固定 J 行为完全不变，回滚干净；
关时本模块全部路径零参与）。状态为纯进程内存（同 precision_adapt 先例：
重启即失、快照不落盘——回滚干净）。

fail-open：本模块所有方法异常静默（不阻塞调用方主路径）。

参数（全部 env 化，参数落盘表见 .env.example 与任务报告 20260817）：
    LMS_J_ALLOSTATIC             0/1     治理开关（默认 0=关）
    LMS_J_TARGET_NORM            7.0     初始设定点（复用现有变量；关=固定值不变）
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

from __future__ import annotations

import logging
import math
import os
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import torch

logger = logging.getLogger(__name__)

# 观测快照保留长度（j_history / events 上限）
_SNAPSHOT_HISTORY = 200


# ================================================================== #
#  治理开关
# ================================================================== #

def allostatic_j_enabled(explicit: Optional[bool] = None) -> bool:
    """治理开关解析：显式参数 > 环境变量 LMS_J_ALLOSTATIC（默认 0=关）。

    布尔接受（不区分大小写）：1/true/yes/on 视为开，其余为关。
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get('LMS_J_ALLOSTATIC', '0')
    return raw.strip().lower() in ('1', 'true', 'yes', 'on')


# ================================================================== #
#  σ 状态统计（越界信号数据源）
# ================================================================== #

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


# ================================================================== #
#  allostatic J 控制器（滑动设定点）
# ================================================================== #

class AllostaticJController:
    """allostatic J 滑动设定点控制器。

    纯进程内存状态（同 precision_adapt 先例：重启即失、快照不落盘、
    回滚干净）。enabled=False 时所有方法为 no-op（返回初始固定值，
    零参与，行为与开关引入前完全一致）。

    机制参数（全部可 env 化；非判定阈值，机制参数允许默认值）:
        init_target: 初始设定点（= 当前 env LMS_J_TARGET_NORM，不另设固定值）。
        window:      surprise 窗口长度（innovation 统计）。
        k:           重锚阈值（z 带 ±k；Mehra 1970 innovation 漂移检测）。
        step:        每次重锚的设定点步长。
        persist:     越带持续轮数（防单轮抖动）。
        j_min/j_max: 设定点上下限（护栏，防设定点 runaway）。
        min_samples: 冷启动样本数（不足 → 不动作，保持初始设定点）。
        sat_frac:    σ 饱和判定阈值（frac_gt0.9 ≥ 此值 → 饱和）。
        col_act:     σ 崩塌判定阈值（act05 ≤ 此值 → 崩塌）。
    """

    def __init__(self, enabled: Optional[bool] = None,
                 init_target: float = 7.0,
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

        # --- σ 越界硬信号 ---
        # （sat/col 已在观测登记时计算）

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

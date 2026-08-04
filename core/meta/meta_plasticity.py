"""
活体记忆系统 - 元可塑性控制器（MetaPlasticityController）
========================================================

这是系统的"最后一公里"——让学习规则参数自身随对话共生演化。

在 FEP（自由能原理）框架中，这是元层级的应用——"精度的精度"：
  - 第 1 层（已有）：每轮最小化自由能（调整 J 矩阵和 precision）
  - 第 2 层（本模块）：最小化"元自由能"（调整学习规则参数本身）

四个元学习规则（全部局部、无需反向传播）：
  1. surprise 趋势 → lr_multiplier:    惊讶趋势上升 → 学得更快
  2. coherence     → orth_multiplier:  目的不一致   → 增加多样性压力
  3. collapse_rate → temp_multiplier:  坍缩频繁     → 提高探索温度
  4. ||J||_F       → cw_multiplier:    J 矩阵饱和   → 加强复杂度惩罚

设计约束：
  - 慢时间尺度：每 meta_interval 轮更新一次（默认 10 轮）
  - 有界调整：所有倍率在 [0.5, 2.0] 范围内
  - 纯局部规则：无需 BPTT，无需全局 loss
  - 状态可持久化：get_state() / set_state()

参考：架构文档第八节《元可塑性与学习规则的演化》
"""

import math
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  辅助函数
# ------------------------------------------------------------------ #

def _sigmoid(x: float) -> float:
    """数值稳定的 sigmoid 函数。

    使用分段计算避免大 |x| 时的数值溢出：
      - x >= 0: 1 / (1 + exp(-x))
      - x <  0: exp(x) / (1 + exp(x))

    参数:
        x: 任意实数。

    返回:
        sigmoid 值，范围 (0, 1)。
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def _clamp(value: float, lo: float, hi: float) -> float:
    """将值限制在 [lo, hi] 范围内。

    参数:
        value: 待限制的值。
        lo: 下限。
        hi: 上限。

    返回:
        限制后的值，保证 lo <= 返回值 <= hi。
    """
    return min(max(value, lo), hi)


def _ema(values: list, alpha: float) -> float:
    """计算指数移动平均（Exponential Moving Average）。

    EMA_t = alpha * value_t + (1 - alpha) * EMA_{t-1}
    初始值 EMA_0 = values[0]

    参数:
        values: 数值序列（按时间顺序）。
        alpha: 平滑因子，越大越敏感于近期值。

    返回:
        EMA 标量。空序列返回 0.0。
    """
    if not values:
        return 0.0
    result = values[0]
    for v in values[1:]:
        result = alpha * v + (1.0 - alpha) * result
    return result


# ------------------------------------------------------------------ #
#  元状态数据结构
# ------------------------------------------------------------------ #

@dataclass
class MetaState:
    """元学习参数状态，可持久化。

    记录四个元参数倍率的当前值，以及用于趋势计算的信号历史。
    倍率因子 1.0 表示使用基准值（不调整）。

    属性:
        lr_multiplier: 学习率倍率。>1 学更快，<1 放慢。
        orth_multiplier: 正交化权重倍率。>1 增加多样性压力。
        temp_multiplier: 温度倍率。>1 提高探索温度。
        cw_multiplier: 复杂度权重倍率。>1 加强复杂度惩罚。
        surprise_history: 最近 N 轮的 surprise 值（用于趋势计算）。
        collapse_count: 累计坍缩次数。
        update_count: 元更新次数（每 meta_interval 轮计一次）。
    """

    # 四个元参数的当前值（倍率因子，1.0 = 基准值）
    lr_multiplier: float = 1.0      # 学习率倍率
    orth_multiplier: float = 1.0    # 正交化权重倍率
    temp_multiplier: float = 1.0   # 温度倍率
    cw_multiplier: float = 1.0      # 复杂度权重倍率

    # 信号历史（用于趋势计算）
    surprise_history: list = field(default_factory=list)  # 最近 N 轮的 surprise 值
    collapse_count: int = 0     # 累计坍缩次数
    update_count: int = 0       # 元更新次数


# ------------------------------------------------------------------ #
#  元可塑性控制器
# ------------------------------------------------------------------ #

class MetaPlasticityController:
    """元可塑性控制器：让学习规则参数随系统状态自演化。

    FEP 元层级应用：
      - 第 1 层（已有）：每轮最小化自由能（调整 J 和 precision）
      - 第 2 层（新增）：最小化"元自由能"（调整学习规则参数本身）

    四个元学习规则（全部局部、无 BP）：
      1. surprise_trend → lr_multiplier:  惊讶趋势上升 → 学更快
      2. coherence → orth_multiplier:       目的不一致   → 增加多样性压力
      3. collapse_rate → temp_multiplier:   坍缩频繁     → 提高探索温度
      4. ||J||_F → cw_multiplier:          J 矩阵饱和   → 加强复杂度惩罚

    设计约束：
      - 慢时间尺度：每 meta_interval 轮更新一次（默认 10 轮）
      - 有界调整：所有倍率在 [bounds_min, bounds_max] 范围内
      - 纯局部规则：无需 BPTT，无需全局 loss
      - 状态可持久化：get_state() / set_state()

    使用方式：
        controller = MetaPlasticityController(config)
        # 每轮对话后调用 update，收集信号
        result = controller.update(surprise, coherence, collapse, j_norm)
        # 获取调整后的有效参数（乘以倍率）
        params = controller.get_adjusted_params(
            base_lr=0.01, base_orth=0.5, base_temp=0.05, base_cw=0.01
        )
        # 将 params 传给 AttractorNetwork.learn() 等方法
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        """初始化元可塑性控制器。

        参数:
            config: 配置字典，支持以下键：
                - meta_interval: 元更新频率（默认 10 轮）
                - meta_lr: 元学习率，控制倍率平滑速度（默认 0.01）
                - bounds_min: 倍率下限（默认 0.5）
                - bounds_max: 倍率上限（默认 2.0）
                - surprise_window: surprise 历史窗口大小（默认 20）
                - orth_alpha: coherence → orth 的系数（默认 1.0）
                - temp_beta: collapse → temp 的系数（默认 5.0）
                - cw_gamma: ||J|| → cw 的系数（默认 1.0）
                - lr_delta: surprise 趋势 → lr 的系数（默认 2.0）
                - shy_target_norm: J 矩阵目标范数，用于计算饱和度（默认 10.0）

        说明：
            meta_lr 作为倍率更新的平滑因子——每次元更新时，
            新倍率 = (1 - meta_lr) * 旧倍率 + meta_lr * 目标倍率。
            值越小变化越缓慢，保证"慢时间尺度"特性。
        """
        config = config or {}

        # --- 元更新时序 ---
        self.meta_interval: int = config.get('meta_interval', 10)
        self.meta_lr: float = config.get('meta_lr', 0.01)

        # --- 倍率边界 ---
        self.bounds_min: float = config.get('bounds_min', 0.5)
        self.bounds_max: float = config.get('bounds_max', 2.0)

        # --- surprise 历史窗口 ---
        self.surprise_window: int = config.get('surprise_window', 20)

        # --- 四个规则的系数 ---
        self.lr_delta: float = config.get('lr_delta', 2.0)
        self.orth_alpha: float = config.get('orth_alpha', 1.0)
        self.temp_beta: float = config.get('temp_beta', 5.0)
        self.cw_gamma: float = config.get('cw_gamma', 1.0)

        # --- J 矩阵目标范数 ---
        self.shy_target_norm: float = config.get('shy_target_norm', 10.0)

        # --- EMA 平滑因子 ---
        # 标准 EMA alpha = 2 / (N + 1)，N 为窗口大小
        self.ema_alpha: float = 2.0 / (self.surprise_window + 1)

        # --- 内部状态 ---
        self.state: MetaState = MetaState()

        # surprise 历史使用 deque，自动丢弃旧值
        # 容量为 2 * window，确保能取出 recent 和 older 两个窗口
        self._surprise_deque: deque = deque(maxlen=self.surprise_window * 2)

        # 总轮数计数器（独立于 state.update_count）
        self._turn_count: int = 0

        logger.info(
            f"MetaPlasticityController 已初始化 "
            f"(interval={self.meta_interval}, meta_lr={self.meta_lr}, "
            f"bounds=[{self.bounds_min}, {self.bounds_max}], "
            f"window={self.surprise_window})"
        )

    # ------------------------------------------------------------------ #
    #  主接口：信号收集与元调整
    # ------------------------------------------------------------------ #

    def update(self, surprise: float, coherence: float,
               collapse_occurred: bool, j_norm: float) -> Optional[dict]:
        """收集信号，达到间隔时执行元调整。

        每轮调用，但仅在 turn_count % meta_interval == 0 时实际调整。
        这是"慢时间尺度"设计的体现——元参数不会每轮抖动，
        而是积累足够信号后才做一次有依据的调整。

        参数:
            surprise: 当前轮的惊讶度（自由能），来自 AttractorNetwork。
            coherence: 当前目的层 coherence (0-1)，来自 PurposeLayer。
            collapse_occurred: 本轮是否发生坍缩（吸引子坍缩到
                已有模式，丧失区分新输入的能力）。
            j_norm: 当前 J 矩阵的 Frobenius 范数，衡量网络饱和度。

        返回:
            - 通常返回 None（仅收集信号）
            - 触发调整时返回调整后的倍率字典:
              {'lr_mult': float, 'orth_mult': float,
               'temp_mult': float, 'cw_mult': float}
        """
        # --- 每轮收集信号 ---
        self._surprise_deque.append(surprise)
        self._turn_count += 1

        if collapse_occurred:
            self.state.collapse_count += 1

        # --- 判断是否触发元调整 ---
        if self.meta_interval <= 0 or self._turn_count % self.meta_interval != 0:
            return None

        # --- 执行元调整 ---
        self.state.update_count += 1

        surprise_list = list(self._surprise_deque)

        # 规则 1：学习率调整（surprise 趋势驱动）
        target_lr = self._compute_lr_multiplier(surprise_list)

        # 规则 2：正交化权重调整（coherence 驱动）
        target_orth = self._compute_orth_multiplier(coherence)

        # 规则 3：温度调整（坍缩率驱动）
        target_temp = self._compute_temp_multiplier()

        # 规则 4：复杂度权重调整（J 矩阵饱和度驱动）
        target_cw = self._compute_cw_multiplier(j_norm)

        # --- 平滑更新：用 meta_lr 渐进逼近目标值 ---
        # 这保证元参数不会突变，符合"慢时间尺度"设计
        self.state.lr_multiplier = self._smooth_update(
            self.state.lr_multiplier, target_lr)
        self.state.orth_multiplier = self._smooth_update(
            self.state.orth_multiplier, target_orth)
        self.state.temp_multiplier = self._smooth_update(
            self.state.temp_multiplier, target_temp)
        self.state.cw_multiplier = self._smooth_update(
            self.state.cw_multiplier, target_cw)

        # 同步 surprise_history 到 state（用于持久化）
        self.state.surprise_history = surprise_list

        result = {
            'lr_mult': self.state.lr_multiplier,
            'orth_mult': self.state.orth_multiplier,
            'temp_mult': self.state.temp_multiplier,
            'cw_mult': self.state.cw_multiplier,
        }

        logger.info(
            f"元更新 #{self.state.update_count} "
            f"(turn={self._turn_count}): "
            f"lr={self.state.lr_multiplier:.4f}, "
            f"orth={self.state.orth_multiplier:.4f}, "
            f"temp={self.state.temp_multiplier:.4f}, "
            f"cw={self.state.cw_multiplier:.4f} | "
            f"signals: surprise={surprise:.4f}, "
            f"coherence={coherence:.4f}, "
            f"collapse={collapse_occurred}, "
            f"j_norm={j_norm:.4f}"
        )

        return result

    # ------------------------------------------------------------------ #
    #  四个元学习规则
    # ------------------------------------------------------------------ #

    def _compute_lr_multiplier(self, surprise_list: list) -> float:
        """规则 1：学习率调整（surprise 趋势驱动）。

        计算 surprise_history 的 EMA 趋势：
            trend = EMA(recent_window) - EMA(older_window)
            lr_multiplier = clamp(1.0 + lr_delta * (2*sigmoid(trend) - 1),
                                  bounds_min, bounds_max)

        使用 2*sigmoid(trend) - 1 将 sigmoid 从 (0,1) 映射到 (-1,1)，
        使得 trend=0 时倍率恰为 1.0（中性基准）。
        这等价于 tanh(trend/2)。

        - 趋势上升 (trend > 0) → 倍率 > 1 → 学得更快
        - 趋势下降 (trend < 0) → 倍率 < 1 → 放慢学习
        - 无趋势     (trend = 0) → 倍率 = 1 → 保持基准

        参数:
            surprise_list: surprise 历史值列表。

        返回:
            学习率倍率的目标值（未经平滑）。
        """
        window = self.surprise_window

        if len(surprise_list) >= window * 2:
            # 有足够历史：计算两个窗口的 EMA 差值
            recent = surprise_list[-window:]
            older = surprise_list[-2 * window:-window]
            trend = (
                _ema(recent, self.ema_alpha)
                - _ema(older, self.ema_alpha)
            )
        else:
            # 历史不足：无趋势
            trend = 0.0

        # 2*sigmoid(trend) - 1 将 trend 从 R 映射到 (-1, 1)
        # trend=0 → 0（中性），trend>0 → 正（加速），trend<0 → 负（减速）
        normalized_trend = 2.0 * _sigmoid(trend) - 1.0
        target = 1.0 + self.lr_delta * normalized_trend
        return _clamp(target, self.bounds_min, self.bounds_max)

    def _compute_orth_multiplier(self, coherence: float) -> float:
        """规则 2：正交化权重调整（coherence 驱动）。

            orth_multiplier = clamp(1.0 + orth_alpha * (1.0 - coherence),
                                    bounds_min, bounds_max)

        - coherence 高（目的一致） → (1-coh) 小 → 倍率接近 1 → 正常正交化
        - coherence 低（目的混乱） → (1-coh) 大 → 倍率 > 1 → 增加多样性压力
          （强制不同模式使用不同连接子集，形成更可区分的吸引子）

        参数:
            coherence: 目的层 coherence 值，范围 [0, 1]。

        返回:
            正交化权重倍率的目标值（未经平滑）。
        """
        target = 1.0 + self.orth_alpha * (1.0 - coherence)
        return _clamp(target, self.bounds_min, self.bounds_max)

    def _compute_temp_multiplier(self) -> float:
        """规则 3：温度调整（坍缩率驱动）。

            collapse_rate = collapse_count / max(update_count, 1)
            temp_multiplier = clamp(1.0 + temp_beta * collapse_rate,
                                     bounds_min, bounds_max)

        - 坍缩频繁 → collapse_rate 高 → 倍率 > 1 → 提高探索温度
          （Langevin 扩散项增强，打破对称性，让相似输入有机会
          收敛到不同吸引子，从而让正交化学习规则发挥作用）
        - 无坍缩   → collapse_rate = 0 → 倍率 = 1 → 保持基准温度

        参数:
            无（使用内部累积的 collapse_count 和 update_count）。

        返回:
            温度倍率的目标值（未经平滑）。
        """
        collapse_rate = (
            self.state.collapse_count / max(self.state.update_count, 1)
        )
        target = 1.0 + self.temp_beta * collapse_rate
        return _clamp(target, self.bounds_min, self.bounds_max)

    def _compute_cw_multiplier(self, j_norm: float) -> float:
        """规则 4：复杂度权重调整（J 矩阵饱和度驱动）。

            saturation = j_norm / shy_target_norm
            cw_multiplier = clamp(
                1.0 + cw_gamma * (sigmoid(saturation) * 2.0 - 1.0),
                bounds_min, bounds_max
            )

        使用 sigmoid(saturation) * 2.0 - 1.0 将 sigmoid 从 (0,1) 映射到 (-1,1)，
        使得 saturation=0 时倍率恰为 1.0（中性基准）。
        当 cw_gamma=1.0 时，等价于 clamp(sigmoid(saturation) * 2.0, ...)。

        - J 矩阵小（saturation → 0） → sigmoid(0)=0.5 → 倍率 ≈ 1 → 正常复杂度
        - J 矩阵大（saturation → ∞） → sigmoid → 1 → 倍率 → 2 → 加强复杂度惩罚
          （L2 权重衰减增强，防止 J 发散，提供抗灾难性遗忘能力）

        参数:
            j_norm: 当前 J 矩阵的 Frobenius 范数。

        返回:
            复杂度权重倍率的目标值（未经平滑）。
        """
        saturation = j_norm / self.shy_target_norm
        # sigmoid(saturation) * 2.0 - 1.0 将 saturation 映射到 (-1, 1)
        # saturation=0 → 0（中性），saturation 大 → 接近 1（加强惩罚）
        normalized_saturation = _sigmoid(saturation) * 2.0 - 1.0
        target = 1.0 + self.cw_gamma * normalized_saturation
        return _clamp(target, self.bounds_min, self.bounds_max)

    # ------------------------------------------------------------------ #
    #  参数应用
    # ------------------------------------------------------------------ #

    def get_adjusted_params(self, base_lr: float, base_orth: float,
                            base_temp: float, base_cw: float) -> dict:
        """返回调整后的有效参数值。

        将基准参数乘以对应的元倍率，得到当前实际应使用的参数值。
        调用方（如 LivingMemoryLoop）应将返回值传给 AttractorNetwork
        和 PurposeLayer 的相关方法。

        参数:
            base_lr: 基准学习率（从 config 或 attractor 读取）。
            base_orth: 基准正交化权重。
            base_temp: 基准温度。
            base_cw: 基准复杂度权重。

        返回:
            调整后的参数字典:
            {
                'learning_rate': base_lr * lr_multiplier,
                'orth_weight': base_orth * orth_multiplier,
                'temperature': base_temp * temp_multiplier,
                'complexity_weight': base_cw * cw_multiplier,
            }
        """
        return {
            'learning_rate': base_lr * self.state.lr_multiplier,
            'orth_weight': base_orth * self.state.orth_multiplier,
            'temperature': base_temp * self.state.temp_multiplier,
            'complexity_weight': base_cw * self.state.cw_multiplier,
        }

    # ------------------------------------------------------------------ #
    #  持久化
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict:
        """返回可持久化的状态字典。

        包含所有元参数倍率、信号历史和计数器，
        可通过 set_state() 完整恢复控制器状态。

        返回:
            状态字典，包含以下键:
            - lr_multiplier, orth_multiplier, temp_multiplier, cw_multiplier
            - surprise_history (list)
            - collapse_count, update_count
            - turn_count
        """
        return {
            'lr_multiplier': self.state.lr_multiplier,
            'orth_multiplier': self.state.orth_multiplier,
            'temp_multiplier': self.state.temp_multiplier,
            'cw_multiplier': self.state.cw_multiplier,
            'surprise_history': list(self._surprise_deque),
            'collapse_count': self.state.collapse_count,
            'update_count': self.state.update_count,
            'turn_count': self._turn_count,
        }

    def set_state(self, state: dict) -> None:
        """从持久化状态恢复。

        参数:
            state: get_state() 返回的状态字典。
                向后兼容：缺失的键使用默认值。
        """
        self.state.lr_multiplier = state.get('lr_multiplier', 1.0)
        self.state.orth_multiplier = state.get('orth_multiplier', 1.0)
        self.state.temp_multiplier = state.get('temp_multiplier', 1.0)
        self.state.cw_multiplier = state.get('cw_multiplier', 1.0)
        self.state.collapse_count = state.get('collapse_count', 0)
        self.state.update_count = state.get('update_count', 0)
        self._turn_count = state.get('turn_count', 0)

        # 恢复 surprise 历史 deque
        history = state.get('surprise_history', [])
        self._surprise_deque = deque(
            history, maxlen=self.surprise_window * 2)
        self.state.surprise_history = list(self._surprise_deque)

        logger.info(
            f"MetaPlasticityController 状态已恢复 "
            f"(turn={self._turn_count}, updates={self.state.update_count})"
        )

    # ------------------------------------------------------------------ #
    #  状态查询
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        """返回状态摘要（用于日志和调试）。

        返回:
            状态字典，包含当前倍率、计数器和下次更新倒计时。
        """
        if self.meta_interval > 0:
            turns_since_update = self._turn_count % self.meta_interval
            next_update_in = self.meta_interval - turns_since_update
        else:
            next_update_in = 0

        return {
            'turn_count': self._turn_count,
            'update_count': self.state.update_count,
            'collapse_count': self.state.collapse_count,
            'lr_multiplier': self.state.lr_multiplier,
            'orth_multiplier': self.state.orth_multiplier,
            'temp_multiplier': self.state.temp_multiplier,
            'cw_multiplier': self.state.cw_multiplier,
            'surprise_history_len': len(self._surprise_deque),
            'next_update_in': next_update_in,
        }

    # ------------------------------------------------------------------ #
    #  内部辅助
    # ------------------------------------------------------------------ #

    def _smooth_update(self, current: float, target: float) -> float:
        """使用 meta_lr 平滑更新倍率值。

        new_value = (1 - meta_lr) * current + meta_lr * target

        meta_lr 越小，变化越缓慢。这是"慢时间尺度"的保障：
        元参数不会因单次信号波动而剧烈跳变，而是渐进地
        向目标值靠拢。

        参数:
            current: 当前倍率值。
            target: 目标倍率值（由规则计算得出）。

        返回:
            平滑后的新倍率值。
        """
        return (1.0 - self.meta_lr) * current + self.meta_lr * target

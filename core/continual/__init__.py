"""
活体记忆系统 - 持续学习（continual learning）
=============================================

ABC 操作规划 §S4「C 条件触发（EWC 保护）」的模块落点（默认关，env 门控）：
  - FisherAccumulator: 对角 Fisher 信息近似（健康窗口上梯度平方的 EMA
    累积，Kirkpatrick 2017 标准做法；绝不在崩塌/过渡态计算）。
  - EwcPenalty: 缩放不变二次罚项 λ·ΣFᵢ(θ̂ᵢ−θ̂*ᵢ)²（θ̂ = θ/‖θ‖_F，
    豁免 A 重锚的整体缩放方向——C 不抵抗 A 重锚）。

治理开关 LMS_EWC_ENABLE（默认 0=关；关 → 零参与）。纯进程内存状态，
无 IO 依赖，只依赖标准库与 torch（不 import 本仓库运行时）。
"""

from core.continual.ewc import EwcPenalty, FisherAccumulator, ewc_enabled

__all__ = ["FisherAccumulator", "EwcPenalty", "ewc_enabled"]

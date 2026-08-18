"""
活体记忆系统 - 数据类型定义
============================

定义核心层各模块间传递的数据结构。
所有数据类型均为 dataclass，保证接口清晰、可序列化。

参考：架构文档 第五节《接口定义》5.1 数据类型
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import torch


# ------------------------------------------------------------------ #
#  设备管理辅助函数（E-P2-1）
# ------------------------------------------------------------------ #

def resolve_device(device: Union[str, torch.device, None] = "auto") -> torch.device:
    """将设备标识解析为 ``torch.device`` 对象。

    统一各核心组件的设备解析逻辑。支持以下输入：

      - ``"auto"``：自动检测 CUDA 可用性，可用则选 ``cuda``，否则 ``cpu``。
      - ``"cpu"``、``"cuda"``、``"cuda:0"`` 等字符串：直接构造 ``torch.device``。
      - 已构造的 ``torch.device`` 对象：原样返回。
      - ``None``：视为 ``"auto"``。

    参数:
        device: 设备标识（str / torch.device / None）。

    返回:
        解析后的 ``torch.device`` 对象。
    """
    if device is None or (isinstance(device, str) and device == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


@dataclass
class SensoryInput:
    """感官输入：从对话文本编码而来的向量信号。

    由桥接层 Encoder 生成，输入到海马体核心的吸引子网络。

    属性:
        vector: 感官向量，形状 [input_dim]，即 embedding 维度。
        metadata: 元数据字典，记录时间戳、来源等附加信息。
    """

    vector: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Activation:
    """海马体激活态：吸引子网络的输出。

    FEP 推断收敛后的网络状态，包含激活值、熵、惊讶度与自由能。

    属性:
        state: 节点激活值，形状 [num_nodes]，取值范围 (-1, 1)（Langevin 函数输出）。
        entropy: 激活熵，衡量状态的不确定性/分散程度。
        surprise: 惊讶度 = 准确性项（precision-weighted prediction error），
            恒 ≥ 0。供报告/回放/目的层/做梦/元可塑性使用。
        free_energy: 自由能（未规范化变分能量，可负；严格 VFE ≥ 0 需
            Bregman 形式 = 后续项 §3.6），= 现有 F 公式。
            仅供学习目标与诊断；默认 0.0 保证旧构造点兼容。
        per_dim_surprise: 逐维惊讶度 π_i·(σ_i−s_i)² [input_dim]，
            供目的层/注意力使用；默认 None（旧构造点兼容）。
        mse: 均方预测误差 (1/input_dim)·Σ(σ_i−s_i)²，跨尺度可比；
            默认 None（旧构造点兼容）。
    """

    state: torch.Tensor
    entropy: float
    surprise: float          # 语义变更：准确性项（原为自由能）
    free_energy: float = 0.0        # 新增（默认值 → 旧构造点零破坏）
    per_dim_surprise: Optional[torch.Tensor] = None   # 新增
    mse: Optional[float] = None     # 新增


@dataclass
class PurposeState:
    """目的层状态：SLM 当前的"兴趣/关注"分布。

    目的不是外部预设的，而是从 SLM 自身活动中涌现的。
    precision 的演化轨迹即为 SLM "兴趣"的成型过程。

    属性:
        precision: 感官精度向量，形状 [input_dim]，每个感官维度的信任度。
        history: precision 演化历史，列表中的每个元素是一份 precision 快照。
        coherence: 目的的内部一致性，标量。衡量 precision 是否稳定、自洽。
        encounter_count: 习惯化计数器，形状 [input_dim]，记录每个感官维度
            被显著激活的累积次数。用于习惯化机制（N3 修复：跨会话持久化）。
            为可选字段，向后兼容旧版快照（无此字段时为 None）。
    """

    precision: torch.Tensor
    history: list[torch.Tensor]
    coherence: float
    encounter_count: Optional[torch.Tensor] = None

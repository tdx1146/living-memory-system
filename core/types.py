"""
活体记忆系统 - 数据类型定义
============================

定义核心层各模块间传递的数据结构。
所有数据类型均为 dataclass，保证接口清晰、可序列化。

参考：架构文档 第五节《接口定义》5.1 数据类型
"""

from dataclasses import dataclass, field
from typing import Any

import torch


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

    FEP 推断收敛后的网络状态，包含激活值、熵和惊讶度（自由能）。

    属性:
        state: 节点激活值，形状 [num_nodes]，取值范围 (-1, 1)（Langevin 函数输出）。
        entropy: 激活熵，衡量状态的不确定性/分散程度。
        surprise: 自由能（惊讶度），标量。越低表示当前状态越符合网络预期。
    """

    state: torch.Tensor
    entropy: float
    surprise: float


@dataclass
class PurposeState:
    """目的层状态：SLM 当前的"兴趣/关注"分布。

    目的不是外部预设的，而是从 SLM 自身活动中涌现的。
    precision 的演化轨迹即为 SLM "兴趣"的成型过程。

    属性:
        precision: 感官精度向量，形状 [input_dim]，每个感官维度的信任度。
        history: precision 演化历史，列表中的每个元素是一份 precision 快照。
        coherence: 目的的内部一致性，标量。衡量 precision 是否稳定、自洽。
    """

    precision: torch.Tensor
    history: list[torch.Tensor]
    coherence: float

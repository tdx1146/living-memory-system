"""
活体记忆系统 - 核心配置
========================

集中管理核心层的所有可配置参数。
使用 dataclass 保证类型安全与可序列化。

参考：架构文档 第三节《模块划分》、第九节《技术栈》
"""

from dataclasses import dataclass


@dataclass
class CoreConfig:
    """核心层统一配置。

    所有核心模块共享此配置实例，保证参数一致性。
    可通过修改此处的默认值来调整整个系统的行为。

    属性:
        num_nodes: 吸引子网络节点数（建议 256-1024 起步）。
        input_dim: 感官输入维度（= embedding 维度）。
            网络的前 input_dim 个节点为感官节点，接收外部 clamping。
        learning_rate: FEP 学习率 η。
        num_infer_steps: FEP 推断的迭代步数 K。
        short_term_decay: 短时记忆衰减系数（越小遗忘越快）。
        long_term_decay: 长时记忆衰减系数（越接近 1 保留越久）。
        precision_min: precision 下限，防止发散。
        precision_max: precision 上限，防止发散。
        precision_lr: 目的层 precision 调整的学习率。
        complexity_weight: 自由能中复杂性项的权重（L2 正则化系数）。
        orth_weight: 正交化压力权重（复杂性梯度的竞争性抑制强度）。
        coherence_threshold: coherence 低于此阈值时触发元目的翻转。
        min_history_length: 触发元目的翻转所需的最短历史长度。
        meta_window: 元目的翻转时回看的历史窗口大小。
        seed: 随机种子，保证可复现性。
    """

    # 网络结构
    num_nodes: int = 256
    input_dim: int = 64

    # 学习与推断
    learning_rate: float = 0.01
    num_infer_steps: int = 10

    # 记忆管理
    short_term_decay: float = 0.8
    long_term_decay: float = 0.999

    # 目的层
    precision_min: float = 0.1
    precision_max: float = 10.0
    precision_lr: float = 0.1
    coherence_threshold: float = 0.3
    min_history_length: int = 5
    meta_window: int = 10

    # FEP 学习规则
    complexity_weight: float = 0.01
    orth_weight: float = 0.5

    # 随机种子
    seed: int = 42

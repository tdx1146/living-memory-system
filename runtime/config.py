"""运行时配置

定义运行时的配置，包括LLM API配置、快照路径、运行模式等。
与core/config.py不同，这里关注的是运行时层面的配置（IO、外部服务）。
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RuntimeConfig:
    """运行时配置（合并 CoreConfig 字段，作为统一配置入口）。

    运行时层面配置（IO、外部服务）与核心层 FEP 参数合并于此，
    使 LivingMemoryLoop 可通过单一配置字典驱动所有组件。
    未在 CoreConfig 中出现的运行时字段（如 llm_api、auto_snapshot）
    保留在本类；CoreConfig 的全部字段以同名方式并入，保证键名一致。

    属性:
        # 网络结构
        num_nodes: 吸引子网络节点数
        input_dim: 感官输入维度
        # 学习与推断
        learning_rate: FEP 学习率 η
        num_infer_steps: FEP 推断的迭代步数 K
        temperature: Langevin 动力学温度（扩散项噪声强度）
        # 记忆管理
        short_term_decay: 短时记忆衰减系数
        long_term_decay: 长时记忆衰减系数
        transfer_rate: 巩固时短时→长时的迁移率
        replay_count: 巩固时回放的条目数
        replay_weight: 巩固时回放的权重
        consolidation_decay: 巩固后短时记忆的衰减系数
        buffer_capacity: 记忆缓冲区容量
        # 目的层
        precision_min: precision 下限
        precision_max: precision 上限
        precision_lr: 目的层 precision 调整的学习率
        coherence_threshold: coherence 低于此阈值时触发元目的翻转
        min_history_length: 触发元目的翻转所需的最短历史长度
        meta_window: 元目的翻转时回看的历史窗口大小
        max_history: 目的层 precision 历史上限
        habituation_rate: 习惯化衰减率
        activation_threshold: 习惯化激活阈值（N4，解耦与 temperature 的耦合）
        # FEP 学习规则
        complexity_weight: 自由能中复杂性项的权重
        orth_weight: 正交化压力权重
        # 随机种子
        seed: 随机种子
        # 运行时
        snapshot_dir: 快照保存目录
        decoder_mode: 解码器模式（'text'或'vector'）
        consolidation_interval: 记忆巩固间隔（每N轮巩固一次）
        auto_snapshot: 是否自动快照
        auto_snapshot_interval: 自动快照间隔（每N轮）
        llm_api: LLM API配置字典
    """
    # 网络结构
    num_nodes: int = 256
    input_dim: int = 64

    # 学习与推断
    learning_rate: float = 0.01
    num_infer_steps: int = 10
    temperature: float = 0.05

    # 记忆管理
    short_term_decay: float = 0.8
    long_term_decay: float = 0.999
    transfer_rate: float = 0.1
    replay_count: int = 10
    replay_weight: float = 0.01
    consolidation_decay: float = 0.5
    buffer_capacity: int = 100

    # 目的层
    precision_min: float = 0.1
    precision_max: float = 10.0
    precision_lr: float = 0.1
    coherence_threshold: float = 0.3
    min_history_length: int = 5
    meta_window: int = 10
    max_history: int = 100
    habituation_rate: float = 0.05
    activation_threshold: float = 0.3

    # FEP 学习规则
    complexity_weight: float = 0.01
    orth_weight: float = 0.5

    # 随机种子
    seed: int = 42

    # 运行时
    snapshot_dir: str = field(
        default_factory=lambda: os.path.expanduser('~/.lms/snapshots')
    )
    decoder_mode: str = 'text'
    consolidation_interval: int = 5
    auto_snapshot: bool = False
    auto_snapshot_interval: int = 50
    llm_api: dict = field(default_factory=lambda: {
        'base_url': os.environ.get('LMS_LLM_BASE_URL', 'https://api.openai.com/v1'),
        'api_key': os.environ.get('LMS_LLM_API_KEY', ''),
        'model': os.environ.get('LMS_LLM_MODEL', 'gpt-3.5-turbo'),
        'max_tokens': 1000,
        'temperature': 0.7,
        'timeout': 30,
        'max_retries': 3,
    })


def default_config() -> dict:
    """返回默认运行时配置字典。

    合并了 CoreConfig 的全部 FEP 参数与运行时配置，
    可直接传给 LivingMemoryLoop。键名与 CoreConfig 保持一致。

    返回:
        包含所有配置项的字典，可直接传给LivingMemoryLoop
    """
    config = RuntimeConfig()
    return {
        # 网络结构
        'num_nodes': config.num_nodes,
        'input_dim': config.input_dim,
        # 学习与推断
        'learning_rate': config.learning_rate,
        'num_infer_steps': config.num_infer_steps,
        'temperature': config.temperature,
        # 记忆管理
        'short_term_decay': config.short_term_decay,
        'long_term_decay': config.long_term_decay,
        'transfer_rate': config.transfer_rate,
        'replay_count': config.replay_count,
        'replay_weight': config.replay_weight,
        'consolidation_decay': config.consolidation_decay,
        'buffer_capacity': config.buffer_capacity,
        # 目的层
        'precision_min': config.precision_min,
        'precision_max': config.precision_max,
        'precision_lr': config.precision_lr,
        'coherence_threshold': config.coherence_threshold,
        'min_history_length': config.min_history_length,
        'meta_window': config.meta_window,
        'max_history': config.max_history,
        'habituation_rate': config.habituation_rate,
        'activation_threshold': config.activation_threshold,
        # FEP 学习规则
        'complexity_weight': config.complexity_weight,
        'orth_weight': config.orth_weight,
        # 随机种子
        'seed': config.seed,
        # 运行时
        'snapshot_dir': config.snapshot_dir,
        'decoder_mode': config.decoder_mode,
        'consolidation_interval': config.consolidation_interval,
        'auto_snapshot': config.auto_snapshot,
        'auto_snapshot_interval': config.auto_snapshot_interval,
        'llm_api': dict(config.llm_api),
    }


def load_config(path: str) -> dict:
    """从JSON文件加载配置。

    参数:
        path: JSON配置文件路径

    返回:
        配置字典（与default_config()合并）
    """
    import json
    config = default_config()
    with open(path, 'r', encoding='utf-8') as f:
        user_config = json.load(f)
    config.update(user_config)
    return config

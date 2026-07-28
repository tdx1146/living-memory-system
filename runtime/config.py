"""运行时配置

定义运行时的配置，包括LLM API配置、快照路径、运行模式等。
与core/config.py不同，这里关注的是运行时层面的配置（IO、外部服务）。
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RuntimeConfig:
    """运行时配置。

    属性:
        num_nodes: 吸引子网络节点数
        input_dim: 感官输入维度
        snapshot_dir: 快照保存目录
        decoder_mode: 解码器模式（'text'或'vector'）
        consolidation_interval: 记忆巩固间隔（每N轮巩固一次）
        auto_snapshot: 是否自动快照
        auto_snapshot_interval: 自动快照间隔（每N轮）
        llm_api: LLM API配置字典
    """
    num_nodes: int = 256
    input_dim: int = 64
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

    返回:
        包含所有配置项的字典，可直接传给LivingMemoryLoop
    """
    config = RuntimeConfig()
    return {
        'num_nodes': config.num_nodes,
        'input_dim': config.input_dim,
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

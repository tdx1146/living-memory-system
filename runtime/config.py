"""运行时配置
================

运行时层面的配置入口。以 ``CoreConfig`` 为单一事实源，
仅在此之上补充运行时专属字段（LLM API、快照目录等）。

设计原则:
  - FEP 参数的默认值由 ``CoreConfig`` 统一管理，本模块不重复定义
  - ``default_config()`` 委托 ``CoreConfig.to_loop_config()``，再叠加运行时字段
  - ``RuntimeConfig`` 保留向后兼容，但内部也委托 ``CoreConfig``
"""

import os
import json
from dataclasses import dataclass, field, fields as dc_fields

from core.config import CoreConfig
from core.paths import get_snapshot_dir


def _default_llm_api() -> dict:
    """从环境变量构建默认 LLM API 配置。"""
    return {
        'base_url': os.environ.get(
            'LMS_LLM_BASE_URL', 'https://api.openai.com/v1'),
        'api_key': os.environ.get('LMS_LLM_API_KEY', ''),
        'model': os.environ.get('LMS_LLM_MODEL', 'gpt-3.5-turbo'),
        'max_tokens': 1000,
        'temperature': 0.7,
        'timeout': 30,
        'max_retries': 3,
    }


@dataclass(init=False)
class RuntimeConfig:
    """运行时配置（向后兼容封装）。

    FEP 参数默认值由 ``CoreConfig`` 提供，本类仅补充运行时专属字段。
    旧代码可直接访问 ``RuntimeConfig().num_nodes`` 等属性，
    行为与旧版一致。

    属性:
        snapshot_dir: 快照保存目录
        decoder_mode: 解码器模式（'text' 或 'vector'）
        consolidation_interval: 记忆巩固间隔（每 N 轮巩固一次）
        auto_snapshot: 是否自动快照
        auto_snapshot_interval: 自动快照间隔（每 N 轮）
        llm_api: LLM API 配置字典
    """
    # 运行时专属字段（CoreConfig 不含这些）
    snapshot_dir: str = field(
        default_factory=lambda: str(get_snapshot_dir())
    )
    decoder_mode: str = 'text'
    consolidation_interval: int = 5
    auto_snapshot: bool = False
    auto_snapshot_interval: int = 50
    llm_api: dict = field(default_factory=_default_llm_api)

    def __init__(self, **kwargs):
        """构造运行时配置。

        接受所有 CoreConfig 字段 + 运行时字段作为关键字参数。
        FEP 参数委托 CoreConfig 管理默认值。
        """
        # 将 FEP 参数委托给 CoreConfig
        core_field_names = {f.name for f in dc_fields(CoreConfig)}
        core_kwargs = {k: v for k, v in kwargs.items() if k in core_field_names}
        rt_kwargs = {k: v for k, v in kwargs.items() if k not in core_field_names}

        self._core = CoreConfig(**core_kwargs)

        # 运行时字段
        self.snapshot_dir = rt_kwargs.get('snapshot_dir', str(get_snapshot_dir()))
        self.decoder_mode = rt_kwargs.get('decoder_mode', 'text')
        self.consolidation_interval = rt_kwargs.get('consolidation_interval', 5)
        self.auto_snapshot = rt_kwargs.get('auto_snapshot', False)
        self.auto_snapshot_interval = rt_kwargs.get('auto_snapshot_interval', 50)
        self.llm_api = rt_kwargs.get('llm_api', _default_llm_api())

    def __getattr__(self, name: str):
        """未在本类中找到的属性，委托给 CoreConfig 实例。"""
        if name.startswith('_'):
            raise AttributeError(name)
        core = self.__dict__.get('_core')
        if core is not None and hasattr(core, name):
            return getattr(core, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


def default_config() -> dict:
    """返回默认运行时配置字典。

    以 ``CoreConfig.to_loop_config()`` 为基础，叠加运行时专属字段。
    可直接传给 ``LivingMemoryLoop``。

    返回:
        包含所有配置项的字典。
    """
    config = CoreConfig().to_loop_config()
    rt = RuntimeConfig()
    # 叠加运行时专属字段（覆盖 to_loop_config 中的默认值）
    config['snapshot_dir'] = rt.snapshot_dir
    config['decoder_mode'] = rt.decoder_mode
    config['consolidation_interval'] = rt.consolidation_interval
    config['auto_snapshot'] = rt.auto_snapshot
    config['auto_snapshot_interval'] = rt.auto_snapshot_interval
    config['llm_api'] = dict(rt.llm_api)
    return config


def load_config(path: str) -> dict:
    """从 JSON 文件加载配置。

    参数:
        path: JSON 配置文件路径。

    返回:
        配置字典（与 ``default_config()`` 合并）。
    """
    config = default_config()
    with open(path, 'r', encoding='utf-8') as f:
        user_config = json.load(f)
    config.update(user_config)
    return config

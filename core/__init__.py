"""
活体记忆系统 - 核心层
======================

核心层是纯计算模块，无 IO 依赖。
包含感官层、海马体核心（吸引子网络、目的层、记忆管理）和配置。
"""

from core.config import CoreConfig
from core.types import Activation, PurposeState, SensoryInput

__all__ = [
    "CoreConfig",
    "SensoryInput",
    "Activation",
    "PurposeState",
]

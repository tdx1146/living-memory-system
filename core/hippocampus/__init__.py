"""
活体记忆系统 - 海马体核心
==========================

海马体核心包含三个关键模块:
  - AttractorNetwork: FEP 吸引子网络（推断 + 学习规则）
  - PurposeLayer: 目的层（precision 自主调节，三层结构）
  - MemoryManager: 多尺度记忆管理（短时/长时/consolidation）
"""

from core.hippocampus.attractor import AttractorNetwork, langevin
from core.hippocampus.memory import MemoryManager
from core.hippocampus.purpose import PurposeLayer

__all__ = [
    "AttractorNetwork",
    "langevin",
    "PurposeLayer",
    "MemoryManager",
]

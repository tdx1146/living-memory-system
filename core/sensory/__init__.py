"""
活体记忆系统 - 感官层
======================

感官层负责将文本转换为感官向量:
  - Tokenizer: 文本 -> token id 序列
  - Embedder: token id -> 冻结 embedding 向量
"""

from core.sensory.embedder import Embedder, SimpleEmbedder
from core.sensory.tokenizer import SimpleTokenizer, Tokenizer

__all__ = [
    "Tokenizer",
    "SimpleTokenizer",
    "Embedder",
    "SimpleEmbedder",
]

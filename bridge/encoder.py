"""编码器：LLM输出文本 -> 海马体感官信号

将对话文本（用户输入或LLM输出）编码为海马体能接收的感官向量。
核心逻辑：text -> tokenize -> embed -> 均值池化(如需) -> SensoryInput

遵循架构文档 5.4 节的接口定义。
依赖core的接口但不依赖具体实现（低耦合）。

注意：不同Embedder实现可能返回不同形状的向量：
  - 有的返回 [num_tokens, dim]（未池化），需要encoder做均值池化
  - 有的返回 [dim]（已池化），encoder直接使用
本编码器自动兼容两种情况。
"""

import time
import torch
from core.types import SensoryInput
from core.sensory.tokenizer import Tokenizer
from core.sensory.embedder import Embedder


class Encoder:
    """LLM输出 -> 海马体输入的编码器。

    将对话文本编码为海马体能接收的感官向量。
    流程：文本 -> tokenize -> embed -> (均值池化如需) -> SensoryInput

    自动兼容返回 [dim] 或 [num_tokens, dim] 的Embedder实现。
    该类是无状态的，可以安全地在多线程环境中使用。
    """

    def encode(self, text: str, tokenizer: Tokenizer,
               embedder: Embedder) -> SensoryInput:
        """将对话文本编码为感官信号。

        参数:
            text: 对话文本（用户输入或LLM输出）
            tokenizer: 分词器实例
            embedder: 嵌入器实例

        返回:
            SensoryInput对象，包含感官向量和元数据
        """
        # 1. 分词
        tokens = tokenizer.tokenize(text)

        # 2. 嵌入 + 池化
        if not tokens:
            # 空文本，返回零向量
            vector = torch.zeros(embedder.dim)
        else:
            # 嵌入：可能是 [dim]（已池化）或 [num_tokens, dim]（未池化）
            embeddings = embedder.embed(tokens)

            if embeddings.dim() == 1:
                # Embedder已做均值池化，返回 [dim]
                vector = embeddings
            elif embeddings.dim() == 2 and embeddings.shape[0] > 0:
                # Embedder返回 [num_tokens, dim]，需要均值池化
                vector = embeddings.mean(dim=0)
            else:
                vector = torch.zeros(embedder.dim)

        # 3. 构造SensoryInput
        return SensoryInput(
            vector=vector,
            metadata={
                'timestamp': time.time(),
                'source': 'encoder',
                'num_tokens': len(tokens),
                'text_length': len(text),
            }
        )

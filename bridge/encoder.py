"""编码器：LLM输出文本 -> 海马体感官信号

将对话文本（用户输入或LLM输出）编码为海马体能接收的感官向量。

自动适配两类 Embedder：
  - 文本路径：若 embedder 提供 embed_text(text)（如 PretrainedEmbedder），
    直接用原始文本编码语义，绕过系统 token id（预训练模型自带分词）。
  - token id 路径：否则走 text -> tokenize -> embed(tokens) -> 均值池化(如需)，
    兼容 SimpleEmbedder 等返回 [dim] 或 [num_tokens, dim] 的实现。

遵循架构文档 5.4 节的接口定义。
依赖core的接口但不依赖具体实现（低耦合）。
"""

import time
import torch
from core.types import SensoryInput
from core.sensory.tokenizer import Tokenizer
from core.sensory.embedder import Embedder


class Encoder:
    """LLM输出 -> 海马体输入的编码器。

    将对话文本编码为海马体能接收的感官向量。
    通过鸭子类型检测 embedder 是否具备 embed_text 方法来分流：
      - 有 embed_text（预训练语义 embedder）：走文本路径
      - 无 embed_text（如 SimpleEmbedder）：走 token id 路径，并自动兼容
        返回 [dim] 或 [num_tokens, dim] 的实现。

    该类是无状态的，可以安全地在多线程环境中使用。
    """

    def encode(self, text: str, tokenizer: Tokenizer,
               embedder: Embedder) -> SensoryInput:
        """将对话文本编码为感官信号。

        参数:
            text: 对话文本（用户输入或LLM输出）
            tokenizer: 分词器实例（文本路径下不参与编码）
            embedder: 嵌入器实例

        返回:
            SensoryInput对象，包含感官向量和元数据
        """
        if hasattr(embedder, "embed_text"):
            # ---- 文本路径：预训练语义 embedder（如 PretrainedEmbedder）----
            # 预训练模型按原始文本编码，自带分词，故绕过系统 token id。
            stripped = text.strip()
            if stripped:
                vector = embedder.embed_text(stripped)
            else:
                vector = torch.zeros(embedder.dim)
            # 文本路径下系统 token id 不参与编码，记为 0
            num_tokens = 0
        else:
            # ---- token id 路径：SimpleEmbedder 等（行为保持不变）----
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

            num_tokens = len(tokens)

        # 3. 构造SensoryInput
        return SensoryInput(
            vector=vector,
            metadata={
                'timestamp': time.time(),
                'source': 'encoder',
                'num_tokens': num_tokens,
                'text_length': len(text),
            }
        )

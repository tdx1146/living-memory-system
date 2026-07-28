"""
活体记忆系统 - 感官层：Token 嵌入（冻结）
==========================================

将 token id 序列转换为稠密向量（"感觉器官"）。
embedding 矩阵是冻结的（不参与 FEP 学习），扮演感官受体的角色。

提供抽象基类 Embedder 定义接口，以及默认实现 SimpleEmbedder。

SimpleEmbedder 策略：
  - 随机初始化的冻结 embedding 矩阵（固定种子，可复现）
  - 对 token 序列做平均池化得到单一感官向量
  - 未来可替换为预训练 SLM 的 embedding 层

参考：架构文档 第五节《接口定义》5.2 感官层接口
"""

from abc import ABC, abstractmethod

import torch


class Embedder(ABC):
    """嵌入器抽象基类。

    定义 token 到向量的统一接口。
    embedding 是冻结的——它不参与 FEP 学习，仅作为感官受体。
    """

    @abstractmethod
    def embed(self, tokens: list[int]) -> torch.Tensor:
        """token id 列表 -> 感官向量。

        参数:
            tokens: token id 列表。

        返回:
            形状 [dim] 的感官向量。
        """
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """embedding 维度。"""
        ...


class SimpleEmbedder(Embedder):
    """简单嵌入器：随机初始化的冻结 embedding。

    使用固定随机种子初始化 embedding 矩阵，然后冻结。
    对输入 token 序列做平均池化，输出单一感官向量。

    适用于测试与原型验证。未来可替换为预训练 SLM 的 embedding：
      - 预训练 embedding 能提供语义先验
      - 但核心 FEP 学习规则不变（仍是涌现的）

    属性:
        embedding: 冻结的 embedding 矩阵，形状 [vocab_size, dim]。
    """

    def __init__(self, dim: int = 64, vocab_size: int = 10000,
                 seed: int = 42) -> None:
        """初始化冻结 embedding。

        参数:
            dim: embedding 维度（= 感官向量维度 = input_dim）。
            vocab_size: 词表大小上限。
            seed: 随机种子，保证可复现。
        """
        self._dim = dim
        generator = torch.Generator()
        generator.manual_seed(seed)
        # 小方差初始化，使初始感官信号处于温和范围
        self.embedding: torch.Tensor = (
            torch.randn(vocab_size, dim, generator=generator) * 0.1
        )
        # 冻结：不参与梯度计算
        self.embedding.requires_grad = False

    def embed(self, tokens: list[int]) -> torch.Tensor:
        """token id 列表 -> 感官向量（平均池化）。

        参数:
            tokens: token id 列表。

        返回:
            形状 [dim] 的感官向量。空输入返回零向量。
        """
        if len(tokens) == 0:
            return torch.zeros(self._dim)
        # 限制 id 在词表范围内
        clamped = [min(max(t, 0), self.embedding.shape[0] - 1) for t in tokens]
        idx = torch.tensor(clamped, dtype=torch.long)
        vecs = self.embedding[idx]  # [num_tokens, dim]
        return vecs.mean(dim=0)  # [dim] 平均池化

    @property
    def dim(self) -> int:
        """embedding 维度。"""
        return self._dim

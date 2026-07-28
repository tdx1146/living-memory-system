"""
活体记忆系统 - 感官层：Token 嵌入（冻结）
==========================================

将 token id 序列转换为稠密向量（"感觉器官"）。
embedding 矩阵是冻结的（不参与 FEP 学习），扮演感官受体的角色。

提供抽象基类 Embedder 定义接口，以及两种实现：
  - SimpleEmbedder：随机初始化的冻结 embedding（无语义先验，用于测试/原型）
  - PretrainedEmbedder：基于 sentence-transformers 的预训练语义 embedding
    （提供跨语言语义先验，扮演真正"训练好的视网膜"）

SimpleEmbedder 策略：
  - 随机初始化的冻结 embedding 矩阵（固定种子，可复现）
  - 对 token 序列做平均池化得到单一感官向量

PretrainedEmbedder 策略：
  - 加载预训练 sentence-transformers 模型（默认 paraphrase-multilingual-
    MiniLM-L12-v2，支持中文，384 维，约 90MB），加载后冻结
  - 预训练模型按"原始文本"编码（而非系统内部的 token id），故提供
    embed_text(text) 方法；Encoder 会检测该方法并走文本路径
  - 预训练输出 384 维，通过一个冻结的随机投影矩阵（Johnson-Lindenstrauss
    缩放，固定种子）降维到系统配置的 input_dim，保持低耦合

参考：架构文档 第五节《接口定义》5.2 感官层接口
"""

from abc import ABC, abstractmethod
import math

import torch

# 可选依赖：sentence-transformers 未安装时优雅降级——
# 模块（及 SimpleEmbedder）仍可正常导入，仅在实例化 PretrainedEmbedder 时
# 抛出带安装提示的 ImportError，避免牵连破坏其它依赖本模块的组件。
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = None
except ImportError as _exc:  # pragma: no cover - 取决于运行环境
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = _exc
    SentenceTransformer = None  # type: ignore[assignment,misc]


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


class PretrainedEmbedder(Embedder):
    """预训练嵌入器：基于 sentence-transformers 的语义"视网膜"。

    与 SimpleEmbedder（随机初始化、无语义先验）不同，PretrainedEmbedder
    加载一个真实的预训练句向量模型，为记忆系统提供跨语言语义先验——
    语义相近的文本会产生相近的感官向量，使海马体能在"语义空间"而非
    "符号空间"中形成吸引子。

    设计要点：
      1. 冻结：模型在 __init__ 加载后立即 requires_grad=False 并 eval()，
         绝不参与 FEP 学习——它只扮演感官受体。
      2. 文本接口：预训练模型按原始文本编码，而 Embedder 抽象接口的
         embed(tokens) 接收的是系统内部 token id。为兼顾两者：
           - 提供 embed_text(text) -> Tensor，由 Encoder 检测并优先使用；
           - embed(tokens) 不适用（抛 NotImplementedError），因为没有原始
             文本时无法产生有意义的语义向量。
      3. 维度适配：预训练输出 384 维，系统 input_dim 通常更小（如 64）。
         用一个冻结的随机投影矩阵（Johnson-Lindenstrauss 缩放 + 固定种子）
         从 raw_dim -> input_dim，既降维又保持低耦合与可复现。
      4. 句向量在投影前做 L2 归一化，使投影后输出幅度稳定在
         ~sqrt(input_dim/raw_dim) 量级（input_dim=64、raw_dim=384 时约 0.41），
         与 SimpleEmbedder 的温和感官信号量级相当。

    依赖：
      需要安装 sentence-transformers（pip install sentence-transformers）。
      未安装时，本模块仍可正常导入（保证 SimpleEmbedder 可用），仅在实例化
      PretrainedEmbedder 时抛出带安装提示的 ImportError。

    属性:
        dim: 投影后的感官向量维度（= input_dim）。
        raw_dim: 预训练模型原始输出维度（如 384）。
    """

    # 默认模型：多语言 MiniLM，支持中文，384 维，约 90MB
    DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, dim: int = 64,
                 model_name: str = DEFAULT_MODEL_NAME,
                 seed: int = 42,
                 device: str = "cpu") -> None:
        """初始化并冻结预训练句向量模型 + 随机投影。

        参数:
            dim: 投影后的感官向量维度（= 系统 input_dim）。
            model_name: sentence-transformers 模型名，默认多语言 MiniLM。
            seed: 随机投影矩阵的种子，保证可复现。
            device: 模型推理设备（"cpu" / "cuda" 等）。

        异常:
            ImportError: sentence-transformers 未安装时抛出，附带安装提示。
        """
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "PretrainedEmbedder 依赖 sentence-transformers，但该库未安装。"
                " 请运行 `pip install sentence-transformers` 安装后重试。"
            ) from _SENTENCE_TRANSFORMERS_IMPORT_ERROR

        self._dim = dim
        self._model_name = model_name
        self._device = device

        # 1. 加载预训练模型（首次会下载约 90MB 权重）
        self._model = SentenceTransformer(model_name, device=device)
        # 2. 冻结：感官受体不参与学习
        self._model.eval()
        for _param in self._model.parameters():
            _param.requires_grad = False

        # 3. 原始输出维度（如 384），从模型动态读取以增强鲁棒性
        self._raw_dim = int(self._model.get_sentence_embedding_dimension())

        # 4. 冻结的随机投影矩阵：raw_dim -> dim
        #    Johnson-Lindenstrauss 缩放（/sqrt(raw_dim)）使单位输入经投影后
        #    幅度保持稳定，无需额外学习即可降维。
        generator = torch.Generator()
        generator.manual_seed(seed)
        self._projection: torch.Tensor = (
            torch.randn(self._raw_dim, dim, generator=generator)
            / math.sqrt(self._raw_dim)
        )
        self._projection.requires_grad = False

    def embed_text(self, text: str) -> torch.Tensor:
        """原始文本 -> 感官向量（预训练语义 + 随机投影）。

        这是 PretrainedEmbedder 的主路径：用预训练模型把文本编码为语义向量，
        再经冻结的随机投影降到 input_dim。Encoder 会检测本方法并优先调用。

        参数:
            text: 原始文本字符串（可中英混合）。

        返回:
            形状 [dim] 的感官向量。空文本返回零向量。
        """
        if not text or not text.strip():
            return torch.zeros(self._dim)

        with torch.no_grad():
            raw = self._model.encode(
                text,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            # sentence-transformers 对单条输入可能返回 [raw_dim] 或 [1, raw_dim]
            if raw.dim() == 2:
                raw = raw.squeeze(0)
            # 统一到 CPU float，避免与投影矩阵的设备/精度不一致
            raw = raw.detach().cpu().float()
            # 随机投影：[raw_dim] @ [raw_dim, dim] -> [dim]
            vector = raw @ self._projection

        return vector.detach()

    def embed(self, tokens: list[int]) -> torch.Tensor:
        """token id 路径（本实现不适用）。

        PretrainedEmbedder 基于原始文本编码语义，无法从系统内部的 token id
        还原出有意义的语义向量。请改用 embed_text(text)，或通过 Encoder
        （会自动检测 embed_text 并走文本路径）调用。

        保留本方法仅为满足 Embedder 抽象接口；调用必抛 NotImplementedError。

        异常:
            NotImplementedError: 始终抛出，提示使用 embed_text。
        """
        raise NotImplementedError(
            "PretrainedEmbedder 基于原始文本编码，不支持 token id 路径。"
            " 请使用 embed_text(text)，或通过 Encoder 调用（自动走文本路径）。"
        )

    @property
    def dim(self) -> int:
        """投影后的感官向量维度（= input_dim）。"""
        return self._dim

    @property
    def raw_dim(self) -> int:
        """预训练模型原始输出维度（如 384）。"""
        return self._raw_dim

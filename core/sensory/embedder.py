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
import os
import time
from typing import Optional

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

    资源管理（E-P2-4 改进）：
      - 懒加载：构造函数只保存模型名/路径，首次调用 embed/embed_text 时
        才实际加载模型权重，避免启动期阻塞与不必要的内存占用。
      - 模型缓存：类级别 ``_model_cache`` 按 "source:model_name:device"
        缓存已加载的 SentenceTransformer 实例，同一模型多实例共享，避免
        重复下载/加载。通过 ``clear_cache()`` 类方法统一释放。
      - 资源释放：``unload()`` 释放当前实例引用，``__del__`` 在对象销毁时
        自动释放，并支持上下文管理器协议（``with PretrainedEmbedder() as e``）。

    设计要点：
      1. 冻结：模型在 load() 后立即 requires_grad=False 并 eval()，
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
      5. 模型路径配置：支持 HuggingFace 模型名、本地目录路径、ModelScope
         模型名（通过环境变量 ``LMS_EMBEDDER_SOURCE`` 选择来源）。

    依赖：
      需要安装 sentence-transformers（pip install sentence-transformers）。
      未安装时，本模块仍可正常导入（保证 SimpleEmbedder 可用），仅在实例化
      PretrainedEmbedder 时抛出带安装提示的 ImportError。

    环境变量:
        LMS_PRETRAINED_MODEL: 覆盖默认模型名/路径（构造时未显式传入时生效）。
        LMS_EMBEDDER_SOURCE: 模型来源，"huggingface"（默认）或 "modelscope"。
        LMS_MODEL_LOAD_TIMEOUT: 模型加载超时（秒），默认 120。

    属性:
        dim: 投影后的感官向量维度（= input_dim）。
        raw_dim: 预训练模型原始输出维度（如 384）。
        is_loaded: 模型是否已加载到内存。
    """

    # 默认模型：多语言 MiniLM，支持中文，384 维，约 90MB
    DEFAULT_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

    # 类级别模型缓存：避免同一模型被重复加载。
    # 键为 "source:model_name:device"，值为已加载的 SentenceTransformer 实例。
    _model_cache: dict = {}

    # 已知模型维度表：在不加载模型的情况下返回预期维度。
    # 键为 HuggingFace 模型名，值为 sentence embedding 维度。
    _KNOWN_DIMS: dict = {
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "paraphrase-mpnet-base-v2": 768,
        "distiluse-base-multilingual-cased-v1": 512,
        "distiluse-base-multilingual-cased-v2": 512,
        "bge-small-zh-v1.5": 512,
        "bge-base-zh-v1.5": 768,
        "bge-large-zh-v1.5": 1024,
        "bge-small-en-v1.5": 384,
        "bge-base-en-v1.5": 768,
        "bge-large-en-v1.5": 1024,
        "text2vec-base-chinese": 768,
        "text2vec-large-chinese": 1024,
        "m3e-small": 384,
        "m3e-base": 768,
        "m3e-large": 1024,
    }

    def __init__(self, dim: int = 64,
                 model_name: Optional[str] = None,
                 seed: int = 42,
                 device: str = "cpu") -> None:
        """初始化预训练嵌入器（懒加载：不立即加载模型）。

        构造函数只保存模型名/路径与配置参数，模型权重在首次调用
        embed/embed_text/embed_text_raw 时才实际加载（懒加载）。
        若 sentence-transformers 未安装，仍在构造时抛出 ImportError
        （保持与既有调用方 ``except ImportError`` 的兼容）。

        参数:
            dim: 投影后的感官向量维度（= 系统 input_dim）。
            model_name: sentence-transformers 模型名或本地目录路径。
                为 None 时依次读取环境变量 ``LMS_PRETRAINED_MODEL``、
                默认值 ``DEFAULT_MODEL_NAME``。
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

        # 模型名称：显式参数 > 环境变量 LMS_PRETRAINED_MODEL > 默认值
        if model_name is None:
            model_name = os.environ.get(
                "LMS_PRETRAINED_MODEL", self.DEFAULT_MODEL_NAME)

        # 模型来源：环境变量 LMS_EMBEDDER_SOURCE（默认 huggingface）
        source = os.environ.get(
            "LMS_EMBEDDER_SOURCE", "huggingface").lower().strip()

        self._dim = dim
        self._model_name = model_name
        self._source = source
        self._device = device
        self._seed = seed

        # 缓存键：source + model_name + device 唯一标识一个已加载模型
        self._cache_key = f"{source}:{model_name}:{device}"

        # 懒加载：模型与投影矩阵在首次使用时才创建
        self._model = None
        self._raw_dim: Optional[int] = None
        self._projection: Optional[torch.Tensor] = None

    # ================================================================== #
    #  资源管理
    # ================================================================== #

    @property
    def is_loaded(self) -> bool:
        """模型是否已加载到内存。"""
        return self._model is not None

    def load(self) -> None:
        """加载预训练模型（懒加载触发点）。

        若模型已在缓存中（被本实例或同配置的其它实例加载过），则直接复用，
        不重复下载/加载。加载完成后冻结模型参数并构建随机投影矩阵。

        本方法是幂等的：对已加载的实例重复调用不会有副作用。

        异常:
            RuntimeError: 加载失败（重试耗尽）时抛出，附带模型名、来源与
                安装/网络提示。
        """
        if self._model is not None:
            return  # 已加载，幂等返回

        # 1. 检查类级别缓存
        if self._cache_key in PretrainedEmbedder._model_cache:
            self._model = PretrainedEmbedder._model_cache[self._cache_key]
        else:
            # 2. 实际加载（含重试与超时控制）
            self._model = self._load_with_retry()
            PretrainedEmbedder._model_cache[self._cache_key] = self._model

        # 3. 冻结：感官受体不参与学习（幂等操作，缓存命中时也无害）
        self._model.eval()
        for _param in self._model.parameters():
            _param.requires_grad = False

        # 4. 原始输出维度（如 384），从模型动态读取以增强鲁棒性
        self._raw_dim = int(self._model.get_sentence_embedding_dimension())

        # 5. 冻结的随机投影矩阵：raw_dim -> dim
        #    Johnson-Lindenstrauss 缩放（/sqrt(raw_dim)）使单位输入经投影后
        #    幅度保持稳定，无需额外学习即可降维。
        generator = torch.Generator()
        generator.manual_seed(self._seed)
        self._projection = (
            torch.randn(self._raw_dim, self._dim, generator=generator)
            / math.sqrt(self._raw_dim)
        )
        self._projection.requires_grad = False

    def _load_with_retry(self):
        """带重试与超时控制的模型加载。

        网络异常时最多重试 3 次，采用指数退避（1s、2s、4s）。
        超时阈值由环境变量 ``LMS_MODEL_LOAD_TIMEOUT`` 控制（默认 120 秒）。

        返回:
            已加载的 SentenceTransformer 实例。

        异常:
            RuntimeError: 全部重试失败后抛出，包含诊断信息。
        """
        timeout = int(os.environ.get("LMS_MODEL_LOAD_TIMEOUT", "120"))
        max_retries = 3
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                # 设置 HuggingFace hub 下载超时（best effort）
                os.environ.setdefault(
                    "HF_HUB_DOWNLOAD_TIMEOUT", str(timeout))
                resolved = self._resolve_source(
                    self._model_name, self._source)
                model = SentenceTransformer(resolved, device=self._device)
                return model
            except Exception as exc:  # noqa: BLE001 - 重试需捕获所有异常
                last_exc = exc
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s
                    time.sleep(wait)
        # 全部重试失败
        raise RuntimeError(
            f"加载预训练模型失败（已重试 {max_retries} 次）："
            f"model_name={self._model_name!r}, source={self._source!r}, "
            f"device={self._device!r}。"
            f"\n原始错误: {last_exc}"
            f"\n提示：若为网络问题，请检查网络连接或设置 "
            f"LMS_EMBEDDER_SOURCE=modelscope 切换为国内镜像源；"
            f"若为依赖缺失，请确认 sentence-transformers 已安装"
            f"（pip install sentence-transformers）。"
        ) from last_exc

    def _resolve_source(self, model_name: str, source: str) -> str:
        """根据来源解析模型为 SentenceTransformer 可接受的标识。

        - huggingface：直接返回模型名，由 sentence-transformers 从 HF Hub 下载。
        - modelscope：若为本地目录路径则直接返回；否则尝试用 modelscope 的
          snapshot_download 下载到本地缓存后返回本地路径。

        参数:
            model_name: 模型名或本地路径。
            source: "huggingface" 或 "modelscope"。

        返回:
            SentenceTransformer 可接受的模型标识（模型名或本地路径）。

        异常:
            ImportError: source 为 modelscope 但未安装 modelscope 库时抛出。
        """
        if source == "modelscope":
            # 本地目录路径：直接加载
            if os.path.isdir(model_name):
                return model_name
            # 需通过 modelscope 下载
            try:
                from modelscope import snapshot_download
            except ImportError as exc:
                raise ImportError(
                    "选择 ModelScope 来源（LMS_EMBEDDER_SOURCE=modelscope）"
                    "但未安装 modelscope 库。"
                    " 请运行 `pip install modelscope` 安装后重试，"
                    " 或设置 LMS_EMBEDDER_SOURCE=huggingface 切换来源。"
                ) from exc
            return snapshot_download(model_name)
        # huggingface（默认）
        return model_name

    def unload(self) -> None:
        """释放当前实例持有的模型引用与投影矩阵。

        仅清除本实例的引用，不影响类级别缓存中的模型（其它实例可能仍在
        使用）。若需彻底释放所有缓存的模型，请调用 ``clear_cache()``。
        卸载后再次调用 embed_text 等方法会重新触发懒加载。

        本方法是幂等的。
        """
        self._model = None
        self._projection = None
        self._raw_dim = None

    @classmethod
    def clear_cache(cls) -> None:
        """释放所有缓存的模型（类级别）。

        清空 ``_model_cache``，使后续所有 PretrainedEmbedder 实例在首次
        使用时重新加载模型。已加载的实例不受影响（它们持有自己的引用），
        但调用 ``unload()`` 后再使用时会重新加载。

        典型用法：在批量推理结束、需要回收显存/内存时调用。
        """
        cls._model_cache.clear()

    @classmethod
    def expected_dim(cls,
                    model_name: Optional[str] = None) -> Optional[int]:
        """在不加载模型的情况下返回预期 sentence embedding 维度。

        查询内置的 ``_KNOWN_DIMS`` 维度表。对本地路径，取其末尾目录名
        再匹配（兼容 ModelScope 缓存路径形式）。

        参数:
            model_name: 模型名或本地路径。为 None 时使用默认模型。

        返回:
            预期维度（如 384）；未知模型返回 None。
        """
        if model_name is None:
            model_name = cls.DEFAULT_MODEL_NAME
        # 精确匹配
        if model_name in cls._KNOWN_DIMS:
            return cls._KNOWN_DIMS[model_name]
        # 末尾目录名匹配（兼容本地缓存路径）
        basename = os.path.basename(model_name.rstrip("/\\"))
        if basename in cls._KNOWN_DIMS:
            return cls._KNOWN_DIMS[basename]
        return None

    def __enter__(self) -> "PretrainedEmbedder":
        """上下文管理器入口：加载模型并返回自身。"""
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口：释放模型资源。不吞没异常。"""
        self.unload()

    def __del__(self) -> None:
        """对象销毁时安全释放模型引用。

        捕获所有异常以避免在解释器关闭期（部分模块已卸载）抛错。
        """
        try:
            self.unload()
        except Exception:  # noqa: BLE001 - __del__ 必须吞没所有异常
            pass

    # ================================================================== #
    #  编码
    # ================================================================== #

    def embed_text(self, text: str) -> torch.Tensor:
        """原始文本 -> 感官向量（预训练语义 + 随机投影）。

        这是 PretrainedEmbedder 的主路径：用预训练模型把文本编码为语义向量，
        再经冻结的随机投影降到 input_dim。Encoder 会检测本方法并优先调用。
        首次调用会触发懒加载（load()）。

        参数:
            text: 原始文本字符串（可中英混合）。

        返回:
            形状 [dim] 的感官向量。空文本返回零向量。
        """
        if not text or not text.strip():
            return torch.zeros(self._dim)

        # 懒加载触发
        self.load()

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

    def embed_text_raw(self, text: str) -> torch.Tensor:
        """原始文本 -> 预训练模型原始语义向量（投影前，384 维）。

        与 embed_text 不同，本方法返回预训练模型的原始输出（未经随机投影
        降维），供 episodic buffer 做高精度语义检索——保留完整 384 维语义
        信息，避免投影造成的信息损失，从而提升检索精度。

        embed_text（投影后 64 维）仍用于吸引子网络输入，两者职责分离：
          - embed_text     -> 吸引子网络感官输入（低维、稳定幅度）
          - embed_text_raw -> episodic buffer 检索（高维、保留语义细节）

        向量在模型编码时已做 L2 归一化（normalize_embeddings=True），
        故余弦相似度等价于点积。首次调用会触发懒加载（load()）。

        参数:
            text: 原始文本字符串（可中英混合）。

        返回:
            形状 [raw_dim] 的 L2 归一化语义向量（如 384 维）。
            空文本返回 raw_dim 维零向量。
        """
        if not text or not text.strip():
            # 空文本：尽量不触发加载——优先用已知维度或已加载维度
            rd = self._raw_dim if self._raw_dim is not None else \
                self.expected_dim(self._model_name)
            if rd is None:
                self.load()
                rd = self._raw_dim
            return torch.zeros(rd)

        # 懒加载触发
        self.load()

        with torch.no_grad():
            raw = self._model.encode(
                text,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
            # sentence-transformers 对单条输入可能返回 [raw_dim] 或 [1, raw_dim]
            if raw.dim() == 2:
                raw = raw.squeeze(0)
            # 统一到 CPU float
            raw = raw.detach().cpu().float()

        return raw.detach()

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
        """预训练模型原始输出维度（如 384）。

        模型未加载时返回已知维度表中的预期维度；未知模型返回 0
        （调用 load() 后可获得准确值）。
        """
        if self._raw_dim is not None:
            return self._raw_dim
        expected = self.expected_dim(self._model_name)
        return expected if expected is not None else 0

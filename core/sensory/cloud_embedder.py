"""
活体记忆系统 - 感官层：云端嵌入适配器（CloudEmbedder）
======================================================

替代 PretrainedEmbedder（本地加载 sentence-transformers 模型），
改为调用远程 embedding API（如 bge-m3, 1024 维）。

设计要点：
  1. 实现 Embedder 抽象基类，提供 embed_text(text) 方法，
     使 Encoder 自动走文本路径（与 PretrainedEmbedder 行为一致）。
  2. 通过 HTTP POST 调用远端 embed 服务，支持 batch 编码。
  3. embed(tokens) 不适用（抛 NotImplementedError），同 PretrainedEmbedder。
  4. 返回 torch.Tensor（不是 numpy），兼容三妹的接口。
  5. 内置简单结果缓存（LRU），减少重复调用。
"""

import math
import time
from collections import OrderedDict
from typing import Optional

import requests
import torch

from core.sensory.embedder import Embedder


class CloudEmbedder(Embedder):
    """云端嵌入适配器：通过 HTTP API 调用远程 embedding 服务。

    与 PretrainedEmbedder 接口兼容（提供 embed_text），
    但不在本地加载模型，改为调用远端 API。

    属性:
        dim: embedding 维度（如 1024）。
        api_url: embed API 完整 URL。
        model: 远端模型名（用于请求体，可选）。
        cache_size: LRU 缓存大小（默认 512）。
    """

    def __init__(self,
                 api_url: str = "https://embed.example.com/v1/embeddings",
                 dim: int = 64,
                 remote_dim: int = 1024,
                 model: Optional[str] = None,
                 cache_size: int = 512,
                 timeout: float = 30.0,
                 retries: int = 3,
                 fallback_url: Optional[str] = None) -> None:
        """初始化云端嵌入器。

        参数:
            api_url: Embed API 完整 URL。
            dim: 输出向量维度（投影后，如 64）。
            remote_dim: 远程embedding维度（如 bge-m3 为 1024）。
            model: 请求体中传递的 model 名称。为 None 时不传 model 字段。
            cache_size: 文本嵌入结果 LRU 缓存大小，0 表示不缓存。
            timeout: HTTP 请求超时秒数。
            retries: 失败重试次数。
            fallback_url: 备用 Embed API URL（主 URL 重试耗尽后自动切换，
                仅本机部署通过环境变量注入；默认 None=不启用）。
        """
        self._dim = dim
        self._remote_dim = remote_dim
        self._api_url = api_url
        self._fallback_url = fallback_url
        self._model = model
        self._timeout = timeout
        self._retries = retries

        # 随机投影矩阵：remote_dim -> dim
        torch.manual_seed(42)  # 固定种子，保证投影一致性
        self._projection = torch.randn(remote_dim, dim) / math.sqrt(remote_dim)

        # LRU 缓存：计算命中显著减少远程调用
        self._cache_size = cache_size
        if cache_size > 0:
            self._cache: OrderedDict = OrderedDict()
        else:
            self._cache = None

    def _call_api(self, text: str) -> list[float]:
        """调用远程 embed API。

        参数:
            text: 编码文本。

        返回:
            float 列表形式的 embedding 向量。

        异常:
            RuntimeError: API 调用失败时抛出。
        """
        payload = {
            "input": text,
        }
        if self._model is not None:
            payload["model"] = self._model

        last_exc = None
        for attempt in range(self._retries):
            try:
                resp = requests.post(
                    self._api_url,
                    json=payload,
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                # OpenAI 兼容格式：data[0].embedding
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0]["embedding"]
                # 兜底：直接取 embedding 字段
                if "embedding" in data:
                    return data["embedding"]
                raise RuntimeError(
                    f"API 返回格式异常：{str(data)[:200]}"
                )
            except (requests.RequestException, ValueError, KeyError,
                    RuntimeError) as exc:
                last_exc = exc
                if attempt < self._retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue

        raise RuntimeError(
            f"Embed API 调用失败（已重试{self._retries}次）：{last_exc}"
        )

    def _call_api_with_fallback(self, text: str) -> list[float]:
        """主 URL 优先，失败自动切 fallback（LAN 直连 → 隧道兜底）。"""
        try:
            return self._call_api(text)
        except RuntimeError:
            if self._fallback_url and self._fallback_url != self._api_url:
                orig = self._api_url
                self._api_url, self._fallback_url = self._fallback_url, None
                try:
                    return self._call_api(text)
                finally:
                    self._api_url = orig
            raise

    def embed_text_raw(self, text: str) -> torch.Tensor:
        """原始文本 -> 远程 API 的原始向量（L2归一化后，不投影）。

        与 PretrainedEmbedder.embed_text_raw 对应，返回 remote_dim 维向量，
        供 recall_episodic 做高精度原始维度检索。缓存与 embed_text 共享。

        参数:
            text: 原始文本字符串。

        返回:
            形状 [remote_dim] 的归一化向量。空文本返回零向量。
        """
        if not text or not text.strip():
            return torch.zeros(self._remote_dim, dtype=torch.float32)

        # 先查缓存（embed_text 已缓存投影向量，这里需要重新获取原始向量）
        # 由于缓存存的是投影后的向量，raw 需要单独调用 API
        raw = self._call_api_with_fallback(text.strip())
        raw_vector = torch.tensor(raw, dtype=torch.float32)
        raw_norm = raw_vector.norm()
        if raw_norm > 0:
            raw_vector = raw_vector / raw_norm
        return raw_vector

    def embed_text(self, text: str) -> torch.Tensor:
        """原始文本 -> 感官向量（通过远程 API 编码 + 随机投影）。

        Encoder 会检测本方法并优先调用，与 PretrainedEmbedder 行为一致。

        参数:
            text: 原始文本字符串。

        返回:
            形状 [dim] 的感官向量（torch.Tensor, float32）。空文本返回零向量。
        """
        if not text or not text.strip():
            return torch.zeros(self._dim, dtype=torch.float32)

        # LRU 缓存查找
        if self._cache is not None:
            if text in self._cache:
                self._cache.move_to_end(text)  # 最近使用
                return self._cache[text]

        # 调用 API
        raw = self._call_api_with_fallback(text.strip())
        raw_vector = torch.tensor(raw, dtype=torch.float32)

        # L2 归一化原始向量（消除不同模型幅度差异）
        raw_norm = raw_vector.norm()
        if raw_norm > 0:
            raw_vector = raw_vector / raw_norm

        # 随机投影：[remote_dim] @ [remote_dim, dim] -> [dim]
        # Johnson-Lindenstrauss 缩放使投影后幅度与 MiniLM 相当（~0.4）
        vector = raw_vector @ self._projection

        # 填充 LRU 缓存
        if self._cache is not None:
            self._cache[text] = vector
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        return vector

    def embed(self, tokens: list[int]) -> torch.Tensor:
        """token id 路径（本实现不适用）。

        与 PretrainedEmbedder 一致，基于原始文本编码，不支持 token id。

        异常:
            NotImplementedError: 始终抛出，提示使用 embed_text。
        """
        raise NotImplementedError(
            "CloudEmbedder 基于 API 编码原始文本，不支持 token id 路径。"
            " 请使用 embed_text(text)，或通过 Encoder 调用（自动走文本路径）。"
        )

    @property
    def dim(self) -> int:
        """embedding 维度。"""
        return self._dim

    @property
    def raw_dim(self) -> int:
        """远程 API 原始输出维度（投影前）。"""
        return self._remote_dim

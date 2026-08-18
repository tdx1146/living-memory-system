"""远程嵌入器：调用外部embed API

调用远程embed服务（如Ollama bge-m3），将文本转换为向量。
支持任意维度的embed服务，内部投影到目标维度。
"""

import logging
import urllib.request
import urllib.error
import json
import torch
import numpy as np

logger = logging.getLogger(__name__)


class RemoteEmbedder:
    """远程嵌入器：调用外部embed API（如Ollama bge-m3）。
    
    通过HTTP调用远程embed服务，支持：
    - Ollama embed服务（如bge-m3）
    - OpenAI兼容embed API
    
    如果远程服务维度与目标dim不同，使用随机投影降维/升维。
    
    属性:
        dim: 输出向量维度（投影后）
        api_url: embed API地址
        model: 模型名
        remote_dim: 远程服务输出维度（首次调用后确定）
    """
    
    def __init__(self, dim: int = 64, api_url: str = None, model: str = "bge-m3", seed: int = 42):
        """初始化远程嵌入器。
        
        参数:
            dim: 目标输出维度（投影后）
            api_url: embed API地址，如 <LAN_IP>:11435/v1/embeddings
            model: 模型名
            seed: 随机投影矩阵的种子
        """
        self._dim = dim
        self._api_url = api_url or "<LAN_IP>:11435/v1/embeddings"
        self._model = model
        self._seed = seed
        self._remote_dim = None
        self._projection = None
        self._initialized = False
        
    @property
    def dim(self) -> int:
        """输出维度（投影后）"""
        return self._dim
    
    def _init_projection(self, remote_dim: int):
        """初始化投影矩阵"""
        if self._initialized and self._remote_dim == remote_dim:
            return
        
        self._remote_dim = remote_dim
        np.random.seed(self._seed)
        
        if remote_dim != self._dim:
            # 随机投影矩阵（Xavier初始化）
            scale = np.sqrt(2.0 / (remote_dim + self._dim))
            self._projection = torch.tensor(
                np.random.randn(remote_dim, self._dim) * scale,
                dtype=torch.float32
            )
            logger.info(f"RemoteEmbedder: 投影矩阵已初始化 {remote_dim} -> {self._dim}")
        else:
            self._projection = None
            logger.info(f"RemoteEmbedder: 无需投影，维度匹配 {self._dim}")
        
        self._initialized = True
    
    def embed_text(self, text: str) -> torch.Tensor:
        """将文本转换为向量
        
        参数:
            text: 输入文本
            
        返回:
            形状 [dim] 的向量
        """
        # 调用远程API
        try:
            data = json.dumps({
                "input": text,
                "model": self._model
            }).encode("utf-8")
            
            req = urllib.request.Request(
                self._api_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            
            # 解析响应
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["embedding"]
            elif "embedding" in result:
                embedding = result["embedding"]
            else:
                raise ValueError(f"无效的embed响应: {result}")
            
            # 转换为tensor
            vector = torch.tensor(embedding, dtype=torch.float32)
            
            # 首次调用时初始化投影
            if not self._initialized:
                self._init_projection(len(embedding))
            
            # 投影
            if self._projection is not None:
                vector = torch.matmul(vector, self._projection)
            
            # L2归一化
            norm = torch.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            return vector
            
        except urllib.error.URLError as e:
            logger.warning(f"远程embed服务不可用: {e}，使用随机向量")
            # 降级：返回随机向量
            np.random.seed(hash(text) % (2**31))
            return torch.tensor(
                np.random.randn(self._dim),
                dtype=torch.float32
            )
        except Exception as e:
            logger.error(f"embed失败: {e}")
            raise
    
    def embed_text_raw(self, text: str) -> torch.Tensor:
        """返回原始向量（投影前），用于episodic buffer高精度检索
        
        参数:
            text: 输入文本
            
        返回:
            形状 [remote_dim] 的原始向量
        """
        try:
            data = json.dumps({
                "input": text,
                "model": self._model
            }).encode("utf-8")
            
            req = urllib.request.Request(
                self._api_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["embedding"]
            elif "embedding" in result:
                embedding = result["embedding"]
            else:
                raise ValueError(f"无效的embed响应: {result}")
            
            # L2归一化
            vector = torch.tensor(embedding, dtype=torch.float32)
            norm = torch.norm(vector)
            if norm > 0:
                vector = vector / norm
            
            return vector
            
        except Exception as e:
            logger.warning(f"获取原始向量失败: {e}")
            return None
    
    def embed(self, tokens: list) -> torch.Tensor:
        """兼容Embedder接口"""
        # 简单处理：将token转为字符串
        text = " ".join(str(t) for t in tokens)
        return self.embed_text(text)
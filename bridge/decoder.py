"""解码器：海马体激活态 -> LLM可理解的context

将海马体的激活态转换为LLM能理解的context文本。
支持两种模式：text模式（描述性文本）和vector模式（向量字符串）。

遵循架构文档 5.4 节的接口定义。
不依赖具体的LLM实现。
"""

from typing import Optional

import torch
from core.types import Activation


class Decoder:
    """海马体 -> LLM可理解context的解码器。

    将海马体的激活态转换为LLM能理解的context。
    支持两种模式：
    - text模式：生成描述性文本（初始用简单的"记忆强度"描述）
    - vector模式：返回向量字符串供prefix tuning使用

    解码时可选择性传入长时记忆检索结果（recalled_memory），
    使输出 context 不仅反映当前激活态，也包含被检索到的长期记忆，
    解决"记忆只写不读"的架构缺陷（对应审计 S1/B8）。

    属性:
        mode: 解码模式（'text' 或 'vector'）
        top_k: text模式下提取的强激活节点数
        threshold: 激活阈值，低于此值的节点不描述
    """

    def __init__(self, mode: str = 'text', top_k: int = 10,
                 threshold: float = 0.01):
        """初始化解码器。

        参数:
            mode: 解码模式，'text'（描述文本）或 'vector'（向量字符串）
            top_k: text模式下提取强激活节点的数量
            threshold: 激活阈值，低于此值的节点不被描述
        """
        assert mode in ('text', 'vector'), f"不支持的模式: {mode}"
        self.mode = mode
        self.top_k = top_k
        self.threshold = threshold

    def decode(self, activation: Activation,
               recalled_memory: Optional[torch.Tensor] = None) -> str:
        """将激活态转换为context文本。

        参数:
            activation: 海马体激活态
            recalled_memory: 可选，长时记忆检索结果，形状 [num_nodes]。
                由 MemoryManager.recall() 返回。非 None 时，解码结果中
                会附加"长时记忆检索"段落（text模式）或 [MEMORY:...] 段
                （vector模式），使长期记忆参与输出路径。
                为 None 时退化为原有行为，保证向后兼容。

        返回:
            context文本字符串
        """
        if self.mode == 'vector':
            return self._decode_vector(activation, recalled_memory)
        return self._decode_text(activation, recalled_memory)

    def _decode_text(self, activation: Activation,
                     recalled_memory: Optional[torch.Tensor] = None) -> str:
        """text模式：将激活值最高的节点映射为关键词描述。

        生成包含激活熵、惊讶度和强激活节点的描述性文本。
        若提供 recalled_memory，则追加"长时记忆检索"段落。

        参数:
            activation: 海马体激活态
            recalled_memory: 可选，长时记忆检索结果。

        返回:
            描述性context文本
        """
        state = activation.state

        # 找到激活值绝对值最高的top_k个节点
        k = min(self.top_k, len(state))
        top_values, top_indices = torch.topk(state.abs(), k=k)

        # 过滤掉低于阈值的
        mask = top_values > self.threshold
        top_values = top_values[mask]
        top_indices = top_indices[mask]

        if len(top_indices) == 0:
            context = ("[记忆context] 无显著激活节点 "
                       f"(熵:{activation.entropy:.3f}, "
                       f"惊讶度:{activation.surprise:.3f})")
        else:
            # 生成节点描述列表
            descriptions = []
            for val, idx in zip(top_values, top_indices):
                strength = float(val)
                node_id = int(idx)
                # 根据激活强度生成描述
                if strength > 0.5:
                    level = "强"
                elif strength > 0.2:
                    level = "中"
                else:
                    level = "弱"
                descriptions.append(f"节点{node_id}({level}:{strength:.3f})")

            context = (
                f"[记忆context] "
                f"熵:{activation.entropy:.3f}, "
                f"惊讶度:{activation.surprise:.3f} | "
                f"激活节点: {', '.join(descriptions)}"
            )

        # 长时记忆检索段落（B8/S1：让长期记忆参与输出路径）
        if recalled_memory is not None:
            memory_section = self._decode_memory_recall(recalled_memory)
            context = f"{context} | {memory_section}"

        return context

    def _decode_memory_recall(self, recalled_memory: torch.Tensor) -> str:
        """将长时记忆检索结果解码为描述性段落。

        计算记忆检索强度（L2 范数），并描述被激活的记忆维度。

        参数:
            recalled_memory: 长时记忆检索结果，形状 [num_nodes]。

        返回:
            "长时记忆检索"描述段落。
        """
        # 记忆检索强度：L2 范数衡量整体激活量，均值衡量平均强度
        strength = float(recalled_memory.norm().item())

        # 空记忆（检索结果全零）
        if strength < 1e-8:
            return "长时记忆检索: 无相关记忆"

        # top-k 激活的记忆维度
        k = min(self.top_k, len(recalled_memory))
        top_values, top_indices = torch.topk(recalled_memory.abs(), k=k)
        mask = top_values > self.threshold
        top_values = top_values[mask]
        top_indices = top_indices[mask]

        if len(top_indices) == 0:
            return (f"长时记忆检索: 强度{strength:.3f} "
                    f"(无显著节点)")

        descriptions = []
        for val, idx in zip(top_values, top_indices):
            descriptions.append(f"维度{int(idx)}({float(val):.3f})")

        return (f"长时记忆检索: 强度{strength:.3f}, "
                f"活跃维度: {', '.join(descriptions)}")

    def _decode_vector(self, activation: Activation,
                       recalled_memory: Optional[torch.Tensor] = None) -> str:
        """vector模式：返回向量字符串供prefix tuning使用。

        若提供 recalled_memory，则追加 [MEMORY:...] 段，使前缀调优
        可以同时利用当前激活态与检索到的长期记忆。

        参数:
            activation: 海马体激活态
            recalled_memory: 可选，长时记忆检索结果。

        返回:
            向量字符串表示
        """
        state_list = activation.state.tolist()
        # 截断到合理精度
        state_str = ",".join(f"{v:.6f}" for v in state_list)
        result = f"[VECTOR:{state_str}]"

        if recalled_memory is not None:
            mem_list = recalled_memory.tolist()
            mem_str = ",".join(f"{v:.6f}" for v in mem_list)
            result = f"{result}[MEMORY:{mem_str}]"

        return result

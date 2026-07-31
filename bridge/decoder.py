"""解码器：海马体激活态 -> LLM可理解的context

将海马体的激活态转换为LLM能理解的context文本。
支持两种模式：text模式（描述性文本）和vector模式（向量字符串）。

遵循架构文档 5.4 节的接口定义。
不依赖具体的LLM实现。
"""

import math
from typing import List, Optional

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

    text 模式下输出包含三部分（A-P1-2）：
    1. "记忆状态解读"——根据熵/惊讶度/coherence 生成自然语言解释，
       让 AI 能理解记忆系统当前所处的状态。
    2. "相关记忆回忆"——检索到的情景记忆文本（如有）。
    3. "详细数据"——原始指标（熵、惊讶度、激活节点等），供程序化解析。

    属性:
        mode: 解码模式（'text' 或 'vector'）
        top_k: text模式下提取的强激活节点数
        threshold: 激活阈值，低于此值的节点不描述
        surprise_history: 历史惊讶度记录，用于判断当前输入是否"意外"
        max_surprise_history: 历史惊讶度记录上限，避免无限增长
    """

    def __init__(self, mode: str = 'text', top_k: int = 10,
                 threshold: float = 0.01,
                 max_surprise_history: int = 100):
        """初始化解码器。

        参数:
            mode: 解码模式，'text'（描述文本）或 'vector'（向量字符串）
            top_k: text模式下提取强激活节点的数量
            threshold: 激活阈值，低于此值的节点不被描述
            max_surprise_history: 惊讶度历史记录上限（默认100）
        """
        assert mode in ('text', 'vector'), f"不支持的模式: {mode}"
        self.mode = mode
        self.top_k = top_k
        self.threshold = threshold
        # 惊讶度历史：用于将当前惊讶度与历史均值比较，生成"意外/吻合"解读
        self.surprise_history: List[float] = []
        self.max_surprise_history = max_surprise_history

    def decode(self, activation: Activation,
               recalled_memory: Optional[torch.Tensor] = None,
               episodic_texts: Optional[List[str]] = None,
               coherence: Optional[float] = None) -> str:
        """将激活态转换为context文本。

        参数:
            activation: 海马体激活态
            recalled_memory: 可选，长时记忆检索结果，形状 [num_nodes]。
                由 MemoryManager.recall() 返回。非 None 时，解码结果中
                会附加"长时记忆检索"段落（text模式）或 [MEMORY:...] 段
                （vector模式），使长期记忆参与输出路径。
                为 None 时退化为原有行为，保证向后兼容。
            episodic_texts: 可选，情景记忆检索到的原始文本列表。
                由 MemoryManager.recall_episodic() 返回的条目文本提取而来。
                非空时在 context 中附加"相关记忆回忆"段落，使 LLM 能直接
                获取此前对话的语义内容。
            coherence: 可选，目的层一致性（purpose.coherence），取值 0~1。
                用于生成"关注方向是否稳定"的自然语言解读。为 None 时跳过
                该项解读（保证向后兼容）。

        返回:
            context文本字符串
        """
        if self.mode == 'vector':
            return self._decode_vector(activation, recalled_memory)
        return self._decode_text(activation, recalled_memory,
                                 episodic_texts, coherence)

    def _decode_text(self, activation: Activation,
                     recalled_memory: Optional[torch.Tensor] = None,
                     episodic_texts: Optional[List[str]] = None,
                     coherence: Optional[float] = None) -> str:
        """text模式：生成结构化的记忆 context（A-P1-2）。

        输出包含三部分，按可读性排序：
        1. "记忆状态解读"——熵/惊讶度/coherence 的自然语言解释（在前，供 AI 直读）。
        2. "相关记忆回忆"——检索到的情景记忆文本（如有）。
        3. "详细数据"——原始指标（熵、惊讶度、激活节点、长时记忆检索等），
           供程序化解析（保留旧格式以向后兼容 regex 解析）。

        参数:
            activation: 海马体激活态
            recalled_memory: 可选，长时记忆检索结果。
            episodic_texts: 可选，情景记忆检索到的原始文本列表。
            coherence: 可选，目的层一致性。

        返回:
            结构化描述性 context 文本
        """
        # === 1. 记忆状态解读（自然语言解释）===
        interpretations: List[str] = []

        # 熵解读（始终输出，因为熵是核心状态指标）
        interpretations.append(self._interpret_entropy(activation))

        # 惊讶度解读（需历史数据支撑，数据不足时跳过）
        surprise_interp = self._interpret_surprise(activation.surprise)
        if surprise_interp:
            interpretations.append(surprise_interp)

        # coherence 解读（未传入时跳过，保证向后兼容）
        coherence_interp = self._interpret_coherence(coherence)
        if coherence_interp:
            interpretations.append(coherence_interp)

        # 记录惊讶度历史：在解读之后记录，避免当前值参与自身均值比较
        self._record_surprise(activation.surprise)

        # === 2. 详细数据（原始指标，保留旧格式以兼容程序化解析）===
        detail = self._build_detail(activation)

        # 长时记忆检索段落（B8/S1：让长期记忆参与输出路径）
        if recalled_memory is not None:
            memory_section = self._decode_memory_recall(recalled_memory)
            detail = f"{detail} | {memory_section}"

        # === 3. 组装输出：解读在前，情景回忆居中，详细数据在后 ===
        lines = ["[记忆context]", "记忆状态解读:"]
        for interp in interpretations:
            lines.append(f"- {interp}")

        # 相关记忆回忆：将检索到的原始文本直接注入 context
        # 这是"记忆有量无质"问题的核心修复——LLM 能直接读到此前对话内容
        if episodic_texts:
            lines.append("相关记忆回忆:")
            for i, text in enumerate(episodic_texts, 1):
                # 截断过长的文本（避免 context 膨胀）
                truncated = text[:200] + "..." if len(text) > 200 else text
                lines.append(f'{i}. "{truncated}"')

        lines.append(f"详细数据: {detail}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 解释性输出辅助方法（A-P1-2）
    # ------------------------------------------------------------------

    def _interpret_entropy(self, activation: Activation) -> str:
        """根据激活熵生成自然语言解读。

        熵反映记忆激活的分散程度：熵高（接近 ln(num_nodes)）表示多个模式
        同时激活（高唤醒）；熵低表示聚焦于少数模式（清晰记忆痕迹）。

        参数:
            activation: 海马体激活态。

        返回:
            熵状态的自然语言描述。
        """
        num_nodes = len(activation.state)
        # 最大熵 = ln(num_nodes)，与 loop.py 在线熵管理一致
        max_entropy = math.log(num_nodes) if num_nodes > 1 else 1.0
        ratio = activation.entropy / max_entropy if max_entropy > 0 else 0.0

        if ratio > 0.9:
            return "当前记忆处于高唤醒状态，多个记忆模式同时激活"
        elif ratio < 0.5:
            return "当前记忆聚焦于少数模式，形成了清晰的记忆痕迹"
        else:
            return "记忆激活适度"

    def _interpret_surprise(self, surprise: float) -> Optional[str]:
        """将当前惊讶度与历史均值比较，生成自然语言解读。

        惊讶度（自由能）越低表示越符合网络预期。本方法将当前值与历史均值
        对比：显著高于均值 → 意外/学习新内容；显著低于均值 → 与已有记忆
        高度吻合。历史数据不足（<3 条）时返回 None，跳过该解读。

        采用相对差（除以历史均值绝对值）避免负数均值下的符号翻转问题。

        参数:
            surprise: 当前轮次的惊讶度。

        返回:
            惊讶度解读文本，或 None（历史数据不足/无明显偏离时）。
        """
        if len(self.surprise_history) < 3:
            return None

        mean_surprise = sum(self.surprise_history) / len(self.surprise_history)
        # 用均值的绝对值作为尺度，避免负数均值下乘法比较的符号翻转
        scale = abs(mean_surprise) if abs(mean_surprise) > 1e-6 else 1.0
        relative_diff = (surprise - mean_surprise) / scale

        if relative_diff > 0.2:
            return "这个输入让我感到意外，正在学习新内容"
        elif relative_diff < -0.2:
            return "这个输入与已有记忆高度吻合"
        return None

    def _interpret_coherence(self, coherence: Optional[float]) -> Optional[str]:
        """根据目的层 coherence 生成自然语言解读。

        coherence 反映 SLM "兴趣/关注"方向的稳定性：高 coherence 表示关注
        方向稳定；低 coherence 表示正在探索新的兴趣点。

        参数:
            coherence: 目的层一致性，取值 0~1。None 时跳过。

        返回:
            coherence 解读文本，或 None（未传入/处于中间区间时）。
        """
        if coherence is None:
            return None
        if coherence < 0.5:
            return "当前关注方向不太明确，正在探索新的兴趣点"
        elif coherence > 0.8:
            return "当前关注方向稳定"
        return None

    def _record_surprise(self, surprise: float) -> None:
        """记录当前惊讶度到历史，并维持历史上限。"""
        self.surprise_history.append(surprise)
        if len(self.surprise_history) > self.max_surprise_history:
            # 仅保留最近的 max_surprise_history 条
            del self.surprise_history[:len(self.surprise_history) -
                                      self.max_surprise_history]

    def _build_detail(self, activation: Activation) -> str:
        """构建"详细数据"段落：熵、惊讶度与强激活节点（保留旧格式）。

        参数:
            activation: 海马体激活态。

        返回:
            原始指标文本（不含 "[记忆context]" 前缀，由调用方统一添加）。
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
            return (f"无显著激活节点 "
                    f"(熵:{activation.entropy:.3f}, "
                    f"惊讶度:{activation.surprise:.3f})")

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

        return (f"熵:{activation.entropy:.3f}, "
                f"惊讶度:{activation.surprise:.3f} | "
                f"激活节点: {', '.join(descriptions)}")

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

"""
活体记忆系统 - 海马体核心：多尺度记忆管理
==========================================

管理短时与长时潜变量，模拟人类记忆的多时间尺度特性:
  - 短时记忆（working memory）: 快速更新，快速遗忘，用于即时上下文
  - 长时记忆（long-term memory）: 缓慢更新，持久保留，形成稳定身份

记忆不是存档——是持续衰减与更新的潜变量（latent state）。
consolidation（巩固）机制将短时记忆迁移到长时记忆，
模拟睡眠期间的记忆巩固过程。

参考：架构文档 第五节 5.3 MemoryManager 接口
"""

import torch
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from core.types import Activation


@dataclass
class EpisodicEntry:
    """情景记忆条目：保存原始文本与语义向量，用于逆向解码。

    海马体不仅存储抽象的吸引子模式，也保存情景记忆的原始内容
    （"发生了什么"），使记忆能够被还原为 LLM 可理解的语义文本。

    属性:
        text: 原始对话文本（用户输入或包含LLM输出的拼接）。
        semantic_vector: 用于检索的语义向量。优先存储预训练模型的原始
            高维向量（PretrainedEmbedder.embed_text_raw 输出，384 维，
            投影前、L2 归一化），以保留完整语义信息、提升检索精度；
            当无法获取原始向量时（如 SimpleEmbedder）退化为存储投影后
            的低维向量（64 维），保证向后兼容。
        surprise: 该条目的惊讶度（自由能），用于重要性加权。
        turn: 对话轮次编号。
    """
    text: str
    semantic_vector: torch.Tensor
    surprise: float
    turn: int


class MemoryManager:
    """多尺度记忆管理器。

    使用两个指数移动平均（EMA）潜变量:
      - short_term_latent: 高衰减率（decay=0.8），快速跟踪近期激活
      - long_term_latent:  低衰减率（decay=0.999），持久保留稳定模式

    属性:
        short_term_latent: 短时潜变量，形状 [num_nodes]。
        long_term_latent:  长时潜变量，形状 [num_nodes]。
    """

    def __init__(self, num_nodes: int,
                 short_term_decay: float = 0.8,
                 long_term_decay: float = 0.999,
                 transfer_rate: float = 0.1,
                 replay_count: int = 10,
                 replay_weight: float = 0.01,
                 consolidation_decay: float = 0.5,
                 buffer_capacity: int = 100,
                 episodic_capacity: int = 200) -> None:
        """初始化记忆管理器。

        参数:
            num_nodes: 节点数（= 激活态维度）。
            short_term_decay: 短时记忆衰减系数。
                越小遗忘越快。0.8 表示每步保留 80%，遗忘 20%。
            long_term_decay: 长时记忆衰减系数。
                越接近 1 保留越久。0.999 表示每步保留 99.9%。
            transfer_rate: 巩固时短时→长时的迁移率。
            replay_count: 巩固时回放的条目数。
            replay_weight: 巩固时回放的权重。
            consolidation_decay: 巩固后短时记忆的衰减系数。
            buffer_capacity: 经验缓冲区容量（用于回放）。
            episodic_capacity: 情景记忆缓冲区容量（保存原始文本+语义向量）。
        """
        self.num_nodes = num_nodes
        self.short_term_decay = short_term_decay
        self.long_term_decay = long_term_decay
        self.transfer_rate = transfer_rate
        self.replay_count = replay_count
        self.replay_weight = replay_weight
        self.consolidation_decay = consolidation_decay
        self._buffer_capacity = buffer_capacity

        # EMA 潜变量：初始为零
        self.short_term_latent: torch.Tensor = torch.zeros(num_nodes)
        self.long_term_latent: torch.Tensor = torch.zeros(num_nodes)

        # 缓冲区：存储 (state, surprise) 元组，用于 consolidation 重要性加权回放
        # 使用 deque 自动处理容量限制（B4 修复），O(1) 追加与淘汰
        self._buffer: deque = deque(maxlen=self._buffer_capacity)

        # 情景记忆缓冲区：保存 (text, semantic_vector, surprise, turn)
        # 用于将记忆逆向解码为 LLM 可理解的语义文本
        self._episodic_buffer: deque = deque(maxlen=episodic_capacity)

    def update(self, activation: Activation, surprise: float = 0.0) -> None:
        """更新短时/长时记忆潜变量。

        使用指数移动平均（EMA）:
            short_term = decay_s * short_term + (1 - decay_s) * activation
            long_term  = decay_l * long_term  + (1 - decay_l) * activation

        短时记忆快速跟踪近期激活（遗忘快），
        长时记忆缓慢积累稳定模式（保留久）。

        参数:
            activation: 当前激活态。
            surprise: 当前激活态的惊讶度（自由能），用于后续 consolidation
                的重要性加权回放。从 activation.surprise 获取。
        """
        state = activation.state
        alpha_s = 1.0 - self.short_term_decay  # 短时 EMA 权重
        alpha_l = 1.0 - self.long_term_decay   # 长时 EMA 权重

        self.short_term_latent = (
            self.short_term_decay * self.short_term_latent + alpha_s * state
        )
        self.long_term_latent = (
            self.long_term_decay * self.long_term_latent + alpha_l * state
        )

        # 缓冲 (state, surprise) 用于回放（deque 自动处理容量限制）
        self._buffer.append((state.clone(), surprise))

    def consolidate(self) -> None:
        """记忆巩固：短时 -> 长时迁移，回放重要经验。

        模拟睡眠期间的记忆巩固:
          1. 将短时潜变量的部分内容迁移到长时潜变量
          2. 按 surprise 排序回放缓冲区中的经验（模拟 REM 睡眠回放）
             优先回放高 surprise（重要/意外）的条目，而非简单的最近条目
          3. 衰减短时潜变量（为新记忆腾出空间）

        这个过程是"身份"稳定的关键——
        短暂的经历被筛选、强化后固化为长期记忆。
        """
        # 短时 -> 长时迁移
        self.long_term_latent = (
            self.long_term_latent + self.transfer_rate * self.short_term_latent
        )

        # 按 surprise 排序回放重要经验（G3 修复：重要性加权而非时间序）
        if self._buffer:
            # 按 surprise 降序排列，优先回放高 surprise 条目
            sorted_buffer = sorted(self._buffer, key=lambda x: x[1], reverse=True)
            replay_count = min(self.replay_count, len(sorted_buffer))
            for state, surprise in sorted_buffer[:replay_count]:
                # 回放权重也按 surprise 加权（高 surprise = 重要经验）
                weight = self.replay_weight * max(surprise, 0.0)
                self.long_term_latent = (
                    self.long_term_latent + weight * state
                )

        # 衰减短时记忆（为新记忆腾出空间）
        self.short_term_latent = self.short_term_latent * self.consolidation_decay

    def recall(self, cue: torch.Tensor) -> torch.Tensor:
        """从记忆中检索：用线索激活相关记忆。

        使用线索对长时潜变量进行门控（gating），
        返回与线索相关的记忆内容。

        门控机制:
            gate = sigmoid(cue)              # 线索决定哪些维度被激活
            recalled = long_term * gate      # 门控后的记忆

        参数:
            cue: 检索线索，形状 [num_nodes]。

        返回:
            检索到的记忆向量，形状 [num_nodes]。
        """
        # 线索门控：cue 决定哪些记忆维度被激活
        gate = torch.sigmoid(cue)
        recalled = self.long_term_latent * gate
        return recalled

    def store_episodic(self, text: str, semantic_vector: torch.Tensor,
                       surprise: float, turn: int,
                       raw_semantic_vector: Optional[torch.Tensor] = None
                       ) -> None:
        """存入情景记忆条目（原始文本 + 语义向量）。

        将对话的原始文本和语义向量一起保存，使记忆系统不仅能
        在吸引子空间中运作，还能将记忆逆向还原为 LLM 可理解的语义文本。

        向量选择策略（高精度优先 + 向后兼容）：
          - 若提供 raw_semantic_vector（预训练模型原始 384 维向量，投影前），
            则存储它——保留完整语义信息，提升检索精度；
          - 否则退化为存储 semantic_vector（投影后低维向量），兼容
            SimpleEmbedder 等无原始向量的嵌入器。

        参数:
            text: 原始对话文本。
            semantic_vector: 投影后的语义向量（低维，用于吸引子网络路径，
                亦作为无 raw 向量时的检索退化存储）。
            surprise: 该条目的惊讶度，用于后续重要性加权。
            turn: 对话轮次编号。
            raw_semantic_vector: 可选，预训练模型原始高维语义向量（投影前、
                L2 归一化）。提供时优先存入 episodic 缓冲区用于高精度检索。
        """
        # 优先存储原始高维向量；无原始向量时退化为投影向量（向后兼容）
        store_vector = (raw_semantic_vector if raw_semantic_vector is not None
                        else semantic_vector)
        entry = EpisodicEntry(
            text=text,
            semantic_vector=store_vector.detach().clone(),
            surprise=surprise,
            turn=turn,
        )
        self._episodic_buffer.append(entry)

    def recall_episodic(self, query_vector: torch.Tensor,
                        top_k: int = 3,
                        fallback_query: Optional[torch.Tensor] = None
                        ) -> List[EpisodicEntry]:
        """基于语义相似度检索情景记忆，返回最相关的文本条目。

        用查询向量与缓冲区中所有条目的语义向量做余弦相似度，
        返回相似度最高的 top_k 个条目。

        维度兼容与向后兼容：
          - 优先用 query_vector（384 维原始向量）检索维度匹配的条目；
          - 若某条目维度与 query_vector 不一致（如旧快照存的 64 维投影
            向量），但与 fallback_query 维度一致，则改用 fallback_query
            计算该条目相似度（退化为原有投影向量检索行为）；
          - 维度均不匹配的条目被跳过。这保证旧快照恢复后检索不崩溃，
            也支持"旧 64 维 + 新 384 维"混合缓冲区。

        参数:
            query_vector: 查询的原始语义向量（384 维，投影前）。
            top_k: 返回的最大条目数。
            fallback_query: 可选，投影后的查询向量（64 维）。用于与旧快照
                中 64 维条目的兼容检索。

        返回:
            最相关的 EpisodicEntry 列表（按相似度降序）。
            缓冲区为空时返回空列表。
        """
        if len(self._episodic_buffer) == 0:
            return []

        # 准备主查询向量（归一化）
        query = query_vector.detach().cpu().float()
        if query.dim() == 1:
            query = query.unsqueeze(0)  # [1, qd]
        query_norm = torch.nn.functional.normalize(query, dim=-1)
        qd = query.shape[-1]

        # 准备 fallback 查询向量（归一化），用于与旧 64 维条目兼容
        if fallback_query is not None:
            fb = fallback_query.detach().cpu().float()
            if fb.dim() == 1:
                fb = fb.unsqueeze(0)  # [1, fd]
            fb_norm = torch.nn.functional.normalize(fb, dim=-1)
            fd = fb.shape[-1]
        else:
            fb_norm = None
            fd = None

        # 逐条计算相似度，按维度自动选择查询向量（支持混合维度缓冲区）
        scored: List[Tuple[float, EpisodicEntry]] = []
        for entry in self._episodic_buffer:
            v = entry.semantic_vector.detach().cpu().float()
            if v.dim() > 1:
                v = v.squeeze()  # 防御性处理，统一为 1-D
            vd = v.shape[-1]
            if vd == qd:
                v_norm = torch.nn.functional.normalize(v, dim=0)
                sim = torch.dot(query_norm.squeeze(0), v_norm).item()
            elif fb_norm is not None and vd == fd:
                v_norm = torch.nn.functional.normalize(v, dim=0)
                sim = torch.dot(fb_norm.squeeze(0), v_norm).item()
            else:
                # 维度均不匹配，跳过该条目（优雅降级，不崩溃）
                continue
            scored.append((sim, entry))

        if not scored:
            return []

        # 按相似度降序取 top_k
        k = min(top_k, len(scored))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:k]]

    def get_state(self) -> dict:
        """返回记忆管理器当前状态（用于快照）。

        返回:
            包含短时/长时潜变量和情景记忆缓冲区的字典。
        """
        return {
            "short_term_latent": self.short_term_latent.clone(),
            "long_term_latent": self.long_term_latent.clone(),
            "num_nodes": self.num_nodes,
            "episodic_buffer": list(self._episodic_buffer),
        }

    def set_state(self, state: dict) -> None:
        """从快照恢复记忆状态。

        参数:
            state: get_state() 返回的字典。
        """
        self.short_term_latent = state["short_term_latent"].clone()
        self.long_term_latent = state["long_term_latent"].clone()
        self.num_nodes = state["num_nodes"]
        # 情景记忆缓冲区恢复（向后兼容：旧快照无此字段时跳过）
        if "episodic_buffer" in state:
            maxlen = self._episodic_buffer.maxlen or 200
            self._episodic_buffer = deque(state["episodic_buffer"], maxlen=maxlen)

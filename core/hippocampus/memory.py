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
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from core.types import Activation, resolve_device


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
        source: 条目来源标记。``'external'`` 表示来自外部对话，
            ``'self_ref'`` 表示来自自指回路。默认 ``'external'``，
            保证向后兼容（自指回路 Phase 0+ 起用）。
    """
    text: str
    semantic_vector: torch.Tensor
    surprise: float
    turn: int
    source: str = 'external'  # 'external' | 'self_ref'


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
                 episodic_capacity: int = 200,
                 device: Union[str, torch.device] = "auto") -> None:
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
            device: 计算设备（E-P2-1）。支持 "auto"/"cpu"/"cuda"/"cuda:0"
                或 torch.device。短时/长时潜变量将创建在该设备上。
                存入缓冲区的激活态和语义向量也会迁移到该设备。
        """
        self.num_nodes = num_nodes
        self.short_term_decay = short_term_decay
        self.long_term_decay = long_term_decay
        self.transfer_rate = transfer_rate
        self.replay_count = replay_count
        self.replay_weight = replay_weight
        self.consolidation_decay = consolidation_decay
        self._buffer_capacity = buffer_capacity

        # E-P2-1: 统一设备管理
        self.device: torch.device = resolve_device(device)

        # EMA 潜变量：初始为零（E-P2-1: 创建在 device 上）
        self.short_term_latent: torch.Tensor = torch.zeros(
            num_nodes, device=self.device)
        self.long_term_latent: torch.Tensor = torch.zeros(
            num_nodes, device=self.device)

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
        # E-P2-1: 迁移到正确 device，防止潜变量与输入张量设备不一致
        state = state.to(self.device)
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
        # E-P2-1: 迁移 cue 到正确 device
        cue = cue.to(self.device)
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
        # E-P2-1: 迁移到正确 device
        store_vector = store_vector.to(self.device)
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

        # 按维度分组（兼容 384 维 + 64 维混合缓冲区）
        # 将同一维度的条目聚集到一起，以便用一次矩阵乘法算完所有相似度，
        # 消除逐条 torch.dot + .item() 带来的 Python 循环与 GPU/CPU 同步开销。
        groups: dict = {}
        for entry in self._episodic_buffer:
            v = entry.semantic_vector.detach().cpu().float()
            if v.dim() > 1:
                v = v.squeeze()  # 防御性处理，统一为 1-D
            vd = v.shape[-1]
            groups.setdefault(vd, []).append((entry, v))

        # 每组用 torch.stack 成矩阵，一次 matmul 算完所有相似度
        scored: List[Tuple[float, EpisodicEntry]] = []
        for vd, items in groups.items():
            # 按维度自动选择查询向量（支持混合维度缓冲区）
            if vd == qd:
                q_vec = query_norm.squeeze(0)  # [qd]
            elif fb_norm is not None and vd == fd:
                q_vec = fb_norm.squeeze(0)  # [fd]
            else:
                # 维度均不匹配，跳过该组（优雅降级，不崩溃）
                continue

            # 堆叠为矩阵 [n, vd]，一次矩阵乘法算完所有相似度
            mat = torch.stack([v for (_, v) in items])
            mat_norm = torch.nn.functional.normalize(mat, dim=-1)
            sims = (mat_norm @ q_vec).tolist()  # 一次同步取回所有相似度
            for sim, (entry, _) in zip(sims, items):
                scored.append((sim, entry))

        if not scored:
            return []

        # 按相似度降序取 top_k
        k = min(top_k, len(scored))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:k]]

    # ------------------------------------------------------------------ #
    #  公开缓冲区访问接口
    #  ------------------------------------------------------------------ #
    #  解耦外部对 _buffer / _episodic_buffer 私有属性的直接访问。
    #  DreamEngine、server、session_manager、dream_scheduler 等上层模块
    #  应通过这些接口查询缓冲区状态，而非直接读取私有 deque。
    # ------------------------------------------------------------------ #

    def buffer_size(self) -> int:
        """返回回放缓冲区当前条目数。

        用于上层模块（如 DreamScheduler、DreamEngine）查询是否有记忆
        可回放，而无需直接访问 ``self._buffer`` 私有属性。
        """
        return len(self._buffer)

    def episodic_size(self) -> int:
        """返回情景记忆缓冲区当前条目数。

        用于上层模块查询情景记忆条目数量（如状态报告、检索前置检查），
        而无需直接访问 ``self._episodic_buffer`` 私有属性。
        """
        return len(self._episodic_buffer)

    def iter_buffer(self):
        """返回回放缓冲区的只读迭代器。

        返回 ``iter(self._buffer)``，调用方可用于遍历 ``(state, surprise)``
        元组进行采样或回放。deque 迭代器是只读的，调用方不应原地修改。
        """
        return iter(self._buffer)

    def iter_episodic(self):
        """返回情景记忆缓冲区的只读迭代器。

        返回 ``iter(self._episodic_buffer)``，调用方可用于遍历
        ``EpisodicEntry`` 条目进行检索或修剪。deque 迭代器是只读的，
        调用方不应原地修改。
        """
        return iter(self._episodic_buffer)

    def get_episodic_maxlen(self) -> int:
        """返回情景记忆缓冲区的最大容量。

        返回 ``self._episodic_buffer.maxlen``，若为 None（无界）则回退到
        默认值 200。用于上层模块在重建缓冲区时获取正确的容量上限。
        """
        return self._episodic_buffer.maxlen or 200

    def replace_episodic_buffer(self, entries) -> int:
        """用 entries 重建情景记忆缓冲区，返回被剔除的条目数。

        在保留原缓冲区容量上限（maxlen）的前提下，用新的条目序列替换
        当前缓冲区内容。当 entries 长度超过 maxlen 时，deque 会自动
        保留最新的条目（丢弃头部旧条目）。

        参数:
            entries: 可迭代的 EpisodicEntry 序列，用于重建缓冲区。

        返回:
            被剔除的条目数 = 旧缓冲区长度 - 新缓冲区长度。
            （注意：若 entries 超过 maxlen，deque 内部淘汰的部分也计入
            新缓冲区长度，因此返回值反映的是净变化量。）
        """
        maxlen = self._episodic_buffer.maxlen or 200
        old_len = len(self._episodic_buffer)
        self._episodic_buffer = deque(entries, maxlen=maxlen)
        return old_len - len(self._episodic_buffer)

    def get_state(self) -> dict:
        """返回记忆管理器当前状态（用于快照）。

        返回:
            包含短时/长时潜变量、回放缓冲区和情景记忆缓冲区的字典。
        """
        return {
            "short_term_latent": self.short_term_latent.clone(),
            "long_term_latent": self.long_term_latent.clone(),
            "num_nodes": self.num_nodes,
            "buffer": list(self._buffer),
            "episodic_buffer": list(self._episodic_buffer),
        }

    def set_state(self, state: dict) -> None:
        """从快照恢复记忆状态。

        E-P2-1: 恢复的张量自动迁移到记忆管理器当前 device。

        参数:
            state: get_state() 返回的字典。
        """
        self.short_term_latent = state["short_term_latent"].clone().to(self.device)
        self.long_term_latent = state["long_term_latent"].clone().to(self.device)
        self.num_nodes = state["num_nodes"]
        # 回放缓冲区恢复（向后兼容：旧快照无此字段时跳过）
        if "buffer" in state:
            maxlen = self._buffer.maxlen or 100
            # E-P2-1: 缓冲区中的 state 张量迁移到当前 device
            migrated_buffer = [
                (s.clone().to(self.device), surp)
                for s, surp in state["buffer"]
            ]
            self._buffer = deque(migrated_buffer, maxlen=maxlen)
        # 情景记忆缓冲区恢复（向后兼容：旧快照无此字段时跳过）
        if "episodic_buffer" in state:
            maxlen = self._episodic_buffer.maxlen or 200
            # E-P2-1: episodic 向量迁移到当前 device
            migrated_episodic = []
            for entry in state["episodic_buffer"]:
                entry.semantic_vector = entry.semantic_vector.to(self.device)
                migrated_episodic.append(entry)
            self._episodic_buffer = deque(migrated_episodic, maxlen=maxlen)

    def to(self, device: Union[str, torch.device]) -> 'MemoryManager':
        """将记忆管理器所有张量迁移到指定设备（E-P2-1）。

        迁移 short_term_latent、long_term_latent、回放缓冲区中的状态
        以及情景缓冲区中的语义向量到目标设备，并更新 self.device。

        参数:
            device: 目标设备（str / torch.device）。

        返回:
            self（支持链式调用）。
        """
        self.device = resolve_device(device)
        self.short_term_latent = self.short_term_latent.to(self.device)
        self.long_term_latent = self.long_term_latent.to(self.device)
        # 迁移回放缓冲区中的 state 张量
        if self._buffer:
            migrated = [(s.to(self.device), surp) for s, surp in self._buffer]
            maxlen = self._buffer.maxlen or 100
            self._buffer = deque(migrated, maxlen=maxlen)
        # 迁移情景缓冲区中的语义向量
        if self._episodic_buffer:
            for entry in self._episodic_buffer:
                entry.semantic_vector = entry.semantic_vector.to(self.device)
        return self

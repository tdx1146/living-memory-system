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

from core.types import Activation


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
                 buffer_capacity: int = 100) -> None:
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
            buffer_capacity: 经验缓冲区容量。
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

    def get_state(self) -> dict:
        """返回记忆管理器当前状态（用于快照）。

        返回:
            包含短时/长时潜变量的字典。
        """
        return {
            "short_term_latent": self.short_term_latent.clone(),
            "long_term_latent": self.long_term_latent.clone(),
            "num_nodes": self.num_nodes,
        }

    def set_state(self, state: dict) -> None:
        """从快照恢复记忆状态。

        参数:
            state: get_state() 返回的字典。
        """
        self.short_term_latent = state["short_term_latent"].clone()
        self.long_term_latent = state["long_term_latent"].clone()
        self.num_nodes = state["num_nodes"]

"""
测试多尺度记忆管理
====================

验证内容:
  1. 多尺度更新（短时 vs 长时）
  2. consolidation（短时 -> 长时迁移）
  3. recall 检索
  4. 状态快照与恢复
"""

import pytest
import torch

from core.hippocampus.memory import MemoryManager
from core.types import Activation


# ----------------------------------------------------------------------- #
#  辅助函数
# ----------------------------------------------------------------------- #


def make_activation(num_nodes: int, pattern: torch.Tensor = None) -> Activation:
    """构造测试用激活态。

    参数:
        num_nodes: 节点数。
        pattern: 激活模式，形状 [num_nodes]。None 时用随机值。

    返回:
        构造的 Activation 对象。
    """
    if pattern is None:
        pattern = torch.randn(num_nodes) * 0.5
    entropy = -torch.sum(
        (pattern.abs() / pattern.abs().sum()) *
        torch.log(pattern.abs() / pattern.abs().sum() + 1e-8)
    ).item()
    return Activation(state=pattern, entropy=entropy, surprise=1.0)


# ----------------------------------------------------------------------- #
#  多尺度更新测试
# ----------------------------------------------------------------------- #


class TestMultiScaleUpdate:
    """短时/长时记忆的多尺度更新。"""

    def test_initial_latents_zero(self):
        """初始潜变量为零。"""
        mem = MemoryManager(num_nodes=32)
        assert torch.allclose(mem.short_term_latent, torch.zeros(32))
        assert torch.allclose(mem.long_term_latent, torch.zeros(32))

    def test_update_changes_short_term(self):
        """update 后短时记忆发生变化。"""
        mem = MemoryManager(num_nodes=32)
        act = make_activation(32, torch.ones(32) * 0.5)

        mem.update(act, act.surprise)
        assert not torch.allclose(mem.short_term_latent, torch.zeros(32))

    def test_update_changes_long_term(self):
        """update 后长时记忆也发生变化（微小）。"""
        mem = MemoryManager(num_nodes=32)
        act = make_activation(32, torch.ones(32) * 0.5)

        mem.update(act, act.surprise)
        assert not torch.allclose(mem.long_term_latent, torch.zeros(32))

    def test_short_term_changes_faster(self):
        """短时记忆比长时记忆变化更快。"""
        mem = MemoryManager(
            num_nodes=32,
            short_term_decay=0.8,
            long_term_decay=0.999,
        )
        act = make_activation(32, torch.ones(32) * 1.0)

        mem.update(act, act.surprise)

        # 短时记忆变化幅度应远大于长时记忆
        short_change = mem.short_term_latent.abs().mean().item()
        long_change = mem.long_term_latent.abs().mean().item()

        assert short_change > long_change * 10  # 短时应远大于长时

    def test_short_term_forgets_quickly(self):
        """短时记忆快速遗忘：无输入后快速衰减。"""
        mem = MemoryManager(
            num_nodes=32,
            short_term_decay=0.5,  # 快速遗忘
            long_term_decay=0.999,
        )

        # 写入一个模式
        pattern = torch.ones(32) * 1.0
        act = make_activation(32, pattern)
        mem.update(act, act.surprise)

        short_after_write = mem.short_term_latent.clone()

        # 然后写入零模式（模拟"无输入"）
        for _ in range(5):
            mem.update(make_activation(32, torch.zeros(32)), 1.0)

        # 短时记忆应大幅衰减
        assert mem.short_term_latent.abs().mean() < short_after_write.abs().mean() * 0.5

    def test_long_term_persists(self):
        """长时记忆持久保留。"""
        mem = MemoryManager(
            num_nodes=32,
            short_term_decay=0.8,
            long_term_decay=0.999,
        )

        # 写入一个模式
        pattern = torch.ones(32) * 1.0
        mem.update(make_activation(32, pattern), 1.0)
        long_after_write = mem.long_term_latent.clone()

        # 然后写入不同模式
        for _ in range(5):
            mem.update(make_activation(32, torch.zeros(32)), 1.0)

        # 长时记忆应保留大部分原始值
        retention = (mem.long_term_latent * long_after_write).sum().item() / (
            long_after_write.norm().item() ** 2 + 1e-8
        )
        assert retention > 0.9  # 保留 90% 以上

    def test_repeated_updates_accumulate(self):
        """重复更新使记忆逐步积累。"""
        mem = MemoryManager(num_nodes=32, long_term_decay=0.99)
        pattern = torch.ones(32) * 0.5
        act = make_activation(32, pattern)

        # 第一次更新
        mem.update(act, act.surprise)
        long_after_1 = mem.long_term_latent.abs().sum().item()

        # 再更新 9 次
        for _ in range(9):
            mem.update(act, act.surprise)

        long_after_10 = mem.long_term_latent.abs().sum().item()

        # 10 次后积累应大于 1 次
        assert long_after_10 > long_after_1


# ----------------------------------------------------------------------- #
#  Consolidation 测试
# ----------------------------------------------------------------------- #


class TestConsolidation:
    """记忆巩固：短时 -> 长时迁移。"""

    def test_consolidate_transfers_to_long_term(self):
        """consolidate 后长时记忆增加（吸收了短时记忆）。"""
        mem = MemoryManager(num_nodes=32, short_term_decay=0.8)

        # 写入短时记忆
        pattern = torch.ones(32) * 0.5
        for _ in range(5):
            mem.update(make_activation(32, pattern), 1.0)

        long_before = mem.long_term_latent.clone()
        short_before = mem.short_term_latent.clone()

        # 巩固
        mem.consolidate()

        # 长时记忆应增加（吸收了短时记忆的内容）
        long_after = mem.long_term_latent
        assert long_after.abs().sum() > long_before.abs().sum()

    def test_consolidate_reduces_short_term(self):
        """consolidate 后短时记忆衰减。"""
        mem = MemoryManager(num_nodes=32, short_term_decay=0.8)

        # 写入短时记忆
        pattern = torch.ones(32) * 0.5
        for _ in range(5):
            mem.update(make_activation(32, pattern), 1.0)

        short_before = mem.short_term_latent.abs().sum().item()

        # 巩固
        mem.consolidate()

        short_after = mem.short_term_latent.abs().sum().item()

        # 短时记忆应衰减
        assert short_after < short_before

    def test_consolidate_preserves_information(self):
        """consolidate 不会丢失信息（迁移到长时）。"""
        mem = MemoryManager(num_nodes=32, short_term_decay=0.8, long_term_decay=0.999)

        # 写入特定模式
        pattern = torch.zeros(32)
        pattern[0] = 1.0
        pattern[1] = 1.0
        for _ in range(5):
            mem.update(make_activation(32, pattern), 1.0)

        total_before = (
            mem.short_term_latent.abs().sum().item()
            + mem.long_term_latent.abs().sum().item()
        )

        mem.consolidate()

        total_after = (
            mem.short_term_latent.abs().sum().item()
            + mem.long_term_latent.abs().sum().item()
        )

        # 总信息量不应大幅减少
        assert total_after > total_before * 0.5

    def test_multiple_consolidations(self):
        """多次 consolidate 不会崩溃。"""
        mem = MemoryManager(num_nodes=32)

        # 持续更新和巩固
        for _ in range(10):
            pattern = torch.randn(32) * 0.5
            mem.update(make_activation(32, pattern), 1.0)
            mem.consolidate()

        # 不应产生 NaN/Inf
        assert not torch.any(torch.isnan(mem.long_term_latent))
        assert not torch.any(torch.isinf(mem.long_term_latent))
        assert not torch.any(torch.isnan(mem.short_term_latent))
        assert not torch.any(torch.isinf(mem.short_term_latent))


# ----------------------------------------------------------------------- #
#  Recall 测试
# ----------------------------------------------------------------------- #


class TestRecall:
    """记忆检索测试。"""

    def test_recall_returns_tensor(self):
        """recall 返回正确形状的张量。"""
        mem = MemoryManager(num_nodes=32)
        cue = torch.randn(32)

        result = mem.recall(cue)
        assert result.shape == (32,)

    def test_recall_empty_memory(self):
        """空记忆时 recall 返回零向量。"""
        mem = MemoryManager(num_nodes=32)
        cue = torch.randn(32)

        result = mem.recall(cue)
        assert torch.allclose(result, torch.zeros(32))

    def test_recall_retrieves_stored_pattern(self):
        """recall 能检索到存储的模式。"""
        mem = MemoryManager(num_nodes=32, long_term_decay=0.9)

        # 存入一个模式
        pattern = torch.zeros(32)
        pattern[0] = 1.0
        pattern[1] = 1.0
        for _ in range(20):
            mem.update(make_activation(32, pattern), 1.0)

        # 用相同模式作为 cue 检索
        cue = pattern.clone()
        recalled = mem.recall(cue)

        # 检索结果在存储模式活跃的维度上应非零
        assert recalled[0].abs() > 0.01
        assert recalled[1].abs() > 0.01

    def test_recall_gated_by_cue(self):
        """recall 结果受 cue 门控。"""
        mem = MemoryManager(num_nodes=32, long_term_decay=0.9)

        # 存入均匀模式
        pattern = torch.ones(32) * 0.5
        for _ in range(20):
            mem.update(make_activation(32, pattern), 1.0)

        # cue 只激活前半部分
        cue = torch.zeros(32)
        cue[:16] = 10.0  # 高值 → sigmoid ≈ 1
        cue[16:] = -10.0  # 低值 → sigmoid ≈ 0

        recalled = mem.recall(cue)

        # 前半部分应被激活，后半部分应被抑制
        front_mean = recalled[:16].abs().mean().item()
        back_mean = recalled[16:].abs().mean().item()
        assert front_mean > back_mean


# ----------------------------------------------------------------------- #
#  状态快照测试
# ----------------------------------------------------------------------- #


class TestMemoryState:
    """记忆状态快照与恢复。"""

    def test_get_state(self):
        """get_state 返回正确的状态字典。"""
        mem = MemoryManager(num_nodes=32)
        state = mem.get_state()

        assert "short_term_latent" in state
        assert "long_term_latent" in state
        assert state["short_term_latent"].shape == (32,)
        assert state["long_term_latent"].shape == (32,)

    def test_set_state_restores(self):
        """set_state 能恢复之前保存的状态。"""
        mem = MemoryManager(num_nodes=32)
        pattern = torch.randn(32) * 0.5
        for _ in range(5):
            mem.update(make_activation(32, pattern), 1.0)

        state = mem.get_state()

        # 创建新管理器并恢复
        mem2 = MemoryManager(num_nodes=32)
        mem2.set_state(state)

        assert torch.allclose(mem.short_term_latent, mem2.short_term_latent)
        assert torch.allclose(mem.long_term_latent, mem2.long_term_latent)

    def test_state_is_copy(self):
        """get_state 返回副本。"""
        mem = MemoryManager(num_nodes=32)
        pattern = torch.ones(32) * 0.5
        mem.update(make_activation(32, pattern), 1.0)

        state = mem.get_state()
        state["short_term_latent"].fill_(0.0)

        # 原始状态不受影响
        assert not torch.allclose(mem.short_term_latent, torch.zeros(32))

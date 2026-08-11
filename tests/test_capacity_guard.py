# -*- coding: utf-8 -*-
"""容量守卫测试：episodic deque 无界化（提取层 v1.4 S1-14，P1-A）

覆盖：
  1. deque 无界化：maxlen is None，时间序 FIFO 自动淘汰不复存在
  2. 软容量满员【不丢】＋capacity_full_events 计数＋告警（丰碑哲学）
  3. 90% 预警线：capacity_warning_events 计数
  4. 硬顶兜底：超顶丢最旧＋capacity_hard_drops 计数（工程兜底非 D1 淘汰）
  5. set_state / replace_episodic_buffer 不再引入 maxlen FIFO 语义
"""

import torch

from core.hippocampus.memory import MemoryManager


def _fill(m: MemoryManager, n: int, start_turn: int = 0) -> None:
    """向 episodic 缓冲区塞 n 条条目。"""
    for i in range(n):
        m.store_episodic(
            f"条目{i}", torch.randn(8), surprise=0.5,
            turn=start_turn + i)


class TestDequeUnbounded:
    def test_deque_has_no_maxlen(self):
        """deque 无界化：maxlen is None（时间序 FIFO 淘汰移除，P1-A 攻击点 1）。"""
        m = MemoryManager(num_nodes=16)
        assert m._episodic_buffer.maxlen is None
        # 有效容量参考 = 软容量
        assert m.get_episodic_maxlen() == 2000

    def test_default_soft_capacity_is_2000(self):
        m = MemoryManager(num_nodes=16)
        assert m.episodic_capacity == 2000


class TestSoftCapacityNoDrop:
    def test_soft_cap_full_no_drop(self):
        """软容量满员不丢＋计数（丰碑哲学：满了再说）。"""
        m = MemoryManager(num_nodes=16, episodic_capacity=50,
                          episodic_hard_cap=100)
        _fill(m, 50)
        assert m.episodic_size() == 50          # 满员不丢
        assert m.capacity_full_events >= 1
        # 继续写入：仍不丢（软容量只是告警线，不是淘汰线）
        _fill(m, 10, start_turn=100)
        assert m.episodic_size() == 60
        assert m.capacity_full_events >= 2
        # 最旧条目仍在（无 FIFO 淘汰）
        turns = sorted(e.turn for e in m.iter_episodic())
        assert turns[0] == 0

    def test_warning_line_counter(self):
        """90% 预警线：接近容量时告警＋计数。"""
        m = MemoryManager(num_nodes=16, episodic_capacity=50,
                          episodic_hard_cap=100)
        _fill(m, 44)  # 44 < 45 (90%) → 未触发
        assert m.capacity_warning_events == 0
        _fill(m, 2)   # 46 ≥ 45 → 触发
        assert m.capacity_warning_events == 1
        assert m.capacity_full_events == 0


class TestHardCapFallback:
    def test_hard_cap_drops_oldest(self):
        """硬顶兜底：超顶丢最旧＋ERROR 计数（工程兜底，非 D1 淘汰）。"""
        m = MemoryManager(num_nodes=16, episodic_capacity=20,
                          episodic_hard_cap=30)
        _fill(m, 30)
        assert m.episodic_size() == 30
        assert m.capacity_hard_drops == 0
        _fill(m, 5, start_turn=1000)  # 35 > 30 → 丢 5 条最旧
        assert m.episodic_size() == 30
        assert m.capacity_hard_drops == 5
        turns = sorted(e.turn for e in m.iter_episodic())
        assert turns[0] == 5      # turn 0-4 被丢（最旧先丢）
        assert turns[-1] == 1004  # 最新保留


class TestRebuildNoFifo:
    def test_set_state_no_maxlen(self):
        """快照恢复不再引入 maxlen FIFO 语义。"""
        m1 = MemoryManager(num_nodes=16, episodic_capacity=20,
                           episodic_hard_cap=100)
        _fill(m1, 25)
        state = m1.get_state()
        m2 = MemoryManager(num_nodes=16, episodic_capacity=20,
                           episodic_hard_cap=100)
        m2.set_state(state)
        assert m2._episodic_buffer.maxlen is None
        # 25 条全部保留（软容量 20 只是告警线，不丢）
        assert m2.episodic_size() == 25
        assert m2.capacity_full_events >= 1

    def test_replace_episodic_buffer_no_fifo(self):
        """重建缓冲区不丢软容量内条目（返回值为净变化量）。"""
        m = MemoryManager(num_nodes=16, episodic_capacity=20,
                          episodic_hard_cap=100)
        _fill(m, 10)
        entries = list(m.iter_episodic())
        removed = m.replace_episodic_buffer(entries)
        assert removed == 0
        assert m.episodic_size() == 10
        assert m._episodic_buffer.maxlen is None

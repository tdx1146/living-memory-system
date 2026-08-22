# -*- coding: utf-8 -*-
"""迁移测试：EpisodicEntry 扩展字段（提取层 v1.4 S1-11，M7/M8 锁死项）

覆盖：
  1. 旧快照条目（无 last_reinforced_turn/info_value/core/ts/gray 字段）加载 →
     getattr 兜底默认值（不崩溃）
  2. store_episodic 初始化新字段：last_reinforced_turn=写入 turn、ts=时间戳、
     info_value/core/gray 透传
  3. 写侧引用加固：recall_episodic(reinforce_turn=…) → 命中条目
     reference_count+1 且 last_reinforced_turn 刷新（P2-A/P2-B：复用
     reference_count 为加固计数唯一权威，无 link_count）
  4. 快照 get_state/set_state 往返：混合新旧条目不崩溃
"""

import time
import torch

from core.hippocampus.memory import MemoryManager, EpisodicEntry


def make_old_style_entry(text="旧条目", turn=5, vec_dim=8):
    """模拟旧快照条目：仅含 v1.3 之前字段（无提取层 v1.4 新增字段）。"""
    e = EpisodicEntry.__new__(EpisodicEntry)
    e.text = text
    e.semantic_vector = torch.zeros(vec_dim)
    e.surprise = 0.5
    e.turn = turn
    e.source = 'external'
    e.confidence = 1.0
    e.rebuttal_count = 0
    e.reference_count = 0
    e.source_trust = 1.0
    e.labile = False
    e.labile_since = None
    e.violated_by = None
    e.last_recalled_at = None
    e.recall_count = 0
    return e


class TestOldSnapshotCompatibility:
    """旧快照条目缺新字段 → getattr 兜底默认值（M7 迁移测试）。"""

    def test_getattr_defaults_for_old_entries(self):
        e = make_old_style_entry()
        # 新字段全部走 getattr 兜底（模拟旧快照 load 后代码路径）
        assert getattr(e, 'last_reinforced_turn', None) is None
        assert getattr(e, 'info_value', 0.0) == 0.0
        assert getattr(e, 'core', None) is None
        assert getattr(e, 'ts', None) is None
        assert getattr(e, 'gray', False) is False
        # 既有字段不受影响
        assert e.turn == 5
        assert e.reference_count == 0

    def test_old_entries_in_buffer_set_state_roundtrip(self):
        """快照往返：混合新旧条目恢复不崩溃，新字段保持。"""
        m = MemoryManager(num_nodes=16)
        old = make_old_style_entry(text="旧", turn=1)
        m._episodic_buffer.append(old)
        new_vec = torch.randn(8)
        m.store_episodic("新", new_vec, surprise=1.0, turn=2,
                         info_value=0.8, core="核心文本", gray=False)
        state = m.get_state()
        assert len(state['episodic_buffer']) == 2

        m2 = MemoryManager(num_nodes=16)
        m2.set_state(state)
        entries = list(m2.iter_episodic())
        by_turn = {e.turn: e for e in entries}
        # 旧条目缺字段 → getattr 兜底
        assert getattr(by_turn[1], 'gray', False) is False
        # 新条目字段保留
        assert by_turn[2].last_reinforced_turn == 2
        assert by_turn[2].info_value == 0.8
        assert by_turn[2].core == "核心文本"
        assert by_turn[2].ts is not None


class TestNewFieldInitialization:
    """store_episodic 初始化新字段（S1-11）。"""

    def test_store_episodic_initializes_fields(self):
        m = MemoryManager(num_nodes=16)
        vec = torch.randn(8)
        m.store_episodic("测试文本", vec, surprise=2.0, turn=7,
                         info_value=0.66, core="压缩核心", gray=True)
        e = list(m.iter_episodic())[-1]
        assert e.last_reinforced_turn == 7          # wear 计时起点 = 写入 turn
        assert e.info_value == 0.66
        assert e.core == "压缩核心"
        assert e.gray is True
        assert e.ts is not None
        assert isinstance(e.ts, float)

    def test_store_episodic_defaults(self):
        m = MemoryManager(num_nodes=16)
        m.store_episodic("默认", torch.randn(8), surprise=0.1, turn=1)
        e = list(m.iter_episodic())[-1]
        assert e.last_reinforced_turn == 1
        assert e.info_value == 0.0
        assert e.core is None
        assert e.gray is False


class TestWriteSideReinforcement:
    """写侧引用加固（P2-B）：reference_count+1 且 last_reinforced_turn 刷新。"""

    def test_reinforce_turn_refreshes_last_reinforced_turn(self):
        m = MemoryManager(num_nodes=16)
        vec_a = torch.randn(8)
        vec_b = torch.randn(8)
        m.store_episodic("条目A", vec_a, surprise=1.0, turn=10)
        m.store_episodic("条目B", vec_b, surprise=1.0, turn=11)
        # 用与 A 相同的向量做写侧引用匹配（新条目向量）；top_k=1 只返回 A
        hits = m.recall_episodic(vec_a, top_k=1, reinforce_turn=42)
        assert len(hits) >= 1
        # A 被命中：reference_count+1 且 last_reinforced_turn 刷新为 42
        a = [e for e in hits if e.text == "条目A"][0]
        assert a.reference_count >= 1
        assert a.last_reinforced_turn == 42
        # [A3] issue A3 修复：先截取 top_k 后加固——只有真正进入 top_k 的条目
        # 被计数/刷新。旧断言编码"count_reference 对全部得分条目计数"的 bug
        # 语义（每轮给整批相似命中、含从未进 LLM context 的条目虚增引用/
        # 置信度/磨损计时）——B 不在 top-1，不再虚增，保持入库原值。
        b = [e for e in m.iter_episodic() if e.text == "条目B"][0]
        assert b.reference_count == 0
        assert b.last_reinforced_turn == 11  # 入库 turn，未被刷新

    def test_reinforce_turn_none_no_refresh(self):
        """reinforce_turn=None（默认）不刷新 last_reinforced_turn（向后兼容）。"""
        m = MemoryManager(num_nodes=16)
        vec = torch.randn(8)
        m.store_episodic("条目", vec, surprise=1.0, turn=10)
        m.recall_episodic(vec, top_k=3)  # reinforce_turn 默认 None
        e = list(m.iter_episodic())[-1]
        assert e.last_reinforced_turn == 10

    def test_no_link_count_field(self):
        """P2-A 定案：不新增 link_count，reference_count 是加固计数唯一权威。"""
        assert not hasattr(EpisodicEntry, 'link_count')
        m = MemoryManager(num_nodes=16)
        m.store_episodic("条目", torch.randn(8), surprise=1.0, turn=1)
        e = list(m.iter_episodic())[-1]
        assert hasattr(e, 'reference_count')

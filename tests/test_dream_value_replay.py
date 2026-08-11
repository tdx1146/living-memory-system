# -*- coding: utf-8 -*-
"""做梦价值重放 + 磨损判据测试（提取层 v1.4 S1-13，D1 丰碑）

覆盖：
  1. D1 反例证明（验证④）：100 轮前高价值条目被重放（时间无关）——
     白盒构造：EpisodicEntry(turn=任意值) → memory._episodic_buffer.append()
     （附 #2 定案：不新增生产接口）
  2. 旧条目在 100 轮后仍存在（wear_threshold=∞，无按轮数修剪）
  3. 重放命中 → reference_count+1 ＋ last_reinforced_turn 刷新（加固事件 1）
  4. gray 条目不参与重放（三重冻结①）
  5. wear_stats 观测（只观测不删除）
"""

import random

import pytest
import torch

from core.hippocampus.attractor import AttractorNetwork
from core.hippocampus.purpose import PurposeLayer
from core.hippocampus.memory import MemoryManager, EpisodicEntry
from core.hippocampus.dream_engine import DreamEngine
from core.sensory.embedder import SimpleEmbedder


@pytest.fixture
def engine(tmp_path):
    """小规模 DreamEngine（与 test_dream_engine 同构）。"""
    config = {
        'num_nodes': 32, 'input_dim': 8,
        'idle_learning_rate': 0.001, 'idle_orth_weight': 1.0,
        'idle_temperature': 0.1, 'idle_num_steps': 5,
        'consolidation_ratio': 0.7, 'collapse_threshold': 0.9,
        'max_idle_steps': 50, 'snapshot_dir': str(tmp_path),
    }
    torch.manual_seed(42)
    attractor = AttractorNetwork(32, 8, seed=42)
    purpose = PurposeLayer(8)
    memory = MemoryManager(32)
    embedder = SimpleEmbedder(8)
    return DreamEngine(attractor, purpose, memory, embedder, config)


def make_entry(text, surprise, turn, info_value=0.0, source='external',
               vec=None):
    e = EpisodicEntry(
        text=text,
        semantic_vector=(vec if vec is not None else torch.zeros(8)),
        surprise=surprise,
        turn=turn,
        source=source,
    )
    e.info_value = info_value
    return e


def fill_buffer(engine, n=3):
    """填充激活态缓冲（dream_mvp 前置：buffer 空会 early return）。"""
    torch.manual_seed(7)
    for i in range(n):
        state = torch.randn(engine.num_nodes) * 0.5
        from core.types import Activation
        engine.memory.update(Activation(state=state, entropy=1.0,
                                        surprise=float(i + 1) * 0.3),
                             surprise=float(i + 1) * 0.3)


class TestD1AntiExample:
    def test_old_high_value_entry_replayed(self, engine, tmp_path):
        """D1 反例证明：100 轮前高价值条目被重放（时间无关）。"""
        random.seed(42)
        old = make_entry("100轮前的核心记忆", surprise=2.0, turn=1,
                         info_value=1.0)
        for i in range(5):
            engine.memory._episodic_buffer.append(
                make_entry(f"今天的低价值{i}", surprise=0.01,
                           turn=101 + i, info_value=0.01))
        engine.memory._episodic_buffer.append(old)
        fill_buffer(engine)

        result = engine.dream_mvp(n_steps=5)
        vr = result.get('value_replay', {})
        ids = vr.get('replay_set_ids', [])
        # 旧条目（turn=1）进入重放集——与 101 轮后的新条目同池竞争
        assert old.turn in ids, f"旧条目未被重放：replay_set_ids={ids}"
        # 被采中 → 加固（reference_count+1 且 last_reinforced_turn 刷新）
        assert old.reference_count >= 1
        assert old.last_reinforced_turn == 101 + 4, (
            f"last_reinforced_turn={old.last_reinforced_turn}")

    def test_old_entry_survives_after_100_turns(self, engine):
        """旧条目在 100 轮后仍存在（wear_threshold=∞，无按轮数修剪）。"""
        engine.memory._episodic_buffer.append(
            make_entry("童年记忆", surprise=0.5, turn=1, info_value=0.5))
        for i in range(5):
            engine.memory._episodic_buffer.append(
                make_entry(f"新条目{i}", surprise=0.5, turn=101 + i,
                           info_value=0.5))
        engine._forgetting_pruning()
        turns = [e.turn for e in engine.memory.iter_episodic()]
        assert 1 in turns, "100 轮前的旧条目被修剪（D1 丰碑哲学失效）"
        assert len(turns) == 6


class TestReinforcement:
    def test_replay_hit_reinforces(self, engine):
        """加固事件 1：重放命中 → reference_count+1 + last_reinforced_turn。"""
        e1 = make_entry("条目1", surprise=1.0, turn=10, info_value=1.0)
        e2 = make_entry("条目2", surprise=1.0, turn=11, info_value=1.0)
        engine.memory._episodic_buffer.append(e1)
        engine.memory._episodic_buffer.append(e2)
        vr = engine._value_replay(k=2)
        # k=2 ≥ 候选数 → 全部被采中
        assert sorted(vr['reinforced_ids']) == sorted([10, 11])
        assert e1.reference_count >= 1
        assert e1.last_reinforced_turn == 11  # 当前最大 turn
        assert e2.last_reinforced_turn == 11

    def test_gray_excluded_from_value_replay(self, engine):
        """灰度三重冻结①：gray 条目不参与重放/加固。"""
        normal = make_entry("正式", surprise=1.0, turn=1, info_value=1.0)
        gray = make_entry("灰度", surprise=1.0, turn=2, info_value=1.0,
                          source='store_gray')
        engine.memory._episodic_buffer.append(normal)
        engine.memory._episodic_buffer.append(gray)
        vr = engine._value_replay(k=5)
        ids = vr['replay_set_ids']
        assert gray.turn not in ids
        assert gray.reference_count == 0
        assert normal.turn in ids

    def test_cluster_failopen_and_representative(self, engine):
        """聚类 fail-open＋代表幸存加固（加固事件 3）。"""
        vec_a = torch.randn(8)
        e1 = make_entry("同类1", surprise=1.0, turn=1, info_value=1.0,
                        vec=vec_a)
        e2 = make_entry("同类2", surprise=0.9, turn=2, info_value=0.9,
                        vec=vec_a)  # 与 e1 相同向量 → 余弦相似度 1.0
        e3 = make_entry("异类3", surprise=1.0, turn=3, info_value=1.0,
                        vec=torch.randn(8))
        for e in (e1, e2, e3):
            engine.memory._episodic_buffer.append(e)
        vr = engine._value_replay(k=3)
        # e1/e2 归一组（相似），代表=e1（分数更高）被额外加固
        assert e1.reference_count >= 2, (
            f"代表幸存加固缺失: e1.reference_count={e1.reference_count}")
        assert e3.reference_count >= 1


class TestWearObservability:
    def test_wear_stats_in_result(self, engine):
        """wear_stats 观测（只观测不删除，远期淘汰分级预埋）。"""
        engine.memory._episodic_buffer.append(
            make_entry("旧", surprise=0.5, turn=1, info_value=0.5))
        engine.memory._episodic_buffer.append(
            make_entry("新", surprise=0.5, turn=50, info_value=0.5))
        fill_buffer(engine)
        result = engine.dream_mvp(n_steps=3)
        vr = result.get('value_replay', {})
        ws = vr.get('wear_stats', {})
        assert ws.get('count') == 2
        assert ws.get('max', 0) > ws.get('min', 0)  # 旧条目磨损更大

    def test_effective_wear_reinforcement_resets(self):
        """加固后重新计时：reference_count 高/加固新 → 磨损慢。"""
        engine = engine_factory()
        e_old = make_entry("被加固的旧条目", surprise=0.5, turn=1,
                           info_value=0.5)
        e_old.reference_count = 5
        e_old.last_reinforced_turn = 40  # 最近被加固
        e_bare = make_entry("无链接旧条目", surprise=0.5, turn=1,
                            info_value=0.5)
        engine.memory._episodic_buffer.append(e_old)
        engine.memory._episodic_buffer.append(e_bare)
        w_old = engine._effective_wear(e_old, 100)
        w_bare = engine._effective_wear(e_bare, 100)
        # 被链接（reference_count=5 → 因子 1-0.5=0.5）且加固后重计时
        # （60 轮）→ 磨损显著小于无链接（99 轮 × 因子 1.0）
        assert w_old < w_bare


def engine_factory():
    """无 fixture 版本（供非 fixture 测试用）。"""
    config = {
        'num_nodes': 32, 'input_dim': 8,
        'idle_learning_rate': 0.001, 'idle_orth_weight': 1.0,
        'idle_temperature': 0.1, 'idle_num_steps': 5,
        'consolidation_ratio': 0.7, 'collapse_threshold': 0.9,
        'max_idle_steps': 50, 'snapshot_dir': '/tmp/lms-test-dream',
    }
    torch.manual_seed(42)
    attractor = AttractorNetwork(32, 8, seed=42)
    purpose = PurposeLayer(8)
    memory = MemoryManager(32)
    embedder = SimpleEmbedder(8)
    return DreamEngine(attractor, purpose, memory, embedder, config)

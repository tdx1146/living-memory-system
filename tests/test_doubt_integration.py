# -*- coding: utf-8 -*-
"""体验层 D（设计 v1.1 §6-§8）：怀疑融入（专注方向版）测试。

覆盖（§6.7 验收）：
  1. 置信度公式（rebuttal≥2→0.1、三路汇入）
  2. labile 窗口开闭（mark_labile / resolve_labile 三种结局）
  3. 反流畅回放权重（同 surprise 下被反驳条目权重更低）
  4. gap 登记（C 类只入 status，A/B 类进怀疑灯）
  5. 配额门控（低相关条目不占席）
  6. doubt_ingest 结构化摄入（fail-open）
  7. record_reference 钩子（正向佐证）
  8. 做梦 doubt_review 复核统计（reviewed/downgraded/rewritten/flagged）
  9. API：/status doubt 字段、/recall 置信度字段
"""

import os
import sys
import time

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
import torch

from core.doubt.confidence_field import (
    compute_confidence, get_source_trust, is_low_confidence,
    record_reference, mark_rebutted, get_rebuttal_rate,
)
from core.doubt.reconsolidation import (
    detect_destabilization, find_violated_entry, mark_labile, resolve_labile,
)
from core.doubt.recall_scheduler import (
    salience, forgetting, select_with_low_confidence_quota,
)
from core.doubt.gap_registry import GapRegistry
from core.doubt import doubt_ingest
from core.hippocampus.memory import MemoryManager, EpisodicEntry


def _make_entry(text="测试记忆", confidence=1.0, **kw):
    kw.setdefault("source_trust", 1.0)
    kw.setdefault("rebuttal_count", 0)
    kw.setdefault("reference_count", 0)
    kw.setdefault("surprise", 1.0)
    kw.setdefault("turn", 0)
    kw.setdefault("confidence", confidence)
    kw.setdefault("labile", False)
    kw.setdefault("labile_since", None)
    kw.setdefault("violated_by", None)
    kw.setdefault("recall_count", 0)
    return EpisodicEntry(text=text, semantic_vector=torch.zeros(8), **kw)


# ============================================================
# 1. 置信度公式
# ============================================================

class TestConfidenceField:
    def test_basic_formula(self):
        assert compute_confidence(0, 0, 1.0) == 1.0
        assert compute_confidence(1, 0, 1.0) == 0.0  # 1/(0+1)=1 → 1-1=0
        assert compute_confidence(1, 9, 1.0) == pytest.approx(0.9)

    def test_rebuttal_two_forces_01(self):
        """rebuttal≥2 → 强制 0.1（memory_trust 工程先例）。"""
        assert compute_confidence(2, 100, 1.0) == 0.1
        assert compute_confidence(5, 100, 1.0) == 0.1

    def test_source_trust_dimension(self):
        assert get_source_trust('external') == 1.0
        assert get_source_trust('doubt') == 0.5
        assert get_source_trust('unknown_source') == 1.0  # fail-open
        # source_trust 与反驳率相乘
        c = compute_confidence(1, 9, get_source_trust('doubt'))
        assert c == pytest.approx(0.9 * 0.5)

    def test_record_reference_positive_evidence(self):
        e = _make_entry(rebuttal_count=1, reference_count=0)
        record_reference(e)
        assert e.reference_count == 1
        assert e.recall_count == 1
        assert e.last_recalled_at is not None
        # 正向佐证后置信度回升：0.5 → (1-1/2)*1 = 0.5
        e2 = _make_entry(rebuttal_count=1, reference_count=9)
        record_reference(e2)
        assert e2.reference_count == 10
        assert e2.confidence == pytest.approx(1 - 1 / 11)

    def test_mark_rebutted_downgrades(self):
        e = _make_entry(reference_count=9)
        mark_rebutted(e, violated_by="证伪证据")
        assert e.rebuttal_count == 1
        assert e.violated_by == "证伪证据"
        assert e.confidence == pytest.approx(0.9)

    def test_is_low_confidence(self):
        assert is_low_confidence(_make_entry(confidence=0.2))
        assert not is_low_confidence(_make_entry(confidence=0.5))


# ============================================================
# 2. labile 窗口
# ============================================================

class TestReconsolidation:
    def test_detect_destabilization(self):
        window = [1.0] * 20
        assert detect_destabilization(1.0, window) == (False, None)  # σ=0 不判
        assert detect_destabilization(0.0, []) == (False, None)      # 窗口不足
        # 构造 z>2
        win = [1.0, 1.1, 0.9, 1.0] * 5
        flag, z = detect_destabilization(5.0, win)
        assert flag and z is not None and z > 2.0

    def test_mark_labile_sets_fields(self):
        e = _make_entry(reference_count=9)
        assert mark_labile(e, violated_by="违反它的输入")
        assert e.labile is True
        assert e.labile_since is not None
        assert e.rebuttal_count == 1  # 三路汇入②
        assert e.confidence == pytest.approx(0.9)

    def test_resolve_labile_three_outcomes(self):
        # 有证伪证据 → rewritten（复位 labile）
        e1 = _make_entry(labile=True, violated_by="证据")
        e1.labile_since = time.time()
        assert resolve_labile(e1) == 'rewritten'
        assert e1.labile is False
        # 窗口内无证据 → kept（×1.02 重巩固）
        e2 = _make_entry(labile=True, confidence=0.5)
        e2.labile_since = time.time()
        assert resolve_labile(e2) == 'kept'
        assert e2.confidence == pytest.approx(0.5 * 1.02)
        # 超时无证据 → downgraded（×0.98 折损）
        e3 = _make_entry(labile=True, confidence=0.5)
        e3.labile_since = time.time() - 999999
        assert resolve_labile(e3) == 'downgraded'
        assert e3.confidence == pytest.approx(0.5 * 0.98)

    def test_find_violated_entry_picks_highest_confidence(self):
        e_low = _make_entry(confidence=0.3)
        e_high = _make_entry(confidence=0.9)
        scored = [(0.8, e_low), (0.7, e_high)]
        assert find_violated_entry(scored) is e_high


# ============================================================
# 3. 反流畅回放权重
# ============================================================

class TestAntiFluency:
    def test_rebutted_entry_lower_replay_weight(self):
        """反流畅乘子：rebuttal_rate 高但置信仍 ≥0.3 的条目被降权但未剔除。"""
        mm = MemoryManager(num_nodes=8, replay_weight=0.01,
                           transfer_rate=0.0)  # 关 transfer，隔离回放权重
        # 正交基：投影可分离各条目自身贡献
        state_a = torch.zeros(8); state_a[0] = 1.0
        state_b = torch.zeros(8); state_b[1] = 1.0
        mm.update(_ActivationStub(state_a), surprise=10.0, turn=1)
        mm.update(_ActivationStub(state_b), surprise=10.0, turn=2)
        # turn1 被反驳：rebuttal=7, ref=10 → rate=7/11≈0.636 →
        # conf=(1−0.636)×1.0≈0.364 ≥0.3 → 仍进回放池但权重 ×0.364
        e1 = _make_entry(text="被反驳的记忆", turn=1, rebuttal_count=7,
                         reference_count=10, confidence=0.364)
        e2 = _make_entry(text="干净记忆", turn=2, rebuttal_count=0,
                         reference_count=0, confidence=1.0)
        mm._episodic_buffer.append(e1)
        mm._episodic_buffer.append(e2)
        before = mm.long_term_latent.clone()
        mm.consolidate()
        after = mm.long_term_latent
        # 正交基投影 = 各条目自身回放贡献
        contrib_clean = float(after[1] - before[1])
        contrib_rebutted = float(after[0] - before[0])
        assert contrib_clean > contrib_rebutted * 2  # 反流畅乘子生效
        assert contrib_clean == pytest.approx(0.1, abs=1e-6)
        assert contrib_rebutted == pytest.approx(
            0.1 * (1 - 7 / 11) * 1.0, abs=1e-6)

    def test_low_confidence_excluded_from_replay(self):
        """低置信（<0.3）条目不进放大回放池（改由 doubt_review 复核）。"""
        mm = MemoryManager(num_nodes=8, replay_weight=0.01,
                           transfer_rate=0.0)
        state = torch.zeros(8); state[0] = 1.0
        mm.update(_ActivationStub(state), surprise=10.0, turn=1)
        e = _make_entry(text="低置信", turn=1, rebuttal_count=2,
                        reference_count=0, confidence=0.1)
        mm._episodic_buffer.append(e)
        before = mm.long_term_latent.clone()
        mm.consolidate()
        after = mm.long_term_latent
        # 低置信（conf=0.1 < 0.3）不进放大回放池
        assert torch.allclose(after, before, atol=1e-9)


class _ActivationStub:
    """update() 只读 activation.state 的最小替身。"""
    def __init__(self, state):
        self.state = state


class _FakeEmbedder:
    """带 embed_text/embed_text_raw 的假嵌入器（PretrainedEmbedder 形态）。"""
    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed_text(self, text: str):
        # 确定性哈希向量：相同文本同向量，可检索
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        v = torch.zeros(self.dim)
        for i in range(self.dim):
            v[i] = (h[i % len(h)] / 255.0) * 2 - 1
        return v

    embed_text_raw = embed_text


# ============================================================
# 4. gap 登记（C 类只入 status，A/B 类进怀疑灯）
# ============================================================

class TestGapRegistry:
    def test_ab_classes_in_lamp_c_class_diagnostic_only(self):
        g = GapRegistry()
        g.register_fok_unresolved("某个未决问题")
        g.register_low_confidence(_make_entry(text="低置信记忆", confidence=0.2))
        g.register_explore_dims([3, 7])
        lamp = g.doubt_lamp()
        # 怀疑灯只有 A/B 类
        assert lamp["fok_unresolved_count"] == 1
        assert lamp["low_confidence_unreviewed_count"] == 1
        assert "explore_dims" not in lamp  # C 类不进灯
        # /status doubt.gaps 含 C 类（诊断保留）
        snap = g.snapshot()
        assert len(snap["explore_dims"]) == 1

    def test_review_clears_b_class(self):
        g = GapRegistry()
        g.register_low_confidence(_make_entry(text="待复核", confidence=0.2))
        assert g.snapshot()["low_confidence_unreviewed"] != []
        g.mark_review({"reviewed": 5, "flagged": 1})
        assert g.snapshot()["low_confidence_unreviewed"] == []
        assert g.last_review() is not None
        assert g.review_stats()["reviewed"] == 5


# ============================================================
# 5. 配额门控
# ============================================================

class TestRecallQuota:
    def test_low_relevance_entry_does_not_occupy_seat(self):
        """低相关（cue_sim < top1×0.5）的低置信条目不占席（宁缺毋滥）。"""
        top = _make_entry(text="top1", confidence=1.0)
        low_conf_far = _make_entry(text="低置信但无关", confidence=0.2)
        scored = [(0.9, top), (0.8, _make_entry(confidence=1.0)),
                  (0.7, _make_entry(confidence=1.0)),
                  (0.3, low_conf_far)]  # 0.3 < 0.45 = 0.9×0.5
        out = select_with_low_confidence_quota(scored, k=3)
        assert out == scored[:3]  # 不加塞

    def test_relevant_low_conf_occupies_one_seat(self):
        """相关（cue_sim ≥ top1×0.5）的低置信条目占 1 席。"""
        top = _make_entry(text="top1", confidence=1.0)
        low_conf_near = _make_entry(text="低置信但相关", confidence=0.2)
        scored = [(0.9, top), (0.85, _make_entry(confidence=1.0)),
                  (0.8, _make_entry(confidence=1.0)),
                  (0.6, low_conf_near)]  # 0.6 ≥ 0.45
        out = select_with_low_confidence_quota(scored, k=3)
        assert len(out) == 3
        assert low_conf_near in [e for _, e in out]

    def test_salience_function(self):
        assert salience(1.0, 1.0, 1.0) == pytest.approx(1.0)
        assert salience(0.0, 1.0, 1.0) == pytest.approx(0.4)  # α 主导
        e = _make_entry(last_recalled_at=time.time(), recall_count=1)
        assert 0.0 <= forgetting(e) <= 1.0


# ============================================================
# 6. doubt_ingest 结构化摄入
# ============================================================

class TestDoubtIngest:
    def test_parse_protocol(self):
        assert doubt_ingest.parse_doubt_event("[doubt] fok: 某个未决问题") == {
            "kind": "fok", "content": "某个未决问题"}
        assert doubt_ingest.parse_doubt_event("[doubt] conflict: 记忆被反驳") == {
            "kind": "conflict", "content": "记忆被反驳"}
        assert doubt_ingest.parse_doubt_event("普通文本") is None  # 无前缀
        assert doubt_ingest.is_doubt_event("[doubt] fok: x") is True
        assert doubt_ingest.is_doubt_event("普通文本") is False

    def test_ingest_fail_open(self):
        """无前缀/异常 → 不崩溃；正常文本返回 None。"""
        assert doubt_ingest.ingest(None, "普通塑形文本") is None
        assert doubt_ingest.ingest(None, "[doubt] fok: 问题") is not None

    def test_ingest_conflict_marks_entry(self):
        mm = MemoryManager(num_nodes=8)
        mm.store_episodic("用户: 我喜欢蓝色小橘猫", torch.zeros(8),
                          surprise=1.0, turn=0)
        class _Loop:
            memory = mm
            gap_registry = GapRegistry()
        ev = doubt_ingest.ingest(_Loop(), "[doubt] conflict: 我喜欢蓝色小橘猫")
        assert ev is not None and ev["kind"] == "conflict"
        entries = list(mm.iter_episodic())
        assert entries[0].rebuttal_count == 1
        assert entries[0].labile is True
        assert _Loop().gap_registry.snapshot()["fok_unresolved"] != []


# ============================================================
# 7. record_reference 钩子（memory 层）
# ============================================================

class TestRecordReferenceHook:
    def test_recall_counts_reference(self):
        mm = MemoryManager(num_nodes=8)
        mm.store_episodic("用户: 记忆A", torch.ones(8), surprise=1.0, turn=0)
        mm.recall_episodic_scored(torch.ones(8), top_k=1, source_filter=None)
        e = list(mm.iter_episodic())[0]
        assert e.reference_count == 1

    def test_recall_no_reference_when_disabled(self):
        """外部只读探针（/react、/recall）传 record_reference=False 零持久化。"""
        mm = MemoryManager(num_nodes=8)
        mm.store_episodic("用户: 记忆B", torch.ones(8), surprise=1.0, turn=0)
        mm.recall_episodic_scored(torch.ones(8), top_k=1,
                                  source_filter=None, count_reference=False)
        e = list(mm.iter_episodic())[0]
        assert e.reference_count == 0


# ============================================================
# 8. 做梦 doubt_review
# ============================================================

class TestDreamDoubtReview:
    def test_doubt_review_stats_and_labile_clearing(self):
        from core.hippocampus.dream_engine import DreamEngine
        from core.hippocampus.attractor import AttractorNetwork
        from core.hippocampus.purpose import PurposeLayer

        attractor = AttractorNetwork(num_nodes=8, input_dim=4, seed=1,
                                     temperature=0.05)
        purpose = PurposeLayer(input_dim=4)
        mm = MemoryManager(num_nodes=8, buffer_capacity=20, episodic_capacity=50)
        # 缓冲一条记忆（做梦不空转）
        state = torch.randn(8)
        mm.update(_ActivationStub(state), surprise=5.0, turn=1)
        # labile 条目（有证据 → rewritten）+ 低置信条目（reviewed）
        e1 = _make_entry(text="被证伪的记忆", turn=1, labile=True,
                         violated_by="矛盾证据", confidence=0.9)
        e2 = _make_entry(text="低置信记忆", turn=2, confidence=0.2)
        mm._episodic_buffer.append(e1)
        mm._episodic_buffer.append(e2)
        # 高频记忆（反教条抽查 → flagged）
        e3 = _make_entry(text="高频记忆", turn=3, reference_count=10,
                         recall_count=10, confidence=0.9)
        mm._episodic_buffer.append(e3)

        de = DreamEngine(attractor=attractor, purpose=purpose, memory=mm,
                         embedder=None, config={"snapshot_dir": "/tmp/dtest"})
        de._doubt_review()
        stats = de.last_doubt_review
        assert stats["rewritten"] >= 1
        assert stats["reviewed"] >= 1
        assert stats["flagged"] >= 1
        # labile 已裁决（复位）
        assert all(not getattr(e, "labile", False) for e in mm.iter_episodic())
        # 改写条目（source='doubt' supersedes）已落库
        sources = [e.source for e in mm.iter_episodic()]
        assert "doubt" in sources

    def test_phase_weights_include_doubt_review(self):
        from core.hippocampus.dream_engine import DreamEngine
        from core.hippocampus.attractor import AttractorNetwork
        from core.hippocampus.purpose import PurposeLayer
        attractor = AttractorNetwork(num_nodes=8, input_dim=4, seed=1,
                                     temperature=0.05)
        purpose = PurposeLayer(input_dim=4)
        mm = MemoryManager(num_nodes=8)
        de = DreamEngine(attractor=attractor, purpose=purpose, memory=mm,
                         embedder=None, config={"snapshot_dir": "/tmp/dtest"})
        assert de.phase_weights["doubt_review"] == 0.10
        assert de.phase_weights["nrem_consolidation"] == 0.30
        assert abs(sum(de.phase_weights.values()) - 1.0) < 1e-6


# ============================================================
# 9. API：/status doubt + /recall 置信度字段
# ============================================================

class TestDoubtApi:
    def test_status_doubt_fields_via_loop(self, monkeypatch, tmp_path):
        from runtime.config import default_config
        from core.sensory.embedder import SimpleEmbedder
        from runtime.loop import LivingMemoryLoop
        cfg = default_config()
        cfg.update(num_nodes=32, input_dim=16, num_infer_steps=5,
                   embedder=SimpleEmbedder(dim=16), auto_snapshot=False,
                   snapshot_dir=str(tmp_path / "s"))
        loop = LivingMemoryLoop(cfg)
        loop.process_turn("用户: 体验层D测试")
        status = loop.get_status()
        assert "doubt" in status
        assert "gaps" in status["doubt"]
        assert "labile_count" in status["doubt"]
        assert "low_confidence_count" in status["doubt"]

    def test_recall_annotates_confidence(self, monkeypatch, tmp_path):
        from runtime.config import default_config
        from runtime.loop import LivingMemoryLoop
        cfg = default_config()
        cfg.update(num_nodes=32, input_dim=16, num_infer_steps=5,
                   embedder=_FakeEmbedder(16), auto_snapshot=False,
                   snapshot_dir=str(tmp_path / "s"),
                   archive_enabled=False)  # 关归档，纯内存路径
        loop = LivingMemoryLoop(cfg)
        loop.process_turn("用户: 我喜欢蓝色的小橘猫")
        results = loop.recall_episodic_readonly("蓝色的小橘猫", k=3)
        assert results  # 至少一条
        item = results[0]
        assert "confidence" in item
        assert "rebuttal_count" in item
        assert "labile" in item
        assert "source_trust" in item
        # 只读：调用后 reference_count 不增（count_reference=False）
        e = list(loop.memory.iter_episodic())[0]
        assert e.reference_count == 0

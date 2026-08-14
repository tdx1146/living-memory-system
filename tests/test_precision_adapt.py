# -*- coding: utf-8 -*-
"""阶段 3（precision 三层动态化，质疑自动校准）测试。

覆盖（任务书 §E 验收 + §C 四机制）：
  1. 治理开关（LMS_PRECISION_ADAPT 默认开/可关；关=零参与）
  2. 场景①：波动性↑（惊讶序列起伏大）→ 全局 precision 下降
     （doubt_baseline↑ / global_precision↓，HGF 波动性调制）
  3. 场景②：同主题多记忆一致 → 条目置信度上升（Koriat 自一致性）；
     簇内分歧 → 置信度下调
  4. 场景③：重复曝光 → 回放权重被降（非升，反流畅性偏误对抗项）；
     verdict_confidence 反流畅折扣
  5. conformal 分位阈值：校准样本累积后怀疑线随经验分布动（非固定值）；
     冷启动回退初始线 0.3
  6. 对称性约束：坏消息 PE 不被低估（负性惊讶 ≥ 正性中位）；负性证据
     比例推高怀疑基线
  7. 域级聚合（compute_domain_precision）+ 域 suspicion 分数秩
  8. 接线观测：/status precision_adapt、/react reaction.doubt、
     /recall 条目 adaptive_confidence/doubt_verdict 字段（开关关 → 空）

测试约定：本文件所有"自适应路径"测试显式 monkeypatch 启用开关
（conftest 已把套件默认置 0 保旧行为）。
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

from core.doubt.precision_adapt import (
    PrecisionAdaptState, compute_consistency, compute_domain_precision,
    fractional_rank, percentile, precision_adapt_enabled,
)
from core.doubt.confidence_field import (
    compute_confidence, mark_rebutted,
)
from core.hippocampus.memory import MemoryManager, EpisodicEntry


@pytest.fixture(autouse=True)
def _precision_adapt_on(monkeypatch):
    """本文件默认启用 LMS_PRECISION_ADAPT=1（conftest 套件默认 0 保旧行为）。

    测试开关关闭路径的用例在测试体内显式 monkeypatch 置 0（覆盖本 fixture）。
    """
    monkeypatch.setenv("LMS_PRECISION_ADAPT", "1")


def _make_entry(text="测试记忆", confidence=1.0, dim=8, **kw):
    kw.setdefault("source_trust", 1.0)
    kw.setdefault("rebuttal_count", 0)
    kw.setdefault("reference_count", 0)
    kw.setdefault("recall_count", 0)
    kw.setdefault("surprise", 1.0)
    kw.setdefault("turn", 0)
    kw.setdefault("confidence", confidence)
    kw.setdefault("labile", False)
    kw.setdefault("labile_since", None)
    kw.setdefault("violated_by", None)
    return EpisodicEntry(
        text=text, semantic_vector=torch.zeros(dim), **kw)


class _ActivationStub:
    """update() 只读 activation.state 的最小替身。"""
    def __init__(self, state):
        self.state = state


# ============================================================
# 1. 治理开关
# ============================================================

class TestGovernanceSwitch:
    def test_default_on(self, monkeypatch):
        """LMS_PRECISION_ADAPT 默认开（任务书：默认开，可回滚关）。"""
        monkeypatch.delenv("LMS_PRECISION_ADAPT", raising=False)
        assert precision_adapt_enabled(None) is True

    def test_explicit_off(self, monkeypatch):
        monkeypatch.setenv("LMS_PRECISION_ADAPT", "0")
        assert precision_adapt_enabled(None) is False
        assert precision_adapt_enabled(True) is True  # 显式参数优先

    def test_disabled_state_zero_participation(self, monkeypatch):
        """开关关 → 所有方法 no-op / 中性值（零参与，回滚路径）。"""
        monkeypatch.setenv("LMS_PRECISION_ADAPT", "0")
        pa = PrecisionAdaptState()
        assert pa.enabled is False
        pa.observe_surprise(10.0, is_negative=True)  # 不崩、不记
        assert len(pa.surprise_history) == 0
        assert pa.doubt_baseline() == 0.5
        assert pa.global_precision() == 0.5
        assert pa.doubt_threshold() == 0.3  # 初始线（判定由旧路径负责）
        # 无折扣：判定置信度 = 基础公式（reference=10 不触发反流畅折扣）
        e = _make_entry(reference_count=10)
        assert pa.verdict_confidence(e) == pytest.approx(
            compute_confidence(0, 10, 1.0))
        assert pa.should_doubt(e) is False
        assert pa.snapshot() == {'enabled': False}

    def test_disabled_constructor_reads_env(self, monkeypatch):
        monkeypatch.setenv("LMS_PRECISION_ADAPT", "0")
        pa = PrecisionAdaptState()
        assert pa.enabled is False


# ============================================================
# 2. 分位工具（零固定阈值基础件）
# ============================================================

class TestQuantileTools:
    def test_percentile(self):
        assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
        assert percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
        assert percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
        with pytest.raises(ValueError):
            percentile([], 0.5)

    def test_fractional_rank_stable_is_neutral(self):
        """稳定序列 → 0.5 中性（不误报高度怀疑）。"""
        assert fractional_rank([1.0] * 10, 1.0) == 0.5
        assert fractional_rank([], 1.0) == 0.5

    def test_fractional_rank_orders(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert fractional_rank(vals, 5.0) > 0.8
        assert fractional_rank(vals, 1.0) < 0.2
        assert fractional_rank(vals, 3.0) == 0.5


# ============================================================
# 3. 场景①：HGF 波动性 → 全局 precision 下降
# ============================================================

class TestHGFVolatility:
    def _feed(self, pa, surprises, negatives=None):
        negatives = negatives or [False] * len(surprises)
        for s, neg in zip(surprises, negatives):
            pa.observe_surprise(s, is_negative=neg)

    def test_volatility_rise_raises_baseline(self):
        """波动↑ → doubt_baseline↑ / global_precision↓（场景①方向）。"""
        pa = PrecisionAdaptState(min_samples=20)
        # 先喂 40 轮稳定惊讶（波动≈0 → rank 中性）
        self._feed(pa, [5.0] * 40)
        baseline_calm = pa.doubt_baseline()
        # 再喂 40 轮剧烈波动惊讶（高方差）→ 波动性↑ → 怀疑↑
        wild = [5.0 + (20.0 if i % 2 else -20.0) for i in range(40)]
        self._feed(pa, wild)
        baseline_wild = pa.doubt_baseline()
        assert baseline_wild > baseline_calm
        # 全局 precision（信任基线）= 1 − doubt_baseline，随波动下降
        assert pa.global_precision() < 1.0 - baseline_calm
        assert pa.volatility is not None and pa.volatility > 0.0

    def test_stable_environment_neutral(self):
        """环境稳定 → baseline 收敛中性 0.5（precision 不虚高不虚低）。"""
        pa = PrecisionAdaptState(min_samples=20)
        self._feed(pa, [5.0] * 60)
        assert 0.3 <= pa.doubt_baseline() <= 0.7

    def test_cold_start_neutral(self):
        pa = PrecisionAdaptState(min_samples=30)
        self._feed(pa, [5.0, 100.0, 5.0, 100.0])  # 样本不足
        assert pa.doubt_baseline() == 0.5
        assert pa.is_cold() is True


# ============================================================
# 4. 场景②：Koriat 自一致性 → 条目置信度上升
# ============================================================

class TestKoriatConsistency:
    def _entry_with_vec(self, text, vec, **kw):
        e = _make_entry(text=text, **kw)
        e.semantic_vector = vec
        return e

    def test_same_topic_cohort_raises_confidence(self):
        """同主题多记忆一致 → 一致性高 → 条目置信度上升（场景②方向）。"""
        # 同主题簇：4 条高度相似向量 + 1 条无关；簇内条目带轻度反驳历史
        # （base=0.9 < 1.0，给一致性留出上升空间）
        base = torch.randn(16)
        cluster = [base + torch.randn(16) * 0.05 for _ in range(4)]
        unrelated = torch.randn(16) * 5.0
        entries = [
            self._entry_with_vec(f"记忆{i}", v,
                                 rebuttal_count=1, reference_count=9)
            for i, v in enumerate(cluster)
        ]
        entries.append(self._entry_with_vec("无关记忆", unrelated))
        scored = [(1.0 - i * 0.01, e) for i, e in enumerate(entries)]

        cons = compute_consistency(scored, top_n=5)
        # 簇内条目应有一致性（≈0.99 附近），孤立条目无
        assert id(entries[0]) in cons
        assert cons[id(entries[0])] > 0.9
        assert id(entries[-1]) not in cons  # 孤立条目（无同簇成员）

        pa = PrecisionAdaptState()
        base_conf = pa.adaptive_confidence(entries[0])  # (1-1/10)*1 = 0.9
        conf_with_cons = pa.adaptive_confidence(
            entries[0], cons[id(entries[0])])
        # 同主题互相印证 → 置信度上升（Koriat 2012：一致性 = 置信度）
        assert conf_with_cons > base_conf

    def test_divergence_lowers_confidence(self):
        """簇内分歧（低一致性）→ 置信度下调（Koriat：分歧即怀疑）。"""
        pa = PrecisionAdaptState()
        e = _make_entry(rebuttal_count=0, reference_count=9,
                        confidence=0.9)  # 基础置信度 0.9
        conf_base = pa.adaptive_confidence(e)
        conf_low = pa.adaptive_confidence(e, consistency=0.2)  # 簇内分歧
        assert conf_low < conf_base

    def test_force_downgrade_not_overridden_by_consistency(self):
        """硬证据（rebuttal ≥ 2 强制 0.1）不被一致性覆盖（软信号不推翻
        硬证伪）。"""
        pa = PrecisionAdaptState()
        e = _make_entry(rebuttal_count=2, reference_count=100,
                        confidence=0.1)
        assert pa.adaptive_confidence(e, consistency=0.95) == pytest.approx(0.1)

    def test_mixed_dim_vectors_no_crash(self):
        """混合维度缓冲区（384+64 维）不崩，维度不匹配对跳过。"""
        e1 = self._entry_with_vec("a", torch.randn(384))
        e2 = self._entry_with_vec("b", torch.randn(384))
        e3 = self._entry_with_vec("c", torch.randn(64))  # 旧快照 64 维
        scored = [(0.9, e1), (0.8, e2), (0.7, e3)]
        cons = compute_consistency(scored, top_n=3)  # 不崩
        assert isinstance(cons, dict)


# ============================================================
# 5. 场景③：重复曝光降权（反流畅性偏误对抗项）
# ============================================================

class TestAntiFluency:
    def test_replay_weight_repetition_factor(self, monkeypatch):
        """回放加权对抗项：同 surprise 下，重复曝光条目权重低于干净条目
        （重复≠真；开关关时两者相等=旧行为）。"""
        monkeypatch.setenv("LMS_PRECISION_ADAPT", "1")
        mm = MemoryManager(num_nodes=8, replay_weight=0.01,
                           transfer_rate=0.0)
        state_a = torch.zeros(8); state_a[0] = 1.0
        state_b = torch.zeros(8); state_b[1] = 1.0
        mm.update(_ActivationStub(state_a), surprise=10.0, turn=1)
        mm.update(_ActivationStub(state_b), surprise=10.0, turn=2)
        # 同 surprise、同反驳率（0）；唯一差异：重复曝光次数
        e_repeated = _make_entry(text="重复曝光", turn=1,
                                 reference_count=10, recall_count=10)
        e_clean = _make_entry(text="干净", turn=2,
                              reference_count=0, recall_count=0)
        mm._episodic_buffer.append(e_repeated)
        mm._episodic_buffer.append(e_clean)
        before = mm.long_term_latent.clone()
        mm.consolidate()
        after = mm.long_term_latent
        contrib_repeated = float(after[0] - before[0])
        contrib_clean = float(after[1] - before[1])
        # 重复曝光被降权：贡献 < 干净条目（无对抗项时两者相等）
        assert contrib_clean > contrib_repeated
        import math
        assert contrib_repeated == pytest.approx(
            0.1 / (1.0 + math.log1p(10)) * 1.0, abs=1e-6)
        assert contrib_clean == pytest.approx(0.1, abs=1e-6)

    def test_replay_factor_off_keeps_old_behavior(self, monkeypatch):
        """开关关（conftest 默认）：同 surprise 同反驳率 → 权重相等（旧行为）。"""
        monkeypatch.setenv("LMS_PRECISION_ADAPT", "0")
        mm = MemoryManager(num_nodes=8, replay_weight=0.01,
                           transfer_rate=0.0)
        state_a = torch.zeros(8); state_a[0] = 1.0
        state_b = torch.zeros(8); state_b[1] = 1.0
        mm.update(_ActivationStub(state_a), surprise=10.0, turn=1)
        mm.update(_ActivationStub(state_b), surprise=10.0, turn=2)
        e_repeated = _make_entry(text="重复", turn=1,
                                 reference_count=10, recall_count=10)
        e_clean = _make_entry(text="干净", turn=2)
        mm._episodic_buffer.append(e_repeated)
        mm._episodic_buffer.append(e_clean)
        before = mm.long_term_latent.clone()
        mm.consolidate()
        after = mm.long_term_latent
        assert float(after[0] - before[0]) == pytest.approx(0.1, abs=1e-6)
        assert float(after[1] - before[1]) == pytest.approx(0.1, abs=1e-6)

    def test_verdict_confidence_fluency_discount(self):
        """判定置信度：重复曝光（reference 高）不升信——log1p 折扣。"""
        pa = PrecisionAdaptState()
        e_clean = _make_entry(reference_count=0, confidence=0.8)
        e_repeated = _make_entry(reference_count=10, confidence=0.8)
        v_clean = pa.verdict_confidence(e_clean)
        v_rep = pa.verdict_confidence(e_repeated)
        assert v_rep < v_clean  # 重复≠真：判定置信度被折扣


# ============================================================
# 6. conformal 分位阈值（随经验分布动）
# ============================================================

class TestConformalThreshold:
    def test_cold_returns_initial_line(self):
        pa = PrecisionAdaptState()
        assert pa.doubt_threshold() == 0.3  # 冷启动初始线（允许的初始值）

    def test_threshold_follows_rebuttal_distribution(self):
        """校准样本累积后：怀疑线 = 被反驳条目反驳前置信度分布 P85——
        随校准集漂移（非固定值）。"""
        pa = PrecisionAdaptState(min_calibration_samples=5)
        # 低置信区间被证伪的条目（反驳前置信度都低）→ 线低（怀疑收敛）
        for c in [0.1, 0.12, 0.14, 0.16, 0.18, 0.2]:
            e = _make_entry(confidence=c)
            e.confidence_before_rebuttal = c
            pa.record_rebuttal(e)
        thr_low = pa.doubt_threshold()
        assert thr_low < 0.3  # 校准集低 → 线低于初始线（分布驱动）

        # 高置信区间也被证伪（反驳前置信度高）→ 线上移（怀疑更积极）
        pa2 = PrecisionAdaptState(min_calibration_samples=5)
        for c in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
            e = _make_entry(confidence=c)
            e.confidence_before_rebuttal = c
            pa2.record_rebuttal(e)
        thr_high = pa2.doubt_threshold()
        assert thr_high > 0.5
        assert thr_high != thr_low  # 随经验分布动，不是固定值

    def test_should_doubt_uses_dynamic_line(self):
        pa = PrecisionAdaptState(min_calibration_samples=5)
        for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            e = _make_entry(confidence=c)
            e.confidence_before_rebuttal = c
            pa.record_rebuttal(e)
        # 校准集 P85 ≈ 0.525：置信度 0.3 的条目（rebuttal=7/ref=9 → 0.3）
        # 该被怀疑；置信度 0.9 的条目（rebuttal=1/ref=9）不该
        low = _make_entry(rebuttal_count=7, reference_count=9,
                          confidence=0.3)
        high = _make_entry(rebuttal_count=1, reference_count=9,
                           confidence=0.9)
        assert pa.should_doubt(low)
        assert not pa.should_doubt(high)


# ============================================================
# 7. 对称性约束（坏消息 PE 不被低估）
# ============================================================

class TestSymmetry:
    def test_negative_pe_boosted_to_positive_median(self):
        """坏消息 PE 不被系统性低估：负性惊讶 < 正性中位时抬升到中位。"""
        pa = PrecisionAdaptState()
        for _ in range(20):
            pa.observe_surprise(10.0, is_negative=False)  # 正性中位 = 10
        pa.observe_surprise(1.0, is_negative=True)  # 坏消息但惊讶低
        # 对称性补偿后：进入历史的是抬升后的值（≥ 正性中位）
        assert pa.surprise_history[-1] >= 10.0

    def test_negative_evidence_raises_baseline(self):
        """负性证据比例（近期反驳多）推高怀疑基线（对称性约束的全局层）。
        确定性构造：惊讶恒定 5.0（vol_rank 两者相同=0.5），唯一差异是
        负性证据比例从 0 升到 1（neg_ratio rank 0.5 → 1.0）。"""
        pa = PrecisionAdaptState(min_samples=20)
        for _ in range(90):
            pa.observe_surprise(5.0, is_negative=False)  # 全程干净
        base_clean = pa.doubt_baseline()

        pa2 = PrecisionAdaptState(min_samples=20)
        for _ in range(60):
            pa2.observe_surprise(5.0, is_negative=False)  # 前 60 轮干净
        for _ in range(30):
            # 后 30 轮全是反驳：负性惊讶 5.0 ≥ 正性中位 5.0（无额外抬升），
            # 序列仍恒定 → 唯一变化是 neg_ratio 0 → 1
            pa2.observe_surprise(5.0, is_negative=True)
        assert pa2.doubt_baseline() > base_clean + 0.05


# ============================================================
# 8. 域级聚合
# ============================================================

class TestDomainPrecision:
    def test_domain_aggregation(self):
        e1 = _make_entry(rebuttal_count=2, reference_count=0)
        e2 = _make_entry(rebuttal_count=0, reference_count=10)
        d = compute_domain_precision([e1, e2])
        assert d['rebuttal_rate'] == pytest.approx(round(2 / 11, 4))
        assert 0.0 <= d['confidence'] <= 1.0
        assert d['suspicion'] == 0.5  # 占位（调用方注入全局分布）

    def test_domain_suspicion_fractional_rank(self, monkeypatch):
        """域反驳率在全局分布中的分数秩（零固定阈值）。"""
        monkeypatch.setenv("LMS_PRECISION_ADAPT", "1")
        pa = PrecisionAdaptState(min_samples=5)
        # 多个域：反驳率 0 的域（干净）
        for _ in range(20):
            e = _make_entry(rebuttal_count=0, reference_count=1)
            pa.record_recall_cohort([(1.0, e)])
        clean_rank = pa.last_domain['suspicion']
        # 高反驳率域
        for _ in range(3):
            e = _make_entry(rebuttal_count=3, reference_count=1)
            pa.record_recall_cohort([(1.0, e)])
        hot_rank = pa.last_domain['suspicion']
        assert hot_rank > clean_rank
        assert pa.last_domain['rebuttal_rate'] > 0.0


class _FakeEmbedder:
    """带 embed_text/embed_text_raw 的假嵌入器（PretrainedEmbedder 形态）。"""
    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed_text(self, text: str):
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        v = torch.zeros(self.dim)
        for i in range(self.dim):
            v[i] = (h[i % len(h)] / 255.0) * 2 - 1
        return v

    embed_text_raw = embed_text


# ============================================================
# 9. 接线观测（loop /status /react /recall）
# ============================================================

class TestWiring:
    def _make_loop(self, monkeypatch, tmp_path, enabled):
        monkeypatch.setenv("LMS_PRECISION_ADAPT", "1" if enabled else "0")
        from runtime.config import default_config
        from runtime.loop import LivingMemoryLoop
        cfg = default_config()
        cfg.update(num_nodes=32, input_dim=16, num_infer_steps=5,
                   embedder=_FakeEmbedder(16), auto_snapshot=False,
                   snapshot_dir=str(tmp_path / "s"))
        return LivingMemoryLoop(cfg)

    def test_status_and_react_doubt_fields(self, monkeypatch, tmp_path):
        loop = self._make_loop(monkeypatch, tmp_path, enabled=True)
        for i in range(5):
            loop.process_turn(f"用户: 第{i}轮对话内容")
        status = loop.get_status()
        pa = status.get('precision_adapt')
        assert pa is not None and pa.get('enabled') is True
        assert 'baseline' in pa and 'threshold' in pa
        assert 'global_precision' in pa
        assert 'volatility' in pa
        # /react reaction.doubt（glue 透传到注入插件的数据源）
        react = loop.react_readonly("测试问题", k=0)
        assert react['reaction']['doubt']['enabled'] is True
        assert 'threshold' in react['reaction']['doubt']

    def test_switch_off_empty_blocks(self, monkeypatch, tmp_path):
        loop = self._make_loop(monkeypatch, tmp_path, enabled=False)
        loop.process_turn("用户: 测试")
        assert loop.get_status().get('precision_adapt') == {}
        assert loop.react_readonly("测试", k=0)['reaction']['doubt'] == {}

    def test_recall_annotates_adaptive_precision(self, monkeypatch, tmp_path):
        """/recall 条目带 adaptive_confidence/consistency/doubt_verdict。"""
        loop = self._make_loop(monkeypatch, tmp_path, enabled=True)
        loop.process_turn("用户: 我喜欢蓝色的小橘猫")
        results = loop.recall_episodic_readonly("蓝色的小橘猫", k=3)
        assert results
        item = results[0]
        assert 'adaptive_confidence' in item
        assert 'doubt_verdict' in item
        assert 'consistency' in item
        # 只读：引用计数不增（零持久化保持）
        e = list(loop.memory.iter_episodic())[0]
        assert e.reference_count == 0

    def test_conflict_event_feeds_calibration(self, monkeypatch, tmp_path):
        """conflict 证伪事件 → conformal 校准集 + 负性证据标记。"""
        loop = self._make_loop(monkeypatch, tmp_path, enabled=True)
        loop.process_turn("用户: 我喜欢蓝色的小橘猫")
        loop.process_turn("[doubt] conflict: 我喜欢蓝色的小橘猫")
        assert loop.precision_adapt is not None
        assert len(loop.precision_adapt.rebuttal_before_confidences) >= 1
        # 条目已标记证伪
        e = list(loop.memory.iter_episodic())[0]
        assert e.rebuttal_count >= 1
        assert getattr(e, 'confidence_before_rebuttal', None) is not None
        # 负性证据标记已消费（不残留到下一轮）
        assert loop._pending_negative_evidence is False

    def test_dream_engine_receives_precision_adapt(self, monkeypatch, tmp_path):
        """做梦引擎拿到 precision_adapt（doubt_review 低置信复核用自适应
        分位线；开关关 → None → 回退 0.3）。"""
        loop = self._make_loop(monkeypatch, tmp_path, enabled=True)
        de = loop.get_dream_engine()
        assert de.precision_adapt is loop.precision_adapt
        assert de.precision_adapt is not None

        loop_off = self._make_loop(monkeypatch, tmp_path, enabled=False)
        de_off = loop_off.get_dream_engine()
        assert de_off.precision_adapt is None


# ============================================================
# 10. memory_trust 打通确认（现状审计）
# ============================================================

class TestTrustInterop:
    def test_entry_confidence_before_rebuttal_captured(self):
        """mark_rebutted 记录反驳前置信度（conformal 校准集数据源）。"""
        e = _make_entry(reference_count=9, confidence=0.9)
        assert getattr(e, 'confidence_before_rebuttal', None) is None
        mark_rebutted(e, violated_by="证据")
        assert e.confidence_before_rebuttal == pytest.approx(0.9)
        assert e.rebuttal_count == 1
        assert e.confidence == pytest.approx(0.9)  # 重算后 (1-1/10)

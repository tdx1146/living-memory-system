# -*- coding: utf-8 -*-
"""单元测试：做梦时怀疑延伸（阶段 4，2026-08-14）

定位（主 AI 修正）：怀疑是本能，不是特定场景的功能——本组测试覆盖的是
**持续怀疑底座**（precision 三层动态化条目级 + 思考链）在记忆巩固期的
**自然延伸/生效**（dream_engine._doubt_review 阶段 4 路径），不是新增怀疑
机制。核心机制（设计 v1.0 §六 阶段 4 + 8/11 报告启示 3/4 + B3/Schiller 2010）：

  1. 治理开关 DREAM_DOUBT_ENABLED（默认 1=开；0=关回退阶段 1 基础行为）
  2. 不完全线索复核（B3：完全重复反而稳定旧记忆——绝不复述全文）
  3. 结果应用（Schiller 2010 更新而非抹除）：有更强替代 → 降权 ×0.9 +
     合并新版本；无替代 → 维持但降低 precision 权重 ×0.98
  4. 反教条 top-N 高频复核下沉（加工在 LMS 内；AgentOS 侧保留表达层）
  5. 冷却守卫（dream idle_threshold=30s 高频触发，防系统性侵蚀）
  6. 逐条明细观测（last_doubt_review_detail → dream 结果透传）

测试设计原则：
  - 小规模实例（num_nodes=8, input_dim=4）加速测试
  - 直接用 DreamEngine（不经过 loop）——precision_adapt 缺省 None 时
    _review_confidence 回退 entry.confidence（fail-open 路径）
  - 固定 seed 保证可复现；fail-open 路径单独覆盖
"""

import os
import time

import pytest
import torch

from core.hippocampus.attractor import AttractorNetwork
from core.hippocampus.purpose import PurposeLayer
from core.hippocampus.memory import MemoryManager, EpisodicEntry
from core.hippocampus.dream_engine import (
    DreamEngine, dream_doubt_enabled, _doubt_param,
)


def _make_entry(text="测试记忆", confidence=1.0, **kw):
    """构造 EpisodicEntry（与 test_doubt_integration._make_entry 同构）。"""
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


def _make_engine(tmp_path, **cfg_overrides):
    torch.manual_seed(42)
    attractor = AttractorNetwork(8, 4, seed=1, temperature=0.05)
    purpose = PurposeLayer(4)
    mm = MemoryManager(8, buffer_capacity=20, episodic_capacity=50)
    cfg = {"snapshot_dir": str(tmp_path)}
    cfg.update(cfg_overrides)
    de = DreamEngine(attractor=attractor, purpose=purpose, memory=mm,
                     embedder=None, config=cfg)
    return de, mm


# ============================================================
# 1. 治理开关（DREAM_DOUBT_ENABLED）
# ============================================================

class TestGovernanceSwitch:
    def test_default_on(self):
        # 默认 1=开（不污染测试环境的既有 env）
        old = os.environ.pop("DREAM_DOUBT_ENABLED", None)
        try:
            assert dream_doubt_enabled() is True
        finally:
            if old is not None:
                os.environ["DREAM_DOUBT_ENABLED"] = old

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("DREAM_DOUBT_ENABLED", "0")
        assert dream_doubt_enabled(None) is False   # env 关
        assert dream_doubt_enabled(True) is True    # 显式开压过 env
        assert dream_doubt_enabled(False) is False  # 显式关

    def test_env_parsing(self, monkeypatch):
        for off in ("0", "false", "no", "off", "FALSE"):
            monkeypatch.setenv("DREAM_DOUBT_ENABLED", off)
            assert dream_doubt_enabled() is False, off
        monkeypatch.setenv("DREAM_DOUBT_ENABLED", "1")
        assert dream_doubt_enabled() is True
        monkeypatch.setenv("DREAM_DOUBT_ENABLED", "garbage")
        assert dream_doubt_enabled() is True  # 非关值一律视为开（fail-open）

    def test_doubt_param_parsing(self, monkeypatch):
        assert _doubt_param({}, "x", "DREAM_DOUBT_X", 0.7) == 0.7
        monkeypatch.setenv("DREAM_DOUBT_X", "0.85")
        assert _doubt_param({}, "x", "DREAM_DOUBT_X", 0.7) == 0.85
        assert _doubt_param({"x": 0.5}, "x", "DREAM_DOUBT_X", 0.7) == 0.5
        monkeypatch.setenv("DREAM_DOUBT_X", "abc")
        assert _doubt_param({}, "x", "DREAM_DOUBT_X", 0.7) == 0.7  # fail-open


# ============================================================
# 2. 不完全线索（B3：绝不复述全文）
# ============================================================

class TestIncompleteClue:
    def test_clue_is_partial_and_interrogative(self, tmp_path):
        de, _ = _make_engine(tmp_path)
        e = _make_entry(text="用户: 我们讨论过行动层应该只产出意向不执行，边界要写清楚",
                        confidence=0.2)
        clue = de._build_incomplete_clue(e)
        assert "这条还成立吗" in clue
        # 不完全：线索 ≠ 全文，且明显短于原文（前缀截断）
        assert clue != e.text
        assert "用户:" not in clue  # 前缀剥离
        assert len(clue) < len(e.text)

    def test_clue_never_repeats_full_text(self, tmp_path):
        de, _ = _make_engine(tmp_path)
        # 短文本：cut 下限 8 字，仍带疑问句；不原样复述整句
        e = _make_entry(text="短记忆", confidence=0.5)
        clue = de._build_incomplete_clue(e)
        assert clue.endswith("这条还成立吗？")
        assert "短记忆" in clue

    def test_clue_empty_text_fallback(self, tmp_path):
        de, _ = _make_engine(tmp_path)
        e = _make_entry(text="", confidence=0.5)
        assert "这条还成立吗" in de._build_incomplete_clue(e)


# ============================================================
# 3. 复核结果应用（Schiller 2010：更新而非抹除）
# ============================================================

class TestReviewOutcome:
    def test_no_alternative_kept_with_decay(self, tmp_path):
        """无替代 → 维持（×0.98 降权被下限 0.5 拦下：0.2 已在下限内不降，
        P2-1 修复）+ reviewed + 明细。"""
        de, mm = _make_engine(tmp_path)
        e = _make_entry(text="低置信记忆：系统总线应该拆成三块", confidence=0.2)
        mm._episodic_buffer.append(e)
        de._doubt_review()
        assert e.confidence == pytest.approx(0.2)  # 下限守卫：不降也不抬升
        assert de.last_doubt_review["reviewed"] >= 1
        assert de.last_doubt_review["kept"] >= 1
        assert de.last_doubt_review.get("merged", 0) == 0
        assert e.labile is False
        # 明细：category/clue/outcome/alt/conf_before/conf_after 齐全
        assert de.last_doubt_review_detail, "应有逐条明细"
        d = de.last_doubt_review_detail[0]
        assert d["outcome"] == "kept"
        assert d["alt"] is False
        assert d["conf_before"] == pytest.approx(0.2, abs=0.001)
        assert d["conf_after"] == pytest.approx(0.2, abs=0.001)
        assert "这条还成立吗" in d["clue"]
        # 冷却标记已写
        assert e.last_doubt_reviewed_at is not None

    def test_alternative_downgrade_and_merge(self, tmp_path):
        """有更强替代（向量相似≥0.7 且置信度更高）→ 降权 ×0.9（被下限
        0.5 拦下，P2-1）+ superseded_by + 合并（替代 reference+1）。"""
        de, mm = _make_engine(tmp_path)
        # 旧记忆：低置信，被证伪过（rebuttal≥1 也是候选条件之一）
        old = _make_entry(text="旧结论：怀疑是独立模块", confidence=0.2,
                          rebuttal_count=1)
        old.semantic_vector = torch.ones(8) * 0.9
        # 更强替代：同主题高置信
        alt = _make_entry(text="新结论：怀疑是记忆内建元层", confidence=0.9,
                          reference_count=3)
        alt.semantic_vector = torch.ones(8)
        mm._episodic_buffer.append(old)
        mm._episodic_buffer.append(alt)
        de._doubt_review()
        # 旧记忆降权（×0.9 被下限 0.5 拦下：0.2 已在下限内不降，P2-1）
        # + superseded_by + 合并（替代 reference+1）
        assert old.confidence == pytest.approx(0.2)
        assert old.superseded_by  # 指向替代
        assert alt.reference_count == 4  # 合并吸收加固
        assert de.last_doubt_review["downgraded"] >= 1
        assert de.last_doubt_review["merged"] >= 1
        d = next(x for x in de.last_doubt_review_detail
                 if x["outcome"] == "downgraded")
        assert d["alt"] is True
        assert d["category"] == "low_conf"

    def test_rebutted_candidate_included(self, tmp_path):
        """被反驳≥1（doubt_ingest 证伪历史）但不低置信 → 仍进复核。"""
        de, mm = _make_engine(tmp_path)
        e = _make_entry(text="被证伪过的记忆：旧方案A", confidence=0.8,
                        rebuttal_count=1)
        mm._episodic_buffer.append(e)
        de._doubt_review()
        assert de.last_doubt_review["reviewed"] >= 1
        assert e.last_doubt_reviewed_at is not None

    def test_anti_dogma_high_freq_reviewed(self, tmp_path):
        """反教条下沉：top-N 高频条目做不完全线索复核（不止 flag）。"""
        de, mm = _make_engine(tmp_path)
        e = _make_entry(text="被高频引用的记忆：长期结论X", confidence=0.9,
                        reference_count=10, recall_count=10)
        mm._episodic_buffer.append(e)
        de._doubt_review()
        assert de.last_doubt_review["flagged"] >= 1
        # 高频条目已被复核（明细 category=anti_dogma）
        cats = [d["category"] for d in de.last_doubt_review_detail]
        assert "anti_dogma" in cats
        assert e.last_doubt_reviewed_at is not None

    def test_cooldown_prevents_repeated_decay(self, tmp_path):
        """冷却守卫：同一轮/短间隔内不重复复核（防高频做梦系统性侵蚀）。"""
        de, mm = _make_engine(tmp_path)
        e = _make_entry(text="冷却测试记忆", confidence=0.2)
        mm._episodic_buffer.append(e)
        de._doubt_review()
        c1 = e.confidence
        de._doubt_review()  # 立即再次复核 → 冷却跳过
        assert e.confidence == c1
        assert de.last_doubt_review["reviewed"] == 1  # 只复核了一次

    def test_no_vector_fail_open(self, tmp_path):
        """条目无 semantic_vector → 替代检查 None → 走保守无替代路径（不炸）。"""
        de, mm = _make_engine(tmp_path)
        e = _make_entry(text="无向量记忆", confidence=0.2)
        e.semantic_vector = None
        mm._episodic_buffer.append(e)
        de._doubt_review()  # 不应抛异常
        # 无向量 → 走保守无替代路径；×0.98 被下限 0.5 拦下（0.2 不降，P2-1）
        assert e.confidence == pytest.approx(0.2)


# ============================================================
# 4. 开关关 → 回退阶段 1 基础行为（可回滚路径）
# ============================================================

class TestRollbackPath:
    def test_disabled_falls_back_to_phase1(self, tmp_path):
        de, mm = _make_engine(tmp_path, dream_doubt_enabled=False)
        e = _make_entry(text="低置信记忆", confidence=0.2)
        mm._episodic_buffer.append(e)
        de._doubt_review()
        # 阶段 1 行为：×1.02 回稳（不是 ×0.98 折损），无明细、无冷却标记
        assert e.confidence == pytest.approx(0.2 * 1.02)
        assert de.last_doubt_review["reviewed"] >= 1
        assert de.last_doubt_review_detail == []
        assert getattr(e, "last_doubt_reviewed_at", None) is None

    def test_disabled_high_freq_only_flagged(self, tmp_path):
        de, mm = _make_engine(tmp_path, dream_doubt_enabled=False)
        e = _make_entry(text="高频记忆", confidence=0.9,
                        reference_count=10, recall_count=10)
        mm._episodic_buffer.append(e)
        de._doubt_review()
        assert de.last_doubt_review["flagged"] >= 1
        # 只 flag 不复核（无明细、无冷却标记）
        assert de.last_doubt_review_detail == []
        assert getattr(e, "last_doubt_reviewed_at", None) is None


# ============================================================
# 5. dream 结果透传 + 既有机制共存
# ============================================================

class TestDreamResultIntegration:
    def test_dream_mvp_returns_detail(self, tmp_path):
        """dream_mvp 结果含 doubt_review_detail（纯增量字段，观测用）。"""
        de, mm = _make_engine(tmp_path)
        state = torch.randn(8)
        from core.types import Activation
        mm.update(Activation(state=state, entropy=1.0, surprise=5.0), surprise=5.0, turn=1)
        e = _make_entry(text="低置信记忆", confidence=0.2)
        mm._episodic_buffer.append(e)
        res = de.dream_mvp(n_steps=5)
        assert "doubt_review_detail" in res
        assert isinstance(res["doubt_review_detail"], list)
        assert "doubt_review" in res

    def test_labile_still_adjudicated(self, tmp_path):
        """① labile 窗口裁决不受阶段 4 影响（既有机制共存）。"""
        de, mm = _make_engine(tmp_path)
        e = _make_entry(text="被证伪的记忆", turn=1, labile=True,
                        violated_by="矛盾证据", confidence=0.9)
        mm._episodic_buffer.append(e)
        de._doubt_review()
        assert de.last_doubt_review["rewritten"] >= 1
        assert e.labile is False
        # labile 条目不重复进 ②/③（was_labile 排除）
        assert all(d["category"] != "low_conf" for d in de.last_doubt_review_detail)

    def test_empty_entries_noop(self, tmp_path):
        de, mm = _make_engine(tmp_path)
        de._doubt_review()  # 空缓冲区不抛异常
        assert de.last_doubt_review["reviewed"] == 0

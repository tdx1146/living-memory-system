# -*- coding: utf-8 -*-
"""单元测试：做梦时怀疑降权下限（P2-1 修复，2026-08-14，审计-阶段4）

背景：审计 P2-1 —— 无替代 ×0.98 降权无下限，长期低置信但为真的记忆最坏
0.98^30≈0.545/月被磨没。修复：新增 env DREAM_DOUBT_MIN_WEIGHT（默认 0.5），
权重 ≤ 下限后不再降（"怀疑的持续表达"有边界：更新而非抹除）。

下限语义（dream_engine._apply_doubt_decay）：
  - 只拦下降不抬升：conf ≤ 下限原样返回（低置信条目留在复核通道）
  - conf > 下限 → max(下限, conf×factor)：一次最多降到下限
  - 两条降权路径共用：无替代 ×0.98（kept）/ 有替代 ×0.9（downgraded+merged）
  - 下限=0 → 无下限（回退旧行为）；证伪 rebuttal≥2→0.1 独立不受限

测试设计原则：
  - 与 test_dream_doubt_phase4 同构（小规模实例、直接 DreamEngine）
  - 模拟多轮降权 = 复核后手动清冷却标记（last_doubt_reviewed_at=None）
    → 再 _doubt_review()（与生产 86400s 冷却解耦，专测下限收敛）
  - 覆盖：默认下限收敛 / 配置可调（config>env>默认，fail-open）/
    下限=0 回退旧行为 / 已在下限不抬升 / ×0.9 合并路径同受下限 /
    开关关不受影响
"""

import pytest
import torch

from core.hippocampus.attractor import AttractorNetwork
from core.hippocampus.purpose import PurposeLayer
from core.hippocampus.memory import MemoryManager, EpisodicEntry
from core.hippocampus.dream_engine import DreamEngine


def _make_entry(text="测试记忆", confidence=1.0, **kw):
    """构造 EpisodicEntry（与 test_dream_doubt_phase4._make_entry 同构）。"""
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


def _review_rounds(de, mm, entry, rounds=40):
    """多轮复核（每轮清冷却标记，模拟跨天反复复核；与冷却守卫解耦）。"""
    for _ in range(rounds):
        entry.last_doubt_reviewed_at = None
        de._doubt_review()


# ============================================================
# 1. 下限收敛（P2-1 核心：多轮降到下限后不再下降）
# ============================================================

class TestDecayFloor:
    def test_default_floor_converges_and_stops(self, tmp_path):
        """默认下限 0.5：高置信高频条目多轮 ×0.98 降到 0.5 后不再降。"""
        de, mm = _make_engine(tmp_path)
        # 反教条候选：高置信高频（无替代 → kept ×0.98 路径）
        e = _make_entry(text="高频但存疑的结论", confidence=0.9,
                        reference_count=10, recall_count=10)
        mm._episodic_buffer.append(e)
        _review_rounds(de, mm, e, rounds=60)
        assert e.confidence == pytest.approx(0.5)   # 收敛到下限
        # 再多轮也不低于下限
        _review_rounds(de, mm, e, rounds=5)
        assert e.confidence == pytest.approx(0.5)

    def test_decay_applies_above_floor(self, tmp_path):
        """0.6（>0.5）首轮正常降 ×0.98 → 0.588，多轮收敛到 0.5。"""
        de, mm = _make_engine(tmp_path)
        # rebuttal≥1 使高置信条目进入复核（候选条件），无替代 → kept
        e = _make_entry(text="边界记忆", confidence=0.6, rebuttal_count=1)
        mm._episodic_buffer.append(e)
        de._doubt_review()
        assert e.confidence == pytest.approx(0.6 * 0.98)  # 未触底：正常降
        _review_rounds(de, mm, e, rounds=40)
        assert e.confidence == pytest.approx(0.5)         # 触底停止

    def test_below_floor_not_raised(self, tmp_path):
        """已在下限内（0.2）：只拦下降不抬升——复核后仍 0.2。"""
        de, mm = _make_engine(tmp_path)
        e = _make_entry(text="低置信记忆", confidence=0.2)
        mm._episodic_buffer.append(e)
        de._doubt_review()
        assert e.confidence == pytest.approx(0.2)
        _review_rounds(de, mm, e, rounds=5)
        assert e.confidence == pytest.approx(0.2)

    def test_floor_zero_restores_old_behavior(self, tmp_path):
        """下限=0 → 无下限（回退旧行为：×0.98 持续降，与 P2 前一致）。"""
        de, mm = _make_engine(tmp_path, doubt_min_weight=0.0)
        e = _make_entry(text="低置信记忆", confidence=0.2)
        mm._episodic_buffer.append(e)
        de._doubt_review()
        assert e.confidence == pytest.approx(0.2 * 0.98)
        _review_rounds(de, mm, e, rounds=30)
        assert e.confidence == pytest.approx(0.2 * 0.98 ** 31)


# ============================================================
# 2. 有替代 ×0.9 合并路径（评估结论：同受下限）
# ============================================================

class TestDowngradePathFloor:
    def test_downgrade_below_floor_not_decayed(self, tmp_path):
        """有替代 ×0.9 合并路径同受下限（0.2 已在下限内 → 不降，合并照常）。"""
        de, mm = _make_engine(tmp_path)
        old = _make_entry(text="旧结论", confidence=0.2, rebuttal_count=1)
        old.semantic_vector = torch.ones(8) * 0.9
        alt = _make_entry(text="新结论", confidence=0.9, reference_count=3)
        alt.semantic_vector = torch.ones(8)
        mm._episodic_buffer.append(old)
        mm._episodic_buffer.append(alt)
        de._doubt_review()
        assert old.confidence == pytest.approx(0.2)  # 下限拦下 ×0.9
        assert old.superseded_by                      # 合并副作用照常
        assert alt.reference_count == 4

    def test_downgrade_above_floor_converges(self, tmp_path):
        """高置信被替代条目：×0.9 降到下限即止（0.8→0.72→…→0.5）。"""
        de, mm = _make_engine(tmp_path)
        old = _make_entry(text="被替代的高置信旧结论", confidence=0.8,
                          rebuttal_count=1)
        old.semantic_vector = torch.ones(8) * 0.9
        alt = _make_entry(text="更强新结论", confidence=0.95, reference_count=3)
        alt.semantic_vector = torch.ones(8)
        mm._episodic_buffer.append(old)
        mm._episodic_buffer.append(alt)
        de._doubt_review()
        assert old.confidence == pytest.approx(0.8 * 0.9)  # 0.72 > 0.5 正常降
        _review_rounds(de, mm, old, rounds=30)
        assert old.confidence == pytest.approx(0.5)        # 触底停止


# ============================================================
# 3. 配置（config > env > 默认，fail-open）+ 开关关不受影响
# ============================================================

class TestFloorConfig:
    def test_default_value(self, tmp_path):
        de, _ = _make_engine(tmp_path)
        assert de.doubt_min_weight == pytest.approx(0.5)

    def test_config_override(self, tmp_path):
        de, _ = _make_engine(tmp_path, doubt_min_weight=0.3)
        assert de.doubt_min_weight == pytest.approx(0.3)

    def test_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DREAM_DOUBT_MIN_WEIGHT", "0.25")
        de, _ = _make_engine(tmp_path)
        assert de.doubt_min_weight == pytest.approx(0.25)

    def test_explicit_config_beats_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DREAM_DOUBT_MIN_WEIGHT", "0.9")
        de, _ = _make_engine(tmp_path, doubt_min_weight=0.4)
        assert de.doubt_min_weight == pytest.approx(0.4)

    def test_env_fail_open(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DREAM_DOUBT_MIN_WEIGHT", "abc")
        de, _ = _make_engine(tmp_path)
        assert de.doubt_min_weight == pytest.approx(0.5)  # fail-open 默认

    def test_disabled_switch_unaffected(self, tmp_path):
        """DREAM_DOUBT_ENABLED=0：回退 ×1.02 回稳路径，不经降权下限。"""
        de, mm = _make_engine(tmp_path, dream_doubt_enabled=False)
        e = _make_entry(text="低置信记忆", confidence=0.2)
        mm._episodic_buffer.append(e)
        de._doubt_review()
        assert e.confidence == pytest.approx(0.2 * 1.02)
        assert de.last_doubt_review_detail == []
        assert getattr(e, "last_doubt_reviewed_at", None) is None

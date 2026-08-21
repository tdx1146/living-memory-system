# -*- coding: utf-8 -*-
"""M6 单测：梦期巩固时相（dream/consolidation）
（核心重建规格 v2 §2.4 / §4.3 / §6.2 / §7.1 M6）

覆盖（§6.2"做梦期 resolve_labile"用例组 + §4.3 事件流发布）：
  1. 梦期 resolve_labile confirm/supersedes 两分支（走 M3 三时相状态机
     consolidation_resolve——写侧唯一转移入口；梦期是唯一批量改写点）；
  2. supersedes 流程（与 M3 状态机衔接）：原条目标 superseded +
     superseded_by 衔接 + source='doubt' [doubt-supersedes] 记录落库 +
     superseded 条目移出 external 检索面（不抹除原文，Schiller 2010 B2）；
  3. 验证链裁决驱动分支（LMS_VERIFICATION_CHAIN_ENABLED=1：CONFLICT →
     supersedes；CONFIRM → confirm）；开关默认关 → 回退证伪证据判定；
  4. 事件流发布（§4.3 落沙必发事件）：梦期完成 lms.dream_complete +
     lms.doubt_consolidation（confirm/supersedes 结果）；验证链事件
     lms.verification（开关关 → 零发布）；
  5. standalone（无状态机注入）兼容既有纯函数 resolve_labile 行为。

运行方式：pytest rewrite-ws/tests/test_dream_consolidation_m6.py -v
（生产 venv：/tmp/repro3-1786556208/living-memory-system-cloud/.venv）。
"""

import json
import os
import sys
import time

# 确保项目根目录在 Python 路径中（可从任意 cwd 运行）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
import torch

from core.doubt.rebuttal_field import get_rebuttal_consistency
from core.doubt.state_machine import DoubtStateMachine, EntryDoubtState
from core.doubt.verification_chain import (
    ConflictKind,
    VerificationChain,
    VerdictType,
    verification_key,
)
from core.hippocampus.memory import MemoryManager, EpisodicEntry
from core.hippocampus.attractor import AttractorNetwork
from core.hippocampus.purpose import PurposeLayer
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


def _make_engine(tmp_path, doubt_state=None, verification_chain=None,
                 **cfg_overrides):
    """构造 DreamEngine（可选注入 M3 状态机/验证链——M6 接线形态）。"""
    torch.manual_seed(42)
    attractor = AttractorNetwork(8, 4, seed=1, temperature=0.05)
    purpose = PurposeLayer(4)
    mm = MemoryManager(8, buffer_capacity=20, episodic_capacity=50)
    cfg = {"snapshot_dir": str(tmp_path)}
    if doubt_state is not None:
        cfg["doubt_state"] = doubt_state
    if verification_chain is not None:
        cfg["verification_chain"] = verification_chain
    cfg.update(cfg_overrides)
    de = DreamEngine(attractor=attractor, purpose=purpose, memory=mm,
                     embedder=None, config=cfg)
    return de, mm


# ============================================================
# 1. 梦期 resolve_labile confirm/supersedes 两分支（走 M3 状态机）
# ============================================================

class TestDreamConsolidationStateMachine:
    """梦期巩固时相：labile/suspect 条目走 consolidation_resolve。"""

    def test_labile_violated_supersedes_via_state_machine(self, tmp_path):
        """supersedes 分支：labile+证伪证据 → 原条目标 superseded +
        superseded_by 衔接 + [doubt-supersedes] 记录落库 + 原生字段
        updated_by='consolidation'（§2.3 合法写者）。"""
        sm = DoubtStateMachine()
        de, mm = _make_engine(tmp_path, doubt_state=sm)
        e = _make_entry(text="被证伪的记忆：旧方案A", turn=1,
                        labile=True, violated_by="矛盾证据", confidence=0.9)
        e.labile_since = time.time()
        mm._episodic_buffer.append(e)
        de._doubt_review()
        # 两分支：supersedes（rewritten）
        assert de.last_doubt_review["rewritten"] >= 1
        assert de.last_doubt_review["reviewed"] >= 1
        # 原条目：持久态 superseded（状态机写侧转移）+ superseded_by 衔接
        assert e.doubt_state == EntryDoubtState.SUPERSEDED.value
        assert e.labile is False          # labile 时相结束
        assert getattr(e, "superseded_by", ""), "superseded_by 应指向新记录"
        assert getattr(e, "superseded_at", None) is not None
        # 原生字段由巩固时相写侧更新（updated_by='consolidation'）
        assert get_rebuttal_consistency(e)["updated_by"] == "consolidation"
        # supersedes 记录（source='doubt' + [doubt-supersedes] 前缀）已落库
        sources = [x.source for x in mm.iter_episodic()]
        assert sources.count("doubt") >= 1
        new = [x for x in mm.iter_episodic() if x.source == "doubt"][-1]
        assert new.text.startswith("[doubt-supersedes]")
        # 逐条明细（category='consolidation'）
        d = next(x for x in de.last_doubt_review_detail
                 if x["category"] == "consolidation"
                 and x["outcome"] == "rewritten")
        assert d["verdict"] == "conflict"
        assert d["conf_before"] == pytest.approx(0.9, abs=1e-3)

    def test_labile_no_evidence_confirmed(self, tmp_path):
        """confirm 分支：labile 无证伪证据（窗口内）→ kept → 回 stable，
        confidence ×1.02 重巩固。"""
        sm = DoubtStateMachine()
        de, mm = _make_engine(tmp_path, doubt_state=sm)
        e = _make_entry(text="窗口内记忆", turn=1,
                        labile=True, confidence=0.5)
        e.labile_since = time.time()
        mm._episodic_buffer.append(e)
        de._doubt_review()
        assert de.last_doubt_review["kept"] >= 1
        assert e.doubt_state == EntryDoubtState.STABLE.value
        assert e.labile is False
        assert e.confidence == pytest.approx(0.5 * 1.02, abs=1e-3)

    def test_labile_timeout_downgraded(self, tmp_path):
        """confirm 分支超时：labile 无证据且窗口超时 → downgraded（×0.98）。"""
        sm = DoubtStateMachine()
        de, mm = _make_engine(tmp_path, doubt_state=sm)
        e = _make_entry(text="超时记忆", turn=1,
                        labile=True, confidence=0.5)
        e.labile_since = time.time() - 999999
        mm._episodic_buffer.append(e)
        de._doubt_review()
        assert de.last_doubt_review["downgraded"] >= 1
        assert e.doubt_state == EntryDoubtState.STABLE.value
        assert e.confidence == pytest.approx(0.5 * 0.98, abs=1e-3)

    def test_suspect_entry_confirmed_to_stable(self, tmp_path):
        """suspect 条目（注入时怀疑持久态）→ confirm → 回 stable（§2.4）。"""
        sm = DoubtStateMachine()
        de, mm = _make_engine(tmp_path, doubt_state=sm)
        e = _make_entry(text="高惊讶新条目", turn=1, confidence=0.6)
        e.doubt_state = EntryDoubtState.SUSPECT.value  # 注入时怀疑标记
        mm._episodic_buffer.append(e)
        de._doubt_review()
        assert de.last_doubt_review["reviewed"] >= 1
        assert e.doubt_state == EntryDoubtState.STABLE.value
        assert e.labile is False

    def test_suspect_entry_violated_superseded(self, tmp_path):
        """suspect + 证伪证据 → supersedes 分支（同 labile 裁决路径）。"""
        sm = DoubtStateMachine()
        de, mm = _make_engine(tmp_path, doubt_state=sm)
        e = _make_entry(text="被证伪的新条目", turn=1,
                        violated_by="矛盾证据", confidence=0.6)
        e.doubt_state = EntryDoubtState.SUSPECT.value
        mm._episodic_buffer.append(e)
        de._doubt_review()
        assert de.last_doubt_review["rewritten"] >= 1
        assert e.doubt_state == EntryDoubtState.SUPERSEDED.value
        assert any(x.source == "doubt" for x in mm.iter_episodic())

    def test_consolidation_excluded_from_low_conf_same_round(self, tmp_path):
        """梦期已巩固（labile/suspect）条目本轮不再进 ②/③ 低置信复核
        （was_consolidated 排除——避免同一轮双重处理）。"""
        sm = DoubtStateMachine()
        de, mm = _make_engine(tmp_path, doubt_state=sm)
        e = _make_entry(text="低置信且被证伪", turn=1, confidence=0.2,
                        labile=True, violated_by="证据")
        e.labile_since = time.time()
        mm._episodic_buffer.append(e)
        de._doubt_review()
        cats = [d["category"] for d in de.last_doubt_review_detail]
        assert "consolidation" in cats
        assert "low_conf" not in cats, "巩固过的条目不应再进低置信复核"

    def test_dream_mvp_result_includes_consolidation_block(self, tmp_path):
        """dream_mvp 结果含 consolidation 观测块（§4.3 event 流发布数据源）。"""
        sm = DoubtStateMachine()
        de, mm = _make_engine(tmp_path, doubt_state=sm)
        state = torch.randn(8)
        from core.types import Activation
        mm.update(Activation(state=state, entropy=1.0, surprise=5.0),
                  surprise=5.0, turn=1)
        e = _make_entry(text="被证伪的记忆", turn=1,
                        labile=True, violated_by="矛盾证据", confidence=0.9)
        e.labile_since = time.time()
        mm._episodic_buffer.append(e)
        res = de.dream_mvp(n_steps=5)
        assert "consolidation" in res
        assert res["doubt_review"]["rewritten"] >= 1
        assert res["consolidation"]["supersedes"], "应有 supersedes 记录明细"
        assert res["consolidation"]["verification"]["enabled"] is False


# ============================================================
# 2. supersedes 流程（与 M3 状态机衔接——loop 集成）
# ============================================================

class TestDreamSupersedesFlow:
    """loop 集成：梦期 supersedes → 原条目持久态 superseded + 检索面排除。"""

    class _MockEmbedder:
        """带 embed_text 的确定性嵌入器（同 test_doubt_m3 先例）。"""

        def __init__(self, dim=16):
            self._dim = dim

        def embed_text(self, text: str) -> torch.Tensor:
            if not text or not text.strip():
                return torch.zeros(self._dim)
            h = abs(hash(text)) % (2 ** 32)
            g = torch.Generator()
            g.manual_seed(h)
            return torch.randn(self._dim, generator=g) * 0.1

        embed_text_raw = embed_text

        @property
        def dim(self) -> int:
            return self._dim

    def _make_loop(self, tmp_path):
        from runtime.config import default_config
        from runtime.loop import LivingMemoryLoop
        cfg = default_config()
        cfg.update(num_nodes=32, input_dim=16, num_infer_steps=5,
                   embedder=self._MockEmbedder(16), auto_snapshot=False,
                   snapshot_dir=str(tmp_path / "s"),
                   archive_enabled=False)
        return LivingMemoryLoop(cfg)

    def test_loop_dream_supersedes_marks_state(self, tmp_path):
        """loop.dream()：labile+证伪证据 → 原条目 superseded + superseded_by
        + [doubt-supersedes] 记录落库（M3 状态机衔接的完整链路）。"""
        loop = self._make_loop(tmp_path)
        loop.process_turn("用户: 蓝色小橘猫很可爱")
        entry = list(loop.memory.iter_episodic())[-1]
        # 写侧证伪路径（与 [doubt] conflict 同源）→ labile
        entry.labile = True
        entry.labile_since = time.time()
        entry.violated_by = "蓝色小橘猫不可爱"
        # 梦期是唯一批量改写点：dream 前无 superseded
        assert getattr(entry, "doubt_state", "stable") == "stable"
        result = loop.dream(n_steps=5)
        assert result["status"] == "dreamed"
        assert result["doubt_review"]["rewritten"] >= 1
        # 原条目持久态 superseded + 衔接
        assert entry.doubt_state == EntryDoubtState.SUPERSEDED.value
        assert entry.labile is False
        assert getattr(entry, "superseded_by", "").startswith(
            "[doubt-supersedes]")
        # supersedes 记录落库（source='doubt'）
        sources = [e.source for e in loop.memory.iter_episodic()]
        assert "doubt" in sources

    def test_loop_dream_superseded_excluded_from_recall(self, tmp_path):
        """superseded 原条目移出 external 检索面（被证伪事实不再权威召回；
        条目保留不抹除——test_doubt_recall_exclusion 口径延伸）。"""
        loop = self._make_loop(tmp_path)
        loop.process_turn("用户: 蓝色小橘猫很可爱")
        entry = list(loop.memory.iter_episodic())[-1]
        entry.labile = True
        entry.labile_since = time.time()
        entry.violated_by = "蓝色小橘猫不可爱"
        # 基线：supersedes 前原条目可召回（低相似度也进 top_k）
        before = loop.recall_episodic_readonly("蓝色小橘猫很可爱", k=5)
        assert any(r["text"] == entry.text for r in before), \
            "supersedes 前原条目应在检索面"
        loop.dream(n_steps=5)
        after = loop.recall_episodic_readonly("蓝色小橘猫很可爱", k=5)
        assert not any(r["text"] == entry.text for r in after), \
            "superseded 原条目应移出 external 检索面"
        # 条目本身保留（不抹除——更新而非抹除）
        assert any(e.text == entry.text for e in loop.memory.iter_episodic())

    def test_loop_dream_confirms_suspect_back_to_stable(self, tmp_path):
        """loop.dream()：suspect 条目（无证伪证据）→ confirm → 回 stable，
        仍可被召回。"""
        loop = self._make_loop(tmp_path)
        loop.process_turn("用户: 普通记忆A")
        entry = list(loop.memory.iter_episodic())[-1]
        entry.doubt_state = EntryDoubtState.SUSPECT.value
        result = loop.dream(n_steps=5)
        assert result["status"] == "dreamed"
        assert entry.doubt_state == EntryDoubtState.STABLE.value
        # 确认后仍可召回（未被排除）
        res = loop.recall_episodic_readonly("普通记忆A", k=5)
        assert any(r["text"] == entry.text for r in res)

    def test_dream_is_only_batch_rewrite_point(self, tmp_path):
        """梦期是唯一批量改写点：process_turn（注入/检索时相）绝不产生
        superseded；只有 dream 期才出现。"""
        loop = self._make_loop(tmp_path)
        loop.process_turn("用户: 记忆一")
        loop.process_turn("用户: 记忆二")
        for e in loop.memory.iter_episodic():
            assert getattr(e, "doubt_state", "stable") != "superseded", \
                "注入/检索时相不得产生 superseded（只读四不变 + 写侧时相）"
        # 注入时怀疑可标 suspect（写侧），但绝不越级 superseded
        target = list(loop.memory.iter_episodic())[-1]
        loop.doubt_state.injection_check(
            target, surprise=999.0, j_target=1.0,
            verification_chain=loop.verification_chain)
        assert target.doubt_state == EntryDoubtState.SUSPECT.value
        assert loop.doubt_state.snapshot()["phases"]["consolidation"][
            "outcomes"].get("rewritten", 0) == 0
        # 只有梦期 resolve_labile supersedes 分支才标 superseded
        target.labile = True
        target.labile_since = time.time()
        target.violated_by = "矛盾证据"
        loop.dream(n_steps=5)
        assert target.doubt_state == EntryDoubtState.SUPERSEDED.value


# ============================================================
# 3. 验证链裁决驱动分支（LMS_VERIFICATION_CHAIN_ENABLED=1）
# ============================================================

class TestVerificationVerdictDrivesBranch:
    """验证链裁决优先：CONFLICT → supersedes；CONFIRM → confirm。"""

    def test_chain_conflict_drives_supersedes(self, tmp_path):
        """注入时登记 + 验证判 CONFLICT → 梦期 supersedes 分支。"""
        chain = VerificationChain(enabled=True, window=60.0)
        sm = DoubtStateMachine(verification_chain=chain)
        de, mm = _make_engine(tmp_path, doubt_state=sm,
                              verification_chain=chain)
        e = _make_entry(text="应开启方案A", turn=1, confidence=0.8)
        mm._episodic_buffer.append(e)
        # 注入时怀疑：登记验证链（suspect）
        sm.injection_check(e, surprise=50.0, j_target=10.0,
                           verification_chain=chain)
        assert e.doubt_state == EntryDoubtState.SUSPECT.value
        # 独立验证：方向矛盾 → CONFLICT（三选一判定，M3-2）
        req = chain.request_lookup(verification_key(
            str(id(e)), e.text, "injection"))
        assert req is not None
        res = chain.verify(req.idempotency_key,
                           reference_claims=["应关闭方案A"])
        assert res.verdict == VerdictType.CONFLICT.value
        assert res.conflict_kind == ConflictKind.DIRECTIONAL.value
        # 梦期巩固：验证链裁决优先 → supersedes 分支
        de._doubt_review()
        assert de.last_doubt_review["rewritten"] >= 1
        assert e.doubt_state == EntryDoubtState.SUPERSEDED.value
        assert any(x.source == "doubt" for x in mm.iter_episodic())
        # 明细记录验证链裁决来源
        d = next(x for x in de.last_doubt_review_detail
                 if x["category"] == "consolidation"
                 and x["outcome"] == "rewritten")
        assert d["verdict"] == VerdictType.CONFLICT.value

    def test_chain_confirm_drives_confirm(self, tmp_path):
        """验证判 CONFIRM（无矛盾）→ 梦期 confirm 分支（回 stable）。"""
        chain = VerificationChain(enabled=True, window=60.0)
        sm = DoubtStateMachine(verification_chain=chain)
        de, mm = _make_engine(tmp_path, doubt_state=sm,
                              verification_chain=chain)
        e = _make_entry(text="应开启方案A", turn=1, confidence=0.8)
        mm._episodic_buffer.append(e)
        sm.injection_check(e, surprise=50.0, j_target=10.0,
                           verification_chain=chain)
        req = chain.request_lookup(verification_key(
            str(id(e)), e.text, "injection"))
        chain.verify(req.idempotency_key,
                     reference_claims=["应开启方案B"])  # 无矛盾
        res = chain.lookup(req.idempotency_key)
        assert res.verdict == VerdictType.CONFIRM.value
        de._doubt_review()
        assert de.last_doubt_review["rewritten"] == 0
        assert e.doubt_state == EntryDoubtState.STABLE.value

    def test_chain_default_off_falls_back_to_evidence(self, tmp_path):
        """验证链开关默认关 → 回退证伪证据判定（violated_by → supersedes）。"""
        chain = VerificationChain()  # 默认关
        sm = DoubtStateMachine(verification_chain=chain)
        de, mm = _make_engine(tmp_path, doubt_state=sm,
                              verification_chain=chain)
        e = _make_entry(text="被证伪的记忆", turn=1, labile=True,
                        violated_by="矛盾证据", confidence=0.8)
        e.labile_since = time.time()
        mm._episodic_buffer.append(e)
        de._doubt_review()
        assert de.last_doubt_review["rewritten"] >= 1
        assert e.doubt_state == EntryDoubtState.SUPERSEDED.value


# ============================================================
# 4. standalone 兼容（无状态机注入 → 纯函数 resolve_labile）
# ============================================================

class TestStandaloneCompat:
    """DreamEngine 无 doubt_state 注入（既有调用形态）行为不变。"""

    def test_standalone_labile_rewritten_still_works(self, tmp_path):
        """独立使用：labile+证伪 → rewritten + [doubt-supersedes] 落库
        （纯函数 resolve_labile 路径；superseded 兜底标记由 _add_doubt_
        supersede 负责）。"""
        de, mm = _make_engine(tmp_path)  # 无 doubt_state / verification_chain
        e = _make_entry(text="被证伪的记忆", turn=1, labile=True,
                        violated_by="矛盾证据", confidence=0.9)
        e.labile_since = time.time()
        mm._episodic_buffer.append(e)
        de._doubt_review()
        assert de.last_doubt_review["rewritten"] >= 1
        assert e.labile is False
        assert e.doubt_state == EntryDoubtState.SUPERSEDED.value
        assert any(x.source == "doubt" for x in mm.iter_episodic())
        # 明细仍记（category='consolidation'——既有计数/明细兼容）
        assert any(d["category"] == "consolidation"
                   for d in de.last_doubt_review_detail)

    def test_standalone_no_state_machine_snapshot_none(self, tmp_path):
        """独立使用：dream 结果 verification 快照 {'enabled': False}。"""
        de, mm = _make_engine(tmp_path)
        state = torch.randn(8)
        from core.types import Activation
        mm.update(Activation(state=state, entropy=1.0, surprise=5.0),
                  surprise=5.0, turn=1)
        res = de.dream_mvp(n_steps=3)
        assert res["consolidation"]["verification"] == {"enabled": False}


# ============================================================
# 5. 事件流发布（§4.3 落沙必发事件——sandglass 事件流断流修复）
# ============================================================

class TestDreamEventStreamPublishing:
    """梦期巩固/验证链事件发布接线（lms.doubt_consolidation / lms.verification）。"""

    class _MockEmbedder:
        dim = 16

        def embed_text(self, text: str) -> torch.Tensor:
            if not text or not text.strip():
                return torch.zeros(self.dim)
            h = abs(hash(text)) % (2 ** 32)
            g = torch.Generator()
            g.manual_seed(h)
            return torch.randn(self.dim, generator=g) * 0.1

        embed_text_raw = embed_text

    def _make_loop(self, tmp_path):
        from runtime.config import default_config
        from runtime.loop import LivingMemoryLoop
        cfg = default_config()
        cfg.update(num_nodes=32, input_dim=16, num_infer_steps=5,
                   embedder=self._MockEmbedder(), auto_snapshot=False,
                   snapshot_dir=str(tmp_path / "s"),
                   archive_enabled=False)
        return LivingMemoryLoop(cfg)

    def _read_events(self, bus_file):
        if not os.path.exists(bus_file):
            return []
        out = []
        with open(bus_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def test_dream_publishes_doubt_consolidation_event(
            self, tmp_path, monkeypatch):
        """梦期巩固结果发事件：lms.dream_complete + lms.doubt_consolidation
        （confirm/supersedes 结果 + supersedes 记录——§4.3 坑 7 根治）。"""
        bus_file = str(tmp_path / "event_bus.jsonl")
        monkeypatch.setenv("LMS_BUS_FILE", bus_file)
        from runtime import bus_events
        bus_events.reset_publisher_for_test()
        try:
            loop = self._make_loop(tmp_path)
            loop.process_turn("用户: 蓝色小橘猫很可爱")
            entry = list(loop.memory.iter_episodic())[-1]
            entry.labile = True
            entry.labile_since = time.time()
            entry.violated_by = "蓝色小橘猫不可爱"
            result = loop.dream(n_steps=5)
            assert result["doubt_review"]["rewritten"] >= 1
            events = self._read_events(bus_file)
            types = [ev["event_type"] for ev in events]
            assert "lms.dream_complete" in types, "梦期完成事件"
            assert "lms.doubt_consolidation" in types, "巩固结果事件"
            dc = next(ev for ev in events
                      if ev["event_type"] == "lms.doubt_consolidation")
            assert dc["result"] == "OK"
            assert dc["schema_version"] == "1.1"
            assert dc["producer"] == "lms"
            assert dc["payload"]["doubt_review"]["rewritten"] >= 1
            assert dc["payload"]["consolidation"]["supersedes"], \
                "payload 应含 supersedes 记录明细"
        finally:
            bus_events.reset_publisher_for_test()

    def test_verification_events_published_when_enabled(
            self, tmp_path, monkeypatch):
        """验证链事件发布（LMS_VERIFICATION_CHAIN_ENABLED=1）：写侧应用
        conflict 后发 lms.verification（verify_requested/result + 计数）。"""
        bus_file = str(tmp_path / "event_bus.jsonl")
        monkeypatch.setenv("LMS_BUS_FILE", bus_file)
        monkeypatch.setenv("LMS_VERIFICATION_CHAIN_ENABLED", "1")
        from runtime import bus_events
        bus_events.reset_publisher_for_test()
        try:
            loop = self._make_loop(tmp_path)
            loop.process_turn("用户: 蓝色小橘猫很可爱")
            entry = list(loop.memory.iter_episodic())[-1]
            req = loop.verification_chain.register(
                entry_ref="e-new", register_query="蓝色小橘猫不可爱")
            loop.verification_chain.verify(
                req.idempotency_key, reference_claims=["蓝色小橘猫很可爱"])
            assert loop.verification_chain.pending_conflicts()
            loop.process_turn("用户: 无关内容")  # 触发 _apply_verification_conflicts
            events = self._read_events(bus_file)
            vtypes = [ev["event_type"] for ev in events
                      if ev["event_type"] == "lms.verification"]
            assert vtypes, "验证链事件应已发布"
            v = [ev for ev in events
                 if ev["event_type"] == "lms.verification"][-1]
            assert v["payload"]["snapshot"]["conflicts_pending"] == 0
            assert v["payload"]["events"], "应含 verify 事件明细"
            # 写侧应用成功：目标 [doubt] conflict → labile → 补入队 → 巩固
            # 期受控改写（E3 根因 3 修复，2026-08-20）→ 终态 superseded
            assert getattr(entry, "doubt_state", "stable") == "superseded"
            assert entry.rebuttal_count == 1
            assert entry.labile is False  # labile 时相已复位（改写已落库）
        finally:
            bus_events.reset_publisher_for_test()

    def test_verification_events_not_published_when_chain_off(
            self, tmp_path, monkeypatch):
        """验证链开关默认关：零发布（lms.verification 不出现在总线）。"""
        bus_file = str(tmp_path / "event_bus.jsonl")
        monkeypatch.setenv("LMS_BUS_FILE", bus_file)
        from runtime import bus_events
        bus_events.reset_publisher_for_test()
        try:
            loop = self._make_loop(tmp_path)
            assert loop.verification_chain.enabled is False
            loop.process_turn("用户: 蓝色小橘猫很可爱")
            loop.process_turn("用户: 无关内容")
            events = self._read_events(bus_file)
            assert not any(ev["event_type"] == "lms.verification"
                           for ev in events), "链关时不应发布验证事件"
        finally:
            bus_events.reset_publisher_for_test()

    def test_bus_events_quick_api(self):
        """bus_events 新增发布接口可直接调用（含 _sanitize 裁剪）。"""
        import tempfile
        from runtime.bus_events import (
            BusEventPublisher, publish_doubt_consolidation,
            publish_verification_events,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pub = BusEventPublisher(bus_file=os.path.join(tmpdir, "b.jsonl"))
            assert pub.publish_doubt_consolidation({
                "doubt_review": {"rewritten": 1},
                "consolidation": {"supersedes": [{"original": "x"}]},
            }) is True
            assert pub.publish_verification_events({
                "events": [{"type": "verify_requested"}],
            }) is True
            recs = pub._get_writer().read_all()
            assert len(recs) == 2
            assert recs[0]["event_type"] == "lms.doubt_consolidation"
            assert recs[1]["event_type"] == "lms.verification"

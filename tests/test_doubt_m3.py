# -*- coding: utf-8 -*-
"""M3-1 单测：三时相怀疑状态机 + rebuttal-consistency 字段原生 + 验证链骨架
（核心重建规格 v2 §2.1 / §2.2 / §2.3 / §5.2 / §6.2）。

本文件是 ``core/doubt/claims.json`` / ``MODULE_CLAIMS`` 全部 claim 的机器
验证（§5.2 D 模式：claim 与实现不一致 → 测试红）。

用例组映射（规格 §6.2"怀疑原生单测"）：
  1. 三时相状态机  → TestThreePhaseStateMachine
       （检索时→labile 投影不落库；注入时→suspect；巩固时→resolve_labile；
         labile 非持久）
  2. rebuttal-consistency 字段 → TestRebuttalConsistencyField
       （检索只读该字段；写入口仅 ingest/consolidation；非法写者被拒）
  3. 验证链骨架（幂等/防伪四维/开关默认关）→ TestVerificationChainSkeleton
  4. loop 集成（原生字段初始化 / 注入时怀疑 / 冲突同步 / labile 窗口）
     → TestLoopIntegrationM3

运行方式：pytest rewrite-ws/tests/test_doubt_m3.py -v（生产 venv）。
"""

import os
import sys
import time

# 确保项目根目录在 Python 路径中（可从任意 cwd 运行）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
import torch

from core.doubt.rebuttal_field import (
    IllegalWriterError,
    get_rebuttal_consistency,
    init_rebuttal_consistency,
    is_legal_writer,
    read_view,
    record_rebuttal_native,
    update_consistency,
)
from core.doubt.state_machine import (
    DoubtPhase,
    DoubtStateMachine,
    EntryDoubtState,
    compute_rebuttal_hit,
    doubt_injection_enabled,
    is_high_surprise,
)
from core.doubt.verification_chain import (
    ConflictKind,
    VerificationChain,
    VerificationEventType,
    VerdictType,
    verification_chain_enabled,
    verification_key,
)
from core.doubt import reconsolidation
from core.recall.guard import (
    _ENTRY_FIELD_NAMES,
    entry_fingerprint,
)
from core.hippocampus.memory import MemoryManager, EpisodicEntry


# ---------------------------------------------------------------------------
# 测试小工具
# ---------------------------------------------------------------------------

def _make_entry(text="测试记忆", **kw):
    """最小条目替身（带 M3-1 原生字段的 EpisodicEntry）。"""
    kw.setdefault("source_trust", 1.0)
    kw.setdefault("rebuttal_count", 0)
    kw.setdefault("reference_count", 0)
    kw.setdefault("surprise", 1.0)
    kw.setdefault("turn", 0)
    kw.setdefault("confidence", 1.0)
    kw.setdefault("labile", False)
    kw.setdefault("labile_since", None)
    kw.setdefault("violated_by", None)
    kw.setdefault("recall_count", 0)
    return EpisodicEntry(text=text, semantic_vector=torch.zeros(8), **kw)


def _entry_fields(entry):
    """条目全字段快照（与守卫指纹同字段集——断言零改写用）。"""
    return tuple((name, getattr(entry, name, None))
                 for name in _ENTRY_FIELD_NAMES)


def _fresh_chain(enabled=True):
    return VerificationChain(enabled=enabled, window=60.0)


# ===========================================================================
# 1. 三时相怀疑状态机
# ===========================================================================

class TestThreePhaseStateMachine:
    """检索时→labile 投影不落库 / 注入时→suspect / 巩固时→resolve_labile。"""

    # -- 检索时相（只读四不变） ------------------------------------------- #

    def test_retrieval_phase_never_mutates_entry(self):
        """检索时相绝不 setattr 条目：retrieval_hit / retrieval_projection
        后条目全字段与执行前一致（§2.1 只读四不变根基）。"""
        sm = DoubtStateMachine()
        e = _make_entry(text="用户: 记忆A", labile=True, rebuttal_count=2)
        before = _entry_fields(e)
        sm.retrieval_hit(e, score=0.9)
        sm.retrieval_projection([(0.9, e)], precision_adapt=None)
        after = _entry_fields(e)
        assert before == after, "检索时相必须零改写条目"

    def test_retrieval_projection_shape_frozen(self):
        """投影结构与 M2 project_suspicion 逐键同构（§4.1 形态冻结）。"""
        sm = DoubtStateMachine()
        e1 = _make_entry(text="labile条目", labile=True)
        e2 = _make_entry(text="稳定条目")
        proj = sm.retrieval_projection([(0.9, e1), (0.8, e2)])
        assert set(proj) == {"labile", "rebuttal_pending",
                             "verdict_suspect", "summary"}
        assert proj["summary"]["total"] == 2
        assert len(proj["labile"]) == 1

    def test_labile_is_transient_not_persistent(self):
        """labile 是时相状态不是持久状态：检索命中只进内存态窗口，条目
        本身 labile/doubt_state 不被改写；持久层只存三态。"""
        sm = DoubtStateMachine(labile_ttl=60.0)
        e = _make_entry(text="记忆")
        sm.retrieval_hit(e, score=0.85)
        # 内存态窗口有记录
        snap = sm.labile_window_snapshot()
        assert any(s["text"] == "记忆" for s in snap)
        # 条目本身零改写（labile 保持 False——时相状态不进持久字段）
        assert e.labile is False
        assert getattr(e, "doubt_state", "stable") == "stable"
        # 持久层只存三态枚举（labile 不在其中）
        assert [s.value for s in EntryDoubtState] == \
            ["stable", "suspect", "superseded"]

    def test_labile_window_ttl_expiry(self):
        """labile 窗口 TTL：超时条目从内存态窗口清除（不落库）。"""
        sm = DoubtStateMachine(labile_ttl=0.01)
        e = _make_entry(text="超时记忆")
        sm.retrieval_hit(e)
        assert len(sm.labile_window_snapshot()) == 1
        time.sleep(0.05)
        assert sm.labile_window_snapshot() == []

    # -- 注入时相（写侧） ------------------------------------------------- #

    def test_injection_high_surprise_marks_suspect(self):
        """高 surprise（> factor×J_target）→ 标 suspect + 验证链登记。"""
        chain = _fresh_chain()
        sm = DoubtStateMachine(verification_chain=chain)
        e = _make_entry(text="高惊讶新条目")
        outcome = sm.injection_check(
            e, surprise=50.0, j_target=10.0,  # 50 > 1.0×10
            verification_chain=chain)
        assert outcome == EntryDoubtState.SUSPECT.value
        assert e.doubt_state == EntryDoubtState.SUSPECT.value
        assert sm.injection_suspect_marked == 1
        # 验证链已登记（开关开）
        assert chain.pending_count() >= 1

    def test_injection_normal_stays_stable(self):
        """无高 surprise 且无 rebuttal 命中 → 保持 stable（不误标）。"""
        sm = DoubtStateMachine()
        e = _make_entry(text="普通新条目")
        outcome = sm.injection_check(
            e, surprise=0.1, j_target=10.0, rebuttal_hit=False)
        assert outcome == EntryDoubtState.STABLE.value
        assert getattr(e, "doubt_state", "stable") == "stable"
        assert sm.injection_suspect_marked == 0

    def test_injection_rebuttal_hit_marks_suspect(self):
        """存在 rebuttal → 标 suspect（即使 surprise 不高）。"""
        sm = DoubtStateMachine()
        e = _make_entry(text="反驳旧记忆的新条目")
        outcome = sm.injection_check(
            e, surprise=0.01, j_target=10.0, rebuttal_hit=True)
        assert outcome == EntryDoubtState.SUSPECT.value

    def test_injection_verification_chain_default_off_no_register(self):
        """验证链开关默认关：注入时 suspect 但 register 返回 None（零参与）。"""
        chain = VerificationChain()  # 默认关（§5.3）
        sm = DoubtStateMachine(verification_chain=chain)
        e = _make_entry(text="高惊讶")
        outcome = sm.injection_check(
            e, surprise=50.0, j_target=10.0, verification_chain=chain)
        assert outcome == EntryDoubtState.SUSPECT.value, \
            "注入时怀疑是机制本体（默认开），suspect 标记不依赖验证链开关"
        assert chain.pending_count() == 0, "验证链关时零登记"

    def test_disabled_state_machine_zero_participation(self):
        """开关关（LMS_DOUBT_INJECTION_ENABLED=0）→ 注入检查零参与。"""
        sm = DoubtStateMachine(enabled=False)
        e = _make_entry(text="任意条目")
        outcome = sm.injection_check(
            e, surprise=999.0, j_target=1.0, rebuttal_hit=True)
        assert outcome == EntryDoubtState.STABLE.value
        assert getattr(e, "doubt_state", "stable") == "stable"
        assert sm.injection_checks == 0

    def test_transitions_only_from_write_side(self):
        """状态转移只能由写侧时相驱动：检索时相零转移（即使条目已 suspect）。"""
        sm = DoubtStateMachine()
        e = _make_entry(text="已suspect条目")
        e.doubt_state = EntryDoubtState.SUSPECT.value
        before = _entry_fields(e)
        sm.retrieval_hit(e, score=0.9)
        sm.retrieval_projection([(0.9, e)])
        assert _entry_fields(e) == before, \
            "检索时相不得驱动任何状态转移（suspect 保持不变）"
        assert e.doubt_state == EntryDoubtState.SUSPECT.value

    def test_compute_rebuttal_hit_overlap(self):
        """rebuttal 命中判定：新条目与已证伪/去稳定化旧条目内容重叠。"""
        old = _make_entry(text="用户: 我喜欢蓝色小橘猫", rebuttal_count=1)
        # 新条目包含旧条目全文（子串重叠——与 doubt_ingest 同工程惯例）
        new = _make_entry(text="用户: 我喜欢蓝色小橘猫 但后来发现不讨喜")
        assert compute_rebuttal_hit(new, [old, new]) is True
        assert compute_rebuttal_hit(_make_entry(text="无关内容"), [old]) is False

    # -- 巩固时相（接口本段定义；M6 做梦循环） ----------------------------- #

    def test_consolidation_rewritten_when_violated(self):
        """巩固时 resolve_labile：有证伪证据 → 'rewritten' → superseded。"""
        sm = DoubtStateMachine()
        e = _make_entry(text="被证伪的记忆", labile=True,
                        violated_by="矛盾证据", confidence=0.9)
        e.labile_since = time.time()
        res = sm.consolidation_resolve(e)
        assert res["outcome"] == "rewritten"
        assert res["doubt_state"] == EntryDoubtState.SUPERSEDED.value
        assert e.labile is False  # labile 复位（时相状态结束）

    def test_consolidation_kept_window_in(self):
        """窗口内无证据 → 'kept'：回 stable，confidence ×1.02 重巩固。"""
        sm = DoubtStateMachine()
        e = _make_entry(text="窗口内记忆", confidence=0.5, labile=True)
        e.labile_since = time.time()
        res = sm.consolidation_resolve(e)
        assert res["outcome"] == "kept"
        assert res["doubt_state"] == EntryDoubtState.STABLE.value
        assert res["confidence"] == pytest.approx(0.5 * 1.02, abs=1e-3)

    def test_consolidation_downgraded_timeout(self):
        """超时无证据 → 'downgraded'：回 stable，confidence ×0.98 折损。"""
        sm = DoubtStateMachine()
        e = _make_entry(text="超时记忆", confidence=0.5, labile=True)
        e.labile_since = time.time() - 999999
        res = sm.consolidation_resolve(e)
        assert res["outcome"] == "downgraded"
        assert res["doubt_state"] == EntryDoubtState.STABLE.value
        assert res["confidence"] == pytest.approx(0.5 * 0.98, abs=1e-3)

    def test_consolidation_verdict_confirm_and_conflict(self):
        """验证裁决驱动巩固：confirm → kept；conflict → superseded。"""
        sm = DoubtStateMachine()
        e1 = _make_entry(text="验证通过的记忆", labile=True)
        e1.labile_since = time.time()
        res1 = sm.consolidation_resolve(e1, verdict=VerdictType.CONFIRM.value)
        assert res1["outcome"] == "kept"
        assert res1["doubt_state"] == EntryDoubtState.STABLE.value

        e2 = _make_entry(text="验证冲突的记忆", labile=True)
        e2.labile_since = time.time()
        res2 = sm.consolidation_resolve(e2, verdict=VerdictType.CONFLICT.value)
        assert res2["outcome"] == "rewritten"
        assert res2["doubt_state"] == EntryDoubtState.SUPERSEDED.value

    def test_consolidation_updates_native_field_writer(self):
        """巩固时相写原生字段：updated_by='consolidation'（§2.3 合法写者）。"""
        sm = DoubtStateMachine()
        e = _make_entry(text="巩固记忆", labile=True)
        e.labile_since = time.time()
        init_rebuttal_consistency(e)  # ingest 初始化（updated_by='ingest'）
        sm.consolidation_resolve(e)
        rc = get_rebuttal_consistency(e)
        assert rc["updated_by"] == "consolidation"
        assert rc["consistency"] == pytest.approx(e.confidence, abs=1e-3)


# ===========================================================================
# 2. rebuttal-consistency 字段原生（§2.3）
# ===========================================================================

class TestRebuttalConsistencyField:
    """检索只读该字段；写入口仅 ingest/consolidation；非法写者被拒。"""

    def test_ingest_init_and_rebuttal_sync(self):
        """写入口初始化：updated_by='ingest'、rebuttals=[]、consistency=0.0；
        证伪登记同步结构化字段与平坦字段。"""
        e = _make_entry()
        rc = init_rebuttal_consistency(e)
        assert rc["rebuttals"] == []
        assert rc["consistency"] == 0.0
        assert rc["updated_by"] == "ingest"
        # 证伪登记（写侧 ingest）：平坦字段 + 结构化字段同步
        ok = record_rebuttal_native(
            e, rebuttal_id="rebuttal-001", violated_by="证据文本")
        assert ok is True
        assert e.rebuttal_count == 1          # 平坦字段（既有行为）
        assert e.violated_by == "证据文本"
        rc2 = get_rebuttal_consistency(e)
        assert rc2["rebuttals"] == ["rebuttal-001"]
        assert rc2["updated_by"] == "ingest"
        assert rc2["consistency"] == pytest.approx(e.confidence, abs=1e-3)

    def test_retrieval_writer_rejected(self):
        """铁律：updated_by='retrieval' 必拒（IllegalWriterError）。"""
        e = _make_entry()
        init_rebuttal_consistency(e)
        with pytest.raises(IllegalWriterError):
            update_consistency(e, consistency=0.5, updated_by="retrieval")
        with pytest.raises(IllegalWriterError):
            record_rebuttal_native(e, updated_by="retrieval")
        assert is_legal_writer("retrieval") is False
        assert is_legal_writer("ingest") is True
        assert is_legal_writer("consolidation") is True

    def test_consolidation_writer_allowed(self):
        """写入口仅 ingest/consolidation：consolidation 合法。"""
        e = _make_entry()
        init_rebuttal_consistency(e)
        assert update_consistency(
            e, consistency=0.8, updated_by="consolidation") is True
        rc = get_rebuttal_consistency(e)
        assert rc["consistency"] == 0.8
        assert rc["updated_by"] == "consolidation"

    def test_read_returns_deep_copy(self):
        """读取返回深拷贝：调用方改副本对条目零影响（读不可能变写）。"""
        e = _make_entry()
        init_rebuttal_consistency(e)
        rc = get_rebuttal_consistency(e)
        rc["rebuttals"].append("伪造")
        rc["updated_by"] = "retrieval"  # 试图伪造写者
        after = get_rebuttal_consistency(e)
        assert after["rebuttals"] == []
        assert after["updated_by"] == "ingest"

    def test_read_view_shape(self):
        """read_view 精简只读视图（suspicion 投影/响应注解用）。"""
        e = _make_entry()
        init_rebuttal_consistency(e)
        v = read_view(e)
        assert set(v) == {"rebuttals", "consistency", "updated_at",
                          "updated_by"}
        assert v["updated_by"] == "ingest"

    def test_memory_store_episodic_initializes_native_field(self):
        """MemoryManager.store_episodic（写入口）初始化原生字段。"""
        mm = MemoryManager(num_nodes=8)
        mm.store_episodic("用户: 原生字段测试", torch.ones(8),
                          surprise=1.0, turn=0)
        e = list(mm.iter_episodic())[0]
        rc = get_rebuttal_consistency(e)
        assert rc["rebuttals"] == []
        assert rc["updated_by"] == "ingest"
        assert getattr(e, "doubt_state", "stable") == "stable"

    def test_mark_labile_syncs_native_field(self):
        """mark_labile（写侧去稳定化）同步原生字段（证伪 + 溯源）。"""
        e = _make_entry(text="旧记忆", reference_count=9)
        init_rebuttal_consistency(e)
        assert reconsolidation.mark_labile(e, violated_by="违反它的输入")
        assert e.labile is True
        assert e.rebuttal_count == 1          # 平坦字段（既有行为）
        rc = get_rebuttal_consistency(e)
        assert rc["updated_by"] == "ingest"   # 写侧时相溯源
        assert rc["consistency"] == pytest.approx(e.confidence, abs=1e-3)

    def test_entry_fingerprint_captures_native_mutation(self):
        """四不变守卫指纹抓原生字段改写（检索泄漏防线延伸）。"""
        e = _make_entry()
        init_rebuttal_consistency(e)
        before = entry_fingerprint(e)
        e.rebuttal_consistency["updated_by"] = "retrieval"  # 模拟泄漏
        after = entry_fingerprint(e)
        assert before != after

    def test_snapshot_scan_no_retrieval_writer(self):
        """§5.1/§2.5 快照一致性扫描：条目无 updated_by='retrieval' 记录。"""
        mm = MemoryManager(num_nodes=8)
        mm.store_episodic("用户: 记忆", torch.ones(8), surprise=1.0, turn=0)
        for e in mm.iter_episodic():
            rc = get_rebuttal_consistency(e)
            assert rc["updated_by"] != "retrieval"
            assert getattr(e, "updated_by", None) != "retrieval"


# ===========================================================================
# 3. 验证链原生骨架（§2.2 接口/事件/幂等键协议/防伪四维）
# ===========================================================================

class TestVerificationChainSkeleton:
    """幂等 / 防伪独立四维 / 开关默认关。"""

    def test_switch_default_off(self):
        """验证链开关默认 false（§5.3 写侧默认保守）；关时 register 返回 None。"""
        chain = VerificationChain()
        assert chain.enabled is False
        req = chain.register(
            entry_ref="e1", register_query="问题A")
        assert req is None, "开关关 → 零参与"

    def test_switch_env_on(self, monkeypatch):
        """显式开启（env LMS_VERIFICATION_CHAIN_ENABLED=1）→ 生效。"""
        monkeypatch.setenv("LMS_VERIFICATION_CHAIN_ENABLED", "1")
        assert verification_chain_enabled() is True
        chain = VerificationChain()
        assert chain.enabled is True

    def test_same_key_resend_no_duplicate(self):
        """同幂等键重发不重复登记：同一（entry_ref+query+phase）命中同一键。"""
        chain = _fresh_chain()
        req1 = chain.register(
            entry_ref="e1", register_query="应开启方案A")
        req2 = chain.register(
            entry_ref="e1", register_query="应开启方案A")
        assert req1 is not None and req2 is not None
        assert req1.idempotency_key == req2.idempotency_key
        assert req1 is req2, "幂等命中必须原样返回原请求（不重复登记）"
        assert chain.pending_count() == 1

    def test_result_idempotent_single_record(self):
        """结果登记本身幂等：同键同验证只产生一条记录。"""
        chain = _fresh_chain()
        req = chain.register(entry_ref="e1", register_query="Q")
        res1 = chain.submit_result(
            req.idempotency_key, verdict=VerdictType.CONFIRM.value)
        res2 = chain.submit_result(
            req.idempotency_key, verdict=VerdictType.CONFIRM.value)
        assert res1 is not None
        assert res1 is res2, "同键同验证只一条记录（客户端超时≠未写入）"
        assert chain.lookup(req.idempotency_key) is res1

    def test_timeout_retry_returns_original(self):
        """"客户端超时重试"场景：重试同键命中原结果，不重复登记
        （60s 竞态教训同构——store 幂等键机制的验证链版本）。"""
        chain = _fresh_chain()
        req = chain.register(entry_ref="e1", register_query="Q")
        # 首次处理
        first = chain.submit_result(
            req.idempotency_key, verdict=VerdictType.CONFLICT.value,
            conflict_kind=ConflictKind.DIRECTIONAL.value)
        # 客户端超时 → 用同一幂等键重试
        retry = chain.submit_result(
            req.idempotency_key, verdict=VerdictType.CONFLICT.value,
            conflict_kind=ConflictKind.DIRECTIONAL.value)
        assert retry is first
        assert chain.pending_count() == 0

    def test_result_for_unknown_key_none(self):
        """未登记的键提交结果 → None（拒绝无登记结果的"空降"验证）。"""
        chain = _fresh_chain()
        assert chain.submit_result(
            "verif_unknown", verdict=VerdictType.CONFIRM.value) is None

    def test_anti_fraud_independence_dimensions(self):
        """防伪独立四维（§2.2）：query/批次/信号各自独立。"""
        chain = _fresh_chain()
        # ② query 独立：验证 query ≠ 登记 query（不同构造）
        req = chain.register(
            entry_ref="e1", register_query="应开启方案A")
        assert req.verify_query != req.register_query
        assert "换个角度" in req.verify_query
        # ③ 批次独立：验证批次 ≠ 登记批次
        verify_batch = chain.independent_batch(req.batch_id)
        assert verify_batch != req.batch_id
        # ④ 信号独立：验证通道 ≠ 登记通道
        assert chain.channel_for("register.injection") == \
            "verify.injection"
        assert chain.channel_for("register.injection") != \
            "register.injection"
        # ① 端点独立：通道命名空间隔离（register.* vs verify.*）
        assert req.channel.startswith("register.")
        assert chain.channel_for(req.channel).startswith("verify.")

    def test_verification_key_deterministic(self):
        """幂等键确定性：同输入同键（协议硬约束——重试必须带同一键）。"""
        k1 = verification_key("e1", "Q", "injection")
        k2 = verification_key("e1", "Q", "injection")
        k3 = verification_key("e1", "Q不同", "injection")
        assert k1 == k2
        assert k1.startswith("verif_")
        assert k1 != k3

    def test_events_recorded(self):
        """事件流：登记/结果事件带类型（§4.3 落沙必发事件数据源）。"""
        chain = _fresh_chain()
        req = chain.register(entry_ref="e1", register_query="Q")
        chain.submit_result(req.idempotency_key,
                            verdict=VerdictType.CONFIRM.value)
        events = chain.events()
        types = [ev["type"] for ev in events]
        assert VerificationEventType.VERIFY_REQUESTED.value in types
        assert VerificationEventType.VERIFY_RESULT.value in types


# ===========================================================================
# 4. loop 集成（M3-1 接线）
# ===========================================================================

class TestLoopIntegrationM3:
    """loop 层：原生字段初始化 / 注入时怀疑 / 冲突同步 / labile 窗口。"""

    class _MockEmbedder:
        """带 embed_text 的确定性嵌入器（SimpleEmbedder 无 embed_text——
        loop 写路径需 hasattr(embedder,'embed_text')；同 test_recall_m2 先例）。"""

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

    def _make_loop(self, tmp_path, monkeypatch=None, archive=False):
        from runtime.config import default_config
        from runtime.loop import LivingMemoryLoop
        cfg = default_config()
        cfg.update(num_nodes=32, input_dim=16, num_infer_steps=5,
                   embedder=self._MockEmbedder(16), auto_snapshot=False,
                   snapshot_dir=str(tmp_path / "s"),
                   archive_enabled=archive)
        return LivingMemoryLoop(cfg)

    def test_loop_exposes_native_doubt(self, tmp_path):
        """loop 暴露三时相状态机 + 验证链；get_status 有 doubt_native 块。"""
        loop = self._make_loop(tmp_path)
        assert loop.doubt_state is not None
        assert loop.verification_chain is not None
        assert loop.verification_chain.enabled is False  # 默认关
        loop.process_turn("用户: 你好")
        status = loop.get_status()
        assert "doubt_native" in status
        assert status["doubt_native"]["enabled"] is True
        assert "verification_chain" in status["doubt_native"]

    def test_store_episodic_via_loop_initializes_native_field(self, tmp_path):
        """loop 写路径（process_turn → store_episodic）初始化原生字段。"""
        loop = self._make_loop(tmp_path)
        loop.process_turn("用户: 原生字段测试")
        e = list(loop.memory.iter_episodic())[-1]
        rc = get_rebuttal_consistency(e)
        assert rc["updated_by"] == "ingest"
        assert rc["rebuttals"] == []
        assert getattr(e, "doubt_state", "stable") == "stable"

    def test_loop_injection_check_marks_suspect(self, tmp_path):
        """注入时怀疑（写侧）经 loop 接线：高 surprise → suspect。"""
        loop = self._make_loop(tmp_path)
        loop.process_turn("用户: 第一条")
        entry = list(loop.memory.iter_episodic())[-1]
        outcome = loop.doubt_state.injection_check(
            entry, surprise=999.0, j_target=1.0,
            verification_chain=loop.verification_chain)
        assert outcome == EntryDoubtState.SUSPECT.value
        assert entry.doubt_state == EntryDoubtState.SUSPECT.value

    def test_loop_recall_populates_labile_window(self, tmp_path):
        """检索时相经 loop 接线：recall 命中进内存态 labile 窗口。"""
        loop = self._make_loop(tmp_path)
        loop.process_turn("用户: 你好，记忆A")
        loop.process_turn("用户: 你好，记忆B")
        loop.recall_episodic_readonly("记忆A", k=5)
        snap = loop.doubt_state.labile_window_snapshot()
        assert len(snap) >= 1, "检索命中必须进 labile 窗口（内存态）"
        # 条目零改写（检索只读）
        for e in loop.memory.iter_episodic():
            assert getattr(e, "doubt_state", "stable") == "stable"

    def test_loop_conflict_syncs_native_field(self, tmp_path):
        """[doubt] conflict 证伪经 loop：平坦字段 + 原生字段同步（写侧）。"""
        loop = self._make_loop(tmp_path)
        loop.process_turn("用户: 我喜欢蓝色小橘猫")
        loop.process_turn("[doubt] conflict: 我喜欢蓝色小橘猫")
        e = list(loop.memory.iter_episodic())[-1]
        assert e.rebuttal_count == 1
        assert e.labile is True
        rc = get_rebuttal_consistency(e)
        assert rc["updated_by"] == "ingest"
        # consistency = 证伪时点快照（mark_rebutted 后置信度=0.0；随后
        # _retrieve_episodic 的正向佐证把平坦 confidence 抬到 0.5——结构化
        # 字段保存的是写侧时相写入值，不被后续只读路径改写）
        assert rc["consistency"] == 0.0
        assert e.confidence == pytest.approx(0.5, abs=1e-3)

    def test_recall_readonly_four_invariants_hold_with_native(self, tmp_path):
        """检索带原生字段后四不变仍成立（recall 零增量——M2 防线延续）。"""
        loop = self._make_loop(tmp_path)
        loop.process_turn("用户: 你好")
        loop.process_turn("用户: 你好")
        before = {id(e): _entry_fields(e)
                  for e in loop.memory.iter_episodic()}
        loop.recall_merged_readonly("用户: 你好", k=5)
        after = {id(e): _entry_fields(e)
                 for e in loop.memory.iter_episodic()}
        assert after == before, "recall 后条目字段必须零改写（含原生字段）"

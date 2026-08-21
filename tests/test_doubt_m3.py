# -*- coding: utf-8 -*-
"""M3-1/M3-2 单测：三时相怀疑状态机 + rebuttal-consistency 字段原生 +
验证链（骨架 + 矛盾判定三选一/元数据排除/幂等/全链）
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
  4. 矛盾判定三选一（方向/数值/否定各一例）→ TestContradictionJudgment
  5. 元数据排除假阳性演练（时间戳前缀/同义复述/来源互补不触发）
     → TestMetadataExclusion
  6. 验证链全链（草稿→独立验证→修正 + provenance + 幂等 60s 竞态）
     → TestVerificationChainFullChain
  7. 验证链写侧（CONFLICT → [doubt] conflict → labile）→ TestVerificationWriteSide
  8. loop 集成（原生字段初始化 / 注入时怀疑 / 冲突同步 / labile 窗口）
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
    VerificationRequest,
    VerificationResult,
    VerdictType,
    is_contradiction,
    judge_contradiction,
    verdict_for,
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
# 3.5 矛盾判定三选一（§2.2 / §6.2 假阳性演练：真矛盾各一例触发）
# ===========================================================================

class TestContradictionJudgment:
    """矛盾判定三选一：方向/数值/否定三类**明确矛盾**才判 conflict。"""

    def test_directional_contradiction(self):
        """方向矛盾：同一事实两条断言结论相反（规格用例"应开启 A vs 应关闭 A"）。"""
        assert is_contradiction("应开启方案A", "应关闭方案A") == \
            ConflictKind.DIRECTIONAL

    def test_directional_opposite_words(self):
        """方向矛盾：支持 vs 反对（同事实同宾语）。"""
        assert is_contradiction("支持采用方案B", "反对采用方案B") == \
            ConflictKind.DIRECTIONAL

    def test_directional_english(self):
        """方向矛盾（英文词对）：enable vs disable。"""
        assert is_contradiction("enable module A", "disable module A") == \
            ConflictKind.DIRECTIONAL

    def test_directional_requires_same_core(self):
        """方向矛盾必须同一事实：开启方案A vs 关闭方案B 核心不同 → 不判。"""
        assert is_contradiction("开启方案A", "关闭方案B") == \
            ConflictKind.NOT_A_CONFLICT

    def test_numeric_contradiction(self):
        """数值矛盾：同一量同量纲区间不相交（规格用例 1500 vs 800）。"""
        assert is_contradiction("成本为1500元", "成本为800元") == \
            ConflictKind.NUMERIC

    def test_numeric_range_disjoint(self):
        """数值矛盾：范围断言 1500-1800 vs 800 区间不相交。"""
        assert is_contradiction("价格1500-1800元", "价格800元") == \
            ConflictKind.NUMERIC

    def test_numeric_overlapping_interval_not_conflict(self):
        """区间相交 → 不判矛盾（1500-1800 与 1600 相交）。"""
        assert is_contradiction("价格1500-1800元", "价格1600元") == \
            ConflictKind.NOT_A_CONFLICT

    def test_numeric_different_dimension_not_conflict(self):
        """不同量纲同单位不判：1500元 vs 800公里（规格"同量纲同单位"门槛）。"""
        assert is_contradiction("成本1500元", "距离800公里") == \
            ConflictKind.NOT_A_CONFLICT

    def test_numeric_no_shared_topic_not_conflict(self):
        """无共享主题的孤立数字不判（无"同一事实"基础）。"""
        assert is_contradiction("价格是1500", "销量是800") == \
            ConflictKind.NOT_A_CONFLICT

    def test_negation_existence(self):
        """否定矛盾：存在 vs 不存在（存在性直接否定）。"""
        assert is_contradiction("存在蓝色小橘猫", "不存在蓝色小橘猫") == \
            ConflictKind.NEGATION

    def test_negation_validity(self):
        """否定矛盾：可行 vs 不可行（成立性直接否定）。"""
        assert is_contradiction("方案A可行", "方案A不可行") == \
            ConflictKind.NEGATION

    def test_negation_preference(self):
        """否定矛盾：喜欢 vs 不喜欢。"""
        assert is_contradiction("我喜欢蓝色", "我不喜欢蓝色") == \
            ConflictKind.NEGATION

    def test_unrelated_claims_not_conflict(self):
        """无关断言（无主题重叠）不判矛盾。"""
        assert is_contradiction("今天天气很好", "股票涨了3个点") == \
            ConflictKind.NOT_A_CONFLICT

    def test_verdict_for_maps_three_way(self):
        """verdict_for：三类真矛盾 → conflict；其余 → confirm。"""
        assert verdict_for(ConflictKind.DIRECTIONAL.value) == \
            VerdictType.CONFLICT.value
        assert verdict_for(ConflictKind.NUMERIC.value) == \
            VerdictType.CONFLICT.value
        assert verdict_for(ConflictKind.NEGATION.value) == \
            VerdictType.CONFLICT.value
        assert verdict_for(ConflictKind.NOT_A_CONFLICT.value) == \
            VerdictType.CONFIRM.value
        assert verdict_for(None) == VerdictType.CONFIRM.value


# ===========================================================================
# 3.6 元数据排除（P0 假冲突根因②根治——假阳性演练红线）
# ===========================================================================

class TestMetadataExclusion:
    """元数据碰撞一律不判矛盾（§2.2 元数据排除 / §6.2 假阳性演练固化）。"""

    def test_timestamp_prefix_collision_not_conflict(self):
        """时间戳前缀碰撞（P0 事故残留形态——"开工吧 vs System Gate"用例）：
        两条文本仅日期/星期前缀不同 → 不判矛盾 + metadata_excluded。"""
        a = "[Thu 2026-08-06 00:11 GMT+8] 开工吧，我还夜猫子"
        b = "[Fri 2026-08-07 00:11 GMT+8] 开工吧，我还夜猫子"
        j = judge_contradiction(a, b)
        assert j["kind"] is ConflictKind.NOT_A_CONFLICT
        assert j["metadata_excluded"] is True
        assert j["reason"] == "datetime_collision"

    def test_iso_date_prefix_collision_not_conflict(self):
        """ISO 日期前缀碰撞（旧 overlapMatch 纯子串碰撞会误判的形态）。"""
        assert is_contradiction("2026-08-06 开工", "2026-08-07 开工") == \
            ConflictKind.NOT_A_CONFLICT

    def test_chinese_date_prefix_collision_not_conflict(self):
        """中文日期前缀碰撞（8月6日 vs 8月7日）。"""
        assert is_contradiction("8月6日 开工", "8月7日 开工") == \
            ConflictKind.NOT_A_CONFLICT

    def test_synonymous_restatement_not_conflict(self):
        """同义复述（不同措辞同一结论）不判矛盾。"""
        j = judge_contradiction("应开启方案A", "应该开启方案A")
        assert j["kind"] is ConflictKind.NOT_A_CONFLICT
        assert j["metadata_excluded"] is True
        assert j["reason"] == "synonym_restatement"

    def test_synonym_degree_adverb_not_conflict(self):
        """程度副词差异（很可爱 vs 可爱）不判矛盾。"""
        assert is_contradiction("方案A很可行", "方案A可行") == \
            ConflictKind.NOT_A_CONFLICT

    def test_source_complementary_not_conflict(self):
        """来源不同但内容互补（不是对立）不判矛盾（元数据排除③）。"""
        j = judge_contradiction(
            "产品支持中文", "产品支持英文",
            meta_a={"source": "s1"}, meta_b={"source": "s2"})
        assert j["kind"] is ConflictKind.NOT_A_CONFLICT
        assert j["metadata_excluded"] is True
        assert j["reason"] == "source_complementary"

    def test_explicit_exclude_wins(self):
        """显式排除（meta.exclude=True）优先于一切判定。"""
        j = judge_contradiction(
            "应开启A", "应关闭A",
            meta_a={"exclude": True}, meta_b={})
        assert j["kind"] is ConflictKind.NOT_A_CONFLICT
        assert j["metadata_excluded"] is True

    def test_both_retrieved_rebuttal_not_conflict(self):
        """hRepro && eRepro 只证"双方都被检索到"，不证矛盾——须落三选一。"""
        j = judge_contradiction("我喜欢蓝色小橘猫", "今天天气不错")
        assert j["kind"] is ConflictKind.NOT_A_CONFLICT
        assert j["metadata_excluded"] is False


# ===========================================================================
# 3.7 验证链全链（M3-2：草稿→独立验证→修正 + provenance + 幂等 60s 竞态）
# ===========================================================================

class TestVerificationChainFullChain:
    """验证链全链：verify / run_pending / pending_conflicts / provenance。"""

    def _register(self, chain, text="应开启方案A", ref="e1"):
        return chain.register(entry_ref=ref, register_query=text)

    def test_verify_confirm_when_no_conflict(self):
        """无冲突参考断言 → CONFIRM（验证通过）。"""
        chain = _fresh_chain()
        req = self._register(chain)
        res = chain.verify(req.idempotency_key,
                           reference_claims=["应开启方案B"])
        assert res is not None
        assert res.verdict == VerdictType.CONFIRM.value
        assert res.conflict_kind is None
        assert chain.pending_count() == 0

    def test_verify_conflict_directional(self):
        """方向矛盾参考断言 → CONFLICT + conflict_kind + 待应用负载。"""
        chain = _fresh_chain()
        req = self._register(chain)
        res = chain.verify(req.idempotency_key,
                           reference_claims=["应关闭方案A"])
        assert res.verdict == VerdictType.CONFLICT.value
        assert res.conflict_kind == ConflictKind.DIRECTIONAL.value
        pend = chain.pending_conflicts()
        assert len(pend) == 1
        assert pend[0]["claim_text"] == "应开启方案A"
        assert pend[0]["target_text"] == "应关闭方案A"
        assert pend[0]["conflict_kind"] == ConflictKind.DIRECTIONAL.value

    def test_verify_conflict_negation(self):
        """否定矛盾参考断言 → CONFLICT + NEGATION。"""
        chain = _fresh_chain()
        req = self._register(chain, "存在蓝色小橘猫")
        res = chain.verify(req.idempotency_key,
                           reference_claims=["不存在蓝色小橘猫"])
        assert res.verdict == VerdictType.CONFLICT.value
        assert res.conflict_kind == ConflictKind.NEGATION.value

    def test_verify_idempotent_same_key(self):
        """"客户端超时重试"同键不双写：重复 verify 命中原结果对象
        （60s 竞态用例——验证结果登记幂等的全链版本）。"""
        chain = _fresh_chain()
        req = self._register(chain)
        first = chain.verify(req.idempotency_key,
                             reference_claims=["应关闭方案A"])
        second = chain.verify(req.idempotency_key,
                              reference_claims=["应关闭方案A"])
        assert first is second
        # 同键同验证只一条结果记录（pending_conflicts 仍 1 条——写侧应用
        # 是独立步骤，幂等不重复产生结果也不清空待应用清单）
        assert len(chain.pending_conflicts()) == 1
        events = chain.events()
        replays = [ev for ev in events
                   if ev.get("replayed") is True]
        assert len(replays) >= 1, "重试必须命中幂等（replayed 事件）"

    def test_verify_unknown_key_none(self):
        """未登记键验证 → None（拒绝"空降"验证）。"""
        chain = _fresh_chain()
        assert chain.verify("verif_unknown",
                            reference_claims=["x"]) is None

    def test_verify_no_reference_pending(self):
        """无独立证据 → 无裁决 NONE（待验证不臆断 confirm）。"""
        chain = _fresh_chain()
        req = self._register(chain)
        res = chain.verify(req.idempotency_key)
        assert res is not None
        assert res.verdict == VerdictType.NONE.value
        assert chain.pending_count() == 0

    def test_verify_default_off_zero_participation(self):
        """验证链默认关（写侧默认保守）：verify/run_pending 零参与。"""
        chain = VerificationChain()  # 默认关
        assert chain.verify("verif_x",
                            reference_claims=["应关闭方案A"]) is None
        assert chain.run_pending() == []
        assert chain.pending_conflicts() == []

    def test_verify_anti_fraud_dimensions_kept(self):
        """全链保持防伪独立四维：验证批次/通道 ≠ 登记批次/通道。"""
        chain = _fresh_chain()
        req = self._register(chain)
        chain.verify(req.idempotency_key,
                     reference_claims=["应关闭方案A"])
        queries = [p for p in chain.provenance()
                   if p["kind"] == "VERIFY-QUERY"]
        assert queries
        q = queries[-1]
        assert q["verify_batch"] != q["register_batch"]
        assert q["verify_channel"] != q["register_channel"]
        assert q["verify_channel"].startswith("verify.")
        assert q["verify_query"] != q["register_query"]

    def test_verify_provenance_full_trail(self):
        """VERIFY-* provenance 全程：REGISTER → QUERY → JUDGE → RESULT。"""
        chain = _fresh_chain()
        req = self._register(chain)
        chain.verify(req.idempotency_key,
                     reference_claims=["应关闭方案A"])
        kinds = [p["kind"] for p in chain.provenance()]
        assert "VERIFY-REGISTER" in kinds
        assert "VERIFY-QUERY" in kinds
        assert "VERIFY-JUDGE" in kinds
        assert "VERIFY-RESULT" in kinds
        # JUDGE 记录判定详情（judged_kind/reason）
        judge_ev = [p for p in chain.provenance()
                    if p["kind"] == "VERIFY-JUDGE"][-1]
        assert judge_ev["judged_kind"] == ConflictKind.DIRECTIONAL.value
        assert judge_ev["reason"] == "directional"

    def test_run_pending_drives_full_chain(self):
        """run_pending 驱动全链：有证据 → 判定；无证据 → NONE；重跑幂等。"""
        chain = _fresh_chain()
        req1 = self._register(chain, "应开启方案A", ref="e1")
        req2 = self._register(chain, "今天天气不错", ref="e2")

        def evidence(r):
            return {"应开启方案A": ["应关闭方案A"],
                    "今天天气不错": []}.get(r.register_query, [])

        out = chain.run_pending(evidence)
        by_key = {r.idempotency_key: r for r in out}
        assert by_key[req1.idempotency_key].verdict == \
            VerdictType.CONFLICT.value
        assert by_key[req2.idempotency_key].verdict == \
            VerdictType.NONE.value  # 无证据 → 待验证
        assert chain.run_pending(evidence) == [], "重跑不产生新结果（幂等）"

    def test_mark_conflict_applied_idempotent(self):
        """写侧应用记账幂等：同键只记一次 VERIFY-CONFLICT-APPLIED。"""
        chain = _fresh_chain()
        req = self._register(chain)
        chain.verify(req.idempotency_key,
                     reference_claims=["应关闭方案A"])
        assert len(chain.pending_conflicts()) == 1
        assert chain.mark_conflict_applied(req.idempotency_key) is True
        assert chain.pending_conflicts() == []
        assert chain.mark_conflict_applied(req.idempotency_key) is True
        kinds = [p["kind"] for p in chain.provenance()]
        assert kinds.count("VERIFY-CONFLICT-APPLIED") == 1

    def test_snapshot_includes_m32_metrics(self):
        """观测块含 provenance / 待应用冲突计数。"""
        chain = _fresh_chain()
        snap = chain.snapshot()
        assert snap["provenance_n"] == 0
        assert snap["conflicts_pending"] == 0
        req = self._register(chain)
        chain.verify(req.idempotency_key,
                     reference_claims=["应关闭方案A"])
        snap = chain.snapshot()
        assert snap["provenance_n"] >= 4
        assert snap["conflicts_pending"] == 1


# ===========================================================================
# 3.8 验证链写侧（M3-2：CONFLICT 结果 → [doubt] conflict → labile）
# ===========================================================================

class TestVerificationWriteSide:
    """验证链全链写侧：结果写 [doubt] conflict → 目标条目 labile（写侧）。"""

    def _make_loop(self, tmp_path):
        from runtime.config import default_config
        from runtime.loop import LivingMemoryLoop
        cfg = default_config()
        cfg.update(num_nodes=32, input_dim=16, num_infer_steps=5,
                   embedder=TestLoopIntegrationM3._MockEmbedder(16),
                   auto_snapshot=False, snapshot_dir=str(tmp_path / "s"),
                   archive_enabled=False)
        return LivingMemoryLoop(cfg)

    def test_loop_conflict_result_marks_labile(self, tmp_path, monkeypatch):
        """CONFLICT 验证结果写侧应用：目标旧记忆 labile + rebuttal_count+1
        （与人工 [doubt] conflict 同一条证伪路径——写侧时相）。"""
        monkeypatch.setenv("LMS_VERIFICATION_CHAIN_ENABLED", "1")
        loop = self._make_loop(tmp_path)
        loop.process_turn("用户: 蓝色小橘猫很可爱")  # 旧记忆
        entry = list(loop.memory.iter_episodic())[-1]
        # 新断言与旧记忆否定冲突 → verify 判 CONFLICT(NEGATION)
        req = loop.verification_chain.register(
            entry_ref="e-new", register_query="蓝色小橘猫不可爱")
        res = loop.verification_chain.verify(
            req.idempotency_key, reference_claims=["蓝色小橘猫很可爱"])
        assert res.verdict == VerdictType.CONFLICT.value
        assert res.conflict_kind == ConflictKind.NEGATION.value
        assert len(loop.verification_chain.pending_conflicts()) == 1
        # 写侧应用：[doubt] conflict → 目标条目 labile（mark_labile 写侧）
        loop._apply_verification_conflicts()
        assert entry.labile is True
        assert entry.rebuttal_count == 1
        assert loop.verification_chain.pending_conflicts() == []
        assert getattr(entry, "doubt_state", "stable") == "stable"

    def test_loop_process_turn_applies_conflicts(self, tmp_path, monkeypatch):
        """process_turn 自动应用：CONFLICT 结果 → [doubt] conflict → 再巩固改写。

        E3（根因 3 修复，2026-08-20）：conflict 证伪命中补入队 → 巩固期
        受控改写消化 → 条目终态 superseded（原断言 labile 停留在中间态，
        现断言证伪→再巩固全链收口的终态——候选不再被丢在半路）。
        """
        monkeypatch.setenv("LMS_VERIFICATION_CHAIN_ENABLED", "1")
        loop = self._make_loop(tmp_path)
        loop.process_turn("用户: 蓝色小橘猫很可爱")
        entry = list(loop.memory.iter_episodic())[-1]
        req = loop.verification_chain.register(
            entry_ref="e-new", register_query="蓝色小橘猫不可爱")
        loop.verification_chain.verify(
            req.idempotency_key, reference_claims=["蓝色小橘猫很可爱"])
        assert loop.verification_chain.pending_conflicts(), "待应用"
        loop.process_turn("用户: 无关内容")
        assert getattr(entry, "doubt_state", "stable") == "superseded"
        assert entry.rebuttal_count == 1
        assert loop.verification_chain.pending_conflicts() == []

    def test_loop_write_side_default_off(self, tmp_path):
        """开关默认关：写侧应用零参与（无异常、无 pending）。"""
        loop = self._make_loop(tmp_path)
        assert loop.verification_chain.enabled is False
        loop.process_turn("用户: 蓝色小橘猫很可爱")
        loop._apply_verification_conflicts()
        assert loop.verification_chain.pending_conflicts() == []
        for e in loop.memory.iter_episodic():
            assert e.labile is False

    def test_loop_verify_confirm_no_labile(self, tmp_path, monkeypatch):
        """CONFIRM 验证结果不触发证伪（目标条目保持 stable 非 labile）。"""
        monkeypatch.setenv("LMS_VERIFICATION_CHAIN_ENABLED", "1")
        loop = self._make_loop(tmp_path)
        loop.process_turn("用户: 蓝色小橘猫很可爱")
        entry = list(loop.memory.iter_episodic())[-1]
        req = loop.verification_chain.register(
            entry_ref="e-new", register_query="蓝色小橘猫很可爱")
        res = loop.verification_chain.verify(
            req.idempotency_key, reference_claims=["蓝色小橘猫很可爱"])
        assert res.verdict == VerdictType.CONFIRM.value
        loop._apply_verification_conflicts()
        assert entry.labile is False
        assert entry.rebuttal_count == 0


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
        """[doubt] conflict 证伪经 loop：平坦字段 + 原生字段同步（写侧）。

        E3（根因 3 修复，2026-08-20）：conflict 证伪命中补入队 → 本回合
        巩固期再巩固消化（E3 关无 min-age 闸门）→ 条目终态 superseded、
        labile 时相复位、原生字段最后一次写者为 consolidation（巩固时相
        合法写者）——证伪→再巩固全链收口（原断言 labile 停留在中间态）。
        """
        loop = self._make_loop(tmp_path)
        loop.process_turn("用户: 我喜欢蓝色小橘猫")
        loop.process_turn("[doubt] conflict: 我喜欢蓝色小橘猫")
        e = list(loop.memory.iter_episodic())[-1]
        assert e.rebuttal_count == 1
        assert getattr(e, "doubt_state", "stable") == "superseded"
        assert e.labile is False  # labile 时相已复位（改写已落库）
        rc = get_rebuttal_consistency(e)
        # 证伪写侧（ingest）与巩固写侧（consolidation）均为合法写者；
        # 终态写者 = 最后一次状态转移（consolidation_resolve 写侧入口）
        assert rc["updated_by"] == "consolidation"
        assert isinstance(rc["consistency"], float)
        # consistency = 巩固写侧时点快照（consolidation_resolve rewritten
        # 分支写入；随后只读路径不改写结构化字段——写者守卫铁律）
        assert rc["consistency"] == pytest.approx(e.confidence, abs=1e-3)
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

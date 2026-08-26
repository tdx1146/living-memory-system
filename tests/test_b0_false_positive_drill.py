# -*- coding: utf-8 -*-
"""B0 假阳性演练固化（判别力恢复的前置测试 · 演练记录判定过程）

对象：``core/doubt/verification_chain.py`` 的 ``VerificationChain``
（``enabled=True`` 显式构造）+ ``judge_contradiction`` /
``is_contradiction`` / ``verify`` / ``run_pending`` /
``pending_conflicts`` / ``mark_conflict_applied``。

目的（§0 映射）：
  - Q1 流动[是]：本文件是 B0 假阳性演练的可复用闸门——判定语义一旦
    回退（假冲突复发）测试即红；
  - Q2 过程[是]：逐例记录"登记 → 全链验证 → 写侧应用"的判定过程；
  - Q4 熵核[是]：surprise 判别校准（见 test_discrimination_recovery.py）
    的前置——矛盾判定不误报，判别器才谈得上恢复；
  - Q5 可回放[是]：纯 stdlib 单测，无 LLM 依赖，可回放。

两组 + 一组幂等/防伪：
  1. 真矛盾触发组（必须判 CONFLICT，走全链
     verify → pending_conflicts 非空 → mark_conflict_applied 幂等）：
     方向矛盾 / 数值矛盾 / 否定矛盾（三选一，metadata_excluded=False）；
  2. 元数据碰撞不触发组（必须 NOT_A_CONFLICT——P0 假冲突根因①②红线：
     时间戳前缀 / 同义复述 / 来源互补 / 显式排除一律不判矛盾，
     metadata_excluded=True）；
  3. 幂等/防伪组：同键重发不重复 / 同键重复 verify 返回原结果 /
     verify_query ≠ register_query / 验证批次 ≠ 登记批次 /
     channel 隔离 / 开关关零参与。

判据与代码语义核对（写前通读 verification_chain.py 确认）：
  - ``judge_contradiction(claim_a, claim_b, *, meta_a=None, meta_b=None)``
    → ``{"kind": ConflictKind, "metadata_excluded": bool, "reason": str}``
    （kind 为枚举实例；判定顺序：显式排除 → 时间戳碰撞 → 同义复述 →
    否定 → 方向 → 数值 → 来源互补 → no_contradiction）；
  - ``register(*, entry_ref, register_query, registered_phase="injection",
    verify_query=None, batch_id=None, channel=None)`` → 开关关返回 None；
  - ``verify(request_key, *, reference_claims=(), reference_meta=None,
    batch_id=None, channel=None)`` → 任一参考断言判真矛盾 → CONFLICT；
    无参考断言 → NONE；否则 CONFIRM；同键重复返回原结果对象；
  - ``run_pending(evidence_fn=None, *, batch_id=None)`` → 驱动全链，重跑
    幂等返回 []；
  - ``pending_conflicts(last_n=20)`` → CONFLICT 且未应用的结果负载；
  - ``mark_conflict_applied(request_key)`` → 写侧应用记账，幂等。

运行方式：pytest rewrite-ws/tests/test_b0_false_positive_drill.py -v
"""

import os
import sys

# 确保项目根目录在 Python 路径中（可从任意 cwd 运行）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest

from core.doubt.verification_chain import (
    ConflictKind,
    VerificationChain,
    VerificationEventType,
    VerdictType,
    is_contradiction,
    judge_contradiction,
    verification_key,
)


def _fresh_chain(enabled=True, window=60.0):
    """验证链（显式开启 + 短窗口——幂等窗口只防"重试竞态"，60s 够测）。"""
    return VerificationChain(enabled=enabled, window=window)


# ===========================================================================
# 1. 真矛盾触发组（§2.2 三选一：方向/数值/否定——必须判 CONFLICT）
# ===========================================================================

class TestTrueConflictTriggerGroup:
    """真矛盾：judge 三选一命中（metadata_excluded=False）+ 全链
    verify → CONFLICT → pending_conflicts 非空 → mark_conflict_applied 幂等。"""

    # -- judge 层：三选一判定（含 中文方向词对 / 共享非数字 shingle /
    #    -- 否定标记不受例外词干扰） --------------------------------------- #

    @pytest.mark.parametrize("claim_a,claim_b,expected", [
        # 1. 方向矛盾：同一事实两条断言结论相反（中文方向词对，参考
        #    _DIRECTIONAL_PAIRS 的（开启,关闭）——"应"被归一剔除后
        #    "开启方案A" vs "关闭方案A" 核心相同 → 方向矛盾）
        ("应开启方案A", "应关闭方案A", ConflictKind.DIRECTIONAL),
        ("支持采用方案B", "反对采用方案B", ConflictKind.DIRECTIONAL),
        # 2. 数值矛盾：同一量同量纲区间不相交（"成本1500元" vs
        #    "成本800元"——数字归一为 '#' 后共享非数字 shingle "成本"）
        ("成本1500元", "成本800元", ConflictKind.NUMERIC),
        # 3. 否定矛盾：一条断言否定另一条存在性（"系统没有X功能"经
        #    "没"标记去除 → "系统有X功能"；句子避开"不错/了不起"等
        #    _NEGATION_EXCEPTION_WORDS 干扰）
        ("系统有X功能", "系统没有X功能", ConflictKind.NEGATION),
        ("存在蓝色小橘猫", "不存在蓝色小橘猫", ConflictKind.NEGATION),
    ])
    def test_judge_kind_three_way(self, claim_a, claim_b, expected):
        """judge_contradiction：三类真矛盾 → kind 命中且非元数据排除。"""
        j = judge_contradiction(claim_a, claim_b)
        assert j["kind"] is expected
        assert j["metadata_excluded"] is False, \
            "真矛盾绝不能被元数据排除误伤（假阳性演练红线反向）"
        # 便捷只读 API 与 judge 同判定
        assert is_contradiction(claim_a, claim_b) is expected

    # -- 全链：草稿→独立验证→修正（verify → pending → applied 幂等） ---- #

    @pytest.mark.parametrize("register_claim,ref_claim,expected_kind", [
        ("应开启方案A", "应关闭方案A", ConflictKind.DIRECTIONAL.value),
        ("成本1500元", "成本800元", ConflictKind.NUMERIC.value),
        ("系统有X功能", "系统没有X功能", ConflictKind.NEGATION.value),
    ])
    def test_full_chain_conflict_and_apply(
            self, register_claim, ref_claim, expected_kind):
        """全链：register → verify(reference_claims=[对照断言]) → CONFLICT
        → pending_conflicts 命中该键 → mark_conflict_applied 幂等。"""
        chain = _fresh_chain()
        req = chain.register(entry_ref="e1", register_query=register_claim)
        assert req is not None
        assert chain.pending_count() == 1

        res = chain.verify(req.idempotency_key,
                           reference_claims=[ref_claim])
        assert res is not None
        assert res.verdict == VerdictType.CONFLICT.value
        assert res.conflict_kind == expected_kind
        assert res.metadata_excluded is False

        # 写侧待应用负载命中该键
        pend = chain.pending_conflicts()
        keys = [p["request_key"] for p in pend]
        assert req.idempotency_key in keys
        assert len(pend) == 1
        assert pend[0]["claim_text"] == register_claim
        assert pend[0]["target_text"] == ref_claim
        assert pend[0]["conflict_kind"] == expected_kind

        # 写侧应用记账（幂等：应用后不再出现；再应用仍 True 不重复记账）
        assert chain.mark_conflict_applied(req.idempotency_key) is True
        assert chain.pending_conflicts() == [], "应用后待应用清单必须清空该键"
        assert chain.mark_conflict_applied(req.idempotency_key) is True, \
            "重复应用幂等（记账不重复）"
        kinds = [p["kind"] for p in chain.provenance()]
        assert kinds.count("VERIFY-CONFLICT-APPLIED") == 1

    def test_pending_conflicts_payload_fields(self):
        """pending_conflicts 负载：request_key / claim_text / target_text /
        conflict_kind / resolved_at 齐全（[doubt] conflict 写侧数据源）。"""
        chain = _fresh_chain()
        req = chain.register(entry_ref="e-p", register_query="应开启方案C")
        chain.verify(req.idempotency_key,
                     reference_claims=["应关闭方案C"])
        pend = chain.pending_conflicts()
        assert len(pend) == 1
        item = pend[0]
        assert item["request_key"] == req.idempotency_key
        assert item["claim_text"] == "应开启方案C"
        assert item["target_text"] == "应关闭方案C"
        assert item["conflict_kind"] == ConflictKind.DIRECTIONAL.value
        assert isinstance(item["resolved_at"], float) and item["resolved_at"] > 0

    def test_run_pending_drives_conflict_then_apply(self):
        """run_pending 全链驱动：evidence_fn 提供矛盾证据 → CONFLICT →
        写侧应用幂等；重跑不产生新结果。"""
        chain = _fresh_chain()
        req = chain.register(entry_ref="e1", register_query="应开启方案A")

        def evidence(r):
            return ["应关闭方案A"] if r.register_query == "应开启方案A" else []

        out = chain.run_pending(evidence)
        assert len(out) == 1
        assert out[0].verdict == VerdictType.CONFLICT.value
        assert out[0].conflict_kind == ConflictKind.DIRECTIONAL.value
        assert len(chain.pending_conflicts()) == 1
        # 写侧应用
        assert chain.mark_conflict_applied(req.idempotency_key) is True
        assert chain.pending_conflicts() == []
        # 重跑幂等：已出结果 → 不重复判定/登记
        assert chain.run_pending(evidence) == []


# ===========================================================================
# 2. 元数据碰撞不触发组（P0 假冲突根因①②红线：一律不判矛盾）
# ===========================================================================

class TestMetadataCollisionNoTrigger:
    """元数据碰撞：时间戳/日期前缀、同义复述、来源互补、显式排除
    → NOT_A_CONFLICT + metadata_excluded=True；全链 CONFIRM、无 pending。"""

    def test_datetime_prefix_collision_chain(self):
        """时间戳/日期前缀碰撞（规格用例形态）：两文本仅星期/日期/时刻
        前缀不同（"[Thu ...] 开工吧" vs "[Fri ...] 开工吧"）→ 剔除时间戳
        后同文 → NOT_A_CONFLICT + metadata_excluded=True（P0 根因②——
        旧 overlapMatch 纯子串碰撞误判形态的根治）。"""
        a = "[Thu 2026-08-06 00:11 GMT+8] 开工吧"
        b = "[Fri 2026-08-07 09:00 GMT+8] 开工吧"
        j = judge_contradiction(a, b)
        assert j["kind"] is ConflictKind.NOT_A_CONFLICT
        assert j["metadata_excluded"] is True
        assert j["reason"] == "datetime_collision"
        # 全链：register 带时间戳前缀的登记断言 → 对照断言同为"开工吧"
        # （仅日期前缀不同）→ 不判矛盾 → CONFIRM，无待应用冲突
        chain = _fresh_chain()
        req = chain.register(entry_ref="e-dt", register_query=a)
        res = chain.verify(req.idempotency_key, reference_claims=[b])
        assert res is not None
        assert res.verdict == VerdictType.CONFIRM.value
        assert res.conflict_kind is None
        assert chain.pending_conflicts() == []

    @pytest.mark.parametrize("a,b", [
        # ISO 日期前缀碰撞（旧 overlapMatch 会误判的形态）
        ("2026-08-06 开工", "2026-08-07 开工"),
        # 中文日期前缀碰撞
        ("8月6日 开工", "8月7日 开工"),
    ])
    def test_datetime_prefix_variants(self, a, b):
        """日期前缀碰撞变体：ISO / 中文日期 → 一律不判矛盾。"""
        j = judge_contradiction(a, b)
        assert j["kind"] is ConflictKind.NOT_A_CONFLICT
        assert j["metadata_excluded"] is True

    def test_synonym_restatement_excluded(self):
        """同义复述：不同措辞同一结论（"应开启方案A" vs "应该开启方案A"
        ——"应/应该"归一剔除后同文）→ NOT_A_CONFLICT + metadata_excluded=True。
        全链同参考断言 → CONFIRM，无 pending。"""
        a, b = "应开启方案A", "应该开启方案A"
        j = judge_contradiction(a, b)
        assert j["kind"] is ConflictKind.NOT_A_CONFLICT
        assert j["metadata_excluded"] is True
        assert j["reason"] == "synonym_restatement"

        chain = _fresh_chain()
        req = chain.register(entry_ref="e-syn", register_query=a)
        res = chain.verify(req.idempotency_key, reference_claims=[b])
        assert res.verdict == VerdictType.CONFIRM.value
        assert res.conflict_kind is None
        assert chain.pending_conflicts() == []

    def test_synonym_degree_adverb_excluded(self):
        """同义复述（程度副词变体）："今天天气很好" vs "今天天气好"
        ——"很"为语气/程度虚词归一剔除 → 同文 → metadata_excluded=True。"""
        j = judge_contradiction("今天天气很好", "今天天气好")
        assert j["kind"] is ConflictKind.NOT_A_CONFLICT
        assert j["metadata_excluded"] is True
        assert j["reason"] == "synonym_restatement"

    def test_task_paraphrase_redline_holds(self):
        """任务建议措辞对「今天天气很好」vs「今天天气真不错」：红线
        （不判矛盾）成立——NOT_A_CONFLICT。

        偏差记录：该措辞对经归一后为 "今天天气好" vs "今天天气真不错"
        （"真/不错"不在 _STRIP_WORDS/_TRAILING_PARTICLES 剔除集），
        不命中 ``synonym_restatement`` 分支，落入 ``no_contradiction``
        兜底 → ``metadata_excluded=False``（P0 红线"一律不判矛盾"仍成立，
        判定方向正确；只是元数据排除标记未置位）。同义复述分支本身的
        metadata_excluded=True 由上面两个用例覆盖。"""
        j = judge_contradiction("今天天气很好", "今天天气真不错")
        assert j["kind"] is ConflictKind.NOT_A_CONFLICT
        assert j["metadata_excluded"] is False, \
            "当前实现：'真不错' 非剔除词 → 走 no_contradiction 兜底（记录偏差）"
        assert j["reason"] == "no_contradiction"

    def test_source_complementary_excluded(self):
        """来源互补：meta_a.source=A、meta_b.source=B 且内容互补（非对立）
        → 不判矛盾 + metadata_excluded=True（元数据排除③）。"""
        j = judge_contradiction(
            "产品支持中文", "产品支持英文",
            meta_a={"source": "A"}, meta_b={"source": "B"})
        assert j["kind"] is ConflictKind.NOT_A_CONFLICT
        assert j["metadata_excluded"] is True
        assert j["reason"] == "source_complementary"

    def test_source_complementary_chain_confirm(self):
        """来源互补全链：register → verify 互补参考断言 → CONFIRM、
        无 pending（verify 不携带登记侧 meta（meta_a 不传入），链级判定
        落 no_contradiction 兜底 → CONFIRM——写侧语义正确，不产生冲突）。"""
        chain = _fresh_chain()
        req = chain.register(entry_ref="e-src", register_query="产品支持中文")
        res = chain.verify(
            req.idempotency_key,
            reference_claims=["产品支持英文"],
            reference_meta=[{"source": "B"}])
        assert res is not None
        assert res.verdict == VerdictType.CONFIRM.value
        assert res.conflict_kind is None
        assert chain.pending_conflicts() == []

    def test_explicit_exclude_chain(self):
        """显式排除：meta.exclude=True 优先于一切判定（即使文本是真矛盾
        词对）→ NOT_A_CONFLICT + metadata_excluded=True；全链 verify 把
        metadata_excluded 传入结果负载。"""
        j = judge_contradiction(
            "应开启A", "应关闭A",
            meta_a={"exclude": True}, meta_b={})
        assert j["kind"] is ConflictKind.NOT_A_CONFLICT
        assert j["metadata_excluded"] is True
        assert j["reason"] == "explicit_exclude"

        chain = _fresh_chain()
        req = chain.register(entry_ref="e-exc", register_query="应开启A")
        res = chain.verify(
            req.idempotency_key,
            reference_claims=["应关闭A"],
            reference_meta=[{"exclude": True}])
        assert res.verdict == VerdictType.CONFIRM.value
        assert res.metadata_excluded is True, "元数据排除标记经全链传播"
        assert chain.pending_conflicts() == []


# ===========================================================================
# 3. 幂等 / 防伪独立四维 / 开关默认关
# ===========================================================================

class TestIdempotencyAntiFraud:
    """幂等（同键重发/重复 verify）+ 防伪独立四维（query/批次/通道）
    + 开关关零参与。"""

    def test_register_same_key_returns_original(self):
        """同键重发 register 返回**原请求**不重复登记（幂等键协议）。"""
        chain = _fresh_chain()
        req1 = chain.register(entry_ref="e1", register_query="应开启方案A")
        req2 = chain.register(entry_ref="e1", register_query="应开启方案A")
        assert req1 is not None and req2 is not None
        assert req1.idempotency_key == req2.idempotency_key
        assert req1 is req2, "同键重发必须原样返回原请求（禁止盲目重发）"
        assert chain.pending_count() == 1, "不重复登记"

    def test_verify_same_key_returns_original_result(self):
        """同键重复 verify 返回**原结果**（"客户端超时"≠"服务端未写入"）。"""
        chain = _fresh_chain()
        req = chain.register(entry_ref="e1", register_query="应开启方案A")
        first = chain.verify(req.idempotency_key,
                             reference_claims=["应关闭方案A"])
        second = chain.verify(req.idempotency_key,
                              reference_claims=["应关闭方案A"])
        assert first is second, "同键同验证只一条结果记录"
        assert chain.lookup(req.idempotency_key) is first
        events = chain.events()
        replays = [ev for ev in events if ev.get("replayed") is True]
        assert any(ev["type"] == VerificationEventType.VERIFY_RESULT.value
                   for ev in replays)
        # 幂等不产生重复 pending（写侧应用是独立步骤）
        assert len(chain.pending_conflicts()) == 1

    def test_verify_query_independent(self):
        """防伪独立四维②：verify_query ≠ register_query（不同构造）。"""
        chain = _fresh_chain()
        req = chain.register(entry_ref="e1", register_query="应开启方案A")
        assert req.verify_query != req.register_query
        assert "换个角度" in req.verify_query

    def test_verify_batch_independent(self):
        """防伪独立四维③：验证批次 ≠ 登记批次（independent_batch）。"""
        chain = _fresh_chain()
        req = chain.register(entry_ref="e1", register_query="应开启方案A")
        verify_batch = chain.independent_batch(req.batch_id)
        assert verify_batch != req.batch_id
        # 同窗内下一次 next_batch 与登记同批 → independent_batch 加后缀错开
        assert chain.independent_batch(chain.next_batch()) != chain.next_batch()

    def test_channel_isolated(self):
        """防伪独立四维④：验证通道 ≠ 登记通道（channel_for 端点隔离）。"""
        chain = _fresh_chain()
        req = chain.register(entry_ref="e1", register_query="应开启方案A")
        assert req.channel == "register.injection"
        assert chain.channel_for(req.channel) == "verify.injection"
        assert chain.channel_for(req.channel) != req.channel
        assert chain.channel_for(req.channel).startswith("verify.")
        # 全链 provenance 记录验证通道/批次与登记隔离
        chain.verify(req.idempotency_key, reference_claims=["应关闭方案A"])
        q = [p for p in chain.provenance() if p["kind"] == "VERIFY-QUERY"][-1]
        assert q["verify_batch"] != q["register_batch"]
        assert q["verify_channel"] != q["register_channel"]
        assert q["verify_channel"].startswith("verify.")
        assert q["verify_query"] != q["register_query"]

    def test_switch_off_zero_participation(self):
        """开关关（默认构造 VerificationChain()）→ register/verify 返回
        None、run_pending/pending_conflicts 空——零参与（§5.3 写侧默认
        保守；假阳性演练通过才 env 开启）。"""
        chain = VerificationChain()  # 默认关
        assert chain.enabled is False
        assert chain.register(entry_ref="e1", register_query="应开启方案A") \
            is None
        assert chain.verify("verif_x", reference_claims=["应关闭方案A"]) \
            is None
        assert chain.run_pending() == []
        assert chain.pending_conflicts() == []
        assert chain.pending_count() == 0
        assert chain.mark_conflict_applied("verif_x") is False
        assert chain.snapshot() == {"enabled": False}

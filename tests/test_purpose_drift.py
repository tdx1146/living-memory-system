# -*- coding: utf-8 -*-
"""目的检查框架（purpose-drift 时相）单测（总任务书 §二.5 + 避免跑偏方案 §2.1）。

覆盖（任务书规格 3）：
  1. 五问映射：每问至少一例「是」与一例「否」判定（构造对应 round_signals）
     → TestFiveQuestionMapping
  2. verdict 计算：任一否 → drifted + purpose_drift=True；全是 → aligned；
     缺信号 → uncertain 且 reasons 说明缺什么 → TestVerdictRules
  3. 无分数保证：judge/snapshot 输出递归检查不含 0-100 类符合度分数键
     （数值计数如 drift_count 除外）→ TestNoScoreGuarantee
  4. 治理开关：LMS_PURPOSE_DRIFT_ENABLED=0 → judge 中性 + snapshot
     enabled=False → TestGovernanceSwitch
  5. dsh-goal 目的来源：LMS_PURPOSE_GOAL_TEXT 设置后 purpose 随之变化
     → TestPurposeSource
  6. 每轮记录 bounded（maxlen 200）+ drift_count 累计 → TestRecording
  7. DoubtStateMachine 集成：purpose_drift_check 返回闸门信号、snapshot()
     含 purpose_drift 块、原快照键全部保留（回归断言）→ TestStateMachineIntegration
  8. fail-open：异常输入不抛 → TestFailOpen

运行方式：pytest rewrite-ws/tests/test_purpose_drift.py -v。
"""

import os
import sys

# 确保项目根目录在 Python 路径中（可从任意 cwd 运行）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest

from core.doubt.purpose_drift import (
    QUESTION_IDS,
    PurposeDriftPhase,
    _DEFAULT_PURPOSE_TEXT,
    purpose_drift_enabled,
)
from core.doubt.state_machine import DoubtStateMachine
from core.doubt import PurposeDriftPhase as ExportedPurposeDriftPhase


# ---------------------------------------------------------------------------
# 测试小工具
# ---------------------------------------------------------------------------

#: 五问全「是」的完整轮信号（用于 aligned 断言与构造偏离轮）
FULL_YES = {
    "episodic_added": True,
    "doubt_events": 2,
    "lifecycle_trace": True,
    "surprise_in_decisions": True,
    "reconsolidated": True,
}

#: 符合度分数形态键（判据：这类键不得携带 0-100 类数值）
_SCORE_LIKE_KEYS = {
    "score", "grade", "compliance", "conformance", "rating",
    "alignment", "adherence", "fitness",
}


def _assert_no_score_fields(obj, path="root"):
    """递归断言：无「符合度分数」键携带数值（0-100 类）。

    数值计数（drift_count、doubt_events 等事件计数）键名不在
    _SCORE_LIKE_KEYS 内——判据是「无符合度分数」，不是「无任何数值」。
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _SCORE_LIKE_KEYS:
                assert not isinstance(v, (int, float)) or isinstance(v, bool), \
                    "%s.%s 携带数值 %r——违反无分数保证" % (path, k, v)
            _assert_no_score_fields(v, "%s.%s" % (path, k))
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _assert_no_score_fields(item, "%s[%d]" % (path, i))


# ===========================================================================
# 1. 五问映射（每问至少一例「是」与一例「否」）
# ===========================================================================

class TestFiveQuestionMapping:
    """Q1-Q5 各自判定：构造对应 round_signals → 该问是/否。"""

    # -- Q1 流动 ---------------------------------------------------------- #

    def test_q1_yes_via_episodic_added(self):
        """Q1 是：episodic_added（写侧活动）→ 流动。"""
        out = PurposeDriftPhase().judge({"episodic_added": True})
        assert out["answers"]["Q1"]["ok"] is True
        assert out["answers"]["Q1"]["state"] == "yes"

    def test_q1_yes_via_consolidation(self):
        """Q1 是：consolidated（reconsolidated / dream_consolidated）→ 流动。"""
        ph = PurposeDriftPhase()
        assert ph.judge({"reconsolidated": True})["answers"]["Q1"]["state"] \
            == "yes"
        assert ph.judge({"dream_consolidated": True})["answers"]["Q1"]["state"] \
            == "yes"

    def test_q1_no_no_write_activity(self):
        """Q1 否：本轮无任何写侧活动 → 产出未流入活结构。"""
        out = PurposeDriftPhase().judge({})
        assert out["answers"]["Q1"]["ok"] is False
        assert out["answers"]["Q1"]["state"] == "no"
        assert "写侧活动" in out["answers"]["Q1"]["reason"]

    # -- Q2 过程 ---------------------------------------------------------- #

    def test_q2_yes_via_doubt_events(self):
        """Q2 是：doubt_events>0 → 有过程记录。"""
        out = PurposeDriftPhase().judge({"doubt_events": 3})
        assert out["answers"]["Q2"]["state"] == "yes"

    def test_q2_yes_via_lifecycle_or_surprise(self):
        """Q2 是：lifecycle_trace 或 surprise_in_decisions 也构成过程记录。"""
        ph = PurposeDriftPhase()
        assert ph.judge({"lifecycle_trace": True})["answers"]["Q2"]["state"] \
            == "yes"
        assert ph.judge({"surprise_in_decisions": True})["answers"]["Q2"]["state"] \
            == "yes"

    def test_q2_no_conclusion_only(self):
        """Q2 否：只记录结论未记录过程（conclusion_only）。"""
        out = PurposeDriftPhase().judge({"conclusion_only": True})
        assert out["answers"]["Q2"]["ok"] is False
        assert out["answers"]["Q2"]["state"] == "no"
        assert "结论" in out["answers"]["Q2"]["reason"]

    # -- Q3 活体 ---------------------------------------------------------- #

    def test_q3_yes_mechanisms_moving(self):
        """Q3 是：doubt_events>0 / reconsolidated / verification_chain_active
        → 记忆机制真在动。"""
        ph = PurposeDriftPhase()
        assert ph.judge({"reconsolidated": True})["answers"]["Q3"]["state"] \
            == "yes"
        assert ph.judge({"doubt_events": 1})["answers"]["Q3"]["state"] == "yes"
        assert ph.judge({"verification_chain_active": True})[
            "answers"]["Q3"]["state"] == "yes"

    def test_q3_no_memory_idle(self):
        """Q3 否：记忆机制「只是存在」本轮没动（memory_idle）。"""
        out = PurposeDriftPhase().judge({"memory_idle": True})
        assert out["answers"]["Q3"]["ok"] is False
        assert out["answers"]["Q3"]["state"] == "no"

    # -- Q4 熵核 ---------------------------------------------------------- #

    def test_q4_yes_surprise_in_decisions(self):
        """Q4 是：熵/惊讶参与决策（surprise_in_decisions）。"""
        out = PurposeDriftPhase().judge({"surprise_in_decisions": True})
        assert out["answers"]["Q4"]["ok"] is True
        assert out["answers"]["Q4"]["state"] == "yes"

    def test_q4_no_surprise_display_only(self):
        """Q4 否：熵/惊讶只作展示（surprise_display_only）。"""
        out = PurposeDriftPhase().judge({"surprise_display_only": True})
        assert out["answers"]["Q4"]["ok"] is False
        assert out["answers"]["Q4"]["state"] == "no"

    # -- Q5 可回放 -------------------------------------------------------- #

    def test_q5_yes_replayable(self):
        """Q5 是：lifecycle_trace / provenance（VERIFY-*）/ 活跃验证链
        → 涌现过程可回放。"""
        ph = PurposeDriftPhase()
        assert ph.judge({"lifecycle_trace": True})["answers"]["Q5"]["state"] \
            == "yes"
        assert ph.judge({"provenance": True})["answers"]["Q5"]["state"] == "yes"
        assert ph.judge({"verification_chain_active": True})[
            "answers"]["Q5"]["state"] == "yes"

    def test_q5_no_not_replayable(self):
        """Q5 否：过程不可回放/注入/审计（not_replayable）。"""
        out = PurposeDriftPhase().judge({"not_replayable": True})
        assert out["answers"]["Q5"]["ok"] is False
        assert out["answers"]["Q5"]["state"] == "no"


# ===========================================================================
# 2. verdict 计算规则（铁律：任一否→drifted；全是→aligned；其余→uncertain）
# ===========================================================================

class TestVerdictRules:
    def test_all_yes_aligned(self):
        """全「是」→ aligned + purpose_drift=False。"""
        ph = PurposeDriftPhase()
        out = ph.judge(dict(FULL_YES))
        assert out["verdict"] == "aligned"
        assert out["purpose_drift"] is False
        assert all(a["state"] == "yes"
                   for a in out["answers"].values())

    def test_any_no_drifted(self):
        """任一「否」（其余全「是」）→ drifted + purpose_drift=True。"""
        # 构造：仅 Q3 否（memory_idle），Q1/Q2/Q4/Q5 全「是」
        round_signals = {
            "episodic_added": True,        # Q1 是（写侧）
            "surprise_in_decisions": True,  # Q2/Q4 是
            "lifecycle_trace": True,        # Q5 是（Q2 也满足）
            "memory_idle": True,            # Q3 否
        }
        out = PurposeDriftPhase().judge(round_signals)
        assert out["verdict"] == "drifted"
        assert out["purpose_drift"] is True
        assert out["answers"]["Q3"]["state"] == "no"
        assert out["answers"]["Q1"]["state"] == "yes"
        assert out["answers"]["Q2"]["state"] == "yes"
        assert out["answers"]["Q4"]["state"] == "yes"
        assert out["answers"]["Q5"]["state"] == "yes"

    def test_q1_no_alone_drifts(self):
        """仅 Q1 否（无写侧活动）且 Q2-Q5 全「是」→ drifted。"""
        round_signals = {
            "doubt_events": 1,               # Q2/Q3 是
            "lifecycle_trace": True,         # Q5 是
            "surprise_in_decisions": True,   # Q4 是
            # 无 episodic_added/consolidation/suspect → Q1 否
        }
        out = PurposeDriftPhase().judge(round_signals)
        assert out["answers"]["Q1"]["state"] == "no"
        assert out["verdict"] == "drifted"
        assert out["purpose_drift"] is True

    def test_missing_signals_uncertain_with_reasons(self):
        """缺信号 → uncertain，且 reasons 说明缺什么。"""
        out = PurposeDriftPhase().judge({"episodic_added": True})
        assert out["verdict"] == "uncertain"
        assert out["purpose_drift"] is True  # 不确定 = 未判，不是通过
        assert out["answers"]["Q1"]["state"] == "yes"
        # Q2-Q5 均不确定，且理由说明缺什么信号
        for q in ("Q2", "Q3", "Q4", "Q5"):
            assert out["answers"][q]["state"] == "uncertain"
            assert "缺" in out["answers"][q]["reason"], q
        joined = " | ".join(out["reasons"])
        assert "缺过程信号" in joined
        assert "缺活体信号" in joined
        assert "缺熵核信号" in joined
        assert "缺可回放信号" in joined

    def test_output_shape_and_question_order(self):
        """闸门信号形状固定：五个键 + 五问 answers 顺序可回放。"""
        out = PurposeDriftPhase().judge(dict(FULL_YES))
        assert set(out) == {"purpose_drift", "verdict", "answers",
                            "reasons", "purpose"}
        assert list(out["answers"]) == list(QUESTION_IDS) == \
            ["Q1", "Q2", "Q3", "Q4", "Q5"]
        assert len(out["reasons"]) == 5
        assert out["purpose"] == _DEFAULT_PURPOSE_TEXT


# ===========================================================================
# 3. 无分数保证（Pan 警示：不输出可被优化的分数）
# ===========================================================================

class TestNoScoreGuarantee:
    def test_judge_output_no_score_fields(self):
        """judge 输出（aligned 与 drifted 各一例）递归无符合度分数。"""
        ph = PurposeDriftPhase()
        _assert_no_score_fields(ph.judge(dict(FULL_YES)))
        _assert_no_score_fields(ph.judge({}))
        _assert_no_score_fields(ph.judge({"memory_idle": True}))
        _assert_no_score_fields(ph.judge({"episodic_added": True}))

    def test_snapshot_no_score_fields(self):
        """snapshot 输出递归无符合度分数（drift_count 是计数，允许）。"""
        ph = PurposeDriftPhase()
        ph.judge(dict(FULL_YES))
        ph.judge({})  # drifted
        snap = ph.snapshot()
        _assert_no_score_fields(snap)
        # drift_count 是事件计数（键名非符合度分数键），保留
        assert snap["drift_count"] == 1

    def test_no_grade_compliance_keys_anywhere(self):
        """显式扫全部键名：无 score/grade/compliance/conformance 形态键。"""
        ph = PurposeDriftPhase()
        ph.judge(dict(FULL_YES))
        ph.judge({})
        collected = []

        def walk(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    collected.append(k)
                    walk(v)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    walk(item)
        walk(ph.judge({}))
        walk(ph.snapshot())
        assert not (_SCORE_LIKE_KEYS & set(collected)), \
            "出现符合度分数形态键: %s" % (_SCORE_LIKE_KEYS & set(collected))


# ===========================================================================
# 4. 治理开关（LMS_PURPOSE_DRIFT_ENABLED 默认 1=开；0=关 → 中性）
# ===========================================================================

class TestGovernanceSwitch:
    def test_default_enabled(self):
        """默认开：LMS_PURPOSE_DRIFT_ENABLED 未设置 → 1=开。"""
        assert purpose_drift_enabled() is True
        assert PurposeDriftPhase().enabled is True

    def test_env_off_judge_neutral(self, monkeypatch):
        """env=0 → judge 返回中性闸门（不判是/否、闸门不亮）。"""
        monkeypatch.setenv("LMS_PURPOSE_DRIFT_ENABLED", "0")
        ph = PurposeDriftPhase()
        assert ph.enabled is False
        out = ph.judge(dict(FULL_YES))  # 即便给全是信号也中性
        assert out["verdict"] == "uncertain"
        assert out["purpose_drift"] is False
        assert all(a["ok"] is None for a in out["answers"].values())
        assert "关闭" in out["reasons"][0]

    def test_env_off_snapshot_enabled_false(self, monkeypatch):
        """env=0 → snapshot enabled=False。"""
        monkeypatch.setenv("LMS_PURPOSE_DRIFT_ENABLED", "0")
        ph = PurposeDriftPhase()
        snap = ph.snapshot()
        assert snap["enabled"] is False
        assert snap["drift_count"] == 0
        assert snap["recent_verdicts"] == []

    def test_constructor_explicit_off(self):
        """显式参数 enabled=False 优先于 env（默认开时仍可关）。"""
        ph = PurposeDriftPhase(enabled=False)
        assert ph.enabled is False
        out = ph.judge(dict(FULL_YES))
        assert out["verdict"] == "uncertain"
        assert out["purpose_drift"] is False

    def test_env_off_zero_participation_no_records(self, monkeypatch):
        """开关关 → 判定零记录（recent_verdicts 空、drift_count 0）。"""
        monkeypatch.setenv("LMS_PURPOSE_DRIFT_ENABLED", "0")
        ph = PurposeDriftPhase()
        for _ in range(3):
            ph.judge({})
        assert ph.drift_count == 0
        assert len(ph._recent_verdicts) == 0
        assert ph.snapshot()["last_verdict"] is None


# ===========================================================================
# 5. dsh-goal 目的来源（LMS_PURPOSE_GOAL_TEXT / 调用方 purpose_text / 内置）
# ===========================================================================

class TestPurposeSource:
    def test_default_purpose_four_anchors(self):
        """缺省目的 = dandan 原始哲学四词锚点。"""
        ph = PurposeDriftPhase()
        assert ph.judge({})["purpose"] == _DEFAULT_PURPOSE_TEXT
        assert "流动上下文" in _DEFAULT_PURPOSE_TEXT
        assert "活体记忆" in _DEFAULT_PURPOSE_TEXT
        assert "记录涌现过程" in _DEFAULT_PURPOSE_TEXT
        assert "熵核心" in _DEFAULT_PURPOSE_TEXT

    def test_env_goal_text_changes_purpose(self, monkeypatch):
        """LMS_PURPOSE_GOAL_TEXT 设置后 snapshot/judge 的 purpose 随之变化。"""
        monkeypatch.setenv("LMS_PURPOSE_GOAL_TEXT", "守卫流动上下文")
        ph = PurposeDriftPhase()
        assert ph.snapshot()["purpose"] == "守卫流动上下文"
        assert ph.judge({})["purpose"] == "守卫流动上下文"

    def test_explicit_purpose_overrides_env(self, monkeypatch):
        """调用方 purpose_text > env（dsh-goal 每轮目的优先）。"""
        monkeypatch.setenv("LMS_PURPOSE_GOAL_TEXT", "env 目的")
        ph = PurposeDriftPhase()
        out = ph.judge({"episodic_added": True}, purpose_text="dsh-goal 目的")
        assert out["purpose"] == "dsh-goal 目的"
        # 后续轮不带 purpose_text → 回落 env 目的
        assert ph.judge({"episodic_added": True})["purpose"] == "env 目的"

    def test_purpose_truncated_in_snapshot(self, monkeypatch):
        """超长目的在快照中截断（purpose(text 截断)——来源经 env 注入）。"""
        long_purpose = "锚" * 500
        monkeypatch.setenv("LMS_PURPOSE_GOAL_TEXT", long_purpose)
        ph = PurposeDriftPhase()
        ph.judge(dict(FULL_YES))
        snap = ph.snapshot()
        assert len(snap["purpose"]) <= 120
        assert "…" in snap["purpose"]
        assert snap["purpose"].startswith("锚" * 100)


# ===========================================================================
# 6. 每轮记录：有界 deque（maxlen 200）+ drift_count 累计
# ===========================================================================

class TestRecording:
    def test_drift_count_accumulates_only_drifted(self):
        """drift_count 只累计 drifted；aligned/uncertain 不计。"""
        ph = PurposeDriftPhase()
        ph.judge({})                    # drifted（Q1 否）
        assert ph.drift_count == 1
        ph.judge({"memory_idle": True})  # drifted（Q3 否）
        assert ph.drift_count == 2
        ph.judge(dict(FULL_YES))        # aligned —— 不计
        assert ph.drift_count == 2
        ph.judge({"episodic_added": True})  # uncertain —— 未判不是偏离
        assert ph.drift_count == 2
        assert len(ph._recent_verdicts) == 4  # 四轮都记录

    def test_recent_verdicts_bounded_maxlen_200(self):
        """有界：超过 200 轮只保留最近 200 条（deque maxlen 200）。"""
        ph = PurposeDriftPhase()
        for _ in range(210):
            ph.judge({})  # 全 drifted
        assert len(ph._recent_verdicts) == 200
        assert ph.drift_count == 210

    def test_snapshot_recent_verdicts_window(self):
        """snapshot recent_verdicts(n)：默认 20 条、可调窗口。"""
        ph = PurposeDriftPhase()
        for _ in range(50):
            ph.judge({})
        assert len(ph.snapshot()["recent_verdicts"]) == 20
        assert len(ph.snapshot(max_n=100)["recent_verdicts"]) == 50
        assert len(ph.snapshot(max_n=5)["recent_verdicts"]) == 5

    def test_last_verdict_and_reasons_tracked(self):
        """snapshot 记录最近一轮 verdict 与 reasons（可回放可审计）。"""
        ph = PurposeDriftPhase()
        out = ph.judge({"memory_idle": True})
        snap = ph.snapshot()
        assert snap["last_verdict"] == out["verdict"] == "drifted"
        assert snap["last_reasons"] == out["reasons"]
        assert snap["recent_verdicts"][-1]["verdict"] == "drifted"
        assert "ts" in snap["recent_verdicts"][-1]


# ===========================================================================
# 7. DoubtStateMachine 集成（只做加法：purpose_drift 块 + 委托方法）
# ===========================================================================

class TestStateMachineIntegration:
    def test_purpose_drift_check_returns_gate(self):
        """purpose_drift_check 返回闸门信号（委托 PurposeDriftPhase.judge）。"""
        sm = DoubtStateMachine()
        gate = sm.purpose_drift_check(dict(FULL_YES))
        assert set(gate) == {"purpose_drift", "verdict", "answers",
                             "reasons", "purpose"}
        assert gate["verdict"] == "aligned"
        assert gate["purpose_drift"] is False
        gate2 = sm.purpose_drift_check({})  # 无写侧活动 → drifted
        assert gate2["verdict"] == "drifted"
        assert gate2["purpose_drift"] is True

    def test_purpose_drift_check_records_via_phase(self):
        """委托的判定进入 state machine 内建 phase 的记录（drift 累计）。"""
        sm = DoubtStateMachine()
        sm.purpose_drift_check({})
        sm.purpose_drift_check({"memory_idle": True})
        sm.purpose_drift_check(dict(FULL_YES))
        assert sm.purpose_drift_phase.drift_count == 2
        assert len(sm.purpose_drift_phase._recent_verdicts) == 3

    def test_snapshot_contains_purpose_drift_block(self):
        """snapshot() 含 purpose_drift 块（观测数据源）。"""
        sm = DoubtStateMachine()
        sm.purpose_drift_check({"episodic_added": True})
        snap = sm.snapshot()
        block = snap["purpose_drift"]
        assert block["enabled"] is True
        assert block["purpose"] == _DEFAULT_PURPOSE_TEXT
        assert block["last_verdict"] == "uncertain"
        assert "drift_count" in block
        assert "recent_verdicts" in block
        assert "last_reasons" in block

    def test_original_snapshot_keys_preserved(self):
        """回归：原快照键全部保留（enabled/phases/verification_chain/params
        + 新增 purpose_drift——键集为原键集并集 {purpose_drift}）。"""
        sm = DoubtStateMachine()
        snap = sm.snapshot()
        original_keys = {"enabled", "phases", "verification_chain", "params"}
        assert original_keys <= set(snap)
        assert set(snap) == original_keys | {"purpose_drift"}
        # 原键语义不变
        assert snap["enabled"] is True
        assert set(snap["phases"]) == {"retrieval", "injection",
                                       "consolidation"}
        assert "surprise_factor" in snap["params"]
        assert "labile_window_ttl" in snap["params"]
        assert "verification_chain" in snap

    def test_lazy_creation_of_phase(self):
        """purpose_drift=None → 内部懒建（首次使用才创建）。"""
        sm = DoubtStateMachine()
        assert sm.purpose_drift_phase is None
        sm.purpose_drift_check(dict(FULL_YES))
        assert sm.purpose_drift_phase is not None
        assert isinstance(sm.purpose_drift_phase, PurposeDriftPhase)

    def test_injected_phase_used_as_is(self):
        """显式注入的 phase 原样使用（不覆盖开关/目的源）。"""
        injected = PurposeDriftPhase()
        sm = DoubtStateMachine(purpose_drift=injected)
        assert sm.purpose_drift_phase is injected
        gate = sm.purpose_drift_check(dict(FULL_YES))
        assert gate["verdict"] == "aligned"

    def test_disabled_state_machine_snapshot_unchanged(self):
        """回归：state machine 开关关时快照仍为最小形态 {"enabled": False}
        （既有默认路径零改动）。"""
        sm = DoubtStateMachine(enabled=False)
        assert sm.snapshot() == {"enabled": False}


# ===========================================================================
# 8. fail-open（异常输入静默降级，绝不阻断调用方）
# ===========================================================================

class TestFailOpen:
    def test_judge_none_round_signals(self):
        """round_signals=None → 中性闸门，不抛。"""
        ph = PurposeDriftPhase()
        out = ph.judge(None)
        assert isinstance(out, dict)
        assert out["verdict"] in ("aligned", "drifted", "uncertain")

    def test_judge_garbage_inputs(self):
        """非 dict 输入（字符串/数字/列表）→ 不抛、返回闸门 dict。"""
        ph = PurposeDriftPhase()
        for garbage in ("abc", 123, [1, 2], object()):
            out = ph.judge(garbage)
            assert isinstance(out, dict)
            assert "verdict" in out
            assert "answers" in out

    def test_judge_weird_signal_values(self):
        """信号值异常（非法计数/不可转布尔的对象）→ 不抛（fail-open）。"""
        ph = PurposeDriftPhase()
        out = ph.judge({"doubt_events": "abc", "episodic_added": object(),
                        "reconsolidated": None})
        assert isinstance(out, dict)
        assert "verdict" in out
        # 非法计数按 0（fail-open）：doubt_events 不构成正信号
        assert out["answers"]["Q2"]["state"] == "uncertain"

    def test_snapshot_weird_max_n(self):
        """snapshot(max_n=非法值) → 不抛（fail-open 降级快照）。"""
        ph = PurposeDriftPhase()
        ph.judge(dict(FULL_YES))
        assert isinstance(ph.snapshot(max_n="abc"), dict)
        assert isinstance(ph.snapshot(max_n=-5), dict)

    def test_state_machine_delegation_fail_open(self):
        """state machine 委托路径异常输入 → 不抛、返回闸门 dict。"""
        sm = DoubtStateMachine()
        gate = sm.purpose_drift_check(None)
        assert isinstance(gate, dict)
        assert "verdict" in gate
        assert "purpose" in gate

    def test_exported_from_package(self):
        """core.doubt 包级导出 PurposeDriftPhase（与模块类同一对象）。"""
        assert ExportedPurposeDriftPhase is PurposeDriftPhase
        assert hasattr(__import__("core.doubt", fromlist=["purpose_drift_enabled"]),
                       "purpose_drift_enabled")

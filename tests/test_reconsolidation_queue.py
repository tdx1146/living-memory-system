# -*- coding: utf-8 -*-
"""再巩固候选队列测试（R3 labile 平衡——语义决策 D-2026-08-18-01 代码化）。

覆盖（core/doubt/reconsolidation_queue.py）:
  1. 三闸门：候选在队（G1）/ 巩固期触发（G2）/ 改写受控（G3）——
     任一不过 → 不改写（机器防线）
  2. 检索不塑形：检索时相入队被拒（零副作用）；只读接口绝不 setattr 条目
  3. 队列持久化：写侧入队即落盘，新实例（重启）加载后候选仍在
  4. 受控改写：巩固完成只留 append-only 演化史（写者 consolidation），
     绝不覆盖历史
  5. 入队/移除往返 + 容量上限 + TTL 惰性清除
  6. 开关关 → 零参与；损坏文件 → 空队列（fail-open）

运行方式：pytest rewrite-ws/tests/test_reconsolidation_queue.py -q
"""

import json
import time

import pytest

from core.doubt.reconsolidation_queue import (
    ReconsolidationQueue,
    entry_key,
    reconsolidation_enabled,
)
from core.doubt.state_machine import DoubtPhase
from core.store.process_core import (
    append_transition,
    get_evolution,
    make_transition,
)

ENV_ENABLED = "LMS_DOUBT_RECONSOLIDATION_ENABLED"
ENV_PATH = "LMS_DOUBT_RECONSOLIDATION_QUEUE_PATH"


class _MiniEntry:
    """最小条目（与 EpisodicEntry 同构的字段子集；getattr/setattr 兼容）。"""

    def __init__(self, text: str, confidence: float = 0.8):
        self.text = text
        self.confidence = confidence
        self.doubt_state = "suspect"


def _make_queue(tmp_path, monkeypatch=None, **kw):
    """构造队列：默认隔离路径（tmp_path），可注入覆盖参数。"""
    path = str(tmp_path / "candidates.json")
    if monkeypatch is not None:
        monkeypatch.setenv(ENV_PATH, path)
        return ReconsolidationQueue(**kw)
    return ReconsolidationQueue(path=path, **kw)


# --------------------------------------------------------------------------- #
#  0) 开关解析（reconsolidation_enabled）
# --------------------------------------------------------------------------- #

class TestEnabled:

    def test_default_on(self):
        assert reconsolidation_enabled() is True

    def test_env_off(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLED, "0")
        assert reconsolidation_enabled() is False

    def test_env_on_variants(self, monkeypatch):
        for v in ("1", "true", "yes", "on"):
            monkeypatch.setenv(ENV_ENABLED, v)
            assert reconsolidation_enabled() is True

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv(ENV_ENABLED, "0")
        assert reconsolidation_enabled(True) is True
        assert reconsolidation_enabled(False) is False


# --------------------------------------------------------------------------- #
#  1) 三闸门（巩固期受控改写机器防线）
# --------------------------------------------------------------------------- #

class TestThreeGates:

    def test_gate1_candidate_in_queue(self, tmp_path):
        """G1：候选不在队 → 不改写（即使巩固期+受控动作就绪）。"""
        q = _make_queue(tmp_path)
        entry = _MiniEntry("从未入队的条目")
        res = q.maybe_rewrite(
            entry, phase=DoubtPhase.CONSOLIDATION.value)
        assert res == {"passed": False, "gate": "candidate_in_queue",
                       "rewritten": False}
        # 队列零变化（零副作用）
        assert q.size() == 0

    def test_gate2_consolidation_triggered(self, tmp_path):
        """G2：候选在队但非巩固期时相 → 不改写（检索/注入期不改写）。"""
        q = _make_queue(tmp_path)
        entry = _MiniEntry("候选条目")
        assert q.enqueue(entry, reason="high_surprise", phase=DoubtPhase.INJECTION.value)
        for phase in (DoubtPhase.RETRIEVAL.value, DoubtPhase.INJECTION.value):
            res = q.maybe_rewrite(entry, phase=phase)
            assert res["passed"] is False
            assert res["gate"] == "consolidation_triggered"
        # 候选仍在队（未消化）
        assert q.contains(entry) is True

    def test_gate3_rewrite_controlled(self, tmp_path):
        """G3：改写动作不受控（rewrite_fn 抛异常）→ 不改写、队列不动。"""
        q = _make_queue(tmp_path)
        entry = _MiniEntry("候选条目")

        def _bad_rewrite(e):
            raise RuntimeError("不受控改写动作")

        assert q.enqueue(entry, reason="rebuttal")
        res = q.maybe_rewrite(
            entry, phase=DoubtPhase.CONSOLIDATION.value,
            rewrite_fn=_bad_rewrite)
        assert res["passed"] is False
        assert res["gate"] == "rewrite_controlled"
        assert "error" in res
        # 队列未动、条目未被写
        assert q.contains(entry) is True

    def test_all_gates_pass_rewrites_and_removes(self, tmp_path):
        """三闸门全过 → 执行受控改写 + 候选出队（再巩固闭环）。"""
        q = _make_queue(tmp_path)
        entry = _MiniEntry("待再巩固条目")
        assert q.enqueue(entry, reason="high_surprise", score=1.25)
        seen = []
        res = q.maybe_rewrite(
            entry, phase=DoubtPhase.CONSOLIDATION.value,
            rewrite_fn=lambda e: seen.append(e), detail="验证通过")
        assert res["passed"] is True
        assert res["gate"] == "all"
        assert res["rewritten"] is True
        assert seen == [entry]  # 受控动作被执行一次
        # 候选被消化（出队）
        assert q.contains(entry) is False
        assert q.size() == 0

    def test_controlled_append_only(self, tmp_path):
        """G3 内部受控动作：只留 append-only 演化史登记，绝不覆盖历史。"""
        q = _make_queue(tmp_path)
        entry = _MiniEntry("再巩固条目")
        # 预置一条既有演化史（append-only 不覆盖）
        append_transition(entry, make_transition("created", detail="ingest"),
                          updated_by="ingest")
        assert q.enqueue(entry, reason="high_surprise")
        res = q.maybe_rewrite(entry, phase=DoubtPhase.CONSOLIDATION.value)
        assert res["passed"] is True
        ev = get_evolution(entry)
        states = [t["state"] for t in ev["history"]]
        assert states == ["created", "reconsolidated"]  # 追加不覆盖
        assert ev["updated_by"] == "consolidation"      # 写者=巩固期
        # 条目文本/置信度零触碰（改写只留登记）
        assert entry.text == "再巩固条目"
        assert entry.confidence == 0.8

    def test_rewrite_fn_delegation_is_controlled(self, tmp_path):
        """带 rewrite_fn 时：闸门全过后执行委托动作一次（写侧入口）。"""
        q = _make_queue(tmp_path)
        entry = _MiniEntry("委托改写")
        assert q.enqueue(entry)
        calls = []

        def _delegate(e):
            calls.append(e)
            e.doubt_state = "superseded"  # 调用方写侧转移（state_machine 入口）

        res = q.maybe_rewrite(
            entry, phase=DoubtPhase.CONSOLIDATION.value, rewrite_fn=_delegate)
        assert res["passed"] is True
        assert calls == [entry]
        assert entry.doubt_state == "superseded"


# --------------------------------------------------------------------------- #
#  2) 检索不塑形（语义决策 D-2026-08-18-01）+ 只读接口零 setattr
# --------------------------------------------------------------------------- #

class TestRetrievalNoReshape:

    def test_retrieval_phase_enqueue_rejected(self, tmp_path):
        """检索时相入队被拒：队列零变化（零副作用）。"""
        q = _make_queue(tmp_path)
        entry = _MiniEntry("检索命中条目")
        assert q.enqueue(
            entry, phase=DoubtPhase.RETRIEVAL.value) is False
        assert q.size() == 0
        assert q.contains(entry) is False

    def test_readonly_interfaces_never_setattr(self, tmp_path):
        """只读接口（peek/candidates/contains/snapshot）绝不 setattr 条目。"""
        q = _make_queue(tmp_path)
        entry = _MiniEntry("只读条目")
        assert q.enqueue(entry)
        before = dict(vars(entry))
        q.peek()
        q.candidates()
        assert q.contains(entry) is True
        q.snapshot()
        assert q.size() == 1
        assert vars(entry) == before  # 零字段变化

    def test_retrieval_phase_maybe_rewrite_gate2(self, tmp_path):
        """检索时相调用 maybe_rewrite → G2 不过（即使候选在队）。"""
        q = _make_queue(tmp_path)
        entry = _MiniEntry("候选条目")
        assert q.enqueue(entry)
        res = q.maybe_rewrite(entry, phase=DoubtPhase.RETRIEVAL.value)
        assert res["gate"] == "consolidation_triggered"
        assert q.contains(entry) is True  # 未消化


# --------------------------------------------------------------------------- #
#  3) 持久化（跨重启）
# --------------------------------------------------------------------------- #

class TestPersistence:

    def test_survives_restart(self, tmp_path):
        """写侧入队即落盘 → 新实例（模拟重启）加载后候选仍在。"""
        path = str(tmp_path / "candidates.json")
        q1 = ReconsolidationQueue(path=path)
        entry = _MiniEntry("重启不丢失的候选")
        assert q1.enqueue(entry, reason="high_surprise", score=1.25)
        # 模拟进程重启：同一路径新建实例
        q2 = ReconsolidationQueue(path=path)
        assert q2.size() == 1
        assert q2.contains(entry) is True
        rec = q2.peek()[0]
        assert rec["reason"] == "high_surprise"
        assert rec["score"] == 1.25

    def test_remove_persists(self, tmp_path):
        """巩固出队落盘 → 重启后不再含该候选。"""
        path = str(tmp_path / "candidates.json")
        q1 = ReconsolidationQueue(path=path)
        e1 = _MiniEntry("条目一")
        e2 = _MiniEntry("条目二")
        assert q1.enqueue(e1)
        assert q1.enqueue(e2)
        assert q1.remove(e1) is True
        q2 = ReconsolidationQueue(path=path)
        assert q2.contains(e1) is False
        assert q2.contains(e2) is True

    def test_corrupt_file_fail_open(self, tmp_path):
        """损坏队列文件 → 空队列 + 计数（fail-open 不崩）。"""
        path = str(tmp_path / "candidates.json")
        path = str(tmp_path / "candidates.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write("{ 这不是合法 JSON")
        q = ReconsolidationQueue(path=path)
        assert q.size() == 0
        assert q.load_failures >= 1


# --------------------------------------------------------------------------- #
#  4) 入队/移除往返 + 容量 + TTL
# --------------------------------------------------------------------------- #

class TestQueueOps:

    def test_enqueue_remove_roundtrip(self, tmp_path):
        q = _make_queue(tmp_path)
        e1 = _MiniEntry("甲")
        e2 = _MiniEntry("乙")
        assert q.enqueue(e1, reason="high_surprise")
        assert q.enqueue(e2, reason="rebuttal")
        assert q.size() == 2
        keys = [r["entry_key"] for r in q.peek()]
        assert keys == [entry_key(e1), entry_key(e2)]  # 入队序
        assert q.remove(e1) is True
        assert q.contains(e1) is False
        assert q.contains(e2) is True
        assert q.remove(e1) is False  # 已移除 → False（幂等）

    def test_duplicate_enqueue_overwrites_same_key(self, tmp_path):
        """同键重复入队 → 覆盖记录（保持唯一键），不双条。"""
        q = _make_queue(tmp_path)
        e = _MiniEntry("同键条目")
        assert q.enqueue(e, reason="high_surprise", score=1.1)
        assert q.enqueue(e, reason="rebuttal", score=1.3)
        assert q.size() == 1
        assert q.peek()[0]["reason"] == "rebuttal"
        assert q.peek()[0]["score"] == 1.3

    def test_max_candidates_fifo_evict(self, tmp_path):
        """容量上限：超出丢弃最旧（FIFO）。"""
        q = _make_queue(tmp_path, max_candidates=3)
        for i in range(5):
            assert q.enqueue(_MiniEntry("条目%d" % i))
        assert q.size() == 3
        keys = [r["entry_key"] for r in q.peek()]
        assert keys == [entry_key(_MiniEntry("条目2")),
                        entry_key(_MiniEntry("条目3")),
                        entry_key(_MiniEntry("条目4"))]

    def test_enqueue_expired_pruned_immediately(self, tmp_path):
        """入队即过期（entered_at 超 TTL）→ enqueue 内触发 prune 立即清除。"""
        q = _make_queue(tmp_path, ttl=10.0)
        stale = _MiniEntry("过期候选")
        assert q.enqueue(stale, now=time.time() - 100.0)
        assert q.size() == 0
        assert q.contains(stale) is False

    def test_load_prunes_expired(self, tmp_path):
        """启动加载时按 TTL 过滤过期候选（跨重启的惰性清除落点）。"""
        now = time.time()
        path = str(tmp_path / "candidates.json")
        payload = {"candidates": [
            {"entry_key": "sha1:stale", "reason": "rebuttal",
             "score": 1.2, "entered_at": now - 1000.0},
            {"entry_key": "sha1:fresh", "reason": "high_surprise",
             "score": 1.3, "entered_at": now},
        ]}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        q = ReconsolidationQueue(path=path, ttl=10.0)
        assert q.size() == 1
        assert q.peek()[0]["entry_key"] == "sha1:fresh"


# --------------------------------------------------------------------------- #
#  5) 开关关 → 零参与
# --------------------------------------------------------------------------- #

class TestDisabled:

    def test_disabled_zero_participation(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_ENABLED, "0")
        q = ReconsolidationQueue(path=str(tmp_path / "q.json"))
        entry = _MiniEntry("开关关条目")
        assert q.enqueue(entry) is False
        assert q.size() == 0
        assert q.contains(entry) is False
        res = q.maybe_rewrite(entry, phase=DoubtPhase.CONSOLIDATION.value)
        assert res["passed"] is False
        assert q.peek() == []
        # 无文件落盘（零参与）
        assert (tmp_path / "q.json").exists() is False

    def test_disabled_snapshot(self, tmp_path, monkeypatch):
        monkeypatch.setenv(ENV_ENABLED, "0")
        q = ReconsolidationQueue(path=str(tmp_path / "q.json"))
        assert q.snapshot() == {"enabled": False}


# --------------------------------------------------------------------------- #
#  6) entry_key
# --------------------------------------------------------------------------- #

class TestEntryKey:

    def test_explicit_key_wins(self):
        e = _MiniEntry("文本")
        assert entry_key(e, key="custom-1") == "custom-1"

    def test_entry_id_when_present(self):
        class _E:
            id = "entry-42"
            text = "文本"

        assert entry_key(_E()) == "entry-42"

    def test_text_sha1_fallback(self):
        k1 = entry_key(text="同一个文本")
        k2 = entry_key(text="同一个文本")
        assert k1 == k2
        assert k1.startswith("sha1:")
        assert entry_key(text="不同文本") != k1

    def test_empty_returns_empty(self):
        assert entry_key(text="") == ""

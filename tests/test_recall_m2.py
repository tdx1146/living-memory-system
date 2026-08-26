# -*- coding: utf-8 -*-
"""M2 · recall 只读化 测试（核心重建规格 v2 §1.4 / §2.1 / §5.1 / §5.2 / §6.2）。

覆盖（按规格 §6.2"只读四不变"用例组 + M2 三任务）：
  1. 只读四不变守卫（纯 stdlib 单测，注入式 state_reader——四量各自
     违反均被抓；条目指纹抓字段改写——旧 _attach_consistency 泄漏形态）
  2. 怀疑信号投影（纯单测：绝不改写条目 / labile / rebuttal / verdict）
  3. loop 集成（precision_adapt 开）：检索不改写条目（consistency 保持
     None）；_attach_consistency 已删除；守卫武装（写入即抛）
  4. API 集成：/recall /react 响应追加 suspicion 区段（§4.2 独立区段）；
     违反 → 500；快照一致性扫描无 updated_by=retrieval
  5. 测试不掩盖（H 模式）：守卫默认强制开启，conftest 未置 0
"""

import os
import sys

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
import torch

from fastapi.testclient import TestClient

import api.server as server_module
from api.session_manager import SessionManager
from runtime.dream_scheduler import DreamScheduler
from runtime.config import default_config
from core.sensory.embedder import SimpleEmbedder
from runtime.loop import LivingMemoryLoop

from core.recall.guard import (
    FourInvariantGuard,
    ReadOnlyViolation,
    entry_fingerprint,
    episodic_fingerprint,
    diff_four_invariants,
    _ENTRY_FIELD_NAMES,
)
from core.recall.suspicion import (
    project_suspicion,
    entry_readonly_view,
    empty_suspicion,
    EMPTY_SUSPICION,
)


class MockEmbedder(SimpleEmbedder):
    """带 embed_text 的确定性嵌入器（与 test_recall_readonly 同款）。"""

    def embed_text(self, text: str) -> torch.Tensor:
        if not text or not text.strip():
            return torch.zeros(self._dim)
        h = abs(hash(text)) % (2 ** 32)
        g = torch.Generator()
        g.manual_seed(h)
        return torch.randn(self._dim, generator=g) * 0.1

    def embed_text_raw(self, text: str) -> torch.Tensor:
        return self.embed_text(text)


def make_config_factory(snapshot_dir: str):
    """轻量级配置工厂（与 test_recall_readonly / test_react_readonly 同款）。"""
    def factory():
        config = default_config()
        config['num_nodes'] = 32
        config['input_dim'] = 16
        config['num_infer_steps'] = 5
        config['consolidation_interval'] = 3
        config['seed'] = 42
        config['llm_api'] = None
        config['embedder'] = MockEmbedder(dim=16)
        config['auto_snapshot'] = False
        config['snapshot_dir'] = snapshot_dir
        config['archive_dir'] = os.path.join(snapshot_dir, 'archive')
        return config
    return factory


def _create_scheduler(sm):
    return DreamScheduler(
        get_loop_fn=lambda sid: sm.get(sid),
        idle_threshold=999999,
        dream_steps=5,
        dream_full_cycle=False,
        check_interval=999999,
    )


@pytest.fixture
def client(monkeypatch, tmp_path):
    """轻量级 TestClient（无 LLM bridge，不自动做梦；归档隔离到 tmp）。"""
    monkeypatch.setenv("LMS_EMBEDDER", "simple")
    monkeypatch.setenv("LMS_NUM_NODES", "32")
    monkeypatch.setenv("LMS_INPUT_DIM", "16")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LMS_LLM_API_KEY", raising=False)

    snapshot_dir = str(tmp_path / "snapshots")
    sm = SessionManager(default_config_factory=make_config_factory(snapshot_dir))
    scheduler = _create_scheduler(sm)

    orig_sm = server_module._session_manager
    orig_sched = server_module._dream_scheduler
    server_module._session_manager = sm
    server_module._dream_scheduler = scheduler
    try:
        with TestClient(server_module.app) as c:
            yield c
    finally:
        server_module._session_manager = orig_sm
        server_module._dream_scheduler = orig_sched


@pytest.fixture
def loop(tmp_path):
    """独立 LivingMemoryLoop 实例（默认 precision_adapt 关，同回归环境）。"""
    cfg = make_config_factory(str(tmp_path / "snapshots"))()
    return LivingMemoryLoop(cfg)


@pytest.fixture
def loop_precision_adapt(tmp_path, monkeypatch):
    """precision_adapt 开启的 loop（旧泄漏路径只在开关开时活动）。"""
    monkeypatch.setenv("LMS_PRECISION_ADAPT", "1")
    cfg = make_config_factory(str(tmp_path / "snapshots"))()
    return LivingMemoryLoop(cfg)


# ============================================================
# 1. 只读四不变守卫（纯 stdlib 单测）
# ============================================================

class _FakeEntry:
    """守卫单测用的最小条目替身（仅标量字段）。"""

    def __init__(self, eid, **kwargs):
        self._eid = eid
        for name in _ENTRY_FIELD_NAMES:
            setattr(self, name, kwargs.get(name))
        self.text = kwargs.get("text", f"entry-{eid}")


class TestFourInvariantGuard:
    """四量（turn / episodic / J / σ）各自违反均被抓；无违反通过。"""

    def _reader(self, state):
        return lambda: dict(state)

    def _base_state(self):
        return {
            "turn": 3,
            "episodic": frozenset([entry_fingerprint(_FakeEntry(1)),
                                   entry_fingerprint(_FakeEntry(2))]),
            "J": [[0.1, 0.2], [0.3, 0.4]],
            "sigma": [0.5, 0.6],
        }

    def test_unchanged_passes(self):
        state = self._base_state()
        guard = FourInvariantGuard(self._reader(state), scope="t")
        guard.assert_unchanged(dict(state))

    def test_turn_increment_detected(self):
        state = self._base_state()
        guard = FourInvariantGuard(self._reader(state), scope="t")
        state["turn"] += 1
        with pytest.raises(ReadOnlyViolation) as ei:
            guard.assert_unchanged(self._base_state())
        assert "turn" in ei.value.changes

    def test_episodic_change_detected(self):
        state = self._base_state()
        guard = FourInvariantGuard(self._reader(state), scope="t")
        # 条目增（写路径偷偷加了一条）
        state["episodic"] = frozenset(
            list(state["episodic"]) + [entry_fingerprint(_FakeEntry(3))])
        with pytest.raises(ReadOnlyViolation) as ei:
            guard.assert_unchanged(self._base_state())
        assert "episodic" in ei.value.changes

    def test_episodic_removal_detected(self):
        state = self._base_state()
        guard = FourInvariantGuard(self._reader(state), scope="t")
        state["episodic"] = frozenset([entry_fingerprint(_FakeEntry(1))])
        with pytest.raises(ReadOnlyViolation) as ei:
            guard.assert_unchanged(self._base_state())
        assert "episodic" in ei.value.changes

    def test_j_change_detected(self):
        state = self._base_state()
        guard = FourInvariantGuard(self._reader(state), scope="t")
        state["J"] = [[0.9, 0.2], [0.3, 0.4]]
        with pytest.raises(ReadOnlyViolation) as ei:
            guard.assert_unchanged(self._base_state())
        assert "J" in ei.value.changes

    def test_sigma_change_detected(self):
        state = self._base_state()
        guard = FourInvariantGuard(self._reader(state), scope="t")
        state["sigma"] = [0.99, 0.6]
        with pytest.raises(ReadOnlyViolation) as ei:
            guard.assert_unchanged(self._base_state())
        assert "sigma" in ei.value.changes

    def test_context_manager_passes(self):
        state = self._base_state()
        guard = FourInvariantGuard(self._reader(state), scope="t")
        with guard:
            assert guard._before == state  # __enter__ 已快照
        # 正常退出：断言通过

    def test_context_manager_raises_on_mutation(self):
        state = self._base_state()
        guard = FourInvariantGuard(self._reader(state), scope="t")
        with pytest.raises(ReadOnlyViolation):
            with guard:
                state["turn"] += 1  # 守卫窗口内的写 → 违反

    def test_body_exception_not_masked(self):
        """body 异常时跳过断言（不掩盖原始异常）。"""
        state = self._base_state()
        guard = FourInvariantGuard(self._reader(state), scope="t")
        with pytest.raises(ValueError, match="boom"):
            with guard:
                state["turn"] += 1  # 即使窗口内被污染，也不掩盖 body 异常
                raise ValueError("boom")

    def test_entry_fingerprint_captures_consistency_mutation(self):
        """条目指纹抓字段改写——旧 _attach_consistency 泄漏形态。"""
        e = _FakeEntry(1)
        before = entry_fingerprint(e)
        e.consistency = 0.731  # 旧泄漏：recall 内改写 entry.consistency
        after = entry_fingerprint(e)
        assert before != after

    def test_readonly_violation_carries_scope_and_changes(self):
        state = self._base_state()
        state["turn"] += 1
        guard = FourInvariantGuard(self._reader(state), scope="recall_x")
        with pytest.raises(ReadOnlyViolation) as ei:
            guard.assert_unchanged(self._base_state())
        assert ei.value.scope == "recall_x"
        assert "turn" in ei.value.changes
        assert ei.value.to_dict()["scope"] == "recall_x"

    def test_guard_always_on(self):
        """§5.1/§6.1：守卫默认强制开启——测试不许置 0 关掉（H 模式）。"""
        guard = FourInvariantGuard(lambda: {}, scope="probe")
        assert guard.enabled is True
        # 守卫无 env 开关（防"环境变量关断言"掩盖生产 bug）
        for var in ("LMS_RECALL_READONLY", "LMS_RECALL_GUARD_ENABLED"):
            assert os.environ.get(var) is None, f"禁止通过 {var} 关闭守卫"

    def test_nan_stable(self):
        """双 NaN 视为相等（不误报）；单侧 NaN 视为变更。"""
        assert diff_four_invariants(
            {"J": [[float("nan"), 1.0]]}, {"J": [[float("nan"), 1.0]]}
        ) == {}
        assert "J" in diff_four_invariants(
            {"J": [[float("nan"), 1.0]]}, {"J": [[1.0, 1.0]]})


# ============================================================
# 2. 怀疑信号投影（纯单测：绝不改写条目）
# ============================================================

class TestSuspicionProjection:
    """project_suspicion 只读：labile/rebuttal/verdict 只进投影不落条目。"""

    def _entry(self, labile=False, rebuttal_count=0, confidence=1.0,
               text="记忆文本"):
        e = _FakeEntry(1)
        e.text = text
        e.labile = labile
        e.rebuttal_count = rebuttal_count
        e.confidence = confidence
        e.source_trust = 1.0
        e.consistency = None
        e.info_value = 0.0
        e.core = None
        e.gray = False
        e.ts = 123.0
        return e

    def test_projection_never_mutates_entries(self):
        """投影后条目全部字段与投影前一致（铁律：检索绝不改写条目）。"""
        e = self._entry(labile=True, rebuttal_count=2, confidence=0.3)
        before = {name: getattr(e, name) for name in _ENTRY_FIELD_NAMES}
        proj = project_suspicion([(0.9, e)])
        after = {name: getattr(e, name) for name in _ENTRY_FIELD_NAMES}
        assert before == after
        # labile 条目进了投影区，但条目本身 labile 未被复位/改写
        assert len(proj["labile"]) == 1
        assert e.labile is True

    def test_labile_listed(self):
        e = self._entry(labile=True)
        proj = project_suspicion([(0.9, e)])
        assert len(proj["labile"]) == 1
        assert proj["labile"][0]["text"] == "记忆文本"

    def test_rebuttal_pending_listed(self):
        e = self._entry(rebuttal_count=3)
        proj = project_suspicion([(0.9, e)])
        assert len(proj["rebuttal_pending"]) == 1
        assert proj["rebuttal_pending"][0]["rebuttal_count"] == 3

    def test_stable_entry_not_flagged(self):
        e = self._entry(labile=False, rebuttal_count=0)
        proj = project_suspicion([(0.9, e)])
        assert proj["labile"] == []
        assert proj["rebuttal_pending"] == []

    def test_verdict_suspect_listed(self):
        class _FakePrecision:
            def verdict_confidence(self, entry, consistency=None):
                return 0.1  # 低判定置信度

            def doubt_threshold(self):
                return 0.5

        e = self._entry(confidence=0.1)
        proj = project_suspicion(
            [(0.9, e)], precision_adapt=_FakePrecision())
        assert len(proj["verdict_suspect"]) == 1

    def test_verdict_high_confidence_not_listed(self):
        class _FakePrecision:
            def verdict_confidence(self, entry, consistency=None):
                return 0.9

            def doubt_threshold(self):
                return 0.5

        e = self._entry(confidence=0.9)
        proj = project_suspicion(
            [(0.9, e)], precision_adapt=_FakePrecision())
        assert proj["verdict_suspect"] == []

    def test_summary_counts(self):
        e1 = self._entry(labile=True, rebuttal_count=1)
        e2 = self._entry(labile=False, rebuttal_count=0)
        proj = project_suspicion([(0.9, e1), (0.8, e2)])
        assert proj["summary"] == {
            "total": 2, "labile": 1, "rebuttal_pending": 1,
            "verdict_suspect": 0,
        }

    def test_empty_scored_stable_structure(self):
        proj = project_suspicion([])
        assert proj == {
            "labile": [], "rebuttal_pending": [], "verdict_suspect": [],
            "summary": {"total": 0, "labile": 0, "rebuttal_pending": 0,
                        "verdict_suspect": 0},
        }
        # empty_suspicion() 每次独立（不共享可变默认值）
        assert empty_suspicion() == project_suspicion([])
        empty_suspicion()["labile"].append("x")
        assert empty_suspicion()["labile"] == []

    def test_entry_readonly_view_fields(self):
        e = self._entry(labile=True, rebuttal_count=2)
        view = entry_readonly_view(e, score=0.88)
        assert view["text"] == "记忆文本"
        assert view["score"] == 0.88
        assert view["labile"] is True
        assert view["rebuttal_count"] == 2


# ============================================================
# 3. loop 集成：检索零改写 + 守卫武装 + 怀疑投影
# ============================================================

def _entry_fields(entry):
    """条目全字段快照（与守卫指纹同字段集，用于断言零改写）。"""
    return tuple((name, getattr(entry, name, None))
                 for name in _ENTRY_FIELD_NAMES)


class TestRecallReadonlyM2:
    """M2 三任务在 loop 层的集成验证。"""

    def test_attach_consistency_removed(self, loop):
        """旧 _attach_consistency 泄漏方法已整体删除。"""
        assert not hasattr(LivingMemoryLoop, "_attach_consistency")
        assert not hasattr(loop, "_attach_consistency")

    def test_recall_does_not_mutate_entries(self, loop_precision_adapt):
        """检索不改写条目（旧泄漏形态：consistency 保持 None）。"""
        loop = loop_precision_adapt
        assert loop.precision_adapt is not None  # 前置：开关必须开
        loop.process_turn("用户: 我们决定采用方案A")
        loop.process_turn("用户: 方案A需要两个备选")
        before = {id(e): _entry_fields(e)
                  for e in loop.memory.iter_episodic()}

        result = loop.recall_merged_readonly("方案A", k=5)
        assert isinstance(result, list)

        after = {id(e): _entry_fields(e)
                 for e in loop.memory.iter_episodic()}
        assert after == before, "recall 执行后条目字段必须零改写"
        # 旧泄漏形态：recall 后 consistency 必须仍为 None（从未被写回）
        for e in loop.memory.iter_episodic():
            assert getattr(e, "consistency", None) is None

    def test_recall_episodic_readonly_no_mutation(self, loop_precision_adapt):
        loop = loop_precision_adapt
        loop.process_turn("用户: 你好")
        before = {id(e): _entry_fields(e)
                  for e in loop.memory.iter_episodic()}
        loop.recall_episodic_readonly("用户: 你好", k=5)
        after = {id(e): _entry_fields(e)
                 for e in loop.memory.iter_episodic()}
        assert after == before

    def test_guard_armed_detects_mutation(self, loop, monkeypatch):
        """守卫武装：检索窗口内任何写（+turn）→ ReadOnlyViolation。"""
        loop.process_turn("用户: 你好")
        orig_encode = loop._encode_query_vector

        def mutating_encode(text):
            loop.turn_count += 1  # 模拟写路径偷偷 +turn
            return orig_encode(text)

        monkeypatch.setattr(loop, "_encode_query_vector", mutating_encode)
        with pytest.raises(ReadOnlyViolation) as ei:
            loop.recall_episodic_readonly("你好", k=5)
        assert ei.value.scope == "recall_episodic_readonly"
        assert "turn" in ei.value.changes

    def test_merged_guard_armed_detects_mutation(self, loop, monkeypatch):
        """recall_merged_readonly（/recall 入口）窗口内写 → 违反。"""
        loop.process_turn("用户: 你好")
        orig_encode = loop._encode_query_vector

        def mutating_encode(text):
            loop.turn_count += 1
            return orig_encode(text)

        monkeypatch.setattr(loop, "_encode_query_vector", mutating_encode)
        with pytest.raises(ReadOnlyViolation):
            loop.recall_merged_readonly("你好", k=5)

    def test_react_readonly_guard_armed(self, loop, monkeypatch):
        """/react 检索段（infer-only 读口）窗口内写 → 违反。"""
        loop.process_turn("用户: 你好")
        orig_encode = loop.encoder.encode

        def mutating_encode(*args, **kwargs):
            loop.turn_count += 1
            return orig_encode(*args, **kwargs)

        monkeypatch.setattr(loop.encoder, "encode", mutating_encode)
        with pytest.raises(ReadOnlyViolation):
            loop.react_readonly("你好", k=0)

    def test_react_readonly_guard_pass_normal(self, loop):
        """正常 /react 路径：守卫通过（四不变零增量）。"""
        loop.process_turn("用户: 你好")
        before = {
            "turn": loop.turn_count,
            "J": loop.attractor.J.clone(),
            "sigma": loop.attractor.sigma.clone(),
            "episodic_size": loop.memory.episodic_size(),
        }
        result = loop.react_readonly("你好", k=3)
        assert result["turn_count"] == before["turn"]
        assert torch.equal(loop.attractor.J, before["J"])
        assert torch.equal(loop.attractor.sigma, before["sigma"])
        assert loop.memory.episodic_size() == before["episodic_size"]

    def test_recall_readonly_guard_pass_normal(self, loop):
        """正常 /recall 路径：守卫通过。"""
        loop.process_turn("用户: 你好")
        before = {
            "turn": loop.turn_count,
            "J": loop.attractor.J.clone(),
            "sigma": loop.attractor.sigma.clone(),
            "episodic_size": loop.memory.episodic_size(),
        }
        loop.recall_merged_readonly("用户: 你好", k=5)
        assert loop.turn_count == before["turn"]
        assert torch.equal(loop.attractor.J, before["J"])
        assert torch.equal(loop.attractor.sigma, before["sigma"])
        assert loop.memory.episodic_size() == before["episodic_size"]

    def test_last_recall_suspicion_populated(self, loop_precision_adapt):
        """怀疑投影进 last_recall_suspicion（内存态），条目零改写。"""
        loop = loop_precision_adapt
        loop.process_turn("用户: 你好")
        # 写侧合法置 labile（模拟去稳定化，写侧时相动作）
        entry = next(iter(loop.memory.iter_episodic()))
        entry.labile = True
        entry.labile_since = 123.0
        labile_before = entry.labile

        loop.recall_merged_readonly("用户: 你好", k=5)
        proj = loop.last_recall_suspicion
        assert set(proj) == {"labile", "rebuttal_pending",
                             "verdict_suspect", "summary"}
        assert proj["summary"]["total"] >= 1
        assert len(proj["labile"]) >= 1
        assert proj["labile"][0]["text"] == "用户: 你好"
        # 投影不落库：条目 labile 保持原值（检索绝不改写）
        assert entry.labile is labile_before

    def test_react_suspicion_projection(self, loop_precision_adapt):
        loop = loop_precision_adapt
        loop.process_turn("用户: 你好")
        result = loop.react_readonly("你好", k=3)
        assert isinstance(loop.last_recall_suspicion, dict)
        assert "summary" in loop.last_recall_suspicion
        assert result["turn_count"] == 1  # 只读锚点保持


# ============================================================
# 4. API 集成：suspicion 区段 + 违反 500 + 快照扫描
# ============================================================

class TestRecallEndpointM2:
    """/recall /react：既有结构同构 + suspicion 独立区段 + 违反 500。"""

    def test_recall_response_has_suspicion_section(self, client):
        """/recall 响应：既有字段逐字节保留，suspicion 追加在最后。"""
        client.post("/chat", json={
            "session_id": "main", "user_input": "用户: 你好"})
        r = client.post("/recall", json={
            "session_id": "main", "query": "用户: 你好", "k": 5})
        assert r.status_code == 200
        body = r.json()
        # 既有字段（§4.1 同构，逐字节未动）
        for key in ("session_id", "query", "k", "count", "results",
                    "duration_ms", "turn_count", "doubt"):
            assert key in body, f"既有字段缺失: {key}"
        # suspicion 独立区段追加在最后（§4.2）
        assert list(body)[-1] == "suspicion"
        assert set(body["suspicion"]) == {
            "labile", "rebuttal_pending", "verdict_suspect", "summary"}
        assert body["suspicion"]["summary"]["total"] >= 1
        # 条目字段不变量（结果条目结构未动）
        for item in body["results"]:
            assert "text" in item and "score" in item

    def test_react_response_has_suspicion_section(self, client):
        r = client.post("/react", json={"user_input": "你好", "k": 0})
        assert r.status_code == 200
        body = r.json()
        for key in ("session_id", "turn_count", "reaction",
                    "interpretation", "recalled", "detail", "duration_ms"):
            assert key in body, f"既有字段缺失: {key}"
        assert list(body)[-1] == "suspicion"
        assert body["suspicion"]["labile"] == []

    def test_recall_violation_maps_500(self, client, monkeypatch):
        """只读四不变违反 → /recall 500（绝不 fail-open 掩盖）。"""
        client.post("/chat", json={
            "session_id": "main", "user_input": "用户: 你好"})
        loop = server_module._session_manager.get("main")
        orig_encode = loop._encode_query_vector

        def mutating_encode(text):
            loop.turn_count += 1  # 模拟写路径偷偷 +turn
            return orig_encode(text)

        monkeypatch.setattr(loop, "_encode_query_vector", mutating_encode)
        r = client.post("/recall", json={
            "session_id": "main", "query": "用户: 你好", "k": 5})
        assert r.status_code == 500
        assert "只读四不变" in r.json()["detail"]

    def test_react_violation_maps_500(self, client, monkeypatch):
        """只读四不变违反 → /react 500。"""
        loop = server_module._session_manager.get_or_create("main")
        orig_encode = loop.encoder.encode

        def mutating_encode(*args, **kwargs):
            loop.turn_count += 1
            return orig_encode(*args, **kwargs)

        monkeypatch.setattr(loop.encoder, "encode", mutating_encode)
        r = client.post("/react", json={"user_input": "你好", "k": 0})
        assert r.status_code == 500
        assert "只读四不变" in r.json()["detail"]

    def test_snapshot_scan_no_updated_by_retrieval(self, client):
        """§5.1 快照一致性扫描：条目无 updated_by=retrieval 记录。"""
        client.post("/chat", json={
            "session_id": "main", "user_input": "用户: 你好"})
        client.post("/recall", json={
            "session_id": "main", "query": "用户: 你好", "k": 5})
        loop = server_module._session_manager.get("main")
        viol = []
        for e in loop.memory.iter_episodic():
            if getattr(e, "updated_by", None) == "retrieval":
                viol.append(e)
            rc = getattr(e, "rebuttal_consistency", None)
            if isinstance(rc, dict) and rc.get("updated_by") == "retrieval":
                viol.append(e)
        assert viol == []
        # 响应条目里也不得出现 updated_by=retrieval
        r = client.post("/recall", json={
            "session_id": "main", "query": "用户: 你好", "k": 5})
        assert r.status_code == 200
        for item in r.json()["results"]:
            assert item.get("updated_by") != "retrieval"

    def test_guard_not_disabled_by_conftest(self, client):
        """H 模式：conftest 未把增量检查置 0（守卫默认强制开启）。"""
        from core.recall.guard import FourInvariantGuard as G
        assert G(lambda: {}).enabled is True
        # 环境里不存在可关闭守卫的变量（防误配）
        assert os.environ.get("LMS_RECALL_READONLY") is None

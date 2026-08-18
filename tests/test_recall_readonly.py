# -*- coding: utf-8 -*-
"""recall 纯只读修复测试（2026-08-17，dandan 点名基础问题）。

背景：/recall 及 lms-http 工具调用每次调用 +turn（754=750+4 实证——turn
计数被工具调用污染，影响 allostatic 统计与 turn 语义）。

根因（代码级实证）：**lms_http_mcp.py::lms_recall 误 POST /chat**（写路径，
内部 process_turn → runtime/loop.py:597 ``self.turn_count += 1``），每次
lms-http 检索 +1 turn。server.py /recall 端点本身纯只读（T2.3 注释与实现
一致：不 process_turn、不调 LLM、不写缓冲、不落盘；仅进程内观测缓存）。

覆盖：
  1. loop.recall_merged_readonly / recall_episodic_readonly 零持久化
     （turn_count / episodic_size / J / sigma 四不变）
  2. API POST /recall 调用前后 turn_count 不变（只读锚点）
  3. 连续多次 /recall 零增量（回归 754=750+4 的形态）
  4. /status、/landscape 只读端点零增量（read-only sweep）
  5. 桥回归：lms_http_mcp.lms_recall 必须走 /recall 只读端点，绝不走 /chat
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


class MockEmbedder(SimpleEmbedder):
    """带 embed_text 的确定性嵌入器（相同文本 → 相同向量）。

    与 test_store_endpoint 同款：SimpleEmbedder 无语义嵌入能力（无
    embed_text），无法让检索路径真正跑起来；MockEmbedder 提供确定性
    语义向量，使 store → recall 全链路可测。
    """

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
    """轻量级配置工厂（与 test_api_server / test_react_readonly 同款）。"""
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
        # 归档隔离到 tmp（防测试导出污染仓库 data/archive/）
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
    """轻量级 TestClient（无 LLM bridge，不自动做梦；归档目录隔离）。"""
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
    """独立的 LivingMemoryLoop 实例（零持久化断言用）。"""
    cfg = make_config_factory(str(tmp_path / "snapshots"))()
    return LivingMemoryLoop(cfg)


# ============================================================
# 1. loop 层：recall 只读实现零持久化
# ============================================================

class TestRecallReadonlyZeroPersistence:
    """/recall 四不变：J / sigma / episodic_size / turn_count。"""

    def _snapshot_state(self, loop):
        return {
            "J": loop.attractor.J.clone(),
            "sigma": loop.attractor.sigma.clone(),
            "episodic_size": loop.memory.episodic_size(),
            "turn_count": loop.turn_count,
        }

    def test_recall_episodic_readonly_zero_persistence(self, loop):
        """先 process_turn 建立状态，再 recall_episodic_readonly——四不变。"""
        loop.process_turn("用户: 你好")
        before = self._snapshot_state(loop)

        result = loop.recall_episodic_readonly("用户: 你好", k=5)

        after = self._snapshot_state(loop)
        assert after["turn_count"] == before["turn_count"]
        assert after["episodic_size"] == before["episodic_size"]
        assert torch.equal(after["J"], before["J"])
        assert torch.equal(after["sigma"], before["sigma"])
        # 只读实现也能正常检索（语义检索不依赖写路径）
        assert isinstance(result, list)

    def test_recall_merged_readonly_zero_persistence(self, loop):
        """合并检索（内存+归档）同样零增量（T2.3 只读语义）。"""
        loop.process_turn("用户: 你好")
        before = self._snapshot_state(loop)

        result = loop.recall_merged_readonly("用户: 你好", k=5)

        after = self._snapshot_state(loop)
        assert after["turn_count"] == before["turn_count"]
        assert after["episodic_size"] == before["episodic_size"]
        assert torch.equal(after["J"], before["J"])
        assert torch.equal(after["sigma"], before["sigma"])
        assert isinstance(result, list)

    def test_recall_returns_texts(self, loop):
        """检索返回文本条目（只读查询面正常）。"""
        loop.process_turn("用户: 我们决定采用方案A")
        result = loop.recall_merged_readonly("方案A", k=5)
        texts = [r["text"] for r in result]
        assert any("方案A" in t for t in texts)


# ============================================================
# 2. API 层：POST /recall 只读锚点
# ============================================================

class TestRecallEndpointReadonly:
    """API /recall 调用前后 turn_count 不变（只读锚点，回归 754=750+4）。"""

    def test_recall_turn_count_anchor(self, client):
        """先写一轮（turn 0→1），再 /recall——turn 保持 1 不变。"""
        client.post("/chat", json={
            "session_id": "main", "user_input": "用户: 你好"})
        tc0 = client.get("/status/main").json()["status"]["turn_count"]
        assert tc0 == 1

        r = client.post("/recall", json={
            "session_id": "main", "query": "你好", "k": 5})
        assert r.status_code == 200
        # 响应自带只读校验锚点
        assert r.json()["turn_count"] == tc0
        # 调用后仍不变
        tc1 = client.get("/status/main").json()["status"]["turn_count"]
        assert tc1 == tc0

    def test_recall_repeated_zero_increment(self, client):
        """连续 4 次 /recall：turn 零增量（实证形态 750→754 的反例）。"""
        client.post("/chat", json={
            "session_id": "main", "user_input": "用户: 你好"})
        tc0 = client.get("/status/main").json()["status"]["turn_count"]

        for _ in range(4):
            r = client.post("/recall", json={
                "session_id": "main", "query": "你好", "k": 5})
            assert r.status_code == 200
            assert r.json()["turn_count"] == tc0

        tc1 = client.get("/status/main").json()["status"]["turn_count"]
        assert tc1 == tc0

    def test_recall_empty_query_400(self, client):
        """空 query → 400（fail-closed，端点既有语义）。"""
        r = client.post("/recall", json={
            "session_id": "main", "query": "  "})
        assert r.status_code == 400

    def test_recall_creates_session_without_turn(self, client):
        """会话不存在时惰性创建：创建本身不产生 turn 增量。"""
        # 先确认会话不存在
        r0 = client.get("/status/new_sid")
        assert r0.status_code == 404
        # /recall 惰性创建
        r = client.post("/recall", json={
            "session_id": "new_sid", "query": "你好"})
        assert r.status_code == 200
        assert r.json()["turn_count"] == 0
        tc1 = client.get("/status/new_sid").json()["status"]["turn_count"]
        assert tc1 == 0

    def test_readonly_endpoints_sweep(self, client):
        """/status + /landscape 也只读：turn 零增量（read-only sweep）。"""
        client.post("/chat", json={
            "session_id": "main", "user_input": "用户: 你好"})
        tc0 = client.get("/status/main").json()["status"]["turn_count"]

        client.get("/status/main")
        client.get("/landscape/main")
        client.get("/landscape/main?raw=0")

        tc1 = client.get("/status/main").json()["status"]["turn_count"]
        assert tc1 == tc0

    def test_landscape_missing_session_fail_open(self, client):
        """/landscape 对不存在会话：不创建、不 404、零副作用。"""
        r = client.get("/landscape/ghost")
        assert r.status_code == 200
        assert r.json()["fail_open"] is True
        # 会话未被创建
        assert client.get("/status/ghost").status_code == 404


# ============================================================
# 3. 桥回归：lms_http_mcp.lms_recall 必须走 /recall 只读端点
# ============================================================

class _FakeResp:
    """requests.Response 的最小替身。"""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class TestLmsHttpBridgeRecall:
    """lms-http 桥回归：检索只走 /recall，绝不走 /chat（+turn 根因）。"""

    def test_bridge_recall_targets_recall_endpoint(self, monkeypatch):
        """POST 目标必须是 /recall；请求体用 query 字段；turn 来自锚点。"""
        import lms_http_mcp as bridge

        calls = []

        def fake_post(url, json=None, timeout=None):
            calls.append((url, json))
            assert url.endswith("/recall"), (
                f"检索必须走 /recall 只读端点，实际: {url}")
            assert "query" in json, "检索请求体必须携带 query 字段"
            assert "/chat" not in url
            return _FakeResp({
                "session_id": "main", "query": json["query"], "k": 5,
                "count": 1,
                "results": [{"text": "用户: 你好", "score": 0.9}],
                "turn_count": 7,
            })

        def fake_get(url, timeout=None):
            assert url.endswith("/status/main")
            return _FakeResp({
                "session_id": "main",
                "status": {"turn_count": 7, "num_nodes": 32},
            })

        monkeypatch.setattr(bridge.requests, "post", fake_post)
        monkeypatch.setattr(bridge.requests, "get", fake_get)

        r = bridge.lms_recall("你好", "main")
        assert r["success"] is True
        assert r["turn_count"] == 7
        assert "记忆 1" in r["memory_context"]
        assert r["memory_state"].get("turn_count") == 7
        # 全程只发生一次 POST 且目标为 /recall
        assert len(calls) == 1
        assert calls[0][0].endswith("/recall")

    def test_bridge_recall_zero_turn_increment(self, monkeypatch):
        """模拟服务端只读语义：连续 4 次桥检索 turn 锚点恒定（754=750+4 回归）。"""
        import lms_http_mcp as bridge

        turn = 5  # 服务端 turn 恒定（/recall 只读，不增量）

        def fake_post(url, json=None, timeout=None):
            assert url.endswith("/recall")
            return _FakeResp({
                "session_id": "main", "results": [], "turn_count": turn})

        def fake_get(url, timeout=None):
            assert url.endswith("/status/main")
            return _FakeResp({
                "session_id": "main",
                "status": {"turn_count": turn},
            })

        monkeypatch.setattr(bridge.requests, "post", fake_post)
        monkeypatch.setattr(bridge.requests, "get", fake_get)

        for _ in range(4):
            r = bridge.lms_recall("查询", "main")
            assert r["success"] is True
            assert r["turn_count"] == turn

    def test_bridge_recall_status_fail_open(self, monkeypatch):
        """状态 GET 失败不影响检索结果（/recall 锚点兜底）。"""
        import lms_http_mcp as bridge

        def fake_post(url, json=None, timeout=None):
            assert url.endswith("/recall")
            return _FakeResp({
                "session_id": "main",
                "results": [{"text": "用户: 你好", "score": 0.8}],
                "turn_count": 3,
            })

        def fake_get(url, timeout=None):
            raise RuntimeError("status 服务不可用")

        monkeypatch.setattr(bridge.requests, "post", fake_post)
        monkeypatch.setattr(bridge.requests, "get", fake_get)

        r = bridge.lms_recall("你好", "main")
        assert r["success"] is True
        assert r["turn_count"] == 3  # 来自 /recall 锚点
        assert "记忆 1" in r["memory_context"]

    def test_bridge_recall_failure_fail_open(self, monkeypatch):
        """检索 POST 异常 → success=False（调用方 fail-open 降级）。"""
        import lms_http_mcp as bridge

        def fake_post(url, json=None, timeout=None):
            raise RuntimeError("连接拒绝")

        monkeypatch.setattr(bridge.requests, "post", fake_post)

        r = bridge.lms_recall("你好", "main")
        assert r["success"] is False
        assert "error" in r

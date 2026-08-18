# -*- coding: utf-8 -*-
"""提取层 /store 端点测试（提取层 v1.4 S1-1，阶段 1 验收①-③ 接口级）

覆盖：
  1. 白名单（默认 main）：非白名单会话 → 422
  2. 幂等去重：同 payload → dedup_hit=true，turn 不重复增长
  3. 提取核心 ≤300 字落库（core_chars ≤ 300，episodic 尾部含核心）
  4. 价值过滤标记（value_filtered；低价值条目不丢——M5 永不整轮丢）
  5. 灰度标记（LMS_STORE_GRAY=1 → gray=true + source='store_gray'，
     且 L1 不可见：/recall 检索不到 gray 条目）
  6. 写侧引用加固（P2-B）：新条目引用旧条目 → 旧条目 reference_count+1
     ＋ last_reinforced_turn 刷新
  7. 熔断降级（P3）：embed 熔断 → 503 + Retry-After + degraded 响应体
  8. 限流（B7 每会话桶）→ 429
"""

import os
import time

import pytest
import torch
from fastapi.testclient import TestClient

import api.server as server_module
from api.session_manager import SessionManager
from runtime.dream_scheduler import DreamScheduler
from core.sensory.embedder import SimpleEmbedder
from runtime.config import default_config


class MockEmbedder(SimpleEmbedder):
    """带 embed_text 的确定性嵌入器（相同文本 → 相同向量）。"""

    def embed_text(self, text: str) -> torch.Tensor:
        if not text or not text.strip():
            return torch.zeros(self._dim)
        h = abs(hash(text)) % (2 ** 32)
        g = torch.Generator()
        g.manual_seed(h)
        return torch.randn(self._dim, generator=g) * 0.1

    def embed_text_raw(self, text: str) -> torch.Tensor:
        return self.embed_text(text)


class FailingEmbedder(SimpleEmbedder):
    def embed_text(self, text: str) -> torch.Tensor:
        raise ConnectionError("embed 服务不可达（测试模拟）")

    def embed_text_raw(self, text: str) -> torch.Tensor:
        raise ConnectionError("embed 服务不可达（测试模拟）")


def make_config_factory(snapshot_dir, embedder=None, **overrides):
    def factory():
        config = default_config()
        config['num_nodes'] = 32
        config['input_dim'] = 16
        config['num_infer_steps'] = 5
        config['consolidation_interval'] = 3
        config['seed'] = 42
        config['snapshot_dir'] = snapshot_dir
        # 归档隔离到 tmp（防测试导出污染仓库 data/archive/）
        config['archive_dir'] = os.path.join(snapshot_dir, 'archive')
        config['embedder'] = embedder or MockEmbedder(dim=16)
        config.update(overrides)
        return config
    return factory


def _create_scheduler(sm):
    return DreamScheduler(
        get_loop_fn=lambda sid: sm.get(sid),
        idle_threshold=3600, dream_steps=3, check_interval=3600,
    )


def _inject_globals(sm, scheduler):
    orig = (
        server_module._session_manager,
        server_module._dream_scheduler,
    )
    server_module._session_manager = sm
    server_module._dream_scheduler = scheduler
    return orig


def _restore_globals(*orig):
    (server_module._session_manager,
     server_module._dream_scheduler) = orig


@pytest.fixture
def client(monkeypatch, tmp_path):
    """带 MockEmbedder 的 /store 测试客户端。"""
    monkeypatch.setenv("LMS_STORE_RATE_LIMIT", "1000")  # 防限流干扰
    monkeypatch.setenv("LMS_STORE_DEDUP_WINDOW", "60")
    monkeypatch.setenv("LMS_STORE_GRAY", "0")
    monkeypatch.setenv("LMS_STORE_SESSION_ALLOWLIST", "main")
    snapshot_dir = str(tmp_path / "snapshots")
    sm = SessionManager(default_config_factory=make_config_factory(snapshot_dir))
    scheduler = _create_scheduler(sm)
    orig = _inject_globals(sm, scheduler)
    try:
        with TestClient(server_module.app) as c:
            yield c
    finally:
        _restore_globals(*orig)


def _sm_of():
    return server_module._session_manager


def _loop_of(sid="main"):
    return _sm_of().get(sid)


class TestStoreAllowlist:
    def test_non_whitelisted_session_422(self, client):
        resp = client.post("/store", json={
            "session_id": "design-test",
            "user_input": "你好",
            "llm_output": "这是一段回复",
        })
        assert resp.status_code == 422

    def test_main_allowed(self, client):
        resp = client.post("/store", json={
            "session_id": "main",
            "user_input": "帮我对比 A/B 方案",
            "llm_output": "实际上方案A更合适，我们决定采用方案A。",
        })
        assert resp.status_code == 200

    def test_empty_input_400(self, client):
        resp = client.post("/store", json={
            "session_id": "main", "user_input": "", "llm_output": ""})
        assert resp.status_code == 400


class TestStorePipeline:
    def test_store_writes_core_entry(self, client):
        """验收①：核心 ≤300 字落库；turn 增量；episodic 尾部含核心。"""
        before_turn = _loop_of("main").turn_count if _loop_of("main") else 0
        resp = client.post("/store", json={
            "session_id": "main",
            "user_input": "帮我对比 A/B 方案",
            "llm_output": "好的。" + "实际上第%d点很关键，正确的是这样做。" * 40,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["stored"] is True
        assert data["core_chars"] <= 300
        assert data["turn_count"] == before_turn + 1
        # episodic 尾部条目含核心
        loop = _loop_of("main")
        entries = list(loop.memory.iter_episodic())
        assert entries[-1].core is not None
        assert len(entries[-1].core) <= 300
        # 条目 meta 齐全（S1-11 字段）
        e = entries[-1]
        assert e.last_reinforced_turn == loop.turn_count - 1
        assert e.ts is not None
        assert e.info_value > 0

    def test_idempotent_dedup(self, client):
        """验收②：同 payload → dedup_hit=true，turn 不重复增长。"""
        payload = {
            "session_id": "main",
            "user_input": "幂等测试",
            "llm_output": "我们决定采用方案B。",
        }
        r1 = client.post("/store", json=payload)
        assert r1.status_code == 200
        t1 = r1.json()["turn_count"]
        r2 = client.post("/store", json=payload)
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["dedup_hit"] is True
        assert d2["turn_count"] == t1  # 不重复处理

    def test_value_filtered_marks_not_kills(self, client):
        """验收③/M5 契约：用户段多短都转发（永不整轮丢），value_filtered
        只做条目标记（数值阈值阶段 2 开闸后按 H-4 数据标定；公式级断言
        见 test_value_filter）。"""
        # 先存一条高惊讶条目（建立 min-max 归一化基准）
        client.post("/store", json={
            "session_id": "main",
            "user_input": "帮我看一个复杂的技术方案问题",
            "llm_output": "实际上这个问题涉及多个关键环节，"
                          "正确的是先验证再实施，我们决定采用分层方案。" * 5,
        })
        resp = client.post("/store", json={
            "session_id": "main",
            "user_input": "嗯",
            "llm_output": "好的。",   # 短确认词 + 客套回复
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["stored"] is True      # 永不整轮丢（M5）
        assert isinstance(data["value_filtered"], bool)  # 标记语义
        assert data["info_value"] >= 0.0
        # episodic 仍含「用户: 嗯」（用户段不丢）
        entries = list(_loop_of("main").memory.iter_episodic())
        assert any("用户: 嗯" in e.text for e in entries)

    def test_gray_marking_and_invisibility(self, client, monkeypatch):
        """灰度标记：gray=true + source='store_gray'；L1 检索不可见。"""
        # 请求时读取 env（灰度"随时可关"语义：改动即时生效）
        monkeypatch.setenv("LMS_STORE_GRAY", "1")
        resp = client.post("/store", json={
            "session_id": "main",
            "user_input": "灰度测试",
            "llm_output": "我们决定采用方案C。",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["gray"] is True
        entries = list(_loop_of("main").memory.iter_episodic())
        last = entries[-1]
        assert last.gray is True
        assert last.source == "store_gray"
        # L1 不可见：/recall 检索不到 gray 条目
        r = client.post("/recall", json={
            "session_id": "main", "query": "灰度测试", "k": 5})
        assert r.status_code == 200
        texts = [item["text"] for item in r.json()["results"]]
        assert all("灰度测试" not in t for t in texts)


class TestWriteSideReinforcement:
    def test_referenced_old_entry_reinforced(self, client):
        """P2-B：新条目引用旧条目 → 旧条目 reference_count+1 且加固刷新。"""
        # 先存一条旧条目
        client.post("/store", json={
            "session_id": "main",
            "user_input": "第一次讨论方案A",
            "llm_output": "我们决定采用方案A。",
        })
        loop = _loop_of("main")
        old_entry = list(loop.memory.iter_episodic())[-1]
        assert old_entry.reference_count == 0
        old_turn = old_entry.turn

        # 再存一条高度相似的新条目（相同语义内容 → 引用匹配命中）
        client.post("/store", json={
            "session_id": "main",
            "user_input": "再次确认方案A",
            "llm_output": "我们决定采用方案A。",
        })
        # 旧条目被加固（reference_count ≥ 1；last_reinforced_turn 刷新）
        assert old_entry.reference_count >= 1
        assert old_entry.last_reinforced_turn > old_turn


class TestStoreDegraded:
    def test_503_on_embed_circuit_open(self, monkeypatch, tmp_path):
        """P3：embed 熔断 → 503 + Retry-After:30 + degraded 响应体。"""
        import core.sensory.circuit_breaker as cb_module
        import bridge.encoder as enc_module
        breaker = cb_module.EmbedCircuitBreaker(
            enabled=True, max_failures=1, cooldown=3600.0)
        breaker.record_failure()  # → OPEN
        monkeypatch.setattr(cb_module, "get_default_embed_circuit",
                            lambda: breaker)
        monkeypatch.setattr(enc_module, "get_default_embed_circuit",
                            lambda: breaker)
        monkeypatch.setenv("LMS_STORE_RATE_LIMIT", "1000")
        snapshot_dir = str(tmp_path / "snapshots")
        sm = SessionManager(default_config_factory=make_config_factory(
            snapshot_dir, embedder=FailingEmbedder(dim=16)))
        scheduler = _create_scheduler(sm)
        orig = _inject_globals(sm, scheduler)
        try:
            with TestClient(server_module.app) as c:
                # 读侧熔断器也 OPEN（生产场景三处全 OPEN）；先确保会话存在
                loop = _sm_of().get_or_create("main")
                for _ in range(3):
                    with pytest.raises(ConnectionError):
                        loop._embed_circuit.call(
                            lambda: (_ for _ in ()).throw(ConnectionError()))
                resp = c.post("/store", json={
                    "session_id": "main",
                    "user_input": "降级测试",
                    "llm_output": "我们决定采用方案D。",
                })
                assert resp.status_code == 503
                assert resp.headers.get("Retry-After") == "30"
                body = resp.json()
                assert body.get("status") == "degraded"
                assert body.get("reason") == "embed_circuit_open"
                assert server_module.store_503_count() >= 1
        finally:
            _restore_globals(*orig)


class TestStoreRateLimit:
    def test_rate_limit_429(self, monkeypatch, tmp_path):
        """B7：每会话限流 → 429 + Retry-After。"""
        monkeypatch.setenv("LMS_STORE_RATE_LIMIT", "2")
        snapshot_dir = str(tmp_path / "snapshots")
        sm = SessionManager(default_config_factory=make_config_factory(snapshot_dir))
        scheduler = _create_scheduler(sm)
        orig = _inject_globals(sm, scheduler)
        try:
            with TestClient(server_module.app) as c:
                payload = {
                    "session_id": "main",
                    "user_input": f"限流测试{time.time()}",
                    "llm_output": "我们决定采用方案E。",
                }
                for _ in range(2):
                    r = c.post("/store", json=payload)
                    assert r.status_code in (200, 429)
                r3 = c.post("/store", json={
                    "session_id": "main",
                    "user_input": f"限流测试再发{time.time()}",
                    "llm_output": "我们决定采用方案E。",
                })
                assert r3.status_code == 429
                assert "Retry-After" in r3.headers
        finally:
            _restore_globals(*orig)


class TestStoreProcessCoreWiring:
    """P1-1 条目级过程字段接线（api/server.py /store writer 路径 → ingest）：

    验收（目的性审计 §3.2 / P1-1 待集成点接线）：
      1. /store 响应携带 process_core / text_snapshot / evolution 区段；
      2. 落库条目携带 process_core 字段（has_process_core=True），演化史以
         created 为开端（append-only）；
      3. 幂等去重命中路径同样携带过程字段（WriteResult 原样返回首次结果）。
    """

    def test_store_response_carries_process_fields(self, client):
        """响应 + 条目双双携带 P1-1 过程字段（提取过程核心优先）。"""
        resp = client.post("/store", json={
            "session_id": "main",
            "user_input": "这让我很惊讶：方案被推翻了，需要修正，"
                          "现在认为应该转向新立场。",
            "llm_output": "确实出乎意料，我们改变了主意。",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["stored"] is True
        # 响应 P1-1 区段
        pc = data["process_core"]
        assert isinstance(pc, dict)
        assert sorted(pc.keys()) == sorted([
            "surprise_trace", "doubt_events", "turns",
            "open_tails", "confidence_curve", "surprise_source",
        ])
        assert isinstance(data["text_snapshot"], str)
        assert data["text_snapshot"]
        assert isinstance(data["evolution"], dict)
        assert data["evolution"]["updated_by"] == "ingest"
        states = [t["state"] for t in data["evolution"]["history"]]
        assert states == ["created"]
        # 条目级：episodic 尾部条目携带过程字段（M7 旧条目兼容由
        # process_core 层保证——getattr 兜底）
        from core.store.process_core import (
            PROCESS_CORE_FIELDS,
            get_evolution,
            get_process_core,
            has_process_core,
        )
        entries = list(_loop_of("main").memory.iter_episodic())
        assert entries
        e = entries[-1]
        assert has_process_core(e) is True
        entry_pc = get_process_core(e)
        assert sorted(entry_pc.keys()) == sorted(PROCESS_CORE_FIELDS)
        # 含惊讶/转向信号 → 提取过程核心优先（§3.2 断言非空段）
        assert entry_pc["surprise_trace"]
        assert entry_pc["turns"]
        ev = get_evolution(e)
        assert ev["updated_by"] == "ingest"
        assert ev["history"][0]["state"] == "created"

    def test_store_dedup_hit_carries_process_fields(self, client):
        """幂等去重命中：dedup_hit=True 且过程字段同构返回（不重复写）。

        process_core/text_snapshot 为确定性解析产物（同文本同结果）；
        演化史 at 时间戳为本次解析值（幂等记录 = 纯 writer 结果——
        core.store 既有契约），断言 created 开端与写者即可。
        """
        payload = {
            "session_id": "main",
            "user_input": "幂等过程字段测试",
            "llm_output": "我们决定采用方案F。",
        }
        r1 = client.post("/store", json=payload)
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["stored"] is True
        assert d1["process_core"] is not None
        r2 = client.post("/store", json=payload)
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["dedup_hit"] is True
        assert d2["turn_count"] == d1["turn_count"]
        # 幂等命中路径同样携带过程字段（确定性产物一致）
        assert d2["process_core"] == d1["process_core"]
        assert d2["text_snapshot"] == d1["text_snapshot"]
        assert d2["evolution"]["updated_by"] == "ingest"
        assert [t["state"] for t in d2["evolution"]["history"]] == ["created"]

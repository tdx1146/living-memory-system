# -*- coding: utf-8 -*-
"""体验层 A（设计 v1.1 §3）：/react 实时反应只读端点测试。

零持久化断言（§3.6 验收 2）：J 矩阵 / sigma / episodic_size / turn_count
四不变——/react 是 infer-only 读口，P0-12 防查询回声零回归。
覆盖：
  1. loop.react_readonly 直接调用（零持久化 + 解读段 + k=0/k>0）
  2. API POST /react（200 / 400 / turn_count 锚点 / 只读检索）
  3. /react 自有惊讶窗口不污染 decoder 共享窗口（/chat 路径零回归）
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


def make_config_factory(snapshot_dir: str):
    """轻量级配置工厂（与 test_api_server 同款：小网络 + SimpleEmbedder）。"""
    def factory():
        config = default_config()
        config['num_nodes'] = 32
        config['input_dim'] = 16
        config['num_infer_steps'] = 5
        config['consolidation_interval'] = 3
        config['seed'] = 42
        config['llm_api'] = None
        config['embedder'] = SimpleEmbedder(dim=16)
        config['auto_snapshot'] = False
        config['snapshot_dir'] = snapshot_dir
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
    """轻量级 TestClient（无 LLM bridge，不自动做梦）。"""
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
# 1. loop.react_readonly 零持久化 + 语义
# ============================================================

class TestReactReadonlyZeroPersistence:
    """/react 四不变：J / sigma / episodic_size / turn_count。"""

    def _snapshot_state(self, loop):
        return {
            "J": loop.attractor.J.clone(),
            "sigma": loop.attractor.sigma.clone(),
            "episodic_size": loop.memory.episodic_size(),
            "turn_count": loop.turn_count,
        }

    def test_react_readonly_zero_persistence(self, loop):
        """先 process_turn 建立状态，再 /react——四不变。"""
        loop.process_turn("用户: 你好")
        before = self._snapshot_state(loop)

        result = loop.react_readonly("用户: 你好，我的记忆系统", k=0)

        after = self._snapshot_state(loop)
        assert after["turn_count"] == before["turn_count"]
        assert after["episodic_size"] == before["episodic_size"]
        assert torch.equal(after["J"], before["J"])
        assert torch.equal(after["sigma"], before["sigma"])
        # 返回契约字段
        assert result["turn_count"] == before["turn_count"]
        assert result["reaction"]["surprise"] >= 0.0  # 准确性项恒≥0
        assert "free_energy" in result["reaction"]
        assert isinstance(result["interpretation"], str)
        assert result["interpretation"]  # 非空
        assert result["recalled"] == []  # k=0 轻量模式
        assert "detail" in result

    def test_react_readonly_interpretation_entropy(self, loop):
        """解读段含熵解读（复用 decoder 模板）。"""
        result = loop.react_readonly("这是一个测试输入", k=0)
        # 熵解读必在（恒有）——三种模板之一
        assert any(
            s in result["interpretation"]
            for s in ("当前记忆处于高唤醒状态", "当前记忆聚焦于少数模式",
                      "记忆激活适度")
        )

    def test_react_readonly_k_returns_recalled(self, loop):
        """k>0 时返回只读检索条目（先存一条记忆）。"""
        loop.process_turn("用户: 我喜欢蓝色的小橘猫")
        result = loop.react_readonly("蓝色的小橘猫", k=3)
        assert result["turn_count"] == 1  # 仍只读：process_turn 只跑过 1 轮
        assert isinstance(result["recalled"], list)
        if result["recalled"]:
            item = result["recalled"][0]
            assert "text" in item and "score" in item and "origin" in item

    def test_react_readonly_does_not_learn(self, loop):
        """infer-only：J 零 diff（不 learn 的硬断言）。"""
        j_before = loop.attractor.J.clone()
        for _ in range(3):
            loop.react_readonly("纯反应轮，不学习", k=0)
        assert torch.equal(loop.attractor.J, j_before)

    def test_react_window_does_not_pollute_decoder(self, loop):
        """/react 自有惊讶窗口不污染 decoder 共享窗口（/chat 路径零回归）。"""
        loop.react_readonly("反应一", k=0)
        loop.react_readonly("反应二", k=0)
        loop.react_readonly("反应三", k=0)
        # decoder 共享窗口保持为空（/react 从不写它）
        assert len(loop.decoder.surprise_history) == 0
        # /react 自有窗口已累积
        assert len(loop.react_surprise_history) == 3


# ============================================================
# 2. API 端点
# ============================================================

class TestReactEndpoint:
    """POST /react HTTP 端点。"""

    def test_react_endpoint_ok(self, client):
        r = client.post("/react", json={"user_input": "你好", "k": 0})
        assert r.status_code == 200
        body = r.json()
        assert body["session_id"] == "main"
        assert body["reaction"]["surprise"] >= 0.0
        assert body["interpretation"]
        assert body["recalled"] == []
        assert "duration_ms" in body

    def test_react_turn_count_anchor(self, client):
        """/react 调用前后 turn_count 不变（只读锚点）。"""
        r0 = client.post("/react", json={"user_input": "锚点一", "k": 0})
        tc0 = r0.json()["turn_count"]
        r1 = client.post("/react", json={"user_input": "锚点二", "k": 0})
        assert r1.json()["turn_count"] == tc0

    def test_react_empty_input_400(self, client):
        r = client.post("/react", json={"user_input": "", "k": 0})
        assert r.status_code == 400

    def test_react_k_clamped(self, client):
        """k 钳制到 [0,10]（超限不报错）。"""
        r = client.post("/react", json={"user_input": "钳制测试", "k": 99})
        assert r.status_code == 200
        assert isinstance(r.json()["recalled"], list)

    def test_react_sid_alias(self, client):
        """旧客户端 sid 兼容映射。"""
        r = client.post("/react", json={"user_input": "sid 测试", "sid": "test-abc"})
        assert r.status_code == 200
        assert r.json()["session_id"] == "test-abc"

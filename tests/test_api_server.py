# -*- coding: utf-8 -*-
"""API 层端点测试：FastAPI 服务 (api/server.py)

使用 FastAPI TestClient 对所有 HTTP 端点进行集成测试，覆盖：
  1. 健康检查 (GET /health)
  2. 会话管理 (GET /sessions, DELETE /sessions/{sid})
  3. 对话端点 (POST /chat, POST /chat/simple)
  4. 状态与快照 (GET /status/{sid}, POST /snapshot/{sid}, POST /restore/{sid})
  5. 做梦端点 (GET /dream/status, POST /dream/{sid})

测试设计原则：
  - 使用轻量级 config（num_nodes=32, input_dim=16, SimpleEmbedder）
    避免加载预训练模型，确保无 GPU / 无网络 / 无预训练模型环境下可运行。
  - 对 LLM 调用使用 MockLLMBridge，不实际调用外部 API。
  - 通过注入自定义 SessionManager 和 DreamScheduler 单例到 api.server 模块，
    隔离全局状态，保证测试独立性。
  - 每个测试独立运行，不依赖其他测试的执行顺序。
  - 使用 tmp_path 处理快照文件目录。
  - DreamScheduler 的 idle_threshold 和 check_interval 设为极大值，
    防止后台线程在测试期间自动触发做梦。
"""

import os
import sys
import threading

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


# ============================================================
# 辅助类
# ============================================================

class MockLLMBridge:
    """模拟 LLM 桥接器，返回固定响应。

    遵循 LLMBridge 的 query(user_input, memory_context) -> str 接口，
    不进行任何实际 API 调用，用于测试中隔离 LLM 依赖。
    """

    def __init__(self):
        self.query_count = 0
        self.last_user_input = None
        self.last_memory_context = None

    def query(self, user_input: str, memory_context: str) -> str:
        """返回包含用户输入摘要的固定响应。"""
        self.query_count += 1
        self.last_user_input = user_input
        self.last_memory_context = memory_context
        return f"[Mock回复] 收到: {user_input[:30]}"


# ============================================================
# 配置工厂
# ============================================================

def make_config_factory(snapshot_dir: str, with_bridge: bool = False):
    """返回轻量级配置工厂函数。

    创建小规模配置（num_nodes=32, input_dim=16）加速测试，
    使用 SimpleEmbedder 避免加载预训练模型，禁用 LLM API 调用。

    参数:
        snapshot_dir: 快照保存目录（使用 tmp_path）。
        with_bridge: 是否注入 MockLLMBridge（用于 /chat/simple 测试）。

    返回:
        配置工厂函数，调用时返回配置字典。
    """
    def factory():
        config = default_config()
        # 小规模网络，加速推断和学习
        config['num_nodes'] = 32
        config['input_dim'] = 16
        config['num_infer_steps'] = 5
        config['consolidation_interval'] = 3
        config['seed'] = 42
        # 禁用 LLM API（不调用外部服务）
        config['llm_api'] = None
        # 使用 SimpleEmbedder（无需预训练模型）
        config['embedder'] = SimpleEmbedder(dim=16)
        # 禁用自动快照（测试中手动控制）
        config['auto_snapshot'] = False
        config['snapshot_dir'] = snapshot_dir
        # 可选：注入 MockLLMBridge
        if with_bridge:
            config['llm_bridge'] = MockLLMBridge()
        return config
    return factory


def _inject_globals(sm: SessionManager, scheduler: DreamScheduler):
    """将自定义 SessionManager 和 DreamScheduler 注入 api.server 模块全局变量。

    返回原始值以便测试结束后恢复。
    """
    orig_sm = server_module._session_manager
    orig_sched = server_module._dream_scheduler
    server_module._session_manager = sm
    server_module._dream_scheduler = scheduler
    return orig_sm, orig_sched


def _restore_globals(orig_sm, orig_sched):
    """恢复 api.server 模块的全局变量。"""
    server_module._session_manager = orig_sm
    server_module._dream_scheduler = orig_sched


def _create_scheduler(sm: SessionManager) -> DreamScheduler:
    """创建测试用 DreamScheduler。

    idle_threshold 和 check_interval 设为极大值，防止后台线程
    在测试期间自动触发做梦。
    """
    return DreamScheduler(
        get_loop_fn=lambda sid: sm.get(sid),
        idle_threshold=999999,
        dream_steps=5,
        dream_full_cycle=False,
        check_interval=999999,
    )


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def client(monkeypatch, tmp_path):
    """提供轻量级 TestClient（无 LLM bridge）。

    - 设置环境变量避免加载预训练模型
    - 注入自定义 SessionManager（小规模 config）
    - 注入自定义 DreamScheduler（不自动做梦）
    - 使用 TestClient 上下文管理器触发 startup/shutdown 事件
    - 测试结束后恢复全局变量
    """
    # 设置环境变量（startup 事件中的 get_api_config() 会读取）
    monkeypatch.setenv("LMS_EMBEDDER", "simple")
    monkeypatch.setenv("LMS_NUM_NODES", "32")
    monkeypatch.setenv("LMS_INPUT_DIM", "16")
    # 确保未设置 API key，禁用 LLM
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LMS_LLM_API_KEY", raising=False)

    snapshot_dir = str(tmp_path / "snapshots")

    sm = SessionManager(
        default_config_factory=make_config_factory(snapshot_dir)
    )
    scheduler = _create_scheduler(sm)

    orig = _inject_globals(sm, scheduler)
    try:
        with TestClient(server_module.app) as c:
            yield c
    finally:
        _restore_globals(*orig)


@pytest.fixture
def client_with_bridge(monkeypatch, tmp_path):
    """提供带 MockLLMBridge 的 TestClient。

    与 client fixture 相同，但配置中注入了 MockLLMBridge，
    使 /chat/simple 端点可用（loop.bridge 不为 None）。
    """
    monkeypatch.setenv("LMS_EMBEDDER", "simple")
    monkeypatch.setenv("LMS_NUM_NODES", "32")
    monkeypatch.setenv("LMS_INPUT_DIM", "16")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("LMS_LLM_API_KEY", raising=False)

    snapshot_dir = str(tmp_path / "snapshots")

    sm = SessionManager(
        default_config_factory=make_config_factory(snapshot_dir, with_bridge=True)
    )
    scheduler = _create_scheduler(sm)

    orig = _inject_globals(sm, scheduler)
    try:
        with TestClient(server_module.app) as c:
            yield c
    finally:
        _restore_globals(*orig)


# ============================================================
# 1. 健康检查
# ============================================================

class TestHealth:
    """健康检查端点测试。"""

    def test_health_returns_ok(self, client):
        """GET /health 返回 200 和正确状态。"""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "living-memory-api"
        assert "version" in data
        assert "active_sessions" in data
        assert "timestamp" in data


# ============================================================
# 2. 会话管理
# ============================================================

class TestSessionManagement:
    """会话管理端点测试。"""

    def test_list_sessions_empty(self, client):
        """GET /sessions 初始返回空列表。"""
        resp = client.get("/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert "count" in data
        assert data["count"] == 0
        assert data["sessions"] == []

    def test_list_sessions_after_chat(self, client):
        """GET /sessions 对话后返回包含会话的列表。"""
        # 通过 /chat 创建一个会话
        client.post("/chat", json={
            "user_input": "测试会话",
            "session_id": "session_a"
        })
        resp = client.get("/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 1
        assert "session_a" in data["sessions"]

    def test_delete_nonexistent_session_404(self, client):
        """DELETE /sessions/{sid} 删除不存在的会话返回 404。"""
        resp = client.delete("/sessions/nonexistent_session")
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]

    def test_delete_existing_session(self, client):
        """DELETE /sessions/{sid} 删除已存在的会话返回 200。"""
        # 先创建会话
        client.post("/chat", json={
            "user_input": "待删除",
            "session_id": "to_delete"
        })
        # 确认会话存在
        resp = client.get("/sessions")
        assert "to_delete" in resp.json()["sessions"]

        # 删除会话
        resp = client.delete("/sessions/to_delete")
        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True
        assert data["session_id"] == "to_delete"

        # 确认会话已删除
        resp = client.get("/sessions")
        assert "to_delete" not in resp.json()["sessions"]


# ============================================================
# 3. 对话端点
# ============================================================

class TestChatEndpoint:
    """POST /chat 端点测试。"""

    def test_chat_normal_flow(self, client):
        """POST /chat 正常对话流程（无 LLM，返回记忆 context）。"""
        resp = client.post("/chat", json={
            "user_input": "你好世界",
            "session_id": "chat_test"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "chat_test"
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0
        assert isinstance(data["memory_context"], str)
        assert len(data["memory_context"]) > 0
        assert isinstance(data["memory_state"], dict)
        # 无 LLM 时 response 应等于 memory_context
        assert data["response"] == data["memory_context"]
        # memory_state 应包含关键字段
        assert "turn_count" in data["memory_state"]
        assert data["memory_state"]["turn_count"] == 1
        assert "llm_enabled" in data["memory_state"]
        assert data["memory_state"]["llm_enabled"] is False

    def test_chat_with_llm_bridge(self, client_with_bridge):
        """POST /chat 配置了 MockLLMBridge 时返回 LLM 回复。"""
        resp = client_with_bridge.post("/chat", json={
            "user_input": "你好世界",
            "session_id": "chat_llm_test"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "chat_llm_test"
        # 有 LLM 时 response 应来自 MockLLMBridge
        assert "[Mock回复]" in data["response"]
        # response 不应等于 memory_context（经过了 LLM 处理）
        assert data["response"] != data["memory_context"]
        assert data["memory_state"]["llm_enabled"] is True

    def test_chat_empty_input_400(self, client):
        """POST /chat 空输入返回 400。"""
        resp = client.post("/chat", json={
            "user_input": "",
            "session_id": "empty_test"
        })
        assert resp.status_code == 400
        assert "不能为空" in resp.json()["detail"]

    def test_chat_concurrent_503(self, client):
        """POST /chat 并发请求时 acquire 失败返回 503。

        持有 _busy_lock 模拟正在做梦/处理中，
        缩短 acquire 超时使测试快速完成。
        """
        scheduler = server_module._dream_scheduler

        # 持有 _busy_lock 模拟做梦正在进行
        scheduler._busy_lock.acquire()

        # 缩短 acquire 超时（0.2 秒），避免测试等待 10 秒
        original_acquire = scheduler.acquire_conversation

        def fast_acquire(session_id, timeout=10.0):
            return original_acquire(session_id, timeout=0.2)

        scheduler.acquire_conversation = fast_acquire

        try:
            resp = client.post("/chat", json={
                "user_input": "并发测试",
                "session_id": "concurrent_test"
            })
            assert resp.status_code == 503
            assert "做梦" in resp.json()["detail"]
        finally:
            scheduler.acquire_conversation = original_acquire
            try:
                scheduler._busy_lock.release()
            except RuntimeError:
                pass

    def test_chat_multi_turn(self, client):
        """POST /chat 多轮对话 turn_count 递增。"""
        for i in range(3):
            resp = client.post("/chat", json={
                "user_input": f"第{i + 1}轮对话",
                "session_id": "multi_turn_test"
            })
            assert resp.status_code == 200
            assert resp.json()["memory_state"]["turn_count"] == i + 1


# ============================================================
# 4. 简化对话端点
# ============================================================

class TestChatSimpleEndpoint:
    """POST /chat/simple 端点测试。"""

    def test_chat_simple_normal(self, client_with_bridge):
        """POST /chat/simple 正常流程（带 MockLLMBridge）。"""
        resp = client_with_bridge.post("/chat/simple", json={
            "user_input": "简化的对话",
            "session_id": "simple_test"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data["response"], str)
        assert len(data["response"]) > 0
        assert "[Mock回复]" in data["response"]
        assert isinstance(data["memory_status"], dict)
        assert "turn_count" in data["memory_status"]

    def test_chat_simple_no_bridge_503(self, client):
        """POST /chat/simple 未配置 LLM 时返回 503。"""
        resp = client.post("/chat/simple", json={
            "user_input": "无LLM测试",
            "session_id": "no_bridge_test"
        })
        assert resp.status_code == 503
        assert "LLM" in resp.json()["detail"] or "Bridge" in resp.json()["detail"]

    def test_chat_simple_empty_input_400(self, client_with_bridge):
        """POST /chat/simple 空输入返回 400。"""
        resp = client_with_bridge.post("/chat/simple", json={
            "user_input": "",
            "session_id": "empty_simple_test"
        })
        assert resp.status_code == 400

    def test_chat_simple_concurrent_rejection(self, client):
        """POST /chat/simple 并发拒绝：无 bridge 时拒绝请求（503）。

        /chat/simple 没有 acquire/release 机制，其"拒绝"路径
        是未配置 LLM 时返回 503。
        """
        # 无 bridge 的 client，/chat/simple 应返回 503
        resp = client.post("/chat/simple", json={
            "user_input": "并发拒绝测试",
            "session_id": "reject_test"
        })
        assert resp.status_code == 503


# ============================================================
# 5. 状态与快照
# ============================================================

class TestStatusAndSnapshot:
    """状态查询与快照端点测试。"""

    def test_get_status(self, client):
        """GET /status/{sid} 返回记忆状态。"""
        # 先通过 /chat 创建会话并处理一轮
        client.post("/chat", json={
            "user_input": "状态测试",
            "session_id": "status_test"
        })
        resp = client.get("/status/status_test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "status_test"
        assert "status" in data
        assert data["status"]["turn_count"] == 1
        assert data["status"]["num_nodes"] == 32
        assert data["status"]["input_dim"] == 16

    def test_get_status_nonexistent_404(self, client):
        """GET /status/{sid} 不存在的会话返回 404。"""
        resp = client.get("/status/nonexistent_status")
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]

    def test_snapshot_save(self, client, tmp_path):
        """POST /snapshot/{sid} 保存快照到指定路径。"""
        # 先创建会话并运行几轮
        for i in range(3):
            client.post("/chat", json={
                "user_input": f"快照前第{i + 1}轮",
                "session_id": "snap_test"
            })
        snap_path = str(tmp_path / "test_snapshot.pt")
        resp = client.post("/snapshot/snap_test", json={"path": snap_path})
        assert resp.status_code == 200
        data = resp.json()
        assert data["saved"] is True
        assert data["session_id"] == "snap_test"
        assert data["turn_count"] == 3
        # 文件应存在
        assert os.path.exists(snap_path)

    def test_snapshot_default_path(self, client):
        """POST /snapshot/{sid} 不指定 path 时使用默认路径。"""
        client.post("/chat", json={
            "user_input": "默认快照测试",
            "session_id": "snap_default_test"
        })
        resp = client.post("/snapshot/snap_default_test", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["saved"] is True
        assert os.path.exists(data["path"])

    def test_snapshot_nonexistent_session_404(self, client):
        """POST /snapshot/{sid} 不存在的会话返回 404。"""
        resp = client.post("/snapshot/nonexistent_snap", json={
            "path": "/tmp/dummy.pt"
        })
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]

    def test_restore_nonexistent_file_404(self, client):
        """POST /restore/{sid} 从不存在的快照文件恢复返回 404。"""
        # 先创建会话
        client.post("/chat", json={
            "user_input": "恢复测试",
            "session_id": "restore_test"
        })
        fake_path = "/nonexistent/path/missing_snapshot.pt"
        resp = client.post("/restore/restore_test", json={"path": fake_path})
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]

    def test_restore_nonexistent_session_404(self, client, tmp_path):
        """POST /restore/{sid} 不存在的会话返回 404。"""
        snap_path = str(tmp_path / "dummy.pt")
        resp = client.post("/restore/nonexistent_restore", json={
            "path": snap_path
        })
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]

    def test_snapshot_and_restore_roundtrip(self, client, tmp_path):
        """POST /snapshot + /restore 往返：保存后恢复成功。

        注意：turn_count 不在快照持久化范围内（save_state 仅保存
        attractor/purpose/memory/tokenizer 状态），因此恢复后 turn_count
        保持当前值。此处通过 precision_mean 验证 purpose 状态确实被恢复。
        """
        # 创建会话并运行多轮
        for i in range(5):
            client.post("/chat", json={
                "user_input": f"往返测试第{i + 1}轮",
                "session_id": "roundtrip_test"
            })

        # 记录保存前的 precision_mean
        resp_status = client.get("/status/roundtrip_test")
        precision_before = resp_status.json()["status"]["precision_mean"]

        # 保存快照
        snap_path = str(tmp_path / "roundtrip.pt")
        resp = client.post("/snapshot/roundtrip_test", json={"path": snap_path})
        assert resp.status_code == 200
        assert resp.json()["saved"] is True

        # 再运行几轮改变状态
        for i in range(3):
            client.post("/chat", json={
                "user_input": f"恢复前第{i + 1}轮",
                "session_id": "roundtrip_test"
            })

        # 恢复快照
        resp = client.post("/restore/roundtrip_test", json={"path": snap_path})
        assert resp.status_code == 200
        data = resp.json()
        assert data["restored"] is True
        assert "status" in data
        # 恢复后 precision_mean 应接近保存前的值（purpose 状态已恢复）
        precision_after_restore = data["status"]["precision_mean"]
        assert precision_after_restore == pytest.approx(
            precision_before, abs=1e-5
        ), (
            f"恢复后 precision_mean({precision_after_restore}) "
            f"应接近保存前({precision_before})"
        )


# ============================================================
# 6. 做梦端点
# ============================================================

class TestDreamEndpoints:
    """做梦端点测试。"""

    def test_dream_status(self, client):
        """GET /dream/status 返回调度器状态。"""
        resp = client.get("/dream/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "running" in data
        assert "is_dreaming" in data
        assert "idle_threshold" in data
        assert "dream_steps" in data
        assert "registered_sessions" in data
        assert "sessions" in data
        # 调度器应在运行（startup 事件启动）
        assert data["running"] is True
        # 当前不应在做梦
        assert data["is_dreaming"] is False

    def test_trigger_dream(self, client):
        """POST /dream/{sid} 手动触发做梦。"""
        # 先通过 /chat 创建会话并积累记忆
        for i in range(3):
            client.post("/chat", json={
                "user_input": f"做梦前第{i + 1}轮对话",
                "session_id": "dream_test"
            })

        resp = client.post("/dream/dream_test", json={
            "steps": 3,
            "full_cycle": False
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "dream_test"
        assert "result" in data
        # 做梦结果应包含 status 字段
        result = data["result"]
        assert isinstance(result, dict)
        assert "status" in result or "steps" in result

    def test_trigger_dream_new_session(self, client):
        """POST /dream/{sid} 对新会话触发做梦（自动创建会话）。"""
        resp = client.post("/dream/dream_new_session", json={
            "steps": 2,
            "full_cycle": False
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "dream_new_session"
        # 新会话无记忆，做梦应返回 no_memories_to_replay 或类似状态
        result = data["result"]
        assert isinstance(result, dict)

    def test_trigger_dream_no_body(self, client):
        """POST /dream/{sid} 不传 body 时使用默认参数。"""
        # 先创建会话
        client.post("/chat", json={
            "user_input": "默认参数做梦",
            "session_id": "dream_default"
        })
        resp = client.post("/dream/dream_default")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "dream_default"

    def test_dream_status_after_trigger(self, client):
        """GET /dream/status 在触发做梦后显示累计做梦次数。"""
        # 创建会话并积累记忆
        for i in range(2):
            client.post("/chat", json={
                "user_input": f"状态验证第{i + 1}轮",
                "session_id": "dream_status_test"
            })

        # 触发做梦
        resp = client.post("/dream/dream_status_test", json={
            "steps": 2,
            "full_cycle": False
        })
        assert resp.status_code == 200

        # 检查状态
        resp = client.get("/dream/status")
        assert resp.status_code == 200
        data = resp.json()
        # 找到对应会话的状态
        session_status = None
        for s in data["sessions"]:
            if s["session_id"] == "dream_status_test":
                session_status = s
                break
        assert session_status is not None, "dream_status_test 应在注册会话中"
        # 做梦次数应 >= 1
        assert session_status["dream_count"] >= 1

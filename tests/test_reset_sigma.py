# -*- coding: utf-8 -*-
"""C3 σ 重置端点测试（2026-08-23 dandan 批准新增）

覆盖三个新实现：
  1. 数据面 POST /reset-sigma/{session_id}（api/server.py）：
     - 200 成功：σ 归零、bias 重锚为 LMS_BIAS_SCALE、J 矩阵保留、
       turn_count 保留、落盘快照同步（σ=0 / bias=重锚值）
     - 404：会话不存在
     - 503：做梦互斥 acquire 失败
     - 503：save_session_state 返回 None（快照写锁超时 fail-open）
     - reanchor=False：纯重置，bias 保持原值
  2. 控制面 POST /control/reset-sigma（api/control.py）：
     - token 鉴权（_require_write）：缺 token/错 token → 401
     - 只读模式（token 未配置）→ 503
     - 转发数据面 + sid 兼容别名（req.sid 非空且 session_id=="main" 时用 sid）
     - 数据面 503 透传
  3. watchdog reset_sigma 的 token 兜底（scripts/lms_entropy_watchdog.py）：
     - env 缺失 LMS_CONTROL_TOKEN 时从 .env 文件读取（含引号剥离）
     - env 已配置时优先用 env，不读 .env
     - 两者皆缺 → 返回 False（跳过交人工）

测试设计原则（参照 tests/test_api_server.py）：
  - 轻量 config（num_nodes=32, input_dim=16, SimpleEmbedder），无 GPU/网络/模型
  - 注入自定义 SessionManager/DreamScheduler 单例到 api.server 模块
  - DreamScheduler idle_threshold/check_interval 极大值防自动做梦
  - tmp_path 隔离快照目录；控制面审计文件重定向到 tmp（不写脏仓库）
  - api.control 导入时会 _load_dotenv(仓库 .env)——夹具导入后立即撤销
    其注入的环境变量，保证测试隔离（不污染本文件/套件其余测试）
"""

import importlib.util
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


# ============================================================
# 辅助类与工厂（与 test_api_server.py 同款模式）
# ============================================================

class MockLLMBridge:
    """模拟 LLM 桥接器（数据面 /chat 可选注入，本文件未使用但保留一致性）。"""

    def __init__(self):
        self.query_count = 0
        self.last_user_input = None
        self.last_memory_context = None

    def query(self, user_input: str, memory_context: str) -> str:
        self.query_count += 1
        self.last_user_input = user_input
        self.last_memory_context = memory_context
        return f"[Mock回复] 收到: {user_input[:30]}"


def make_config_factory(snapshot_dir: str, with_bridge: bool = False):
    """轻量配置工厂（小网络 + SimpleEmbedder + 禁 LLM + 禁自动快照）。"""
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
        if with_bridge:
            config['llm_bridge'] = MockLLMBridge()
        return config
    return factory


def _inject_globals(sm: SessionManager, scheduler: DreamScheduler):
    orig_sm = server_module._session_manager
    orig_sched = server_module._dream_scheduler
    server_module._session_manager = sm
    server_module._dream_scheduler = scheduler
    return orig_sm, orig_sched


def _restore_globals(orig_sm, orig_sched):
    server_module._session_manager = orig_sm
    server_module._dream_scheduler = orig_sched


def _create_scheduler(sm: SessionManager) -> DreamScheduler:
    """idle_threshold/check_interval 极大值，防止后台线程自动做梦。"""
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
    """数据面 TestClient（轻量 SessionManager + 不自动做梦的调度器）。"""
    monkeypatch.setenv("LMS_EMBEDDER", "simple")
    monkeypatch.setenv("LMS_NUM_NODES", "32")
    monkeypatch.setenv("LMS_INPUT_DIM", "16")
    monkeypatch.setenv("LMS_DREAM_STATE_PATH",
                       str(tmp_path / "dream_state.json"))
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
def control_env(monkeypatch, tmp_path):
    """控制面隔离环境。

    返回 (TestClient, control_module)。
    api.control 模块级 _load_dotenv(仓库 .env) 会在首次导入时把仓库 .env
    的键灌进 os.environ——夹具导入后立即删除这些新增键，保证本文件
    其余测试（尤其数据面）与后续测试文件不被污染。随后强制 token/模式/
    审计文件为测试值，全程不读写仓库真实文件。
    """
    before = set(os.environ.keys())
    import api.control as control_module
    for k in set(os.environ.keys()) - before:
        del os.environ[k]

    monkeypatch.setattr(control_module, "CONTROL_TOKEN", "test-control-token")
    monkeypatch.setattr(control_module, "READ_ONLY_MODE", False)
    monkeypatch.setattr(control_module, "AUDIT_FILE",
                        tmp_path / "control-audit.jsonl")
    monkeypatch.setattr(control_module, "ACCESS_FILE",
                        tmp_path / "access.jsonl")
    with TestClient(control_module.app) as c:
        yield c, control_module


# ============================================================
# 1. 数据面 POST /reset-sigma/{session_id}
# ============================================================

class TestDataPlaneResetSigma:
    """数据面 σ 重置端点。"""

    @staticmethod
    def _create_session(client, sid: str, turns: int = 3) -> int:
        """通过 /chat 创建会话并跑 turns 轮，返回最终 turn_count。"""
        for i in range(turns):
            r = client.post("/chat", json={
                "user_input": f"第{i + 1}轮对话",
                "session_id": sid,
            })
            assert r.status_code == 200, r.text
        return turns

    def test_reset_sigma_200_reanchor_default(self, client, monkeypatch):
        """POST /reset-sigma/{sid} 成功：σ 归零、bias 重锚为 LMS_BIAS_SCALE、
        J 矩阵与 turn_count 保留，且落盘快照同步新状态。"""
        monkeypatch.setenv("LMS_BIAS_SCALE", "0.7")
        self._create_session(client, "rs200", turns=3)

        loop = server_module._session_manager.get("rs200")
        net = loop.attractor
        # 制造饱和 σ 与偏离重锚值的 bias，验证重置语义（确定性）
        net.sigma = torch.full((net.num_nodes,), 0.99)
        net.bias = torch.full((net.num_nodes,), 0.5)
        j_before = net.J.clone()

        resp = client.post("/reset-sigma/rs200")
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["reset"] is True
        assert data["session_id"] == "rs200"
        assert data["sigma_max_before"] == pytest.approx(0.99, abs=1e-6)
        assert data["bias_reanchored"] is True
        assert data["j_preserved"] is True
        assert data["turn_count"] == 3  # turn_count 保留，不因重置清零

        # 重置后：σ 全零
        assert float(net.sigma.abs().max().item()) == 0.0
        # bias 重锚为 LMS_BIAS_SCALE（0.7）
        assert torch.allclose(
            net.bias, torch.full((net.num_nodes,), 0.7))
        # J 矩阵不变（记忆保留）
        assert torch.equal(net.J, j_before)

        # 落盘快照同步（P1-1：重置结果写入磁盘，重启后不回到饱和态）
        snap_path = loop.latest_snapshot_path()
        assert os.path.isfile(snap_path)
        snap = torch.load(snap_path, map_location="cpu", weights_only=False)
        assert float(snap["attractor"]["sigma"].abs().max().item()) == 0.0
        assert torch.allclose(
            snap["attractor"]["bias"],
            torch.full((net.num_nodes,), 0.7))

    def test_reset_sigma_nonexistent_session_404(self, client):
        """不存在的会话 → 404（先于 acquire）。"""
        resp = client.post("/reset-sigma/ghost_session")
        assert resp.status_code == 404
        assert "不存在" in resp.json()["detail"]

    def test_reset_sigma_503_when_acquire_fails(self, client):
        """做梦互斥：acquire_conversation 返回 False → 503。"""
        self._create_session(client, "rs_busy", turns=1)
        scheduler = server_module._dream_scheduler

        original_acquire = scheduler.acquire_conversation
        scheduler.acquire_conversation = lambda sid, timeout=10.0: False
        try:
            resp = client.post("/reset-sigma/rs_busy")
            assert resp.status_code == 503
            assert "做梦" in resp.json()["detail"]
        finally:
            scheduler.acquire_conversation = original_acquire

    def test_reset_sigma_503_when_save_skipped(self, client):
        """save_session_state 返回 None（快照写锁超时 fail-open）→ 503。

        契约（P1-1）：内存 σ 已清零但磁盘未更新时不得谎报成功，
        显式 503（此路径曾因 except Exception 兜底被改写成 500，2026-08-23
        单测实证后修复：except HTTPException 原样透传）。
        """
        self._create_session(client, "rs_savefail", turns=1)
        loop = server_module._session_manager.get("rs_savefail")
        loop.attractor.sigma = torch.full((loop.attractor.num_nodes,), 0.9)

        original_save = loop.save_session_state
        loop.save_session_state = lambda *a, **k: None  # 模拟写锁超时
        try:
            resp = client.post("/reset-sigma/rs_savefail")
            assert resp.status_code == 503, resp.text
            detail = resp.json()["detail"]
            assert "写锁" in detail or "落盘" in detail
        finally:
            loop.save_session_state = original_save

    def test_reset_sigma_reanchor_false_keeps_bias(self, client):
        """reanchor=False 纯重置：σ 归零但 bias 保持原值（不重锚）。"""
        self._create_session(client, "rs_false", turns=2)
        loop = server_module._session_manager.get("rs_false")
        net = loop.attractor
        net.sigma = torch.full((net.num_nodes,), 0.9)
        net.bias = torch.full((net.num_nodes,), 0.5)

        resp = client.post("/reset-sigma/rs_false",
                           json={"reanchor": False})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["reset"] is True
        assert data["bias_reanchored"] is False
        assert data["sigma_max_before"] == pytest.approx(0.9, abs=1e-6)
        assert data["turn_count"] == 2

        # σ 归零，bias 未被重锚（仍为原值 0.5）
        assert float(net.sigma.abs().max().item()) == 0.0
        assert torch.allclose(
            net.bias, torch.full((net.num_nodes,), 0.5))


# ============================================================
# 2. 控制面 POST /control/reset-sigma（api/control.py）
# ============================================================

class TestControlPlaneResetSigma:
    """控制面 σ 重置端点（token 鉴权 + 转发数据面 + sid 别名）。"""

    @staticmethod
    def _post(client, body=None, token="test-control-token"):
        headers = {"X-Control-Token": token} if token is not None else {}
        return client.post("/control/reset-sigma", json=body, headers=headers)

    def _install_fake_data_plane(self, control_module, calls, rc=200, body=None):
        """把数据面转发替换为记录型假实现，返回假响应体。"""
        if body is None:
            body = {"reset": True, "sigma_max_before": 0.99,
                    "bias_reanchored": True, "turn_count": 3,
                    "j_preserved": True}

        async def fake_ahttp(method, path, payload=None, timeout=5.0):
            calls.append((method, path, payload))
            return rc, body

        control_module._ahttp_json = fake_ahttp
        return body

    def test_control_reset_sigma_forwards_ok(self, control_env):
        """带 token 调用成功：转发数据面 /reset-sigma/main，合并返回。"""
        client, control_module = control_env
        calls = []
        self._install_fake_data_plane(control_module, calls)

        resp = self._post(client, {"session_id": "main"})
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["status"] == "ok"
        assert data["control"] is True
        assert data["reset"] is True
        assert data["bias_reanchored"] is True
        # 转发到数据面，默认 reanchor=True
        assert calls == [("POST", "/reset-sigma/main", {"reanchor": True})]

    def test_control_reset_sigma_sid_alias(self, control_env):
        """sid 兼容别名：req.sid 非空且 session_id=="main" 时用 sid。"""
        client, control_module = control_env
        calls = []
        self._install_fake_data_plane(control_module, calls)

        resp = self._post(client, {"sid": "legacy-sess"})
        assert resp.status_code == 200, resp.text
        assert calls[0][1] == "/reset-sigma/legacy-sess"
        assert calls[0][2] == {"reanchor": True}

    def test_control_reset_sigma_session_id_overrides_sid(self, control_env):
        """session_id 显式指定时优先于 sid 别名。"""
        client, control_module = control_env
        calls = []
        self._install_fake_data_plane(control_module, calls)

        resp = self._post(client, {"sid": "alias-sess",
                                   "session_id": "real-sess"})
        assert resp.status_code == 200, resp.text
        assert calls[0][1] == "/reset-sigma/real-sess"

    def test_control_reset_sigma_reanchor_passthrough(self, control_env):
        """reanchor=False 原样透传给数据面。"""
        client, control_module = control_env
        calls = []
        self._install_fake_data_plane(control_module, calls)

        resp = self._post(client, {"session_id": "main", "reanchor": False})
        assert resp.status_code == 200, resp.text
        assert calls[0][2] == {"reanchor": False}

    def test_control_reset_sigma_missing_token_401(self, control_env):
        """缺 token → 401（_require_write 拒绝）。"""
        client, _ = control_env
        resp = self._post(client, {"session_id": "main"}, token=None)
        assert resp.status_code == 401
        assert "令牌" in resp.json()["detail"]

    def test_control_reset_sigma_wrong_token_401(self, control_env):
        """错误 token → 401。"""
        client, _ = control_env
        resp = self._post(client, {"session_id": "main"}, token="wrong-token")
        assert resp.status_code == 401
        assert "令牌" in resp.json()["detail"]

    def test_control_reset_sigma_read_only_503(self, control_env):
        """只读模式（token 未配置）→ 503 写端点禁用。"""
        client, control_module = control_env
        control_module.READ_ONLY_MODE = True
        resp = self._post(client, {"session_id": "main"})
        assert resp.status_code == 503
        assert "只读" in resp.json()["detail"]

    def test_control_reset_sigma_data_plane_503_passthrough(self, control_env):
        """数据面 503（做梦互斥）→ 控制面原样 503。"""
        client, control_module = control_env
        calls = []
        self._install_fake_data_plane(
            control_module, calls, rc=503,
            body={"detail": "系统正在做梦（记忆巩固中），请稍后重试。"})
        resp = self._post(client, {"session_id": "main"})
        assert resp.status_code == 503
        assert "数据面" in resp.json()["detail"]


# ============================================================
# 3. watchdog reset_sigma token 兜底（scripts/lms_entropy_watchdog.py）
# ============================================================

class TestWatchdogTokenFallback:
    """watchdog reset_sigma 的 LMS_CONTROL_TOKEN 兜底读取。"""

    @staticmethod
    def _load_watchdog():
        """以独立模块名加载 scripts/lms_entropy_watchdog.py（scripts 非包）。"""
        path = os.path.join(_PROJECT_ROOT, "scripts",
                            "lms_entropy_watchdog.py")
        spec = importlib.util.spec_from_file_location(
            "lms_entropy_watchdog_under_test", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    @staticmethod
    def _redirect_env_file(monkeypatch, wd, tmp_path):
        """把 watchdog 的 __file__ 指到 tmp/scripts/ 下，使 "../.env"
        解析为 tmp/.env（隔离测试，绝不读仓库真实 .env）。"""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        monkeypatch.setattr(
            wd, "__file__", str(scripts_dir / "lms_entropy_watchdog.py"))
        monkeypatch.setattr(wd, "LOG_FILE", str(tmp_path / "watchdog.log"))
        return tmp_path / ".env"

    @staticmethod
    def _fake_urlopen(monkeypatch, wd, captured):
        """捕获请求头/URL 并返回 reset 成功的假响应。"""
        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{"reset": true}'

        def fake_urlopen(req, timeout=None):
            # urllib.request 会把头名规范化（如 X-Control-Token → X-control-token），
            # 且本版本 get_header/has_header 是精确匹配——手工大小写不敏感取回，
            # 与真实服务端（hmac 校验头值）解析语义一致
            captured["token"] = next(
                (v for k, v in req.headers.items()
                 if k.lower() == "x-control-token"), None)
            captured["url"] = req.full_url
            return FakeResp()

        monkeypatch.setattr(wd.urllib.request, "urlopen", fake_urlopen)

    def test_token_fallback_from_env_file(self, monkeypatch, tmp_path):
        """env 缺失 LMS_CONTROL_TOKEN 时，从 .env 文件读取（含引号剥离）。"""
        wd = self._load_watchdog()
        monkeypatch.delenv("LMS_CONTROL_TOKEN", raising=False)
        env_file = self._redirect_env_file(monkeypatch, wd, tmp_path)
        env_file.write_text('LMS_CONTROL_TOKEN="fallback-token-123"\n',
                            encoding="utf-8")

        captured = {}
        self._fake_urlopen(monkeypatch, wd, captured)

        ok = wd.reset_sigma({"streak": 2}, "σmax 0.99 / 熵 0.9999")
        assert ok is True
        # 兜底读到的 token 被用作控制面鉴权头
        assert captured["token"] == "fallback-token-123"
        assert captured["url"] == "http://127.0.0.1:8191/control/reset-sigma"

    def test_env_token_wins_over_env_file(self, monkeypatch, tmp_path):
        """env 已配置时直接使用，不读 .env（env 优先）。"""
        wd = self._load_watchdog()
        monkeypatch.setenv("LMS_CONTROL_TOKEN", "env-token-456")
        env_file = self._redirect_env_file(monkeypatch, wd, tmp_path)
        env_file.write_text('LMS_CONTROL_TOKEN="fallback-token-123"\n',
                            encoding="utf-8")

        captured = {}
        self._fake_urlopen(monkeypatch, wd, captured)

        ok = wd.reset_sigma({"streak": 2}, "test-reason")
        assert ok is True
        assert captured["token"] == "env-token-456"

    def test_no_token_anywhere_returns_false(self, monkeypatch, tmp_path):
        """env 与 .env 均无 token → 返回 False（跳过交人工，不发请求）。"""
        wd = self._load_watchdog()
        monkeypatch.delenv("LMS_CONTROL_TOKEN", raising=False)
        env_file = self._redirect_env_file(monkeypatch, wd, tmp_path)
        env_file.write_text("OTHER_KEY=1\n", encoding="utf-8")

        called = []

        def _explode(*a, **k):
            called.append(True)
            raise AssertionError("不应发起 HTTP 请求")

        monkeypatch.setattr(wd.urllib.request, "urlopen", _explode)

        ok = wd.reset_sigma({"streak": 2}, "test-reason")
        assert ok is False
        assert called == []

    def test_missing_env_file_returns_false(self, monkeypatch, tmp_path):
        """env 缺失且 .env 文件不存在 → 返回 False（fail-open 交人工）。"""
        wd = self._load_watchdog()
        monkeypatch.delenv("LMS_CONTROL_TOKEN", raising=False)
        # tmp/scripts/ 下没有 .env
        self._redirect_env_file(monkeypatch, wd, tmp_path)

        ok = wd.reset_sigma({"streak": 2}, "test-reason")
        assert ok is False

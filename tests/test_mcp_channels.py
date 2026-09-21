"""tests/test_mcp_channels.py — MCP 写通道 v2 契约修复（2026-09-21 修复单）。

覆盖两个"坏掉的 MCP 写通道"的修法与调用方预算：
  · lms_http_mcp.lms_store  → 端点 /chat(404) 修到 /store，解析 v2 平铺字段
  · mcp_memory_server.do_store_memory → /feed 超时放宽 + 快照改后台（不阻塞）
另覆盖同文件同族 v2 漂移：lms_recall/lms_status/lms_dream 的字段名/值域对齐。

全部用假 requests / 假 _http_post 注入，零网络、零生产写入。
"""

import json
import time

import pytest


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP %d" % self.status_code)


class _FakeRequests:
    """按 URL 子串匹配的假 requests；记录所有调用（方法/URL/超时）。"""

    def __init__(self, post_routes=None, get_routes=None):
        self.post_routes = post_routes or {}
        self.get_routes = get_routes or {}
        self.calls = []

    def _pick(self, routes, url):
        for key, resp in routes.items():
            if key in url:
                return resp
        raise AssertionError("unexpected request: " + url)

    def post(self, url, json=None, timeout=None):
        self.calls.append(("POST", url, json, timeout))
        return self._pick(self.post_routes, url)

    def get(self, url, timeout=None):
        self.calls.append(("GET", url, None, timeout))
        return self._pick(self.get_routes, url)


# ---------------------------------------------------------------------------
# lms_http_mcp：lms_store 修端点 + 解析
# ---------------------------------------------------------------------------

def test_lms_store_uses_v2_store_endpoint(monkeypatch):
    import lms_http_mcp as m

    fake = _FakeRequests(post_routes={
        "/store": _Resp({"turn": 42, "stored": True, "dedup_hit": False,
                         "surprise": 1.5}),
    })
    monkeypatch.setattr(m, "requests", fake)

    r = m.lms_store("用户输入", "助手回复", "main")
    assert r["success"] is True
    assert r["turn_count"] == 42, "应读 v2 平铺 turn（原读 memory_state.turn_count ⇒ 恒 0）"
    assert r["dedup_hit"] is False
    method, url, payload, timeout = fake.calls[0]
    assert url.endswith("/store"), "写通道必须打 v2 真实写口 /store"
    assert "/chat" not in url, "旧 /chat 在 v2 不存在（404）"
    assert payload == {"session_id": "main", "user_input": "用户输入",
                       "llm_output": "助手回复"}
    assert timeout == 60, "/store 实测 11-17s ⇒ 预算须 >=60s"


def test_lms_recall_flat_status_and_turn(monkeypatch):
    import lms_http_mcp as m

    fake = _FakeRequests(
        post_routes={"/recall": _Resp({"results": [{"text": "命中", "score": 0.9}],
                                       "turn": 7})},
        get_routes={"/status/main": _Resp({"session_id": "main", "turn": 9,
                                           "capacity": {"entries": 3}})},
    )
    monkeypatch.setattr(m, "requests", fake)

    r = m.lms_recall("查询", "main")
    assert r["success"] is True
    assert r["turn_count"] == 9, "应读 v2 平铺 /status.turn"
    assert r["memory_state"]["turn"] == 9, "/status 平铺体应原样进 memory_state"
    assert "命中" in r["memory_context"]


def test_lms_status_v2_flat_fields(monkeypatch):
    import lms_http_mcp as m

    fake = _FakeRequests(get_routes={"/status/main": _Resp({
        "session_id": "main", "turn": 100, "j_target": 8.5, "surprise": 76.0,
        "entropy": 7.3, "capacity": {"entries": 5725, "num_nodes": 1536}})})
    monkeypatch.setattr(m, "requests", fake)

    r = m.lms_status("main")
    assert r["success"] is True
    assert r["turn_count"] == 100, "不再恒 0"
    assert r["num_nodes"] == 1536, "不再恒兜底 256"
    assert r["entries"] == 5725
    assert r["j_target"] == 8.5 and r["surprise"] == 76.0


@pytest.mark.parametrize("status,expected", [
    ("dreamed", True), ("skipped", False), ("failed", False), (None, False)])
def test_lms_dream_reads_real_status(monkeypatch, status, expected):
    import lms_http_mcp as m

    result = {"status": status} if status is not None else {}
    fake = _FakeRequests(post_routes={
        "/dream/main": _Resp({"session_id": "main", "result": result})})
    monkeypatch.setattr(m, "requests", fake)

    r = m.lms_dream("main")
    assert r["success"] is True
    assert r["dream_completed"] is expected, "不再'只要 200 就报完成'"
    assert r["dream_status"] == status


def test_lms_snapshot_timeout_widened(monkeypatch):
    import lms_http_mcp as m

    fake = _FakeRequests(post_routes={"/snapshot/main": _Resp({"path": "/x"})})
    monkeypatch.setattr(m, "requests", fake)
    m.lms_snapshot("main")
    assert fake.calls[0][3] == 180, "快照实测 35-70s ⇒ 原 10s 必然假失败"


# ---------------------------------------------------------------------------
# mcp_memory_server：do_store_memory（超时/快照解耦）与 do_recall_memory
# ---------------------------------------------------------------------------

def test_store_memory_flat_status_and_no_sync_snapshot(monkeypatch):
    import mcp_memory_server as srv

    calls = []

    def fake_post(path, payload, timeout):
        calls.append(("POST", path, timeout))
        assert path == "/feed", "写通道主路径应只 POST /feed（快照异步）"
        return {"turn_count": 5}

    def fake_get(path, timeout):
        calls.append(("GET", path, timeout))
        return {"turn": 5, "entropy": 7.0, "surprise": 3.0,
                "capacity": {"entries": 9, "num_nodes": 64}}

    monkeypatch.setattr(srv, "_http_post", fake_post)
    monkeypatch.setattr(srv, "_http_get", fake_get)
    monkeypatch.setattr(srv, "_schedule_snapshot", lambda: True)

    out = json.loads(srv.do_store_memory("用户: hi\n助手: yo"))
    assert out["status"] == "已存储"
    assert out["turn_count"] == 5
    assert out["episodic_buffer_size"] == 9, "v2 条目数在 capacity.entries"
    assert out["entropy"] == 7.0 and out["surprise"] == 3.0
    assert out["snapshot_scheduled"] is True
    assert all(c[1] != "/snapshot/main" for c in calls), \
        "快照不得再在关键路径同步等待（大会话 35-70s ⇒ 必超时）"


def test_schedule_snapshot_returns_promptly(monkeypatch):
    import mcp_memory_server as srv

    monkeypatch.setattr(srv, "_snapshot_inflight", False)

    def slow_post(path, payload, timeout):
        time.sleep(0.5)
        return {"saved": True, "path": "/tmp/x"}

    monkeypatch.setattr(srv, "_http_post", slow_post)
    t0 = time.time()
    ok = srv._schedule_snapshot()
    dt = time.time() - t0
    assert ok is True
    assert dt < 0.2, "后台触发必须立即返回（不阻塞工具调用）"
    time.sleep(0.7)
    assert srv._snapshot_inflight is False, "后台结束后应复位在跑标志"


def test_schedule_snapshot_skips_when_inflight(monkeypatch):
    import mcp_memory_server as srv

    monkeypatch.setattr(srv, "_snapshot_inflight", True)
    assert srv._schedule_snapshot() is False, "已在跑应跳过（防堆积）"


def test_recall_memory_flat_status(monkeypatch):
    import mcp_memory_server as srv

    monkeypatch.setattr(srv, "_http_post", lambda p, pl, t: {
        "results": [{"text": "x", "score": 0.9}], "turn": 3})
    monkeypatch.setattr(srv, "_http_get", lambda p, t: {
        "turn": 3, "capacity": {"entries": 7}})

    out = srv.do_recall_memory("查询")
    assert "检索到 1 条相关记忆" in out
    assert "情景记忆缓冲区大小: 7" in out

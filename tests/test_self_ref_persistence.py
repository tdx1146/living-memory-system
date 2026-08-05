"""单元测试：自指回路自述跨重启持久化（Phase 3.4 反思回流收尾）

覆盖范围：
  1. observe() 蒸馏出的自述追加到持久化 JSONL（只写不读回 → 无循环）
  2. 重启恢复：新实例 backfill_voice_history() 从持久层回填内存历史
  3. 内存优先：回填与内存历史按文本去重，不重复追加
  4. 防循环：回填只恢复文本历史，绝不触碰嵌入/编码器（不回注）
  5. fail-open：持久化目录不可写 / 文件损坏不崩溃
  6. 路径解析：cfg 显式 > 环境变量 LMS_SELF_VOICE_PATH/DIR > 仓库默认
  7. API 层：/self-ref/voice 重启后（会话惰性重建 + 内存空）仍能返回自述

设计依据：docs/SELF_REF_INTEGRATED_DESIGN.md（反思回流收尾）。
"""

import json
import os

import pytest

try:
    from core.hippocampus.self_referential import (
        SelfVoiceDistiller,
        SelfReferentialLoop,
    )
    _SELF_REF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SELF_REF_AVAILABLE = False
    SelfReferentialLoop = None  # type: ignore[assignment]

from tests.test_self_referential import (  # noqa: E402
    FakeEncoder,
    FakeEmbedder,
    FakeTokenizer,
    make_activation,
    make_memory_context,
)


def _require_self_ref():
    if not _SELF_REF_AVAILABLE:
        pytest.skip("core.hippocampus.self_referential 尚未就绪")


def _make_loop(persist_path, session_id="main", history_cap=20, **cfg_extra):
    """构造带显式持久化路径的 SelfReferentialLoop。"""
    _require_self_ref()
    enc = FakeEncoder(dim=16)
    tok = FakeTokenizer()
    emb = FakeEmbedder(dim=16)
    config = {
        "self_ref_alpha_base": 0.15,
        "self_ref_history_cap": history_cap,
        "session_id": session_id,
        "self_ref_voice_persist_path": persist_path,
    }
    config.update(cfg_extra)
    loop = SelfReferentialLoop(enc, tok, emb, config=config)
    return loop, enc


def _voice_text():
    """与 distiller 实际产物一致的预期自述（解读 + 激活节点片段）。"""
    return SelfVoiceDistiller.distill(_ctx())


def _ctx():
    return make_memory_context(
        interpretations=["我同时唤醒多条记忆，但关注方向稳定。"],
        detail="熵:1.2, 惊讶度:0.3 | 激活节点: 节点0(强:0.8)",
    )


def _read_persisted(path):
    """读取持久化 JSONL，返回 text 列表（按文件顺序）。"""
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict) and rec.get("text"):
                out.append(rec["text"])
    return out


# ============================================================
# 1. observe 持久化
# ============================================================

class TestObservePersist:
    def test_observe_persists_voice(self, tmp_path):
        """observe 后自述应追加到持久化文件（含文本与会话标记）。"""
        path = str(tmp_path / "self_voice_main.jsonl")
        loop, _ = _make_loop(path)
        loop.observe(_ctx(), make_activation())

        assert os.path.exists(path)
        lines = _read_persisted(path)
        assert lines == [_voice_text()]

        # 记录结构：含 ts / session_id / text 三字段
        with open(path, "r", encoding="utf-8") as f:
            rec = json.loads(f.readline())
        assert rec["text"] == _voice_text()
        assert rec["session_id"] == "main"
        assert isinstance(rec["ts"], (int, float))

    def test_observe_appends_multiple(self, tmp_path):
        """多轮 observe 逐条追加（含重复文本也保留——忠实还原轮次）。"""
        path = str(tmp_path / "v.jsonl")
        loop, _ = _make_loop(path)
        loop.observe(_ctx(), make_activation())
        loop.observe(_ctx(), make_activation())
        ctx2 = make_memory_context(
            interpretations=["另一条自述。"],
            detail="熵:0.9, 惊讶度:0.2 | 激活节点: 节点1(中:0.5)",
        )
        loop.observe(ctx2, make_activation(seed=7))

        expected = [_voice_text(), _voice_text(),
                    SelfVoiceDistiller.distill(ctx2)]
        assert _read_persisted(path) == expected

    def test_empty_voice_not_persisted(self, tmp_path):
        """空自述（vector 模式/无解读）不落盘。"""
        path = str(tmp_path / "v.jsonl")
        loop, _ = _make_loop(path)
        loop.observe("无法识别的格式", make_activation())
        assert not os.path.exists(path)

    def test_persist_fail_open(self, tmp_path):
        """持久化目录不可写 → observe 不崩溃（静默降级）。"""
        blocked = tmp_path / "no_write" / "v.jsonl"
        path = str(blocked)
        loop, _ = _make_loop(path)
        # 强制不可写：父目录不存在且创建失败场景用只读目录模拟
        ro_dir = tmp_path / "ro"
        ro_dir.mkdir()
        os.chmod(ro_dir, 0o500)
        try:
            loop2, _ = _make_loop(str(ro_dir / "v.jsonl"))
            loop2.observe(_ctx(), make_activation())  # 不应抛异常
            assert loop2.self_voice_history == [_voice_text()]  # 内存照常
        finally:
            os.chmod(ro_dir, 0o700)

    def test_persist_disabled_when_false(self, tmp_path):
        """cfg 显式 self_ref_voice_persist_path=False → 禁用持久化。"""
        loop, _ = _make_loop("ignored.jsonl")
        loop.voice_persist_path = None  # 显式禁用等价路径
        loop.observe(_ctx(), make_activation())
        # 不写任何文件（路径为 None 直接跳过）
        assert not os.path.exists("ignored.jsonl")


# ============================================================
# 2. 重启恢复 / 回填
# ============================================================

class TestBackfill:
    def test_restart_backfill_restores_history(self, tmp_path):
        """模拟重启：新实例（内存空）+ 同一持久化文件 → 回填全部历史。"""
        path = str(tmp_path / "v.jsonl")
        loop1, _ = _make_loop(path)
        loop1.observe(_ctx(), make_activation())
        loop1.observe(_ctx(), make_activation())
        ctx2 = make_memory_context(
            interpretations=["另一条自述。"], detail="熵:0.9, 惊讶度:0.2")
        loop1.observe(ctx2, make_activation(seed=7))

        # 重启：全新实例
        loop2, _ = _make_loop(path)
        assert loop2.self_voice_history == []
        added = loop2.backfill_voice_history()

        assert added == 3
        assert loop2.self_voice_history == [
            _voice_text(), _voice_text(), "另一条自述。"]

    def test_backfill_memory_first_dedup(self, tmp_path):
        """内存优先：内存已有条目不回填重复；只合并缺失条目。"""
        path = str(tmp_path / "v.jsonl")
        loop1, _ = _make_loop(path)
        loop1.observe(_ctx(), make_activation())
        ctx2 = make_memory_context(
            interpretations=["另一条自述。"], detail="熵:0.9, 惊讶度:0.2")
        loop1.observe(ctx2, make_activation(seed=7))

        # 新实例先有 1 条新自述（模拟重启后先对话了一轮），再回填
        loop2, _ = _make_loop(path)
        ctx3 = make_memory_context(
            interpretations=["重启后的新自述。"], detail="熵:1.0, 惊讶度:0.4")
        loop2.observe(ctx3, make_activation(seed=9))
        assert loop2.self_voice_history == ["重启后的新自述。"]

        added = loop2.backfill_voice_history()
        # 文件两条（含内存中缺失的两条）全部合并，不重复内存已有条目
        assert added == 2
        assert _voice_text() in loop2.self_voice_history
        assert "另一条自述。" in loop2.self_voice_history
        assert loop2.self_voice_history.count("重启后的新自述。") == 1

    def test_backfill_respects_capacity(self, tmp_path):
        """回填后裁剪到 history_capacity（保留最近条目）。"""
        path = str(tmp_path / "v.jsonl")
        loop1, _ = _make_loop(path, history_cap=10)
        for i in range(15):
            loop1.observe(make_memory_context(
                interpretations=[f"自述第{i}条。"],
                detail="熵:1.0, 惊讶度:0.1"), make_activation(seed=i))

        loop2, _ = _make_loop(path, history_cap=10)
        added = loop2.backfill_voice_history()
        assert added == 10  # 只回填容量内的最近 10 条
        assert len(loop2.self_voice_history) == 10
        assert loop2.self_voice_history[-1] == "自述第14条。"

    def test_backfill_no_file_or_corrupt(self, tmp_path):
        """文件不存在 / 含损坏行 → 不崩溃，损坏行跳过。"""
        path = str(tmp_path / "v.jsonl")
        loop, _ = _make_loop(path)
        assert loop.backfill_voice_history() == 0  # 文件不存在

        with open(path, "w", encoding="utf-8") as f:
            f.write("{broken json\n")
            f.write(json.dumps({"ts": 1.0, "session_id": "main",
                                "text": "有效自述。"}) + "\n")
            f.write("not json at all\n")
        added = loop.backfill_voice_history()
        assert added == 1
        assert loop.self_voice_history == ["有效自述。"]


# ============================================================
# 3. 防循环：回填不回注
# ============================================================

class TestAntiLoop:
    def test_backfill_never_reinjects_echo(self, tmp_path):
        """回填只恢复文本历史：编码器零调用、sensory_self_prev 仍为 None。"""
        path = str(tmp_path / "v.jsonl")
        loop1, _ = _make_loop(path)
        loop1.observe(_ctx(), make_activation())

        loop2, enc2 = _make_loop(path)
        assert enc2.encoded == []
        loop2.backfill_voice_history()
        # 回填后：文本历史已恢复，但嵌入/回注源完全未被触碰
        assert loop2.self_voice_history == [_voice_text()]
        assert enc2.encoded == []
        assert loop2.sensory_self_prev is None

    def test_persist_is_write_only(self, tmp_path):
        """持久化本身不触发任何编码/回注（observe 外的副作用为零）。"""
        path = str(tmp_path / "v.jsonl")
        loop1, enc1 = _make_loop(path)
        loop1.observe(_ctx(), make_activation())
        # observe 内的 encode 只有 1 次（蒸馏文本编码，供回声计算），
        # 持久化不增加任何编码调用
        assert len(enc1.encoded) == 1
        # 且文件只增不改（回读不会写入新内容）
        before = _read_persisted(path)
        loop1.persisted_voice_count()
        loop1.backfill_voice_history()
        assert _read_persisted(path) == before


# ============================================================
# 4. 路径解析
# ============================================================

class TestPathResolution:
    def test_cfg_path_wins(self, tmp_path, monkeypatch):
        """cfg 显式路径 > 环境变量。"""
        monkeypatch.setenv("LMS_SELF_VOICE_PATH", "/env/file.jsonl")
        loop, _ = _make_loop(str(tmp_path / "cfg.jsonl"))
        assert loop.voice_persist_path == str(tmp_path / "cfg.jsonl")

    def test_env_file_path(self, tmp_path, monkeypatch):
        """无 cfg 路径时，LMS_SELF_VOICE_PATH 生效。"""
        monkeypatch.delenv("LMS_SELF_VOICE_DIR", raising=False)
        monkeypatch.setenv("LMS_SELF_VOICE_PATH", str(tmp_path / "env.jsonl"))
        loop, _ = _make_loop(None)
        assert loop.voice_persist_path == str(tmp_path / "env.jsonl")

    def test_env_dir_with_session(self, tmp_path, monkeypatch):
        """LMS_SELF_VOICE_DIR + session_id → per-session 文件。"""
        monkeypatch.delenv("LMS_SELF_VOICE_PATH", raising=False)
        monkeypatch.setenv("LMS_SELF_VOICE_DIR", str(tmp_path))
        loop, _ = _make_loop(None, session_id="alice")
        assert loop.voice_persist_path == str(
            tmp_path / "self_voice_alice.jsonl")

    def test_default_path_in_repo_data(self, monkeypatch):
        """无任何配置 → 仓库默认 data/self_voice/self_voice_main.jsonl。"""
        import inspect
        monkeypatch.delenv("LMS_SELF_VOICE_PATH", raising=False)
        monkeypatch.delenv("LMS_SELF_VOICE_DIR", raising=False)
        _require_self_ref()
        enc = FakeEncoder(dim=16)
        loop = SelfReferentialLoop(enc, FakeTokenizer(), FakeEmbedder(dim=16),
                                   config={"session_id": "main"})
        src_file = inspect.getfile(SelfReferentialLoop)
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(src_file))))
        assert loop.voice_persist_path == os.path.join(
            repo_root, "data", "self_voice", "self_voice_main.jsonl")


# ============================================================
# 5. API 层：/self-ref/voice 重启后回填
# ============================================================

@pytest.mark.skipif(not _SELF_REF_AVAILABLE, reason="self_ref 未就绪")
class TestVoiceEndpointRestart:
    def test_voice_endpoint_backfills_after_restart(self, tmp_path):
        """模拟重启：会话重建 + 内存空 → /self-ref/voice 从持久层回填。"""
        from fastapi.testclient import TestClient

        import api.server as server_module
        from api.session_manager import SessionManager
        from core.sensory.embedder import SimpleEmbedder
        from runtime.config import default_config
        from runtime.dream_scheduler import DreamScheduler

        voice_file = tmp_path / "self_voice_main.jsonl"
        # 预置"重启前"的持久化自述（由旧实例 observe 产生）
        with open(voice_file, "w", encoding="utf-8") as f:
            for text in ("重启前的自述一。", "重启前的自述二。"):
                f.write(json.dumps(
                    {"ts": 1000.0, "session_id": "main", "text": text})
                    + "\n")

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
            config['snapshot_dir'] = str(tmp_path / "snapshots")
            config['self_ref_enabled'] = True
            config['self_ref_voice_persist_path'] = str(voice_file)
            return config

        sm = SessionManager(default_config_factory=factory)
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: sm.get(sid),
            idle_threshold=999999, dream_steps=5,
            dream_full_cycle=False, check_interval=999999,
        )
        orig_sm = server_module._session_manager
        orig_sched = server_module._dream_scheduler
        server_module._session_manager = sm
        server_module._dream_scheduler = scheduler
        try:
            with TestClient(server_module.app) as client:
                # 重启后首次读取：会话尚不存在（惰性创建），内存为空
                r = client.get("/self-ref/voice?session_id=main&limit=5")
                assert r.status_code == 200
                data = r.json()
                assert data["enabled"] is True
                assert data["count"] >= 2
                # 最近在前：最后写入的“自述二”在最前
                assert data["voices"][0] == "重启前的自述二。"
                assert set(data["voices"]) == {
                    "重启前的自述一。", "重启前的自述二。"}
                assert data["persisted_count"] == 2
        finally:
            server_module._session_manager = orig_sm
            server_module._dream_scheduler = orig_sched

    def test_voice_endpoint_memory_first(self, tmp_path):
        """内存优先：内存已有历史时，回填不重复、不覆盖。"""
        from fastapi.testclient import TestClient

        import api.server as server_module
        from api.session_manager import SessionManager
        from core.sensory.embedder import SimpleEmbedder
        from runtime.config import default_config
        from runtime.dream_scheduler import DreamScheduler

        voice_file = tmp_path / "self_voice_main.jsonl"
        with open(voice_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(
                {"ts": 1000.0, "session_id": "main", "text": "旧自述。"})
                + "\n")

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
            config['snapshot_dir'] = str(tmp_path / "snapshots")
            config['self_ref_enabled'] = True
            config['self_ref_voice_persist_path'] = str(voice_file)
            return config

        sm = SessionManager(default_config_factory=factory)
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: sm.get(sid),
            idle_threshold=999999, dream_steps=5,
            dream_full_cycle=False, check_interval=999999,
        )
        orig_sm = server_module._session_manager
        orig_sched = server_module._dream_scheduler
        server_module._session_manager = sm
        server_module._dream_scheduler = scheduler
        try:
            with TestClient(server_module.app) as client:
                # 先让会话产生一条新自述（内存优先源）
                from core.hippocampus.self_referential import (
                    SelfVoiceDistiller)
                loop = sm.get_or_create("main")
                loop.self_ref.self_voice_history.append("内存中的新自述。")
                r = client.get("/self-ref/voice?session_id=main&limit=5")
                data = r.json()
                # 内存条目在（且不重复），旧自述被合并进来（内容优先，
                # 合并追加不改变内存条目）
                assert "内存中的新自述。" in data["voices"]
                assert "旧自述。" in data["voices"]
                assert data["voices"].count("旧自述。") == 1
        finally:
            server_module._session_manager = orig_sm
            server_module._dream_scheduler = orig_sched

# -*- coding: utf-8 -*-
"""E3 闭环测试（自我怀疑驱动的主动调节，dandan 拍板 2026-08-20 22:14）。

规格依据：`方案-E3自我怀疑驱动调节-20260820.md` §3.2/§3.3/§3.4（A1-A7） +
`开工令-E3实施-给四妹-20260820.md`（拍板 1-5：证伪收编 conflict、无 LLM
自动生成、surprise 判据留观测不设阈值、节流放宽 satiety 12h/单次上限 2）。

覆盖（按开工令验收准备）：
  1. 选择器排序（纯函数：score = α·epistemic + β·progress + γ·reachable；
     excluded / superseded 过滤）
  2. 重激活落库（e3_reactivate 路径 A → mark_labile + 候选进队列）
  3. min-age 闸门（E3 开：同轮入队不消化、下一轮消化；E3 关：同轮即消保持）
  4. satiety 待机（冷却 / 累计闭环达上限待机 / rearm 恢复 / judge_resolved）
  5. 开关关零参与（LMS_E3_ENABLED=0 → e3_reactivate {enabled:false}、
     /e3/review 端点零参与）
  6. doubt_ingest 协议扩展（'证伪'/'reactivate' 别名 → conflict；conflict
     命中补入队——根因 1/3 修复）
  7. gap_registry 消解（mark_resolved / fok_resolved / snapshot / mark_review 联动）

运行方式：pytest rewrite-ws/tests/test_e3_reactivation.py -v。
"""

import os
import sys

# 确保项目根目录在 Python 路径中（可从任意 cwd 运行）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from types import SimpleNamespace

import pytest
import torch

from runtime.config import default_config
from runtime.loop import LivingMemoryLoop, _e3_min_age_turns


# --------------------------------------------------------------------------- #
#  测试小工具（与 test_loop_reconsolidation_labile 同款风格）
# --------------------------------------------------------------------------- #

class FakeEmbedder:
    """带 embed_text / embed_text_raw 的确定性嵌入器（16 维）。"""

    def __init__(self, dim: int = 16):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, tokens: list) -> torch.Tensor:
        return torch.zeros(self._dim)

    def embed_text(self, text: str) -> torch.Tensor:
        vec = torch.zeros(self._dim)
        for i, ch in enumerate(text):
            vec[i % self._dim] += ord(ch) / 1000.0
        return vec

    embed_text_raw = embed_text


def make_loop(tmp_path, **overrides) -> LivingMemoryLoop:
    """小规模测试 loop：隔离快照目录 + 隔离再巩固候选队列路径。"""
    torch.manual_seed(42)
    cfg = default_config()
    cfg.update({
        'num_nodes': 32,
        'input_dim': 16,
        'num_infer_steps': 5,
        'seed': 42,
        'embedder': FakeEmbedder(dim=16),
        'auto_snapshot': False,
        'snapshot_dir': str(tmp_path / 'snaps'),
        'archive_enabled': False,
        'reconsolidation_queue_path': str(tmp_path / 'recon' / 'candidates.json'),
    })
    cfg.update(overrides)
    return LivingMemoryLoop(cfg)


def _entry(text: str, confidence: float = 0.7, recall_count: int = 0,
           doubt_state: str = "stable", has_vector: bool = True):
    """轻量条目替身（选择器纯函数测试用——不依赖 EpisodicEntry 类）。"""
    return SimpleNamespace(
        text=text, confidence=confidence, recall_count=recall_count,
        semantic_vector=(object() if has_vector else None),
        doubt_state=doubt_state)


def _seed_entry(loop, text: str) -> None:
    """直接经 memory.store_episodic 落一条旧记忆（E3 重激活的目标）。

    不经 process_turn → 不触发注入 suspect/labile/入队——队列保持干净，
    min-age 闸门与重激活路径的因果清晰可测。
    """
    loop.memory.store_episodic(
        text, torch.randn(16), surprise=0.5, turn=loop.turn_count)


# =========================================================================== #
#  1. 选择器排序（纯函数）
# =========================================================================== #

class TestSelectorPure:

    def test_select_cases_sorted_by_score(self):
        """排序：score 降序；低 recall（罕见度）→ 高 epistemic → 高分。"""
        from core.doubt.epistemic_selector import select_cases
        e_rare = _entry("旧记忆A 框架", confidence=0.7, recall_count=0)
        e_common = _entry("旧记忆B 框架", confidence=0.7, recall_count=50)
        fok = [
            {"topic": "旧记忆A 框架", "detail": "疑点 1"},
            {"topic": "旧记忆B 框架", "detail": "疑点 2"},
        ]
        cands = select_cases(fok, [], episodic=[e_rare, e_common])
        assert len(cands) == 2
        assert cands[0]["topic"].startswith("旧记忆A")
        assert cands[0]["score"] >= cands[1]["score"]
        assert cands[0]["epistemic"] > cands[1]["epistemic"]
        assert cands[0]["target_found"] is True
        assert cands[0]["entry_key"]  # 稳定键非空
        assert cands[0]["reachable"] == 1.0  # 语义向量在场

    def test_select_cases_excludes_resolved_and_superseded(self):
        """过滤：excluded_topics（resolved/冷却）与 superseded 目标跳过。"""
        from core.doubt.epistemic_selector import select_cases
        e_stable = _entry("旧记忆A", confidence=0.7)
        e_sup = _entry("旧记忆B", confidence=0.7, doubt_state="superseded")
        fok = [
            {"topic": "旧记忆A", "detail": ""},
            {"topic": "旧记忆B", "detail": ""},
            {"topic": "旧记忆C", "detail": ""},
        ]
        cands = select_cases(
            fok, [], episodic=[e_stable, e_sup],
            excluded_topics=["旧记忆C"])
        topics = [c["topic"] for c in cands]
        assert "旧记忆A" in topics
        assert "旧记忆B" not in topics   # 目标已 superseded → 跳过
        assert "旧记忆C" not in topics   # excluded（resolved/冷却）→ 跳过

    def test_progress_rising_preferred_over_falling(self):
        """progress：近 N 轮 surprise 差分 >0（学习进度）> 差分 <0。"""
        from core.doubt.epistemic_selector import select_cases
        e = _entry("记忆X", confidence=0.7)
        fok = [{"topic": "记忆X", "detail": ""}]
        rising = [3.0, 3.0, 3.0, 4.0, 4.5, 5.0]
        falling = [5.0, 4.5, 4.0, 3.0, 3.0, 3.0]
        c_r = select_cases(fok, [], episodic=[e],
                           surprise_window=rising)[0]
        c_f = select_cases(fok, [], episodic=[e],
                           surprise_window=falling)[0]
        assert c_r["progress"] > c_f["progress"]

    def test_unreachable_candidate_low_reachable(self):
        """reachable：无目标条目 → 0（不可达悬案排后）。"""
        from core.doubt.epistemic_selector import select_cases
        fok = [{"topic": "找不到的悬案", "detail": ""}]
        cands = select_cases(fok, [], episodic=[], max_candidates=5)
        assert cands
        assert cands[0]["target_found"] is False
        assert cands[0]["reachable"] == 0.0
        assert cands[0]["entry_key"] == ""


# =========================================================================== #
#  2. doubt_ingest 协议扩展（根因 1/3 修复）
# =========================================================================== #

class TestDoubtIngestE3:

    def test_alias_zhengwei_maps_to_conflict(self):
        """拍板 2：'证伪' 别名 → conflict 语义（证伪收编）。"""
        from core.doubt.doubt_ingest import parse_doubt_event
        ev = parse_doubt_event("[doubt] 证伪: 意志=上下文塑形 的框架错了")
        assert ev is not None
        assert ev["kind"] == "conflict"

    def test_alias_reactivate_maps_to_conflict(self):
        """拍板 2：'reactivate' 别名 → conflict 语义（重激活收编）。"""
        from core.doubt.doubt_ingest import parse_doubt_event
        ev = parse_doubt_event("[doubt] reactivate: 尹丽娜角色重塑决策")
        assert ev is not None
        assert ev["kind"] == "conflict"
        # 原 conflict 不受影响
        assert parse_doubt_event(
            "[doubt] conflict: 旧记忆")["kind"] == "conflict"
        # 未知 kind 仍 fail-open 返回 None
        assert parse_doubt_event("[doubt] 未知协议: x") is None

    def test_conflict_hit_enqueues_candidate(self, tmp_path, monkeypatch):
        """根因 3 修复：conflict 证伪命中 → 候选进队列（写侧时相）。

        E3 关（无 min-age 闸门）时同轮即消化：候选入队 → 本回合巩固期
        consolidation_resolve → superseded（labile 复位）。可观测 = 条目
        被受控改写（证明"入队→三闸门→消化"链路通）。
        """
        monkeypatch.setenv("LMS_E3_ENABLED", "0")  # 入队修复不受 E3 总开关治理
        loop = make_loop(tmp_path)
        _seed_entry(loop, "三娘角色重塑决策 值得重新审视")
        target = list(loop.memory.iter_episodic())[-1]
        assert loop.reconsolidation_queue.size() == 0
        loop.process_turn("[doubt] conflict: 三娘角色重塑决策")
        # conflict 命中 → mark_labile + enqueue → 本回合消化（E3 关无闸门）
        assert getattr(target, "doubt_state", "stable") == "superseded"
        assert target.rebuttal_count >= 1
        assert target.violated_by is not None

    def test_ingest_target_entry_explicit(self, tmp_path, monkeypatch):
        """E3 重激活专用：显式 target_entry 跳过重叠匹配（fail-open）。"""
        from core.doubt.doubt_ingest import ingest
        monkeypatch.setenv("LMS_E3_ENABLED", "0")
        loop = make_loop(tmp_path)
        _seed_entry(loop, "目标旧记忆 甲乙丙")
        target = list(loop.memory.iter_episodic())[-1]
        ev = ingest(loop, "[doubt] conflict: 完全无关的线索文本",
                    target_entry=target)
        assert ev is not None
        assert ev["action"] == "rebutted"
        assert ev["entry"] is target
        assert target.labile is True


# =========================================================================== #
#  3. min-age 闸门（根因 4 修复）
# =========================================================================== #

class TestMinAgeGate:

    def test_e3_on_candidate_survives_same_turn(self, tmp_path, monkeypatch):
        """E3 开（默认 min-age=1）：同轮入队不消化，下一轮才消费。

        根治"同轮即消"：候选至少活过当轮，做梦 M6 能扫到 labile/suspect。
        """
        monkeypatch.setenv("LMS_E3_ENABLED", "1")
        loop = make_loop(tmp_path)
        _seed_entry(loop, "三娘角色重塑决策 值得重新审视")
        target = list(loop.memory.iter_episodic())[-1]
        assert loop.reconsolidation_queue.size() == 0
        # conflict 事件在 process_turn ingest 步入队；巩固期在本轮内
        loop.process_turn("[doubt] conflict: 三娘角色重塑决策")
        assert loop.reconsolidation_queue.contains(target) is True, \
            "同轮入队候选必须活过当轮（min-age=1）"
        assert getattr(target, "doubt_state", "stable") == "stable"
        assert target.labile is True
        # 下一轮：turn 起始快照已含该候选 → 巩固期消费（受控改写落库）
        loop.process_turn("用户: 无关内容")
        assert loop.reconsolidation_queue.contains(target) is False, \
            "候选须在下一轮被巩固期消化"
        assert getattr(target, "doubt_state", "stable") == "superseded"

    def test_e3_off_same_turn_consumption_preserved(self, tmp_path,
                                                    monkeypatch):
        """E3 关：min-age 闸门不生效——行为与 E3 引入前逐位一致。"""
        monkeypatch.setenv("LMS_E3_ENABLED", "0")
        assert _e3_min_age_turns() == 0
        loop = make_loop(tmp_path)
        _seed_entry(loop, "三娘角色重塑决策 值得重新审视")
        target = list(loop.memory.iter_episodic())[-1]
        loop.process_turn("[doubt] conflict: 三娘角色重塑决策")
        # E3 关：无闸门 → 本回合巩固期即消化（同轮即消保持）
        assert loop.reconsolidation_queue.contains(target) is False
        assert getattr(target, "doubt_state", "stable") == "superseded"


# =========================================================================== #
#  4. satiety（judge_resolved + SatietyGate）
# =========================================================================== #

class TestSatietyGate:

    def test_gate_cooldown_and_expiry(self):
        """per-topic 冷却：重激活后冷却期内 is_cooldown，到期自动失效。"""
        from core.doubt.satiety import SatietyGate
        g = SatietyGate(cooldown_h=12.0, max_cycles=3, standby_h=24.0)
        g.set_clock(1000.0)
        assert g.is_armed()
        g.record_reactivation("悬案A")
        assert g.is_cooldown("悬案A")
        assert g.cooldown_remaining("悬案A") > 0
        g.set_clock(1000.0 + 12 * 3600)  # 12h 后（dandan 拍板放宽）
        assert not g.is_cooldown("悬案A")

    def test_gate_standby_after_max_cycles_and_rearm(self):
        """3 次闭环后待机（终止信号防 OCD 化）；rearm 重新武装。"""
        from core.doubt.satiety import SatietyGate
        g = SatietyGate(cooldown_h=1.0, max_cycles=3, standby_h=24.0)
        g.set_clock(5000.0)
        for _ in range(3):
            g.close_cycle()
        assert g.in_standby()
        assert not g.is_armed()
        assert g.snapshot()["in_standby"] is True
        g.rearm()
        assert g.is_armed()
        assert g.cycle_count == 0

    def test_judge_resolved(self):
        """消解判定：superseded / rewritten ≥1 / kept ≥N → resolved。"""
        from core.doubt.satiety import judge_resolved
        assert judge_resolved(SimpleNamespace(doubt_state="superseded")) \
            == "resolved"
        stable = SimpleNamespace(doubt_state="stable")
        assert judge_resolved(stable, outcomes={"rewritten": 1}) == "resolved"
        assert judge_resolved(stable, outcomes={"kept": 3}) == "resolved"
        assert judge_resolved(stable, outcomes={"kept": 1}) == "pending"
        assert judge_resolved(None, outcomes={}) == "pending"


# =========================================================================== #
#  5. gap_registry 消解（satiety 落点）
# =========================================================================== #

class TestGapRegistryResolved:

    def test_mark_resolved_lifecycle(self):
        """mark_resolved：fok_resolved 追加 + 未决移除 + snapshot/联动。"""
        from core.doubt.gap_registry import GapRegistry
        g = GapRegistry()
        g.register_fok_unresolved(topic="悬案A")
        assert g.snapshot()["fok_unresolved"] != []
        g.mark_resolved("悬案A")
        assert g.is_resolved("悬案A")
        assert g.resolved_count() == 1
        assert all(r["topic"] != "悬案A"
                   for r in g.snapshot()["fok_unresolved"])
        assert any(r["topic"] == "悬案A"
                   for r in g.snapshot()["fok_resolved"])
        # mark_review 联动：resolved_topics 一并消解；fok_resolved 不清空
        g.register_fok_unresolved(topic="悬案B")
        g.mark_review({"rewritten": 1}, resolved_topics=["悬案B"])
        assert g.is_resolved("悬案B")
        assert g.resolved_count() == 2


# =========================================================================== #
#  6. loop.e3_reactivate（重激活落库 / 待机 / 开关关零参与）
# =========================================================================== #

class TestE3ReactivationLoop:

    def test_switch_off_zero_participation(self, tmp_path, monkeypatch):
        """总开关关：e3_reactivate 返回 {enabled:false} 零参与（A6）。"""
        monkeypatch.setenv("LMS_E3_ENABLED", "0")
        loop = make_loop(tmp_path)
        res = loop.e3_reactivate()
        assert res["enabled"] is False
        assert loop.reconsolidation_queue.size() == 0
        assert loop.gap_registry.snapshot()["fok_resolved"] == []

    def test_reactivation_path_a_persists_to_queue(self, tmp_path,
                                                   monkeypatch):
        """重激活落库（路径 A）：选悬案 → conflict → labile + 候选进队列。"""
        monkeypatch.setenv("LMS_E3_ENABLED", "1")
        loop = make_loop(tmp_path)
        _seed_entry(loop, "三娘角色重塑决策 值得重新审视")
        loop.gap_registry.register_fok_unresolved(
            topic="三娘角色重塑决策", detail="角色重塑方向存疑")
        res = loop.e3_reactivate()
        assert res["enabled"] is True
        assert res["armed"] is True
        assert res["selected"] >= 1
        assert res["activated"] >= 1
        assert res["activated_items"][0]["path"] == "A"
        assert loop.reconsolidation_queue.size() >= 1, \
            "重激活候选必须带证据进队列（根因 3 修复链路）"
        target = list(loop.memory.iter_episodic())[-1]
        assert target.labile is True
        assert "surprise_before" in res  # A2 观测字段（不设阈值判定）

    def test_reactivation_dry_run_no_activation(self, tmp_path, monkeypatch):
        """dry_run：只选择不激活（A1 观测）——队列/条目零改动。"""
        monkeypatch.setenv("LMS_E3_ENABLED", "1")
        loop = make_loop(tmp_path)
        _seed_entry(loop, "三娘角色重塑决策 值得重新审视")
        loop.gap_registry.register_fok_unresolved(
            topic="三娘角色重塑决策")
        res = loop.e3_reactivate(dry_run=True)
        assert res["enabled"] is True
        assert res["dry_run"] is True
        assert res["selected"] >= 1
        assert res["activated"] == 0
        assert loop.reconsolidation_queue.size() == 0
        assert list(loop.memory.iter_episodic())[-1].labile is False

    def test_satiety_standby_blocks_reactivation(self, tmp_path, monkeypatch):
        """satiety 待机：累计闭环达上限 → 后续触发不参与（A5）。"""
        monkeypatch.setenv("LMS_E3_ENABLED", "1")
        monkeypatch.setenv("LMS_E3_SATIETY_MAX_CYCLES", "1")
        loop = make_loop(tmp_path)
        _seed_entry(loop, "三娘角色重塑决策 值得重新审视")
        loop.gap_registry.register_fok_unresolved(topic="三娘角色重塑决策")
        res1 = loop.e3_reactivate()
        assert res1["activated"] >= 1
        assert loop.satiety_gate.in_standby() is True, \
            "累计 1/1 闭环 → 待机"
        res2 = loop.e3_reactivate()
        assert res2["enabled"] is True
        assert res2["armed"] is False
        assert res2["activated"] == 0
        # 新 [doubt] 事件 → rearm 恢复（设计 §3.2 ⑤ 恢复路径）
        loop.process_turn("[doubt] fok: 一个新的问题")
        assert loop.satiety_gate.is_armed() is True

    def test_satiety_resolution_sweep_marks_fok_resolved(self, tmp_path,
                                                         monkeypatch):
        """消解扫：上轮重激活目标 superseded → fok_resolved +1（A5）。"""
        monkeypatch.setenv("LMS_E3_ENABLED", "1")
        loop = make_loop(tmp_path)
        _seed_entry(loop, "三娘角色重塑决策 值得重新审视")
        loop.gap_registry.register_fok_unresolved(topic="三娘角色重塑决策")
        res = loop.e3_reactivate()
        assert res["activated"] >= 1
        # 模拟梦期/巩固期改写落库：目标转 superseded
        target = loop._e3_find_target_by_key(
            loop._e3_pending_targets[0]["entry_key"])
        from core.doubt.state_machine import (
            DoubtPhase, EntryDoubtState)
        loop.doubt_state._set_doubt_state(
            target, EntryDoubtState.SUPERSEDED,
            phase=DoubtPhase.CONSOLIDATION)
        # 下一次触发：消解扫收口 → fok_resolved +1
        loop.e3_reactivate(dry_run=True)
        assert loop.gap_registry.resolved_count() >= 1
        assert loop.gap_registry.is_resolved("三娘角色重塑决策")

    def test_status_e3_observation(self, tmp_path, monkeypatch):
        """/status e3 观测块（闭环计数/待机态/fok_resolved 计数）。"""
        monkeypatch.setenv("LMS_E3_ENABLED", "1")
        loop = make_loop(tmp_path)
        obs = loop.e3_observation()
        assert obs["enabled"] is True
        assert "satiety" in obs
        assert "closed_loops" in obs
        assert "fok_resolved_count" in obs
        assert obs["satiety"]["cooldown_h"] == 12.0  # dandan 拍板放宽


# =========================================================================== #
#  7. /e3/review 端点（A6 开关回归）
# =========================================================================== #

class TestE3ReviewEndpoint:

    def test_endpoint_disabled_zero_participation(self, monkeypatch):
        """LMS_E3_ENABLED=0 → /e3/review 返回 {enabled:false}，不建会话。"""
        import asyncio
        import api.server as server_module
        monkeypatch.setenv("LMS_E3_ENABLED", "0")
        req = server_module.E3ReviewRequest(session_id="main")
        body = asyncio.run(server_module.e3_review(req))
        assert body["enabled"] is False

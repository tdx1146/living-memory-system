# -*- coding: utf-8 -*-
"""R3（C1）labile 运行时接线测试：loop 巩固期调用 maybe_rewrite 三闸门。

覆盖（语义决策 D-2026-08-18-01 运行时落地——reconsolidation_queue 已在
core/doubt/，本文件验证 loop 接线）：
  1. loop 暴露 reconsolidation_queue（跨重启持久化队列实例）
  2. 巩固期（doubt_check 后、state_update 前）调用 maybe_rewrite：候选
     被三闸门消化（写侧委托 state_machine 巩固时相入口）——队列排空 +
     条目受控改写 + state_update 观测计数 + 9 步生命周期不破
  3. 写侧入队接线：注入 suspect 标记后登记（commit 步）/ 去稳定化 labile
     标记后登记（体验层 D）——只允许写侧时相
  4. 检索路径零改写保持：/recall、/react 只读口不碰队列、不消化候选、
     只读四不变不破坏
  5. 治理开关 LMS_DOUBT_RECONSOLIDATION_ENABLED=0 → 零参与
  6. C3（P2-B 口径修正）接线：loop 目的检查消费 readonly_round /
     reconsolidated 信号——只读轮 Q1 豁免、再巩固轮 Q1 判"是"

运行方式：pytest rewrite-ws/tests/test_loop_reconsolidation_labile.py -v。
"""

import os
import sys

# 确保项目根目录在 Python 路径中（可从任意 cwd 运行）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
import torch

from core.doubt.reconsolidation_queue import ReconsolidationQueue
from core.doubt.state_machine import DoubtPhase, EntryDoubtState
from runtime.config import default_config
from runtime.loop import LivingMemoryLoop, MODULE_CLAIMS


# --------------------------------------------------------------------------- #
#  测试小工具（与 test_loop_m5 / test_doubt_m3 同款风格）
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


def _enqueue_suspect_candidate(loop, entry=None, reason="test_candidate"):
    """把一条条目标 suspect 并入队（模拟写侧注入 suspect 登记）。"""
    if entry is None:
        entry = list(loop.memory.iter_episodic())[-1]
    loop.doubt_state.injection_check(
        entry, surprise=999.0, j_target=1.0,
        verification_chain=loop.verification_chain)
    assert entry.doubt_state == EntryDoubtState.SUSPECT.value
    assert loop.reconsolidation_queue.enqueue(
        entry, reason=reason, score=999.0,
        phase=DoubtPhase.INJECTION.value) is True
    return entry


# =========================================================================== #
#  1. loop 暴露队列实例 + 巩固期 maybe_rewrite 三闸门消化
# =========================================================================== #

class TestLoopReconsolidationWiring:

    def test_loop_exposes_reconsolidation_queue(self, tmp_path):
        """loop 持有再巩固候选队列实例（R3 C1 接线载体）。"""
        loop = make_loop(tmp_path)
        assert isinstance(loop.reconsolidation_queue, ReconsolidationQueue)
        assert loop.reconsolidation_queue.enabled is True
        assert loop._last_turn_reconsolidated is False

    def test_consolidation_phase_drains_queue_via_maybe_rewrite(self, tmp_path):
        """巩固期（doubt_check 后、state_update 前）调用 maybe_rewrite：
        候选被消化（队列排空 + 条目受控改写 + 观测计数）。"""
        loop = make_loop(tmp_path)
        loop.process_turn('用户: 第一条记忆')
        entry = _enqueue_suspect_candidate(loop, reason="high_surprise")
        assert loop.reconsolidation_queue.size() == 1
        # 下一轮 process_turn：巩固期 maybe_rewrite 三闸门全过 → 消化
        loop.process_turn('用户: 无关内容')
        assert loop.reconsolidation_queue.size() == 0, "候选必须被消化"
        # 写侧委托（state_machine 巩固时相入口）真实发生：suspect 被裁决
        assert entry.doubt_state in (
            EntryDoubtState.STABLE.value,
            EntryDoubtState.SUPERSEDED.value), entry.doubt_state
        # state_update 观测计数 + 9 步生命周期不破
        assert loop.last_turn_lifecycle['steps']['state_update'][
            'reconsolidated'] >= 1
        assert loop.last_turn_lifecycle['broken'] is False
        # 下一轮目的时相 reconsolidated 信号（诚实信号：再巩固真实发生）
        assert loop._last_turn_reconsolidated is True

    def test_consolidation_phase_leaves_non_candidates_untouched(self, tmp_path):
        """G1 候选在队：不在队列的条目经 process_turn 零改写（机器防线）。"""
        loop = make_loop(tmp_path)
        loop.process_turn('用户: 普通记忆')
        entry = list(loop.memory.iter_episodic())[-1]
        assert loop.reconsolidation_queue.contains(entry) is False
        loop.process_turn('用户: 又一轮')
        # 未入队条目：doubt_state / labile 零触碰
        assert getattr(entry, "doubt_state", "stable") == "stable"
        assert getattr(entry, "labile", False) is False

    def test_retrieval_paths_zero_rewrite(self, tmp_path):
        """检索路径零改写保持：/recall 与 /react 只读口不碰队列、不消化
        候选、只读四不变不破坏（C1 铁律）。"""
        loop = make_loop(tmp_path)
        loop.process_turn('用户: 蓝色小橘猫很可爱')
        entry = _enqueue_suspect_candidate(loop, reason="suspect_candidate")
        assert loop.reconsolidation_queue.size() == 1
        before = loop._readonly_state_snapshot()
        loop.recall_episodic_readonly('蓝色小橘猫', k=3)
        loop.react_readonly('蓝色小橘猫')
        loop.recall_merged_readonly('蓝色小橘猫', k=3)
        assert loop._readonly_state_snapshot() == before  # 只读四不变
        # 队列零变化：候选未被只读路径消化
        assert loop.reconsolidation_queue.size() == 1
        assert loop.reconsolidation_queue.contains(entry) is True
        assert entry.doubt_state == EntryDoubtState.SUSPECT.value
        # 只读口不产生生命周期记录
        assert loop.lifecycle_trace is None

    def test_switch_off_zero_participation(self, tmp_path, monkeypatch):
        """治理开关关（LMS_DOUBT_RECONSOLIDATION_ENABLED=0）→ 队列零参与，
        process_turn 照常、观测计数 0。"""
        monkeypatch.setenv("LMS_DOUBT_RECONSOLIDATION_ENABLED", "0")
        loop = make_loop(tmp_path)
        assert loop.reconsolidation_queue.enabled is False
        loop.process_turn('用户: 开关关轮')
        assert loop.last_turn_lifecycle['broken'] is False
        assert loop.last_turn_lifecycle['steps']['state_update'][
            'reconsolidated'] == 0
        assert loop._last_turn_reconsolidated is False
        # 队列路径零落盘（无 candidates.json 生成）
        assert (tmp_path / 'recon' / 'candidates.json').exists() is False


# =========================================================================== #
#  2. 写侧入队接线（只允许写侧时相）
# =========================================================================== #

class TestWriteSideEnqueueWiring:

    def test_injection_suspect_marks_enqueues_candidate(self, tmp_path,
                                                        monkeypatch):
        """commit 步注入 suspect 标记后登记再巩固候选（写侧入队即落盘）。

        用 LMS_DOUBT_SURPRISE_FACTOR=0 让本轮新条目必然高 surprise → 标
        suspect → 经 loop 接线入队（spy 验证 reason=injection_suspect）。
        """
        monkeypatch.setenv("LMS_DOUBT_SURPRISE_FACTOR", "0")
        loop = make_loop(tmp_path)
        calls = []
        orig_enqueue = loop.reconsolidation_queue.enqueue

        def _spy(*args, **kwargs):
            calls.append((args, kwargs))
            return orig_enqueue(*args, **kwargs)

        loop.reconsolidation_queue.enqueue = _spy
        loop.process_turn('用户: 高惊讶新条目')
        assert len(calls) >= 1, "注入 suspect 后必须登记再巩固候选"
        assert calls[0][1].get('reason') == "injection_suspect"
        assert calls[0][1].get('phase') == DoubtPhase.INJECTION.value
        # 写侧入队即落盘（持久化契约）：队列文件已生成
        assert (tmp_path / 'recon' / 'candidates.json').exists() is True
        # 条目经本 turn 巩固期被受控裁决（suspect → stable/superseded——
        # 写侧委托 state_machine 巩固时相入口真实发生）
        entry = list(loop.memory.iter_episodic())[-1]
        assert entry.doubt_state in (
            EntryDoubtState.STABLE.value,
            EntryDoubtState.SUPERSEDED.value), entry.doubt_state

    def test_destabilize_labile_enqueues_candidate(self, tmp_path):
        """体验层 D 去稳定化（mark_labile）后登记再巩固候选（写侧时相）。"""
        loop = make_loop(tmp_path)
        loop.process_turn('用户: 去稳定化目标记忆')
        # 填满 20 轮窗口（小 surprise 基线），再注入高 surprise → z>2
        loop.destab_surprise_window.clear()
        for i in range(20):
            loop.destab_surprise_window.append(0.005 if i % 2 else 0.015)
        calls = []
        orig_enqueue = loop.reconsolidation_queue.enqueue

        def _spy(*args, **kwargs):
            calls.append((args, kwargs))
            return orig_enqueue(*args, **kwargs)

        loop.reconsolidation_queue.enqueue = _spy
        sem = loop.embedder.embed_text('去稳定化目标记忆')
        act = type("Act", (), {"surprise": 0.1})()
        loop._destabilize_if_high_surprise(act, sem, sem, '去稳定化目标记忆')
        assert len(calls) >= 1, "labile 标记后必须登记再巩固候选"
        assert calls[0][1].get('reason') == "destabilized_labile"
        assert calls[0][1].get('phase') == DoubtPhase.INJECTION.value


# =========================================================================== #
#  3. C3（P2-B 口径修正）接线：loop 目的检查消费 readonly_round /
#     reconsolidated 信号
# =========================================================================== #

class TestPurposeDriftWiring:

    def test_readonly_round_signal_wired(self, tmp_path):
        """只读轮（无写侧活动）→ readonly_round=True → Q1 豁免不判 drifted。"""
        loop = make_loop(tmp_path)
        gate = loop._purpose_drift_check(
            episodic_added=False, doubt_events=0, suspect_marked=False)
        assert gate["answers"]["Q1"]["state"] == "exempt"
        assert gate["verdict"] != "drifted"

    def test_write_round_signal_wired(self, tmp_path):
        """写轮（episodic_added=True）→ readonly_round=False → Q1 判"是"。"""
        loop = make_loop(tmp_path)
        gate = loop._purpose_drift_check(
            episodic_added=True, doubt_events=0, suspect_marked=False)
        assert gate["answers"]["Q1"]["state"] == "yes"

    def test_reconsolidated_signal_flows_to_next_round(self, tmp_path):
        """再巩固真实发生 → 下一轮目的检查 reconsolidated=True → Q1 判"是"
        （诚实信号：不因"本轮无 episodic 写入"误判只读轮）。"""
        loop = make_loop(tmp_path)
        loop.process_turn('用户: 记忆A')
        entry = _enqueue_suspect_candidate(loop)   # 保证下一轮巩固期必有消化
        # 本轮巩固期消化候选 → _last_turn_reconsolidated=True
        loop.process_turn('用户: 记忆B')
        assert loop._last_turn_reconsolidated is True
        # 下一轮目的检查：无 episodic 写入但有再巩固 → 写侧活动 → Q1 是
        gate = loop._purpose_drift_check(
            episodic_added=False, doubt_events=0, suspect_marked=False)
        assert gate["answers"]["Q1"]["state"] == "yes"
        # 对照：无再巩固活动的新 loop → 无写侧活动 → Q1 豁免（只读轮）
        loop2 = make_loop(tmp_path)
        gate2 = loop2._purpose_drift_check(
            episodic_added=False, doubt_events=0, suspect_marked=False)
        assert gate2["answers"]["Q1"]["state"] == "exempt"

    def test_purpose_check_still_records_drift_gap_for_write_idle(self,
                                                                  tmp_path):
        """写轮无任何写侧活动 → Q1 否 → drifted + gap 登记（口径不放松——
        只读轮豁免不放松写轮判定）。"""
        loop = make_loop(tmp_path)
        # 直接构造：无写侧活动且明确是写轮（round_type 由 loop 按信号推导；
        # 此处用 .judge 验证 write 轮语义在 loop 之外的判定仍为否）
        from core.doubt.purpose_drift import PurposeDriftPhase
        out = PurposeDriftPhase().judge({"round_type": "write"})
        assert out["answers"]["Q1"]["state"] == "no"
        assert out["verdict"] == "drifted"


# =========================================================================== #
#  4. claim 登记一致（runtime/claims.json 与 MODULE_CLAIMS 同源）
# =========================================================================== #

class TestClaims:

    def test_module_claims_include_wiring(self):
        """MODULE_CLAIMS 登记 C1 接线 claim（machine-readable，§5.2）。"""
        assert "reconsolidation_wired_in_consolidation_phase" in \
            MODULE_CLAIMS["claims"]
        claim = MODULE_CLAIMS["claims"][
            "reconsolidation_wired_in_consolidation_phase"]
        assert claim["statement"].strip()
        assert claim["verified_by"] == "tests/test_loop_reconsolidation_labile.py"

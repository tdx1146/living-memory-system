# -*- coding: utf-8 -*-
"""判别力恢复 · SURPRISE_FACTOR 校准测试（总任务书 §二.1）

依据：``LMS_DOUBT_SURPRISE_FACTOR`` 校准 1.2~1.3 —— 门槛
factor×J_target = 48~52（j_target=40）> surprise_mean+2.5σ≈47.8 →
常态条目放行、异常条目仍拦；判别器饱和（17/17 全 suspect）修复。

对象：``core/doubt/state_machine.py`` 的
  - ``is_high_surprise(surprise, j_target, factor=None)``
  - ``DoubtStateMachine(surprise_factor=...)`` 与 ``injection_check(...)``

判据与代码语义核对（写前通读 state_machine.py 确认）：
  - ``is_high_surprise``：``surprise > factor × J_target``（**严格大于**，
    等于门槛 → False）；缺 surprise/j_target → False（fail-open）；
    factor=None → 每调用读 env ``LMS_DOUBT_SURPRISE_FACTOR``
    （``_surprise_factor()``：max(0.0, float(env))，非法值回退默认 1.0）；
    显式 factor → max(0.0, float(factor)) 钳制；
  - ``DoubtStateMachine.__init__``：``surprise_factor`` 缺省 → 构造时读
    env（非模块级缓存）；显式 → max(0.0, float(...)) 钳制；
  - ``injection_check(entry, *, surprise, j_target, rebuttal_hit,
    verification_chain, register_query, registered_phase)``：
    ``is_high_surprise(surprise, j_target, self.surprise_factor)``
    为真或 rebuttal_hit → 标 suspect + 登记验证链（链开关默认关→跳过）；
    否则返回条目当前态（stable）。

校准目标：factor=1.25 × j_target=40 → 门槛 50；
  - surprise=45（常态，<48 门槛且 <50）→ 放行（False）；
  - surprise=60（异常，>52）→ 拦截（True）；
  - surprise=50（门槛附近，=50）→ 严格大于 → False（实现定义）。

运行方式：pytest rewrite-ws/tests/test_discrimination_recovery.py -v
"""

import os
import sys

# 确保项目根目录在 Python 路径中（可从任意 cwd 运行）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
import torch

from core.doubt.state_machine import (
    DoubtStateMachine,
    EntryDoubtState,
    _surprise_factor,
    is_high_surprise,
)
from core.doubt.verification_chain import VerificationChain
from core.hippocampus.memory import EpisodicEntry


def _make_entry(text="测试记忆"):
    """最小条目替身（同 tests/test_doubt_m3.py fixture 风格——EpisodicEntry）。"""
    return EpisodicEntry(
        text=text, semantic_vector=torch.zeros(8), surprise=1.0, turn=0)


# 饱和场景批：17 条常态（均 45）+ 1 条异常（60），j_target=40。
# 均 45 < mean+2.5σ≈47.8（常态带）；异常 60 明显越界。
_NORMAL_SURPRISE = 45.0
_ANOMALY_SURPRISE = 60.0
_J_TARGET = 40.0
_SATURATION_BATCH = [_NORMAL_SURPRISE] * 17 + [_ANOMALY_SURPRISE]


# ===========================================================================
# 1. is_high_surprise 语义（factor=1.25 × j_target=40 → 门槛 50）
# ===========================================================================

class TestIsHighSurpriseSemantics:
    """校准语义：常态放行 / 异常拦截 / 门槛附近按实现定义 / 缺值 fail-open。"""

    def test_normal_below_threshold_not_high(self):
        """factor=1.25、j_target=40：surprise=45（常态，<48 最低校准门槛）
        → False（放行——判别力恢复目标）。"""
        assert is_high_surprise(45, 40, 1.25) is False

    def test_anomaly_above_threshold_high(self):
        """surprise=60（异常，>52 最高校准门槛）→ True（仍拦——异常不放过）。"""
        assert is_high_surprise(60, 40, 1.25) is True

    def test_boundary_equal_threshold_not_high(self):
        """surprise=50（门槛附近，等于 factor×j_target=1.25×40=50）→
        按实现定义（严格大于）→ False。"""
        assert is_high_surprise(50, 40, 1.25) is False, \
            "实现为严格 '>'：surprise == 门槛 不判高（实现定义断言）"

    def test_missing_values_fail_open(self):
        """缺 surprise / j_target → False（fail-open：信息不足不判高）。"""
        assert is_high_surprise(None, 40, 1.25) is False
        assert is_high_surprise(45, None, 1.25) is False

    def test_default_factor_baseline_saturation(self):
        """默认 factor=1.0 基线（当前饱和行为的文档化）：surprise=41 >
        1.0×40 → True——旧行为下常态 45 也全部越线（饱和根因）。"""
        assert is_high_surprise(41, 40, 1.0) is True
        assert is_high_surprise(_NORMAL_SURPRISE, _J_TARGET, 1.0) is True


# ===========================================================================
# 2. env 解析（LMS_DOUBT_SURPRISE_FACTOR 读取时机）
# ===========================================================================

class TestEnvParsing:
    """env 解析：构造时读取 / 每调用读取 / 非法值回退 / 显式优先。"""

    def test_env_read_at_construction(self, monkeypatch):
        """monkeypatch env=1.25 → 新建 DoubtStateMachine() →
        surprise_factor == 1.25（__init__ 时经 _surprise_factor() 读取 env）。"""
        monkeypatch.setenv("LMS_DOUBT_SURPRISE_FACTOR", "1.25")
        sm = DoubtStateMachine()
        assert sm.surprise_factor == 1.25

    def test_env_read_per_call_is_high_surprise(self, monkeypatch):
        """is_high_surprise factor=None → **每调用**读 env（模块级函数
        非缓存）：env=1.25 下 45 放行、60 拦截。"""
        monkeypatch.setenv("LMS_DOUBT_SURPRISE_FACTOR", "1.25")
        assert is_high_surprise(45, 40) is False
        assert is_high_surprise(60, 40) is True
        # 换 env 即时生效（运行时读取，改动即时生效——§5.3）
        monkeypatch.setenv("LMS_DOUBT_SURPRISE_FACTOR", "1.0")
        assert is_high_surprise(45, 40) is True

    def test_invalid_env_falls_back_default(self, monkeypatch):
        """非法 env（非数值）→ _surprise_factor() 回退默认 1.0；构造与
        每调用两条路径一致。"""
        monkeypatch.setenv("LMS_DOUBT_SURPRISE_FACTOR", "abc")
        assert _surprise_factor() == 1.0
        assert DoubtStateMachine().surprise_factor == 1.0
        assert is_high_surprise(45, 40) is True, "回退 1.0 → 45>40 判高"

    def test_explicit_surprise_factor_overrides_env(self, monkeypatch):
        """显式 surprise_factor 优先于 env（缺省才读 env）。"""
        monkeypatch.setenv("LMS_DOUBT_SURPRISE_FACTOR", "1.25")
        sm = DoubtStateMachine(surprise_factor=1.0)
        assert sm.surprise_factor == 1.0
        assert sm.surprise_factor != 1.25


# ===========================================================================
# 3. 饱和修复场景（17/17 全 suspect → 校准后不全 suspect）
# ===========================================================================

class TestSaturationRecovery:
    """判别器饱和修复：均 45 的常态批在 factor=1.0 下全 suspect（17/17
    饱和形态复现——文档化）；factor=1.25 下仅异常值 True（常态放行）。"""

    def test_factor_125_batch_not_all_suspect(self):
        """factor=1.25：同一批（17 常态 45 + 1 异常 60）逐个判定 →
        不全 suspect——17 条 False、仅异常 1 条 True（饱和修复）。"""
        marks = [is_high_surprise(s, _J_TARGET, 1.25)
                 for s in _SATURATION_BATCH]
        assert sum(marks) == 1, "仅异常值（60）判高"
        assert marks[-1] is True
        assert all(not m for m in marks[:-1]), "17 条常态全部放行"

    def test_factor_100_batch_all_suspect(self):
        """factor=1.0（旧行为基线）：同一批 → 全 suspect（17/17 饱和形态
        复现——文档化：均 45 > 40 全部越线）。"""
        marks = [is_high_surprise(s, _J_TARGET, 1.0)
                 for s in _SATURATION_BATCH]
        assert all(marks), "17/17 全 suspect（饱和根因基线）"

    @pytest.mark.parametrize("factor,threshold", [
        (1.2, 48.0),   # 校准下限：1.2 × 40 = 48
        (1.3, 52.0),   # 校准上限：1.3 × 40 = 52
    ])
    def test_threshold_band_48_52(self, factor, threshold):
        """边界断言：门槛 = factor × j_target（48~52 区间）；surprise ==
        门槛 → False（严格大于）；门槛+0.1 → True。"""
        assert factor * _J_TARGET == pytest.approx(threshold)
        assert is_high_surprise(threshold, _J_TARGET, factor) is False
        assert is_high_surprise(threshold + 0.1, _J_TARGET, factor) is True
        # 门槛带内下沿（常态带 47.8 之上、门槛之下——如 47.9 < 48）不判高
        assert is_high_surprise(threshold - 0.1, _J_TARGET, factor) is False


# ===========================================================================
# 4. DoubtStateMachine 集成（injection_check 接线）
# ===========================================================================

class TestInjectionCheckIntegration:
    """DoubtStateMachine(surprise_factor=1.25) 集成：高 surprise → suspect
    标记 + 验证链登记；常态 surprise → 不标记（不误标）。"""

    def _machine(self, surprise_factor=1.25, chain=None):
        return DoubtStateMachine(surprise_factor=surprise_factor,
                                 verification_chain=chain)

    def test_injection_anomaly_marks_suspect(self):
        """injection_check：surprise=60（异常）→ 'suspect' 标记 + 验证链
        登记（链显式开）。"""
        chain = VerificationChain(enabled=True, window=60.0)
        sm = self._machine(surprise_factor=1.25, chain=chain)
        e = _make_entry(text="异常新条目")
        outcome = sm.injection_check(e, surprise=60.0, j_target=40.0,
                                     verification_chain=chain)
        assert outcome == EntryDoubtState.SUSPECT.value
        assert e.doubt_state == EntryDoubtState.SUSPECT.value
        assert sm.injection_suspect_marked == 1
        assert chain.pending_count() == 1, "suspect 条目登记验证链"

    def test_injection_normal_stays_stable(self):
        """injection_check：surprise=45（常态，<50 门槛）→ 不标记
        （保持 stable——判别力恢复的写侧语义）。"""
        sm = self._machine(surprise_factor=1.25)
        e = _make_entry(text="常态新条目")
        outcome = sm.injection_check(e, surprise=45.0, j_target=40.0)
        assert outcome == EntryDoubtState.STABLE.value
        assert getattr(e, "doubt_state", "stable") == "stable"
        assert sm.injection_suspect_marked == 0

    def test_injection_batch_saturation_recovery(self):
        """饱和修复集成：17 条常态（45）+ 1 条异常（60）逐条
        injection_check —— factor=1.25 下 17 条 stable、仅异常 suspect；
        factor=1.0 下同批 17/17 全 suspect（饱和形态文档化）。"""
        # 校准后：常态放行、异常仍拦
        sm = self._machine(surprise_factor=1.25)
        for i in range(17):
            e = _make_entry(text=f"常态条目{i}")
            assert sm.injection_check(e, surprise=45.0, j_target=40.0) \
                == EntryDoubtState.STABLE.value
        e_anomaly = _make_entry(text="异常条目")
        assert sm.injection_check(e_anomaly, surprise=60.0, j_target=40.0) \
            == EntryDoubtState.SUSPECT.value
        assert sm.injection_suspect_marked == 1

        # 旧行为基线：同一批全 suspect（17/17 饱和形态复现）
        sm_base = self._machine(surprise_factor=1.0)
        for i in range(17):
            e = _make_entry(text=f"基线条目{i}")
            assert sm_base.injection_check(e, surprise=45.0, j_target=40.0) \
                == EntryDoubtState.SUSPECT.value
        assert sm_base.injection_suspect_marked == 17

    def test_factor_clamp_non_positive(self):
        """参数合法性：factor<=0 → max(0.0, ...) 钳制（构造与纯函数一致）；
        钳到 0 后门槛为 0——surprise>0 判高、surprise==0 不判高。"""
        assert DoubtStateMachine(surprise_factor=-5).surprise_factor == 0.0
        assert DoubtStateMachine(surprise_factor=0).surprise_factor == 0.0
        assert is_high_surprise(1, 40, -5) is True   # 1 > 0×40
        assert is_high_surprise(0, 40, -5) is False  # 0 > 0×40 不成立

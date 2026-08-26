# -*- coding: utf-8 -*-
"""ABC §S3「B 落地（precision 学习化）」模块单测：π = 1/Var(surprise) 估计。

覆盖（任务书 §S3 模块部分 + 四妹-ABC操作规划-20260817.md §S3）：
  1. 治理开关默认关（LMS_PRECISION_LEARN 未设置）：estimate()=None、
     lr_multiplier 原样、snapshot enabled=False、observe 零参与不记录
  2. 开关开（显式参数或 env）：常数 surprise 序列 → var→0 → π 钳到 MIN
     （不除零、无 NaN）；高波动 → π 低、低波动 → π 高（方向断言）
  3. 窗口生效：滑动窗口长度限制，窗口外旧样本不参与
  4. EMA 低通：突变后估计渐进收敛（不跳变），方向断言
  5. 钳制：π 不超过 MAX / 不低于 MIN
  6. lr_multiplier：开关关原样；π 高 → 倍率>1、π 低 → 倍率<1（方向）
  7. 样本不足（< min_samples=5）→ estimate None（冷启动保护）
  8. snapshot 形状字段齐全；fail-open（喂 NaN/非数值不抛）

测试约定：纯 stdlib 单测（不依赖 torch）。默认关路径显式
monkeypatch.delenv("LMS_PRECISION_LEARN") 防 env 泄漏；开路径用显式参数
或 monkeypatch.setenv。
"""

import math
import os
import sys

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest

from core.doubt.precision_adapt import (
    PrecisionLearnState, precision_learn_enabled,
)


def _feed(st, surprises):
    """批量喂入 surprise 序列。"""
    for s in surprises:
        st.observe(s)
    return st


def _low_var_seq(n, base=10.0, jitter=0.1):
    """低波动序列：base±jitter 交替（方差 = jitter²，非退化）。"""
    return [base + jitter if i % 2 else base - jitter for i in range(n)]


def _high_var_seq(n, base=10.0, amp=10.0):
    """高波动序列：base±amp 交替（方差 = amp²）。"""
    return [base + amp if i % 2 else base - amp for i in range(n)]


# ============================================================
# 1. 治理开关默认关（零参与，行为与开关引入前完全一致）
# ============================================================

class TestSwitchDefaultOff:
    def test_default_off_estimate_none(self, monkeypatch):
        """LMS_PRECISION_LEARN 未设置 → 开关关：estimate 恒 None。"""
        monkeypatch.delenv("LMS_PRECISION_LEARN", raising=False)
        st = PrecisionLearnState()
        _feed(st, [5.0] * 10)
        assert st.enabled is False
        assert st.estimate() is None

    def test_default_off_lr_multiplier_unchanged(self, monkeypatch):
        """开关关 → lr_multiplier 原样返回（行为与现状完全一致）。"""
        monkeypatch.delenv("LMS_PRECISION_LEARN", raising=False)
        st = PrecisionLearnState()
        assert st.lr_multiplier(0.01) == 0.01
        assert st.lr_multiplier(0.5) == 0.5

    def test_default_off_snapshot_enabled_false_zero_samples(self, monkeypatch):
        """开关关 → snapshot enabled=False；observe 可调用但不记录。"""
        monkeypatch.delenv("LMS_PRECISION_LEARN", raising=False)
        st = PrecisionLearnState()
        _feed(st, [5.0] * 10)  # observe 仍可被调用（零参与）
        snap = st.snapshot()
        assert snap['enabled'] is False
        assert snap['pi_estimate'] is None
        assert snap['variance'] is None
        assert snap['samples'] == 0  # 零参与：未记录任何观测
        assert snap['var_window'] == 200
        assert snap['ema'] == 0.02

    def test_explicit_false_overrides_env(self, monkeypatch):
        """显式 enabled=False 优先于 env 开。"""
        monkeypatch.setenv("LMS_PRECISION_LEARN", "1")
        st = PrecisionLearnState(enabled=False)
        assert st.enabled is False
        assert precision_learn_enabled(False) is False


# ============================================================
# 2. 开关解析（显式参数 / env）+ env 参数族
# ============================================================

class TestSwitchEnable:
    def test_explicit_enable_overrides_env_off(self, monkeypatch):
        """显式 enabled=True 优先于 env 关。"""
        monkeypatch.setenv("LMS_PRECISION_LEARN", "0")
        st = PrecisionLearnState(enabled=True)
        assert st.enabled is True
        assert precision_learn_enabled(True) is True

    def test_env_enable(self, monkeypatch):
        """LMS_PRECISION_LEARN=1 → 开关开（1/true/yes/on 均视为开）。"""
        for raw in ("1", "true", "yes", "on", "TRUE"):
            monkeypatch.setenv("LMS_PRECISION_LEARN", raw)
            assert precision_learn_enabled(None) is True

    def test_env_params_authoritative(self, monkeypatch):
        """LMS_PRECISION_VAR_* 参数族可配（env 为唯一权威）。"""
        monkeypatch.setenv("LMS_PRECISION_LEARN", "1")
        monkeypatch.setenv("LMS_PRECISION_VAR_WINDOW", "50")
        monkeypatch.setenv("LMS_PRECISION_VAR_EMA", "0.5")
        monkeypatch.setenv("LMS_PRECISION_VAR_MIN", "0.1")
        monkeypatch.setenv("LMS_PRECISION_VAR_MAX", "10.0")
        st = PrecisionLearnState()
        assert st.enabled is True
        assert st.var_window == 50
        assert st.ema == 0.5
        assert st.pi_min == 0.1
        assert st.pi_max == 10.0

    def test_explicit_params_override_env(self, monkeypatch):
        """显式构造参数 > env（参数级覆盖）。"""
        monkeypatch.setenv("LMS_PRECISION_LEARN", "1")
        monkeypatch.setenv("LMS_PRECISION_VAR_WINDOW", "50")
        st = PrecisionLearnState(var_window=123, ema=0.1)
        assert st.var_window == 123
        assert st.ema == 0.1


# ============================================================
# 3. π 估计（常数序列 / 方向 / 钳制 / 窗口）
# ============================================================

class TestPiEstimate:
    def test_constant_sequence_pi_min_no_nan(self, monkeypatch):
        """全等序列 → var→0 → π 钳到 MIN（不除零、无 NaN）。

        源规格原文：「var 过小（如全等序列）→ 用 MIN 保护」——退化方差
        不做 1/0，保守低 precision。
        """
        monkeypatch.setenv("LMS_PRECISION_LEARN", "1")
        st = _feed(PrecisionLearnState(), [5.0] * 250)
        pi = st.estimate()
        assert pi == pytest.approx(st.pi_min)  # 0.05
        assert math.isfinite(pi)
        assert st.snapshot()['variance'] == pytest.approx(0.0, abs=1e-9)

    def test_high_vs_low_volatility_direction(self, monkeypatch):
        """高波动 → π 低；低波动（非退化）→ π 高（方向断言）。"""
        monkeypatch.setenv("LMS_PRECISION_LEARN", "1")
        st_high = _feed(PrecisionLearnState(), _high_var_seq(250))
        st_low = _feed(PrecisionLearnState(), _low_var_seq(250))
        pi_high, pi_low = st_high.estimate(), st_low.estimate()
        assert pi_low > pi_high
        assert pi_low == pytest.approx(st_low.pi_max)    # 低波动 → 钳 MAX
        assert pi_high == pytest.approx(st_high.pi_min)  # 高波动 → 钳 MIN

    def test_clamp_bounds_never_exceeded(self, monkeypatch):
        """任意输入序列：π ∈ [MIN, MAX] 恒成立（防除零/爆值）。"""
        monkeypatch.setenv("LMS_PRECISION_LEARN", "1")
        for seq in (_high_var_seq(300), _low_var_seq(300), [10.0] * 300,
                    _low_var_seq(300, jitter=1e-9)):  # 近全等（非退化）→ 爆值路径
            st = _feed(PrecisionLearnState(), seq)
            pi = st.estimate()
            assert pi is not None
            assert st.pi_min - 1e-12 <= pi <= st.pi_max + 1e-12

    def test_window_rollover_excludes_old_samples(self, monkeypatch):
        """滑动窗口生效：窗口外旧样本不参与（尾窗长度限制）。

        用 ema=1.0（无低通，估计 = 1/尾窗原始方差）隔离窗口效应，排除
        EMA 滞后干扰。
        """
        monkeypatch.setenv("LMS_PRECISION_LEARN", "1")
        # 300 轮高波动灌满窗口 → 再 50 轮低波动：尾窗 = 150 高 + 50 低（混合）
        st_mixed = _feed(PrecisionLearnState(ema=1.0), _high_var_seq(300))
        _feed(st_mixed, _low_var_seq(50))
        pi_mixed = st_mixed.estimate()
        assert pi_mixed < st_mixed.pi_max  # 旧高波动样本仍在窗口内 → 未到 MAX
        # 同样 300 高波动 → 再 250 低波动：尾窗 200 条全部为低波动（旧样本排除）
        st_pure = _feed(PrecisionLearnState(ema=1.0), _high_var_seq(300))
        _feed(st_pure, _low_var_seq(250))
        pi_pure = st_pure.estimate()
        assert pi_pure == pytest.approx(st_pure.pi_max)  # 窗口完全滚动 → 钳 MAX
        assert pi_pure > pi_mixed


# ============================================================
# 4. EMA 低通（突变后渐进收敛，不跳变）
# ============================================================

class TestEmaLowPass:
    def test_regime_switch_converges_gradually(self, monkeypatch):
        """低波动 → 高波动突变：π 渐进收敛（不瞬间跳变到新稳态）。"""
        monkeypatch.setenv("LMS_PRECISION_LEARN", "1")
        st = _feed(PrecisionLearnState(), _low_var_seq(250))
        pi_before = st.estimate()
        assert pi_before == pytest.approx(st.pi_max)
        # 突变：进入高波动；逐检查点采样（累计 10/30/60 轮）
        _feed(st, _high_var_seq(10))
        pi_10 = st.estimate()
        _feed(st, _high_var_seq(20))
        pi_30 = st.estimate()
        _feed(st, _high_var_seq(30))
        pi_60 = st.estimate()
        # 不跳变：10 轮后仍远离新稳态（未直接落到 MIN）
        assert pi_10 > st.pi_min
        assert pi_10 < pi_before
        # 渐进收敛方向：单调下降
        assert pi_10 > pi_30 > pi_60
        assert pi_60 >= st.pi_min


# ============================================================
# 5. lr_multiplier（π 加权学习率倍率）
# ============================================================

class TestLrMultiplier:
    def test_off_returns_prev_lr(self, monkeypatch):
        """开关关 → 原样返回 prev_lr。"""
        monkeypatch.delenv("LMS_PRECISION_LEARN", raising=False)
        st = PrecisionLearnState()
        assert st.lr_multiplier(0.01) == 0.01
        assert st.lr_multiplier(1e-5) == 1e-5

    def test_cold_returns_prev_lr(self, monkeypatch):
        """开关开但样本不足（冷启动）→ 原样返回 prev_lr（不给倍率）。"""
        monkeypatch.setenv("LMS_PRECISION_LEARN", "1")
        st = PrecisionLearnState(min_samples=5)
        _feed(st, _low_var_seq(4))
        assert st.estimate() is None
        assert st.lr_multiplier(0.02) == 0.02

    def test_high_pi_amplifies_low_pi_attenuates(self, monkeypatch):
        """开关开：π 高 → 倍率>1；π 低 → 倍率<1（自然梯度方向）。"""
        monkeypatch.setenv("LMS_PRECISION_LEARN", "1")
        st_high_pi = _feed(PrecisionLearnState(), _low_var_seq(250))
        st_low_pi = _feed(PrecisionLearnState(), _high_var_seq(250))
        base = 0.01
        mult_high = st_high_pi.lr_multiplier(base)
        mult_low = st_low_pi.lr_multiplier(base)
        assert mult_high > base  # π 高 → 倍率 > 1
        assert mult_low < base   # π 低 → 倍率 < 1
        # 精确关系：effective_lr = prev_lr × π̄
        assert mult_high == pytest.approx(base * st_high_pi.pi_max)
        assert mult_low == pytest.approx(base * st_low_pi.pi_min)


# ============================================================
# 6. 冷启动保护 + snapshot 形状
# ============================================================

class TestColdStartAndSnapshot:
    def test_cold_start_min_samples(self, monkeypatch):
        """样本不足（< min_samples=5）→ estimate None；足量后给出估计。"""
        monkeypatch.setenv("LMS_PRECISION_LEARN", "1")
        st = PrecisionLearnState(min_samples=5)
        _feed(st, _low_var_seq(4))
        assert st.estimate() is None
        _feed(st, _low_var_seq(1))
        assert st.estimate() is not None

    def test_snapshot_shape_full(self, monkeypatch):
        """snapshot 形状字段齐全（开关开、样本充足）。"""
        monkeypatch.setenv("LMS_PRECISION_LEARN", "1")
        st = _feed(PrecisionLearnState(), _low_var_seq(60))
        snap = st.snapshot()
        assert set(snap) == {'enabled', 'var_window', 'ema',
                             'pi_estimate', 'variance', 'samples'}
        assert snap['enabled'] is True
        assert snap['var_window'] == 200
        assert snap['ema'] == 0.02
        assert snap['pi_estimate'] is not None
        assert snap['variance'] is not None
        assert snap['samples'] == 60

    def test_snapshot_cold_fields(self, monkeypatch):
        """冷启动：pi_estimate=None；samples 准确；方差可有值（样本 ≥2）。"""
        monkeypatch.setenv("LMS_PRECISION_LEARN", "1")
        st = PrecisionLearnState()
        _feed(st, [10.0, 12.0])  # 2 样本：方差已定义但样本不足
        snap = st.snapshot()
        assert snap['pi_estimate'] is None
        assert snap['samples'] == 2
        assert snap['variance'] == pytest.approx(1.0)  # var({10,12}) = 1


# ============================================================
# 7. fail-open（异常输入不抛、不污染）
# ============================================================

class TestFailOpen:
    def test_nan_inf_no_raise_no_pollution(self, monkeypatch):
        """喂 NaN/inf 不抛，且不污染方差估计（样本不计入）。"""
        monkeypatch.setenv("LMS_PRECISION_LEARN", "1")
        st = PrecisionLearnState()
        st.observe(float('nan'))
        st.observe(float('inf'))
        st.observe(float('-inf'))
        assert st.snapshot()['samples'] == 0
        _feed(st, _low_var_seq(60))  # 之后正常观测不受影响
        assert st.estimate() == pytest.approx(st.pi_max)

    def test_non_numeric_no_raise(self, monkeypatch):
        """非数值输入（字符串/None/列表）不抛（fail-open 静默丢弃）。"""
        monkeypatch.setenv("LMS_PRECISION_LEARN", "1")
        st = PrecisionLearnState()
        st.observe("abc")
        st.observe(None)
        st.observe([1.0])
        assert st.snapshot()['samples'] == 0
        st.observe(10.0)
        assert st.snapshot()['samples'] == 1

    def test_disabled_observe_still_callable(self, monkeypatch):
        """开关关：observe 仍可调用（零参与，不记录、不抛）。"""
        monkeypatch.setenv("LMS_PRECISION_LEARN", "0")
        st = PrecisionLearnState()
        st.observe(5.0)
        st.observe(float('nan'))
        st.observe("garbage")
        assert st.snapshot()['samples'] == 0
        assert st.estimate() is None

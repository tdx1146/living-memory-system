"""allostatic J 滑动设定点（论文机制 A）单元测试。

被测模块: runtime/allostatic_j.py

覆盖:
  1. 治理开关解析（默认关 / env 开 / 显式参数）      (TestSwitch)
  2. 冷启动：样本不足不动作                          (TestColdStart)
  3. 三态重锚方向：饱和降 / 崩塌升 / 动态稳（任务书单测核心）(TestReanchorDirection)
  4. surprise 带漂移（Mehra 1970 innovation）        (TestSurpriseBand)
  5. 设定点护栏 [j_min, j_max]                       (TestBounds)
  6. σ 统计计算 + NaN/空中性                         (TestSigmaStats)
  7. 全链路：controller 驱动 attractor 钳制（动态设定点）(TestAttractorIntegration)
  8. 关时零参与                                       (TestDisabled)

设计约束:
  - 不改动其他源文件；仅新增本测试文件
  - 与既有风格一致：纯逻辑断言 + 最小 torch 依赖
"""

import os

import pytest
import torch

from runtime.allostatic_j import (
    AllostaticJController,
    SigmaStats,
    allostatic_j_enabled,
    compute_sigma_stats,
)


# ================================================================== #
#  1. 治理开关
# ================================================================== #

class TestSwitch:
    """治理开关解析：默认 0=关（回滚干净），1 开。"""

    def test_default_off(self, monkeypatch):
        monkeypatch.delenv("LMS_J_ALLOSTATIC", raising=False)
        assert allostatic_j_enabled() is False

    def test_env_on(self, monkeypatch):
        monkeypatch.setenv("LMS_J_ALLOSTATIC", "1")
        assert allostatic_j_enabled() is True

    def test_env_off_values(self, monkeypatch):
        for v in ("0", "false", "no", "off"):
            monkeypatch.setenv("LMS_J_ALLOSTATIC", v)
            assert allostatic_j_enabled() is False

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("LMS_J_ALLOSTATIC", "1")
        assert allostatic_j_enabled(False) is False

    def test_disabled_controller_zero_participation(self):
        ctl = AllostaticJController(enabled=False, init_target=7.0)
        assert ctl.update(123.0, SigmaStats(frac_gt0_9=1.0, act05=0)) == 7.0
        assert ctl.j_target == 7.0
        assert ctl.snapshot() == {"enabled": False}
        assert len(ctl.surprise_window) == 0  # 连观测都不登记（零参与）


# ================================================================== #
#  2. 冷启动
# ================================================================== #

class TestColdStart:
    """窗口样本不足 min_samples → 保持初始设定点，不动作。"""

    def test_cold_start_keeps_init(self):
        ctl = AllostaticJController(
            enabled=True, init_target=7.0, min_samples=30)
        # 即使 σ 饱和信号在场，冷启动也不重锚
        for _ in range(29):
            t = ctl.update(5.0, SigmaStats(frac_gt0_9=1.0, act05=0))
            assert t == 7.0
        assert ctl.j_target == 7.0
        assert len(ctl.surprise_window) == 29

    def test_warm_after_min_samples(self):
        ctl = AllostaticJController(
            enabled=True, init_target=7.0, min_samples=30)
        for _ in range(30):
            ctl.update(5.0, SigmaStats(frac_gt0_9=0.0, act05=256))
        # 第 31 轮：饱和信号 → 下降
        t = ctl.update(5.0, SigmaStats(frac_gt0_9=1.0, act05=256))
        assert t == pytest.approx(6.5)


# ================================================================== #
#  3. 三态重锚方向（任务书单测核心：饱和降/崩塌升/动态稳）
# ================================================================== #

class TestReanchorDirection:
    """构造饱和/动态/崩塌三态 → J 重估行为正确。"""

    def _warm(self, ctl, n=35, surprise=5.0):
        """用中性 σ + 稳定 surprise 暖机过冷启动（不触发重锚）。"""
        neutral = SigmaStats(frac_gt0_9=0.0, act05=1000000)
        for _ in range(n):
            ctl.update(surprise, neutral)

    def test_saturation_decreases_target(self):
        """饱和态（frac_gt0.9 高）→ 设定点下降。"""
        ctl = AllostaticJController(
            enabled=True, init_target=7.0, min_samples=30,
            step=0.5, sat_frac=0.9)
        self._warm(ctl)
        t = ctl.update(5.0, SigmaStats(frac_gt0_9=1.0, act05=256))
        assert t == pytest.approx(6.5)
        # 持续饱和 → 继续下降
        t = ctl.update(5.0, SigmaStats(frac_gt0_9=0.95, act05=256))
        assert t == pytest.approx(6.0)

    def test_collapse_increases_target(self):
        """崩塌态（act05 极低）→ 设定点上升。"""
        ctl = AllostaticJController(
            enabled=True, init_target=7.0, min_samples=30,
            step=0.5, col_act=5)
        self._warm(ctl)
        t = ctl.update(5.0, SigmaStats(frac_gt0_9=0.0, act05=2))
        assert t == pytest.approx(7.5)
        # 持续崩塌 → 继续上升
        t = ctl.update(5.0, SigmaStats(frac_gt0_9=0.0, act05=1))
        assert t == pytest.approx(8.0)

    def test_dynamic_stable(self):
        """动态态（健康波动 + 中性 σ）→ 设定点稳定不动。"""
        ctl = AllostaticJController(
            enabled=True, init_target=7.0, min_samples=30,
            k=2.0, persist=5, step=0.5)
        neutral = SigmaStats(frac_gt0_9=0.0, act05=1000000)
        # 长时间健康波动（确定性正弦：均值 5.0 ± 0.05，|z|max≈1.4 ≪ k=2）
        import math
        for i in range(300):
            s = 5.0 + 0.05 * math.sin(i * 0.3)
            t = ctl.update(s, neutral)
            assert t == pytest.approx(7.0)
        assert ctl.j_target == pytest.approx(7.0)
        # 越界触发未发生（无重锚事件）
        assert len(ctl.events) == 0

    def test_saturation_beats_band(self):
        """σ 硬信号优先级高于 surprise 带（饱和在场时即使 z 在带内也降）。"""
        ctl = AllostaticJController(
            enabled=True, init_target=7.0, min_samples=30,
            k=2.0, persist=5, step=0.5)
        self._warm(ctl)
        # z≈0（带内）但 σ 饱和 → 仍下降
        t = ctl.update(5.0, SigmaStats(frac_gt0_9=0.95, act05=256))
        assert t == pytest.approx(6.5)


# ================================================================== #
#  4. surprise 带漂移（Mehra 1970 innovation 法）
# ================================================================== #

class TestSurpriseBand:
    """持续越带（±k, persist 轮）→ 重锚；带内 → 稳。"""

    def test_persistent_high_surprise_decreases(self):
        ctl = AllostaticJController(
            enabled=True, init_target=7.0, min_samples=30,
            k=2.0, persist=5, step=0.5)
        neutral = SigmaStats(frac_gt0_9=0.0, act05=1000000)
        for _ in range(35):
            ctl.update(5.0, neutral)
        # 5 轮持续高惊讶（>> 窗口均值）→ 越过 persist → 下降
        for i in range(5):
            t = ctl.update(50.0, neutral)
            if i < 4:
                assert t == pytest.approx(7.0)  # 未达 persist，不动
            else:
                assert t == pytest.approx(6.5)  # 第 5 轮触发

    def test_persistent_low_surprise_increases(self):
        ctl = AllostaticJController(
            enabled=True, init_target=7.0, min_samples=30,
            k=2.0, persist=5, step=0.5)
        neutral = SigmaStats(frac_gt0_9=0.0, act05=1000000)
        for _ in range(35):
            ctl.update(5.0, neutral)
        for i in range(5):
            t = ctl.update(0.01, neutral)
            if i < 4:
                assert t == pytest.approx(7.0)
            else:
                assert t == pytest.approx(7.5)

    def test_single_spike_no_reanchor(self):
        """单轮越带不重锚（persist 防单轮抖动）。"""
        ctl = AllostaticJController(
            enabled=True, init_target=7.0, min_samples=30,
            k=2.0, persist=5, step=0.5)
        neutral = SigmaStats(frac_gt0_9=0.0, act05=1000000)
        for _ in range(35):
            ctl.update(5.0, neutral)
        t = ctl.update(50.0, neutral)  # 单轮尖峰
        assert t == pytest.approx(7.0)
        t = ctl.update(5.0, neutral)  # 恢复正常 → streak 清零
        assert t == pytest.approx(7.0)


# ================================================================== #
#  5. 设定点护栏
# ================================================================== #

class TestBounds:
    """设定点永远钳在 [j_min, j_max] 内。"""

    def test_never_below_min(self):
        ctl = AllostaticJController(
            enabled=True, init_target=3.5, min_samples=1,
            j_min=3.0, j_max=40.0, step=0.5)
        for _ in range(10):  # 持续饱和
            ctl.update(5.0, SigmaStats(frac_gt0_9=1.0, act05=256))
        assert ctl.j_target == pytest.approx(3.0)

    def test_never_above_max(self):
        ctl = AllostaticJController(
            enabled=True, init_target=39.5, min_samples=1,
            j_min=3.0, j_max=40.0, step=0.5)
        for _ in range(10):  # 持续崩塌
            ctl.update(5.0, SigmaStats(frac_gt0_9=0.0, act05=0))
        assert ctl.j_target == pytest.approx(40.0)

    def test_init_clamped_to_range(self):
        ctl = AllostaticJController(
            enabled=True, init_target=100.0, min_samples=1,
            j_min=3.0, j_max=40.0)
        assert ctl.j_target == pytest.approx(40.0)
        ctl2 = AllostaticJController(
            enabled=True, init_target=0.5, min_samples=1,
            j_min=3.0, j_max=40.0)
        assert ctl2.j_target == pytest.approx(3.0)


# ================================================================== #
#  6. σ 统计计算
# ================================================================== #

class TestSigmaStats:
    """compute_sigma_stats：饱和/崩塌计数 + NaN/空中性。"""

    def test_counts(self):
        s = torch.tensor([0.95, 0.91, 0.1, 0.0, -0.99, 0.05, 0.06])
        st = compute_sigma_stats(s)
        assert st.frac_gt0_9 == pytest.approx(3 / 7)  # 0.95/0.91/0.99
        assert st.act05 == 5  # 0.95/0.91/0.1/0.99/0.06
        assert st.valid is True

    def test_nan_neutral(self):
        st = compute_sigma_stats(torch.tensor([float("nan"), 0.5]))
        assert st.valid is False
        # 无效统计不触发 σ 信号（仅 surprise 带生效）——中性不误重锚
        ctl = AllostaticJController(
            enabled=True, init_target=7.0, min_samples=1,
            k=2.0, persist=5)
        t = ctl.update(5.0, st)
        assert t == pytest.approx(7.0)

    def test_empty_neutral(self):
        st = compute_sigma_stats(torch.zeros(0))
        assert st.valid is False

    def test_invalid_keeps_last_valid_signal_off(self):
        """无效 σ 数据轮：σ 信号不参与（只有带漂移能触发）。"""
        ctl = AllostaticJController(
            enabled=True, init_target=7.0, min_samples=1,
            k=2.0, persist=5, step=0.5)
        # 先饱和一轮（有效）→ 降
        t = ctl.update(5.0, SigmaStats(frac_gt0_9=1.0, act05=256))
        assert t == pytest.approx(6.5)
        # 再来一轮 NaN（无效）→ 不因 σ 再降（surprise 带内）
        t = ctl.update(5.0, SigmaStats(frac_gt0_9=0.0, act05=0, valid=False))
        assert t == pytest.approx(6.5)


# ================================================================== #
#  7. 全链路：controller 驱动 attractor 钳制（动态设定点）
# ================================================================== #

class TestAttractorIntegration:
    """loop 契约：update → 写 attractor.j_target_norm → learn 按动态值钳制。"""

    @staticmethod
    def _strong_net(seed=42):
        """构造 J 范数远超设定点的吸引子网络（模拟长期学习后的强 J）。"""
        from core.hippocampus.attractor import AttractorNetwork
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=seed)
        net.J = torch.randn(32, 32) * 3.0
        net.J = (net.J + net.J.T) / 2
        net.J.fill_diagonal_(0)
        return net

    def test_learn_clamps_to_rising_target(self):
        """崩塌升：设定点 7.0→7.5，learn 按 7.5 钳制（动态而非固定 7）。"""
        net = self._strong_net()
        ctl = AllostaticJController(
            enabled=True, init_target=7.0, min_samples=1,
            persist=1, step=0.5)
        # 崩塌信号 → 设定点上升
        target = ctl.update(0.1, SigmaStats(frac_gt0_9=0.0, act05=2))
        assert target == pytest.approx(7.5)
        net.j_target_norm = target
        act = net.infer(torch.randn(16) * 0.5, torch.ones(16))
        net.learn(act, torch.randn(16) * 0.5, learning_rate=0.1)
        j_norm = float(torch.norm(net.J, p="fro").item())
        assert j_norm <= 7.5 * (1 + 1e-3), f"‖J‖_F={j_norm} 未按动态设定点 7.5 钳制"

    def test_learn_clamps_to_falling_target(self):
        """饱和降：设定点 7.0→6.5，learn 按 6.5 钳制。"""
        net = self._strong_net()
        ctl = AllostaticJController(
            enabled=True, init_target=7.0, min_samples=1,
            persist=1, step=0.5)
        target = ctl.update(5.0, SigmaStats(frac_gt0_9=1.0, act05=32))
        assert target == pytest.approx(6.5)
        net.j_target_norm = target
        act = net.infer(torch.randn(16) * 0.5, torch.ones(16))
        net.learn(act, torch.randn(16) * 0.5, learning_rate=0.1)
        j_norm = float(torch.norm(net.J, p="fro").item())
        assert j_norm <= 6.5 * (1 + 1e-3), f"‖J‖_F={j_norm} 未按动态设定点 6.5 钳制"

    def test_snapshot_exposes_dynamic_sequence(self):
        """灵魂指标②/③：j_history 动态（非固定）+ events 越界触发可观测。"""
        ctl = AllostaticJController(
            enabled=True, init_target=7.0, min_samples=1,
            persist=1, step=0.5)
        for _ in range(3):
            ctl.update(5.0, SigmaStats(frac_gt0_9=1.0, act05=32))  # 饱和 ×3
        snap = ctl.snapshot(turn_count=42)
        assert snap["enabled"] is True
        assert snap["j_target"] == pytest.approx(5.5)  # 7.0 → 6.5 → 6.0 → 5.5
        assert len(snap["j_history"]) == 3
        assert snap["j_history"] == [6.5, 6.0, 5.5]  # 非固定序列（指标②）
        assert len(snap["events"]) == 3
        assert snap["events"][-1]["reason"] == "saturation"  # 越界可观测（指标③）
        assert snap["events"][-1]["ts_turn"] == 42

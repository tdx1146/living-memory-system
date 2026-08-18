"""
测试 S4 C 条件触发（EWC 保护）—— ABC 操作规划 §S4
====================================================

覆盖（≥12 例）：
  1. 开关默认关 → 零参与（penalty=0、Fisher 不更新、snapshot enabled=False）
  2. Fisher EMA 积累方向（健康窗口喂梯度 → 非零；未积累 → None）
  3. 健康窗口门控（healthy=False 轮次不更新 Fisher——崩塌期保护）
  4. 罚项数值正确性（手算小案例断言 λ·ΣFᵢ(θ̂ᵢ−θ̂*ᵢ)²，θ̂=θ/‖θ‖_F）
  5. 缩放不变性（A 重锚豁免红线：P(cθ)=P(θ)，c=2/0.5/10）
  6. 罚项对方向偏离敏感（同方向 → 0；偏离 → 增大）
  7. 开关关时 AttractorNetwork.learn 输出与开关引入前一致（逐字节）
  8. 开关开时 learn 正常跑通（不崩、数值有界、保护生效）
  9. snapshot 形状；fail-open（NaN/零范数/非法梯度 → 中性零参与）

纪律：只验证 S4 新行为 + 既有默认路径零回归；不触碰其他模块。
"""

import pytest
import torch

from core.continual.ewc import EwcPenalty, FisherAccumulator, ewc_enabled
from core.hippocampus.attractor import AttractorNetwork
from core.types import Activation


@pytest.fixture(autouse=True)
def _ewc_env_clean(monkeypatch):
    """隔离 LMS_EWC_ENABLE：默认移除（=关）；需要 env 开的测试自行 setenv。

    保证「开关默认关」断言不依赖外部环境（conftest 不设此变量，此处防御）。
    """
    monkeypatch.delenv("LMS_EWC_ENABLE", raising=False)


def _make_net(seed=42, ewc=None):
    """小规模吸引子网络（32 节点 / 16 维输入），构造参数与既有测试一致。"""
    return AttractorNetwork(num_nodes=32, input_dim=16, seed=seed, ewc=ewc)


def _healthy_activation(n=32, scale=0.5):
    """健康激活态（σ 统计：非崩塌 act05>5、非饱和 frac_gt0_9<0.9、valid）。"""
    return Activation(state=torch.randn(n) * scale, entropy=1.0, surprise=0.5)


# ======================================================================= #
#  FisherAccumulator（单元）
# ======================================================================= #


class TestFisherAccumulator:
    """对角 Fisher 近似：健康窗口 EMA 累积 + 崩塌期门控。"""

    def test_fisher_none_before_accumulation(self):
        """未积累 → fisher() 返回 None（保护未就绪，零参与）。"""
        acc = FisherAccumulator((2, 2), "cpu", enabled=True)
        assert acc.fisher() is None
        assert acc.reference() is None

    def test_fisher_single_healthy_update_squared_grad(self):
        """单次健康更新：F = g²（Kirkpatrick 对角 Fisher 的标准对象）。"""
        acc = FisherAccumulator((2, 2), "cpu", enabled=True)
        g = torch.full((2, 2), 3.0)
        acc.update(weights=torch.ones(2, 2), grads=g, healthy=True)
        F = acc.fisher()
        assert F is not None
        assert torch.allclose(F, torch.full((2, 2), 9.0))
        assert acc.count == 1

    def test_fisher_healthy_gate_skips_unhealthy(self):
        """healthy=False（崩塌/过渡态）→ 不更新 Fisher（关键约束）。"""
        acc = FisherAccumulator((2, 2), "cpu", enabled=True)
        g = torch.full((2, 2), 2.0)
        acc.update(weights=torch.ones(2, 2), grads=g, healthy=False)
        acc.update(weights=torch.ones(2, 2), grads=g, healthy=False)
        assert acc.fisher() is None          # 崩塌期零积累
        assert acc.count == 0
        assert acc.skipped == 2              # 可观测：被门控跳过的轮次
        # 恢复健康后照常积累（保护不被坏工作点污染）
        acc.update(weights=torch.ones(2, 2), grads=g, healthy=True)
        assert torch.allclose(acc.fisher(), torch.full((2, 2), 4.0))
        assert acc.count == 1

    def test_fisher_window_running_mean(self):
        """健康窗口内（count < window）：运行平均（无偏估计）。"""
        acc = FisherAccumulator((2, 2), "cpu", enabled=True, window=5)
        g = torch.full((2, 2), 2.0)
        for _ in range(3):
            acc.update(weights=torch.ones(2, 2), grads=g, healthy=True)
        assert torch.allclose(acc.fisher(), torch.full((2, 2), 4.0))
        assert acc.count == 3

    def test_fisher_ema_after_window(self):
        """窗口满后（count ≥ window）：EMA 追踪，旧样本指数衰减。

        window=1, ema=0.9：g=[2,0] → F 依次 4 → 0.1·4+0.9·0 = 0.4
        （运行平均会得 2.0——EMA 的衰减行为与均值可区分）。
        """
        acc = FisherAccumulator((2, 2), "cpu", enabled=True, window=1, ema=0.9)
        acc.update(weights=torch.ones(2, 2), grads=torch.full((2, 2), 2.0),
                   healthy=True)
        acc.update(weights=torch.ones(2, 2), grads=torch.zeros(2, 2),
                   healthy=True)
        assert torch.allclose(acc.fisher(), torch.full((2, 2), 0.4))
        assert acc.count == 2

    def test_fisher_disabled_noop(self):
        """开关关（enabled=False）→ update 全 no-op，Fisher 永不积累。"""
        acc = FisherAccumulator((2, 2), "cpu", enabled=False)
        acc.update(weights=torch.ones(2, 2), grads=torch.ones(2, 2),
                   healthy=True)
        assert acc.fisher() is None
        assert acc.count == 0 and acc.skipped == 0

    def test_fisher_reference_frozen_on_first_healthy(self):
        """保护锚点 θ* 在首个健康更新定格，后续 weights 不覆盖。"""
        acc = FisherAccumulator((2,), "cpu", enabled=True)
        w1 = torch.tensor([1.0, 2.0])
        w2 = torch.tensor([9.0, 9.0])
        acc.update(weights=w1, grads=torch.tensor([1.0, 1.0]), healthy=True)
        acc.update(weights=w2, grads=torch.tensor([1.0, 1.0]), healthy=True)
        assert torch.equal(acc.reference(), w1)

    def test_fisher_skips_nan_grads_fail_open(self):
        """非法梯度（NaN）→ fail-open 跳过，不污染已积累的 Fisher。"""
        acc = FisherAccumulator((2,), "cpu", enabled=True)
        acc.update(weights=torch.ones(2), grads=torch.tensor([1.0, 1.0]),
                   healthy=True)
        nan_grad = torch.tensor([float("nan"), 1.0])
        acc.update(weights=torch.ones(2), grads=nan_grad, healthy=True)
        assert torch.allclose(acc.fisher(), torch.ones(2))  # 未被污染
        assert acc.count == 1


# ======================================================================= #
#  EwcPenalty（单元）
# ======================================================================= #


class TestEwcPenalty:
    """缩放不变二次罚项：λ·ΣFᵢ(θ̂ᵢ−θ̂*ᵢ)²（θ̂=θ/‖θ‖_F，豁免 A 重锚缩放方向）。"""

    def test_penalty_zero_when_disabled(self):
        """开关默认关（env 未设）→ 零参与：penalty=0、梯度全 0、不更新 Fisher。"""
        pen = EwcPenalty(shape=(2,), device="cpu", enabled=None)  # env 关
        assert pen.enabled is False
        assert ewc_enabled() is False
        pen.update(weights=torch.ones(2), grads=torch.ones(2), healthy=True)
        assert pen.penalty(torch.ones(2)) == 0.0
        assert torch.equal(pen.gradient(torch.ones(2)), torch.zeros(2))
        assert pen.snapshot()["fisher_ready"] is False  # Fisher 未更新
        assert pen.snapshot()["enabled"] is False

    def test_penalty_hand_computed_small_case(self):
        """数值正确性：手算 2 参数小案例。

        θ*=[3,4]（‖θ*‖=5 → θ̂*=[0.6,0.8]），F=[1,1]，λ=0.1：
          θ=[1,0]（‖θ‖=1 → θ̂=[1,0]）：
          dev=[0.4,−0.8] → P = 0.1·(1·0.16 + 1·0.64) = 0.08
        """
        pen = EwcPenalty(shape=(2,), device="cpu", enabled=True, lam=0.1,
                         reference=torch.tensor([3.0, 4.0]))
        pen.update(weights=torch.tensor([3.0, 4.0]),
                   grads=torch.tensor([1.0, 1.0]), healthy=True)  # F=[1,1]
        P = pen.penalty(torch.tensor([1.0, 0.0]))
        # θ̂*=[0.6,0.8] 为浮点非精确值 → 允许 1e-6 舍入容差（相对误差 ~1e-7）
        assert abs(P - 0.08) < 1e-6

    def test_penalty_scale_invariant(self):
        """A 重锚豁免红线：整体标量缩放 c·θ 不改方向 → 罚项不变。"""
        pen = EwcPenalty(shape=(2,), device="cpu", enabled=True, lam=0.1,
                         reference=torch.tensor([3.0, 4.0]))
        pen.update(weights=torch.tensor([3.0, 4.0]),
                   grads=torch.tensor([1.0, 1.0]), healthy=True)
        theta = torch.tensor([1.0, 0.0])
        p1 = pen.penalty(theta)
        p2 = pen.penalty(2.0 * theta)     # c=2
        p3 = pen.penalty(0.5 * theta)     # c=0.5
        p4 = pen.penalty(10.0 * theta)    # c=10
        assert abs(p1 - p2) < 1e-12 and abs(p1 - p3) < 1e-12
        assert abs(p1 - p4) < 1e-12

    def test_penalty_sensitive_to_direction_deviation(self):
        """罚项对「方向偏离」敏感：同方向 → 0；偏离越大 → 罚项越大。"""
        pen = EwcPenalty(shape=(2,), device="cpu", enabled=True, lam=0.1,
                         reference=torch.tensor([3.0, 4.0]))
        pen.update(weights=torch.tensor([3.0, 4.0]),
                   grads=torch.tensor([1.0, 1.0]), healthy=True)
        on_dir = pen.penalty(torch.tensor([6.0, 8.0]))    # = 2·θ*，同方向
        small_off = pen.penalty(torch.tensor([3.0, 4.05]))  # 微小偏离
        far_off = pen.penalty(torch.tensor([1.0, 0.0]))   # 大偏离
        assert on_dir == 0.0
        assert small_off > 0.0
        assert far_off > small_off

    def test_gradient_matches_finite_difference(self):
        """梯度 = ∂P/∂θ 的数值验证（中心差分）；兼验径向自由。"""
        pen = EwcPenalty(shape=(2,), device="cpu", enabled=True, lam=0.1,
                         reference=torch.tensor([3.0, 4.0]))
        pen.update(weights=torch.tensor([3.0, 4.0]),
                   grads=torch.tensor([1.0, 1.0]), healthy=True)
        theta = torch.tensor([1.0, 0.3])
        g = pen.gradient(theta)
        # 径向分量 ⟨∇P, θ̂⟩ ≈ 0（EWC 力永不沿整体缩放方向 → 不抵抗 A 重锚）
        th_hat = theta / torch.linalg.vector_norm(theta)
        assert abs(torch.dot(g, th_hat).item()) < 1e-6
        # 中心差分校验各分量
        h = 1e-4
        for i in range(2):
            ep = torch.zeros(2); ep[i] = h
            num = (pen.penalty(theta + ep) - pen.penalty(theta - ep)) / (2 * h)
            assert abs(num - g[i].item()) < 1e-4

    def test_penalty_gradient_zero_before_fisher(self):
        """Fisher 未积累 / θ* 未定格 → 罚项 0、梯度全 0（保护未就绪零参与）。"""
        pen = EwcPenalty(shape=(2,), device="cpu", enabled=True)
        assert pen.penalty(torch.ones(2)) == 0.0
        assert torch.equal(pen.gradient(torch.ones(2)), torch.zeros(2))

    def test_penalty_fail_open(self):
        """fail-open：NaN 输入 / 零范数方向 / 异常 → 中性零参与，不抛异常。"""
        pen = EwcPenalty(shape=(2,), device="cpu", enabled=True, lam=0.1,
                         reference=torch.tensor([3.0, 4.0]))
        pen.update(weights=torch.tensor([3.0, 4.0]),
                   grads=torch.tensor([1.0, 1.0]), healthy=True)
        nan_theta = torch.tensor([float("nan"), 1.0])
        assert pen.penalty(nan_theta) == 0.0
        assert torch.equal(pen.gradient(nan_theta), torch.zeros(2))
        zero_theta = torch.zeros(2)   # 零范数 → 方向未定义 → 中性
        assert pen.penalty(zero_theta) == 0.0
        assert torch.equal(pen.gradient(zero_theta), torch.zeros(2))

    def test_snapshot_shape(self):
        """snapshot 固定键集合 {enabled, lambda, fisher_window, fisher_ready,
        penalty_last}，值随状态更新。"""
        pen = EwcPenalty(shape=(2,), device="cpu", enabled=True, lam=0.1,
                         window=500, ema=0.02,
                         reference=torch.tensor([3.0, 4.0]))
        snap0 = pen.snapshot()
        assert set(snap0.keys()) == {
            "enabled", "lambda", "fisher_window", "fisher_ready",
            "penalty_last"}
        assert snap0["enabled"] is True
        assert snap0["lambda"] == 0.1
        assert snap0["fisher_window"] == 500
        assert snap0["fisher_ready"] is False
        assert snap0["penalty_last"] == 0.0
        # 积累 + 计算罚项后
        pen.update(weights=torch.tensor([3.0, 4.0]),
                   grads=torch.tensor([1.0, 1.0]), healthy=True)
        pen.penalty(torch.tensor([1.0, 0.0]))
        snap1 = pen.snapshot()
        assert snap1["fisher_ready"] is True
        assert abs(snap1["penalty_last"] - 0.08) < 1e-6


# ======================================================================= #
#  AttractorNetwork 集成（learn 挂载点）
# ======================================================================= #


class TestAttractorEwcIntegration:
    """learn() 只做加法：开关关零参与（逐字节一致）；开关开挂载罚项。"""

    def test_learn_ewc_off_identical_default_path(self):
        """开关关（默认 None 与显式 enabled=False）→ 输出与开关引入前一致。

        同种子双网络（默认 vs 显式关闭 EWC），同一学习序列 → J 逐字节相同。
        """
        a = _make_net(seed=42)                      # 默认：env 关 → ewc None
        b = _make_net(seed=42, ewc=EwcPenalty(
            shape=(32, 32), device="cpu", enabled=False))
        assert a.ewc is None                        # 默认零参与
        assert b.ewc is not None and b.ewc.enabled is False
        act = _healthy_activation()
        for _ in range(3):
            a.learn(act, torch.randn(16), learning_rate=0.01)
            b.learn(act, torch.randn(16), learning_rate=0.01)
        assert torch.equal(a.J, b.J)                # 逐字节一致（零参与）

    def test_env_gate_wiring(self, monkeypatch):
        """env 门控：LMS_EWC_ENABLE=1 → 内部构造并启用；=0 → None。"""
        monkeypatch.setenv("LMS_EWC_ENABLE", "1")
        net_on = _make_net(seed=7)
        assert net_on.ewc is not None and net_on.ewc.enabled is True
        monkeypatch.setenv("LMS_EWC_ENABLE", "0")
        net_off = _make_net(seed=7)
        assert net_off.ewc is None

    def test_learn_ewc_on_runs_bounded_and_bites(self):
        """开关开 → learn 正常跑通（不崩、数值有界），且保护生效（J 轨迹
        与关闭路径分叉——罚项确实挂载到学习更新上）。"""
        base = _make_net(seed=42)
        on = _make_net(seed=42, ewc=EwcPenalty(
            shape=(32, 32), device="cpu", enabled=True, lam=0.1))
        act = _healthy_activation()
        for _ in range(4):
            base.learn(act, torch.randn(16), learning_rate=0.01)
            on.learn(act, torch.randn(16), learning_rate=0.01)
        # 不崩 + 数值有界（J 有限；范数钳制目标仍成立）
        assert bool(torch.isfinite(on.J).all())
        assert float(torch.norm(on.J, p="fro")) <= on.j_target_norm + 1e-6
        assert float(torch.norm(on.J, p="fro")) < 1.0
        # 保护生效：J 轨迹分叉（θ̂ 偏离 θ* 后罚项梯度非零）
        assert not torch.equal(base.J, on.J)
        snap = on.ewc.snapshot()
        assert snap["fisher_ready"] is True        # 健康窗口已积累 Fisher
        assert snap["penalty_last"] > 0.0          # 罚项确在动

    def test_learn_ewc_collapse_skips_fisher(self):
        """崩塌态（σ≈0 → act05=0 ≤ col_act）→ learn 内 healthy=False →
        Fisher 不积累（绝不在坏工作点计算）。"""
        net = _make_net(seed=42, ewc=EwcPenalty(
            shape=(32, 32), device="cpu", enabled=True, lam=0.1))
        collapse_act = Activation(state=torch.zeros(32), entropy=0.0,
                                  surprise=0.0)
        for _ in range(3):
            net.learn(collapse_act, torch.zeros(16), learning_rate=0.01)
        assert bool(torch.isfinite(net.J).all())   # 主路径不崩
        snap = net.ewc.snapshot()
        assert snap["fisher_ready"] is False       # 崩塌期零积累
        assert net.ewc._acc.count == 0
        assert net.ewc._acc.skipped == 3

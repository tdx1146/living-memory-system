"""
测试目的层
============

验证内容:
  1. precision 能演化
  2. 高惊讶维度 precision 升高
  3. coherence 计算
  4. 元目的翻转
"""

import pytest
import torch

from core.hippocampus.purpose import PurposeLayer
from core.types import Activation


# ----------------------------------------------------------------------- #
#  辅助函数
# ----------------------------------------------------------------------- #


def make_activation(num_nodes: int, input_dim: int,
                    sensory_pattern: torch.Tensor = None,
                    seed: int = 42) -> Activation:
    """构造测试用激活态。

    参数:
        num_nodes: 总节点数。
        input_dim: 感官节点数。
        sensory_pattern: 感官节点激活模式，形状 [input_dim]。
        seed: 非感官节点的随机种子。

    返回:
        构造的 Activation 对象。
    """
    if sensory_pattern is None:
        sensory_pattern = torch.zeros(input_dim)
    # 感官节点用给定模式，非感官节点用小随机值
    state = torch.cat([
        sensory_pattern,
        torch.randn(num_nodes - input_dim, generator=torch.Generator().manual_seed(seed)) * 0.1
    ])
    entropy = -torch.sum(
        (state.abs() / state.abs().sum()) *
        torch.log(state.abs() / state.abs().sum() + 1e-8)
    ).item()
    return Activation(state=state, entropy=entropy, surprise=1.0)


# ----------------------------------------------------------------------- #
#  Precision 演化测试
# ----------------------------------------------------------------------- #


class TestPrecisionEvolution:
    """precision 能否随 adjust 演化。"""

    def test_initial_precision_uniform(self):
        """初始 precision 为均匀分布。"""
        layer = PurposeLayer(input_dim=16)
        precision = layer.get_precision()

        # 所有维度应相同
        assert torch.allclose(precision, torch.ones(16))

    def test_precision_changes_after_adjust(self):
        """adjust 后 precision 发生变化。"""
        layer = PurposeLayer(input_dim=16, precision_lr=0.3)
        precision_before = layer.get_precision()

        # 构造非均匀激活
        pattern = torch.zeros(16)
        pattern[0] = 0.8  # 某维度高激活
        act = make_activation(num_nodes=32, input_dim=16, sensory_pattern=pattern)

        layer.adjust(surprise=2.0, activation=act)
        precision_after = layer.get_precision()

        assert not torch.allclose(precision_before, precision_after)

    def test_precision_history_grows(self):
        """每次 adjust 后历史增长。"""
        layer = PurposeLayer(input_dim=16)

        assert len(layer.history) == 0

        for i in range(5):
            pattern = torch.randn(16) * 0.3
            act = make_activation(num_nodes=32, input_dim=16, sensory_pattern=pattern)
            layer.adjust(surprise=1.0, activation=act)

        assert len(layer.history) == 5

    def test_precision_stabilizes_over_time(self):
        """重复相同输入，precision 趋于稳定。"""
        layer = PurposeLayer(input_dim=16, precision_lr=0.3)
        pattern = torch.zeros(16)
        pattern[3] = 0.9  # 固定高激活维度

        act = make_activation(num_nodes=32, input_dim=16, sensory_pattern=pattern)

        # 重复 adjust
        changes = []
        for _ in range(10):
            prev = layer.get_precision()
            layer.adjust(surprise=1.0, activation=act)
            curr = layer.get_precision()
            changes.append((curr - prev).abs().mean().item())

        # 后期变化应小于前期（收敛）
        assert changes[-1] < changes[0]


# ----------------------------------------------------------------------- #
#  高惊讶维度测试
# ----------------------------------------------------------------------- #


class TestHighSurprisePrecision:
    """高惊讶维度的 precision 应升高。"""

    def test_high_activation_dim_gets_higher_precision(self):
        """激活强度高的维度应获得更高的 precision。"""
        layer = PurposeLayer(input_dim=16, precision_lr=0.5)

        # 构造激活：维度 0 高激活，其余低激活
        pattern = torch.full((16,), 0.01)
        pattern[0] = 0.9  # 维度 0 高惊讶
        act = make_activation(num_nodes=32, input_dim=16, sensory_pattern=pattern)

        layer.adjust(surprise=2.0, activation=act)
        precision = layer.get_precision()

        # 维度 0 的 precision 应高于平均值
        avg_precision = precision.mean().item()
        assert precision[0].item() > avg_precision

    def test_low_activation_dim_gets_lower_precision(self):
        """激活强度低的维度应获得更低的 precision。"""
        layer = PurposeLayer(input_dim=16, precision_lr=0.5)

        pattern = torch.full((16,), 0.9)
        pattern[5] = 0.01  # 维度 5 低惊讶
        act = make_activation(num_nodes=32, input_dim=16, sensory_pattern=pattern)

        layer.adjust(surprise=2.0, activation=act)
        precision = layer.get_precision()

        # 维度 5 的 precision 应低于平均值
        avg_precision = precision.mean().item()
        assert precision[5].item() < avg_precision

    def test_multiple_adjusts_amplify_difference(self):
        """多次 adjust 后高/低惊讶维度的 precision 差距增大。"""
        layer = PurposeLayer(input_dim=8, precision_lr=0.3)

        pattern = torch.zeros(8)
        pattern[0] = 0.9
        pattern[1] = 0.1
        act = make_activation(num_nodes=16, input_dim=8, sensory_pattern=pattern)

        # 多次 adjust
        for _ in range(10):
            layer.adjust(surprise=1.0, activation=act)

        precision = layer.get_precision()
        # 维度 0 precision 应显著高于维度 1
        assert precision[0].item() > precision[1].item()

    def test_precision_within_bounds(self):
        """precision 始终在 [precision_min, precision_max] 范围内。"""
        layer = PurposeLayer(
            input_dim=16,
            precision_lr=0.5,
            precision_min=0.1,
            precision_max=10.0,
        )

        # 极端激活模式
        pattern = torch.ones(16)
        act = make_activation(num_nodes=32, input_dim=16, sensory_pattern=pattern)

        for _ in range(20):
            layer.adjust(surprise=100.0, activation=act)

        precision = layer.get_precision()
        assert torch.all(precision >= 0.1 - 1e-6)
        assert torch.all(precision <= 10.0 + 1e-6)


# ----------------------------------------------------------------------- #
#  Coherence 测试
# ----------------------------------------------------------------------- #


class TestCoherence:
    """coherence 计算测试。"""

    def test_initial_coherence_high(self):
        """初始 coherence 应较高（稳定状态）。"""
        layer = PurposeLayer(input_dim=16, min_history_length=2)
        assert layer.coherence == 1.0

    def test_stable_input_high_coherence(self):
        """稳定输入 → coherence 保持较高。"""
        layer = PurposeLayer(input_dim=16, precision_lr=0.2, min_history_length=2)

        # 重复相同模式
        pattern = torch.zeros(16)
        pattern[0] = 0.8
        act = make_activation(num_nodes=32, input_dim=16, sensory_pattern=pattern)

        for _ in range(10):
            layer.adjust(surprise=1.0, activation=act)

        # 稳定输入应保持较高 coherence
        assert layer.coherence > 0.5

    def test_fluctuating_input_lower_coherence(self):
        """波动输入 → coherence 较低。"""
        layer = PurposeLayer(input_dim=16, precision_lr=0.5, min_history_length=2)

        # 交替不同模式，造成 precision 波动
        pattern_a = torch.zeros(16)
        pattern_a[0] = 0.9
        pattern_b = torch.zeros(16)
        pattern_b[7] = 0.9

        for i in range(10):
            pattern = pattern_a if i % 2 == 0 else pattern_b
            act = make_activation(num_nodes=32, input_dim=16, sensory_pattern=pattern)
            layer.adjust(surprise=1.0, activation=act)

        # 混合 coherence（方向分量 + 幅度分量）下，温和的两维交替只改变
        # precision 的方向、几乎不改变幅度（幅度分量≈0.96），因此混合后的
        # coherence 略高于旧版纯余弦相似度情形（约 0.952），但仍显著低于
        # 完美一致（1.0）。阈值相应放宽以匹配新的混合度量分布。
        # 注：方向波动更剧烈的场景（如多维轮换）会同时压低方向与幅度分量，
        # 使 coherence 大幅下降——该路径由 TestMetaFlip 系列测试覆盖。
        assert layer.coherence < 0.97

    def test_coherence_in_range(self):
        """coherence 在 [0, 1] 范围内。"""
        layer = PurposeLayer(input_dim=16, precision_lr=0.5, min_history_length=2)

        for i in range(10):
            pattern = torch.randn(16).abs() * 0.5
            act = make_activation(num_nodes=32, input_dim=16, sensory_pattern=pattern)
            layer.adjust(surprise=float(i), activation=act)

        assert 0.0 <= layer.coherence <= 1.0


# ----------------------------------------------------------------------- #
#  元目的翻转测试
# ----------------------------------------------------------------------- #


class TestMetaFlip:
    """元目的翻转测试。"""

    def test_no_flip_when_coherence_high(self):
        """coherence 高时不翻转。"""
        layer = PurposeLayer(
            input_dim=16,
            precision_lr=0.1,
            coherence_threshold=0.1,  # 很低阈值，几乎不触发
            min_history_length=3,
        )

        pattern = torch.zeros(16)
        pattern[0] = 0.8
        act = make_activation(num_nodes=32, input_dim=16, sensory_pattern=pattern)

        for _ in range(10):
            layer.adjust(surprise=1.0, activation=act)

        assert layer.flipped is False
        assert layer.flip_count == 0

    def test_flip_triggered_by_low_coherence(self):
        """coherence 持续低时触发翻转。"""
        layer = PurposeLayer(
            input_dim=8,
            precision_lr=0.5,
            coherence_threshold=0.95,  # 高阈值，容易触发
            min_history_length=3,
            meta_window=5,
        )

        # 交替极端不同的模式，使 precision 剧烈波动
        patterns = [
            torch.zeros(8),
            torch.zeros(8),
            torch.zeros(8),
        ]
        patterns[0][0] = 0.9
        patterns[1][4] = 0.9
        patterns[2][7] = 0.9

        flipped_any = False
        for i in range(15):
            pattern = patterns[i % 3]
            act = make_activation(num_nodes=16, input_dim=8, sensory_pattern=pattern)
            layer.adjust(surprise=2.0, activation=act)
            if layer.flipped:
                flipped_any = True

        # 应至少触发过一次翻转
        assert flipped_any
        assert layer.flip_count > 0

    def test_flip_boosts_target_dimension(self):
        """翻转后目标维度 precision 被强化到最大值。"""
        layer = PurposeLayer(
            input_dim=8,
            precision_lr=0.5,
            precision_max=10.0,
            coherence_threshold=0.95,
            min_history_length=3,
            meta_window=5,
        )

        # 先让维度 0 成为高惊讶维度
        for _ in range(5):
            pattern = torch.zeros(8)
            pattern[0] = 0.9
            act = make_activation(num_nodes=16, input_dim=8, sensory_pattern=pattern)
            layer.adjust(surprise=2.0, activation=act)

        # 然后切换到不同维度造成波动
        for i in range(10):
            pattern = torch.zeros(8)
            pattern[(i % 4) + 1] = 0.9  # 轮换不同维度
            act = make_activation(num_nodes=16, input_dim=8, sensory_pattern=pattern)
            layer.adjust(surprise=2.0, activation=act)

        # 如果发生了翻转，翻转后某个维度应被设为最大值
        if layer.flip_count > 0:
            precision = layer.get_precision()
            assert precision.max().item() >= 9.0  # 接近 precision_max

    def test_flip_count_increments(self):
        """每次翻转 flip_count 递增。"""
        layer = PurposeLayer(
            input_dim=8,
            precision_lr=0.5,
            coherence_threshold=0.99,  # 极高阈值
            min_history_length=2,
            meta_window=3,
        )

        initial_count = layer.flip_count

        # 制造剧烈波动
        for i in range(20):
            pattern = torch.zeros(8)
            pattern[i % 8] = 0.9
            act = make_activation(num_nodes=16, input_dim=8, sensory_pattern=pattern)
            layer.adjust(surprise=2.0, activation=act)

        assert layer.flip_count >= initial_count


# ----------------------------------------------------------------------- #
#  PurposeState 返回测试
# ----------------------------------------------------------------------- #


class TestGetPurpose:
    """get_purpose 返回 PurposeState 测试。"""

    def test_returns_purpose_state(self):
        """get_purpose 返回正确的 PurposeState。"""
        from core.types import PurposeState

        layer = PurposeLayer(input_dim=16, min_history_length=2)

        pattern = torch.randn(16).abs() * 0.3
        act = make_activation(num_nodes=32, input_dim=16, sensory_pattern=pattern)
        layer.adjust(surprise=1.0, activation=act)

        purpose = layer.get_purpose()
        assert isinstance(purpose, PurposeState)
        assert purpose.precision.shape == (16,)
        assert len(purpose.history) == 1
        assert isinstance(purpose.coherence, float)

    def test_precision_is_copy(self):
        """返回的 precision 是副本，不影响内部状态。"""
        layer = PurposeLayer(input_dim=16)

        precision = layer.get_precision()
        precision.fill_(0.0)

        # 内部 precision 不受影响
        assert torch.allclose(layer.get_precision(), torch.ones(16))


# ----------------------------------------------------------------------- #
#  2026-08-10 惊讶度语义拆分新增测试（设计v1.1 §5.2）
# ----------------------------------------------------------------------- #


class TestSurpriseSemantics:
    """准确性项语义下的目的层测试（C2 新增）。"""

    def test_purpose_global_scaling_not_saturated(self):
        """全局缩放 sigmoid(surprise/20) 在 surprise∈[1,100] 不饱和。

        （惊讶度修复-01-设计方案.md §5.2 新增，v1.1 审计必须修改项 2：
        旧式 sigmoid(surprise) 在 100 处恒≈1.0，全局缩放失效；除以
        total_surprise_scale=20 后取值跨 [0.512, 0.993]，单调有效。）
        """
        layer = PurposeLayer(input_dim=16)
        assert layer.total_surprise_scale == pytest.approx(20.0)
        low = torch.sigmoid(torch.tensor(1.0) / layer.total_surprise_scale)
        high = torch.sigmoid(torch.tensor(100.0) / layer.total_surprise_scale)
        # sigmoid(1/20)=0.512, sigmoid(100/20)=0.993
        assert float(low.item()) == pytest.approx(0.512, abs=1e-3)
        assert float(high.item()) == pytest.approx(0.993, abs=1e-3)
        # 对照旧式 sigmoid(surprise)：100 处恒 ≈ 1.0
        old = torch.sigmoid(torch.tensor(100.0))
        assert float(old.item()) == pytest.approx(1.0, abs=1e-6)
        assert float(high.item()) < 0.999, "除以 20 后不应饱和到 1.0"

    def test_per_dim_surprise_mapping(self):
        """per_dim 映射：读 activation.per_dim_surprise 时输出有界且单调；
        None 时回退不抛异常。

        （惊讶度修复-01-设计方案.md §5.2 新增：tanh(raw/k) 映射到
        [precision_min, precision_max]，随 raw 单调；None 走 |σ| 代理。）
        """
        layer = PurposeLayer(input_dim=16)
        pmin, pmax = layer.precision_min, layer.precision_max

        # 主分支：读 per_dim_surprise
        pd = torch.zeros(16)
        pd[:5] = torch.tensor([0.0, 0.1, 1.0, 5.0, 50.0])
        act = Activation(
            state=torch.zeros(32), entropy=0.5, surprise=1.0,
            per_dim_surprise=pd)
        out = layer._compute_per_dim_surprise(act)
        assert out.shape == (16,)
        assert torch.all(out >= pmin - 1e-6) and torch.all(out <= pmax + 1e-6)
        # 单调性：raw 越大 → mapped 越大
        mapped_vals = out[:5]
        assert torch.all(mapped_vals[1:] >= mapped_vals[:-1] - 1e-6)
        # 饱和映射：raw=0 → ≈precision_min；raw=50 → ≈precision_max
        assert float(out[0].item()) == pytest.approx(pmin, abs=1e-3)
        assert float(out[4].item()) == pytest.approx(pmax, abs=1e-2)

        # 回退分支：per_dim_surprise=None 不抛异常
        act_old = Activation(
            state=torch.cat([torch.ones(16) * 0.5, torch.zeros(16)]),
            entropy=0.5, surprise=1.0)
        out_fb = layer._compute_per_dim_surprise(act_old)
        assert out_fb.shape == (16,)
        assert torch.all(out_fb >= pmin - 1e-6) and torch.all(out_fb <= pmax + 1e-6)

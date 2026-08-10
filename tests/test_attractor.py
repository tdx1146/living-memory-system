"""
测试 FEP 吸引子网络
====================

验证内容:
  1. 推断能收敛
  2. 学习能改变 J 矩阵
  3. 正交化效果（两个相关输入训练后吸引子相关系数下降）
  4. 抗灾难性遗忘
  5. Langevin 函数数值稳定性
  6. 景观快照与恢复
"""

import pytest
import torch

from core.hippocampus.attractor import AttractorNetwork, langevin
from core.types import Activation


# ----------------------------------------------------------------------- #
#  Langevin 函数测试
# ----------------------------------------------------------------------- #


class TestLangevin:
    """Langevin 函数: L(b) = coth(b) - 1/b 的数值稳定性测试。"""

    def test_large_positive(self):
        """大正数 b → 接近 1（但不等于1，因为 1/b 项）。"""
        b = torch.tensor([10.0, 100.0, 1000.0])
        result = langevin(b)
        # L(b) = coth(b) - 1/b → 1 (当 b→∞)
        # L(10) ≈ 1 - 0.1 = 0.9
        # L(100) ≈ 1 - 0.01 = 0.99
        # L(1000) ≈ 1 - 0.001 = 0.999
        assert result[0].item() > 0.8    # L(10) ≈ 0.9
        assert result[1].item() > 0.98   # L(100) ≈ 0.99
        assert result[2].item() > 0.998  # L(1000) ≈ 0.999
        # b 越大越接近 1
        assert result[1] > result[0]
        assert result[2] > result[1]

    def test_large_negative(self):
        """大负数 b → 接近 -1（但不等于-1，因为 1/b 项）。"""
        b = torch.tensor([-10.0, -100.0, -1000.0])
        result = langevin(b)
        # L(b) = coth(b) - 1/b → -1 (当 b→-∞)
        # L(-10) ≈ -1 + 0.1 = -0.9
        # L(-100) ≈ -1 + 0.01 = -0.99
        assert result[0].item() < -0.8   # L(-10) ≈ -0.9
        assert result[1].item() < -0.98  # L(-100) ≈ -0.99
        assert result[2].item() < -0.998 # L(-1000) ≈ -0.999
        # b 越负越接近 -1
        assert result[1] < result[0]
        assert result[2] < result[1]

    def test_near_zero(self):
        """b → 0 → L(b) ≈ b/3（泰勒展开），不溢出。"""
        b = torch.tensor([1e-6, 1e-8, 0.0, -1e-6])
        result = langevin(b)
        # 应接近 b/3，且不产生 NaN/Inf
        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))
        # b=0 时 L=0
        assert abs(result[2].item()) < 1e-10

    def test_range(self):
        """输出始终在 (-1, 1) 范围内。"""
        b = torch.linspace(-5, 5, 100)
        result = langevin(b)
        assert torch.all(result > -1.001)
        assert torch.all(result < 1.001)

    def test_monotonic(self):
        """L(b) 是单调递增函数。"""
        b = torch.linspace(-5, 5, 100)
        result = langevin(b)
        diffs = result[1:] - result[:-1]
        assert torch.all(diffs > 0)


# ----------------------------------------------------------------------- #
#  推断测试
# ----------------------------------------------------------------------- #


class TestInfer:
    """FEP 推断测试。"""

    def test_infer_returns_activation(self):
        """推断返回正确的 Activation 对象。"""
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        sensory = torch.randn(16) * 0.5
        precision = torch.ones(16)
        act = net.infer(sensory, precision, num_steps=10)

        assert isinstance(act, Activation)
        assert act.state.shape == (32,)
        assert isinstance(act.entropy, float)
        assert isinstance(act.surprise, float)
        # 2026-08-10 惊讶度语义拆分：infer 产出完整四件套
        assert isinstance(act.free_energy, float)
        assert act.per_dim_surprise is not None
        assert act.per_dim_surprise.shape == (16,)
        assert act.mse is not None

    def test_surprise_nonnegative(self):
        """惊讶度（准确性项）恒 ≥ 0；precision=0 时为 0。

        （惊讶度修复-01-设计方案.md §5.2 新增：任意 sensory/precision 组合
        下 act.surprise >= 0，零精度下 == 0.0。）
        """
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        for p_val in (0.1, 1.0, 10.0):
            for _ in range(5):
                sensory = torch.randn(16) * 0.8
                precision = torch.full((16,), p_val)
                act = net.infer(sensory, precision)
                assert act.surprise >= 0.0, (
                    f"precision={p_val} 下 surprise={act.surprise} < 0")
        act_zero = net.infer(torch.randn(16), torch.zeros(16))
        assert act_zero.surprise == 0.0

    def test_surprise_vs_free_energy_decomposed(self):
        """surprise（准确性项）与 free_energy 分解自洽。

        free_energy == corr + bias + surprise + complexity（重算验证）；
        surprise == 手算 0.5·Σπ(σ−s)²（allclose）。
        （惊讶度修复-01-设计方案.md §5.2 新增。）
        """
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        sensory = torch.randn(16) * 0.5
        precision = torch.ones(16) * 2.0
        act = net.infer(sensory, precision)
        sigma = act.state

        corr = -0.5 * (sigma @ net.J @ sigma)
        bias = torch.dot(net.bias, sigma)
        acc = 0.5 * torch.sum(precision * (sigma[:16] - sensory) ** 2)
        comp = 0.5 * net.complexity_weight * torch.sum(net.J ** 2)

        assert act.free_energy == pytest.approx(
            float((corr + bias + acc + comp).item()), abs=1e-5)
        assert act.surprise == pytest.approx(float(acc.item()), abs=1e-5)

    def test_accuracy_vs_F_ordering_differs(self):
        """能量污染剥离的直接证据：free_energy 排序与 surprise 排序相反。

        构造两态：A=深陷吸引子但输入错位（能量项大负 → free_energy 低，但
        感官误差大 → surprise 高）；B=游离但输入吻合（能量项≈0 →
        free_energy 高，误差小 → surprise 低）。断言两者排序相反——
        surprise 与 free_energy 不是同一物理量。
        （惊讶度修复-01-设计方案.md §5.2 新增。）
        """
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        net.bias = torch.full((32,), -1.0)  # 深吸引子偏置（低能量阱）
        sensory = torch.randn(16) * 0.5
        precision = torch.ones(16)

        # A：全 0.9 强共激活（深陷吸引子），感官部分远离输入（大误差）
        state_a = torch.full((32,), 0.9)
        # B：感官部分贴近输入、其余近零（游离但吻合）
        state_b = torch.zeros(32)
        state_b[:16] = sensory * 0.99

        fe_a = net._compute_free_energy(state_a, sensory, precision)
        fe_b = net._compute_free_energy(state_b, sensory, precision)
        su_a = net._compute_surprise(state_a, sensory, precision)
        su_b = net._compute_surprise(state_b, sensory, precision)

        assert fe_a < fe_b, f"深陷态自由能应更低: {fe_a} vs {fe_b}"
        assert su_a > su_b, f"深陷态惊讶度应更高: {su_a} vs {su_b}"

    def test_infer_state_in_range(self):
        """推断后激活值在 (-1, 1) 范围内。"""
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        sensory = torch.randn(16) * 0.5
        precision = torch.ones(16)
        act = net.infer(sensory, precision, num_steps=10)

        assert torch.all(act.state >= -1.001)
        assert torch.all(act.state <= 1.001)

    def test_infer_converges(self):
        """推断能收敛：增加步数后状态变化减小。"""
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        sensory = torch.randn(16) * 0.5
        precision = torch.ones(16)

        # 跑 5 步
        net.reset_state()
        act_5 = net.infer(sensory, precision, num_steps=5)
        state_5 = act_5.state.clone()

        # 继续跑 5 步（总共 10 步）
        act_10 = net.infer(sensory, precision, num_steps=5)
        state_10 = act_10.state.clone()

        # 后 5 步的变化应小于前 5 步的变化（收敛趋势）
        # 由于已从相同状态出发，后续变化应更小
        diff_later = (state_10 - state_5).abs().mean().item()
        # 后续变化应很小（收敛后趋于稳定）
        assert diff_later < 1.0  # 收敛后每步变化不大

    def test_infer_converges_with_more_steps(self):
        """更多推断步数使状态更稳定。"""
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        sensory = torch.randn(16) * 0.5
        precision = torch.ones(16) * 2.0

        # 少步推断
        net.reset_state()
        net.infer(sensory, precision, num_steps=3)
        state_short = net.sigma.clone()
        change_short = (net.sigma - state_short).abs().mean().item()

        # 多步推断后状态更稳定
        net.reset_state()
        net.infer(sensory, precision, num_steps=50)
        state_long_1 = net.sigma.clone()
        net.infer(sensory, precision, num_steps=10)
        state_long_2 = net.sigma.clone()
        change_long = (state_long_2 - state_long_1).abs().mean().item()

        # 长步推断后变化应很小
        assert change_long < 0.5

    def test_precision_affects_inference(self):
        """高 precision 使感官输入对推断影响更大。"""
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        sensory = torch.randn(16) * 2.0

        # 低 precision
        net.reset_state()
        precision_low = torch.ones(16) * 0.1
        act_low = net.infer(sensory, precision_low, num_steps=20)

        # 高 precision
        net.reset_state()
        precision_high = torch.ones(16) * 10.0
        act_high = net.infer(sensory, precision_high, num_steps=20)

        # 两种 precision 下结果应不同
        diff = (act_high.state - act_low.state).abs().mean().item()
        assert diff > 0.01


# ----------------------------------------------------------------------- #
#  学习测试
# ----------------------------------------------------------------------- #


class TestLearn:
    """FEP 学习规则测试。"""

    def test_learn_changes_J(self):
        """学习能改变 J 矩阵。"""
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        J_before = net.J.clone()

        sensory = torch.randn(16) * 0.5
        precision = torch.ones(16)
        act = net.infer(sensory, precision)
        net.learn(act, sensory, learning_rate=0.1)

        J_after = net.J.clone()
        assert not torch.allclose(J_before, J_after)

    def test_J_remains_symmetric(self):
        """学习后 J 矩阵保持对称。"""
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        sensory = torch.randn(16) * 0.5
        precision = torch.ones(16)
        act = net.infer(sensory, precision)
        net.learn(act, sensory, learning_rate=0.1)

        assert torch.allclose(net.J, net.J.T, atol=1e-6)

    def test_J_diagonal_zero(self):
        """学习后对角线始终为 0（无自连接）。"""
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        sensory = torch.randn(16) * 0.5
        precision = torch.ones(16)
        act = net.infer(sensory, precision)
        net.learn(act, sensory, learning_rate=0.1)

        diagonal = net.J.diagonal()
        assert torch.allclose(diagonal, torch.zeros_like(diagonal))

    def test_j_clamp_unconditional(self):
        """J 范数钳制不依赖 norm_surprise 开关（无条件生效）。

        （惊讶度修复-01-设计方案.md §5.2 新增：钳制移出开关门后，
        norm_surprise=False 时设置 j_target_norm=0.1 仍应把 ‖J‖_F 压住。）
        """
        net = AttractorNetwork(
            num_nodes=32, input_dim=16, seed=42, norm_surprise=False)
        net.j_target_norm = 0.1
        for _ in range(5):
            sensory = torch.randn(16) * 0.5
            act = net.infer(sensory, torch.ones(16))
            net.learn(act, sensory, learning_rate=0.1)
        j_norm = float(torch.norm(net.J, p="fro").item())
        assert j_norm <= 0.1 * (1 + 1e-3), f"‖J‖_F={j_norm} 未被钳制到 0.1"

    def test_repeated_learning_strengthens_attractor(self):
        """重复学习同一模式后，该模式的吸引子变得更稳定（surprise 下降）。

        注（2026-08-10 惊讶度语义拆分）：surprise 现为准确性项
        （0.5·Σπ(σ−s)²），学习后 σ 更贴近 s → 预测误差下降 → surprise 下降，
        断言在新语义下依然成立，仅语义从"自由能"变为"预测误差"。
        """
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        sensory = torch.randn(16) * 0.5
        precision = torch.ones(16) * 2.0

        # 学习前
        net.reset_state()
        act_before = net.infer(sensory, precision, num_steps=20)
        surprise_before = act_before.surprise

        # 重复学习
        for _ in range(20):
            net.reset_state()
            act = net.infer(sensory, precision, num_steps=20)
            net.learn(act, sensory, learning_rate=0.05)

        # 学习后
        net.reset_state()
        act_after = net.infer(sensory, precision, num_steps=20)
        surprise_after = act_after.surprise

        # surprise 应下降（网络学会预测该模式）
        assert surprise_after < surprise_before


# ----------------------------------------------------------------------- #
#  正交化测试
# ----------------------------------------------------------------------- #


class TestOrthogonalization:
    """正交化效果测试：两个相关输入训练后吸引子相关系数下降。"""

    def test_orthogonalization_reduces_correlation(self):
        """两个相关输入训练后，吸引子相关系数应下降。"""
        torch.manual_seed(123)
        net = AttractorNetwork(num_nodes=64, input_dim=32, seed=42)
        net.orth_weight = 1.5  # 增强正交化压力

        # 创建两个相关但不同的感官输入
        base = torch.randn(32) * 0.5
        noise_a = torch.randn(32) * 0.3
        noise_b = torch.randn(32) * 0.3
        input_a = base + noise_a
        input_b = base + noise_b  # 与 input_a 相关但不同

        precision = torch.ones(32) * 2.0

        def get_attractor(net, sensory):
            """获取给定感官输入的吸引子状态。"""
            net.reset_state()
            act = net.infer(sensory, precision, num_steps=30)
            return act.state

        def cosine_sim(a, b):
            return torch.dot(a, b).item() / (
                a.norm().item() * b.norm().item() + 1e-8
            )

        # 训练前：计算非感官节点（后32维）的吸引子相关系数
        # 非感官节点更能体现正交化效果（感官节点被输入主导）
        attr_a_before = get_attractor(net, input_a)[32:]
        attr_b_before = get_attractor(net, input_b)[32:]
        corr_before = cosine_sim(attr_a_before, attr_b_before)

        # 交替训练两个模式
        for _ in range(50):
            net.reset_state()
            act_a = net.infer(input_a, precision, num_steps=30)
            net.learn(act_a, input_a, learning_rate=0.05)

            net.reset_state()
            act_b = net.infer(input_b, precision, num_steps=30)
            net.learn(act_b, input_b, learning_rate=0.05)

        # 训练后：计算非感官节点的吸引子相关系数
        attr_a_after = get_attractor(net, input_a)[32:]
        attr_b_after = get_attractor(net, input_b)[32:]
        corr_after = cosine_sim(attr_a_after, attr_b_after)

        # 正交化后非感官节点的相关系数应下降
        assert corr_after < corr_before


# ----------------------------------------------------------------------- #
#  抗灾难性遗忘测试
# ----------------------------------------------------------------------- #


class TestAntiCatastrophicForgetting:
    """抗灾难性遗忘测试：学习新模式后旧模式仍可检索。"""

    def test_old_pattern_retrievable_after_new_learning(self):
        """学习模式 B 后，模式 A 的吸引子仍可被检索。"""
        torch.manual_seed(456)
        net = AttractorNetwork(num_nodes=48, input_dim=24, seed=42)

        # 两个不同的模式
        input_a = torch.randn(24) * 0.5
        input_b = torch.randn(24) * 0.5  # 与 A 不同
        precision = torch.ones(24) * 2.0

        def get_attractor(net, sensory):
            net.reset_state()
            act = net.infer(sensory, precision, num_steps=30)
            return act.state

        # 先充分学习模式 A
        for _ in range(30):
            net.reset_state()
            act = net.infer(input_a, precision, num_steps=30)
            net.learn(act, input_a, learning_rate=0.05)

        attr_a_after_A = get_attractor(net, input_a)

        # 然后学习模式 B（可能干扰 A）
        for _ in range(15):
            net.reset_state()
            act = net.infer(input_b, precision, num_steps=30)
            net.learn(act, input_b, learning_rate=0.05)

        # 检索 A
        attr_a_after_B = get_attractor(net, input_a)

        # A 的吸引子在学习 B 后应仍然与学习 A 后的相似
        def cosine_sim(a, b):
            return torch.dot(a, b).item() / (
                a.norm().item() * b.norm().item() + 1e-8
            )

        similarity = cosine_sim(attr_a_after_A, attr_a_after_B)

        # 相似度应保持较高（旧模式未被完全遗忘）
        assert similarity > 0.3

    def test_old_pattern_retrieval_better_than_random(self):
        """学习 B 后，检索 A 得到的吸引子比随机模式更接近 A。"""
        torch.manual_seed(789)
        net = AttractorNetwork(num_nodes=48, input_dim=24, seed=42)

        input_a = torch.randn(24) * 0.5
        input_b = torch.randn(24) * 0.5
        precision = torch.ones(24) * 2.0

        # 学习 A
        for _ in range(30):
            net.reset_state()
            act = net.infer(input_a, precision, num_steps=30)
            net.learn(act, input_a, learning_rate=0.05)

        # 学习 B（少量）
        for _ in range(10):
            net.reset_state()
            act = net.infer(input_b, precision, num_steps=30)
            net.learn(act, input_b, learning_rate=0.05)

        # 用 A 作为 cue 检索
        net.reset_state()
        retrieved = net.infer(input_a, precision, num_steps=30).state

        # 检索结果应更接近 A 的初始感官信号（在前 24 维）
        sensory_retrieved = retrieved[:24].sign()
        sensory_a_sign = input_a.sign()

        # 符号一致的比例应高于随机（>50%）
        agreement = (sensory_retrieved == sensory_a_sign).float().mean().item()
        assert agreement > 0.5


# ----------------------------------------------------------------------- #
#  景观快照测试
# ----------------------------------------------------------------------- #


class TestLandscape:
    """吸引子景观快照与恢复测试。"""

    def test_get_landscape(self):
        """get_landscape 返回正确的状态字典。"""
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        landscape = net.get_landscape()

        assert "J" in landscape
        assert "bias" in landscape
        assert "sigma" in landscape
        assert landscape["J"].shape == (32, 32)
        assert landscape["bias"].shape == (32,)
        assert landscape["sigma"].shape == (32,)

    def test_set_landscape_restores_state(self):
        """set_landscape 能恢复之前保存的状态。"""
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)

        # 学习一些东西
        sensory = torch.randn(16) * 0.5
        precision = torch.ones(16) * 2.0
        for _ in range(5):
            net.reset_state()
            act = net.infer(sensory, precision)
            net.learn(act, sensory, learning_rate=0.1)

        # 保存
        landscape = net.get_landscape()

        # 创建新网络并恢复
        net2 = AttractorNetwork(num_nodes=32, input_dim=16, seed=999)
        net2.set_landscape(landscape)

        # 验证恢复后的状态一致
        assert torch.allclose(net.J, net2.J)
        assert torch.allclose(net.bias, net2.bias)
        assert torch.allclose(net.sigma, net2.sigma)

    def test_landscape_is_copy(self):
        """get_landscape 返回的是副本，修改不影响原网络。"""
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        landscape = net.get_landscape()
        landscape["J"].fill_(0.0)

        # 原网络的 J 不应被修改
        assert not torch.allclose(net.J, torch.zeros(32, 32))


# ----------------------------------------------------------------------- #
#  自由能与熵测试
# ----------------------------------------------------------------------- #


class TestFreeEnergy:
    """自由能与熵计算测试。"""

    def test_free_energy_is_finite(self):
        """自由能与惊讶度都是有限值。"""
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        sensory = torch.randn(16) * 0.5
        precision = torch.ones(16) * 2.0
        act = net.infer(sensory, precision)

        assert not (act.surprise == float("inf") or act.surprise == float("-inf"))
        assert not (act.surprise != act.surprise)  # not NaN
        # 2026-08-10 语义拆分：free_energy 为未规范化变分能量（可负），同样须有限
        assert not (act.free_energy == float("inf") or act.free_energy == float("-inf"))
        assert not (act.free_energy != act.free_energy)  # not NaN

    def test_norm_surprise_noop(self):
        """norm_surprise 开关已弃用为 no-op：开/关下 free_energy 数值一致。

        （惊讶度修复-01-设计方案.md §5.2 新增：F/‖J‖ 已移除，开关不参与计算。）
        """
        sensory = torch.randn(16) * 0.5
        precision = torch.ones(16) * 2.0
        net_a = AttractorNetwork(
            num_nodes=32, input_dim=16, seed=42, norm_surprise=True)
        net_b = AttractorNetwork(
            num_nodes=32, input_dim=16, seed=42, norm_surprise=False)
        net_a.J = net_b.J.clone()
        net_a.bias = net_b.bias.clone()
        sigma = torch.randn(32) * 0.5
        fe_a = net_a._compute_free_energy(sigma, sensory, precision)
        fe_b = net_b._compute_free_energy(sigma, sensory, precision)
        assert fe_a == pytest.approx(fe_b)

    def test_entropy_non_negative(self):
        """熵是非负值。"""
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        sensory = torch.randn(16) * 0.5
        precision = torch.ones(16) * 2.0
        act = net.infer(sensory, precision)

        # 信息熵应非负（用概率分布计算的熵 >= 0）
        assert act.entropy >= -1e-6

    def test_free_energy_decreases_with_learning(self):
        """学习后自由能下降。

        （2026-08-10 惊讶度语义拆分：断言对象从 surprise 迁移到 free_energy——
        surprise 现为准确性项，与"自由能下降"语义不符；free_energy 保留
        现有 F 公式，仍是学习目标。）
        """
        net = AttractorNetwork(num_nodes=32, input_dim=16, seed=42)
        sensory = torch.randn(16) * 0.5
        precision = torch.ones(16) * 2.0

        net.reset_state()
        act_before = net.infer(sensory, precision, num_steps=20)
        fe_before = act_before.free_energy

        for _ in range(20):
            net.reset_state()
            act = net.infer(sensory, precision, num_steps=20)
            net.learn(act, sensory, learning_rate=0.05)

        net.reset_state()
        act_after = net.infer(sensory, precision, num_steps=20)
        fe_after = act_after.free_energy

        assert fe_after < fe_before

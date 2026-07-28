"""
活体记忆系统 - 海马体核心：FEP 吸引子网络
==========================================

这是整个系统的核心记忆引擎。基于自由能原理（Free Energy Principle, FEP）
实现记忆的"塑形"——不是存档，而是持续改变网络的吸引子景观。

核心机制:
  1. 推断（infer）: Langevin 激活动力学 + 感官 clamping，收敛到吸引子态
  2. 学习（learn）: FEP 学习规则 ΔJ = -η∂F/∂J，准确性(Hebbian) + 复杂性(正交化)
  3. 无反向传播，无全局 loss——规则从第一性原理推导

数学基础:
  - Langevin 函数: L(b) = coth(b) - 1/b，是 CB（Chandrasekhar-Brink）分布的激活函数
  - 自由能: F = 准确性项 + 复杂性项(KL散度)
  - 准确性梯度 ≈ Hebbian 相关 (σ_i * σ_j)，强化共激活连接
  - 复杂性梯度 = 正交化压力，驱散相似表示，形成可区分的吸引子

参考：架构文档 第五节 5.3、第七节《FEP学习规则实现要点》
"""

import torch

from core.types import Activation


def langevin(b: torch.Tensor) -> torch.Tensor:
    """Langevin 函数：CB 分布的激活函数。

    L(b) = coth(b) - 1/b = 1/tanh(b) - 1/b

    性质:
      - b → +∞: L(b) → 1
      - b → -∞: L(b) → -1
      - b → 0:  L(b) → b/3（泰勒展开极限，需特殊处理避免数值溢出）

    使用 1/tanh(b) 代替 cosh(b)/sinh(b) 以避免大 b 时的数值溢出
    （tanh 在 ±1 处饱和，不会溢出）。

    参数:
        b: 任意形状的张量，对数几率（log-odds）。

    返回:
        与 b 同形状的张量，取值范围 (-1, 1)。
    """
    result = torch.empty_like(b)
    # 小 b 区域使用泰勒展开 L(b) ≈ b/3，避免 1/tanh(b) 和 1/b 的除零问题
    small_mask = b.abs() < 1e-4
    large_mask = ~small_mask

    # 大 b 区域：使用 1/tanh(b) - 1/b（避免 cosh/sinh 溢出）
    b_large = b[large_mask]
    result[large_mask] = 1.0 / torch.tanh(b_large) - 1.0 / b_large

    # 小 b 区域：泰勒展开 L(b) ≈ b/3
    result[small_mask] = b[small_mask] / 3.0

    return result


class AttractorNetwork:
    """FEP 吸引子网络：核心记忆引擎。

    网络结构:
      - num_nodes 个节点，前 input_dim 个为感官节点（接收外部 clamping）
      - J: 耦合矩阵 [num_nodes, num_nodes]，对称，对角线为 0
      - bias: 偏置向量 [num_nodes]
      - sigma: 当前状态 [num_nodes]，取值 (-1, 1)

    推断时，感官节点被外部信号 clamping，其余节点通过 Langevin 动力学
    收敛到吸引子态。学习时，根据 FEP 规则更新 J 矩阵。

    关键: 无反向传播，无全局 loss。学习规则从自由能原理推导。
    """

    def __init__(self, num_nodes: int, input_dim: int,
                 seed: int = 42,
                 temperature: float = 0.05) -> None:
        """初始化吸引子网络。

        参数:
            num_nodes: 网络节点总数（建议 256-1024）。
            input_dim: 感官输入维度。前 input_dim 个节点为感官节点。
            seed: 随机种子，保证可复现。
            temperature: Langevin 动力学温度（扩散项噪声强度）。
                T > 0 时相似但不同的输入有机会收敛到不同吸引子，
                使正交化学习规则能够发挥作用。设为 0 则退化为纯平均场推断。
        """
        assert input_dim <= num_nodes, (
            f"input_dim({input_dim}) 不能大于 num_nodes({num_nodes})"
        )
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.seed = seed

        generator = torch.Generator()
        generator.manual_seed(seed)

        # 耦合矩阵 J：初始为小随机值，对称化，对角线为 0
        # 小初始化保证初始状态接近中性，学习逐步塑造结构
        self.J: torch.Tensor = (
            torch.randn(num_nodes, num_nodes, generator=generator) * 0.01
        )
        # 对称化（非序列模式：J_ij = J_ji）
        self.J = (self.J + self.J.T) / 2
        # 无自连接
        self.J.fill_diagonal_(0)

        # 偏置向量：初始为 0
        self.bias: torch.Tensor = torch.zeros(num_nodes)

        # 当前状态 sigma：初始为 0（中性状态）
        self.sigma: torch.Tensor = torch.zeros(num_nodes)

        # 复杂性/正交化参数（可外部调整）
        self.complexity_weight: float = 0.01
        self.orth_weight: float = 0.5

        # Langevin 温度：扩散项的噪声强度（G7 修复：从构造参数获取）
        # 物理上 Langevin 动力学 = 漂移(确定性) + 扩散(随机性)
        # temperature > 0 时，相似但不同的输入有机会收敛到不同吸引子，
        # 使正交化学习规则能够发挥作用。设为 0 则退化为纯平均场推断。
        self.temperature: float = temperature

    # ------------------------------------------------------------------ #
    #  推断
    # ------------------------------------------------------------------ #

    def infer(self, sensory_input: torch.Tensor, precision: torch.Tensor,
              num_steps: int = 10) -> Activation:
        """FEP 推断：给定感官输入和 precision，跑 K 步收敛到吸引子态。

        推断规则（从 FEP 推导）:
            b_q = bias + J @ sigma           # 对数几率（内部模型预测）
            sigma = langevin(b_q)             # Langevin 激活: coth(b) - 1/b
            sigma += sqrt(2*T) * noise        # Langevin 扩散项（随机性）
            感官节点被 sensory_input * precision clamping（外部证据注入）

        Langevin 动力学物理上包含漂移项（确定性激活）和扩散项（随机噪声）。
        扩散项的作用:
          - 打破对称性：相似但不同的输入有机会收敛到不同吸引子
          - 探索景观：避免陷入次优局部最小值
          - 使正交化学习规则生效：不同吸引子产生不同的 Hebbian 相关

        高 precision = 高信任 = 感官证据主导推断。
        低 precision = 低信任 = 内部模型主导推断。

        参数:
            sensory_input: 感官向量，形状 [input_dim]。
            precision: 感官精度向量，形状 [input_dim]。
            num_steps: 推断迭代步数 K。

        返回:
            收敛后的 Activation（激活态 + 熵 + 惊讶度）。
        """
        # B7 修复：输入形状校验，防止晦涩的广播错误
        assert sensory_input.shape == (self.input_dim,), (
            f"sensory_input shape mismatch: {sensory_input.shape} "
            f"!= ({self.input_dim},)"
        )
        assert precision.shape == (self.input_dim,), (
            f"precision shape mismatch: {precision.shape} "
            f"!= ({self.input_dim},)"
        )

        sigma = self.sigma.clone()

        for _step in range(num_steps):
            # 对数几率：内部模型的预测
            b_q = self.bias + self.J @ sigma  # [num_nodes]

            # 感官 clamping：将外部证据注入感官节点
            # precision 权衡感官证据与内部预测的信任程度
            b_q[:self.input_dim] = (
                b_q[:self.input_dim] + sensory_input * precision
            )

            # Langevin 漂移项：确定性激活
            sigma = langevin(b_q)

            # Langevin 扩散项：小量随机扰动
            # 这不是数值噪声，而是 Langevin 方程的物理组成部分
            # 温度 T 控制噪声强度，T=0 时退化为平均场推断
            if self.temperature > 0:
                noise = torch.randn_like(sigma) * self.temperature
                sigma = sigma + noise
                # 保持激活值在 (-1, 1) 范围内
                sigma = torch.clamp(sigma, -0.999, 0.999)

        # 更新内部状态
        self.sigma = sigma.clone()

        # 计算熵与惊讶度
        surprise = self._compute_free_energy(sigma, sensory_input, precision)
        entropy = self._compute_entropy(sigma)

        return Activation(state=sigma, entropy=entropy, surprise=surprise)

    # ------------------------------------------------------------------ #
    #  学习
    # ------------------------------------------------------------------ #

    def learn(self, activation: Activation, sensory_input: torch.Tensor,
              learning_rate: float = 0.01) -> None:
        """FEP 学习：更新耦合矩阵 J。

        学习规则（从 FEP 推导）:
            ΔJ = -η * ∂F/∂J
            F = 准确性项 + 复杂性项(KL)

        准确性项的梯度:
            ∂F_accuracy/∂J_ij = -σ_i * σ_j  （共激活应增强连接以降低自由能）
            因此 -∂F_accuracy/∂J = +σ_i * σ_j = Hebbian 相关

        复杂性项的实现（双层正交化）:

            层 1 — sigma 正交化（方向性）:
                在 Hebbian 学习之前，从当前 sigma 中减去与
                已学结构（J @ sigma，即先验）重叠的投影分量。
                这样新模式只在"新颖方向"上学习，不与已有吸引子叠加。
                这是 KL(后验||先验) 的直接体现。

                sigma_orth = sigma - orth_weight * proj * (J @ sigma)

            层 2 — 饱和度压力（幅度性）:
                对 J 中已经大的连接施加额外学习阻力，防止发散。
                见 _complexity_gradient()。

        最终更新:
            ΔJ = Hebbian(sigma_orth) - 复杂性梯度
            J += η * (ΔJ + ΔJ^T) / 2   （对称化）
            J.fill_diagonal_(0)          （无自连接）

        无反向传播。无全局 loss。规则从第一性原理推导。

        参数:
            activation: 推断得到的激活态。
            sensory_input: 产生该激活态的感官输入。
            learning_rate: 学习率 η。
        """
        sigma = activation.state

        # --- 层 1: sigma 正交化（改变学习方向） ---
        # 从当前模式中减去与已学结构（先验）重叠的部分
        # 只在"新颖"方向上做 Hebbian 学习
        if self.orth_weight > 0:
            recall = self.J @ sigma  # 网络对当前模式的回忆（先验）
            recall_energy = torch.dot(recall, recall)
            if recall_energy > 1e-10:
                # 当前 sigma 在回忆方向上的投影系数
                proj_coef = torch.dot(recall, sigma) / recall_energy
                # 减去投影（保留正交分量）
                sigma = sigma - self.orth_weight * proj_coef * recall
                # 保持在 Langevin 激活值的有效范围
                sigma = torch.clamp(sigma, -0.999, 0.999)

        # --- 准确性梯度：Hebbian 相关（在正交化 sigma 上） ---
        # 共激活的节点增强连接，降低自由能（形成吸引子）
        hebbian = torch.outer(sigma, sigma)  # [num_nodes, num_nodes]

        # --- 层 2: 复杂性梯度：饱和度压力 + 权重衰减 ---
        complexity_grad = self._complexity_gradient(sigma)

        # --- 总更新: ΔJ = -∂F/∂J = Hebbian - 复杂性 ---
        # 注意符号: ∂F_accuracy/∂J = -Hebbian（负号因为增强连接降低自由能）
        #          ∂F_complexity/∂J = +complexity_grad（正号因为复杂性增加自由能）
        #          ΔJ = -(∂F_accuracy + ∂F_complexity) = Hebbian - complexity_grad
        delta_J = hebbian - complexity_grad

        # 对称化（非序列模式）
        self.J = self.J + learning_rate * (delta_J + delta_J.T) / 2

        # 对角线置零（无自连接）
        self.J.fill_diagonal_(0)

    def _complexity_gradient(self, sigma: torch.Tensor) -> torch.Tensor:
        """复杂性梯度：正交化压力 + 权重衰减。

        正交化压力（核心创新）:
            基于 J 矩阵的已有结构（饱和度）施加学习阻力。

            J 中已经大的连接代表被之前模式强化的方向。
            新模式在这些"已占用"方向上的 Hebbian 学习应该被抑制，
            迫使新模式使用网络中尚未被占用的节点子空间。

            这是一种"饱和度门控学习"：
                saturation = |J_ij| / max(|J|)     # 已学习程度
                orth_pressure = orth_weight * saturation * |Hebbian|

            结果：不同模式使用不同的连接子集，形成近似正交的吸引子。

            与 Langevin 扩散项配合使用：
            - 扩散项让相似输入产生不同的 sigma（打破对称性）
            - 正交化让不同 sigma 的学习不重叠（形成独立吸引子）

        权重衰减:
            complexity_weight * J，防止 J 发散，提供抗灾难性遗忘能力。
            保持 J 有界使得旧模式不会被新模式完全覆盖。

        参数:
            sigma: 当前激活态，形状 [num_nodes]。

        返回:
            复杂性梯度矩阵，形状 [num_nodes, num_nodes]。
        """
        # J 矩阵的饱和度：连接强度的归一化
        J_strength = self.J.abs()
        max_strength = J_strength.max()
        if max_strength > 1e-8:
            saturation = J_strength / max_strength  # 归一化到 [0, 1]
        else:
            saturation = torch.zeros_like(J_strength)

        # 当前 Hebbian 项的方向
        hebbian = torch.outer(sigma, sigma)

        # 正交化压力：在已饱和（已学习）的方向上施加阻力
        # saturation 高 = 这些连接已被之前的模式强化
        # 新模式在这些方向的学习被抑制，被迫寻找新的方向
        orth_pressure = self.orth_weight * saturation * hebbian.abs()

        # L2 权重衰减（防止 J 发散，抗灾难性遗忘）
        weight_decay = self.complexity_weight * self.J

        return orth_pressure + weight_decay

    # ------------------------------------------------------------------ #
    #  自由能与熵
    # ------------------------------------------------------------------ #

    def _compute_free_energy(self, sigma: torch.Tensor,
                             sensory_input: torch.Tensor,
                             precision: torch.Tensor) -> float:
        """计算自由能（惊讶度）。

        F = 能量项 + 准确性项 + 复杂性项

        能量项（网络内部能量，负号因为共激活降低能量）:
            -0.5 * σ^T J σ - b^T σ
            高共激活 → 低能量 → 低自由能（状态更"自然"）

        准确性项（感官预测误差）:
            0.5 * Σ precision_i * (σ_i - sensory_i)^2
            预测越准 → 误差越小 → 自由能越低

        复杂性项（模型复杂度正则化）:
            0.5 * complexity_weight * ||J||^2
            模型越简单 → 自由能越低

        参数:
            sigma: 激活态，形状 [num_nodes]。
            sensory_input: 感官输入，形状 [input_dim]。
            precision: 精度向量，形状 [input_dim]。

        返回:
            自由能标量（惊讶度）。越低表示状态越符合网络预期。
        """
        # 能量项：共激活越强，自由能越低
        corr_term = -0.5 * (sigma @ self.J @ sigma)
        bias_term = torch.dot(self.bias, sigma)

        # 准确性项：感官预测误差
        sensory_error = sigma[:self.input_dim] - sensory_input
        accuracy = 0.5 * torch.sum(precision * sensory_error ** 2)

        # 复杂性项：J 的 L2 正则化
        complexity = 0.5 * self.complexity_weight * torch.sum(self.J ** 2)

        free_energy = corr_term + bias_term + accuracy + complexity
        return float(free_energy)

    def _compute_entropy(self, sigma: torch.Tensor) -> float:
        """计算激活熵。

        熵衡量激活状态的分散程度:
          - 高熵：激活均匀分布（不确定状态）
          - 低熵：激活集中于少数节点（确定状态/明确吸引子）

        使用 |σ| 的信息熵近似（sigma 取值 (-1,1)，用绝对值度量激活强度）。

        参数:
            sigma: 激活态，形状 [num_nodes]。

        返回:
            熵标量。
        """
        abs_sigma = sigma.abs()
        # 归一化为概率分布
        total = abs_sigma.sum()
        if total < 1e-8:
            return 0.0
        p = abs_sigma / total
        entropy = -torch.sum(p * torch.log(p + 1e-8))
        return float(entropy)

    # ------------------------------------------------------------------ #
    #  景观快照
    # ------------------------------------------------------------------ #

    def get_landscape(self) -> dict:
        """返回当前吸引子景观状态（用于快照）。

        景观 = J矩阵 + bias + sigma 的完整状态。
        这就是"火种"——保存它即可在任何地方重新点燃同一身份。

        返回:
            包含 J、bias、sigma 的字典（均为张量副本）。
        """
        return {
            "J": self.J.clone(),
            "bias": self.bias.clone(),
            "sigma": self.sigma.clone(),
            "num_nodes": self.num_nodes,
            "input_dim": self.input_dim,
        }

    def set_landscape(self, landscape: dict) -> None:
        """从快照恢复吸引子景观（重新点燃火种）。

        参数:
            landscape: get_landscape() 返回的字典。
        """
        self.J = landscape["J"].clone()
        self.bias = landscape["bias"].clone()
        self.sigma = landscape["sigma"].clone()
        self.num_nodes = landscape["num_nodes"]
        self.input_dim = landscape["input_dim"]

    def reset_state(self) -> None:
        """重置内部状态 sigma 为零（不影响已学习的 J 矩阵）。"""
        self.sigma = torch.zeros(self.num_nodes)

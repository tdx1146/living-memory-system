"""
活体记忆系统 - 海马体核心：目的层（核心创新）
================================================

目的层是本系统区别于传统记忆系统的核心设计。
在 FEP 框架中，precision 决定对感官证据的信任程度。
如果 precision 固定，SLM 只能被动最小化惊讶；
如果 precision 可变，SLM 可以主动选择关注什么——
这就是"目的"的萌芽。

三层结构:
  Layer 1 - Sensory Precision（感官精度）:
      每个 sensory 维度的信任度，初始均匀，根据惊讶度调整
      常遇到的 → 低 precision（已熟悉）
      意外的   → 高 precision（值得关注）

  Layer 2 - Attention Allocation（注意力分配）:
      从 sensory precision 派生的"兴趣分布"
      coherence 参数衡量目的的内部一致性

  Layer 3 - Meta-Purpose（元目的）:
      对"自己的 precision 是否合适"的元层面评估
      允许目的的"翻转"——不是渐进调整，是质变
      当 coherence 持续低时，在历史中找最高惊讶维度强化

对应 dandan 的概念:
  - "目的是随对话共生" → precision 在每轮对话中被调整
  - "逐步成型" → precision 的历史轨迹，越后期越稳定
  - "有权选择不同兴趣爱好" → Layer 3 的元目的翻转

参考：架构文档 第六节《目的层详细设计》
"""

import torch

from core.types import Activation, PurposeState


class PurposeLayer:
    """目的层：SLM 自主调整 precision。

    precision 不是超参数——它是一个可以被 SLM 自身调整的状态。
    precision 的演化轨迹 = SLM "兴趣"的成型过程 = 目的的涌现。

    三层结构协同工作:
      Layer 1 根据惊讶度更新 precision（渐变）
      Layer 2 派生注意力分配并计算 coherence（自洽性度量）
      Layer 3 在 coherence 持续低时触发元目的翻转（质变）
    """

    def __init__(self, input_dim: int,
                 precision_lr: float = 0.1,
                 precision_min: float = 0.1,
                 precision_max: float = 10.0,
                 coherence_threshold: float = 0.3,
                 min_history_length: int = 5,
                 meta_window: int = 10) -> None:
        """初始化目的层。

        参数:
            input_dim: 感官输入维度（= precision 向量长度）。
            precision_lr: precision 调整的学习率。
            precision_min: precision 下限，防止发散。
            precision_max: precision 上限，防止发散。
            coherence_threshold: coherence 低于此值时考虑触发元目的翻转。
            min_history_length: 触发翻转所需的最短历史长度。
            meta_window: 元目的翻转时回看的历史窗口大小。
        """
        self.input_dim = input_dim
        self.precision_lr = precision_lr
        self.precision_min = precision_min
        self.precision_max = precision_max
        self.coherence_threshold = coherence_threshold
        self.min_history_length = min_history_length
        self.meta_window = meta_window

        # Layer 1: Sensory Precision —— 初始均匀分布
        self.sensory_precision: torch.Tensor = torch.ones(input_dim)

        # Layer 2: Attention Allocation —— 从 precision 派生
        self.attention: torch.Tensor = torch.softmax(self.sensory_precision, dim=0)
        self.coherence: float = 1.0  # 初始 coherence 满分

        # Layer 3: Meta-Purpose —— 元目的状态
        self.flipped: bool = False  # 是否发生过翻转
        self.flip_count: int = 0    # 翻转次数

        # precision 演化历史
        self.history: list[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    #  Layer 1: 感官精度更新
    # ------------------------------------------------------------------ #

    def _compute_per_dim_surprise(self, activation: Activation) -> torch.Tensor:
        """计算每个感官维度的惊讶度。

        惊讶度代理：感官节点的激活强度 |σ_i|。
        高激活 = 该维度承载了信息 = 令人惊讶。
        低激活 = 该维度无信息 = 不令人惊讶。

        将惊讶度缩放到 precision 的合理范围 [precision_min, precision_max]。

        参数:
            activation: 吸引子网络的激活态。

        返回:
            每个感官维度的惊讶度，形状 [input_dim]，值域映射到 precision 范围。
        """
        # 感官节点的激活强度（取绝对值，范围 (0, 1)）
        sensory_activation = activation.state[:self.input_dim].abs()

        # 缩放到 precision 范围：0 → precision_min，1 → precision_max
        scale = self.precision_max - self.precision_min
        per_dim_surprise = self.precision_min + sensory_activation * scale
        return per_dim_surprise

    # ------------------------------------------------------------------ #
    #  Layer 2: 注意力分配 + 一致性计算
    # ------------------------------------------------------------------ #

    def _compute_coherence(self) -> float:
        """计算目的的内部一致性（coherence）。

        coherence 衡量当前 precision 与近期历史的吻合程度:
          - coherence 高：precision 稳定，目的明确
          - coherence 低：precision 波动，目的不明确

        使用当前 precision 与历史均值的余弦相似度。

        返回:
            coherence 标量，范围 [0, 1]。
        """
        if len(self.history) < 2:
            return 1.0  # 历史不足时默认一致

        # 取近期历史窗口
        window = min(self.meta_window, len(self.history))
        recent = self.history[-window:]
        history_tensor = torch.stack(recent)  # [window, input_dim]

        # 历史均值
        mean_precision = history_tensor.mean(dim=0)  # [input_dim]

        # 余弦相似度
        current = self.sensory_precision
        norm_curr = current.norm()
        norm_mean = mean_precision.norm()
        if norm_curr < 1e-8 or norm_mean < 1e-8:
            return 0.0

        cosine = torch.dot(current, mean_precision) / (norm_curr * norm_mean)
        # 余弦相似度范围 [-1, 1]，映射到 [0, 1]
        coherence = float((cosine + 1.0) / 2.0)
        return coherence

    # ------------------------------------------------------------------ #
    #  Layer 3: 元目的翻转
    # ------------------------------------------------------------------ #

    def _meta_adjust(self) -> None:
        """元目的翻转：当 coherence 持续低时触发。

        不是立刻翻转，而是在历史中找到最高惊讶维度（平均 precision 最高），
        强化它——这对应"突然对某个方向产生兴趣"。

        翻转策略:
          1. 回看近期历史，找出平均 precision 最高的维度
          2. 将该维度 precision 设为最大值（大幅强化）
          3. 适当衰减其他维度（腾出注意力空间）
          4. 标记翻转发生
        """
        if len(self.history) < 1:
            return

        # 回看近期历史窗口
        window = min(self.meta_window, len(self.history))
        recent = self.history[-window:]
        history_tensor = torch.stack(recent)  # [window, input_dim]

        # 找历史中平均 precision 最高的维度
        avg_precision = history_tensor.mean(dim=0)  # [input_dim]
        target_dim = int(avg_precision.argmax().item())

        # 强化目标维度：设为最大值
        self.sensory_precision[target_dim] = self.precision_max

        # 轻微衰减其他维度（腾出注意力空间，但不至于归零）
        mask = torch.ones(self.input_dim)
        mask[target_dim] = 0.0
        self.sensory_precision = (
            self.sensory_precision * (1.0 - 0.3 * mask)
            + self.precision_min * 0.3 * mask
        )

        # 重新 clamp
        self.sensory_precision = torch.clamp(
            self.sensory_precision, self.precision_min, self.precision_max
        )

        # 标记翻转
        self.flipped = True
        self.flip_count += 1

    # ------------------------------------------------------------------ #
    #  主接口
    # ------------------------------------------------------------------ #

    def adjust(self, surprise: float, activation: Activation) -> None:
        """根据惊讶度调整 precision。

        核心思想:
          - 高惊讶维度 → 提高 precision（更关注，学得更快）
          - 低惊讶维度 → 降低 precision（已熟悉，减少关注）
          - 但不是简单的贪心调整——有自身一致性约束（coherence）

        调整规则遵循 FEP 元层面:
          - precision 的更新 = 最小化"元自由能"
          - 元自由能 = 对"我的 precision 是否合适"的惊讶

        这就是"目的随对话共生"的机制。

        参数:
            surprise: 标量惊讶度（自由能），来自吸引子网络。
            activation: 激活态，用于提取每个感官维度的惊讶度。
        """
        # --- Layer 1: 基于 per-dimension 惊讶度调整 sensory precision ---
        per_dim_surprise = self._compute_per_dim_surprise(activation)

        # 指数移动平均：precision 向 per_dim_surprise 靠拢
        # surprise 全局缩放因子：总惊讶度越高，调整幅度越大
        surprise_scale = torch.sigmoid(torch.tensor(surprise))  # (0, 1)
        effective_lr = self.precision_lr * (0.5 + surprise_scale.item())

        self.sensory_precision = (
            self.sensory_precision
            + effective_lr * (per_dim_surprise - self.sensory_precision)
        )

        # Clamp 防止发散
        self.sensory_precision = torch.clamp(
            self.sensory_precision, self.precision_min, self.precision_max
        )

        # --- Layer 2: 计算注意力分配 + coherence ---
        self.attention = torch.softmax(self.sensory_precision, dim=0)
        self.coherence = self._compute_coherence()

        # --- Layer 3: 元目的翻转 ---
        # 重置翻转标记（每次 adjust 重新判断）
        self.flipped = False
        if (self.coherence < self.coherence_threshold
                and len(self.history) >= self.min_history_length):
            self._meta_adjust()
            # 翻转后重新计算 attention
            self.attention = torch.softmax(self.sensory_precision, dim=0)

        # --- 记录历史 ---
        self.history.append(self.sensory_precision.clone())

    def get_precision(self) -> torch.Tensor:
        """返回当前 precision 向量。

        返回:
            precision 副本，形状 [input_dim]。
        """
        return self.sensory_precision.clone()

    def get_purpose(self) -> PurposeState:
        """返回当前目的状态。

        返回:
            PurposeState，包含 precision、history 和 coherence。
        """
        return PurposeState(
            precision=self.sensory_precision.clone(),
            history=[p.clone() for p in self.history],
            coherence=self.coherence,
        )

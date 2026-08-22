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
      允许目的的"方向翻转"——不是渐进调整，是质变
      当 coherence 持续低时，强化历史平均 precision 最高的维度
      （越关注越深入，回初版语义 01e6482）

对应 dandan 的概念:
  - "目的是随对话共生" → precision 在每轮对话中被调整
  - "逐步成型" → precision 的历史轨迹，越后期越稳定
  - "有权选择不同兴趣爱好" → Layer 3 的元目的翻转

参考：架构文档 第六节《目的层详细设计》
"""

import torch
from typing import Union

from core.types import Activation, PurposeState, resolve_device


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
                 coherence_threshold: float = 0.5,
                 min_history_length: int = 5,
                 meta_window: int = 10,
                 max_history: int = 100,
                 habituation_rate: float = 0.05,
                 activation_threshold: float = 0.3,
                 coherence_direction_weight: float = 0.5,
                 coherence_magnitude_weight: float = 0.5,
                 per_dim_surprise_scale: float = 0.1,
                 total_surprise_scale: float = 20.0,
                 device: Union[str, torch.device] = "auto") -> None:
        """初始化目的层。

        参数:
            input_dim: 感官输入维度（= precision 向量长度）。
            precision_lr: precision 调整的学习率。
            precision_min: precision 下限，防止发散。
            precision_max: precision 上限，防止发散。
            coherence_threshold: coherence 低于此值时考虑触发元目的翻转。
                默认 0.5（混合 coherence 分布不同于纯余弦相似度，
                0.3 过低会导致元目的翻转始终不触发）。
            min_history_length: 触发翻转所需的最短历史长度。
            meta_window: 元目的翻转时回看的历史窗口大小。
            max_history: precision 历史上限，防止无界增长（默认 meta_window*10）。
            habituation_rate: 习惯化衰减率，控制"常遇到→低precision"的速度。
            activation_threshold: 习惯化激活阈值，感官节点激活绝对值超过此值
                才被计入 encounter_count（N4 修复）。当 temperature 较低时
                （如 temperature=0 确定性推断），激活值偏小，应适当降低此阈值
                以保证习惯化机制生效。
            coherence_direction_weight: coherence 方向分量权重（余弦相似度），
                衡量 precision 方向的稳定性。默认 0.5。
            coherence_magnitude_weight: coherence 幅度分量权重（范数比值），
                衡量 precision 幅度的稳定性。默认 0.5。
                方向权重与幅度权重之和无需为 1，内部会归一化以保证
                coherence 落在 [0, 1]。
            per_dim_surprise_scale: 逐维惊讶度映射的尺度常数 k（tanh(raw/k)）。
                默认 0.1（C2 落地时按实测 per_dim_surprise 分布校准：生产景观
                1280 样本 median=0.97/P10=0.70/P90=1.24；k=1.0 拍脑袋初值
                会因 habituation(≈0.13) 压目标 + π 乘性反馈螺旋，使 precision
                百轮内崩塌到下限，已证伪；k=0.1 落在审计预估 0.1~0.3 区间，
                300 轮 A/B 与旧 |σ| 代理行为最接近（0.44 vs 0.36，std 0.045
                vs 0.037），详见惊讶度修复-01-设计方案.md §4.4 校准记录）。
            total_surprise_scale: 全局惊讶度缩放常数（sigmoid(surprise/scale)），
                默认 20.0。生产 surprise（准确性项）实测 26~32，除以 20 后
                sigmoid 取值 0.79~0.83，不饱和；语义"总惊讶度越高，调整幅度
                越大"单调保留。
            device: 计算设备（E-P2-1）。支持 "auto"/"cpu"/"cuda"/"cuda:0"
                或 torch.device。precision 向量、attention 和 encounter_count
                将创建在该设备上。
        """
        self.input_dim = input_dim
        self.precision_lr = precision_lr
        self.precision_min = precision_min
        self.precision_max = precision_max
        self.coherence_threshold = coherence_threshold
        self.min_history_length = min_history_length
        self.meta_window = meta_window
        self.max_history = max_history
        self.habituation_rate = habituation_rate
        self.activation_threshold = activation_threshold
        # 逐维惊讶度映射尺度 k（tanh(raw/k)）与全局缩放常数（sigmoid(surprise/scale)）
        self.per_dim_surprise_scale = per_dim_surprise_scale
        self.total_surprise_scale = total_surprise_scale
        # coherence 混合权重（方向分量 + 幅度分量）
        self.coherence_direction_weight = coherence_direction_weight
        self.coherence_magnitude_weight = coherence_magnitude_weight

        # E-P2-1: 统一设备管理
        self.device: torch.device = resolve_device(device)

        # Layer 1: Sensory Precision —— 初始均匀分布（E-P2-1: 创建在 device 上）
        self.sensory_precision: torch.Tensor = torch.ones(
            input_dim, device=self.device)

        # Layer 2: Attention Allocation —— 从 precision 派生
        self.attention: torch.Tensor = torch.softmax(self.sensory_precision, dim=0)
        self.coherence: float = 1.0  # 初始 coherence 满分

        # Layer 3: Meta-Purpose —— 元目的状态
        self.flipped: bool = False  # 是否发生过翻转
        self.flip_count: int = 0    # 翻转次数

        # 习惯化计数器：跟踪每个维度被"遇到"的次数（G2 修复）
        # E-P2-1: 创建在 device 上
        self.encounter_count: torch.Tensor = torch.zeros(
            input_dim, device=self.device)

        # precision 演化历史
        self.history: list[torch.Tensor] = []

    # ------------------------------------------------------------------ #
    #  Layer 1: 感官精度更新
    # ------------------------------------------------------------------ #

    def _compute_per_dim_surprise(self, activation: Activation) -> torch.Tensor:
        """逐维惊讶度 = π_i·(σ_i−s_i)²（由 attractor.infer 填充），映射到
        [precision_min, precision_max]。

        回退：activation.per_dim_surprise 为 None（旧构造点/旧测试手搓的
        Activation）时，沿用 |σ_i| 激活强度代理（行为降级，不崩溃）。

        映射为单调饱和函数 tanh(raw/k)：surprise_i ∈ [0,∞) →
        [precision_min, precision_max]。k = per_dim_surprise_scale 为尺度
        常数（C2 落地时实测校准，使典型误差落在 tanh 灵敏区）。

        参数:
            activation: 吸引子网络的激活态。

        返回:
            每个感官维度的惊讶度，形状 [input_dim]，值域映射到 precision 范围。
        """
        raw = activation.per_dim_surprise
        if raw is None:
            # 回退分支：旧构造点无 per_dim_surprise，用 |σ_i| 激活强度代理
            # （E-P2-1: 迁移到正确 device）
            sensory_activation = (
                activation.state[:self.input_dim].abs().to(self.device))
            scale = self.precision_max - self.precision_min
            return self.precision_min + sensory_activation * scale
        raw = raw.to(self.device)
        # 单调饱和映射：surprise_i ∈ [0,∞) → [precision_min, precision_max]
        # tanh(raw / k)：k 为尺度常数（per_dim_surprise_scale）。
        # 理由：|σ| 代理天然 ∈(0,1)，而 π(σ−s)² 无量纲上限，需有界单调映射
        # 才能继续喂给 EMA 与 clamp。
        k = self.per_dim_surprise_scale
        mapped = (self.precision_min
                  + (self.precision_max - self.precision_min)
                  * torch.tanh(raw / k))
        return mapped

    # ------------------------------------------------------------------ #
    #  Layer 2: 注意力分配 + 一致性计算
    # ------------------------------------------------------------------ #

    def _compute_coherence(self) -> float:
        """计算目的的内部一致性（coherence）。

        coherence 衡量当前 precision 与近期历史的吻合程度:
          - coherence 高：precision 稳定，目的明确
          - coherence 低：precision 波动，目的不明确

        采用「方向分量 + 幅度分量」的混合度量，以同时反映 precision
        的方向变化与幅度变化:

          1. 方向分量（direction_component）:
             当前 precision 与历史均值的余弦相似度，映射到 [0, 1]。
             衡量 precision **方向**的稳定性——方向越一致，分量越高。
             局限：余弦相似度只对方向敏感，对幅度不敏感。

          2. 幅度分量（magnitude_component）:
             当前 precision 范数与历史均值范数的比值:
                 magnitude_ratio = min(||current||, ||mean||)
                                   / max(||current||, ||mean||)
             范围 (0, 1]。衡量 precision **幅度**的稳定性——幅度越接近，
             分量越高。当习惯化（habituation）导致所有高频维度 precision
             同步下降时，方向几乎不变（余弦≈1）但幅度显著缩小，此时
             幅度分量能正确下降，从而拉低整体 coherence。

          3. 混合 coherence:
                 coherence = w_direction * direction_component
                           + w_magnitude * magnitude_component
             权重在构造函数中配置（coherence_direction_weight /
             coherence_magnitude_weight，默认 0.5/0.5）。内部对权重做
             归一化，保证 coherence 始终落在 [0, 1]。

        这一修复解决了"习惯化导致 precision 全局下降但方向不变，使余弦
        相似度恒为 ~1.0、coherence 恒为 ~1.0、元目的翻转永不触发"的缺陷。

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

        current = self.sensory_precision
        norm_curr = current.norm()
        norm_mean = mean_precision.norm()
        # 范数过小（接近零向量）时方向与幅度均不可靠，视为低一致性
        if norm_curr < 1e-8 or norm_mean < 1e-8:
            return 0.0

        # --- 方向分量：余弦相似度映射到 [0, 1] ---
        cosine = torch.dot(current, mean_precision) / (norm_curr * norm_mean)
        direction_component = float((cosine + 1.0) / 2.0)

        # --- 幅度分量：范数比值，范围 (0, 1] ---
        # min/max 比值保证对称性：无论当前幅度大于还是小于历史均值，
        # 偏离越大分量越低。幅度变化大（如习惯化导致全局下降）时下降。
        norm_curr_val = float(norm_curr)
        norm_mean_val = float(norm_mean)
        magnitude_component = (
            min(norm_curr_val, norm_mean_val)
            / max(norm_curr_val, norm_mean_val)
        )

        # --- 混合 coherence（权重归一化以保证落在 [0, 1]）---
        w_dir = self.coherence_direction_weight
        w_mag = self.coherence_magnitude_weight
        total_w = w_dir + w_mag
        if total_w <= 0.0:
            # 两个权重都为 0 的退化情形：退化为纯方向分量
            coherence = direction_component
        else:
            coherence = (w_dir * direction_component
                         + w_mag * magnitude_component) / total_w

        # 数值保护，确保落在 [0, 1]
        return max(0.0, min(1.0, coherence))

    # ------------------------------------------------------------------ #
    #  Layer 3: 元目的翻转
    # ------------------------------------------------------------------ #

    def _meta_adjust(self) -> None:
        """元目的方向翻转：当 coherence 持续低时触发。

        翻转策略（强化已关注方向，回初版语义 01e6482；
        dandan 拍板 1：元目的翻转 = 强化已关注方向，越关注越专注；
        8/3 起"切换最未探索维度"作废）：
          1. 取历史窗口内平均 precision 最高的维度（已关注方向）
          2. 将该维度 precision 设为最大值（大幅强化）
          3. 适当衰减其他维度（腾出注意力空间）
          4. 标记翻转发生

        依据：high precision deepens attractor basins（母本附录7）——
        越关注越深入；怀疑质检在更深的盆地里做（阶段 D 语义闭环）。

        注意：方法名保持 _meta_adjust 以兼容外部调用，
        语义是"强化已关注方向"（meta-flip，回初版）。
        """
        if len(self.history) < 1:
            return

        # 强化已关注方向：历史窗口内平均 precision 最高的维度
        # （初版 01e6482 语义；precision 三层结构零触碰，单函数内 1 行替换
        # 5 行，git revert 单 commit 可回滚）
        window = min(self.meta_window, len(self.history))
        recent = self.history[-window:]
        history_tensor = torch.stack(recent)          # [window, input_dim]
        avg_precision = history_tensor.mean(dim=0)    # [input_dim]
        target_dim = int(avg_precision.argmax().item())

        # [A15] 强化已关注方向：将历史窗口内平均 precision 最高的维度设为
        # 最大值（meta-flip 强化，非"探索新方向"——旧注释与实现语义矛盾，
        # 实际是强化已关注方向而非探索；只改注释，代码不动）
        self.sensory_precision[target_dim] = self.precision_max

        # 轻微衰减其他维度（腾出注意力空间，但不至于归零）
        # E-P2-1: mask 创建在 device 上
        mask = torch.ones(self.input_dim, device=self.device)
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
          - 习惯化机制：常遇到的维度 → encounter_count 高 → habituation 衰减
            → per_dim_surprise 降低 → precision 降低（实现"已熟悉→低precision"）

        调整规则遵循 FEP 元层面:
          - precision 的更新 = 最小化"元自由能"
          - 元自由能 = 对"我的 precision 是否合适"的惊讶

        这就是"目的随对话共生"的机制。

        参数:
            surprise: 标量惊讶度（准确性项，恒≥0），来自吸引子网络。
            activation: 激活态，用于提取每个感官维度的惊讶度。
        """
        # --- G2: 习惯化计数器更新 ---
        # 统计每个感官维度被"遇到"（显著激活）的次数
        # encounter_count 是累积计数，不随 history 裁剪而重置
        # N4: 阈值可配置（self.activation_threshold），不再硬编码 0.3
        # E-P2-1: 迁移激活态到正确 device
        self.encounter_count += (
            activation.state[:self.input_dim].abs().to(self.device)
            > self.activation_threshold
        ).float()

        # --- Layer 1: 基于 per-dimension 惊讶度调整 sensory precision ---
        per_dim_surprise = self._compute_per_dim_surprise(activation)

        # G2: 习惯化衰减——常遇到的维度 surprise 被抑制
        # habituation = 1 / (1 + encounter_count * rate)
        # encounter_count=0 → habituation=1.0（无衰减）
        # encounter_count=20 → habituation=1/(1+20*0.05)=0.5（衰减一半）
        # 注：误差小本身已反映熟悉（per_dim_surprise 已低），habituation 再乘
        # 一次属双重复合，冗余不有害（惊讶度修复-01-设计方案.md §4.4 注）。
        habituation = 1.0 / (1.0 + self.encounter_count * self.habituation_rate)
        per_dim_surprise = per_dim_surprise * habituation

        # 指数移动平均：precision 向 per_dim_surprise 靠拢
        # surprise 全局缩放因子：总惊讶度越高，调整幅度越大。
        # 现 surprise 为准确性项（量级 O(1~100)），除以 total_surprise_scale
        # （默认 20.0）后落在 sigmoid 灵敏区；旧式 sigmoid(surprise) 在
        # 大输入下恒≈1，全局缩放饱和失效（惊讶度修复-01-设计方案.md §4.4）。
        surprise_scale = torch.sigmoid(
            torch.tensor(surprise) / self.total_surprise_scale)
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

        # --- Layer 3: 元目的方向翻转 ---
        # 重置翻转标记（每次 adjust 重新判断）
        self.flipped = False
        if (self.coherence < self.coherence_threshold
                and len(self.history) >= self.min_history_length):
            self._meta_adjust()
            # 翻转后重新计算 attention 和 coherence（G6 修复）
            self.attention = torch.softmax(self.sensory_precision, dim=0)
            self.coherence = self._compute_coherence()

        # --- 记录历史 ---
        self.history.append(self.sensory_precision.clone())
        # S3: 裁剪历史到上限，防止无界增长
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def get_precision(self) -> torch.Tensor:
        """返回当前 precision 向量。

        返回:
            precision 副本，形状 [input_dim]。
        """
        return self.sensory_precision.clone()

    def get_purpose(self) -> PurposeState:
        """返回当前目的状态。

        返回:
            PurposeState，包含 precision、history、coherence 和 encounter_count。
        """
        return PurposeState(
            precision=self.sensory_precision.clone(),
            history=[p.clone() for p in self.history],
            coherence=self.coherence,
            encounter_count=self.encounter_count.clone(),
        )

    def set_purpose(self, state: PurposeState) -> None:
        """从 PurposeState 恢复目的层状态。

        封装内部属性恢复逻辑，使 persistence 层通过公开接口操作，
        而非依赖脆弱的属性直写。

        E-P2-1: 恢复的张量自动迁移到目的层当前 device。

        参数:
            state: PurposeState，包含 precision、history、coherence 和
                encounter_count（可选，向后兼容旧版快照）。
        """
        self.sensory_precision = state.precision.clone().to(self.device)
        self.history = [p.clone().to(self.device) for p in state.history]
        self.coherence = state.coherence
        # N3: 恢复 encounter_count（向后兼容：旧版快照无此字段时保持初始零值）
        if state.encounter_count is not None:
            self.encounter_count = state.encounter_count.clone().to(self.device)
        # 重算 attention（从恢复的 precision 派生）
        self.attention = torch.softmax(self.sensory_precision, dim=0)

    # ------------------------------------------------------------------ #
    #  设备迁移（E-P2-1）
    # ------------------------------------------------------------------ #

    def to(self, device: Union[str, torch.device]) -> 'PurposeLayer':
        """将目的层所有张量迁移到指定设备（E-P2-1）。

        迁移 sensory_precision、attention、encounter_count 和历史
        precision 快照到目标设备，并更新 self.device。

        参数:
            device: 目标设备（str / torch.device）。

        返回:
            self（支持链式调用）。
        """
        self.device = resolve_device(device)
        self.sensory_precision = self.sensory_precision.to(self.device)
        self.attention = self.attention.to(self.device)
        self.encounter_count = self.encounter_count.to(self.device)
        self.history = [p.to(self.device) for p in self.history]
        return self

    # ------------------------------------------------------------------ #
    #  公开演化接口（供 DreamEngine 等上层模块调用）
    # ------------------------------------------------------------------ #

    def nudge_low_encounter_dim(self, nudge: float) -> None:
        """向 encounter_count 最低的维度偏移 precision（好奇心萌芽）。

        找出 ``encounter_count`` 最低的维度（最未被探索），将其
        ``sensory_precision`` 增加 ``nudge``，鼓励探索尚未被关注的
        方向。偏移后重新 clamp 防止发散，并重算 attention。

        该方法封装了 DreamEngine ``_purpose_evolve`` 中"向低探索维度
        偏移 precision"的逻辑，使上层模块通过公开接口操作，而非直接
        读写 ``sensory_precision`` / ``attention`` 私有属性。

        参数:
            nudge: precision 偏移幅度（正值增加）。通常为小量
                （如 0.05），实现温和的好奇心漂移。
        """
        encounter = self.encounter_count
        if encounter.numel() == 0:
            return
        target_dim = int(encounter.argmin().item())
        self.sensory_precision[target_dim] = (
            self.sensory_precision[target_dim] + nudge
        )
        # clamp 防止发散
        self.sensory_precision = torch.clamp(
            self.sensory_precision,
            self.precision_min,
            self.precision_max,
        )
        # 重新计算 attention
        self.attention = torch.softmax(self.sensory_precision, dim=0)

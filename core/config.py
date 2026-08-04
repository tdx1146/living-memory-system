"""
活体记忆系统 - 核心配置
========================

集中管理核心层的所有可配置参数。
使用 dataclass 保证类型安全与可序列化。

参考：架构文档 第三节《模块划分》、第九节《技术栈》
"""

import os
from pathlib import Path
from dataclasses import dataclass, fields


@dataclass
class CoreConfig:
    """核心层统一配置。

    所有核心模块共享此配置实例，保证参数一致性。
    可通过修改此处的默认值来调整整个系统的行为。

    属性:
        num_nodes: 吸引子网络节点数（建议 256-1024 起步）。
        input_dim: 感官输入维度（= embedding 维度）。
            网络的前 input_dim 个节点为感官节点，接收外部 clamping。
        learning_rate: FEP 学习率 η。
        num_infer_steps: FEP 推断的迭代步数 K。
        short_term_decay: 短时记忆衰减系数（越小遗忘越快）。
        long_term_decay: 长时记忆衰减系数（越接近 1 保留越久）。
        precision_min: precision 下限，防止发散。
        precision_max: precision 上限，防止发散。
        precision_lr: 目的层 precision 调整的学习率。
        complexity_weight: 自由能中复杂性项的权重（L2 正则化系数）。
        orth_weight: 正交化压力权重（复杂性梯度的竞争性抑制强度）。
        coherence_threshold: coherence 低于此阈值时触发元目的翻转。
        min_history_length: 触发元目的翻转所需的最短历史长度。
        meta_window: 元目的翻转时回看的历史窗口大小。
        seed: 随机种子，保证可复现性。
        temperature: Langevin 动力学温度（扩散项噪声强度）。
        max_history: 目的层 precision 历史上限，防止无界增长。
        habituation_rate: 习惯化衰减率，控制"常遇到→低precision"的速度。
        activation_threshold: 习惯化激活阈值，感官节点激活绝对值超过此值
            才被计入 encounter_count。当 temperature 较低时应适当降低此值
            （N4 修复：解耦阈值与 temperature 的硬编码耦合）。
        transfer_rate: 记忆巩固时短时→长时的迁移率。
        replay_count: 记忆巩固时回放的条目数。
        replay_weight: 记忆巩固时回放的权重。
        consolidation_decay: 巩固后短时记忆的衰减系数。
        buffer_capacity: 记忆缓冲区容量。
        meta_enabled: 是否启用元学习（元可塑性）。
        meta_interval: 元更新频率（每 N 轮触发一次元参数调整）。
        meta_lr: 元学习率。
        meta_bounds_min: 元倍率下限（钳制学习规则参数的自适应倍率）。
        meta_bounds_max: 元倍率上限。
        meta_surprise_window: surprise 历史窗口大小。
        meta_orth_alpha: coherence→orth 系数。
        meta_temp_beta: collapse→temp 系数。
        meta_cw_gamma: ||J||→cw 系数。
        meta_lr_delta: surprise 趋势→lr 系数。
        meta_shy_target_norm: SHY 目标范数（元层用）。
        device: 计算设备标识。支持 "auto"(自动检测 CUDA)、"cpu"、"cuda"、
            "cuda:0" 等。组件构造时通过 resolve_device() 解析为 torch.device。
    """

    # 网络结构
    num_nodes: int = 256
    input_dim: int = 64

    # 学习与推断
    learning_rate: float = 0.01
    num_infer_steps: int = 10
    temperature: float = 0.05

    # 记忆管理
    short_term_decay: float = 0.8
    long_term_decay: float = 0.999
    transfer_rate: float = 0.1
    replay_count: int = 10
    replay_weight: float = 0.01
    consolidation_decay: float = 0.5
    buffer_capacity: int = 100

    # 目的层
    precision_min: float = 0.1
    precision_max: float = 10.0
    precision_lr: float = 0.1
    coherence_threshold: float = 0.3
    coherence_direction_weight: float = 0.5
    coherence_magnitude_weight: float = 0.5
    min_history_length: int = 5
    meta_window: int = 10
    max_history: int = 100
    habituation_rate: float = 0.05
    activation_threshold: float = 0.3

    # FEP 学习规则
    complexity_weight: float = 0.01
    orth_weight: float = 0.5

    # 在线熵管理
    entropy_high_threshold: float = 0.9
    entropy_low_threshold: float = 0.5

    # 元可塑性（Meta-Plasticity）
    meta_enabled: bool = True              # 是否启用元学习
    meta_interval: int = 10               # 元更新频率（每N轮）
    meta_lr: float = 0.01                 # 元学习率
    meta_bounds_min: float = 0.5          # 倍率下限
    meta_bounds_max: float = 2.0          # 倍率上限
    meta_surprise_window: int = 20        # surprise历史窗口
    meta_orth_alpha: float = 1.0          # coherence→orth 系数
    meta_temp_beta: float = 5.0           # collapse→temp 系数
    meta_cw_gamma: float = 1.0            # ||J||→cw 系数
    meta_lr_delta: float = 2.0            # surprise趋势→lr 系数
    meta_shy_target_norm: float = 10.0   # SHY目标范数（元层用）

    # 随机种子
    seed: int = 42

    # 设备管理（E-P2-1）：支持 "auto"(自动检测 CUDA)/"cpu"/"cuda"/"cuda:0" 等
    device: str = "auto"

    # ================================================================== #
    #  参数校验
    # ================================================================== #

    def validate(self) -> None:
        """校验全部配置参数的取值约束。

        校验失败时抛出 ValueError，异常信息包含具体字段名与无效值，
        便于快速定位配置错误。

        此方法是可选调用的（不在 __post_init__ 中自动调用），
        以避免破坏依赖默认值或宽松构造的现有代码。建议在部署入口、
        加载外部配置后或测试中显式调用。

        异常:
            ValueError: 任一参数违反约束时抛出。
        """
        def _fail(field_name, value, msg):
            raise ValueError(
                f"CoreConfig 字段 '{field_name}' 无效: "
                f"{value!r}（{msg}）"
            )

        # --- 网络结构 ---
        if not (self.num_nodes > 0):
            _fail('num_nodes', self.num_nodes, '必须 > 0')
        if not (self.input_dim > 0):
            _fail('input_dim', self.input_dim, '必须 > 0')
        if not (self.input_dim <= self.num_nodes):
            _fail('input_dim', self.input_dim,
                  f'必须 <= num_nodes({self.num_nodes})')

        # --- 学习与推断 ---
        if not (self.learning_rate > 0):
            _fail('learning_rate', self.learning_rate, '必须 > 0')
        if not (self.num_infer_steps > 0):
            _fail('num_infer_steps', self.num_infer_steps, '必须 > 0')
        if not (0 < self.temperature < 1):
            _fail('temperature', self.temperature, '必须满足 0 < temperature < 1')

        # --- 记忆管理 ---
        if not (0 < self.short_term_decay < 1):
            _fail('short_term_decay', self.short_term_decay,
                  '必须满足 0 < x < 1')
        if not (0 < self.long_term_decay < 1):
            _fail('long_term_decay', self.long_term_decay,
                  '必须满足 0 < x < 1')
        if not (0 < self.transfer_rate < 1):
            _fail('transfer_rate', self.transfer_rate, '必须满足 0 < x < 1')
        if not (self.replay_count > 0):
            _fail('replay_count', self.replay_count, '必须 > 0')
        if not (0 < self.replay_weight < 1):
            _fail('replay_weight', self.replay_weight, '必须满足 0 < x < 1')
        if not (0 < self.consolidation_decay < 1):
            _fail('consolidation_decay', self.consolidation_decay,
                  '必须满足 0 < x < 1')
        if not (self.buffer_capacity > 0):
            _fail('buffer_capacity', self.buffer_capacity, '必须 > 0')

        # --- 目的层 ---
        if not (self.precision_min > 0):
            _fail('precision_min', self.precision_min, '必须 > 0')
        if not (self.precision_max > self.precision_min):
            _fail('precision_max', self.precision_max,
                  f'必须 > precision_min({self.precision_min})')
        if not (self.precision_lr > 0):
            _fail('precision_lr', self.precision_lr, '必须 > 0')
        if not (0 <= self.coherence_threshold <= 1):
            _fail('coherence_threshold', self.coherence_threshold,
                  '必须满足 0 <= x <= 1')
        if not (self.coherence_direction_weight >= 0):
            _fail('coherence_direction_weight',
                  self.coherence_direction_weight, '必须 >= 0')
        if not (self.coherence_magnitude_weight >= 0):
            _fail('coherence_magnitude_weight',
                  self.coherence_magnitude_weight, '必须 >= 0')
        if not (self.coherence_direction_weight
                + self.coherence_magnitude_weight > 0):
            _fail('coherence_direction_weight + coherence_magnitude_weight',
                  (self.coherence_direction_weight,
                   self.coherence_magnitude_weight),
                  '两者之和必须 > 0')
        if not (self.min_history_length > 0):
            _fail('min_history_length', self.min_history_length, '必须 > 0')
        if not (self.meta_window > 0):
            _fail('meta_window', self.meta_window, '必须 > 0')
        if not (self.max_history > 0):
            _fail('max_history', self.max_history, '必须 > 0')
        if not (self.habituation_rate > 0):
            _fail('habituation_rate', self.habituation_rate, '必须 > 0')
        if not (self.activation_threshold > 0):
            _fail('activation_threshold', self.activation_threshold, '必须 > 0')

        # --- FEP 学习规则 ---
        if not (self.complexity_weight >= 0):
            _fail('complexity_weight', self.complexity_weight, '必须 >= 0')
        if not (self.orth_weight >= 0):
            _fail('orth_weight', self.orth_weight, '必须 >= 0')

        # --- 在线熵管理 ---
        if not (0 < self.entropy_low_threshold
                < self.entropy_high_threshold <= 1):
            _fail('entropy_low_threshold / entropy_high_threshold',
                  (self.entropy_low_threshold, self.entropy_high_threshold),
                  '必须满足 0 < entropy_low_threshold '
                  '< entropy_high_threshold <= 1')

        # --- 元可塑性 ---
        if not (self.meta_interval > 0):
            _fail('meta_interval', self.meta_interval, '必须 > 0')
        if not (self.meta_lr > 0):
            _fail('meta_lr', self.meta_lr, '必须 > 0')
        if not (0 < self.meta_bounds_min < self.meta_bounds_max):
            _fail('meta_bounds_min / meta_bounds_max',
                  (self.meta_bounds_min, self.meta_bounds_max),
                  '必须满足 0 < meta_bounds_min < meta_bounds_max')
        if not (self.meta_surprise_window > 0):
            _fail('meta_surprise_window', self.meta_surprise_window, '必须 > 0')

        # --- 随机种子 ---
        # bool 是 int 的子类，需排除（True/False 不应作为 seed）
        if not (isinstance(self.seed, int)
                and not isinstance(self.seed, bool)):
            _fail('seed', self.seed, '必须为整数')

        # --- 设备管理 ---
        if not (isinstance(self.device, str) and self.device):
            _fail('device', self.device,
                  '必须为非空字符串（如 "auto"/"cpu"/"cuda"/"cuda:0"）')

    # ================================================================== #
    #  设备解析（E-P2-1）
    # ================================================================== #

    def resolve_device(self):
        """解析 device 配置为 ``torch.device`` 对象。

        ``"auto"`` 时自动检测 CUDA 可用性；``"cpu"``/``"cuda"``/``"cuda:0"``
        等直接构造 ``torch.device``。延迟导入 torch，保证本模块在未安装
        torch 的纯 dataclass 场景下仍可被导入。

        返回:
            torch.device 对象。
        """
        import torch
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    # ================================================================== #
    #  配置转换
    # ================================================================== #

    def to_loop_config(self) -> dict:
        """转换为 LivingMemoryLoop 需要的 config dict。

        覆盖 runtime/loop.py 中所有 config.get() 读取的标量键名，
        使返回的字典可直接传给 LivingMemoryLoop(config=...)。

        CoreConfig 不含的运行时字段（consolidation_interval、decoder_mode、
        auto_snapshot 等）以与 loop.py 默认值一致的值补齐，
        保证单一 dict 即可驱动整个循环。

        返回:
            可直接传给 LivingMemoryLoop 的配置字典。
        """
        return {
            # 网络结构
            'num_nodes': self.num_nodes,
            'input_dim': self.input_dim,
            # 学习与推断
            'learning_rate': self.learning_rate,
            'num_infer_steps': self.num_infer_steps,
            'temperature': self.temperature,
            # 记忆管理
            'short_term_decay': self.short_term_decay,
            'long_term_decay': self.long_term_decay,
            'transfer_rate': self.transfer_rate,
            'replay_count': self.replay_count,
            'replay_weight': self.replay_weight,
            'consolidation_decay': self.consolidation_decay,
            'buffer_capacity': self.buffer_capacity,
            # 目的层
            'precision_min': self.precision_min,
            'precision_max': self.precision_max,
            'precision_lr': self.precision_lr,
            'coherence_threshold': self.coherence_threshold,
            'coherence_direction_weight': self.coherence_direction_weight,
            'coherence_magnitude_weight': self.coherence_magnitude_weight,
            'min_history_length': self.min_history_length,
            'meta_window': self.meta_window,
            'max_history': self.max_history,
            'habituation_rate': self.habituation_rate,
            'activation_threshold': self.activation_threshold,
            # FEP 学习规则
            'complexity_weight': self.complexity_weight,
            'orth_weight': self.orth_weight,
            # 在线熵管理
            'entropy_high_threshold': self.entropy_high_threshold,
            'entropy_low_threshold': self.entropy_low_threshold,
            # 元可塑性
            'meta_enabled': self.meta_enabled,
            'meta_interval': self.meta_interval,
            'meta_lr': self.meta_lr,
            'meta_bounds_min': self.meta_bounds_min,
            'meta_bounds_max': self.meta_bounds_max,
            'meta_surprise_window': self.meta_surprise_window,
            'meta_orth_alpha': self.meta_orth_alpha,
            'meta_temp_beta': self.meta_temp_beta,
            'meta_cw_gamma': self.meta_cw_gamma,
            'meta_lr_delta': self.meta_lr_delta,
            'meta_shy_target_norm': self.meta_shy_target_norm,
            # 随机种子
            'seed': self.seed,
            # 设备管理（E-P2-1）
            'device': self.device,
            # 运行时默认值（CoreConfig 不含这些字段，补齐 loop.py 默认值）
            'consolidation_interval': 5,
            'decoder_mode': 'text',
            'auto_snapshot': False,
            'auto_snapshot_interval': 50,
            'snapshot_dir': str(Path.home() / '.lms' / 'snapshots'),
        }

    # ================================================================== #
    #  环境变量加载
    # ================================================================== #

    @classmethod
    def from_env(cls, **overrides) -> 'CoreConfig':
        """从环境变量读取配置覆盖值，构造 CoreConfig 实例。

        环境变量命名规则：字段名大写并加前缀 ``LMS_``。例如：
            - LMS_NUM_NODES -> num_nodes
            - LMS_LEARNING_RATE -> learning_rate
            - LMS_TEMPERATURE -> temperature
            - LMS_META_ENABLED -> meta_enabled
            - LMS_SEED -> seed

        类型转换依据 dataclass 字段类型注解自动完成（int / float / bool）。
        布尔值接受（不区分大小写）：1/true/yes/on 视为 True，其余为 False。

        优先级：显式 overrides > 环境变量 > dataclass 默认值。

        参数:
            **overrides: 显式覆盖值，优先级高于环境变量。

        返回:
            CoreConfig 实例。
        """
        kwargs = {}
        for f in fields(cls):
            env_name = 'LMS_' + f.name.upper()
            raw = os.environ.get(env_name)
            if raw is None:
                continue
            ftype = f.type
            # 兼容字符串形式类型注解（from __future__ import annotations）
            if isinstance(ftype, str):
                if ftype == 'bool':
                    ftype = bool
                elif ftype == 'int':
                    ftype = int
                elif ftype == 'float':
                    ftype = float
            if ftype is bool:
                kwargs[f.name] = raw.strip().lower() in (
                    '1', 'true', 'yes', 'on')
            elif ftype is int:
                kwargs[f.name] = int(raw)
            elif ftype is float:
                kwargs[f.name] = float(raw)
            else:
                kwargs[f.name] = raw
        # 显式 overrides 优先级最高
        kwargs.update(overrides)
        return cls(**kwargs)

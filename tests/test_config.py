"""
CoreConfig 校验与统一配置体系测试
==================================

被测模块: core/config.py

覆盖方面:
  1. validate() 默认配置通过        (TestValidateDefault)
  2. 各种非法参数被正确拒绝          (TestValidateRejectsInvalid)
  3. validate() 不在 __post_init__ 自动调用 (TestOptionalValidation)
  4. to_loop_config() 键完整性       (TestToLoopConfig)
  5. from_env() 环境变量覆盖         (TestFromEnv)

约束:
  - 不依赖 torch（纯 dataclass 逻辑）
  - 不修改其他源文件
  - validate() 为可选调用，不影响现有构造行为
"""

import re

import pytest

from core.config import CoreConfig


# ==================================================================
#  LivingMemoryLoop 通过 config.get() 读取的全部标量键名
#  （摘自 runtime/loop.py 的 __init__ 与 process_turn，
#   排除对象注入键：attractor/purpose/memory/tokenizer/embedder/
#   encoder/decoder/llm_bridge/llm_api）
# ==================================================================
LOOP_SCALAR_KEYS = {
    # 网络结构
    'num_nodes', 'input_dim',
    # 学习与推断
    'learning_rate', 'num_infer_steps', 'temperature',
    # 记忆管理
    'short_term_decay', 'long_term_decay', 'transfer_rate',
    'replay_count', 'replay_weight', 'consolidation_decay',
    'buffer_capacity',
    # 目的层
    'precision_min', 'precision_max', 'precision_lr',
    'coherence_threshold', 'coherence_direction_weight',
    'coherence_magnitude_weight', 'min_history_length',
    'meta_window', 'max_history', 'habituation_rate',
    'activation_threshold',
    # FEP 学习规则
    'complexity_weight', 'orth_weight',
    # 在线熵管理
    'entropy_high_threshold', 'entropy_low_threshold',
    # 元可塑性
    'meta_enabled', 'meta_interval', 'meta_lr',
    'meta_bounds_min', 'meta_bounds_max', 'meta_surprise_window',
    'meta_orth_alpha', 'meta_temp_beta', 'meta_cw_gamma',
    'meta_lr_delta', 'meta_shy_target_norm',
    # 随机种子
    'seed',
    # 运行时（CoreConfig 不含，由 to_loop_config 补齐默认值）
    'consolidation_interval', 'decoder_mode',
    'auto_snapshot', 'auto_snapshot_interval', 'snapshot_dir',
}


# ==================================================================
#  1. 默认配置校验通过
# ==================================================================
class TestValidateDefault:
    """验证默认 CoreConfig 能通过 validate()。"""

    def test_default_config_validates(self):
        """默认配置的全部参数都满足约束，validate() 不抛异常。"""
        config = CoreConfig()
        config.validate()  # 不抛异常即通过

    def test_validate_returns_none(self):
        """validate() 成功时返回 None。"""
        config = CoreConfig()
        assert config.validate() is None

    def test_custom_valid_config(self):
        """自定义但合法的配置也能通过校验。"""
        config = CoreConfig(
            num_nodes=128,
            input_dim=64,
            learning_rate=0.001,
            temperature=0.1,
            short_term_decay=0.5,
            long_term_decay=0.99,
            entropy_low_threshold=0.3,
            entropy_high_threshold=0.8,
            meta_bounds_min=0.6,
            meta_bounds_max=1.5,
            coherence_direction_weight=1.0,
            coherence_magnitude_weight=0.0,
        )
        config.validate()


# ==================================================================
#  2. 非法参数被正确拒绝
# ==================================================================
class TestValidateRejectsInvalid:
    """验证各种非法参数触发 ValueError，且异常信息含字段名。"""

    @pytest.mark.parametrize("overrides, expected_field", [
        # --- 网络结构 ---
        ({'num_nodes': 0}, 'num_nodes'),
        ({'num_nodes': -1}, 'num_nodes'),
        ({'input_dim': 0}, 'input_dim'),
        ({'num_nodes': 64, 'input_dim': 128}, 'input_dim'),  # > num_nodes
        # --- 学习与推断 ---
        ({'learning_rate': 0}, 'learning_rate'),
        ({'learning_rate': -0.01}, 'learning_rate'),
        ({'num_infer_steps': 0}, 'num_infer_steps'),
        ({'num_infer_steps': -5}, 'num_infer_steps'),
        ({'temperature': 0}, 'temperature'),
        ({'temperature': 1}, 'temperature'),
        ({'temperature': 1.5}, 'temperature'),
        ({'temperature': -0.1}, 'temperature'),
        # --- 记忆管理 ---
        ({'short_term_decay': 0}, 'short_term_decay'),
        ({'short_term_decay': 1}, 'short_term_decay'),
        ({'short_term_decay': 1.5}, 'short_term_decay'),
        ({'long_term_decay': 0}, 'long_term_decay'),
        ({'long_term_decay': 1.5}, 'long_term_decay'),
        ({'transfer_rate': 0}, 'transfer_rate'),
        ({'transfer_rate': 1}, 'transfer_rate'),
        ({'replay_count': 0}, 'replay_count'),
        ({'replay_count': -3}, 'replay_count'),
        ({'replay_weight': 0}, 'replay_weight'),
        ({'replay_weight': 1}, 'replay_weight'),
        ({'consolidation_decay': 0}, 'consolidation_decay'),
        ({'consolidation_decay': 1.5}, 'consolidation_decay'),
        ({'buffer_capacity': 0}, 'buffer_capacity'),
        ({'buffer_capacity': -1}, 'buffer_capacity'),
        # --- 目的层 ---
        ({'precision_min': 0}, 'precision_min'),
        ({'precision_min': -0.1}, 'precision_min'),
        ({'precision_max': 0.05, 'precision_min': 0.1}, 'precision_max'),
        ({'precision_lr': 0}, 'precision_lr'),
        ({'coherence_threshold': -0.1}, 'coherence_threshold'),
        ({'coherence_threshold': 1.5}, 'coherence_threshold'),
        ({'coherence_direction_weight': -0.1}, 'coherence_direction_weight'),
        ({'coherence_magnitude_weight': -0.1}, 'coherence_magnitude_weight'),
        # 两者之和为 0
        ({'coherence_direction_weight': 0,
          'coherence_magnitude_weight': 0},
         'coherence_direction_weight + coherence_magnitude_weight'),
        ({'min_history_length': 0}, 'min_history_length'),
        ({'meta_window': 0}, 'meta_window'),
        ({'max_history': 0}, 'max_history'),
        ({'habituation_rate': 0}, 'habituation_rate'),
        ({'activation_threshold': 0}, 'activation_threshold'),
        # --- FEP 学习规则 ---
        ({'complexity_weight': -0.01}, 'complexity_weight'),
        ({'orth_weight': -0.1}, 'orth_weight'),
        # --- 在线熵管理 ---
        ({'entropy_low_threshold': 0},
         'entropy_low_threshold / entropy_high_threshold'),
        ({'entropy_high_threshold': 1.5},
         'entropy_low_threshold / entropy_high_threshold'),
        ({'entropy_low_threshold': 0.9, 'entropy_high_threshold': 0.5},
         'entropy_low_threshold / entropy_high_threshold'),
        ({'entropy_low_threshold': 0.5, 'entropy_high_threshold': 0.5},
         'entropy_low_threshold / entropy_high_threshold'),
        # --- 元可塑性 ---
        ({'meta_interval': 0}, 'meta_interval'),
        ({'meta_lr': 0}, 'meta_lr'),
        ({'meta_bounds_min': 0}, 'meta_bounds_min / meta_bounds_max'),
        ({'meta_bounds_min': 2.0, 'meta_bounds_max': 0.5},
         'meta_bounds_min / meta_bounds_max'),
        ({'meta_bounds_min': 1.0, 'meta_bounds_max': 1.0},
         'meta_bounds_min / meta_bounds_max'),
        ({'meta_surprise_window': 0}, 'meta_surprise_window'),
        # --- 随机种子 ---
        ({'seed': 1.5}, 'seed'),
        ({'seed': 'abc'}, 'seed'),
        ({'seed': True}, 'seed'),
    ])
    def test_invalid_param_raises(self, overrides, expected_field):
        """非法参数触发 ValueError，且异常信息包含字段名。"""
        config = CoreConfig(**overrides)
        with pytest.raises(ValueError, match=re.escape(expected_field)):
            config.validate()

    def test_error_message_contains_value(self):
        """异常信息包含无效值本身，便于定位。"""
        config = CoreConfig(num_nodes=0)
        with pytest.raises(ValueError) as exc_info:
            config.validate()
        msg = str(exc_info.value)
        assert 'num_nodes' in msg
        assert '0' in msg

    def test_boundary_values_accepted(self):
        """边界合法值应被接受（coherence_threshold=0 和 =1）。"""
        CoreConfig(coherence_threshold=0.0).validate()
        CoreConfig(coherence_threshold=1.0).validate()
        # complexity_weight=0 / orth_weight=0 合法
        CoreConfig(complexity_weight=0.0, orth_weight=0.0).validate()
        # entropy_high_threshold=1.0 合法（<= 1）
        CoreConfig(entropy_low_threshold=0.5,
                   entropy_high_threshold=1.0).validate()
        # coherence 一个为 0、另一个 > 0，合法
        CoreConfig(coherence_direction_weight=0.0,
                   coherence_magnitude_weight=0.5).validate()


# ==================================================================
#  3. validate() 为可选调用
# ==================================================================
class TestOptionalValidation:
    """验证 validate() 不在 __post_init__ 中自动调用。"""

    def test_invalid_config_constructable(self):
        """非法配置仍可正常构造（不自动校验），保持向后兼容。"""
        # 这些值若自动校验会抛异常，但构造时不抛
        config = CoreConfig(num_nodes=-1, temperature=5.0)
        assert config.num_nodes == -1
        assert config.temperature == 5.0

    def test_validate_must_be_called_explicitly(self):
        """只有显式调用 validate() 才会抛异常。"""
        config = CoreConfig(learning_rate=-1)
        # 不调用 validate() 不抛
        assert config.learning_rate == -1
        # 调用后才抛
        with pytest.raises(ValueError, match='learning_rate'):
            config.validate()


# ==================================================================
#  4. to_loop_config() 键完整性
# ==================================================================
class TestToLoopConfig:
    """验证 to_loop_config() 返回的 dict 覆盖 loop.py 全部标量键。"""

    def test_contains_all_loop_keys(self):
        """返回的 dict 包含 loop.py 需要的所有标量键。"""
        config = CoreConfig()
        result = config.to_loop_config()
        missing = LOOP_SCALAR_KEYS - set(result.keys())
        assert not missing, f"to_loop_config() 缺少键: {missing}"

    def test_no_extra_object_injection_keys(self):
        """不应包含对象注入键（attractor/purpose/memory 等）。"""
        config = CoreConfig()
        result = config.to_loop_config()
        injection_keys = {
            'attractor', 'purpose', 'memory', 'tokenizer', 'embedder',
            'encoder', 'decoder', 'llm_bridge', 'llm_api',
        }
        extra = injection_keys & set(result.keys())
        assert not extra, f"to_loop_config() 不应包含注入键: {extra}"

    def test_values_match_config_fields(self):
        """返回 dict 的值与 CoreConfig 字段值一致。"""
        config = CoreConfig(
            num_nodes=128,
            input_dim=32,
            learning_rate=0.02,
            temperature=0.1,
            seed=7,
        )
        result = config.to_loop_config()
        assert result['num_nodes'] == 128
        assert result['input_dim'] == 32
        assert result['learning_rate'] == 0.02
        assert result['temperature'] == 0.1
        assert result['seed'] == 7

    def test_runtime_defaults_present(self):
        """运行时默认键以与 loop.py 一致的默认值补齐。"""
        config = CoreConfig()
        result = config.to_loop_config()
        assert result['consolidation_interval'] == 5
        assert result['decoder_mode'] == 'text'
        assert result['auto_snapshot'] is False
        assert result['auto_snapshot_interval'] == 50
        assert isinstance(result['snapshot_dir'], str)

    def test_meta_keys_present(self):
        """元可塑性相关键全部存在。"""
        config = CoreConfig()
        result = config.to_loop_config()
        meta_keys = [
            'meta_enabled', 'meta_interval', 'meta_lr',
            'meta_bounds_min', 'meta_bounds_max', 'meta_surprise_window',
            'meta_orth_alpha', 'meta_temp_beta', 'meta_cw_gamma',
            'meta_lr_delta', 'meta_shy_target_norm',
        ]
        for k in meta_keys:
            assert k in result, f"缺少元可塑性键: {k}"

    def test_can_drive_living_memory_loop(self):
        """to_loop_config() 的输出能实际驱动 LivingMemoryLoop 构造。

        （需 torch；跳过条件由 conftest 的导入隐式保证。）
        """
        config = CoreConfig(num_nodes=64, input_dim=32, num_infer_steps=3)
        loop_dict = config.to_loop_config()
        from runtime.loop import LivingMemoryLoop
        loop = LivingMemoryLoop(loop_dict)
        assert loop.attractor.num_nodes == 64
        assert loop.attractor.input_dim == 32
        # 清理避免残留状态
        del loop


# ==================================================================
#  5. from_env() 环境变量覆盖
# ==================================================================
class TestFromEnv:
    """验证 from_env() 能从环境变量读取覆盖值。"""

    def test_no_env_returns_defaults(self, monkeypatch):
        """无任何 LMS_ 环境变量时返回默认配置。"""
        # 清除所有可能的 LMS_ 变量（基于 CoreConfig 字段名生成）
        from dataclasses import fields as dc_fields
        for f in dc_fields(CoreConfig):
            monkeypatch.delenv('LMS_' + f.name.upper(), raising=False)
        config = CoreConfig.from_env()
        default = CoreConfig()
        assert config.num_nodes == default.num_nodes
        assert config.learning_rate == default.learning_rate
        assert config.seed == default.seed

    def test_int_env_override(self, monkeypatch):
        """整数类型环境变量被正确读取。"""
        monkeypatch.setenv('LMS_NUM_NODES', '512')
        monkeypatch.setenv('LMS_INPUT_DIM', '128')
        monkeypatch.setenv('LMS_SEED', '99')
        monkeypatch.setenv('LMS_REPLAY_COUNT', '20')
        config = CoreConfig.from_env()
        assert config.num_nodes == 512
        assert config.input_dim == 128
        assert config.seed == 99
        assert config.replay_count == 20
        assert isinstance(config.num_nodes, int)
        assert isinstance(config.seed, int)

    def test_float_env_override(self, monkeypatch):
        """浮点类型环境变量被正确读取。"""
        monkeypatch.setenv('LMS_LEARNING_RATE', '0.05')
        monkeypatch.setenv('LMS_TEMPERATURE', '0.2')
        monkeypatch.setenv('LMS_SHORT_TERM_DECAY', '0.7')
        monkeypatch.setenv('LMS_PRECISION_MIN', '0.5')
        config = CoreConfig.from_env()
        assert config.learning_rate == pytest.approx(0.05)
        assert config.temperature == pytest.approx(0.2)
        assert config.short_term_decay == pytest.approx(0.7)
        assert config.precision_min == pytest.approx(0.5)
        assert isinstance(config.learning_rate, float)
        assert isinstance(config.temperature, float)

    @pytest.mark.parametrize("raw, expected", [
        ('1', True),
        ('true', True),
        ('True', True),
        ('TRUE', True),
        ('yes', True),
        ('on', True),
        ('0', False),
        ('false', False),
        ('no', False),
        ('off', False),
        ('', False),
        ('anything_else', False),
    ])
    def test_bool_env_override(self, monkeypatch, raw, expected):
        """布尔类型环境变量的各种取值都被正确解析。"""
        monkeypatch.setenv('LMS_META_ENABLED', raw)
        config = CoreConfig.from_env()
        assert config.meta_enabled is expected
        assert isinstance(config.meta_enabled, bool)

    def test_overrides_take_priority_over_env(self, monkeypatch):
        """显式 overrides 优先级高于环境变量。"""
        monkeypatch.setenv('LMS_NUM_NODES', '512')
        config = CoreConfig.from_env(num_nodes=256)
        assert config.num_nodes == 256  # overrides 胜出

    def test_partial_env_override(self, monkeypatch):
        """仅设置部分环境变量时，其余字段保持默认值。"""
        monkeypatch.setenv('LMS_TEMPERATURE', '0.15')
        config = CoreConfig.from_env()
        default = CoreConfig()
        assert config.temperature == pytest.approx(0.15)
        # 未设置的保持默认
        assert config.num_nodes == default.num_nodes
        assert config.learning_rate == default.learning_rate

    def test_env_config_validates(self, monkeypatch):
        """从环境变量加载的合法配置能通过 validate()。"""
        monkeypatch.setenv('LMS_NUM_NODES', '128')
        monkeypatch.setenv('LMS_INPUT_DIM', '64')
        monkeypatch.setenv('LMS_LEARNING_RATE', '0.02')
        monkeypatch.setenv('LMS_TEMPERATURE', '0.1')
        config = CoreConfig.from_env()
        config.validate()  # 不抛异常即通过

    def test_env_prefix_is_lms(self, monkeypatch):
        """非 LMS_ 前缀的环境变量被忽略。"""
        monkeypatch.setenv('NUM_NODES', '999')  # 无前缀
        monkeypatch.setenv('LMS_NUM_NODES', '128')
        config = CoreConfig.from_env()
        assert config.num_nodes == 128  # 只读 LMS_ 前缀的

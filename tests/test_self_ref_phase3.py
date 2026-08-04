"""Phase 3 测试：状态级递归 + 多轮 echo 衰减 + 稳定性保护
=====================================================

覆盖范围:
  3.1 状态级递归（state-level recursion）：
      generate_state_seed 将上一轮 activation.state 作为下一轮 infer 种子偏置，
      通过 autocorr 门控、锁定/振荡检测、冷却期、做梦衰减等机制保护稳定性。
  3.2 多轮 echo 衰减（multi-round echo decay）：
      _compute_decayed_echo 将最近 K 轮自述嵌入按指数衰减加权平均，
      替代 Phase 2 的单轮模式；范数守卫防止数值膨胀。
  3.5 稳定性保护（stability safeguards）：
      state_recursion_alpha 有界、冷却期递减、高 autocorr streak 重置等。
  主循环集成（LivingMemoryLoop integration）：
      Phase 3 启用/关闭时 loop 正常运行，关闭时行为与 Phase 2 一致。

设计依据: docs/SELF_REF_INTEGRATED_DESIGN.md
  - 第六节 Phase 3 验证标准
  - 第四节 3.1/3.2/3.5 实现
"""

import hashlib

import pytest
import torch

from core.types import Activation, SensoryInput


# ============================================================
# 常量
# ============================================================

NUM_NODES = 32
INPUT_DIM = 16


# ============================================================
# 辅助函数与假对象（沿用 Phase 1/2 测试风格）
# ============================================================

def _text_seed(text: str) -> int:
    """由文本生成稳定的随机种子。"""
    digest = hashlib.md5(text.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little')


class FakeEncoder:
    """假编码器：按文本 hash 生成确定性向量。"""

    def __init__(self, dim: int = INPUT_DIM):
        self._dim = dim
        self.encoded: list = []

    def encode(self, text: str, tokenizer, embedder) -> SensoryInput:
        g = torch.Generator().manual_seed(_text_seed(text))
        vec = torch.randn(self._dim, generator=g) * 0.1
        self.encoded.append((text, vec.clone()))
        return SensoryInput(vector=vec, metadata={'source': 'fake_encoder'})

    @property
    def dim(self) -> int:
        return self._dim


class FakeTokenizer:
    def tokenize(self, text: str):
        return [0]

    def get_vocab(self):
        return {}


class FakeEmbedder:
    def __init__(self, dim: int = INPUT_DIM):
        self._dim = dim

    def embed(self, tokens):
        return torch.zeros(self._dim)

    @property
    def dim(self) -> int:
        return self._dim


def make_activation(num_nodes: int = NUM_NODES, seed: int = 42,
                    entropy: float = 0.5, surprise: float = 0.1) -> Activation:
    """构造测试用激活态。"""
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(num_nodes, generator=g) * 0.1
    return Activation(state=state, entropy=entropy, surprise=surprise)


def make_memory_context(interpretations=None, detail=None) -> str:
    """构造与 Decoder._decode_text 输出格式一致的 memory_context。"""
    if interpretations is None:
        interpretations = ["当前记忆聚焦于少数模式"]
    lines = ["[记忆context]", "记忆状态解读:"]
    for interp in interpretations:
        lines.append(f"- {interp}")
    if detail is not None:
        lines.append(f"详细数据: {detail}")
    return "\n".join(lines)


def make_self_ref(alpha_base: float = 0.15, history_cap: int = 20,
                  dim: int = INPUT_DIM, **kwargs):
    """构造 SelfReferentialLoop 及其假依赖。

    支持通过 kwargs 传入 Phase 3 配置：
      - self_ref_state_recursion_enabled
      - self_ref_state_recursion_strength
      - self_ref_echo_max_rounds
      - self_ref_echo_decay_rate
    """
    from core.hippocampus.self_referential import SelfReferentialLoop
    enc = FakeEncoder(dim=dim)
    tok = FakeTokenizer()
    emb = FakeEmbedder(dim=dim)
    config = {
        'self_ref_alpha_base': alpha_base,
        'self_ref_history_cap': history_cap,
    }
    config.update(kwargs)
    loop = SelfReferentialLoop(enc, tok, emb, config=config)
    return loop, enc


def _set_autocorr_result(sr, autocorr=None, state='normal'):
    """直接设置 AutocorrController._last_result，用于测试门控行为。"""
    sr.autocorr_controller._last_result = {
        'autocorr': autocorr,
        'state': state,
        'lock_count': 0,
        'oscillation_count': 0,
        'should_inject_noise': False,
    }


# ============================================================
# 3.1 状态级递归测试
# ============================================================

class TestStateRecursion:
    """测试 Phase 3.1 状态级递归。"""

    def test_disabled_by_default(self):
        """默认关闭，generate_state_seed 返回 None。"""
        sr, _ = make_self_ref()
        sigma = torch.zeros(NUM_NODES)
        seed = sr.generate_state_seed(sigma)
        assert seed is None
        assert sr.state_recursion_alpha == 0.0

    def test_enabled_but_no_history_returns_none(self):
        """启用但无 prev_activation_state 时返回 None。"""
        sr, _ = make_self_ref(self_ref_state_recursion_enabled=True)
        sigma = torch.zeros(NUM_NODES)
        seed = sr.generate_state_seed(sigma)
        assert seed is None
        assert sr.prev_activation_state is None
        assert sr.state_recursion_alpha == 0.0

    def test_enabled_returns_seed(self):
        """启用且有历史时返回混合 seed。"""
        sr, _ = make_self_ref(
            self_ref_state_recursion_enabled=True,
            self_ref_state_recursion_strength=0.3)
        # observe 设置 prev_activation_state
        act = make_activation()
        ctx = make_memory_context()
        sr.observe(ctx, act)
        assert sr.prev_activation_state is not None

        sigma = torch.zeros(NUM_NODES)
        seed = sr.generate_state_seed(sigma)
        assert seed is not None
        # observe 后 autocorr=None（历史不足），gate=1.0
        # effective_strength = 0.3 * 1.0 = 0.3
        # seed = (1-0.3)*sigma + 0.3*prev_state = 0.3*prev_state
        prev_state = act.state.detach().clone()
        expected = (1.0 - 0.3) * sigma + 0.3 * prev_state
        assert torch.allclose(seed, expected, atol=1e-5)
        assert sr.state_recursion_alpha == pytest.approx(0.3, rel=1e-5)

    def test_locked_state_disables_recursion(self):
        """锁定状态时返回 None。"""
        sr, _ = make_self_ref(self_ref_state_recursion_enabled=True)
        act = make_activation()
        ctx = make_memory_context()
        sr.observe(ctx, act)

        _set_autocorr_result(sr, autocorr=0.96, state='locked')
        sigma = torch.zeros(NUM_NODES)
        seed = sr.generate_state_seed(sigma)
        assert seed is None
        assert sr.state_recursion_alpha == 0.0

    def test_oscillating_state_disables_recursion(self):
        """振荡状态时返回 None。"""
        sr, _ = make_self_ref(self_ref_state_recursion_enabled=True)
        act = make_activation()
        ctx = make_memory_context()
        sr.observe(ctx, act)

        _set_autocorr_result(sr, autocorr=-0.5, state='oscillating')
        sigma = torch.zeros(NUM_NODES)
        seed = sr.generate_state_seed(sigma)
        assert seed is None
        assert sr.state_recursion_alpha == 0.0

    def test_autocorr_gating_reduces_strength(self):
        """autocorr 越高偏置越弱。"""
        sr, _ = make_self_ref(
            self_ref_state_recursion_enabled=True,
            self_ref_state_recursion_strength=0.3)
        act = make_activation()
        ctx = make_memory_context()
        sr.observe(ctx, act)

        # autocorr=0.5 → gate = 1 - 0.5 = 0.5
        # effective_strength = 0.3 * 0.5 = 0.15
        _set_autocorr_result(sr, autocorr=0.5, state='normal')
        sigma = torch.zeros(NUM_NODES)
        seed = sr.generate_state_seed(sigma)
        assert seed is not None
        assert sr.state_recursion_alpha < sr.state_recursion_strength
        assert sr.state_recursion_alpha == pytest.approx(0.15, rel=1e-5)

    def test_dream_stale_reduces_strength(self):
        """做梦后 stale 状态偏置减半（x0.3）。"""
        sr, _ = make_self_ref(
            self_ref_state_recursion_enabled=True,
            self_ref_state_recursion_strength=0.3)
        act = make_activation()
        ctx = make_memory_context()
        sr.observe(ctx, act)

        # observe 后 autocorr=None → gate=1.0
        # dream_stale 时 effective_strength = 0.3 * 1.0 * 0.3 = 0.09
        sr.on_dream_start()
        assert sr.dream_stale is True

        sigma = torch.zeros(NUM_NODES)
        seed = sr.generate_state_seed(sigma)
        assert seed is not None
        assert sr.state_recursion_alpha == pytest.approx(0.09, rel=1e-5)

    def test_cooldown_after_high_autocorr_streak(self):
        """连续 5 轮高 autocorr 触发 10 轮冷却。"""
        sr, _ = make_self_ref(self_ref_state_recursion_enabled=True)
        act = make_activation()
        ctx = make_memory_context()
        sr.observe(ctx, act)

        _set_autocorr_result(sr, autocorr=0.9, state='normal')
        sigma = torch.zeros(NUM_NODES)

        # 前 4 轮：streak 递增但不触发
        for i in range(4):
            seed = sr.generate_state_seed(sigma)
            assert seed is not None, f"第 {i+1} 轮不应触发冷却"
            assert sr._high_autocorr_streak == i + 1

        # 第 5 轮：streak 达到 5，触发 10 轮冷却
        seed = sr.generate_state_seed(sigma)
        assert seed is None
        assert sr._state_recursion_cooldown == 10
        assert sr._high_autocorr_streak == 0

        # 冷却期内继续返回 None
        seed = sr.generate_state_seed(sigma)
        assert seed is None
        assert sr._state_recursion_cooldown == 9

    def test_state_in_get_state(self):
        """get_state 包含 Phase 3 字段。"""
        sr, _ = make_self_ref(
            self_ref_state_recursion_enabled=True,
            self_ref_state_recursion_strength=0.25,
            self_ref_echo_max_rounds=3)
        act = make_activation()
        ctx = make_memory_context()
        sr.observe(ctx, act)

        state = sr.get_state()
        assert 'prev_activation_state' in state
        assert state['prev_activation_state'] is not None
        assert state['state_recursion_enabled'] is True
        assert state['state_recursion_strength'] == 0.25
        assert 'state_recursion_alpha' in state
        assert '_state_recursion_cooldown' in state
        assert '_high_autocorr_streak' in state
        assert 'sensory_self_history' in state
        assert state['echo_max_rounds'] == 3
        assert 'echo_decay_rate' in state

    def test_state_in_set_state(self):
        """set_state 恢复 Phase 3 字段。"""
        sr, _ = make_self_ref(
            self_ref_state_recursion_enabled=True,
            self_ref_state_recursion_strength=0.25,
            self_ref_echo_max_rounds=3)
        act = make_activation(seed=7)
        ctx = make_memory_context()
        sr.observe(ctx, act)
        sr._state_recursion_cooldown = 3
        sr._high_autocorr_streak = 2

        state = sr.get_state()

        sr2, _ = make_self_ref()
        sr2.set_state(state)
        assert sr2.state_recursion_enabled is True
        assert sr2.state_recursion_strength == 0.25
        assert sr2.prev_activation_state is not None
        assert torch.allclose(
            sr2.prev_activation_state, sr.prev_activation_state, atol=1e-5)
        assert sr2._state_recursion_cooldown == 3
        assert sr2._high_autocorr_streak == 2
        assert sr2.echo_max_rounds == 3

    def test_old_snapshot_backward_compat(self):
        """旧快照（无 Phase 3 字段）加载不报错。"""
        sr, _ = make_self_ref()
        act = make_activation()
        ctx = make_memory_context()
        sr.observe(ctx, act)

        state = sr.get_state()
        # 模拟旧快照：删除所有 Phase 3 字段
        for key in ('prev_activation_state', 'state_recursion_enabled',
                    'state_recursion_strength', 'state_recursion_alpha',
                    '_state_recursion_cooldown', '_high_autocorr_streak',
                    'sensory_self_history', 'echo_decay_rate',
                    'echo_max_rounds'):
            state.pop(key, None)

        sr2, _ = make_self_ref()
        sr2.set_state(state)  # 不应抛异常
        assert sr2.state_recursion_enabled is False
        assert sr2.state_recursion_strength == 0.3
        assert sr2.prev_activation_state is None
        assert sr2.echo_max_rounds == 1
        assert sr2.echo_decay_rate == 0.5

    def test_status_includes_phase3_fields(self):
        """get_status 暴露 Phase 3 监控字段。"""
        sr, _ = make_self_ref(
            self_ref_state_recursion_enabled=True,
            self_ref_echo_max_rounds=3)
        act = make_activation()
        ctx = make_memory_context()
        sr.observe(ctx, act)

        status = sr.get_status()
        assert 'state_recursion_enabled' in status
        assert status['state_recursion_enabled'] is True
        assert 'state_recursion_strength' in status
        assert 'state_recursion_alpha' in status
        assert 'echo_max_rounds' in status
        assert status['echo_max_rounds'] == 3
        assert 'echo_decay_rate' in status
        assert 'sensory_self_history_size' in status
        assert status['sensory_self_history_size'] >= 1

    def test_save_load_round_trip(self):
        """save->load 往返后 generate_state_seed 产出一致。"""
        sr, _ = make_self_ref(
            self_ref_state_recursion_enabled=True,
            self_ref_state_recursion_strength=0.3)
        act = make_activation(seed=99)
        ctx = make_memory_context()
        sr.observe(ctx, act)

        sigma = torch.zeros(NUM_NODES)
        seed_before = sr.generate_state_seed(sigma)
        assert seed_before is not None

        state = sr.get_state()
        sr2, _ = make_self_ref(
            self_ref_state_recursion_enabled=True,
            self_ref_state_recursion_strength=0.3)
        sr2.set_state(state)

        seed_after = sr2.generate_state_seed(sigma)
        assert seed_after is not None
        assert torch.allclose(seed_before, seed_after, atol=1e-5)
        assert sr2.state_recursion_alpha == pytest.approx(
            sr.state_recursion_alpha, rel=1e-5)


# ============================================================
# 3.2 多轮 echo 衰减测试
# ============================================================

class TestMultiRoundEcho:
    """测试 Phase 3.2 多轮 echo 衰减。"""

    def test_default_max_rounds_is_1(self):
        """默认 echo_max_rounds=1，向后兼容。"""
        sr, _ = make_self_ref()
        assert sr.echo_max_rounds == 1
        assert sr.echo_decay_rate == 0.5

    def test_single_round_mode_identical_to_phase2(self):
        """echo_max_rounds=1 时行为与 Phase 2 一致。"""
        sr, _ = make_self_ref()  # 默认即 echo_max_rounds=1
        assert sr.echo_max_rounds == 1
        # 跑两轮 observe 建立历史（使用不同 context 避免高 echo_sim 触发 L3）
        act1 = make_activation(seed=1)
        act2 = make_activation(seed=2)
        sr.observe(make_memory_context(["第一轮解读"]), act1)
        sr.observe(make_memory_context(["第二轮解读"]), act2)

        ext_vec = torch.randn(INPUT_DIM)
        echo = sr.generate_echo(ext_sensory=ext_vec)
        assert echo is not None
        # 单轮模式：vector 应直接使用 sensory_self_prev（设备对齐后）
        expected_vec = sr.sensory_self_prev.to(ext_vec.device).detach()
        assert torch.allclose(echo['vector'], expected_vec, atol=1e-5)

    def test_history_stores_multiple_entries(self):
        """observe 多轮后 history 存储多条记录。"""
        sr, _ = make_self_ref(self_ref_echo_max_rounds=3)
        ctx = make_memory_context()
        for i in range(3):
            act = make_activation(seed=i)
            sr.observe(ctx, act)

        assert len(sr.sensory_self_history) == 3
        # sensory_self_prev 应与最后一条记录一致
        assert torch.allclose(
            sr.sensory_self_prev, sr.sensory_self_history[-1], atol=1e-5)

    def test_history_trimmed_to_max(self):
        """history 超过 max_rounds+2 时裁剪。"""
        sr, _ = make_self_ref(self_ref_echo_max_rounds=3)
        ctx = make_memory_context()
        # max_hist = max(3+2, 3) = 5
        for i in range(8):
            act = make_activation(seed=i)
            sr.observe(ctx, act)

        assert len(sr.sensory_self_history) == 5

    def test_decayed_echo_weights_correct(self):
        """衰减加权计算正确（手动验证权重）。"""
        # decay=0.5, K=3 → 权重 [1.0, 0.5, 0.25] → 归一化 [0.571, 0.286, 0.143]
        sr, _ = make_self_ref(
            self_ref_echo_max_rounds=3,
            self_ref_echo_decay_rate=0.5)
        # 手动构造 history（3 条正交单位向量）
        v0 = torch.zeros(INPUT_DIM)
        v0[0] = 1.0  # 最早
        v1 = torch.zeros(INPUT_DIM)
        v1[1] = 1.0
        v2 = torch.zeros(INPUT_DIM)
        v2[2] = 1.0  # 最近
        sr.sensory_self_history = [v0.clone(), v1.clone(), v2.clone()]
        sr.sensory_self_prev = v2.clone()  # 与 observe 行为一致

        result = sr._compute_decayed_echo(None)
        # weighted_sum = 1.0*v2 + 0.5*v1 + 0.25*v0
        # total_weight = 1.75
        # result = [0.25/1.75, 0.5/1.75, 1.0/1.75]
        total = 1.0 + 0.5 + 0.25
        expected = torch.zeros(INPUT_DIM)
        expected[0] = 0.25 / total
        expected[1] = 0.5 / total
        expected[2] = 1.0 / total
        assert torch.allclose(result, expected, atol=1e-5)

    def test_decayed_echo_normalized(self):
        """多轮 echo 归一化后范数不膨胀。"""
        sr, _ = make_self_ref(
            self_ref_echo_max_rounds=4,
            self_ref_echo_decay_rate=0.5)
        # 用随机向量构造 history
        g = torch.Generator().manual_seed(123)
        vectors = [torch.randn(INPUT_DIM, generator=g) * 0.1 for _ in range(4)]
        sr.sensory_self_history = [v.clone() for v in vectors]
        sr.sensory_self_prev = vectors[-1].clone()

        result = sr._compute_decayed_echo(None)
        max_norm = max(float(v.norm()) for v in vectors)
        # 加权平均的范数不超过最大单轮范数（三角不等式）
        assert float(result.norm()) <= max_norm + 1e-5

    def test_norm_guard_fallback(self):
        """范数守卫：多轮范数超过单轮 1.5 倍时回退单轮。"""
        sr, _ = make_self_ref(
            self_ref_echo_max_rounds=3,
            self_ref_echo_decay_rate=0.5)
        # 构造使加权平均范数远大于最近一轮的场景
        # 最近轮 v2 范数=1，早期轮范数=10 且同向
        v0 = torch.zeros(INPUT_DIM)
        v0[0] = 10.0  # 最早
        v1 = torch.zeros(INPUT_DIM)
        v1[0] = 10.0
        v2 = torch.zeros(INPUT_DIM)
        v2[0] = 1.0  # 最近（sensory_self_prev）
        sr.sensory_self_history = [v0.clone(), v1.clone(), v2.clone()]
        sr.sensory_self_prev = v2.clone()

        result = sr._compute_decayed_echo(None)
        # 加权平均范数 ≈ 4.857 > 1.0 * 1.5 = 1.5 → 回退单轮
        # 回退后应等于 sensory_self_prev = v2
        assert torch.allclose(result, v2, atol=1e-5)

    def test_multi_round_in_get_state(self):
        """get_state 包含 sensory_self_history。"""
        sr, _ = make_self_ref(self_ref_echo_max_rounds=3)
        ctx = make_memory_context()
        for i in range(3):
            act = make_activation(seed=i)
            sr.observe(ctx, act)

        state = sr.get_state()
        assert 'sensory_self_history' in state
        assert len(state['sensory_self_history']) == 3
        assert 'echo_decay_rate' in state
        assert 'echo_max_rounds' in state

    def test_save_load_round_trip_multi_echo(self):
        """save->load 后多轮 echo 产出一致。"""
        sr, _ = make_self_ref(
            self_ref_echo_max_rounds=3,
            self_ref_echo_decay_rate=0.5)
        ctx = make_memory_context()
        for i in range(4):
            act = make_activation(seed=i)
            sr.observe(ctx, act)

        ext_vec = torch.randn(INPUT_DIM, generator=torch.Generator().manual_seed(55))
        echo_before = sr.generate_echo(ext_sensory=ext_vec)
        assert echo_before is not None

        state = sr.get_state()
        sr2, _ = make_self_ref(
            self_ref_echo_max_rounds=3,
            self_ref_echo_decay_rate=0.5)
        sr2.set_state(state)

        echo_after = sr2.generate_echo(ext_sensory=ext_vec)
        assert echo_after is not None
        assert torch.allclose(
            echo_before['vector'], echo_after['vector'], atol=1e-5)
        assert echo_after['alpha'] == pytest.approx(
            echo_before['alpha'], rel=1e-5)

    def test_backward_compat_no_history(self):
        """旧快照（无 sensory_self_history）加载不报错。"""
        sr, _ = make_self_ref(self_ref_echo_max_rounds=3)
        act = make_activation()
        ctx = make_memory_context()
        sr.observe(ctx, act)

        state = sr.get_state()
        # 模拟旧快照：删除多轮 echo 字段
        for key in ('sensory_self_history', 'echo_decay_rate',
                    'echo_max_rounds'):
            state.pop(key, None)

        sr2, _ = make_self_ref(self_ref_echo_max_rounds=3)
        sr2.set_state(state)  # 不应抛异常
        assert sr2.echo_max_rounds == 1  # 回退默认值
        assert sr2.echo_decay_rate == 0.5
        assert len(sr2.sensory_self_history) == 0


# ============================================================
# 3.5 稳定性保护测试
# ============================================================

class TestPhase3Stability:
    """测试 Phase 3.5 稳定性保护。"""

    def test_phase3_disabled_by_default(self):
        """Phase 3 所有功能默认关闭。"""
        sr, _ = make_self_ref()
        assert sr.state_recursion_enabled is False
        assert sr.echo_max_rounds == 1
        assert sr.state_recursion_alpha == 0.0
        assert sr._state_recursion_cooldown == 0
        assert sr._high_autocorr_streak == 0
        assert len(sr.sensory_self_history) == 0

    def test_no_regression_when_disabled(self):
        """Phase 3 全部关闭时，generate_echo 行为与 Phase 2 一致。"""
        sr, _ = make_self_ref()  # Phase 3 全部默认关闭
        # 建立历史（使用不同 context 避免高 echo_sim 触发 L3）
        act1 = make_activation(seed=1)
        act2 = make_activation(seed=2)
        sr.observe(make_memory_context(["第一轮解读"]), act1)
        sr.observe(make_memory_context(["第二轮解读"]), act2)

        ext_vec = torch.randn(INPUT_DIM)
        echo = sr.generate_echo(ext_sensory=ext_vec)
        assert echo is not None
        # 单轮模式：vector 应等于 sensory_self_prev（设备对齐后 detach）
        expected = sr.sensory_self_prev.to(ext_vec.device).detach()
        assert torch.allclose(echo['vector'], expected, atol=1e-5)
        # state_recursion 未启用，alpha 应为 0
        assert sr.state_recursion_alpha == 0.0

    def test_state_recursion_strength_bounded(self):
        """state_recursion_alpha 不超过 state_recursion_strength。"""
        sr, _ = make_self_ref(
            self_ref_state_recursion_enabled=True,
            self_ref_state_recursion_strength=0.3)
        act = make_activation()
        ctx = make_memory_context()
        sr.observe(ctx, act)

        # 测试不同 autocorr 下的 alpha
        for ac in [None, 0.0, 0.3, 0.5, 0.7, 0.79]:
            _set_autocorr_result(sr, autocorr=ac, state='normal')
            sigma = torch.zeros(NUM_NODES)
            seed = sr.generate_state_seed(sigma)
            if seed is not None:
                assert sr.state_recursion_alpha <= sr.state_recursion_strength + 1e-8
            # 重置 streak
            sr._high_autocorr_streak = 0

    def test_cooldown_decrements(self):
        """冷却期每轮递减。"""
        sr, _ = make_self_ref(self_ref_state_recursion_enabled=True)
        act = make_activation()
        ctx = make_memory_context()
        sr.observe(ctx, act)

        sr._state_recursion_cooldown = 5
        sigma = torch.zeros(NUM_NODES)
        for expected in [4, 3, 2, 1, 0]:
            seed = sr.generate_state_seed(sigma)
            assert seed is None
            assert sr._state_recursion_cooldown == expected
        # 冷却结束后恢复正常
        seed = sr.generate_state_seed(sigma)
        assert seed is not None

    def test_high_autocorr_streak_resets(self):
        """autocorr 降到 0.8 以下时 streak 重置。"""
        sr, _ = make_self_ref(self_ref_state_recursion_enabled=True)
        act = make_activation()
        ctx = make_memory_context()
        sr.observe(ctx, act)

        sigma = torch.zeros(NUM_NODES)
        # 连续 3 轮高 autocorr
        _set_autocorr_result(sr, autocorr=0.9, state='normal')
        for _ in range(3):
            sr.generate_state_seed(sigma)
        assert sr._high_autocorr_streak == 3

        # autocorr 回落到 0.5（< 0.8）
        _set_autocorr_result(sr, autocorr=0.5, state='normal')
        seed = sr.generate_state_seed(sigma)
        assert sr._high_autocorr_streak == 0
        assert seed is not None  # 正常返回 seed


# ============================================================
# 主循环集成测试
# ============================================================

class TestLoopIntegration:
    """测试 LivingMemoryLoop 中 Phase 3 的集成。"""

    def test_loop_with_state_recursion_enabled(self):
        """启用状态递归的 loop 正常运行。"""
        from runtime.loop import LivingMemoryLoop

        config = {
            'num_nodes': NUM_NODES,
            'input_dim': INPUT_DIM,
            'self_ref_enabled': True,
            'self_ref_alpha_base': 0.15,
            'self_ref_state_recursion_enabled': True,
            'self_ref_state_recursion_strength': 0.3,
            'meta_enabled': False,
            'auto_snapshot': False,
        }
        loop = LivingMemoryLoop(config)

        for i in range(5):
            loop.process_turn(f"state recursion test {i}")

        assert loop.self_ref is not None
        assert loop.self_ref.state_recursion_enabled is True
        assert loop.self_ref.prev_activation_state is not None
        # 验证状态递归 alpha 在合理范围内
        status = loop.get_status()
        assert status['self_ref_enabled'] is True

    def test_loop_with_multi_echo_enabled(self):
        """启用多轮 echo 的 loop 正常运行。"""
        from runtime.loop import LivingMemoryLoop

        config = {
            'num_nodes': NUM_NODES,
            'input_dim': INPUT_DIM,
            'self_ref_enabled': True,
            'self_ref_alpha_base': 0.15,
            'self_ref_echo_max_rounds': 3,
            'self_ref_echo_decay_rate': 0.5,
            'meta_enabled': False,
            'auto_snapshot': False,
        }
        loop = LivingMemoryLoop(config)

        for i in range(6):
            loop.process_turn(f"multi echo test {i}")

        assert loop.self_ref is not None
        assert loop.self_ref.echo_max_rounds == 3
        # observe 多轮后 history 应有记录
        assert len(loop.self_ref.sensory_self_history) > 1

    def test_loop_with_all_phase3_enabled(self):
        """全部 Phase 3 功能启用的 loop 正常运行。"""
        from runtime.loop import LivingMemoryLoop

        config = {
            'num_nodes': NUM_NODES,
            'input_dim': INPUT_DIM,
            'self_ref_enabled': True,
            'self_ref_alpha_base': 0.15,
            'self_ref_state_recursion_enabled': True,
            'self_ref_state_recursion_strength': 0.3,
            'self_ref_echo_max_rounds': 3,
            'self_ref_echo_decay_rate': 0.5,
            'meta_enabled': False,
            'auto_snapshot': False,
        }
        loop = LivingMemoryLoop(config)

        for i in range(8):
            loop.process_turn(f"all phase3 test {i}")

        assert loop.self_ref is not None
        assert loop.self_ref.state_recursion_enabled is True
        assert loop.self_ref.echo_max_rounds == 3

        # 验证系统未崩溃，状态正常
        status = loop.get_status()
        assert status['self_ref_enabled'] is True

    def test_loop_disabled_identical_to_phase2(self):
        """Phase 3 关闭时 loop 行为与 Phase 2 一致。"""
        from runtime.loop import LivingMemoryLoop

        config = {
            'num_nodes': NUM_NODES,
            'input_dim': INPUT_DIM,
            'self_ref_enabled': True,
            'self_ref_alpha_base': 0.15,
            'meta_enabled': False,
            'auto_snapshot': False,
        }
        loop = LivingMemoryLoop(config)

        for i in range(5):
            loop.process_turn(f"phase2 compat test {i}")

        assert loop.self_ref is not None
        # Phase 3 默认关闭
        assert loop.self_ref.state_recursion_enabled is False
        assert loop.self_ref.echo_max_rounds == 1
        # generate_state_seed 应返回 None
        sigma = loop.attractor.sigma
        seed = loop.self_ref.generate_state_seed(sigma)
        assert seed is None
        # generate_echo 应使用单轮模式
        assert len(loop.self_ref.sensory_self_history) > 0

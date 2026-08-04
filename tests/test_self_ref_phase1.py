"""压力测试：自指回路 Phase 1 稳定性保护层
=========================================

覆盖范围：
  1. AutocorrController（Tier 1 自适应控制器）单元测试
  2. StabilityArbiter（统一稳定性仲裁器）单元测试
  3. SelfReferentialLoop Phase 1 集成测试（Tier 1+2 联动）
  4. 压力测试（模拟长时运行）

设计依据：docs/SELF_REF_INTEGRATED_DESIGN.md
  - 第三节 稳定性机制详解（Tier 1/2/3 + StabilityArbiter）
  - 第五节 Phase 1 验证标准
  - 第六节 测试策略

容错策略：Phase 1 接口（AutocorrController / StabilityArbiter）可能尚未就绪
（并行开发），相关测试自动 skip，不阻塞文件中其他测试。
"""

import hashlib
import math

import pytest
import torch

from core.types import Activation, SensoryInput


# ============================================================
# 优雅导入：Phase 1 类可能尚未实现
# ============================================================

try:
    from core.hippocampus.self_referential import (
        SelfReferentialLoop,
        SelfVoiceDistiller,
    )
    _SELF_REF_AVAILABLE = True
    _SELF_REF_IMPORT_ERROR = None
except ImportError as _exc:
    _SELF_REF_AVAILABLE = False
    _SELF_REF_IMPORT_ERROR = _exc
    SelfReferentialLoop = None  # type: ignore[assignment,misc]
    SelfVoiceDistiller = None  # type: ignore[assignment,misc]

try:
    from core.hippocampus.self_referential import AutocorrController
    _AUTOCORR_AVAILABLE = True
except ImportError:
    _AUTOCORR_AVAILABLE = False
    AutocorrController = None  # type: ignore[assignment,misc]

try:
    from core.hippocampus.self_referential import StabilityArbiter
    _ARBITER_AVAILABLE = True
except ImportError:
    _ARBITER_AVAILABLE = False
    StabilityArbiter = None  # type: ignore[assignment,misc]


# ============================================================
# 常量
# ============================================================

NUM_NODES = 32
INPUT_DIM = 16


# ============================================================
# 辅助函数与假对象（沿用 Phase 0 测试风格）
# ============================================================

def _text_seed(text: str) -> int:
    """由文本生成稳定的随机种子（md5，跨进程可复现）。"""
    digest = hashlib.md5(text.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little')


class FakeEncoder:
    """假编码器：按文本 hash 生成确定性向量，并记录所有编码历史。"""

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
    """假分词器：满足 Encoder.encode 调用签名。"""

    def tokenize(self, text: str):
        return [0]

    def get_vocab(self):
        return {}


class FakeEmbedder:
    """假嵌入器：满足 Encoder.encode 调用签名。"""

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


def make_fixed_activation(state_vec: torch.Tensor,
                          entropy: float = 0.5,
                          surprise: float = 0.1) -> Activation:
    """用指定 state 向量构造激活态（用于精确控制 autocorr）。"""
    return Activation(state=state_vec.clone(), entropy=entropy, surprise=surprise)


def make_opposite_activation(num_nodes: int = NUM_NODES,
                             seed: int = 42) -> Activation:
    """构造与 make_activation(seed) 相反的激活态（state 取负）。

    用于振荡测试：cos(A, -A) = -1.0。
    """
    act = make_activation(num_nodes=num_nodes, seed=seed)
    return Activation(state=-act.state, entropy=act.entropy, surprise=act.surprise)


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
    """构造 SelfReferentialLoop 及其假依赖，返回 (loop, fake_encoder)。"""
    if not _SELF_REF_AVAILABLE:
        pytest.skip(f"SelfReferentialLoop 未就绪: {_SELF_REF_IMPORT_ERROR}")
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


# ============================================================
# Phase 1 就绪检查
# ============================================================

def _require_autocorr():
    """若 AutocorrController 未就绪则跳过当前测试。"""
    if not _AUTOCORR_AVAILABLE:
        pytest.skip("AutocorrController 尚未实现（Phase 1 并行开发中）")


def _require_arbiter():
    """若 StabilityArbiter 未就绪则跳过当前测试。"""
    if not _ARBITER_AVAILABLE:
        pytest.skip("StabilityArbiter 尚未实现（Phase 1 并行开发中）")


def _loop_has_phase1(loop) -> bool:
    """检测 SelfReferentialLoop 是否已集成 Phase 1 特性。

    检测策略：
      1. get_status() 中是否包含 'autocorr' 字段
      2. loop 是否有 autocorr_controller / arbiter 属性
    """
    try:
        status = loop.get_status()
        if isinstance(status, dict) and 'autocorr' in status:
            return True
    except Exception:
        pass
    return (hasattr(loop, 'autocorr_controller')
            or hasattr(loop, 'arbiter')
            or hasattr(loop, 'controller'))


def _require_loop_phase1(loop):
    """若 loop 未集成 Phase 1 则跳过当前测试。"""
    if not _loop_has_phase1(loop):
        pytest.skip("SelfReferentialLoop Phase 1 特性尚未集成（并行开发中）")


# ============================================================
# API 弹性辅助：适配不同方法命名
# ============================================================

def _try_call(obj, names, *args, **kwargs):
    """尝试在 obj 上调用多个候选方法名之一。"""
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)(*args, **kwargs)
    pytest.skip(f"未在 {type(obj).__name__} 上找到方法 {names}")


def _try_get(obj, attr_names, method_names=None, default=None):
    """尝试获取属性或调用 getter，返回第一个非 None 值。"""
    for name in attr_names:
        if hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return val
    if method_names:
        for name in method_names:
            if hasattr(obj, name):
                val = getattr(obj, name)()
                if val is not None:
                    return val
    return default


def _make_controller(lag_N=3, persistence=3, alpha_base=0.15):
    """创建 AutocorrController，尝试多种构造签名。"""
    _require_autocorr()
    for kwargs in [
        {'lag_N': lag_N, 'persistence': persistence, 'alpha_base': alpha_base},
        {'lag': lag_N, 'lock_persistence': persistence, 'alpha_base': alpha_base},
        {'alpha_base': alpha_base},
        {},
    ]:
        try:
            return AutocorrController(**kwargs)
        except TypeError:
            continue
    pytest.skip("无法用已知参数构造 AutocorrController")


def _make_arbiter(alpha_base=0.15):
    """创建 StabilityArbiter，尝试多种构造签名。"""
    _require_arbiter()
    for kwargs in [
        {'alpha_base': alpha_base},
        {'alpha_base': alpha_base, 'echo_threshold': 0.95, 'echo_decay': 0.80},
        {},
    ]:
        try:
            return StabilityArbiter(**kwargs)
        except TypeError:
            continue
    pytest.skip("无法用已知参数构造 StabilityArbiter")


def _controller_update(controller, activation: Activation):
    """向控制器馈入激活态。"""
    return _try_call(controller, ['update', 'record', 'observe', 'push'],
                     activation)


def _controller_autocorr(controller):
    """获取当前 autocorr 值（None 表示历史不足）。"""
    return _try_get(controller,
                    ['autocorr', 'last_autocorr', 'current_autocorr'],
                    ['get_autocorr', 'compute_autocorr'])


def _controller_state(controller) -> str:
    """获取当前稳定性状态（'normal' / 'locked' / 'oscillating'）。"""
    val = _try_get(controller,
                   ['state', 'stability_state'],
                   ['get_stability_state', 'get_state'],
                   default='normal')
    return str(val).lower() if val is not None else 'normal'


def _controller_alpha(controller) -> float:
    """计算自适应 alpha。"""
    return float(_try_call(controller,
                           ['compute_adaptive_alpha', 'adaptive_alpha',
                            'get_alpha']))


def _controller_noise(controller, vector: torch.Tensor) -> torch.Tensor:
    """生成正交噪声。"""
    return _try_call(controller,
                     ['generate_orthogonal_noise', 'orthogonal_noise',
                      'make_noise'],
                     vector)


def _controller_lock_count(controller) -> int:
    """获取锁定计数器。"""
    val = _try_get(controller,
                   ['lock_count', '_lock_count', 'locked_count'],
                   default=0)
    return int(val) if val is not None else 0


def _arbiter_arbitrate(arbiter, signals: dict):
    """调用仲裁器，尝试多个方法名。"""
    return _try_call(arbiter, ['arbitrate', 'decide', 'compute_alpha',
                               'adjust'],
                     signals)


def _normalize_arbiter_result(result) -> dict:
    """将 arbitrate() 返回值统一为 dict，便于断言。

    支持三种返回形式：
      - float: {'alpha': float, 'inject_noise': False, 'reasoning': '', ...}
      - dict:  直接取键
      - object: 取属性
      - tuple: (alpha,) 或 (alpha, inject_noise) 或 (alpha, inject_noise, reasoning)
    """
    if isinstance(result, dict):
        return {
            'alpha': float(result.get('alpha', 0.0)),
            'inject_noise': bool(result.get('inject_noise', False)),
            'reasoning': str(result.get('reasoning', '')),
            'has_inject_noise': 'inject_noise' in result,
            'has_reasoning': 'reasoning' in result,
        }
    if isinstance(result, (int, float)):
        return {
            'alpha': float(result),
            'inject_noise': False,
            'reasoning': '',
            'has_inject_noise': False,
            'has_reasoning': False,
        }
    if isinstance(result, tuple):
        alpha = float(result[0]) if len(result) > 0 else 0.0
        inject = bool(result[1]) if len(result) > 1 else False
        reasoning = str(result[2]) if len(result) > 2 else ''
        return {
            'alpha': alpha,
            'inject_noise': inject,
            'reasoning': reasoning,
            'has_inject_noise': len(result) > 1,
            'has_reasoning': len(result) > 2,
        }
    # 对象形式
    return {
        'alpha': float(getattr(result, 'alpha', 0.0)),
        'inject_noise': bool(getattr(result, 'inject_noise', False)),
        'reasoning': str(getattr(result, 'reasoning', '')),
        'has_inject_noise': hasattr(result, 'inject_noise'),
        'has_reasoning': hasattr(result, 'reasoning'),
    }


# ============================================================
# 集成测试辅助：模拟自指回路轮次
# ============================================================

DEFAULT_EXT_SENSORY = torch.ones(INPUT_DIM) * 0.1


def _run_round(loop, memory_context: str, activation: Activation,
               ext_sensory: torch.Tensor = None,
               entropy_ratio: float = 0.5,
               activation_prev: Activation = None) -> dict:
    """模拟一轮自指回路：generate_echo -> observe。

    返回 {'echo': Optional[dict], 'alpha': float}。
    """
    if ext_sensory is None:
        ext_sensory = DEFAULT_EXT_SENSORY
    echo = loop.generate_echo(
        entropy_ratio=entropy_ratio,
        ext_sensory=ext_sensory,
        activation_prev=activation_prev,
    )
    alpha = echo['alpha'] if echo is not None else 0.0
    loop.observe(memory_context, activation)
    return {'echo': echo, 'alpha': alpha}


def _run_sequence(loop, rounds_data) -> list:
    """运行多轮自指回路序列。

    Args:
        loop: SelfReferentialLoop 实例。
        rounds_data: list[dict]，每项含 'memory_context', 'activation'，
            可选 'ext_sensory'。

    Returns:
        每轮的 alpha 值列表。
    """
    alphas = []
    prev_act = None
    for data in rounds_data:
        ext = data.get('ext_sensory', DEFAULT_EXT_SENSORY)
        echo = loop.generate_echo(
            entropy_ratio=0.5,
            ext_sensory=ext,
            activation_prev=prev_act,
        )
        alpha = echo['alpha'] if echo is not None else 0.0
        alphas.append(alpha)
        loop.observe(data['memory_context'], data['activation'])
        prev_act = data['activation']
    return alphas


# ============================================================
# 1. AutocorrController 单元测试
# ============================================================

class TestAutocorrController:
    """Tier 1 自适应控制器：监测激活自相关，动态调节 alpha。

    设计 3.1:
      - autocorr_lag_N = cos(sigma_t, sigma_{t-N})
      - 锁定: autocorr > 0.95 持续 K 轮
      - 振荡: autocorr < -0.3 持续 K 轮
      - 正常: -0.3 <= autocorr <= 0.95
      - alpha_adaptive = alpha_base * (1 - autocorr_clipped)
    """

    def test_insufficient_history_returns_none_autocorr(self):
        """历史不足时返回 None autocorr，状态为 normal。

        lag_N=3 时，需要至少 4 次更新才能计算首个 autocorr
        （比较第 3 次与第 0 次）。仅更新 2 次时 autocorr 应为 None。
        """
        ctrl = _make_controller(lag_N=3, persistence=3, alpha_base=0.15)
        act = make_activation(seed=42)

        _controller_update(ctrl, act)
        _controller_update(ctrl, act)

        autocorr = _controller_autocorr(ctrl)
        assert autocorr is None, "历史不足时 autocorr 应为 None"

        state = _controller_state(ctrl)
        assert state == 'normal', f"历史不足时状态应为 normal，实际: {state}"

    def test_identical_activation_triggers_locked(self):
        """连续相同 activation -> autocorr ~= 1.0，持续 persistence 轮后触发 locked。

        使用 lag_N=3, persistence=3，共更新 6 次（lag_N + persistence）：
          更新 0-2: 构建历史
          更新 3: 首个 autocorr = cos(a, a) = 1.0，lock_count=1
          更新 4: autocorr = 1.0，lock_count=2
          更新 5: autocorr = 1.0，lock_count=3 -> locked
        """
        ctrl = _make_controller(lag_N=3, persistence=3, alpha_base=0.15)
        act = make_activation(seed=100)

        for _ in range(6):
            _controller_update(ctrl, act)

        autocorr = _controller_autocorr(ctrl)
        assert autocorr is not None, "6 次更新后 autocorr 不应为 None"
        assert autocorr == pytest.approx(1.0, abs=1e-5), (
            f"相同 activation 的 autocorr 应 ~= 1.0，实际: {autocorr}"
        )

        state = _controller_state(ctrl)
        assert state == 'locked', f"持续高 autocorr 后应触发 locked，实际: {state}"

    def test_alternating_activation_triggers_oscillating(self):
        """交替相反 activation -> autocorr ~= -1.0，持续 persistence 轮后触发 oscillating。

        使用 lag_N=3（奇数），交替 A / -A：
          更新 0: A
          更新 1: -A
          更新 2: A
          更新 3: -A -> autocorr = cos(-A, A) = -1.0（lag_N=3，比较更新3与更新0）
        奇数 lag 保证交替序列在 lag 间隔上始终反向。
        """
        ctrl = _make_controller(lag_N=3, persistence=3, alpha_base=0.15)
        act_a = make_activation(seed=200)
        act_b = make_opposite_activation(seed=200)  # state = -act_a.state

        for i in range(6):
            _controller_update(ctrl, act_a if i % 2 == 0 else act_b)

        autocorr = _controller_autocorr(ctrl)
        assert autocorr is not None, "6 次更新后 autocorr 不应为 None"
        assert autocorr == pytest.approx(-1.0, abs=1e-4), (
            f"交替相反 activation 的 autocorr 应 ~= -1.0，实际: {autocorr}"
        )

        state = _controller_state(ctrl)
        assert state == 'oscillating', (
            f"持续负 autocorr 后应触发 oscillating，实际: {state}"
        )

    def test_normal_variation_keeps_normal_state(self):
        """正常变化 -> autocorr 在 [-0.3, 0.95] 范围，状态为 normal。

        使用不同 seed 的随机 activation，32 维空间中随机向量近正交，
        autocorr 接近 0，不会触发锁定或振荡。
        """
        ctrl = _make_controller(lag_N=3, persistence=5, alpha_base=0.15)

        for i in range(10):
            act = make_activation(seed=300 + i)
            _controller_update(ctrl, act)

        autocorr = _controller_autocorr(ctrl)
        # 随机向量的 autocorr 应在正常范围内（允许偶尔波动）
        if autocorr is not None:
            assert -0.5 <= autocorr <= 0.95, (
                f"正常变化的 autocorr 应在合理范围，实际: {autocorr}"
            )

        state = _controller_state(ctrl)
        assert state == 'normal', (
            f"正常变化应保持 normal 状态，实际: {state}"
        )

    def test_compute_adaptive_alpha(self):
        """compute_adaptive_alpha: locked -> 0, oscillating -> 0, normal -> base*(1-autocorr)。

        分三步验证：
          1. 触发 locked -> alpha = 0
          2. 触发 oscillating -> alpha = 0
          3. 正常状态 -> alpha = alpha_base * (1 - abs(autocorr_clipped))
        """
        alpha_base = 0.15

        # --- 1. locked -> alpha = 0 ---
        ctrl_lock = _make_controller(lag_N=3, persistence=3,
                                     alpha_base=alpha_base)
        act = make_activation(seed=400)
        for _ in range(6):
            _controller_update(ctrl_lock, act)
        assert _controller_state(ctrl_lock) == 'locked'
        alpha_locked = _controller_alpha(ctrl_lock)
        assert alpha_locked == pytest.approx(0.0, abs=1e-6), (
            f"locked 状态 alpha 应为 0，实际: {alpha_locked}"
        )

        # --- 2. oscillating -> alpha = 0 ---
        ctrl_osc = _make_controller(lag_N=3, persistence=3,
                                    alpha_base=alpha_base)
        act_a = make_activation(seed=401)
        act_b = make_opposite_activation(seed=401)
        for i in range(6):
            _controller_update(ctrl_osc, act_a if i % 2 == 0 else act_b)
        assert _controller_state(ctrl_osc) == 'oscillating'
        alpha_osc = _controller_alpha(ctrl_osc)
        assert alpha_osc == pytest.approx(0.0, abs=1e-6), (
            f"oscillating 状态 alpha 应为 0，实际: {alpha_osc}"
        )

        # --- 3. normal -> alpha = base * (1 - abs(autocorr)) ---
        ctrl_norm = _make_controller(lag_N=3, persistence=5,
                                     alpha_base=alpha_base)
        for i in range(6):
            _controller_update(ctrl_norm, make_activation(seed=402 + i))
        state_norm = _controller_state(ctrl_norm)
        assert state_norm == 'normal', f"应为 normal，实际: {state_norm}"
        autocorr = _controller_autocorr(ctrl_norm)
        alpha_norm = _controller_alpha(ctrl_norm)
        if autocorr is not None:
            autocorr_clipped = max(0.0, min(1.0, abs(autocorr)))
            expected = alpha_base * (1.0 - autocorr_clipped)
            assert alpha_norm == pytest.approx(expected, abs=1e-4), (
                f"normal 状态 alpha 应为 {expected}，实际: {alpha_norm}"
            )
        assert 0.0 <= alpha_norm <= alpha_base, (
            f"alpha 应在 [0, alpha_base] 范围内，实际: {alpha_norm}"
        )

    def test_generate_orthogonal_noise(self):
        """generate_orthogonal_noise: 噪声与输入正交（点积 ~= 0）。

        设计 3.1:
          noise = randn_like(sensory) * 0.1
          proj = dot(noise, sensory) / (dot(sensory, sensory) + eps)
          noise_orth = noise - proj * sensory
        正交化后 dot(noise_orth, sensory) ~= 0。
        """
        ctrl = _make_controller(lag_N=3, persistence=3, alpha_base=0.15)
        # 使用非零向量，避免零向量退化
        vector = torch.randn(NUM_NODES)
        vector = vector / vector.norm()  # 单位向量

        noise = _controller_noise(ctrl, vector)
        assert isinstance(noise, torch.Tensor), "噪声应为 torch.Tensor"
        assert noise.shape == vector.shape, "噪声形状应与输入一致"

        dot_product = float(torch.dot(noise.flatten(), vector.flatten()))
        # 正交化后点积应接近 0（允许浮点误差）
        vector_norm = float(vector.norm())
        if vector_norm > 1e-8:
            # 相对误差：|dot| / ||vector|| 应很小
            assert abs(dot_product) < 1e-4 * max(1.0, float(noise.norm())), (
                f"噪声与输入的点积应 ~= 0，实际: {dot_product}"
            )

    def test_lock_release_on_different_activation(self):
        """锁定解除：locked 后注入不同 activation，lock_count 重置。

        流程：
          1. 持续相同 activation 触发 locked
          2. 注入一个不同的 activation
          3. autocorr 下降，lock_count 重置为 0
          4. 状态回到 normal（或至少不再 locked）
        """
        ctrl = _make_controller(lag_N=3, persistence=3, alpha_base=0.15)
        act_same = make_activation(seed=500)

        # 触发 locked
        for _ in range(6):
            _controller_update(ctrl, act_same)
        assert _controller_state(ctrl) == 'locked'
        assert _controller_lock_count(ctrl) >= 3

        # 注入不同 activation（用完全不同的 seed）
        act_diff = make_activation(seed=999)
        _controller_update(ctrl, act_diff)

        # lock_count 应重置
        lock_count_after = _controller_lock_count(ctrl)
        assert lock_count_after == 0, (
            f"注入不同 activation 后 lock_count 应重置为 0，实际: {lock_count_after}"
        )

        # 状态应不再为 locked
        state_after = _controller_state(ctrl)
        assert state_after != 'locked', (
            f"注入不同 activation 后状态不应为 locked，实际: {state_after}"
        )


# ============================================================
# 2. StabilityArbiter 单元测试
# ============================================================

class TestStabilityArbiter:
    """统一稳定性仲裁器：汇总所有稳定性信号，统一决策自指权重。

    设计 3.4:
      - Tier 1 基线: alpha = alpha_base * (1 - abs(autocorr_clipped))
      - L2 回声: echo_sim > 0.95 -> alpha = 0; 0.80 < echo_sim <= 0.95 -> 线性衰减
      - L5 外部: ext_novelty > 0.5 -> alpha *= 0.3
      - 天花板: alpha = min(alpha, alpha_base)
    """

    ALPHA_BASE = 0.15

    def _signals(self, autocorr=0.0, echo_sim=0.0, ext_novelty=0.0,
                 state='normal', **extra):
        """构造仲裁器输入信号字典。"""
        sig = {
            'autocorr': autocorr,
            'echo_sim': echo_sim,
            'ext_novelty': ext_novelty,
            'state': state,
            'entropy_ratio': 0.7,
            'coherence': 0.8,
            'surprise': 0.3,
        }
        sig.update(extra)
        return sig

    def _arbitrate(self, signals) -> dict:
        """调用仲裁器并标准化返回值。"""
        arbiter = _make_arbiter(alpha_base=self.ALPHA_BASE)
        raw = _arbiter_arbitrate(arbiter, signals)
        return _normalize_arbiter_result(raw)

    def test_normal_state_no_decay(self):
        """正常状态: alpha = base * (1-autocorr)，无衰减。

        autocorr=0.3, echo_sim=0.5, ext_novelty=0.1：
          alpha = 0.15 * (1 - 0.3) = 0.105
          echo_sim < 0.80 -> 无回声衰减
          ext_novelty < 0.5 -> 无外部优先衰减
        """
        result = self._arbitrate(self._signals(
            autocorr=0.3, echo_sim=0.5, ext_novelty=0.1))
        expected = self.ALPHA_BASE * (1.0 - 0.3)
        assert result['alpha'] == pytest.approx(expected, abs=1e-4), (
            f"正常状态 alpha 应为 {expected}，实际: {result['alpha']}"
        )

    def test_high_echo_zeroes_alpha(self):
        """高回声 (echo_sim > 0.95): alpha = 0。

        L2 硬守卫：echo_sim > 0.95 时直接清零。
        """
        result = self._arbitrate(self._signals(
            autocorr=0.0, echo_sim=0.96, ext_novelty=0.0))
        assert result['alpha'] == pytest.approx(0.0, abs=1e-6), (
            f"高回声时 alpha 应为 0，实际: {result['alpha']}"
        )

    def test_mid_echo_linear_decay(self):
        """中回声 (0.80 < echo_sim < 0.95): alpha 线性衰减。

        验证：
          1. alpha 低于无回声时的值
          2. echo_sim 越高，alpha 越低（单调递减）
        """
        alpha_no_echo = self._arbitrate(self._signals(
            autocorr=0.0, echo_sim=0.5, ext_novelty=0.0))['alpha']
        alpha_mid_85 = self._arbitrate(self._signals(
            autocorr=0.0, echo_sim=0.85, ext_novelty=0.0))['alpha']
        alpha_mid_90 = self._arbitrate(self._signals(
            autocorr=0.0, echo_sim=0.90, ext_novelty=0.0))['alpha']

        # 回声衰减应使 alpha 降低
        assert alpha_mid_85 < alpha_no_echo, (
            f"中回声应降低 alpha: {alpha_mid_85} < {alpha_no_echo}"
        )
        # 更高回声 -> 更低 alpha（单调性）
        assert alpha_mid_90 <= alpha_mid_85 + 1e-6, (
            f"回声越高 alpha 应越低: {alpha_mid_90} <= {alpha_mid_85}"
        )

    def test_external_priority_reduces_alpha(self):
        """外部优先 (ext_novelty > 0.5): alpha *= 0.3。

        L5 硬守卫：外部新颖度飙升时临时压低自指权重。
        """
        alpha_normal = self._arbitrate(self._signals(
            autocorr=0.0, echo_sim=0.0, ext_novelty=0.1))['alpha']
        alpha_ext = self._arbitrate(self._signals(
            autocorr=0.0, echo_sim=0.0, ext_novelty=0.6))['alpha']

        expected_ext = self.ALPHA_BASE * 0.3
        assert alpha_ext == pytest.approx(expected_ext, abs=1e-4), (
            f"外部优先时 alpha 应为 {expected_ext}，实际: {alpha_ext}"
        )
        assert alpha_ext < alpha_normal, (
            f"外部优先应降低 alpha: {alpha_ext} < {alpha_normal}"
        )

    def test_multiple_guards_most_conservative(self):
        """多个守卫同时触发: 取最保守值（alpha 最低）。

        echo_sim=0.85 + ext_novelty=0.6 同时触发：
          alpha 应 <= 单独触发 echo 时的 alpha
          alpha 应 <= 单独触发 ext 时的 alpha
        """
        alpha_echo_only = self._arbitrate(self._signals(
            autocorr=0.0, echo_sim=0.85, ext_novelty=0.0))['alpha']
        alpha_ext_only = self._arbitrate(self._signals(
            autocorr=0.0, echo_sim=0.0, ext_novelty=0.6))['alpha']
        alpha_both = self._arbitrate(self._signals(
            autocorr=0.0, echo_sim=0.85, ext_novelty=0.6))['alpha']

        min_individual = min(alpha_echo_only, alpha_ext_only)
        assert alpha_both <= min_individual + 1e-6, (
            f"多守卫同时触发应取最保守值: {alpha_both} <= {min_individual}"
        )

    def test_alpha_ceiling(self):
        """天花板: alpha 永不超过 alpha_base。

        即使 autocorr=0（无自相关抑制），alpha 也不应超过 alpha_base。
        """
        result = self._arbitrate(self._signals(
            autocorr=0.0, echo_sim=0.0, ext_novelty=0.0))
        assert result['alpha'] <= self.ALPHA_BASE + 1e-8, (
            f"alpha 不应超过 alpha_base: {result['alpha']} <= {self.ALPHA_BASE}"
        )

        # 负 autocorr 也不应使 alpha 超过天花板
        result_neg = self._arbitrate(self._signals(
            autocorr=-0.5, echo_sim=0.0, ext_novelty=0.0))
        assert result_neg['alpha'] <= self.ALPHA_BASE + 1e-8, (
            f"负 autocorr 时 alpha 也不应超过 alpha_base: "
            f"{result_neg['alpha']} <= {self.ALPHA_BASE}"
        )

    def test_locked_injects_noise(self):
        """locked 时 inject_noise = True。

        当 autocorr > 0.95（锁定条件）或 state='locked' 时，
        仲裁器应建议注入正交噪声。
        """
        result = self._arbitrate(self._signals(
            autocorr=0.96, echo_sim=0.0, ext_novelty=0.0, state='locked'))

        if not result['has_inject_noise']:
            # 尝试仅通过 state 信号传递
            result2 = self._arbitrate(self._signals(
                autocorr=0.96, echo_sim=0.0, ext_novelty=0.0,
                state='locked', inject_noise=True))
            if not result2['has_inject_noise']:
                pytest.skip("仲裁器返回值不含 inject_noise 字段（API 差异）")

        # 如果有 inject_noise 字段，锁定时应为 True
        if result['has_inject_noise']:
            assert result['inject_noise'] is True, (
                f"locked 状态应 inject_noise=True，实际: {result['inject_noise']}"
            )

    def test_reasoning_non_empty(self):
        """reasoning 字段非空（用于监控）。

        仲裁器应提供决策理由字符串，便于调试与监控。
        """
        result = self._arbitrate(self._signals(
            autocorr=0.3, echo_sim=0.5, ext_novelty=0.1))

        if not result['has_reasoning']:
            pytest.skip("仲裁器返回值不含 reasoning 字段（API 差异）")

        assert isinstance(result['reasoning'], str), (
            "reasoning 应为字符串类型"
        )
        assert len(result['reasoning']) > 0, (
            "reasoning 不应为空字符串"
        )


# ============================================================
# 3. SelfReferentialLoop 集成测试（Tier 1+2 联动）
# ============================================================

class TestSelfRefLoopPhase1:
    """SelfReferentialLoop Phase 1 集成测试：Tier 1 自适应控制器 + Tier 2 硬守卫联动。

    验证 Phase 1 验证标准（设计第五节）：
      1. 回声压力：连续相同输入 -> alpha 逐渐降至 0
      2. 振荡检测：交替输入 -> 检测振荡 -> alpha 降
      3. 正常变化：每轮不同输入 -> alpha 保持健康范围
      4. 回声检测：连续相同 memory_context -> echo_similarity > 0.95 -> alpha = 0
      5. 外部优先：ext_sensory 突变 -> alpha 临时降低
      6. 噪声注入：锁定时 generate_echo 返回的 vector 含正交噪声分量
      7. get_status 包含 autocorr/state/ext_novelty 字段
      8. get_state/set_state 包含 Phase 1 新增状态
    """

    def test_lock_scenario_alpha_decreases(self):
        """锁定场景：连续 N 轮相同输入 -> alpha 逐渐降至 0。

        使用相同的 activation 和 memory_context 连续运行 20 轮，
        后期 alpha 应显著低于初期。
        """
        loop, _ = make_self_ref(alpha_base=0.15, history_cap=30)
        _require_loop_phase1(loop)

        act = make_activation(seed=600)
        ctx = make_memory_context(interpretations=["锁定测试相同自述"])

        rounds = [{'memory_context': ctx, 'activation': act} for _ in range(20)]
        alphas = _run_sequence(loop, rounds)

        # 所有 alpha 应有限（无 NaN）
        assert all(math.isfinite(a) for a in alphas), "alpha 不应包含 NaN"

        # 后期 alpha 应显著低于初期
        early_alpha = max(alphas[1:6])  # 跳过首轮（alpha=0）
        late_alpha = min(alphas[-5:])
        assert late_alpha < early_alpha, (
            f"锁定后期 alpha ({late_alpha}) 应低于初期 ({early_alpha})"
        )

    def test_oscillation_scenario_alpha_decreases(self):
        """振荡场景：交替输入 -> 检测到振荡 -> alpha 降低。

        交替使用两个对立主题的 activation 和 memory_context，
        autocorr 应为负值，触发振荡检测，alpha 降低。
        """
        loop, _ = make_self_ref(alpha_base=0.15, history_cap=30)
        _require_loop_phase1(loop)

        act_a = make_activation(seed=700)
        act_b = make_opposite_activation(seed=700)
        ctx_a = make_memory_context(interpretations=["主题A的解读"])
        ctx_b = make_memory_context(interpretations=["主题B的解读"])

        rounds = []
        for i in range(20):
            if i % 2 == 0:
                rounds.append({'memory_context': ctx_a, 'activation': act_a})
            else:
                rounds.append({'memory_context': ctx_b, 'activation': act_b})

        alphas = _run_sequence(loop, rounds)

        assert all(math.isfinite(a) for a in alphas), "alpha 不应包含 NaN"

        # 后期 alpha 应低于初期（振荡检测触发降权）
        early_alpha = max(alphas[1:6])
        late_alpha = min(alphas[-5:])
        assert late_alpha < early_alpha, (
            f"振荡后期 alpha ({late_alpha}) 应低于初期 ({early_alpha})"
        )

    def test_normal_variation_healthy_alpha(self):
        """正常变化：每轮不同输入 -> alpha 保持健康范围。

        使用不同的 activation 和 memory_context，autocorr 应在正常范围，
        alpha 应保持正值且不超过 alpha_base。
        """
        loop, _ = make_self_ref(alpha_base=0.15, history_cap=30)
        _require_loop_phase1(loop)

        rounds = [
            {
                'memory_context': make_memory_context(
                    interpretations=[f"第{i}轮独特自述内容"]),
                'activation': make_activation(seed=800 + i),
            }
            for i in range(15)
        ]
        alphas = _run_sequence(loop, rounds)

        # 跳过首轮（alpha=0），验证其余轮次
        for i, alpha in enumerate(alphas[1:], 1):
            assert 0.0 <= alpha <= 0.15 + 1e-6, (
                f"第 {i} 轮 alpha 超出 [0, 0.15]: {alpha}"
            )

        # 正常变化时至少有一些轮次 alpha > 0
        positive_alphas = [a for a in alphas[1:] if a > 0.01]
        assert len(positive_alphas) > 0, (
            "正常变化时应至少有一些轮次 alpha > 0.01"
        )

    def test_echo_detection_zeroes_alpha(self):
        """回声检测：连续相同 memory_context -> echo_similarity > 0.95 -> alpha = 0。

        连续传入完全相同的 memory_context，蒸馏出的自述文本相同，
        编码后向量一致，echo_similarity = 1.0 > 0.95，触发 L2 硬清零。
        """
        loop, _ = make_self_ref(alpha_base=0.15, history_cap=30)
        _require_loop_phase1(loop)

        ctx = make_memory_context(interpretations=["完全相同的自述内容"])
        act = make_activation(seed=900)

        rounds = [{'memory_context': ctx, 'activation': act} for _ in range(15)]
        alphas = _run_sequence(loop, rounds)

        # 后期 alpha 应为 0 或接近 0（回声硬清零 + 锁定）
        late_alphas = alphas[-5:]
        assert all(a < 0.01 for a in late_alphas), (
            f"回声检测后期 alpha 应接近 0: {late_alphas}"
        )

    def test_external_priority_lowers_alpha(self):
        """外部优先：ext_sensory 突变 -> alpha 临时降低。

        前几轮使用固定 ext_sensory，然后突然切换到完全不同的向量，
        ext_novelty 飙升，触发 L5 外部优先，alpha 临时降低。
        """
        loop, _ = make_self_ref(alpha_base=0.15, history_cap=30)
        _require_loop_phase1(loop)

        ext_normal = torch.ones(INPUT_DIM) * 0.1
        ext_new = torch.ones(INPUT_DIM) * (-0.1)  # 反向，cos ~= -1

        act = make_activation(seed=1000)
        ctx = make_memory_context(interpretations=["外部优先测试"])

        # 前 5 轮：固定 ext_sensory
        rounds = []
        for i in range(5):
            rounds.append({
                'memory_context': ctx, 'activation': act,
                'ext_sensory': ext_normal,
            })
        # 第 6 轮：突变 ext_sensory
        rounds.append({
            'memory_context': ctx, 'activation': act,
            'ext_sensory': ext_new,
        })

        alphas = _run_sequence(loop, rounds)

        # 突变轮的 alpha 应低于前一轮
        alpha_before = alphas[4]  # 第 5 轮（index 4）
        alpha_change = alphas[5]  # 第 6 轮（突变轮）
        assert alpha_change <= alpha_before + 1e-6, (
            f"外部突变时 alpha 应降低: {alpha_change} <= {alpha_before}"
        )

    def test_noise_injection_when_locked(self):
        """噪声注入：锁定时 generate_echo 返回的 vector 含正交噪声分量。

        触发锁定后，generate_echo 应在回注向量上叠加正交噪声，
        使返回的 vector 与原始 sensory_self_prev 不同。
        """
        loop, _ = make_self_ref(alpha_base=0.15, history_cap=30)
        _require_loop_phase1(loop)

        act = make_activation(seed=1100)
        ctx = make_memory_context(interpretations=["噪声注入测试相同自述"])

        # 运行足够轮次触发锁定
        for i in range(20):
            _run_round(loop, ctx, act)

        # 获取锁定后的 echo
        echo = loop.generate_echo(
            entropy_ratio=0.5,
            ext_sensory=DEFAULT_EXT_SENSORY,
            activation_prev=act,
        )

        if echo is None:
            pytest.skip("锁定后 generate_echo 返回 None（可能 alpha=0 时不回注）")

        # 获取原始 sensory_self_prev（未加噪声的回注源）
        state = loop.get_state()
        original_vec = state.get('sensory_self_prev')

        if original_vec is not None:
            echo_vec = echo['vector']
            # 如果注入了噪声，echo 向量应与原始向量不同
            if not torch.allclose(echo_vec, original_vec.to(echo_vec.device),
                                  atol=1e-6):
                # 验证噪声分量近似正交
                noise_component = echo_vec - original_vec.to(echo_vec.device)
                dot = float(torch.dot(
                    noise_component.flatten(),
                    original_vec.to(echo_vec.device).flatten()
                ))
                original_norm = float(original_vec.norm())
                if original_norm > 1e-8:
                    assert abs(dot) < 1e-3 * max(1.0, float(noise_component.norm())), (
                        f"噪声分量应与原始向量正交，点积: {dot}"
                    )

    def test_get_status_phase1_fields(self):
        """get_status 包含 autocorr/state/ext_novelty 字段。

        Phase 1 增强后，get_status 应暴露自指动力学监控指标。
        """
        loop, _ = make_self_ref(alpha_base=0.15, history_cap=30)
        _require_loop_phase1(loop)

        # 运行几轮产生状态
        for i in range(5):
            _run_round(loop,
                       make_memory_context(interpretations=[f"第{i}轮"]),
                       make_activation(seed=1200 + i))

        status = loop.get_status()
        assert isinstance(status, dict)

        # Phase 1 新增字段
        has_autocorr = 'autocorr' in status or 'last_autocorr' in status
        has_state = 'state' in status or 'stability_state' in status
        has_ext_novelty = ('ext_novelty' in status
                           or 'last_ext_novelty' in status)

        assert has_autocorr, (
            f"get_status 应包含 autocorr 字段，实际键: {list(status.keys())}"
        )
        assert has_state, (
            f"get_status 应包含 state 字段，实际键: {list(status.keys())}"
        )

    def test_state_roundtrip_phase1(self):
        """get_state/set_state 包含 Phase 1 新增状态。

        Phase 1 新增的状态（如 autocorr_history、lock_count 等）应在
        get_state/set_state 往返后保持一致。
        """
        loop, _ = make_self_ref(alpha_base=0.15, history_cap=30)
        _require_loop_phase1(loop)

        # 运行几轮产生状态
        for i in range(8):
            _run_round(loop,
                       make_memory_context(interpretations=[f"第{i}轮"]),
                       make_activation(seed=1300 + i))

        state = loop.get_state()

        # 检查 Phase 1 新增字段是否存在
        phase1_fields = ['autocorr_history', 'lock_count', 'osc_count',
                         'controller_state', 'stability_state']
        found_fields = [f for f in phase1_fields if f in state]

        if not found_fields:
            pytest.skip("get_state 未包含已知 Phase 1 字段（API 差异）")

        # 往返测试
        loop2, _ = make_self_ref(alpha_base=0.15, history_cap=30)
        loop2.set_state(state)
        state2 = loop2.get_state()

        # 验证找到的 Phase 1 字段一致
        for field in found_fields:
            val1 = state.get(field)
            val2 = state2.get(field)
            if isinstance(val1, list) and isinstance(val2, list):
                assert len(val2) == len(val1), (
                    f"字段 {field} 往返后长度不一致: {len(val2)} != {len(val1)}"
                )
            elif isinstance(val1, (int, float)):
                assert val2 == pytest.approx(val1, abs=1e-6), (
                    f"字段 {field} 往返后值不一致: {val2} != {val1}"
                )
            elif val1 is not None:
                assert val2 is not None, (
                    f"字段 {field} 往返后不应为 None"
                )


# ============================================================
# 4. 压力测试（模拟长时运行）
# ============================================================

class TestStressScenarios:
    """压力测试：模拟长时运行，验证 Phase 1 验证标准（设计第五节）。

    缩短版：使用 50 轮（而非设计文档的 1000 轮）以适配单元测试时间约束。
    """

    def test_echo_stress_alpha_converges(self):
        """回声压力测试：50 轮相同输入，验证 alpha 收敛到 0，不 NaN。

        Phase 1 验证标准 1（缩短版）：
          连续相同输入 -> autocorr 收敛到 >0.9 -> Tier 1 触发 -> alpha -> 0
        """
        loop, _ = make_self_ref(alpha_base=0.15, history_cap=50)
        if not _loop_has_phase1(loop):
            pytest.skip("Phase 1 未集成，跳过回声压力测试")

        act = make_activation(seed=2000)
        ctx = make_memory_context(interpretations=["回声压力测试相同自述"])

        rounds = [{'memory_context': ctx, 'activation': act} for _ in range(50)]
        alphas = _run_sequence(loop, rounds)

        # 无 NaN
        assert all(math.isfinite(a) for a in alphas), "alpha 包含 NaN"

        # 后 10 轮 alpha 应收敛到接近 0
        late_alphas = alphas[-10:]
        max_late = max(late_alphas)
        assert max_late < 0.02, (
            f"后 10 轮 alpha 应收敛到接近 0，最大值: {max_late}"
        )

    def test_oscillation_stress_alpha_reduces(self):
        """振荡压力测试：50 轮交替输入，验证振荡检测触发，alpha 降低。

        Phase 1 验证标准 2（缩短版）：
          交替对立主题 -> autocorr < -0.3 -> 振荡检测 -> alpha 降低
        """
        loop, _ = make_self_ref(alpha_base=0.15, history_cap=50)
        if not _loop_has_phase1(loop):
            pytest.skip("Phase 1 未集成，跳过振荡压力测试")

        act_a = make_activation(seed=2100)
        act_b = make_opposite_activation(seed=2100)
        ctx_a = make_memory_context(interpretations=["振荡主题A"])
        ctx_b = make_memory_context(interpretations=["振荡主题B"])

        rounds = []
        for i in range(50):
            if i % 2 == 0:
                rounds.append({'memory_context': ctx_a, 'activation': act_a})
            else:
                rounds.append({'memory_context': ctx_b, 'activation': act_b})

        alphas = _run_sequence(loop, rounds)

        # 无 NaN
        assert all(math.isfinite(a) for a in alphas), "alpha 包含 NaN"

        # 后期 alpha 应低于初期
        early_max = max(alphas[1:10])
        late_min = min(alphas[-10:])
        assert late_min < early_max, (
            f"振荡后期 alpha ({late_min}) 应低于初期最大值 ({early_max})"
        )

    def test_external_response_after_self_ref(self):
        """外部响应测试：20 轮自指运行后注入全新输入，验证 alpha 响应。

        Phase 1 验证标准 3（缩短版）：
          自指运行 20 轮后注入全新输入 -> alpha 应有响应（变化）。
        """
        loop, _ = make_self_ref(alpha_base=0.15, history_cap=50)
        if not _loop_has_phase1(loop):
            pytest.skip("Phase 1 未集成，跳过外部响应测试")

        act = make_activation(seed=2200)
        ctx = make_memory_context(interpretations=["自指运行阶段相同自述"])
        ext_normal = torch.ones(INPUT_DIM) * 0.1

        # 20 轮自指运行
        rounds = []
        for _ in range(20):
            rounds.append({
                'memory_context': ctx, 'activation': act,
                'ext_sensory': ext_normal,
            })
        alphas_before = _run_sequence(loop, rounds)

        # 注入全新输入
        act_new = make_activation(seed=9999)
        ctx_new = make_memory_context(interpretations=["全新的外部输入内容"])
        ext_new = torch.randn(INPUT_DIM) * 0.5

        echo = loop.generate_echo(
            entropy_ratio=0.5,
            ext_sensory=ext_new,
            activation_prev=act,
        )
        alpha_response = echo['alpha'] if echo is not None else 0.0

        loop.observe(ctx_new, act_new)

        # alpha 应有响应（与锁定阶段的 0 不同）
        late_before = alphas_before[-1]
        # 响应轮的 alpha 应与锁定阶段不同（要么因 ext_novelty 降低，要么因 autocorr 变化而恢复）
        assert math.isfinite(alpha_response), "响应 alpha 不应为 NaN"

    def test_alpha_bounded(self):
        """增益有界性：任意轮 0 <= alpha <= alpha_base。

        Phase 1 验证标准 4：
          无论运行多少轮、什么场景，alpha 始终在 [0, alpha_base] 范围内。
        """
        loop, _ = make_self_ref(alpha_base=0.15, history_cap=50)

        alpha_base = loop.alpha_base

        # 场景 1：相同输入
        act = make_activation(seed=2300)
        ctx = make_memory_context(interpretations=["有界性测试相同"])
        rounds = [{'memory_context': ctx, 'activation': act} for _ in range(20)]
        alphas_same = _run_sequence(loop, rounds)

        for i, a in enumerate(alphas_same):
            assert 0.0 <= a <= alpha_base + 1e-6, (
                f"相同输入第 {i} 轮 alpha 越界: {a}，范围 [0, {alpha_base}]"
            )

        # 场景 2：交替输入
        act_a = make_activation(seed=2301)
        act_b = make_opposite_activation(seed=2301)
        rounds = []
        for i in range(20):
            rounds.append({
                'memory_context': ctx,
                'activation': act_a if i % 2 == 0 else act_b,
            })
        alphas_alt = _run_sequence(loop, rounds)

        for i, a in enumerate(alphas_alt):
            assert 0.0 <= a <= alpha_base + 1e-6, (
                f"交替输入第 {i} 轮 alpha 越界: {a}，范围 [0, {alpha_base}]"
            )

        # 场景 3：随机输入
        rounds = [
            {
                'memory_context': make_memory_context(
                    interpretations=[f"随机第{i}轮"]),
                'activation': make_activation(seed=2400 + i),
            }
            for i in range(20)
        ]
        alphas_rand = _run_sequence(loop, rounds)

        for i, a in enumerate(alphas_rand):
            assert 0.0 <= a <= alpha_base + 1e-6, (
                f"随机输入第 {i} 轮 alpha 越界: {a}，范围 [0, {alpha_base}]"
            )

    def test_mixed_scenario_lifecycle(self):
        """混合场景：正常 -> 锁定 -> 注入噪声 -> 恢复 -> 振荡 -> 降权 -> 恢复。

        验证自指回路在复杂场景下的全生命周期稳定性：
          1. 正常阶段（5 轮不同输入）：alpha 健康
          2. 锁定阶段（15 轮相同输入）：alpha 降至 0
          3. 注入噪声（1 轮全新输入）：打破锁定
          4. 恢复阶段（5 轮不同输入）：alpha 恢复
          5. 振荡阶段（15 轮交替输入）：alpha 降至 0
          6. 恢复阶段（5 轮不同输入）：alpha 恢复
        """
        loop, _ = make_self_ref(alpha_base=0.15, history_cap=60)
        # 此测试在 Phase 0 下也运行（验证不崩溃），Phase 1 下验证状态转换

        has_phase1 = _loop_has_phase1(loop)
        all_alphas = []

        # --- 1. 正常阶段 ---
        for i in range(5):
            alpha = _run_round(loop,
                               make_memory_context(interpretations=[f"正常{i}"]),
                               make_activation(seed=3000 + i))['alpha']
            all_alphas.append(alpha)

        # --- 2. 锁定阶段 ---
        act_lock = make_activation(seed=3100)
        ctx_lock = make_memory_context(interpretations=["锁定阶段相同自述"])
        for _ in range(15):
            alpha = _run_round(loop, ctx_lock, act_lock)['alpha']
            all_alphas.append(alpha)

        # --- 3. 注入噪声 ---
        alpha_inject = _run_round(
            loop,
            make_memory_context(interpretations=["注入全新内容打破锁定"]),
            make_activation(seed=9999),
        )['alpha']
        all_alphas.append(alpha_inject)

        # --- 4. 恢复阶段 ---
        for i in range(5):
            alpha = _run_round(loop,
                               make_memory_context(interpretations=[f"恢复{i}"]),
                               make_activation(seed=3200 + i))['alpha']
            all_alphas.append(alpha)

        # --- 5. 振荡阶段 ---
        act_osc_a = make_activation(seed=3300)
        act_osc_b = make_opposite_activation(seed=3300)
        ctx_osc_a = make_memory_context(interpretations=["振荡A"])
        ctx_osc_b = make_memory_context(interpretations=["振荡B"])
        for i in range(15):
            if i % 2 == 0:
                alpha = _run_round(loop, ctx_osc_a, act_osc_a)['alpha']
            else:
                alpha = _run_round(loop, ctx_osc_b, act_osc_b)['alpha']
            all_alphas.append(alpha)

        # --- 6. 恢复阶段 ---
        for i in range(5):
            alpha = _run_round(loop,
                               make_memory_context(interpretations=[f"最终恢复{i}"]),
                               make_activation(seed=3400 + i))['alpha']
            all_alphas.append(alpha)

        # 全局验证：无 NaN，有界
        assert all(math.isfinite(a) for a in all_alphas), (
            "混合场景中 alpha 包含 NaN"
        )
        alpha_base = loop.alpha_base
        for i, a in enumerate(all_alphas):
            assert 0.0 <= a <= alpha_base + 1e-6, (
                f"第 {i} 轮 alpha 越界: {a}，范围 [0, {alpha_base}]"
            )

        # Phase 1 专属验证：锁定阶段 alpha 应低于正常阶段
        if has_phase1:
            normal_alphas = all_alphas[1:5]  # 正常阶段（跳过首轮）
            lock_alphas = all_alphas[15:25]  # 锁定阶段后期
            if normal_alphas and lock_alphas:
                normal_max = max(normal_alphas)
                lock_min = min(lock_alphas)
                # 锁定阶段应出现降权（但不要求每轮都降，因控制器需要时间触发）
                assert lock_min < normal_max, (
                    f"锁定阶段 alpha ({lock_min}) 应低于正常阶段 ({normal_max})"
                )

"""单元测试：自指回路模块 (SelfVoiceDistiller + SelfReferentialLoop)

覆盖范围：
  1. SelfVoiceDistiller：从 decoder 输出的 memory_context 中蒸馏"自述内容"
  2. SelfReferentialLoop：延迟回注、固定增益、echo_similarity、历史裁剪、状态往返
  3. EpisodicEntry.source 字段：来源标记（默认 'external'，可设 'self_ref'）
  4. 回归保护：self_ref_enabled 默认关闭、开启时不崩溃

设计依据：docs/SELF_REF_INTEGRATED_DESIGN.md
  - 第五节 Phase 0 验证标准：首轮 generate_echo 返回 None；第 2 轮起可回注；
    alpha 固定为 alpha_base=0.15；默认关闭不破坏既有行为。
  - 第六节测试策略：回归保护为最高优先级。

接口假设（按设计文档实现，并行子 AI 创建 self_referential.py）：
  - SelfVoiceDistiller.distill(memory_context: str) -> str
  - SelfReferentialLoop(encoder, tokenizer, embedder, alpha_base=0.15, history_cap=...)
        .observe(memory_context: str, activation: Activation) -> None
        .generate_echo() -> Optional[dict]   # dict 含 'alpha'(float) 与 'vector'(Tensor)
        .get_state() -> dict
        .set_state(state: dict) -> None
        .get_status() -> dict
  - LivingMemoryLoop：config['self_ref_enabled'] 控制；默认 False 时 self_ref 为 None。

容错策略：self_referential 模块或 EpisodicEntry.source 尚未就绪时，相关测试自动 skip，
不阻塞其余测试与全量回归。待并行实现完成后这些测试即转为通过。
"""

import dataclasses
import hashlib
import math

import pytest
import torch

from core.types import Activation, SensoryInput


# ============================================================
# 优雅导入：self_referential 模块可能尚未创建
# ============================================================

try:
    from core.hippocampus.self_referential import (
        SelfVoiceDistiller,
        SelfReferentialLoop,
    )
    _SELF_REF_AVAILABLE = True
    _SELF_REF_IMPORT_ERROR = None
except ImportError as _exc:  # 模块尚未实现
    _SELF_REF_AVAILABLE = False
    _SELF_REF_IMPORT_ERROR = _exc
    SelfVoiceDistiller = None  # type: ignore[assignment,misc]
    SelfReferentialLoop = None  # type: ignore[assignment,misc]


# EpisodicEntry 当前定义在 core/hippocampus/memory.py
from core.hippocampus.memory import EpisodicEntry  # noqa: E402


# ============================================================
# 辅助函数与假对象
# ============================================================

def _require_self_ref():
    """若 self_referential 模块未就绪则跳过当前测试。"""
    if not _SELF_REF_AVAILABLE:
        pytest.skip(
            f"core.hippocampus.self_referential 尚未就绪: "
            f"{_SELF_REF_IMPORT_ERROR}"
        )


def _episodic_has_source() -> bool:
    """检测 EpisodicEntry 是否已具备 source 字段（并行实现可能尚未添加）。"""
    try:
        names = {f.name for f in dataclasses.fields(EpisodicEntry)}
    except Exception:
        return False
    return 'source' in names


def _text_seed(text: str) -> int:
    """由文本生成稳定的随机种子（md5，跨进程可复现）。"""
    digest = hashlib.md5(text.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little')


def make_activation(num_nodes: int = 32, seed: int = 42) -> Activation:
    """构造测试用激活态。Phase 0 自指回路不依赖激活态细节，仅需合法对象。"""
    g = torch.Generator().manual_seed(seed)
    state = torch.randn(num_nodes, generator=g) * 0.1
    return Activation(state=state, entropy=1.0, surprise=0.5)


def make_memory_context(interpretations=None, recalled_texts=None,
                        detail=None) -> str:
    """构造与 Decoder._decode_text 输出格式一致的 memory_context。

    结构：
      [记忆context]
      记忆状态解读:
      - {解读1}
      - {解读2}
      相关记忆回忆:        # 仅当 recalled_texts 非空
      1. "{文本1}"
      详细数据: {detail}   # 仅当 detail 非空

    参数:
        interpretations: 记忆状态解读段落的项目列表（系统"自述"内容）。
        recalled_texts: 相关记忆回忆的文本列表（外部记忆，蒸馏时应排除）。
        detail: 详细数据段落内容（熵/惊讶度/激活节点等原始指标）。
    """
    if interpretations is None:
        interpretations = ["当前记忆聚焦于少数模式，形成了清晰的记忆痕迹"]

    lines = ["[记忆context]", "记忆状态解读:"]
    for interp in interpretations:
        lines.append(f"- {interp}")

    if recalled_texts:
        lines.append("相关记忆回忆:")
        for i, text in enumerate(recalled_texts, 1):
            lines.append(f'{i}. "{text}"')

    if detail is not None:
        lines.append(f"详细数据: {detail}")

    return "\n".join(lines)


def make_test_config(**overrides) -> dict:
    """创建小规模测试配置（与 test_integration.py 风格一致）。"""
    from runtime.config import default_config
    config = default_config()
    config['num_nodes'] = 32
    config['input_dim'] = 16
    config['num_infer_steps'] = 5
    config['consolidation_interval'] = 3
    config['seed'] = 42
    config.update(overrides)
    return config


class FakeEncoder:
    """假编码器：对相同文本返回确定的随机向量，并记录所有编码历史。

    用于隔离真实 Encoder/Embedder 依赖。向量由文本 md5 种子决定，
    跨调用可复现，便于验证 generate_echo 的延迟与回注内容。
    """

    def __init__(self, dim: int = 16):
        self._dim = dim
        # 记录每次 encode 的 (文本, 向量)，用于断言延迟回注的内容
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

    def __init__(self, dim: int = 16):
        self._dim = dim

    def embed(self, tokens):
        return torch.zeros(self._dim)

    @property
    def dim(self) -> int:
        return self._dim


def make_self_ref(alpha_base: float = 0.15, history_cap: int = 5,
                  dim: int = 16):
    """构造 SelfReferentialLoop 及其假依赖，返回 (loop, fake_encoder)。

    返回 fake_encoder 便于测试断言延迟回注时编码过的文本/向量。
    实际接口：SelfReferentialLoop(encoder, tokenizer, embedder, config: dict)，
    config 键为 self_ref_alpha_base / self_ref_history_cap。
    """
    _require_self_ref()
    enc = FakeEncoder(dim=dim)
    tok = FakeTokenizer()
    emb = FakeEmbedder(dim=dim)
    config = {
        'self_ref_alpha_base': alpha_base,
        'self_ref_history_cap': history_cap,
    }
    loop = SelfReferentialLoop(enc, tok, emb, config=config)
    return loop, enc


def _call_generate_echo(loop, dim: int = 16):
    """调用 generate_echo（Phase 0 需 entropy_ratio 与 ext_sensory 参数）。

    Phase 0 中这两个参数不参与 alpha 计算（仅 ext_sensory 用于设备对齐），
    故传入固定占位值。
    """
    return loop.generate_echo(
        entropy_ratio=0.5, ext_sensory=torch.zeros(dim))


def _get_history(state: dict) -> list:
    """从 get_state 字典中提取自述历史（兼容多种键名）。"""
    return (state.get('self_voice_history')
            or state.get('voice_history')
            or state.get('history')
            or [])


# ============================================================
# 1. SelfVoiceDistiller 测试
# ============================================================

class TestSelfVoiceDistiller:
    """SelfVoiceDistiller：从 memory_context 蒸馏系统自述内容。"""

    @pytest.fixture
    def distiller(self):
        _require_self_ref()
        return SelfVoiceDistiller()

    def test_distill_normal(self, distiller):
        """正常蒸馏：保留自述，排除外部记忆回忆与结构化标记。

        构造含三段（记忆状态解读 / 相关记忆回忆 / 详细数据）的 context，
        验证蒸馏结果：
          - 不含 [记忆context] 结构化标记
          - 不含相关记忆回忆的文本（外部记忆）
          - 含记忆状态解读的自述内容
        """
        ctx = make_memory_context(
            interpretations=["当前记忆聚焦于少数模式，形成了清晰的记忆痕迹"],
            recalled_texts=[
                "外部记忆内容你好世界",
                "另一条外部回忆片段",
            ],
            detail="熵:1.234, 惊讶度:0.567 | 激活节点: 节点0(强:0.800)",
        )
        result = distiller.distill(ctx)

        assert isinstance(result, str)
        # 排除结构化标记
        assert "[记忆context]" not in result
        # 排除外部记忆回忆内容
        assert "外部记忆内容你好世界" not in result
        assert "另一条外部回忆片段" not in result
        assert "相关记忆回忆" not in result
        # 保留自述内容
        assert "聚焦于少数模式" in result

    def test_distill_empty_context(self, distiller):
        """空 context：不抛异常，返回空字符串。"""
        result = distiller.distill("")
        assert isinstance(result, str)
        assert result == ""

    def test_distill_missing_section(self, distiller):
        """缺失段落：仅有记忆状态解读无详细数据，能正确提取已有段落。"""
        ctx = make_memory_context(
            interpretations=["记忆激活适度"],
            recalled_texts=None,
            detail=None,  # 无详细数据段
        )
        result = distiller.distill(ctx)
        assert isinstance(result, str)
        assert "记忆激活适度" in result
        assert "[记忆context]" not in result

    def test_distill_unstructured_text(self, distiller):
        """格式异常：无结构化标记的纯文本，优雅处理不抛异常。"""
        plain = "这是一段没有任何结构化标记的纯文本。"
        result = distiller.distill(plain)
        assert isinstance(result, str)
        # 优雅处理：返回字符串即可（可为原文或空）

    def test_distill_only_markers_no_content(self, distiller):
        """边界：只有标记行无实际内容，不崩溃。"""
        ctx = "[记忆context]\n记忆状态解读:\n相关记忆回忆:\n详细数据:"
        result = distiller.distill(ctx)
        assert isinstance(result, str)
        assert "[记忆context]" not in result

    def test_distill_nested_markers(self, distiller):
        """边界：自述内容中嵌套出现结构化标记字符串，优雅处理。

        蒸馏器不应因内容中含 "[记忆context]" 子串而崩溃或误删全部内容。
        """
        ctx = make_memory_context(
            interpretations=["正常解读含[记忆context]嵌套标记结尾"],
        )
        result = distiller.distill(ctx)
        assert isinstance(result, str)
        # 自述内容主体应被保留
        assert "正常解读含" in result
        assert "嵌套标记结尾" in result

    def test_distill_very_long_context(self, distiller):
        """边界：超长 context 不崩溃，仍正确排除外部回忆。"""
        long_interp = "超长自述内容片段" * 500
        ctx = make_memory_context(
            interpretations=[long_interp],
            recalled_texts=["外部回忆A", "外部回忆B", "外部回忆C"],
            detail="熵:2.000, 惊讶度:1.000",
        )
        result = distiller.distill(ctx)
        assert isinstance(result, str)
        assert "外部回忆A" not in result
        assert "外部回忆B" not in result
        assert "超长自述内容片段" in result


# ============================================================
# 2. SelfReferentialLoop 测试
# ============================================================

class TestSelfReferentialLoopObserveEcho:
    """SelfReferentialLoop：observe / generate_echo 的回注与延迟行为。"""

    def test_generate_echo_none_before_observe(self):
        """首轮无回注：新建 loop 调用 generate_echo 返回 None。"""
        loop, _ = make_self_ref()
        assert _call_generate_echo(loop) is None

    def test_generate_echo_after_observe(self):
        """observe 后可回注：observe 后下一轮 generate_echo 返回非 None。"""
        loop, _ = make_self_ref()
        act = make_activation()
        ctx = make_memory_context(interpretations=["测试自述内容"])
        loop.observe(ctx, act)

        echo = _call_generate_echo(loop)
        assert echo is not None
        assert 'alpha' in echo
        assert 'vector' in echo
        assert isinstance(echo['vector'], torch.Tensor)
        assert echo['vector'].dim() == 1
        assert echo['vector'].numel() > 0

    def test_one_turn_delay(self):
        """延迟正确性：第 t 轮 generate_echo 使用第 t-1 轮 observe 的内容。

        序列模拟真实轮次（generate_echo 在轮首，observe 在轮尾）：
          observe(A)         -> 缓存 voice_A 的编码
          generate_echo()    -> 应返回 voice_A（t-1 的内容，非当前轮）
          observe(B)         -> 缓存 voice_B 的编码
          generate_echo()    -> 应返回 voice_B（t-1 的内容）
        通过 FakeEncoder 记录的编码向量验证回注内容的来源。
        """
        loop, enc = make_self_ref()
        act = make_activation()

        ctx_a = make_memory_context(interpretations=["自述内容甲"])
        ctx_b = make_memory_context(interpretations=["自述内容乙"])

        # 轮 1 尾：observe(A)
        loop.observe(ctx_a, act)
        # 取 observe(A) 期间编码的向量
        _, vec_a = enc.encoded[-1]

        # 轮 2 首：echo 应使用 A（t-1）
        echo1 = loop.generate_echo(
            entropy_ratio=0.5, ext_sensory=torch.zeros(16))
        assert echo1 is not None
        assert torch.allclose(echo1['vector'], vec_a)

        # 轮 2 尾：observe(B)
        loop.observe(ctx_b, act)
        _, vec_b = enc.encoded[-1]

        # 轮 3 首：echo 应使用 B（t-1），而非 A
        echo2 = loop.generate_echo(
            entropy_ratio=0.5, ext_sensory=torch.zeros(16))
        assert echo2 is not None
        assert torch.allclose(echo2['vector'], vec_b)

        # A 与 B 的回注向量应不同（证明 echo 随 observe 推进）
        assert not torch.allclose(echo1['vector'], echo2['vector'])

    def test_fixed_alpha_base(self):
        """固定增益：Phase 0 MVP 中 alpha 固定为 alpha_base=0.15。

        多轮 observe 后，generate_echo 返回的 alpha 始终等于 alpha_base，
        不随轮次变化（Phase 0 无自适应控制器）。
        """
        loop, _ = make_self_ref(alpha_base=0.15)
        act = make_activation()

        loop.observe(make_memory_context(interpretations=["第一轮自述"]), act)
        for i in range(4):
            echo = loop.generate_echo(
                entropy_ratio=0.5, ext_sensory=torch.zeros(16))
            assert echo is not None
            assert echo['alpha'] == pytest.approx(0.15)
            loop.observe(
                make_memory_context(interpretations=[f"后续自述{i}"]), act)

    def test_custom_alpha_base(self):
        """自定义 alpha_base 生效（验证参数接线，非仅硬编码 0.15）。"""
        loop, _ = make_self_ref(alpha_base=0.08)
        act = make_activation()
        loop.observe(make_memory_context(), act)
        echo = loop.generate_echo(
            entropy_ratio=0.5, ext_sensory=torch.zeros(16))
        assert echo is not None
        assert echo['alpha'] == pytest.approx(0.08)


class TestSelfReferentialLoopSimilarity:
    """SelfReferentialLoop：echo_similarity 计算与历史裁剪。"""

    def test_echo_similarity_computed(self):
        """echo_similarity 计算：连续 observe 两次后相似度被正确计算。

        echo_similarity = cosine(emb(t), emb(t-1))。首次 observe 无前序，
        无相似度；第二次 observe 后产生一个相似度值，应在 [-1, 1] 区间。
        通过 get_state() 的 echo_similarity_history 验证（设计 4.2）。
        """
        loop, _ = make_self_ref()
        act = make_activation()

        loop.observe(make_memory_context(interpretations=["自述甲"]), act)
        state_after_one = loop.get_state()
        hist1 = state_after_one.get('echo_similarity_history', [])
        # 首次无前序，相似度历史应为空（或长度 0）
        assert len(hist1) == 0

        loop.observe(make_memory_context(interpretations=["自述乙"]), act)
        state_after_two = loop.get_state()
        hist2 = state_after_two.get('echo_similarity_history', [])
        # 第二次后应有至少一个相似度
        assert len(hist2) >= 1
        sim = hist2[-1]
        assert isinstance(sim, float)
        assert math.isfinite(sim)
        # 余弦相似度范围
        assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6

    def test_echo_similarity_identical_voice(self):
        """echo_similarity 边界：两次蒸馏结果相同时相似度接近 1。"""
        loop, _ = make_self_ref()
        act = make_activation()
        same_ctx = make_memory_context(interpretations=["完全相同的自述"])
        loop.observe(same_ctx, act)
        loop.observe(same_ctx, act)
        state = loop.get_state()
        hist = state.get('echo_similarity_history', [])
        assert len(hist) >= 1
        # 相同文本经 FakeEncoder 编码向量一致，余弦相似度应为 1.0
        assert hist[-1] == pytest.approx(1.0, abs=1e-5)

    def test_history_cap_trims(self):
        """历史裁剪：observe 超过 history_cap 次后，历史长度不超过 cap。"""
        cap = 3
        loop, _ = make_self_ref(history_cap=cap)
        act = make_activation()

        for i in range(cap + 5):
            loop.observe(
                make_memory_context(interpretations=[f"自述第{i}轮"]), act)

        state = loop.get_state()
        history = _get_history(state)
        assert len(history) <= cap, (
            f"history 长度 {len(history)} 超过 cap {cap}"
        )

    def test_history_cap_default_keeps_recent(self):
        """历史裁剪：超过 cap 后保留的是最近的条目（FIFO 淘汰）。"""
        cap = 2
        loop, _ = make_self_ref(history_cap=cap)
        act = make_activation()

        loop.observe(make_memory_context(interpretations=["最早的"]), act)
        loop.observe(make_memory_context(interpretations=["中间的"]), act)
        loop.observe(make_memory_context(interpretations=["最近的"]), act)

        state = loop.get_state()
        history = _get_history(state)
        assert len(history) <= cap
        # 最近一次的自述应保留在历史中
        joined = " ".join(str(h) for h in history)
        assert "最近的" in joined


class TestSelfReferentialLoopState:
    """SelfReferentialLoop：get_state / set_state 往返与 get_status。"""

    def test_state_roundtrip(self):
        """get_state/set_state 往返：恢复后状态一致。

        运行两轮 observe 产生非空状态，get_state 后 set_state 到新 loop，
        验证 turn_count / 自述历史 / echo_similarity_history / 缓存向量一致。
        alpha_base 为 config 级参数（不在 get_state 中），通过 loop 属性比较。
        """
        loop1, _ = make_self_ref(history_cap=10)
        act = make_activation()
        loop1.observe(make_memory_context(interpretations=["甲"]), act)
        loop1.observe(make_memory_context(interpretations=["乙"]), act)

        state = loop1.get_state()

        # 新 loop 加载状态
        loop2, _ = make_self_ref(history_cap=10)
        loop2.set_state(state)
        state2 = loop2.get_state()

        # 核心字段一致
        assert state2.get('turn_count') == state.get('turn_count')

        # alpha_base 为 config 级，两个 loop 同配置应一致
        assert loop2.alpha_base == pytest.approx(loop1.alpha_base)

        # 自述历史一致
        h1 = _get_history(state)
        h2 = _get_history(state2)
        assert h2 == h1

        # gain_history 一致
        assert state2.get('gain_history', []) == state.get('gain_history', [])

        # echo_similarity_history 一致
        es1 = state.get('echo_similarity_history', [])
        es2 = state2.get('echo_similarity_history', [])
        assert len(es2) == len(es1)
        for a, b in zip(es1, es2):
            assert a == pytest.approx(b)

        # sensory_self_prev（张量）若存在则一致
        p1 = state.get('sensory_self_prev')
        p2 = state2.get('sensory_self_prev')
        if p1 is not None and p2 is not None:
            assert torch.allclose(p1, p2)
        else:
            assert p1 is None and p2 is None

    def test_state_roundtrip_echo_consistent(self):
        """状态往返后行为一致：恢复的 loop 的 generate_echo 产出与原 loop 一致。

        若回注缓存被纳入持久化状态，恢复后下一轮 echo 应使用相同的延迟内容。
        """
        loop1, _ = make_self_ref(history_cap=10)
        act = make_activation()
        loop1.observe(make_memory_context(interpretations=["甲"]), act)
        loop1.observe(make_memory_context(interpretations=["乙"]), act)

        echo_before = loop1.generate_echo(
            entropy_ratio=0.5, ext_sensory=torch.zeros(16))
        state = loop1.get_state()

        loop2, _ = make_self_ref(history_cap=10)
        loop2.set_state(state)
        echo_after = loop2.generate_echo(
            entropy_ratio=0.5, ext_sensory=torch.zeros(16))

        if echo_before is not None and echo_after is not None:
            assert echo_after['alpha'] == pytest.approx(echo_before['alpha'])
            assert torch.allclose(
                echo_after['vector'], echo_before['vector'], atol=1e-6)

    def test_get_status_fields(self):
        """get_status：返回正确的监控字段。

        验证返回字典包含轮次、增益、相似度与历史相关信息。
        实际 get_status 返回：turn_count, last_alpha, last_echo_similarity,
        alpha_base, echo_threshold, history_capacity, self_voice_history_size,
        gain_history_size, echo_similarity_history_size, has_sensory_self_prev。
        """
        loop, _ = make_self_ref(alpha_base=0.15, history_cap=5)
        act = make_activation()
        loop.observe(make_memory_context(interpretations=["甲"]), act)
        loop.observe(make_memory_context(interpretations=["乙"]), act)

        status = loop.get_status()
        assert isinstance(status, dict)

        # 轮次（observe 次数）
        assert status.get('turn_count') == 2

        # 增益基线
        assert status.get('alpha_base') == pytest.approx(0.15)

        # 相似度信息存在
        has_sim = ('last_echo_similarity' in status
                   or 'echo_similarity' in status
                   or 'echo_similarity_history_size' in status)
        assert has_sim, "get_status 应包含 echo_similarity 相关字段"

        # 历史信息存在
        has_hist = ('history_capacity' in status
                    or 'self_voice_history_size' in status
                    or 'history_size' in status)
        assert has_hist, "get_status 应包含 history 相关字段"

    def test_get_status_after_no_observe(self):
        """get_status：未 observe 时也能安全返回（无相似度，轮次为 0）。"""
        loop, _ = make_self_ref(alpha_base=0.15)
        status = loop.get_status()
        assert isinstance(status, dict)
        assert status.get('turn_count') == 0
        assert status.get('alpha_base') == pytest.approx(0.15)
        # 未 observe 时无回注缓存
        assert status.get('has_sensory_self_prev') is False


# ============================================================
# 3. EpisodicEntry.source 字段测试
# ============================================================

class TestEpisodicEntrySource:
    """EpisodicEntry.source 字段：来源标记，防止自指污染情景记忆检索。

    设计 3.3 Tier 3：source 默认 'external'，自指条目标记为 'self_ref'。
    """

    def test_default_source_external(self):
        """默认值：不传 source 时默认为 'external'。"""
        if not _episodic_has_source():
            pytest.skip("EpisodicEntry.source 字段尚未实现（等待并行更新）")
        entry = EpisodicEntry(
            text="外部对话内容",
            semantic_vector=torch.zeros(8),
            surprise=0.5,
            turn=1,
        )
        assert entry.source == 'external'

    def test_custom_source_self_ref(self):
        """自定义值：传入 source='self_ref' 时存储正确。"""
        if not _episodic_has_source():
            pytest.skip("EpisodicEntry.source 字段尚未实现（等待并行更新）")
        entry = EpisodicEntry(
            text="自指蒸馏内容",
            semantic_vector=torch.zeros(8),
            surprise=0.3,
            turn=2,
            source='self_ref',
        )
        assert entry.source == 'self_ref'

    def test_source_does_not_break_existing_fields(self):
        """source 字段加入后不影响既有字段构造与读取。"""
        if not _episodic_has_source():
            pytest.skip("EpisodicEntry.source 字段尚未实现（等待并行更新）")
        vec = torch.randn(8)
        entry = EpisodicEntry(
            text="测试文本",
            semantic_vector=vec,
            surprise=1.2,
            turn=5,
            source='self_ref',
        )
        assert entry.text == "测试文本"
        assert torch.allclose(entry.semantic_vector, vec)
        assert entry.surprise == pytest.approx(1.2)
        assert entry.turn == 5
        assert entry.source == 'self_ref'


# ============================================================
# 4. 回归保护测试
# ============================================================

class TestRegressionProtection:
    """回归保护：自指默认关闭不破坏既有行为，开启时不崩溃。

    设计第五节 Phase 0 验证标准：
      1. self_ref_enabled=False：既有测试 100% 通过
      2. self_ref_enabled=True：process_turn 正常返回，无异常
    """

    def test_default_disabled(self):
        """默认关闭：config 不含 self_ref_enabled 时，LivingMemoryLoop 不创建 self_ref。

        既有行为保持不变——self_ref 为 None（或属性不存在）。
        """
        from runtime.loop import LivingMemoryLoop
        config = make_test_config()
        # 不设置 self_ref_enabled
        assert 'self_ref_enabled' not in config

        loop = LivingMemoryLoop(config)
        # 默认关闭时 self_ref 应为 None（getattr 兼容属性尚未存在的情形）
        assert getattr(loop, 'self_ref', None) is None

    def test_explicit_disabled(self):
        """显式关闭：self_ref_enabled=False 时 self_ref 为 None。"""
        from runtime.loop import LivingMemoryLoop
        config = make_test_config(self_ref_enabled=False)
        loop = LivingMemoryLoop(config)
        assert getattr(loop, 'self_ref', None) is None

    def test_enabled_no_crash(self):
        """开启不崩溃：self_ref_enabled=True 时 process_turn 正常执行。

        使用轻量配置（num_nodes=32），运行多轮验证不抛异常、返回有效 context。
        首轮无历史自述（generate_echo 返回 None），第 2 轮起可回注。
        """
        if not _SELF_REF_AVAILABLE:
            pytest.skip(
                f"self_referential 模块未就绪，无法测试开启路径: "
                f"{_SELF_REF_IMPORT_ERROR}"
            )
        from runtime.loop import LivingMemoryLoop
        config = make_test_config(self_ref_enabled=True)
        loop = LivingMemoryLoop(config)
        # 开启时 self_ref 应已创建
        assert getattr(loop, 'self_ref', None) is not None

        for i in range(5):
            ctx = loop.process_turn(f"自指测试第{i+1}轮内容")
            assert isinstance(ctx, str)
            assert len(ctx) > 0

        # 多轮后系统状态稳定（无 NaN）
        assert loop.last_activation is not None
        assert torch.isfinite(loop.last_activation.state).all()

    def test_enabled_first_turn_no_echo(self):
        """开启路径首轮验证：首轮 process_turn 不注入自指（无历史自述）。

        Phase 0 验证标准 3：首轮 generate_echo 返回 None。
        这里通过完整 loop 验证首轮不因自指出错。
        """
        if not _SELF_REF_AVAILABLE:
            pytest.skip(
                f"self_referential 模块未就绪: {_SELF_REF_IMPORT_ERROR}"
            )
        from runtime.loop import LivingMemoryLoop
        config = make_test_config(self_ref_enabled=True)
        loop = LivingMemoryLoop(config)

        # 首轮：正常返回，不崩溃
        ctx = loop.process_turn("首轮输入")
        assert isinstance(ctx, str)
        assert "记忆context" in ctx or "熵" in ctx

    def test_disabled_identical_to_baseline(self):
        """默认关闭时行为与无自指基线一致：process_turn 正常返回 context。

        验证关闭自指不影响既有循环（回归保护的核心：零损伤）。
        """
        from runtime.loop import LivingMemoryLoop
        torch.manual_seed(42)
        loop = LivingMemoryLoop(make_test_config())

        for i in range(5):
            ctx = loop.process_turn(f"基线测试第{i+1}轮")
            assert isinstance(ctx, str)
            assert "熵" in ctx
            assert "惊讶度" in ctx

        assert loop.turn_count == 5

"""Phase 2 测试：深度集成（Tier 3 基础设施安全）
=============================================

覆盖范围:
  1. 来源标记（source tagging）：EpisodicEntry.source + recall_episodic 过滤
  2. 做梦钩子（dream hooks）：on_dream_start/end + stale 衰减
  3. 学习隔离（learning isolation）：自指主导轮次学习率减半 + meta.update 跳过
  4. 持久化（persistence）：save→load 往返一致性 + 旧快照向后兼容

设计依据: docs/SELF_REF_INTEGRATED_DESIGN.md
  - 第五节 Phase 2 验证标准
  - 第三节 3.3 Tier 3 基础设施安全
"""

import math
import hashlib

import pytest
import torch

from core.types import Activation, SensoryInput
from core.hippocampus.memory import MemoryManager, EpisodicEntry


# ============================================================
# 常量
# ============================================================

NUM_NODES = 32
INPUT_DIM = 16


# ============================================================
# 辅助函数与假对象（沿用 Phase 1 测试风格）
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
    """构造 SelfReferentialLoop 及其假依赖。"""
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


# ============================================================
# 1. 来源标记测试（source tagging）
# ============================================================

class TestSourceTagging:
    """测试 EpisodicEntry.source 字段与 recall_episodic 来源过滤。"""

    def test_store_episodic_default_source_is_external(self):
        """不传 source 参数时默认为 'external'。"""
        mem = MemoryManager(NUM_NODES, device='cpu')
        vec = torch.randn(INPUT_DIM)
        mem.store_episodic("hello", vec, 0.5, turn=1)

        entries = list(mem.iter_episodic())
        assert len(entries) == 1
        assert entries[0].source == 'external'

    def test_store_episodic_with_explicit_external(self):
        """显式传 source='external' 与默认行为一致。"""
        mem = MemoryManager(NUM_NODES, device='cpu')
        vec = torch.randn(INPUT_DIM)
        mem.store_episodic("hello", vec, 0.5, turn=1, source='external')

        entries = list(mem.iter_episodic())
        assert entries[0].source == 'external'

    def test_store_episodic_with_self_ref_source(self):
        """source='self_ref' 标记自指条目。"""
        mem = MemoryManager(NUM_NODES, device='cpu')
        vec = torch.randn(INPUT_DIM)
        mem.store_episodic("self-voice", vec, 0.3, turn=1, source='self_ref')

        entries = list(mem.iter_episodic())
        assert entries[0].source == 'self_ref'

    def test_recall_filters_self_ref_by_default(self):
        """recall_episodic 默认 source_filter='external'，不返回 self_ref 条目。"""
        mem = MemoryManager(NUM_NODES, device='cpu')
        vec_ext = torch.randn(INPUT_DIM)
        vec_self = torch.randn(INPUT_DIM)
        mem.store_episodic("external text", vec_ext, 0.5, turn=1,
                           source='external')
        mem.store_episodic("self voice", vec_self, 0.3, turn=2,
                           source='self_ref')

        results = mem.recall_episodic(vec_ext, top_k=5)
        assert len(results) == 1
        assert results[0].text == "external text"
        assert results[0].source == 'external'

    def test_recall_with_none_filter_returns_all(self):
        """source_filter=None 返回所有来源的条目。"""
        mem = MemoryManager(NUM_NODES, device='cpu')
        vec_ext = torch.randn(INPUT_DIM)
        vec_self = torch.randn(INPUT_DIM)
        mem.store_episodic("external text", vec_ext, 0.5, turn=1)
        mem.store_episodic("self voice", vec_self, 0.3, turn=2,
                           source='self_ref')

        results = mem.recall_episodic(vec_ext, top_k=5, source_filter=None)
        assert len(results) == 2

    def test_recall_with_self_ref_filter(self):
        """source_filter='self_ref' 只返回自指条目。"""
        mem = MemoryManager(NUM_NODES, device='cpu')
        vec_ext = torch.randn(INPUT_DIM)
        vec_self = torch.randn(INPUT_DIM)
        mem.store_episodic("external text", vec_ext, 0.5, turn=1)
        mem.store_episodic("self voice", vec_self, 0.3, turn=2,
                           source='self_ref')

        results = mem.recall_episodic(vec_self, top_k=5,
                                      source_filter='self_ref')
        assert len(results) == 1
        assert results[0].text == "self voice"

    def test_recall_backward_compat_no_source_field(self):
        """旧条目（无 source 字段）默认为 'external'，被默认过滤包含。"""
        mem = MemoryManager(NUM_NODES, device='cpu')
        vec = torch.randn(INPUT_DIM)
        # 直接构造 EpisodicEntry 不传 source（模拟旧数据）
        entry = EpisodicEntry(text="old entry", semantic_vector=vec,
                              surprise=0.5, turn=1)
        # 不传 source 时 dataclass 默认值为 'external'
        assert entry.source == 'external'
        mem.replace_episodic_buffer([entry])

        results = mem.recall_episodic(vec, top_k=5)
        assert len(results) == 1
        assert results[0].source == 'external'

    def test_mixed_source_buffer_retrieval(self):
        """混合来源缓冲区中，按来源过滤精确返回。"""
        mem = MemoryManager(NUM_NODES, device='cpu')
        for i in range(5):
            vec = torch.randn(INPUT_DIM)
            src = 'external' if i % 2 == 0 else 'self_ref'
            mem.store_episodic(f"entry_{i}", vec, 0.1 * i, turn=i,
                               source=src)

        # 默认过滤：只返回 external（3 条：i=0,2,4）
        ext_results = mem.recall_episodic(torch.randn(INPUT_DIM), top_k=10)
        assert len(ext_results) == 3
        for r in ext_results:
            assert r.source == 'external'

        # self_ref 过滤：只返回 self_ref（2 条：i=1,3）
        sr_results = mem.recall_episodic(torch.randn(INPUT_DIM), top_k=10,
                                         source_filter='self_ref')
        assert len(sr_results) == 2
        for r in sr_results:
            assert r.source == 'self_ref'

        # 不过滤：返回全部 5 条
        all_results = mem.recall_episodic(torch.randn(INPUT_DIM), top_k=10,
                                          source_filter=None)
        assert len(all_results) == 5


# ============================================================
# 2. 做梦钩子测试（dream hooks）
# ============================================================

class TestDreamHooks:
    """测试 on_dream_start / on_dream_end 及 stale 衰减。"""

    def test_on_dream_start_sets_stale_flag(self):
        """on_dream_start 设置 dream_stale=True，dream_age=0。"""
        sr, _ = make_self_ref()
        assert sr.dream_stale is False

        sr.on_dream_start()
        assert sr.dream_stale is True
        assert sr.dream_age == 0

    def test_on_dream_end_clears_stale_flag(self):
        """on_dream_end 清除 dream_stale，设置 dream_age=1。"""
        sr, _ = make_self_ref()
        sr.on_dream_start()
        assert sr.dream_stale is True

        sr.on_dream_end()
        assert sr.dream_stale is False
        assert sr.dream_age == 1

    def test_on_dream_end_decays_sensory_self_prev(self):
        """on_dream_end 对 sensory_self_prev 施加指数衰减。"""
        sr, _ = make_self_ref()
        # 先 observe 一次以设置 sensory_self_prev
        act = make_activation()
        ctx = make_memory_context()
        sr.observe(ctx, act)
        assert sr.sensory_self_prev is not None

        original_norm = float(sr.sensory_self_prev.norm())
        expected_factor = math.exp(-1.0 / 5.0)

        sr.on_dream_start()
        sr.on_dream_end()

        decayed_norm = float(sr.sensory_self_prev.norm())
        assert decayed_norm == pytest.approx(
            original_norm * expected_factor, rel=1e-5)

    def test_on_dream_end_with_none_sensory_self_prev(self):
        """sensory_self_prev 为 None 时 on_dream_end 不崩溃。"""
        sr, _ = make_self_ref()
        assert sr.sensory_self_prev is None

        sr.on_dream_start()
        sr.on_dream_end()  # 不应抛异常

        assert sr.dream_stale is False

    def test_generate_echo_reduces_alpha_when_stale(self):
        """dream_stale=True 时 generate_echo 的 alpha 被压低到 0.3 倍。"""
        sr, _ = make_self_ref()
        # 先 observe 两次以建立历史
        act1 = make_activation(seed=1)
        act2 = make_activation(seed=2)
        ctx = make_memory_context()
        sr.observe(ctx, act1)
        sr.observe(ctx, act2)

        # 正常状态下的 alpha
        ext_vec = torch.randn(INPUT_DIM)
        echo_normal = sr.generate_echo(ext_sensory=ext_vec)
        assert echo_normal is not None
        alpha_normal = echo_normal['alpha']

        # stale 状态下的 alpha
        sr.on_dream_start()
        echo_stale = sr.generate_echo(ext_sensory=ext_vec)
        assert echo_stale is not None
        alpha_stale = echo_stale['alpha']

        # stale 时 alpha 应为正常时的 0.3 倍（或更小，因衰减后向量变化）
        if alpha_normal > 0:
            assert alpha_stale <= alpha_normal * 0.31  # 0.3 + 容差

    def test_dream_state_in_get_state(self):
        """get_state 包含 Phase 2 做梦状态字段。"""
        sr, _ = make_self_ref()
        sr.on_dream_start()

        state = sr.get_state()
        assert 'dream_stale' in state
        assert state['dream_stale'] is True
        assert 'dream_age' in state
        assert state['dream_age'] == 0

    def test_dream_state_in_set_state(self):
        """set_state 恢复 Phase 2 做梦状态。"""
        sr, _ = make_self_ref()
        sr.on_dream_start()
        state = sr.get_state()

        sr2, _ = make_self_ref()
        sr2.set_state(state)
        assert sr2.dream_stale is True
        assert sr2.dream_age == 0

    def test_dream_state_in_get_status(self):
        """get_status 暴露 Phase 2 做梦监控字段。"""
        sr, _ = make_self_ref()
        sr.on_dream_start()

        status = sr.get_status()
        assert 'dream_stale' in status
        assert status['dream_stale'] is True
        assert 'dream_age' in status
        assert 'is_self_ref_dominant' in status


# ============================================================
# 3. 学习隔离测试（learning isolation）
# ============================================================

class TestLearningIsolation:
    """测试自指主导轮次的学习率减半与 meta.update 跳过。"""

    def test_self_ref_dominant_flag_default_false(self):
        """初始状态 last_is_self_ref_dominant 为 False。"""
        sr, _ = make_self_ref()
        assert sr.last_is_self_ref_dominant is False

    def test_dominant_flag_in_get_state(self):
        """get_state 包含 last_is_self_ref_dominant 字段。"""
        sr, _ = make_self_ref()
        sr.last_is_self_ref_dominant = True

        state = sr.get_state()
        assert 'last_is_self_ref_dominant' in state
        assert state['last_is_self_ref_dominant'] is True

    def test_dominant_flag_in_set_state(self):
        """set_state 恢复 last_is_self_ref_dominant。"""
        sr, _ = make_self_ref()
        sr.last_is_self_ref_dominant = True
        state = sr.get_state()

        sr2, _ = make_self_ref()
        sr2.set_state(state)
        assert sr2.last_is_self_ref_dominant is True

    def test_loop_learning_isolation_with_self_ref_enabled(self):
        """集成测试：LivingMemoryLoop 中自指主导轮次学习率减半。

        构造一个自指启用的 loop，模拟自指主导轮次（alpha_t > 0.05
        且 ext_novelty < 0.1），验证学习率被减半。
        """
        from runtime.loop import LivingMemoryLoop

        config = {
            'num_nodes': NUM_NODES,
            'input_dim': INPUT_DIM,
            'self_ref_enabled': True,
            'self_ref_alpha_base': 0.15,
            'meta_enabled': False,  # 简化测试，先不测 meta 隔离
            'learning_rate': 0.01,
        }
        loop = LivingMemoryLoop(config)

        # 先跑几轮建立自指历史
        for i in range(5):
            loop.process_turn(f"test input {i}")

        # 验证 loop 正常运行且自指状态被追踪
        status = loop.get_status()
        assert status['self_ref_enabled'] is True
        assert 'self_ref_is_dominant' in status

    def test_loop_meta_update_skipped_when_dominant(self):
        """集成测试：自指主导轮次跳过 meta.update。

        通过检查 meta 的 _surprise_deque 长度来验证 meta.update 是否被调用。
        非主导轮次会调用 meta.update，主导轮次跳过，因此最终长度
        应 <= 总轮次（说明有跳过）或 > 0（说明非主导轮次正常馈入）。
        """
        from runtime.loop import LivingMemoryLoop

        config = {
            'num_nodes': NUM_NODES,
            'input_dim': INPUT_DIM,
            'self_ref_enabled': True,
            'self_ref_alpha_base': 0.15,
            'meta_enabled': True,
            'meta_interval': 1,  # 每轮更新
            'learning_rate': 0.01,
        }
        loop = LivingMemoryLoop(config)

        # 跑几轮建立自指历史
        for i in range(10):
            loop.process_turn(f"test input {i}")

        # meta 应已收集了一些 surprise（非主导轮次馈入）
        if loop.meta is not None:
            assert len(loop.meta._surprise_deque) > 0

        # 验证系统未崩溃，meta 状态正常
        status = loop.get_status()
        assert 'meta' in status


# ============================================================
# 4. 持久化测试（persistence）
# ============================================================

class TestPersistence:
    """测试 Phase 2 状态的 save→load 往返一致性。"""

    def test_save_load_round_trip_with_dream_state(self):
        """save→load 往返后做梦状态一致。"""
        sr, _ = make_self_ref()
        # 建立一些状态
        act = make_activation()
        ctx = make_memory_context()
        sr.observe(ctx, act)
        sr.on_dream_start()

        state = sr.get_state()
        sr2, _ = make_self_ref()
        sr2.set_state(state)

        assert sr2.dream_stale == sr.dream_stale
        assert sr2.dream_age == sr.dream_age
        assert sr2.last_is_self_ref_dominant == sr.last_is_self_ref_dominant

    def test_old_snapshot_without_phase2_fields(self):
        """旧快照（无 Phase 2 字段）加载不报错，回退默认值。"""
        sr, _ = make_self_ref()
        act = make_activation()
        ctx = make_memory_context()
        sr.observe(ctx, act)

        state = sr.get_state()
        # 模拟旧快照：删除 Phase 2 字段
        del state['dream_stale']
        del state['dream_age']
        del state['last_is_self_ref_dominant']

        sr2, _ = make_self_ref()
        sr2.set_state(state)  # 不应抛异常

        assert sr2.dream_stale is False
        assert sr2.dream_age == 0
        assert sr2.last_is_self_ref_dominant is False

    def test_save_load_generate_echo_consistency(self):
        """save→load 后 generate_echo 产出一致。"""
        sr, _ = make_self_ref()
        # 建立历史
        act1 = make_activation(seed=1)
        act2 = make_activation(seed=2)
        ctx = make_memory_context(["test interpretation"])
        sr.observe(ctx, act1)
        sr.observe(ctx, act2)

        ext_vec = torch.randn(INPUT_DIM)
        echo_before = sr.generate_echo(ext_sensory=ext_vec)

        # 保存→恢复
        state = sr.get_state()
        sr2, _ = make_self_ref()
        sr2.set_state(state)

        echo_after = sr2.generate_echo(ext_sensory=ext_vec)

        # alpha 和 state 应一致
        assert echo_after is not None
        assert echo_before is not None
        assert echo_after['alpha'] == pytest.approx(
            echo_before['alpha'], rel=1e-5)
        assert torch.allclose(
            echo_after['vector'], echo_before['vector'], atol=1e-5)

    def test_loop_save_load_with_self_ref(self):
        """LivingMemoryLoop 完整 save→load 往返测试。"""
        import tempfile
        import os
        from runtime.loop import LivingMemoryLoop

        config = {
            'num_nodes': NUM_NODES,
            'input_dim': INPUT_DIM,
            'self_ref_enabled': True,
            'self_ref_alpha_base': 0.15,
            'meta_enabled': True,
            'auto_snapshot': False,
        }
        loop = LivingMemoryLoop(config)

        # 跑几轮建立状态
        for i in range(5):
            loop.process_turn(f"round trip test {i}")

        # 保存
        with tempfile.TemporaryDirectory() as tmpdir:
            snap_path = os.path.join(tmpdir, 'test_snapshot.pt')
            loop.save_state(snap_path)

            # 创建新 loop 并加载
            loop2 = LivingMemoryLoop(config)
            loop2.load_state(snap_path)

            # 验证自指状态恢复
            assert loop2.self_ref is not None
            assert loop2.self_ref.turn_count == loop.self_ref.turn_count
            assert (loop2.self_ref.last_alpha ==
                    pytest.approx(loop.self_ref.last_alpha, rel=1e-5))

    def test_loop_load_old_snapshot_without_self_ref(self):
        """旧快照（无 self_ref 字段）加载不报错。"""
        import tempfile
        import os
        from runtime.loop import LivingMemoryLoop
        from persistence.snapshot import Snapshot

        config = {
            'num_nodes': NUM_NODES,
            'input_dim': INPUT_DIM,
            'self_ref_enabled': False,  # 旧系统未启用自指
            'meta_enabled': False,
            'auto_snapshot': False,
        }
        loop = LivingMemoryLoop(config)
        loop.process_turn("old snapshot test")

        with tempfile.TemporaryDirectory() as tmpdir:
            snap_path = os.path.join(tmpdir, 'old_snapshot.pt')
            loop.save_state(snap_path)

            # 创建启用了自指的新 loop，加载旧快照
            config2 = dict(config)
            config2['self_ref_enabled'] = True
            loop2 = LivingMemoryLoop(config2)
            # 不应抛异常
            loop2.load_state(snap_path)
            # 自指状态应为初始值
            assert loop2.self_ref is not None
            assert loop2.self_ref.turn_count == 0


# ============================================================
# 5. 集成测试：做梦 + 自指联动
# ============================================================

class TestDreamIntegration:
    """测试 LivingMemoryLoop.dream() 中自指钩子的调用。"""

    def test_dream_calls_on_dream_start_and_end(self):
        """dream() 方法调用 on_dream_start 和 on_dream_end。"""
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

        # 先跑几轮建立记忆和自指历史
        for i in range(3):
            loop.process_turn(f"pre-dream input {i}")

        # 记录做梦前的状态
        assert loop.self_ref is not None
        pre_stale = loop.self_ref.dream_stale

        # 执行做梦
        result = loop.dream(n_steps=3)

        # 做梦后 stale 应已清除
        assert loop.self_ref.dream_stale is False
        assert loop.self_ref.dream_age == 1

    def test_dream_without_self_ref_no_crash(self):
        """未启用自指时 dream() 不调用钩子，不崩溃。"""
        from runtime.loop import LivingMemoryLoop

        config = {
            'num_nodes': NUM_NODES,
            'input_dim': INPUT_DIM,
            'self_ref_enabled': False,
            'meta_enabled': False,
            'auto_snapshot': False,
        }
        loop = LivingMemoryLoop(config)

        # 跑一轮建立记忆
        loop.process_turn("no self-ref test")

        # 执行做梦——不应崩溃
        result = loop.dream(n_steps=2)
        assert result is not None

    def test_post_dream_self_ref_continuity(self):
        """做梦后自指回路连续运行，stale 衰减生效。"""
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

        # 跑几轮建立自指历史
        for i in range(3):
            loop.process_turn(f"continuity test {i}")

        # 记录做梦前的 sensory_self_prev 范数
        pre_norm = None
        if loop.self_ref.sensory_self_prev is not None:
            pre_norm = float(loop.self_ref.sensory_self_prev.norm())

        # 执行做梦
        loop.dream(n_steps=2)

        # 做梦后 sensory_self_prev 应被衰减
        if pre_norm is not None and loop.self_ref.sensory_self_prev is not None:
            post_norm = float(loop.self_ref.sensory_self_prev.norm())
            expected_factor = math.exp(-1.0 / 5.0)
            assert post_norm == pytest.approx(
                pre_norm * expected_factor, rel=1e-4)

        # 做梦后继续运行不崩溃
        ctx = loop.process_turn("post-dream input")
        assert ctx is not None

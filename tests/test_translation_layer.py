"""
LMS↔沙漏翻译层测试
====================

测试 TranslationLayer 的双向转换功能：
  1. extract_explicit_memories: 从记忆管理器提取显式记忆条目
  2. extract_activation_pattern: 激活态模式描述
  3. inject_explicit_memory: 文本 -> 感官输入
  4. create_context_injection: 格式化上下文输出
  5. 双向同步（sync_to_hourglass + recall_from_hourglass）
  6. 空记忆与边界情况

使用轻量级配置（num_nodes=32, input_dim=16），不依赖外部服务或预训练模型。
"""

import json
import pytest
import torch

from bridge.translation_layer import (
    TranslationLayer,
    HourglassClient,
    InMemoryHourglassClient,
)
from bridge.encoder import Encoder
from core.types import Activation, SensoryInput
from core.hippocampus.memory import MemoryManager
from core.sensory.tokenizer import SimpleTokenizer
from core.sensory.embedder import SimpleEmbedder


# ============================================================
# 测试辅助：轻量级确定性文本嵌入器
# ============================================================

class FakeTextEmbedder:
    """测试用文本嵌入器：基于字符 ord 的确定性嵌入。

    无需预训练模型，同一段文本始终产生相同向量，
    相似文本（共享字符）产生相似向量，适合测试检索功能。

    提供 embed_text 方法（与 PretrainedEmbedder 接口兼容），
    但不继承 Embedder 抽象基类（避免强制实现 embed(tokens)）。
    """

    def __init__(self, dim: int = 16, seed: int = 42) -> None:
        self._dim = dim
        self._seed = seed

    def embed_text(self, text: str) -> torch.Tensor:
        """文本 -> 确定性语义向量（L2 归一化）。"""
        if not text or not text.strip():
            return torch.zeros(self._dim)
        vec = torch.zeros(self._dim)
        for i, ch in enumerate(text):
            # 用 ord 和位置构造确定性值（不依赖 PYTHONHASHSEED）
            val = (ord(ch) * (i + 1) + self._seed) % 9973
            vec[val % self._dim] += (val / 9973.0) * 2.0 - 1.0
        norm = vec.norm()
        if norm > 1e-8:
            vec = vec / norm
        return vec

    @property
    def dim(self) -> int:
        return self._dim


# ============================================================
# 测试常量
# ============================================================

NUM_NODES = 32
INPUT_DIM = 16


# ============================================================
# 辅助函数
# ============================================================

def make_memory_manager_with_entries(
    num_entries: int = 3,
    num_nodes: int = NUM_NODES,
    input_dim: int = INPUT_DIM,
) -> MemoryManager:
    """构造已填充情景记忆条目的 MemoryManager。"""
    mem = MemoryManager(num_nodes=num_nodes)
    texts = [f"测试记忆条目_{i}" for i in range(num_entries)]
    surprises = [0.5 + i * 0.5 for i in range(num_entries)]  # 0.5, 1.0, 1.5...
    for i, (text, surprise) in enumerate(zip(texts, surprises)):
        vec = torch.randn(input_dim)
        mem.store_episodic(text, vec, surprise=surprise, turn=i)
    return mem


# ============================================================
# HourglassClient 抽象接口测试
# ============================================================

class TestHourglassClientInterface:
    """HourglassClient 抽象基类与 InMemoryHourglassClient 测试。"""

    def test_cannot_instantiate_abstract(self):
        """抽象基类不能直接实例化。"""
        with pytest.raises(TypeError):
            HourglassClient()

    def test_in_memory_store_returns_id(self):
        """store 返回字符串 ID。"""
        client = InMemoryHourglassClient()
        record_id = client.store({
            'text': 'hello',
            'semantic_vector': [0.1, 0.2, 0.3],
        })
        assert isinstance(record_id, str)
        assert len(record_id) > 0

    def test_in_memory_search_by_vector(self):
        """search 按余弦相似度检索。"""
        client = InMemoryHourglassClient()
        vec = torch.tensor([1.0, 0.0, 0.0, 0.0])
        client.store({
            'text': 'target',
            'semantic_vector': vec.tolist(),
        })
        client.store({
            'text': 'other',
            'semantic_vector': [0.0, 1.0, 0.0, 0.0],
        })

        # 查询与第一条完全一致
        results = client.search(vec, top_k=2)
        assert len(results) == 2
        assert results[0]['text'] == 'target'
        assert results[0]['score'] > results[1]['score']

    def test_in_memory_search_empty(self):
        """空存储时 search 返回空列表。"""
        client = InMemoryHourglassClient()
        results = client.search(torch.randn(4), top_k=3)
        assert results == []

    def test_in_memory_delete(self):
        """delete 能删除记录。"""
        client = InMemoryHourglassClient()
        record_id = client.store({
            'text': 'to_delete',
            'semantic_vector': [1.0, 0.0],
        })
        assert client.size() == 1
        assert client.delete(record_id) is True
        assert client.size() == 0

    def test_in_memory_delete_nonexistent(self):
        """删除不存在的 ID 返回 False。"""
        client = InMemoryHourglassClient()
        assert client.delete('nonexistent-id') is False

    def test_in_memory_search_dimension_mismatch(self):
        """维度不匹配的条目被跳过（优雅降级）。"""
        client = InMemoryHourglassClient()
        client.store({
            'text': 'dim4',
            'semantic_vector': [1.0, 0.0, 0.0, 0.0],
        })
        client.store({
            'text': 'dim8',
            'semantic_vector': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        })
        # 用 4 维查询，应只匹配 dim4 条目
        results = client.search(torch.tensor([1.0, 0.0, 0.0, 0.0]), top_k=5)
        assert len(results) == 1
        assert results[0]['text'] == 'dim4'

    def test_in_memory_search_results_have_score(self):
        """检索结果包含 score 字段。"""
        client = InMemoryHourglassClient()
        client.store({
            'text': 'test',
            'semantic_vector': [1.0, 0.0],
        })
        results = client.search(torch.tensor([1.0, 0.0]), top_k=1)
        assert 'score' in results[0]
        assert 'record_id' in results[0]

    def test_in_memory_search_top_k_limit(self):
        """top_k 限制返回数量。"""
        client = InMemoryHourglassClient()
        for i in range(5):
            client.store({
                'text': f'item_{i}',
                'semantic_vector': torch.randn(4).tolist(),
            })
        results = client.search(torch.randn(4), top_k=3)
        assert len(results) == 3


# ============================================================
# extract_explicit_memories 测试
# ============================================================

class TestExtractExplicitMemories:
    """从记忆管理器提取显式记忆条目。"""

    def test_extract_basic(self):
        """基本提取：返回正确数量与格式的条目。"""
        mem = make_memory_manager_with_entries(num_entries=3)
        tl = TranslationLayer()
        results = tl.extract_explicit_memories(mem, top_k=5)

        assert len(results) == 3
        for entry in results:
            assert 'text' in entry
            assert 'semantic_vector' in entry
            assert 'timestamp' in entry
            assert 'surprise' in entry
            assert 'activation_strength' in entry
            assert 'source' in entry
            assert entry['source'] == 'lms_episodic'

    def test_extract_sorted_by_surprise(self):
        """按惊讶度降序排列。"""
        mem = make_memory_manager_with_entries(num_entries=3)
        tl = TranslationLayer()
        results = tl.extract_explicit_memories(mem, top_k=5)

        surprises = [r['surprise'] for r in results]
        assert surprises == sorted(surprises, reverse=True)
        # 第一条惊讶度最高（1.5）
        assert results[0]['surprise'] == pytest.approx(1.5)

    def test_extract_top_k_limit(self):
        """top_k 限制提取数量。"""
        mem = make_memory_manager_with_entries(num_entries=5)
        tl = TranslationLayer()
        results = tl.extract_explicit_memories(mem, top_k=3)
        assert len(results) == 3

    def test_extract_top_k_exceeds_size(self):
        """top_k 大于条目数时返回全部。"""
        mem = make_memory_manager_with_entries(num_entries=2)
        tl = TranslationLayer()
        results = tl.extract_explicit_memories(mem, top_k=10)
        assert len(results) == 2

    def test_extract_all_with_zero_top_k(self):
        """top_k=0 提取全部条目。"""
        mem = make_memory_manager_with_entries(num_entries=4)
        tl = TranslationLayer()
        results = tl.extract_explicit_memories(mem, top_k=0)
        assert len(results) == 4

    def test_extract_empty_memory(self):
        """空记忆缓冲区返回空列表。"""
        mem = MemoryManager(num_nodes=NUM_NODES)
        tl = TranslationLayer()
        results = tl.extract_explicit_memories(mem, top_k=5)
        assert results == []

    def test_extract_serializable(self):
        """提取结果可 JSON 序列化。"""
        mem = make_memory_manager_with_entries(num_entries=2)
        tl = TranslationLayer()
        results = tl.extract_explicit_memories(mem, top_k=5)

        # 应能成功 JSON 序列化（无 tensor 残留）
        serialized = json.dumps(results, ensure_ascii=False)
        restored = json.loads(serialized)
        assert len(restored) == 2
        assert isinstance(restored[0]['semantic_vector'], list)

    def test_extract_semantic_vector_is_list(self):
        """semantic_vector 转换为 list（非 tensor）。"""
        mem = make_memory_manager_with_entries(num_entries=1)
        tl = TranslationLayer()
        results = tl.extract_explicit_memories(mem, top_k=5)
        assert isinstance(results[0]['semantic_vector'], list)

    def test_extract_activation_strength_positive(self):
        """activation_strength 为正数（L2 范数）。"""
        mem = make_memory_manager_with_entries(num_entries=1)
        tl = TranslationLayer()
        results = tl.extract_explicit_memories(mem, top_k=5)
        assert results[0]['activation_strength'] > 0

    def test_extract_timestamp_matches_turn(self):
        """timestamp 与条目的 turn 一致。"""
        mem = MemoryManager(num_nodes=NUM_NODES)
        vec = torch.randn(INPUT_DIM)
        mem.store_episodic("turn5_text", vec, surprise=1.0, turn=5)
        tl = TranslationLayer()
        results = tl.extract_explicit_memories(mem, top_k=5)
        assert results[0]['timestamp'] == 5


# ============================================================
# extract_activation_pattern 测试
# ============================================================

class TestExtractActivationPattern:
    """激活态模式描述测试。"""

    def test_basic_pattern(self):
        """基本模式提取。"""
        state = torch.zeros(NUM_NODES)
        state[0] = 0.8
        state[5] = 0.3
        state[10] = -0.6
        activation = Activation(state=state, entropy=2.0, surprise=0.5)

        tl = TranslationLayer()
        pattern = tl.extract_activation_pattern(activation)

        assert pattern['num_nodes'] == NUM_NODES
        assert pattern['entropy'] == pytest.approx(2.0)
        assert pattern['surprise'] == pytest.approx(0.5)
        assert pattern['num_active'] >= 1
        assert isinstance(pattern['active_nodes'], list)

    def test_active_nodes_contain_top(self):
        """active_nodes 包含最强激活节点。"""
        state = torch.zeros(NUM_NODES)
        state[3] = 0.9
        activation = Activation(state=state, entropy=1.0, surprise=0.2)

        tl = TranslationLayer()
        pattern = tl.extract_activation_pattern(activation, top_k=5)

        node_ids = [n['node_id'] for n in pattern['active_nodes']]
        assert 3 in node_ids
        # 节点3的 strength 应为 0.9
        node3 = [n for n in pattern['active_nodes'] if n['node_id'] == 3][0]
        assert node3['strength'] == pytest.approx(0.9)
        assert node3['activation'] == pytest.approx(0.9)

    def test_threshold_filtering(self):
        """低于阈值的节点被过滤。"""
        state = torch.zeros(NUM_NODES)
        state[0] = 0.5
        state[1] = 0.001  # 低于默认阈值 0.01
        activation = Activation(state=state, entropy=0.5, surprise=0.1)

        tl = TranslationLayer()
        pattern = tl.extract_activation_pattern(activation, top_k=10,
                                                threshold=0.01)
        node_ids = [n['node_id'] for n in pattern['active_nodes']]
        assert 0 in node_ids
        assert 1 not in node_ids

    def test_zero_activation(self):
        """零激活态：无显著节点。"""
        activation = Activation(
            state=torch.zeros(NUM_NODES), entropy=0.0, surprise=0.0)
        tl = TranslationLayer()
        pattern = tl.extract_activation_pattern(activation)

        assert pattern['num_active'] == 0
        assert pattern['active_nodes'] == []
        assert pattern['entropy'] == pytest.approx(0.0)
        assert pattern['activation_norm'] == pytest.approx(0.0)

    def test_entropy_ratio(self):
        """entropy_ratio 在 [0, 1] 范围内。"""
        import math
        state = torch.randn(NUM_NODES) * 0.3
        activation = Activation(
            state=state, entropy=math.log(NUM_NODES) * 0.5, surprise=0.3)

        tl = TranslationLayer()
        pattern = tl.extract_activation_pattern(activation)

        assert 0.0 <= pattern['entropy_ratio'] <= 1.0
        assert pattern['entropy_ratio'] == pytest.approx(0.5, abs=0.01)

    def test_negative_activation(self):
        """负激活值正确记录带符号 activation。"""
        state = torch.zeros(NUM_NODES)
        state[7] = -0.7
        activation = Activation(state=state, entropy=0.3, surprise=0.4)

        tl = TranslationLayer()
        pattern = tl.extract_activation_pattern(activation)

        node7 = [n for n in pattern['active_nodes'] if n['node_id'] == 7][0]
        assert node7['activation'] == pytest.approx(-0.7)
        assert node7['strength'] == pytest.approx(0.7)

    def test_serializable(self):
        """模式描述可 JSON 序列化。"""
        state = torch.randn(NUM_NODES) * 0.5
        activation = Activation(state=state, entropy=1.5, surprise=0.8)
        tl = TranslationLayer()
        pattern = tl.extract_activation_pattern(activation)

        serialized = json.dumps(pattern)
        restored = json.loads(serialized)
        assert restored['num_nodes'] == NUM_NODES


# ============================================================
# inject_explicit_memory 测试
# ============================================================

class TestInjectExplicitMemory:
    """文本 -> 感官输入转换测试。"""

    def test_returns_sensory_input(self):
        """返回 SensoryInput 实例。"""
        encoder = Encoder()
        tokenizer = SimpleTokenizer()
        embedder = SimpleEmbedder(dim=INPUT_DIM)
        tl = TranslationLayer()

        result = tl.inject_explicit_memory(
            "hello world", encoder, tokenizer, embedder)

        assert isinstance(result, SensoryInput)

    def test_vector_dimension(self):
        """向量维度与 embedder 一致。"""
        encoder = Encoder()
        tokenizer = SimpleTokenizer()
        embedder = SimpleEmbedder(dim=INPUT_DIM)
        tl = TranslationLayer()

        result = tl.inject_explicit_memory(
            "test text", encoder, tokenizer, embedder)
        assert result.vector.shape == (INPUT_DIM,)

    def test_source_metadata(self):
        """metadata 标注 source 为 hourglass_injection。"""
        encoder = Encoder()
        tokenizer = SimpleTokenizer()
        embedder = SimpleEmbedder(dim=INPUT_DIM)
        tl = TranslationLayer()

        result = tl.inject_explicit_memory(
            "injected memory", encoder, tokenizer, embedder)
        assert result.metadata['source'] == 'hourglass_injection'

    def test_injected_text_metadata(self):
        """metadata 包含截断的原始文本。"""
        encoder = Encoder()
        tokenizer = SimpleTokenizer()
        embedder = SimpleEmbedder(dim=INPUT_DIM)
        tl = TranslationLayer()

        long_text = "a" * 300
        result = tl.inject_explicit_memory(
            long_text, encoder, tokenizer, embedder)
        # 截断到 200 字符
        assert len(result.metadata['injected_text']) == 200

    def test_empty_text(self):
        """空文本返回零向量。"""
        encoder = Encoder()
        tokenizer = SimpleTokenizer()
        embedder = SimpleEmbedder(dim=INPUT_DIM)
        tl = TranslationLayer()

        result = tl.inject_explicit_memory(
            "", encoder, tokenizer, embedder)
        assert torch.allclose(result.vector, torch.zeros(INPUT_DIM))

    def test_with_text_embedder(self):
        """使用具备 embed_text 的嵌入器（文本路径）。"""
        encoder = Encoder()
        tokenizer = SimpleTokenizer()
        embedder = FakeTextEmbedder(dim=INPUT_DIM)
        tl = TranslationLayer()

        result = tl.inject_explicit_memory(
            "语义文本", encoder, tokenizer, embedder)
        assert result.vector.shape == (INPUT_DIM,)
        assert result.metadata['source'] == 'hourglass_injection'

    def test_consistency(self):
        """相同文本产生相同向量。"""
        encoder = Encoder()
        tokenizer = SimpleTokenizer()
        embedder = SimpleEmbedder(dim=INPUT_DIM)
        tl = TranslationLayer()

        r1 = tl.inject_explicit_memory(
            "same text", encoder, tokenizer, embedder)
        r2 = tl.inject_explicit_memory(
            "same text", encoder, tokenizer, embedder)
        assert torch.allclose(r1.vector, r2.vector)


# ============================================================
# create_context_injection 测试
# ============================================================

class TestCreateContextInjection:
    """上下文格式化输出测试。"""

    def test_basic_formatting(self):
        """基本格式化输出。"""
        memories = [
            {'text': '记忆一', 'surprise': 0.5, 'timestamp': 0,
             'activation_strength': 1.2},
            {'text': '记忆二', 'surprise': 1.0, 'timestamp': 1,
             'activation_strength': 0.8},
        ]
        tl = TranslationLayer()
        result = tl.create_context_injection(memories, max_tokens=500)

        assert '[显式记忆回忆]' in result
        assert '记忆一' in result
        assert '记忆二' in result
        assert '惊讶度' in result
        assert '强度' in result

    def test_empty_memories(self):
        """空列表返回空字符串。"""
        tl = TranslationLayer()
        result = tl.create_context_injection([], max_tokens=500)
        assert result == ""

    def test_max_tokens_truncation(self):
        """max_tokens 限制输出长度。"""
        memories = [
            {'text': f'记忆条目_{i}', 'surprise': 0.5,
             'timestamp': i, 'activation_strength': 1.0}
            for i in range(20)
        ]
        tl = TranslationLayer()
        result = tl.create_context_injection(memories, max_tokens=100)

        # 输出应被截断（包含标题但不包含全部20条）
        assert '[显式记忆回忆]' in result
        assert '记忆条目_19' not in result  # 最后一条因预算不足被截断

    def test_single_memory(self):
        """单条记忆格式化。"""
        memories = [
            {'text': '唯一记忆', 'surprise': 0.3, 'timestamp': 0,
             'activation_strength': 0.5},
        ]
        tl = TranslationLayer()
        result = tl.create_context_injection(memories, max_tokens=500)

        assert '唯一记忆' in result
        assert '0.30' in result  # surprise 格式化

    def test_missing_fields_default(self):
        """缺失字段以默认值填充。"""
        memories = [{'text': 'partial'}]  # 缺 surprise/timestamp/strength
        tl = TranslationLayer()
        result = tl.create_context_injection(memories, max_tokens=500)

        assert 'partial' in result
        assert '惊讶度:0.00' in result

    def test_long_text_truncation(self):
        """单条文本超 200 字符时截断。"""
        long_text = "x" * 300
        memories = [
            {'text': long_text, 'surprise': 0.5, 'timestamp': 0,
             'activation_strength': 1.0},
        ]
        tl = TranslationLayer()
        result = tl.create_context_injection(memories, max_tokens=1000)

        # 截断后应包含 "..." 且不含完整的 300 字符
        assert '...' in result
        assert result.count('x') < 300

    def test_numbering(self):
        """条目正确编号。"""
        memories = [
            {'text': f'mem_{i}', 'surprise': 0.1 * i, 'timestamp': i,
             'activation_strength': 1.0}
            for i in range(3)
        ]
        tl = TranslationLayer()
        result = tl.create_context_injection(memories, max_tokens=500)

        assert '1.' in result
        assert '2.' in result
        assert '3.' in result


# ============================================================
# 双向同步测试
# ============================================================

class TestBidirectionalSync:
    """sync_to_hourglass + recall_from_hourglass 双向同步。"""

    def test_sync_returns_count(self):
        """sync 返回同步条目数。"""
        mem = make_memory_manager_with_entries(num_entries=3)
        client = InMemoryHourglassClient()
        tl = TranslationLayer()

        count = tl.sync_to_hourglass(mem, client)
        assert count == 3
        assert client.size() == 3

    def test_sync_empty_memory(self):
        """空记忆同步返回 0。"""
        mem = MemoryManager(num_nodes=NUM_NODES)
        client = InMemoryHourglassClient()
        tl = TranslationLayer()

        count = tl.sync_to_hourglass(mem, client)
        assert count == 0

    def test_recall_after_sync(self):
        """同步后能检索到记忆。"""
        embedder = FakeTextEmbedder(dim=INPUT_DIM)
        mem = MemoryManager(num_nodes=NUM_NODES)

        texts = ["你好世界", "今天天气不错", "机器学习很有趣"]
        for i, text in enumerate(texts):
            vec = embedder.embed_text(text)
            mem.store_episodic(text, vec, surprise=float(i), turn=i)

        client = InMemoryHourglassClient()
        tl = TranslationLayer(hourglass_client=client)

        # 同步
        count = tl.sync_to_hourglass(mem, client)
        assert count == 3

        # 用相同文本检索——应精确匹配
        results = tl.recall_from_hourglass("你好世界", embedder, top_k=3)
        assert len(results) >= 1
        assert results[0]['text'] == "你好世界"
        assert results[0]['score'] == pytest.approx(1.0, abs=1e-5)

    def test_recall_top_k(self):
        """recall 的 top_k 限制返回数量。"""
        embedder = FakeTextEmbedder(dim=INPUT_DIM)
        mem = MemoryManager(num_nodes=NUM_NODES)

        texts = [f"unique_text_{i}" for i in range(5)]
        for i, text in enumerate(texts):
            vec = embedder.embed_text(text)
            mem.store_episodic(text, vec, surprise=1.0, turn=i)

        client = InMemoryHourglassClient()
        tl = TranslationLayer(hourglass_client=client)
        tl.sync_to_hourglass(mem, client)

        results = tl.recall_from_hourglass("unique_text_0", embedder, top_k=2)
        assert len(results) <= 2

    def test_recall_no_client_raises(self):
        """未配置 client 时 recall 抛出 RuntimeError。"""
        embedder = FakeTextEmbedder(dim=INPUT_DIM)
        tl = TranslationLayer()  # 无 client
        with pytest.raises(RuntimeError, match="未配置 hourglass_client"):
            tl.recall_from_hourglass("query", embedder)

    def test_recall_unsupported_embedder_raises(self):
        """embedder 无 embed_text 时抛出 ValueError。"""
        client = InMemoryHourglassClient()
        tl = TranslationLayer(hourglass_client=client)
        embedder = SimpleEmbedder(dim=INPUT_DIM)  # 无 embed_text

        with pytest.raises(ValueError, match="embed_text"):
            tl.recall_from_hourglass("query", embedder)

    def test_recall_empty_hourglass(self):
        """空沙漏检索返回空列表。"""
        embedder = FakeTextEmbedder(dim=INPUT_DIM)
        client = InMemoryHourglassClient()
        tl = TranslationLayer(hourglass_client=client)

        results = tl.recall_from_hourglass("anything", embedder, top_k=3)
        assert results == []

    def test_full_round_trip(self):
        """完整往返：提取 -> 同步 -> 检索 -> 注入。"""
        embedder = FakeTextEmbedder(dim=INPUT_DIM)
        mem = MemoryManager(num_nodes=NUM_NODES)

        # 存入记忆
        text = "地球绕着太阳转"
        vec = embedder.embed_text(text)
        mem.store_episodic(text, vec, surprise=1.5, turn=0)

        client = InMemoryHourglassClient()
        tl = TranslationLayer(hourglass_client=client)

        # 1. 同步到沙漏
        assert tl.sync_to_hourglass(mem, client) == 1

        # 2. 从沙漏检索
        results = tl.recall_from_hourglass(text, embedder, top_k=1)
        assert len(results) == 1
        assert results[0]['text'] == text

        # 3. 格式化为上下文
        context = tl.create_context_injection(results, max_tokens=500)
        assert text in context

        # 4. 注入为感官输入
        encoder = Encoder()
        tokenizer = SimpleTokenizer()
        sensory = tl.inject_explicit_memory(
            results[0]['text'], encoder, tokenizer, embedder)
        assert isinstance(sensory, SensoryInput)
        assert sensory.metadata['source'] == 'hourglass_injection'

    def test_sync_preserves_all_entries(self):
        """同步保留全部条目（不限 top_k）。"""
        mem = make_memory_manager_with_entries(num_entries=10)
        client = InMemoryHourglassClient()
        tl = TranslationLayer()

        count = tl.sync_to_hourglass(mem, client)
        assert count == 10


# ============================================================
# 边界情况与容错测试
# ============================================================

class TestEdgeCases:
    """空记忆、边界情况与容错测试。"""

    def test_extract_from_fresh_memory(self):
        """全新 MemoryManager（无任何记忆）提取返回空。"""
        mem = MemoryManager(num_nodes=NUM_NODES)
        tl = TranslationLayer()
        assert tl.extract_explicit_memories(mem) == []

    def test_extract_activation_single_node(self):
        """单节点激活态。"""
        activation = Activation(
            state=torch.tensor([0.5]), entropy=0.0, surprise=0.1)
        tl = TranslationLayer()
        pattern = tl.extract_activation_pattern(activation)
        assert pattern['num_nodes'] == 1
        assert pattern['num_active'] == 1

    def test_create_context_very_small_budget(self):
        """极小 max_tokens 预算。"""
        memories = [
            {'text': 'short', 'surprise': 0.1, 'timestamp': 0,
             'activation_strength': 0.5},
        ]
        tl = TranslationLayer()
        result = tl.create_context_injection(memories, max_tokens=5)
        # 预算太小，可能返回空字符串或仅标题
        assert isinstance(result, str)

    def test_create_context_with_none_values(self):
        """记忆条目含 None 值时不崩溃。"""
        memories = [
            {'text': None, 'surprise': None, 'timestamp': None,
             'activation_strength': None},
        ]
        tl = TranslationLayer()
        result = tl.create_context_injection(memories, max_tokens=500)
        assert isinstance(result, str)

    def test_sync_with_failing_client(self):
        """client.store 失败时容错（不影响其他条目）。"""
        mem = make_memory_manager_with_entries(num_entries=3)
        tl = TranslationLayer()

        # 创建一个部分失败的 client
        class PartialFailClient(HourglassClient):
            def __init__(self):
                self.stored = 0
                self._call_count = 0

            def store(self, record):
                self._call_count += 1
                if self._call_count == 2:
                    raise Exception("模拟存储失败")
                self.stored += 1
                return f"id_{self.stored}"

            def search(self, query_vector, top_k=3):
                return []

            def delete(self, record_id):
                return False

        client = PartialFailClient()
        count = tl.sync_to_hourglass(mem, client)
        # 3 条中第2条失败，应成功 2 条
        assert count == 2

    def test_in_memory_client_store_tensor_vector(self):
        """InMemoryHourglassClient 支持 tensor 向量存储。"""
        client = InMemoryHourglassClient()
        vec = torch.tensor([1.0, 0.0, 0.0])
        client.store({'text': 'tensor_vec', 'semantic_vector': vec})

        results = client.search(torch.tensor([1.0, 0.0, 0.0]), top_k=1)
        assert len(results) == 1
        assert results[0]['text'] == 'tensor_vec'

    def test_in_memory_client_store_list_vector(self):
        """InMemoryHourglassClient 支持 list 向量存储。"""
        client = InMemoryHourglassClient()
        client.store({'text': 'list_vec', 'semantic_vector': [1.0, 0.0, 0.0]})

        results = client.search(torch.tensor([1.0, 0.0, 0.0]), top_k=1)
        assert len(results) == 1
        assert results[0]['text'] == 'list_vec'

    def test_translation_layer_no_client_init(self):
        """无 client 初始化不报错。"""
        tl = TranslationLayer()
        assert tl.hourglass_client is None

    def test_translation_layer_with_client_init(self):
        """带 client 初始化正确存储。"""
        client = InMemoryHourglassClient()
        tl = TranslationLayer(hourglass_client=client)
        assert tl.hourglass_client is client

    def test_extract_top_k_negative(self):
        """top_k 为负数时提取全部。"""
        mem = make_memory_manager_with_entries(num_entries=3)
        tl = TranslationLayer()
        results = tl.extract_explicit_memories(mem, top_k=-1)
        assert len(results) == 3

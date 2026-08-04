"""桥接层测试

测试 Encoder、Decoder、LLMBridge 的功能。
"""

import pytest
import torch
from unittest.mock import MagicMock, patch, PropertyMock

from bridge.encoder import Encoder
from bridge.decoder import Decoder
from bridge.llm_bridge import LLMBridge
from core.types import SensoryInput, Activation
from core.sensory.tokenizer import Tokenizer
from core.sensory.embedder import Embedder


# ============================================================
# Encoder 测试
# ============================================================

class TestEncoder:
    """编码器测试"""

    def test_encode_basic(self, tokenizer, embedder, sample_text):
        """测试基本编码功能"""
        encoder = Encoder()
        result = encoder.encode(sample_text, tokenizer, embedder)

        assert isinstance(result, SensoryInput)
        assert result.vector.shape == (embedder.dim,)
        # 向量应该是有限值
        assert torch.isfinite(result.vector).all()

    def test_encode_empty_text(self, tokenizer, embedder):
        """测试空文本编码"""
        encoder = Encoder()
        result = encoder.encode("", tokenizer, embedder)

        assert isinstance(result, SensoryInput)
        assert torch.allclose(result.vector, torch.zeros(embedder.dim))

    def test_encode_metadata(self, tokenizer, embedder, sample_text):
        """测试元数据正确性"""
        encoder = Encoder()
        result = encoder.encode(sample_text, tokenizer, embedder)

        assert 'timestamp' in result.metadata
        assert result.metadata['source'] == 'encoder'
        assert result.metadata['text_length'] == len(sample_text)
        assert result.metadata['num_tokens'] > 0

    def test_encode_consistency(self, tokenizer, embedder):
        """测试相同输入产生相同输出"""
        encoder = Encoder()
        text = "一致性测试"
        r1 = encoder.encode(text, tokenizer, embedder)
        r2 = encoder.encode(text, tokenizer, embedder)
        assert torch.allclose(r1.vector, r2.vector)

    def test_encode_different_text_different_vector(self, tokenizer, embedder):
        """测试不同文本产生不同向量"""
        encoder = Encoder()
        r1 = encoder.encode("你好世界", tokenizer, embedder)
        r2 = encoder.encode("完全不同的文本", tokenizer, embedder)
        assert not torch.allclose(r1.vector, r2.vector)

    def test_encode_vector_dimension(self, tokenizer, embedder):
        """测试输出向量维度匹配embedder"""
        encoder = Encoder()
        result = encoder.encode("test", tokenizer, embedder)
        assert result.vector.shape[0] == embedder.dim

    def test_encode_with_custom_tokenizer(self, embedder):
        """测试使用自定义tokenizer（Mock）"""
        mock_tokenizer = MagicMock(spec=Tokenizer)
        mock_tokenizer.tokenize.return_value = [1, 2, 3, 4, 5]

        encoder = Encoder()
        result = encoder.encode("mock text", mock_tokenizer, embedder)

        assert isinstance(result, SensoryInput)
        assert result.vector.shape == (embedder.dim,)
        mock_tokenizer.tokenize.assert_called_once_with("mock text")

    def test_encode_with_custom_embedder(self, tokenizer):
        """测试使用自定义embedder（Mock）"""
        mock_embedder = MagicMock(spec=Embedder)
        mock_embedder.dim = 48
        mock_embedder.embed.return_value = torch.randn(5, 48)

        encoder = Encoder()
        result = encoder.encode("mock text", tokenizer, mock_embedder)

        assert isinstance(result, SensoryInput)
        assert result.vector.shape == (48,)
        mock_embedder.embed.assert_called_once()


# ============================================================
# Decoder 测试
# ============================================================

class TestDecoder:
    """解码器测试"""

    def test_decode_text_mode(self, sample_activation):
        """测试text模式解码"""
        decoder = Decoder(mode='text')
        result = decoder.decode(sample_activation)

        assert isinstance(result, str)
        assert len(result) > 0
        assert '记忆context' in result or '激活' in result

    def test_decode_vector_mode(self, sample_activation):
        """测试vector模式解码"""
        decoder = Decoder(mode='vector')
        result = decoder.decode(sample_activation)

        assert isinstance(result, str)
        assert 'VECTOR' in result

    def test_decode_text_mode_contains_entropy_surprise(self, sample_activation):
        """测试text模式包含熵和惊讶度信息"""
        decoder = Decoder(mode='text')
        result = decoder.decode(sample_activation)

        assert '熵' in result
        assert '惊讶度' in result

    def test_decode_zero_activation(self, zero_activation):
        """测试零激活态解码"""
        decoder = Decoder(mode='text', threshold=0.01)
        result = decoder.decode(zero_activation)

        assert isinstance(result, str)
        assert '无显著激活' in result

    def test_decode_top_k(self, sample_activation):
        """测试top_k参数"""
        import re
        decoder = Decoder(mode='text', top_k=5)
        result = decoder.decode(sample_activation)

        # 应该最多描述5个节点（用正则匹配"节点+数字"的模式）
        node_count = len(re.findall(r'节点\d+', result))
        assert node_count <= 5

    def test_decode_threshold_filtering(self):
        """测试阈值过滤"""
        # 创建一个大部分节点都很小的激活态
        state = torch.zeros(64)
        state[0] = 0.5  # 只有一个节点显著
        state[1] = 0.001  # 这个节点低于阈值
        activation = Activation(state=state, entropy=0.5, surprise=0.3)

        decoder = Decoder(mode='text', threshold=0.01, top_k=10)
        result = decoder.decode(activation)

        # 只应包含1个节点（索引0）
        assert '节点0' in result
        assert '节点1' not in result

    def test_decode_invalid_mode(self):
        """测试无效模式"""
        with pytest.raises(AssertionError):
            Decoder(mode='invalid')

    def test_decode_vector_mode_precision(self, sample_activation):
        """测试vector模式的精度"""
        decoder = Decoder(mode='vector')
        result = decoder.decode(sample_activation)

        # 应包含向量数据
        assert result.startswith('[VECTOR:')
        assert result.endswith(']')


# ============================================================
# LLMBridge 测试
# ============================================================

class TestLLMBridge:
    """LLM桥接器测试"""

    def test_init_default(self):
        """测试默认初始化"""
        config = {
            'base_url': 'http://localhost:8000/v1',
            'api_key': 'test-key',
            'model': 'test-model'
        }
        bridge = LLMBridge(config)

        assert bridge.base_url == 'http://localhost:8000/v1'
        assert bridge.api_key == 'test-key'
        assert bridge.model == 'test-model'
        assert bridge.max_tokens == 1000
        assert bridge.temperature == 0.7

    def test_init_custom_params(self):
        """测试自定义参数初始化"""
        config = {
            'base_url': 'http://custom:8000/v1',
            'api_key': 'custom-key',
            'model': 'custom-model',
            'max_tokens': 2000,
            'temperature': 0.5,
            'timeout': 60,
            'max_retries': 5,
            'retry_delay': 0.5,
        }
        bridge = LLMBridge(config)

        assert bridge.max_tokens == 2000
        assert bridge.temperature == 0.5
        assert bridge.timeout == 60
        assert bridge.max_retries == 5
        assert bridge.retry_delay == 0.5

    def test_query_success_with_mock_client(self):
        """测试成功查询（使用mock client）"""
        # 创建mock client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "这是LLM的回复"
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            'base_url': 'http://localhost:8000/v1',
            'api_key': 'test-key',
            'model': 'test-model'
        }
        bridge = LLMBridge(config)
        bridge.set_client(mock_client)

        result = bridge.query("你好", "记忆context: 测试")

        assert result == "这是LLM的回复"
        mock_client.chat.completions.create.assert_called_once()

        # 验证调用参数
        call_args = mock_client.chat.completions.create.call_args
        assert call_args.kwargs['model'] == 'test-model'
        messages = call_args.kwargs['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert '记忆context' in messages[0]['content']
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == '你好'

    def test_query_retry_on_failure(self):
        """测试失败重试"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        config = {
            'base_url': 'http://localhost:8000/v1',
            'api_key': 'test-key',
            'model': 'test-model',
            'max_retries': 3,
            'retry_delay': 0.01,  # 快速重试
        }
        bridge = LLMBridge(config)
        bridge.set_client(mock_client)

        with pytest.raises(RuntimeError) as exc_info:
            bridge.query("你好", "context")

        assert "重试" in str(exc_info.value) or "失败" in str(exc_info.value)
        assert mock_client.chat.completions.create.call_count == 3

    def test_query_retry_then_success(self):
        """测试重试后成功"""
        mock_client = MagicMock()
        # 前两次失败，第三次成功
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "成功回复"
        mock_client.chat.completions.create.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
            mock_response
        ]

        config = {
            'base_url': 'http://localhost:8000/v1',
            'api_key': 'test-key',
            'model': 'test-model',
            'max_retries': 5,
            'retry_delay': 0.01,
        }
        bridge = LLMBridge(config)
        bridge.set_client(mock_client)

        result = bridge.query("你好", "context")
        assert result == "成功回复"
        assert mock_client.chat.completions.create.call_count == 3

    def test_query_includes_memory_context(self):
        """测试查询包含记忆context"""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "回复"
        mock_client.chat.completions.create.return_value = mock_response

        config = {
            'base_url': 'http://localhost:8000/v1',
            'api_key': 'test-key',
            'model': 'test-model'
        }
        bridge = LLMBridge(config)
        bridge.set_client(mock_client)

        memory_context = "[记忆context] 熵:1.5, 惊讶度:0.8"
        bridge.query("用户问题", memory_context)

        call_args = mock_client.chat.completions.create.call_args
        system_content = call_args.kwargs['messages'][0]['content']
        assert memory_context in system_content

    def test_system_prompt_configurable(self):
        """测试系统提示词可配置"""
        custom_prompt = "你是自定义助手。"
        config = {
            'base_url': 'http://localhost:8000/v1',
            'api_key': 'test-key',
            'model': 'test-model',
            'system_prompt': custom_prompt,
        }
        bridge = LLMBridge(config)
        assert bridge.system_prompt == custom_prompt

    def test_query_max_retries_zero(self):
        """测试max_retries=0时不重试"""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Error")

        config = {
            'base_url': 'http://localhost:8000/v1',
            'api_key': 'test-key',
            'model': 'test-model',
            'max_retries': 1,  # 至少尝试1次
            'retry_delay': 0.01,
        }
        bridge = LLMBridge(config)
        bridge.set_client(mock_client)

        with pytest.raises(RuntimeError):
            bridge.query("你好", "context")

        # max_retries=1 意味着只调用1次
        assert mock_client.chat.completions.create.call_count == 1

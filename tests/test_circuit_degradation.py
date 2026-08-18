# -*- coding: utf-8 -*-
"""写侧 embed 熔断降级测试（提取层 v1.4 S1-7，P3 降级路径）

覆盖（P3 降级路径表，熔断 OPEN 时三处统一 catch CircuitOpenError）：
  ① encoder.encode → 零向量 SensoryInput ＋ degraded 标记（FEP 照常安全）
  ② loop L342 semantic_vector → None ＋ degraded（本轮不写 episodic）
  ③ loop L347 raw_semantic_vector → None（退化为投影向量）
- turn_count 照常增量（B3 顶层提取一致）
- degraded_events / last_turn_degraded 观测
- 读侧（/recall 路径）不受影响：仍返回空（fail-open）
"""

import pytest
import torch

from bridge.encoder import Encoder
from core.sensory.circuit_breaker import EmbedCircuitBreaker
from core.sensory.tokenizer import SimpleTokenizer
from core.sensory.embedder import SimpleEmbedder
from runtime.loop import LivingMemoryLoop
from runtime.config import default_config


class FailingEmbedder(SimpleEmbedder):
    """embed 服务不可达（每次调用都抛连接异常）。"""

    def embed_text(self, text: str) -> torch.Tensor:
        raise ConnectionError("embed 服务不可达（测试模拟）")

    def embed_text_raw(self, text: str) -> torch.Tensor:
        raise ConnectionError("embed 服务不可达（测试模拟）")


def make_test_config(**overrides) -> dict:
    config = default_config()
    config['num_nodes'] = 32
    config['input_dim'] = 16
    config['num_infer_steps'] = 5
    config['consolidation_interval'] = 3
    config['seed'] = 42
    config['embed_circuit'] = True
    config.update(overrides)
    return config


@pytest.fixture
def open_breaker():
    """已熔断 OPEN 的测试熔断器。"""
    b = EmbedCircuitBreaker(enabled=True, max_failures=1, cooldown=3600.0)
    b.record_failure()  # 连续失败 ≥ max_failures → OPEN
    assert b.is_open
    return b


class TestEncoderDegraded:
    def test_encode_returns_zero_vector_on_open(self, open_breaker, monkeypatch):
        """P3 ①：encoder.encode 熔断 OPEN → 零向量 + degraded 标记。"""
        import bridge.encoder as enc_module
        monkeypatch.setattr(enc_module, "get_default_embed_circuit",
                            lambda: open_breaker)
        enc = Encoder()
        out = enc.encode("测试文本", SimpleTokenizer(), FailingEmbedder(dim=8))
        assert out.vector.shape == (8,)
        assert torch.all(out.vector == 0)
        assert out.metadata.get('degraded') is True


class TestProcessTurnDegraded:
    def test_process_turn_degraded_no_episodic_write(self, open_breaker,
                                                     monkeypatch):
        """P3 ②③：写侧熔断 → 本轮不写 episodic，turn 照常增量，观测计数。

        在 core.sensory.circuit_breaker 模块级替换 get_default_embed_circuit
        （encoder 与 loop 写侧共用的真实单例入口），保证三处 embed 同一
        熔断状态（S1-7 设计：写侧共用同一熔断器单例）。
        """
        import bridge.encoder as enc_module
        import core.sensory.circuit_breaker as cb_module
        # encoder 在模块级 from-import（构造时绑定）→ 需在 enc_module 替换；
        # loop 写侧在函数内 from-import（每次调用现查）→ 在 cb_module 替换。
        # 两者指向同一 open_breaker，保证三处 embed 同一熔断状态。
        monkeypatch.setattr(enc_module, "get_default_embed_circuit",
                            lambda: open_breaker)
        monkeypatch.setattr(cb_module, "get_default_embed_circuit",
                            lambda: open_breaker)
        config = make_test_config()
        config['embedder'] = FailingEmbedder(dim=16)
        loop = LivingMemoryLoop(config)

        # 读侧熔断器（loop 自建实例，_encode_query_vector 用）也需 OPEN：
        # 生产场景 embed 挂 3 次后三处全部 OPEN（写侧共用单例＋读侧自建）；
        # CLOSED 窗口内裸抛原始异常为既有行为（熔断前 3 次失败期，设计不改）
        for _ in range(3):
            with pytest.raises(ConnectionError):
                loop._embed_circuit.call(
                    lambda: (_ for _ in ()).throw(ConnectionError()))
        assert loop._embed_circuit.is_open

        before = loop.memory.episodic_size()
        before_turn = loop.turn_count
        ctx = loop.process_turn("你好", "这是一段长回复")
        assert ctx is not None                      # 流程不断（返回 context）
        assert loop.memory.episodic_size() == before  # 本轮不写 episodic
        assert loop.turn_count == before_turn + 1   # turn_count 照常增量（B3）
        assert loop.last_turn_degraded is True
        assert loop.degraded_events >= 1


class TestStatusObservability:
    def test_get_status_exposes_degraded_and_capacity(self):
        """观测字段进 get_status（灰度仪表数据源）。"""
        config = make_test_config()
        loop = LivingMemoryLoop(config)
        status = loop.get_status()
        assert 'degraded_events' in status
        assert 'last_turn_degraded' in status
        assert 'capacity_usage' in status
        assert 'capacity_soft_limit' in status
        assert status['capacity_soft_limit'] == 2000

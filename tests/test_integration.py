"""集成测试：活体记忆系统主循环 (LivingMemoryLoop)

测试 runtime/loop.py 的 LivingMemoryLoop 类，覆盖以下场景：
  1. process_turn() 完整循环
  2. save_state() / load_state() 往返一致性
  3. query_llm() 端到端
  4. 配置参数生效 (S2 验证)
  5. 长对话场景 (S3 验证)
  6. 自动快照 (G4 验证)

测试设计原则:
  - 不修改源代码，仅验证行为
  - 使用固定 seed 保证可复现
  - 小规模配置加速测试
  - 所有文件操作使用 tmp_path
  - 浮点比较使用 pytest.approx
  - 已知限制（memory 潜变量未持久化）以测试确认而非掩盖
"""

import os
import re
import pytest
import torch

from runtime.loop import LivingMemoryLoop
from runtime.config import default_config
from core.types import Activation
from core.sensory.tokenizer import SimpleTokenizer


# ============================================================
# 辅助类和函数
# ============================================================

class MockLLMBridge:
    """模拟 LLM 桥接器，返回固定响应。

    遵循 LLMBridge 的 query(user_input, memory_context) -> str 接口，
    不进行任何实际 API 调用，用于集成测试中隔离 LLM 依赖。
    """

    def __init__(self):
        self.query_count = 0
        self.last_user_input = None
        self.last_memory_context = None

    def query(self, user_input: str, memory_context: str) -> str:
        """返回包含用户输入和记忆上下文摘要的固定响应。"""
        self.query_count += 1
        self.last_user_input = user_input
        self.last_memory_context = memory_context
        return f"收到: {user_input[:20]}... [记忆: {memory_context[:30]}...]"


def parse_memory_strength(context: str):
    """从 context 文本中解析长时记忆检索强度。

    解码器输出的格式有三种：
      - "长时记忆检索: 无相关记忆"           → 返回 0.0
      - "长时记忆检索: 强度X.XXX (无显著节点)" → 返回 X.XXX
      - "长时记忆检索: 强度X.XXX, 活跃维度: ..." → 返回 X.XXX

    返回:
        float: 记忆强度值
        None:  如果 context 中不包含"长时记忆检索"段落
    """
    if '长时记忆检索' not in context:
        return None
    if '无相关记忆' in context:
        return 0.0
    match = re.search(r'强度([\d.]+)', context)
    if match:
        return float(match.group(1))
    return None


def parse_entropy(context: str):
    """从 context 文本中解析熵值。"""
    match = re.search(r'熵:([\d.]+)', context)
    if match:
        return float(match.group(1))
    return None


def parse_surprise(context: str):
    """从 context 文本中解析惊讶度值。"""
    match = re.search(r'惊讶度:([\d.]+)', context)
    if match:
        return float(match.group(1))
    return None


def make_test_config(**overrides) -> dict:
    """创建小规模测试配置，可覆盖默认值。

    返回的配置使用小的 num_nodes/input_dim 加速测试，
    并设置固定 seed 保证可复现。
    """
    config = default_config()
    config['num_nodes'] = 32
    config['input_dim'] = 16
    config['num_infer_steps'] = 5
    config['consolidation_interval'] = 3
    config['seed'] = 42
    config.update(overrides)
    return config


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def loop():
    """标准测试 loop，小规模配置。

    使用 num_nodes=32, input_dim=16 加速测试。
    固定 seed=42 保证可复现。
    consolidation_interval=3 使巩固较快发生（第4轮触发）。
    """
    torch.manual_seed(42)
    config = make_test_config()
    return LivingMemoryLoop(config)


@pytest.fixture
def loop_with_mock_bridge():
    """带 mock LLM 的 loop。

    注入 MockLLMBridge，使 query_llm 可用且不依赖外部 API。
    """
    torch.manual_seed(42)
    config = make_test_config(llm_bridge=MockLLMBridge())
    return LivingMemoryLoop(config)


@pytest.fixture
def loop_no_bridge():
    """不带 LLM bridge 的 loop。

    显式移除 llm_api 配置，确保 self.bridge 为 None。
    用于测试 query_llm 在无桥接器时的 RuntimeError 行为。
    """
    torch.manual_seed(42)
    config = make_test_config()
    config.pop('llm_api', None)
    return LivingMemoryLoop(config)


@pytest.fixture
def loop_deterministic():
    """确定性测试 loop（temperature=0），用于需要精确比较的场景。

    temperature=0 消除 Langevin 扩散项的随机噪声，
    使 infer() 在相同输入下产生完全一致的输出。
    """
    torch.manual_seed(42)
    config = make_test_config(temperature=0.0)
    return LivingMemoryLoop(config)


# ============================================================
# 1. process_turn() 完整循环测试
# ============================================================

class TestProcessTurn:
    """测试 process_turn() 的完整记忆循环。"""

    def test_single_turn_returns_string(self, loop):
        """单轮调用返回字符串 context。"""
        context = loop.process_turn("你好世界")
        assert isinstance(context, str)
        assert len(context) > 0

    def test_multi_turn_no_exception(self, loop):
        """连续10轮调用不抛异常。"""
        inputs = [
            "你好世界",
            "今天天气怎么样",
            "我喜欢编程",
            "机器学习很有趣",
            "深度学习是AI的子集",
            "自然语言处理",
            "计算机视觉",
            "强化学习",
            "生成对抗网络",
            "Transformer架构",
        ]
        for i, text in enumerate(inputs):
            context = loop.process_turn(text)
            assert isinstance(context, str), f"第{i+1}轮返回非字符串"

    def test_turn_count_increments(self, loop):
        """turn_count 正确递增。"""
        assert loop.turn_count == 0
        loop.process_turn("第一轮")
        assert loop.turn_count == 1
        loop.process_turn("第二轮")
        assert loop.turn_count == 2
        loop.process_turn("第三轮")
        assert loop.turn_count == 3

    def test_last_activation_updated(self, loop):
        """last_activation 在每轮后被更新。"""
        assert loop.last_activation is None

        loop.process_turn("第一轮")
        assert loop.last_activation is not None
        assert isinstance(loop.last_activation, Activation)
        first_activation = loop.last_activation

        loop.process_turn("第二轮")
        assert loop.last_activation is not None
        # 第二轮后 last_activation 应该是新的对象（已更新）
        assert loop.last_activation is not first_activation

    def test_entropy_and_surprise_in_context(self, loop):
        """context 文本中包含熵和惊讶度信息。"""
        context = loop.process_turn("测试文本")
        assert '熵' in context
        assert '惊讶度' in context

        # 解析出的数值应该是有限浮点数
        entropy = parse_entropy(context)
        surprise = parse_surprise(context)
        assert entropy is not None
        assert surprise is not None
        assert isinstance(entropy, float)
        assert isinstance(surprise, float)

    def test_memory_recall_in_context(self, loop):
        """S1 验证：多轮后 context 中包含"长时记忆检索"信息且强度非零。

        关键验证点：
          - 首轮 context 中记忆强度接近 0
          - 多轮后（至少5轮）context 中出现非零记忆强度
        """
        # 首轮：long_term_latent 初始为零，仅被 EMA 微量更新
        first_context = loop.process_turn("首轮输入")
        first_strength = parse_memory_strength(first_context)
        assert first_strength is not None, "首轮应包含长时记忆检索段落"
        # 首轮记忆强度应接近 0（< 1e-3 或为 0）
        assert first_strength < 1e-2, (
            f"首轮记忆强度应接近0，实际为 {first_strength}"
        )

        # 多轮后：经过巩固（consolidation_interval=3，第4轮触发），
        # long_term_latent 通过迁移获得显著内容
        for i in range(7):
            loop.process_turn(f"对话第{i+2}轮，内容各不相同")

        final_context = loop.process_turn("第八轮测试")
        final_strength = parse_memory_strength(final_context)
        assert final_strength is not None, "多轮后应包含长时记忆检索段落"
        # 多轮后记忆强度应明显大于首轮
        assert final_strength > first_strength, (
            f"多轮后记忆强度({final_strength})应大于首轮({first_strength})"
        )
        # 经过巩固后记忆强度应非零且可观
        assert final_strength > 0.0, "多轮后记忆强度应为正值"

    def test_first_turn_memory_near_zero(self, loop):
        """首轮记忆强度接近零的独立验证。

        首轮时 long_term_latent 初始为零，memory.update 仅注入
        (1 - long_term_decay) = 0.001 比例的激活态，recall 结果极小。
        """
        context = loop.process_turn("首轮记忆测试")
        strength = parse_memory_strength(context)
        assert strength is not None
        # 首轮记忆强度应非常小（EMA 权重 0.001 * 激活态 * sigmoid 门控）
        assert strength < 1e-2, f"首轮记忆强度应 < 0.01，实际为 {strength}"

    def test_get_status_after_turns(self, loop):
        """get_status() 在多轮后返回正确状态。"""
        for i in range(5):
            loop.process_turn(f"状态测试第{i+1}轮")

        status = loop.get_status()
        assert status['turn_count'] == 5
        assert status['num_nodes'] == 32
        assert status['input_dim'] == 16
        assert 'last_entropy' in status
        assert 'last_surprise' in status
        assert 'purpose_coherence' in status
        assert 'precision_mean' in status
        assert 'precision_std' in status
        assert isinstance(status['last_entropy'], float)
        assert isinstance(status['last_surprise'], float)


# ============================================================
# 2. save_state() / load_state() 往返一致性
# ============================================================

class TestSaveLoadState:
    """测试 save_state() 和 load_state() 的往返一致性。

    注意已知限制：当前 save_state/load_state 只保存 attractor 和 purpose，
    不保存 memory 潜变量（short_term_latent / long_term_latent）。
    测试应确认此行为而非掩盖。
    """

    def test_save_creates_file(self, loop, tmp_path):
        """save_state 创建 .pt 文件。"""
        # 运行几轮以产生非初始状态
        for i in range(3):
            loop.process_turn(f"保存测试第{i+1}轮")

        path = str(tmp_path / "test_snapshot.pt")
        loop.save_state(path)

        assert os.path.exists(path)
        assert path.endswith('.pt')

    def test_load_restores_attractor(self, loop, tmp_path):
        """load_state 恢复 J 矩阵和 sigma。"""
        # 运行多轮使 J 和 sigma 偏离初始值
        for i in range(5):
            loop.process_turn(f"吸引子测试第{i+1}轮")

        # 保存原始状态
        original_J = loop.attractor.J.clone()
        original_sigma = loop.attractor.sigma.clone()
        original_bias = loop.attractor.bias.clone()

        path = str(tmp_path / "attractor_snapshot.pt")
        loop.save_state(path)

        # 创建新 loop 并加载
        torch.manual_seed(999)  # 不同 seed，初始 J 不同
        new_loop = LivingMemoryLoop(make_test_config(seed=999))
        # 确认新 loop 的 J 确实不同
        assert not torch.allclose(new_loop.attractor.J, original_J)

        new_loop.load_state(path)

        # 验证 attractor 状态恢复一致
        assert torch.allclose(new_loop.attractor.J, original_J)
        assert torch.allclose(new_loop.attractor.sigma, original_sigma)
        assert torch.allclose(new_loop.attractor.bias, original_bias)

    def test_load_restores_purpose(self, loop, tmp_path):
        """load_state 恢复 precision 和 coherence。"""
        # 运行多轮使 precision 和 coherence 偏离初始值
        for i in range(5):
            loop.process_turn(f"目的层测试第{i+1}轮")

        # 保存原始状态
        original_precision = loop.purpose.get_precision()
        original_coherence = loop.purpose.coherence
        original_history_len = len(loop.purpose.history)

        path = str(tmp_path / "purpose_snapshot.pt")
        loop.save_state(path)

        # 创建新 loop 并加载
        new_loop = LivingMemoryLoop(make_test_config())
        new_loop.load_state(path)

        # 验证 purpose 状态恢复一致
        assert torch.allclose(new_loop.purpose.get_precision(), original_precision)
        assert new_loop.purpose.coherence == pytest.approx(original_coherence)
        assert len(new_loop.purpose.history) == original_history_len
        # 逐条比较 history
        for h1, h2 in zip(new_loop.purpose.history, loop.purpose.history):
            assert torch.allclose(h1, h2)

    def test_roundtrip_behavior_consistency(self, tmp_path):
        """保存→恢复后，相同输入产生相似的激活态输出。

        使用 temperature=0 消除推断随机性，确保可比性。
        共享 tokenizer 以消除词表差异（tokenizer 词表未持久化是已知限制）。
        验证：
          - attractor 和 purpose 恢复后，推断产生的熵/惊讶度一致
          - memory 未恢复导致记忆检索部分不同（已知限制）
        """
        torch.manual_seed(42)
        # 共享 tokenizer：tokenizer 词表是动态构建的，未持久化。
        # 不共享时，两个 loop 对相同文本产生不同 token IDs，
        # 导致不同感官输入。共享后可隔离测试 attractor/purpose 的往返一致性。
        shared_tokenizer = SimpleTokenizer()

        config_a = make_test_config(temperature=0.0, tokenizer=shared_tokenizer)
        loop_a = LivingMemoryLoop(config_a)

        # 运行5轮积累状态
        for i in range(5):
            loop_a.process_turn(f"往返测试第{i+1}轮")

        # 保存状态
        path = str(tmp_path / "roundtrip.pt")
        loop_a.save_state(path)

        # 创建新 loop 并加载（共享同一个 tokenizer）
        config_b = make_test_config(temperature=0.0, tokenizer=shared_tokenizer)
        loop_b = LivingMemoryLoop(config_b)
        loop_b.load_state(path)

        # 验证 attractor 和 purpose 一致
        assert torch.allclose(loop_b.attractor.J, loop_a.attractor.J)
        assert torch.allclose(loop_b.attractor.sigma, loop_a.attractor.sigma)
        assert torch.allclose(
            loop_b.purpose.get_precision(), loop_a.purpose.get_precision()
        )

        # 相同输入产生相同激活态（temperature=0 保证确定性）
        torch.manual_seed(123)
        context_a = loop_a.process_turn("往返验证输入")
        entropy_a = parse_entropy(context_a)
        surprise_a = parse_surprise(context_a)

        torch.manual_seed(123)
        context_b = loop_b.process_turn("往返验证输入")
        entropy_b = parse_entropy(context_b)
        surprise_b = parse_surprise(context_b)

        # 激活态部分（熵和惊讶度）应一致
        assert entropy_a == pytest.approx(entropy_b, abs=1e-6), (
            f"熵不一致: A={entropy_a}, B={entropy_b}"
        )
        assert surprise_a == pytest.approx(surprise_b, abs=1e-6), (
            f"惊讶度不一致: A={surprise_a}, B={surprise_b}"
        )

        # 记忆检索部分应不同（loop_a 有积累，loop_b 记忆归零）
        strength_a = parse_memory_strength(context_a)
        strength_b = parse_memory_strength(context_b)
        assert strength_a is not None and strength_b is not None
        # loop_a 经过5轮积累，记忆强度应大于 loop_b（记忆未恢复）
        assert strength_a >= strength_b, (
            f"已保存的系统记忆强度({strength_a})应 >= 恢复后的({strength_b})"
        )

    def test_memory_latents_not_persisted(self, loop, tmp_path):
        """已知限制验证：memory 潜变量未被 save_state/load_state 持久化。

        save_state 只保存 attractor（J, bias, sigma）和 purpose（precision,
        history, coherence），不保存 memory 的 short_term_latent 和
        long_term_latent。恢复后这些潜变量归零。

        此测试确认该已知行为，而非视为 bug。Wave 4 应评估是否需要
        将 memory 状态纳入持久化范围。
        """
        # 运行多轮积累记忆
        for i in range(6):
            loop.process_turn(f"记忆持久化测试第{i+1}轮")

        # 确认 loop 有非零记忆
        assert loop.memory.long_term_latent.abs().sum() > 0, (
            "运行多轮后 long_term_latent 应非零"
        )
        assert loop.memory.short_term_latent.abs().sum() > 0, (
            "运行多轮后 short_term_latent 应非零"
        )

        # 保存
        path = str(tmp_path / "memory_test.pt")
        loop.save_state(path)

        # 创建新 loop 并加载
        new_loop = LivingMemoryLoop(make_test_config())
        new_loop.load_state(path)

        # 确认 memory 潜变量归零（未持久化）
        assert new_loop.memory.long_term_latent.abs().sum() == 0, (
            "load_state 后 long_term_latent 应归零（未持久化）"
        )
        assert new_loop.memory.short_term_latent.abs().sum() == 0, (
            "load_state 后 short_term_latent 应归零（未持久化）"
        )

        # 同时确认 encounter_count 也未持久化
        assert new_loop.purpose.encounter_count.abs().sum() == 0, (
            "load_state 后 encounter_count 应归零（未持久化）"
        )


# ============================================================
# 3. query_llm() 端到端测试
# ============================================================

class TestQueryLLM:
    """测试 query_llm() 的端到端行为。"""

    def test_query_without_bridge_raises(self, loop_no_bridge):
        """未配置 LLM Bridge 时抛出 RuntimeError。"""
        assert loop_no_bridge.bridge is None

        with pytest.raises(RuntimeError, match="未配置LLM Bridge"):
            loop_no_bridge.query_llm("测试输入")

    def test_query_with_mock_bridge(self, loop_with_mock_bridge):
        """mock LLM 返回响应，记忆系统被更新。"""
        loop = loop_with_mock_bridge
        assert loop.bridge is not None

        response = loop.query_llm("你好世界")

        # 验证返回了 mock 响应
        assert isinstance(response, str)
        assert "收到" in response
        assert "你好世界" in response

        # 验证 mock bridge 被调用
        assert loop.bridge.query_count == 1
        assert loop.bridge.last_user_input == "你好世界"
        # memory_context 被传入（首轮时 last_activation 为 None，使用"[无记忆]"）
        assert loop.bridge.last_memory_context is not None

    def test_query_updates_memory(self, loop_with_mock_bridge):
        """query_llm 后 turn_count 增加（内部调用 process_turn）。"""
        loop = loop_with_mock_bridge
        assert loop.turn_count == 0

        loop.query_llm("第一轮对话")

        # query_llm 内部调用 process_turn，turn_count 应增加
        assert loop.turn_count == 1
        assert loop.last_activation is not None

        loop.query_llm("第二轮对话")
        assert loop.turn_count == 2

    def test_query_passes_memory_context(self, loop_with_mock_bridge):
        """query_llm 将记忆 context 传递给 bridge。

        首轮时 last_activation 为 None，context 为"[无记忆]"。
        非首轮时，context 来自 decoder.decode()。
        """
        loop = loop_with_mock_bridge

        # 首轮：last_activation 为 None
        loop.query_llm("首轮输入")
        assert loop.bridge.last_memory_context == "[无记忆]"

        # 第二轮：last_activation 已设置，context 来自 decoder
        loop.query_llm("第二轮输入")
        assert loop.bridge.last_memory_context != "[无记忆]"
        assert "记忆context" in loop.bridge.last_memory_context


# ============================================================
# 4. 配置参数生效测试 (S2 验证)
# ============================================================

class TestConfigWiring:
    """测试配置参数正确传递到各组件 (S2 验证)。

    S2 修复的核心：自建组件时 config 中的 FEP 参数应正确接线到
    AttractorNetwork 和 PurposeLayer。
    """

    def test_custom_temperature_affects_attractor(self):
        """自定义 temperature 传递到 AttractorNetwork。"""
        # 高温配置
        config_hot = make_test_config(temperature=0.5)
        loop_hot = LivingMemoryLoop(config_hot)
        assert loop_hot.attractor.temperature == pytest.approx(0.5)

        # 低温配置
        config_cold = make_test_config(temperature=0.01)
        loop_cold = LivingMemoryLoop(config_cold)
        assert loop_cold.attractor.temperature == pytest.approx(0.01)

        # 零温配置（确定性推断）
        config_zero = make_test_config(temperature=0.0)
        loop_zero = LivingMemoryLoop(config_zero)
        assert loop_zero.attractor.temperature == pytest.approx(0.0)

    def test_custom_orth_weight_affects_behavior(self):
        """自定义 orth_weight 影响学习行为。

        orth_weight=0 时无正交化压力，orth_weight=1 时有强正交化。
        两者在相同输入下应产生不同的 J 矩阵。
        """
        # 创建两个配置，仅 orth_weight 不同
        config_low = make_test_config(orth_weight=0.0, temperature=0.0)
        config_high = make_test_config(orth_weight=1.0, temperature=0.0)

        loop_low = LivingMemoryLoop(config_low)
        loop_high = LivingMemoryLoop(config_high)

        # 验证参数已传递
        assert loop_low.attractor.orth_weight == pytest.approx(0.0)
        assert loop_high.attractor.orth_weight == pytest.approx(1.0)

        # 初始 J 应相同（相同 seed）
        assert torch.allclose(loop_low.attractor.J, loop_high.attractor.J)

        # 运行相同输入
        inputs = ["测试正交化", "第二轮输入", "第三轮数据"]
        for text in inputs:
            torch.manual_seed(42)
            loop_low.process_turn(text)
            torch.manual_seed(42)
            loop_high.process_turn(text)

        # 多轮学习后 J 矩阵应不同（正交化压力不同）
        assert not torch.allclose(loop_low.attractor.J, loop_high.attractor.J), (
            "不同 orth_weight 应产生不同的 J 矩阵"
        )

    def test_custom_max_history_limits_growth(self):
        """S3 验证：自定义 max_history 限制 history 增长。"""
        custom_max = 10
        config = make_test_config(max_history=custom_max)
        loop = LivingMemoryLoop(config)

        assert loop.purpose.max_history == custom_max

        # 运行超过 max_history 轮
        for i in range(custom_max + 5):
            loop.process_turn(f"历史限制测试第{i+1}轮")

        # history 长度不应超过 max_history
        assert len(loop.purpose.history) <= custom_max, (
            f"history 长度 {len(loop.purpose.history)} 超过上限 {custom_max}"
        )

    def test_custom_habituation_rate_affects_purpose(self):
        """自定义 habituation_rate 传递到 PurposeLayer 并影响行为。

        habituation_rate 控制习惯化衰减速度：
          habituation = 1 / (1 + encounter_count * habituation_rate)

        需要 temperature > 0 使激活值超过 0.3 阈值，触发 encounter_count
        累积，从而使 habituation_rate 对 precision 产生影响。
        temperature=0 时激活值过小，encounter_count 始终为 0，
        habituation 机制无法生效。
        """
        config = make_test_config(habituation_rate=0.5)
        loop = LivingMemoryLoop(config)

        assert loop.purpose.habituation_rate == pytest.approx(0.5)

        # 验证不同 habituation_rate 产生不同行为
        # 使用较高 temperature（0.5）使激活值超过 0.3 阈值，
        # 触发 encounter_count 累积和习惯化机制
        config_low = make_test_config(habituation_rate=0.01, temperature=0.5)
        config_high = make_test_config(habituation_rate=1.0, temperature=0.5)

        loop_low = LivingMemoryLoop(config_low)
        loop_high = LivingMemoryLoop(config_high)

        # 运行相同输入（设置相同 seed 保证 Langevin 噪声一致）
        for i in range(5):
            text = f"习惯化测试第{i+1}轮"
            torch.manual_seed(42)
            loop_low.process_turn(text)
            torch.manual_seed(42)
            loop_high.process_turn(text)

        # 确认 encounter_count 已累积（习惯化机制已激活）
        assert loop_low.purpose.encounter_count.sum() > 0, (
            "低 habituation_rate 的 encounter_count 应已累积"
        )
        assert loop_high.purpose.encounter_count.sum() > 0, (
            "高 habituation_rate 的 encounter_count 应已累积"
        )

        # 高 habituation_rate 应导致 encounter_count 对 precision 影响更大
        # 即高 habituation_rate 下 precision 应更趋向均匀（衰减更快）
        precision_low = loop_low.purpose.get_precision()
        precision_high = loop_high.purpose.get_precision()

        # 两组 precision 不应完全相同
        assert not torch.allclose(precision_low, precision_high), (
            "不同 habituation_rate 应产生不同的 precision"
        )


# ============================================================
# 5. 长对话场景测试 (S3 验证)
# ============================================================

class TestLongConversation:
    """测试长对话场景下的系统稳定性和资源约束 (S3 验证)。

    S3 修复：purpose.history 裁剪到 max_history 上限，防止无界增长。
    """

    def test_history_bounded_after_many_turns(self, loop):
        """S3 验证：500轮后 history 长度不超过 max_history（默认100）。"""
        max_history = loop.purpose.max_history
        assert max_history == 100  # 默认值

        for i in range(500):
            loop.process_turn(f"长对话第{i+1}轮，这是一段测试文本")

        assert len(loop.purpose.history) <= max_history, (
            f"500轮后 history 长度 {len(loop.purpose.history)} "
            f"超过上限 {max_history}"
        )
        # 经过500轮，history 应恰好等于 max_history（持续追加+裁剪）
        assert len(loop.purpose.history) == max_history

    def test_no_memory_leak_in_long_run(self, loop):
        """长对话不导致内存持续增长。

        检查所有可能增长的数据结构在长对话后仍受约束：
          - purpose.history: 受 max_history 约束
          - memory._buffer: 受 buffer_capacity (deque maxlen) 约束
          - 所有张量形状保持不变
        """
        num_nodes = loop.attractor.num_nodes
        buffer_cap = loop.memory._buffer_capacity

        for i in range(200):
            loop.process_turn(f"内存泄漏测试第{i+1}轮")

        # purpose.history 受约束
        assert len(loop.purpose.history) <= loop.purpose.max_history

        # memory._buffer 受 deque maxlen 约束
        assert len(loop.memory._buffer) <= buffer_cap

        # 张量形状不变（无维度膨胀）
        assert loop.attractor.J.shape == (num_nodes, num_nodes)
        assert loop.attractor.sigma.shape == (num_nodes,)
        assert loop.memory.short_term_latent.shape == (num_nodes,)
        assert loop.memory.long_term_latent.shape == (num_nodes,)
        assert loop.purpose.sensory_precision.shape == (loop.attractor.input_dim,)
        assert loop.purpose.encounter_count.shape == (loop.attractor.input_dim,)

    def test_system_stable_after_100_turns(self, loop):
        """100轮后系统仍正常工作，无 NaN/Inf。

        验证所有关键状态变量在长对话后保持数值稳定性。
        """
        for i in range(100):
            loop.process_turn(f"稳定性测试第{i+1}轮")

        # 检查 attractor 状态无 NaN/Inf
        assert torch.isfinite(loop.attractor.J).all(), "J 矩阵含 NaN/Inf"
        assert torch.isfinite(loop.attractor.sigma).all(), "sigma 含 NaN/Inf"
        assert torch.isfinite(loop.attractor.bias).all(), "bias 含 NaN/Inf"

        # 检查 purpose 状态无 NaN/Inf
        assert torch.isfinite(loop.purpose.sensory_precision).all(), \
            "precision 含 NaN/Inf"
        assert torch.isfinite(loop.purpose.encounter_count).all(), \
            "encounter_count 含 NaN/Inf"
        assert isinstance(loop.purpose.coherence, float)
        assert not (loop.purpose.coherence != loop.purpose.coherence), \
            "coherence 是 NaN"

        # 检查 memory 状态无 NaN/Inf
        assert torch.isfinite(loop.memory.short_term_latent).all(), \
            "short_term_latent 含 NaN/Inf"
        assert torch.isfinite(loop.memory.long_term_latent).all(), \
            "long_term_latent 含 NaN/Inf"

        # 检查 last_activation 无 NaN/Inf
        assert loop.last_activation is not None
        assert torch.isfinite(loop.last_activation.state).all(), \
            "last_activation.state 含 NaN/Inf"
        assert isinstance(loop.last_activation.entropy, float)
        assert isinstance(loop.last_activation.surprise, float)

        # 系统仍能正常处理新输入
        context = loop.process_turn("第101轮验证")
        assert isinstance(context, str)
        assert len(context) > 0


# ============================================================
# 6. 自动快照测试 (G4 验证)
# ============================================================

class TestAutoSnapshot:
    """测试自动快照功能 (G4 验证)。

    G4 修复：process_turn 中按 auto_snapshot_interval 自动保存状态。
    """

    def test_auto_snapshot_creates_file(self, tmp_path):
        """G4 验证：auto_snapshot 开启时自动创建快照文件。"""
        config = make_test_config(
            auto_snapshot=True,
            auto_snapshot_interval=5,
            snapshot_dir=str(tmp_path),
        )
        loop = LivingMemoryLoop(config)

        # 运行5轮（第5轮 turn_count=5，5%5==0 → 触发快照）
        for i in range(5):
            loop.process_turn(f"快照测试第{i+1}轮")

        snapshot_path = os.path.join(str(tmp_path), "snapshot_5.pt")
        assert os.path.exists(snapshot_path), (
            f"第5轮应创建快照文件: {snapshot_path}"
        )

    def test_auto_snapshot_interval(self, tmp_path):
        """快照按间隔创建，不是每轮都创建。

        设置 interval=5，运行7轮：
          - 第5轮：turn_count=5, 5%5==0 → 创建 snapshot_5.pt
          - 第1-4轮, 6-7轮：不创建快照
        """
        config = make_test_config(
            auto_snapshot=True,
            auto_snapshot_interval=5,
            snapshot_dir=str(tmp_path),
        )
        loop = LivingMemoryLoop(config)

        for i in range(7):
            loop.process_turn(f"间隔测试第{i+1}轮")

        # 第5轮应创建快照
        assert os.path.exists(os.path.join(str(tmp_path), "snapshot_5.pt"))

        # 非间隔轮次不应创建快照
        for n in [1, 2, 3, 4, 6, 7]:
            path = os.path.join(str(tmp_path), f"snapshot_{n}.pt")
            assert not os.path.exists(path), (
                f"第{n}轮不应创建快照，但文件存在: {path}"
            )

        # 继续运行到第10轮，应创建第二个快照
        for i in range(7, 10):
            loop.process_turn(f"间隔测试第{i+1}轮")

        assert os.path.exists(os.path.join(str(tmp_path), "snapshot_10.pt"))

    def test_auto_snapshot_disabled_by_default(self, loop):
        """默认不开启 auto_snapshot。"""
        # 检查默认配置
        default_cfg = default_config()
        assert default_cfg['auto_snapshot'] is False

        # 检查 loop 实例的配置
        assert loop.config.get('auto_snapshot', False) is False

        # 运行多轮，确认不创建任何快照文件
        # （默认 snapshot_dir 是 ~/.lms/snapshots，不应在此测试中产生文件）
        for i in range(55):  # 超过默认 interval=50
            loop.process_turn(f"默认配置测试第{i+1}轮")

        # turn_count 应正常递增
        assert loop.turn_count == 55
        # 系统应正常运行
        assert loop.last_activation is not None

    def test_auto_snapshot_failure_does_not_crash(self, tmp_path):
        """快照保存失败时不影响主循环（异常被捕获）。"""
        # 使用一个不存在的根目录路径，使保存失败
        invalid_dir = os.path.join(str(tmp_path), "nonexistent_root",
                                   "deeply", "nested", "path")
        # 注意：Snapshot.save 会调用 os.makedirs(exist_ok=True)，
        # 所以需要用一个真正无法写入的方式触发失败。
        # 这里用文件路径（而非目录）作为 snapshot_dir 来触发失败。
        invalid_file = str(tmp_path / "blocking_file")
        with open(invalid_file, 'w') as f:
            f.write("block")

        config = make_test_config(
            auto_snapshot=True,
            auto_snapshot_interval=2,
            snapshot_dir=invalid_file,  # 这是一个文件，不是目录
        )
        loop = LivingMemoryLoop(config)

        # 第2轮应触发快照，但保存会失败（路径是文件不是目录）
        # 主循环不应崩溃
        for i in range(3):
            context = loop.process_turn(f"失败测试第{i+1}轮")
            assert isinstance(context, str)

        # turn_count 应正常递增
        assert loop.turn_count == 3

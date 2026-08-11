# -*- coding: utf-8 -*-
"""单元测试：做梦引擎 (DreamEngine)

测试 core/hippocampus/dream_engine.py 的 DreamEngine 类，覆盖：
  1. 初始化与配置读取
  2. MVP 做梦 (dream_mvp)：空缓冲区、有记忆、J 变化、快照保存
  3. 惊讶度加权采样 (_sample_by_surprise)
  4. 空闲态推断/学习 (_idle_infer / _idle_learn)
  5. 七阶段各阶段：SHY、遗忘修剪、景观漂移、目的演化、坍缩检测、扰动注入
  6. 完整做梦周期 (dream_cycle)
  7. 状态查询 (get_status)
  8. 稳定性（连续做梦周期不崩溃）
  9. 集成测试：LivingMemoryLoop.dream() 端到端

测试设计原则：
  - 小规模实例（num_nodes=32, input_dim=8）加速测试
  - 固定 seed 保证可复现
  - 所有文件操作使用 tmp_path
  - 浮点比较使用 pytest.approx
"""

import os
import pytest
import torch

from core.hippocampus.attractor import AttractorNetwork
from core.hippocampus.purpose import PurposeLayer
from core.hippocampus.memory import MemoryManager
from core.hippocampus.dream_engine import DreamEngine
from core.sensory.embedder import SimpleEmbedder
from core.types import Activation
from runtime.loop import LivingMemoryLoop
from runtime.config import default_config


# ============================================================
# 辅助函数与 Mock
# ============================================================

def make_test_config(**overrides) -> dict:
    """创建小规模测试配置（用于集成测试的 LivingMemoryLoop）。"""
    config = default_config()
    config['num_nodes'] = 32
    config['input_dim'] = 16
    config['num_infer_steps'] = 5
    config['consolidation_interval'] = 3
    config['seed'] = 42
    config.update(overrides)
    return config


class MockEmbedder(SimpleEmbedder):
    """带 embed_text 的嵌入器，用于情景记忆存储与检索测试。

    SimpleEmbedder 仅有 embed(tokens)，无法触发 process_turn 中的
    情景记忆存储路径。本类补充 embed_text，基于文本哈希生成确定性
    向量，使相同文本产生相同向量、不同文本产生不同向量。
    """

    def embed_text(self, text: str) -> torch.Tensor:
        if not text or not text.strip():
            return torch.zeros(self._dim)
        h = abs(hash(text)) % (2 ** 32)
        g = torch.Generator()
        g.manual_seed(h)
        return torch.randn(self._dim, generator=g) * 0.1


def populate_buffer(engine: DreamEngine, n: int = 5) -> None:
    """向 memory._buffer 填充 (state, surprise) 条目。

    使用 memory.update 走真实路径，surprise 各不相同以便测试加权采样。
    """
    torch.manual_seed(42)
    for i in range(n):
        state = torch.randn(engine.num_nodes) * 0.5
        surprise = float(i + 1) * 0.5  # 0.5, 1.0, 1.5, ...
        act = Activation(state=state, entropy=1.0, surprise=surprise)
        engine.memory.update(act, surprise)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def small_config():
    """小规模做梦引擎配置（32 节点，8 维输入）。"""
    return {
        'num_nodes': 32,
        'input_dim': 8,
        'idle_learning_rate': 0.001,
        'idle_orth_weight': 1.0,
        'idle_temperature': 0.1,
        'idle_num_steps': 5,
        'consolidation_ratio': 0.7,
        'collapse_threshold': 0.9,
        'max_idle_steps': 50,
        'snapshot_dir': 'test_snapshots',
    }


@pytest.fixture
def dream_engine(small_config, tmp_path):
    """小规模 DreamEngine 实例。

    使用 num_nodes=32, input_dim=8 加速测试。
    snapshot_dir 指向 tmp_path 以避免污染工作目录。
    """
    config = dict(small_config)
    config['snapshot_dir'] = str(tmp_path)
    torch.manual_seed(42)
    attractor = AttractorNetwork(32, 8, seed=42)
    purpose = PurposeLayer(8)
    memory = MemoryManager(32)
    embedder = SimpleEmbedder(8)
    return DreamEngine(attractor, purpose, memory, embedder, config)


@pytest.fixture
def dream_engine_with_memories(dream_engine):
    """预填充了记忆的 DreamEngine。"""
    populate_buffer(dream_engine, n=5)
    return dream_engine


# ============================================================
# 1. 初始化与 MVP 做梦
# ============================================================

class TestDreamEngine:
    """DreamEngine 单元测试。"""

    def test_init(self, dream_engine, small_config):
        """DreamEngine 初始化正确，参数读取正确。"""
        assert dream_engine.num_nodes == small_config['num_nodes']
        assert dream_engine.input_dim == small_config['input_dim']
        assert dream_engine.idle_lr == pytest.approx(
            small_config['idle_learning_rate'])
        assert dream_engine.idle_orth_weight == pytest.approx(
            small_config['idle_orth_weight'])
        assert dream_engine.idle_temperature == pytest.approx(
            small_config['idle_temperature'])
        assert dream_engine.idle_num_steps == small_config['idle_num_steps']
        assert dream_engine.consolidation_ratio == pytest.approx(
            small_config['consolidation_ratio'])
        assert dream_engine.collapse_threshold == pytest.approx(
            small_config['collapse_threshold'])
        assert dream_engine.max_idle_steps == small_config['max_idle_steps']

        # 零精度向量
        assert dream_engine.idle_precision.shape == (small_config['input_dim'],)
        assert torch.all(dream_engine.idle_precision == 0)

        # 阶段权重齐全
        assert set(dream_engine.phase_weights.keys()) == {
            'nrem_consolidation', 'synaptic_homeostasis', 'forgetting_pruning',
            'landscape_drift', 'purpose_evolution', 'rem_integration',
            'snapshot', 'doubt_review',  # 体验层 D：怀疑复核阶段（nrem 让渡 0.10）
        }

        # 组件引用正确
        assert dream_engine.attractor is not None
        assert dream_engine.purpose is not None
        assert dream_engine.memory is not None
        assert dream_engine.embedder is not None

    def test_dream_mvp_empty_buffer(self, dream_engine):
        """空缓冲区时 dream_mvp 优雅处理。"""
        assert len(dream_engine.memory._buffer) == 0
        result = dream_engine.dream_mvp(n_steps=10)
        assert isinstance(result, dict)
        assert result['status'] == 'no_memories_to_replay'
        # 不应抛异常，且 step_count 不增加
        assert dream_engine.step_count == 0

    def test_dream_mvp_with_memories(self, dream_engine_with_memories):
        """有记忆时 dream_mvp 正常运行，返回包含统计信息的 dict。"""
        engine = dream_engine_with_memories
        result = engine.dream_mvp(n_steps=10)
        assert isinstance(result, dict)
        assert result['status'] == 'dreamed'
        assert result['steps'] == 10
        assert 'avg_surprise' in result
        assert 'max_surprise' in result
        assert 'j_change' in result
        assert 'buffer_size' in result
        assert isinstance(result['avg_surprise'], float)
        assert result['buffer_size'] > 0

    def test_dream_mvp_modifies_j(self, dream_engine_with_memories):
        """dream_mvp 后 J 矩阵发生变化。"""
        engine = dream_engine_with_memories
        J_before = engine.attractor.J.clone()
        engine.dream_mvp(n_steps=10)
        J_after = engine.attractor.J.clone()
        # J 应发生变化（低学习率也会累积修改）
        assert not torch.equal(J_before, J_after)
        change = torch.linalg.norm(J_after - J_before).item()
        assert change > 0

    # ------------------------------------------------------------
    # 2. 惊讶度加权采样
    # ------------------------------------------------------------

    def test_sample_by_surprise(self, dream_engine_with_memories):
        """surprise 加权采样返回有效条目。"""
        engine = dream_engine_with_memories
        torch.manual_seed(42)
        state, surprise = engine._sample_by_surprise()
        assert state is not None
        assert state.shape == (engine.num_nodes,)
        assert isinstance(surprise, float)

    def test_sample_by_surprise_weights(self, dream_engine):
        """高 surprise 条目被采样概率更高（多次采样统计验证）。"""
        engine = dream_engine
        engine.memory._buffer.clear()
        # 两条条目：高 surprise 与低 surprise
        state_high = torch.ones(32) * 0.5
        state_low = torch.ones(32) * (-0.5)
        engine.memory._buffer.append((state_high.clone(), 5.0))
        engine.memory._buffer.append((state_low.clone(), 0.0))

        torch.manual_seed(42)
        high_count = 0
        n = 200
        for _ in range(n):
            _, surprise = engine._sample_by_surprise()
            if surprise == pytest.approx(5.0):
                high_count += 1

        # 高 surprise 应被采样更多（softmax([5,0]) ≈ 0.993）
        assert high_count > n / 2, (
            f"高 surprise 应被采样更多，实际 high_count={high_count}/{n}"
        )

    def test_dream_softmax_not_degenerate(self, dream_engine):
        """surprise 跨度 [1, 100] 时 softmax 不退化（标准化后非 argmax）。

        （惊讶度修复-01-设计方案.md §5.2 新增：z-score 标准化 + z 截断 +
        β=1.0 下，权重应有可辨方差，且最高 surprise 采样概率仍最高。）
        """
        engine = dream_engine
        # 构造跨度 [1, 100] 的缓冲区
        buf = []
        for i in range(20):
            state = torch.ones(engine.num_nodes) * (i + 1) / 20.0
            surprise = 1.0 + i * (99.0 / 19.0)
            buf.append((state, surprise))
        # 计算权重分布（复用引擎内部逻辑）
        surprises = torch.tensor([s for (_, s) in buf], dtype=torch.float32)
        z = (surprises - surprises.mean()) / surprises.std()
        z = z.clamp(-3.0, 3.0)
        weights = torch.exp(1.0 * z)
        weights = weights / weights.sum()
        assert float(weights.std()) > 0.01, (
            f"权重方差过小（{float(weights.std()):.4f}），采样退化为均匀")
        # 最高 surprise 项权重仍最高
        assert float(weights.argmax().item()) == int(surprises.argmax().item())

    def test_dream_softmax_single_outlier(self, dream_engine):
        """重尾单离群（99 条 ~0.1 + 1 条 100）不退化 argmax。

        （惊讶度修复-01-设计方案.md §5.2 新增，v1.1 审计必须修改项 4：
        z 截断 clamp(±3) + β=1.0 下离群项权重 < 0.9（对照：无截断时
        exp(β·z≈10) 退化为 argmax ≈1.0），且离群项采样概率仍最高。）
        """
        engine = dream_engine
        buf = [(torch.zeros(engine.num_nodes), 0.1) for _ in range(99)]
        buf.append((torch.ones(engine.num_nodes), 100.0))

        # 权重计算（z 截断 + β=1.0）
        surprises = torch.tensor([s for (_, s) in buf], dtype=torch.float32)
        z = (surprises - surprises.mean()) / surprises.std()
        z = z.clamp(-3.0, 3.0)
        weights = torch.exp(1.0 * z)
        weights = weights / weights.sum()
        outlier_w = float(weights[-1].item())
        assert outlier_w < 0.9, (
            f"离群项权重 {outlier_w:.4f} ≥ 0.9，softmax 退化为 argmax")
        assert float(weights.std()) > 0.01
        assert float(weights.argmax().item()) == 99, (
            "离群项（最高 surprise）采样概率应仍最高")

        # 端到端采样验证（多次采样统计，离群项应被采到但非垄断）
        torch.manual_seed(3)
        high_count = 0
        n = 1000
        for _ in range(n):
            state, _ = engine._sample_by_surprise(buffer=buf)
            if float(state.sum()) > 0:
                high_count += 1
        assert 0.05 < high_count / n < 0.9, (
            f"离群项采样率 {high_count / n:.3f} 异常（应 >0 且 <0.9）")

    # ------------------------------------------------------------
    # 3. 空闲态推断与学习
    # ------------------------------------------------------------

    def test_idle_infer_zero_precision(self, dream_engine):
        """零精度推断返回有效 Activation。"""
        engine = dream_engine
        # _idle_infer 用该输入作为 sigma 种子（num_nodes 维），而非感官输入
        sensory = torch.randn(engine.num_nodes) * 0.5
        torch.manual_seed(42)
        activation = engine._idle_infer(sensory)
        assert isinstance(activation, Activation)
        assert activation.state.shape == (engine.num_nodes,)
        assert torch.isfinite(activation.state).all()
        assert isinstance(activation.entropy, float)
        assert isinstance(activation.surprise, float)
        # 零精度向量全零
        assert torch.all(engine.idle_precision == 0)

    def test_dream_idle_surprise_reconstruction(self, dream_engine):
        """做梦 surprise = 对种子记忆的重构误差（零精度陷阱回归钉死）。

        （惊讶度修复-01-设计方案.md §5.2 新增：零 precision 下准确性项恒为 0，
        做梦 surprise 须改用 0.5·Σ(σ−seed)²；断言 > 0 且等于手算值。）
        """
        engine = dream_engine
        seed = torch.randn(engine.num_nodes) * 0.5
        torch.manual_seed(42)
        activation = engine._idle_infer(seed)
        # 手算重构误差（感官部分 = 前 input_dim 个节点）
        replay_error = (
            activation.state[:engine.input_dim] - seed[:engine.input_dim])
        manual = float(0.5 * torch.sum(replay_error ** 2).item())
        assert activation.surprise > 0.0, (
            f"做梦 surprise={activation.surprise} 应为正（重构误差）")
        assert activation.surprise == pytest.approx(manual, abs=1e-6)
        # per_dim 同步填充（供目的层/回填使用）
        assert activation.per_dim_surprise is not None
        assert activation.per_dim_surprise.shape == (engine.input_dim,)

    def test_idle_learn_low_lr(self, dream_engine):
        """低学习率学习后 J 变化小于在线学习。"""
        engine = dream_engine
        # _idle_infer 用该输入作为 sigma 种子（num_nodes 维），而非感官输入
        sensory = torch.randn(engine.num_nodes) * 0.5
        torch.manual_seed(42)
        # 先用零精度推断产生一个激活态
        engine.attractor.sigma = torch.zeros(engine.num_nodes)
        activation = engine._idle_infer(sensory)

        # 空闲态学习（idle_lr=0.001）
        J0 = engine.attractor.J.clone()
        engine._idle_learn(activation, sensory)
        idle_change = torch.linalg.norm(
            engine.attractor.J - J0).item()

        # 重置 J 后做在线学习（lr=0.01，在线的 10 倍）
        engine.attractor.J = J0.clone()
        engine.attractor.learn(activation, sensory, 0.01)
        online_change = torch.linalg.norm(
            engine.attractor.J - J0).item()

        assert idle_change < online_change, (
            f"空闲态学习变化({idle_change})应小于在线学习({online_change})"
        )

    # ------------------------------------------------------------
    # 4. 突触稳态下调 (SHY)
    # ------------------------------------------------------------

    def test_synaptic_homeostasis(self, dream_engine):
        """SHY 后 J 矩阵 Frobenius 范数不超过 target。"""
        engine = dream_engine
        # 构造一个范数远超 target 的大 J
        torch.manual_seed(42)
        big_J = torch.randn(32, 32) * 0.5
        big_J = (big_J + big_J.T) / 2
        big_J.fill_diagonal_(0)
        engine.attractor.J = big_J

        target = engine.shy_target_norm
        assert torch.linalg.norm(engine.attractor.J).item() > target

        engine._synaptic_homeostasis()
        norm = torch.linalg.norm(engine.attractor.J).item()
        assert norm <= target + 1e-6, (
            f"SHY 后 J 范数({norm})应不超过 target({target})"
        )

    def test_synaptic_homeostasis_preserves_structure(self, dream_engine):
        """SHY 后 J 的符号模式不变。"""
        engine = dream_engine
        torch.manual_seed(42)
        big_J = torch.randn(32, 32) * 0.5
        big_J = (big_J + big_J.T) / 2
        big_J.fill_diagonal_(0)
        engine.attractor.J = big_J

        signs_before = torch.sign(engine.attractor.J)
        engine._synaptic_homeostasis()
        signs_after = torch.sign(engine.attractor.J)

        assert torch.equal(signs_before, signs_after), (
            "SHY 缩放为正因子，J 的符号模式应保持不变"
        )

    # ------------------------------------------------------------
    # 5. 遗忘修剪
    # ------------------------------------------------------------

    def test_forgetting_pruning(self, dream_engine):
        """遗忘修剪后低 surprise 记忆在 long_term_latent 中痕迹衰减。"""
        engine = dream_engine
        engine.memory._buffer.clear()
        state_high = torch.ones(32) * 0.8
        state_low = torch.ones(32) * 0.8
        engine.memory._buffer.append((state_high.clone(), 5.0))
        engine.memory._buffer.append((state_low.clone(), 0.0))

        # 存储初始 long_term_latent
        ltl_before = engine.memory.long_term_latent.clone()

        engine._forgetting_pruning()

        # 验证 long_term_latent 发生了变化（低 surprise 记忆被衰减）
        ltl_after = engine.memory.long_term_latent
        assert not torch.equal(ltl_before, ltl_after), (
            "遗忘修剪后 long_term_latent 应发生变化（低 surprise 记忆被衰减）"
        )

    # ------------------------------------------------------------
    # 6. 景观漂移
    # ------------------------------------------------------------

    def test_landscape_drift(self, dream_engine):
        """景观漂移后 J 变化微小（范数变化 < 阈值）。"""
        engine = dream_engine
        J_before = engine.attractor.J.clone()
        torch.manual_seed(42)
        engine._landscape_drift()
        J_after = engine.attractor.J.clone()

        change = torch.linalg.norm(J_after - J_before).item()
        # drift_scale=0.001，漂移应远小于 0.1
        assert change < 0.1, f"景观漂移变化({change})应微小(<0.1)"

    def test_landscape_drift_symmetry(self, dream_engine):
        """漂移后 J 保持对称。"""
        engine = dream_engine
        torch.manual_seed(42)
        engine._landscape_drift()
        J = engine.attractor.J
        assert torch.allclose(J, J.T, atol=1e-10), "漂移后 J 应保持对称"
        # 对角线应为 0
        assert torch.allclose(J.diagonal(), torch.zeros_like(J.diagonal()))

    # ------------------------------------------------------------
    # 7. 目的层演化
    # ------------------------------------------------------------

    def test_purpose_evolve(self, dream_engine):
        """目的演化后 precision 发生变化。"""
        engine = dream_engine
        precision_before = engine.purpose.get_precision().clone()

        # 构造一个非平凡的激活态触发 precision 调整
        torch.manual_seed(42)
        state = torch.randn(engine.num_nodes) * 0.5
        activation = Activation(state=state, entropy=1.0, surprise=0.5)
        engine._purpose_evolve(activation)

        precision_after = engine.purpose.get_precision()
        assert not torch.equal(precision_before, precision_after), (
            "目的演化后 precision 应发生变化"
        )

    # ------------------------------------------------------------
    # 8. 坍缩检测
    # ------------------------------------------------------------

    def test_check_collapse_no_collapse(self, dream_engine):
        """正常情况下（多样激活态）不检测到坍缩。"""
        engine = dream_engine
        torch.manual_seed(42)
        for _ in range(5):
            state = torch.randn(engine.num_nodes) * 0.5
            act = Activation(state=state, entropy=1.0, surprise=0.5)
            assert engine._check_collapse(act) is False

    def test_check_collapse_detected(self, dream_engine):
        """重复相同激活态时检测到坍缩。"""
        engine = dream_engine
        torch.manual_seed(42)
        state = torch.randn(engine.num_nodes) * 0.5
        act = Activation(state=state, entropy=1.0, surprise=0.5)

        # 第一次：历史为空，无坍缩
        assert engine._check_collapse(act) is False
        # 第二次：相同状态，余弦相似度=1.0 > 阈值 → 坍缩
        assert engine._check_collapse(act) is True

    # ------------------------------------------------------------
    # 9. 扰动注入
    # ------------------------------------------------------------

    def test_inject_perturbation(self, dream_engine):
        """注入扰动后 sigma 和 J 发生变化。"""
        engine = dream_engine
        sigma_before = engine.attractor.sigma.clone()
        J_before = engine.attractor.J.clone()

        torch.manual_seed(42)
        engine._inject_perturbation()

        assert not torch.equal(sigma_before, engine.attractor.sigma), (
            "注入扰动后 sigma 应变化"
        )
        assert not torch.equal(J_before, engine.attractor.J), (
            "注入扰动后 J 应变化"
        )

    # ------------------------------------------------------------
    # 10. 完整做梦周期
    # ------------------------------------------------------------

    def test_dream_cycle(self, dream_engine_with_memories):
        """完整做梦周期正常运行，返回统计 dict。"""
        engine = dream_engine_with_memories
        torch.manual_seed(42)
        result = engine.dream_cycle(max_steps=30)
        assert isinstance(result, dict)
        assert result['status'] == 'dreamed'
        assert result['cycles'] == 30
        assert 'avg_surprise' in result
        assert 'phases' in result
        assert isinstance(result['phases'], dict)
        # 阶段计数之和应等于总步数
        assert sum(result['phases'].values()) == 30

    def test_dream_cycle_preserves_components(self, dream_engine_with_memories):
        """做梦后组件状态被正确恢复（temperature, orth_weight）。"""
        engine = dream_engine_with_memories
        orig_temp = engine.attractor.temperature
        orig_orth = engine.attractor.orth_weight

        torch.manual_seed(42)
        engine.dream_cycle(max_steps=20)

        assert engine.attractor.temperature == pytest.approx(orig_temp), (
            "做梦后 temperature 应恢复"
        )
        assert engine.attractor.orth_weight == pytest.approx(orig_orth), (
            "做梦后 orth_weight 应恢复"
        )

    # ------------------------------------------------------------
    # 11. 状态查询
    # ------------------------------------------------------------

    def test_get_status(self, dream_engine_with_memories):
        """get_status 返回正确的字段。"""
        engine = dream_engine_with_memories
        status = engine.get_status()
        assert isinstance(status, dict)
        # 关键字段存在
        for key in ['step_count', 'num_nodes', 'input_dim',
                    'idle_learning_rate', 'idle_orth_weight',
                    'idle_temperature', 'consolidation_ratio',
                    'collapse_threshold', 'buffer_size', 'j_norm',
                    'precision_mean']:
            assert key in status, f"get_status 缺少字段 {key}"
        assert status['num_nodes'] == 32
        assert status['input_dim'] == 8
        assert status['buffer_size'] > 0

    # ------------------------------------------------------------
    # 12. 快照保存
    # ------------------------------------------------------------

    def test_dream_mvp_saves_snapshot(self, dream_engine, tmp_path):
        """dream_mvp 后快照文件被创建（使用 tmp_path）。

        T1.1/P0-5：快照存入会话子目录 snapshots/{session}/latest_{session}.pt。
        """
        engine = dream_engine
        populate_buffer(engine, n=5)
        engine.dream_mvp(n_steps=5)

        snapshots = list(tmp_path.glob('default/*.pt'))
        assert len(snapshots) >= 1, (
            f"dream_mvp 后应创建会话级快照文件，实际 {len(snapshots)} 个"
        )
        assert os.path.exists(
            os.path.join(str(tmp_path), 'default', 'latest_default.pt'))

    # ------------------------------------------------------------
    # 13. 稳定性
    # ------------------------------------------------------------

    def test_repeated_dream_cycles_stable(self, dream_engine_with_memories):
        """连续多次做梦周期系统不崩溃。"""
        engine = dream_engine_with_memories
        torch.manual_seed(42)
        for i in range(3):
            result = engine.dream_cycle(max_steps=15)
            assert result['status'] == 'dreamed'

        # 系统状态数值稳定（无 NaN/Inf）
        assert torch.isfinite(engine.attractor.J).all()
        assert torch.isfinite(engine.attractor.sigma).all()
        assert torch.isfinite(engine.purpose.sensory_precision).all()
        # 步数累计正确
        assert engine.step_count == 45


# ============================================================
# 集成测试
# ============================================================

class TestDreamEngineIntegration:
    """DreamEngine 与 LivingMemoryLoop 的集成测试。"""

    def test_loop_dream_integration(self, tmp_path):
        """LivingMemoryLoop.dream() 方法正常工作。"""
        torch.manual_seed(42)
        config = make_test_config(snapshot_dir=str(tmp_path))
        loop = LivingMemoryLoop(config)

        # 积累一些记忆
        for i in range(5):
            loop.process_turn(f"做梦集成测试第{i+1}轮")

        result = loop.dream(n_steps=10)
        assert isinstance(result, dict)
        assert result['status'] in ('dreamed', 'no_memories_to_replay')
        if result['status'] == 'dreamed':
            assert result['steps'] == 10

    def test_dream_after_store(self, tmp_path):
        """存储记忆后做梦，再检索，记忆仍可被找到。"""
        torch.manual_seed(42)
        config = make_test_config(
            snapshot_dir=str(tmp_path),
            embedder=MockEmbedder(dim=16),
        )
        loop = LivingMemoryLoop(config)

        # 存储一条情景记忆
        loop.process_turn("重要的AI记忆内容")

        # 做梦
        loop.dream(n_steps=5)

        # 检索：情景记忆仍可被找到
        query = loop.embedder.embed_text("重要的AI记忆内容")
        entries = loop.memory.recall_episodic(query, top_k=3)
        assert len(entries) > 0, "做梦后情景记忆仍应可被检索"
        assert "AI" in entries[0].text

    def test_dream_improves_recall(self, tmp_path):
        """做梦后记忆检索质量不下降（巩固不破坏已有记忆）。"""
        torch.manual_seed(42)
        config = make_test_config(
            snapshot_dir=str(tmp_path),
            embedder=MockEmbedder(dim=16),
        )
        loop = LivingMemoryLoop(config)

        # 存储多条记忆
        texts = ["人工智能", "机器学习", "深度学习网络"]
        for t in texts:
            loop.process_turn(t)

        # 做梦前检索："人工智能" 应是 top-1
        query = loop.embedder.embed_text("人工智能")
        entries_before = loop.memory.recall_episodic(query, top_k=1)
        assert len(entries_before) == 1
        assert "人工智能" in entries_before[0].text

        # 做梦
        loop.dream(n_steps=8)

        # 做梦后检索：top-1 仍应是 "人工智能"（巩固不破坏已有记忆）
        entries_after = loop.memory.recall_episodic(query, top_k=1)
        assert len(entries_after) == 1
        assert "人工智能" in entries_after[0].text, (
            "做梦后记忆检索质量不应下降"
        )

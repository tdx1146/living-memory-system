# -*- coding: utf-8 -*-
"""阶段 1 弥散态专项（v1.2 §三 A 级）：per-node precision 差异性重标定测试。

背景（v1.2 §一/§三）：弥散态根因链 = J@σ 主导（J_norm 40 自 8/10 钳制）→
σ 输入不变（inter-input cosine ≈ 1.0）→ precision 塌缩（极差 ≈1.2%）→
surprise 退化为 mse 线性函数（surprise ≈ 0.5·π̄·d·mse，方向响应丢失）。
A 级修复 = 每个 sensory 节点的 precision 按其激活模式单独重标定
（恢复 node 间差异，非全局缩放），均值恢复到 8/10 有结构期水平（π̄≈1.5）。

覆盖（任务书 §三 A 级 + v1.2 验收 2 判据）：
  1. 治理开关（LMS_PRECISION_RECALIB 默认关；关=零行为变化，开=重标定介入）
  2. 开关开 → node 间 precision 差异恢复（flat precision → std > 0）
  3. 均值恢复目标（默认 1.5；LMS_PRECISION_RECALIB_MEAN 可配）
  4. 强度参数（STRENGTH=0 → 无差异；>0 → 差异随强度放大）
  5. clamp 合法范围 [0.1, 10]
  6. 不同输入簇 → 重标定后 precision 模式可分（组间激活模式可分判据）
  7. 无结构输入（全等）→ 原样返回（不编造差异）
  8. 灵魂机制：开关关时 surprise = 0.5·π̄·d·mse 线性成立（v1.2 §一公式）；
     开关开时线性被打破（π 恢复 node 间差异 → 方向响应不再被压平）

测试约定：重标定路径显式 monkeypatch 启用开关（开关默认关，零行为变化，
既有测试不受影响——与 test_precision_adapt.py 同约定）。
"""

import os
import sys

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
import torch

from core.hippocampus.attractor import AttractorNetwork


# ----------------------------------------------------------------------- #
#  治理开关（LMS_PRECISION_RECALIB 开关族）
# ----------------------------------------------------------------------- #


class TestRecalibSwitch:
    """开关族：默认关（零行为变化）/ env 开 / 参数可配。"""

    def test_default_off(self):
        """未设 env → 开关关（零行为变化，可回滚）。"""
        net = AttractorNetwork(num_nodes=16, input_dim=8, seed=1)
        assert net.precision_recalib_enabled is False

    def test_env_on(self, monkeypatch):
        """LMS_PRECISION_RECALIB=1 → 开关开，默认参数就位。"""
        monkeypatch.setenv("LMS_PRECISION_RECALIB", "1")
        net = AttractorNetwork(num_nodes=16, input_dim=8, seed=1)
        assert net.precision_recalib_enabled is True
        assert net.precision_recalib_mean == 1.5  # 8/10 有结构期均值
        assert net.precision_recalib_strength == 1.0

    def test_env_params(self, monkeypatch):
        """MEAN/STRENGTH 可配（env 族完整）。"""
        monkeypatch.setenv("LMS_PRECISION_RECALIB", "1")
        monkeypatch.setenv("LMS_PRECISION_RECALIB_MEAN", "2.5")
        monkeypatch.setenv("LMS_PRECISION_RECALIB_STRENGTH", "0.5")
        net = AttractorNetwork(num_nodes=16, input_dim=8, seed=1)
        assert net.precision_recalib_enabled is True
        assert net.precision_recalib_mean == 2.5
        assert net.precision_recalib_strength == 0.5

    def test_env_off_takes_precedence(self, monkeypatch):
        """置 0 即回退（生产回滚路径：快照回滚 + env 置 0）。"""
        monkeypatch.setenv("LMS_PRECISION_RECALIB", "0")
        monkeypatch.setenv("LMS_PRECISION_RECALIB_MEAN", "9.9")
        net = AttractorNetwork(num_nodes=16, input_dim=8, seed=1)
        assert net.precision_recalib_enabled is False


# ----------------------------------------------------------------------- #
#  per-node 重标定单元行为
# ----------------------------------------------------------------------- #


class TestRecalibratePrecisionPerNode:
    """_recalibrate_precision_per_node 的单元行为。"""

    def _net(self, monkeypatch):
        monkeypatch.setenv("LMS_PRECISION_RECALIB", "1")
        return AttractorNetwork(num_nodes=16, input_dim=8, seed=1)

    def test_flat_precision_gets_differentiation(self, monkeypatch):
        """弥散态场景：precision 全平（π̄≈0.3469）→ 重标定后 node 间差异恢复。

        v1.2 §三 判据：恢复 node 间差异（不是全局乘一个系数）。
        """
        net = self._net(monkeypatch)
        precision = torch.full((8,), 0.3469)
        sensory = torch.tensor([0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2])
        out = net._recalibrate_precision_per_node(precision, sensory)
        assert out.shape == precision.shape
        # node 间差异恢复：std 显著 > 0（弥散态 flat 的 std ≈ 0）
        assert out.std().item() > 0.05, f"重标定后应有 node 间差异，std={out.std().item():.4f}"
        # 均值恢复到目标（默认 1.5）。注意：clamp [0.1,10] 会截掉下尾
        # （极端负 z → 0.1），使均值略高于目标——容忍 clamp 畸变（10%）。
        assert out.mean().item() == pytest.approx(1.5, rel=0.1)
        # 范围合法（clamp [0.1, 10]，对齐 purpose 层）
        assert out.min().item() >= 0.1 - 1e-6
        assert out.max().item() <= 10.0 + 1e-6
        # 原 precision 不被篡改（克隆语义）
        assert precision.std().item() == 0.0

    def test_mean_follows_target_env(self, monkeypatch):
        """均值恢复目标可配：MEAN=2.5 → 重标定后均值 ≈2.5。"""
        monkeypatch.setenv("LMS_PRECISION_RECALIB", "1")
        monkeypatch.setenv("LMS_PRECISION_RECALIB_MEAN", "2.5")
        net = AttractorNetwork(num_nodes=16, input_dim=8, seed=1)
        precision = torch.full((8,), 0.34)
        sensory = torch.tensor([0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2])
        out = net._recalibrate_precision_per_node(precision, sensory)
        assert out.mean().item() == pytest.approx(2.5, rel=0.1)

    def test_strength_zero_no_differentiation(self, monkeypatch):
        """STRENGTH=0 → 只恢复均值、不制造差异（强度语义）。"""
        monkeypatch.setenv("LMS_PRECISION_RECALIB", "1")
        monkeypatch.setenv("LMS_PRECISION_RECALIB_STRENGTH", "0")
        net = AttractorNetwork(num_nodes=16, input_dim=8, seed=1)
        precision = torch.full((8,), 0.34)
        sensory = torch.tensor([0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2])
        out = net._recalibrate_precision_per_node(precision, sensory)
        assert out.std().item() < 1e-6, "STRENGTH=0 不应制造 node 间差异"
        assert out.mean().item() == pytest.approx(1.5, abs=0.05)

    def test_strength_amplifies_differentiation(self, monkeypatch):
        """STRENGTH 放大 node 间差异（1.0 vs 3.0 → std 更大）。"""
        monkeypatch.setenv("LMS_PRECISION_RECALIB", "1")
        sensory = torch.tensor([0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2])
        precision = torch.full((8,), 0.34)

        monkeypatch.setenv("LMS_PRECISION_RECALIB_STRENGTH", "1.0")
        net1 = AttractorNetwork(num_nodes=16, input_dim=8, seed=1)
        out1 = net1._recalibrate_precision_per_node(precision, sensory)

        monkeypatch.setenv("LMS_PRECISION_RECALIB_STRENGTH", "3.0")
        net3 = AttractorNetwork(num_nodes=16, input_dim=8, seed=1)
        out3 = net3._recalibrate_precision_per_node(precision, sensory)

        assert out3.std().item() > out1.std().item()

    def test_structureless_input_unchanged(self, monkeypatch):
        """无结构输入（各维全等）→ 原样返回（不编造差异）。"""
        net = self._net(monkeypatch)
        precision = torch.full((8,), 0.3469)
        sensory = torch.full((8,), 0.5)  # 无激活模式结构
        out = net._recalibrate_precision_per_node(precision, sensory)
        assert torch.allclose(out, precision)

    def test_zero_input_unchanged(self, monkeypatch):
        """零输入（act_sum ≈ 0）→ 原样返回（除零保护）。"""
        net = self._net(monkeypatch)
        precision = torch.full((8,), 0.3469)
        sensory = torch.zeros(8)
        out = net._recalibrate_precision_per_node(precision, sensory)
        assert torch.allclose(out, precision)

    def test_different_clusters_distinguishable(self, monkeypatch):
        """组间激活模式可分判据（v1.2 §三）：不同输入簇 → 重标定模式可分。

        弥散态下 inter-input cosine ≈ 1.0（不可分）；重标定后应恢复可分。
        """
        net = self._net(monkeypatch)
        precision = torch.full((8,), 0.3469)
        cluster_a = torch.tensor([0.9, -0.8, 0.7, -0.6, 0.5, -0.4, 0.3, -0.2])
        cluster_b = torch.tensor([-0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8])
        pa = net._recalibrate_precision_per_node(precision, cluster_a)
        pb = net._recalibrate_precision_per_node(precision, cluster_b)
        cos = torch.nn.functional.cosine_similarity(
            pa - pa.mean(), pb - pb.mean(), dim=0).item()
        assert cos < 0.99, f"不同输入簇的重标定模式应可分，cosine={cos:.4f}"


# ----------------------------------------------------------------------- #
#  灵魂机制：infer 集成（开关差异 + 线性公式被打破）
# ----------------------------------------------------------------------- #


class TestRecalibInferIntegration:
    """经 infer 的开关行为差异（v1.2 §一 公式 + §三 判据）。"""

    def test_off_surprise_is_linear_in_mse(self):
        """开关关：surprise = 0.5·π̄·d·mse 精确成立（v1.2 §一 公式本体）。

        弥散态的"平"= precision 恒定提出 Σ 外 → 惊讶度被压成 mse 线性函数。
        """
        net = AttractorNetwork(num_nodes=16, input_dim=8, seed=1)
        precision = torch.full((8,), 0.3469)
        sensory = torch.randn(8) * 0.5
        act = net.infer(sensory, precision, update_internal_state=False)
        d = 8
        expect = 0.5 * 0.3469 * d * act.mse
        assert act.surprise == pytest.approx(expect, rel=0.01), (
            f"开关关时 surprise 应为 mse 线性函数: {act.surprise} vs {expect}")

    def test_on_breaks_linear_formula(self, monkeypatch):
        """开关开：线性公式被打破（π 恢复 node 间差异 → 方向响应不被压平）。

        v1.2 §三 灵魂判据：不是绝对值回范围，是"π̄ 提不出 Σ 外"。
        """
        monkeypatch.setenv("LMS_PRECISION_RECALIB", "1")
        net = AttractorNetwork(num_nodes=16, input_dim=8, seed=1)
        precision = torch.full((8,), 0.3469)
        sensory = torch.randn(8) * 0.5
        act = net.infer(sensory, precision, update_internal_state=False)
        linear_expect = 0.5 * 0.3469 * 8 * act.mse
        assert act.surprise != pytest.approx(linear_expect, rel=0.01), (
            "开关开时 surprise 不应再是 0.5·π̄·d·mse 线性函数")
        # 重标定后 π 的 node 间差异确实参与了惊讶度（per_dim 与 mse 同形性打破）
        assert net.precision_recalib_enabled is True

    def test_on_vs_off_differ_same_input(self, monkeypatch):
        """同一输入：开关开/关 → surprise 不同（开关行为差异，控制变量）。"""
        # off
        net_off = AttractorNetwork(num_nodes=16, input_dim=8, seed=42)
        # on
        monkeypatch.setenv("LMS_PRECISION_RECALIB", "1")
        net_on = AttractorNetwork(num_nodes=16, input_dim=8, seed=42)

        precision = torch.full((8,), 0.3469)
        sensory = torch.randn(8, generator=torch.Generator().manual_seed(7)) * 0.5
        act_off = net_off.infer(sensory, precision, update_internal_state=False)
        act_on = net_on.infer(sensory, precision, update_internal_state=False)
        assert act_off.surprise != pytest.approx(act_on.surprise, rel=0.01), (
            "开关开/关对同一输入应产生不同惊讶度（重标定真实介入）")

    def test_on_precision_std_restored_in_effect(self, monkeypatch):
        """开关开：有效 precision（重标定后）极差/标准差恢复（弥散态 1.2% 的病）。"""
        monkeypatch.setenv("LMS_PRECISION_RECALIB", "1")
        net = AttractorNetwork(num_nodes=16, input_dim=8, seed=1)
        precision = torch.full((8,), 0.3469)
        sensory = torch.randn(8, generator=torch.Generator().manual_seed(3)) * 0.5
        effective = net._recalibrate_precision_per_node(precision, sensory)
        # 弥散态 flat precision 极差/均值 ≈ 1.2%；重标定后应拉开一个量级以上
        spread = (effective.max().item() - effective.min().item()) / effective.mean().item()
        assert spread > 0.2, f"重标定后 precision 相对极差应 >20%，实际 {spread*100:.1f}%"

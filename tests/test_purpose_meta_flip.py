# -*- coding: utf-8 -*-
"""体验层 C（设计 v1.1 §5）：元目的翻转回初版（强化已关注方向）测试。

验收（§5.3）：
  1. 单测：构造低 coherence 场景 → _meta_adjust 后
     sensory_precision.argmax() = 历史平均 precision 最高维度（非 encounter 最低）；
  2. 200 轮演化：目的层行为向"越关注越深入"演化（不切换到未探索维度）；
  3. 现有 test_purpose.py 翻转用例不锁 argmin 语义 → 全绿（回归）。
"""

import os
import sys

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
import torch

from core.hippocampus.purpose import PurposeLayer
from core.types import Activation


class TestMetaAdjustArgmaxSemantics:
    """翻转目标 = 历史平均 precision 最高维度（非 encounter 最低）。"""

    def test_meta_adjust_targets_most_attended_dim(self):
        p = PurposeLayer(
            input_dim=8, coherence_threshold=0.5, min_history_length=3,
            meta_window=5)
        # 历史：维度 2 平均 precision 最高（8.0），其余 1.0
        base = torch.ones(8) * 1.0
        for i in range(5):
            h = base.clone()
            h[2] = 6.0 + i  # 平均 = 8.0
            p.history.append(h)
        # encounter_count：维度 2 高（已关注），维度 7 为 0（最未探索）
        # → argmin 语义会选 7，argmax（初版）语义应选 2
        p.encounter_count = torch.zeros(8)
        p.encounter_count[2] = 50.0
        p.coherence = 0.2  # 低 coherence（触发条件）

        p._meta_adjust()

        assert p.flipped and p.flip_count == 1
        # 目标维度 = 历史平均 precision 最高的维度 2
        assert p.sensory_precision[2] == p.precision_max
        # 未探索维度 7 不被强化
        assert p.sensory_precision[7] < p.precision_max

    def test_meta_adjust_avg_precision_math(self):
        """argmax 语义的数学验证：window/recent/stack/mean/argmax 与手算一致。"""
        p = PurposeLayer(input_dim=4, meta_window=3)
        # 历史 5 条，最近 3 条平均：dim1 最高
        for i, vals in enumerate([
            [1.0, 1.0, 1.0, 1.0],
            [2.0, 2.0, 1.0, 1.0],
            [3.0, 5.0, 1.0, 1.0],
            [1.0, 6.0, 1.0, 1.0],
            [1.0, 7.0, 1.0, 1.0],
        ]):
            p.history.append(torch.tensor(vals, dtype=torch.float32))
        p.encounter_count = torch.tensor([0.0, 100.0, 0.0, 0.0])
        p.coherence = 0.1
        p._meta_adjust()
        # 最近 3 条（索引 2/3/4）dim1 均值 = (5+6+7)/3 = 6.0 → argmax = 1
        assert p.sensory_precision[1] == p.precision_max
        # 维度 3 是 encounter 最低（0），但不应被选
        assert p.sensory_precision[3] < p.precision_max


class TestPurposeEvolutionDeepensFocus:
    """200 轮演化：目的层向"越关注越深入"演化（不切换到未探索维度）。"""

    def test_200_rounds_deepen_attended_dims(self):
        torch.manual_seed(42)
        p = PurposeLayer(
            input_dim=8, coherence_threshold=0.5, min_history_length=5,
            meta_window=10, habituation_rate=0.05)
        # 固定输入模式：强激活集中在维度 0-1（已关注方向），维度 6-7 从不激活
        pattern = torch.zeros(8)
        pattern[0] = 0.9
        pattern[1] = 0.7
        pattern[2] = 0.2

        for _ in range(200):
            act = Activation(
                state=pattern.clone(), entropy=0.5, surprise=1.0)
            p.adjust(1.0, act)

        # 已关注维度 precision 不低于未探索维度（越关注越深入，不跑偏）
        assert p.sensory_precision[0] >= p.sensory_precision[6] - 1e-6
        assert p.sensory_precision[1] >= p.sensory_precision[7] - 1e-6
        # 翻转后强化的是高平均 precision 维度（若发生过翻转）
        if p.flip_count > 0:
            window = min(p.meta_window, len(p.history))
            avg = torch.stack(p.history[-window:]).mean(dim=0)
            target = int(avg.argmax().item())
            assert p.sensory_precision[target] >= p.sensory_precision.max() - 1e-6

    def test_meta_adjust_via_adjust_low_coherence(self):
        """通过 adjust 触发的端到端：低 coherence → 翻转强化已关注方向。"""
        p = PurposeLayer(
            input_dim=6, coherence_threshold=0.99, min_history_length=3,
            meta_window=4)  # 阈值拉高，确保触发
        # 制造历史：dim3 平均 precision 最高
        for i in range(5):
            h = torch.ones(6) * 1.0
            h[3] = 4.0 + i
            p.history.append(h)
        p.coherence = 0.1
        p.encounter_count = torch.zeros(6)
        p.encounter_count[5] = 0.0  # 最未探索
        act = Activation(state=torch.zeros(6), entropy=0.5, surprise=1.0)
        p.adjust(1.0, act)
        # flip 后 dim3 被强化到最大值（或接近——EMA 后再翻转置 max）
        assert p.sensory_precision[3] == p.precision_max

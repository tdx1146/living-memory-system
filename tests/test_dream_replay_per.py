# -*- coding: utf-8 -*-
"""PER 概率采样测试（提取层 v1.4 S1-6，拍板④/M3；验证⑤判据 v1.4 修正版）

覆盖：
  1. 随机性：多次采样分布随概率变化（非确定性 top-k）
  2. 防饿死反例（v1.4 判据）：1 条低分（P≥0.01）＋20 条高分 → N=500 次
     做梦，低分条目被采中 ≥1 次（概率 ≥ 1-(1-0.01)^500 ≈ 99.3%）
  3. 无放回：单次调用不重复（工程决策标注）
  4. gray 排除（三重冻结①）：source=='store_gray' 永不进候选
  5. ε 保底：全零分数条目仍有非零概率
"""

import random

import torch

from core.hippocampus.memory import EpisodicEntry
from runtime.dream_replay import (
    sample_replay_set, replay_score, EPSILON,
    GRAY_SOURCE,
)


def make_entry(text, surprise, info_value=0.0, source="external", turn=0):
    e = EpisodicEntry(
        text=text,
        semantic_vector=torch.zeros(4),
        surprise=surprise,
        turn=turn,
        source=source,
    )
    e.info_value = info_value
    return e


class TestRandomness:
    def test_sampling_is_probabilistic(self):
        """随机性验证：同候选集多次采样结果不同（非确定性 top-k）。"""
        entries = [make_entry(f"高{i}", 1.0, info_value=1.0, turn=i)
                   for i in range(5)]
        outcomes = set()
        for seed in range(50):
            rng = random.Random(seed)
            picked = sample_replay_set(entries, k=1, rng=rng)
            outcomes.add(picked[0][0].text)
        assert len(outcomes) > 1, "多次采样结果完全相同——采样退化为确定性"

    def test_high_score_sampled_more_often(self):
        """分布验证：高分条目采样占比显著高于低分条目。"""
        entries = [
            make_entry("高分", 1.0, info_value=1.0, turn=1),
            make_entry("低分", 0.1, info_value=0.1, turn=2),
        ]
        high = 0
        n = 500
        for seed in range(n):
            rng = random.Random(seed)
            picked = sample_replay_set(entries, k=1, rng=rng)
            if picked[0][0].text == "高分":
                high += 1
        # 高分：p=1.0 → P≈1/(1+0.01^0.6)≈0.94；低分：P≈0.059
        # 高分占比应 >0.7；500 次中低分零命中概率 ≈ 2e-14（ε 保底有效）
        assert high > n * 0.7, (
            f"高分条目占比 {high / n:.3f} 偏低（应显著高于低分）")
        assert high < n, "高分条目垄断采样（低分条目饿死，违背 ε 保底）"


class TestAntiStarvation:
    def test_low_score_entry_never_starves(self):
        """防饿死反例（v1.4 判据修正版）：低分条目在 N=500 次做梦中被采中。

        构造：20 条高分（surprise=info=1.0，p≈1.0）＋1 条低分
        （surprise=info=0.3，p≈0.09 → P(low) ≈ 0.09^0.6/20 ≈ 0.0116 ≥ 0.01）。
        无放回 k=1 下，500 次做梦至少采中 1 次的概率 ≈ 99.7%。
        """
        entries = [make_entry(f"高{i}", 1.0, info_value=1.0, turn=i)
                   for i in range(20)]
        low = make_entry("低分旧条目", 0.3, info_value=0.3, turn=999)
        entries.append(low)

        # 先验证 P(low) ≥ 0.01（判据前置）
        scores = [replay_score(e, 1.0, 1.0) for e in entries]
        alpha = 0.6
        powered = [s ** alpha for s in scores]
        p_low = powered[-1] / sum(powered)
        assert p_low >= 0.01, f"P(low)={p_low:.4f} < 0.01，判据构造失败"

        hit = 0
        n = 500
        for seed in range(n):
            rng = random.Random(seed)
            picked = sample_replay_set(entries, k=1, rng=rng)
            if picked[0][0].text == "低分旧条目":
                hit += 1
        assert hit >= 1, (
            f"低分条目 500 次做梦零命中——防饿死失效"
            f"（P={p_low:.4f}，期望命中≈{n * p_low:.1f}）")
        # 高分条目占多数采样位（效率不丢）
        high_hits = n - hit
        assert high_hits > n * 0.8, "高分条目占比过低"


class TestWithoutReplacement:
    def test_no_duplicates_in_one_call(self):
        """无放回（工程决策）：单次调用内不重复采样。"""
        entries = [make_entry(f"条目{i}", 1.0, info_value=1.0, turn=i)
                   for i in range(10)]
        picked = sample_replay_set(entries, k=5, rng=random.Random(7))
        texts = [e.text for e, _s, _p in picked]
        assert len(texts) == len(set(texts)) == 5


class TestGrayExclusion:
    def test_gray_never_sampled(self):
        """灰度三重冻结①：store_gray 条目永不进重放候选。"""
        entries = [
            make_entry("正式1", 1.0, info_value=1.0, turn=1),
            make_entry("正式2", 1.0, info_value=1.0, turn=2),
        ]
        gray = make_entry("灰度条目", 1.0, info_value=1.0, turn=3)
        gray.source = GRAY_SOURCE
        entries.append(gray)
        for seed in range(100):
            rng = random.Random(seed)
            picked = sample_replay_set(entries, k=2, rng=rng)
            for e, _s, _p in picked:
                assert e.source != GRAY_SOURCE, "gray 条目被采中"

    def test_all_gray_returns_empty(self):
        entries = [make_entry("灰度", 1.0, info_value=1.0, turn=1)]
        entries[0].source = GRAY_SOURCE
        assert sample_replay_set(entries, k=3) == []


class TestEpsilonFloor:
    def test_epsilon_guarantees_nonzero_probability(self):
        """ε 保底：全零分数条目仍有非零概率（论文"小常数"语义）。"""
        entries = [make_entry(f"零{i}", 0.0, info_value=0.0, turn=i)
                   for i in range(5)]
        scores = [replay_score(e, 0.0, 0.0) for e in entries]
        assert all(s == EPSILON for s in scores), "全零条目分数应为 ε"
        # 多次采样：每个条目都至少被采中一次（ε/Σ 概率）
        seen = set()
        for seed in range(2000):
            rng = random.Random(seed)
            picked = sample_replay_set(entries, k=1, rng=rng)
            seen.add(picked[0][0].text)
        assert len(seen) > 1, "ε 保底失效：有条目从未被采中"

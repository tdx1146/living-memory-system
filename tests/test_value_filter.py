# -*- coding: utf-8 -*-
"""价值过滤测试（提取层 v1.4 S1-4，按论文重做判据）

覆盖：
  1. info_value = 0.7×surprise_norm + 0.3×recall_hit_score（论文公式）
  2. min-max 归一化防除零（surprise_max/ref_max ≤ 0 → 0）
  3. 冷启动垫片（非承重）：无历史时启发式打底；纯确认词 → 0 分
  4. value_filtered 阈值判据（标记语义，不决定整轮生死）
"""

from api.value_filter import (
    compute_info_value, value_filtered,
    normalize_minmax, DEFAULT_VALUE_THRESHOLD,
)


class TestInfoValueFormula:
    def test_paper_formula(self):
        """info_value = 0.7×surprise_norm + 0.3×recall_hit。"""
        # surprise=0.5/1.0=0.5；ref=2/4=0.5 → 0.7×0.5+0.3×0.5=0.5
        iv = compute_info_value(surprise=0.5, reference_count=2,
                                surprise_max=1.0, ref_count_max=4)
        assert abs(iv - 0.5) < 1e-9

    def test_surprise_dominant(self):
        iv_hi_surprise = compute_info_value(
            surprise=1.0, reference_count=0, surprise_max=1.0,
            ref_count_max=10)
        iv_lo_surprise = compute_info_value(
            surprise=0.0, reference_count=0, surprise_max=1.0,
            ref_count_max=10)
        assert iv_hi_surprise > iv_lo_surprise

    def test_recall_hit_second_signal(self):
        """被引用次数越高价值越高（Mattar & Daw 未来效用）。"""
        iv0 = compute_info_value(surprise=0.5, reference_count=0,
                                 surprise_max=1.0, ref_count_max=10)
        iv5 = compute_info_value(surprise=0.5, reference_count=5,
                                 surprise_max=1.0, ref_count_max=10)
        assert iv5 > iv0


class TestNormalization:
    def test_normalize_minmax_zero_max(self):
        assert normalize_minmax(1.0, 0) == 0.0
        assert normalize_minmax(1.0, None) == 0.0
        assert normalize_minmax(0.5, 2.0) == 0.25

    def test_clamped_to_unit(self):
        assert normalize_minmax(3.0, 1.0) == 1.0
        assert normalize_minmax(-1.0, 1.0) == 0.0


class TestColdStartPad:
    def test_pad_when_no_history(self):
        """无 surprise/recall 历史 → 冷启动垫片打底（非承重）。"""
        iv = compute_info_value(surprise=0.0, reference_count=0,
                                surprise_max=1.0, ref_count_max=1.0,
                                text="这是一个包含方案和决定的较长句子，"
                                     "长度超过五十个字符以确保启发式分数。")
        assert iv > 0.0

    def test_pleasantry_zero_score(self):
        """纯确认词 → 0 分（客套过滤语义）。"""
        iv = compute_info_value(surprise=0.0, reference_count=0,
                                surprise_max=1.0, ref_count_max=1.0,
                                text="好的")
        assert iv == 0.0


class TestValueFiltered:
    def test_threshold_judgement(self):
        assert value_filtered(0.5) is True
        assert value_filtered(0.1) is False

    def test_default_threshold(self):
        assert DEFAULT_VALUE_THRESHOLD == 0.3

    def test_marks_but_does_not_kill(self):
        """M5：value_filtered 是条目标记，不决定整轮生死（丰碑哲学）。"""
        # 低价值条目仍可存储（调用方决定），判据只出标记
        assert value_filtered(0.2, threshold=0.3) is False
        assert value_filtered(0.2, threshold=0.1) is True

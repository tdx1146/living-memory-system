# -*- coding: utf-8 -*-
"""分段标记 + 提取式核心压缩测试（提取层 v1.4 S1-3/S1-5）

覆盖：
  1. segment_reply：句子切分、六类分类（事实修正/决策/偏好/情感/过程/客套）
  2. valuable_sentences：只保留有价值四类（H-4：无过程/客套污染）
  3. extract_core：长文本 → ≤300 字核心；两段式（4 类各取 top-1）
  4. 非生成式：核心只含原文子串
"""

import pytest

from api.segment_reply import (
    segment_reply, segment_sentences, classify_sentence,
    valuable_sentences, CATEGORY_FACT, CATEGORY_DECISION,
    CATEGORY_PREFERENCE, CATEGORY_EMOTION,
    CATEGORY_PROCESS, CATEGORY_PLEASANTRY,
)
from api.extract_core import extract_core, core_stats


class TestSegmentReply:
    def test_segment_sentences(self):
        sents = segment_sentences("第一句。第二句！第三句？第四句；第五句…第六句")
        assert len(sents) == 6
        assert sents[0] == "第一句。"

    def test_classify_fact(self):
        assert classify_sentence("实际上这个方案是错的，正确的是另一种。") \
            == CATEGORY_FACT

    def test_classify_decision(self):
        assert classify_sentence("我们决定采用方案B，下一步先做迁移。") \
            == CATEGORY_DECISION

    def test_classify_preference(self):
        assert classify_sentence("我更喜欢简洁的做法，希望少一点配置。") \
            == CATEGORY_PREFERENCE

    def test_classify_emotion(self):
        assert classify_sentence("真的很感谢你，我很开心。") == CATEGORY_EMOTION

    def test_classify_pleasantry(self):
        assert classify_sentence("好的。") == CATEGORY_PLEASANTRY

    def test_classify_process_fallback(self):
        # 无关键词命中的叙述长句 → 过程
        assert classify_sentence("这个模块包含了一些基础功能。") \
            == CATEGORY_PROCESS

    def test_valuable_sentences_excludes_process_and_pleasantry(self):
        reply = ("好的。"
                 "实际上方案A有性能问题，正确的是方案B。"
                 "我们决定采用方案B。"
                 "另外这个模块包含一些基础功能。")
        vals = valuable_sentences(reply)
        cats = {v["category"] for v in vals}
        assert CATEGORY_FACT in cats
        assert CATEGORY_DECISION in cats
        assert CATEGORY_PROCESS not in cats
        assert CATEGORY_PLEASANTRY not in cats


class TestExtractCore:
    def test_long_text_compressed_under_300(self):
        """长文本 → 核心 ≤300 字（稀疏化验证）。"""
        long_reply = "好的，我先看一下。\n" + "。".join(
            [f"实际上第{i}点很关键，正确的是这样做。" for i in range(80)])
        core = extract_core(long_reply, max_chars=300)
        assert 0 < len(core) <= 300

    def test_core_is_extractive_not_generative(self):
        """非生成式：核心只由原文句子拼接（集合运算不生成）。"""
        reply = ("我们决定采用方案A。实际上方案B有缺陷，正确的是先做验证。"
                 "我很期待这个结果。")
        core = extract_core(reply, max_chars=300)
        for sent in segment_sentences(core):
            assert sent in reply, f"核心出现原文不存在的句子: {sent}"

    def test_two_stage_picks_top_per_category(self):
        """两段式：4 类各取 top-1（H-8 定案）。"""
        reply = ("我们决定采用方案A。"
                 "实际上方案B有缺陷，正确的是先做验证，这个点非常关键。"
                 "我更喜欢稳扎稳打的节奏。"
                 "感谢你的耐心。"
                 "然后我们还需要处理一些基础设置。")
        core = extract_core(reply, max_chars=300)
        # 三类有价值句都进入核心（事实修正/决策/偏好）
        assert "决定采用方案A" in core
        assert "方案B有缺陷" in core
        assert "稳扎稳打" in core
        # 过程句不进入
        assert "基础设置" not in core

    def test_pleasantry_only_falls_back_to_prefix(self):
        """纯客套回复 → 回退原文前缀截断（不空）。"""
        core = extract_core("好的。没问题。谢谢！", max_chars=300)
        assert core != ""

    def test_empty_input(self):
        assert extract_core("") == ""
        assert extract_core(None) == ""

    def test_core_stats(self):
        core = extract_core("我们决定采用方案A。", max_chars=300)
        stats = core_stats(core)
        assert stats["core_chars"] == len(core)
        assert stats["core_sentence_count"] >= 1

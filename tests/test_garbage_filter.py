# -*- coding: utf-8 -*-
"""体验层 B（设计 v1.1 §4）：子代理样板污染过滤测试。

验收（§4.3）：
  1. 8 个样板案例 → 全部过滤（不落库）；
  2. 真实对话（"用户: 你好"、"命名 毛毛"）照常入库可召回（回归）；
  3. 现有 4 条垃圾过滤行为不变（回归）；
  4. store_episodic 入口命中即丢（fail-open）+ _GARBAGE_FILTERED 计数。
"""

import os
import sys

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
import torch

from core.hippocampus.memory import (
    MemoryManager, _is_garbage_text, _GARBAGE_TEXT_RE, _GARBAGE_FILTERED,
)


# 8 个样板案例（含本次会话实证变体：无 [Subagent Context] 前缀的
# [Subagent Task] 直接开头）
SUPPLANT_TEMPLATES = [
    # 1. 完整子代理派发样板（实证 7/8 案例主体）
    "[Subagent Context] You are running as a subagent (depth 1/1). "
    "Results auto-announce to your requester; do not busy-poll for status. "
    "Begin. Your assigned task is in the system prompt.",
    # 2. 无 [Subagent Context] 前缀的 [Subagent Task] 变体（实证 sandglass 条目）
    "[Subagent Task] — M1：Agent OS 本地模块功能审计（系统论角度）",
    # 3. [Subagent Task] 重复前缀变体（实证条目 4/5）
    "[Subagent Task] [Subagent Task] — 复核 Agent OS 全局设计文件是否完整落盘",
    # 4. 心跳回执
    "HEARTBEAT_OK",
    # 5. 子代理指令样板（截断变体）
    "You are running as a subagent (depth 1/1). Results auto-announce",
    # 6. 结果通知样板（截断变体）
    "Results auto-announce to your requester; do not busy-poll for status.",
    # 7. 任务派发样板（截断变体）
    "Your assigned task is in the system prompt under **Your Role**; "
    "execute it to completion.",
    # 8. [Subagent Context] 前缀单独出现（调度元数据）
    "[Subagent Context] This content was routed by OpenClaw from",
]

# 真实对话（不应被误杀）
NORMAL_TEXTS = [
    "用户: 你好",
    "命名 毛毛",
    "用户: 我喜欢蓝色的小橘猫\n助手: 我也喜欢，它很可爱",
    "用户: 方法论说遇到问题要先调研",
    "用户: 你还记得我们上次聊的体验层设计吗",
]


def _make_memory():
    return MemoryManager(num_nodes=32, episodic_capacity=200)


class TestSubagentGarbageFilter:
    """体验层 B：子代理样板 8 案例全部过滤。"""

    def test_eight_templates_all_garbage(self):
        for t in SUPPLANT_TEMPLATES:
            assert _is_garbage_text(t), f"样板未命中: {t[:60]}"

    def test_store_episodic_filters_templates(self):
        mm = _make_memory()
        vec = torch.zeros(16)
        for t in SUPPLANT_TEMPLATES:
            mm.store_episodic(t, vec, surprise=1.0, turn=0)
        assert mm.episodic_size() == 0, "样板文本不应落库"

    def test_normal_conversation_not_filtered(self):
        for t in NORMAL_TEXTS:
            assert not _is_garbage_text(t), f"真实对话被误杀: {t[:40]}"

    def test_normal_conversation_stored_and_recallable(self):
        mm = _make_memory()
        vec = torch.zeros(16)
        for t in NORMAL_TEXTS:
            mm.store_episodic(t, vec, surprise=1.0, turn=0)
        assert mm.episodic_size() == len(NORMAL_TEXTS)
        # 可被检索（回归：真实对话照常进入记忆流）
        scored = mm.recall_episodic_scored(
            torch.zeros(16), top_k=5, source_filter=None)
        assert len(scored) == len(NORMAL_TEXTS)


class TestLegacyGarbageUnchanged:
    """现有 4 条垃圾过滤行为不变（回归）。"""

    LEGACY = [
        "Sender (untrusted metadata) 12345",
        "System (untrusted) message",
        "System: 2026-08-11 端口探测",
        "端口探测 22/tcp 开放",
    ]

    def test_legacy_still_filtered(self):
        for t in self.LEGACY:
            assert _is_garbage_text(t), f"旧垃圾规则失效: {t[:40]}"

    def test_garbage_re_count(self):
        """正则条数 = 4 旧 + 6 新 + 2（v1.4 S1-8: [Inter-session message]/[梦醒]）
        = 12（纯增量，无删改）。"""
        assert len(_GARBAGE_TEXT_RE) == 12

    def test_garbage_filtered_counter(self):
        """_GARBAGE_FILTERED 计数可用（进程内，可被 status 读取）。"""
        assert isinstance(_GARBAGE_FILTERED, int)
        assert _GARBAGE_FILTERED >= 0

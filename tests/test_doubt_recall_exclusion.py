# -*- coding: utf-8 -*-
"""P0 污染处置（2026-08-17）：[doubt 系统事件检索排除 + 写侧守卫测试。

覆盖（方案「子AI任务-污染条目处置方案-20260817.md」路径 b 三处修复）：
  1. memory._recall_episodic_scored：source_filter='external' 时显式排除
     [doubt] conflict（source='external'，此前满权重进检索面）与
     [doubt-supersedes]（source='doubt'，此前被排除纯属来源过滤巧合）；
  2. archive.query_archive：归档路径 [doubt 行不参与检索（此前无任何过滤，
     合并检索中满权重可见）；
  3. loop.process_turn：写侧守卫——[doubt 系统事件不再作为普通对话入库
     （对齐 doubt_ingest docstring 已声明未执行的「系统事件不是对话」）；
  4. 不误伤回归：正文提及 [doubt 的真实对话照常入库可检索（锚定行首）。
"""

import base64
import os
import sys

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pytest
import torch

from core.hippocampus.memory import MemoryManager
from core.archive.archive_store import query_archive, rebuild_archive
from core.doubt.doubt_ingest import is_doubt_event

# ── 测试数据 ──────────────────────────────────────────────
# [doubt] conflict 事件（source=external 入库的污染条目——P0 事故残留）
DOUBT_CONFLICT_TEXTS = [
    "[doubt] conflict: dandan：你们都完全跑偏了，动用所有搜索手段",
    "[doubt] conflict: [Thu 2026-08-06 00:11 GMT+8] 开工吧，我还夜猫子",
]
# [doubt-supersedes] 证伪标记（source='doubt'，梦期 resolve_labile 改写产物）
SUPERSEDE_TEXTS = [
    "[doubt-supersedes] 原记忆被证伪: 你们都完全跑偏了 —— 原: dandan：完全跑偏",
    "[doubt-supersedes] 原记忆被证伪: 开工吧夜猫子 —— 原: [Thu 2026-08-06] 开工吧",
]
# 真实对话（不应被排除/误杀）
NORMAL_TEXTS = [
    "用户: 开工吧，我还夜猫子你还不知道么\n助手: 好的，我早日完成",
    "用户: 你们都完全跑偏了，重新搜索记忆\n助手: 收到，我重新核实",
    "正文提及 [doubt 机制设计的讨论（行首非 [doubt，锚定不误伤）",
]


def _make_memory() -> MemoryManager:
    return MemoryManager(num_nodes=32, episodic_capacity=200)


def _store_all(mm: MemoryManager):
    v = torch.ones(16)
    for t in NORMAL_TEXTS:
        mm.store_episodic(t, v, surprise=1.0, turn=1,
                          raw_semantic_vector=v, source='external')
    for t in DOUBT_CONFLICT_TEXTS:
        mm.store_episodic(t, v, surprise=1.0, turn=2,
                          raw_semantic_vector=v, source='external')
    for t in SUPERSEDE_TEXTS:
        mm.store_episodic(t, v, surprise=1.0, turn=3,
                          raw_semantic_vector=v, source='doubt')


# ============================================================
# 1. 内存路径：_recall_episodic_scored 排除 [doubt 系统事件
# ============================================================
class TestMemoryRecallExclusion:
    def test_external_recall_excludes_doubt_events(self):
        """source_filter='external'：真实对话全召回，[doubt 条目零出现。"""
        mm = _make_memory()
        _store_all(mm)
        scored = mm.recall_episodic_scored(
            torch.ones(16), top_k=20, source_filter='external',
            count_reference=False)
        texts = [e.text for _, e in scored]
        assert len(texts) == len(NORMAL_TEXTS)
        for t in NORMAL_TEXTS:
            assert t in texts, f"真实对话被排除: {t[:40]}"
        for t in DOUBT_CONFLICT_TEXTS + SUPERSEDE_TEXTS:
            assert t not in texts, f"[doubt 条目未排除: {t[:40]}"

    def test_unfiltered_recall_still_sees_doubt_entries(self):
        """source_filter=None（全来源，内部诊断）：[doubt 条目仍可见——
        只排除检索面，不抹除条目（只读语义不变）。"""
        mm = _make_memory()
        _store_all(mm)
        scored = mm.recall_episodic_scored(
            torch.ones(16), top_k=20, source_filter=None,
            count_reference=False)
        texts = [e.text for _, e in scored]
        assert len(texts) == len(NORMAL_TEXTS) + len(DOUBT_CONFLICT_TEXTS) \
            + len(SUPERSEDE_TEXTS)
        for t in DOUBT_CONFLICT_TEXTS + SUPERSEDE_TEXTS:
            assert t in texts

    def test_self_ref_filter_unchanged(self):
        """source_filter='self_ref' 语义不变（回归：Phase 2 来源过滤未受影响）。"""
        mm = _make_memory()
        _store_all(mm)
        # 无 self_ref 条目 → 空结果（[doubt 排除不改变 self_ref 面）
        scored = mm.recall_episodic_scored(
            torch.ones(16), top_k=20, source_filter='self_ref',
            count_reference=False)
        assert scored == []

    def test_doubt_entries_still_in_buffer(self):
        """排除只发生在检索面，缓冲区条目本身保留（备份/诊断可查）。"""
        mm = _make_memory()
        _store_all(mm)
        assert mm.episodic_size() == len(NORMAL_TEXTS) \
            + len(DOUBT_CONFLICT_TEXTS) + len(SUPERSEDE_TEXTS)


# ============================================================
# 2. 归档路径：query_archive 排除 [doubt 行
# ============================================================
class TestArchiveQueryExclusion:
    def _b64(self, vec):
        return base64.b64encode(
            np.asarray(vec, dtype=np.float32).tobytes()).decode()

    def _records(self):
        recs = []
        for i, t in enumerate(NORMAL_TEXTS):
            recs.append({
                "version": 1, "session_id": "main", "turn": 100 + i,
                "text": t, "source": "external", "surprise": 1.0,
                "text_hash": f"n{i}", "vector_b64": self._b64([1.0] * 8),
                "exported_at": 1.0, "origin": "archive",
            })
        for i, t in enumerate(DOUBT_CONFLICT_TEXTS + SUPERSEDE_TEXTS):
            recs.append({
                "version": 1, "session_id": "main", "turn": 200 + i,
                "text": t, "source": "external" if i < len(DOUBT_CONFLICT_TEXTS) else "doubt",
                "surprise": 1.0, "text_hash": f"d{i}",
                "vector_b64": self._b64([1.0] * 8),
                "exported_at": 1.0, "origin": "archive",
            })
        return recs

    def test_archive_query_excludes_doubt_lines(self, tmp_path):
        """归档检索：真实对话可召回，[doubt 行零出现（含正文提及不误伤）。"""
        rebuild_archive("main", self._records(), archive_dir=str(tmp_path))
        res = query_archive(
            "main", np.ones(8, dtype=np.float32), k=20,
            archive_dir=str(tmp_path))
        texts = [r["text"] for r in res]
        assert len(res) == len(NORMAL_TEXTS)
        for t in NORMAL_TEXTS:
            assert t in texts, f"归档真实对话被排除: {t[:40]}"
        for t in DOUBT_CONFLICT_TEXTS + SUPERSEDE_TEXTS:
            assert t not in texts, f"归档 [doubt 行未排除: {t[:40]}"


# ============================================================
# 3. 写侧守卫：loop.process_turn 不存 [doubt 系统事件
# ============================================================
class _StubEmbedder:
    """带 embed_text / embed_text_raw 的桩嵌入器（触发写侧语义向量路径）。"""
    dim = 16

    def embed_text(self, text: str) -> torch.Tensor:
        return torch.ones(self.dim)

    def embed_text_raw(self, text: str) -> torch.Tensor:
        return torch.ones(16)


def _make_loop():
    from runtime.loop import LivingMemoryLoop
    config = {
        "num_nodes": 32, "input_dim": 16, "num_infer_steps": 5,
        "consolidation_interval": 3, "seed": 42,
        "auto_snapshot": False,
        "embedder": _StubEmbedder(),
    }
    return LivingMemoryLoop(config)


class TestProcessTurnStoreGuard:
    def test_doubt_event_not_stored(self):
        """[doubt 系统事件走结构化摄入，不再作为普通对话入库。"""
        loop = _make_loop()
        loop.process_turn(
            "[doubt] conflict: dandan：你们都完全跑偏了", llm_output="")
        assert loop.memory.episodic_size() == 0, \
            "[doubt 系统事件不应写入 episodic 缓冲区"

    def test_normal_text_stored(self):
        """回归：真实对话照常入库。"""
        loop = _make_loop()
        loop.process_turn("开工吧，我还夜猫子你还不知道么", llm_output="")
        assert loop.memory.episodic_size() == 1

    def test_is_doubt_event_helper(self):
        """判定助手：仅 [doubt 前缀（锚定行首）命中，正文提及不误伤。"""
        assert is_doubt_event("[doubt] conflict: xxx")
        assert is_doubt_event("[doubt] fok: 未决问题")
        assert not is_doubt_event("用户: [doubt] 是什么意思？")
        assert not is_doubt_event("正文讨论 [doubt 机制")
        assert not is_doubt_event("")

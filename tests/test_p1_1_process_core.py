# -*- coding: utf-8 -*-
"""P1-1 单测：存储过程化——条目主结构改"过程核心"（结论降为派生视图）

规格依据：`四妹-更新版目的性审计-20260818.md` §三（P1-1 裁决全文，§3.2
可执行方案 / §3.3 边界）；`四妹-LMS核心重写规格v2-20260817.md` §1.5。

覆盖（任务书 + claims.json 机器验证，§5.2 D 模式）：
  1. 新条目含 process_core / text_snapshot / evolution（§3.2 schema）
     → TestNewEntryFields（含 ingest 写路径附加 + 直接附加两路）；
  2. 旧条目（M7 迁移，无新字段）getattr 兜底可读：空结构 + text_snapshot
     回退 text + 迁移数据读取不崩溃 → TestOldEntryCompat（含快照往返）；
  3. 提取过程核心优先：含惊讶/怀疑/转向/悬案痕迹的文本 → process_core 捕获、
     text_snapshot 派生（非本体）→ TestExtractProcessCorePriority；
  4. evolution append-only 不覆盖（可回放可审计）→ TestEvolutionAppendOnly；
  5. surprise_source 链接 thought id（存在即有字段，不强制）→
     TestSurpriseSourceLinks；
  6. 既有 store 行为回归不破坏（幂等/语义分离/怀疑钩子 fail-open/幂等记录
     不被过程字段污染/entry 弹出不进 journal）→
     TestExistingStoreBehaviorRegression + TestFailOpen。

运行方式：pytest rewrite-ws/tests/test_p1_1_process_core.py -q
"""

import json
import os
import sys
import tempfile
from types import SimpleNamespace

# -- 路径引导：本文件可从任意 cwd 运行 ----------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest  # noqa: E402
import torch  # noqa: E402

from core.store import (  # noqa: E402
    IdempotencyRegistry,
    PROCESS_CORE_FIELDS,
    SemanticViolation,
    WriteRequest,
    WriteSemantics,
    append_transition,
    attach_entry_fields,
    build_text_snapshot,
    empty_process_core,
    extract_process_core,
    get_evolution,
    get_process_core,
    get_text_snapshot,
    has_process_core,
    init_evolution,
    make_transition,
    ingest,
)
from core.hippocampus.memory import MemoryManager  # noqa: E402

# --------------------------------------------------------------------------- #
#  小工具
# --------------------------------------------------------------------------- #


def make_old_style_entry(text="旧条目", turn=5, vec_dim=8):
    """模拟旧快照条目（M7 迁移：无 process_core/text_snapshot/evolution）。"""
    from core.hippocampus.memory import EpisodicEntry
    e = EpisodicEntry.__new__(EpisodicEntry)
    e.text = text
    e.semantic_vector = torch.zeros(vec_dim)
    e.surprise = 0.5
    e.turn = turn
    e.source = 'external'
    e.confidence = 1.0
    e.rebuttal_count = 0
    e.reference_count = 0
    e.source_trust = 1.0
    e.labile = False
    e.labile_since = None
    e.violated_by = None
    e.last_recalled_at = None
    e.recall_count = 0
    return e


class _Entry:
    """极简条目替身（无 torch 依赖；setattr 附加字段）。"""

    def __init__(self, text="", confidence=1.0, source="external"):
        self.text = text
        self.confidence = confidence
        self.source = source
        self.gray = False
        self.doubt_state = "stable"


def _writer_returning_entry(entry, stored=True):
    """模拟 api 层 writer：落条目后把新条目放进结果（待集成点形态）。"""
    def writer(req):
        return {"stored": stored, "turn_count": 1, "entry": entry}
    return writer


# --------------------------------------------------------------------------- #
#  ① 新条目含 process_core / text_snapshot / evolution
# --------------------------------------------------------------------------- #


class TestNewEntryFields:
    """§3.2：新条目主结构 = 过程核心；text_snapshot 派生；evolution 演化史。"""

    def test_ingest_attaches_process_fields_to_entry(self):
        """写路径（ingest + writer 报告 entry）→ 新条目携带三新字段。"""
        entry = _Entry(text="我没想到方案A是错的，需要修正。")
        writer = _writer_returning_entry(entry)
        req = WriteRequest(
            semantics=WriteSemantics.STORE, text="我没想到方案A是错的，需要修正。",
            llm_output="我推翻了之前的看法。", metadata={"surprise": 0.9},
            idempotency_key="k-new-entry")
        res = ingest(req, writer, registry=IdempotencyRegistry())
        assert res.stored is True
        # 条目已附加过程字段（strict 增量：既有字段保持）
        assert has_process_core(entry) is True
        assert entry.text == "我没想到方案A是错的，需要修正。"  # 既有字段不动
        assert entry.doubt_state == "stable"                   # 既有字段不动
        pc = get_process_core(entry)
        assert sorted(pc.keys()) == sorted(PROCESS_CORE_FIELDS)
        assert all(isinstance(pc[f], list) for f in PROCESS_CORE_FIELDS)
        # 惊讶/转向痕迹被捕获（提取过程核心优先）
        assert any("惊讶" in t or "surprise" in t.lower()
                   for t in pc["surprise_trace"])
        assert len(pc["turns"]) >= 1
        # text_snapshot 派生视图（非空，可重建）
        assert isinstance(get_text_snapshot(entry), str)
        assert get_text_snapshot(entry) != ""
        # evolution：append-only 演化史，以 created 为起点，写者 ingest
        ev = get_evolution(entry)
        assert isinstance(ev["history"], list)
        assert ev["history"][0]["state"] == "created"
        assert ev["updated_by"] == "ingest"

    def test_write_result_carries_process_fields(self):
        """WriteResult 携带 process_core/text_snapshot/evolution（观测）。"""
        req = WriteRequest(semantics=WriteSemantics.STORE, text="hello",
                           idempotency_key="k-res")

        def writer(r):
            return {"stored": True, "turn_count": 1}

        res = ingest(req, writer, registry=IdempotencyRegistry())
        assert isinstance(res.process_core, dict)
        assert sorted(res.process_core.keys()) == sorted(PROCESS_CORE_FIELDS)
        assert isinstance(res.text_snapshot, str)
        assert isinstance(res.evolution, dict)
        assert res.evolution["history"][0]["state"] == "created"
        # 无信号文本 → process_core 全空 + 快照回退 text（不臆造）
        assert res.process_core == empty_process_core()
        assert res.text_snapshot == "hello"

    def test_attach_entry_fields_direct_incremental(self):
        """直接附加（process_core.py 助手）：严格增量，既有字段不被触碰。"""
        entry = _Entry(text="原始文本", confidence=0.9, source="external")
        pc = extract_process_core(text="这里有个惊讶：发现错了，需要修正。",
                                  surprise=0.8, turn=7)
        attach_entry_fields(entry, process_core=pc)
        assert entry.text == "原始文本"
        assert entry.confidence == 0.9
        assert entry.source == "external"
        assert entry.gray is False
        assert entry.doubt_state == "stable"
        assert get_process_core(entry)["surprise_trace"]
        # 快照由过程核心派生（非原始文本）
        assert get_text_snapshot(entry) != "原始文本"

    def test_dict_entry_supported(self):
        """dict 形态条目同样支持（persistence 反序列化兜底）。"""
        entry = {"text": "旧文本"}
        attach_entry_fields(entry, process_core=extract_process_core(
            text="我改变了主意，认为B更好。"))
        assert "process_core" in entry and "text_snapshot" in entry
        assert "evolution" in entry
        assert entry["text"] == "旧文本"
        assert get_process_core(entry)["turns"]
        assert get_text_snapshot(entry) != ""


# --------------------------------------------------------------------------- #
#  ② M7 旧条目 getattr 兜底可读（迁移数据读取不崩溃）
# --------------------------------------------------------------------------- #


class TestOldEntryCompat:
    """旧条目（无新字段）→ getattr 默认值：空结构 + text_snapshot 回退 text。"""

    def test_old_entry_readable_via_getattr_defaults(self):
        e = make_old_style_entry(text="旧条目文本", turn=9)
        # 无字段 → 不崩溃，空结构
        assert has_process_core(e) is False
        assert get_process_core(e) == empty_process_core()
        # text_snapshot 回退旧 text 字段（M7 向后兼容映射）
        assert get_text_snapshot(e) == "旧条目文本"
        # evolution 回退空演化史（updated_by 默认 ingest）
        ev = get_evolution(e)
        assert ev == {"history": [], "updated_by": "ingest"}
        # 旧条目首次演化写入不崩溃（自动初始化演化史）
        append_transition(e, make_transition("suspect", detail="M3 注入时怀疑"))
        ev2 = get_evolution(e)
        assert len(ev2["history"]) == 1
        assert ev2["history"][0]["state"] == "suspect"
        assert ev2["updated_by"] == "ingest"

    def test_old_entry_text_absent_fallback_empty(self):
        """text 也缺失的畸形条目 → 快照空串，不抛异常。"""
        e = SimpleNamespace(turn=1)
        assert get_text_snapshot(e) == ""
        assert get_process_core(e) == empty_process_core()

    def test_old_and_new_entries_snapshot_roundtrip(self):
        """MemoryManager 快照往返：混合新旧条目不崩溃，新字段保持。"""
        m = MemoryManager(num_nodes=16)
        old = make_old_style_entry(text="旧条目无过程字段", turn=1)
        m._episodic_buffer.append(old)
        m.store_episodic("新条目", torch.randn(8), surprise=1.0, turn=2)
        new = list(m.iter_episodic())[-1]
        attach_entry_fields(new, process_core=extract_process_core(
            text="新条目带来惊讶", surprise=1.2, turn=2))
        state = m.get_state()

        m2 = MemoryManager(num_nodes=16)
        m2.set_state(state)
        by_turn = {e.turn: e for e in m2.iter_episodic()}
        # 旧条目：getattr 兜底（迁移数据读取不崩溃）
        assert get_process_core(by_turn[1]) == empty_process_core()
        assert get_text_snapshot(by_turn[1]) == "旧条目无过程字段"
        # 新条目：字段保留
        assert has_process_core(by_turn[2]) is True
        assert len(get_process_core(by_turn[2])["surprise_trace"]) >= 1
        assert get_text_snapshot(by_turn[2]) != ""


# --------------------------------------------------------------------------- #
#  ③ 提取过程核心优先（惊讶来源 / 怀疑轨迹 / 转向 / 悬案 → process_core）
# --------------------------------------------------------------------------- #


class TestExtractProcessCorePriority:
    """§3.2：提取"过程核心优先"，text_snapshot 降级为派生投影。"""

    def test_extraction_captures_process_traces(self):
        text = ("我之前以为方案A是对的，现在很惊讶发现它错了，需要修正为方案B。"
                "我怀疑这个结论，这里还有个悬案没解决。")
        llm_output = "我推翻了之前的看法，现在认为方案B才对。"
        pc = extract_process_core(text=text, llm_output=llm_output,
                                  surprise=0.9, confidence=0.65, turn=42)
        # 惊讶来源：输入→惊讶值→修正动作
        assert pc["surprise_trace"], "惊讶来源必须被捕获"
        joined = " ".join(pc["surprise_trace"])
        assert "惊讶值=0.90" in joined
        assert "修正:有" in joined
        assert "@turn=42" in joined
        # 怀疑轨迹
        assert any("suspect" in d for d in pc["doubt_events"])
        # 转向
        assert any("转向" in t for t in pc["turns"])
        # 悬案尾巴
        assert any("悬案" in t for t in pc["open_tails"])
        # 置信度曲线（单点起点）
        assert pc["confidence_curve"] == [0.65]

    def test_text_snapshot_derived_from_process_core(self):
        text = "我很惊讶之前的结论是错的，需要修正。这是个悬案。"
        pc = extract_process_core(text=text, surprise=0.7, turn=3)
        snap = build_text_snapshot(pc, text=text, core="")
        # 派生视图：来自过程核心（悬案/惊讶轨迹），不是原文逐字
        assert "悬案" in snap
        assert "惊讶" in snap
        assert snap != text
        # 可压缩、≤300 字（投影非本体）
        assert len(snap) <= 300

    def test_snapshot_evolves_with_process_core(self):
        """快照随演化更新：过程核心追加轨迹 → 新快照重建（不是定格文本）。"""
        pc = extract_process_core(text="我改变了主意，认为B更好。", turn=1)
        snap1 = build_text_snapshot(pc)
        pc["open_tails"].append("悬案: 待考证的遗留问题")
        snap2 = build_text_snapshot(pc)
        assert snap2 != snap1
        assert "待考证" in snap2

    def test_no_signal_input_empty_core_and_fallback(self):
        """无过程信号 → 空结构 + 快照回退 core/text（不臆造）。"""
        pc = extract_process_core(text="普通的记录文本。", llm_output="")
        assert pc == empty_process_core()
        assert build_text_snapshot(pc, text="原文", core="压缩核心") == "压缩核心"
        assert build_text_snapshot(pc, text="原文", core="") == "原文"

    def test_metadata_confidence_curve(self):
        pc = extract_process_core(
            text="惊讶：结论被推翻", metadata={
                "confidence_curve": [0.72, 0.65, 0.81]})
        assert pc["confidence_curve"] == [0.72, 0.65, 0.81]


# --------------------------------------------------------------------------- #
#  ④ evolution append-only（不覆盖历史，可回放可审计）
# --------------------------------------------------------------------------- #


class TestEvolutionAppendOnly:
    """§3.2/§3.3：evolution.history 为 append-only 状态转移。"""

    def test_history_grows_without_overwrite(self):
        entry = _Entry(text="记忆")
        ev = init_evolution(updated_by="ingest")
        from core.store.process_core import record_transition
        record_transition(ev, make_transition("created", detail="ingest"))
        attach_entry_fields(entry, evolution=ev)
        # 追加两条状态转移
        append_transition(entry, make_transition("suspect", detail="注入时怀疑"))
        append_transition(entry, make_transition(
            "confirm", detail="验证链通过"))
        hist = get_evolution(entry)["history"]
        assert [h["state"] for h in hist] == ["created", "suspect", "confirm"]
        # 前两条记录原样保留（append-only：不覆盖历史）
        assert hist[0]["state"] == "created"
        assert hist[1]["detail"] == "注入时怀疑"

    def test_updated_by_transition_and_invalid_writer(self):
        entry = _Entry(text="记忆")
        attach_entry_fields(entry, evolution=init_evolution("ingest"))
        append_transition(entry, make_transition("suspect"),
                          updated_by="consolidation")
        assert get_evolution(entry)["updated_by"] == "consolidation"
        # 非法写者（retrieval——只读铁律同款）被拒绝，保持原写者
        append_transition(entry, make_transition("reconsolidated"),
                          updated_by="retrieval")
        assert get_evolution(entry)["updated_by"] == "consolidation"

    def test_evolution_never_shrinks(self):
        """演化史只增不减（回放/审计完整性）。"""
        entry = _Entry(text="记忆")
        attach_entry_fields(entry)
        n0 = len(get_evolution(entry)["history"])
        for i in range(5):
            append_transition(entry, make_transition("state-%d" % i))
        hist = get_evolution(entry)["history"]
        assert len(hist) == n0 + 5
        assert hist[-1]["state"] == "state-4"


# --------------------------------------------------------------------------- #
#  ⑤ surprise_source 链接 thought id（对话侧与思考侧同构）
# --------------------------------------------------------------------------- #


class TestSurpriseSourceLinks:
    """§3.2：surprise_source 支持链接 thought id——存在即有字段，不强制。"""

    def test_metadata_links_normalized(self):
        pc = extract_process_core(
            text="惊讶内容", metadata={"surprise_source": ["th-1", "thought:th-2"]})
        assert pc["surprise_source"] == ["thought:th-1", "thought:th-2"]

    def test_text_pattern_links(self):
        pc = extract_process_core(
            text="这个惊讶来自 thought:th-99 的怀疑，还有 thought_88 也相关。")
        assert "thought:th-99" in pc["surprise_source"]
        assert "thought:88" in pc["surprise_source"]

    def test_no_link_empty_list(self):
        pc = extract_process_core(text="普通文本没有链接。")
        assert pc["surprise_source"] == []  # 存在即有字段，不强制


# --------------------------------------------------------------------------- #
#  ⑥ 既有 store 行为回归不破坏（幂等 / 语义 / 钩子 / 记录不被污染）
# --------------------------------------------------------------------------- #


class TestExistingStoreBehaviorRegression:
    """P1-1 严格增量：既有写侧行为零回归（claims 机器验证）。"""

    def test_existing_store_semantics_untouched(self):
        """语义分离/幂等/钩子核心断言回归（与 test_store_m1 同口径）。"""
        registry = IdempotencyRegistry(window=60.0)
        calls = []

        def writer(req):
            calls.append(req)
            return {"stored": True, "turn_count": 1}

        req1 = WriteRequest(semantics=WriteSemantics.STORE, text="hello",
                            idempotency_key="k-reg")
        req2 = WriteRequest(semantics=WriteSemantics.STORE, text="hello",
                            idempotency_key="k-reg")
        res1 = ingest(req1, writer, registry=registry)
        res2 = ingest(req2, writer, registry=registry)
        assert len(calls) == 1, "同键重发必须不双写（幂等回归）"
        assert res1.stored is True and res2.dedup_hit is True
        assert res2.data == res1.data, "dedup 原样返回首次结果"
        # 语义分离回归：feed 灰化拒绝 / store_gray 不可召回
        with pytest.raises(SemanticViolation):
            WriteRequest(semantics=WriteSemantics.FEED, text="x",
                         gray=True).resolve()
        from core.store import recallable
        assert recallable("store_gray") is False

    def test_entry_popped_not_in_record_or_response(self):
        """writer 报告 entry → 弹出：不进幂等记录/响应（JSON 安全）。"""
        registry = IdempotencyRegistry(window=60.0)
        entry = _Entry(text="x")
        res = ingest(
            WriteRequest(semantics=WriteSemantics.STORE, text="x",
                         idempotency_key="k-pop"),
            _writer_returning_entry(entry), registry=registry)
        assert res.stored is True
        assert "entry" not in res.data, "entry 不得进响应"
        rec = registry.lookup("k-pop")
        assert rec is not None
        assert "entry" not in rec.result, "entry 不得进幂等记录"
        assert has_process_core(entry) is True, "字段仍附加到条目"

    def test_journal_line_json_safe_with_entry(self):
        """journal 配置下 writer 报告 entry → 记录行仍可解析（不炸序列化）。"""
        with tempfile.TemporaryDirectory() as td:
            journal = os.path.join(td, "idem.jsonl")
            registry = IdempotencyRegistry(window=60.0, journal_path=journal)
            entry = _Entry(text="x")
            res = ingest(
                WriteRequest(semantics=WriteSemantics.STORE, text="x",
                             idempotency_key="k-j"),
                _writer_returning_entry(entry), registry=registry)
            assert res.stored is True
            assert has_process_core(entry) is True
            with open(journal, encoding="utf-8") as f:
                obj = json.loads(f.readline())
            assert "entry" not in obj["result"], "journal 不得含 entry 对象"
            assert obj["result"]["stored"] is True

    def test_doubt_hook_fail_open_regression(self):
        """注入时怀疑钩子异常 fail-open 回归：不阻断写侧。"""
        import importlib
        mod = importlib.import_module("core.store.ingest")
        with mod._doubt_hooks_lock:
            saved = list(mod._doubt_hooks)
            mod._doubt_hooks.clear()
        try:
            def bad_hook(req, res):
                raise RuntimeError("怀疑逻辑故障")
            mod.register_doubt_hook(bad_hook)
            registry = IdempotencyRegistry(window=60.0)
            calls = []

            def writer(req):
                calls.append(req)
                return {"stored": True, "turn_count": 1}

            res = ingest(WriteRequest(
                semantics=WriteSemantics.STORE, text="x",
                idempotency_key="k-hook"), writer, registry=registry)
            assert res.stored is True, "钩子异常 fail-open：写侧不被阻断"
        finally:
            with mod._doubt_hooks_lock:
                mod._doubt_hooks[:] = saved


# --------------------------------------------------------------------------- #
#  fail-open：提取/附加异常绝不阻断写侧
# --------------------------------------------------------------------------- #


class TestFailOpen:
    """P1-1 防线：过程核心任何异常 → 空结构/文本回退，写侧照常。"""

    def test_extraction_failure_never_blocks_write(self, monkeypatch):
        import importlib
        ingest_mod = importlib.import_module("core.store.ingest")

        def boom(req):
            raise RuntimeError("提取器故障")

        monkeypatch.setattr(ingest_mod, "extract_process_core_for_request", boom)
        entry = _Entry(text="x")
        registry = IdempotencyRegistry(window=60.0)
        res = ingest(
            WriteRequest(semantics=WriteSemantics.STORE, text="x",
                         idempotency_key="k-fo"),
            _writer_returning_entry(entry), registry=registry)
        assert res.stored is True, "提取异常 fail-open：写侧不被阻断"
        assert res.process_core == empty_process_core()
        assert res.text_snapshot == "x"  # 文本回退
        assert has_process_core(entry) is True, "条目仍附加（空结构）"
        assert get_text_snapshot(entry) == "x"

    def test_attach_failure_never_blocks_write(self, monkeypatch):
        """附加异常 fail-open：条目已写（缺过程字段仅告警），写侧不阻断。"""
        import core.store.ingest as ingest_mod

        class StubbornEntry(_Entry):
            def __setattr__(self, name, value):
                if name in ("process_core", "text_snapshot", "evolution"):
                    raise RuntimeError("只读条目拒绝附加")
                super().__setattr__(name, value)

        entry = StubbornEntry(text="x")
        registry = IdempotencyRegistry(window=60.0)
        res = ingest(
            WriteRequest(semantics=WriteSemantics.STORE, text="x",
                         idempotency_key="k-fo2"),
            _writer_returning_entry(entry), registry=registry)
        assert res.stored is True, "附加异常 fail-open：写侧不被阻断"
        assert res.process_core == empty_process_core()

# -*- coding: utf-8 -*-
"""core/store · 写侧提取层（核心重建 M1 第一段）

规格依据：`四妹-LMS核心重写规格v2-20260817.md` §1.5 / §2.2。
重写总纲：**核心重建，血管不换**——本包是写侧唯一入口（ingest/feed/store
语义分离 + 幂等键机制），api 层端点签名与返回结构不变，只换内部实现。

包结构（M1 第一段交付 + P1-1 过程核心）：
  - ``semantics.py``  写语义定义：feed（知识补存双通道·可召回）/
                      store（普通存储·含 store_gray 灰度口径）——语义唯一权威；
  - ``idempotency.py`` 幂等键机制：查重 → 写 → 登记（写成功才登记）、
                      60s 竞态防护、同键并发串行化、journal 窗口内重启重放；
  - ``ingest.py``      写侧统一入口：ingest()/feed()/store() 分派 +
                      注入时怀疑钩子（M3 接入点）+ 写侧默认保守 +
                      P1-1 提取过程核心优先（§3.2）；
  - ``process_core.py`` P1-1 过程核心提取层：条目新字段
                      （process_core / text_snapshot / evolution）的提取、
                      派生、append-only 演化史与 getattr 兜底读取
                      （规格 `四妹-更新版目的性审计-20260818.md` §三）。

依赖注入设计：本包**不 import 任何 LMS 运行时模块**（纯 stdlib）——
可被 python 直接 import/运行；落库由 api 层注入的 writer 回调完成
（内部走 loop.process_turn，血管不换）。

注意（核对结论，2026-08-17）：``ingest`` 既是子模块名又是入口函数名——
包级属性 ``core.store.ingest`` 被**入口函数**遮蔽（by design，调用方要的是
函数）。需要子模块成员时用 ``importlib.import_module("core.store.ingest")``
或 ``from core.store.ingest import ...``（importlib 经 sys.modules，不受遮蔽
影响）。本包 ``__all__`` 已覆盖三个子模块的全部公开成员（含 Writer/
DoubtHook/store_gray_default/validate_semantics），包级 import 后无导出缺口。

M1 后续段落（不在本段范围）：注入时怀疑逻辑（M3 接入钩子）、
写侧保守标记 suspect 进验证链、与 api/server.py 端点接线（M8）。
"""

from __future__ import annotations

#: 模块语义 claim 登记（§5.2：machine-readable；实现与 claims.json 同源）
MODULE_CLAIMS: dict = {
    "module": "core/store",
    "milestone": "M1-1",
    "claims": {
        "idempotency": {
            "statement": "写侧所有入口（ingest/feed/store）幂等：同键重发不双写，"
                         "已处理键原样返回原结果",
            "verified_by": "tests/test_store_m1.py::"
                           "test_idempotent_key_resend_no_double_write",
        },
        "register_after_success": {
            "statement": "写成功才登记幂等键；writer 失败不登记，重试同键可重写"
                         "（修复旧实现失败也占用窗口的假 dedup_hit）",
            "verified_by": "tests/test_store_m1.py::"
                           "test_failed_write_not_registered_retry_allowed",
        },
        "concurrent_same_key_serialized": {
            "statement": "同键并发（首写与超时重试同时在途）由同键锁串行化，"
                         "全程只写一次",
            "verified_by": "tests/test_store_m1.py::"
                           "test_timeout_retry_race_single_write",
        },
        "feed_store_separation": {
            "statement": "feed 永不灰化且 source 必须可召回；store 灰度走 "
                         "store_gray（L1 不可召回——口径登记，非 bug）",
            "verified_by": "tests/test_store_m1.py::"
                           "test_feed_store_semantic_separation",
        },
        "dedup_hit_no_side_effect": {
            "statement": "幂等命中路径零写入、零状态变更（无副作用）",
            "verified_by": "tests/test_store_m1.py::"
                           "test_dedup_hit_returns_original_result",
        },
        "doubt_hook_fail_open": {
            "statement": "注入时怀疑钩子异常 fail-open，绝不阻断写侧",
            "verified_by": "tests/test_store_m1.py::"
                           "test_doubt_hook_fail_open",
        },
        "doubt_hook_labile_not_persisted": {
            "statement": "注入时怀疑信号是 labile 内存态，绝不落库：幂等记录/"
                         "写结果/响应不被怀疑标记污染（§2.2 只读防线同款）",
            "verified_by": "tests/test_store_m1.py::"
                           "test_doubt_hook_labile_not_persisted",
        },
        "window_is_race_guard_not_permanent_dedup": {
            "statement": "60s 窗口是竞态防护不是永久去重；窗口外同键视为新写"
                         "（旧结果自然过期）",
            "verified_by": "tests/test_store_m1.py::"
                           "test_window_expiry_allows_new_write",
        },
        "journal_replay_survives_restart": {
            "statement": "journal 配置时窗口内重启后重放：重启后同键重试不双写"
                         "（写侧默认保守）",
            "verified_by": "tests/test_store_m1.py::"
                           "test_journal_replay_after_restart",
        },
        # -- P1-1 过程核心（目的性审计 §三：存储主体是过程，结论是派生视图）--
        "process_core_extraction_priority": {
            "statement": "/store 提取层'提取过程核心优先'：惊讶来源/怀疑轨迹/"
                         "转向/悬案/置信度曲线/thought 链接进 process_core，"
                         "text_snapshot 为派生视图（非本体）",
            "verified_by": "tests/test_p1_1_process_core.py::"
                           "TestExtractProcessCorePriority::"
                           "test_extraction_captures_process_traces",
        },
        "old_entry_getattr_compat": {
            "statement": "M7 旧条目（无 process_core/text_snapshot/evolution）"
                         "必须仍可读：新字段访问走 getattr 默认值——空结构 + "
                         "text_snapshot 回退 text，迁移数据读取不崩溃",
            "verified_by": "tests/test_p1_1_process_core.py::"
                           "TestOldEntryCompat::"
                           "test_old_entry_readable_via_getattr_defaults",
        },
        "evolution_append_only": {
            "statement": "evolution.history 为 append-only 状态转移：只追加，"
                         "绝不覆盖历史（每条记忆演化史可回放可审计）",
            "verified_by": "tests/test_p1_1_process_core.py::"
                           "TestEvolutionAppendOnly::"
                           "test_history_grows_without_overwrite",
        },
        "process_core_fail_open": {
            "statement": "过程核心提取/解析/附加全程 fail-open：异常绝不阻断"
                         "写侧（空过程核心 + 文本回退兜底）",
            "verified_by": "tests/test_p1_1_process_core.py::"
                           "TestFailOpen::test_extraction_failure_never_blocks_write",
        },
        "strict_incremental_no_breakage": {
            "statement": "P1-1 严格增量：不删除/不破坏既有字段（text/confidence/"
                         "source/gray/doubt_state 等写侧既有语义保持）；幂等记录/"
                         "响应不被过程字段污染（entry 弹出，不进幂等记录）",
            "verified_by": "tests/test_p1_1_process_core.py::"
                           "TestExistingStoreBehaviorRegression::"
                           "test_existing_store_semantics_untouched",
        },
    },
}

from .semantics import (  # noqa: E402
    NON_RECALLABLE_SOURCES,
    FeedChannel,
    SemanticViolation,
    WriteSemantics,
    recallable,
    resolve_source,
    store_gray_default,
    validate_semantics,
)
from .idempotency import (  # noqa: E402
    IdempotencyRecord,
    IdempotencyRegistry,
    fingerprint_key,
    generate_idempotency_key,
    normalize_key,
)
from .process_core import (  # noqa: E402
    DEFAULT_SNAPSHOT_MAX_CHARS,
    DEFAULT_TRACE_MAX_CHARS,
    EVOLUTION_WRITERS,
    PROCESS_CORE_FIELDS,
    append_transition,
    attach_entry_fields,
    build_text_snapshot,
    empty_process_core,
    extract_process_core,
    extract_process_core_for_request,
    get_evolution,
    get_process_core,
    get_text_snapshot,
    has_process_core,
    init_evolution,
    make_transition,
    record_transition,
)
from .ingest import (  # noqa: E402
    DoubtHook,
    Writer,
    WriteRequest,
    WriteResult,
    feed,
    get_default_registry,
    ingest,
    register_doubt_hook,
    store,
)

__all__ = [
    "MODULE_CLAIMS",
    "NON_RECALLABLE_SOURCES",
    "FeedChannel",
    "SemanticViolation",
    "WriteSemantics",
    "recallable",
    "resolve_source",
    "store_gray_default",
    "validate_semantics",
    "IdempotencyRecord",
    "IdempotencyRegistry",
    "fingerprint_key",
    "generate_idempotency_key",
    "normalize_key",
    "DEFAULT_SNAPSHOT_MAX_CHARS",
    "DEFAULT_TRACE_MAX_CHARS",
    "EVOLUTION_WRITERS",
    "PROCESS_CORE_FIELDS",
    "append_transition",
    "attach_entry_fields",
    "build_text_snapshot",
    "empty_process_core",
    "extract_process_core",
    "extract_process_core_for_request",
    "get_evolution",
    "get_process_core",
    "get_text_snapshot",
    "has_process_core",
    "init_evolution",
    "make_transition",
    "record_transition",
    "DoubtHook",
    "Writer",
    "WriteRequest",
    "WriteResult",
    "feed",
    "get_default_registry",
    "ingest",
    "register_doubt_hook",
    "store",
]

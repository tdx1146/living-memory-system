"""怀疑融入（体验层 D，设计 v1.1 §6）—— 专注方向版 + 核心重建 M3-1 原生怀疑。

怀疑 = 专注的质检员（dandan 拍板 2）：怀疑修正"已关注方向"内记忆的
信任权重（置信度），不改变 precision 的方向（专注）。所有机制对照
P1 原设计做"专注化审查"，两处修订（gap C 类降级 + 配额 relevance-gated），
其余按 P1 §2.1-2.5 落地。

M3-1（核心重建规格 v2 §二，自我怀疑原生机制）新增：
  - state_machine:        三时相怀疑状态机（检索时 labile 窗口投影 /
                          注入时 suspect / 巩固时 resolve_labile 接口）
  - rebuttal_field:       rebuttal-consistency 字段原生（§2.3 存取接口 +
                          updated_by 写者守卫铁律）
  - verification_chain:  验证链原生骨架（§2.2 接口/事件类型/幂等键协议/
                         防伪独立四维脚手架）

M3-2（同一规格 §2.2 判定细节）：
  - verification_chain:  矛盾判定三选一（方向/数值/否定）+ 元数据排除
                         （时间戳碰撞/同义复述/来源互补——P0 假冲突根因
                         ①②根治）+ 验证链全链（草稿→独立验证→修正：
                         verify/run_pending + VERIFY-* provenance +
                         CONFLICT 结果写 [doubt] conflict → labile）

目的检查时相（总任务书 §二.5，dandan 16:57 指示——目的时相与质疑系统
融合）：
  - purpose_drift:  每轮 [doubt] purpose-drift 判定（消费 dsh-goal 目的
                    + 目的手册五问 Q1-Q5）——输出「是否偏离+理由」闸门
                    信号，不输出可被优化的分数（Pan 警示）；经
                    DoubtStateMachine.purpose_drift_check 委托接入

模块（既有）：
  - confidence_field: 置信度场纯函数（更新/重算/强制降权/来源信任）
  - reconsolidation:  惊讶度双角色-角色2（去稳定化，labile 标记）
  - recall_scheduler: 召回唤起 salience（贝叶斯三项 + 低置信配额门控）
  - doubt_ingest:     /feed 结构化怀疑摄入（fail-open）
  - gap_registry:     信息缺口登记（A/B 类进怀疑灯，C 类仅诊断）

红线：surprise/free_energy/per_dim 语义、attractor infer/learn、
precision 三层结构、G1 归一化、doubt-system 四脚本 —— 全部零触碰。
"""

from __future__ import annotations

#: 模块语义 claim 登记（§5.2：machine-readable；实现与 claims.json 同源）
MODULE_CLAIMS: dict = {
    "module": "core/doubt",
    "milestone": "M3-1/M3-2",
    "rewrite_spec": "四妹-LMS核心重写规格v2-20260817.md §2.1/§2.2/§2.3/§5.2",
    "claims": {
        "retrieval_phase_readonly": {
            "statement": "检索时怀疑（labile 窗口 / 原生字段视图）只进内存态"
                         "投影；retrieval_hit / retrieval_projection 绝不"
                         "setattr 条目（只读四不变与怀疑机制共存的根基）",
            "verified_by": "tests/test_doubt_m3.py::"
                           "TestThreePhaseStateMachine::"
                           "test_retrieval_phase_never_mutates_entry",
        },
        "transition_write_side_only": {
            "statement": "条目怀疑态转移（stable→suspect→superseded）只能由"
                         "写侧时相驱动（injection_check / "
                         "consolidation_resolve）；检索时相零转移",
            "verified_by": "tests/test_doubt_m3.py::"
                           "TestThreePhaseStateMachine::"
                           "test_transitions_only_from_write_side",
        },
        "injection_suspect_and_verify": {
            "statement": "注入时怀疑：高 surprise（>factor×J_target）或 "
                         "rebuttal 命中 → 标 suspect + 登记验证链"
                         "（验证链开关默认关）",
            "verified_by": "tests/test_doubt_m3.py::"
                           "TestThreePhaseStateMachine::"
                           "test_injection_high_surprise_marks_suspect",
        },
        "labile_not_persistent": {
            "statement": "labile 是时相状态不是持久状态：检索时相只进内存态"
                         "labile 窗口，绝不落库；持久层只存 "
                         "stable/suspect/superseded 三态",
            "verified_by": "tests/test_doubt_m3.py::"
                           "TestThreePhaseStateMachine::"
                           "test_labile_is_transient_not_persistent",
        },
        "rebuttal_field_writer_guard": {
            "statement": "rebuttal_consistency 写者守卫：updated_by 仅允许 "
                         "ingest/consolidation；retrieval 必拒"
                         "（IllegalWriterError——§2.3 铁律）；读取返回深拷贝",
            "verified_by": "tests/test_doubt_m3.py::"
                           "TestRebuttalConsistencyField::"
                           "test_retrieval_writer_rejected",
        },
        "rebuttal_field_ingest_init": {
            "statement": "写入口（ingest/store_episodic）初始化原生字段："
                         "updated_by='ingest'、rebuttals=[]；证伪标记"
                         "（mark_labile）同步结构化字段与平坦字段",
            "verified_by": "tests/test_doubt_m3.py::"
                           "TestRebuttalConsistencyField::"
                           "test_ingest_init_and_rebuttal_sync",
        },
        "verification_chain_idempotent": {
            "statement": "验证链幂等：同幂等键重发不重复登记；结果登记同键"
                         "同验证只一条记录（客户端超时≠服务端未写入）",
            "verified_by": "tests/test_doubt_m3.py::"
                           "TestVerificationChainSkeleton::"
                           "test_same_key_resend_no_duplicate",
        },
        "verification_chain_default_off": {
            "statement": "验证链开关默认 false（§5.3 写侧默认保守：默认关闭、"
                         "显式开启；开启必须过假阳性演练）；关时 register "
                         "返回 None 零参与",
            "verified_by": "tests/test_doubt_m3.py::"
                           "TestVerificationChainSkeleton::"
                           "test_switch_default_off",
        },
        "verification_anti_fraud_four_dimensions": {
            "statement": "防伪独立四维：verify_query ≠ register_query；验证"
                         "批次 ≠ 登记批次；验证通道 ≠ 登记通道（独立端点/"
                         "query/批次/信号）",
            "verified_by": "tests/test_doubt_m3.py::"
                           "TestVerificationChainSkeleton::"
                           "test_anti_fraud_independence_dimensions",
        },
        "contradiction_three_way": {
            "statement": "矛盾判定三选一（§2.2）：只有方向矛盾（结论相反）/"
                         "数值矛盾（同量纲区间不相交）/否定矛盾（存在性/"
                         "成立性否定）三类明确矛盾才判 conflict；未命中"
                         "三选一一律不判矛盾（P0 假冲突根因①根治）",
            "verified_by": "tests/test_doubt_m3.py::"
                           "TestContradictionJudgment::"
                           "test_directional_contradiction",
        },
        "metadata_exclusion_no_false_conflict": {
            "statement": "元数据排除（P0 假冲突根因②根治）：时间戳/日期前缀"
                         "碰撞、同义复述、来源互补一律不判矛盾"
                         "（metadata_excluded=True——假阳性演练红线）",
            "verified_by": "tests/test_doubt_m3.py::"
                           "TestMetadataExclusion::"
                           "test_timestamp_prefix_collision_not_conflict",
        },
        "verification_full_chain": {
            "statement": "验证链全链（草稿→独立验证→修正）：verify/"
                         "run_pending 驱动，防伪独立四维全程保持，VERIFY-* "
                         "provenance 全程记录；CONFLICT 结果由写侧应用为 "
                         "[doubt] conflict → 目标条目 labile（幂等记账）",
            "verified_by": "tests/test_doubt_m3.py::"
                           "TestVerificationChainFullChain::"
                           "test_verify_provenance_full_trail",
        },
    },
}

from core.doubt.confidence_field import (  # noqa: E402
    FORCE_DOWNGRADE_CONFIDENCE,
    FORCE_DOWNGRADE_REBUTTALS,
    LOW_CONFIDENCE_THRESHOLD,
    compute_confidence,
    get_rebuttal_rate,
    get_source_trust,
    is_low_confidence,
    mark_rebutted,
    record_reference,
    refresh_confidence,
)
from core.doubt.reconsolidation import (  # noqa: E402
    detect_destabilization,
    find_violated_entry,
    mark_labile,
    resolve_labile,
)
from core.doubt.recall_scheduler import (  # noqa: E402
    DEFAULT_WEIGHTS,
    forgetting,
    salience,
    select_with_low_confidence_quota,
)
from core.doubt.gap_registry import GapRegistry  # noqa: E402
from core.doubt import doubt_ingest  # noqa: E402
from core.doubt.rebuttal_field import (  # noqa: E402
    FIELD_NAME,
    LEGAL_WRITERS,
    IllegalWriterError,
    empty_rebuttal_consistency,
    get_rebuttal_consistency,
    init_rebuttal_consistency,
    is_legal_writer,
    read_view,
    record_rebuttal_native,
    update_consistency,
)
from core.doubt.verification_chain import (  # noqa: E402
    ConflictKind,
    VerificationChain,
    VerificationEventType,
    VerificationRequest,
    VerificationResult,
    VerdictType,
    is_contradiction,
    judge_contradiction,
    verdict_for,
    verification_chain_enabled,
    verification_key,
)
from core.doubt.state_machine import (  # noqa: E402
    DoubtPhase,
    DoubtStateMachine,
    EntryDoubtState,
    compute_rebuttal_hit,
    doubt_injection_enabled,
    is_high_surprise,
)
from core.doubt.purpose_drift import (  # noqa: E402
    PurposeDriftPhase,
    purpose_drift_enabled,
)
from core.doubt.precision_adapt import (  # noqa: E402
    PrecisionLearnState,
    precision_learn_enabled,
)

__all__ = [
    "MODULE_CLAIMS",
    # confidence_field
    "FORCE_DOWNGRADE_CONFIDENCE",
    "FORCE_DOWNGRADE_REBUTTALS",
    "LOW_CONFIDENCE_THRESHOLD",
    "compute_confidence",
    "get_rebuttal_rate",
    "get_source_trust",
    "is_low_confidence",
    "mark_rebutted",
    "record_reference",
    "refresh_confidence",
    # reconsolidation
    "detect_destabilization",
    "find_violated_entry",
    "mark_labile",
    "resolve_labile",
    # recall_scheduler
    "DEFAULT_WEIGHTS",
    "forgetting",
    "salience",
    "select_with_low_confidence_quota",
    # gap_registry / doubt_ingest
    "GapRegistry",
    "doubt_ingest",
    # rebuttal_field（M3-1）
    "FIELD_NAME",
    "LEGAL_WRITERS",
    "IllegalWriterError",
    "empty_rebuttal_consistency",
    "get_rebuttal_consistency",
    "init_rebuttal_consistency",
    "is_legal_writer",
    "read_view",
    "record_rebuttal_native",
    "update_consistency",
    # verification_chain（M3-1/M3-2）
    "ConflictKind",
    "VerificationChain",
    "VerificationEventType",
    "VerificationRequest",
    "VerificationResult",
    "VerdictType",
    "is_contradiction",
    "judge_contradiction",
    "verdict_for",
    "verification_chain_enabled",
    "verification_key",
    # state_machine（M3-1）
    "DoubtPhase",
    "DoubtStateMachine",
    "EntryDoubtState",
    "compute_rebuttal_hit",
    "doubt_injection_enabled",
    "is_high_surprise",
    # purpose_drift（总任务书 §二.5：目的检查时相）
    "PurposeDriftPhase",
    "purpose_drift_enabled",
    # precision_learn（ABC §S3：π = 1/Var 估计 + lr_multiplier 跟随 π）
    "PrecisionLearnState",
    "precision_learn_enabled",
]

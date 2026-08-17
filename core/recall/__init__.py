# -*- coding: utf-8 -*-
"""core/recall · 纯只读检索模块（核心重建 M2 · recall 只读化）

规格依据：`四妹-LMS核心重写规格v2-20260817.md` §1.4（recall 纯只读——
只读四不变硬约束）/ §2.1（检索时怀疑·labile 内存态投影）/ §5.1（只读
四不变自动化，坑 9 根治）/ §5.2（claim 登记）。

重写总纲：**核心重建，血管不换**——本包是 recall 的机器防线与怀疑投影
层，端点签名/返回结构冻结，只换内部实现（删除只读泄漏，四不变从"注释
承诺"变"机器断言"）。

包结构（M2 交付）：
  - ``guard.py``     只读四不变守卫：FourInvariantGuard / ReadOnlyViolation /
                     entry_fingerprint / diff_four_invariants（纯 stdlib，
                     状态读取由调用方注入 state_reader——loop 层张量转 list）；
  - ``suspicion.py`` 检索时怀疑投影：project_suspicion / entry_readonly_view
                     （只读扫描，labile/rebuttal/verdict 只进内存态投影，
                     绝不 setattr 条目、绝不进快照）。

依赖注入设计：本包**不 import 任何 LMS 运行时模块**（纯 stdlib）——
可被 python 直接 import/运行；loop 层通过注入 state_reader /
precision_adapt / consistency_provider 接入（M1 core/store 同款约束）。

与 M1（core/store 写侧）的边界：store 是写侧唯一入口（幂等/注入时怀疑），
recall 是纯只读侧（四不变 + 检索时怀疑投影）——两侧由 loop 9 步生命周期
（M5 重组）在 turn 层编排，本包不触碰写侧。
"""

from __future__ import annotations

from core.recall.guard import (
    ReadOnlyViolation,
    FourInvariantGuard,
    entry_fingerprint,
    episodic_fingerprint,
    diff_four_invariants,
)
from core.recall.suspicion import (
    project_suspicion,
    entry_readonly_view,
    empty_suspicion,
    EMPTY_SUSPICION,
)

__all__ = [
    "ReadOnlyViolation",
    "FourInvariantGuard",
    "entry_fingerprint",
    "episodic_fingerprint",
    "diff_four_invariants",
    "project_suspicion",
    "entry_readonly_view",
    "empty_suspicion",
    "EMPTY_SUSPICION",
    "MODULE_CLAIMS",
]

#: 模块语义 claim 登记（§5.2：machine-readable；实现与 claims.json 同源）
MODULE_CLAIMS: dict = {
    "module": "core/recall",
    "milestone": "M2",
    "claims": {
        "read_only_four_invariants": {
            "statement": "一次 recall（/recall、/react 检索段、MCP 直连）执行"
                         "前后 turn / episodic 条目集 / J / σ 四量零增量；"
                         "违反抛 ReadOnlyViolation → API 500 + 告警（G 模式"
                         "禁止静默）",
            "verified_by": "tests/test_recall_m2.py::TestFourInvariantGuard + "
                           "tests/test_recall_m2.py::TestRecallReadonlyM2::"
                           "test_guard_armed_detects_mutation + "
                           "tests/test_recall_m2.py::TestRecallEndpointM2::"
                           "test_recall_violation_maps_500",
        },
        "no_entry_mutation": {
            "statement": "检索路径绝不改写条目（旧 _attach_consistency 的 "
                         "entry.consistency 改写已删除）；条目指纹含全部可变"
                         "标量字段——字段改写同样触发守卫",
            "verified_by": "tests/test_recall_m2.py::TestRecallReadonlyM2::"
                           "test_recall_does_not_mutate_entries + "
                           "tests/test_recall_m2.py::TestFourInvariantGuard::"
                           "test_entry_fingerprint_captures_consistency_mutation",
        },
        "suspicion_projection_only": {
            "statement": "检索时怀疑（labile 窗口 / rebuttal 待核查 / verdict "
                         "可疑）只进内存态投影随响应返回；绝不写条目、绝不进"
                         "快照",
            "verified_by": "tests/test_recall_m2.py::TestSuspicionProjection::"
                           "test_projection_never_mutates_entries",
        },
        "attach_consistency_removed": {
            "statement": "旧 _attach_consistency 整体删除（loop 无此方法），"
                         "一致性计算保留为只读投影（仅进程内观测缓存）",
            "verified_by": "tests/test_recall_m2.py::TestRecallReadonlyM2::"
                           "test_attach_consistency_removed",
        },
        "endpoint_contract_frozen": {
            "statement": "端点签名/返回结构同构：/recall 与 /react 既有字段"
                         "逐字节不变，怀疑信号只追加在既有结构之后独立区段"
                         "（§4.2）",
            "verified_by": "tests/test_recall_m2.py::TestRecallEndpointM2::"
                           "test_recall_response_has_suspicion_section + "
                           "tests/test_recall_readonly.py（回归）",
        },
    },
}

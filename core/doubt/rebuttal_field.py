# -*- coding: utf-8 -*-
"""core/doubt · rebuttal-consistency 字段原生（M3-1 · §2.3）

规格依据：`四妹-LMS核心重写规格v2-20260817.md` §2.3（rebuttal-consistency
字段原生——只读，检索绝不改写）。

每个条目原生携带（结构化字段 ``rebuttal_consistency``）：

.. code-block:: json

    {
      "rebuttals": ["<rebuttal 条目 id>", "..."],
      "consistency": 0.0,            // 0..1；供 precision 调制（写侧原生值）
      "updated_at": "<ts>",
      "updated_by": "ingest"         // 只允许 ingest / consolidation（写侧时相）
    }

铁律（§2.3）：``updated_by`` **永不为 ``retrieval``**——检索路径只读该字段
进 π 调制与怀疑信号，写该字段的唯一合法入口是写侧（注入 ingest / 巩固
consolidation）。本模块提供**存取接口**：读取返回深拷贝（绝不暴露内部引用
防检索路径误写）；写入带 writer 守卫（非法写者直接拒绝）。

与既有平坦字段的关系（任务书"条目原生携带 rebuttal_count/consistency/
violated_by"）：``entry.rebuttal_count`` / ``entry.violated_by`` /
``entry.confidence`` 仍是标量权威（confidence_field 公式不变）；结构化
``rebuttal_consistency`` 是它们的**写侧溯源封装**（rebuttals 列表 /
consistency 原生值 / updated_at / updated_by），由本模块在写入口同步。
平坦 ``entry.consistency``（Koriat 召回时自一致性投影）**不**由本模块
触碰——M2 铁律：召回只读投影、绝不写回条目（两者概念正交）。

设计约束（M1 ``core/store`` 同款：纯 stdlib，可被轻量单测直接 import）：
  - 不 import torch / fastapi / LMS 运行时模块；
  - 对条目对象只做 getattr/setattr（条目可为任意持有这些字段的对象）。
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

#: 结构化字段名（条目上的属性名）
FIELD_NAME = "rebuttal_consistency"

#: 合法写者（§2.3：只允许 ingest / consolidation——写侧时相）。
#: **retrieval 永不在其中**（铁律：检索只读）。
LEGAL_WRITERS: frozenset = frozenset({"ingest", "consolidation"})

#: 结构化字段键（§2.3 schema）
_KEY_REBUTTALS = "rebuttals"
_KEY_CONSISTENCY = "consistency"
_KEY_UPDATED_AT = "updated_at"
_KEY_UPDATED_BY = "updated_by"

#: 一致性钳制区间
_CONS_MIN, _CONS_MAX = 0.0, 1.0


class IllegalWriterError(ValueError):
    """非法写者（如 ``updated_by='retrieval'``）——§2.3 铁律违反。

    写侧守卫拒绝写入并抛本异常（G 模式：非法写者必须可见，不静默吞掉）；
    调用方（写侧）应捕获并按 fail-open 降级（不阻断主流程），但不得把
    非法写者"悄悄改成合法写者"。
    """


def is_legal_writer(updated_by: str) -> bool:
    """写者合法性判定（§2.3 铁律：retrieval 永非法）。"""
    return updated_by in LEGAL_WRITERS


def empty_rebuttal_consistency() -> Dict[str, Any]:
    """返回一份独立的空结构（避免共享可变默认值）。"""
    return {
        _KEY_REBUTTALS: [],
        _KEY_CONSISTENCY: 0.0,
        _KEY_UPDATED_AT: None,
        _KEY_UPDATED_BY: None,
    }


def _coerce_consistency(value: Any) -> float:
    """一致性归一化 [0,1]（fail-open：畸形值 → 0.0）。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return max(_CONS_MIN, min(_CONS_MAX, v))


# ---------------------------------------------------------------------- #
#  读接口（检索路径唯一合法操作）
# ---------------------------------------------------------------------- #

def get_rebuttal_consistency(entry: Any) -> Dict[str, Any]:
    """只读读取结构化字段（返回**深拷贝**——绝不暴露条目内部引用）。

    检索路径（recall / suspicion 投影 / precision 调制）只能经本接口读取；
    返回拷贝保证"读"不可能变成"写"（检索路径误改副本对条目零影响）。
    缺失/畸形字段 → 空结构（fail-open，向后兼容旧条目/旧快照）。
    """
    rc = getattr(entry, FIELD_NAME, None)
    if not isinstance(rc, dict):
        return empty_rebuttal_consistency()
    return {
        _KEY_REBUTTALS: list(rc.get(_KEY_REBUTTALS) or []),
        _KEY_CONSISTENCY: _coerce_consistency(
            rc.get(_KEY_CONSISTENCY, 0.0)),
        _KEY_UPDATED_AT: rc.get(_KEY_UPDATED_AT),
        _KEY_UPDATED_BY: rc.get(_KEY_UPDATED_BY),
    }


def read_view(entry: Any) -> Dict[str, Any]:
    """怀疑投影/响应注解用的精简只读视图（深拷贝语义同 get）。"""
    rc = get_rebuttal_consistency(entry)
    return {
        "rebuttals": rc[_KEY_REBUTTALS],
        "consistency": rc[_KEY_CONSISTENCY],
        "updated_at": rc[_KEY_UPDATED_AT],
        "updated_by": rc[_KEY_UPDATED_BY],
    }


# ---------------------------------------------------------------------- #
#  写接口（写侧时相专用：ingest / consolidation）
# ---------------------------------------------------------------------- #

def _set_field(entry: Any, rc: Dict[str, Any]) -> None:
    """写回条目（本模块唯一 setattr 点——全部经写者守卫后到达）。"""
    setattr(entry, FIELD_NAME, rc)


def init_rebuttal_consistency(
    entry: Any, *, now: Optional[float] = None,
    consistency: Optional[float] = None,
) -> Dict[str, Any]:
    """写侧初始化（ingest 写入新条目时调用——store_episodic 内建）。

    幂等：字段已存在（旧条目/重复调用）→ 原样返回，不覆盖既有证据。
    ``updated_by`` 初始化为 ``'ingest'``（写入口创建条目）。
    """
    existing = getattr(entry, FIELD_NAME, None)
    if isinstance(existing, dict):
        return get_rebuttal_consistency(entry)
    rc = {
        _KEY_REBUTTALS: [],
        _KEY_CONSISTENCY: _coerce_consistency(
            consistency if consistency is not None else 0.0),
        _KEY_UPDATED_AT: now if now is not None else time.time(),
        _KEY_UPDATED_BY: "ingest",
    }
    _set_field(entry, rc)
    return dict(rc)


def update_consistency(
    entry: Any, *, consistency: float, updated_by: str,
    now: Optional[float] = None,
) -> bool:
    """写侧更新原生 consistency（供 precision 调制；巩固期重算走本接口）。

    例：``update_consistency(entry, consistency=0.8, updated_by='consolidation')``。

    Raises:
        IllegalWriterError: ``updated_by`` 不在 {ingest, consolidation}——
            尤其 ``'retrieval'`` 必拒（§2.3 铁律）。
    """
    if not is_legal_writer(updated_by):
        raise IllegalWriterError(
            f"rebuttal_consistency 非法写者 updated_by={updated_by!r}"
            f"（仅允许 {sorted(LEGAL_WRITERS)}；retrieval 永不可写）")
    rc = get_rebuttal_consistency(entry)
    rc[_KEY_CONSISTENCY] = _coerce_consistency(consistency)
    rc[_KEY_UPDATED_AT] = now if now is not None else time.time()
    rc[_KEY_UPDATED_BY] = updated_by
    _set_field(entry, rc)
    return True


def record_rebuttal_native(
    entry: Any, *, rebuttal_id: Optional[str] = None,
    violated_by: Optional[str] = None, updated_by: str = "ingest",
    now: Optional[float] = None,
) -> bool:
    """证伪登记（写侧时相——注入时怀疑 / 巩固期统一入口）。

    副作用（与既有 ``confidence_field.mark_rebutted`` 完全同源）：
      - 平坦字段：``rebuttal_count +1``、``violated_by`` 记录、置信度重算；
      - 结构化字段：rebuttal id 追加（去重）、consistency = 证伪后置信度、
        updated_at / updated_by 更新。

    Raises:
        IllegalWriterError: 非法写者（retrieval 必拒）。
    """
    if not is_legal_writer(updated_by):
        raise IllegalWriterError(
            f"rebuttal_consistency 非法写者 updated_by={updated_by!r}"
            f"（仅允许 {sorted(LEGAL_WRITERS)}；retrieval 永不可写）")
    from core.doubt.confidence_field import mark_rebutted
    # 平坦字段权威更新（既有行为逐字节保留；异常 fail-open）
    try:
        mark_rebutted(entry, violated_by=violated_by)
    except Exception:  # pylint: disable=broad-except
        pass
    rc = get_rebuttal_consistency(entry)
    if rebuttal_id:
        rebuttals = list(rc[_KEY_REBUTTALS])
        if rebuttal_id not in rebuttals:
            rebuttals.append(rebuttal_id)
        rc[_KEY_REBUTTALS] = rebuttals[-64:]  # 防膨胀（上限 64 条）
    rc[_KEY_CONSISTENCY] = _coerce_consistency(
        getattr(entry, "confidence", 0.0))
    rc[_KEY_UPDATED_AT] = now if now is not None else time.time()
    rc[_KEY_UPDATED_BY] = updated_by
    _set_field(entry, rc)
    return True

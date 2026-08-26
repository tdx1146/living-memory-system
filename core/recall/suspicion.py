# -*- coding: utf-8 -*-
"""检索时怀疑投影（M2 · §2.1 检索时怀疑 / §1.4 只读四不变）。

规格依据：`四妹-LMS核心重写规格v2-20260817.md` §1.4（检索时怀疑产生的
labile/怀疑信号**只进内存态投影**，随响应返回；绝不写条目、绝不进快照）、
§2.1（三时相怀疑状态机——检索时相只产生投影，这是只读四不变与怀疑机制
共存的根基）。

铁律：本模块的投影函数**绝不 setattr 条目**——labile / rebuttal /
verdict 全部只读自条目与精度观测对象，产出独立的投影结构返回调用方；
条目持久层只存 ``stable / suspect / superseded`` 三态（M3 落地），
``labile`` 是时相状态不是持久状态。

设计约束（M1 ``core/store`` 同款：纯 stdlib，可被轻量单测直接 import）：
  - 不 import torch / fastapi / LMS 运行时模块；
  - 精度观测（verdict_confidence / doubt_threshold / consistency 缓存）
    通过注入的 ``precision_adapt`` 对象只读调用（None → 相应区段为空）。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------- #
#  只读条目视图
# ---------------------------------------------------------------------- #

#: 响应条目/怀疑信号区的只读字段（全部 getattr 只读，绝不 setattr）。
_VIEW_FIELDS: tuple = (
    "text",
    "confidence",
    "rebuttal_count",
    "labile",
    "source_trust",
    "consistency",
    "info_value",
    "core",
    "gray",
    "ts",
)


def entry_readonly_view(entry: Any, score: Optional[float] = None) -> dict:
    """条目只读视图（检索/怀疑信号共用；绝不改写条目）。"""
    view: Dict[str, Any] = {}
    for name in _VIEW_FIELDS:
        view[name] = getattr(entry, name, None)
    if score is not None:
        view["score"] = float(score)
    # 置信度归一（与 /recall 响应口径一致：None → 1.0）
    conf = view.get("confidence")
    view["confidence"] = round(float(conf or 1.0), 3) if conf is not None else 1.0
    view["rebuttal_count"] = int(view.get("rebuttal_count") or 0)
    view["labile"] = bool(view.get("labile", False))
    return view


# ---------------------------------------------------------------------- #
#  怀疑信号投影
# ---------------------------------------------------------------------- #

#: 空投影（稳定结构：无检索/无命中时返回同构区段，消费端零特判）。
EMPTY_SUSPICION: dict = {
    "labile": [],
    "rebuttal_pending": [],
    "verdict_suspect": [],
    "summary": {
        "total": 0,
        "labile": 0,
        "rebuttal_pending": 0,
        "verdict_suspect": 0,
    },
}


def empty_suspicion() -> dict:
    """返回一份独立的空投影结构（避免共享可变默认值）。"""
    return {
        "labile": [],
        "rebuttal_pending": [],
        "verdict_suspect": [],
        "summary": dict(EMPTY_SUSPICION["summary"]),
    }


def project_suspicion(
        scored: Sequence[Tuple[float, Any]],
        precision_adapt: Any = None,
        consistency_provider: Optional[Callable[[Any], Optional[float]]] = None,
        max_items: int = 20,
) -> dict:
    """检索时怀疑投影（只读；绝不改写条目）。

    参数:
        scored: ``[(score, entry), ...]``（recall 命中簇，按相关度降序）。
        precision_adapt: 可选的精度观测对象——只读调用其
            ``verdict_confidence(entry, cons)`` 与 ``doubt_threshold()``；
            None → ``verdict_suspect`` 区段为空。
        consistency_provider: 可选的逐条目一致性读取回调（如
            ``consistency_cache.get(id(entry))``）；None → 一致性用 None。
        max_items: 各区条目视图数上限（防响应膨胀）。

    返回:
        ``{labile: [...], rebuttal_pending: [...], verdict_suspect: [...],
        summary: {total, labile, rebuttal_pending, verdict_suspect}}``——
        全部为内存态投影，随响应返回，绝不落库。
    """
    labile: List[dict] = []
    rebuttal: List[dict] = []
    verdict: List[dict] = []
    total = len(scored)
    for score, entry in scored:
        view = entry_readonly_view(entry, score)
        if bool(getattr(entry, "labile", False)):
            labile.append(view)
        if int(getattr(entry, "rebuttal_count", 0) or 0) > 0:
            rebuttal.append(view)
        if precision_adapt is not None:
            try:
                cons = (consistency_provider(entry)
                        if consistency_provider is not None else None)
                vconf = precision_adapt.verdict_confidence(entry, cons)
                if vconf < precision_adapt.doubt_threshold():
                    verdict.append(view)
            except Exception:  # pylint: disable=broad-except
                # 精度观测异常：该条目不进 verdict 区（fail-open，只读）
                pass
    return {
        "labile": labile[:max_items],
        "rebuttal_pending": rebuttal[:max_items],
        "verdict_suspect": verdict[:max_items],
        "summary": {
            "total": total,
            "labile": len(labile),
            "rebuttal_pending": len(rebuttal),
            "verdict_suspect": len(verdict),
        },
    }

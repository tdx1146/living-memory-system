# -*- coding: utf-8 -*-
"""只读四不变守卫（M2 · recall 只读化 · 机器防线）。

规格依据：`四妹-LMS核心重写规格v2-20260817.md` §1.4（recall 纯只读）/
§5.1（只读四不变自动化，坑 9 根治：从"注释承诺"到"机器防线"）。

只读四不变：一次 recall（含 /recall 与 /react 的检索段）执行前后，
``turn / episodic 条目集 / J / σ`` 四个量**零增量**；违反即抛
:class:`ReadOnlyViolation`（API 层映射 500 + 告警——G 模式：禁止静默，
绝不 fail-open 掩盖）。

设计约束（M1 ``core/store`` 同款：可被轻量单测直接 import/运行）：
  - 本模块**纯 stdlib**：不 import torch / fastapi / LMS 运行时模块。
  - 状态读取通过调用方注入的 ``state_reader`` 回调完成（loop 层注入
    ``_readonly_state_snapshot``，张量在注入侧转 list——本模块只做比较）。
  - ``state_reader`` 返回 dict：``{turn: int, episodic: frozenset, J: 可比值,
    sigma: 可比值}``；``episodic`` 由调用方用 :func:`entry_fingerprint`
    构造（含全部可变标量字段指纹——不仅能抓条目增删，也能抓字段改写，
    旧 ``_attach_consistency`` 泄漏形态就是字段改写）。

机器防线三层（§5.1）：
  1. loop 层：``recall_episodic_readonly`` / ``react_readonly`` /
     ``recall_merged_readonly`` 全部包守卫（覆盖 /recall、/react 检索段、
     MCP stdio 直连路径——所有调用面都汇入这三个方法）；
  2. API 层：捕获 :class:`ReadOnlyViolation` → 500 + ``logger.error`` 告警；
  3. 测试层：test_recall_m2.py 对每条 claim 有对应用例；conftest 禁止把
     守卫置 0 关掉（H 模式：测试绿不能掩盖生产 bug）。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional


# ---------------------------------------------------------------------- #
#  异常
# ---------------------------------------------------------------------- #

class ReadOnlyViolation(RuntimeError):
    """只读四不变违反。

    一次 recall 执行前后 ``turn / episodic 条目集 / J / σ`` 出现非零增量。
    API 层捕获后映射 500 并告警——绝不静默、绝不 fail-open 掩盖。
    """

    def __init__(self, scope: str, changes: Dict[str, Any]):
        self.scope = scope
        self.changes = changes
        super().__init__(
            f"[recall只读四不变违反] scope={scope} 变更: {changes}")

    def to_dict(self) -> dict:
        """序列化违反详情（供 API 层 500 detail / 告警日志）。"""
        return {"scope": self.scope, "changes": self.changes}


# ---------------------------------------------------------------------- #
#  条目指纹（episodic 条目集的可比较形态）
# ---------------------------------------------------------------------- #

#: 参与条目指纹的可变标量字段（读 getattr + 默认值兜底；绝不 setattr）。
#: 覆盖旧泄漏形态（consistency 改写）+ 写侧会触碰的全部标量字段——
#: 条目集指纹既抓增删又抓字段改写。
#: M3-1 追加 rebuttal_consistency（结构化 dict，_norm_value 已支持）与
#: doubt_state——检索路径对原生怀疑字段的任何改写同样触发守卫（§2.3
#: 检索只读铁律的机器防线）。
_ENTRY_FIELD_NAMES: tuple = (
    "text",
    "surprise",
    "turn",
    "source",
    "confidence",
    "rebuttal_count",
    "reference_count",
    "source_trust",
    "labile",
    "labile_since",
    "violated_by",
    "last_recalled_at",
    "recall_count",
    "consistency",
    "confidence_before_rebuttal",
    "last_reinforced_turn",
    "info_value",
    "core",
    "ts",
    "gray",
    "rebuttal_consistency",
    "doubt_state",
)


def _norm_value(value: Any) -> Any:
    """把值归一化为可哈希、可稳定比较的形态。

    - None / bool / str / int 原样；
    - float：NaN → 哨兵字符串（NaN != NaN 会破坏 frozenset 比较），
      其余 round(v, 9)（平滑嵌入噪声，但保留 ≥1e-9 的真实改写）；
    - list/tuple → 递归归一化后的 tuple（可哈希）；
    - dict → 按键排序的 (key, value) 元组序列（可哈希——M3-1 追加的
      rebuttal_consistency 结构化字段由此可进指纹；键统一转 str 排序，
      防异构键比较崩溃）。
    """
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if value != value:  # NaN
            return "nan"
        return round(value, 9)
    if isinstance(value, (list, tuple)):
        return tuple(_norm_value(v) for v in value)
    if isinstance(value, dict):
        return tuple(
            (str(k), _norm_value(v))
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        )
    # 其他类型（不应出现在标量字段）：原样返回，__eq__ 由调用方保证可比
    return value


def entry_fingerprint(entry: Any) -> tuple:
    """条目指纹（id + 全部可变标量字段归一化值）。

    id(entry) 区分对象（同内容不同条目不误判），字段值抓改写。
    """
    return tuple([id(entry)] + [
        _norm_value(getattr(entry, name, None)) for name in _ENTRY_FIELD_NAMES
    ])


def episodic_fingerprint(entries) -> frozenset:
    """条目集指纹（frozenset：零增量 = 无增删 + 无字段改写）。"""
    return frozenset(entry_fingerprint(e) for e in entries)


# ---------------------------------------------------------------------- #
#  四不变守卫
# ---------------------------------------------------------------------- #

class FourInvariantGuard:
    """只读四不变守卫（上下文管理器 + 显式两步）。

    用法（loop 层）：

    .. code-block:: python

        guard = FourInvariantGuard(state_reader=self._readonly_state_snapshot,
                                   scope="recall_merged_readonly")
        with guard:
            ... recall 执行 ...
            return results

    守卫在 ``__exit__``（正常路径）比较前后快照；body 异常时跳过断言
    （不掩盖原始异常，原始异常本身已足够响亮）。
    """

    #: 默认强制开启（§5.1 机器防线）。不提供 env 开关——测试不许把
    #: 增量检查置 0 关掉（H 模式）。
    enabled: bool = True

    def __init__(self, state_reader: Callable[[], dict],
                 scope: str = "recall"):
        if not callable(state_reader):
            raise TypeError("state_reader 必须可调用（返回四不变快照 dict）")
        self.state_reader = state_reader
        self.scope = scope
        self._before: Optional[dict] = None

    # -- 上下文管理器 ------------------------------------------------ #

    def __enter__(self) -> "FourInvariantGuard":
        self._before = self.snapshot()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.assert_unchanged(self._before, scope=self.scope)
        return False  # 不吞异常

    # -- 显式两步（非 with 场景）------------------------------------- #

    def snapshot(self) -> dict:
        """读取当前四不变状态快照。"""
        return self.state_reader()

    def assert_unchanged(self, before: dict,
                         scope: Optional[str] = None) -> None:
        """比较 before 与当前状态；出现增量 → 抛 ReadOnlyViolation。"""
        if not self.enabled:
            return
        after = self.state_reader()
        changes = diff_four_invariants(before, after)
        if changes:
            raise ReadOnlyViolation(scope or self.scope, changes)


# ---------------------------------------------------------------------- #
#  四量比较
# ---------------------------------------------------------------------- #

def _values_equal(a: Any, b: Any) -> bool:
    """嵌套 list/tuple/标量的深度相等（NaN 视为相等）。"""
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:  # 双 NaN
            return True
    try:
        return bool(a == b)
    except Exception:  # pylint: disable=broad-except
        return str(a) == str(b)


def diff_four_invariants(before: dict, after: dict) -> dict:
    """对比两个四不变快照，返回变更明细（无变更 → {}）。

    返回形如 ``{"turn": (before, after), "J": (before, after), ...}``；
    episodic 变更附条目数差异（指纹集增量在守卫侧已可比）。
    """
    changes: Dict[str, Any] = {}
    for key in ("turn", "episodic", "J", "sigma"):
        b = before.get(key)
        a = after.get(key)
        if not _values_equal(b, a):
            changes[key] = {
                "before": _describe(b),
                "after": _describe(a),
            }
    return changes


def _describe(value: Any) -> Any:
    """快照值的可读描述（episodic 指纹集 → 条目数，避免巨串日志）。"""
    if isinstance(value, frozenset):
        return f"<episodic entries={len(value)}>"
    if isinstance(value, (list, tuple)) and len(value) > 16:
        return f"<{type(value).__name__} len={len(value)}>"
    return value

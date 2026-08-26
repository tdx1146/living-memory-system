# -*- coding: utf-8 -*-
"""core/doubt · 三时相怀疑状态机（M3-1 · §2.1）

规格依据：`四妹-LMS核心重写规格v2-20260817.md` §2.1（三时相怀疑状态机
——Nader 2000：记忆被检索即回到可修改的易变态——labile 窗口）。

三时相：

+----------+----------------------+-------------------------------+------------+
| 时相      | 触发点               | 动作                           | 落库？     |
+==========+======================+===============================+============+
| 检索时    | recall 命中条目       | 条目进 labile 窗口（内存态）； | **否**    |
| 怀疑      |                      | 检查 rebuttal-consistency；   |（只读     |
|          |                      | 输出怀疑信号（衔接 M2         |  四不变）  |
|          |                      | ``project_suspicion``）       |            |
| 注入时    | store/feed 写入新条目 | 计算 surprise 与 J_target 比较；| 是（写侧）|
| 怀疑      |（loop process_turn    | 高 surprise 或存在 rebuttal   |            |
|          |  / core.store 钩子）  | → 标 ``suspect`` 并登记验证链 |            |
| 巩固时    | dream 期（做梦）      | 对 labile/suspect 条目        | 是（写侧）|
| 怀疑      |                      | ``consolidation_resolve``：   |            |
|          |                      | 验证通过→确认（stable），     |            |
|          |                      | 冲突→supersedes；更新         |            |
|          |                      | confidence（**接口本段定义，   |            |
|          |                      | 完整做梦循环属 M6**）         |            |
+----------+----------------------+-------------------------------+------------+

条目怀疑态（§2.1 状态图）：持久层只存 ``stable / suspect / superseded``
三态（``EntryDoubtState``）；``labile`` 是**时相状态不是持久状态**
（只在检索/注入时相存在，随响应生命周期结束，不落库）。

铁律：**状态转移只能由写侧时相驱动**（注入/巩固）——检索时相只产生
投影。本模块 ``retrieval_hit`` / ``retrieval_projection`` 绝不 setattr
条目（只读四不变）；``injection_check`` / ``consolidation_resolve`` 是
仅有的两个写侧状态转移入口。

治理开关：``LMS_DOUBT_INJECTION_ENABLED`` 默认 1（注入时怀疑是机制本体
——写侧默认保守：宁可拦错不轻信）；验证链登记开关独立见
``verification_chain.py``（默认 0）。开关关 → 全部路径零参与，行为与
开关引入前完全一致（8/10 治理开关先例风格）。

设计约束（M1 ``core/store`` 同款：纯 stdlib，可被轻量单测直接 import）：
  - 不 import torch / fastapi / LMS 运行时模块；
  - 精度观测/验证链通过注入对象只读调用（None → 相应路径为空）。
"""

from __future__ import annotations

import enum
import logging
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from core.doubt.purpose_drift import PurposeDriftPhase
from core.doubt.rebuttal_field import (
    get_rebuttal_consistency,
    update_consistency,
)
from core.doubt.verification_chain import (
    VerificationChain,
    VerdictType,
    verification_key,
)

logger = logging.getLogger("core.doubt.state_machine")

# ---------------------------------------------------------------------- #
#  时相 / 条目怀疑态
# ---------------------------------------------------------------------- #

class DoubtPhase(str, enum.Enum):
    """三时相（§2.1）。"""

    RETRIEVAL = "retrieval"           # 检索时怀疑（只读投影，不落库）
    INJECTION = "injection"           # 注入时怀疑（写侧）
    CONSOLIDATION = "consolidation"   # 巩固时怀疑（写侧，做梦期）


class EntryDoubtState(str, enum.Enum):
    """条目持久怀疑态（§2.1：持久层只存三态；labile 是时相状态不进本枚举）。"""

    STABLE = "stable"
    SUSPECT = "suspect"
    SUPERSEDED = "superseded"


# ---------------------------------------------------------------------- #
#  env 参数（§5.3：单一默认值来源；运行时读取，改动即时生效）
# ---------------------------------------------------------------------- #

_ENV_INJECTION_ENABLED = "LMS_DOUBT_INJECTION_ENABLED"
_ENV_SURPRISE_FACTOR = "LMS_DOUBT_SURPRISE_FACTOR"
_ENV_LABILE_TTL = "LMS_DOUBT_LABILE_WINDOW_TTL"

#: 注入时怀疑开关默认 **1**（机制本体，写侧默认保守——宁可拦错不轻信；
#: 与验证链开关（默认 0）正交）
_DEFAULT_INJECTION_ENABLED = True

#: 高 surprise 判定：surprise > factor × J_target（§2.1"计算 surprise 与
#: J_target 比较"；factor 默认 1.0，env 可调——判定参数 env 化，M8 校准）
_DEFAULT_SURPRISE_FACTOR = 1.0

#: labile 窗口 TTL（秒）：检索时相命中的内存态窗口存活期（§2.1 labile
#: 窗口 / §2.4 做梦期待核查清单的累积口径；默认 1h，env 可调）
_DEFAULT_LABILE_TTL = 3600.0


def doubt_injection_enabled(explicit: Optional[bool] = None) -> bool:
    """注入时怀疑开关解析：显式参数 > 环境变量 LMS_DOUBT_INJECTION_ENABLED。

    布尔接受（不区分大小写）：1/true/yes/on 视为开，其余为关。
    默认开（机制本体）；0=关 → 全部路径零参与，行为与开关引入前完全一致。
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get(_ENV_INJECTION_ENABLED, "1")
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _surprise_factor() -> float:
    try:
        return max(0.0, float(os.environ.get(
            _ENV_SURPRISE_FACTOR, str(_DEFAULT_SURPRISE_FACTOR))))
    except (TypeError, ValueError):
        return _DEFAULT_SURPRISE_FACTOR


def _labile_ttl() -> float:
    try:
        return max(1.0, float(os.environ.get(
            _ENV_LABILE_TTL, str(_DEFAULT_LABILE_TTL))))
    except (TypeError, ValueError):
        return _DEFAULT_LABILE_TTL


def _neutral_purpose_gate() -> dict:
    """目的检查双保险 fail-open 的中性闸门（不判是/否，闸门不亮）。"""
    from core.doubt.purpose_drift import _DEFAULT_PURPOSE_TEXT
    return {
        "purpose_drift": False,
        "verdict": "uncertain",
        "answers": {q: {"ok": None, "state": "uncertain",
                        "reason": "目的检查委托失败（fail-open）——闸门中性"}
                    for q in ("Q1", "Q2", "Q3", "Q4", "Q5")},
        "reasons": ["目的检查委托失败（fail-open）——闸门中性，未判"],
        "purpose": _DEFAULT_PURPOSE_TEXT,
    }


# ---------------------------------------------------------------------- #
#  纯函数：注入时判定
# ---------------------------------------------------------------------- #

def is_high_surprise(surprise: Optional[float], j_target: Optional[float],
                     factor: Optional[float] = None) -> bool:
    """高 surprise 判定（§2.1：surprise 与 J_target 比较）。

    ``surprise > factor × J_target``；缺 surprise / j_target →
    False（fail-open 保守：信息不足不判高）。
    """
    if surprise is None or j_target is None:
        return False
    f = _surprise_factor() if factor is None else max(0.0, float(factor))
    try:
        return float(surprise) > f * float(j_target)
    except (TypeError, ValueError):
        return False


def compute_rebuttal_hit(entry: Any, iter_episodic: Iterable[Any]) -> bool:
    """注入时"存在 rebuttal"判定（工程近似，M3-2 语义化）。

    新条目文本与**已被证伪/去稳定化的旧条目**（rebuttal_count>0 或
    labile）内容重叠（子串/包含）→ 视为对该旧记忆的反驳证据存在。
    无 LLM（同 doubt_ingest._find_overlapping_entry 工程惯例，溯源
    §4.2 标注）；异常 → False（fail-open）。
    """
    if entry is None:
        return False
    try:
        needle = (getattr(entry, "text", "") or "")[:120]
        if not needle:
            return False
        for other in iter_episodic:
            if other is entry:
                continue
            if (int(getattr(other, "rebuttal_count", 0) or 0) > 0
                    or bool(getattr(other, "labile", False))):
                otext = (getattr(other, "text", "") or "")[:120]
                if needle and (needle in otext or otext in needle):
                    return True
    except Exception:  # pylint: disable=broad-except
        return False
    return False


# ---------------------------------------------------------------------- #
#  三时相怀疑状态机
# ---------------------------------------------------------------------- #

class DoubtStateMachine:
    """三时相怀疑状态机（§2.1）。

    内存态：
      - ``_labile_window``：检索时相命中的 labile 窗口（{id(entry): 记录}）
        ——**纯内存**（同 react_surprise_history 先例：不落盘、不进快照、
        重启即失）；供 /status 观测与 M6 做梦期待核查清单累积。
      - 注入/巩固计数（观测）。

    Claim（§5.2，见 claims.json / MODULE_CLAIMS）：
      - 检索只读：retrieval_hit / retrieval_projection 绝不 setattr 条目；
      - 状态转移仅写侧：suspect/superseded 只能由 injection_check /
        consolidation_resolve（写侧时相）驱动；
      - 注入时怀疑：高 surprise 或 rebuttal 命中 → suspect + 验证链登记
        （验证链开关默认关——见 verification_chain）；
      - labile 非持久：labile 窗口是内存态，绝不落库。
    """

    def __init__(self, enabled: Optional[bool] = None,
                 verification_chain: Optional[VerificationChain] = None,
                 labile_ttl: Optional[float] = None,
                 surprise_factor: Optional[float] = None,
                 purpose_drift: Optional[PurposeDriftPhase] = None) -> None:
        # 治理开关：显式参数 > 环境变量（默认 1=开；0 → 零参与）
        self.enabled = doubt_injection_enabled(enabled)
        self.verification_chain = verification_chain
        self._labile_ttl = _labile_ttl() if labile_ttl is None else float(labile_ttl)
        self.surprise_factor = (
            _surprise_factor() if surprise_factor is None
            else max(0.0, float(surprise_factor)))
        # 目的检查时相（总任务书 §二.5：每轮 [doubt] purpose-drift 判定）。
        # None → 内部懒建（首次 purpose_drift_check / snapshot 时创建）；
        # 显式传入可注入共享实例或带自定义开关/目的源的实例。
        self.purpose_drift_phase = purpose_drift
        # labile 窗口（内存态，不落库）：id(entry) -> {entry, entered_at, score}
        self._labile_window: Dict[int, Dict[str, Any]] = {}
        # 观测计数
        self.injection_checks: int = 0
        self.injection_suspect_marked: int = 0
        self.consolidation_outcomes: Dict[str, int] = {}
        self._last_consolidation: Optional[dict] = None

    # ================================================================== #
    #  目的检查时相（总任务书 §二.5：每轮 [doubt] purpose-drift 判定）
    # ================================================================== #

    def _ensure_purpose_drift(self) -> PurposeDriftPhase:
        """内部懒建（None → 首次使用时创建；显式注入实例原样返回）。"""
        if self.purpose_drift_phase is None:
            self.purpose_drift_phase = PurposeDriftPhase()
        return self.purpose_drift_phase

    def purpose_drift_check(self, round_signals: dict,
                            purpose_text: str = "") -> dict:
        """每轮目的检查：委托 ``PurposeDriftPhase.judge``，返回闸门信号。

        闸门信号（Pan 警示：**无任何可被优化的分数**，只有是否偏离+理由）：
          ``{purpose_drift, verdict, answers{Q1..Q5}, reasons, purpose}``
        ——verdict 为 "drifted"/"uncertain" 时闸门亮（purpose_drift=True）；
        "uncertain"= 未判不是通过，reasons 说明缺什么信号，调用方据此补判。

        ``round_signals`` 键表与五问映射规则见 ``purpose_drift.py`` 模块
        docstring（Q1 流动 / Q2 过程 / Q3 活体 / Q4 熵核 / Q5 可回放——
        避免跑偏方案 §2.1 判据表）。

        fail-open：judge 自身已静默降级；此处再兜一层，异常绝不阻断调用方。
        """
        try:
            return self._ensure_purpose_drift().judge(
                round_signals, purpose_text)
        except Exception:  # pylint: disable=broad-except
            # 双保险 fail-open：目的检查异常 → 中性闸门（不阻断）
            logger.error("目的检查委托失败（fail-open）", exc_info=True)
            return _neutral_purpose_gate()

    # ================================================================== #
    #  检索时相（只读投影——绝不 setattr 条目）
    # ================================================================== #

    def retrieval_hit(self, entry: Any, score: Optional[float] = None) -> dict:
        """检索命中 → 条目进 labile 窗口（内存态）+ 返回原生字段只读视图。

        **绝不 setattr 条目**（只读四不变：labile 是时相状态，只存在于
        本窗口的内存记录里，不写条目、不进快照）。窗口记录供 M6 做梦期
        待核查清单累积（§2.4）。
        """
        view = get_rebuttal_consistency(entry)
        if self.enabled and entry is not None:
            self._labile_window[id(entry)] = {
                "entry": entry,
                "entered_at": time.time(),
                "score": float(score) if score is not None else None,
            }
            # 窗口 TTL 惰性清理（防无限增长）
            if len(self._labile_window) > 512:
                self._prune_labile_window()
        return view

    def retrieval_projection(
        self, scored: Sequence[Tuple[float, Any]],
        projection_fn: Optional[Callable[..., dict]] = None,
        precision_adapt: Any = None,
        consistency_provider: Optional[Callable[[Any], Optional[float]]] = None,
        max_items: int = 20,
    ) -> dict:
        """检索时怀疑投影（衔接 M2 ``project_suspicion``）。

        - ``projection_fn`` 缺省 → 内部调用 ``core.recall.suspicion.
          project_suspicion``（M2 只读投影：labile/rebuttal/verdict 区段）；
        - 每个命中条目先经 ``retrieval_hit`` 进 labile 窗口（内存态）；
        - 返回结构与 M2 ``project_suspicion`` **逐键同构**（§4.1 血管不换
          ——/recall 响应 suspicion 区段形态冻结，M2 测试断言精确键集）。

        原生字段（§2.3）的只读视图不注入本投影（形态冻结）；消费方经
        ``core.doubt.rebuttal_field.read_view / get_rebuttal_consistency``
        读取，或经 ``labile_window_snapshot``（内存态待核查清单）取用。
        """
        if projection_fn is None:
            from core.recall.suspicion import project_suspicion
            projection_fn = project_suspicion
        # 检索命中 → labile 窗口（内存态；绝不 setattr）
        for _score, entry in scored:
            self.retrieval_hit(entry, score=_score)
        return projection_fn(
            scored, precision_adapt=precision_adapt,
            consistency_provider=consistency_provider,
            max_items=max_items)

    def labile_window_snapshot(self, max_items: int = 20) -> List[dict]:
        """当前 labile 窗口（内存态待核查清单——§2.4 M6 做梦期消费）。

        只读；条目对象不返回（防调用方误写）——只返回只读视图。
        """
        self._prune_labile_window()
        out = []
        for rec in list(self._labile_window.values())[-max_items:]:
            entry = rec.get("entry")
            view = get_rebuttal_consistency(entry) if entry is not None else {}
            out.append({
                "text": (getattr(entry, "text", "") or "")[:120]
                if entry is not None else "",
                "score": rec.get("score"),
                "entered_at": rec.get("entered_at"),
                "rebuttal_consistency": view,
            })
        return out

    def _prune_labile_window(self) -> None:
        now = time.time()
        expired = [
            k for k, rec in self._labile_window.items()
            if (now - rec.get("entered_at", 0.0)) > self._labile_ttl
        ]
        for k in expired:
            self._labile_window.pop(k, None)

    # ================================================================== #
    #  注入时相（写侧：唯一合法驱动 stable → suspect）
    # ================================================================== #

    def injection_check(
        self, entry: Any, *, surprise: Optional[float] = None,
        j_target: Optional[float] = None, rebuttal_hit: bool = False,
        verification_chain: Optional[VerificationChain] = None,
        register_query: Optional[str] = None,
        registered_phase: str = "injection",
    ) -> str:
        """注入时怀疑（§2.1：store/feed 写入新条目时，写侧时相）。

        判定：高 surprise（surprise > factor × J_target）**或** rebuttal
        命中 → 标 ``suspect``（条目持久怀疑态，写侧）+ 登记验证链
        （验证链开关默认关——登记为 None 即跳过）。

        参数:
            entry: 新写入的条目（写侧路径已持有）。
            surprise: 条目惊讶度（process_turn 原生量）。
            j_target: 吸引子 J 设定点（attractor.j_target_norm；缺省则
                不做高 surprise 判定——信息不足不判）。
            rebuttal_hit: 调用方计算的反驳命中（compute_rebuttal_hit）。
            verification_chain: 验证链（None → 不登记；内部亦尊重其开关）。
            register_query: 验证链登记的 query（缺省 = entry.text）。

        返回:
            ``'suspect'`` | ``'stable'``（条目当前持久态）。
        """
        if not self.enabled:
            return getattr(entry, "doubt_state",
                           EntryDoubtState.STABLE.value) or EntryDoubtState.STABLE.value
        self.injection_checks += 1
        high = is_high_surprise(surprise, j_target, self.surprise_factor)
        if not (high or rebuttal_hit):
            return self._current_doubt_state(entry)
        self._set_doubt_state(entry, EntryDoubtState.SUSPECT,
                              phase=DoubtPhase.INJECTION)
        self.injection_suspect_marked += 1
        # 验证链登记（开关默认关 → register 返回 None，跳过）
        chain = verification_chain if verification_chain is not None \
            else self.verification_chain
        if chain is not None:
            try:
                chain.register(
                    entry_ref=str(id(entry)),
                    register_query=register_query
                    if register_query is not None
                    else (getattr(entry, "text", "") or ""),
                    registered_phase=registered_phase,
                )
            except Exception:  # pylint: disable=broad-except
                # fail-open：验证链登记异常绝不阻断写侧（G 模式以日志可见）
                logger.error("验证链登记失败（fail-open）", exc_info=True)
        return self._current_doubt_state(entry)

    # ================================================================== #
    #  巩固时相（写侧：接口本段定义；完整做梦循环属 M6）
    # ================================================================== #

    def consolidation_resolve(
        self, entry: Any, *, now: Optional[float] = None,
        window_seconds: float = 86400.0,
        reconsolidate_gain: float = 1.02, decay: float = 0.98,
        verdict: Optional[str] = None,
        verification_chain: Optional[VerificationChain] = None,
    ) -> dict:
        """巩固时 resolve_labile 接口契约（§2.4——M6 在梦期调用）。

        三种结局（与 ``reconsolidation.resolve_labile`` 同源）：
          - 冲突裁决（``verdict='conflict'`` 或条目有证伪证据 violated_by）
            → ``'rewritten'``：条目转 ``superseded``（调用方落 supersedes
            记录——M6 实现），confidence 按验证结果调整；
          - 验证通过（``verdict='confirm'``）或窗口内无证据
            → ``'kept'``：条目回 ``stable``，confidence ×1.02 重巩固；
          - 超时无证据 → ``'downgraded'``：条目回 ``stable``，confidence
            ×0.98 折损（轻微，保守）。

        全部动作为写侧时相：原生 ``rebuttal_consistency`` 以
        ``updated_by='consolidation'`` 更新（§2.3 铁律——合法写者）；
        状态转移经 ``_set_doubt_state``（唯一写侧入口）。

        返回:
            ``{outcome, doubt_state, confidence, rebuttal_consistency}``。
        """
        from core.doubt.reconsolidation import resolve_labile

        now = now if now is not None else time.time()
        violated_by = getattr(entry, "violated_by", None)
        # 裁决优先级：显式验证裁决 > 证伪证据 > 时间窗纯函数
        if verdict == VerdictType.CONFLICT.value or (
                verdict is None and violated_by):
            outcome = "rewritten"
            self._set_doubt_state(entry, EntryDoubtState.SUPERSEDED,
                                  phase=DoubtPhase.CONSOLIDATION)
            # labile 时相结束（复位——与 reconsolidation.resolve_labile
            # 的 rewritten 分支同行为）
            try:
                entry.labile = False
                entry.labile_since = None
            except Exception:  # pylint: disable=broad-except
                pass
            # 写侧更新原生字段（巩固期一致性重算——M6 语义化，本段用置信度）
            try:
                update_consistency(
                    entry, consistency=getattr(entry, "confidence", 0.0),
                    updated_by="consolidation", now=now)
            except Exception:  # pylint: disable=broad-except
                pass
        else:
            outcome = resolve_labile(
                entry, now=now, window_seconds=window_seconds,
                reconsolidate_gain=reconsolidate_gain, decay=decay)
            if verdict == VerdictType.CONFIRM.value and outcome != "rewritten":
                outcome = "kept"
            self._set_doubt_state(entry, EntryDoubtState.STABLE,
                                  phase=DoubtPhase.CONSOLIDATION)
            if outcome == "rewritten":
                self._set_doubt_state(entry, EntryDoubtState.SUPERSEDED,
                                      phase=DoubtPhase.CONSOLIDATION)
            try:
                update_consistency(
                    entry, consistency=getattr(entry, "confidence", 0.0),
                    updated_by="consolidation", now=now)
            except Exception:  # pylint: disable=broad-except
                pass
        self.consolidation_outcomes[outcome] = \
            self.consolidation_outcomes.get(outcome, 0) + 1
        # 验证结果登记（幂等；开关默认关 → None 跳过）
        chain = verification_chain if verification_chain is not None \
            else self.verification_chain
        if chain is not None:
            try:
                key = verification_key(
                    str(id(entry)),
                    getattr(entry, "text", "") or "", "injection")
                chain.submit_result(
                    key, verdict=verdict
                    if verdict is not None
                    else (VerdictType.CONFLICT.value
                          if outcome == "rewritten"
                          else VerdictType.CONFIRM.value))
            except Exception:  # pylint: disable=broad-except
                pass
        result = {
            "outcome": outcome,
            "doubt_state": self._current_doubt_state(entry),
            "confidence": round(float(getattr(entry, "confidence", 1.0) or 1.0), 4),
            "rebuttal_consistency": get_rebuttal_consistency(entry),
        }
        self._last_consolidation = result
        return result

    # ================================================================== #
    #  内部：状态转移（仅写侧时相调用）
    # ================================================================== #

    def _set_doubt_state(self, entry: Any, state: EntryDoubtState,
                         phase: DoubtPhase) -> None:
        """写侧状态转移（**仅 injection_check / consolidation_resolve 调用**）。

        检索时相绝不调用本方法（只读四不变）——这是"状态转移只能由写侧
        时相驱动"的机器防线。条目无 doubt_state 字段（旧条目）时 getattr
        兜底；setattr 失败 fail-open。
        """
        try:
            setattr(entry, "doubt_state", state.value)
        except Exception:  # pylint: disable=broad-except
            pass

    @staticmethod
    def _current_doubt_state(entry: Any) -> str:
        """条目当前持久怀疑态（缺字段回退 stable——旧条目/旧快照）。"""
        v = getattr(entry, "doubt_state", None)
        return v if v in (
            EntryDoubtState.STABLE.value,
            EntryDoubtState.SUSPECT.value,
            EntryDoubtState.SUPERSEDED.value,
        ) else EntryDoubtState.STABLE.value

    # ================================================================== #
    #  观测
    # ================================================================== #

    def snapshot(self) -> dict:
        """观测块（/status doubt_native 数据源）。"""
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "phases": {
                "retrieval": {
                    "labile_window_size": len(self._labile_window),
                },
                "injection": {
                    "checks": self.injection_checks,
                    "suspect_marked": self.injection_suspect_marked,
                },
                "consolidation": {
                    "outcomes": dict(self.consolidation_outcomes),
                },
            },
            "verification_chain": (
                self.verification_chain.snapshot()
                if self.verification_chain is not None else {"enabled": False}),
            "purpose_drift": self._ensure_purpose_drift().snapshot(),
            "params": {
                "surprise_factor": self.surprise_factor,
                "labile_window_ttl": self._labile_ttl,
            },
        }

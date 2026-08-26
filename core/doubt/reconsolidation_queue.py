# -*- coding: utf-8 -*-
"""core/doubt · 再巩固候选队列（R3 labile 平衡——语义决策 D-2026-08-18-01 代码化）

规格依据：
  - `四妹-更新版目的性审计-20260818.md` §二（labile 平衡 C：再巩固候选队列
    持久化（跨重启）+ 巩固期受控改写（三闸门）——阶段五/六）；
  - `总任务书-重编一口气-四妹-20260818.md` §二.3（C labile 平衡：再巩固
    候选队列持久化 + 巩固期受控改写——语义决策 D-2026-08-18-01 落地：
    **检索不塑形 + 巩固期再巩固**）。

语义决策 D-2026-08-18-01（本模块代码化的核心）：
  - **检索不塑形**：检索时相绝不产生任何改写/入队副作用（只读四不变与
    labile 窗口共存——与 state_machine 铁律同款）；
  - **巩固期再巩固**：再巩固只允许在巩固期（写侧时相）发生，且必须过
    三闸门。

三闸门（巩固期受控改写必须**全过**，任一不过 → 不改写）：
  G1 候选在队（``candidate_in_queue``）：条目（entry_key）在再巩固候选
     队列中——队列是再巩固的输入前提（判别力恢复 B 是队列质量的物理
     前提，审计 §二依赖链 B→C）；
  G2 巩固期触发（``consolidation_triggered``）：当前时相是巩固期写侧
     时相（``DoubtPhase.CONSOLIDATION``）——改写只能由巩固调度方触发；
  G3 改写受控（``rewrite_controlled``）：改写动作受限——只允许（a）写侧
     状态转移（经 state_machine 写侧入口，如 consolidation_resolve）与
     （b）append-only 演化史登记（process_core.append_transition——
     存储记过程、改写留演化史，P1-1 §3.3 同构）；绝不覆盖历史、绝不
     触碰检索路径。

队列持久化（跨重启）：候选队列存 JSON 文件（默认
``<data_dir>/reconsolidation/candidates.json``，env
``LMS_DOUBT_RECONSOLIDATION_QUEUE_PATH`` 可覆盖）——**写侧入队即落盘**
（原子写 tmp+rename，G 模式失败告警不静默）；启动时加载；重启后候选
仍在——再巩固不因进程重启丢失（labile 窗口是内存态可失，候选队列是
持久态不失——两者互补：窗口=时相观测，队列=再巩固计划）。

铁律（与 state_machine 同款）：
  - 入队只允许写侧时相（注入 suspect 标记后登记 / 巩固期补充登记）；
    检索时相调用入队 → 拒绝（返回 False，零副作用）；
  - 本模块所有只读接口（contains / peek / candidates / snapshot /
    size）绝不 setattr 条目（检索路径零改写保持）；
  - 三闸门裁决是机器防线：``maybe_rewrite`` 任一闸门不过 → 返回
    ``{"passed": False, "gate": <未过的闸门>}``，不执行任何改写动作。

治理开关：``LMS_DOUBT_RECONSOLIDATION_ENABLED`` 默认 1（labile 平衡是
R3 核心机制本体）；开关关 → 全部路径零参与（行为与开关引入前完全一致，
8/10 治理开关先例风格）。

设计约束（M1 core/store 同款）：**纯 stdlib**，不 import torch / fastapi /
LMS 运行时模块——可被轻量单测直接 import；对条目的读写经 getattr/setattr
（dict 亦兼容），不依赖具体条目类。全程 fail-open：任何异常 → 相应操作
安全跳过（日志可见），绝不阻断调用方。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional

from core.doubt.state_machine import DoubtPhase

logger = logging.getLogger("core.doubt.reconsolidation_queue")

# ---------------------------------------------------------------------- #
#  env 参数（§5.3：单一默认值来源；运行时读取，改动即时生效）
# ---------------------------------------------------------------------- #

_ENV_QUEUE_ENABLED = "LMS_DOUBT_RECONSOLIDATION_ENABLED"
_ENV_QUEUE_PATH = "LMS_DOUBT_RECONSOLIDATION_QUEUE_PATH"
_ENV_QUEUE_MAX = "LMS_DOUBT_RECONSOLIDATION_MAX"
_ENV_QUEUE_TTL = "LMS_DOUBT_RECONSOLIDATION_TTL"

#: 再巩固开关默认 **1**（labile 平衡是 R3 核心机制本体——审计排期阶段五/六）
_DEFAULT_ENABLED = True

#: 候选队列容量上限（防无限增长；超出丢弃最旧——FIFO 淘汰）
_DEFAULT_MAX_CANDIDATES = 256

#: 候选 TTL（秒）：入队后超过 TTL 未被巩固 → 惰性清除（默认 7 天——
#: 再巩固窗口不是无限等待；判别力恢复后 suspect 应在一周内被消化）
_DEFAULT_TTL = 7 * 86400.0

#: 持久化文件名（默认数据目录下）
_DEFAULT_QUEUE_FILENAME = "candidates.json"

#: 检索时相标记（入队拒绝用——语义决策 D-2026-08-18-01：检索不塑形）
_RETRIEVAL_PHASE = DoubtPhase.RETRIEVAL.value


def reconsolidation_enabled(explicit: Optional[bool] = None) -> bool:
    """再巩固开关解析：显式参数 > 环境变量 LMS_DOUBT_RECONSOLIDATION_ENABLED。

    布尔接受（不区分大小写）：1/true/yes/on 视为开，其余为关。
    默认开（机制本体）；0=关 → 全部路径零参与，行为与开关引入前完全一致。
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get(_ENV_QUEUE_ENABLED, "1")
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _queue_max() -> int:
    try:
        return max(1, int(os.environ.get(_ENV_QUEUE_MAX, "256") or 256))
    except (TypeError, ValueError):
        return _DEFAULT_MAX_CANDIDATES


def _queue_ttl() -> float:
    try:
        return max(1.0, float(os.environ.get(_ENV_QUEUE_TTL, "604800") or 604800))
    except (TypeError, ValueError):
        return _DEFAULT_TTL


def queue_path_default() -> Any:
    """候选队列默认路径：env 覆盖 > 数据目录。

    优先级：
      1. ``LMS_DOUBT_RECONSOLIDATION_QUEUE_PATH``（绝对路径直接用；
         相对路径按项目根解析——m7_dual_write 同工程惯例）；
      2. ``core.paths.get_data_dir()/reconsolidation/candidates.json``
         （LMS_DATA_DIR 可覆盖数据根）。
    """
    env = os.environ.get(_ENV_QUEUE_PATH, "").strip()
    if env:
        p = os.path.abspath(env)
        return p
    from core.paths import get_data_dir
    return os.path.join(
        str(get_data_dir()), "reconsolidation", _DEFAULT_QUEUE_FILENAME)


# ---------------------------------------------------------------------- #
#  纯函数：entry_key / 记录结构
# ---------------------------------------------------------------------- #

def entry_key(entry: Any = None, *, key: Optional[str] = None,
              text: str = "") -> str:
    """候选条目稳定键（跨重启稳定：内容哈希——条目无 id 字段）。

    优先级：显式 ``key`` > 条目 ``id`` 字段（若存在）> 文本 sha1 前缀。
    文本为空且无键 → 返回空串（调用方视为不可入队）。
    """
    if key is not None and str(key).strip():
        return str(key).strip()
    if entry is not None:
        eid = getattr(entry, "id", None)
        if eid is not None and str(eid).strip():
            return str(eid).strip()
    t = (text or "").strip()
    if not t and entry is not None:
        t = str(getattr(entry, "text", "") or "").strip()
    if not t:
        return ""
    return "sha1:" + hashlib.sha1(t.encode("utf-8", "replace")).hexdigest()[:24]

#: 模块函数别名（方法内 entry_key 参数遮蔽函数名——统一经别名调用）
_entry_key_of = entry_key


def _record(entry_key: str, *, reason: str, score: Optional[float],
            entered_at: float, detail: str = "") -> Dict[str, Any]:
    """构造一条候选记录（JSON 可序列化——不含条目对象，只存稳定键）。"""
    rec: Dict[str, Any] = {
        "entry_key": entry_key,
        "reason": reason or "",
        "score": round(float(score), 4) if score is not None else None,
        "entered_at": float(entered_at),
    }
    if detail:
        rec["detail"] = str(detail)
    return rec


# ---------------------------------------------------------------------- #
#  再巩固候选队列（持久化 + 三闸门受控改写）
# ---------------------------------------------------------------------- #

class ReconsolidationQueue:
    """再巩固候选队列（R3 labile 平衡；跨重启持久化）。

    内存态：
      - ``_candidates``：{entry_key: 候选记录}（有序，插入序=入队序）。
      - 持久化：``_path``（JSON 文件）——写侧入队/移除/巩固完成即落盘
        （原子写）；启动加载；损坏文件 → 空队列 + 告警（fail-open）。

    Claim（§5.2，见 claims.json / MODULE_CLAIMS）：
      - 检索不塑形：检索时相入队被拒（零副作用）；所有只读接口绝不
        setattr 条目（语义决策 D-2026-08-18-01）；
      - 三闸门机器防线：maybe_rewrite 任一闸门不过 → 不改写；
      - 队列持久化：写侧入队即落盘，重启后候选仍在；
      - 受控改写：巩固完成只留 append-only 演化史（process_core
        append_transition），绝不覆盖历史。
    """

    def __init__(self, enabled: Optional[bool] = None,
                 path: Optional[str] = None,
                 max_candidates: Optional[int] = None,
                 ttl: Optional[float] = None) -> None:
        # 治理开关：显式参数 > 环境变量（默认 1=开；0 → 零参与）
        self.enabled = reconsolidation_enabled(enabled)
        self.path = path if path is not None else queue_path_default()
        self.max_candidates = (
            _queue_max() if max_candidates is None else max(1, int(max_candidates)))
        self.ttl = _queue_ttl() if ttl is None else max(1.0, float(ttl))
        # 候选有序字典：entry_key -> 记录（插入序 = 入队序）
        self._candidates: Dict[str, Dict[str, Any]] = {}
        self.load_failures: int = 0
        self._loaded = False
        self._load()

    # ================================================================== #
    #  持久化（原子写：tmp + rename；G 模式失败告警不静默）
    # ================================================================== #

    def _load(self) -> None:
        """启动/构造时加载候选队列（损坏 → 空队列 + 告警，fail-open）。"""
        if not self.enabled:
            self._loaded = True
            return
        self._loaded = True
        if not self.path:
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                raise ValueError("queue 根结构非对象")
            items = raw.get("candidates", [])
            if not isinstance(items, list):
                raise ValueError("candidates 非列表")
            now = time.time()
            for rec in items:
                if not isinstance(rec, dict) or not rec.get("entry_key"):
                    continue
                if (now - float(rec.get("entered_at", 0.0))) > self.ttl:
                    continue  # 过期候选不加载
                self._candidates[str(rec["entry_key"])] = dict(rec)
        except FileNotFoundError:
            pass  # 首次运行：无文件 = 空队列
        except Exception as e:  # pylint: disable=broad-except
            self.load_failures += 1
            logger.error(
                "再巩固候选队列加载失败（fail-open→空队列）：%s——检查 %s",
                e, self.path)

    def _save(self) -> bool:
        """原子落盘（tmp + rename）；失败告警（G 模式），返回是否成功。"""
        if not self.enabled or not self.path:
            return True
        try:
            payload = json.dumps(
                {"candidates": list(self._candidates.values())},
                ensure_ascii=False)
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(
                prefix=".candidates-", suffix=".tmp", dir=d or ".")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(payload)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.path)
            finally:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
            return True
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "再巩固候选队列落盘失败（fail-open，队列仅内存态）：%s", e)
            return False

    # ================================================================== #
    #  入队（**只允许写侧时相**——语义决策 D-2026-08-18-01：检索不塑形）
    # ================================================================== #

    def enqueue(self, entry: Any = None, *, entry_key: Optional[str] = None,
                text: str = "", reason: str = "",
                score: Optional[float] = None, detail: str = "",
                phase: Any = DoubtPhase.INJECTION,
                now: Optional[float] = None) -> bool:
        """把条目登记为再巩固候选（写侧时相；**写侧入队即落盘**）。

        闸门（写侧铁律）：
          - 开关关 → False（零参与）；
          - ``phase`` 是检索时相（DoubtPhase.RETRIEVAL）→ 拒绝
            （检索不塑形——语义决策 D-2026-08-18-01；零副作用）；
          - entry_key 解析失败（无键无文本）→ False（fail-open 不臆造）。

        入队后持久化；超出容量上限 → 丢弃最旧候选（FIFO）。
        返回是否成功入队。
        """
        if not self.enabled:
            return False
        if phase == _RETRIEVAL_PHASE:
            logger.warning(
                "再巩固入队被拒：检索时相零副作用（检索不塑形——"
                "D-2026-08-18-01）")
            return False
        k = _entry_key_of(entry, key=entry_key, text=text)
        if not k:
            return False
        now = now if now is not None else time.time()
        self._candidates[k] = _record(
            k, reason=reason, score=score, entered_at=now, detail=detail)
        self._prune()
        self._save()
        return True

    # ================================================================== #
    #  移除 / 查询（只读接口绝不 setattr 条目——检索路径零改写保持）
    # ================================================================== #

    def remove(self, entry: Any = None, *, entry_key: Optional[str] = None,
               text: str = "") -> bool:
        """把条目移出候选队列（巩固完成/放弃时调用；落盘）。"""
        if not self.enabled:
            return False
        k = _entry_key_of(entry, key=entry_key, text=text)
        if not k or k not in self._candidates:
            return False
        self._candidates.pop(k, None)
        self._save()
        return True

    def contains(self, entry: Any = None, *, entry_key: Optional[str] = None,
                 text: str = "") -> bool:
        """G1 闸门：条目是否在候选队列中（只读，绝不 setattr 条目）。"""
        if not self.enabled:
            return False
        k = _entry_key_of(entry, key=entry_key, text=text)
        return bool(k) and k in self._candidates

    def size(self) -> int:
        """当前候选数（只读）。"""
        return len(self._candidates) if self.enabled else 0

    def peek(self, max_items: int = 20) -> List[Dict[str, Any]]:
        """候选记录视图（只读；按入队序返回拷贝，不含条目对象）。"""
        if not self.enabled:
            return []
        self._prune()
        return [dict(r) for r in list(self._candidates.values())[:max_items]]

    # 与 recall_scheduler 命名习惯一致的别名
    def candidates(self, max_items: int = 20) -> List[Dict[str, Any]]:
        """候选记录视图别名（只读；与 peek 同构）。"""
        return self.peek(max_items=max_items)

    # ================================================================== #
    #  巩固期受控改写（三闸门机器防线）
    # ================================================================== #

    def maybe_rewrite(self, entry: Any = None, *, entry_key: Optional[str] = None,
                      text: str = "", phase: Any = DoubtPhase.CONSOLIDATION,
                      rewrite_fn: Optional[Callable[[Any], Any]] = None,
                      detail: str = "", now: Optional[float] = None) -> Dict[str, Any]:
        """巩固期受控改写（**三闸门全过才执行**，任一不过 → 不改写）。

        闸门（顺序裁决，返回未过的第一个闸门）：
          G1 ``candidate_in_queue``     ：条目在再巩固候选队列中；
          G2 ``consolidation_triggered``：``phase`` 是巩固期写侧时相
            （DoubtPhase.CONSOLIDATION）——改写只能由巩固调度方触发；
          G3 ``rewrite_controlled``     ：改写动作受控——``rewrite_fn``
            可调用（调用方经 state_machine 写侧入口执行状态转移，如
            consolidation_resolve）**或**走内部受控动作（append-only
            演化史登记，process_core.append_transition——绝不覆盖历史）。

        全部闸门过 → 执行改写动作，然后：
          - 演化史登记（append-only；存储记过程、改写留演化史——P1-1 同构）；
          - 条目移出候选队列（巩固完成）并落盘。

        返回：
          - 未过闸门：``{"passed": False, "gate": "candidate_in_queue" |
            "consolidation_triggered" | "rewrite_controlled"}``；
          - 全过：``{"passed": True, "gate": "all",
            "rewritten": True, "entry_key": <k>}``。

        本方法**绝不 setattr 条目**（改写动作委托给 rewrite_fn / 纯追加
        登记）；检索路径零改写保持。
        """
        if not self.enabled:
            return {"passed": False, "gate": "reconsolidation_enabled",
                    "rewritten": False}
        k = _entry_key_of(entry, key=entry_key, text=text)
        # G1 候选在队
        if not k or k not in self._candidates:
            return {"passed": False, "gate": "candidate_in_queue",
                    "rewritten": False}
        # G2 巩固期触发（写侧时相）
        if phase != DoubtPhase.CONSOLIDATION.value:
            return {"passed": False, "gate": "consolidation_triggered",
                    "rewritten": False}
        # G3 改写受控：rewrite_fn 缺省 → 内部受控动作（append-only 登记）
        try:
            if rewrite_fn is not None:
                rewrite_fn(entry)
            else:
                self._controlled_append(entry, detail=detail, now=now)
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "巩固期受控改写执行失败（fail-open，不动队列）：%s", e)
            return {"passed": False, "gate": "rewrite_controlled",
                    "rewritten": False, "error": str(e)}
        # 巩固完成：移出队列（候选被消化——再巩固闭环）并落盘
        self._candidates.pop(k, None)
        self._save()
        return {"passed": True, "gate": "all", "rewritten": True,
                "entry_key": k}

    @staticmethod
    def _controlled_append(entry: Any, *, detail: str = "",
                           now: Optional[float] = None) -> None:
        """内部受控动作：append-only 演化史登记（绝不覆盖历史）。

        经 ``core.store.process_core`` 的 ``append_transition`` 落
        ``evolution.history``（写者 consolidation——P1-1 演化史合法写者）。
        条目无 evolution 字段（旧条目）→ 自动初始化（M7 兼容）。
        纯追加：不改文本/置信度/怀疑态——真正的状态转移由调用方经
        state_machine 写侧入口完成（本模块只做再巩固的登记与裁决）。
        """
        from core.store.process_core import append_transition, make_transition
        append_transition(
            entry,
            make_transition("reconsolidated", at=now,
                            detail=detail or "再巩固候选消化（受控登记）"),
            updated_by="consolidation")

    # ================================================================== #
    #  内部：TTL 清理（惰性）
    # ================================================================== #

    def _prune(self) -> None:
        """惰性 TTL 清理 + 容量上限（FIFO 淘汰最旧）。"""
        if not self.enabled:
            return
        now = time.time()
        expired = [
            k for k, rec in self._candidates.items()
            if (now - float(rec.get("entered_at", 0.0))) > self.ttl
        ]
        for k in expired:
            self._candidates.pop(k, None)
        while len(self._candidates) > self.max_candidates:
            # dict 保插入序：弹最旧（首个）
            self._candidates.pop(next(iter(self._candidates)))

    # ================================================================== #
    #  观测
    # ================================================================== #

    def snapshot(self) -> dict:
        """观测块（/status doubt_native 数据源候选区段）。"""
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "path": self.path,
            "size": self.size(),
            "max_candidates": self.max_candidates,
            "ttl_seconds": self.ttl,
            "load_failures": self.load_failures,
        }

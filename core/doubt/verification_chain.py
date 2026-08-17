# -*- coding: utf-8 -*-
"""core/doubt · 验证链原生骨架（M3-1 · §2.2 接口/事件/幂等键协议）

规格依据：`四妹-LMS核心重写规格v2-20260817.md` §2.2（验证链原生——
防伪独立四维 + 矛盾判定三选一 + 幂等防伪）。

本段（M3-1）交付**骨架**：接口、事件类型、幂等键协议、防伪独立四维的
脚手架。**判定细节（矛盾判定三选一 / 元数据排除的语义实现）属 M3-2**——
本模块的 ``submit_result`` 只按协议登记结果，不做语义判定；调用方
（M3-2 / 外部验证器）给出 verdict。

防伪独立四维（SelfCheckGPT 工程同构，§2.2）：
  1. 端点独立   —— 验证请求走独立路由/通道（本模块 channel 字段承载）；
  2. query 独立 —— 验证 query 与登记 query **不同构造**（换措辞/换角度）：
     ``build_independent_query`` 保证 verify_query ≠ register_query；
  3. 批次独立   —— 验证分批错开（``next_batch`` 时间窗批次；
     验证必须用与登记**不同**的批次 id）；
  4. 信号独立   —— 验证信号与登记信号物理隔离（``channel_for``：
     登记走 ``register.*``，验证走 ``verify.*``，互不盖章）。

幂等防伪协议（根因③根治）：
  - 验证请求带幂等键（**登记时生成**，随验证请求回传）；
  - 服务端收到验证请求先按幂等键查重：已处理 → 直接返回原结果，
    不重复登记（``lookup`` / ``submit_result`` 幂等）；
  - **"客户端超时" ≠ "服务端未写入"**：客户端重试必须带同一幂等键，
    禁止盲目重发（同键重试命中原结果）；
  - 验证结果登记本身也幂等：同一条目同一次验证只产生一条记录。

治理开关（§5.3 / §7.2）：``LMS_VERIFICATION_CHAIN_ENABLED`` 默认 **0**
（关）——写侧默认保守：验证链默认关闭推进，假阳性演练通过才 env 开启。
开关关 → ``register`` 返回 None（零参与，行为与开关引入前完全一致）。

设计约束（M1 ``core/store`` 同款：纯 stdlib，可被轻量单测直接 import）：
  - 不 import torch / fastapi / LMS 运行时模块。
"""

from __future__ import annotations

import enum
import hashlib
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger("core.doubt.verification_chain")

# ---------------------------------------------------------------------- #
#  事件类型 / 判定类型（协议层）
# ---------------------------------------------------------------------- #

class VerificationEventType(str, enum.Enum):
    """验证链事件类型（观测/事件流，§4.3 落沙必发事件的数据源）。"""

    VERIFY_REQUESTED = "verify_requested"   # 验证请求已登记（条目标 suspect 时）
    VERIFY_RESULT = "verify_result"         # 验证结果已登记（幂等）
    VERIFY_RESOLVED = "verify_resolved"     # 条目已裁决（confirm/supersedes）


class VerdictType(str, enum.Enum):
    """验证裁决（§2.4 做梦期 resolve_labile 两分支）。

    M3-2 实现矛盾判定三选一后给出 verdict；本段只定义协议。
    """

    CONFIRM = "confirm"      # 验证通过 → 条目回 stable
    CONFLICT = "conflict"    # 验证冲突 → 登记 supersedes
    NONE = "none"            # 无裁决（未验证/待验证）


class ConflictKind(str, enum.Enum):
    """矛盾类型（§2.2 矛盾判定三选一——M3-2 实现判定细节）。

    只有三类明确矛盾才登记 conflict；元数据碰撞（时间戳前缀/同义复述/
    来源互补）一律 ``NOT_A_CONFLICT``（假阳性演练红线）。
    """

    DIRECTIONAL = "directional"   # 方向矛盾：同一事实两条断言结论相反
    NUMERIC = "numeric"           # 数值矛盾：同一量数值断言区间不相交
    NEGATION = "negation"         # 否定矛盾：一条断言否定另一条的存在性
    NOT_A_CONFLICT = "not_a_conflict"  # 元数据碰撞/同义——不判矛盾


# ---------------------------------------------------------------------- #
#  env 参数（§5.3：单一默认值来源；运行时读取，改动即时生效）
# ---------------------------------------------------------------------- #

_ENV_ENABLED = "LMS_VERIFICATION_CHAIN_ENABLED"

#: 验证链开关默认 **false**（§5.3：写侧默认保守 + 坑 4 教训：
#: 默认关闭、显式开启；开启必须过假阳性演练）
_DEFAULT_ENABLED = False

#: 窗口内同键去重（幂等防护窗口；同登记/同验证的"重试"在窗口内命中原结果）
_DEFAULT_WINDOW = 3600.0

#: 批次时间窗（防伪独立四维③：同窗登记同批，验证必须换批）
_DEFAULT_BATCH_SPAN = 300.0


def verification_chain_enabled(explicit: Optional[bool] = None) -> bool:
    """治理开关解析：显式参数 > 环境变量 LMS_VERIFICATION_CHAIN_ENABLED。

    布尔接受（不区分大小写）：1/true/yes/on 视为开，其余为关。
    默认关（§5.3：写侧默认保守——宁可拦错不轻信，但默认不开新机制）。
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get(_ENV_ENABLED, "0")
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _batch_span() -> float:
    try:
        return max(1.0, float(os.environ.get(
            "LMS_VERIFICATION_BATCH_SPAN", str(_DEFAULT_BATCH_SPAN))))
    except (TypeError, ValueError):
        return _DEFAULT_BATCH_SPAN


# ---------------------------------------------------------------------- #
#  幂等键协议
# ---------------------------------------------------------------------- #

#: 幂等键前缀（与 store 幂等键语义同族：查重 → 处理 → 登记）
_KEY_PREFIX = "verif_"


def verification_key(entry_ref: str, register_query: str,
                     registered_phase: str) -> str:
    """验证请求幂等键（登记时生成，随验证请求回传）。

    键 = 确定性指纹（entry_ref + register_query + phase）——同一条目
    同一次验证（同 query 同阶段）重发命中同一键；重试必须带同一键
    （禁止盲目重发——盲目重发 = 换键 = 服务端视为新登记）。
    """
    canonical = "\x00".join([
        str(entry_ref or ""), str(register_query or ""),
        str(registered_phase or ""),
    ])
    return _KEY_PREFIX + hashlib.sha256(
        canonical.encode("utf-8")).hexdigest()


def generate_request_id() -> str:
    """请求 id（随机；供 rebuttal_id / 事件去重）。"""
    return uuid.uuid4().hex


# ---------------------------------------------------------------------- #
#  数据模型
# ---------------------------------------------------------------------- #

@dataclass
class VerificationRequest:
    """一条验证请求（登记时生成，随验证请求回传——幂等键协议载体）。

    防伪独立四维字段（§2.2）：
      - ``register_query``：登记 query（原始措辞）；
      - ``verify_query``  ：独立验证 query（换措辞/换角度，**不同构造**）；
      - ``batch_id``      ：登记批次（验证必须用**不同**批次，防批量污染）；
      - ``channel``       ：信号通道（``register.*``；验证信号走 ``verify.*``）。
    """

    idempotency_key: str
    entry_ref: str                       # 条目标识（id/文本指纹）
    registered_at: float
    registered_phase: str                # 'injection' | 'consolidation'
    register_query: str
    verify_query: str
    batch_id: str
    channel: str
    request_id: str = field(default_factory=generate_request_id)


@dataclass
class VerificationResult:
    """一条验证结果（登记本身幂等：同键同验证只一条记录）。

    ``conflict_kind`` 为 None 表示无矛盾判定信息（M3-2 填三选一）；
    ``metadata_excluded=True`` 表示元数据排除命中（不判矛盾——假阳性红线）。
    """

    idempotency_key: str
    verdict: str                         # VerdictType 值
    conflict_kind: Optional[str] = None  # ConflictKind 值（None=未判定）
    metadata_excluded: bool = False
    resolved_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------- #
#  验证链（进程内注册表；幂等；事件记录）
# ---------------------------------------------------------------------- #

class VerificationChain:
    """验证链骨架（进程内状态；同 gap_registry 先例：重启即失、快照不落盘）。

    Claim（§5.2，见 claims.json / MODULE_CLAIMS）：
      - 幂等：同幂等键重发不重复登记；已处理键直接返回原结果；
      - 结果登记幂等：同一条目同一次验证只产生一条记录；
      - 防伪独立：verify_query ≠ register_query；验证批次 ≠ 登记批次；
        验证通道 ≠ 登记通道。
    """

    def __init__(self, enabled: Optional[bool] = None,
                 window: Optional[float] = None) -> None:
        self._enabled = verification_chain_enabled(enabled)
        self._window = _DEFAULT_WINDOW if window is None else float(window)
        self._requests: Dict[str, VerificationRequest] = {}
        self._results: Dict[str, VerificationResult] = {}
        self._events: Deque[dict] = deque(maxlen=200)
        self._batch_span = _batch_span()

    # -- 开关 / 观测 ---------------------------------------------------- #

    @property
    def enabled(self) -> bool:
        """验证链开关（默认 false——§5.3 写侧默认保守）。"""
        return self._enabled

    def _expired(self, ts: float) -> bool:
        return (time.time() - ts) > self._window

    # -- 登记（幂等键协议：登记时生成，随验证请求回传） ------------------- #

    def register(
        self, *, entry_ref: str, register_query: str,
        registered_phase: str = "injection",
        verify_query: Optional[str] = None,
        batch_id: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> Optional[VerificationRequest]:
        """登记一条验证请求（开关关 → None，零参与）。

        - 幂等键 = ``verification_key(entry_ref, register_query, phase)``：
          同键重发（窗口内）返回**原请求**，不重复登记；
        - ``verify_query`` 缺省 → ``build_independent_query(register_query)``
          （防伪独立四维②：与登记 query 不同构造）；
        - ``batch_id`` 缺省 → 当前时间窗批次（防伪独立四维③）；
        - ``channel`` 缺省 → ``register.{phase}``（防伪独立四维④）。
        """
        if not self._enabled:
            return None
        key = verification_key(entry_ref, register_query, registered_phase)
        existing = self._requests.get(key)
        if existing is not None and not self._expired(existing.registered_at):
            # 幂等命中：同键重发不重复登记，原样返回原请求
            self._record_event(VerificationEventType.VERIFY_REQUESTED, {
                "idempotency_key": key, "replayed": True,
                "entry_ref": entry_ref,
            })
            return existing
        req = VerificationRequest(
            idempotency_key=key,
            entry_ref=str(entry_ref)[:200],
            registered_at=time.time(),
            registered_phase=registered_phase,
            register_query=str(register_query)[:300],
            verify_query=str(verify_query or self.build_independent_query(
                register_query))[:300],
            batch_id=batch_id or self.next_batch(),
            channel=channel or f"register.{registered_phase}",
        )
        self._requests[key] = req
        self._record_event(VerificationEventType.VERIFY_REQUESTED, {
            "idempotency_key": key, "replayed": False,
            "entry_ref": req.entry_ref, "phase": registered_phase,
            "batch_id": req.batch_id, "channel": req.channel,
        })
        return req

    # -- 结果登记（幂等：同键同验证只一条记录） --------------------------- #

    def submit_result(
        self, request_key: str, *, verdict: str,
        conflict_kind: Optional[str] = None,
        metadata_excluded: bool = False,
    ) -> Optional[VerificationResult]:
        """登记验证结果（**服务端先按幂等键查重**：已处理 → 直接返回
        原结果，不重复登记——"客户端超时"≠"服务端未写入"，重试同键
        命中原结果）。

        Args:
            request_key: 验证请求幂等键（登记时生成、随验证请求回传）。
            verdict: ``VerdictType`` 值（M3-2 判定细节给出）。
            conflict_kind: ``ConflictKind`` 值（None=未做三选一判定）。
            metadata_excluded: 元数据排除命中（不判矛盾——假阳性红线）。

        Returns:
            VerificationResult；开关关 / 键未登记 → None。
        """
        if not self._enabled:
            return None
        if request_key not in self._requests:
            return None
        existing = self._results.get(request_key)
        if existing is not None:
            # 幂等：同键同验证只产生一条记录
            self._record_event(VerificationEventType.VERIFY_RESULT, {
                "idempotency_key": request_key, "replayed": True,
            })
            return existing
        res = VerificationResult(
            idempotency_key=request_key,
            verdict=verdict,
            conflict_kind=conflict_kind,
            metadata_excluded=bool(metadata_excluded),
        )
        self._results[request_key] = res
        self._record_event(VerificationEventType.VERIFY_RESULT, {
            "idempotency_key": request_key, "replayed": False,
            "verdict": verdict, "conflict_kind": conflict_kind,
            "metadata_excluded": bool(metadata_excluded),
        })
        return res

    # -- 查询（只读） ----------------------------------------------------- #

    def lookup(self, request_key: str) -> Optional[VerificationResult]:
        """按幂等键查重（只读、无副作用——供 §5.2 测试断言）。"""
        return self._results.get(request_key)

    def request_lookup(self, request_key: str) -> Optional[VerificationRequest]:
        """按幂等键查请求（只读）。"""
        return self._requests.get(request_key)

    def pending_count(self) -> int:
        """当前待验证请求数（有请求无结果）。"""
        if not self._enabled:
            return 0
        return sum(
            1 for key in self._requests if key not in self._results)

    # -- 防伪独立四维脚手架（M3-2 细化判定；本段定义协议） ------------------ #

    @staticmethod
    def build_independent_query(register_query: str) -> str:
        """独立验证 query（防伪独立四维②：换措辞/换角度，不同构造）。

        M3-2 将实现真正的语义换问（LLM/规则重述）；本段提供确定性
        脚手架：登记 query 原样保留 + 独立角度后缀——保证
        ``verify_query != register_query``（协议硬约束），并标注
        待 M3-2 语义化。
        """
        q = str(register_query or "").strip()
        if not q:
            return "（独立核查：该记忆是否仍被相信？换个角度重述）"
        return f"{q[:200]}？换个角度重述核查"

    def next_batch(self) -> str:
        """批次 id（时间窗：同窗登记同批——防批量污染的分批基础）。

        验证请求必须用与登记**不同**的批次 id（调用方在验证时取
        ``next_batch()`` 的新值即可天然错开）。
        """
        return f"b{int(time.time() // self._batch_span)}"

    def independent_batch(self, registered_batch: Optional[str]) -> str:
        """为验证请求生成与登记**不同**的批次（防伪独立四维③）。"""
        b = self.next_batch()
        if registered_batch is not None and b == registered_batch:
            return f"{b}.v"
        return b

    @staticmethod
    def channel_for(signal: str) -> str:
        """信号通道（防伪独立四维④：验证信号与登记信号物理隔离）。

        - 登记信号：``register.{phase}``（如 ``register.injection``）；
        - 验证信号：``verify.{kind}``（如 ``verify.result`` / ``verify.request``）。
        登记与验证**永不共享通道**——验证结果不给自己盖章。
        """
        return str(signal).replace("register.", "verify.", 1) \
            if str(signal).startswith("register.") else f"verify.{signal}"

    # -- 事件（观测/事件流数据源；§4.3 落沙必发事件） ---------------------- #

    def _record_event(self, event_type: VerificationEventType,
                      payload: Dict[str, Any]) -> None:
        self._events.append({
            "type": event_type.value,
            "ts": time.time(),
            **payload,
        })

    def events(self, last_n: int = 20) -> List[dict]:
        """最近事件（观测；事件流发布方取用——§4.3 落沙必发事件）。"""
        return list(self._events)[-last_n:]

    def snapshot(self) -> dict:
        """观测块（/status doubt_native.verification_chain 数据源）。"""
        if not self._enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "pending": self.pending_count(),
            "resolved": len(self._results),
            "registered": len(self._requests),
            "events_n": len(self._events),
        }

    def clear(self) -> None:
        """清空注册表（测试用；生产不调用——窗口自动过期）。"""
        self._requests.clear()
        self._results.clear()
        self._events.clear()

# -*- coding: utf-8 -*-
"""core/doubt · 目的检查框架——purpose-drift 时相（总任务书 §二.5）

规格依据：
  - `总任务书-重编一口气-四妹-20260818.md` §二.5（dandan 指示：目的时相
    与质疑系统融合——每轮 ``[doubt] purpose-drift`` 判定：消费 dsh-goal
    目的 + 目的手册五问）；
  - `四妹-避免跑偏方案-20260818.md` §2.1（任务级五问判据表 Q1-Q5 + 判定
    规则铁律：任一判"否"→ 偏离；"不确定"= 未判，不是通过，必须补判到
    "是/否"——堵住"灵活解释"的洞）。

论文依据（三行）：
  1. 负反馈比较器——目的 = 设定点，每轮把当前活动与目的设定点比较，
     偏差即偏离信号（gate 输出"是否偏离+理由"，不是误差幅值）；
  2. FEP prior 吸引子——目的先验是吸引子，活动状态投影回目的空间，
     偏离 = 状态落出 prior 吸引盆（判定用状态隶属，不用连续分数）；
  3. StateAct 每步重申——每步执行前重申目的（本模块每轮 judge 即
     "重申"动作，输出闸门信号供调用方决定继续/转向/补判）。

**Pan 警示（硬约束）**：本模块**严禁输出任何可被优化的分数**（如
0-100 符合度/评分）。输出只有"是否偏离 + 理由"两类语义：``verdict``
（aligned/drifted/uncertain）+ 每题 ``ok``/``reason``。数值计数
（``drift_count``、``doubt_events``）是**事件计数**不是符合度分数——
判别器不优化这些计数。

env 参数表（§5.3：运行时读取，改动即时生效）：
  - ``LMS_PURPOSE_DRIFT_ENABLED`` 默认 **1**（开）；0=关 → ``judge`` 返回
    中性闸门、``snapshot`` 的 ``enabled`` 为 False（开关关 → 零参与，
    行为与开关引入前完全一致——8/10 治理开关先例风格）。
  - ``LMS_PURPOSE_GOAL_TEXT`` dsh-goal 目的文本来源；缺省用内置总目的
    原文：dandan 原始哲学四词锚点——「流动上下文 / 活体记忆 / 记录涌现
    过程 / 熵核心」。

五问定义（Q1-Q5，源自避免跑偏方案 §2.1 判据表）：
  - Q1 流动：产出是"可被下次使用、随状态演化的活结构"，非一次性静态档案；
  - Q2 过程：记录"怎么到这一步"（惊讶/怀疑/转向/悬案），非只记录结论；
  - Q3 活体：记忆机制真在动（怀疑→修正/检索→再巩固/遗忘→重巩固真实发生），
    而非"只是存在"（接口接了 ≠ 活着）；
  - Q4 熵核：熵/惊讶参与决策（唤醒/思考/注入调度），而非只作展示；
  - Q5 可回放：涌现过程可回放、可注入、可审计（有记忆形态投影）。

round_signals → 每问映射规则（judge 的确定性判据；全部缺省视为中性，
即缺键按 False/0 处理）：
  - 键表（调用方提供，可扩展；未知键忽略）：
      episodic_added(bool)           本轮有经历条目写入（store/feed 写侧）
      doubt_events(int)              本轮怀疑事件数（>0 = 有过程）
      verification_chain_active(bool) 验证链本轮参与（登记/验证/结果——
                                      活跃即产生 VERIFY-* provenance）
      suspect_marked(bool)           本轮有条目被标 suspect（注入时写侧
                                      状态转移）
      surprise_in_decisions(bool)    熵/惊讶参与了决策（allostatic /
                                      injection 用了 surprise）
      lifecycle_trace(bool)          M5 9 步生命周期记录存在
      reconsolidated(bool)           再巩固真实发生
      retrieval_projected(bool)      检索投影发生（只读侧——不构成写侧）
      dream_consolidated(bool)       梦期巩固发生
      provenance(bool)               VERIFY-* provenance 存在（验证链记录
                                      侧；等价于 verification_chain_active）
      conclusion_only(bool)          本轮只记录结论未记录过程 → Q2 判"否"
      memory_idle(bool)              记忆机制"只是存在"本轮没动 → Q3 判"否"
      surprise_display_only(bool)    熵/惊讶只作展示未参与决策 → Q4 判"否"
      not_replayable(bool)           涌现过程不可回放/注入/审计 → Q5 判"否"
  - Q1 流动：任一写侧活动（episodic_added / reconsolidated /
      dream_consolidated / suspect_marked）→ 是；本轮无任何写侧活动 → 否
      （规格：episodic_added 或 consolidated → 是；本轮无任何写侧活动
      → 否/不确定——本实现取"否"，理由见下方"判定规则"）；
  - Q2 过程：doubt_events>0 或 lifecycle_trace 或 surprise_in_decisions
      → 是；elif conclusion_only → 否；else → 不确定；
  - Q3 活体：doubt_events>0 或 reconsolidated 或 verification_chain_active
      → 是；elif memory_idle → 否；else → 不确定；
  - Q4 熵核：surprise_in_decisions → 是；elif surprise_display_only → 否；
      else → 不确定；
  - Q5 可回放：lifecycle_trace 或 provenance 或 verification_chain_active
      （活跃验证链记录 VERIFY-* provenance）→ 是；elif not_replayable
      → 否；else → 不确定。

判定规则（铁律，与避免跑偏方案 §2.1 一致）：
  - 每题三分：是（ok=True）/ 否（ok=False）/ 不确定（ok=None）。有正信号
    判"是"、有显式负信号判"否"、无信号判"不确定 + 缺什么信号的理由"；
    正信号优先于负信号（一轮确实发生过过程即判"是"，不因补记结论而误否）。
  - **"不确定"= 未判，不是通过**：judge 在无信号时不臆造，返回
    "不确定 + 缺什么信号的理由"——调用方据此**补判**（补齐信号重跑 judge）。
  - 最终 verdict：存在任一"否" → ``drifted``；全"是" → ``aligned``；
    其余（含未补判的"不确定"）→ ``uncertain``。
  - 闸门信号 ``purpose_drift`` = (verdict != "aligned")：**drifted 与
    uncertain 都亮闸门**——"不确定"不是通过，必须补判到"是/否"后才算
    对齐（宁可拦错不轻信，写侧默认保守同款）。
  - Q1 特殊：无任何写侧活动直接判"否"（不是"不确定"）——"本轮什么都没
    写"本身是确定的负信号（Q1 反例：结论定格/静态档案），不需要补判。

设计约束（M1 core/store 同款：纯 stdlib，可被轻量单测直接 import）：
  - 不 import torch / fastapi / LMS 运行时模块；
  - fail-open：judge/snapshot 任何异常静默降级为中性闸门，绝不阻断调用方。
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger("core.doubt.purpose_drift")

# ---------------------------------------------------------------------- #
#  五问常量（Q1-Q5 定义——避免跑偏方案 §2.1 判据表）
# ---------------------------------------------------------------------- #

QUESTIONS: Dict[str, str] = {
    "Q1": "流动：产出是可被下次使用、随状态演化的活结构，非一次性静态档案",
    "Q2": "过程：记录「怎么到这一步」（惊讶/怀疑/转向/悬案），非只记录结论",
    "Q3": "活体：记忆机制真在动（怀疑→修正/检索→再巩固/遗忘→重巩固真实发生），"
          "而非「只是存在」",
    "Q4": "熵核：熵/惊讶参与决策（唤醒/思考/注入调度），而非只作展示",
    "Q5": "可回放：涌现过程可回放、可注入、可审计（有记忆形态投影）",
}

#: 五问判定顺序（judge 输出与 reasons 的固定顺序——可回放可审计）
QUESTION_IDS: List[str] = ["Q1", "Q2", "Q3", "Q4", "Q5"]

# ---------------------------------------------------------------------- #
#  env 参数（§5.3：单一默认值来源；运行时读取，改动即时生效）
# ---------------------------------------------------------------------- #

_ENV_PURPOSE_DRIFT_ENABLED = "LMS_PURPOSE_DRIFT_ENABLED"
_ENV_PURPOSE_GOAL_TEXT = "LMS_PURPOSE_GOAL_TEXT"

#: 目的检查开关默认 **1**（开——总任务书 §二.5：目的时相是机制本体）
_DEFAULT_ENABLED = True

#: 缺省总目的原文：dandan 原始哲学四词锚点（§0 目的锚定；只有 dandan
#: 可改——通过 LMS_PURPOSE_GOAL_TEXT 或调用方 purpose_text 显式传入）
_DEFAULT_PURPOSE_TEXT = "流动上下文 / 活体记忆 / 记录涌现过程 / 熵核心"

#: 每轮判定记录的有界窗口（deque maxlen）
_RECENT_MAXLEN = 200

#: snapshot 中 purpose 文本截断长度（超长截断 + 省略号）
_PURPOSE_TRUNCATE = 120

#: 中性闸门（开关关 / fail-open 用）的每题理由
_DISABLED_REASON = "目的检查关闭（LMS_PURPOSE_DRIFT_ENABLED=0）——闸门中性"
_FAILOPEN_REASON_PREFIX = "目的检查内部异常（fail-open）——闸门中性，未判"


def purpose_drift_enabled(explicit: Optional[bool] = None) -> bool:
    """目的检查开关解析：显式参数 > 环境变量 LMS_PURPOSE_DRIFT_ENABLED。

    布尔接受（不区分大小写）：1/true/yes/on 视为开，其余为关。
    默认开（机制本体）；0=关 → judge 返回中性、snapshot enabled=False。
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get(_ENV_PURPOSE_DRIFT_ENABLED, "1")
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _resolve_purpose(purpose_text: str = "") -> str:
    """dsh-goal 目的文本来源：调用方 purpose_text > env LMS_PURPOSE_GOAL_TEXT
    > 内置总目的原文（dandan 四词锚点）。"""
    if isinstance(purpose_text, str) and purpose_text.strip():
        return purpose_text.strip()
    env = os.environ.get(_ENV_PURPOSE_GOAL_TEXT, "")
    if env and env.strip():
        return env.strip()
    return _DEFAULT_PURPOSE_TEXT


def _truncate(text: str, limit: int = _PURPOSE_TRUNCATE) -> str:
    """快照用文本截断（purpose 字段；超长加省略号）。"""
    text = text if isinstance(text, str) else str(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _neutral_gate(purpose: str, reason: str,
                  verdict: str = "uncertain") -> dict:
    """中性闸门（开关关 / fail-open 共用）：不判是/否，闸门不亮（fail-open
    不阻断调用方；开关关零参与），每题给统一理由。"""
    answers = {q: {"ok": None, "state": "uncertain", "reason": reason}
               for q in QUESTION_IDS}
    return {
        "purpose_drift": False,
        "verdict": verdict,
        "answers": answers,
        "reasons": [reason],
        "purpose": purpose,
    }


# ---------------------------------------------------------------------- #
#  目的检查时相（PurposeDriftPhase）
# ---------------------------------------------------------------------- #

class PurposeDriftPhase:
    """目的检查时相（总任务书 §二.5：每轮 [doubt] purpose-drift 判定）。

    与质疑系统（core/doubt）同处一个模块族：消费同一轮活动信号
    （round_signals），输出闸门信号「是否偏离 + 理由」——**不输出任何
    可被优化的分数**（Pan 警示）。

    输入: ``judge(round_signals, purpose_text="")``
       - round_signals: 本轮活动信号 dict（键表与映射规则见模块 docstring；
         全部缺省视为中性）。
       - purpose_text: 本轮目的文本（dsh-goal 目的）；空 → env
         LMS_PURPOSE_GOAL_TEXT → 内置总目的原文。

    输出: 闸门信号 dict
       - purpose_drift: bool —— 闸门是否亮（verdict != "aligned"）；
       - verdict: "aligned" | "drifted" | "uncertain"；
       - answers: {Q1..Q5: {"ok": bool|None, "state": "yes"|"no"|"uncertain",
         "reason": str}}；
       - reasons: [str] —— 五问逐条判定（可回放可审计的完整轨迹）；
       - purpose: 本轮判定的目的文本。

    每轮记录：有界 deque（maxlen 200）``_recent_verdicts``（每轮
    {verdict, ts}）+ ``drift_count`` 累计（只计 verdict == "drifted"；
    uncertain 是"未判"不是"偏离"，记入 recent_verdicts 但不计 drift）。

    治理开关：LMS_PURPOSE_DRIFT_ENABLED（默认 1=开；0=关 → judge 返回
    中性闸门、snapshot enabled=False）。fail-open：异常静默降级，绝不
    阻断调用方。
    """

    def __init__(self, enabled: Optional[bool] = None) -> None:
        # 治理开关：显式参数 > 环境变量（默认 1=开；0 → 中性闸门）
        self.enabled = purpose_drift_enabled(enabled)
        # 每轮判定记录（有界 deque——可回放近 200 轮）
        self._recent_verdicts: Deque[Dict[str, Any]] = deque(maxlen=_RECENT_MAXLEN)
        #: drifted 轮次累计（只计 verdict == "drifted"）
        self.drift_count: int = 0
        self._last_verdict: Optional[str] = None
        self._last_reasons: List[str] = []

    # ------------------------------------------------------------------ #
    #  主入口：每轮判定
    # ------------------------------------------------------------------ #

    def judge(self, round_signals: dict, purpose_text: str = "") -> dict:
        """本轮目的检查：输入活动信号，输出闸门信号（是否偏离 + 理由）。

        开关关 → 返回中性闸门（不记录）；异常 → fail-open 中性闸门（不
        记录、不阻断）。判定规则与映射表见模块 docstring。
        """
        purpose = _resolve_purpose(purpose_text)
        if not self.enabled:
            return _neutral_gate(purpose, _DISABLED_REASON)
        try:
            return self._judge_impl(round_signals, purpose)
        except Exception as exc:  # pylint: disable=broad-except
            # fail-open：异常静默，绝不阻断调用方（以日志可见）
            logger.error("目的检查失败（fail-open）", exc_info=True)
            return _neutral_gate(
                purpose, "%s：%s" % (_FAILOPEN_REASON_PREFIX, exc))

    def _judge_impl(self, round_signals: dict, purpose: str) -> dict:
        """judge 的非防护实现（enabled 已保证；round_signals 可能非 dict）。"""
        sig = round_signals if isinstance(round_signals, dict) else {}

        # 信号提取（缺省视为中性：False / 0）
        episodic_added = _truthy(sig, "episodic_added")
        doubt_events = _count(sig, "doubt_events")
        verification_chain_active = _truthy(sig, "verification_chain_active")
        suspect_marked = _truthy(sig, "suspect_marked")
        surprise_in_decisions = _truthy(sig, "surprise_in_decisions")
        lifecycle_trace = _truthy(sig, "lifecycle_trace")
        reconsolidated = _truthy(sig, "reconsolidated")
        dream_consolidated = _truthy(sig, "dream_consolidated")
        provenance = _truthy(sig, "provenance")
        conclusion_only = _truthy(sig, "conclusion_only")
        memory_idle = _truthy(sig, "memory_idle")
        surprise_display_only = _truthy(sig, "surprise_display_only")
        not_replayable = _truthy(sig, "not_replayable")

        # -- 五问判定（映射规则见模块 docstring） ------------------------ #
        answers: Dict[str, Dict[str, Any]] = {}

        # Q1 流动：任一写侧活动 → 是；无任何写侧活动 → 否
        write_activity = (episodic_added or reconsolidated
                          or dream_consolidated or suspect_marked)
        if write_activity:
            answers["Q1"] = _yes(
                "Q1",
                "本轮有写侧活动（episodic_added / reconsolidated / "
                "dream_consolidated / suspect_marked）——产出流入可复用、"
                "随状态演化的活结构")
        else:
            answers["Q1"] = _no(
                "Q1",
                "本轮无任何写侧活动（episodic_added / consolidated / "
                "suspect_marked 均无）——产出未流入活结构，疑似一次性"
                "静态档案（Q1 反例：结论定格）")

        # Q2 过程：doubt_events>0 或 lifecycle_trace 或
        #         surprise_in_decisions → 是；elif conclusion_only → 否
        if doubt_events > 0 or lifecycle_trace or surprise_in_decisions:
            answers["Q2"] = _yes(
                "Q2",
                "本轮有过程记录（doubt_events=%d / lifecycle_trace / "
                "surprise_in_decisions）——记录了「怎么到这一步」"
                % doubt_events)
        elif conclusion_only:
            answers["Q2"] = _no(
                "Q2",
                "本轮只记录了结论未记录过程（conclusion_only）——"
                "惊讶/怀疑/转向/悬案的过程痕迹缺失（Q2 反例：只交结论）")
        else:
            answers["Q2"] = _uncertain(
                "Q2",
                "缺过程信号（doubt_events / lifecycle_trace / "
                "surprise_in_decisions 均无）——无法判定本轮是否记录"
                "「怎么到这一步」；请补信号后补判")

        # Q3 活体：doubt_events>0 或 reconsolidated 或
        #          verification_chain_active → 是；elif memory_idle → 否
        if doubt_events > 0 or reconsolidated or verification_chain_active:
            answers["Q3"] = _yes(
                "Q3",
                "本轮记忆机制真在动（doubt_events=%d / reconsolidated / "
                "verification_chain_active）——怀疑→修正/再巩固/验证链"
                "真实发生" % doubt_events)
        elif memory_idle:
            answers["Q3"] = _no(
                "Q3",
                "记忆机制「只是存在」本轮未动（memory_idle）——接口接了"
                "不等于活着，怀疑/再巩固/验证链闭环断（Q3 反例：机制空转）")
        else:
            answers["Q3"] = _uncertain(
                "Q3",
                "缺活体信号（doubt_events / reconsolidated / "
                "verification_chain_active 均无）——无法判定记忆机制"
                "是否真在动；请补信号后补判")

        # Q4 熵核：surprise_in_decisions → 是；elif surprise_display_only
        if surprise_in_decisions:
            answers["Q4"] = _yes(
                "Q4",
                "熵/惊讶参与了决策（surprise_in_decisions）——唤醒/思考/"
                "注入调度被熵信号驱动")
        elif surprise_display_only:
            answers["Q4"] = _no(
                "Q4",
                "熵/惊讶只作展示未参与决策（surprise_display_only）——"
                "指标不驱动行为（Q4 反例：surprise 只是日志字段）")
        else:
            answers["Q4"] = _uncertain(
                "Q4",
                "缺熵核信号（surprise_in_decisions 无）——无法判定熵/惊讶"
                "是否参与决策；请补信号后补判")

        # Q5 可回放：lifecycle_trace 或 provenance 或
        #            verification_chain_active（活跃链记录 VERIFY-*）
        if lifecycle_trace or provenance or verification_chain_active:
            answers["Q5"] = _yes(
                "Q5",
                "涌现过程可回放（lifecycle_trace / provenance（VERIFY-*）/ "
                "verification_chain_active）——有记忆形态投影可注入可审计")
        elif not_replayable:
            answers["Q5"] = _no(
                "Q5",
                "涌现过程不可回放/不可注入/不可审计（not_replayable）——"
                "过程只在日志里、无记忆形态投影（Q5 反例：无法重现）")
        else:
            answers["Q5"] = _uncertain(
                "Q5",
                "缺可回放信号（lifecycle_trace / provenance（VERIFY-*）均无）"
                "——无法判定涌现过程是否可回放；请补信号后补判")

        # -- 最终 verdict（铁律：任一否→drifted；全是→aligned；其余→uncertain）
        states = [answers[q]["state"] for q in QUESTION_IDS]
        if "no" in states:
            verdict = "drifted"
        elif all(s == "yes" for s in states):
            verdict = "aligned"
        else:
            verdict = "uncertain"

        # 闸门信号：drifted 与 uncertain 都亮（"不确定"= 未判，不是通过）
        purpose_drift = verdict != "aligned"
        reasons = ["%s %s：%s" % (q, answers[q]["state"], answers[q]["reason"])
                   for q in QUESTION_IDS]

        # 每轮记录（可回放可审计）
        self._recent_verdicts.append({"verdict": verdict, "ts": time.time()})
        if verdict == "drifted":
            self.drift_count += 1
        self._last_verdict = verdict
        self._last_reasons = reasons

        return {
            "purpose_drift": purpose_drift,
            "verdict": verdict,
            "answers": answers,
            "reasons": reasons,
            "purpose": purpose,
        }

    # ------------------------------------------------------------------ #
    #  观测
    # ------------------------------------------------------------------ #

    def snapshot(self, max_n: int = 20) -> dict:
        """观测块（/status purpose 数据源）。

        返回: {enabled, purpose(截断), last_verdict, last_reasons,
        drift_count, recent_verdicts(n)}——**无任何符合度分数**。
        开关关 → enabled=False（其余字段保留：零值/空，供审计）。
        """
        try:
            recent = list(self._recent_verdicts)
            if max_n is not None and int(max_n) > 0:
                recent = recent[-int(max_n):]
            return {
                "enabled": self.enabled,
                "purpose": _truncate(_resolve_purpose()),
                "last_verdict": self._last_verdict,
                "last_reasons": list(self._last_reasons),
                "drift_count": self.drift_count,
                "recent_verdicts": [dict(r) for r in recent],
            }
        except Exception:  # pylint: disable=broad-except
            # fail-open：快照异常静默降级（绝不阻断调用方）
            logger.error("目的检查快照失败（fail-open）", exc_info=True)
            return {"enabled": self.enabled, "purpose": _truncate(
                _DEFAULT_PURPOSE_TEXT), "last_verdict": None,
                "last_reasons": [], "drift_count": self.drift_count,
                "recent_verdicts": []}


# ---------------------------------------------------------------------- #
#  内部小工具
# ---------------------------------------------------------------------- #

def _truthy(sig: dict, key: str) -> bool:
    """信号取值：缺省 False（中性）；非 dict 输入由 judge 防护层兜底。"""
    return bool(sig.get(key, False))


def _count(sig: dict, key: str) -> int:
    """计数信号取值：缺省 0；非法数值按 0（fail-open，不抛）。"""
    try:
        return int(sig.get(key, 0))
    except (TypeError, ValueError):
        return 0


def _yes(q: str, reason: str) -> dict:
    return {"ok": True, "state": "yes", "reason": reason}


def _no(q: str, reason: str) -> dict:
    return {"ok": False, "state": "no", "reason": reason}


def _uncertain(q: str, reason: str) -> dict:
    return {"ok": None, "state": "uncertain", "reason": reason}

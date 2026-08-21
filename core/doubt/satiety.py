# -*- coding: utf-8 -*-
"""E3 satiety 消解（core/doubt · 纯函数 + 轻状态）——终止信号防 OCD 化。

规格依据：`方案-E3自我怀疑驱动调节-20260820.md` §3.2 ⑤ / §3.3；dandan 拍板
5（2026-08-20 22:14，节流放宽）：
  - satiety 冷却 24h → 12h（LMS_E3_SATIETY_COOLDOWN_H，默认 12）；
  - 单次重激活上限 1 → 2 条（LMS_E3_REACTIVATE_MAX，loop 侧读取）；
  - 不加日预算（三重节流太多）。

判定（``judge_resolved``，尽力而为不无限怀疑，Szechtman & Woody 2004）：
  - 目标条目 rewritten（doubt_state == 'superseded'，或复核报告
    ``outcomes['rewritten'] ≥ 1``）→ 'resolved'；
  - 连续 N 轮复核 kept（无新证据）→ 'resolved'（N = kept_threshold，
    默认 3；调用方传 gap_registry.review_stats() 的 kept 计数）。

待机（``SatietyGate``，冷却/计数/武装-待机状态机）：
  - 同一悬案重激活/消解后 ``LMS_E3_SATIETY_COOLDOWN_H`` 内不重选
    （per-topic 冷却；``is_cooldown(topic)`` 判）；
  - 累计闭环 ≥ ``LMS_E3_SATIETY_MAX_CYCLES``（默认 3）→ 待机
    ``LMS_E3_SATIETY_STANDBY_H``（默认 24）不参与（``is_armed()`` 判）；
  - 新 [doubt] 事件 / 新 plateau 触发 → ``rearm()`` 重新武装（loop 侧
    process_turn 在 doubt 事件后调用）。

约束：纯 stdlib、可单测、fail-open（任何异常 → 保守返回，绝不抛）。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

_ENV_COOLDOWN_H = "LMS_E3_SATIETY_COOLDOWN_H"
_ENV_MAX_CYCLES = "LMS_E3_SATIETY_MAX_CYCLES"
_ENV_STANDBY_H = "LMS_E3_SATIETY_STANDBY_H"

#: 默认值（dandan 拍板 5 放宽：冷却 12h；设计 §3.2 ⑤：3 次闭环后待机 24h）
_DEFAULT_COOLDOWN_H = 12.0
_DEFAULT_MAX_CYCLES = 3
_DEFAULT_STANDBY_H = 24.0

#: 闭环历史记录上限（观测）
_HISTORY_LIMIT = 20


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, "") or default)
    except (TypeError, ValueError):
        return float(default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, "") or default)
    except (TypeError, ValueError):
        return int(default)


# ---------------------------------------------------------------------- #
#  纯函数：消解判定
# ---------------------------------------------------------------------- #

def judge_resolved(entry: Any = None, outcomes: Optional[Dict] = None,
                   kept_threshold: int = 3) -> str:
    """判定悬案是否消解（纯函数，fail-open）→ 'resolved' | 'pending'。

    尽力而为不无限怀疑（Szechtman & Woody 2004 终止信号）：
      - 目标条目已 superseded（重巩固改写落库）→ resolved；
      - 复核报告 rewritten ≥ 1 → resolved；
      - 复核报告 kept ≥ kept_threshold（连续 N 轮无新证据）→ resolved。
    其余 → pending（继续观察）。
    """
    if entry is not None:
        try:
            if str(getattr(entry, "doubt_state", "stable") or "stable") \
                    == "superseded":
                return "resolved"
        except Exception:  # pylint: disable=broad-except
            pass
    stats = outcomes or {}
    try:
        if int(stats.get("rewritten", 0) or 0) >= 1:
            return "resolved"
        if int(stats.get("kept", 0) or 0) >= max(1, int(kept_threshold)):
            return "resolved"
    except (TypeError, ValueError):
        pass
    return "pending"


# ---------------------------------------------------------------------- #
#  SatietyGate：冷却 / 计数 / 武装-待机状态机（轻状态，纯内存）
# ---------------------------------------------------------------------- #

class SatietyGate:
    """E3 satiety 状态机（轻状态纯内存，重启即失——同 gap_registry 先例）。

    显式参数 > 环境变量 > 默认值；``now`` 可注入（测试确定性时钟）。
    """

    def __init__(self, cooldown_h: Optional[float] = None,
                 max_cycles: Optional[int] = None,
                 standby_h: Optional[float] = None,
                 now: Optional[float] = None) -> None:
        self.cooldown_h = max(
            0.0, float(cooldown_h if cooldown_h is not None
                       else _env_float(_ENV_COOLDOWN_H, _DEFAULT_COOLDOWN_H)))
        self.max_cycles = max(
            1, int(max_cycles if max_cycles is not None
                   else _env_int(_ENV_MAX_CYCLES, _DEFAULT_MAX_CYCLES)))
        self.standby_h = max(
            0.0, float(standby_h if standby_h is not None
                       else _env_float(_ENV_STANDBY_H, _DEFAULT_STANDBY_H)))
        # per-topic 冷却：topic -> 冷却截止时间戳
        self._cooldown_until: Dict[str, float] = {}
        # 闭环计数与待机
        self.cycle_count: int = 0
        self.standby_until: Optional[float] = None
        self.closed_cycles: List[Dict] = []
        # 观测计数
        self.armed_events: int = 0
        self.standby_events: int = 0
        self._clock_now: Optional[float] = now

    # ------------------------------------------------------------------ #
    #  时钟（测试注入）
    # ------------------------------------------------------------------ #

    def _ts(self, now: Optional[float] = None) -> float:
        if now is not None:
            return float(now)
        if self._clock_now is not None:
            return float(self._clock_now)
        return time.time()

    def set_clock(self, now: float) -> None:
        """测试用：固定时钟（None 恢复真实时钟）。"""
        self._clock_now = now

    # ------------------------------------------------------------------ #
    #  查询
    # ------------------------------------------------------------------ #

    def is_armed(self, now: Optional[float] = None) -> bool:
        """机制可否参与：不在待机中即为武装（per-topic 冷却另行判）。"""
        return not self.in_standby(now)

    def in_standby(self, now: Optional[float] = None) -> bool:
        now = self._ts(now)
        return self.standby_until is not None and now < self.standby_until

    def is_cooldown(self, topic: Any, now: Optional[float] = None) -> bool:
        """同一悬案冷却中？（冷却到期自动清除——惰性）"""
        now = self._ts(now)
        key = str(topic)
        until = self._cooldown_until.get(key)
        if until is None:
            return False
        if now >= until:
            self._cooldown_until.pop(key, None)
            return False
        return True

    def cooldown_topics(self, now: Optional[float] = None) -> List[str]:
        """当前冷却中的 topic 列表（选择器排除集数据源，只读）。"""
        now = self._ts(now)
        out = []
        for key, until in list(self._cooldown_until.items()):
            if now < until:
                out.append(key)
            else:
                self._cooldown_until.pop(key, None)
        return out

    def cooldown_remaining(self, topic: Any, now: Optional[float] = None) -> float:
        """指定悬案剩余冷却秒数（0 = 不在冷却）。"""
        now = self._ts(now)
        until = self._cooldown_until.get(str(topic))
        if until is None:
            return 0.0
        return max(0.0, until - now)

    # ------------------------------------------------------------------ #
    #  记账
    # ------------------------------------------------------------------ #

    def record_reactivation(self, topic: Any, now: Optional[float] = None) -> None:
        """重激活后 per-topic 冷却（冷却期内不重选同一悬案）。"""
        now = self._ts(now)
        self._cooldown_until[str(topic)] = now + self.cooldown_h

    def record_resolved(self, topic: Any, detail: str = "",
                        now: Optional[float] = None) -> Dict:
        """消解记账（观测 + 冷却；fok_resolved 落点 = gap_registry.mark_resolved）。"""
        now = self._ts(now)
        self._cooldown_until[str(topic)] = now + self.cooldown_h
        return {"topic": str(topic), "ts": now,
                "detail": str(detail)[:300]}

    def close_cycle(self, outcome: str = "activated", detail: str = "",
                    now: Optional[float] = None) -> Dict:
        """闭环记账：cycle_count +1；达 max_cycles → 待机 standby_h。

        3 次闭环后待机（设计 §3.2 ⑤——Szechtman & Woody 2004 终止信号防
        OCD 化）；新 [doubt] 事件 / 新 plateau 触发 → rearm() 恢复。
        """
        now = self._ts(now)
        self.cycle_count += 1
        rec = {"ts": now, "cycle": self.cycle_count,
               "outcome": str(outcome), "detail": str(detail)[:300]}
        self.closed_cycles.append(rec)
        self.closed_cycles = self.closed_cycles[-_HISTORY_LIMIT:]
        if self.cycle_count >= self.max_cycles:
            self.standby_until = now + self.standby_h
            self.standby_events += 1
        return rec

    def rearm(self, now: Optional[float] = None) -> bool:
        """重新武装：新 [doubt] 事件 / 新 plateau 触发 → 清待机 + 重置计数。"""
        now = self._ts(now)
        self.standby_until = None
        self.cycle_count = 0
        self.armed_events += 1
        return True

    # ------------------------------------------------------------------ #
    #  观测
    # ------------------------------------------------------------------ #

    def snapshot(self, now: Optional[float] = None) -> Dict:
        now = self._ts(now)
        return {
            "cooldown_h": self.cooldown_h,
            "max_cycles": self.max_cycles,
            "standby_h": self.standby_h,
            "armed": self.is_armed(now),
            "in_standby": self.in_standby(now),
            "standby_until": self.standby_until,
            "cycle_count": self.cycle_count,
            "cooldown_count": len(self.cooldown_topics(now)),
            "closed_cycles": list(self.closed_cycles)[-5:],
            "armed_events": self.armed_events,
            "standby_events": self.standby_events,
        }

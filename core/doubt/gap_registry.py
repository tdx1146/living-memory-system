"""信息缺口登记（体验层 D，设计 v1.1 §6.6）—— 怀疑灯数据源。

**数据源链路（审计建议 4）**：doubt_ingest（/feed 结构化摄入）→
gap_registry（登记）→ /status doubt.gaps + 回魂怀疑灯。

缺口分类：
  - A 类 fok_unresolved：first-order knowledge 未决（已关注方向内的悬而未决）
  - B 类 low_confidence_unreviewed：低置信未复核条目（已关注方向内的质检欠账）
  - C 类 explore_dims：探索缺口（encounter_count 最低维度）——**仅 /status
    诊断观测，不进回魂怀疑灯、不进注入**（专注化修订：暴露探索缺口=引导
    主 AI 探索新方向=跑偏，违反拍板 2）。

模块是纯内存状态（重启即失）；doubt_ingest 摄入 → 本登记 → /status 读出。
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional


class GapRegistry:
    """信息缺口登记（进程内内存状态）。"""

    def __init__(self) -> None:
        # A/B/C 三类缺口（C 类仅诊断）
        self._fok_unresolved: List[Dict] = []          # A 类
        self._low_confidence_unreviewed: List[Dict] = []  # B 类
        self._explore_dims: List[Dict] = []            # C 类（仅诊断）
        # E3 satiety：已消解悬案（fok_resolved——消解历史，不随复核轮清空）
        self._fok_resolved: List[Dict] = []
        self._last_review: Optional[float] = None      # 最近一次做梦复核时间
        self._review_stats: Dict = {                   # 最近复核报告
            'reviewed': 0, 'downgraded': 0, 'rewritten': 0,
            'kept': 0, 'flagged': 0,
        }

    # ------------------------------------------------------------------ #
    #  登记
    # ------------------------------------------------------------------ #

    def register_fok_unresolved(self, topic: str, detail: str = "",
                                ts: Optional[float] = None) -> None:
        """A 类：fok 未决登记（上限 50 条，防膨胀）。"""
        self._fok_unresolved.append({
            'topic': str(topic)[:120],
            'detail': str(detail)[:300],
            'ts': ts if ts is not None else time.time(),
        })
        self._fok_unresolved = self._fok_unresolved[-50:]

    def register_low_confidence(self, entry, ts: Optional[float] = None) -> None:
        """B 类：低置信未复核条目登记（按 entry 文本去重，上限 50 条）。"""
        try:
            text = getattr(entry, 'text', '') or ''
            conf = float(getattr(entry, 'confidence', 1.0) or 1.0)
            rebuttal = int(getattr(entry, 'rebuttal_count', 0) or 0)
        except (TypeError, ValueError):
            return
        record = {
            'text': str(text)[:120],
            'confidence': round(conf, 3),
            'rebuttal_count': rebuttal,
            'ts': ts if ts is not None else time.time(),
        }
        # 去重（同文本只留最新）
        self._low_confidence_unreviewed = [
            r for r in self._low_confidence_unreviewed
            if r.get('text') != record['text']
        ]
        self._low_confidence_unreviewed.append(record)
        self._low_confidence_unreviewed = self._low_confidence_unreviewed[-50:]

    def register_explore_dims(self, dims: List[int],
                              ts: Optional[float] = None) -> None:
        """C 类：探索缺口（encounter_count 最低维度）——仅诊断观测。

        不进怀疑灯、不进注入（专注化修订，设计 v1.1 §6.6）。
        """
        self._explore_dims = [{
            'dims': [int(d) for d in dims][:10],
            'ts': ts if ts is not None else time.time(),
        }]

    # ------------------------------------------------------------------ #
    #  E3 satiety：消解登记（fok_resolved）
    # ------------------------------------------------------------------ #

    def mark_resolved(self, topic: str, detail: str = "",
                      ts: Optional[float] = None) -> bool:
        """E3 satiety：悬案消解登记（尽力而为不无限怀疑的落点）。

        追加消解记录到 ``_fok_resolved``（上限 100 条，防膨胀），并同步
        从 A/B 类缺口移除同 topic 记录（消解后不再出现在怀疑灯）。幂等：
        同 topic 重复登记只留最新。返回是否新增登记。
        """
        normalized = str(topic)[:120]
        if not normalized:
            return False
        record = {
            'topic': normalized,
            'detail': str(detail)[:300],
            'ts': ts if ts is not None else time.time(),
        }
        existed = any(
            r.get('topic') == normalized for r in self._fok_resolved)
        self._fok_resolved = [
            r for r in self._fok_resolved if r.get('topic') != normalized]
        self._fok_resolved.append(record)
        self._fok_resolved = self._fok_resolved[-100:]
        # 同步移除未决（A 类）与低置信待复核（B 类）中的同 topic 记录
        self._fok_unresolved = [
            r for r in self._fok_unresolved
            if r.get('topic') != normalized]
        self._low_confidence_unreviewed = [
            r for r in self._low_confidence_unreviewed
            if (r.get('text') or '')[:120] != normalized]
        return not existed

    def is_resolved(self, topic: str) -> bool:
        """该悬案是否已消解（选择器排除集数据源，只读）。"""
        normalized = str(topic)[:120]
        return any(
            r.get('topic') == normalized for r in self._fok_resolved)

    def resolved_list(self) -> List[Dict]:
        """已消解悬案记录（只读拷贝）。"""
        return list(self._fok_resolved)

    def resolved_count(self) -> int:
        """已消解悬案计数（A5 观测数据源）。"""
        return len(self._fok_resolved)

    # ------------------------------------------------------------------ #
    #  复核联动
    # ------------------------------------------------------------------ #

    def mark_review(self, stats: Optional[Dict] = None,
                    resolved_topics: Optional[List[str]] = None) -> None:
        """做梦 doubt_review 完成后调用：记录时间 + 复核报告。

        E3 联动（2026-08-20）：
          - ``resolved_topics``（可选）：本次复核判定已消解的悬案列表，
            一并 ``mark_resolved``（satiety 落点；调用方也可逐条调用）。
          - ``_fok_resolved`` 是消解历史，**不**随复核轮清空（与 B 类
            不同——B 类清空是"已复核一轮"，消解史是"已闭环"）。
        """
        self._last_review = time.time()
        if stats:
            for k in ('reviewed', 'downgraded', 'rewritten', 'kept', 'flagged'):
                if k in stats:
                    self._review_stats[k] = int(stats.get(k, 0) or 0)
        for topic in (resolved_topics or []):
            try:
                self.mark_resolved(topic, detail='做梦复核判定消解')
            except Exception:  # pylint: disable=broad-except
                pass
        # B 类清空（已复核一轮）；A 类保留（fok 未决不随单轮复核消失）
        self._low_confidence_unreviewed = []

    # ------------------------------------------------------------------ #
    #  读出
    # ------------------------------------------------------------------ #

    def snapshot(self) -> Dict:
        """/status doubt.gaps 数据源（A/B 类 + C 类诊断 + E3 消解史）。"""
        return {
            'fok_unresolved': list(self._fok_unresolved),
            'low_confidence_unreviewed': list(self._low_confidence_unreviewed),
            'explore_dims': list(self._explore_dims),  # C 类：仅诊断
            'fok_resolved': list(self._fok_resolved),  # E3 satiety 消解史
        }

    def doubt_lamp(self) -> Dict:
        """回魂怀疑灯数据（**仅 A/B 类**，C 类不进——专注化修订）。"""
        return {
            'fok_unresolved_count': len(self._fok_unresolved),
            'low_confidence_unreviewed_count': len(
                self._low_confidence_unreviewed),
            'last_review': self._last_review,
        }

    def review_stats(self) -> Dict:
        return dict(self._review_stats)

    def last_review(self) -> Optional[float]:
        return self._last_review

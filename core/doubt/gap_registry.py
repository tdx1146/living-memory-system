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

import json
import os
import re
import time
from typing import Dict, List, Optional


class GapRegistry:
    """信息缺口登记（进程内内存状态 + 持久化镜像）。

    E3 持久化（2026-08-21 部署验证发现：重启后 fok_unresolved 清零 → E3
    候选源丢失）：A/B/C + resolved 列表经 _PERSIST_PATH jsonl 落盘，
    启动时 load 恢复；写侧 fail-open（落盘失败不影响登记）。
    """

    _PERSIST_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))),
        'data', 'gap_registry.json')

    def __init__(self, persist_path: Optional[str] = None) -> None:
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
        if persist_path:
            self._PERSIST_PATH = persist_path
        self._load()

    # ------------------------------------------------------------------ #
    #  持久化（E3 2026-08-21：重启不丢候选源）
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """启动恢复：读 jsonl 镜像（缺省/损坏 → 空，fail-open）。"""
        try:
            if not os.path.exists(self._PERSIST_PATH):
                return
            with open(self._PERSIST_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self._fok_unresolved = list(
                data.get('fok_unresolved', []) or [])[-50:]
            self._low_confidence_unreviewed = list(
                data.get('low_confidence_unreviewed', []) or [])[-50:]
            self._explore_dims = list(
                data.get('explore_dims', []) or [])
            self._fok_resolved = list(
                data.get('fok_resolved', []) or [])[-100:]
            self._last_review = data.get('last_review')
        except Exception:  # pylint: disable=broad-except
            pass

    def _persist(self) -> None:
        """落盘镜像（fail-open：写失败不影响登记主流程）。"""
        try:
            data = {
                'fok_unresolved': self._fok_unresolved,
                'low_confidence_unreviewed':
                    self._low_confidence_unreviewed,
                'explore_dims': self._explore_dims,
                'fok_resolved': self._fok_resolved,
                'last_review': self._last_review,
                'ts': time.time(),
            }
            tmp = self._PERSIST_PATH + '.tmp'
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self._PERSIST_PATH)
        except Exception:  # pylint: disable=broad-except
            pass

    # ------------------------------------------------------------------ #
    #  登记
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fok_core(topic: str) -> str:
        """归一化核心：剥包装前缀（用户:/助手:）与 [doubt] 协议前缀。"""
        t = re.sub(r'^(?:用户|助手)\s*[:：]\s*', '', str(topic))
        t = re.sub(r'^\[doubt\]\s*\w+\s*[:：]\s*', '', t)
        return t.strip()

    @staticmethod
    def _fok_similar(a: str, b: str) -> bool:
        """同一悬案判定：共享前缀 ≥12 字，或字符二元组 Jaccard 相似度 ≥ 0.4。

        (2026-08-22 修复实证：同一悬案三种形态共享前 16 字前缀——"旧结论说
        等待只是被动，但机械援军独白"——但尾部在第 16 字分叉，长主题稀释
        Jaccard；故用"前缀长匹配 OR 整体相似"。独立悬案共享 ≤7 字，不误并。)
        """
        if not a or not b:
            return a == b
        if a == b:
            return True
        if len(a) < 8 or len(b) < 8:
            return a == b
        # 共享前缀长度
        n = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                break
            n += 1
        if n >= 12:
            return True
        # 字符二元组 Jaccard
        ga = set(zip(a, a[1:]))
        gb = set(zip(b, b[1:]))
        inter = len(ga & gb)
        union = len(ga | gb)
        return union > 0 and inter / union >= 0.4

    def _same_case(self, a_topic: str, b_topic: str) -> bool:
        """[A4][B7] 消解/判重与登记侧同构的归一化判定（_fok_core + _fok_similar）。

        登记侧（register_fok_unresolved）用归一化核心 + 前缀/Jaccard 模糊去重；
        旧消解/判重侧（mark_resolved/is_resolved）用原始 topic 精确相等 →
        同一悬案换包装登记后永远无法被消解移除，E3 反复选中已闭环主题。
        统一后三处共用同构判定；_fok_core 剥"用户:/助手:"与 [doubt] 前缀，
        _fok_similar 已含精确相等短路，不会误并独立悬案（实测共享 ≤7 字不误并）。
        """
        ca, cb = self._fok_core(a_topic), self._fok_core(b_topic)
        return bool(ca) and bool(cb) and self._fok_similar(ca, cb)

    def register_fok_unresolved(self, topic: str, detail: str = "",
                                ts: Optional[float] = None) -> None:
        """A 类：fok 未决登记（上限 50 条，防膨胀）。

        2026-08-22 去重修复：E3 重激活/路径 B 会让同一悬案以不同包装重复登记
        （gap_registry 实锤：机械援军主题 3 形态、3080 主题 2 形态，间隔 ~21h）。
        归一化去重键（剥前缀取核心前 20 字）命中已有条目 → 更新 ts/detail
        （"重新面对"语义），不新增。
        """
        topic = str(topic)[:120]
        core = self._fok_core(topic)
        now = ts if ts is not None else time.time()
        for existing in self._fok_unresolved:
            # [A4] 统一走 _same_case（原内联 _fok_similar 判定，语义等价；
            # 消解侧 mark_resolved/is_resolved 现用同一判定助手，避免漂移）。
            if core and self._same_case(core, existing['topic']):
                existing['ts'] = now
                if detail:
                    existing['detail'] = str(detail)[:300]
                self._persist()
                return  # 去重：更新已有，不新增
        self._fok_unresolved.append({
            'topic': topic,
            'detail': str(detail)[:300],
            'ts': now,
        })
        self._fok_unresolved = self._fok_unresolved[-50:]
        self._persist()

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
        self._persist()

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
        # [A4][B7] 幂等去重改用 _same_case（与登记侧同构）：登记侧模糊去重、
        # 消解侧精确匹配 → 同悬案换包装登记后永远无法被消解移除。统一判定后
        # 消解移除/判重与登记去重行为一致（B7 同根，一并覆盖）。
        existed = any(
            self._same_case(r.get('topic', ''), normalized)
            for r in self._fok_resolved)
        self._fok_resolved = [
            r for r in self._fok_resolved
            if not self._same_case(r.get('topic', ''), normalized)]
        self._fok_resolved.append(record)
        self._fok_resolved = self._fok_resolved[-100:]
        self._persist()
        # 同步移除未决（A 类）与低置信待复核（B 类）中的同 topic 记录
        # [A4] A 类移除同样用 _same_case（旧精确匹配无法移除换包装悬案）；
        # B 类按 text 去重（register_low_confidence 语义），保持原样。
        self._fok_unresolved = [
            r for r in self._fok_unresolved
            if not self._same_case(r.get('topic', ''), normalized)]
        self._low_confidence_unreviewed = [
            r for r in self._low_confidence_unreviewed
            if (r.get('text') or '')[:120] != normalized]
        return not existed

    def is_resolved(self, topic: str) -> bool:
        """该悬案是否已消解（选择器排除集数据源，只读）。"""
        normalized = str(topic)[:120]
        # [A4][B7] 与登记侧同构的归一化判定：精确匹配无法识别换包装的已消解
        # 悬案 → 已闭环主题被 E3 反复选中。
        return any(
            self._same_case(normalized, r.get('topic', ''))
            for r in self._fok_resolved)

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

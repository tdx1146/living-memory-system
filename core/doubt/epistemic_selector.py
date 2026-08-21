# -*- coding: utf-8 -*-
"""E3 悬案选择器（core/doubt · 纯函数选料器）——自我怀疑驱动的主动调节。

规格依据：`方案-E3自我怀疑驱动调节-20260820.md` §3.2 ① / §3.3（dandan 拍板
2026-08-20 22:14：免费版重激活材料用无 LLM 自动生成——本模块纯 stdlib，
零 LLM、零深想子代理）。

选料（候选源）：
  - A 类 ``gaps.fok_unresolved``（fok 未决）
  - B 类 ``gaps.low_confidence_unreviewed``（低置信未复核）
  - suspect 未决条目（doubt_state == 'suspect'）——由调用方（loop）提取
    为 fok 形态记录传入，本模块不直接遍历 memory（保持纯函数、可单测）

排序（全部从既有字段算）：
  ``score = α·epistemic + β·progress + γ·reachable``
    epistemic ≈ 条目罕见度（recall_count 低）+ 置信度中高（有可学空间，
      Ten 2021 学习进度——中高置信 = 有可学空间，过低/过高都低分）；
    progress  ≈ 近 N 轮 surprise 差分 >0 且未饱和（学习进度信号；窗口
      不足/缺省 → 中性 0.5）；
    reachable ≈ 存在可检索的旧条目（语义向量在场——Dubey & Griffiths
      2020 理性好奇排序：不可达的悬案学了也用不上）。

过滤：目标条目已 superseded / 悬案已 resolved / satiety 冷却中
（excluded_topics 由调用方传入）→ 跳过。

约束（同 reconsolidation_queue）：纯 stdlib、可单测、fail-open（任何
异常 → 该候选跳过，绝不抛给调用方）；对条目只读（getattr，绝不 setattr）。
"""

from __future__ import annotations

import hashlib
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence

# ---------------------------------------------------------------------- #
#  env 权重（§5.3：单一默认值来源；运行时读取，改动即时生效）
# ---------------------------------------------------------------------- #

_ENV_ALPHA = "LMS_E3_SELECTOR_ALPHA"
_ENV_BETA = "LMS_E3_SELECTOR_BETA"
_ENV_GAMMA = "LMS_E3_SELECTOR_GAMMA"
_ENV_TOPIC_LEN = "LMS_E3_REACTIVATE_TOPIC_LEN"

_DEFAULT_WEIGHTS = {"alpha": 0.5, "beta": 0.3, "gamma": 0.2}
_DEFAULT_TOPIC_LEN = 120

#: 内部子项比例（epistemic = 0.5×罕见度 + 0.5×置信度可学空间；工程假设，
#: 溯源 §4.2 标注——不参与 env，验收后按需标定）
_EPISTEMIC_RARITY_W = 0.5
_EPISTEMIC_CONF_W = 0.5

#: 置信度"中高可学空间"峰值（Ten 2021 学习进度：中高置信最有可学空间）
_CONF_PEAK = 0.7

#: progress 的 surprise 差分尺度与饱和线（工程假设，溯源 §4.2 标注）
_PROGRESS_RISING_SCALE = 2.0
_PROGRESS_SAT_CEILING = 8.0

#: 重叠匹配窗口（与 doubt_ingest._find_overlapping_entry 同口径）
_OVERLAP_LEN = 120


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, "") or default)
    except (TypeError, ValueError):
        return float(default)


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def default_weights() -> Dict[str, float]:
    """选择器权重：env（LMS_E3_SELECTOR_ALPHA/BETA/GAMMA）> 默认 0.5/0.3/0.2。

    返回归一化后的 {alpha, beta, gamma}；全零/非法 → 默认权重（fail-open）。
    """
    a = _env_float(_ENV_ALPHA, _DEFAULT_WEIGHTS["alpha"])
    b = _env_float(_ENV_BETA, _DEFAULT_WEIGHTS["beta"])
    g = _env_float(_ENV_GAMMA, _DEFAULT_WEIGHTS["gamma"])
    total = a + b + g
    if total <= 0 or not all(math.isfinite(v) for v in (a, b, g)):
        return dict(_DEFAULT_WEIGHTS)
    return {"alpha": a / total, "beta": b / total, "gamma": g / total}


def default_topic_len() -> int:
    """线索/证据摘要长度上限（LMS_E3_REACTIVATE_TOPIC_LEN，默认 120）。"""
    try:
        return max(40, int(os.environ.get(_ENV_TOPIC_LEN, "") or _DEFAULT_TOPIC_LEN))
    except (TypeError, ValueError):
        return _DEFAULT_TOPIC_LEN


# ---------------------------------------------------------------------- #
#  纯函数：稳定键 / 字段读取 / 重叠匹配
# ---------------------------------------------------------------------- #

def _stable_key(text: str) -> str:
    """悬案稳定键（与 reconsolidation_queue.entry_key 同约定：sha1 前缀）。

    只用于选择器内部去重/匹配；真正入队时由 loop 用目标条目调用
    ``reconsolidation_queue.entry_key(entry)`` 得同构键（条目无 id 字段
    → 同为文本 sha1 前缀，两端一致）。
    """
    t = (text or "").strip()
    if not t:
        return ""
    return "sha1:" + hashlib.sha1(t.encode("utf-8", "replace")).hexdigest()[:24]


def _entry_field(entry: Any, name: str, default: Any = None) -> Any:
    """条目字段安全读取（dict / 对象 / 缺失字段 → default，fail-open）。"""
    if entry is None:
        return default
    try:
        if isinstance(entry, dict):
            return entry.get(name, default)
        return getattr(entry, name, default)
    except Exception:  # pylint: disable=broad-except
        return default


def _entry_text(entry: Any) -> str:
    return str(_entry_field(entry, "text", "") or "").strip()


def _find_target_entry(episodic: Sequence[Any], needle: str) -> Optional[Any]:
    """在 episodic 条目中找与 needle 内容重叠的目标条目（纯工程匹配）。

    与 doubt_ingest._find_overlapping_entry 同口径：needle 出现在条目文本
    中，或条目文本前缀出现在 needle 中。找不到 → None（fail-open）。
    """
    if not episodic:
        return None
    needle = (needle or "").strip()[:_OVERLAP_LEN]
    if not needle:
        return None
    try:
        for entry in episodic:
            text = _entry_text(entry)
            if text and (needle in text or text[:_OVERLAP_LEN] in needle):
                return entry
    except Exception:  # pylint: disable=broad-except
        return None
    return None


def _is_superseded(entry: Any) -> bool:
    return str(_entry_field(entry, "doubt_state", "stable") or "stable") \
        == "superseded"


# ---------------------------------------------------------------------- #
#  三项评分（全从既有字段算，无 LLM）
# ---------------------------------------------------------------------- #

def _epistemic_score(*, recall_count: Optional[float] = None,
                     confidence: Optional[float] = None) -> float:
    """epistemic 项：罕见度（recall_count 低）+ 置信度中高（有可学空间）。

    罕见度 = 1/(1+recall_count)：recall 0 → 1.0，recall 10 → ~0.09。
    可学空间 = 1 - |conf - 0.7|×2（clamp01）：峰值 0.7 → 1.0；过低置信
    （<0.2）→ 0（已在低置信通道）；过高（1.0）→ 0.4（仍有空间）。
    """
    try:
        recall = max(0.0, float(recall_count if recall_count is not None else 0.0))
    except (TypeError, ValueError):
        recall = 0.0
    try:
        conf = float(confidence if confidence is not None else _CONF_PEAK)
    except (TypeError, ValueError):
        conf = _CONF_PEAK
    rarity = 1.0 / (1.0 + recall)
    headroom = _clamp01(1.0 - abs(conf - _CONF_PEAK) * 2.0)
    return _clamp01(
        _EPISTEMIC_RARITY_W * rarity + _EPISTEMIC_CONF_W * headroom)


def _progress_score(surprise_window: Optional[Sequence[float]]) -> float:
    """progress 项：近 N 轮 surprise 差分 >0 且未饱和（学习进度信号）。

    差分 = 后半窗均值 - 前半窗均值（窗口不足 3 条 → 中性 0.5，fail-open）；
    progress = 0.5 + 0.5×tanh(差分 / 尺度)：差分 +2 → ~0.88，-2 → ~0.12；
    未饱和判据：近期均值 ≤ 饱和线（默认 8.0），超过 → ×0.5 折半（已饱和
    无可学）。全局窗口是逐 topic 差分的工程代理（loop 侧无 per-topic
    窗口，溯源 §4.2 标注）。
    """
    window = list(surprise_window or [])
    if len(window) < 3:
        return 0.5
    half = max(1, len(window) // 2)
    older = window[:half]
    recent = window[-half:]
    try:
        mean_older = sum(older) / len(older)
        mean_recent = sum(recent) / len(recent)
        rising = float(mean_recent) - float(mean_older)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(rising):
        return 0.5
    progress = 0.5 + 0.5 * math.tanh(rising / _PROGRESS_RISING_SCALE)
    if mean_recent > _PROGRESS_SAT_CEILING:
        progress *= 0.5
    return _clamp01(progress)


def _reachable_score(entry: Any) -> float:
    """reachable 项：存在可检索的旧条目（语义向量在场）。

    目标条目存在且带 semantic_vector → 1.0（检索可达，学了有用）；
    存在但无向量 → 0.5（文本仍在，检索弱）；不存在 → 0.0。
    """
    if entry is None:
        return 0.0
    vec = _entry_field(entry, "semantic_vector", None)
    return 1.0 if vec is not None else 0.5


# ---------------------------------------------------------------------- #
#  候选构造
# ---------------------------------------------------------------------- #

def _build_candidate(*, kind: str, topic: str, detail: str, entry: Any,
                     episodic: Sequence[Any], surprise_window: Sequence[float],
                     weights: Dict[str, float]) -> Dict[str, Any]:
    """构造一个候选 dict（纯函数；评分三项全算）。"""
    target_text = _entry_text(entry) if entry is not None else ""
    if entry is not None:
        recall = _entry_field(entry, "recall_count", 0)
        conf = _entry_field(entry, "confidence", _CONF_PEAK)
    else:
        recall = 0
        conf = _CONF_PEAK
    epi = _epistemic_score(recall_count=recall, confidence=conf)
    prog = _progress_score(surprise_window)
    reach = _reachable_score(entry)
    score = (weights["alpha"] * epi + weights["beta"] * prog
             + weights["gamma"] * reach)
    return {
        "kind": kind,
        "topic": str(topic)[:default_topic_len()],
        "detail": str(detail)[:300],
        "entry_key": _stable_key(target_text) if target_text else "",
        "target_text": target_text,
        "target_found": entry is not None,
        "score": round(float(score), 4),
        "epistemic": round(float(epi), 4),
        "progress": round(float(prog), 4),
        "reachable": round(float(reach), 4),
    }


# ---------------------------------------------------------------------- #
#  主入口
# ---------------------------------------------------------------------- #

def select_cases(
    fok_list: Iterable[Dict[str, Any]],
    lowconf_list: Iterable[Dict[str, Any]],
    episodic: Optional[Sequence[Any]] = None,
    surprise_window: Optional[Sequence[float]] = None,
    weights: Optional[Dict[str, float]] = None,
    excluded_topics: Optional[Iterable[str]] = None,
    max_candidates: int = 20,
) -> List[Dict[str, Any]]:
    """E3 选料：候选源（fok ∪ lowconf）→ 过滤 → 排序 → top-N。

    参数:
        fok_list: A 类 fok_unresolved 记录（含 'topic'/'detail'）——调用方
            可把 suspect 未决条目按 fok 形态混入（loop 侧处理）。
        lowconf_list: B 类 low_confidence_unreviewed 记录（含
            'text'/'confidence'/'rebuttal_count'）。
        episodic: 可检索的旧条目序列（None → reachable=0，仅文本匹配）。
        surprise_window: 近 N 轮 surprise 观测（None → progress 中性）。
        weights: {alpha, beta, gamma}（None → default_weights()）。
        excluded_topics: 已 resolved / satiety 冷却中的 topic 集合（跳过）。
        max_candidates: 返回条数上限（默认 20）。

    返回:
        按 score 降序的候选 dict 列表（无 LLM、纯 stdlib、fail-open）。
    """
    weights = weights if isinstance(weights, dict) else default_weights()
    weights = {
        "alpha": _clamp01(float(weights.get("alpha", 0.5))),
        "beta": _clamp01(float(weights.get("beta", 0.3))),
        "gamma": _clamp01(float(weights.get("gamma", 0.2))),
    }
    if weights["alpha"] + weights["beta"] + weights["gamma"] <= 0:
        weights = dict(_DEFAULT_WEIGHTS)
    episodic = list(episodic) if episodic is not None else []
    excluded = set(str(t) for t in (excluded_topics or []))
    candidates: List[Dict[str, Any]] = []
    seen_topics = set()

    def _append(cand: Dict[str, Any]) -> None:
        topic = str(cand.get("topic", "")).strip()
        if not topic or topic in seen_topics:
            return  # 去重（同 topic 只留首个）
        seen_topics.add(topic)
        candidates.append(cand)

    try:
        for rec in fok_list:
            try:
                topic = str(rec.get("topic", "") or "").strip()
                if not topic or topic in excluded:
                    continue
                detail = str(rec.get("detail", "") or "")
                entry = _find_target_entry(episodic, topic)
                if entry is not None and _is_superseded(entry):
                    continue  # 已 superseded → 跳过
                _append(_build_candidate(
                    kind="fok", topic=topic, detail=detail, entry=entry,
                    episodic=episodic, surprise_window=surprise_window,
                    weights=weights))
            except Exception:  # pylint: disable=broad-except
                continue
        for rec in lowconf_list:
            try:
                topic = str(rec.get("text", "") or "").strip()
                if not topic or topic in excluded:
                    continue
                detail = "低置信未复核（confidence=%s）" % str(
                    rec.get("confidence", ""))
                entry = _find_target_entry(episodic, topic)
                if entry is not None and _is_superseded(entry):
                    continue
                _append(_build_candidate(
                    kind="lowconf", topic=topic, detail=detail, entry=entry,
                    episodic=episodic, surprise_window=surprise_window,
                    weights=weights))
            except Exception:  # pylint: disable=broad-except
                continue
    except Exception:  # pylint: disable=broad-except
        pass
    candidates.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    return candidates[:max(1, int(max_candidates))]

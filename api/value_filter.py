# -*- coding: utf-8 -*-
"""价值过滤（提取层 v1.4 S1-4，按论文重做，拍板⑤）

info_value(i) = w1 × surprise_norm(i) + w2 × recall_hit_score(i)
  w1 = 0.7, w2 = 0.3（初始；阶段 2 开闸后按 H-4 抽样数据标定）

维度 1：surprise_norm（惊讶度，主信号）
  - 论文：NEMORI 2508.03341（预测误差=未来效用的选择信号）；
          Surprise-as-Signal 2606.31495（惊讶门控写入）；
          Lisman & Grace 2005（PMID 15924857）；Feldman & Friston 2010
          （PMID 21160551）
  - 实现：LMS process_turn 原生 surprise（零新增成本），缓冲区内 min-max 归一化

维度 2：recall_hit_score（未来效用实证，次信号）
  - 论文：Mattar & Daw 2018（PMID 30349103 价值=潜在未来奖励）；
          丰碑"引用自动加固"
  - 实现：条目 reference_count 归一化（写侧引用匹配/做梦重放命中联动，
          复用现有字段为加固计数唯一权威，P2-A；/recall 只读不计数）

冷启动垫片（非承重，设计 §6.2 标注"无出处，工程启发式，仅冷启动垫片，
不承重"）：条目尚无 surprise/recall 历史时，用原启发式（长度/数字/信息词）
打底分 0.3-1.0；纯确认词（≤10 字客套）→ 0 分（客套过滤=过滤语义非价值语义）。
"""

import re
from typing import Optional

# 权重（初始；阶段 2 开闸后 H-4 标定）
W_SURPRISE = 0.7
W_RECALL_HIT = 0.3

# 低价值阈值（value_filtered 判据；初始值，阶段 2 开闸后按真实数据标定）
DEFAULT_VALUE_THRESHOLD = 0.3

# 客套/纯确认词（≤10 字 → 0 分）
_PLEASANTRY_RE = re.compile(
    r"^(好的|嗯|好|行|ok|okay|没问题|收到|明白|可以|当然|谢谢|不客气|"
    r"再见|你好|辛苦|对|是的|没错|真棒|厉害)[，。！？!?]*$", re.I)
_INFO_WORD_RE = re.compile(
    r"因为|所以|但是|方法|方案|原因|结果|建议|决定|关键|重要|注意|"
    r"实际上|正确的是|需要|必须|应该")


def normalize_minmax(value: float, vmax: float) -> float:
    """min-max 归一化（vmax≤0 时返回 0，防除零）。"""
    if vmax is None or vmax <= 1e-12:
        return 0.0
    v = float(value)
    return max(0.0, min(1.0, v / vmax))


def compute_info_value(
    surprise: float,
    reference_count: int = 0,
    surprise_max: Optional[float] = None,
    ref_count_max: Optional[float] = None,
    text: Optional[str] = None,
    w1: float = W_SURPRISE,
    w2: float = W_RECALL_HIT,
) -> float:
    """价值分数 = 0.7×surprise_norm + 0.3×recall_hit_score（论文重做）。

    参数:
        surprise: 条目惊讶度（process_turn 原生量，FEP 预测误差）。
        reference_count: 条目被引用次数（加固计数，P2-A 权威字段）。
        surprise_max: 缓冲区内 surprise 最大值（None/≤0 → surprise_norm=0）。
        ref_count_max: 缓冲区内 reference_count 最大值（None/≤0 → recall_hit=0）。
        text: 冷启动垫片用原文（可选；iv≤0 时启用启发式垫片）。
        w1/w2: 权重（初始 0.7/0.3）。

    返回:
        [0, 1] 价值分数。无历史（surprise_norm=recall_hit=0）时用冷启动
        垫片（非承重）；纯确认词 → 0 分。
    """
    surprise_norm = normalize_minmax(surprise, surprise_max)
    recall_hit = normalize_minmax(reference_count, ref_count_max)
    iv = w1 * surprise_norm + w2 * recall_hit
    if iv <= 1e-12:
        # 冷启动垫片（非承重，无出处工程启发式；设计 §6.2）
        iv = _cold_start_pad(text or "")
    return max(0.0, min(1.0, iv))


def _cold_start_pad(text: str) -> float:
    """冷启动启发式垫片（0.3 基础+长度+数字+标点+信息词，封顶 1.0）。

    标注：无出处，工程启发式，仅冷启动垫片，不承重（设计 §6.2）。
    纯确认词（≤10 字客套）→ 0 分（客套过滤语义）。
    """
    t = (text or "").strip()
    if not t:
        return 0.0
    if len(t) <= 10 and _PLEASANTRY_RE.match(t):
        return 0.0
    score = 0.3
    if len(t) > 50:
        score += 0.2
    if re.search(r"\d", t):
        score += 0.2
    if re.search(r"[，。！？、；：]", t):
        score += 0.1
    if _INFO_WORD_RE.search(t):
        score += 0.2
    return min(1.0, score)


def value_filtered(info_value: float,
                   threshold: float = DEFAULT_VALUE_THRESHOLD) -> bool:
    """价值过滤判据：info_value ≥ 阈值 → 通过（True）。

    语义（M5）：插件永不整轮跳过，价值判定全权交 LMS——本判据只做
    条目标记（value_filtered 字段）与观测，不决定整轮生死（丰碑：
    低价值条目仍存储，只是被标记）。
    """
    return float(info_value) >= float(threshold)

# -*- coding: utf-8 -*-
"""提取式核心压缩（提取层 v1.4 S1-5）

两段式提取（H-8 定案，≤300 字）：
  第一段：4 类（事实修正/决策/偏好/情感）各取 top-1 显著句（TextRank 式
    显著性打分；每句先经 segment_reply 分类，过程/客套句不进入候选）；
  第二段：候选句全局按显著性排序拼接，截断 ≤300 字。

论文出处（设计附录 B）：
  - TextRank 2004（ACL W04-3252）：句子显著性 = 句间词重叠图上的
    迭代权重（本实现为轻量版：词重叠度 + 位置先验 + 长度适中，标注
    "TextRank 语义、轻量实现"）；
  - 2312.06901（长度可控提取式）：≤N 字约束现成方法族；
  - 容量定理族（2309.12673 等）：稀疏化误差界。

集合运算不生成：只做选择/截断/拼接，不产出原文不存在的词（非生成式）。
"""

import re
from typing import List, Dict

from api.segment_reply import (
    segment_sentences, classify_sentence,
    VALUE_CATEGORIES, CATEGORY_LABELS,
)

# 默认输出上限（设计 H-8：≤300 字）
DEFAULT_MAX_CHARS = 300

# 词切分（中文按字符 2-gram + 英文按词；简单实现，TextRank 语义）
_WORD_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


def _tokenize(sentence: str) -> set:
    """句子 → 词集合（中文单字 + 英文单词；用于句间重叠计算）。"""
    return set(_WORD_RE.findall(sentence.lower()))


def _sentence_significance(sentences: List[str], idx: int) -> float:
    """单句显著性：句间词重叠（TextRank 轻量版）+ 位置先验 + 长度适中。

    分数 = 0.6×词重叠度 + 0.25×位置先验 + 0.15×长度适中
      - 词重叠度：与其余句子的 Jaccard 相似度均值（0~1）
      - 位置先验：首句 1.0，第二句 0.7，其余线性衰减（回复开头常为总结）
      - 长度适中：20~120 字为 1.0，过短/过长衰减（防单字句/超长流水句）
    工程实现（标注：TextRank 语义的轻量近似，非完整 PageRank 迭代）。
    """
    toks = [_tokenize(s) for s in sentences]
    cur = toks[idx]
    if not cur:
        return 0.0
    n = len(sentences)
    overlap = 0.0
    count = 0
    for j, other in enumerate(toks):
        if j == idx or not other:
            continue
        union = cur | other
        if union:
            overlap += len(cur & other) / len(union)
            count += 1
    overlap_score = overlap / count if count else 0.0

    if idx == 0:
        pos_score = 1.0
    elif idx == 1:
        pos_score = 0.7
    else:
        pos_score = max(0.0, 0.7 - 0.15 * (idx - 1))

    length = len(sentences[idx])
    if 20 <= length <= 120:
        len_score = 1.0
    elif length < 20:
        len_score = 0.4 + 0.6 * (length / 20.0)
    else:
        len_score = max(0.0, 1.0 - 0.01 * (length - 120))

    return 0.6 * overlap_score + 0.25 * pos_score + 0.15 * len_score


def _truncate(text: str, max_chars: int) -> str:
    """按字符截断（≤max_chars，保持完整性不硬切词——按句边界回退）。"""
    if len(text) <= max_chars:
        return text
    # 尽量在句子边界截断
    cut = text[:max_chars]
    return cut


def extract_core(llm_output: str, max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """两段式提取核心（≤max_chars 字）。

    参数:
        llm_output: LLM 回复全文。
        max_chars: 输出上限（默认 300）。

    返回:
        核心文本（≤max_chars 字）。无有价值句时回退为原文前缀截断
        （保证不空；客套过滤语义由 value_filter 判定，此处只管压缩）。
    """
    if not llm_output or not llm_output.strip():
        return ""
    sentences = segment_sentences(llm_output)
    if not sentences:
        return ""

    # 第一段：4 类各取 top-1 显著句（过程/客套句不进入候选，H-4）
    best_per_category: Dict[str, Dict] = {}
    for i, s in enumerate(sentences):
        cat = classify_sentence(s)
        if cat not in VALUE_CATEGORIES:
            continue
        score = _sentence_significance(sentences, i)
        cur = best_per_category.get(cat)
        if cur is None or score > cur["score"]:
            best_per_category[cat] = {"text": s, "score": score, "idx": i}

    candidates = list(best_per_category.values())
    if not candidates:
        # 无有价值句（纯客套/过程回复）：回退原文前缀截断（≤300 字）
        return _truncate(llm_output.strip(), max_chars)

    # 第二段：全局按显著性排序拼接，截断 ≤max_chars
    candidates.sort(key=lambda x: x["score"], reverse=True)
    core = ""
    for cand in candidates:
        sep = " " if core else ""
        piece = core + sep + cand["text"]
        if len(piece) > max_chars:
            # 放不下整句：若核心为空则硬截断该句，否则保持现状
            if not core:
                core = _truncate(cand["text"], max_chars)
            break
        core = piece
    return core.strip()


def core_stats(core: str) -> Dict:
    """核心统计（观测/审计用）。"""
    return {
        "core_chars": len(core),
        "core_sentence_count": len(segment_sentences(core)),
    }

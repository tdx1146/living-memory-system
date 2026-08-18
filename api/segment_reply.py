# -*- coding: utf-8 -*-
"""分段标记：规则分类器（提取层 v1.4 S1-3）

把 LLM 回复按句子分段并打上类别标记，供提取式压缩（S1-5 extract_core）
选择"有价值"的句子（事实修正/决策/偏好/情感），排除过程句/客套句污染
（H-4 语义：提取核心无过程句/客套句）。

类别（论文语义参照，规则本身为工程实现、无独立论文，按设计标注）：
  - fact_correction 事实修正   （TiMem 2601.02845 层级抽象；修正=高信息量）
  - decision         决策      （AMD 2608.07169 中间粒度最有价值）
  - preference       偏好      （同上）
  - emotion          情感      （NEMORI 叙事整合）
  - process          过程      （衔接/叙述性，低提取价值）
  - pleasantry       客套      （纯确认/客套，过滤语义——非价值语义）

VALUE_CATEGORIES = 前四类（提取候选）；过程/客套不进入核心提取。
"""

import re
from typing import List, Dict

# 类别常量（提取候选 = 有价值四类）
CATEGORY_FACT = "fact_correction"
CATEGORY_DECISION = "decision"
CATEGORY_PREFERENCE = "preference"
CATEGORY_EMOTION = "emotion"
CATEGORY_PROCESS = "process"
CATEGORY_PLEASANTRY = "pleasantry"

VALUE_CATEGORIES = (
    CATEGORY_FACT, CATEGORY_DECISION,
    CATEGORY_PREFERENCE, CATEGORY_EMOTION,
)

# 类别名 → 中文（观测/审计用）
CATEGORY_LABELS = {
    CATEGORY_FACT: "事实修正",
    CATEGORY_DECISION: "决策",
    CATEGORY_PREFERENCE: "偏好",
    CATEGORY_EMOTION: "情感",
    CATEGORY_PROCESS: "过程",
    CATEGORY_PLEASANTRY: "客套",
}

# 句子切分：中文句号/感叹/问号/分号/省略号 + 英文句点/问号/感叹
_SENT_SPLIT_RE = re.compile(
    r"(?<=[。！？!?；;…])\s*|(?<=[.])\s+(?=[A-Z0-9\u4e00-\u9fff])")

# 类别关键词规则（工程规则，按设计标注"规则本身为工程实现（无独立论文）"）
_RULES: List[tuple] = [
    (CATEGORY_FACT, re.compile(
        r"修正|更正|其实|实际上|不对|错了|正确的是|准确|纠正|重要|关键|"
        r"注意|提醒|真实|事实|正确做法|应该是")),
    (CATEGORY_DECISION, re.compile(
        r"决定|选择|采用|方案|计划|安排|建议|推荐|应该|可以|将|后续|下一步|"
        r"优先|策略|做法是|准备|开始|停止|取消")),
    (CATEGORY_PREFERENCE, re.compile(
        r"喜欢|偏好|更愿意|希望|想要|倾向|在意|重视|在乎|觉得\S*好|不喜欢|"
        r"最想|理想|期望")),
    (CATEGORY_EMOTION, re.compile(
        r"开心|高兴|难过|伤心|生气|愤怒|担心|焦虑|感谢|抱歉|对不起|爱你|"
        r"感动|失望|欣慰|紧张|害怕|兴奋")),
    (CATEGORY_PLEASANTRY, re.compile(
        r"^(好的|嗯|没问题|不客气|再见|你好|请问|辛苦|谢谢|明白|收到|"
        r"没问题|可以啊|当然|对的|是的)[，。！？!?]*$")),
]

# 纯客套短语（短确认词，分类器兜底）
_PURE_PLEASANTRY_RE = re.compile(
    r"^(好的|嗯|好|行|ok|okay|没问题|收到|明白|可以|当然|谢谢|不客气|"
    r"再见|你好|辛苦|对|是的|没错|真棒|厉害)[，。！？!?]*$", re.I)


def segment_sentences(text: str) -> List[str]:
    """把回复切分为句子列表（保留原文，去除首尾空白）。

    中文标点（。！？；…）直接切；英文句点后跟大写/中文才切（防缩写误切）。
    """
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text)]
    return [p for p in parts if p]


def classify_sentence(sentence: str) -> str:
    """规则分类单句 → 类别（工程规则实现；客套短语优先于关键词命中）。

    返回:
        六类之一（VALUE_CATEGORIES 为提取候选；process/pleasantry 不进入核心）。
    """
    s = sentence.strip()
    if not s:
        return CATEGORY_PROCESS
    # 纯客套短语兜底（短确认词优先，防"好的，我们可以用方案A"误判客套）
    if _PURE_PLEASANTRY_RE.match(s) and len(s) <= 12:
        return CATEGORY_PLEASANTRY
    for cat, pattern in _RULES:
        if pattern.search(s):
            return cat
    # 无关键词命中：短句（≤8 字）倾向客套/过程，长句归过程（叙述性）
    if len(s) <= 8:
        return CATEGORY_PLEASANTRY
    return CATEGORY_PROCESS


def segment_reply(text: str) -> List[Dict]:
    """分段标记入口：回复 → [{'text': 句子, 'category': 类别}, ...]。

    参数:
        text: LLM 回复全文。

    返回:
        句子分段列表（按原文顺序）；空输入返回空列表。
    """
    return [
        {"text": s, "category": classify_sentence(s)}
        for s in segment_sentences(text)
    ]


def valuable_sentences(text: str) -> List[Dict]:
    """只保留有价值类别的句子（事实修正/决策/偏好/情感）。

    extract_core 的两段式第一段候选来源；过程/客套句被排除（H-4）。
    """
    return [seg for seg in segment_reply(text)
            if seg["category"] in VALUE_CATEGORIES]

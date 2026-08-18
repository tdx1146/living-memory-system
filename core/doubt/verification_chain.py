# -*- coding: utf-8 -*-
"""core/doubt · 验证链原生骨架（M3-1 · §2.2 接口/事件/幂等键协议）

规格依据：`四妹-LMS核心重写规格v2-20260817.md` §2.2（验证链原生——
防伪独立四维 + 矛盾判定三选一 + 幂等防伪）。

M3-1 交付**骨架**：接口、事件类型、幂等键协议、防伪独立四维的脚手架。
**M3-2 交付判定细节**：矛盾判定三选一（方向/数值/否定）+ 元数据排除
（时间戳碰撞/同义复述/来源互补——P0 假冲突根因①②根治）+ 验证链全链
（草稿→独立验证→修正：``verify`` / ``run_pending`` 驱动，VERIFY-*
provenance 全程记录，CONFLICT 结果由写侧经 ``[doubt] conflict`` → labile）。
本模块 ``submit_result`` 按协议登记结果；判定由 ``is_contradiction`` /
``judge_contradiction``（纯 stdlib 规则语义，无 LLM 依赖）给出。

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
import re
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

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
#  矛盾判定三选一 + 元数据排除（M3-2：P0 假冲突根因①②的语义根治）
# ---------------------------------------------------------------------- #
#
# 设计约束：纯 stdlib 规则判定（无 LLM 依赖——同 doubt_ingest 工程惯例，
# 溯源 §4.2 标注"无 LLM 用规则近似"）。判定顺序：
#   1. 元数据排除先行：时间戳/日期前缀碰撞、同义复述、来源互补、显式
#      排除 → 一律不判矛盾（metadata_excluded=True——假阳性演练红线）；
#   2. 三选一：否定矛盾 → 方向矛盾 → 数值矛盾（存在性/成立性否定优先
#      于方向对立——"不应开启 A"是"应开启 A"的否定而非方向冲突）；
#   3. 其余一律 NOT_A_CONFLICT（宁可漏判不误判——P0 的失败方向是假冲突）。

#: 时间戳/日期形态（元数据排除①：剔除后再做语义判定——旧 overlapMatch
#: 纯子串碰撞把"[Thu 2026-08-06 00:11 GMT+8] 开工吧"类日期前缀判成冲突
#: 的根因修复）
_DATETIME_PATTERNS = [
    re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}(?:[T\s]\d{1,2}:\d{2}(?::\d{2})?)?\b", re.I),  # ISO 日期/时间
    re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b"),                                        # 时钟时间戳
    re.compile(r"\d{4}年\d{1,2}月\d{1,2}[日号]?"),                                      # 中文全日期
    re.compile(r"\d{1,2}月\d{1,2}[日号]?"),                                              # 中文月日
    re.compile(r"[\[\(][^\]\)]*?(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[^\]\)]*?[\]\)]", re.I),  # 英文星期括号
    re.compile(r"[\[\(][^\]\)]*?\d{4}-\d{1,2}-\d{1,2}[^\]\)]*?[\]\)]"),                  # 含 ISO 日期括号
]

#: 语气/程度虚词（归一剔除——"不同措辞同一结论"的同义复述判定基础；
#: 否定标记（不/没/无/非/未/勿）**绝不**在归一阶段剔除，留给否定矛盾判定）
_STRIP_WORDS = ("应该", "应当", "应", "很", "非常", "十分", "特别")

#: 句尾语气词（仅句尾剔除——避免误伤词内字："目的/的确"里的"的"不动）
_TRAILING_PARTICLES = "的了吗呢吧啊呀哦嗯"

#: 否定标记（否定矛盾：一条断言直接否定另一条的存在性/成立性）。
#: 长标记在前（"没有"先于"没"）。"别"不入列（"别人/别墅"歧义过大）。
_NEGATION_MARKERS = ("没有", "不", "没", "无", "非", "未", "勿")

#: 否定标记的例外词（标记字不是否定的词——"不错/不客气/了不起"等）
_NEGATION_EXCEPTION_WORDS = ("不错", "不客气", "了不起", "非但", "非常",
                             "未央", "未来", "未免", "无非", "无论",
                             "无妨", "无须", "无所")

#: 方向矛盾词对（同一事实两条断言结论相反）。长词/专指词在前——
#: （开启,关闭）先于（开,关），避免"开启"里的"开"被单字词对误拆。
_DIRECTIONAL_PAIRS = (
    ("开启", "关闭"), ("打开", "关闭"), ("启用", "停用"), ("启用", "禁用"),
    ("支持", "反对"), ("赞成", "反对"), ("同意", "反对"), ("支持", "抵制"),
    ("增加", "减少"), ("提高", "降低"), ("上升", "下降"), ("上涨", "下跌"),
    ("开始", "停止"), ("继续", "停止"), ("保留", "删除"), ("保留", "移除"),
    ("通过", "拒绝"), ("接受", "拒绝"), ("允许", "禁止"), ("允许", "拒绝"),
    ("买", "卖"), ("涨", "跌"), ("前进", "后退"),
    ("开", "关"), ("是", "否"),
    ("enable", "disable"), ("support", "oppose"), ("agree", "disagree"),
    ("accept", "reject"), ("increase", "decrease"), ("raise", "lower"),
    ("on", "off"), ("yes", "no"), ("true", "false"), ("up", "down"),
    ("buy", "sell"), ("allow", "forbid"), ("start", "stop"),
    ("open", "close"),
)

#: 数值断言单位（量纲归一：同量纲同单位才可比；"1500元 vs 800公里"
#: 不同量纲不判矛盾）。乘子：万/亿/千为数量级修饰（1500万 → 1.5e7）。
_UNIT_TABLE = (
    ("平方公里", "area_km2", 1.0), ("平方米", "area_m2", 1.0),
    ("公里", "km", 1.0), ("千米", "km", 1.0), ("kg", "kg", 1.0),
    ("公斤", "kg", 1.0), ("厘米", "cm", 1.0), ("毫米", "mm", 1.0),
    ("米", "m", 1.0),
    ("分钟", "min", 1.0), ("小时", "h", 1.0), ("秒钟", "s", 1.0),
    ("秒", "s", 1.0), ("天", "day", 1.0), ("个月", "month", 1.0),
    ("年", "year", 1.0), ("岁", "age", 1.0),
    ("万元", "money", 1e4), ("亿元", "money", 1e8),
    ("元", "money", 1.0), ("块", "money", 1.0), ("美元", "money_usd", 1.0),
    ("%", "percent", 1.0), ("百分比", "percent", 1.0),
    ("个", "count", 1.0), ("人", "count", 1.0), ("次", "count", 1.0),
    ("家", "count", 1.0), ("台", "count", 1.0), ("辆", "count", 1.0),
    ("条", "count", 1.0), ("名", "count", 1.0),
    ("万", "number", 1e4), ("亿", "number", 1e8), ("千", "number", 1e3),
)
_UNIT_TABLE_SORTED = tuple(sorted(_UNIT_TABLE, key=lambda x: -len(x[0])))

_PUNCT_CHARS = "，。！？；：、,.!?;:()（）[]【】{}<>《》\"'“”‘’…—·-–~～"

_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
_RANGE_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(?:[-~～至到])\s*(\d[\d,]*(?:\.\d+)?)")


def _strip_datetime_tokens(text: str) -> str:
    """剔除时间戳/日期形态（元数据排除①）。"""
    s = str(text or "")
    for pat in _DATETIME_PATTERNS:
        s = pat.sub("", s)
    return s


def normalize_claim(text: str) -> str:
    """语义归一：剔除时间戳 → 小写/去标点空白 → 去语气虚词/句尾语气词。

    否定标记（不/没/无/非/未/勿）**保留**——否定矛盾判定依赖它们。
    """
    s = _strip_datetime_tokens(text)
    s = s.lower()
    for ch in _PUNCT_CHARS:
        s = s.replace(ch, " ")
    s = "".join(s.split())
    for w in _STRIP_WORDS:
        s = s.replace(w, "")
    while s and s[-1] in _TRAILING_PARTICLES:
        s = s[:-1]
    return s


def _shingles(s: str) -> set:
    if not s:
        return set()
    if len(s) <= 2:
        return {s}
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _overlap_ratio(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa), len(sb))


def _remove_negation_marker(s: str, marker: str) -> Optional[str]:
    """去掉一次否定标记（跳过例外词内的标记字）；无 → None。"""
    start = 0
    while True:
        i = s.find(marker, start)
        if i < 0:
            return None
        span = s[max(0, i - 1): i + len(marker) + 1]
        if any(exp in span for exp in _NEGATION_EXCEPTION_WORDS):
            start = i + 1
            continue
        return s[:i] + s[i + len(marker):]


def _is_negation(a: str, b: str) -> bool:
    """否定矛盾（三选一②）：一条断言直接否定另一条的存在性/成立性。"""
    if not a or not b or a == b:
        return False
    for marker in _NEGATION_MARKERS:
        if _remove_negation_marker(a, marker) == b:
            return True
        if _remove_negation_marker(b, marker) == a:
            return True
    return False


def _is_directional(a: str, b: str) -> bool:
    """方向矛盾（三选一①）：同一事实两条断言结论相反。"""
    for w1, w2 in _DIRECTIONAL_PAIRS:
        # "开关/是否"式自含词防误判：任一侧同时含词对两词 → 跳过
        if w1 in a and w2 in a:
            continue
        if w1 in b and w2 in b:
            continue
        if not ((w1 in a and w2 in b) or (w2 in a and w1 in b)):
            continue
        ra = a.replace(w1, "").replace(w2, "")
        rb = b.replace(w1, "").replace(w2, "")
        if not ra or not rb:
            continue
        if ra == rb or _overlap_ratio(ra, rb) >= 0.6:
            return True
    return False


def _match_unit(text: str, pos: int) -> Tuple[str, str, float]:
    """取数字后的单位（最长匹配）→ (单位串, 量纲, 乘子)。"""
    rest = text[pos:]
    for unit, dim, mult in _UNIT_TABLE_SORTED:
        if rest.startswith(unit):
            return unit, dim, mult
    return "", "dimensionless", 1.0


def _extract_numbers(text: str) -> List[Tuple[float, float, str]]:
    """提取数值断言（范围优先）→ [(lo, hi, 量纲), ...]。"""
    s = _strip_datetime_tokens(text)
    out = []
    for m in _RANGE_RE.finditer(s):
        lo = float(m.group(1).replace(",", ""))
        hi = float(m.group(2).replace(",", ""))
        if lo > hi:
            lo, hi = hi, lo
        _, dim, mult = _match_unit(s, m.end())
        out.append((lo * mult, hi * mult, dim))
    s2 = _RANGE_RE.sub("", s)
    for m in _NUMBER_RE.finditer(s2):
        val = float(m.group(0).replace(",", ""))
        _, dim, mult = _match_unit(s2, m.end())
        out.append((val * mult, val * mult, dim))
    return out


def _topic_overlap(a: str, b: str) -> bool:
    """同主题门槛：数字归一为 '#' 后共享**非数字派生**的 2-gram。

    必要条件（"同一事实"）：共享 shingle 必须含非 '#' 字符——"价格是1500"
    vs "销量是800" 只共享 '是#'（数字派生）→ 不判；"成本1500元" vs
    "成本800元" 共享 '成本' → 判（同一事实的不同数值断言）。
    """
    ka = re.sub(r"\d+", "#", a)
    kb = re.sub(r"\d+", "#", b)
    shared = _shingles(ka) & _shingles(kb)
    return any("#" not in s for s in shared)


def _is_numeric_contradiction(a: str, b: str) -> bool:
    """数值矛盾（三选一③）：同一量、同量纲同单位、区间不相交。"""
    if not _topic_overlap(a, b):
        return False
    na = _extract_numbers(a)
    nb = _extract_numbers(b)
    if not na or not nb:
        return False
    for la, ha, da in na:
        for lb, hb, db in nb:
            if da != db:
                continue
            if ha < lb or hb < la:  # 区间不相交
                return True
    return False


def judge_contradiction(claim_a: str, claim_b: str, *,
                        meta_a: Optional[dict] = None,
                        meta_b: Optional[dict] = None) -> dict:
    """矛盾判定（§2.2 三选一 + 元数据排除）→ 判定详情。

    返回 ``{"kind": ConflictKind, "metadata_excluded": bool, "reason": str}``：
      - kind=NOT_A_CONFLICT 且 metadata_excluded=True → 假阳性演练红线命中
        （时间戳碰撞/同义复述/来源互补/显式排除）；
      - kind ∈ {DIRECTIONAL, NUMERIC, NEGATION} → 真矛盾，登记 conflict。
    """
    a = str(claim_a or "").strip()
    b = str(claim_b or "").strip()
    if not a or not b:
        return {"kind": ConflictKind.NOT_A_CONFLICT,
                "metadata_excluded": False, "reason": "empty_claim"}
    ma = meta_a or {}
    mb = meta_b or {}
    if ma.get("exclude") or mb.get("exclude"):
        return {"kind": ConflictKind.NOT_A_CONFLICT,
                "metadata_excluded": True, "reason": "explicit_exclude"}
    a_stripped = _strip_datetime_tokens(a)
    b_stripped = _strip_datetime_tokens(b)
    # 元数据排除①：时间戳/日期前缀碰撞——剔除时间戳后同文（仅时间戳不同）
    if a_stripped == b_stripped and (a_stripped != a or b_stripped != b):
        return {"kind": ConflictKind.NOT_A_CONFLICT,
                "metadata_excluded": True, "reason": "datetime_collision"}
    a_norm = normalize_claim(a)
    b_norm = normalize_claim(b)
    if not a_norm or not b_norm:
        return {"kind": ConflictKind.NOT_A_CONFLICT,
                "metadata_excluded": False, "reason": "empty_after_normalize"}
    # 元数据排除②：同义复述（不同措辞同一结论）→ 不判矛盾
    if a_norm == b_norm:
        return {"kind": ConflictKind.NOT_A_CONFLICT,
                "metadata_excluded": True, "reason": "synonym_restatement"}
    # 三选一②：否定矛盾（存在性/成立性直接否定——优先于方向对立）
    if _is_negation(a_norm, b_norm):
        return {"kind": ConflictKind.NEGATION,
                "metadata_excluded": False, "reason": "negation"}
    # 三选一①：方向矛盾（同一事实两条断言结论相反）
    if _is_directional(a_norm, b_norm):
        return {"kind": ConflictKind.DIRECTIONAL,
                "metadata_excluded": False, "reason": "directional"}
    # 三选一③：数值矛盾（同一量同量纲区间不相交）
    if _is_numeric_contradiction(a_stripped, b_stripped):
        return {"kind": ConflictKind.NUMERIC,
                "metadata_excluded": False, "reason": "numeric"}
    # 元数据排除③：来源不同且内容互补（未命中三选一）→ 不判矛盾
    if (ma.get("source") and mb.get("source")
            and ma.get("source") != mb.get("source")):
        return {"kind": ConflictKind.NOT_A_CONFLICT,
                "metadata_excluded": True, "reason": "source_complementary"}
    return {"kind": ConflictKind.NOT_A_CONFLICT,
            "metadata_excluded": False, "reason": "no_contradiction"}


def is_contradiction(claim_a: str, claim_b: str, *,
                     meta_a: Optional[dict] = None,
                     meta_b: Optional[dict] = None) -> ConflictKind:
    """矛盾判定三选一（§2.2）→ ConflictKind（便捷只读 API）。"""
    return judge_contradiction(claim_a, claim_b, meta_a=meta_a,
                               meta_b=meta_b)["kind"]


def verdict_for(conflict_kind: Optional[str]) -> str:
    """矛盾类型 → 验证裁决：三类真矛盾 → conflict；其余 → confirm。"""
    if conflict_kind in (
            ConflictKind.DIRECTIONAL.value,
            ConflictKind.NUMERIC.value,
            ConflictKind.NEGATION.value,
    ):
        return VerdictType.CONFLICT.value
    return VerdictType.CONFIRM.value


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
        验证通道 ≠ 登记通道；
      - 判定三选一（M3-2）：只有方向/数值/否定三类明确矛盾才判 conflict；
        元数据碰撞（时间戳前缀/同义复述/来源互补）一律不判（假阳性红线）；
      - 全链（M3-2）：草稿→独立验证→修正（``verify`` / ``run_pending``），
        VERIFY-* provenance 全程记录；CONFLICT 结果由写侧应用
        （``pending_conflicts`` → ``[doubt] conflict`` → labile）。
    """

    def __init__(self, enabled: Optional[bool] = None,
                 window: Optional[float] = None) -> None:
        self._enabled = verification_chain_enabled(enabled)
        self._window = _DEFAULT_WINDOW if window is None else float(window)
        self._requests: Dict[str, VerificationRequest] = {}
        self._results: Dict[str, VerificationResult] = {}
        self._events: Deque[dict] = deque(maxlen=200)
        # M3-2：VERIFY-* provenance 链（§4.3 观测/审计数据源；进程内存态）
        self._provenance: Deque[dict] = deque(maxlen=200)
        # M3-2：conflict 结果写侧应用记账（[doubt] conflict → labile 幂等）
        self._applied_conflicts: set = set()
        # M3-2：conflict 判定的目标断言文本（pending_conflicts 的 target）
        self._conflict_targets: Dict[str, str] = {}
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
        self._record_provenance("VERIFY-REGISTER", idempotency_key=key,
                                entry_ref=req.entry_ref,
                                phase=registered_phase,
                                register_query=req.register_query,
                                register_batch=req.batch_id,
                                register_channel=req.channel)
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

    # -- 验证链全链（M3-2：草稿→独立验证→修正） --------------------------- #

    def verify(
        self, request_key: str, *,
        reference_claims: Sequence[str] = (),
        reference_meta: Optional[Sequence[Optional[dict]]] = None,
        batch_id: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> Optional[VerificationResult]:
        """验证链全链：草稿→独立验证→修正（§2.2 防伪独立四维 + 幂等）。

        草稿:     请求登记时携带的 register_query（注册表已有）；
        独立验证: 验证 query（登记时生成、与登记**不同构造**）、验证批次
                  （≠登记批次——``independent_batch``）、验证通道
                  （≠登记通道——``channel_for`` 独立端点命名空间），
                  四维全部与登记隔离；
        修正:     以参考断言（独立证据）经 ``judge_contradiction`` 做
                  三选一判定：命中方向/数值/否定 → CONFLICT（写侧经
                  ``[doubt] conflict`` → labile）；全部 NOT_A_CONFLICT
                  → CONFIRM；无参考断言 → NONE（待验证）。

        幂等:     同键重复 verify 命中原结果，不重复判定/登记
                  （"客户端超时"≠"服务端未写入"——60s 竞态防护）；
        provenance: VERIFY-QUERY / VERIFY-JUDGE / VERIFY-RESULT 全程记录。
        """
        if not self._enabled:
            return None
        req = self._requests.get(request_key)
        if req is None:
            return None
        existing = self._results.get(request_key)
        if existing is not None:
            # 幂等：同键重试命中原结果，不重复判定/登记
            self._record_event(VerificationEventType.VERIFY_RESULT, {
                "idempotency_key": request_key, "replayed": True,
            })
            self._record_provenance("VERIFY-RESULT", idempotency_key=request_key,
                                    replayed=True)
            return existing
        # 防伪独立四维：验证批次/通道与登记隔离（query 登记时已独立构造）
        verify_batch = batch_id or self.independent_batch(req.batch_id)
        verify_channel = channel or self.channel_for(req.channel)
        self._record_provenance(
            "VERIFY-QUERY", idempotency_key=request_key,
            entry_ref=req.entry_ref,
            verify_query=req.verify_query,
            verify_batch=verify_batch,
            verify_channel=verify_channel,
            register_batch=req.batch_id,
            register_channel=req.channel,
            register_query=req.register_query,
        )
        refs = list(reference_claims or [])
        metas = (list(reference_meta) if reference_meta is not None
                 else [None] * len(refs))
        if not refs:
            # 无独立证据 → 无裁决（待验证；不臆断 confirm）
            res = self.submit_result(request_key, verdict=VerdictType.NONE.value)
            self._record_provenance(
                "VERIFY-RESULT", idempotency_key=request_key,
                verdict=VerdictType.NONE.value, note="no_reference_pending")
            return res
        verdict = VerdictType.CONFIRM.value
        kind: Optional[str] = None
        excluded = False
        target_text: Optional[str] = None
        for ref, meta in zip(refs, metas):
            j = judge_contradiction(req.register_query, ref, meta_b=meta or {})
            excluded = excluded or bool(j["metadata_excluded"])
            self._record_provenance(
                "VERIFY-JUDGE", idempotency_key=request_key,
                reference=str(ref)[:80],
                judged_kind=j["kind"].value,
                metadata_excluded=j["metadata_excluded"],
                reason=j["reason"],
            )
            if j["kind"] is not ConflictKind.NOT_A_CONFLICT:
                verdict = VerdictType.CONFLICT.value
                kind = j["kind"].value
                target_text = str(ref)
                break
        res = self.submit_result(
            request_key, verdict=verdict,
            conflict_kind=kind, metadata_excluded=excluded)
        if kind is not None and target_text is not None:
            self._conflict_targets[request_key] = target_text
        self._record_provenance(
            "VERIFY-RESULT", idempotency_key=request_key,
            verdict=verdict, conflict_kind=kind,
            metadata_excluded=excluded, reference_n=len(refs))
        return res

    def run_pending(
        self,
        evidence_fn: Optional[Callable[[VerificationRequest], Sequence[str]]] = None,
        *, batch_id: Optional[str] = None,
    ) -> List[VerificationResult]:
        """全链驱动：对每个待验证请求执行 草稿→独立验证→修正（幂等）。

        ``evidence_fn(request)`` → 独立证据断言列表（None/空 → 无裁决
        NONE——待验证不臆断）。全部结果登记幂等；重跑不产生新结果。
        """
        if not self._enabled:
            return []
        out: List[VerificationResult] = []
        for key in list(self._requests):
            if key in self._results:
                continue
            refs: Sequence[str] = []
            if evidence_fn is not None:
                try:
                    refs = list(evidence_fn(self._requests[key]) or [])
                except Exception:  # pylint: disable=broad-except
                    refs = []
            res = self.verify(key, reference_claims=refs, batch_id=batch_id)
            if res is not None:
                out.append(res)
        return out

    # -- conflict 结果写侧应用（[doubt] conflict → labile） ----------------- #

    def pending_conflicts(self, last_n: int = 20) -> List[dict]:
        """待写侧应用的 conflict 结果（结果写 ``[doubt] conflict`` → labile
        的数据源——loop 写侧取用，应用后 ``mark_conflict_applied`` 记账）。

        负载：request_key / claim_text（登记断言，新主张）/ target_text
        （被否定的旧记忆文本——``[doubt] conflict`` 的目标）/
        conflict_kind / resolved_at。
        """
        if not self._enabled:
            return []
        out: List[dict] = []
        for key, res in list(self._results.items()):
            if key in self._applied_conflicts:
                continue
            if res.verdict != VerdictType.CONFLICT.value:
                continue
            req = self._requests.get(key)
            out.append({
                "request_key": key,
                "claim_text": req.register_query if req is not None else "",
                "target_text": self._conflict_targets.get(key, ""),
                "conflict_kind": res.conflict_kind,
                "resolved_at": res.resolved_at,
            })
        return out[-last_n:]

    def mark_conflict_applied(self, request_key: str) -> bool:
        """写侧已应用 conflict（[doubt] conflict → labile）后记账（幂等）。"""
        if not self._enabled:
            return False
        if request_key not in self._results:
            return False
        if request_key in self._applied_conflicts:
            return True
        self._applied_conflicts.add(request_key)
        self._record_provenance("VERIFY-CONFLICT-APPLIED",
                                idempotency_key=request_key)
        return True

    # -- provenance（VERIFY-* 日志；§4.3 事件流同源数据源） ----------------- #

    def provenance(self, last_n: int = 20) -> List[dict]:
        """VERIFY-* provenance 链（观测/审计）：VERIFY-REGISTER /
        VERIFY-QUERY / VERIFY-JUDGE / VERIFY-RESULT / VERIFY-CONFLICT-APPLIED。"""
        return list(self._provenance)[-last_n:]

    def _record_provenance(self, kind: str, **fields: Any) -> None:
        entry = {"kind": kind, "ts": time.time(), **fields}
        self._provenance.append(entry)
        # VERIFY-* 结构化日志（§2.2 provenance；reference 内容截断防噪）
        log_fields = {k: v for k, v in fields.items() if k != "reference"}
        logger.info("VERIFY-%s %s", kind, log_fields)

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

        M3-2 语义化完成：登记 query 原样保留 + 独立角度后缀——保证
        ``verify_query != register_query``（协议硬约束）；真正的独立换问
        由调用方（外部验证器/规则证据源）在参考断言层面实现（§2.2
        "同一事实不同问法"的工程落地是 judge_contradiction 的语义判定，
        验证 query 的构造独立性由本方法保证）。
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
            "provenance_n": len(self._provenance),
            "conflicts_pending": len(self.pending_conflicts()),
        }

    def clear(self) -> None:
        """清空注册表（测试用；生产不调用——窗口自动过期）。"""
        self._requests.clear()
        self._results.clear()
        self._events.clear()
        self._provenance.clear()
        self._applied_conflicts.clear()
        self._conflict_targets.clear()

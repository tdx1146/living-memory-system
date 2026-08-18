# -*- coding: utf-8 -*-
"""core/store · 写语义定义（M1 第一段：feed/store 语义分离）

规格依据：`四妹-LMS核心重写规格v2-20260817.md` §1.5（store 提取层）。
本模块是**写语义的单一权威定义点**（§4.5：默认值同源，禁止 schema/代码/文档
三处各写各的）：

  - ``feed``：知识补存双通道（可召回层，source 语义 = 可召回）。补存类知识
    一律走 feed，**禁止**走 store_gray 导致探针必 miss（坑 6 教训）。
  - ``store``：普通存储。可带灰度标记（``gray=True`` → ``source='store_gray'``），
    ``store_gray`` 不进入 L1 可召回层——这是**口径不是 bug**（灰度口径，登记于此）。

本模块只做语义判定与口径登记，**不 import 任何 LMS 运行时模块**
（依赖注入，纯 stdlib）——保证可被 python 直接 import/运行，也与血管（api 层）
解耦。写侧的真正落库由调用方注入的 writer 完成（见 ingest.py）。
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional


class WriteSemantics(str, Enum):
    """写语义（写侧统一入口的两种语义分离）。"""

    FEED = "feed"    # 知识补存双通道：可召回层（source 语义 = 可召回）
    STORE = "store"  # 普通存储（含 store_gray 灰度标记，不可被 L1 召回）


class FeedChannel(str, Enum):
    """feed 双通道（知识补存：可召回层为主，扩展层为辅）。

    双通道语义（规格 §1.5 / §4.3）：
      - ``primary``：主通道 = episodic 可召回层（L1 检索直接可达）；
      - ``archive`` ：扩展通道 = 窗口外归档补充层（tier1，recall_merged_readonly
        合并检索补充；与 loop._export_episodic_to_archive 同一口径）。
    通道是**语义标记**：具体落哪一层由调用方 writer 实现，本层只保证
    语义合法（feed 永不灰化）并透传给 writer。
    """

    PRIMARY = "primary"
    ARCHIVE = "archive"


# ---------------------------------------------------------------------------
# 灰度口径登记（坑 6：灰度冻结——口径不是 bug，迁移与测试必须保留）
# ---------------------------------------------------------------------------
#: 不可被 L1 召回的 source 集合。``store_gray`` 是唯一成员；命中即不可召回。
NON_RECALLABLE_SOURCES: frozenset[str] = frozenset({"store_gray"})

#: feed 双通道默认 source（可召回；与现有 /feed 落库 source='external' 同源）。
DEFAULT_FEED_SOURCE = "external"
#: store 普通存储默认 source（可召回；gray=True 时被 store_gray 覆盖）。
DEFAULT_STORE_SOURCE = "external"

#: 灰度默认开关 env（沿用现有 env 名，§4.5 命名同源；请求显式传 gray 时优先）。
_ENV_STORE_GRAY = "LMS_STORE_GRAY"


class SemanticViolation(ValueError):
    """写语义违规（api 层映射 422；写侧默认保守：宁可拦错不轻信——坑 3）。"""


def store_gray_default() -> bool:
    """store 灰度默认值（请求未显式指定 gray 时生效）。

    运行时读 env（灰度"随时可关"语义，与现有 api/server.py 的
    ``_store_gray_enabled()`` 同款：改动即时生效，无需重启）。
    """
    return os.environ.get(_ENV_STORE_GRAY, "0") == "1"


def recallable(source: Optional[str]) -> bool:
    """L1 可召回判据（口径：store_gray 不可召回，其余可召回）。"""
    return source not in NON_RECALLABLE_SOURCES


def resolve_source(semantics: WriteSemantics,
                   source: Optional[str],
                   gray: bool) -> str:
    """按写语义推导条目最终 source（写侧唯一权威，调用方 writer 必须采用）。

    语义约束（§1.5）：
      - ``feed``：永不灰化（gray 传 True 直接拒绝）；source 必须可召回
        （不得为 store_gray 等不可召回 source）——补存知识禁止走灰度口径；
      - ``store``：gray=True → ``store_gray``（不可召回，口径保留）；
        gray=False → 调用方 source 或默认 ``external``。

    Raises:
        SemanticViolation: 语义冲突（feed+gray / feed+不可召回 source）。
    """
    if semantics is WriteSemantics.FEED:
        if gray:
            raise SemanticViolation(
                "feed 语义禁止灰化（知识补存必须可召回，坑 6 口径）；"
                "补存类知识请走 feed，禁止走 store_gray")
        s = source or DEFAULT_FEED_SOURCE
        if s in NON_RECALLABLE_SOURCES:
            raise SemanticViolation(
                f"feed 语义禁止使用不可召回 source='{s}'"
                f"（{sorted(NON_RECALLABLE_SOURCES)} 为 store 灰度专属口径）")
        return s
    # STORE
    if gray:
        return "store_gray"
    return source or DEFAULT_STORE_SOURCE


def validate_semantics(semantics: WriteSemantics,
                       source: Optional[str],
                       gray: bool) -> None:
    """写前语义校验（ingest 入口第一步；违规抛 SemanticViolation）。

    只做语义判定，不落库、无副作用（claim：无副作用——供 §5.2 测试断言）。
    """
    if semantics not in (WriteSemantics.FEED, WriteSemantics.STORE):
        raise SemanticViolation(f"未知写语义: {semantics!r}")
    if source is not None and not isinstance(source, str):
        raise SemanticViolation(f"source 必须是字符串: {source!r}")
    if not isinstance(gray, bool):
        raise SemanticViolation(f"gray 必须是布尔: {gray!r}")
    # 触发 source 推导即完成全部冲突校验
    resolve_source(semantics, source, gray)

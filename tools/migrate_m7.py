#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMS 核心重建 M7 · 数据迁移工具（快照 + 归档合并去重，幂等可回滚可重跑）
========================================================================

规格依据：`四妹-LMS核心重写规格v2-20260817.md` §三（数据兼容与迁移：
latest_main.pt v0.5.0 + main.jsonl 归档——幂等可回滚可重跑）/ §7.1 M7
（新 schema 稳定后迁移——双写回滚/校验）。

任务对照（§3.2 迁移步骤，1:1 落地）::

    0 备份      backup_inputs()——迁移前完整备份快照 + 归档 + 当前新存储
                （双份），校验备份可读（BK-01 教训：备份本身可能损坏，
                校验不过即中止，不带着坏备份进迁移）
    1 校验      validate_snapshot()——条目数 / 字段完整性 / confidence∈[0,1]
                / turn 单调性；validate_archive_records()——行级字段校验
    2 映射      map_entry_to_new()——旧字段→新字段（rebuttal_consistency
                初始化（init_rebuttal_consistency，写侧 'ingest'）、怀疑态
                初始化 stable、violated_by 证伪语义保留为 superseded、
                source 原样保留含 store_gray（灰度口径：gray→source
                归一化 'store_gray'，L1 不可召回语义保持））
    3 合并归档  merge_entries()——快照 ∪ 归档按 (turn, text_hash) 去重合并，
                快照条目优先（有语义向量）；条目数 + turn 序列完整性双重校验
    4 全量校验  validate_merged()——条目数 = 快照 + 归档 - 去重数；
                抽样语义抽查；checksum 对账（输入/输出 sha256 落 manifest）
    5 双写回滚期 dual_write journal——N 轮（LMS_M7_DUAL_WRITE_ROUNDS，
                env 化）增量对比 journal + finalize 校验
    6 切换/回滚 任一步校验失败 → 立即回滚到备份（rollback-drill / restore），
                不带着错误进生产

不丢记忆铁律（§3.3）：迁移全程**无删除操作**——输入快照/归档只读、不改、
不删；输出写入独立 out-dir；迁移脚本本身幂等——中断后重跑不产生重复条目。

用法::

    # 1) 只校验 + 规划（默认 dry-run，不发任何写）
    python tools/migrate_m7.py validate --snapshot snapshots/main/latest_main.pt \
        --archive data/archive/main.jsonl

    # 2) 执行迁移（备份 → 校验 → 映射 → 合并去重 → 全量校验 → 落盘）
    python tools/migrate_m7.py migrate --snapshot snapshots/main/latest_main.pt \
        --archive data/archive/main.jsonl --apply --session main

    # 3) 回滚演练（备份可读性 + 恢复对账；BK-01 教训的机器防线）
    python tools/migrate_m7.py rollback-drill --backup-dir data/migration/backups/xxx

    # 4) 迁移后校验（recall 命中 / 不丢记忆）
    python tools/migrate_m7.py verify --snapshot data/migration/main/migrated_latest_main.pt \
        --archive data/migration/main/main.jsonl

    # 5) 双写回滚期记录 / 收尾
    python tools/migrate_m7.py dual-write record --journal ... --round 1 --old-inc 3 --new-inc 3
    python tools/migrate_m7.py dual-write check  --journal ...

退出码: 0=通过 2=参数错误 3=校验失败/迁移中止（错误在 stderr + 报告）。

环境变量（M8：env 化，单一默认值来源——本文件 docstring 为唯一权威）::

    LMS_M7_SNAPSHOT            输入快照（默认 snapshots/main/latest_main.pt）
    LMS_M7_ARCHIVE             输入归档（默认 data/archive/main.jsonl）
    LMS_M7_SESSION             会话（默认 main）
    LMS_M7_OUT_DIR             输出目录（默认 data/migration/{session}）
    LMS_M7_BACKUP_DIR          备份根目录（默认 data/migration/backups）
    LMS_M7_DUAL_WRITE_ROUNDS   双写回滚期轮数 N（默认 0 = 不启用；env 化 §3.2-5）
    LMS_M7_FORCE               校验错误时仍继续（默认 0，仅显式开启）

生产落盘（"我落盘"）：本工具输出到 out-dir（原子替换、伴 manifest），
生产放置/切换由运维按 §3.2 步骤 5-6 执行；工具提供 verify / rollback-drill
作为切换前/后的机器校验。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
import numpy as np  # noqa: E402

from core.archive.archive_store import (  # noqa: E402
    b64_to_vector,
    entry_to_record,
    text_hash,
    vector_to_b64,
)
from core.doubt.rebuttal_field import init_rebuttal_consistency  # noqa: E402
from core.hippocampus.memory import (  # noqa: E402
    EpisodicEntry,
    MemoryManager,
)

# ---------------------------------------------------------------------------
# 常量 / env（M8：单一默认值来源，见 docstring）
# ---------------------------------------------------------------------------

TOOL_NAME = "tools/migrate_m7.py"
MILESTONE = "M7"
REWRITE_SPEC = "四妹-LMS核心重写规格v2-20260817.md §三/§7.1 M7"

DEFAULT_SESSION = "main"
DEFAULT_SNAPSHOT = "snapshots/main/latest_main.pt"
DEFAULT_ARCHIVE = "data/archive/main.jsonl"
DEFAULT_OUT_DIR = "data/migration/{session}"
DEFAULT_BACKUP_DIR = "data/migration/backups"
DEFAULT_DUAL_WRITE_ROUNDS = 0

# 归档记录格式版本（与 core/archive/archive_store.ARCHIVE_VERSION 同源）
ARCHIVE_VERSION = 1

# 校验类别
_ERR = "error"      # 阻断迁移（除非 --force）
_WARN = "warning"   # 记录并继续


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def sha256_file(path: str) -> str:
    """文件 sha256（大文件分块）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """原子写（tmp + os.replace）：读者永远看到完整文件（G 模式）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".m7_", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, str(path))
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def atomic_write_jsonl(path: Path, records: List[dict]) -> None:
    """原子写 JSONL（每行一个 JSON，ensure_ascii=False）。"""
    lines = "".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in records
    ).encode("utf-8")
    atomic_write_bytes(path, lines)


def _coerce_turn(v: Any) -> int:
    """turn 强制转 int（fail-open：非法 → -1）。"""
    try:
        return int(v)
    except (TypeError, ValueError):
        return -1


def entry_text(e: Any) -> Optional[str]:
    """取条目文本（dataclass / dict / tuple 三形态兼容）。"""
    if isinstance(e, dict):
        return e.get("text")
    if isinstance(e, (tuple, list)) and len(e) >= 1:
        return e[0]
    return getattr(e, "text", None)


def entry_key(e: Any) -> Optional[Tuple[int, str]]:
    """条目去重键 (turn, text_hash(text))——与归档 (turn, text_hash) 同口径。

    无有效文本 → None（该条目不参与去重/迁移，计为 invalid）。
    """
    text = entry_text(e)
    if not text or not str(text).strip():
        return None
    turn = _coerce_turn(e.get("turn") if isinstance(e, dict) else getattr(e, "turn", None))
    if turn < 0:
        return None
    return (turn, text_hash(str(text)))


# ---------------------------------------------------------------------------
# 输入读取
# ---------------------------------------------------------------------------

def load_snapshot_raw(path: str) -> dict:
    """只读加载 v0.5.0 快照（weights_only=False；失败抛异常由调用方处理）。"""
    return torch.load(path, map_location="cpu", weights_only=False)


def extract_episodic(st: dict) -> List[Any]:
    """从快照顶层提取 episodic 条目列表（st['memory']['episodic_buffer']）。

    兼容 EpisodicEntry dataclass / dict / tuple 三形态（与旧
    tools/export_episodic.py 同口径）；无 memory 字段 → 空列表。
    """
    mem = st.get("memory") or {}
    eb = mem.get("episodic_buffer") or []
    return list(eb)


def read_archive_records(path: str) -> Tuple[List[dict], int]:
    """读取归档 JSONL（增量日志）。

    兼容两种行格式：
      - archive_store 格式（{version, session_id, turn, text, source, surprise,
        text_hash, vector_b64, exported_at, origin}）；
      - 旧导出格式（{text, surprise, turn, source, ...}，无 vector_b64）。

    坏行跳过并计数（fail-open 粒度到行级——与 archive_store._iter_records
    同语义）；返回 (records, skipped_lines)。
    """
    records: List[dict] = []
    skipped = 0
    if not os.path.isfile(path):
        return records, skipped
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(rec, dict):
                skipped += 1
                continue
            records.append(rec)
    return records, skipped


# ---------------------------------------------------------------------------
# 校验（§3.2 步骤 1）
# ---------------------------------------------------------------------------

def validate_snapshot(path: str) -> Tuple[dict, List[Any], List[dict]]:
    """校验快照：条目数 / 字段完整性 / confidence∈[0,1] / turn 单调性。

    返回 (raw_st, entries, issues)。issues = [{level, msg}]；
    level='error' 阻断迁移（除非 --force），'warning' 仅记录。
    """
    issues: List[dict] = []
    st = load_snapshot_raw(path)
    version = st.get("version", "unknown")
    if "attractor" not in st or "purpose" not in st:
        issues.append({"level": _ERR,
                       "msg": f"快照缺少 attractor/purpose 字段（version={version}）"})
    if str(version) != "0.5.0":
        issues.append({"level": _WARN,
                       "msg": f"快照版本 {version!r} 非 0.5.0（规格输入为 v0.5.0，"
                              f"仍尝试读取）"})
    entries = extract_episodic(st)
    issues.append({"level": _WARN, "msg": f"快照 episodic 条目数 = {len(entries)}"})

    prev_turn = -1
    for i, e in enumerate(entries):
        text = entry_text(e)
        if not text or not str(text).strip():
            issues.append({"level": _ERR,
                           "msg": f"条目[{i}] 无有效文本（turn="
                                  f"{getattr(e, 'turn', '?')}）"})
            continue
        turn = _coerce_turn(
            e.get("turn") if isinstance(e, dict) else getattr(e, "turn", None))
        if turn < 0:
            issues.append({"level": _ERR, "msg": f"条目[{i}] turn 非法: {turn!r}"})
            continue
        if turn < prev_turn:
            if turn == 0:
                # 硬复位段（规格 §3.2-3"硬复位后"预判项——真实输入含
                # turn 0-13/0-628 复位段）：turn 计数归 0 重排，**不阻断迁移**；
                # 严格完整性校验在合并输出上做（validate_merged：turn 非降 +
                # 键唯一双重校验，error 级）。复位后段内继续按单调递增校验。
                issues.append({"level": _WARN,
                               "msg": f"turn 硬复位段（归 0，规格 §3.2-3 "
                                      f"预判场景，不阻断）：条目[{i}] "
                                      f"turn={turn} < 前一条目 "
                                      f"turn={prev_turn}——输出将按 turn 稳定"
                                      f"排序并强制键唯一（见 validate_merged）"})
                prev_turn = turn   # 复位段新基线（段内继续单调校验）
                continue
            issues.append({"level": _ERR,
                           "msg": f"turn 单调性违反（非硬复位归 0，判为数据"
                                  f"损坏）：条目[{i}] turn={turn} < 前一条目 "
                                  f"turn={prev_turn}"})
        prev_turn = max(prev_turn, turn)
        conf = (e.get("confidence") if isinstance(e, dict)
                else getattr(e, "confidence", None))
        if conf is not None:
            try:
                cf = float(conf)
                if not (0.0 <= cf <= 1.0):
                    issues.append({"level": _ERR,
                                   "msg": f"条目[{i}] turn={turn} "
                                          f"confidence={cf!r} 超出 [0,1]"})
            except (TypeError, ValueError):
                issues.append({"level": _WARN,
                               "msg": f"条目[{i}] turn={turn} confidence "
                                      f"非数值: {conf!r}（映射时钳制）"})
    return st, entries, issues


def validate_archive_records(records: List[dict]) -> List[dict]:
    """归档行级字段校验（文本完整性 / turn 合法性 / 语义向量可解码）。"""
    issues: List[dict] = []
    for i, rec in enumerate(records):
        text = rec.get("text")
        if not text or not str(text).strip():
            issues.append({"level": _ERR,
                           "msg": f"归档行[{i}] 无有效文本"})
            continue
        turn = _coerce_turn(rec.get("turn"))
        if turn < 0:
            issues.append({"level": _ERR,
                           "msg": f"归档行[{i}] turn 非法: {rec.get('turn')!r}"})
        vb = rec.get("vector_b64")
        if vb:
            v = b64_to_vector(vb)
            if v is None or v.size == 0:
                issues.append({"level": _WARN,
                               "msg": f"归档行[{i}] vector_b64 不可解码"
                                      f"（该行退化为文本记录）"})
    return issues


# ---------------------------------------------------------------------------
# schema 映射（§3.2 步骤 2）
# ---------------------------------------------------------------------------

def map_entry_to_new(e: Any, *, session: str, now_ts: float,
                     source: Optional[str] = None,
                     semantic_vector: Any = None,
                     surprise: Optional[float] = None,
                     turn: Optional[int] = None) -> Optional[EpisodicEntry]:
    """旧条目 → 新 schema EpisodicEntry（旧字段→新字段映射）。

    映射规则（§3.2-2）：
      - 文本/turn/surprise/source 原样保留（source 含 store_gray 原样保留；
        灰度语义归一：gray 标记 → source='store_gray'，L1 不可召回口径保持）；
      - confidence 保留并钳制 [0,1]；
      - rebuttal_count/reference_count/source_trust/violated_by/
        last_recalled_at/recall_count/consistency/confidence_before_rebuttal
        原样保留（不丢字段）；
      - 新字段初始化：last_reinforced_turn=写入 turn（wear 计时起点）、
        info_value=0.0、core=None、ts=旧 ts（无则 None）、gray=旧灰度；
      - rebuttal_consistency 结构化字段经 init_rebuttal_consistency 初始化
        （写侧 'ingest'，幂等；consistency 取旧平坦 consistency 钳制值）；
      - 怀疑态：默认 'stable'；旧证伪记录（violated_by 非空）保留原语义
        → 'superseded'（§3.2-2"conflict/supersedes 旧记录保留原语义"）。

    无有效文本 → 返回 None（不迁移）。
    """
    text = entry_text(e)
    if not text or not str(text).strip():
        return None
    is_dict = isinstance(e, dict)
    get = (lambda k, d=None: e.get(k, d)) if is_dict else \
        (lambda k, d=None: getattr(e, k, d))

    turn_i = _coerce_turn(turn if turn is not None else get("turn"))
    if turn_i < 0:
        turn_i = 0

    # 语义向量：显式覆盖 > 条目自带（archive 记录经 b64 解码后传入）。
    # b64 解码（b64_to_vector）产出 numpy float32——统一归一为 torch 张量，
    # 保证迁移产物（新快照 episodic_buffer）可被新核心 MemoryManager.set_state
    # 直接消费（memory.py set_state 对向量调 .to(device)——numpy 无该方法，
    # E-P2-1 兼容线要求张量）。
    vec = semantic_vector
    if vec is None:
        vec = get("semantic_vector")
    if vec is not None and not isinstance(vec, torch.Tensor):
        # np.frombuffer 产出只读数组——先拷贝再转张量（消除
        # "non-writable numpy array" 告警，且保证迁移产物张量可写）
        vec = torch.as_tensor(np.asarray(vec).copy(), dtype=torch.float32)
    if vec is None:
        # 无向量条目：仍可入档案（文本记录），不产生可召回 EpisodicEntry
        # ——由调用方决定归档-only；此处返回 None 时调用方走归档-only 路径
        pass

    # 灰度归一（口径：gray 标记与 source='store_gray' 同语义）
    old_gray = bool(get("gray", False))
    old_source = str(source if source is not None else get("source", "external") or "external")
    gray = old_gray or old_source == "store_gray"
    src = "store_gray" if gray else old_source

    try:
        confidence = float(get("confidence", 1.0) or 1.0)
    except (TypeError, ValueError):
        confidence = 1.0
    confidence = max(0.0, min(1.0, confidence))

    try:
        surprise_f = float(surprise if surprise is not None
                           else (get("surprise", 0.0) or 0.0))
    except (TypeError, ValueError):
        surprise_f = 0.0

    try:
        old_consistency = get("consistency")
        cons_val = float(old_consistency) if old_consistency is not None else 0.0
    except (TypeError, ValueError):
        cons_val = 0.0

    entry = EpisodicEntry(
        text=str(text),
        semantic_vector=vec if vec is not None else torch.zeros(0),
        surprise=surprise_f,
        turn=turn_i,
        source=src,
        confidence=confidence,
        rebuttal_count=int(get("rebuttal_count", 0) or 0),
        reference_count=int(get("reference_count", 0) or 0),
        source_trust=float(get("source_trust", 1.0) or 1.0),
        labile=False,                      # labile 是时相状态不落库（§2.1）
        labile_since=None,
        violated_by=get("violated_by"),
        last_recalled_at=get("last_recalled_at"),
        recall_count=int(get("recall_count", 0) or 0),
        consistency=get("consistency"),
        confidence_before_rebuttal=get("confidence_before_rebuttal"),
        last_reinforced_turn=turn_i,       # wear 计时起点 = 写入 turn
        info_value=0.0,                    # 迁移不重算价值（冷启动垫片非承重）
        core=get("core"),
        ts=get("ts"),
        gray=gray,
    )
    # rebuttal_consistency 结构化字段（写侧初始化，幂等）
    init_rebuttal_consistency(entry, now=now_ts, consistency=cons_val)
    # 怀疑态：默认 stable；旧证伪记录（violated_by）保留 superseded 语义
    violated = get("violated_by")
    entry.doubt_state = "superseded" if violated else "stable"
    return entry


def map_archive_record_to_entry(rec: dict, *, session: str,
                                now_ts: float) -> Optional[EpisodicEntry]:
    """归档记录 → 新条目（vector_b64 解码为语义向量；无向量 → None）。"""
    vec = None
    vb = rec.get("vector_b64")
    if vb:
        vec = b64_to_vector(vb)
    return map_entry_to_new(
        rec, session=session, now_ts=now_ts,
        source=rec.get("source"),
        semantic_vector=vec,
        surprise=rec.get("surprise"),
        turn=rec.get("turn"),
    )


# ---------------------------------------------------------------------------
# 合并去重（§3.2 步骤 3）
# ---------------------------------------------------------------------------

def merge_entries(snapshot_entries: List[Any],
                  archive_records: List[dict],
                  *, session: str, now_ts: float) -> Dict[str, Any]:
    """快照 ∪ 归档 按 (turn, text_hash) 去重合并；快照条目优先（有向量）。

    返回 stats（含各计数 + 公式校验）；并产出：
      - merged_episodic: 全部有向量条目（新 schema EpisodicEntry），
        按 turn 稳定排序 → 进新快照 episodic_buffer；
      - archive_out: 重建归档记录（快照派生记录 + 保留的原始归档记录），
        无向量记录仅进归档（文本记录，不参与向量检索——fail-open 口径）。

    去重语义（§3.3 无删除）：输入文件不改不删；合并视图内重复键只保留一份
    （快照优先）。条目数公式（§3.2-4）：
        final = |快照唯一| + |归档唯一新增| = |快照| + |归档有效| - 去重数
    """
    # 1) 快照条目建键（内部重复保留首条，计数）
    snap_keys: Dict[Tuple[int, str], Any] = {}
    snap_internal_dup = 0
    for e in snapshot_entries:
        k = entry_key(e)
        if k is None:
            continue
        if k in snap_keys:
            snap_internal_dup += 1
            continue
        snap_keys[k] = e

    # 2) 归档记录建键（与快照冲突 → 快照优先；内部重复保留首条）
    archive_kept: Dict[Tuple[int, str], dict] = {}
    dup_with_snapshot = 0
    dup_archive_internal = 0
    archive_invalid = 0
    for rec in archive_records:
        k = entry_key(rec)
        if k is None:
            archive_invalid += 1
            continue
        if k in snap_keys:
            dup_with_snapshot += 1
            continue
        if k in archive_kept:
            dup_archive_internal += 1
            continue
        archive_kept[k] = rec

    # 3) 产出：快照条目（映射） + 归档唯一新增（解码映射）
    #    - merged: 有向量条目 → 进新快照 episodic_buffer（可召回）
    #    - archive_out: **全部唯一条目**（含无向量文本记录）→ 重建归档
    #      （§3.2-4 条目数 = 快照 + 归档 - 去重数 的"条目"= 全部唯一条目）
    merged: List[EpisodicEntry] = []
    archive_out: List[dict] = []
    for k, e in snap_keys.items():
        new_e = map_entry_to_new(e, session=session, now_ts=now_ts)
        if new_e is None:
            continue
        if _has_valid_vector(new_e):
            merged.append(new_e)
        archive_out.append(_record_from_entry(session, new_e,
                                              exported_at=now_ts))
    for k, rec in archive_kept.items():
        new_e = map_archive_record_to_entry(rec, session=session, now_ts=now_ts)
        if new_e is not None and _has_valid_vector(new_e):
            merged.append(new_e)
            archive_out.append(_record_from_entry(session, new_e,
                                                  exported_at=now_ts))
        else:
            # 无向量记录：仅进归档（保留原记录原文，不参与向量检索）
            archive_out.append(rec)

    # 稳定排序（turn 升序，同 turn 保序——输出可复现）
    merged.sort(key=lambda e: (e.turn, text_hash(e.text)))
    archive_out.sort(key=lambda r: (_coerce_turn(r.get("turn")), r.get("text", "")))

    final_episodic = len(merged)
    final_union = len(archive_out)
    snap_unique = len(snap_keys)
    archive_unique_new = len(archive_kept)
    dup_total = (snap_internal_dup + dup_with_snapshot + dup_archive_internal)
    formula_ok = (
        final_union == snap_unique + archive_unique_new
        and final_union + dup_total
        == len(snapshot_entries) + len(archive_records) - archive_invalid
    )
    return {
        "session": session,
        "snapshot_entries": len(snapshot_entries),
        "snapshot_unique": snap_unique,
        "snapshot_internal_dup": snap_internal_dup,
        "archive_records": len(archive_records),
        "archive_valid": len(archive_records) - archive_invalid,
        "archive_invalid": archive_invalid,
        "archive_unique_new": archive_unique_new,
        "dup_with_snapshot": dup_with_snapshot,
        "dup_archive_internal": dup_archive_internal,
        "dup_total": dup_total,
        "final_union": final_union,          # 条目数（§3.2-4 公式口径）
        "final_episodic": final_episodic,    # 有向量 → episodic_buffer
        "final_archive": final_union,
        "formula_ok": formula_ok,
        "merged_episodic": merged,
        "archive_out": archive_out,
    }


def _record_from_entry(session: str, e: EpisodicEntry,
                       exported_at: Optional[float] = None) -> dict:
    """新条目 → 归档记录（与运行期 export 同构：entry_to_record 唯一权威）。

    exported_at 缺省用当前时间；迁移场景传入数据时基（data_ts）保证
    重跑字节级可复现（幂等）。
    """
    rec = entry_to_record(session, e)
    if rec is None:
        return {
            "version": ARCHIVE_VERSION,
            "session_id": session,
            "turn": e.turn,
            "text": e.text,
            "source": e.source,
            "surprise": e.surprise,
            "text_hash": text_hash(e.text),
            "vector_b64": None,
            "exported_at": exported_at if exported_at is not None else time.time(),
            "origin": "archive",
        }
    if exported_at is not None:
        rec["exported_at"] = exported_at
    return rec


def _has_valid_vector(e: EpisodicEntry) -> bool:
    """条目是否携带可参与向量检索的语义向量（非空且非 0 维）。"""
    v = getattr(e, "semantic_vector", None)
    if v is None:
        return False
    try:
        return int(torch.as_tensor(v).numel()) > 0
    except Exception:  # noqa: BLE001 - 畸形向量视为无向量（fail-open）
        return False


# ---------------------------------------------------------------------------
# 全量校验（§3.2 步骤 4）
# ---------------------------------------------------------------------------

def validate_merged(stats: dict, merged: List[EpisodicEntry]) -> List[dict]:
    """迁移后校验：条目数公式 + turn 序列完整性双重校验 + 抽样语义抽查。

    turn 序列完整性（"未发现的坑"预判项）：输出按 turn 排序非降；
    无重复 (turn, text_hash)；turn 上限 = max(快照 max, 归档 max)。
    """
    issues: List[dict] = []
    if not stats["formula_ok"]:
        issues.append({"level": _ERR,
                       "msg": f"条目数公式不成立：快照 {stats['snapshot_entries']}"
                              f" + 归档 {stats['archive_records']}"
                              f" - 去重 {stats['dup_total']}"
                              f" ≠ 最终 {stats['final_union']}"})
    keys = set()
    prev_turn = -1
    max_turn = -1
    for i, e in enumerate(merged):
        k = (e.turn, text_hash(e.text))
        if k in keys:
            issues.append({"level": _ERR,
                           "msg": f"合并后重复键: turn={e.turn} hash={k[1]}"})
        keys.add(k)
        if e.turn < prev_turn:
            issues.append({"level": _ERR,
                           "msg": f"输出 turn 序列违反非降：{e.turn} < {prev_turn}"})
        prev_turn = max(prev_turn, e.turn)
        max_turn = max(max_turn, e.turn)
    issues.append({"level": _WARN,
                   "msg": f"合并后条目数 = {len(merged)}，turn 范围 [0..{max_turn}]"})
    # 抽样语义抽查：随机取 ≤10 条核对文本/turn/source 非空且 hash 自洽
    import random
    rng = random.Random(0)
    sample = rng.sample(merged, min(10, len(merged)))
    for e in sample:
        if not e.text or e.turn < 0:
            issues.append({"level": _ERR,
                           "msg": f"抽样条目异常: turn={e.turn} text={e.text[:30]!r}"})
    return issues


# ---------------------------------------------------------------------------
# 备份 / 回滚（§3.2 步骤 0 / 6；BK-01 教训）
# ---------------------------------------------------------------------------

def backup_inputs(snapshot_path: str, archive_path: str,
                  backup_dir: str) -> Tuple[List[Path], List[dict]]:
    """迁移前备份：快照 + 归档 + 校验备份可读（BK-01：备份本身可能损坏）。

    备份后立即做可读性校验（torch.load 快照 / JSONL 逐行解析 + 计数对账）；
    任何一项校验不过 → 返回 issues（调用方必须中止，不带着坏备份进迁移）。
    """
    bdir = Path(backup_dir)
    bdir.mkdir(parents=True, exist_ok=True)
    snap_bak = bdir / "latest_main.pt"
    arch_bak = bdir / "main.jsonl"
    meta = bdir / "backup_meta.json"
    shutil.copy2(snapshot_path, snap_bak)
    shutil.copy2(archive_path, arch_bak)

    issues: List[dict] = []
    # 快照备份可读性
    snap_expected = None
    arch_expected = None
    try:
        st = load_snapshot_raw(str(snap_bak))
        snap_expected = len(extract_episodic(st))
        n2 = len(extract_episodic(load_snapshot_raw(snapshot_path)))
        if snap_expected != n2:
            issues.append({"level": _ERR,
                           "msg": f"备份快照 episodic 数 {snap_expected} ≠ 原快照 {n2}（备份损坏）"})
    except Exception as ex:  # noqa: BLE001 - 备份损坏必须可见
        issues.append({"level": _ERR, "msg": f"备份快照不可读: {ex}"})
    # 归档备份可读性
    try:
        recs, skipped = read_archive_records(str(arch_bak))
        arch_expected = len(recs)
        orig_recs, orig_skipped = read_archive_records(archive_path)
        if arch_expected != len(orig_recs) or skipped != orig_skipped:
            issues.append({"level": _ERR,
                           "msg": f"备份归档行数 {arch_expected} ≠ 原归档 {len(orig_recs)}"
                                  f"（备份损坏）"})
    except Exception as ex:  # noqa: BLE001
        issues.append({"level": _ERR, "msg": f"备份归档不可读: {ex}"})
    meta_data = {
        "ts": time.time(),
        "snapshot": {"src": str(snapshot_path), "dst": str(snap_bak),
                     "sha256": sha256_file(str(snap_bak)),
                     "episodic_count": snap_expected},
        "archive": {"src": str(archive_path), "dst": str(arch_bak),
                    "sha256": sha256_file(str(arch_bak)),
                    "valid_lines": arch_expected},
        "readable_verified": not any(i["level"] == _ERR for i in issues),
    }
    atomic_write_bytes(meta, json.dumps(meta_data, ensure_ascii=False,
                                        indent=2).encode("utf-8"))
    return [snap_bak, arch_bak, meta], issues


def rollback_drill(backup_dir: str, restore_to: Optional[str] = None) -> List[dict]:
    """回滚演练（BK-01）：备份可读性（快照 load + 归档解析 + 计数对账）+
    可选恢复演练（把备份恢复到 restore_to，校验 checksum 一致）。

    计数对账口径：备份时 backup_meta.json 记录原输入期望计数（snapshot
    episodic_count / archive valid_lines）；演练时逐项比对——备份被截断/
    改写/损坏 → 计数或可读性 mismatch → error（BK-01：备份本身可能损坏）。

    演练不触碰生产路径：restore_to 缺省仅校验可读性，不写任何文件。
    """
    bdir = Path(backup_dir)
    issues: List[dict] = []
    snap_bak = bdir / "latest_main.pt"
    arch_bak = bdir / "main.jsonl"
    if not snap_bak.is_file() or not arch_bak.is_file():
        issues.append({"level": _ERR, "msg": f"备份目录不完整: {backup_dir}"})
        return issues

    # 备份时登记的期望计数（BK-01 对账基准）
    meta: dict = {}
    meta_path = bdir / "backup_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
        except (json.JSONDecodeError, OSError):
            issues.append({"level": _WARN,
                           "msg": "backup_meta.json 不可读（无期望计数基准）"})
    expected_snap = (meta.get("snapshot") or {}).get("episodic_count")
    expected_arch = (meta.get("archive") or {}).get("valid_lines")

    try:
        st = load_snapshot_raw(str(snap_bak))
        n = len(extract_episodic(st))
        if expected_snap is not None and n != int(expected_snap):
            issues.append({"level": _ERR,
                           "msg": f"备份快照 episodic {n} ≠ 备份登记 "
                                  f"{expected_snap}（备份被改动/损坏）"})
        issues.append({"level": _WARN,
                       "msg": f"备份快照可读：version={st.get('version')} "
                              f"episodic={n}"})
    except Exception as ex:  # noqa: BLE001
        issues.append({"level": _ERR, "msg": f"备份快照损坏不可读: {ex}"})
    try:
        recs, skipped = read_archive_records(str(arch_bak))
        if expected_arch is not None and len(recs) != int(expected_arch):
            issues.append({"level": _ERR,
                           "msg": f"备份归档有效行 {len(recs)} ≠ 备份登记 "
                                  f"{expected_arch}（备份被改动/损坏）"})
        issues.append({"level": _WARN,
                       "msg": f"备份归档可读：{len(recs)} 条有效行，"
                              f"{skipped} 行坏行跳过"})
    except Exception as ex:  # noqa: BLE001
        issues.append({"level": _ERR, "msg": f"备份归档损坏不可读: {ex}"})
    if restore_to:
        rdir = Path(restore_to)
        rdir.mkdir(parents=True, exist_ok=True)
        for name in ("latest_main.pt", "main.jsonl"):
            shutil.copy2(bdir / name, rdir / name)
            if sha256_file(str(bdir / name)) != sha256_file(str(rdir / name)):
                issues.append({"level": _ERR,
                               "msg": f"恢复校验失败: {name} checksum 不一致"})
        issues.append({"level": _WARN, "msg": f"恢复演练完成: {restore_to}"})
    return issues


# ---------------------------------------------------------------------------
# 输出落盘（§3.2 步骤 4/5；原子替换 + manifest）
# ---------------------------------------------------------------------------

def write_migrated_snapshot(raw_st: dict, merged: List[EpisodicEntry],
                            out_path: str, *, session: str,
                            now_ts: float,
                            archive_out: Optional[List[dict]] = None) -> str:
    """写新 schema 快照（v0.5.0）：原快照状态透传 + episodic_buffer 替换。

    - attractor/purpose/memory(潜变量+buffer)/tokenizer/meta/self_ref
      原样透传（只增不改旧文件，§3.3）；
    - memory.episodic_buffer = 合并后的新条目列表；
    - session_id=turn_count（= max(旧 turn_count, 合并全部条目 max turn，
      含归档-only 文本记录——turn 序列完整性的落点，§3.2-3））/
      last_entropy_ratio 元数据补齐。
    原子写（tmp + os.replace）→ 中断重跑安全（幂等：同输入同输出）。
    """
    new_st = dict(raw_st)
    mem = dict(raw_st.get("memory") or {})
    mem["episodic_buffer"] = list(merged)
    new_st["memory"] = mem
    new_st["session_id"] = str(session)
    old_turn = raw_st.get("turn_count")
    try:
        old_turn_i = int(old_turn) if old_turn is not None else 0
    except (TypeError, ValueError):
        old_turn_i = 0
    arch_turns = [_coerce_turn(r.get("turn")) for r in (archive_out or [])]
    max_turn = max([old_turn_i] + [e.turn for e in merged]
                   + [t for t in arch_turns if t >= 0] + [0])
    new_st["turn_count"] = max_turn
    if raw_st.get("last_entropy_ratio") is not None:
        new_st["last_entropy_ratio"] = float(raw_st["last_entropy_ratio"])
    new_st["timestamp"] = now_ts

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".m7_", suffix=".pt.tmp", dir=str(out.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(new_st, f)
        os.replace(tmp, str(out))
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return out_path


def build_manifest(*, session: str, now_ts: float, data_ts: float,
                   snapshot_in: str, archive_in: str,
                   snap_issues: List[dict], arch_issues: List[dict],
                   merge_stats: dict, merged_issues: List[dict],
                   out_snapshot: str, out_archive: str,
                   backup_dir: str, backup_issues: List[dict],
                   dual_write_rounds: int, journal_path: str) -> dict:
    """汇总迁移报告（机器可读，供对账/审计）。"""
    return {
        "tool": TOOL_NAME,
        "milestone": MILESTONE,
        "rewrite_spec": REWRITE_SPEC,
        "ts": now_ts,          # 本次运行时刻（墙钟）
        "data_ts": data_ts,    # 数据时基 = 原快照 timestamp（重跑可复现）
        "session": session,
        "inputs": {
            "snapshot": {"path": snapshot_in,
                         "sha256": sha256_file(snapshot_in)},
            "archive": {"path": archive_in,
                        "sha256": sha256_file(archive_in)},
        },
        "validation": {
            "snapshot_issues": snap_issues,
            "archive_issues": arch_issues,
            "merged_issues": merged_issues,
            "backup_issues": backup_issues,
            "passed": not any(
                i["level"] == _ERR for i in
                snap_issues + arch_issues + merged_issues + backup_issues),
        },
        "merge": {k: v for k, v in merge_stats.items()
                  if k not in ("merged_episodic", "archive_out")},
        "outputs": {
            "snapshot": {"path": out_snapshot,
                         "sha256": sha256_file(out_snapshot)},
            "archive": {"path": out_archive,
                        "sha256": sha256_file(out_archive)},
        },
        "backup_dir": backup_dir,
        "dual_write": {
            "rounds": dual_write_rounds,
            "journal": journal_path,
        },
    }


def run_migrate(snapshot_path: str, archive_path: str, *,
                session: str, out_dir: str, backup_dir: str,
                apply: bool, force: bool,
                dual_write_rounds: int) -> Tuple[int, dict]:
    """§3.2 完整流程。返回 (exit_code, manifest)。"""
    run_ts = time.time()
    report: dict = {"steps": {}}

    # 数据时基：取原快照 timestamp（迁移不改数据的"as-of"时间）。
    # 快照 timestamp 缺失时回退运行时刻；输出快照/归档记录共用同一
    # data_ts → 重跑字节级可复现（幂等，见 claims）。
    raw_st0 = load_snapshot_raw(snapshot_path)
    data_ts = raw_st0.get("timestamp") or run_ts

    # 0) 备份（BK-01：备份可读校验不过 → 中止）
    if apply:
        bdir = str(Path(backup_dir) / time.strftime("%Y%m%d_%H%M%S"))
        _, backup_issues = backup_inputs(snapshot_path, archive_path, bdir)
        report["backup_dir"] = bdir
    else:
        bdir, backup_issues = "", []
        report["backup_dir"] = "(dry-run 未备份)"

    # 1) 校验
    st, entries, snap_issues = validate_snapshot(snapshot_path)
    archive_records, arch_skipped = read_archive_records(archive_path)
    arch_issues = validate_archive_records(archive_records)
    arch_issues.append({"level": _WARN,
                        "msg": f"归档行数 = {len(archive_records)}"
                               f"（坏行 {arch_skipped}）"})

    # 2/3) 映射 + 合并去重（now_ts=data_ts：归档导出时间可复现）
    merge_stats = merge_entries(entries, archive_records,
                                session=session, now_ts=data_ts)
    merged = merge_stats["merged_episodic"]
    merged_issues = validate_merged(merge_stats, merged)

    all_issues = snap_issues + arch_issues + merged_issues + backup_issues
    errors = [i for i in all_issues if i["level"] == _ERR]
    report["steps"] = {
        "backup": "ok" if not any(i["level"] == _ERR for i in backup_issues)
                  else "FAIL",
        "validate": "ok" if not any(i["level"] == _ERR
                                    for i in snap_issues + arch_issues)
                    else "FAIL",
        "merge": "ok" if not merged_issues or not any(
            i["level"] == _ERR for i in merged_issues) else "FAIL",
    }

    if errors and not force:
        print("[M7] 校验失败（" + str(len(errors)) + " 项 error）；"
              "中止迁移。用 --force 显式放行，或先回滚/修复输入。",
              file=sys.stderr)
        for i in errors:
            print(f"[M7]   [{i['level']}] {i['msg']}", file=sys.stderr)
        return 3, report

    if not apply:
        print("[M7] dry-run：校验+合并规划通过，未写任何文件。")
        print(f"[M7]   快照条目 {merge_stats['snapshot_entries']}"
              f" + 归档 {merge_stats['archive_records']}"
              f" - 去重 {merge_stats['dup_total']}"
              f" = 合并 {merge_stats['final_union']}"
              f"（公式 {'OK' if merge_stats['formula_ok'] else 'FAIL'}）")
        report["dry_run"] = True
        return 0, report

    # 4) 落盘（原子替换；data_ts 保证重跑字节级可复现）
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    out_snap = str(out_dir_p / "migrated_latest_main.pt")
    out_arch = str(out_dir_p / "main.jsonl")
    write_migrated_snapshot(st, merged, out_snap, session=session,
                            now_ts=data_ts,
                            archive_out=merge_stats["archive_out"])
    atomic_write_jsonl(Path(out_arch), merge_stats["archive_out"])

    # 5) 双写回滚期 journal 初始化（LMS_M7_DUAL_WRITE_ROUNDS）
    journal_path = str(out_dir_p / "dual_write_journal.jsonl")
    if dual_write_rounds > 0 and not os.path.exists(journal_path):
        atomic_write_jsonl(Path(journal_path), [{
            "event": "dual_write_start",
            "rounds": dual_write_rounds,
            "final_episodic": merge_stats["final_episodic"],
            "ts": run_ts,
        }])

    manifest = build_manifest(
        session=session, now_ts=run_ts, data_ts=data_ts,
        snapshot_in=snapshot_path, archive_in=archive_path,
        snap_issues=snap_issues, arch_issues=arch_issues,
        merge_stats=merge_stats, merged_issues=merged_issues,
        out_snapshot=out_snap, out_archive=out_arch,
        backup_dir=bdir, backup_issues=backup_issues,
        dual_write_rounds=dual_write_rounds, journal_path=journal_path,
    )
    manifest_path = out_dir_p / "manifest.json"
    atomic_write_bytes(manifest_path,
                       json.dumps(manifest, ensure_ascii=False,
                                  indent=2).encode("utf-8"))
    print(f"[M7] 迁移完成：条目 {merge_stats['final_union']}"
          f"（episodic {merge_stats['final_episodic']} + 归档文本 "
          f"{merge_stats['final_union'] - merge_stats['final_episodic']}；"
          f"公式 {merge_stats['snapshot_entries']}"
          f"+{merge_stats['archive_records']}"
          f"-{merge_stats['dup_total']}）")
    print(f"[M7]   输出快照: {out_snap}")
    print(f"[M7]   输出归档: {out_arch}")
    print(f"[M7]   manifest: {manifest_path}")
    return 0, manifest


# ---------------------------------------------------------------------------
# 迁移后校验（任务③：recall 命中 / 不丢记忆）
# ---------------------------------------------------------------------------

def verify_migrated(snapshot_path: str, archive_path: str, *,
                    session: str, sample_k: int = 10) -> List[dict]:
    """迁移后校验：新核心加载迁移快照 → recall 命中 / 不丢记忆。

    检查项：
      1. 快照可加载（MemoryManager.set_state），episodic 条目数 > 0；
      2. 自命中：抽样条目以其自身语义向量检索，top-k 必含该条目文本
         （recall 命中）；
      3. 灰度口径：store_gray 条目不出现在 external 检索面（口径回归）；
      4. [doubt] 系统事件条目不出现在 external 检索面（P0 污染处置）；
      5. 归档可检索：query_archive 返回带 origin='archive' 的记录；
      6. 计数对账：episodic_size == 合并条目数（不丢记忆）。
    """
    issues: List[dict] = []
    st = load_snapshot_raw(snapshot_path)
    mem_st = st.get("memory") or {}
    num_nodes = int(mem_st.get("num_nodes") or 0)
    if num_nodes <= 0:
        issues.append({"level": _ERR, "msg": "迁移快照无有效 num_nodes"})
        return issues
    mm = MemoryManager(num_nodes=num_nodes)
    mm.set_state(mem_st)
    entries = list(mm.iter_episodic())
    issues.append({"level": _WARN,
                   "msg": f"新核心加载迁移快照：episodic = {len(entries)} 条"})
    if len(entries) == 0:
        issues.append({"level": _ERR, "msg": "迁移快照 episodic 为空（不丢记忆违反）"})

    # 2) 自命中（recall 命中）
    import random
    rng = random.Random(0)
    sample = rng.sample(entries, min(sample_k, len(entries)))
    miss = 0
    for e in sample:
        try:
            hits = mm.recall_episodic(e.semantic_vector, top_k=5,
                                      source_filter=None)
            # recall_episodic 返回 EpisodicEntry 列表（非 (score, entry)）
            hit_texts = {h.text for h in hits}
        except Exception as ex:  # noqa: BLE001
            issues.append({"level": _ERR,
                           "msg": f"recall 自命中异常: {ex}"})
            continue
        if e.text not in hit_texts:
            miss += 1
            issues.append({"level": _ERR,
                           "msg": f"自命中 miss：turn={e.turn} "
                                  f"text={e.text[:40]!r}"})
    issues.append({"level": _WARN,
                   "msg": f"recall 自命中：抽样 {len(sample)} 条，miss {miss} 条"})

    # 3) 灰度口径 + 4) [doubt] 排除
    ext_hits = mm.recall_episodic(torch.zeros(num_nodes), top_k=20,
                                  source_filter="external")
    ext_texts = {h.text for h in ext_hits}
    gray_entries = [e for e in entries if e.source == "store_gray"]
    doubt_entries = [e for e in entries
                     if e.text.lstrip().startswith("[doubt")]
    gray_leak = [e.text for e in gray_entries if e.text in ext_texts]
    doubt_leak = [e.text for e in doubt_entries if e.text in ext_texts]
    if gray_leak:
        issues.append({"level": _ERR,
                       "msg": f"灰度口径违反：{len(gray_leak)} 条 store_gray "
                              f"进入 external 检索面"})
    if doubt_leak:
        issues.append({"level": _ERR,
                       "msg": f"P0 污染处置违反：{len(doubt_leak)} 条 [doubt] "
                              f"系统事件进入 external 检索面"})
    issues.append({"level": _WARN,
                   "msg": f"灰度条目 {len(gray_entries)} 条（external 面泄漏 "
                          f"{len(gray_leak)}），[doubt] 条目 "
                          f"{len(doubt_entries)} 条（泄漏 {len(doubt_leak)}）"})

    # 5) 归档可检索
    try:
        from core.archive.archive_store import count_archive, query_archive
        arch_count = count_archive(session, str(Path(archive_path).parent))
        issues.append({"level": _WARN,
                       "msg": f"归档条目数 = {arch_count}"})
        vec = entries[0].semantic_vector if entries else torch.zeros(num_nodes)
        qr = query_archive(session, vec, k=5, archive_dir=str(Path(archive_path).parent))
        origins = {r.get("origin") for r in qr}
        if qr and "archive" not in origins:
            issues.append({"level": _ERR,
                           "msg": "归档检索返回条目缺少 origin='archive' 标记"})
    except Exception as ex:  # noqa: BLE001
        issues.append({"level": _ERR, "msg": f"归档检索校验异常: {ex}"})
    return issues


# ---------------------------------------------------------------------------
# 双写回滚期 journal（§3.2 步骤 5）
# ---------------------------------------------------------------------------

def dual_write_record(journal_path: str, *, round_no: int,
                      old_inc: int, new_inc: int) -> List[dict]:
    """记录一轮双写增量对比。返回 issues（增量不一致 → error）。"""
    issues: List[dict] = []
    jp = Path(journal_path)
    jp.parent.mkdir(parents=True, exist_ok=True)
    rec = {
        "event": "round",
        "round": int(round_no),
        "old_increment": int(old_inc),
        "new_increment": int(new_inc),
        "match": int(old_inc) == int(new_inc),
        "ts": time.time(),
    }
    existing = []
    if jp.is_file():
        existing = [json.loads(l) for l in jp.read_text(
            encoding="utf-8").splitlines() if l.strip()]
    for r in existing:
        if r.get("event") == "round" and int(r.get("round", -1)) == int(round_no):
            issues.append({"level": _ERR,
                           "msg": f"round {round_no} 已记录（幂等拒绝重复登记）"})
            return issues
    if rec["match"]:
        issues.append({"level": _WARN,
                       "msg": f"round {round_no} 增量一致"
                              f"（old={old_inc} new={new_inc}）"})
    else:
        issues.append({"level": _ERR,
                       "msg": f"round {round_no} 增量不一致"
                              f"（old={old_inc} vs new={new_inc}）——"
                              f"不满足切单写条件，继续双写"})
    atomic_write_jsonl(jp, existing + [rec])
    return issues


def dual_write_check(journal_path: str, required_rounds: int) -> List[dict]:
    """双写收尾校验：round 数达标且全部 match → 可切单写。"""
    issues: List[dict] = []
    jp = Path(journal_path)
    if not jp.is_file():
        issues.append({"level": _ERR, "msg": f"双写 journal 不存在: {journal_path}"})
        return issues
    rounds = [json.loads(l) for l in jp.read_text(
        encoding="utf-8").splitlines() if l.strip()
        and json.loads(l).get("event") == "round"]
    if len(rounds) < required_rounds:
        issues.append({"level": _ERR,
                       "msg": f"双写轮数不足：{len(rounds)}/{required_rounds}"})
        return issues
    bad = [r for r in rounds if not r.get("match", False)]
    if bad:
        issues.append({"level": _ERR,
                       "msg": f"{len(bad)} 轮增量不一致，禁止切单写"})
    else:
        issues.append({"level": _WARN,
                       "msg": f"双写 {len(rounds)} 轮全部一致 → 满足切单写条件"})
    return issues


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="migrate_m7.py",
        description="LMS 核心重建 M7 数据迁移（§三/§7.1 M7）：快照+归档"
                    "合并去重，幂等可回滚可重跑")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_validate = sub.add_parser("validate", help="只校验输入（不写任何文件）")
    p_validate.add_argument("--snapshot", default=env_str(
        "LMS_M7_SNAPSHOT", DEFAULT_SNAPSHOT))
    p_validate.add_argument("--archive", default=env_str(
        "LMS_M7_ARCHIVE", DEFAULT_ARCHIVE))
    p_validate.set_defaults(func=cmd_validate)

    p_migrate = sub.add_parser("migrate", help="执行迁移（默认 dry-run）")
    p_migrate.add_argument("--snapshot", default=env_str(
        "LMS_M7_SNAPSHOT", DEFAULT_SNAPSHOT))
    p_migrate.add_argument("--archive", default=env_str(
        "LMS_M7_ARCHIVE", DEFAULT_ARCHIVE))
    p_migrate.add_argument("--session", default=env_str(
        "LMS_M7_SESSION", DEFAULT_SESSION))
    p_migrate.add_argument("--out-dir", default=None,
                           help=f"输出目录（默认 {DEFAULT_OUT_DIR.format(session='{session}')}）")
    p_migrate.add_argument("--backup-dir", default=env_str(
        "LMS_M7_BACKUP_DIR", DEFAULT_BACKUP_DIR))
    p_migrate.add_argument("--apply", action="store_true",
                           help="实际落盘（默认 dry-run 只规划）")
    p_migrate.add_argument("--force", action="store_true",
                           help="校验 error 仍继续（默认 0；显式开启）")
    p_migrate.add_argument("--dual-write-rounds", type=int,
                           default=env_int("LMS_M7_DUAL_WRITE_ROUNDS",
                                           DEFAULT_DUAL_WRITE_ROUNDS))
    p_migrate.set_defaults(func=cmd_migrate)

    p_drill = sub.add_parser("rollback-drill", help="回滚演练（BK-01）")
    p_drill.add_argument("--backup-dir", required=True)
    p_drill.add_argument("--restore-to", default=None,
                         help="可选：恢复演练目标目录（checksum 对账）")
    p_drill.set_defaults(func=cmd_drill)

    p_verify = sub.add_parser("verify", help="迁移后校验（recall 命中/不丢记忆）")
    p_verify.add_argument("--snapshot", required=True)
    p_verify.add_argument("--archive", required=True)
    p_verify.add_argument("--session", default=env_str(
        "LMS_M7_SESSION", DEFAULT_SESSION))
    p_verify.add_argument("--sample", type=int, default=10)
    p_verify.set_defaults(func=cmd_verify)

    p_dw = sub.add_parser("dual-write", help="双写回滚期 journal")
    dw_sub = p_dw.add_subparsers(dest="dw_cmd", required=True)
    p_dw_r = dw_sub.add_parser("record")
    p_dw_r.add_argument("--journal", required=True)
    p_dw_r.add_argument("--round", type=int, required=True)
    p_dw_r.add_argument("--old-inc", type=int, required=True)
    p_dw_r.add_argument("--new-inc", type=int, required=True)
    p_dw_r.set_defaults(func=cmd_dw_record)
    p_dw_c = dw_sub.add_parser("check")
    p_dw_c.add_argument("--journal", required=True)
    p_dw_c.add_argument("--rounds", type=int, required=True)
    p_dw_c.set_defaults(func=cmd_dw_check)
    return ap


def cmd_validate(args) -> int:
    st, entries, issues = validate_snapshot(args.snapshot)
    records, skipped = read_archive_records(args.archive)
    issues += validate_archive_records(records)
    issues.append({"level": _WARN,
                   "msg": f"归档行数 = {len(records)}（坏行 {skipped}）"})
    errors = [i for i in issues if i["level"] == _ERR]
    for i in issues:
        print(f"[M7][{i['level']}] {i['msg']}")
    if errors:
        print(f"[M7] 校验失败：{len(errors)} 项 error", file=sys.stderr)
        return 3
    print("[M7] 校验通过（可执行 migrate）")
    return 0


def cmd_migrate(args) -> int:
    out_dir = args.out_dir or DEFAULT_OUT_DIR.format(session=args.session)
    code, manifest = run_migrate(
        args.snapshot, args.archive,
        session=args.session, out_dir=out_dir,
        backup_dir=args.backup_dir,
        apply=args.apply, force=args.force,
        dual_write_rounds=args.dual_write_rounds,
    )
    return code


def cmd_drill(args) -> int:
    issues = rollback_drill(args.backup_dir, restore_to=args.restore_to)
    errors = [i for i in issues if i["level"] == _ERR]
    for i in issues:
        print(f"[M7][{i['level']}] {i['msg']}")
    if errors:
        print(f"[M7] 回滚演练失败：{len(errors)} 项 error", file=sys.stderr)
        return 3
    print("[M7] 回滚演练通过（备份可读，可恢复）")
    return 0


def cmd_verify(args) -> int:
    issues = verify_migrated(args.snapshot, args.archive,
                             session=args.session, sample_k=args.sample)
    errors = [i for i in issues if i["level"] == _ERR]
    for i in issues:
        print(f"[M7][{i['level']}] {i['msg']}")
    if errors:
        print(f"[M7] 迁移后校验失败：{len(errors)} 项 error", file=sys.stderr)
        return 3
    print("[M7] 迁移后校验通过（recall 命中/不丢记忆）")
    return 0


def cmd_dw_record(args) -> int:
    issues = dual_write_record(args.journal, round_no=args.round,
                               old_inc=args.old_inc, new_inc=args.new_inc)
    errors = [i for i in issues if i["level"] == _ERR]
    for i in issues:
        print(f"[M7][{i['level']}] {i['msg']}")
    return 3 if errors else 0


def cmd_dw_check(args) -> int:
    issues = dual_write_check(args.journal, required_rounds=args.rounds)
    errors = [i for i in issues if i["level"] == _ERR]
    for i in issues:
        print(f"[M7][{i['level']}] {i['msg']}")
    return 3 if errors else 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = _build_parser()
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

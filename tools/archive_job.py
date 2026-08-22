#!/usr/bin/env python3
"""
T2.3 归档重建工具：手动/定时触发归档重建（扫描快照 -> 重建索引）
================================================================

背景：episodic 只保留最近 200 条滚动窗口，窗口外记忆靠归档 JSONL 补充检索。
本工具把**快照里的 episodic 历史**回填进归档（含已被 200 条窗口挤掉的旧记忆），
供 /recall 合并检索命中。

用法::

    # 重建指定会话的归档（扫描 snapshots/{session}/ 下全部轮次快照）
    python tools/archive_job.py rebuild --session main

    # 连同旧平铺快照（snapshots/snapshot_*.pt、snapshots/latest.pt）一起扫描
    python tools/archive_job.py rebuild --session main --legacy

    # 自定义目录（快照目录/归档目录）
    python tools/archive_job.py rebuild --session main \\
        --snapshot-dir /path/to/snapshots --archive-dir /path/to/archive

    # 查看归档统计
    python tools/archive_job.py stats [--session main]

定时触发示例（cron 每日一次，重建 main 会话索引）::

    0 3 * * * cd <LMS_ROOT> && \\
        .venv/bin/python tools/archive_job.py rebuild --session main >> logs/archive_job.log 2>&1

安全说明：
  - 只读快照、只写归档（data/archive/{session}.jsonl），不触碰运行实例；
  - [B10] 重建是"扫描快照合并去重，只增不删"（按 (turn, text_hash) 去重）：
    旧实现整体替换归档会把不在保留快照中的窗口外旧条目永久丢失；
    现实现先保留现有归档行再追加新记录，旧条目不再丢失；
  - 快照读取失败/条目损坏一律跳过并告警（fail-open），绝不中断整体重建。
"""

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

# 让脚本可以在任意工作目录下直接运行（python tools/archive_job.py）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.archive.archive_store import (  # noqa: E402
    archive_path_for,
    count_archive,
    entry_to_record,
    get_archive_dir,
    rebuild_archive,
)
from core.paths import get_snapshot_dir  # noqa: E402
from persistence.snapshot import (  # noqa: E402
    Snapshot,
    sanitize_session_id,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("archive_job")


# ----------------------------------------------------------------------
# 快照收集
# ----------------------------------------------------------------------

def _file_sort_key(p: Path):
    """按文件名中的 (turn, ts) 排序；解析不出轮次的大轮次排在最后。"""
    m = re.search(r"_(\d+)_(\d{8}_\d{6})\.pt$", p.name)
    if m:
        return (int(m.group(1)), m.group(2), p.name)
    return (10**9, p.name, "")


def collect_snapshot_files(snapshot_dir: Path, session_id: str,
                           legacy: bool = False) -> list:
    """收集某会话的全部快照文件（新命名规范 + 可选旧平铺格式）。

    新命名：snapshots/{session}/snapshot_{session}_{turn}_{ts}.pt 与
    latest_{session}.pt；旧格式（--legacy）：snapshots/snapshot_*.pt、
    snapshots/latest.pt（T1.1 之前平铺，无会话归属，按指定会话处理）。
    """
    sid = sanitize_session_id(session_id)
    files: dict = {}  # 按 resolved path 去重

    session_dir = snapshot_dir / sid
    if session_dir.is_dir():
        for p in sorted(session_dir.glob(f"snapshot_{sid}_*.pt"),
                        key=_file_sort_key):
            files[str(p.resolve())] = p
        latest = session_dir / f"latest_{sid}.pt"
        if latest.exists():
            files[str(latest.resolve())] = latest

    if legacy:
        for p in sorted(snapshot_dir.glob("snapshot_*.pt"),
                        key=_file_sort_key):
            files[str(p.resolve())] = p
        flat_latest = snapshot_dir / "latest.pt"
        if flat_latest.exists():
            files[str(flat_latest.resolve())] = flat_latest

    return list(files.values())


def extract_records_from_snapshot(path: Path, session_id: str) -> list:
    """从单个快照 .pt 中提取 episodic 条目并转换为归档记录。

    快照 memory 字段缺失/无 episodic_buffer（旧版本）时返回空列表；
    单条目转换失败跳过（fail-open）。
    """
    try:
        data = Snapshot().load_raw(str(path))
    except Exception as e:
        logger.warning("跳过损坏/不可读快照 %s: %s", path, e)
        return []

    mem = data.get("memory") or {}
    entries = mem.get("episodic_buffer") or []
    records = []
    for entry in entries:
        try:
            rec = entry_to_record(session_id, entry)
            if rec is not None:
                records.append(rec)
        except Exception as e:
            logger.warning("快照 %s 中条目转换失败（跳过）: %s", path, e)
    return records


# ----------------------------------------------------------------------
# 子命令
# ----------------------------------------------------------------------

def cmd_rebuild(args) -> int:
    """扫描快照 -> 合并更新归档索引（[B10] 只增不删，按 (turn, text_hash) 去重）。"""
    if args.snapshot_dir:
        snap_dir = Path(args.snapshot_dir).expanduser()
    else:
        snap_dir = Path(os.environ.get("LMS_SNAPSHOT_DIR", "").strip()) \
            if os.environ.get("LMS_SNAPSHOT_DIR", "").strip() \
            else get_snapshot_dir()

    sessions = args.session.split(",") if args.session else ["main"]
    t0 = time.time()
    for sid in sessions:
        sid = sanitize_session_id(sid.strip() or "main")
        files = collect_snapshot_files(snap_dir, sid, legacy=args.legacy)
        if not files:
            logger.warning("会话 %s 无可扫描快照（目录: %s），跳过", sid, snap_dir)
            continue

        records = []
        for p in files:
            recs = extract_records_from_snapshot(p, sid)
            if recs:
                logger.info("  %s: 提取 %d 条", p.name, len(recs))
            records.extend(recs)

        n = rebuild_archive(sid, records, archive_dir=args.archive_dir)
        arch = archive_path_for(sid, args.archive_dir)
        # [B10] 文案同步合并语义：不再宣称"重建完成"（会误导为整体替换）
        print(f"[{sid}] 归档合并完成：扫描 {len(files)} 个快照，"
              f"归档 {n} 条（只增不删，既有条目保留）-> {arch}")
    logger.info("归档合并耗时 %.2fs", time.time() - t0)
    return 0


def cmd_stats(args) -> int:
    """打印归档统计（条数/路径/大小）。"""
    sessions = args.session.split(",") if args.session else ["main"]
    total = 0
    for sid in sessions:
        sid = sanitize_session_id(sid.strip() or "main")
        n = count_archive(sid, args.archive_dir)
        path = archive_path_for(sid, args.archive_dir)
        size = path.stat().st_size if path.exists() else 0
        print(f"[{sid}] 归档条目: {n} 条，文件: {path}（{size} 字节）")
        total += n
    print(f"总计: {total} 条")
    return 0


# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="LMS T2.3 归档重建工具（扫描快照 -> 重建 data/archive 索引）")
    sub = parser.add_subparsers(dest="command", required=True)

    p_rebuild = sub.add_parser("rebuild", help="扫描快照重建归档索引")
    p_rebuild.add_argument("--session", default="main",
                           help="会话标识（逗号分隔多个；默认 main）")
    p_rebuild.add_argument("--snapshot-dir", default=None,
                           help="快照根目录（默认 LMS_SNAPSHOT_DIR 或 get_snapshot_dir()）")
    p_rebuild.add_argument("--archive-dir", default=None,
                           help="归档目录（默认 LMS_ARCHIVE_DIR 或 data/archive）")
    p_rebuild.add_argument("--legacy", action="store_true",
                           help="同时扫描旧平铺快照（snapshots/snapshot_*.pt、latest.pt）")
    p_rebuild.set_defaults(func=cmd_rebuild)

    p_stats = sub.add_parser("stats", help="查看归档统计")
    p_stats.add_argument("--session", default="main",
                         help="会话标识（逗号分隔多个；默认 main）")
    p_stats.add_argument("--archive-dir", default=None,
                         help="归档目录（默认 LMS_ARCHIVE_DIR 或 data/archive）")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

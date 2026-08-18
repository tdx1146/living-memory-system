#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMS 数据抢救工具 ①：export_episodic —— 从快照导出 episodic 情景记忆 → JSONL
============================================================================
背景（总体方案 §6 / T2.5）：snapshot_50/100/150/200 + latest.pt 内含 161 条
真实对话记忆，淹没在 bus 垃圾与查询回声里。本工具是抢救流程第一步：
只读导出，绝不写回快照。

用法:
    python tools/export_episodic.py --snapshots snapshots/rescue-backup-20260810 \
        --out /tmp/export_raw.jsonl
    python tools/export_episodic.py --files snapshots/rescue-backup-20260810/latest.pt \
        --out -                                   # 指定单个文件，输出到 stdout
    python tools/export_episodic.py --dry-run     # 只统计不写文件（安全）

行为:
    * torch.load(map_location='cpu', weights_only=False) 只读加载；
    * 提取顶层 st['memory']['episodic_buffer']（EpisodicEntry dataclass，
      兼容 dict / (text, vec, surprise, turn, source) 元组形态）；
    * 单文件加载失败 → WARNING 跳过（fail-open），不中断整体；
    * --dedup（默认开）：按 text 的 sha256 全局去重，保留首次出现
      （snapshot_50/100/150/200 轮次重叠，去重后即 161 条唯一文本）；
    * 输出 JSONL 每行: {"session_guess","turn","surprise","source",
      "ts","snapshot_file","text"}；--dry-run 只打印统计。

退出码: 0=成功（含空结果） 2=参数错误。
"""

import argparse
import hashlib
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 以脚本方式运行时 sys.path[0] = tools/，需显式加回项目根，
# 否则 torch.load 反序列化 EpisodicEntry（core.hippocampus.memory）会失败
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 默认扫描目录：历史抢救备份（旧版扁平命名快照，只读封存）
DEFAULT_SNAPSHOTS = os.path.join(PROJECT_ROOT, "snapshots", "rescue-backup-20260810")


def _load_entries(path: str) -> list:
    """从单个快照文件提取 episodic 条目列表（只读，失败抛异常由调用方处理）。"""
    import torch
    st = torch.load(path, map_location="cpu", weights_only=False)
    mem = st.get("memory") or {}
    eb = mem.get("episodic_buffer") or []
    entries = []
    for e in eb:
        if hasattr(e, "text"):  # EpisodicEntry dataclass
            text = e.text
            surprise = getattr(e, "surprise", 0.0)
            turn = getattr(e, "turn", 0)
            source = getattr(e, "source", "external")
        elif isinstance(e, dict):
            text = e.get("text", "")
            surprise = e.get("surprise", 0.0)
            turn = e.get("turn", 0)
            source = e.get("source", "external")
        elif isinstance(e, (tuple, list)) and len(e) >= 1:
            text = e[0]
            surprise = e[2] if len(e) > 2 else 0.0
            turn = e[3] if len(e) > 3 else 0
            source = e[4] if len(e) > 4 else "external"
        else:
            continue
        if not isinstance(text, str) or not text.strip():
            continue
        entries.append({
            "text": text,
            "surprise": float(surprise) if surprise is not None else 0.0,
            "turn": int(turn) if turn is not None else 0,
            "source": str(source),
        })
    return entries


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="从 LMS 快照导出 episodic 情景记忆 → JSONL（T2.5 抢救第一步）")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--snapshots", default=DEFAULT_SNAPSHOTS,
                     help=f"快照目录（扫描 *.pt，默认 {DEFAULT_SNAPSHOTS}）")
    src.add_argument("--files", nargs="+", help="显式指定快照文件清单（替代 --snapshots）")
    ap.add_argument("--out", default=None,
                    help="输出 JSONL 路径（默认 export_raw.jsonl；'-' = stdout）")
    ap.add_argument("--session-guess", default="main",
                    help="session_guess 字段值（默认 main，快照无 session_id 时的归属猜测）")
    ap.add_argument("--dedup", dest="dedup", action="store_true", default=True,
                    help="按 text sha256 去重（默认开）")
    ap.add_argument("--no-dedup", dest="dedup", action="store_false")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印统计，不写任何文件")
    ap.add_argument("--limit", type=int, default=0,
                    help="最多导出的唯一条目数（0=不限）")
    args = ap.parse_args()

    if args.files:
        files = sorted(args.files)
    else:
        if not os.path.isdir(args.snapshots):
            print(f"[export] 快照目录不存在: {args.snapshots}", file=sys.stderr)
            return 2
        files = sorted(
            os.path.join(args.snapshots, f)
            for f in os.listdir(args.snapshots)
            if f.endswith(".pt") and not f.endswith(".lock")
        )

    out_path = args.out or os.path.join(PROJECT_ROOT, "export_raw.jsonl")
    seen: set = set()
    rows = []
    stats = {"files": 0, "raw_entries": 0, "load_fail": 0}
    ts = time.time()

    for path in files:
        if not os.path.isfile(path):
            print(f"[export][WARN] 跳过（非文件）: {path}")
            continue
        try:
            entries = _load_entries(path)
        except Exception as e:  # noqa: BLE001 - fail-open：坏文件跳过不中断
            stats["load_fail"] += 1
            print(f"[export][WARN] 加载失败，跳过: {path}: {type(e).__name__}: {e}")
            continue
        stats["files"] += 1
        stats["raw_entries"] += len(entries)
        for e in entries:
            if args.dedup:
                h = _sha256(e["text"])
                if h in seen:
                    continue
                seen.add(h)
            rows.append({
                "session_guess": args.session_guess,
                "turn": e["turn"],
                "surprise": e["surprise"],
                "source": e["source"],
                "ts": ts,
                "snapshot_file": os.path.basename(path),
                "text": e["text"],
            })
            if args.limit and len(rows) >= args.limit:
                break
        if args.limit and len(rows) >= args.limit:
            break

    # 统计
    total_chars = sum(len(r["text"]) for r in rows)
    sample = rows[0]["text"][:60] if rows else ""
    print(f"[export] 扫描 {stats['files']} 个快照，原始条目 {stats['raw_entries']}，"
          f"加载失败 {stats['load_fail']}，去重后导出 {len(rows)} 条"
          f"（{total_chars} 字符）")
    if sample:
        print(f"[export] 样例: {sample!r}")
    if args.dry_run:
        print("[export] --dry-run：未写任何文件")
        return 0

    # 写输出
    if out_path == "-":
        for r in rows:
            print(json.dumps(r, ensure_ascii=False))
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[export] 已写入 {out_path}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

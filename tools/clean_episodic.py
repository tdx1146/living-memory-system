#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMS 数据抢救工具 ②：clean_episodic —— 清洗导出的 episodic JSONL
====================================================================
背景（总体方案 §6 / T2.5 第三步）：导出后的条目混杂 bus 系统事件、
查询回声、测试/种子文本、元数据包裹等污染。本工具按规则清洗：

过滤规则（按序执行，命中任一即丢弃，并计入 rule 统计）:
  1. system_garbage : bus_heartbeat | 任务 .* 完成 \(exit=\d+\) |
                      heartbeat poll | HEARTBEAT_OK | 事件总线心跳
  2. test_seed      : 开头命中 记忆模式 | 我同时唤醒多条记忆 | 你好，我叫小明
                      | 系统提示 | Pre-compaction memory flush | Store durable
  3. metadata_wrap  : 含 'Sender (untrusted metadata)' 的元数据包裹 →
                      剥壳取正文（保留最后一个 'metadata)' 之后的内容）；
                      若剥壳后为空则丢弃
  4. query_echo     : 文本长度 < --min-len（默认 5）→ 丢弃；
                      --query-echoes FILE 提供已知查询清单（每行一条）→
                      规范化后完全匹配/包含命中 → 丢弃
  5. truncate       : 超过 --max-len（默认 2000）字 → 截断（不丢弃）

去重: 规范化（strip + 空白折叠）后按 sha256 去重，保留首次出现（默认开）。

用法:
    python tools/clean_episodic.py --in /tmp/export_raw.jsonl --out /tmp/export_clean.jsonl
    python tools/clean_episodic.py --in - --out - --dry-run
    python tools/clean_episodic.py --in /tmp/export_raw.jsonl --query-echoes queries.txt

输出: 与输入同 schema 的 JSONL；--dry-run 只打印规则命中统计。

退出码: 0=成功 2=参数错误。
"""

import argparse
import hashlib
import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 规则 1：系统事件/bus 垃圾
RE_SYSTEM = re.compile(
    r"bus_heartbeat|任务 .* 完成 \(exit=\d+\)|heartbeat poll|HEARTBEAT_OK|"
    r"事件总线心跳|轮询|poll 事件",
    re.IGNORECASE,
)
# 规则 2：测试/种子/系统提示片段（开头命中）
RE_TEST_SEED = re.compile(
    r"^(记忆模式|我同时唤醒多条记忆|你好，我叫小明|系统提示|"
    r"Pre-compaction memory flush|Store durable memories only|"
    r"这是测试|测试消息)",
    re.IGNORECASE,
)
# 规则 3：元数据包裹标记
META_MARKER = "Sender (untrusted metadata)"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


def _load_query_echoes(path: str) -> list:
    if not path:
        return []
    out = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            t = _norm(line)
            if t:
                out.append(t)
    return out


def clean_row(row: dict, query_echoes: list, min_len: int,
              max_len: int, stats: dict) -> dict | None:
    """清洗单行；返回清洗后的 dict（或 None=丢弃）。"""
    text = row.get("text", "")
    if not isinstance(text, str):
        stats["dropped"]["empty"] += 1
        return None

    # 规则 3a：元数据包裹剥壳
    if META_MARKER in text:
        parts = text.split(META_MARKER)
        body = parts[-1]
        # 剥掉标记后的常见分隔符（: 或 > 或换行）
        body = re.sub(r"^[\s:>\-]+", "", body)
        if not body.strip():
            stats["dropped"]["metadata_wrap"] += 1
            return None
        text = body
        stats["stripped"]["metadata_wrap"] += 1

    # 规则 1：系统事件
    if RE_SYSTEM.search(text):
        stats["dropped"]["system_garbage"] += 1
        return None
    # 规则 2：测试/种子/套话
    if RE_TEST_SEED.match(text):
        stats["dropped"]["test_seed"] += 1
        return None
    # 规则 4：长度下限
    if len(_norm(text)) < min_len:
        stats["dropped"]["too_short"] += 1
        return None
    # 规则 4b：查询回声（与已知查询重复）
    if query_echoes:
        nt = _norm(text)
        if any(nt == q or q in nt for q in query_echoes):
            stats["dropped"]["query_echo"] += 1
            return None
    # 规则 5：截断
    if len(text) > max_len:
        text = text[:max_len] + "…[截断]"
        stats["truncated"] += 1

    out = dict(row)
    out["text"] = text
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="清洗 export_episodic 导出的 JSONL（T2.5 抢救第二步）")
    ap.add_argument("--in", dest="in_path", default=None,
                    help="输入 JSONL（默认 export_raw.jsonl；'-' = stdin）")
    ap.add_argument("--out", default=None,
                    help="输出 JSONL（默认 export_clean.jsonl；'-' = stdout）")
    ap.add_argument("--dry-run", action="store_true", help="只打印统计，不写文件")
    ap.add_argument("--dedup", dest="dedup", action="store_true", default=True)
    ap.add_argument("--no-dedup", dest="dedup", action="store_false")
    ap.add_argument("--min-len", type=int, default=5, help="最短文本长度（默认 5）")
    ap.add_argument("--max-len", type=int, default=2000, help="最长文本长度（默认 2000）")
    ap.add_argument("--query-echoes", default=None,
                    help="已知查询清单文件（每行一条），命中即视为查询回声丢弃")
    args = ap.parse_args()

    in_path = args.in_path or os.path.join(PROJECT_ROOT, "export_raw.jsonl")
    out_path = args.out or os.path.join(PROJECT_ROOT, "export_clean.jsonl")
    query_echoes = _load_query_echoes(args.query_echoes) if args.query_echoes else []

    stats = {
        "input": 0, "kept": 0,
        "dropped": {"system_garbage": 0, "test_seed": 0, "metadata_wrap": 0,
                    "too_short": 0, "query_echo": 0, "empty": 0},
        "stripped": {"metadata_wrap": 0},
        "truncated": 0,
    }

    # 读取输入
    if in_path == "-":
        lines = sys.stdin
    else:
        if not os.path.isfile(in_path):
            print(f"[clean] 输入文件不存在: {in_path}", file=sys.stderr)
            return 2
        lines = open(in_path, encoding="utf-8", errors="replace")
    rows = []
    try:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                stats["dropped"]["empty"] += 1  # 坏行计为丢弃
    finally:
        if in_path != "-":
            lines.close()

    stats["input"] = len(rows)
    seen: set = set()
    kept_rows = []
    for row in rows:
        cleaned = clean_row(row, query_echoes, args.min_len, args.max_len, stats)
        if cleaned is None:
            continue
        if args.dedup:
            h = _sha256(_norm(cleaned["text"]))
            if h in seen:
                stats["dropped"]["dup"] = stats["dropped"].get("dup", 0) + 1
                continue
            seen.add(h)
        kept_rows.append(cleaned)

    stats["kept"] = len(kept_rows)
    print(f"[clean] 输入 {stats['input']} 条 → 保留 {stats['kept']} 条")
    for rule, n in stats["dropped"].items():
        if n:
            print(f"[clean]   丢弃[{rule}]: {n}")
    if stats["stripped"]["metadata_wrap"]:
        print(f"[clean]   剥壳[metadata_wrap]: {stats['stripped']['metadata_wrap']}")
    if stats["truncated"]:
        print(f"[clean]   截断: {stats['truncated']}")
    if args.dry_run:
        print("[clean] --dry-run：未写任何文件")
        return 0

    if out_path == "-":
        for r in kept_rows:
            print(json.dumps(r, ensure_ascii=False))
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in kept_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[clean] 已写入 {out_path}（{len(kept_rows)} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

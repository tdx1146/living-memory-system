#!/usr/bin/env python3
"""
clean_episodic.py — LMS 记忆垃圾清洗（2026-08-10 设计回归）
============================================================
清理 main 脑 episodic 缓冲区里的非对话垃圾：
  - Sender (untrusted metadata) JSON 包装（消息元数据被误存）
  - System (untrusted) 端口探测/系统事件
  - 测试数据（"我叫小明"等 lms_store 验证用例）

用法：
    python3 tools/clean_episodic.py --session main [--dry-run] [--backup]

设计原则（dandan：不允许乱改）：
  - 默认 --dry-run 只报告不清理
  - --backup 清理前把原快照备份到 snapshots/main/*.pre-clean.pt
  - 只删明确垃圾（正则匹配），真实对话一律保留
  - [B18] fail-closed：探测 :8190 存活即拒绝执行（live 内存每 ~34s 覆盖磁盘，
    清洗不生效且无锁直写有撕裂风险）——"前置：服务已停止"从约定升级为强制
"""

import argparse
import glob
import os
import re
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

# ── 垃圾判定规则（保守：宁可漏不可误删）────────────────────────
_GARBAGE_PATTERNS = [
    re.compile(r"Sender \(untrusted metadata\)", re.I),      # metadata 包装
    re.compile(r"System \(untrusted\)", re.I),               # 系统事件
    re.compile(r"^System:", re.I),                           # System 前缀
    re.compile(r"端口探测|自主唤醒.*冒烟测试", re.I),           # 探测日志
    re.compile(r"用户: 我叫小明|装饰器确实是Python", re.I),      # 测试数据
    # P0 污染处置（2026-08-17）：[doubt 系统事件（conflict 事件 /
    # [doubt-supersedes] 证伪标记）不是对话，清理出 episodic。锚定行首：
    # 正文提及 [doubt 的真实条目不误伤（与 memory.py/archive_store.py
    # 检索排除同正则语义）。
    re.compile(r"^\s*\[doubt(?:-[a-z]+)?\]", re.I),
]


def _is_garbage(text: str) -> bool:
    return any(p.search(text) for p in _GARBAGE_PATTERNS)


def _find_episodic(state: dict):
    """递归找 episodic 列表。"""
    def walk(o, depth=0):
        if depth > 6 or not isinstance(o, dict):
            return None
        for k, v in o.items():
            if "episodic" in k.lower() and isinstance(v, list):
                return v
            r = walk(v, depth + 1)
            if r is not None:
                return r
        return None
    return walk(state)


def main() -> int:
    # [B18] fail-closed 探测：live :8190 运行中清洗会被内存覆盖（每 ~34s
    # 写一次快照）且本工具无锁直写有撕裂风险 → 拒绝执行（不可达才允许继续）
    import urllib.request
    try:
        with urllib.request.urlopen(
                "http://127.0.0.1:8190/health", timeout=5) as r:
            if r.status == 200:
                print("❌ LMS :8190 正在运行：清洗会被内存覆盖且无锁直写有"
                      "撕裂风险，请先停服再执行")
                return 1
    except Exception:
        pass  # 不可达 → 允许继续

    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="main")
    ap.add_argument("--dry-run", action="store_true", default=True,
                    help="只报告不清理（默认）")
    ap.add_argument("--apply", action="store_true", help="实际清理（需显式指定）")
    ap.add_argument("--backup", action="store_true", help="清理前备份快照")
    args = ap.parse_args()

    snap_dir = f"snapshots/{args.session}"
    cands = sorted(glob.glob(os.path.join(snap_dir, "latest_*.pt")),
                   key=os.path.getmtime)
    if not cands:
        print(f"❌ 无快照: {snap_dir}")
        return 1
    latest = cands[-1]
    print(f"加载: {latest}")

    d = torch.load(latest, map_location="cpu", weights_only=False)
    st = d.get("state", d)
    ep = _find_episodic(st)
    if ep is None:
        print("❌ 未找到 episodic 缓冲区")
        return 1

    total = len(ep)
    keep, drop = [], []
    for e in ep:
        # 兼容 dict 与 EpisodicEntry dataclass（0.5.0/T1.3 起快照存类实例，
        # 旧版 str(e) 取 repr 导致锚定正则永不命中——2026-08-17 dry-run 实锤）
        txt = str(e.get("text", "")) if isinstance(e, dict) else str(
            getattr(e, "text", "") or "")
        if _is_garbage(txt):
            drop.append(e)
        else:
            keep.append(e)

    print(f"episodic 总数: {total} | 保留: {len(keep)} | 清理: {len(drop)}")
    if drop:
        print("清理清单:")
        for e in drop:
            txt = str(e.get("text", "")) if isinstance(e, dict) else str(
                getattr(e, "text", "") or "")
            print(f"  - {txt[:60]}")

    if args.dry_run and not args.apply:
        print("（dry-run：未改动。加 --apply 实际清理，--backup 先备份）")
        return 0

    if args.apply:
        if args.backup:
            bak = latest.replace(".pt", f".pre-clean-{datetime.now():%Y%m%d-%H%M}.pt")
            shutil.copy2(latest, bak)
            print(f"已备份: {bak}")
        # 替换 episodic 列表
        def walk_replace(o, depth=0):
            if depth > 6 or not isinstance(o, dict):
                return
            for k, v in list(o.items()):
                if "episodic" in k.lower() and isinstance(v, list):
                    o[k] = keep
                    return
                walk_replace(v, depth + 1)
        walk_replace(st)
        # 写回（原子）
        tmp = latest + ".tmp"
        torch.save(d, tmp)
        os.replace(tmp, latest)
        print(f"✅ 已清理 {len(drop)} 条垃圾，快照已更新: {latest}")
        print("   ⚠️ 需重启 :8190 让内存中的缓冲区重新加载")
    return 0


if __name__ == "__main__":
    sys.exit(main())

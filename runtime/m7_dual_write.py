#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMS M7 · 双写回滚期运行时登记模块（dual-write journal runtime registration）
================================================================================

背景（M7-数据迁移-20260818.md §生产切换说明 + 规格 §三.3.2-5）::

    双写回滚期（LMS_M7_DUAL_WRITE_ROUNDS=30）——新核心先以"读新写旧"或
    "双写"运行 N 轮，**每轮登记旧/新两侧的增量**（old_increment /
    new_increment），N 轮全部一致（dual_write_check）后才切单写；
    任一不一致 → error → 禁止切单写。journal schema（与 tools 完全一致）::

        {"event": "round", "round": N, "old_increment": ..,
         "new_increment": .., "match": bool, "ts": ..}

本模块是运行期（runtime/loop.py 接线）的双写登记入口——纯 stdlib，
单源真相优先：``tools.migrate_m7`` 可导入时直接复用其
``dual_write_record`` / ``dual_write_check``（journal schema 与 tools
逐字节同构）；tools 不可导入（如无 torch 环境）时用**完全相同** schema
的 stdlib 重实现兜底——保证 tools 的 check CLI 永远可读本模块写的 journal。

env 化默认值（M8 约定——本模块 docstring 为运行时侧唯一权威；tools 侧
默认值以 tools/migrate_m7.py docstring 为准）::

    LMS_M7_DUAL_WRITE_ROUNDS   双写回滚期轮数 N（整型，>0 → 开；
                               缺省/≤0 → 关；默认 0 = 不启用）。
                               显式参数 > 环境变量。
    LMS_M7_DUAL_WRITE_JOURNAL  双写 journal 路径（本模块新增 env；
                               默认 data/migration/dual_write_journal.jsonl，
                               相对路径按项目根（本文件上级）解析为绝对路径；
                               tools 的 CLI 仍用 --journal 显式传参）。

公开 API::

    dual_write_enabled(explicit=None) -> bool
    journal_path_default() -> Path
    record_round(*, round_no, old_inc, new_inc, journal_path=None) -> List[dict]
    check_rounds(required_rounds, journal_path=None) -> List[dict]

CLI（退出码 0=通过 2=参数错误 3=校验失败）::

    # 从 rewrite-ws 项目根运行（与 runtime.cli 同款约定：`-m` 需要包可导入）；
    # 任意 CWD 可用绝对路径脚本方式调用：python <rewrite-ws>/runtime/m7_dual_write.py
    python -m runtime.m7_dual_write --record-round N --old-inc A --new-inc B [--journal PATH]
    python -m runtime.m7_dual_write --check --rounds N [--journal PATH]

fail-open 纪律（G 模式：登记/校验异常绝不抛——以 error issue 返回，
调用方（主循环）记日志继续；双写登记失败 = 该轮不可验证 = 闸门自然
不放行，语义安全）::

    - 非法 round/inc（不可整型强转）→ error issue，零写入；
    - round_no < 1（round 编号从 1 起）→ error issue，零写入；
    - journal 目录不可写 / 路径是目录 / 任何 IO 异常 → error issue；
    - check 的 required_rounds 非法 → error issue。

运行期接入点说明（loop.py 接线，父代理执行——本模块不修改 loop.py）::

    loop.py 的 commit 步（step 7）已计算本轮新核心写入增量::

        _epi_before = self.memory.episodic_size()          # 约 line 762
        ...
        self.memory.store_episodic(...)                    # 约 line 766
        _epi_delta = self.memory.episodic_size() - _epi_before   # line 794

    ★ round 编号：`self.turn_count` 在 **emit 步**（_increment_turn，
    line 909）才 +1 —— 若在 commit 步接线，首轮 turn_count=0。因此推荐
    **在 emit 步（line 909 之后）接线**，round_no == self.turn_count
    （≥1）；若坚持在 commit 步接线，须传 round_no = self.turn_count + 1。

    ★ new_inc（新核心本轮增量）: `_epi_delta`（commit 步 line 794，
    `self.memory.episodic_size() - _epi_before`——store_episodic 可能被
    垃圾过滤跳过，条目数差分是唯一可靠口径，与 doubt/目的时相同款）。

    ★ old_inc（旧核心本轮增量）: 双写期旧存储与 new 侧接收同一批写入；
    old_inc = 本轮写入旧存储的条目数（旧核心 store_episodic 的写入计数，
    或旧存储条目数差分）。生产旧存储环境 /tmp/repro3-1786556208/
    living-memory-system-cloud 当前无 live 数据（snapshots/main/ 为空、
    无 data/archive/）→ 旧侧计数须在真实部署处由接线提供；rewrite-ws
    内首轮以迁移产物口径实跑（见 data/migration/dual_write_journal.jsonl
    的 note 事件）。

    ★ 零参与保证（Q3）: env 未设/≤0 → dual_write_enabled() 为 False →
    接线处不调用 record_round → 零参与、零 IO；开关开时每轮参与。

    ★ 切单写闸门: 满 N 轮后 check_rounds(N)（或 CLI --check --rounds N）；
    issues 无 error → 切单写；任一 error（轮数不足/含不一致/登记失败）→
    禁止切单写，继续双写。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

# 项目根自举（与 tools/migrate_m7.py 同款）：保证 `python -m runtime.m7_dual_write`
# 与 `from runtime.m7_dual_write import ...` 从任意 CWD 可用。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# 常量 / env（单一默认值来源 = 本模块 docstring）
# ---------------------------------------------------------------------------

DUAL_WRITE_ROUNDS_ENV = "LMS_M7_DUAL_WRITE_ROUNDS"
JOURNAL_ENV = "LMS_M7_DUAL_WRITE_JOURNAL"
DEFAULT_JOURNAL = "data/migration/dual_write_journal.jsonl"

# tools 后端是否可用（单源真相：可用即复用 tools；否则 stdlib 兜底）
try:
    from tools.migrate_m7 import (  # noqa: E402
        dual_write_record as _tools_dual_write_record,
        dual_write_check as _tools_dual_write_check,
    )
    BACKEND = "tools.migrate_m7"
except Exception:  # noqa: BLE001 —— 无 torch 等环境：stdlib 兜底
    _tools_dual_write_record = None
    _tools_dual_write_check = None
    BACKEND = "stdlib-fallback"

# 与 tools/migrate_m7.py 同款级别
_ERR = "error"
_WARN = "warning"


# ---------------------------------------------------------------------------
# 开关解析
# ---------------------------------------------------------------------------

def dual_write_enabled(explicit=None) -> bool:
    """双写回滚期开关：显式参数 > 环境变量。

    规则: 值解析为整数后 ``>0`` → 开；缺省 / ``<=0`` / 非整型 → 关（保守
    ——避免误启双写导致意外 IO；env 开时每轮参与，关时零参与）。

    参数:
        explicit: 显式覆盖值（int/bool/str 均可；``None`` = 读
            ``LMS_M7_DUAL_WRITE_ROUNDS`` env）。bool 直接映射
            （True→开，False→关）；其余按整型解析。

    返回:
        bool：是否启用双写登记。
    """
    if explicit is None:
        raw = os.environ.get(DUAL_WRITE_ROUNDS_ENV, "").strip()
    elif isinstance(explicit, bool):
        return explicit
    else:
        raw = str(explicit).strip()
    if not raw:
        return False
    try:
        return int(raw) > 0
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# journal 路径
# ---------------------------------------------------------------------------

def journal_path_default() -> Path:
    """默认双写 journal 路径（绝对路径）。

    优先级: ``LMS_M7_DUAL_WRITE_JOURNAL`` env（若设）> 默认
    ``data/migration/dual_write_journal.jsonl``（项目根相对）。
    相对路径统一解析为绝对路径（项目根 = 本文件上级目录），与 CWD 无关
    ——运行期（loop 接线）与 CLI 行为一致、可复现。
    """
    raw = os.environ.get(JOURNAL_ENV, "").strip() or DEFAULT_JOURNAL
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def _resolve_journal(journal_path: Optional[str]) -> Path:
    """解析 journal 路径（None → journal_path_default()；相对 → 项目根绝对）。"""
    if journal_path is None:
        return journal_path_default()
    p = Path(str(journal_path))
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


# ---------------------------------------------------------------------------
# stdlib 兜底实现（schema 与 tools/migrate_m7.py 逐字节同构）
# ---------------------------------------------------------------------------

def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """原子写（tmp + os.replace）：读者永远看到完整文件（G 模式）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".m7dw_", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, str(path))
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _atomic_write_jsonl(path: Path, records: List[dict]) -> None:
    """原子写 JSONL（每行一个 JSON，ensure_ascii=False）。"""
    lines = "".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in records
    ).encode("utf-8")
    _atomic_write_bytes(path, lines)


def _read_journal_lines(journal_path: str) -> List[dict]:
    """读 journal 全部行（含非 round 事件，如 dual_write_start / note）。"""
    jp = Path(journal_path)
    if not jp.is_file():
        return []
    return [json.loads(l) for l in jp.read_text(
        encoding="utf-8").splitlines() if l.strip()]


def _fallback_record(journal_path: str, *, round_no: int,
                     old_inc: int, new_inc: int) -> List[dict]:
    """tools.dual_write_record 的同 schema stdlib 重实现（兜底路径）。"""
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
    existing = _read_journal_lines(journal_path)
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
    _atomic_write_jsonl(jp, existing + [rec])
    return issues


def _fallback_check(journal_path: str, required_rounds: int) -> List[dict]:
    """tools.dual_write_check 的同 schema stdlib 重实现（兜底路径）。"""
    issues: List[dict] = []
    jp = Path(journal_path)
    if not jp.is_file():
        issues.append({"level": _ERR, "msg": f"双写 journal 不存在: {journal_path}"})
        return issues
    rounds = [r for r in _read_journal_lines(journal_path)
              if r.get("event") == "round"]
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
# 公开 API：登记 / 校验
# ---------------------------------------------------------------------------

def record_round(*, round_no, old_inc, new_inc, journal_path=None) -> List[dict]:
    """登记一轮双写增量对比（委托 tools.dual_write_record 或同 schema 兜底）。

    语义:
        - 幂等: 同 round 重复登记 → error issue（"已记录"），零写入；
        - 原子追加: 读-改-写整文件（tmp + os.replace），读者永远看到完整
          journal；既有事件（dual_write_start / note）原样保留；
        - fail-open: 非法参数 / IO 异常 → error issue，**绝不抛**；
        - 增量一致（old_inc == new_inc）→ warning；不一致 → error
          （不满足切单写条件）。

    参数:
        round_no: 轮号（int 可强转，**必须 ≥ 1**；run 期建议 = emit 步后
            的 self.turn_count）。
        old_inc: 旧存储本轮增量（int 可强转）。
        new_inc: 新存储本轮增量（int 可强转）。
        journal_path: journal 路径（str/Path；None → journal_path_default()）。

    返回:
        List[dict] issues（与 tools 同构: {"level": "warning"|"error",
        "msg": str}）。
    """
    try:
        rn = int(round_no)
        oi = int(old_inc)
        ni = int(new_inc)
    except (TypeError, ValueError):
        return [{"level": _ERR,
                 "msg": f"非法 round/inc 参数（round_no={round_no!r} "
                        f"old_inc={old_inc!r} new_inc={new_inc!r}）——拒绝登记"}]
    if rn < 1:
        return [{"level": _ERR,
                 "msg": f"非法 round 号 {rn}（必须 ≥ 1）——拒绝登记"}]
    jp = _resolve_journal(journal_path)
    try:
        if _tools_dual_write_record is not None:
            return list(_tools_dual_write_record(
                str(jp), round_no=rn, old_inc=oi, new_inc=ni))
        return _fallback_record(str(jp), round_no=rn, old_inc=oi, new_inc=ni)
    except Exception as ex:  # noqa: BLE001
        return [{"level": _ERR,
                 "msg": f"record_round 异常（fail-open 不抛）: {ex}"}]


def check_rounds(required_rounds, journal_path=None) -> List[dict]:
    """双写收尾校验（委托 tools.dual_write_check 或同 schema 兜底）。

    语义: round 数达标且全部 match → warning（满足切单写条件）；轮数不足
    → error；含不一致轮 → error（禁止切单写）；journal 不存在 → error。
    fail-open: required_rounds 非法 / 异常 → error issue，不抛。

    参数:
        required_rounds: 闸门所需轮数 N（int 可强转；应为
            LMS_M7_DUAL_WRITE_ROUNDS 值）。
        journal_path: journal 路径（str/Path；None → journal_path_default()）。

    返回:
        List[dict] issues（与 tools 同构）。
    """
    try:
        rr = int(required_rounds)
    except (TypeError, ValueError):
        return [{"level": _ERR,
                 "msg": f"非法 required_rounds={required_rounds!r}"
                        f"（必须为整数）"}]
    jp = _resolve_journal(journal_path)
    try:
        if _tools_dual_write_check is not None:
            return list(_tools_dual_write_check(str(jp), required_rounds=rr))
        return _fallback_check(str(jp), required_rounds=rr)
    except Exception as ex:  # noqa: BLE001
        return [{"level": _ERR,
                 "msg": f"check_rounds 异常（fail-open 不抛）: {ex}"}]


# ---------------------------------------------------------------------------
# CLI（python -m runtime.m7_dual_write）
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """CLI 入口。退出码: 0=通过 2=参数错误 3=校验失败（issues 含 error）。"""
    ap = argparse.ArgumentParser(
        prog="python -m runtime.m7_dual_write",
        description="M7 双写回滚期运行时登记：--record-round 登记一轮；"
                    "--check 收尾校验（N 轮一致才可切单写）。")
    ap.add_argument("--record-round", type=int, default=None,
                    help="登记第 N 轮（需 --old-inc 与 --new-inc）")
    ap.add_argument("--old-inc", type=int, default=None,
                    help="旧存储本轮增量")
    ap.add_argument("--new-inc", type=int, default=None,
                    help="新存储本轮增量")
    ap.add_argument("--check", action="store_true",
                    help="执行双写收尾校验（需 --rounds）")
    ap.add_argument("--rounds", type=int, default=None,
                    help="校验所需轮数 N")
    ap.add_argument("--journal", default=None,
                    help=f"journal 路径（默认 {DEFAULT_JOURNAL}；"
                         f"env {JOURNAL_ENV} 可覆盖）")
    args = ap.parse_args(argv)

    if args.record_round is not None and args.check:
        print("[M7-DW] 参数错误：--record-round 与 --check 互斥", file=sys.stderr)
        return 2
    if args.record_round is not None:
        if args.old_inc is None or args.new_inc is None:
            print("[M7-DW] 参数错误：--record-round 需要 --old-inc 与 --new-inc",
                  file=sys.stderr)
            return 2
        issues = record_round(round_no=args.record_round,
                              old_inc=args.old_inc, new_inc=args.new_inc,
                              journal_path=args.journal)
    elif args.check:
        if args.rounds is None:
            print("[M7-DW] 参数错误：--check 需要 --rounds N", file=sys.stderr)
            return 2
        issues = check_rounds(args.rounds, journal_path=args.journal)
    else:
        print("[M7-DW] 参数错误：必须指定 --record-round ... 或 --check --rounds N",
              file=sys.stderr)
        return 2

    for i in issues:
        print(f"[M7-DW][{i['level']}] {i['msg']}")
    return 3 if any(i["level"] == _ERR for i in issues) else 0


if __name__ == "__main__":
    sys.exit(main())

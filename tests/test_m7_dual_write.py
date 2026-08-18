# -*- coding: utf-8 -*-
"""M7 双写回滚期运行时登记模块测试（runtime/m7_dual_write.py）。

覆盖（任务书 M7 双写 journal 首轮）:
  - 开关解析：env=0 / 缺省 / 30 / 非法值；显式参数 > env；bool 直映射
  - journal 默认路径（data/migration/ 约定 + env 覆盖 + 相对路径解析）
  - record：schema 与 tools 同构 / 幂等拒绝重复 / 增量一致·不一致 / 原子追加
  - 互通：本模块写的 journal 可被 tools.migrate_m7.dual_write_check 读取
    （双向）
  - check：轮数不足 / 全一致 / 含不一致 / journal 不存在
  - fail-open：非法 round/inc 不抛、round<1 拒绝、目录路径不抛
  - CLI：--record-round / --check 退出码与落盘
"""

import json
import os
from pathlib import Path

import pytest

from runtime import m7_dual_write as dw
from tools.migrate_m7 import dual_write_check as tools_dual_write_check
from tools.migrate_m7 import dual_write_record as tools_dual_write_record

JOURNAL_ENV = "LMS_M7_DUAL_WRITE_JOURNAL"
ROUNDS_ENV = "LMS_M7_DUAL_WRITE_ROUNDS"


def _round_events(path: Path) -> list:
    """读 journal 中全部 round 事件（与 tools.dual_write_check 同口径）。"""
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(
        encoding="utf-8").splitlines() if l.strip()
        and json.loads(l).get("event") == "round"]


# --------------------------------------------------------------------------- #
# 1) 开关解析（dual_write_enabled）
# --------------------------------------------------------------------------- #

class TestDualWriteEnabled:

    def test_env_missing_is_off(self, monkeypatch):
        monkeypatch.delenv(ROUNDS_ENV, raising=False)
        assert dw.dual_write_enabled() is False

    def test_env_zero_is_off(self, monkeypatch):
        monkeypatch.setenv(ROUNDS_ENV, "0")
        assert dw.dual_write_enabled() is False

    def test_env_negative_is_off(self, monkeypatch):
        monkeypatch.setenv(ROUNDS_ENV, "-5")
        assert dw.dual_write_enabled() is False

    def test_env_30_is_on(self, monkeypatch):
        monkeypatch.setenv(ROUNDS_ENV, "30")
        assert dw.dual_write_enabled() is True

    def test_env_invalid_is_off_conservative(self, monkeypatch):
        monkeypatch.setenv(ROUNDS_ENV, "abc")
        assert dw.dual_write_enabled() is False

    def test_explicit_overrides_env_on(self, monkeypatch):
        monkeypatch.setenv(ROUNDS_ENV, "30")
        assert dw.dual_write_enabled(explicit=0) is False

    def test_explicit_overrides_env_off(self, monkeypatch):
        monkeypatch.setenv(ROUNDS_ENV, "0")
        assert dw.dual_write_enabled(explicit=5) is True

    def test_explicit_bool_direct_mapping(self, monkeypatch):
        monkeypatch.setenv(ROUNDS_ENV, "0")
        assert dw.dual_write_enabled(explicit=True) is True
        assert dw.dual_write_enabled(explicit=False) is False

    def test_explicit_str_parsed(self, monkeypatch):
        monkeypatch.setenv(ROUNDS_ENV, "0")
        assert dw.dual_write_enabled(explicit="30") is True


# --------------------------------------------------------------------------- #
# 2) journal 默认路径
# --------------------------------------------------------------------------- #

class TestJournalPathDefault:

    def test_default_under_data_migration(self, monkeypatch):
        monkeypatch.delenv(JOURNAL_ENV, raising=False)
        p = dw.journal_path_default()
        assert p.is_absolute()
        assert p.name == "dual_write_journal.jsonl"
        assert p.parent.name == "migration"
        assert p == dw.PROJECT_ROOT / "data/migration/dual_write_journal.jsonl"

    def test_env_override(self, monkeypatch, tmp_path):
        target = tmp_path / "custom" / "journal.jsonl"
        monkeypatch.setenv(JOURNAL_ENV, str(target))
        assert dw.journal_path_default() == target

    def test_env_relative_resolved_to_project_root(self, monkeypatch):
        monkeypatch.setenv(JOURNAL_ENV, "data/migration/custom.jsonl")
        assert dw.journal_path_default() == \
            dw.PROJECT_ROOT / "data/migration/custom.jsonl"


# --------------------------------------------------------------------------- #
# 3) record_round：schema / 幂等 / 一致·不一致 / 原子追加
# --------------------------------------------------------------------------- #

class TestRecordRound:

    def test_record_schema_matches_tools(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        dw.record_round(round_no=1, old_inc=3, new_inc=3, journal_path=jp)
        rec = _round_events(jp)[0]
        assert set(rec.keys()) == {"event", "round", "old_increment",
                                   "new_increment", "match", "ts"}
        assert rec["event"] == "round"
        assert rec["round"] == 1
        assert rec["old_increment"] == 3
        assert rec["new_increment"] == 3
        assert rec["match"] is True

    def test_consistent_round_no_error(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        issues = dw.record_round(round_no=1, old_inc=3, new_inc=3, journal_path=jp)
        assert not any(i["level"] == "error" for i in issues)
        assert any(i["level"] == "warning" and "增量一致" in i["msg"]
                   for i in issues)

    def test_duplicate_round_rejected_idempotent(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        dw.record_round(round_no=1, old_inc=3, new_inc=3, journal_path=jp)
        issues = dw.record_round(round_no=1, old_inc=9, new_inc=9,
                                 journal_path=jp)
        assert any(i["level"] == "error" and "已记录" in i["msg"]
                   for i in issues)
        # journal 中 round 1 仍是首登值（拒绝重复 = 零修改）
        recs = _round_events(jp)
        assert len(recs) == 1
        assert recs[0]["old_increment"] == 3
        assert recs[0]["new_increment"] == 3

    def test_inconsistent_round_error(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        issues = dw.record_round(round_no=1, old_inc=3, new_inc=5, journal_path=jp)
        assert any(i["level"] == "error" and "不一致" in i["msg"]
                   for i in issues)
        # 不一致轮也真实落盘（可审计），match=false
        rec = _round_events(jp)[0]
        assert rec["match"] is False
        assert rec["old_increment"] == 3 and rec["new_increment"] == 5

    def test_atomic_append_preserves_prior_records(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        for rnd, a, b in [(1, 1, 1), (2, 2, 2), (3, 4, 4)]:
            dw.record_round(round_no=rnd, old_inc=a, new_inc=b, journal_path=jp)
        recs = _round_events(jp)
        assert [r["round"] for r in recs] == [1, 2, 3]
        assert all(r["match"] for r in recs)
        # 每行都是合法 JSON（无半行/截断——原子写）
        for line in jp.read_text(encoding="utf-8").splitlines():
            assert isinstance(json.loads(line), dict)

    def test_atomic_append_preserves_non_round_events(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        jp.write_text(
            json.dumps({"event": "note", "kind": "caliber", "msg": "x"}) + "\n",
            encoding="utf-8")
        dw.record_round(round_no=1, old_inc=1, new_inc=1, journal_path=jp)
        lines = [json.loads(l) for l in jp.read_text(
            encoding="utf-8").splitlines() if l.strip()]
        assert lines[0]["event"] == "note"          # 既有事件保留
        assert lines[1]["event"] == "round"
        assert lines[1]["round"] == 1

    def test_missing_parent_dir_created(self, tmp_path):
        jp = tmp_path / "a" / "b" / "c" / "dw.jsonl"
        issues = dw.record_round(round_no=1, old_inc=1, new_inc=1, journal_path=jp)
        assert not any(i["level"] == "error" for i in issues)
        assert jp.is_file()

    def test_explicit_journal_path_used(self, tmp_path, monkeypatch):
        # 默认路径指向 tmp（隔离工作区状态），显式路径必须生效且默认不落盘
        monkeypatch.setenv(JOURNAL_ENV, str(tmp_path / "default.jsonl"))
        explicit = tmp_path / "explicit.jsonl"
        dw.record_round(round_no=1, old_inc=1, new_inc=1, journal_path=explicit)
        assert explicit.is_file()
        assert not (tmp_path / "default.jsonl").exists()

    # ---- fail-open：非法 round/inc 不抛，零写入 ----

    def test_failopen_invalid_round_no_raises_nothing(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        issues = dw.record_round(round_no="abc", old_inc=1, new_inc=1,
                                 journal_path=jp)
        assert any(i["level"] == "error" and "非法" in i["msg"]
                   for i in issues)
        assert not jp.exists()

    def test_failopen_invalid_inc_raises_nothing(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        issues = dw.record_round(round_no=1, old_inc=None, new_inc="x",
                                 journal_path=jp)
        assert any(i["level"] == "error" for i in issues)
        assert not jp.exists()

    def test_failopen_round_zero_rejected(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        issues = dw.record_round(round_no=0, old_inc=1, new_inc=1, journal_path=jp)
        assert any(i["level"] == "error" and "≥ 1" in i["msg"]
                   for i in issues)
        assert not jp.exists()

    def test_failopen_journal_path_is_dir(self, tmp_path):
        issues = dw.record_round(round_no=1, old_inc=1, new_inc=1,
                                 journal_path=tmp_path)
        assert any(i["level"] == "error" and "fail-open" in i["msg"]
                   for i in issues)

    def test_failopen_unreadable_journal_line(self, tmp_path):
        # 手写一行坏 JSON —— record 应 fail-open 返回 error 而不抛
        jp = tmp_path / "dw.jsonl"
        jp.write_text("{bad json\n", encoding="utf-8")
        issues = dw.record_round(round_no=1, old_inc=1, new_inc=1, journal_path=jp)
        assert any(i["level"] == "error" for i in issues)


# --------------------------------------------------------------------------- #
# 4) check_rounds：轮数不足 / 全一致 / 含不一致 / 不存在
# --------------------------------------------------------------------------- #

class TestCheckRounds:

    def test_insufficient_rounds_error(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        dw.record_round(round_no=1, old_inc=1, new_inc=1, journal_path=jp)
        issues = dw.check_rounds(3, journal_path=jp)
        assert any(i["level"] == "error" and "轮数不足" in i["msg"]
                   for i in issues)

    def test_all_consistent_passes(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        for rnd in (1, 2):
            dw.record_round(round_no=rnd, old_inc=2, new_inc=2, journal_path=jp)
        issues = dw.check_rounds(2, journal_path=jp)
        assert not any(i["level"] == "error" for i in issues)
        assert any("切单写" in i["msg"] for i in issues)

    def test_any_mismatch_blocks_switch(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        dw.record_round(round_no=1, old_inc=1, new_inc=1, journal_path=jp)
        dw.record_round(round_no=2, old_inc=1, new_inc=3, journal_path=jp)
        issues = dw.check_rounds(2, journal_path=jp)
        assert any(i["level"] == "error" and "禁止切单写" in i["msg"]
                   for i in issues)

    def test_missing_journal_error(self, tmp_path):
        issues = dw.check_rounds(30, journal_path=tmp_path / "nope.jsonl")
        assert any(i["level"] == "error" and "不存在" in i["msg"]
                   for i in issues)

    def test_failopen_invalid_required_rounds(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        dw.record_round(round_no=1, old_inc=1, new_inc=1, journal_path=jp)
        issues = dw.check_rounds("abc", journal_path=jp)
        assert any(i["level"] == "error" and "非法" in i["msg"]
                   for i in issues)


# --------------------------------------------------------------------------- #
# 5) 与 tools.migrate_m7 互通（同一 journal 文件双向可读）
# --------------------------------------------------------------------------- #

class TestToolsInterop:

    def test_runtime_journal_readable_by_tools_check(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        dw.record_round(round_no=1, old_inc=3, new_inc=3, journal_path=jp)
        dw.record_round(round_no=2, old_inc=0, new_inc=0, journal_path=jp)
        # tools 的 check 直接读本模块写的 journal
        issues = tools_dual_write_check(str(jp), required_rounds=2)
        assert not any(i["level"] == "error" for i in issues)
        assert any("切单写" in i["msg"] for i in issues)

    def test_runtime_check_reads_tools_written_journal(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        tools_dual_write_record(str(jp), round_no=1, old_inc=3, new_inc=3)
        tools_dual_write_record(str(jp), round_no=2, old_inc=5, new_inc=5)
        issues = dw.check_rounds(2, journal_path=jp)
        assert not any(i["level"] == "error" for i in issues)
        assert any("切单写" in i["msg"] for i in issues)

    def test_mixed_tools_and_runtime_records_single_journal(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        tools_dual_write_record(str(jp), round_no=1, old_inc=2, new_inc=2)
        dw.record_round(round_no=2, old_inc=4, new_inc=4, journal_path=jp)
        tools_dual_write_record(str(jp), round_no=3, old_inc=1, new_inc=1)
        # tools check 全量可读且通过
        issues = tools_dual_write_check(str(jp), required_rounds=3)
        assert not any(i["level"] == "error" for i in issues)
        # runtime check 全量可读且通过
        issues = dw.check_rounds(3, journal_path=jp)
        assert not any(i["level"] == "error" for i in issues)

    def test_runtime_duplicate_rejection_seen_by_tools(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        dw.record_round(round_no=1, old_inc=2, new_inc=2, journal_path=jp)
        issues = tools_dual_write_record(str(jp), round_no=1, old_inc=9, new_inc=9)
        assert any(i["level"] == "error" and "已记录" in i["msg"]
                   for i in issues)
        assert len(_round_events(jp)) == 1


# --------------------------------------------------------------------------- #
# 6) CLI（python -m runtime.m7_dual_write）
# --------------------------------------------------------------------------- #

class TestCLI:

    def test_cli_record_round_writes_journal(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        code = dw.main(["--record-round", "1", "--old-inc", "3",
                        "--new-inc", "3", "--journal", str(jp)])
        assert code == 0
        recs = _round_events(jp)
        assert len(recs) == 1 and recs[0]["round"] == 1 and recs[0]["match"]

    def test_cli_check_ok_exit_zero(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        dw.record_round(round_no=1, old_inc=2, new_inc=2, journal_path=jp)
        code = dw.main(["--check", "--rounds", "1", "--journal", str(jp)])
        assert code == 0

    def test_cli_check_mismatch_exit_three(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        dw.record_round(round_no=1, old_inc=2, new_inc=9, journal_path=jp)
        code = dw.main(["--check", "--rounds", "1", "--journal", str(jp)])
        assert code == 3

    def test_cli_record_missing_incs_exit_two(self, tmp_path):
        code = dw.main(["--record-round", "1", "--old-inc", "3"])
        assert code == 2

    def test_cli_check_missing_rounds_exit_two(self, tmp_path):
        code = dw.main(["--check"])
        assert code == 2

    def test_cli_conflicting_modes_exit_two(self, tmp_path):
        code = dw.main(["--record-round", "1", "--old-inc", "1",
                        "--new-inc", "1", "--check"])
        assert code == 2

    def test_cli_no_mode_exit_two(self, tmp_path):
        code = dw.main([])
        assert code == 2

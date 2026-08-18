# -*- coding: utf-8 -*-
"""M7 单测：数据迁移（快照 + 归档合并去重，幂等可回滚可重跑）
（核心重建规格 v2 §三 / §6.2"迁移"用例组 / §7.1 M7）

覆盖（§6.2 迁移组 + 任务书四步）：
  1. 快照解析校验：条目数 / 字段完整性 / confidence∈[0,1] / turn 单调性
     （§3.2-1，校验错误 → 阻断迁移）
  2. 归档读取：坏行跳过（fail-open 行级）；archive_store 格式 + 旧导出
     格式双兼容
  3. 合并去重（§3.2-3/4）：快照∪归档按 (turn, text_hash)；条目数公式
     final = 快照 + 归档 - 去重数；快照条目优先；归档内部重复去重；
     turn 序列完整性
  4. schema 映射（§3.2-2）：rebuttal_consistency 初始化（写侧 'ingest'）、
     怀疑态 stable、violated_by 证伪语义保留为 superseded、source 原样
     保留含 store_gray（灰度归一 L1 不可召回口径）、confidence 钳制、
     last_reinforced_turn=写入 turn
  5. 回滚演练（BK-01 教训）：备份可读性校验（快照 load + 归档解析 + 计数
     对账）；损坏备份被检出；恢复演练 checksum 对账
  6. 中断重跑幂等：migrate 重跑输出一致（manifest 计数/checksum 稳定）；
     输出文件被截断（模拟中断）→ 重跑原子修复，不产生重复条目
  7. 迁移后校验（任务③）：新核心加载迁移快照 → recall 自命中 / 灰度不入
     external 检索面 / [doubt] 事件不入检索面 / 归档可检索 / 条目数不丢
  8. 双写回滚期 journal（§3.2-5）：增量一致记录 / 不一致 error /
     轮数不足禁止切单写

运行方式：pytest rewrite-ws/tests/test_migration_m7.py -v
（生产 venv：/tmp/repro3-1786556208/living-memory-system-cloud/.venv）。
"""

import json
import os
import sys

# 确保项目根目录在 Python 路径中（可从任意 cwd 运行）
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import pytest
import torch

from core.archive.archive_store import text_hash, vector_to_b64
from core.hippocampus.memory import EpisodicEntry
from tools.migrate_m7 import (
    backup_inputs,
    dual_write_check,
    dual_write_record,
    extract_episodic,
    load_snapshot_raw,
    map_entry_to_new,
    merge_entries,
    read_archive_records,
    rollback_drill,
    run_migrate,
    validate_archive_records,
    validate_snapshot,
    verify_migrated,
    write_migrated_snapshot,
)


# --------------------------------------------------------------------------- #
#  夹具：旧 v0.5.0 风格快照 + main.jsonl 归档
# --------------------------------------------------------------------------- #

def make_old_entry(text, turn, vec, source="external", confidence=1.0,
                   surprise=0.5, **kw):
    """旧 schema EpisodicEntry（仅 v1.3 之前字段——模拟旧快照条目）。"""
    e = EpisodicEntry.__new__(EpisodicEntry)
    e.text = text
    e.semantic_vector = vec
    e.surprise = surprise
    e.turn = turn
    e.source = source
    e.confidence = confidence
    e.rebuttal_count = 0
    e.reference_count = 0
    e.source_trust = 1.0
    e.labile = False
    e.labile_since = None
    e.violated_by = None
    e.last_recalled_at = None
    e.recall_count = 0
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def make_fixture_snapshot(path, entries, turn_count=629, num_nodes=16,
                          dim=8, version="0.5.0"):
    """构造 v0.5.0 风格快照文件（attractor/purpose/memory/tokenizer/meta/
    self_ref 齐备 + episodic_buffer）。"""
    st = {
        "version": version,
        "timestamp": 1786979736.0,
        "session_id": "main",
        "turn_count": turn_count,
        "last_entropy_ratio": 0.9888,
        "attractor": {
            "J": torch.randn(num_nodes, num_nodes) * 0.1,
            "bias": torch.zeros(num_nodes),
            "sigma": torch.full((num_nodes,), 0.5),
            "num_nodes": num_nodes,
            "input_dim": dim,
        },
        "purpose": {
            "precision": torch.full((dim,), 0.3469),
            "history": [torch.full((dim,), 0.3)],
            "coherence": 0.5,
            "encounter_count": torch.zeros(dim),
        },
        "memory": {
            "short_term_latent": torch.randn(num_nodes) * 0.1,
            "long_term_latent": torch.randn(num_nodes) * 0.1,
            "num_nodes": num_nodes,
            "buffer": [],
            "episodic_buffer": list(entries),
        },
        "tokenizer": {"vocab": {"<unk>": 0}},
        "meta": {"scale": 1.0},
        "self_ref": {"voice_history": []},
    }
    torch.save(st, str(path))
    return st


def make_archive(path, lines, bad_line=True):
    """写 main.jsonl 归档（archive_store 格式 + 旧导出格式 + 可选坏行）。"""
    with open(str(path), "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        if bad_line:
            f.write("{this is not json}\n")   # 坏行（行级 fail-open 跳过）
            f.write("\n")                      # 空行跳过
    return path


def archive_rec(text, turn, source="external", surprise=1.0,
                vec=None, legacy=False):
    """构造一行归档记录（默认 archive_store 格式；legacy=True 旧导出格式）。"""
    if legacy:
        return {"text": text, "turn": turn, "surprise": surprise,
                "source": source, "ts": 1786979736.0, "snapshot_file": "x.pt"}
    return {
        "version": 1, "session_id": "main", "turn": turn, "text": text,
        "source": source, "surprise": surprise,
        "text_hash": text_hash(text),
        "vector_b64": vector_to_b64(vec) if vec is not None else None,
        "exported_at": 1786979737.0, "origin": "archive",
    }


@pytest.fixture()
def fixture_pair(tmp_path):
    """标准夹具：6 条快照条目 + 6 行归档（含 2 行与快照重复、1 行归档内部
    重复、1 行旧导出格式、1 行坏行）。返回 (snapshot_path, archive_path, ctx)。"""
    dim = 8
    rng = torch.Generator().manual_seed(42)

    def vec():
        return torch.randn(dim, generator=rng)

    snap_entries = [
        make_old_entry("用户: 你好\n助手: 你好！", 600, vec()),
        make_old_entry("用户: 开启A功能\n助手: 已开启", 605, vec()),
        make_old_entry("[doubt] conflict: 应开启A vs 应关闭A", 610, vec()),
        make_old_entry("灰度测试条目", 615, vec(), source="store_gray"),
        make_old_entry("被证伪的旧记录", 620, vec(), violated_by="证据X"),
        make_old_entry("测试数据 我叫小明", 625, vec()),
    ]
    snap_path = tmp_path / "latest_main.pt"
    make_fixture_snapshot(snap_path, snap_entries)

    v630 = vec()
    arch_lines = [
        archive_rec("用户: 你好\n助手: 你好！", 600, vec=vec()),  # 与快照重复
        archive_rec("用户: 新归档记忆\n助手: 已记录", 630, vec=v630),
        archive_rec("用户: 旧导出格式行\n助手: 仅文本", 635, legacy=True),
        archive_rec("用户: 新归档记忆\n助手: 已记录", 630, vec=v630),  # 内部重复
        archive_rec("被证伪的旧记录", 620, vec=vec()),  # 与快照重复
    ]
    arch_path = tmp_path / "main.jsonl"
    make_archive(arch_path, arch_lines)

    ctx = {
        "snap_entries": snap_entries,
        "dim": dim,
        "snapshot": snap_path,
        "archive": arch_path,
        # 期望：快照唯一 6 + 归档唯一新增 2（630 有向量 / 635 无向量文本）
        #   - 重复：与快照 2 + 归档内部 1 = 3
        #   - 坏行 1（跳过）
        #   final_union = 6 + 5 - 3 = 8；final_episodic = 7（635 无向量仅归档）
        "expect_union": 8,
        "expect_episodic": 7,
        "expect_dup": 3,
        "expect_archive_valid": 5,
        "expect_archive_skipped": 1,
    }
    return snap_path, arch_path, ctx


# --------------------------------------------------------------------------- #
#  1) 快照校验（§3.2-1）
# --------------------------------------------------------------------------- #

class TestSnapshotValidation:

    def test_valid_snapshot_no_errors(self, fixture_pair):
        snap, arch, ctx = fixture_pair
        st, entries, issues = validate_snapshot(str(snap))
        errors = [i for i in issues if i["level"] == "error"]
        assert not errors, issues
        assert len(entries) == 6

    def test_confidence_out_of_range_is_error(self, tmp_path):
        entries = [make_old_entry("x", 1, torch.zeros(8), confidence=1.5)]
        snap = tmp_path / "s.pt"
        make_fixture_snapshot(snap, entries)
        _, _, issues = validate_snapshot(str(snap))
        assert any("confidence" in i["msg"] and i["level"] == "error"
                   for i in issues)

    def test_turn_monotonicity_violation_is_error(self, tmp_path):
        entries = [make_old_entry("a", 5, torch.zeros(8)),
                   make_old_entry("b", 3, torch.zeros(8))]
        snap = tmp_path / "s.pt"
        make_fixture_snapshot(snap, entries)
        _, _, issues = validate_snapshot(str(snap))
        assert any("单调" in i["msg"] and i["level"] == "error"
                   for i in issues)

    def test_missing_attractor_purpose_is_error(self, tmp_path):
        st = {"version": "0.5.0", "memory": {"episodic_buffer": []}}
        snap = tmp_path / "s.pt"
        torch.save(st, str(snap))
        _, _, issues = validate_snapshot(str(snap))
        assert any("attractor" in i["msg"] and i["level"] == "error"
                   for i in issues)

    def test_old_version_is_warning_only(self, tmp_path):
        entries = [make_old_entry("a", 1, torch.zeros(8))]
        snap = tmp_path / "s.pt"
        make_fixture_snapshot(snap, entries, version="0.4.0")
        _, _, issues = validate_snapshot(str(snap))
        assert not any(i["level"] == "error" for i in issues)
        assert any("0.4.0" in i["msg"] for i in issues)


# --------------------------------------------------------------------------- #
#  2) 归档读取（fail-open 行级 + 双格式兼容）
# --------------------------------------------------------------------------- #

class TestArchiveParsing:

    def test_bad_lines_skipped_counted(self, tmp_path):
        p = tmp_path / "a.jsonl"
        make_archive(p, [archive_rec("ok", 1, vec=torch.zeros(8))])
        records, skipped = read_archive_records(str(p))
        assert len(records) == 1
        assert skipped == 1  # 坏行

    def test_legacy_export_format_accepted(self, tmp_path):
        p = tmp_path / "a.jsonl"
        make_archive(p, [archive_rec("旧格式", 3, legacy=True)], bad_line=False)
        records, skipped = read_archive_records(str(p))
        assert len(records) == 1
        assert records[0]["text"] == "旧格式"

    def test_missing_archive_is_empty(self, tmp_path):
        records, skipped = read_archive_records(str(tmp_path / "nope.jsonl"))
        assert records == [] and skipped == 0

    def test_archive_field_validation(self):
        issues = validate_archive_records([
            {"text": "", "turn": 1},
            {"text": "ok", "turn": "bad"},
            {"text": "ok2", "turn": 2, "vector_b64": "!!!not-base64!!!"},
        ])
        errors = [i for i in issues if i["level"] == "error"]
        assert len(errors) == 2  # 空文本 + turn 非法


# --------------------------------------------------------------------------- #
#  3) 合并去重 + 条目数公式（§3.2-3/4）
# --------------------------------------------------------------------------- #

class TestMergeDedup:

    def test_merge_counts_and_formula(self, fixture_pair):
        snap, arch, ctx = fixture_pair
        st, entries, _ = validate_snapshot(str(snap))
        records, _ = read_archive_records(str(arch))
        stats = merge_entries(entries, records, session="main",
                              now_ts=1786979738.0)
        assert stats["snapshot_entries"] == 6
        assert stats["snapshot_unique"] == 6
        assert stats["archive_valid"] == ctx["expect_archive_valid"]
        assert stats["dup_with_snapshot"] == 2
        assert stats["dup_archive_internal"] == 1
        assert stats["dup_total"] == ctx["expect_dup"]
        assert stats["final_union"] == ctx["expect_union"]
        assert stats["final_episodic"] == ctx["expect_episodic"]
        assert stats["formula_ok"] is True

    def test_merge_formula_snapshot_wins(self, fixture_pair):
        """同 (turn, text_hash) 重复时快照条目优先（有向量）。"""
        snap, arch, ctx = fixture_pair
        st, entries, _ = validate_snapshot(str(snap))
        records, _ = read_archive_records(str(arch))
        stats = merge_entries(entries, records, session="main",
                              now_ts=1.0)
        by_turn = {e.turn: e for e in stats["merged_episodic"]}
        # turn=600 重复：快照版本 surprise=0.5 保留（归档版 surprise=1.0 被去）
        assert by_turn[600].surprise == 0.5
        # turn=620 重复：快照版本 violated_by 语义保留
        assert by_turn[620].violated_by == "证据X"

    def test_no_vector_record_stays_archive_only(self, fixture_pair):
        """旧导出格式（无向量）→ 仅进归档（文本记录），不进 episodic。"""
        snap, arch, ctx = fixture_pair
        st, entries, _ = validate_snapshot(str(snap))
        records, _ = read_archive_records(str(arch))
        stats = merge_entries(entries, records, session="main",
                              now_ts=1.0)
        turns_episodic = {e.turn for e in stats["merged_episodic"]}
        assert 635 not in turns_episodic          # 无向量 → 不进 episodic
        assert 635 in {r.get("turn") for r in stats["archive_out"]}
        arc635 = [r for r in stats["archive_out"] if r.get("turn") == 635][0]
        assert arc635.get("vector_b64") is None   # 文本记录（无向量）

    def test_merge_turn_sequence_integrity(self, fixture_pair):
        """输出按 turn 非降排序、无重复键（turn 序列完整性双重校验）。"""
        snap, arch, ctx = fixture_pair
        st, entries, _ = validate_snapshot(str(snap))
        records, _ = read_archive_records(str(arch))
        stats = merge_entries(entries, records, session="main", now_ts=1.0)
        merged = stats["merged_episodic"]
        turns = [e.turn for e in merged]
        assert turns == sorted(turns)
        keys = [(e.turn, text_hash(e.text)) for e in merged]
        assert len(set(keys)) == len(keys)


# --------------------------------------------------------------------------- #
#  4) schema 映射（§3.2-2）
# --------------------------------------------------------------------------- #

class TestSchemaMapping:

    def test_new_fields_initialized(self):
        e = make_old_entry("文本", 7, torch.zeros(8), confidence=0.8)
        new = map_entry_to_new(e, session="main", now_ts=123.0)
        assert new.doubt_state == "stable"          # 怀疑态初始化 stable
        assert new.last_reinforced_turn == 7        # wear 计时起点 = 写入 turn
        assert new.info_value == 0.0
        assert new.core is None
        assert new.gray is False
        assert new.ts is None
        rc = new.rebuttal_consistency
        assert rc["rebuttals"] == []
        assert rc["consistency"] == 0.0
        assert rc["updated_by"] == "ingest"         # 写侧时相（非 retrieval）
        assert new.confidence == 0.8

    def test_confidence_clamped(self):
        e = make_old_entry("文本", 1, torch.zeros(8), confidence=1.5)
        new = map_entry_to_new(e, session="main", now_ts=1.0)
        assert new.confidence == 1.0

    def test_store_gray_preserved_and_normalized(self):
        e = make_old_entry("灰度", 2, torch.zeros(8), source="store_gray")
        new = map_entry_to_new(e, session="main", now_ts=1.0)
        assert new.source == "store_gray"           # 原样保留
        assert new.gray is True                     # 灰度归一
        e2 = make_old_entry("灰度2", 3, torch.zeros(8), gray=True)
        new2 = map_entry_to_new(e2, session="main", now_ts=1.0)
        assert new2.source == "store_gray"          # gray 标记 → source 归一
        assert new2.gray is True

    def test_violated_by_preserves_superseded_semantics(self):
        e = make_old_entry("证伪", 4, torch.zeros(8), violated_by="证据")
        new = map_entry_to_new(e, session="main", now_ts=1.0)
        assert new.doubt_state == "superseded"      # 旧证伪记录保留原语义
        assert new.violated_by == "证据"

    def test_legacy_dict_and_tuple_entries(self):
        d = {"text": "dict条目", "surprise": 1.0, "turn": 9, "source": "external"}
        new = map_entry_to_new(d, session="main", now_ts=1.0)
        assert new.text == "dict条目" and new.turn == 9
        assert new.doubt_state == "stable"


# --------------------------------------------------------------------------- #
#  5) 回滚演练（② + BK-01 教训：备份本身可能损坏）
# --------------------------------------------------------------------------- #

class TestBackupAndRollback:

    def test_backup_readable_verified(self, tmp_path, fixture_pair):
        snap, arch, ctx = fixture_pair
        bdir = tmp_path / "backups"
        paths, issues = backup_inputs(str(snap), str(arch), str(bdir))
        assert not any(i["level"] == "error" for i in issues), issues
        meta = json.loads((bdir / "backup_meta.json").read_text("utf-8"))
        assert meta["readable_verified"] is True
        # 备份快照可 load 且条目数一致
        st = load_snapshot_raw(str(bdir / "latest_main.pt"))
        assert len(extract_episodic(st)) == 6

    def test_corrupted_backup_detected(self, tmp_path, fixture_pair):
        """BK-01：备份损坏必须被检出（截断备份快照 → rollback-drill 报错）。"""
        snap, arch, ctx = fixture_pair
        bdir = tmp_path / "backups"
        backup_inputs(str(snap), str(arch), str(bdir))
        # 模拟损坏：截断备份快照
        with open(bdir / "latest_main.pt", "rb") as f:
            data = f.read()
        with open(bdir / "latest_main.pt", "wb") as f:
            f.write(data[: len(data) // 2])
        issues = rollback_drill(str(bdir))
        assert any(i["level"] == "error" and "快照" in i["msg"]
                   for i in issues)

    def test_corrupted_archive_backup_detected(self, tmp_path, fixture_pair):
        snap, arch, ctx = fixture_pair
        bdir = tmp_path / "backups"
        backup_inputs(str(snap), str(arch), str(bdir))
        with open(bdir / "main.jsonl", "w", encoding="utf-8") as f:
            f.write("{broken\n")  # 全部坏行 → 行数对账失败
        issues = rollback_drill(str(bdir))
        assert any(i["level"] == "error" and "归档" in i["msg"]
                   for i in issues)

    def test_restore_checksum_reconcile(self, tmp_path, fixture_pair):
        snap, arch, ctx = fixture_pair
        bdir = tmp_path / "backups"
        backup_inputs(str(snap), str(arch), str(bdir))
        rest = tmp_path / "restore"
        issues = rollback_drill(str(bdir), restore_to=str(rest))
        assert not any(i["level"] == "error" for i in issues), issues
        # 恢复文件与原输入逐字节一致
        import hashlib

        def sha(p):
            return hashlib.sha256(open(str(p), "rb").read()).hexdigest()
        assert sha(rest / "latest_main.pt") == sha(snap)
        assert sha(rest / "main.jsonl") == sha(arch)


# --------------------------------------------------------------------------- #
#  6) 中断重跑幂等（③/§3.3：迁移脚本本身幂等）
# --------------------------------------------------------------------------- #

class TestIdempotency:

    def _migrate(self, out_dir, fixture_pair, **kw):
        snap, arch, ctx = fixture_pair
        code, manifest = run_migrate(
            str(snap), str(arch), session="main", out_dir=str(out_dir),
            backup_dir=str(out_dir.parent / "backups"),
            apply=True, force=False, dual_write_rounds=0, **kw)
        assert code == 0, manifest
        return out_dir, manifest

    def test_rerun_produces_identical_outputs(self, tmp_path, fixture_pair):
        out1 = tmp_path / "m1"
        out2 = tmp_path / "m2"
        self._migrate(out1, fixture_pair)
        self._migrate(out2, fixture_pair)
        m1 = json.loads((out1 / "manifest.json").read_text("utf-8"))
        m2 = json.loads((out2 / "manifest.json").read_text("utf-8"))
        assert m1["merge"]["final_union"] == m2["merge"]["final_union"]
        assert m1["merge"]["final_episodic"] == m2["merge"]["final_episodic"]
        assert m1["outputs"]["snapshot"]["sha256"] == \
            m2["outputs"]["snapshot"]["sha256"]
        assert m1["outputs"]["archive"]["sha256"] == \
            m2["outputs"]["archive"]["sha256"]

    def test_interrupted_output_repaired_on_rerun(self, tmp_path, fixture_pair):
        """模拟中断：输出快照被截断 → 重跑原子修复，条目数不重复不丢失。"""
        out = tmp_path / "m"
        snap, arch, ctx = fixture_pair
        code, _ = run_migrate(str(snap), str(arch), session="main",
                              out_dir=str(out),
                              backup_dir=str(out.parent / "backups"),
                              apply=True, force=False, dual_write_rounds=0)
        assert code == 0
        out_snap = out / "migrated_latest_main.pt"
        # 模拟中断：截断输出快照
        data = out_snap.read_bytes()
        out_snap.write_bytes(data[: len(data) // 3])
        # 重跑（同一 out-dir）：应原子修复
        code2, _ = run_migrate(str(snap), str(arch), session="main",
                               out_dir=str(out),
                               backup_dir=str(out.parent / "backups"),
                               apply=True, force=False, dual_write_rounds=0)
        assert code2 == 0
        st = load_snapshot_raw(str(out_snap))
        merged = extract_episodic(st)
        assert len(merged) == ctx["expect_episodic"]   # 不重复不丢失
        keys = [(e.turn, text_hash(e.text)) for e in merged]
        assert len(set(keys)) == len(keys)

    def test_migrate_dry_run_writes_nothing(self, tmp_path, fixture_pair):
        snap, arch, ctx = fixture_pair
        out = tmp_path / "m"
        code, manifest = run_migrate(
            str(snap), str(arch), session="main", out_dir=str(out),
            backup_dir=str(out.parent / "backups"),
            apply=False, force=False, dual_write_rounds=0)
        assert code == 0
        assert manifest.get("dry_run") is True
        assert not out.exists()   # dry-run 不落盘

    def test_validation_error_blocks_migrate(self, tmp_path):
        """校验 error（confidence 超界）→ 默认阻断；--force 放行。"""
        entries = [make_old_entry("x", 1, torch.zeros(8), confidence=2.0)]
        snap = tmp_path / "s.pt"
        make_fixture_snapshot(snap, entries)
        arch = tmp_path / "a.jsonl"
        make_archive(arch, [archive_rec("y", 2, vec=torch.zeros(8))],
                     bad_line=False)
        out = tmp_path / "m"
        code, _ = run_migrate(str(snap), str(arch), session="main",
                              out_dir=str(out),
                              backup_dir=str(out.parent / "backups"),
                              apply=True, force=False, dual_write_rounds=0)
        assert code == 3
        assert not (out / "migrated_latest_main.pt").exists()
        # --force 显式放行
        code2, manifest = run_migrate(
            str(snap), str(arch), session="main", out_dir=str(out),
            backup_dir=str(out.parent / "backups"),
            apply=True, force=True, dual_write_rounds=0)
        assert code2 == 0


# --------------------------------------------------------------------------- #
#  7) 迁移后校验（任务③：recall 命中 / 不丢记忆）
# --------------------------------------------------------------------------- #

class TestPostMigrationVerify:

    def _migrate_and_verify(self, tmp_path, fixture_pair):
        snap, arch, ctx = fixture_pair
        out = tmp_path / "m"
        code, _ = run_migrate(str(snap), str(arch), session="main",
                              out_dir=str(out),
                              backup_dir=str(out.parent / "backups"),
                              apply=True, force=False, dual_write_rounds=0)
        assert code == 0
        out_snap = out / "migrated_latest_main.pt"
        out_arch = out / "main.jsonl"
        issues = verify_migrated(str(out_snap), str(out_arch), session="main",
                                 sample_k=10)
        return out, issues

    def test_recall_self_hit_no_memory_loss(self, tmp_path, fixture_pair):
        out, issues = self._migrate_and_verify(tmp_path, fixture_pair)
        assert not any(i["level"] == "error" for i in issues), issues
        # 条目数不丢（episodic 7 = 快照 6 + 归档有向量 1）
        assert any("episodic = 7" in i["msg"] for i in issues)
        assert any("自命中" in i["msg"] and "miss 0" in i["msg"] for i in issues)

    def test_gray_and_doubt_excluded_from_recall(self, tmp_path, fixture_pair):
        out, issues = self._migrate_and_verify(tmp_path, fixture_pair)
        assert not any(i["level"] == "error" for i in issues), issues
        assert any("灰度条目 1 条" in i["msg"] and "泄漏 0" in i["msg"]
                   for i in issues)
        assert any("[doubt] 条目 1 条" in i["msg"] and "泄漏 0" in i["msg"]
                   for i in issues)

    def test_archive_queryable(self, tmp_path, fixture_pair):
        out, issues = self._migrate_and_verify(tmp_path, fixture_pair)
        assert not any(i["level"] == "error" for i in issues), issues
        assert any("归档条目数 = 8" in i["msg"] for i in issues)

    def test_verify_detects_empty_snapshot(self, tmp_path):
        """空迁移快照（不丢记忆违反）→ verify 报错。"""
        st = {
            "version": "0.5.0",
            "attractor": {"num_nodes": 8},
            "purpose": {},
            "memory": {"num_nodes": 8,
                       "short_term_latent": torch.zeros(8),
                       "long_term_latent": torch.zeros(8),
                       "buffer": [],
                       "episodic_buffer": []},
        }
        snap = tmp_path / "empty.pt"
        torch.save(st, str(snap))
        arch = tmp_path / "a.jsonl"
        make_archive(arch, [], bad_line=False)
        issues = verify_migrated(str(snap), str(arch), session="main")
        assert any(i["level"] == "error" and "为空" in i["msg"]
                   for i in issues)


# --------------------------------------------------------------------------- #
#  8) 双写回滚期 journal（§3.2-5）
# --------------------------------------------------------------------------- #

class TestDualWriteJournal:

    def test_consistent_rounds_pass(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        issues = dual_write_record(str(jp), round_no=1, old_inc=3, new_inc=3)
        assert not any(i["level"] == "error" for i in issues)
        issues = dual_write_record(str(jp), round_no=2, old_inc=0, new_inc=0)
        assert not any(i["level"] == "error" for i in issues)
        check = dual_write_check(str(jp), required_rounds=2)
        assert not any(i["level"] == "error" for i in check)
        assert any("切单写" in i["msg"] for i in check)

    def test_mismatch_round_blocks_switch(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        issues = dual_write_record(str(jp), round_no=1, old_inc=3, new_inc=5)
        assert any(i["level"] == "error" and "不一致" in i["msg"]
                   for i in issues)
        check = dual_write_check(str(jp), required_rounds=1)
        assert any(i["level"] == "error" and "禁止切单写" in i["msg"]
                   for i in check)

    def test_insufficient_rounds_blocks_switch(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        dual_write_record(str(jp), round_no=1, old_inc=1, new_inc=1)
        check = dual_write_check(str(jp), required_rounds=3)
        assert any(i["level"] == "error" and "轮数不足" in i["msg"]
                   for i in check)

    def test_duplicate_round_rejected(self, tmp_path):
        jp = tmp_path / "dw.jsonl"
        dual_write_record(str(jp), round_no=1, old_inc=1, new_inc=1)
        issues = dual_write_record(str(jp), round_no=1, old_inc=2, new_inc=2)
        assert any(i["level"] == "error" and "已记录" in i["msg"]
                   for i in issues)


# --------------------------------------------------------------------------- #
#  端到端：migrate 落盘产物可直接被新核心加载
# --------------------------------------------------------------------------- #

class TestEndToEnd:

    def test_migrated_snapshot_loadable_by_new_core(self, tmp_path,
                                                    fixture_pair):
        """迁移输出 → MemoryManager.set_state 恢复 + recall 可用（闭环）。"""
        snap, arch, ctx = fixture_pair
        out = tmp_path / "m"
        code, _ = run_migrate(str(snap), str(arch), session="main",
                              out_dir=str(out),
                              backup_dir=str(out.parent / "backups"),
                              apply=True, force=False, dual_write_rounds=0)
        assert code == 0
        st = load_snapshot_raw(str(out / "migrated_latest_main.pt"))
        assert st["session_id"] == "main"
        assert st["turn_count"] == 635          # max(629, 最大合并 turn=635)
        from core.hippocampus.memory import MemoryManager
        mm = MemoryManager(num_nodes=16)
        mm.set_state(st["memory"])
        assert mm.episodic_size() == ctx["expect_episodic"]
        # 抽查：以"你好"条目的向量检索，命中该条目
        target = [e for e in mm.iter_episodic()
                  if e.text == "用户: 你好\n助手: 你好！"][0]
        hits = mm.recall_episodic(target.semantic_vector, top_k=3,
                                  source_filter=None)
        assert any(h.text == target.text for h in hits)

"""归档存储层（T2.3 检索扩容）。

提供窗口外情景记忆的 JSONL 归档导出/检索/重建：
  - export_episodic: 快照时把 episodic 追加导出 data/archive/{session}.jsonl
  - query_archive:   numpy 余弦向量检索（origin="archive" 来源标记）
  - rebuild_archive: 扫描快照重建归档（tools/archive_job.py 使用）
"""

from core.archive.archive_store import (
    archive_path_for,
    count_archive,
    entry_to_record,
    export_episodic,
    get_archive_dir,
    query_archive,
    rebuild_archive,
    text_hash,
    vector_to_b64,
)

__all__ = [
    "archive_path_for",
    "count_archive",
    "entry_to_record",
    "export_episodic",
    "get_archive_dir",
    "query_archive",
    "rebuild_archive",
    "text_hash",
    "vector_to_b64",
]

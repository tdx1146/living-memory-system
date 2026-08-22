"""
活体记忆系统 - 归档存储层（T2.3 检索扩容）
==========================================

背景（P0-10 结构性根因）：``_episodic_buffer`` 是 ``deque(maxlen=200)`` 滚动窗口，
窗口外记忆永久不可检索——"存了但检索不到"。本模块为窗口外记忆提供**归档补充检索**：

  - 每次快照（``save_session_state``）时把 episodic 缓冲区追加导出到
    ``data/archive/{session}.jsonl``（按 (turn, text_hash) 去重）；
  - ``query_archive()`` 用 numpy 余弦相似度检索归档（当前数据量下全量扫描
    可接受；量大时再做分块/索引，见总体方案 T2.3 后续项）；
  - 返回条目显式携带 ``origin="archive"`` 来源标记。

设计护栏（设计原理核对报告 2026-08-10 §5 R1）：
  - **合并检索必须内存（活体）优先、归档补充、带来源标记**——本模块只负责
    "归档补充"这一半；合并顺序由调用方（runtime/loop.py ``recall_merged_readonly``）
    保证"内存 tier0 优先展示"，归档永远只是补充，绝不替代活体检索。

工程约束：
  - 纯 stdlib + numpy，不依赖 torch / persistence / runtime（保持 core 层纯净，
    依赖图无环：core <- persistence <- runtime；core/archive 只依赖自身层）；
  - fail-open：任何 IO/解析异常不允许拖垮调用方主流程，由调用方兜底；
  - 写路径用 fcntl 伴生锁串行化（与 persistence/snapshot 同款模式，此处独立
    实现以避免 core 反向依赖 persistence）；非 POSIX 平台无锁直写（fail-open）。

记录格式（每行一个 JSON 对象）::

    {
        "version": 1,
        "session_id": "main",
        "turn": 123,
        "text": "用户: ...\\n助手: ...",
        "source": "external",        # 'external' | 'self_ref'（沿用 Phase 2 来源标记）
        "surprise": 1.23,
        "text_hash": "sha256 前 16 位 hex（去重键之一）",
        "vector_b64": "base64(float32 小端)（语义向量，供 numpy 余弦检索）",
        "exported_at": 1723...,
        "origin": "archive"
    }
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from core.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

# 归档记录版本（将来字段变更时递增，读取端做向后兼容）
ARCHIVE_VERSION = 1

# P0 污染处置（2026-08-17）：系统事件不是对话——[doubt 前缀条目
# （[doubt] conflict 事件 / [doubt-supersedes] 证伪标记）不参与归档检索。
# 锚定行首：正文提及 [doubt 的真实条目不误伤。与
# core/hippocampus/memory.py 的 _DOUBT_EVENT_RE 同语义（各自模块内定义，
# 保持 core 层依赖无环）。
_DOUBT_EVENT_RE = re.compile(r"^\s*\[doubt(?:-[a-z]+)?\]", re.I)

# 归档根目录默认：项目根/data/archive（可用 LMS_ARCHIVE_DIR 环境变量覆盖）
_DEFAULT_ARCHIVE_DIR = PROJECT_ROOT / "data" / "archive"

# fcntl 锁超时（写路径串行化；与 persistence/snapshot 的 fail-open 语义一致：
# 锁超时跳过本次写，绝不阻塞主流程）
_LOCK_TIMEOUT = float(os.environ.get("LMS_ARCHIVE_LOCK_TIMEOUT", "5"))


# ======================================================================
# 路径与基础工具
# ======================================================================

def get_archive_dir() -> Path:
    """归档根目录：``LMS_ARCHIVE_DIR`` 环境变量 > 项目根/data/archive。"""
    env_dir = os.environ.get("LMS_ARCHIVE_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser()
    return _DEFAULT_ARCHIVE_DIR


def archive_path_for(session_id: str, archive_dir: Optional[str] = None) -> Path:
    """归档文件路径：data/archive/{session}.jsonl。"""
    sid = _sanitize_session_id(session_id)
    base = Path(archive_dir).expanduser() if archive_dir else get_archive_dir()
    return base / f"{sid}.jsonl"


def _sanitize_session_id(session_id) -> str:
    """把 session_id 清洗为文件系统安全字符集（与 persistence/snapshot 同规则）。"""
    import re
    s = str(session_id or "default").strip()
    s = re.sub(r"[^A-Za-z0-9_-]", "_", s)
    return s or "session"


def text_hash(text: str) -> str:
    """条目文本去重哈希：sha256 前 16 位 hex。"""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def vector_to_b64(vec) -> Optional[str]:
    """向量 -> base64(float32 小端) 字符串。

    兼容 torch.Tensor / numpy 数组 / 列表；失败返回 None（调用方跳过该条目，
    不影响其他条目——fail-open 粒度到条目级）。
    """
    try:
        if vec is None:
            return None
        if hasattr(vec, "detach"):  # torch.Tensor
            arr = vec.detach().cpu().float().numpy()
        else:
            arr = np.asarray(vec)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        return base64.b64encode(arr.tobytes()).decode("ascii")
    except Exception as e:  # pragma: no cover - 防御性
        logger.debug("向量编码失败（跳过该条目）: %s", e)
        return None


def b64_to_vector(b64: str) -> Optional[np.ndarray]:
    """base64(float32 小端) -> numpy 向量；失败返回 None。"""
    try:
        raw = base64.b64decode(b64)
        return np.frombuffer(raw, dtype=np.float32)
    except Exception as e:  # pragma: no cover - 防御性
        logger.debug("向量解码失败（跳过该条目）: %s", e)
        return None


def entry_to_record(session_id: str, entry, turn: Optional[int] = None,
                    text: Optional[str] = None,
                    text_hash_value: Optional[str] = None) -> Optional[dict]:
    """把一条情景记忆条目（EpisodicEntry 或等价 dict）转换为归档记录。

    统一了运行期导出（runtime/loop.py）与归档重建（tools/archive_job.py）
    的记录构造逻辑，保证两处格式完全一致。

    参数:
        session_id: 会话标识。
        entry: EpisodicEntry 数据类实例，或含相同字段的 dict。
        turn: 轮次覆盖（缺省时从 entry 读取）。
        text: 文本覆盖（缺省时从 entry 读取）。
        text_hash_value: 文本哈希覆盖（缺省时按 text 计算）。

    返回:
        归档记录 dict；条目无有效文本/向量时返回 None。
    """
    if isinstance(entry, dict):
        text = text if text is not None else entry.get("text")
        turn = turn if turn is not None else entry.get("turn")
        vec = entry.get("semantic_vector")
        source = entry.get("source", "external") or "external"
        surprise = float(entry.get("surprise", 0.0) or 0.0)
    else:
        text = text if text is not None else getattr(entry, "text", None)
        turn = turn if turn is not None else getattr(entry, "turn", None)
        vec = getattr(entry, "semantic_vector", None)
        source = getattr(entry, "source", "external") or "external"
        surprise = float(getattr(entry, "surprise", 0.0) or 0.0)

    if not text or not str(text).strip():
        return None
    try:
        turn_i = int(turn) if turn is not None else -1
    except (TypeError, ValueError):
        turn_i = -1
    h = text_hash_value or text_hash(str(text))
    vector_b64 = vector_to_b64(vec)
    if vector_b64 is None:
        # 无向量 = 无法参与向量检索；仍导出文本供将来重建/全文检索，
        # 检索侧会自动跳过无向量条目（fail-open 粒度到条目级）。
        logger.debug("条目无有效向量，仅导出文本: turn=%s hash=%s", turn_i, h)
    return {
        "version": ARCHIVE_VERSION,
        "session_id": str(session_id),
        "turn": turn_i,
        "text": str(text),
        "source": source,
        "surprise": surprise,
        "text_hash": h,
        "vector_b64": vector_b64,
        "exported_at": time.time(),
        "origin": "archive",
    }


# ======================================================================
# fcntl 伴生锁（POSIX；非 POSIX 无锁直写，fail-open）
# ======================================================================

try:
    import fcntl
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - 非 POSIX 平台
    fcntl = None
    _HAVE_FCNTL = False


def _lock_path_for(path: Path) -> str:
    return str(path) + ".lock"


def _acquire_lock(lock_path: str, exclusive: bool = True,
                  timeout: float = _LOCK_TIMEOUT):
    """获取伴生文件锁（与 persistence/snapshot 同款模式：超时返回 None）。"""
    if not _HAVE_FCNTL:
        return None
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    except OSError as e:  # pragma: no cover - 防御性
        logger.warning("归档锁文件创建失败（无锁直写，fail-open）: %s", e)
        return None
    op = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    deadline = time.time() + timeout
    while True:
        try:
            fcntl.flock(fd, op | fcntl.LOCK_NB)
            return fd
        except OSError:
            if time.time() >= deadline:
                os.close(fd)
                return None
            time.sleep(0.05)


def _release_lock(fd) -> None:
    """释放伴生文件锁（尽力而为，失败仅告警）。"""
    if fd is None:
        return
    try:
        if _HAVE_FCNTL:
            fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError as e:  # pragma: no cover - 防御性
        logger.warning("归档锁释放失败（忽略）: %s", e)
    os.close(fd)


# ======================================================================
# 读写
# ======================================================================

def _iter_records(path: Path):
    """逐行读取归档 JSONL（坏行跳过，fail-open 粒度到行级）。"""
    if not path.exists():
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    logger.debug("归档坏行跳过: %s", line[:80])
                    continue
                if isinstance(rec, dict) and rec.get("text"):
                    yield rec
    except OSError as e:
        logger.warning("归档读取失败（视为空归档，fail-open）: %s", e)


def _load_keys(path: Path) -> set:
    """读取归档中已存在的去重键集合 {(turn, text_hash)}。"""
    keys = set()
    for rec in _iter_records(path):
        turn = rec.get("turn")
        h = rec.get("text_hash")
        if turn is not None and h:
            keys.add((int(turn), h))
    return keys


def export_episodic(session_id: str, entries: Iterable,
                    archive_dir: Optional[str] = None) -> int:
    """把 episodic 条目追加导出到归档文件（按 (turn, text_hash) 去重）。

    快照（``save_session_state``）落盘后调用；条目级 fail-open：
    单条构造失败不影响其他条目；IO 失败抛出由调用方兜底（快照主流程不受影响）。

    参数:
        session_id: 会话标识。
        entries: EpisodicEntry 可迭代对象（通常来自 memory.iter_episodic()）。
        archive_dir: 归档目录覆盖（缺省用 get_archive_dir()）。

    返回:
        本次实际新增的条目数。
    """
    path = archive_path_for(session_id, archive_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd = _acquire_lock(_lock_path_for(path), exclusive=True)
    try:
        keys = _load_keys(path)
        new_records = []
        for entry in entries:
            rec = entry_to_record(session_id, entry)
            if rec is None:
                continue
            key = (rec["turn"], rec["text_hash"])
            if key in keys:
                continue
            keys.add(key)
            new_records.append(rec)
        if new_records:
            # 追加写：单次 write 一行，避免与重建（原子替换）之外的并发写者交错
            with open(path, "a", encoding="utf-8") as f:
                for rec in new_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(
                "[%s] 归档导出 +%d 条（累计 %d 条）-> %s",
                session_id, len(new_records), len(keys), path)
        return len(new_records)
    finally:
        _release_lock(fd)


def query_archive(session_id: str, query_vec, k: int = 5,
                  archive_dir: Optional[str] = None) -> List[dict]:
    """向量相似度检索归档（numpy 余弦，全量扫描）。

    参数:
        session_id: 会话标识（归档按会话分文件，天然按 session_id 过滤）。
        query_vec: 查询语义向量（torch.Tensor / numpy 数组；与存储向量同维）。
        k: 返回条数上限。
        archive_dir: 归档目录覆盖。

    返回:
        [{'origin': 'archive', 'session_id', 'turn', 'text', 'source',
          'surprise', 'score'}, ...]（按余弦相似度降序，最多 k 条）；
        归档不存在/为空时返回空列表。
    """
    path = archive_path_for(session_id, archive_dir)
    if not path.exists():
        return []

    # 查询向量归一化（与内存检索路径同语义：余弦相似度）
    q = _as_flat_float32(query_vec)
    if q is None or q.size == 0:
        return []
    q_norm = q / (np.linalg.norm(q) + 1e-9)
    qd = int(q.shape[0])

    scored: List[tuple] = []
    for rec in _iter_records(path):
        # P0 污染处置（2026-08-17）：[doubt 系统事件（conflict 事件 /
        # 证伪标记）不是对话，不参与归档检索——归档路径此前无任何
        # source/前缀过滤，[doubt 条目在合并检索（recall_merged_readonly）
        # 中满权重可见。锚定行首，正文提及 [doubt 的条目不误伤。
        if _DOUBT_EVENT_RE.match(rec.get("text", "") or ""):
            continue
        vb = rec.get("vector_b64")
        if not vb:
            continue  # 无向量条目无法参与向量检索（导出时已记录）
        v = b64_to_vector(vb)
        if v is None:
            continue
        if v.shape[0] != qd:
            # 维度不匹配（如旧快照 64 维 vs 新 1024 维）：跳过该组
            # （与内存检索 _recall_episodic_scored 的降级策略一致）
            continue
        v_norm = v / (np.linalg.norm(v) + 1e-9)
        score = float(np.dot(v_norm, q_norm))
        scored.append((score, rec))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, rec in scored[: max(1, int(k))]:
        results.append({
            "origin": "archive",
            "session_id": rec.get("session_id", str(session_id)),
            "turn": rec.get("turn"),
            "text": rec.get("text", ""),
            "source": rec.get("source", "external"),
            "surprise": rec.get("surprise"),
            "score": score,
        })
    return results


def rebuild_archive(session_id: str, records: Iterable[dict],
                    archive_dir: Optional[str] = None) -> int:
    """合并去重、只增不删地更新归档文件（原子替换，受伴生锁保护）。

    供 tools/archive_job.py 扫描快照后重建索引使用；按 (turn, text_hash)
    去重后整文件重写。

    [B10] 语义变更：旧实现用扫描到的 records **整体替换**归档文件 → 不在
    保留快照中的旧归档条目（窗口外记忆）被工具重新引入丢失。现改为合并：
    先读现有归档行并入 seen（保留），再追加新记录，只增不删。

    参数:
        session_id: 会话标识。
        records: 归档记录 dict 可迭代对象（entry_to_record 的输出）。
        archive_dir: 归档目录覆盖。

    返回:
        更新后归档中的唯一条目数。
    """
    path = archive_path_for(session_id, archive_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd = _acquire_lock(_lock_path_for(path), exclusive=True, timeout=10.0)
    try:
        # [B10] 先读现有归档行（坏行由 _iter_records 跳过），并入去重建 seen——
        # 窗口外旧归档条目保留，不再被整体替换丢失
        existing = list(_iter_records(path)) if path.exists() else []
        seen: set = set()
        uniq: List[dict] = []
        for rec in existing:
            turn = int(rec.get("turn", -1) or -1)
            h = rec.get("text_hash") or text_hash(str(rec.get("text", "")))
            key = (turn, h)
            if key not in seen:
                seen.add(key)
                uniq.append(rec)
        for rec in records:
            if not isinstance(rec, dict) or not rec.get("text"):
                continue
            turn = int(rec.get("turn", -1) or -1)
            h = rec.get("text_hash") or text_hash(str(rec.get("text", "")))
            key = (turn, h)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(rec)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in uniq:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, path)  # 原子替换
        logger.info("[%s] 归档合并完成：%d 条（含既有 %d 条）-> %s",
                    session_id, len(uniq), len(existing), path)
        return len(uniq)
    finally:
        _release_lock(fd)


def count_archive(session_id: str, archive_dir: Optional[str] = None) -> int:
    """统计归档条目数（坏行不计）。"""
    path = archive_path_for(session_id, archive_dir)
    return sum(1 for _ in _iter_records(path))


def _as_flat_float32(vec) -> Optional[np.ndarray]:
    """把 torch.Tensor / numpy 数组 / 列表统一为 1-D float32 numpy 数组。"""
    try:
        if vec is None:
            return None
        if hasattr(vec, "detach"):  # torch.Tensor
            arr = vec.detach().cpu().float().numpy()
        else:
            arr = np.asarray(vec)
        return np.asarray(arr, dtype=np.float32).reshape(-1)
    except Exception as e:  # pragma: no cover - 防御性
        logger.debug("查询向量解析失败: %s", e)
        return None

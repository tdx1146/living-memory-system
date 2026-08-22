"""吸引子景观快照

保存和加载吸引子网络状态（J矩阵、bias、sigma）和目的层状态（precision、history）。
格式：PyTorch的.pt文件（torch.save/load）。

遵循架构文档 5.5 节的接口定义。
保存内容：J矩阵、bias、sigma状态、precision向量、目的层历史。
可选保存：memory潜变量、tokenizer词表（N1+N2 修复）、meta元可塑性状态（0.3.0）、
        self_ref自指回路状态（0.4.0）。
包含版本号和时间戳。

版本历史:
    - 0.2.0: 增加 memory 和 tokenizer 可选字段（N1+N2 修复）。
    - 0.3.0: 增加 meta 元可塑性可选字段（向后兼容：旧快照无此字段时跳过）。
    - 0.4.0: 增加 self_ref 自指回路可选字段（向后兼容：旧快照无此字段时跳过）。
    - 0.5.0（阶段 1-A/P0-5+P0-4）:
        * 顶层新增可选元数据 session_id / turn_count / last_entropy_ratio
          （向后兼容：旧快照无这些字段也能正常 load）；
        * save()/load 全路径（含 _torch_load）加 fcntl.flock 伴生 `.lock` 文件锁
          —— 写排他锁（拿不到等待重试，超时告警跳过、不崩溃），读共享锁
          （超时告警后无锁直读，fail-open）。过渡期兜底，阶段 1-B 单写者收口；
        * 新增会话级路径助手：snapshot_path_for() / latest_path_for() /
          sanitize_session_id()，命名 `snapshot_{session}_{turn}_{ts}.pt` 存于
          `snapshots/{session}/` 子目录，每会话维护 `latest_{session}.pt`。
"""

import os
import re
import time
import shutil
import tempfile
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 快照格式版本号（N1+N2: 0.2.0 增加 memory 和 tokenizer 可选字段；
#                  0.3.0 增加 meta 元可塑性可选字段；
#                  0.4.0 增加 self_ref 自指回路可选字段；
#                  0.5.0 增加顶层元数据 session_id/turn_count/last_entropy_ratio
#                  + fcntl 伴生锁 + 会话级命名助手，见上方版本历史）
SNAPSHOT_VERSION = "0.5.0"

# ---------------------------------------------------------------------------
# fcntl 伴生锁（P0-4 过渡期兜底 / T1.2）
# ---------------------------------------------------------------------------
# 锁文件 = 目标快照路径 + ".lock"（伴生文件，不随快照原子替换而消失）。
# 写路径持排他锁（LOCK_EX），读路径持共享锁（LOCK_SH）：
#   - 多进程并发写同一快照时互相串行，杜绝"读旧写新"覆盖整段演化；
#   - 锁拿不到（如另一写者占用）→ 短间隔重试至超时 → 写超时告警并跳过保存
#     （fail-open：宁可这次不落盘，也不崩溃）；读超时告警后无锁直读
#     （原子替换保证读到的一定是完整文件，读侧风险极低）。
# 非 POSIX 平台（无 fcntl）自动降级为无锁，保持原有行为（fail-open）。
try:
    import fcntl
    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - 仅非 POSIX 平台
    fcntl = None  # type: ignore[assignment]
    _HAVE_FCNTL = False

# 写锁等待超时（秒，T1.2 要求 ~5s）与重试间隔
WRITE_LOCK_TIMEOUT = float(os.environ.get("LMS_SNAPSHOT_LOCK_TIMEOUT", "5.0"))
READ_LOCK_TIMEOUT = WRITE_LOCK_TIMEOUT
_LOCK_RETRY_INTERVAL = 0.05


def _lock_path_for(path: str) -> str:
    """返回快照路径的伴生锁文件路径（path + .lock）。"""
    return path + ".lock"


def _acquire_lock(lock_path: str, exclusive: bool, timeout: float) -> Optional[int]:
    """对伴生锁文件加 flock（非阻塞 + 重试至超时）。

    参数:
        lock_path: 锁文件路径。
        exclusive: True=排他锁（写），False=共享锁（读）。
        timeout: 总等待超时（秒）。

    返回:
        已加锁的文件描述符（持有期间必须保持打开）；超时返回 None。
        调用方负责 finally 中 _release_lock 释放。
    """
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    op = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    deadline = time.time() + timeout
    while True:
        try:
            fcntl.flock(fd, op | fcntl.LOCK_NB)
            return fd
        except (BlockingIOError, OSError):
            if time.time() >= deadline:
                os.close(fd)
                return None
            time.sleep(_LOCK_RETRY_INTERVAL)


def _release_lock(fd: int) -> None:
    """释放 flock 并关闭锁文件描述符（尽力而为，失败仅告警）。"""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError as e:  # pragma: no cover - 极边缘场景
        logger.warning(f"释放快照锁失败（忽略）: {e}")
    os.close(fd)


# ---------------------------------------------------------------------------
# 会话级快照路径助手（P0-5 / T1.1 新命名规范）
# ---------------------------------------------------------------------------
# 新命名：snapshots/{session}/snapshot_{session}_{turn}_{ts}.pt
# 最新指针：snapshots/{session}/latest_{session}.pt
# 旧平铺文件（latest.pt / snapshot_{n}.pt / {sid}_{ts}.pt）继续可读，不迁移。


def sanitize_session_id(session_id) -> str:
    """把 session_id 清洗为文件系统安全字符集（防路径穿越/特殊字符）。

    与 api/server.py P0-7 的清洗规则保持一致：仅保留字母数字、下划线、连字符；
    空值回退 "default"，全非法字符回退 "session"。
    """
    s = str(session_id or "default").strip()
    s = re.sub(r"[^A-Za-z0-9_-]", "_", s)
    return s or "session"


def snapshot_path_for(snapshot_dir: str, session_id: str,
                      turn_count: int, ts: Optional[str] = None) -> str:
    """生成轮次快照路径：snapshots/{session}/snapshot_{session}_{turn}_{ts}.pt。

    参数:
        snapshot_dir: 快照根目录。
        session_id: 会话标识（自动清洗）。
        turn_count: 当前轮次（写入文件名，保证按轮次归档）。
        ts: 时间戳字符串（默认当前时间 %Y%m%d_%H%M%S）。

    返回:
        完整快照文件路径。
    """
    sid = sanitize_session_id(session_id)
    ts = ts or time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(snapshot_dir, sid,
                        f"snapshot_{sid}_{int(turn_count)}_{ts}.pt")


def latest_path_for(snapshot_dir: str, session_id: str) -> str:
    """生成会话最新快照路径：snapshots/{session}/latest_{session}.pt。

    写新快照时同步更新该文件，作为该会话"最新状态指针"。
    """
    sid = sanitize_session_id(session_id)
    return os.path.join(snapshot_dir, sid, f"latest_{sid}.pt")


# ---------------------------------------------------------------------------
# 快照修剪（T1.4 补 / P2-8：prune keep=20，防止快照目录无限增长）
# ---------------------------------------------------------------------------
# 只匹配新命名规范 `snapshot_{session}_{turn}_{ts}.pt`（ts = %Y%m%d_%H%M%S）。
# 旧版扁平文件（snapshot_NNN.pt / latest.pt 等散落在快照根目录）不在此正则
# 范围内，由历史/迁移流程另行处理，prune 一律不动。
_SNAPSHOT_RE = re.compile(
    r"^snapshot_(?P<sid>[A-Za-z0-9_-]+)_(?P<turn>\d+)_(?P<ts>\d{8}_\d{6})\.pt$"
)


def _prune_keep_default() -> int:
    """读取 LMS_SNAPSHOT_PRUNE_KEEP 环境变量（默认 20，非法值回退默认）。"""
    raw = os.environ.get("LMS_SNAPSHOT_PRUNE_KEEP", "").strip()
    if raw:
        try:
            v = int(raw)
            if v >= 1:
                return v
        except ValueError:
            pass
        logger.warning(f"LMS_SNAPSHOT_PRUNE_KEEP 非法（{raw!r}），使用默认 20")
    return 20


def prune_snapshots(snapshot_dir: str, keep: Optional[int] = None) -> dict:
    """按会话子目录修剪轮次快照，每个会话只保留最近 keep 个（T1.4 补 / P2-8）。

    规则:
      * 只处理 ``snapshots/{session}/`` 一级子目录中符合新命名规范
        ``snapshot_{session}_{turn}_{ts}.pt`` 的文件（正则匹配）；
      * 排序键 (ts, turn)：ts 为主序（文件名时间戳 %Y%m%d_%H%M%S，字典序即
        时间序），同 ts 再按 turn 递增；保留最近 keep 个，其余删除；
      * ``latest_{session}.pt``（会话"最新状态指针"）、``*.lock`` 伴生锁、
        根目录散落的旧版文件一律不删；
      * 删除前校验：仅删正则命中的 .pt 且非符号链接（防误删/防链接逃逸）；
        单文件删除失败仅告警（fail-open，不中断其余会话）；
      * 幂等：任意时刻可安全重跑；调整 keep 即可改变保留深度。

    参数:
        snapshot_dir: 快照根目录（如 ./snapshots）。
        keep: 每个会话保留的轮次快照数；None 时读 LMS_SNAPSHOT_PRUNE_KEEP
              （默认 20，<1 的非法值回退默认）。

    返回:
        {session: {"kept": int, "removed": int}} 统计字典；目录不存在/空返回 {}。
    """
    if keep is None:
        keep = _prune_keep_default()
    keep = max(1, int(keep))
    snapshot_dir = str(snapshot_dir)
    if not os.path.isdir(snapshot_dir):
        logger.warning(f"prune_snapshots: 快照目录不存在，跳过: {snapshot_dir}")
        return {}

    try:
        entries = sorted(os.listdir(snapshot_dir))
    except OSError as e:
        logger.warning(f"prune_snapshots: 读取快照目录失败 {snapshot_dir}: {e}")
        return {}

    stats: dict = {}
    for name in entries:
        sub = os.path.join(snapshot_dir, name)
        # 只处理一级子目录（跳过根目录散落文件与符号链接）
        if not os.path.isdir(sub) or os.path.islink(sub):
            continue
        try:
            files = [f for f in os.listdir(sub)
                     if _SNAPSHOT_RE.match(f)
                     and os.path.isfile(os.path.join(sub, f))]
        except OSError as e:
            logger.warning(f"prune_snapshots: 读取会话目录失败 {sub}: {e}")
            continue
        if not files:
            continue

        def _sort_key(fn: str) -> tuple:
            m = _SNAPSHOT_RE.match(fn)
            return (m.group("ts"), int(m.group("turn")))

        files.sort(key=_sort_key)
        drop = files[:-keep] if len(files) > keep else []
        removed = 0
        for fn in drop:
            path = os.path.join(sub, fn)
            if os.path.islink(path):
                continue  # 符号链接不删（防逃逸）
            try:
                os.remove(path)
                removed += 1
                logger.info(f"prune_snapshots: 删除旧快照 {path}")
            except OSError as e:
                logger.warning(f"prune_snapshots: 删除失败 {path}: {e}")
        stats[name] = {"kept": len(files) - removed, "removed": removed}
    return stats


class Snapshot:
    """吸引子景观快照管理器。

    保存"火种"（J矩阵 + precision状态），恢复时"重新点燃"。

    快照格式（v0.4.0）：
        {
            'version': str,           # 快照版本号
            'timestamp': float,       # 保存时间戳
            'attractor': dict,        # 吸引子景观状态
            'purpose': dict,          # 目的层状态
            'memory': dict,           # 记忆潜变量（可选，N1）
            'tokenizer': dict,        # 分词器词表（可选，N2）
            'meta': dict,             # 元可塑性状态（可选，0.3.0）
            'self_ref': dict,         # 自指回路状态（可选，0.4.0）
        }

    attractor字典包含：
        - J: 耦合矩阵 [num_nodes, num_nodes]
        - bias: 偏置向量 [num_nodes]
        - sigma: 当前激活状态 [num_nodes]
        - num_nodes: 节点数
        - input_dim: 输入维度

    purpose字典包含：
        - precision: precision向量 [input_dim]
        - history: precision历史列表
        - coherence: 一致性值
        - encounter_count: 习惯化计数器 [input_dim]（N3，可选）

    memory字典包含（N1）：
        - short_term_latent: 短时潜变量 [num_nodes]
        - long_term_latent: 长时潜变量 [num_nodes]
        - num_nodes: 节点数

    tokenizer字典包含（N2）：
        - vocab: 词表字典 {token_str: token_id}

    meta字典包含（0.3.0，可选）：
        - 元可塑性层的自适应状态（如各学习规则参数的倍率、surprise历史等）。
          具体结构由 core.meta 层定义；persistence 层仅做透传，不解释其内容。
          旧版快照（v0.2.0）无此字段，加载时优雅跳过（向后兼容）。

    self_ref字典包含（0.4.0，可选）：
        - 自指回路的状态（如自述历史、延迟缓冲区、增益历史等）。
          具体结构由 core.hippocampus.self_referential 层定义；persistence 层
          仅做透传，不解释其内容。旧版快照（v0.3.0）无此字段，加载时优雅跳过
          （向后兼容）。

    接口设计说明（G5）:
        save()/load() 采用 **dict 接口** 而非直接接受 core 对象。
        这是有意的架构决策，目的是彻底解耦 persistence 与 core 层
        （架构约束：依赖图为无环DAG core <- persistence <- runtime，
        persistence 只负责序列化/反序列化，不做业务逻辑）。

        - 使用 dict 接口：persistence 无需在运行时导入 core 的具体类
          （如 AttractorNetwork / PurposeLayer），避免循环依赖与层间耦合。
          符合"高内聚 + 低耦合"原则。
        - 调用方负责对象 <-> dict 的转换：
            * attractor_landscape 由调用方通过
              AttractorNetwork.get_landscape() 获取后传入；
            * purpose_state 由调用方构造为包含 precision / history /
              coherence 键的 dict（可由 PurposeLayer.get_purpose()
              返回的 PurposeState 拆解得到）。
        - 架构文档 5.5 节虽描述为"接受对象"的接口，但 dict 实现是更优的
          解耦设计，本实现以 dict 为准，并在 persistence 层内部保持一致。
    """

    def save(self, path: str, attractor_landscape: dict,
             purpose_state: dict,
             memory_state: Optional[dict] = None,
             tokenizer_state: Optional[dict] = None,
             meta_state: Optional[dict] = None,
             self_ref_state: Optional[dict] = None,
             session_id: Optional[str] = None,
             turn_count: Optional[int] = None,
             last_entropy_ratio: Optional[float] = None,
             last_activation: Optional[dict] = None) -> bool:
        """保存吸引子景观和目的层状态到文件。

        0.5.0 新增：
          - 顶层元数据 session_id / turn_count / last_entropy_ratio（可选，
            向后兼容：旧快照无这些字段也能正常 load）；
          - fcntl 排他锁（伴生 .lock 文件）：拿不到锁时等待重试
            （默认 5s，LMS_SNAPSHOT_LOCK_TIMEOUT 可覆盖），超时记录告警
            并跳过保存（fail-open，不崩溃）；
          - 返回 bool：True=已保存，False=因写锁超时被跳过。

        参数:
            path: 保存路径（.pt文件）
            attractor_landscape: 吸引子景观状态字典，由调用方通过
                AttractorNetwork.get_landscape() 获取后传入。应包含键：
                J / bias / sigma（torch.Tensor）与 num_nodes / input_dim（int）。
            purpose_state: 目的层状态字典，由调用方构造。应包含键：
                precision（torch.Tensor）/ history（list[torch.Tensor]）/
                coherence（float）。可由 PurposeLayer.get_purpose() 返回的
                PurposeState 拆解得到。
            memory_state: 记忆潜变量状态字典（可选，N1）。由 MemoryManager
                的 get_state() 获取后传入。包含 short_term_latent /
                long_term_latent / num_nodes 键。为 None 时不保存此字段。
            tokenizer_state: 分词器词表字典（可选，N2）。由 SimpleTokenizer
                的 get_vocab() 获取后传入。为 None 时不保存此字段。
            meta_state: 元可塑性状态字典（可选，0.3.0）。由 core.meta 层的
                get_state() 获取后传入。persistence 层仅透传，不解释其内容。
                为 None 时不保存此字段。
            self_ref_state: 自指回路状态字典（可选，0.4.0）。由
                core.hippocampus.self_referential 层的 get_state() 获取后传入。
                persistence 层仅透传，不解释其内容。为 None 时不保存此字段。
            session_id: 会话标识（可选，0.5.0 顶层元数据）。
            turn_count: 当前轮次（可选，0.5.0 顶层元数据，供重启后连续）。
            last_entropy_ratio: 最近一轮熵比（可选，0.5.0 顶层元数据）。
            last_activation: 最近一轮激活态序列化 dict（可选，[A10]——内部
                参数，非端点签名）。为 None 时不保存该字段（旧快照无此键 →
                load 侧优雅跳过，向后兼容）。

        返回:
            True 表示已保存；False 表示因写锁超时被跳过（fail-open）。
        """
        data = {
            'version': SNAPSHOT_VERSION,
            'timestamp': time.time(),
            'attractor': attractor_landscape,
            'purpose': purpose_state,
        }

        # N1: 可选保存 memory 潜变量
        if memory_state is not None:
            data['memory'] = memory_state

        # N2: 可选保存 tokenizer 词表
        if tokenizer_state is not None:
            data['tokenizer'] = tokenizer_state

        # 0.3.0: 可选保存 meta 元可塑性状态
        if meta_state is not None:
            data['meta'] = meta_state

        # 0.4.0: 可选保存 self_ref 自指回路状态
        if self_ref_state is not None:
            data['self_ref'] = self_ref_state

        # 0.5.0: 顶层元数据（可选；旧快照无这些字段也能 load）
        if session_id is not None:
            data['session_id'] = str(session_id)
        if turn_count is not None:
            data['turn_count'] = int(turn_count)
        if last_entropy_ratio is not None:
            data['last_entropy_ratio'] = float(last_entropy_ratio)
        # [A10] 可选保存 last_activation（重启后 query_llm 首轮不再
        # "[无记忆]"；None 不写键 → 旧快照无该键，load 侧优雅跳过）。
        if last_activation is not None:
            data['last_activation'] = last_activation

        # 确保目录存在
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # 0.5.0: fcntl 排他锁（写路径串行化，P0-4 过渡期兜底）
        if _HAVE_FCNTL:
            lock_fd = _acquire_lock(
                _lock_path_for(path), exclusive=True,
                timeout=WRITE_LOCK_TIMEOUT)
            if lock_fd is None:
                # 写锁拿不到：等待重试已超时 → 告警并跳过保存（fail-open，不崩溃）
                logger.warning(
                    f"快照写锁超时（{WRITE_LOCK_TIMEOUT}s），本次保存跳过: {path}")
                return False
        else:
            lock_fd = None  # 非 POSIX 平台无 fcntl：无锁直写（fail-open）

        try:
            # 原子写入：先写临时文件，再原子替换，避免崩溃时截断原有快照
            fd, tmp_path = tempfile.mkstemp(
                prefix=".snap_", suffix=".tmp",
                dir=dir_path if dir_path else ".")
            try:
                with os.fdopen(fd, "wb") as f:
                    torch.save(data, f)  # 写到文件句柄而非路径
                os.replace(tmp_path, path)  # 原子替换
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
        finally:
            if lock_fd is not None:
                _release_lock(lock_fd)
        logger.info(f"快照已保存到 {path}")
        return True

    def save_copy(self, src_path: str, dst_path: str) -> bool:
        """把已保存的快照文件原子复制到另一路径（T1.1：同步 latest_{session}.pt）。

        与 save() 同样受 fcntl 排他锁保护（锁目标路径的伴生 .lock 文件）；
        写锁超时则告警跳过（fail-open，不崩溃）。复制为 tmp + os.replace，
        保证读者永远看到完整文件。

        参数:
            src_path: 源快照文件（已完整落盘）。
            dst_path: 目标路径（如 snapshots/{session}/latest_{session}.pt）。

        返回:
            True 表示已复制；False 表示因写锁超时被跳过。
        """
        dir_path = os.path.dirname(dst_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        if _HAVE_FCNTL:
            lock_fd = _acquire_lock(
                _lock_path_for(dst_path), exclusive=True,
                timeout=WRITE_LOCK_TIMEOUT)
            if lock_fd is None:
                logger.warning(
                    f"快照写锁超时（{WRITE_LOCK_TIMEOUT}s），同步 latest 跳过: "
                    f"{dst_path}")
                return False
        else:
            lock_fd = None

        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=".snap_", suffix=".tmp",
                dir=dir_path if dir_path else ".")
            try:
                with os.fdopen(fd, "wb") as f:
                    with open(src_path, "rb") as sf:
                        shutil.copyfileobj(sf, f)
                os.replace(tmp_path, dst_path)  # 原子替换
            except Exception:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise
        finally:
            if lock_fd is not None:
                _release_lock(lock_fd)
        logger.debug(f"快照已同步复制到 {dst_path}")
        return True

    def load(self, path: str) -> tuple[dict, dict]:
        """从文件加载吸引子景观和目的层状态。

        仅返回必需的 attractor / purpose 两个状态；可选字段
        （memory / tokenizer / meta）不在返回值中，需通过 load_raw()
        获取（与 N1/N2 既定模式一致；保持本方法二元组返回以兼容
        recovery.py 及现有测试的解包调用）。

        向后兼容：旧版快照（v0.3.0，无 self_ref 字段）可正常加载——
        本方法仅读取 attractor / purpose，不依赖任何可选字段。

        参数:
            path: 快照文件路径（.pt文件）

        返回:
            元组 (attractor_landscape, purpose_state)

        异常:
            FileNotFoundError: 文件不存在
            KeyError: 文件格式不正确
        """
        data = _torch_load(path)

        if 'attractor' not in data or 'purpose' not in data:
            raise KeyError("快照文件格式不正确：缺少attractor或purpose字段")

        version = data.get('version', 'unknown')
        logger.info(
            f"从 {path} 加载快照 "
            f"(版本: {version}, 时间: {data.get('timestamp', 'unknown')})"
        )

        # 可选字段（memory/tokenizer/meta/self_ref）不在此返回，由调用方通过
        # load_raw() 按需取用；旧快照无 meta/self_ref 字段时自动跳过（向后兼容）。
        return data['attractor'], data['purpose']

    def load_raw(self, path: str) -> dict:
        """加载完整快照数据（包括可选字段 memory/tokenizer/meta/self_ref）。

        供 runtime 层获取 persistence 层未直接恢复的可选状态
        （如 tokenizer 词表——N2: tokenizer 在 runtime 层直接处理；
        meta 元可塑性状态——0.3.0: 由 core.meta 层通过
        raw_data.get('meta') 取用，旧快照无此字段时返回 None；
        self_ref 自指回路状态——0.4.0: 由 runtime 层通过
        raw_data.get('self_ref') 取用，旧快照无此字段时返回 None）。

        参数:
            path: 快照文件路径

        返回:
            完整的快照数据字典
        """
        return _torch_load(path)

    def get_metadata(self, path: str) -> dict:
        """获取快照元数据（不加载完整状态）。

        参数:
            path: 快照文件路径

        返回:
            元数据字典，包含version和timestamp
        """
        data = _torch_load(path)
        return {
            'version': data.get('version', 'unknown'),
            'timestamp': data.get('timestamp', 0),
            # [B3] 追加顶层 session_id：/restore 归属校验需要（轻量读顶层键，
            # 避免整包 load；旧快照无该键 → None，调用方放行向后兼容）
            'session_id': data.get('session_id'),
        }


def _torch_load(path: str) -> dict:
    """兼容不同PyTorch版本的torch.load封装（0.5.0 起带共享读锁）。

    PyTorch 2.6+ 默认 weights_only=True，需要显式设置为False
    以加载包含Python对象的字典。

    0.5.0/T1.2：读取前对伴生 .lock 文件加 flock 共享锁（与写锁互斥），
    保证读到的不是写者正在替换中的半截文件；锁超时告警后无锁直读
    （fail-open——原子替换已保证读到的一定是完整文件）。

    参数:
        path: 文件路径

    返回:
        加载的数据字典
    """
    if _HAVE_FCNTL:
        lock_fd = _acquire_lock(
            _lock_path_for(path), exclusive=False, timeout=READ_LOCK_TIMEOUT)
        if lock_fd is None:
            logger.warning(
                f"快照读锁超时（{READ_LOCK_TIMEOUT}s），无锁直读: {path}")
    else:
        lock_fd = None  # 非 POSIX 平台：无锁直读（fail-open）
    try:
        try:
            return torch.load(path, weights_only=False)
        except TypeError:
            # 旧版本PyTorch不支持weights_only参数
            return torch.load(path)
    finally:
        if lock_fd is not None:
            _release_lock(lock_fd)

"""活体记忆系统 - 会话管理器

维护多个会话（session）的 LivingMemoryLoop 实例，使不同 TRAE/OpenClaw
对话窗口拥有相互独立的记忆状态。每个 session_id 对应一个独立的"大脑"。

设计要点:
    - 线程安全：内部加锁，因为 torch 推断非 async，多请求可能并发。
    - 惰性创建：首次访问 session 时才初始化记忆环（加载预训练模型较重）。
    - 配置覆盖：get_or_create 支持传入自定义 config 覆盖默认配置。
"""

import logging
import os
import threading
from typing import Dict, List, Optional

from runtime.loop import LivingMemoryLoop
from api.config import get_api_config
from persistence.snapshot import sanitize_session_id, latest_path_for
from persistence.audit import audit

logger = logging.getLogger(__name__)


class SessionManager:
    """会话管理器：维护 {session_id: LivingMemoryLoop} 字典。

    属性:
        _sessions: session_id -> LivingMemoryLoop 映射。
        _configs:  session_id -> 该 session 使用的 config 字典（便于重建）。
        _lock:     读写锁，保证并发安全。
        _default_config: 默认配置工厂函数。
    """

    def __init__(self, default_config_factory=None) -> None:
        """初始化会话管理器。

        参数:
            default_config_factory: 返回默认配置字典的可调用对象。
                若为 None，则使用 api.config.get_api_config。
        """
        self._sessions: Dict[str, LivingMemoryLoop] = {}
        self._configs: Dict[str, dict] = {}
        self._lock = threading.RLock()
        # [B19] 会话级锁表：首访的 LivingMemoryLoop 构造 + torch.load 自动恢复
        # 只在会话级锁内执行，避免一个会话恢复阻塞所有会话的 get_or_create
        self._session_locks: Dict[str, threading.Lock] = {}
        self._default_config_factory = (
            default_config_factory or get_api_config
        )

    def get_or_create(
        self,
        session_id: str,
        config: Optional[dict] = None,
    ) -> LivingMemoryLoop:
        """获取或创建指定 session 的 LivingMemoryLoop。

        若 session 已存在则直接返回（忽略传入的 config，保持一致性）；
        若不存在则用传入 config（或默认 config）创建新实例。

        参数:
            session_id: 会话标识。
            config: 可选的配置覆盖。若为 None 则使用默认配置。

        返回:
            该 session 的 LivingMemoryLoop 实例。
        """
        if not session_id:
            # [B6] 空 session_id 静默兜底到 "default" 无日志 → 拼错字段的客户端
            # 流量全部污染 default 脑；补 WARNING 提升可见性（行为不变）
            logger.warning(
                "收到空 session_id，兜底为 'default'"
                "（客户端字段拼写错误会污染 default 脑）")
            session_id = "default"

        with self._lock:
            existing = self._sessions.get(session_id)
        if existing is not None:
            return existing

        # [B19] 会话级锁：首访的 LivingMemoryLoop 构造 + torch.load 自动恢复
        # （可能秒级）只阻塞同 sid 的并发首访，不阻塞其他会话（旧实现全程持
        # 全局 RLock → 一个会话恢复时所有会话的 get_or_create 全部被阻塞）。
        with self._session_lock(session_id):
            with self._lock:
                loop = self._sessions.get(session_id)
                if loop is not None:
                    return loop
                # 新建 session
                cfg = dict(config) if config else self._default_config_factory()
                # 注入 session_id：供 per-session 持久化使用
                # （如 self_voice 自述文件按会话隔离），对记忆环本体无副作用
                cfg['session_id'] = session_id
                logger.info(f"创建新会话: {session_id}")
                loop = LivingMemoryLoop(cfg)
            # 全局锁外执行自动恢复（仍在会话级锁内，只阻塞同 sid）：
            # 阶段1-A 补丁（P0-13 优雅停机的配套），进程重启后自动恢复会话快照
            self._try_auto_restore(loop, session_id)
            with self._lock:
                # [B19] 恢复完成后才注册可见（防并发访问读到半恢复状态；
                # 与旧实现"恢复完成后才返回"的语义一致；双检防极端路径）
                existing = self._sessions.get(session_id)
                if existing is not None:
                    return existing
                self._sessions[session_id] = loop
                self._configs[session_id] = cfg
                # T2.6：会话创建审计（只加日志，不改业务逻辑）
                audit("session_created", session_id=session_id,
                      turn_count=loop.turn_count)
            return loop

    def _session_lock(self, session_id: str) -> threading.Lock:
        """[B19] 会话级锁（get-or-create）：全局锁内 get-or-create 会话锁。"""
        with self._lock:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_id] = lock
            return lock

    def _try_auto_restore(self, loop: LivingMemoryLoop, session_id: str) -> None:
        """启动自动恢复：在候选快照中选择"最有内容"者加载。

        候选（按优先级）：
          1）snapshots/{session}/latest_{session}.pt（新命名规范）
          2）snapshots/latest.pt（存量旧格式，向后兼容）
        但"新规范存在"不能排除"旧格式更丰富"——优雅停机可能保存
        turn=0 的空新格式快照（首次启动即停机时），此时应回退到
        旧格式真实快照。策略：peek 各候选的 meta.turn_count，
        选择 turn 最大的恢复（内容最丰富的状态）。
        任何失败仅告警不阻断（fail-open，不因快照损坏拒绝服务）。
        """
        try:
            import torch
            snap_dir = loop._snapshot_dir_path()
            sid = sanitize_session_id(session_id)
            candidates = [
                latest_path_for(str(snap_dir), sid),   # 新规范 latest_{sid}.pt
                str(snap_dir / "latest.pt"),          # 旧格式根目录 latest.pt
            ]
            best_path: Optional[str] = None
            best_turn = -1
            for path in candidates:
                if not os.path.isfile(path):
                    continue
                try:
                    st = torch.load(path, map_location="cpu", weights_only=False)
                    # [B2] 归属校验：顶层/嵌套 session_id 非空且与当前会话不符 →
                    # 跳过该候选（旧实现把全局 latest.pt 当任意会话候选，load_state
                    # 对 session_id 不一致仅告警 → 跨会话污染/静默回退过期脑）；
                    # 旧快照无 session_id 字段 → 放行（向后兼容，保留旧格式恢复能力）
                    owner = st.get("session_id")
                    if not owner and isinstance(st.get("meta"), dict):
                        owner = st.get("meta", {}).get("session_id")
                    if owner and str(owner) != str(session_id):
                        logger.warning(
                            f"[{session_id}] 候选快照 {path} 归属会话 {owner}，"
                            "跳过（防跨会话污染）")
                        continue
                    turn = int((st.get("meta") or {}).get("turn_count", 0))
                    if turn > best_turn:
                        best_path, best_turn = path, turn
                except Exception as e:
                    logger.warning(
                        f"[{session_id}] 快照探测失败 {path}: {e}")
            if best_path is None:
                return
            try:
                loop.load_state(best_path)
                logger.info(
                    f"[{session_id}] 启动自动恢复快照: {best_path} "
                    f"(turn={loop.turn_count})")
                # T2.6：快照加载审计（自动恢复路径）
                audit("snapshot_loaded", session_id=session_id,
                      path=best_path, turn_count=loop.turn_count,
                      mode="auto_restore")
            except Exception as e:
                logger.warning(
                    f"[{session_id}] 快照恢复失败 {best_path}: {e}")
                # T2.6：关键错误审计（自动恢复失败）
                audit("critical_error", component="auto_restore",
                      session_id=session_id, path=best_path,
                      error=str(e)[:200])
        except Exception as e:
            logger.warning(f"[{session_id}] 自动恢复流程异常（跳过）: {e}")

    def get(self, session_id: str) -> Optional[LivingMemoryLoop]:
        """获取指定 session（不存在返回 None）。"""
        with self._lock:
            return self._sessions.get(session_id)

    def remove(self, session_id: str) -> bool:
        """删除指定 session。

        返回:
            True 若 session 存在并已删除；False 若不存在。
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                self._configs.pop(session_id, None)
                logger.info(f"已删除会话: {session_id}")
                # T2.6：会话删除审计（只加日志，不改业务逻辑）
                audit("session_deleted", session_id=session_id)
                return True
            return False

    def list_sessions(self) -> List[str]:
        """列出所有活跃的 session_id。"""
        with self._lock:
            return list(self._sessions.keys())

    def get_config(self, session_id: str) -> Optional[dict]:
        """获取指定 session 使用的配置字典。"""
        with self._lock:
            return self._configs.get(session_id)

    def get_status(self, session_id: str) -> Optional[dict]:
        """获取指定 session 的状态（基于 loop.get_status() 并补充情景缓冲区）。

        返回:
            状态字典；若 session 不存在返回 None。
        """
        loop = self.get(session_id)
        if loop is None:
            return None

        status = loop.get_status()
        # 补充情景记忆缓冲区大小
        try:
            status['episodic_buffer_size'] = loop.memory.episodic_size()
        except Exception:
            status['episodic_buffer_size'] = 0
        # 标记是否配置了 LLM
        status['llm_enabled'] = loop.bridge is not None
        return status

    def clear(self) -> int:
        """清空所有会话。

        返回:
            被清除的会话数量。
        """
        with self._lock:
            n = len(self._sessions)
            self._sessions.clear()
            self._configs.clear()
            logger.info(f"已清空所有会话（共 {n} 个）")
            return n

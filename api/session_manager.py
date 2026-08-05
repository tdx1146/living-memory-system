"""活体记忆系统 - 会话管理器

维护多个会话（session）的 LivingMemoryLoop 实例，使不同 TRAE/OpenClaw
对话窗口拥有相互独立的记忆状态。每个 session_id 对应一个独立的"大脑"。

设计要点:
    - 线程安全：内部加锁，因为 torch 推断非 async，多请求可能并发。
    - 惰性创建：首次访问 session 时才初始化记忆环（加载预训练模型较重）。
    - 配置覆盖：get_or_create 支持传入自定义 config 覆盖默认配置。
"""

import logging
import threading
from typing import Dict, List, Optional

from runtime.loop import LivingMemoryLoop
from api.config import get_api_config

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
            session_id = "default"

        with self._lock:
            if session_id not in self._sessions:
                # 新建 session
                cfg = dict(config) if config else self._default_config_factory()
                # 注入 session_id：供 per-session 持久化使用
                # （如 self_voice 自述文件按会话隔离），对记忆环本体无副作用
                cfg['session_id'] = session_id
                logger.info(f"创建新会话: {session_id}")
                loop = LivingMemoryLoop(cfg)
                self._sessions[session_id] = loop
                self._configs[session_id] = cfg
            return self._sessions[session_id]

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

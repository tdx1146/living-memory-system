"""活体记忆系统 - 做梦调度器（DreamScheduler）
================================================

常驻后台线程，监控会话空闲时间，在无对话时自动触发做梦引擎。

核心机制：
  1. 后台守护线程定期检查每个会话的最后活动时间
  2. 当空闲时间超过阈值（默认 30 秒），自动触发做梦周期
  3. 收到对话请求时立即暂停做梦（通过 _busy 标志协调）
  4. 做梦期间记忆系统被锁定，对话请求等待做梦完成或超时

设计依据：DREAM_ENGINE_DESIGN.md 第四节"常驻进程方案"
  - CPU：限制为 1 线程，做梦时 CPU 占用 < 5%
  - 做梦频率：无对话时每 idle_threshold 秒触发一次
  - 对话优先：收到对话请求时立即中断做梦

线程安全：
  - _lock: 保护 _sessions 字典的读写
  - _busy_lock: 做梦期间锁定，对话请求通过 acquire_conversation() 等待
"""

import time
import logging
import threading
from typing import Dict, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger("dream_scheduler")


@dataclass
class SessionActivity:
    """单个会话的活动状态跟踪。

    属性:
        session_id: 会话标识。
        last_active_ts: 最后一次对话活动的时间戳。
        dream_count: 累计做梦次数。
        last_dream_ts: 上次做梦的时间戳。
        last_dream_result: 上次做梦的统计结果。
    """
    session_id: str
    last_active_ts: float = field(default_factory=time.time)
    dream_count: int = 0
    last_dream_ts: float = 0.0
    last_dream_result: dict = field(default_factory=dict)


class DreamScheduler:
    """做梦调度器：后台守护线程，空闲时自动触发做梦。

    监控所有注册会话的空闲时间，当空闲超过阈值时自动触发该会话的
    DreamEngine 做梦周期。对话请求通过 acquire_conversation() 暂停做梦，
    release_conversation() 恢复做梦。

    属性:
        idle_threshold: 空闲触发阈值（秒，默认 30）。
        dream_steps: 每次自动做梦的步数（默认 20）。
        dream_full_cycle: 是否使用完整七阶段周期（默认 False，MVP 模式）。
        check_interval: 检查间隔（秒，默认 5）。
        max_dream_duration: 单次做梦最大时长（秒，默认 60）。
    """

    def __init__(
        self,
        get_loop_fn: Callable[[str], object],
        idle_threshold: float = 30.0,
        dream_steps: int = 20,
        dream_full_cycle: bool = False,
        check_interval: float = 5.0,
        max_dream_duration: float = 60.0,
    ) -> None:
        """初始化做梦调度器。

        参数:
            get_loop_fn: 通过 session_id 获取 LivingMemoryLoop 的回调函数。
                返回 None 时表示会话不存在。
            idle_threshold: 空闲多少秒后触发做梦（默认 30）。
            dream_steps: 每次自动做梦步数（默认 20）。
            dream_full_cycle: True 使用完整七阶段周期，False 使用 MVP。
            check_interval: 检查空闲的间隔（秒，默认 5）。
            max_dream_duration: 单次做梦最大时长（秒，默认 60）。
        """
        self._get_loop_fn = get_loop_fn
        self.idle_threshold = idle_threshold
        self.dream_steps = dream_steps
        self.dream_full_cycle = dream_full_cycle
        self.check_interval = check_interval
        self.max_dream_duration = max_dream_duration

        # 会话活动记录
        self._activities: Dict[str, SessionActivity] = {}
        self._lock = threading.RLock()

        # 做梦锁定：做梦期间持有，对话请求需要等待释放
        self._busy_lock = threading.Lock()
        self._is_dreaming = False
        self._dreaming_session: Optional[str] = None

        # 线程控制
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()

        logger.info(
            f"DreamScheduler 已初始化 "
            f"(idle={idle_threshold}s, steps={dream_steps}, "
            f"full_cycle={dream_full_cycle}, check={check_interval}s)"
        )

    # ------------------------------------------------------------------ #
    #  会话管理
    # ------------------------------------------------------------------ #

    def register_session(self, session_id: str) -> None:
        """注册一个会话到调度器监控。

        参数:
            session_id: 会话标识。
        """
        with self._lock:
            if session_id not in self._activities:
                self._activities[session_id] = SessionActivity(
                    session_id=session_id)
                logger.debug(f"已注册会话: {session_id}")

    def unregister_session(self, session_id: str) -> None:
        """从调度器移除会话。

        参数:
            session_id: 会话标识。
        """
        with self._lock:
            self._activities.pop(session_id, None)
            logger.debug(f"已移除会话: {session_id}")

    def touch(self, session_id: str) -> None:
        """更新会话的最后活动时间（对话发生时调用）。

        参数:
            session_id: 会话标识。
        """
        with self._lock:
            act = self._activities.get(session_id)
            if act is None:
                act = SessionActivity(session_id=session_id)
                self._activities[session_id] = act
            act.last_active_ts = time.time()

    # ------------------------------------------------------------------ #
    #  对话/做梦协调
    # ------------------------------------------------------------------ #

    def acquire_conversation(self, session_id: str, timeout: float = 10.0
                             ) -> bool:
        """获取对话权限（等待做梦完成）。

        对话请求在处理前调用此方法，确保做梦不会同时修改记忆状态。
        使用带超时的锁等待，避免长时间阻塞。

        参数:
            session_id: 会话标识（用于日志）。
            timeout: 等待超时（秒，默认 10）。

        返回:
            True 若成功获取（做梦已完成或超时），False 若超时。
        """
        acquired = self._busy_lock.acquire(timeout=timeout)
        if acquired:
            logger.debug(f"[{session_id}] 对话权限已获取")
        else:
            logger.warning(
                f"[{session_id}] 等待做梦完成超时({timeout}s)，强制继续")
        return acquired

    def release_conversation(self, session_id: str) -> None:
        """释放对话权限（对话结束后调用）。

        参数:
            session_id: 会话标识（用于日志）。
        """
        try:
            self._busy_lock.release()
            logger.debug(f"[{session_id}] 对话权限已释放")
        except RuntimeError:
            # 锁未被持有（可能 acquire 超时后仍调用了 release）
            pass

    def is_dreaming(self) -> bool:
        """返回当前是否正在做梦。"""
        return self._is_dreaming

    def get_dreaming_session(self) -> Optional[str]:
        """返回正在做梦的会话 ID（无则 None）。"""
        return self._dreaming_session

    # ------------------------------------------------------------------ #
    #  线程生命周期
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """启动做梦调度器后台线程。"""
        if self._running:
            logger.warning("DreamScheduler 已在运行")
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="DreamScheduler",
            daemon=True,
        )
        self._thread.start()
        logger.info("DreamScheduler 后台线程已启动")

    def stop(self, timeout: float = 5.0) -> None:
        """停止做梦调度器后台线程。

        参数:
            timeout: 等待线程结束的超时（秒）。
        """
        if not self._running:
            return

        self._running = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("DreamScheduler 后台线程已停止")

    # ------------------------------------------------------------------ #
    #  后台主循环
    # ------------------------------------------------------------------ #

    def _run_loop(self) -> None:
        """后台线程主循环：定期检查空闲并触发做梦。"""
        logger.info("DreamScheduler 主循环开始")

        while not self._stop_event.is_set():
            try:
                self._check_and_dream()
            except Exception as e:
                logger.error(f"做梦调度异常: {e}", exc_info=True)

            # 等待下一次检查（可被 stop 提前唤醒）
            self._stop_event.wait(self.check_interval)

        logger.info("DreamScheduler 主循环结束")

    def _check_and_dream(self) -> None:
        """检查所有会话的空闲状态，对空闲会话触发做梦。"""
        now = time.time()

        with self._lock:
            # 快照当前活动列表（避免长时间持锁）
            sessions_to_check = list(self._activities.values())

        for activity in sessions_to_check:
            if not self._running:
                break

            idle_time = now - activity.last_active_ts
            if idle_time < self.idle_threshold:
                continue  # 仍在活跃，跳过

            # 检查上次做梦到现在是否够一个周期
            if activity.last_dream_ts > 0:
                since_last_dream = now - activity.last_dream_ts
                if since_last_dream < self.idle_threshold:
                    continue  # 距上次做梦太近

            # 获取 loop 实例
            loop = self._get_loop_fn(activity.session_id)
            if loop is None:
                continue

            # 检查是否有记忆可回放
            try:
                buf_size = loop.memory.buffer_size()
            except Exception:
                buf_size = 0

            if buf_size == 0:
                continue  # 无记忆可回放

            # 尝试获取做梦锁（非阻塞：拿不到说明有对话正在进行）
            acquired = self._busy_lock.acquire(blocking=False)
            if not acquired:
                logger.debug(
                    f"[{activity.session_id}] 对话进行中，跳过做梦")
                continue

            try:
                with self._lock:
                    self._is_dreaming = True
                    self._dreaming_session = activity.session_id

                logger.info(
                    f"[{activity.session_id}] 空闲 {idle_time:.0f}s，"
                    f"触发自动做梦({self.dream_steps}步)")

                start_time = time.time()
                result = loop.dream(
                    n_steps=self.dream_steps,
                    full_cycle=self.dream_full_cycle,
                )
                duration = time.time() - start_time

                # 更新活动记录
                with self._lock:
                    act = self._activities.get(activity.session_id)
                    if act is not None:
                        act.dream_count += 1
                        act.last_dream_ts = time.time()
                        act.last_dream_result = result

                logger.info(
                    f"[{activity.session_id}] 做梦完成: "
                    f"{result.get('steps', 0)}步, "
                    f"耗时{duration:.1f}s, "
                    f"累计{act.dream_count}次")

            except Exception as e:
                logger.error(
                    f"[{activity.session_id}] 做梦失败: {e}",
                    exc_info=True)
            finally:
                with self._lock:
                    self._is_dreaming = False
                    self._dreaming_session = None
                self._busy_lock.release()

    # ------------------------------------------------------------------ #
    #  手动触发
    # ------------------------------------------------------------------ #

    def trigger_dream(self, session_id: str, steps: int = 20,
                      full_cycle: bool = False,
                      timeout: float = 120.0) -> dict:
        """手动触发指定会话的做梦。

        获取对话锁后执行做梦，阻塞直到完成或超时。

        参数:
            session_id: 会话标识。
            steps: 做梦步数。
            full_cycle: 是否完整七阶段周期。
            timeout: 最大等待时间（秒）。

        返回:
            做梦结果字典。失败时返回 {'status': 'error', 'error': ...}。
        """
        loop = self._get_loop_fn(session_id)
        if loop is None:
            return {'status': 'error', 'error': f'会话 {session_id} 不存在'}

        acquired = self._busy_lock.acquire(timeout=timeout)
        if not acquired:
            return {'status': 'error', 'error': '获取做梦锁超时'}

        try:
            with self._lock:
                self._is_dreaming = True
                self._dreaming_session = session_id

            result = loop.dream(n_steps=steps, full_cycle=full_cycle)

            with self._lock:
                act = self._activities.get(session_id)
                if act is not None:
                    act.dream_count += 1
                    act.last_dream_ts = time.time()
                    act.last_dream_result = result

            return result
        except Exception as e:
            logger.error(f"[{session_id}] 手动做梦失败: {e}", exc_info=True)
            return {'status': 'error', 'error': str(e)}
        finally:
            with self._lock:
                self._is_dreaming = False
                self._dreaming_session = None
            self._busy_lock.release()

    # ------------------------------------------------------------------ #
    #  状态查询
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        """返回调度器状态摘要。"""
        with self._lock:
            sessions = []
            for sid, act in self._activities.items():
                sessions.append({
                    'session_id': sid,
                    'idle_seconds': time.time() - act.last_active_ts,
                    'dream_count': act.dream_count,
                    'last_dream_ts': act.last_dream_ts,
                    'last_dream_steps': act.last_dream_result.get('steps', 0),
                })

            # _is_dreaming / _dreaming_session 在 _check_and_dream 与
            # trigger_dream 中写入，此处需在同一把锁下读取以避免数据竞争
            return {
                'running': self._running,
                'is_dreaming': self._is_dreaming,
                'dreaming_session': self._dreaming_session,
                'idle_threshold': self.idle_threshold,
                'dream_steps': self.dream_steps,
                'dream_full_cycle': self.dream_full_cycle,
                'check_interval': self.check_interval,
                'registered_sessions': len(self._activities),
                'sessions': sessions,
            }

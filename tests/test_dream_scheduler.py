# -*- coding: utf-8 -*-
"""单元测试：做梦调度器 (DreamScheduler)

测试 runtime/dream_scheduler.py 的 DreamScheduler 类，覆盖：
  1. 基础功能：初始化、注册/移除会话、touch、状态查询
  2. 自动做梦触发：空闲触发、忙时不触发、空缓冲区不触发、对话期间暂停
  3. 手动触发：手动做梦、不存在的会话
  4. 对话/做梦协调：获取/释放对话权限、超时恢复
  5. 生命周期：启动/停止线程、做梦中停止不崩溃
  6. 集成测试：用真实 LivingMemoryLoop 测试自动做梦端到端

测试设计原则：
  - 小规模实例（num_nodes=32, input_dim=8）加速测试
  - 固定 seed 保证可复现
  - 使用 Mock 隔离 DreamScheduler 与真实记忆系统（多数测试）
  - 使用真实 LivingMemoryLoop 做端到端集成测试
  - 时间敏感测试使用轮询（poll）而非固定 sleep，提升鲁棒性
"""

import sys
import os
import time
import threading

# 确保项目根目录在 Python 路径中
# tests/ 的父目录就是项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pytest
import torch

from runtime.dream_scheduler import DreamScheduler, SessionActivity
from runtime.loop import LivingMemoryLoop
from runtime.config import default_config


# ============================================================
# 辅助类与函数
# ============================================================

class MockMemory:
    """模拟记忆管理器，暴露 _buffer 与 buffer_size() 供调度器检查长度。"""

    def __init__(self, buffer_size=5):
        self._buffer = [
            (torch.randn(32), float(i) * 0.5) for i in range(buffer_size)
        ]

    def buffer_size(self):
        """公开接口，与 MemoryManager.buffer_size() 对齐。"""
        return len(self._buffer)


class MockLoop:
    """模拟 LivingMemoryLoop，用于隔离测试 DreamScheduler。

    提供 DreamScheduler 所需的最小接口：
      - memory._buffer: 记忆缓冲区（调度器检查其长度决定是否做梦）
      - dream(n_steps, full_cycle): 做梦方法，返回统计字典

    dream() 的调用次数与参数被记录，便于断言。
    可选地向缓冲区追加条目以模拟真实做梦行为（做梦会新增记忆）。
    """

    def __init__(self, buffer_size=5, dream_result=None, grow_buffer=True):
        self.memory = MockMemory(buffer_size)
        self.dream_count = 0
        self.last_dream_args = None
        self._dream_result = dream_result or {
            'status': 'dreamed',
            'steps': 5,
            'avg_surprise': 0.5,
            'buffer_size': buffer_size,
        }
        self._grow_buffer = grow_buffer

    def dream(self, n_steps=20, full_cycle=False):
        self.dream_count += 1
        self.last_dream_args = (n_steps, full_cycle)
        if self._grow_buffer:
            for _ in range(n_steps):
                self.memory._buffer.append((torch.randn(32), 0.5))
        return dict(self._dream_result)


class MockEmbedder:
    """带 embed_text 的嵌入器，用于情景记忆存储与检索。

    SimpleEmbedder 仅有 embed(tokens)，无法触发 process_turn 中的
    情景记忆存储路径。本类补充 embed_text，基于文本哈希生成确定性
    向量，使相同文本产生相同向量、不同文本产生不同向量。
    """

    def __init__(self, dim=8):
        self._dim = dim

    def embed(self, tokens):
        return torch.randn(self._dim)

    def embed_text(self, text):
        if not text or not text.strip():
            return torch.zeros(self._dim)
        h = abs(hash(text)) % (2 ** 32)
        g = torch.Generator()
        g.manual_seed(h)
        return torch.randn(self._dim, generator=g) * 0.1


def make_test_config(**overrides):
    """创建小规模测试配置（用于集成测试的 LivingMemoryLoop）。

    使用 num_nodes=32, input_dim=8 加速测试，固定 seed=42 保证可复现。
    """
    config = default_config()
    config['num_nodes'] = 32
    config['input_dim'] = 8
    config['num_infer_steps'] = 5
    config['consolidation_interval'] = 3
    config['seed'] = 42
    config.update(overrides)
    return config


def wait_for_condition(predicate, timeout=5.0, interval=0.1):
    """轮询等待条件成立，超时返回 False。

    用于时间敏感的后台线程测试，避免固定 sleep 带来的脆弱性。

    参数:
        predicate: 无参可调用对象，返回 bool。
        timeout: 最大等待秒数。
        interval: 轮询间隔秒数。

    返回:
        True 若在超时前条件成立，False 若超时。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def mock_loop():
    """带 5 条记忆的 MockLoop。"""
    torch.manual_seed(42)
    return MockLoop(buffer_size=5)


@pytest.fixture
def empty_mock_loop():
    """空缓冲区的 MockLoop。"""
    torch.manual_seed(42)
    return MockLoop(buffer_size=0)


@pytest.fixture(autouse=True)
def _seed(tmp_path, monkeypatch):
    """每个测试前固定随机种子，保证可复现。"""
    torch.manual_seed(42)
    # 做梦观测文件隔离到 tmp（防真实 Loop 集成测试写脏仓库
    # runtime/dream_state.json，S1-9）
    monkeypatch.setenv("LMS_DREAM_STATE_PATH",
                       str(tmp_path / "dream_state.json"))


# ============================================================
# 1. 基础功能测试
# ============================================================

class TestDreamSchedulerBasics:
    """DreamScheduler 基础功能：初始化、会话管理、touch、状态查询。"""

    def test_init(self):
        """初始化参数正确读取。"""
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: None,
            idle_threshold=15.0,
            dream_steps=10,
            dream_full_cycle=True,
            check_interval=2.0,
            max_dream_duration=30.0,
        )
        assert scheduler.idle_threshold == 15.0
        assert scheduler.dream_steps == 10
        assert scheduler.dream_full_cycle is True
        assert scheduler.check_interval == 2.0
        assert scheduler.max_dream_duration == 30.0

        # 内部状态初始化
        assert scheduler._activities == {}
        assert scheduler._is_dreaming is False
        assert scheduler._dreaming_session is None
        assert scheduler._running is False
        assert scheduler._thread is None

    def test_init_defaults(self):
        """默认参数值正确。"""
        scheduler = DreamScheduler(get_loop_fn=lambda sid: None)
        assert scheduler.idle_threshold == 30.0
        assert scheduler.dream_steps == 20
        assert scheduler.dream_full_cycle is False
        assert scheduler.check_interval == 5.0
        assert scheduler.max_dream_duration == 60.0

    def test_register_unregister(self):
        """注册/移除会话。"""
        scheduler = DreamScheduler(get_loop_fn=lambda sid: None)

        # 初始无会话
        assert scheduler.get_status()['registered_sessions'] == 0

        # 注册两个会话
        scheduler.register_session("s1")
        scheduler.register_session("s2")
        status = scheduler.get_status()
        assert status['registered_sessions'] == 2
        session_ids = {s['session_id'] for s in status['sessions']}
        assert session_ids == {"s1", "s2"}

        # 重复注册不报错（幂等）
        scheduler.register_session("s1")
        assert scheduler.get_status()['registered_sessions'] == 2

        # 移除一个会话
        scheduler.unregister_session("s1")
        status = scheduler.get_status()
        assert status['registered_sessions'] == 1
        assert status['sessions'][0]['session_id'] == "s2"

        # 移除不存在的会话不报错
        scheduler.unregister_session("nonexistent")
        assert scheduler.get_status()['registered_sessions'] == 1

    def test_touch(self):
        """touch 更新活动时间。"""
        scheduler = DreamScheduler(get_loop_fn=lambda sid: None)
        scheduler.register_session("s1")

        # 获取注册时的初始时间戳
        old_ts = scheduler._activities["s1"].last_active_ts

        # 等待一小段时间使时间戳可区分
        time.sleep(0.05)
        scheduler.touch("s1")

        new_ts = scheduler._activities["s1"].last_active_ts
        assert new_ts > old_ts, (
            f"touch 后时间戳({new_ts})应大于之前({old_ts})"
        )

    def test_touch_unregistered(self):
        """touch 未注册的会话时自动创建。"""
        scheduler = DreamScheduler(get_loop_fn=lambda sid: None)
        assert scheduler.get_status()['registered_sessions'] == 0

        scheduler.touch("auto_created")
        assert scheduler.get_status()['registered_sessions'] == 1
        assert "auto_created" in scheduler._activities

    def test_get_status(self):
        """状态查询返回正确字段。"""
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: None,
            idle_threshold=12.0,
            dream_steps=8,
            dream_full_cycle=True,
            check_interval=3.0,
        )
        scheduler.register_session("s1")
        scheduler.register_session("s2")

        status = scheduler.get_status()

        # 顶层字段
        expected_keys = {
            'running', 'is_dreaming', 'dreaming_session',
            'idle_threshold', 'dream_steps', 'dream_full_cycle',
            'check_interval', 'registered_sessions', 'sessions',
        }
        assert set(status.keys()) == expected_keys

        # 值正确
        assert status['running'] is False
        assert status['is_dreaming'] is False
        assert status['dreaming_session'] is None
        assert status['idle_threshold'] == 12.0
        assert status['dream_steps'] == 8
        assert status['dream_full_cycle'] is True
        assert status['check_interval'] == 3.0
        assert status['registered_sessions'] == 2

        # 每个会话条目字段
        for sess in status['sessions']:
            assert set(sess.keys()) == {
                'session_id', 'idle_seconds', 'dream_count',
                'last_dream_ts', 'last_dream_steps',
            }
            assert sess['dream_count'] == 0
            assert sess['last_dream_ts'] == 0.0
            assert sess['last_dream_steps'] == 0


# ============================================================
# 2. 自动做梦触发测试
# ============================================================

class TestAutoDream:
    """自动做梦触发：空闲触发、忙时不触发、空缓冲区、对话暂停。"""

    def test_auto_dream_triggers(self, mock_loop):
        """空闲超过阈值后自动触发做梦。

        设置极短的 idle_threshold(1秒) 和 check_interval(0.5秒)，
        注册有记忆的会话后等待，验证 dream_count > 0。
        """
        loops = {"s1": mock_loop}
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: loops.get(sid),
            idle_threshold=1.0,
            dream_steps=5,
            check_interval=0.5,
        )
        scheduler.register_session("s1")
        # 将最后活动时间设到过去，使首次检查即满足空闲条件
        scheduler._activities["s1"].last_active_ts = time.time() - 10

        scheduler.start()
        try:
            # 轮询等待做梦触发（最多等待 5 秒）
            triggered = wait_for_condition(
                lambda: scheduler.get_status()['sessions'][0]['dream_count'] > 0,
                timeout=5.0,
                interval=0.2,
            )
            assert triggered, "空闲超时后应自动触发做梦"

            status = scheduler.get_status()
            assert status['sessions'][0]['dream_count'] > 0
            # MockLoop 的 dream 被调用
            assert mock_loop.dream_count >= 1
        finally:
            scheduler.stop()

    def test_no_dream_when_busy(self, mock_loop):
        """当 _busy_lock 被持有时不触发做梦。"""
        loops = {"s1": mock_loop}
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: loops.get(sid),
            idle_threshold=0.5,
            dream_steps=3,
            check_interval=0.3,
        )
        scheduler.register_session("s1")
        scheduler._activities["s1"].last_active_ts = time.time() - 10

        # 持有对话锁（模拟正在进行的对话）
        acquired = scheduler.acquire_conversation("s1", timeout=1.0)
        assert acquired is True

        scheduler.start()
        try:
            # 等待多个检查周期
            time.sleep(2.0)

            status = scheduler.get_status()
            assert all(
                s['dream_count'] == 0 for s in status['sessions']
            ), "对话进行中（_busy_lock 持有）时不应触发做梦"
            # MockLoop 的 dream 未被调用
            assert mock_loop.dream_count == 0
        finally:
            scheduler.release_conversation("s1")
            scheduler.stop()

    def test_no_dream_empty_buffer(self, empty_mock_loop):
        """缓冲区为空时不触发做梦。"""
        loops = {"s1": empty_mock_loop}
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: loops.get(sid),
            idle_threshold=0.5,
            dream_steps=3,
            check_interval=0.3,
        )
        scheduler.register_session("s1")
        scheduler._activities["s1"].last_active_ts = time.time() - 10

        scheduler.start()
        try:
            time.sleep(2.0)

            status = scheduler.get_status()
            assert all(
                s['dream_count'] == 0 for s in status['sessions']
            ), "缓冲区为空时不应触发做梦"
            assert empty_mock_loop.dream_count == 0
        finally:
            scheduler.stop()

    def test_no_dream_when_loop_missing(self):
        """get_loop_fn 返回 None 时跳过该会话。"""
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: None,
            idle_threshold=0.5,
            dream_steps=3,
            check_interval=0.3,
        )
        scheduler.register_session("s1")
        scheduler._activities["s1"].last_active_ts = time.time() - 10

        scheduler.start()
        try:
            time.sleep(2.0)
            status = scheduler.get_status()
            assert all(
                s['dream_count'] == 0 for s in status['sessions']
            )
        finally:
            scheduler.stop()

    def test_dream_paused_on_conversation(self, mock_loop):
        """acquire_conversation 后不触发自动做梦，释放后恢复。"""
        loops = {"s1": mock_loop}
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: loops.get(sid),
            idle_threshold=0.5,
            dream_steps=3,
            check_interval=0.3,
        )
        scheduler.register_session("s1")
        scheduler._activities["s1"].last_active_ts = time.time() - 10

        # 先获取对话权限，再启动调度器，确保做梦线程启动时锁已被持有
        acquired = scheduler.acquire_conversation("s1", timeout=1.0)
        assert acquired is True

        scheduler.start()
        try:
            time.sleep(1.5)
            status = scheduler.get_status()
            assert all(
                s['dream_count'] == 0 for s in status['sessions']
            ), "对话期间不应触发做梦"

            # 释放对话权限 - 做梦应恢复
            scheduler.release_conversation("s1")

            triggered = wait_for_condition(
                lambda: scheduler.get_status()['sessions'][0]['dream_count'] > 0,
                timeout=5.0,
                interval=0.2,
            )
            assert triggered, "释放对话权限后应恢复自动做梦"
        finally:
            scheduler.stop()


# ============================================================
# 3. 手动触发测试
# ============================================================

class TestManualTrigger:
    """手动触发做梦。"""

    def test_trigger_dream_manual(self, mock_loop):
        """手动触发做梦返回结果。"""
        loops = {"s1": mock_loop}
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: loops.get(sid),
            dream_steps=5,
        )
        scheduler.register_session("s1")

        result = scheduler.trigger_dream("s1", steps=5)
        assert isinstance(result, dict)
        assert result['status'] == 'dreamed'

        # MockLoop.dream 被调用一次
        assert mock_loop.dream_count == 1
        assert mock_loop.last_dream_args == (5, False)

        # 活动记录更新
        status = scheduler.get_status()
        sess = [s for s in status['sessions'] if s['session_id'] == "s1"][0]
        assert sess['dream_count'] == 1
        assert sess['last_dream_ts'] > 0
        assert sess['last_dream_steps'] == 5

    def test_trigger_dream_full_cycle(self, mock_loop):
        """手动触发完整周期做梦。"""
        mock_loop._dream_result = {
            'status': 'dreamed', 'steps': 10, 'cycles': 10,
        }
        loops = {"s1": mock_loop}
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: loops.get(sid),
        )
        scheduler.register_session("s1")

        result = scheduler.trigger_dream("s1", steps=10, full_cycle=True)
        assert result['status'] == 'dreamed'
        assert mock_loop.last_dream_args == (10, True)

    def test_trigger_dream_nonexistent(self):
        """不存在的会话返回错误。"""
        scheduler = DreamScheduler(get_loop_fn=lambda sid: None)

        result = scheduler.trigger_dream("nonexistent_session", steps=5)
        assert isinstance(result, dict)
        assert result['status'] == 'error'
        assert '不存在' in result['error']

    def test_trigger_dream_unregistered_updates_no_activity(self, mock_loop):
        """手动触发未注册会话时仍执行做梦，但不更新活动记录。"""
        loops = {"s1": mock_loop}
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: loops.get(sid),
        )
        # 不注册 s1
        assert scheduler.get_status()['registered_sessions'] == 0

        result = scheduler.trigger_dream("s1", steps=3)
        assert result['status'] == 'dreamed'
        assert mock_loop.dream_count == 1
        # 未注册，活动记录不增加
        assert scheduler.get_status()['registered_sessions'] == 0


# ============================================================
# 4. 对话/做梦协调测试
# ============================================================

class TestConversationCoordination:
    """对话权限的获取与释放、超时恢复。"""

    def test_acquire_release_conversation(self):
        """获取和释放对话权限。"""
        scheduler = DreamScheduler(get_loop_fn=lambda sid: None)

        # 获取权限
        acquired = scheduler.acquire_conversation("s1", timeout=1.0)
        assert acquired is True

        # 做梦期间 _busy_lock 被持有，第二次获取应等待
        # （在另一线程中验证非阻塞获取会失败）
        second_acquired = []
        def try_acquire():
            second_acquired.append(
                scheduler._busy_lock.acquire(blocking=False))
        t = threading.Thread(target=try_acquire)
        t.start()
        t.join()
        assert second_acquired == [False], "持有对话权限时 _busy_lock 不可重入"

        # 释放权限
        scheduler.release_conversation("s1")

        # 释放后可再次获取
        acquired2 = scheduler.acquire_conversation("s1", timeout=1.0)
        assert acquired2 is True
        scheduler.release_conversation("s1")

    def test_acquire_timeout(self):
        """对话超时后仍可继续（恢复后可重新获取）。"""
        scheduler = DreamScheduler(get_loop_fn=lambda sid: None)

        # 在另一线程持有锁
        holder_done = threading.Event()
        def hold_lock():
            scheduler._busy_lock.acquire()
            holder_done.wait(timeout=5)
            scheduler._busy_lock.release()

        holder = threading.Thread(target=hold_lock)
        holder.start()
        time.sleep(0.1)  # 确保锁已被持有

        try:
            # 获取超时 - 返回 False
            acquired = scheduler.acquire_conversation("s1", timeout=0.3)
            assert acquired is False, "锁被持有时应超时返回 False"
        finally:
            # 释放持有者（不在此处调用 release_conversation，
            # 否则会释放持有者的锁；release 的容错性由单独的测试覆盖）
            holder_done.set()
            holder.join()

        # 持有者释放后，应能重新获取（系统恢复正常）
        acquired = scheduler.acquire_conversation("s1", timeout=2.0)
        assert acquired is True, "超时后锁释放应能重新获取"
        scheduler.release_conversation("s1")

    def test_release_without_acquire_no_crash(self):
        """未获取锁时调用 release 不崩溃。"""
        scheduler = DreamScheduler(get_loop_fn=lambda sid: None)
        # 直接 release 不应抛异常
        scheduler.release_conversation("s1")

    def test_is_dreaming_reflects_state(self, mock_loop):
        """is_dreaming / get_dreaming_session 在做梦期间反映状态。"""
        dream_started = threading.Event()
        dream_can_finish = threading.Event()

        original_dream = mock_loop.dream

        def blocking_dream(n_steps=20, full_cycle=False):
            dream_started.set()
            dream_can_finish.wait(timeout=5)
            return original_dream(n_steps, full_cycle)

        mock_loop.dream = blocking_dream
        loops = {"s1": mock_loop}
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: loops.get(sid),
            idle_threshold=0.1,
            dream_steps=3,
            check_interval=0.2,
        )
        scheduler.register_session("s1")
        scheduler._activities["s1"].last_active_ts = time.time() - 10

        scheduler.start()
        try:
            assert dream_started.wait(timeout=3), "应在 3 秒内开始做梦"
            assert scheduler.is_dreaming() is True
            assert scheduler.get_dreaming_session() == "s1"
        finally:
            dream_can_finish.set()
            scheduler.stop(timeout=2.0)

        # 做梦结束后状态恢复
        assert scheduler.is_dreaming() is False
        assert scheduler.get_dreaming_session() is None


# ============================================================
# 5. 生命周期测试
# ============================================================

class TestLifecycle:
    """线程生命周期：启动/停止、做梦中停止。"""

    def test_start_stop(self):
        """启动和停止线程。"""
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: None,
            idle_threshold=1000,  # 高阈值避免触发做梦
            check_interval=0.5,
        )
        assert scheduler._running is False
        assert scheduler._thread is None

        scheduler.start()
        assert scheduler._running is True
        assert scheduler._thread is not None
        assert scheduler._thread.is_alive()

        scheduler.stop(timeout=2)
        assert scheduler._running is False
        assert scheduler._thread is None

    def test_start_idempotent(self):
        """重复启动不创建多个线程。"""
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: None,
            idle_threshold=1000,
            check_interval=0.5,
        )
        scheduler.start()
        thread1 = scheduler._thread

        scheduler.start()  # 重复启动
        assert scheduler._thread is thread1, "重复启动不应创建新线程"

        scheduler.stop()

    def test_stop_when_not_running(self):
        """未启动时 stop 不报错。"""
        scheduler = DreamScheduler(get_loop_fn=lambda sid: None)
        scheduler.stop()  # 不应抛异常

    def test_stop_during_dream(self):
        """做梦中停止不崩溃。

        后台线程正在执行 loop.dream() 时调用 stop()，
        验证 stop() 正常返回且不抛异常。
        """
        dream_started = threading.Event()
        dream_can_finish = threading.Event()

        class BlockingLoop:
            def __init__(self):
                self.memory = MockMemory(5)

            def dream(self, n_steps=20, full_cycle=False):
                dream_started.set()
                # 阻塞直到测试释放
                dream_can_finish.wait(timeout=10)
                return {'status': 'dreamed', 'steps': n_steps}

        loop = BlockingLoop()
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: loop,
            idle_threshold=0.1,
            dream_steps=5,
            check_interval=0.2,
        )
        scheduler.register_session("s1")
        scheduler._activities["s1"].last_active_ts = time.time() - 10

        scheduler.start()
        try:
            # 等待做梦开始
            assert dream_started.wait(timeout=3), "应在 3 秒内开始做梦"
            assert scheduler.is_dreaming() is True

            # 做梦中停止 - 不应抛异常
            scheduler.stop(timeout=1.0)
        finally:
            # 释放阻塞的做梦，让守护线程退出
            dream_can_finish.set()
            scheduler.stop()

        # 到达此处说明未崩溃
        assert scheduler._running is False

    def test_stop_wakes_sleeping_thread(self):
        """stop 能唤醒等待中的后台线程（_stop_event.wait 可被提前唤醒）。"""
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: None,
            idle_threshold=1000,
            check_interval=2.0,  # 长检查间隔
        )
        scheduler.start()
        time.sleep(0.2)  # 确保线程进入等待

        # stop 应在远小于 check_interval 的时间内完成
        start = time.time()
        scheduler.stop(timeout=2.0)
        elapsed = time.time() - start
        assert elapsed < 1.5, (
            f"stop 应通过 _stop_event 提前唤醒线程，耗时 {elapsed:.2f}s"
        )


# ============================================================
# 6. 集成测试
# ============================================================

class TestSchedulerIntegration:
    """用真实 LivingMemoryLoop 测试 DreamScheduler 端到端。"""

    def test_scheduler_with_real_loop(self, tmp_path):
        """真实 LivingMemoryLoop 集成测试。

        流程：存储记忆 -> 启动调度器 -> 等待自动做梦 ->
        验证做梦次数 > 0 -> 验证缓冲区增长 -> 检索记忆仍可用。
        """
        torch.manual_seed(42)
        config = make_test_config(
            snapshot_dir=str(tmp_path),
            embedder=MockEmbedder(dim=8),
        )
        loop = LivingMemoryLoop(config)

        # 存储几条记忆（通过 process_turn 走完整记忆路径）
        texts = [
            "重要的AI记忆内容",
            "机器学习基础知识",
            "深度学习网络架构",
        ]
        for text in texts:
            loop.process_turn(text)

        # 记录做梦前的缓冲区大小
        buffer_before = len(loop.memory._buffer)
        assert buffer_before > 0, "process_turn 后缓冲区应有记忆"

        # 创建调度器：极短 idle_threshold 加速触发
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: loop,
            idle_threshold=1.0,
            dream_steps=5,
            check_interval=0.5,
        )
        scheduler.register_session("integration_session")
        # 将最后活动时间设到过去，使首次检查即满足空闲条件
        scheduler._activities["integration_session"].last_active_ts = (
            time.time() - 10
        )

        scheduler.start()
        try:
            # 轮询等待做梦触发
            triggered = wait_for_condition(
                lambda: scheduler.get_status()['sessions'][0]['dream_count'] > 0,
                timeout=8.0,
                interval=0.2,
            )
            assert triggered, "真实 loop 应在空闲后自动触发做梦"

            # 验证做梦次数 > 0
            status = scheduler.get_status()
            sess = status['sessions'][0]
            assert sess['dream_count'] > 0
            assert sess['last_dream_ts'] > 0
            assert sess['last_dream_steps'] > 0

            # 验证缓冲区增长（做梦会新增条目）
            buffer_after = len(loop.memory._buffer)
            assert buffer_after > buffer_before, (
                f"做梦后缓冲区({buffer_after})应大于做梦前({buffer_before})"
            )
        finally:
            scheduler.stop()

        # 验证记忆仍可被检索（做梦不破坏已有记忆）
        # [A16] 存储文本统一带"用户:"前缀（issue A16）→ 检索 query 需与
        # 存储格式同构（裸文本 query 与新前缀存储不再同向量）
        query = loop.embedder.embed_text("用户: 重要的AI记忆内容")
        entries = loop.memory.recall_episodic(query, top_k=3)
        assert len(entries) > 0, "做梦后情景记忆仍应可被检索"
        assert "AI" in entries[0].text, (
            "检索到的记忆应包含存储的文本内容"
        )

    def test_manual_dream_with_real_loop(self, tmp_path):
        """手动触发真实 loop 做梦，验证结果与记忆可用性。"""
        torch.manual_seed(42)
        config = make_test_config(
            snapshot_dir=str(tmp_path),
            embedder=MockEmbedder(dim=8),
        )
        loop = LivingMemoryLoop(config)

        # 存储记忆
        loop.process_turn("手动做梦测试记忆")

        buffer_before = len(loop.memory._buffer)
        assert buffer_before > 0

        # 手动触发做梦
        scheduler = DreamScheduler(
            get_loop_fn=lambda sid: loop,
            dream_steps=5,
        )
        scheduler.register_session("s1")

        result = scheduler.trigger_dream("s1", steps=5)
        assert isinstance(result, dict)
        assert result['status'] in ('dreamed', 'no_memories_to_replay')

        if result['status'] == 'dreamed':
            # 缓冲区应增长
            buffer_after = len(loop.memory._buffer)
            assert buffer_after > buffer_before, (
                "手动做梦后缓冲区应增长"
            )

        # 记忆仍可检索
        query = loop.embedder.embed_text("手动做梦测试记忆")
        entries = loop.memory.recall_episodic(query, top_k=3)
        assert len(entries) > 0

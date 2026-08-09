"""
活体记忆系统 - 感官层：embed 熔断器（T2.8/P2-6）
================================================

问题：夜间 embed 服务（如手机 Ollama / 公网隧道）不可达时，每次请求都
走完整重试链（CloudEmbedder retries=3 × timeout=30s）→ /recall 每次卡
5-10s 后 500/空，且 3 天无人察觉（R7 S6 实测）。

方案：连续 N 次失败 → 熔断 OPEN 冷却 cooldown 秒；冷却期内所有调用
快速失败（不触网），冷却结束自动半开（放行一次探测）。

设计约束（与总体方案 T2.8 一致）:
  - 开关：LMS_EMBED_CIRCUIT=1 启用（默认 0 = 关闭，行为与原来完全一致）；
  - 线程安全：内部 threading.Lock 保护状态；
  - fail-open：熔断期快速失败（抛 CircuitOpenError），调用方（loop 的
    _encode_query_vector）捕获后返回空检索结果，绝不卡请求；
  - 可测：clock 可注入（测试用假时钟模拟冷却期结束）。
"""

import os
import time
import logging
import threading
from typing import Optional, Callable, Any

logger = logging.getLogger("core.sensory.circuit_breaker")


class CircuitOpenError(RuntimeError):
    """熔断器处于 OPEN 状态时的快速失败异常。"""


class EmbedCircuitBreaker:
    """embed 调用熔断器。

    状态机:
      - CLOSED（默认）: 正常调用；连续失败计数 < max_failures。
      - OPEN: 连续失败达 max_failures → 熔断 cooldown 秒；期间 call()
        直接抛 CircuitOpenError（不触网）。
      - HALF-OPEN（隐式）: 冷却期结束后的下一次调用放行探测；成功 →
        回到 CLOSED（计数清零），失败 → 重新 OPEN。

    参数:
        enabled: 是否启用熔断（None 时读环境变量 LMS_EMBED_CIRCUIT，
            默认 0 = 关闭）。
        max_failures: 连续失败多少次后熔断（默认 3）。
        cooldown: 熔断冷却秒数（默认 300 = 5 分钟）。
        clock: 时间源（测试注入；默认 time.time）。
    """

    def __init__(self, enabled: Optional[bool] = None,
                 max_failures: int = 3,
                 cooldown: float = 300.0,
                 clock: Optional[Callable[[], float]] = None) -> None:
        if enabled is None:
            enabled = os.environ.get("LMS_EMBED_CIRCUIT", "0") == "1"
        self.enabled: bool = bool(enabled)
        self.max_failures: int = int(max_failures)
        self.cooldown: float = float(cooldown)
        self._clock = clock or time.time

        self._lock = threading.Lock()
        self._failures: int = 0
        self._open_until: float = 0.0

        # 统计（诊断用）
        self.tripped_count: int = 0   # 累计熔断次数
        self.blocked_count: int = 0   # 熔断期被快速拒绝的调用次数

    # ------------------------------------------------------------------ #
    #  状态
    # ------------------------------------------------------------------ #

    @property
    def is_open(self) -> bool:
        """当前是否处于熔断 OPEN 状态。

        冷却期已结束则自动转半开（返回 False，放行一次探测），
        并清零失败计数——探测失败会重新计数并再次熔断。
        """
        if not self.enabled:
            return False
        with self._lock:
            if self._open_until > 0 and self._clock() >= self._open_until:
                # 冷却结束：进入半开（放行探测）
                self._open_until = 0.0
                self._failures = 0
                return False
            return self._open_until > 0

    @property
    def failure_count(self) -> int:
        """当前连续失败次数（诊断用）。"""
        with self._lock:
            return self._failures

    # ------------------------------------------------------------------ #
    #  记录
    # ------------------------------------------------------------------ #

    def record_success(self) -> None:
        """记录一次成功调用（清零连续失败计数）。"""
        with self._lock:
            self._failures = 0

    def record_failure(self) -> None:
        """记录一次失败；连续失败达阈值 → 熔断 OPEN cooldown 秒。"""
        with self._lock:
            self._failures += 1
            if self._failures >= self.max_failures:
                self._open_until = self._clock() + self.cooldown
                self.tripped_count += 1
                logger.warning(
                    "embed 熔断器已打开: 连续 %d 次失败, "
                    "熔断 %.0fs（LMS_EMBED_CIRCUIT 已启用）",
                    self._failures, self.cooldown)

    # ------------------------------------------------------------------ #
    #  调用包装
    # ------------------------------------------------------------------ #

    def call(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """带熔断保护的函数调用。

        - 关闭状态: 正常调用；成功 record_success，异常 record_failure 后上抛；
        - 熔断 OPEN: 快速抛 CircuitOpenError（不触网），blocked_count +1；
        - 半开探测: 放行一次；失败会重新熔断。

        返回:
            fn 的返回值。

        异常:
            CircuitOpenError: 熔断期间快速失败。
            fn 自身异常: 关闭状态下照常上抛（行为与无熔断一致）。
        """
        if self.is_open:
            self.blocked_count += 1
            raise CircuitOpenError(
                "embed 服务熔断中（连续失败 {} 次），快速失败".format(
                    self.max_failures))
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result

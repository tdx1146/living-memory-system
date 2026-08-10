"""bus_events.py — LMS → 主总线 发布侧（Phase 4 / D0 双向反馈中枢的"发出"侧）
================================================================================

只做**外围发布**，绝不侵入 LMS 核心算法（core/hippocampus/*、J 矩阵、
dream_engine 内部算法、self_ref 自指回路本体一律零改动）。

设计要点（与《全局设计-统合设计方案.md》第四节 D0 对齐）：
  1. **发事件 = 软参考**：LMS 发布的事件只是通知/参考信号，订阅方可忽略；
     总线是哑管道，本模块只负责"以统一契约写入事件文件"。
  2. **契约对齐 v1.1**：schema_version="1.1" + event_id=uuid4 + trace_id + t
     + event_type + producer + result；与 iso-sand LogWriter 同源语义
     （append + fcntl.flock + fsync）。
  3. **熔断闸门**：发布侧连续失败自动暂停（熔断），绝不把异常抛回 LMS 主循环
     ——任何发布异常 try/except 静默降级，只记日志。
  4. **self_ref 最敏感**：默认关闭（LMS_SELF_REF_PUBLISH 默认 "off"），
     开启后才发布"蒸馏后的可发布摘要"（≤200 字），且限频（≥30 分钟一条）。

环境变量（均可覆盖默认值）：
    LMS_BUS_FILE                -- 总线事件文件路径（默认 iso-sand/data/event_bus.jsonl）
    LMS_BUS_MAX_FAILURES        -- 熔断阈值：连续失败 N 次触发熔断（默认 5）
    LMS_BUS_COOLDOWN_SECONDS    -- 熔断冷却时间（默认 600 = 10 分钟）
    LMS_SELF_REF_PUBLISH        -- self_ref 发布开关："on" 才发布（默认 off）
    LMS_SELF_REF_MIN_INTERVAL   -- self_ref 发布最小间隔秒数（默认 1800 = 30 分钟）

安全红线：
  - 本模块不 import core.hippocampus 内部算法、不读 J 矩阵、不碰自指回路。
  - 所有发布函数对外保证**永不抛异常**（静默降级 + 日志）。
"""

import fcntl
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Optional

logger = logging.getLogger("bus_events")

__all__ = [
    "CircuitBreaker",
    "BusEventPublisher",
    "get_publisher",
    "publish_plastified",
    "publish_dream_complete",
    "publish_self_ref",
    "reset_publisher_for_test",
    "get_publisher_status",
    "quick_test",
]

_BJT = timezone(timedelta(hours=8))

# ---------------------------------------------------------------------------
# 默认配置（环境变量可覆盖）
# ---------------------------------------------------------------------------
_DEFAULT_BUS_FILE = os.environ.get(
    "LMS_BUS_FILE",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "Agent OS", "iso-sand", "data", "event_bus.jsonl",
    ),
)
_PRODUCER = "lms"

_MAX_PAYLOAD_DEPTH = 4          # payload 序列化最大嵌套深度（防意外递归）
_MAX_PAYLOAD_ITEMS = 50         # 列表/字典最大条目数（只发数值摘要，不发达量原始态）
_SELF_REF_SUMMARY_MAX_CHARS = 200  # self_ref 蒸馏摘要上限（≤200 字）


def _env_int(key: str, default: int) -> int:
    """读环境变量为 int，非法时回退默认。"""
    try:
        return int(os.environ.get(key, str(default)).strip())
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    """读环境变量为 float，非法时回退默认。"""
    try:
        return float(os.environ.get(key, str(default)).strip())
    except (TypeError, ValueError):
        return default


def _env_bool_on(key: str, default: bool = False) -> bool:
    """读环境变量为布尔开关（"on"/"1"/"true"/"yes" → True）。"""
    val = os.environ.get(key, "").strip().lower()
    if not val:
        return default
    return val in ("on", "1", "true", "yes")


def _now_iso() -> str:
    """当前时间 ISO 8601（北京时区，与总线既有事件格式一致）。"""
    return datetime.now(_BJT).isoformat()


def _sanitize(value: Any, depth: int = 0) -> Any:
    """把任意值转换为 JSON 安全的基本类型（数值摘要专用，防原始激活态泄漏）。

    - torch/numpy 标量 → float/int
    - dict/list 递归裁剪（深度/条目上限）
    - 其他不可序列化对象 → 类型名（不发原始对象）
    """
    if depth > _MAX_PAYLOAD_DEPTH:
        return "..."
    # torch.Tensor / numpy 数组 → 只发标量摘要，绝不发原始向量
    if hasattr(value, "item") and callable(value.item):
        try:
            return _sanitize(value.item(), depth + 1)
        except Exception:
            return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        # float 非有限值 → None（JSON 不允许 NaN/Infinity）
        if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
            return None
        return value
    if value is None:
        return None
    if isinstance(value, dict):
        out = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= _MAX_PAYLOAD_ITEMS:
                break
            out[str(k)] = _sanitize(v, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for i, v in enumerate(value):
            if i >= _MAX_PAYLOAD_ITEMS:
                break
            out.append(_sanitize(v, depth + 1))
        return out
    return f"<{type(value).__name__}>"


# ---------------------------------------------------------------------------
# 熔断器
# ---------------------------------------------------------------------------
class CircuitBreaker:
    """发布侧熔断器：连续失败达阈值 → 熔断暂停发布，冷却后自动尝试恢复。

    语义（与设计"熔断闸门"一致）：
      - allow():     熔断中（冷却未到）→ False（暂停发布）；否则 True（含冷却
                     结束后的半开状态，允许一次试探性发布）
      - record_success(): 发布成功 → 失败计数清零
      - record_failure(): 发布失败 → 失败计数 +1；达阈值 → 熔断（记录熔断时刻）
      - 任何时刻调用方都可 get_status() 查看熔断状态（观测用）
    """

    def __init__(self, max_failures: int = 5, cooldown_seconds: float = 600.0):
        self.max_failures = max(max_failures, 1)
        self.cooldown_seconds = max(cooldown_seconds, 1.0)
        self._failures = 0
        self._open_until = 0.0
        self._tripped_count = 0
        self._total_failures = 0
        self._total_successes = 0
        self._lock = threading.Lock()

    def allow(self) -> bool:
        """是否允许发布。熔断冷却未到 → False。"""
        with self._lock:
            if self._open_until > 0.0:
                if time.time() < self._open_until:
                    return False
                # 冷却结束：半开，放行一次试探
                logger.warning(
                    "[bus_events] 熔断冷却结束，尝试恢复发布 "
                    f"(已暂停 {self.cooldown_seconds:.0f}s)")
                self._open_until = 0.0
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._total_successes += 1

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._total_failures += 1
            if self._failures >= self.max_failures:
                self._open_until = time.time() + self.cooldown_seconds
                self._tripped_count += 1
                logger.warning(
                    "[bus_events] 🔴 熔断触发：连续失败 %d 次，"
                    "暂停发布 %.0f 秒（已熔断 %d 次）",
                    self._failures, self.cooldown_seconds, self._tripped_count)

    def is_open(self) -> bool:
        with self._lock:
            return self._open_until > 0.0 and time.time() < self._open_until

    def reset(self) -> None:
        """手动复位（测试/运维用）。"""
        with self._lock:
            self._failures = 0
            self._open_until = 0.0

    def status(self) -> dict:
        with self._lock:
            return {
                "max_failures": self.max_failures,
                "cooldown_seconds": self.cooldown_seconds,
                "consecutive_failures": self._failures,
                "open": self.is_open(),
                "tripped_count": self._tripped_count,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
            }


# ---------------------------------------------------------------------------
# 写入器（LogWriter 等价物：append + fcntl.flock + fsync，契约 v1.1）
# ---------------------------------------------------------------------------
class _BusWriter:
    """线程安全 + 进程安全的 JSONL 追加写入器（与 iso-sand LogWriter 语义等价）。

    append + fcntl.flock + fsync：与总线消费者/调度器共用同一把跨进程文件锁，
    保证并发写不互相撕裂。
    """

    def __init__(self, filepath: str):
        self._filepath = os.path.abspath(filepath)
        self._lock = threading.Lock()
        self._ensure_dir()

    def _ensure_dir(self):
        os.makedirs(os.path.dirname(self._filepath), exist_ok=True)

    @property
    def filepath(self) -> str:
        return self._filepath

    def write(self, record: dict, validate: bool = True) -> dict:
        """追加写入一条 JSONL 记录（缺 t 自动补；必填字段校验）。"""
        if "t" not in record:
            record["t"] = _now_iso()
        if validate:
            for field in ("event_type", "producer", "result"):
                if field not in record:
                    raise ValueError(f"缺少必填字段 '{field}'，记录: {record}")
            if record.get("result") not in ("OK", "FAIL", "WARN", "TIMEOUT"):
                raise ValueError(
                    f"result 必须为 OK/FAIL/WARN/TIMEOUT，实际: {record.get('result')}")
        with self._lock:
            with open(self._filepath, "a", encoding="utf-8") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    line = json.dumps(record, ensure_ascii=False)
                    f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        return record

    def read_all(self) -> list:
        """读取全部记录（测试/审计用；非 JSON 行跳过）。"""
        if not os.path.exists(self._filepath):
            return []
        records = []
        with open(self._filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records


# ---------------------------------------------------------------------------
# 发布器（单例）
# ---------------------------------------------------------------------------
class BusEventPublisher:
    """LMS 总线发布器：v1.1 事件构造 + 熔断 + self_ref 开关/限频。

    对外保证：**任何方法永不抛异常**（发布失败只记日志 + 喂熔断器）。
    """

    def __init__(self, bus_file: Optional[str] = None):
        self.bus_file = bus_file or os.environ.get(
            "LMS_BUS_FILE", _DEFAULT_BUS_FILE)
        self.breaker = CircuitBreaker(
            max_failures=_env_int("LMS_BUS_MAX_FAILURES", 5),
            cooldown_seconds=_env_float("LMS_BUS_COOLDOWN_SECONDS", 600.0),
        )
        self.self_ref_publish_enabled = _env_bool_on(
            "LMS_SELF_REF_PUBLISH", default=False)
        self.self_ref_min_interval = _env_float(
            "LMS_SELF_REF_MIN_INTERVAL", 1800.0)
        # self_ref 限频状态
        self._self_ref_lock = threading.Lock()
        self._self_ref_last_ts = 0.0
        self._self_ref_published = 0
        self._writer: Optional[_BusWriter] = None

    def _get_writer(self) -> Optional[_BusWriter]:
        """惰性创建写入器；失败（如路径不可写）返回 None 并喂熔断器。"""
        if self._writer is None:
            try:
                self._writer = _BusWriter(self.bus_file)
            except Exception as e:
                logger.warning("[bus_events] 写入器创建失败: %s", e)
                self.breaker.record_failure()
                return None
        return self._writer

    # -- 通用发布 --------------------------------------------------------
    def publish(self, event_type: str, payload: dict,
                trace_id: Optional[str] = None,
                result: str = "OK",
                detail: str = "") -> bool:
        """发布一条 v1.1 契约事件。返回 True=已写入总线；False=未发布（熔断/失败）。"""
        # 熔断闸门：熔断中直接拒绝（静默，日志）
        if not self.breaker.allow():
            logger.info(
                "[bus_events] 熔断中，跳过发布 %s "
                "(冷却剩余 %.0fs)", event_type,
                max(0.0, self.breaker._open_until - time.time()))
            return False

        writer = self._get_writer()
        if writer is None:
            # 写入器不可用已在 _get_writer 内喂过熔断器
            return False

        record = {
            "t": _now_iso(),
            "schema_version": "1.1",
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "producer": _PRODUCER,
            "result": result,
            "trace_id": trace_id or f"lms:{event_type}:{uuid.uuid4().hex[:8]}",
            "payload": _sanitize(payload),
        }
        if detail:
            record["detail"] = str(detail)[:500]

        try:
            writer.write(record)
            self.breaker.record_success()
            logger.info(
                "[bus_events] ✅ 已发布 %s (event_id=%s trace=%s)",
                event_type, record["event_id"], record["trace_id"])
            return True
        except Exception as e:
            # 静默降级：绝不让发布异常影响 LMS 主循环
            logger.warning(
                "[bus_events] ⚠️ 发布 %s 失败: %s:%s "
                "(失败计数 %d/%d)",
                event_type, type(e).__name__, e,
                self.breaker._failures + 1, self.breaker.max_failures)
            self.breaker.record_failure()
            return False

    # -- lms.plastified（塑形倾向/状态反哺）-----------------------------
    def publish_plastified(self, state: dict) -> bool:
        """发布 lms.plastified：只发数值摘要（entropy/surprise/激活节点数/目的层 precision），
        绝不发原始激活态/J 矩阵。

        参数 state 期望键（由 runtime/loop.py 钩子构造）：
            entropy, surprise, active_nodes, precision_mean, precision_std,
            coherence, turn_count, entropy_ratio
        """
        safe = _sanitize(state)
        return self.publish(
            "lms.plastified",
            payload=safe,
            detail="LMS 塑形倾向/状态摘要（软参考信号，订阅方可忽略）",
        )

    # -- lms.dream_complete（做梦完成通知）-------------------------------
    def publish_dream_complete(self, stats: dict) -> bool:
        """发布 lms.dream_complete：步数/耗时/结果（可观测性信号）。"""
        safe = _sanitize(stats)
        return self.publish(
            "lms.dream_complete",
            payload=safe,
            detail="LMS 做梦完成通知（可观测性信号，软参考）",
        )

    # -- lms.self_ref（自我认知透视，最敏感，默认关闭 + 限频）-------------
    def publish_self_ref(self, summary: str, guard: Optional[dict] = None) -> bool:
        """发布 lms.self_ref：蒸馏后文本摘要（≤200 字）+ 护栏状态。

        受 LMS_SELF_REF_PUBLISH 开关控制（默认 off）；开启后限频
        （≥ LMS_SELF_REF_MIN_INTERVAL 秒一条，默认 30 分钟）。

        参数:
            summary: 蒸馏后的可发布摘要文本（超长自动截断到 ≤200 字）
            guard:   护栏状态字典（state/autocorr/alpha/coherence 等数值摘要）
        """
        if not self.self_ref_publish_enabled:
            return False  # 默认关闭：看护人式第一步（保守护着，不发）

        # 限频
        with self._self_ref_lock:
            now = time.time()
            if now - self._self_ref_last_ts < self.self_ref_min_interval:
                logger.info(
                    "[bus_events] self_ref 限频中，跳过发布 "
                    "(距上次 %.0fs < %.0fs)",
                    now - self._self_ref_last_ts, self.self_ref_min_interval)
                return False
            self._self_ref_last_ts = now
            self._self_ref_published += 1

        text = str(summary or "").strip()
        if len(text) > _SELF_REF_SUMMARY_MAX_CHARS:
            # 截断后含省略号总长仍 ≤200 字
            text = text[:_SELF_REF_SUMMARY_MAX_CHARS - 1] + "…"
        payload = {
            "summary": text,
            "guard": _sanitize(guard or {}),
        }
        return self.publish(
            "lms.self_ref",
            payload=payload,
            detail="LMS 自我认知透视（蒸馏后限量发布，最高敏感度，软参考）",
        )

    # -- 状态 -------------------------------------------------------------
    def status(self) -> dict:
        st = self.breaker.status()
        st.update({
            "bus_file": self.bus_file,
            "self_ref_publish_enabled": self.self_ref_publish_enabled,
            "self_ref_min_interval": self.self_ref_min_interval,
            "self_ref_published_total": self._self_ref_published,
        })
        return st

    def reset_for_test(self) -> None:
        """测试用：复位熔断器 + self_ref 限频状态（不动总线文件）。"""
        self.breaker.reset()
        with self._self_ref_lock:
            self._self_ref_last_ts = 0.0
            self._self_ref_published = 0


# ---------------------------------------------------------------------------
# 模块级单例 + 便捷函数（供 runtime/loop.py、api/server.py 钩子调用）
# ---------------------------------------------------------------------------
_publisher: Optional[BusEventPublisher] = None
_publisher_lock = threading.Lock()


def get_publisher() -> BusEventPublisher:
    """获取模块级单例发布器（首次调用时按环境变量构造）。"""
    global _publisher
    if _publisher is None:
        with _publisher_lock:
            if _publisher is None:
                _publisher = BusEventPublisher()
    return _publisher


def publish_plastified(state: dict) -> bool:
    """模块级便捷函数：发布 lms.plastified（永不抛异常）。"""
    try:
        return get_publisher().publish_plastified(state)
    except Exception as e:  # 最后一道防线：任何异常静默降级
        logger.warning("[bus_events] publish_plastified 静默降级: %s", e)
        return False


def publish_dream_complete(stats: dict) -> bool:
    """模块级便捷函数：发布 lms.dream_complete（永不抛异常）。"""
    try:
        return get_publisher().publish_dream_complete(stats)
    except Exception as e:
        logger.warning("[bus_events] publish_dream_complete 静默降级: %s", e)
        return False


def publish_self_ref(summary: str, guard: Optional[dict] = None) -> bool:
    """模块级便捷函数：发布 lms.self_ref（开关+限频控制，永不抛异常）。"""
    try:
        return get_publisher().publish_self_ref(summary, guard)
    except Exception as e:
        logger.warning("[bus_events] publish_self_ref 静默降级: %s", e)
        return False


def reset_publisher_for_test() -> BusEventPublisher:
    """测试/运维用：强制重建单例（读取最新环境变量）。"""
    global _publisher
    with _publisher_lock:
        _publisher = BusEventPublisher()
    return _publisher


def get_publisher_status() -> dict:
    """观测用：当前发布器状态（熔断 + self_ref 开关）。"""
    try:
        return get_publisher().status()
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# 快速自测（不依赖真实总线文件）
# ---------------------------------------------------------------------------
def quick_test():
    """bus_events.py 自测：写入/契约字段/熔断/self_ref 开关与限频/摘要裁剪。"""
    import tempfile
    print("=" * 50)
    print("🧪 bus_events.py 快速自测")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        bus_file = os.path.join(tmpdir, "event_bus.jsonl")

        # ── 1. 基础发布 + v1.1 契约字段 ──
        pub = BusEventPublisher(bus_file=bus_file)
        ok = pub.publish_plastified({
            "entropy": 2.5, "surprise": 0.3, "active_nodes": 42,
            "precision_mean": 1.2, "precision_std": 0.1,
            "coherence": 0.7, "turn_count": 10, "entropy_ratio": 0.45,
        })
        assert ok, "基础发布应成功"
        recs = pub._get_writer().read_all()
        assert len(recs) == 1
        rec = recs[0]
        for field in ("schema_version", "event_id", "trace_id", "t",
                      "event_type", "producer", "result"):
            assert field in rec, f"缺字段 {field}"
        assert rec["schema_version"] == "1.1"
        assert rec["event_type"] == "lms.plastified"
        assert rec["producer"] == "lms"
        assert rec["result"] == "OK"
        assert rec["payload"]["active_nodes"] == 42
        print("✅ v1.1 契约字段齐全 + lms.plastified 发布成功")

        # ── 2. self_ref：默认关闭（不发布）──
        pub.reset_for_test()
        ok = pub.publish_self_ref("这是一段蒸馏后的自述摘要", {"state": "normal"})
        assert not ok, "self_ref 默认应关闭不发布"
        assert len(pub._get_writer().read_all()) == 1, "关闭时不应新增事件"
        print("✅ self_ref 默认关闭（看护人式第一步）")

        # ── 3. self_ref：开启后发布 + 摘要裁剪 ≤200 字 + 限频 ──
        pub.self_ref_publish_enabled = True
        long_text = "自" * 500
        ok = pub.publish_self_ref(long_text, {"state": "normal", "autocorr": 0.5})
        assert ok, "开启后应发布"
        recs = pub._get_writer().read_all()
        sr = [r for r in recs if r["event_type"] == "lms.self_ref"]
        assert len(sr) == 1
        assert len(sr[0]["payload"]["summary"]) <= 200, "摘要应 ≤200 字"
        assert sr[0]["payload"]["guard"]["state"] == "normal"
        # 限频：立即再发应被拒
        ok2 = pub.publish_self_ref("第二条", {"state": "normal"})
        assert not ok2, "限频内第二条应被拒"
        print("✅ self_ref 开启后发布 + ≤200字裁剪 + 限频生效")

        # ── 4. 熔断：连续失败 5 次 → 熔断 → 冷却后恢复 ──
        pub2 = BusEventPublisher(
            bus_file=os.path.join(tmpdir, "no_such_dir", "bus.jsonl"))
        pub2.breaker = CircuitBreaker(max_failures=3, cooldown_seconds=1.0)
        # 用一个必然失败的路径：bus_file 指向已存在目录
        pub3 = BusEventPublisher(bus_file=tmpdir)  # tmpdir 是目录 → open 失败
        pub3.breaker = CircuitBreaker(max_failures=3, cooldown_seconds=1.0)
        for i in range(3):
            pub3.publish_plastified({"entropy": 1.0})
        assert pub3.breaker.is_open(), "连续 3 次失败应熔断"
        st = pub3.breaker.status()
        assert st["tripped_count"] == 1 and st["open"] is True
        # 熔断中发布被拒（不新增文件、不抛异常）
        ok3 = pub3.publish_plastified({"entropy": 2.0})
        assert not ok3, "熔断中应拒绝发布"
        # 冷却结束 → 自动恢复尝试
        time.sleep(1.2)
        ok4 = pub3.publish_plastified({"entropy": 3.0})
        assert not ok4, "恢复后仍失败（路径仍错）但不再熔断中拒绝（半开放行）"
        assert not pub3.breaker.is_open(), "冷却结束后熔断应解除"
        # 修复路径 → 发布成功 → 失败计数清零
        pub4 = BusEventPublisher(bus_file=bus_file)
        pub4.breaker = CircuitBreaker(max_failures=3, cooldown_seconds=1.0)
        pub4.publish_plastified({"entropy": 1.0})  # 1 次失败（路径修复前不可能）
        pub4.bus_file = bus_file  # 修复路径
        pub4._writer = _BusWriter(bus_file)  # 重建写入器指向正确路径
        ok5 = pub4.publish_plastified({"entropy": 9.9})
        assert ok5, "路径修复后应发布成功"
        assert pub4.breaker._failures == 0, "成功后失败计数应清零"
        print("✅ 熔断：连续失败→熔断→冷却→恢复，全程不抛异常")

        # ── 5. dream_complete ──
        pub5 = BusEventPublisher(bus_file=bus_file)
        ok6 = pub5.publish_dream_complete({
            "steps": 20, "duration_seconds": 3.5, "snapshot_saved": True,
        })
        assert ok6
        recs = pub5._get_writer().read_all()
        dc = [r for r in recs if r["event_type"] == "lms.dream_complete"]
        assert len(dc) == 1 and dc[0]["payload"]["steps"] == 20
        print("✅ lms.dream_complete 发布成功")

    print("=" * 50)
    print("✅ bus_events.py 快速自测通过")
    print("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    quick_test()

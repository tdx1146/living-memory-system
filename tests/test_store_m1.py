# -*- coding: utf-8 -*-
"""M1-1 收尾单测：core/store 写侧提取层（ingest/feed/store 语义分离 + 幂等键机制）

规格依据：`四妹-LMS核心重写规格v2-20260817.md` §1.5 / §2.2 / §5.2。
本文件是 ``claims.json`` / ``MODULE_CLAIMS`` 全部 claim 的机器验证
（§5.2 D 模式：claim 与实现不一致 → 测试红）。

任务书四项 → 用例映射：
  ① 幂等键重发不双写     → test_idempotent_key_resend_no_double_write
                           + test_dedup_hit_returns_original_result（命中零副作用）
                           + test_failed_write_not_registered_retry_allowed（写成功才登记）
  ② 客户端超时重试竞态(60s) → test_timeout_retry_race_single_write（同键锁串行化，全程只写一次）
                           + test_window_expiry_allows_new_write（60s 是竞态防护不是永久去重）
                           + test_journal_replay_after_restart（窗口内重启重放，重试不双写）
  ③ feed/store 语义分离   → test_feed_store_semantic_separation
  ④ 注入时怀疑入口        → test_doubt_hook_fail_open（fail-open，绝不阻断写侧）
     （labile 内存态不落库） → test_doubt_hook_labile_not_persisted（怀疑信号只进内存态，绝不落库）

运行方式（任选其一，本文件不依赖 pytest fixtures）：
  - pytest rewrite-M1/tests/test_store_m1.py -v
  - .venv-m1/bin/python rewrite-M1/tests/test_store_m1.py   （纯 stdlib 兜底 runner）
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

# -- 路径引导：本文件可从任意 cwd 运行（纯 stdlib） -----------------------------
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.store import (  # noqa: E402
    IdempotencyRegistry,
    SemanticViolation,
    WriteRequest,
    WriteSemantics,
    feed,
    ingest,
    recallable,
    register_doubt_hook,
    store,
    store_gray_default,
    validate_semantics,
)

# ---------------------------------------------------------------------------
# 测试小工具（无 pytest 依赖）
# ---------------------------------------------------------------------------


def _writer_result(text: str = "hello", **over) -> dict:
    """与 api 层 StoreResponse 同构的 writer 返回（字段超集）。"""
    base = {
        "stored": True,
        "turn_count": 1,
        "value_filtered": False,
        "core_chars": len(text),
        "gray": False,
        "surprise": 0.3,
        "info_value": 0.5,
        "reason": "ok",
    }
    base.update(over)
    return base


def _counting_writer(calls):
    """记录每次真实写的 writer（calls 长度 = 实际写次数，断言不双写用）。"""
    def writer(req):
        calls.append(req)
        return _writer_result(req.text)
    return writer


@contextmanager
def raises(exc_type):
    """极简 pytest.raises 替代（纯 stdlib）。"""
    try:
        yield
    except exc_type:
        return
    raise AssertionError(f"期望抛出 {exc_type.__name__}，但未抛出")


def _ingest_mod():
    """core.store.ingest 子模块（包级属性 ingest 被入口函数遮蔽，用 importlib 取）。"""
    return importlib.import_module("core.store.ingest")


class _DoubtHooksGuard:
    """测试隔离：临时清空全局怀疑钩子，结束后恢复（避免用例间污染）。"""

    def __init__(self):
        self._mod = _ingest_mod()
        with self._mod._doubt_hooks_lock:
            self._saved = list(self._mod._doubt_hooks)
            self._mod._doubt_hooks.clear()

    def __enter__(self):
        return self._mod

    def __exit__(self, *exc):
        with self._mod._doubt_hooks_lock:
            self._mod._doubt_hooks[:] = self._saved
        return False


class _EnvGuard:
    """保存/恢复指定 env 变量（LMS_STORE_GRAY 灰度默认开关）。"""

    def __init__(self, *names):
        self._names = names
        self._saved = {n: os.environ.get(n) for n in names}

    def __enter__(self):
        for n in self._names:
            os.environ.pop(n, None)
        return self

    def __exit__(self, *exc):
        for n, v in self._saved.items():
            if v is None:
                os.environ.pop(n, None)
            else:
                os.environ[n] = v
        return False


# ---------------------------------------------------------------------------
# ① 幂等键重发不双写
# ---------------------------------------------------------------------------


def test_idempotent_key_resend_no_double_write():
    """同键重发不双写：第二次请求命中缓存原样返回，writer 只被调用一次。"""
    registry = IdempotencyRegistry(window=60.0)
    calls = []
    writer = _counting_writer(calls)
    req1 = WriteRequest(semantics=WriteSemantics.STORE, text="hello",
                        idempotency_key="k-1")
    req2 = WriteRequest(semantics=WriteSemantics.STORE, text="hello",
                        idempotency_key="k-1")
    res1 = ingest(req1, writer, registry=registry)
    res2 = ingest(req2, writer, registry=registry)
    assert res1.stored is True
    assert res1.dedup_hit is False
    assert res2.stored is True
    assert res2.dedup_hit is True
    assert len(calls) == 1, "同键重发必须不双写"
    assert res2.idempotency_key == "k-1"
    # 原样返回原结果：底层 writer 结果逐字段一致（dedup_hit/replayed 是
    # 命中元信息，语义上应当置位，不算结果差异）
    assert res2.data == res1.data
    assert (res2.turn_count, res2.core_chars, res2.surprise, res2.info_value,
            res2.reason) == (res1.turn_count, res1.core_chars, res1.surprise,
                             res1.info_value, res1.reason)


def test_failed_write_not_registered_retry_allowed():
    """写成功才登记：writer 抛异常 → 不登记；重试同键可重写（修复旧实现假 dedup_hit）。"""
    registry = IdempotencyRegistry(window=60.0)
    calls = []

    def flaky_writer(req):
        calls.append(req)
        if len(calls) == 1:
            raise RuntimeError("simulated write failure")
        return _writer_result(req.text)

    req = WriteRequest(semantics=WriteSemantics.STORE, text="hello",
                       idempotency_key="k-fail")
    with raises(RuntimeError):
        ingest(req, flaky_writer, registry=registry)
    assert registry.lookup("k-fail") is None, "写失败不得登记幂等键"
    res = ingest(req, flaky_writer, registry=registry)  # 重试同键 → 可重写
    assert res.stored is True
    assert res.dedup_hit is False
    assert len(calls) == 2
    assert registry.lookup("k-fail") is not None, "重写成功后才登记"


def test_dedup_hit_returns_original_result():
    """幂等命中路径零写入、零状态变更（无副作用）——dedup_hit 命中
    只是读缓存，writer 不执行、注册表不新增。"""
    registry = IdempotencyRegistry(window=60.0)
    calls = []
    writer = _counting_writer(calls)
    req1 = WriteRequest(semantics=WriteSemantics.STORE, text="hello",
                        idempotency_key="k-dedup")
    req2 = WriteRequest(semantics=WriteSemantics.STORE, text="hello",
                        idempotency_key="k-dedup")
    res1 = ingest(req1, writer, registry=registry)
    size_before = registry.size()
    res2 = ingest(req2, writer, registry=registry)
    assert res2.dedup_hit is True
    assert len(calls) == 1, "命中路径零写入"
    assert registry.size() == size_before, "命中路径零状态变更"
    assert res2.data == res1.data, "命中路径原样返回首次结果"


# ---------------------------------------------------------------------------
# ② 客户端超时重试竞态（60s 窗口）
# ---------------------------------------------------------------------------


def test_timeout_retry_race_single_write():
    """客户端超时 ≠ 未写入：首写在途时重试并发到达 → 同键锁串行化，
    重试等待首写完成并命中缓存，全程只写一次（60s 幂等竞态教训）。"""
    registry = IdempotencyRegistry(window=60.0)  # 60s 竞态防护窗口
    calls = []
    started = threading.Event()
    release = threading.Event()
    results = {}
    errors = []

    def slow_writer(req):
        calls.append(req)
        started.set()
        release.wait(timeout=10)  # 模拟慢写：客户端等不及 → 超时
        return _writer_result(req.text)

    def run(tag):
        try:
            req = WriteRequest(semantics=WriteSemantics.STORE, text="race",
                               idempotency_key="k-race")
            results[tag] = ingest(req, slow_writer, registry=registry)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    t1 = threading.Thread(target=lambda: run("first"), name="first-write")
    t1.start()
    assert started.wait(timeout=5), "首写未在 5s 内开始"
    t2 = threading.Thread(target=lambda: run("retry"), name="timeout-retry")
    t2.start()
    time.sleep(0.2)  # 给重试到达同键锁的时间（此时首写仍被锁内慢写挡住）
    assert len(calls) == 1, "首写完成前重试不得再次写（同键锁必须挡住）"
    release.set()
    t1.join(timeout=10)
    t2.join(timeout=10)
    assert not errors, f"竞态期间出现异常: {errors}"
    assert not t1.is_alive() and not t2.is_alive()
    assert len(calls) == 1, "同键并发必须全程只写一次"
    assert results["first"].stored is True
    assert results["first"].dedup_hit is False
    assert results["retry"].stored is True
    assert results["retry"].dedup_hit is True, "重试必须命中首次结果"
    # 原样返回原结果（dedup_hit 为命中元信息，理应置位）
    assert results["retry"].data == results["first"].data
    assert results["retry"].turn_count == results["first"].turn_count
    assert results["retry"].reason == results["first"].reason


def test_window_expiry_allows_new_write():
    """60s 窗口是竞态防护不是永久去重：窗口外同键视为新写（旧结果自然过期）。"""
    registry = IdempotencyRegistry(window=0.05)  # 缩短窗口加速测试（语义同 60s）
    calls = []

    def writer(req):
        calls.append(req)
        return _writer_result(req.text, turn_count=len(calls))

    req = WriteRequest(semantics=WriteSemantics.STORE, text="hello",
                       idempotency_key="k-exp")
    res1 = ingest(req, writer, registry=registry)
    assert res1.dedup_hit is False
    time.sleep(0.2)  # > window（0.05s）→ 记录过期
    res2 = ingest(req, writer, registry=registry)
    assert res2.dedup_hit is False, "窗口外同键 = 新写"
    assert len(calls) == 2


def test_journal_replay_after_restart():
    """journal 配置时窗口内重启后重放：重启后同键重试不双写（写侧默认保守）。"""
    with tempfile.TemporaryDirectory() as td:
        journal = os.path.join(td, "idem.journal")
        calls = []
        writer = _counting_writer(calls)
        # 第一次运行：写 + 登记（登记结果追加进 journal）
        r1 = IdempotencyRegistry(window=60.0, journal_path=journal)
        req = WriteRequest(semantics=WriteSemantics.STORE, text="hello",
                           idempotency_key="k-journal")
        res1 = ingest(req, writer, registry=r1)
        assert res1.stored is True and res1.dedup_hit is False
        # 模拟重启：新注册表同一 journal → 重放窗口内记录
        r2 = IdempotencyRegistry(window=60.0, journal_path=journal)
        assert r2.replayed_records == 1
        rec = r2.lookup("k-journal")
        assert rec is not None and rec.replayed is True
        res2 = ingest(req, writer, registry=r2)
        assert res2.dedup_hit is True, "重启后同键重试不双写"
        assert res2.stored is True
        assert res2.replayed is True, "命中 journal 重放记录（replayed 元信息）"
        assert res2.data == res1.data, "重启后原样返回首次结果"
        assert res2.turn_count == res1.turn_count
        assert len(calls) == 1
        # 窗口外记录不重放（竞态防护不是永久去重）
        time.sleep(0.15)
        r3 = IdempotencyRegistry(window=0.02, journal_path=journal)
        assert r3.replayed_records == 0
        assert r3.lookup("k-journal") is None


# ---------------------------------------------------------------------------
# ③ feed/store 语义分离
# ---------------------------------------------------------------------------


def test_feed_store_semantic_separation():
    """feed 永不灰化且 source 必须可召回；store 灰度走 store_gray
    （L1 不可召回——口径登记，非 bug）；store gray 默认读 env LMS_STORE_GRAY。"""
    with _EnvGuard("LMS_STORE_GRAY"):
        # -- feed：永不灰化 / 不可召回 source 直接拒绝 --
        with raises(SemanticViolation):
            WriteRequest(semantics=WriteSemantics.FEED, text="x", gray=True).resolve()
        with raises(SemanticViolation):
            WriteRequest(semantics=WriteSemantics.FEED, text="x",
                         source="store_gray").resolve()
        with raises(SemanticViolation):
            validate_semantics(WriteSemantics.FEED, None, True)
        r = WriteRequest(semantics=WriteSemantics.FEED, text="x").resolve()
        assert r.resolved_gray is False
        assert r.resolved_source == "external"
        assert recallable(r.resolved_source) is True
        # -- store：灰度口径保留 --
        g = WriteRequest(semantics=WriteSemantics.STORE, text="x",
                         gray=True).resolve()
        assert g.resolved_gray is True
        assert g.resolved_source == "store_gray"
        assert recallable(g.resolved_source) is False, "store_gray 不可 L1 召回（口径）"
        n = WriteRequest(semantics=WriteSemantics.STORE, text="x", gray=False,
                         source="memo").resolve()
        assert n.resolved_source == "memo"
        # -- store gray 默认读 env（LMS_STORE_GRAY，随时可关） --
        os.environ["LMS_STORE_GRAY"] = "1"
        assert store_gray_default() is True
        d = WriteRequest(semantics=WriteSemantics.STORE, text="x").resolve()
        assert d.resolved_gray is True and d.resolved_source == "store_gray"
        os.environ["LMS_STORE_GRAY"] = "0"
        assert store_gray_default() is False
        d2 = WriteRequest(semantics=WriteSemantics.STORE, text="x").resolve()
        assert d2.resolved_gray is False and d2.resolved_source == "external"
        # -- 便捷入口把解析口径传给 writer（写侧唯一权威） --
        seen = {}

        def capture_writer(req):
            seen["source"] = req.resolved_source
            seen["gray"] = req.resolved_gray
            return _writer_result(req.text, gray=req.resolved_gray)

        feed("补存知识", writer=capture_writer, registry=IdempotencyRegistry())
        assert seen == {"source": "external", "gray": False}
        store("普通存储", gray=True, writer=capture_writer,
              registry=IdempotencyRegistry())
        assert seen == {"source": "store_gray", "gray": True}


# ---------------------------------------------------------------------------
# ④ 注入时怀疑入口（labile 内存态不落库）
# ---------------------------------------------------------------------------


def test_doubt_hook_fail_open():
    """注入时怀疑钩子异常 fail-open，绝不阻断写侧；dedup 命中不重复触发。"""
    with _DoubtHooksGuard() as mod:
        assert mod is not None  # 子模块可达（包级 import 不受遮蔽影响）
        registry = IdempotencyRegistry(window=60.0)
        calls = []
        writer = _counting_writer(calls)
        hook_calls = []

        def good_hook(req, res):
            hook_calls.append((req.semantics.value, res.stored,
                               res.idempotency_key))

        def bad_hook(req, res):
            raise RuntimeError("怀疑逻辑故障（M3 未接入前不应存在，防御）")

        register_doubt_hook(good_hook)
        register_doubt_hook(bad_hook)
        req = WriteRequest(semantics=WriteSemantics.STORE, text="x",
                           idempotency_key="k-hook")
        res = ingest(req, writer, registry=registry)
        assert res.stored is True, "钩子异常 fail-open：写侧不被阻断"
        assert len(hook_calls) == 1
        assert hook_calls[0] == ("store", True, "k-hook")
        res2 = ingest(req, writer, registry=registry)  # dedup 命中
        assert res2.dedup_hit is True
        assert len(hook_calls) == 1, "dedup 命中不重复触发怀疑钩子"


def test_doubt_hook_labile_not_persisted():
    """注入时怀疑（§2.2）：怀疑信号是 labile 内存态——只进内存，
    绝不落库（幂等记录/写结果/响应不被怀疑标记污染）。"""
    with _DoubtHooksGuard():
        registry = IdempotencyRegistry(window=60.0)
        calls = []
        writer = _counting_writer(calls)
        labile = []  # 内存态怀疑信号（模拟 M3 的 labile 窗口）

        def suspect_hook(req, res):
            labile.append({"key": res.idempotency_key, "suspect": True})

        register_doubt_hook(suspect_hook)
        req = WriteRequest(semantics=WriteSemantics.STORE, text="x",
                           idempotency_key="k-labile")
        res = ingest(req, writer, registry=registry)
        assert len(labile) == 1, "钩子触发：怀疑信号进入内存态"
        assert labile[0]["suspect"] is True
        assert "suspect" not in res.data, "响应结果不被怀疑信号污染"
        rec = registry.lookup("k-labile")
        assert rec is not None
        assert rec.result == _writer_result("x"), \
            "幂等记录 = 纯 writer 结果（不落怀疑标记）"
        assert "suspect" not in rec.result


# ---------------------------------------------------------------------------
# 纯 stdlib 兜底 runner（无 pytest 环境时使用）
# ---------------------------------------------------------------------------


def _run_stdlib() -> None:
    import traceback
    funcs = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in funcs:
        t0 = time.time()
        try:
            fn()
            print(f"PASS  {name}  ({time.time() - t0:.2f}s)")
        except Exception:  # noqa: BLE001
            failures.append(name)
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(funcs) - len(failures)}/{len(funcs)} passed")
    if failures:
        print("FAILED:", ", ".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    _run_stdlib()

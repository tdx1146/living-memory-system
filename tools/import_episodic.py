#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMS 数据抢救工具 ③：import_episodic —— 导入清洗后的 JSONL 到 :8190
====================================================================
背景（总体方案 §6 / T2.5 第四步）：把清洗后的记忆文本喂入隔离会话
（推荐 main-v2）。本工具限流感知：尊重服务端 429/Retry-After，
失败条目落 fail-log，不中断整体。

端点选择:
  --endpoint feed（默认）: POST /feed {"text","session_id","source"}。
      塑形输入侧，不调 LLM，适合批量导入；但服务端有滑动窗口限流
      （LMS_FEED_RATE_LIMIT，默认 10 次/分钟）→ 本工具按 --rate 限速，
      429 时按 Retry-After 等待重试。
  --endpoint chat: POST /chat {"user_input","session_id"}。
      process_turn 全流程；若 .env 配置了 LLM 会逐条调 LLM（慢且花钱），
      批量导入慎用。llm_output 留空。

用法:
    python tools/import_episodic.py --in /tmp/export_clean.jsonl \
        --session-id main-v2 --endpoint feed --rate 8
    python tools/import_episodic.py --in /tmp/export_clean.jsonl --dry-run
    python tools/import_episodic.py --in /tmp/export_clean.jsonl \
        --endpoint chat --rate 20 --fail-log /tmp/import_fail.jsonl

参数:
    --in        输入 JSONL（默认 export_clean.jsonl；'-' = stdin）
    --base-url  API 地址（默认 http://127.0.0.1:8190）
    --session-id 目标会话（默认 main-v2，隔离导入）
    --endpoint  feed|chat（默认 feed）
    --rate      每分钟最大请求数（feed 默认 8，chat 默认 20；
                环境变量 LMS_FEED_RATE_LIMIT 可作 feed 默认参考）
    --max-retries 429/503 重试次数（默认 3）
    --fail-log  失败条目记录文件（默认 logs/import_episodic_fail.jsonl）
    --dry-run   只打印导入计划，不发任何请求

退出码: 0=全部成功或 dry-run 2=参数错误 3=部分失败（见 fail-log）。
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASE = "http://127.0.0.1:8190"
DEFAULT_FAIL_LOG = os.path.join(PROJECT_ROOT, "logs", "import_episodic_fail.jsonl")


def _post_json(url: str, payload: dict, timeout: float = 15.0):
    """POST JSON，返回 (status, body_dict)；网络错误抛 URLError。"""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                return resp.status, json.loads(body) if body else {}
            except json.JSONDecodeError:
                return resp.status, {"raw": body[:200]}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = {"raw": body[:200]}
        retry_after = e.headers.get("Retry-After") if e.headers else None
        if retry_after:
            try:
                parsed["retry_after"] = float(retry_after)
            except ValueError:
                parsed["retry_after"] = None
        return e.code, parsed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="导入清洗后的 episodic JSONL 到 LMS :8190（T2.5 抢救第三步）")
    ap.add_argument("--in", dest="in_path", default=None,
                    help="输入 JSONL（默认 export_clean.jsonl；'-' = stdin）")
    ap.add_argument("--base-url", default=os.environ.get("LMS_API_BASE", DEFAULT_BASE),
                    help=f"API 地址（默认 {DEFAULT_BASE}）")
    ap.add_argument("--session-id", default="main-v2",
                    help="目标会话（默认 main-v2，隔离导入不扰动现役脑）")
    ap.add_argument("--endpoint", choices=["feed", "chat"], default="feed")
    ap.add_argument("--rate", type=int, default=None,
                    help="每分钟最大请求数（feed 默认 8，chat 默认 20）")
    ap.add_argument("--max-retries", type=int, default=3,
                    help="429/503 重试次数（默认 3）")
    ap.add_argument("--fail-log", default=DEFAULT_FAIL_LOG,
                    help=f"失败条目记录（默认 {DEFAULT_FAIL_LOG}；'' 禁用）")
    ap.add_argument("--dry-run", action="store_true", help="只打印计划，不发请求")
    args = ap.parse_args()

    # 默认限速：feed 参照服务端 LMS_FEED_RATE_LIMIT（默认 10/分钟）留裕量；
    # chat 无服务端限流，但逐条 process_turn 较重，取 20/分钟。
    if args.rate is None:
        if args.endpoint == "feed":
            server_limit = int(os.environ.get("LMS_FEED_RATE_LIMIT", "10"))
            args.rate = max(1, server_limit - 2)
        else:
            args.rate = 20

    in_path = args.in_path or os.path.join(PROJECT_ROOT, "export_clean.jsonl")
    if in_path == "-":
        lines = sys.stdin
    else:
        if not os.path.isfile(in_path):
            print(f"[import] 输入文件不存在: {in_path}", file=sys.stderr)
            return 2
        lines = open(in_path, encoding="utf-8", errors="replace")
    rows = []
    try:
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    finally:
        if in_path != "-":
            lines.close()

    if not rows:
        print("[import] 输入为空，无可导入条目")
        return 0

    url = f"{args.base_url.rstrip('/')}/{'feed' if args.endpoint == 'feed' else 'chat'}"
    print(f"[import] 计划: {len(rows)} 条 → {args.endpoint} {url} "
          f"session={args.session_id} rate={args.rate}/min")
    if args.dry_run:
        print("[import] --dry-run：未发送任何请求")
        return 0

    ok = 0
    failed = 0
    fails = []
    interval = 60.0 / args.rate
    next_slot = time.monotonic()
    t0 = time.time()

    for i, row in enumerate(rows):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        payload = (
            {"text": text, "session_id": args.session_id,
             "source": row.get("source", "rescue_import")}
            if args.endpoint == "feed"
            else {"user_input": text, "session_id": args.session_id, "llm_output": ""}
        )
        # 限速：均匀间隔（简单令牌桶）
        now = time.monotonic()
        if now < next_slot:
            time.sleep(next_slot - now)
        next_slot = max(next_slot + interval, time.monotonic())

        status, body = _post_json(url, payload)
        attempts = 1
        while status in (429, 503) and attempts <= args.max_retries:
            ra = body.get("retry_after") if isinstance(body, dict) else None
            wait = ra if isinstance(ra, (int, float)) and ra > 0 else 5.0
            print(f"[import][{i + 1}/{len(rows)}] HTTP {status} 限流/服务不可用，"
                  f"等待 {wait:.0f}s 重试（{attempts}/{args.max_retries}）")
            time.sleep(min(wait, 30.0))
            status, body = _post_json(url, payload)
            attempts += 1

        if status == 200:
            ok += 1
            if ok % 20 == 0:
                print(f"[import] 进度 {ok}/{len(rows)}（{time.time() - t0:.0f}s）")
        else:
            failed += 1
            fails.append({
                "ts": time.time(),
                "status": status,
                "session_id": args.session_id,
                "endpoint": args.endpoint,
                "text": text,
                "response": body if isinstance(body, dict) else {"raw": str(body)[:200]},
            })
            print(f"[import][WARN] 第 {i + 1} 条失败（HTTP {status}）: {text[:40]!r}")

    # 失败条目落盘（append-only，不覆盖历史）
    if fails and args.fail_log:
        os.makedirs(os.path.dirname(args.fail_log) or ".", exist_ok=True)
        with open(args.fail_log, "a", encoding="utf-8") as f:
            for fl in fails:
                f.write(json.dumps(fl, ensure_ascii=False) + "\n")
        print(f"[import] 失败条目已记录: {args.fail_log}")

    dur = time.time() - t0
    print(f"[import] 完成: 成功 {ok}，失败 {failed}，耗时 {dur:.0f}s "
          f"（{ok / max(dur, 0.1) * 60:.1f} 条/分钟）")
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    sys.exit(main())

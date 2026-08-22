#!/usr/bin/env python3
"""lms_dsh_feed.py — DSH 会话 → LMS 自动喂入 watcher（2026-08-22）

背景：dandan 拍板"每轮自动 lms_store"。LMS 的 /store 带 gray 标记（外部检索
不可见），/chat 才是可检索的写路径。本脚本扫描 DSH 的 session.jsonl.zstd
（zstd 流式解压），把「用户消息 + 对应助手最终文本」配对 POST /chat 喂入
LMS main 脑。幂等：按 (session 文件, seq) 水位线推进；失败不推进，下轮重试。

用法（cron 每 2 分钟）：
  */2 * * * * /vol2/1000/AI专用/living-memory-system-cloud/.venv/bin/python \
      /vol2/1000/AI专用/living-memory-system-cloud/scripts/lms_dsh_feed.py \
      >> /vol2/1000/AI专用/living-memory-system-cloud/logs/lms_dsh_feed.log 2>&1
"""
import glob
import json
import os
import sys
import time

import requests
import zstandard

DSH_SESSIONS_DIR = "/vol1/@apphome/trim.openclaw/data/home/.dsh/sessions"
WORKSPACE_DIR = "--vol1-~0040apphome-trim.openclaw-data-dsh-package--"
LMS_API_URL = "http://localhost:8190"
WATERMARK_FILE = "/vol2/1000/AI专用/living-memory-system-cloud/runtime/lms_dsh_feed_watermark.json"
SID = "main"
USER_TEXT_MAX = 2000
ASST_TEXT_MAX = 2500
BACKFILL_MAX_TURNS = 20  # 首次见到某会话时最多回填的对话轮数


def find_newest_session_file():
    """返回最新被修改的 session.jsonl.zstd 绝对路径（工作区会话目录内）。"""
    base = os.path.join(DSH_SESSIONS_DIR, WORKSPACE_DIR)
    candidates = glob.glob(os.path.join(base, "session-*", "session.jsonl.zstd"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def read_events(path):
    """流式解压 zstd，返回全部 JSON 对象（带 seq）。"""
    dctx = zstandard.ZstdDecompressor()
    with open(path, "rb") as f, dctx.stream_reader(f) as r:
        text = r.read().decode("utf-8", errors="replace")
    events = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events


def extract_text(obj, kind):
    """从 user/message 或 assistant/message 中提取纯文本。"""
    if kind == "user":
        content = (obj.get("data") or {}).get("content") or []
        return "".join(c.get("text", "") for c in content if c.get("type") == "text")
    # assistant: data.message.content = [{type: reasoning|text|tool-call}, ...]
    msg = (obj.get("data") or {}).get("message") or {}
    content = msg.get("content") or []
    return "".join(c.get("text", "") for c in content if c.get("type") == "text")


def chat(user_input, llm_output):
    """POST /chat；503（做梦）抛异常由调用方处理。"""
    r = requests.post(
        f"{LMS_API_URL}/chat",
        json={"session_id": SID, "user_input": user_input, "llm_output": llm_output},
        timeout=30,
    )
    if r.status_code == 503:
        raise RuntimeError(f"503 dreaming: {r.text[:120]}")
    r.raise_for_status()
    return r.json()


def main():
    path = find_newest_session_file()
    if not path:
        print(f"[{time.strftime('%H:%M:%S')}] no session file found")
        return 0

    watermark = {}
    if os.path.exists(WATERMARK_FILE):
        try:
            with open(WATERMARK_FILE) as f:
                watermark = json.load(f)
        except Exception:
            watermark = {}

    prev_file = watermark.get("file")
    prev_seq = int(watermark.get("seq", 0) or 0)
    backfilled = watermark.get("backfilled", False)

    events = read_events(path)
    if prev_file != path:
        # 新会话：只回填最近 N 条用户消息，避免首次全量冲刷
        prev_seq = 0
        backfilled = False

    pending_user = None   # (seq, text)
    posted = 0
    max_seq = prev_seq
    turns = 0

    for ev in events:
        seq = ev.get("seq")
        if seq is None:
            continue
        max_seq = max(max_seq, seq)
        if seq <= prev_seq:
            continue
        etype = ev.get("type", "")

        if etype == "user/message":
            # 只喂真用户消息（source.kind == 'user'），跳过系统注入的
            # runtime context 快照等 harness 噪音
            src = ((ev.get("data") or {}).get("source") or {}).get("kind")
            if src != "user":
                continue
            text = extract_text(ev, "user").strip()
            if text:
                pending_user = (seq, text[:USER_TEXT_MAX])
        elif etype == "assistant/message":
            text = extract_text(ev, "assistant").strip()
            if not text or pending_user is None:
                continue
            if not backfilled and turns >= BACKFILL_MAX_TURNS:
                # 首次回填已达上限：推进水位但不 POST 剩余
                continue
            u_seq, u_text = pending_user
            try:
                chat(u_text, text[:ASST_TEXT_MAX])
                posted += 1
                turns += 1
                print(f"[{time.strftime('%H:%M:%S')}] fed turn (u:{u_seq}->a:{seq}): "
                      f"{u_text[:40]!r}...")
                # 推进水位到当前 assistant seq
                watermark = {"file": path, "seq": seq,
                             "backfilled": backfilled or turns >= BACKFILL_MAX_TURNS}
                with open(WATERMARK_FILE, "w") as f:
                    json.dump(watermark, f)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] POST failed at seq {seq}: {e}; "
                      f"水位不推进，下轮重试")
                return 1
            pending_user = None

    # 没有新内容也把水位对齐到最新 seq（跳过 chunk/tool 等噪音事件）
    if max_seq > prev_seq and posted == 0:
        watermark = {"file": path, "seq": max_seq,
                     "backfilled": backfilled}
        with open(WATERMARK_FILE, "w") as f:
            json.dump(watermark, f)
        print(f"[{time.strftime('%H:%M:%S')}] no feedable pairs; watermark -> {max_seq}")

    print(f"[{time.strftime('%H:%M:%S')}] done: posted={posted} file={os.path.basename(os.path.dirname(path))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

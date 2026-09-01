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
# [2026-08-30 修复] V1 LMS :8190 已停用（四妹 V2 切换：agentos-v2/lms-api 监听 :8191）。
# 旧硬编码导致本 watcher 每 2 分钟失败（当日 1426 次）→ 四妹的对话进不了共享脑。
# 另：V1 的 /chat（可检索写路径）在 V2 不存在；V2 的 /store 就是完整写入路径
# （走 ingest→encode→retrieve→commit 全生命周期，entries 增长且可检索，已实测），
# payload 字段与 /chat 同构（session_id/user_input/llm_output）。两者均可 env 覆盖。
LMS_API_URL = os.environ.get("LMS_URL", "http://localhost:8191")
LMS_WRITE_PATH = os.environ.get("LMS_WRITE_PATH", "/store")
WATERMARK_FILE = "/vol2/1000/AI专用/living-memory-system-cloud/runtime/lms_dsh_feed_watermark.json"
SID = "main"
USER_TEXT_MAX = 2000
ASST_TEXT_MAX = 2500
BACKFILL_MAX_TURNS = 20  # 首次见到某会话时最多回填的对话轮数


def find_newest_session_file():
    """返回最新被修改的 session.jsonl.zstd 绝对路径。
    2026-08-25 改造：遍历 sessions 下所有工作区目录（不再硬编码单一工作区），
    使新建工作区（如 AI专用）的会话也能被喂入 LMS。"""
    base = DSH_SESSIONS_DIR
    candidates = glob.glob(os.path.join(base, "*", "session-*", "session.jsonl.zstd"))
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
    """POST 写入端点（V1 /chat → V2 /store，env LMS_WRITE_PATH 可覆盖）；503（做梦）抛异常由调用方处理。"""
    r = requests.post(
        f"{LMS_API_URL}{LMS_WRITE_PATH}",
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
    # [C7] 水位只推进到最后一个成功 POST 的 assistant seq（未配对 user 不跳）
    last_paired_seq = prev_seq
    cap_hit = False   # [C8] 本轮是否触达回填上限（触达 → 剩余轮次不 POST、不推进水位）

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
                # [C8] 达上限：本轮剩余轮次跳过 POST 且**不推进水位**（下轮继续按上限回填）——
                # 上限语义是"每轮最多 N 轮"，不是"全局只回填 N 轮"。
                # 原实现"推进水位 + 置 backfilled=True"：① 置 True 反而解除上限 → 同轮全量冲刷；
                # ② 水位越过未 POST 的轮次 → 该批对话永久丢失。修复：跳过但水位不动。
                cap_hit = True
                pending_user = None
                continue
            u_seq, u_text = pending_user
            try:
                chat(u_text, text[:ASST_TEXT_MAX])
                posted += 1
                turns += 1
                print(f"[{time.strftime('%H:%M:%S')}] fed turn (u:{u_seq}->a:{seq}): "
                      f"{u_text[:40]!r}...")
                # [C7] 水位只推进到当前已配对 assistant seq
                last_paired_seq = seq
                # [C8] backfilled 只在"完整扫完"时置 True（见收尾段）；触上限的轮次保持 False
                watermark = {"file": path, "seq": seq,
                             "backfilled": backfilled}
                with open(WATERMARK_FILE, "w") as f:
                    json.dump(watermark, f)
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] POST failed at seq {seq}: {e}; "
                      f"水位不推进，下轮重试")
                return 1
            pending_user = None

    # [C8] 完整扫完未触上限（posted>0 且全程未触发 cap 分支）→ 初始回填已排空，
    # 置 backfilled=True（解除每轮上限；只有"完整扫完"才置 True，触上限的轮次保持 False）
    if not backfilled and posted > 0 and not cap_hit:
        backfilled = True
        watermark = {"file": path, "seq": last_paired_seq, "backfilled": True}
        with open(WATERMARK_FILE, "w") as f:
            json.dump(watermark, f)
        print(f"[{time.strftime('%H:%M:%S')}] backfill complete; watermark -> {last_paired_seq}")

    # [C7] 无配对可喂（posted==0，如末尾是未配对 user / 纯噪音事件）：
    # 水位只推进到最后一个已配对 assistant seq（即 prev_seq），绝不跳到未配对 user 的 seq
    # ——未配对 user 下轮续配（seq 未推进 → 下轮重新扫描该事件），该轮对话不再永久丢失
    if max_seq > prev_seq and posted == 0:
        watermark = {"file": path, "seq": last_paired_seq,
                     "backfilled": backfilled}
        with open(WATERMARK_FILE, "w") as f:
            json.dump(watermark, f)
        print(f"[{time.strftime('%H:%M:%S')}] no feedable pairs; watermark 停在已配对 seq {last_paired_seq}")

    print(f"[{time.strftime('%H:%M:%S')}] done: posted={posted} file={os.path.basename(os.path.dirname(path))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""lms_feed_docs_chunked.py — 分块喂文档（2026-08-22，修复 bge-m3 长文本嵌入稀释）。

早前整段喂入（~1000 字）实测余弦仅 0.416（被稀释），必须分块 ≤300 字。
每块走 /chat 入库（可检索路径）。幂等：记录已喂 (文件, 块号)。
"""
import json
import os
import re
import sys
import time

import requests

WS = "/vol1/@apphome/trim.openclaw/data/dsh/package/"
STATE = "/vol2/1000/AI专用/living-memory-system-cloud/runtime/lms_feed_docs_state.json"
CHUNK = 260  # 每块字符数（bge-m3 对 250 字级文本嵌入稳定）
DOCS = [
    "LMS系统全面说明-20260822.md",
    "全局审视-LMS与AgentOS-给dandan-20260822.md",
]


def split_chunks(text: str, size: int = CHUNK):
    """按句子边界分块；无句边界的超长段按硬长度二次切分（防文档尾部永久丢失）。"""
    parts = re.split(r"(?<=[。！？!?\n])", text)
    chunks, cur = [], ""
    for p in parts:
        # [C9] 单段超长（无句边界，如超长代码/URL 段）→ 按硬长度二次切分，
        # 否则原实现 chunk[:CHUNK] 截断后剩余部分被丢弃 → 文档尾部永久丢失
        while len(p) > size:
            if cur:
                chunks.append(cur.strip())
                cur = ""
            chunks.append(p[:size].strip())
            p = p[size:]
        if len(cur) + len(p) > size and cur:
            chunks.append(cur.strip())
            cur = p
        else:
            cur += p
    if cur.strip():
        chunks.append(cur.strip())
    return [c for c in chunks if c]


def main():
    state = {}
    if os.path.exists(STATE):
        try:
            with open(STATE) as f:
                state = json.load(f)
        except Exception:
            state = {}
    ok, total, failed = 0, 0, 0
    for fname in DOCS:
        try:
            with open(WS + fname) as f:
                text = f.read()
        except Exception as e:
            print(f"SKIP {fname}: {e}", flush=True)
            continue
        chunks = split_chunks(text)
        done = set(state.get(fname, []))
        for i, chunk in enumerate(chunks):
            if i in done:
                continue
            user_input = f"【文档喂入·分块 {i}】{fname}\n{chunk[:CHUNK]}"
            try:
                r = requests.post(
                    "http://localhost:8190/chat",
                    json={"session_id": "main", "user_input": user_input,
                          "llm_output": "（文档喂入分块）"},
                    timeout=40,
                )
                r.raise_for_status()
                done.add(i)
                state[fname] = sorted(done)
                with open(STATE, "w") as f:
                    json.dump(state, f)
                ok += 1
                total += 1
                print(f"OK {fname} chunk {i} ({len(chunk)} chars)", flush=True)
            except Exception as e:
                print(f"FAIL {fname} chunk {i}: {e}", flush=True)
                failed += 1   # [C9] 失败计数：任何块失败 → 退出码非零，运维可见（原实现失败也 exit 0）
                time.sleep(2)
            time.sleep(0.5)
    print(f"done: fed={ok} total_chunks={total} failed={failed}", flush=True)
    return 1 if failed else 0   # [C9] 失败必须非零退出；失败块不写 done → 下轮重试


if __name__ == "__main__":
    sys.exit(main())

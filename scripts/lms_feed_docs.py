#!/usr/bin/env python3
"""lms_feed_docs.py — 把工作区关键文档喂入 LMS main 脑（2026-08-22）"""
import re
import sys
import time

import requests

WS = "/vol1/@apphome/trim.openclaw/data/dsh/package/"
DOCS = [
    ("四妹-LMS核心重写规格v2-20260817.md", "总纲：", 900),
    ("审视-A2-surprise未回升-20260821.md", "一句话摘要", 900),
    ("调研-最新DSH插件异地登录-20260822.md", "结论先行", 900),
    ("评估-LMS与AgentOS真实价值-给dandan-20260822.md", "一句话结论", 900),
    ("交接-LMS接线与对照实验-20260822.md", "已完成", 700),
    ("对照实验-LMS-vs-grep-20260822.md", "逐题结果", 700),
]


def main():
    ok = 0
    for fname, anchor, length in DOCS:
        try:
            with open(WS + fname) as f:
                text = f.read()
        except Exception as e:
            print(f"SKIP {fname}: {e}", flush=True)
            continue
        m = re.search(re.escape(anchor), text) or re.search(anchor, text)
        start = m.start() if m else 0
        excerpt = re.sub(r"\n{2,}", "\n", text[start:start + length]).strip()
        user_input = f"【文档喂入】{fname}\n{excerpt}"
        try:
            r = requests.post(
                "http://localhost:8190/chat",
                json={"session_id": "main", "user_input": user_input,
                      "llm_output": "（文档喂入完成，来源：" + fname + "）"},
                timeout=40,
            )
            r.raise_for_status()
            ok += 1
            print(f"OK {fname} ({len(user_input)} chars)", flush=True)
        except Exception as e:
            print(f"FAIL {fname}: {e}", flush=True)
        time.sleep(1)
    print(f"done: ok={ok}/{len(DOCS)}", flush=True)
    return 0 if ok == len(DOCS) else 1


if __name__ == "__main__":
    sys.exit(main())

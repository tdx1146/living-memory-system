#!/usr/bin/env python3
"""migrate_gray_visible.py — 灰池可见性迁移（2026-08-22，dandan 拍板"结束灰度"）。

把 store_gray 条目中**非垃圾**的（真实对话）source 改为 external（检索可见），
垃圾条目（中文/英文子代理样板等）保持 gray（不可见但保留，丰碑哲学不丢）。

前置：LMS 服务已停止（避免快照双写）。
强制前置（fail-closed）：入口探测 :8190，live 可达即退出（见 main()）。
用法：env LMS_SNAPSHOT_DIR=... .venv/bin/python tools/migrate_gray_visible.py
"""
import sys
import urllib.request
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from api.session_manager import SessionManager  # noqa: E402
from api.config import get_api_config  # noqa: E402
from core.hippocampus.memory import _is_garbage_text  # noqa: E402


def main():
    # [C14] fail-closed：原"前置：服务已停止"仅为文档声明，未强制——
    # live 运行时本工具会与 live 进程双写同一快照（竞态/撕裂）。
    # 入口先探测 :8190，可达即报错退出（fail-closed）。
    try:
        with urllib.request.urlopen("http://127.0.0.1:8190/health", timeout=5) as r:
            if r.status == 200:
                print("❌ LMS :8190 正在运行：灰池迁移会与 live 双写竞态，请先停服")
                return 1
    except Exception:
        pass  # 不可达 → 允许执行
    sm = SessionManager(default_config_factory=get_api_config)
    loop = sm.get_or_create("main")
    entries = list(loop.memory.iter_episodic())
    gray = [e for e in entries if getattr(e, "source", "") == "store_gray"]
    flipped = 0
    kept_gray = 0
    for e in gray:
        text = getattr(e, "text", "") or ""
        if _is_garbage_text(text):
            kept_gray += 1
            continue
        e.source = "external"
        e.gray = False
        flipped += 1
    print(f"total={len(entries)} gray={len(gray)} flipped_to_external={flipped} kept_gray_junk={kept_gray}")
    path = loop.save_session_state()
    print(f"snapshot saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

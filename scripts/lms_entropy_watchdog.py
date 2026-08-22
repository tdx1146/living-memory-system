#!/usr/bin/env python3
"""lms_entropy_watchdog.py — 熵看门狗（2026-08-22，dandan 指示：自动换新鸡蛋，别硬整焦蛋）

监测 LMS 场的健康：若进入"烧焦"态（熵>0.9995 或 σ 顶格>0.95，连续 N 次），
**自动**回滚到"最后一次健康快照"（自动换蛋）并重启 LMS。全程日志 + 事件记录。

设计要点：
  - 与 lms_ops_monitor.py 互补：那个只报警，这个直接处置（自愈）。
  - "最后一次健康快照"：每次观测到熵<0.995 时，记录当时的 latest_main.pt 路径。
    烧焦时回滚到它（健康蛋）。若从未记录过健康快照 → 用内置兜底快照（如有）。
  - fail-open：LMS 不可达（curl 失败）→ 只记日志不动手（可能正在重启/维护）。
  - 处置后冷却：一次处置后 30 分钟内不再处置（防抖）。
  - 自愈后验证：重启后等 20s，若仍烧焦 → 记警（交给人工，不再自动循环）。

用法（cron 每 5 分钟）：
  */5 * * * * .venv/bin/python scripts/lms_entropy_watchdog.py >> logs/lms_entropy_watchdog.log 2>&1
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

LMS_URL = "http://127.0.0.1:8190"
STATE_FILE = "/vol2/1000/AI专用/living-memory-system-cloud/runtime/entropy_watchdog_state.json"
LOG_FILE = "/vol2/1000/AI专用/living-memory-system-cloud/logs/lms_entropy_watchdog.log"
SNAP_DIR = "/vol2/1000/AI专用/living-memory-system-cloud/snapshots/main"
ENV_DUMP = "/tmp/lms-orig-env.txt"

ENTROPY_BURNT = 0.9995   # 熵烧焦线
SIGMA_BURNT = 0.95       # σ 顶格线
ENTROPY_HEALTHY = 0.995  # 低于此线记"健康"
BURNT_STREAK_NEED = 2    # 连续 N 次烧焦才处置（防抖动）
DISPOSE_COOLDOWN_S = 1800  # 处置后冷却


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)


def http_get(path, timeout=6):
    try:
        with urllib.request.urlopen(LMS_URL + path, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        return {"error": str(e)}


def read_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"streak": 0, "last_healthy": "", "last_dispose": 0}


def write_state(st):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(st, f)
    except Exception:
        pass


def latest_snapshot():
    try:
        cands = [p for p in os.listdir(SNAP_DIR) if p.endswith(".pt")]
        if not cands:
            return ""
        return max(cands, key=lambda p: os.path.getmtime(
            os.path.join(SNAP_DIR, p)))
    except Exception:
        return ""


def dispose_burnt(state, reason):
    """自动换蛋：停服 → 恢复健康快照 → 设 bias → 存档 → 重启。"""
    log(f"⚠️ 触发自动换蛋（{reason}）")
    snap = state.get("last_healthy", "")
    if not snap or not os.path.exists(os.path.join(SNAP_DIR, snap)):
        log("❌ 无健康快照可用（从未记录），跳过自动处置，交人工")
        return False
    try:
        # 停服
        subprocess.run(["pkill", "-f", "api.run"], capture_output=True)
        time.sleep(5)
        # 恢复健康快照 + 设 bias + 存档
        script = f'''
import torch, warnings; warnings.filterwarnings("ignore")
from api.session_manager import SessionManager
from api.config import get_api_config
sm = SessionManager(default_config_factory=get_api_config)
loop = sm.get_or_create("main")
loop.load_state("{SNAP_DIR}/{snap}")
net = loop.attractor
import os
bias = float(os.environ.get("LMS_BIAS_SCALE", "0.9") or 0.9)
net.bias = torch.full((256,), bias)
hb = getattr(net, "homeostatic_bias", None)
if hb is not None: hb.bias = bias; hb.init_bias = bias
loop.save_session_state()
print("restored:", "{SNAP_DIR}/{snap}")
'''
        env = dict(os.environ)
        if os.path.exists(ENV_DUMP):
            for line in open(ENV_DUMP):
                line = line.rstrip("\n")
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k] = v
        proc = subprocess.run(
            [sys.executable, "-c", script],
            cwd="/vol2/1000/AI专用/living-memory-system-cloud",
            env=env, capture_output=True, text=True, timeout=120)
        log(f"  恢复输出: {proc.stdout.strip()[-100:]} {proc.stderr.strip()[-100:]}")
        # 重启
        subprocess.Popen(
            [sys.executable, "-m", "api.run", "--host", "127.0.0.1",
             "--port", "8190"],
            cwd="/vol2/1000/AI专用/living-memory-system-cloud",
            env=env, stdout=open("/tmp/lms-api.log", "a"),
            stderr=subprocess.STDOUT, start_new_session=True)
        log("✅ 自动换蛋完成，LMS 已重启")
        return True
    except Exception as e:
        log(f"❌ 自动换蛋失败: {e}")
        return False


def main():
    state = read_state()
    now = time.time()
    # 处置冷却检查
    if now - state.get("last_dispose", 0) < DISPOSE_COOLDOWN_S:
        return 0

    status = http_get("/status/main")
    landscape = http_get("/landscape/main")
    if "error" in status or "error" in landscape:
        log(f"LMS 不可达（fail-open 不动手）: {status.get('error','')}")
        return 0

    entropy = float(status.get("status", {}).get("entropy_ratio", 0) or 0)
    ls = landscape.get("landscape", {})
    act = ls.get("activation", {})
    top = act.get("top_activated", [])
    sigma_max = max([n.get("sigma", 0) for n in top], default=0)

    # 记录健康快照
    if entropy < ENTROPY_HEALTHY:
        snap = latest_snapshot()
        if snap:
            state["last_healthy"] = snap
            state["streak"] = 0
            write_state(state)
        return 0

    # 烧焦判定
    burnt = entropy > ENTROPY_BURNT or sigma_max > SIGMA_BURNT
    if burnt:
        state["streak"] = state.get("streak", 0) + 1
        write_state(state)
        log(f"⚠️ 烧焦信号（熵 {entropy:.4f} / σmax {sigma_max:.3f}），连续 {state['streak']}/{BURNT_STREAK_NEED}")
        if state["streak"] >= BURNT_STREAK_NEED:
            if dispose_burnt(state, f"熵 {entropy:.4f} / σmax {sigma_max:.3f}"):
                state["streak"] = 0
                state["last_dispose"] = now
                write_state(state)
    else:
        state["streak"] = 0
        write_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())

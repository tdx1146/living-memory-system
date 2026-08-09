#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LMS 活体记忆系统 - 自我配置端口（Control Plane）启动脚本

用法:
    python scripts/run_control.py
    python scripts/run_control.py --port 8191 --host 127.0.0.1

命令行参数:
    --host    监听地址（默认读取 LMS_CTRL_HOST 或 127.0.0.1）
    --port    监听端口（默认读取 LMS_CTRL_PORT 或 8191）

说明:
    * api.control 导入时会自动加载项目 .env（含 LMS_CONTROL_TOKEN），
      无需手动 source；token 缺失 → 只读模式（写端点 503，见 api/control.py）。
    * 本脚本只启动管理面；数据面 :8190 由 api/run.py / lms-api.service 负责。
"""

import os
import sys
import argparse

# 确保项目根目录在 sys.path 中（支持任意 cwd 启动）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 切换工作目录到项目根，使 logs/ data/ 等相对路径落在项目内
os.chdir(_PROJECT_ROOT)


def main() -> None:
    """解析参数并启动 uvicorn 控制面服务。"""
    parser = argparse.ArgumentParser(
        description="LMS 自我配置端口（Control Plane :8191）启动脚本")
    parser.add_argument("--host", type=str, default=None,
                        help="监听地址（默认 LMS_CTRL_HOST 或 127.0.0.1）")
    parser.add_argument("--port", type=int, default=None,
                        help="监听端口（默认 LMS_CTRL_PORT 或 8191）")
    args = parser.parse_args()

    # 延迟导入：让 .env 加载（api.control 导入时完成）先于配置读取
    from api.control import app, CTRL_HOST, CTRL_PORT, READ_ONLY_MODE

    host = args.host or os.environ.get("LMS_CTRL_HOST", CTRL_HOST)
    port = args.port or int(os.environ.get("LMS_CTRL_PORT", CTRL_PORT))

    import uvicorn
    print("=" * 56)
    print("LMS Control Plane（自我配置端口）")
    print(f"  地址: http://{host}:{port}")
    print(f"  文档: http://{host}:{port}/docs  |  OpenAPI: /openapi.json")
    print(f"  模式: {'只读（LMS_CONTROL_TOKEN 未配置）' if READ_ONLY_MODE else '完整（token 已配置）'}")
    print("  提示: 写端点需 X-Control-Token 请求头；审计写 logs/control-audit.jsonl")
    print("=" * 56)
    uvicorn.run(
        app,
        host=host,
        port=port,
        timeout_graceful_shutdown=10,
    )


if __name__ == "__main__":
    main()

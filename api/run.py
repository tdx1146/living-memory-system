"""活体记忆系统 API - 启动脚本

用法:
    python -m api.run
    python api/run.py
    python api/run.py --host 0.0.0.0 --port 8190 --reload

命令行参数:
    --host    监听地址（默认 127.0.0.1，可由 LMS_API_HOST 环境变量覆盖）
    --port    监听端口（默认 8190，可由 LMS_API_PORT 环境变量覆盖）
    --reload  开发热重载
"""

import os
import sys
import argparse

# 确保项目根目录在 sys.path 中（支持 python api/run.py 启动方式）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 切换工作目录到项目根，使 ./snapshots 等相对路径落在项目内
os.chdir(_PROJECT_ROOT)


def main() -> None:
    """解析参数并启动 uvicorn 服务。"""
    parser = argparse.ArgumentParser(
        description="活体记忆系统 FastAPI 服务启动脚本")
    parser.add_argument(
        "--host", type=str, default=None,
        help="监听地址（默认读取 LMS_API_HOST 或 127.0.0.1）")
    parser.add_argument(
        "--port", type=int, default=None,
        help="监听端口（默认读取 LMS_API_PORT 或 8190）")
    parser.add_argument(
        "--reload", action="store_true",
        help="开启热重载（开发模式）")
    args = parser.parse_args()

    # 延迟导入，确保 sys.path 已就绪
    from api.config import get_server_config

    server_cfg = get_server_config()
    host = args.host or server_cfg['host']
    port = args.port or server_cfg['port']

    # 热重载模式下必须以字符串形式指定 app 路径
    if args.reload:
        import uvicorn
        print("=" * 50)
        print("活体记忆系统 API（开发模式，热重载）")
        print(f"  地址: http://{host}:{port}")
        print(f"  文档: http://{host}:{port}/docs")
        print("=" * 50)
        uvicorn.run(
            "api.server:app",
            host=host,
            port=port,
            reload=True,
        )
    else:
        # 非重载模式直接传入 app 对象，减少一次导入开销
        from api.server import app
        import uvicorn
        print("=" * 50)
        print("活体记忆系统 API")
        print(f"  地址: http://{host}:{port}")
        print(f"  文档: http://{host}:{port}/docs")
        print("  按 Ctrl+C 停止")
        print("=" * 50)
        uvicorn.run(
            app,
            host=host,
            port=port,
        )


if __name__ == "__main__":
    main()

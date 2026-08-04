"""活体记忆系统 - 命令行调用接口
====================================

供 TRAE Skill 调用的命令行工具，提供记忆检索、存储、状态查询和做梦功能。

用法:
    python memory_cli.py recall "查询文本"
    python memory_cli.py store "要存储的对话文本"
    python memory_cli.py status
    python memory_cli.py store_and_recall "要存储的对话文本" "检索查询"
    python memory_cli.py dream              # 默认 MVP 做梦 20 步
    python memory_cli.py dream 50           # MVP 做梦 50 步
    python memory_cli.py dream --steps 10   # MVP 做梦 10 步
    python memory_cli.py dream --full 100   # 完整做梦周期 100 步
    python memory_cli.py dream --full --steps 100  # 完整做梦周期 100 步

输出:
    JSON 格式的结果，包含 status 字段和记忆内容。
"""

import os
import sys
import json
import logging
from typing import Any

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# 抑制详细日志（只保留错误），避免干扰 JSON 输出
logging.basicConfig(
    level=logging.WARNING,
    stream=sys.stderr,
    format="%(levelname)s: %(message)s",
)

# 检查 LLM API Key（从环境变量读取，不再硬编码）
# 未配置时仅打印警告，不退出，避免被 import 时阻塞；
# 下游 mcp_memory_server 会自行处理缺失的 Key（记忆检索/存储不受影响）。
if not (os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("LMS_LLM_API_KEY")):
    logging.warning(
        "未配置 DEEPSEEK_API_KEY / LMS_LLM_API_KEY 环境变量，"
        "LLM 相关功能将不可用。请参考 .env.example 配置环境变量。"
    )

# ---------------------------------------------------------------------------
# 导入记忆系统
# ---------------------------------------------------------------------------
try:
    from mcp_memory_server import (
        do_recall_memory,
        do_store_memory,
        do_get_memory_status,
    )
    _INIT_OK = True
    _INIT_ERROR = None
except Exception as e:
    _INIT_OK = False
    _INIT_ERROR = str(e)

# ---------------------------------------------------------------------------
# 做梦引擎导入（可选，DreamEngine 由子AI-1创建中，可能尚不存在）
# ---------------------------------------------------------------------------
try:
    from core.hippocampus.dream_engine import DreamEngine
    _DREAM_OK = True
    _DREAM_ERROR = None
except ImportError as e:
    DreamEngine = None
    _DREAM_OK = False
    _DREAM_ERROR = str(e)


def cmd_recall(query: str) -> str:
    """检索记忆"""
    if not _INIT_OK:
        return json.dumps({
            "status": "error",
            "error": f"记忆系统初始化失败: {_INIT_ERROR}",
        }, ensure_ascii=False)

    try:
        result = do_recall_memory(query)
        return json.dumps({
            "status": "success",
            "result": result,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
        }, ensure_ascii=False)


def cmd_store(text: str) -> str:
    """存储记忆"""
    if not _INIT_OK:
        return json.dumps({
            "status": "error",
            "error": f"记忆系统初始化失败: {_INIT_ERROR}",
        }, ensure_ascii=False)

    try:
        result = do_store_memory(text)
        return json.dumps({
            "status": "success",
            "result": result,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
        }, ensure_ascii=False)


def cmd_status() -> str:
    """获取记忆系统状态"""
    if not _INIT_OK:
        return json.dumps({
            "status": "error",
            "error": f"记忆系统初始化失败: {_INIT_ERROR}",
        }, ensure_ascii=False)

    try:
        result = do_get_memory_status()
        return json.dumps({
            "status": "success",
            "result": json.loads(result),
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
        }, ensure_ascii=False)


def cmd_store_and_recall(text: str, query: str) -> str:
    """存储记忆并立即检索"""
    if not _INIT_OK:
        return json.dumps({
            "status": "error",
            "error": f"记忆系统初始化失败: {_INIT_ERROR}",
        }, ensure_ascii=False)

    try:
        # 先存储
        store_result = do_store_memory(text)
        # 再检索
        recall_result = do_recall_memory(query)
        return json.dumps({
            "status": "success",
            "store_result": store_result,
            "recall_result": recall_result,
        }, ensure_ascii=False, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
        }, ensure_ascii=False)


def cmd_dream(steps: int, full_cycle: bool) -> str:
    """执行做梦周期。

    初始化 LivingMemoryLoop（自动加载快照恢复记忆），通过 loop.dream()
    调用 DreamEngine，根据模式执行 MVP 做梦（dream_mvp）或完整做梦周期
    （dream_cycle）。做梦后自动保存快照。

    参数:
        steps: 做梦步数。
        full_cycle: True 执行完整做梦周期（dream_cycle），
                    False 执行 MVP 做梦（dream_mvp）。

    返回:
        JSON 格式的做梦结果。
    """
    # 检查 DreamEngine 是否可用
    if not _DREAM_OK:
        return json.dumps({
            "status": "error",
            "error": (
                "做梦引擎（DreamEngine）不可用。"
                f"导入失败: {_DREAM_ERROR}\n"
                "请确认 core/hippocampus/dream_engine.py 已创建。"
            ),
        }, ensure_ascii=False)

    try:
        # 通过 mcp_memory_server 获取 LivingMemoryLoop（自动加载快照）
        from mcp_memory_server import get_loop
        loop = get_loop()

        # 调用 loop.dream()（内部懒加载 DreamEngine + 做梦后自动快照）
        result = loop.dream(n_steps=steps, full_cycle=full_cycle)
        mode = "full_cycle" if full_cycle else "mvp"

        return json.dumps({
            "status": "success",
            "mode": mode,
            "steps": steps,
            "result": result,
        }, ensure_ascii=False, indent=2, default=_json_default)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e),
        }, ensure_ascii=False)


def _json_default(obj: Any) -> Any:
    """JSON 序列化的 default 回调，处理 torch.Tensor / numpy 标量。"""
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return str(obj)


def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({
            "status": "error",
            "error": "用法: python memory_cli.py <command> [args]\n"
                    "命令: recall <query>, store <text>, status, "
                    "store_and_recall <text> <query>, "
                    "dream [steps] [--full <steps>]"
        }, ensure_ascii=False))
        sys.exit(1)

    command = sys.argv[1]

    if command == "recall":
        if len(sys.argv) < 3:
            print(json.dumps({
                "status": "error",
                "error": "用法: python memory_cli.py recall <query>"
            }, ensure_ascii=False))
            sys.exit(1)
        print(cmd_recall(sys.argv[2]))

    elif command == "store":
        if len(sys.argv) < 3:
            print(json.dumps({
                "status": "error",
                "error": "用法: python memory_cli.py store <text>"
            }, ensure_ascii=False))
            sys.exit(1)
        print(cmd_store(sys.argv[2]))

    elif command == "status":
        print(cmd_status())

    elif command == "store_and_recall":
        if len(sys.argv) < 4:
            print(json.dumps({
                "status": "error",
                "error": "用法: python memory_cli.py store_and_recall <text> <query>"
            }, ensure_ascii=False))
            sys.exit(1)
        print(cmd_store_and_recall(sys.argv[2], sys.argv[3]))

    elif command == "dream":
        # 解析 dream 命令参数
        # 用法:
        #   dream                 → MVP 做梦，默认 20 步
        #   dream 50              → MVP 做梦，50 步
        #   dream --steps 10      → MVP 做梦，10 步
        #   dream --full          → 完整做梦周期，默认 100 步
        #   dream --full 100      → 完整做梦周期，100 步
        #   dream --full --steps 100  → 完整做梦周期，100 步
        full_cycle = False
        steps = None  # 未指定时按模式取默认值
        dream_args = sys.argv[2:]
        i = 0
        while i < len(dream_args):
            arg = dream_args[i]
            if arg == "--full":
                full_cycle = True
            elif arg == "--steps":
                if i + 1 >= len(dream_args):
                    print(json.dumps({
                        "status": "error",
                        "error": "--steps 需要一个步数参数"
                    }, ensure_ascii=False))
                    sys.exit(1)
                try:
                    steps = int(dream_args[i + 1])
                except ValueError:
                    print(json.dumps({
                        "status": "error",
                        "error": f"无效的步数: {dream_args[i + 1]}"
                    }, ensure_ascii=False))
                    sys.exit(1)
                i += 1  # 跳过步数值
            else:
                # 位置参数：步数
                try:
                    steps = int(arg)
                except ValueError:
                    print(json.dumps({
                        "status": "error",
                        "error": f"无效的步数或选项: {arg}\n"
                                 "用法: dream [steps] | dream --steps N | "
                                 "dream --full [steps]"
                    }, ensure_ascii=False))
                    sys.exit(1)
            i += 1

        # 未指定步数时按模式取默认值
        if steps is None:
            steps = 100 if full_cycle else 20

        print(cmd_dream(steps, full_cycle))

    else:
        print(json.dumps({
            "status": "error",
            "error": f"未知命令: {command}\n"
                    "可用命令: recall, store, status, "
                    "store_and_recall, dream"
        }, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()

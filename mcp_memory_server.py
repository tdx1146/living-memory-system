"""活体记忆系统 - MCP (Model Context Protocol) 服务器
=====================================================

通过 MCP 协议向 TRAE IDE 暴露活体记忆系统的核心能力，使 AI 助手能够：
  1. recall_memory  — 检索与查询相关的历史对话记忆
  2. store_memory   — 将当前对话存储到记忆系统
  3. get_memory_status — 获取记忆系统运行状态
  4. dream_memory   — 触发记忆系统"做梦"（空闲态记忆巩固与整合）

基于 mcp 2.0.0 的 Server 类（构造器注册 handler，非装饰器模式）。
通过 stdio 传输与 MCP 客户端通信。

启动方式:
    python mcp_memory_server.py

注意: 日志输出到 stderr（stdout 专用于 MCP 协议通信）。
"""

import os
import re
import sys
import json
import logging
import asyncio
from typing import Any

# ---------------------------------------------------------------------------
# 路径设置：确保项目根目录在 sys.path 上，使 runtime/core 等包可被导入
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ---------------------------------------------------------------------------
# MCP 2.0.0 导入
# ---------------------------------------------------------------------------
import mcp.server.stdio as stdio  # noqa: E402
from mcp.server import Server  # noqa: E402
import mcp.types as types  # noqa: E402

# ---------------------------------------------------------------------------
# 活体记忆系统导入
# ---------------------------------------------------------------------------
from runtime.config import default_config  # noqa: E402
from runtime.loop import LivingMemoryLoop  # noqa: E402
# 共享配置构建逻辑：embedder / LLM API 配置 / 默认 LLM 常量均来自 api.config，
# 消除与 api/config.py 的重复代码。
from api.config import (  # noqa: E402
    build_embedder,
    build_llm_api_config,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)

logger = logging.getLogger("mcp_memory_server")

# ---------------------------------------------------------------------------
# 全局 LivingMemoryLoop 实例（懒加载）
# ---------------------------------------------------------------------------
_loop: LivingMemoryLoop | None = None
_init_error: str | None = None


def _build_config() -> dict:
    """构建 LivingMemoryLoop 配置字典。

    合并 runtime.default_config()，通过 api.config 共享的 build_embedder()
    注入 PretrainedEmbedder（失败时降级为 SimpleEmbedder），并通过
    build_llm_api_config() 按环境变量配置 LLM API 与快照。
    """
    config = default_config()

    # --- Embedder：从环境变量读取类型，支持 cloud / pretrained / simple ---
    # 共享构建逻辑见 api.config.build_embedder（支持 cloud/pretrained/simple，
    # 依赖缺失或加载失败时自动降级）。
    embedder_type = os.environ.get("LMS_EMBEDDER", "pretrained").strip()
    embedder = build_embedder(embedder_type, config['input_dim'])
    config['embedder'] = embedder
    # 预训练/云端 embedder 输出幅度较小，习惯化阈值适配
    # （仅当 embedder 提供 embed_text 方法时设置；SimpleEmbedder 不需要）
    if hasattr(embedder, 'embed_text'):
        config['activation_threshold'] = 0.02

    # --- LLM API 配置（可选，工具本身不需要 LLM，但保持系统完整） ---
    # 共享构建逻辑见 api.config.build_llm_api_config()
    config['llm_api'] = build_llm_api_config(
        DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL)
    if config['llm_api']:
        logger.info("LLM API 已配置 (DeepSeek)")
    else:
        logger.info("未配置 LLM API key，LLM 功能已禁用（工具仍可正常使用）")

    # --- 快照配置 ---
    config['auto_snapshot'] = True
    config['auto_snapshot_interval'] = 50
    config['snapshot_dir'] = os.path.join(_SCRIPT_DIR, 'snapshots')

    # --- 会话标识（T1.1/P0-5：快照按会话命名与元数据持久化需要） ---
    # 本 MCP 直连进程代表单一脑；默认 "main"（与 lms-http 桥的 sid=main
    # 对齐），可用 LMS_MCP_SESSION_ID 覆盖。快照读写将落在
    # snapshots/{session}/latest_{session}.pt（新命名规范）。
    config['session_id'] = os.environ.get("LMS_MCP_SESSION_ID", "main").strip() or "main"

    return config


def _get_snapshot_dir() -> str:
    """获取快照目录路径。"""
    return os.path.join(_SCRIPT_DIR, 'snapshots')


def _get_latest_snapshot(session_id: str | None = None) -> str | None:
    """按新命名规范查找最新快照（T1.1/P0-5 跟随新路径）。

    查找顺序（fail-open，逐步回退）：
      1. snapshots/{session}/latest_{session}.pt（会话专属最新指针，优先）；
      2. 各会话子目录下 mtime 最新的 latest_*.pt（未指定/无专属文件时）；
      3. 根目录平铺旧格式 *.pt（存量 latest.pt / snapshot_{n}.pt 等，mtime 最新）。
    """
    from persistence.snapshot import sanitize_session_id, latest_path_for

    snap_dir = _get_snapshot_dir()
    if not os.path.isdir(snap_dir):
        return None

    # 1. 会话专属最新指针（新命名规范）
    if session_id:
        sid = sanitize_session_id(session_id)
        cand = latest_path_for(snap_dir, sid)
        if os.path.isfile(cand):
            return cand

    # 2. 各会话子目录下的 latest_*.pt（按 mtime 取最新）
    best: str | None = None
    for sub in sorted(os.listdir(snap_dir)):
        sub_path = os.path.join(snap_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        for fname in os.listdir(sub_path):
            if fname.startswith("latest_") and fname.endswith(".pt"):
                cand = os.path.join(sub_path, fname)
                if best is None or os.path.getmtime(cand) > os.path.getmtime(best):
                    best = cand
    if best is not None:
        return best

    # 3. 旧格式平铺（存量兼容）
    import glob
    snapshots = glob.glob(os.path.join(snap_dir, '*.pt'))
    if not snapshots:
        return None
    snapshots.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return snapshots[0]


def get_loop() -> LivingMemoryLoop:
    """获取全局 LivingMemoryLoop 实例（懒加载）。

    首次调用时初始化记忆系统（加载预训练模型等），并自动加载最新快照
    以恢复跨会话记忆。若初始化失败，记录错误并抛出 RuntimeError；
    后续调用直接抛出已记录的错误。

    返回:
        已初始化的 LivingMemoryLoop 实例。

    异常:
        RuntimeError: 记忆系统初始化失败时抛出。
    """
    global _loop, _init_error

    if _loop is not None:
        return _loop

    if _init_error is not None:
        raise RuntimeError(f"记忆系统初始化失败: {_init_error}")

    try:
        logger.info("正在初始化 LivingMemoryLoop ...")
        config = _build_config()
        _loop = LivingMemoryLoop(config)
        logger.info("LivingMemoryLoop 初始化完成")

        # 自动加载最新快照，恢复跨会话记忆
        # T1.1/P0-5：按本进程会话标识查找新命名规范下的最新快照
        latest_snap = _get_latest_snapshot(_loop.session_id)
        if latest_snap:
            try:
                _loop.load_state(latest_snap)
                logger.info(f"已从快照恢复记忆: {latest_snap}")
            except Exception as e:
                logger.warning(f"快照加载失败（忽略，使用空白记忆）: {e}")
        else:
            logger.info("未找到快照文件，使用空白记忆")

        return _loop
    except Exception as e:
        _init_error = str(e)
        logger.error(f"记忆系统初始化失败: {e}", exc_info=True)
        raise RuntimeError(f"记忆系统初始化失败: {_init_error}") from e


# ---------------------------------------------------------------------------
# JSON 序列化辅助：处理 torch.Tensor / numpy 标量
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """JSON 序列化的 default 回调，处理非原生类型。"""
    # torch.Tensor / numpy scalar -> Python 标量
    if hasattr(obj, 'item'):
        return obj.item()
    # numpy 数组
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return str(obj)


def _to_json(obj: Any) -> str:
    """安全 JSON 序列化，处理 torch/numpy 类型。"""
    return json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default)


# ---------------------------------------------------------------------------
# 工具逻辑实现（纯函数，可独立测试）
# ---------------------------------------------------------------------------

def do_recall_memory(query: str) -> str:
    """检索与查询相关的历史对话记忆。

    使用预训练模型将查询编码为语义向量，在情景记忆缓冲区中检索最相关的
    top 3 条记忆，并返回记忆文本与系统状态。

    参数:
        query: 检索查询文本。

    返回:
        格式化的记忆检索结果文本。
    """
    loop = get_loop()
    embedder = loop.embedder

    # 获取查询向量：优先 384 维原始向量（高精度），fallback 64 维投影向量
    raw_vec = None
    proj_vec = None
    if hasattr(embedder, 'embed_text_raw'):
        raw_vec = embedder.embed_text_raw(query)
    if hasattr(embedder, 'embed_text'):
        proj_vec = embedder.embed_text(query)

    # 选择查询向量与 fallback
    if raw_vec is not None:
        query_vector = raw_vec
        fallback_query = proj_vec  # 用于兼容旧 64 维条目
    elif proj_vec is not None:
        query_vector = proj_vec
        fallback_query = None
    else:
        # SimpleEmbedder：不支持语义检索
        buffer_size = loop.memory.episodic_size()
        status = loop.get_status()
        return (
            f"当前嵌入器不支持语义检索（SimpleEmbedder），"
            f"无法进行语义记忆检索。\n\n"
            f"情景记忆缓冲区大小: {buffer_size}\n"
            f"记忆系统状态:\n{_to_json(status)}"
        )

    # 语义检索
    entries = loop.memory.recall_episodic(
        query_vector, top_k=3, fallback_query=fallback_query)

    status = loop.get_status()
    buffer_size = loop.memory.episodic_size()

    if not entries:
        return (
            f"未检索到相关记忆（查询: \"{query}\"）。\n"
            f"情景记忆缓冲区大小: {buffer_size}\n\n"
            f"记忆系统状态:\n{_to_json(status)}"
        )

    # 格式化检索结果
    lines = [
        f"检索到 {len(entries)} 条相关记忆（查询: \"{query}\"）:",
        f"情景记忆缓冲区大小: {buffer_size}",
        "",
    ]
    for i, entry in enumerate(entries, 1):
        lines.append(f"--- 记忆 {i}（轮次 {entry.turn}，惊讶度 {entry.surprise:.3f}）---")
        lines.append(entry.text)
        lines.append("")

    lines.append(f"记忆系统状态:\n{_to_json(status)}")
    return "\n".join(lines)


def _parse_conversation(text: str) -> tuple[str, str]:
    """从对话文本中解析用户输入和助手回复。

    process_turn 的设计意图是分开接收 user_input 和 llm_output，再在内部
    拼接为 "用户: {user_input}\\n助手: {llm_output}"。若直接把整段对话文本
    塞进 user_input，会导致 "用户:/助手:" 标记被重复嵌套，记忆编码失真。

    本函数将外部传入的对话文本拆分为 (user_input, llm_output)：

    1. 若文本包含 "用户:" / "助手:" 标记，解析出两部分分别返回。
    2. 若不含标记（纯文本），原样作为 user_input 返回，llm_output 留空，
       保持与旧调用方的兼容。

    解析逻辑健壮性：
    - 支持 "用户:" 前面有换行/空格（先 strip 整体空白再匹配）。
    - 支持 "助手:" 不存在的情况（llm_output 返回空串）。
    - 同时兼容半角 ":" 与全角 "：" 冒号。

    参数:
        text: 原始对话文本。

    返回:
        (user_input, llm_output) 元组。
    """
    if not text:
        return "", ""

    # 尝试匹配 "用户: xxx\n助手: xxx" 格式
    # - 前导 \s* 容忍 "用户:" 前的换行/空格
    # - (.*?) 非贪婪，匹配用户输入直到遇到 "助手:" 行或文本结束
    # - 助手部分可选，支持半角/全角冒号
    pattern = r'\s*用户[:：]\s*(.*?)(?:\n\s*助手[:：]\s*(.*))?$'
    match = re.match(pattern, text, re.DOTALL)
    if match:
        user_input = match.group(1).strip()
        llm_output = match.group(2).strip() if match.group(2) else ""
        return user_input, llm_output

    # 不含标记的纯文本：整段作为 user_input，llm_output 留空（保持原行为）
    return text, ""


def do_store_memory(text: str) -> str:
    """将对话内容存储到记忆系统。

    调用 LivingMemoryLoop.process_turn() 执行完整的记忆循环（编码、推断、
    学习、巩固、存储），将对话文本写入情景记忆缓冲区。

    输入文本会被解析为用户输入与助手回复两部分，分别传入 process_turn 的
    user_input / llm_output 参数（A-P1-1 修复）。若文本不含 "用户:/助手:"
    标记，则作为纯文本整段传入 user_input，llm_output 留空。

    参数:
        text: 要存储的对话文本（建议格式："用户: xxx\\n助手: xxx"）。

    返回:
        存储确认信息与记忆状态（JSON 格式）。
    """
    loop = get_loop()

    # 解析对话文本：分离用户输入与助手回复，避免标记重复嵌套（A-P1-1）
    user_input, llm_output = _parse_conversation(text)

    # process_turn 执行完整记忆循环并返回记忆 context
    memory_context = loop.process_turn(user_input, llm_output)

    status = loop.get_status()
    buffer_size = loop.memory.episodic_size()

    # 自动保存快照，确保持久化跨会话记忆
    # T1.1/P0-5：会话级命名规范（snapshots/{session}/... + latest_{session}.pt）
    snapshot_saved = False
    try:
        path = loop.save_session_state()
        if path:
            snapshot_saved = True
            logger.info(f"记忆已保存到快照: {path}")
        else:
            logger.warning("快照保存被写锁超时跳过（不影响当前操作）")
    except Exception as e:
        logger.warning(f"快照保存失败（不影响当前操作）: {e}")

    result = {
        "status": "已存储",
        "turn_count": status.get('turn_count', 0),
        "episodic_buffer_size": buffer_size,
        "last_entropy": status.get('last_entropy'),
        "last_surprise": status.get('last_surprise'),
        "precision_mean": status.get('precision_mean'),
        "purpose_coherence": status.get('purpose_coherence'),
        "snapshot_saved": snapshot_saved,
        "memory_context": memory_context,
    }
    return _to_json(result)


def do_get_memory_status() -> str:
    """获取记忆系统的当前运行状态。

    返回包含轮次数、熵、惊讶度、precision、情景记忆缓冲区大小等的状态字典。

    返回:
        状态字典（JSON 格式）。
    """
    loop = get_loop()
    status = loop.get_status()
    # 补充情景记忆缓冲区大小
    status['episodic_buffer_size'] = loop.memory.episodic_size()
    # 补充 embedder 类型信息
    embedder = loop.embedder
    status['embedder_type'] = type(embedder).__name__
    if hasattr(embedder, 'raw_dim'):
        status['embedder_raw_dim'] = embedder.raw_dim
    return _to_json(status)


def do_dream_memory(steps: int = 20, full_cycle: bool = False) -> str:
    """触发记忆系统的"做梦"过程。

    在空闲时进行记忆巩固、遗忘和整合，让记忆系统在无对话输入时持续运转。
    建议在对话间歇时调用。做梦会精炼吸引子景观、衰减低价值记忆、
    演化目的层，并自动保存快照。

    参数:
        steps: 做梦步数（默认 20）。full_cycle=False 时为 MVP 做梦步数，
            full_cycle=True 时为完整做梦周期步数。
        full_cycle: True 执行完整七阶段做梦周期（NREM巩固/SHY/遗忘修剪/
            景观漂移/目的演化/REM整合/快照），False 执行 MVP 简化版做梦。

    返回:
        做梦统计（JSON 格式），包含模式、步数、平均惊讶度、坍缩次数、
        快照保存状态等。
    """
    loop = get_loop()
    result = loop.dream(n_steps=steps, full_cycle=full_cycle)
    return _to_json(result)


# ---------------------------------------------------------------------------
# MCP 工具定义
# ---------------------------------------------------------------------------

RECALL_MEMORY_TOOL = types.Tool(
    name="recall_memory",
    description=(
        "检索与查询相关的历史对话记忆。当你需要回忆之前与用户讨论的内容时"
        "调用此工具。例如：用户问'你还记得我之前说的吗'，或当你需要之前的"
        "上下文信息时。建议在每次对话开始时主动调用。"
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "检索查询文本",
            }
        },
        "required": ["query"],
    },
)

STORE_MEMORY_TOOL = types.Tool(
    name="store_memory",
    description=(
        "将当前对话内容存储到记忆系统。在每次与用户完成一轮对话后调用此工具，"
        "保存用户的输入和你的回复。这使系统能在未来检索到本次对话内容。"
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "要存储的对话文本"
                    "（可以是'用户: xxx\\n助手: xxx'格式）"
                ),
            }
        },
        "required": ["text"],
    },
)

GET_MEMORY_STATUS_TOOL = types.Tool(
    name="get_memory_status",
    description=(
        "获取记忆系统的当前状态（轮次数、熵、惊讶度、precision等）。"
        "用于了解记忆系统的运作情况。"
    ),
    inputSchema={
        "type": "object",
        "properties": {},
        "required": [],
    },
)

DREAM_MEMORY_TOOL = types.Tool(
    name="dream_memory",
    description=(
        "触发记忆系统的'做梦'过程，在空闲时进行记忆巩固、遗忘和整合。"
        "建议在对话间歇时调用。做梦会让记忆系统在无输入时持续运转，"
        "精炼吸引子景观、衰减低价值记忆、演化目的层，使记忆从'冷工具'"
        "变为'活体'。"
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "steps": {
                "type": "integer",
                "description": "做梦步数（默认 20）",
                "default": 20,
            },
            "full_cycle": {
                "type": "boolean",
                "description": (
                    "True 执行完整七阶段做梦周期，False 执行 MVP 简化版"
                    "（默认 false）"
                ),
                "default": False,
            },
        },
        "required": [],
    },
)

ALL_TOOLS = [
    RECALL_MEMORY_TOOL,
    STORE_MEMORY_TOOL,
    GET_MEMORY_STATUS_TOOL,
    DREAM_MEMORY_TOOL,
]


# ---------------------------------------------------------------------------
# MCP Handler 函数（mcp 2.0.0 构造器注册模式）
# ---------------------------------------------------------------------------

async def handle_list_tools(ctx, params) -> types.ListToolsResult:
    """处理 tools/list 请求，返回可用工具列表。"""
    return types.ListToolsResult(tools=ALL_TOOLS)


async def handle_call_tool(
    ctx, params: types.CallToolRequestParams
) -> types.CallToolResult:
    """处理 tools/call 请求，分派到对应工具逻辑。"""
    name = params.name
    arguments = params.arguments or {}

    try:
        if name == "recall_memory":
            query = arguments.get("query", "")
            if not query:
                return types.CallToolResult(
                    content=[types.TextContent(
                        type="text",
                        text="错误：缺少必需参数 'query'")],
                    is_error=True,
                )
            result_text = do_recall_memory(query)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=result_text)]
            )

        elif name == "store_memory":
            text = arguments.get("text", "")
            if not text:
                return types.CallToolResult(
                    content=[types.TextContent(
                        type="text",
                        text="错误：缺少必需参数 'text'")],
                    is_error=True,
                )
            result_text = do_store_memory(text)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=result_text)]
            )

        elif name == "get_memory_status":
            result_text = do_get_memory_status()
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=result_text)]
            )

        elif name == "dream_memory":
            steps = arguments.get("steps", 20)
            full_cycle = arguments.get("full_cycle", False)
            result_text = do_dream_memory(
                steps=steps, full_cycle=full_cycle)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=result_text)]
            )

        else:
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=f"错误：未知工具 '{name}'。"
                         f"可用工具: recall_memory, store_memory, "
                         f"get_memory_status, dream_memory")],
                is_error=True,
            )

    except Exception as e:
        logger.error(f"工具 '{name}' 执行出错: {e}", exc_info=True)
        return types.CallToolResult(
            content=[types.TextContent(
                type="text",
                text=f"工具执行出错: {e}")],
            is_error=True,
        )


# ---------------------------------------------------------------------------
# 服务器创建与启动
# ---------------------------------------------------------------------------

def create_server() -> Server:
    """创建并配置 MCP 服务器实例。

    使用 mcp 2.0.0 的构造器注册模式（on_list_tools / on_call_tool），
    而非旧版的 @server.list_tools() 装饰器。
    """
    server = Server(
        "living-memory",
        on_list_tools=handle_list_tools,
        on_call_tool=handle_call_tool,
    )
    return server


async def main() -> None:
    """MCP 服务器主入口：通过 stdio 启动服务。"""
    server = create_server()
    logger.info("MCP 服务器 'living-memory' 正在启动 (stdio 传输)...")
    async with stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    # 日志输出到 stderr（stdout 专用于 MCP 协议通信）
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(main())

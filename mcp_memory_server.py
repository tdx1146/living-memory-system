"""活体记忆系统 - MCP (Model Context Protocol) 服务器
=====================================================

通过 MCP 协议向 TRAE IDE 暴露活体记忆系统的核心能力，使 AI 助手能够：
  1. recall_memory  — 检索与查询相关的历史对话记忆
  2. store_memory   — 将当前对话存储到记忆系统
  3. get_memory_status — 获取记忆系统运行状态

基于 mcp 2.0.0 的 Server 类（构造器注册 handler，非装饰器模式）。
通过 stdio 传输与 MCP 客户端通信。

启动方式:
    python mcp_memory_server.py

注意: 日志输出到 stderr（stdout 专用于 MCP 协议通信）。
"""

import os
import sys
import json
import logging
import asyncio

# ---------------------------------------------------------------------------
# 路径设置：确保项目根目录在 sys.path 上，使 runtime/core 等包可被导入
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ---------------------------------------------------------------------------
# MCP 2.0.0 导入
# ---------------------------------------------------------------------------
import mcp.server.stdio as stdio
from mcp.server import Server
import mcp.types as types

# ---------------------------------------------------------------------------
# 活体记忆系统导入
# ---------------------------------------------------------------------------
from runtime.config import default_config
from runtime.loop import LivingMemoryLoop

logger = logging.getLogger("mcp_memory_server")

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 预训练 sentence-transformers 模型本地缓存路径（modelscope 下载）
PRETRAINED_MODEL_PATH = (
    r"C:\Users\dandan\.cache\modelscope\models"
    r"\sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
    r"\snapshots\master"
)

# 默认 LLM 配置（DeepSeek API）
DEFAULT_LLM_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_LLM_MODEL = "deepseek-chat"

# ---------------------------------------------------------------------------
# 全局 LivingMemoryLoop 实例（懒加载）
# ---------------------------------------------------------------------------
_loop: LivingMemoryLoop | None = None
_init_error: str | None = None


def _build_config() -> dict:
    """构建 LivingMemoryLoop 配置字典。

    合并 runtime.default_config()，注入 PretrainedEmbedder（失败时降级为
    SimpleEmbedder），并按环境变量配置 LLM API 与快照。
    """
    config = default_config()

    # --- Embedder：优先使用预训练语义嵌入器 ---
    try:
        from core.sensory.embedder import PretrainedEmbedder
        model_path = os.environ.get(
            "LMS_PRETRAINED_MODEL", PRETRAINED_MODEL_PATH)
        embedder = PretrainedEmbedder(
            dim=config['input_dim'], model_name=model_path)
        config['embedder'] = embedder
        # 预训练 embedder 输出幅度较小，习惯化阈值适配
        config['activation_threshold'] = 0.02
        logger.info(f"PretrainedEmbedder 已加载，模型路径: {model_path}")
    except Exception as e:
        logger.warning(
            f"PretrainedEmbedder 加载失败（{e}），降级为 SimpleEmbedder")
        # default_config() 已含 SimpleEmbedder，此处不覆盖

    # --- LLM API 配置（可选，工具本身不需要 LLM，但保持系统完整） ---
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip() or \
        os.environ.get("LMS_LLM_API_KEY", "").strip()
    if api_key:
        config['llm_api'] = {
            'base_url': os.environ.get(
                "LMS_LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
            'api_key': api_key,
            'model': os.environ.get("LMS_LLM_MODEL", DEFAULT_LLM_MODEL),
            'temperature': 0.7,
            'max_tokens': 1000,
            'timeout': 30,
            'max_retries': 3,
            'system_prompt': (
                '你是一个有记忆能力的AI助手。系统会为你提供海马体的记忆context，'
                '其中包含激活节点信息和长时记忆检索结果，'
                '这代表了你对此前对话的记忆状态。'
                '请结合记忆context和当前用户输入来回答问题。'
                '如果记忆context中有相关信息，请尝试利用它来回答；'
                '同时保持自然流畅的对话。'
            ),
        }
        logger.info("LLM API 已配置 (DeepSeek)")
    else:
        config['llm_api'] = None
        logger.info("未配置 LLM API key，LLM 功能已禁用（工具仍可正常使用）")

    # --- 快照配置 ---
    config['auto_snapshot'] = True
    config['auto_snapshot_interval'] = 50
    config['snapshot_dir'] = os.path.join(_SCRIPT_DIR, 'snapshots')

    return config


def get_loop() -> LivingMemoryLoop:
    """获取全局 LivingMemoryLoop 实例（懒加载）。

    首次调用时初始化记忆系统（加载预训练模型等）。
    若初始化失败，记录错误并抛出 RuntimeError；后续调用直接抛出已记录的错误。

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
        return _loop
    except Exception as e:
        _init_error = str(e)
        logger.error(f"记忆系统初始化失败: {e}", exc_info=True)
        raise RuntimeError(f"记忆系统初始化失败: {_init_error}") from e


# ---------------------------------------------------------------------------
# JSON 序列化辅助：处理 torch.Tensor / numpy 标量
# ---------------------------------------------------------------------------

def _json_default(obj):
    """JSON 序列化的 default 回调，处理非原生类型。"""
    # torch.Tensor / numpy scalar -> Python 标量
    if hasattr(obj, 'item'):
        return obj.item()
    # numpy 数组
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    return str(obj)


def _to_json(obj) -> str:
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
        buffer_size = len(loop.memory._episodic_buffer)
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
    buffer_size = len(loop.memory._episodic_buffer)

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


def do_store_memory(text: str) -> str:
    """将对话内容存储到记忆系统。

    调用 LivingMemoryLoop.process_turn() 执行完整的记忆循环（编码、推断、
    学习、巩固、存储），将对话文本写入情景记忆缓冲区。

    参数:
        text: 要存储的对话文本（建议格式："用户: xxx\\n助手: xxx"）。

    返回:
        存储确认信息与记忆状态（JSON 格式）。
    """
    loop = get_loop()

    # process_turn 执行完整记忆循环并返回记忆 context
    # text 作为 user_input 传入，llm_output 留空（text 已包含完整对话）
    memory_context = loop.process_turn(text)

    status = loop.get_status()
    buffer_size = len(loop.memory._episodic_buffer)

    result = {
        "status": "已存储",
        "turn_count": status.get('turn_count', 0),
        "episodic_buffer_size": buffer_size,
        "last_entropy": status.get('last_entropy'),
        "last_surprise": status.get('last_surprise'),
        "precision_mean": status.get('precision_mean'),
        "purpose_coherence": status.get('purpose_coherence'),
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
    status['episodic_buffer_size'] = len(loop.memory._episodic_buffer)
    # 补充 embedder 类型信息
    embedder = loop.embedder
    status['embedder_type'] = type(embedder).__name__
    if hasattr(embedder, 'raw_dim'):
        status['embedder_raw_dim'] = embedder.raw_dim
    return _to_json(status)


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

ALL_TOOLS = [RECALL_MEMORY_TOOL, STORE_MEMORY_TOOL, GET_MEMORY_STATUS_TOOL]


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

        else:
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=f"错误：未知工具 '{name}'。"
                         f"可用工具: recall_memory, store_memory, get_memory_status")],
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

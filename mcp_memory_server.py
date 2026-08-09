"""活体记忆系统 - MCP (Model Context Protocol) 服务器（阶段1-B 薄桥版）
=====================================================

通过 MCP 协议向 AI 助手（OpenClaw/TRAE 等）暴露活体记忆系统的核心能力：
  1. recall_memory  — 检索与查询相关的历史对话记忆
  2. store_memory   — 将当前对话存储到记忆系统
  3. get_memory_status — 获取记忆系统运行状态
  4. dream_memory   — 触发记忆系统"做梦"（空闲态记忆巩固与整合）

阶段1-B（T1.5 / P0-4 收口 + P1-3）架构变更：
  - **从"直连 torch 重型进程"改为"stdio 薄桥"**：所有工具调用转发 HTTP :8190
    （store→POST /feed + /snapshot、recall→POST /recall、status→GET /status/{sid}、
    dream→POST /dream/{sid}），**本地不加载 torch / 快照 / 不直写文件**。
  - **单写者收口**：记忆状态唯一写者是常驻 HTTP API 进程（:8190），本进程只做
    请求转发，不做任何本地持久化（P0-4：4×MCP 无锁互踩 latest.pt 的问题根除）。
  - **内存目标达成**：不依赖 mcp SDK（其 pydantic 依赖较重），采用与
    lms_http_mcp.py 相同的**手写 stdio JSON-RPC 薄协议**（~26MB 级，冷启动
    <100ms；lms-http 已在网关长期运行验证兼容）。
  - **契约不变**：MCP 工具名/参数保持原样（recall_memory/store_memory/
    get_memory_status/dream_memory），openclaw.json 无需改动；
    LMS_MCP_SESSION_ID 默认 "main" 语义保留，所有转发请求携带该会话标识。
  - **fail-open**：HTTP 调用失败时抛出明确异常（MCP 侧显示 is_error 结果），
    绝不静默返回假数据；日志全部输出到 stderr（stdout 专用于 MCP 协议）。

启动方式:
    python mcp_memory_server.py

参考实现：lms_http_mcp.py（同目录，既有 HTTP 转发薄桥范式）。
"""

import os
import sys
import json
import logging
from typing import Any

logger = logging.getLogger("mcp_memory_server")

# HTTP 转发（与 lms_http_mcp.py 同款依赖，requirements.txt 已含）
try:
    import requests  # noqa: E402
except Exception:  # pragma: no cover - 依赖缺失兜底
    requests = None

# ---------------------------------------------------------------------------
# 转发目标配置（阶段1-B：单写者 = API 进程）
# ---------------------------------------------------------------------------
LMS_API_URL = os.environ.get("LMS_URL", "http://localhost:8190").rstrip("/")
# 会话标识：默认 "main"（与 lms-http 桥的 sid=main 对齐），可用 LMS_MCP_SESSION_ID 覆盖
SESSION_ID = os.environ.get("LMS_MCP_SESSION_ID", "main").strip() or "main"

# 各端点超时（秒）：与 lms_http_mcp.py 的约定对齐（检索/存储放宽到 15s）
_TIMEOUTS = {
    "recall": 15.0,   # /recall 只读检索（目标 ~1s，放宽兜底）
    "feed": 15.0,     # /feed 塑形写（process_turn，无 LLM）
    "snapshot": 15.0,  # /snapshot 触发 API 落盘
    "status": 8.0,    # /status 状态查询
    "dream": 60.0,    # /dream 做梦（最慢路径，放宽）
}


def _http_post(path: str, payload: dict, timeout: float) -> dict:
    """POST 转发到 :8190（fail-open：失败抛异常，由调用方转 is_error）。"""
    if requests is None:
        raise RuntimeError("requests 依赖缺失，无法转发到 LMS API")
    resp = requests.post(
        f"{LMS_API_URL}{path}", json=payload, timeout=timeout)
    if resp.status_code >= 400:
        # 显式暴露错误（含 /feed 429 限流 / 404 会话不存在等），绝不静默吞掉
        try:
            detail = resp.json().get("detail", resp.text[:200])
        except Exception:
            detail = resp.text[:200]
        raise RuntimeError(
            f"LMS API {path} 返回 HTTP {resp.status_code}: {detail}")
    return resp.json()


def _http_get(path: str, timeout: float) -> dict:
    """GET 转发到 :8190（fail-open：失败抛异常，由调用方转 is_error）。"""
    if requests is None:
        raise RuntimeError("requests 依赖缺失，无法转发到 LMS API")
    resp = requests.get(f"{LMS_API_URL}{path}", timeout=timeout)
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text[:200])
        except Exception:
            detail = resp.text[:200]
        raise RuntimeError(
            f"LMS API {path} 返回 HTTP {resp.status_code}: {detail}")
    return resp.json()


# ---------------------------------------------------------------------------
# JSON 序列化辅助（防御性保留，兼容无 torch 环境的纯 Python 类型）
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """JSON 序列化的 default 回调，处理非原生类型。"""
    # torch.Tensor / numpy scalar -> Python 标量（薄桥下通常不会出现，防御性保留）
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
# 工具逻辑实现（薄桥转发，纯函数可独立测试）
# ---------------------------------------------------------------------------

def do_recall_memory(query: str) -> str:
    """检索与查询相关的历史对话记忆（转发 :8190 /recall 只读端点）。

    阶段1-B：由 HTTP API 完成编码+检索（不 process_turn、不调 LLM、不写缓冲），
    本进程只做请求转发，返回格式与旧直连版保持同构
    （记忆列表 + 记忆系统状态）。
    """
    data = _http_post("/recall", {
        "session_id": SESSION_ID,
        "query": query,
        "k": 3,  # 与旧直连版 top_k=3 保持一致
    }, _TIMEOUTS["recall"])

    results = data.get("results", []) if isinstance(data, dict) else []
    buffer_size = 0
    status = {}
    # 附加当前会话状态（只读 GET，失败不影响检索结果本身）
    try:
        st_data = _http_get(f"/status/{SESSION_ID}", _TIMEOUTS["status"])
        status = st_data.get("status", {}) if isinstance(st_data, dict) else {}
        buffer_size = status.get("episodic_buffer_size", 0)
    except Exception as e:
        logger.warning(f"检索后获取状态失败（忽略）: {e}")

    if not results:
        return (
            f"未检索到相关记忆（查询: \"{query}\"）。\n"
            f"情景记忆缓冲区大小: {buffer_size}\n\n"
            f"记忆系统状态:\n{_to_json(status)}"
        )

    lines = [
        f"检索到 {len(results)} 条相关记忆（查询: \"{query}\"）:",
        f"情景记忆缓冲区大小: {buffer_size}",
        "",
    ]
    for i, entry in enumerate(results, 1):
        text = (entry or {}).get("text", "")
        score = (entry or {}).get("score", 0.0)
        lines.append(f"--- 记忆 {i}（相关度 {score:.3f}）---")
        lines.append(text)
        lines.append("")

    lines.append(f"记忆系统状态:\n{_to_json(status)}")
    return "\n".join(lines)


def _parse_conversation(text: str) -> tuple[str, str]:
    """从对话文本中解析用户输入和助手回复（保留原逻辑，供 /feed 语义对齐）。

    /feed 的 process_turn(text, llm_output="") 与原直连版
    process_turn(user_input, llm_output) 在 llm_output 非空时拼接出的
    编码文本完全一致（"用户: {user_input}\\n助手: {llm_output}"），
    因此直接整段转发即可保持等价；本解析函数保留以便校验/兜底。
    """
    import re
    if not text:
        return "", ""
    pattern = r'\s*用户[:：]\s*(.*?)(?:\n\s*助手[:：]\s*(.*))?$'
    match = re.match(pattern, text, re.DOTALL)
    if match:
        user_input = match.group(1).strip()
        llm_output = match.group(2).strip() if match.group(2) else ""
        return user_input, llm_output
    return text, ""


def do_store_memory(text: str) -> str:
    """将对话内容存储到记忆系统（转发 :8190 /feed 塑形写 + /snapshot 落盘）。

    阶段1-B（单写者收口）：不再本地 process_turn / 不再本地写快照——
      1. POST /feed    → API 进程内 process_turn 写入该会话大脑（无 LLM，快）；
      2. POST /snapshot/{sid} → 触发 API 统一落盘（快照写者只有 API，P0-4 根除）；
      3. GET /status/{sid}    → 回填状态字段，保持旧返回契约的信息量。
    任一步失败均不静默：抛异常 → MCP 侧 is_error 可见。
    """
    # 语义校验（保留原解析函数，确认文本非空）
    user_input, _ = _parse_conversation(text)
    if not user_input and not text.strip():
        raise RuntimeError("store_memory: 输入文本为空")

    # 1. 塑形写入（process_turn，source 标记为 mcp 以区分总线）
    feed = _http_post("/feed", {
        "text": text,
        "session_id": SESSION_ID,
        "source": "mcp",
    }, _TIMEOUTS["feed"])

    # 2. 触发 API 落盘快照（保持旧"存储即保存快照"语义；失败仅告警不阻断）
    snapshot_saved = False
    snapshot_path = ""
    try:
        snap = _http_post(f"/snapshot/{SESSION_ID}", {}, _TIMEOUTS["snapshot"])
        snapshot_saved = bool(snap.get("saved", False))
        snapshot_path = snap.get("path", "")
    except Exception as e:
        logger.warning(f"快照触发失败（不影响存储本身）: {e}")

    # 3. 回填状态
    status = {}
    try:
        st_data = _http_get(f"/status/{SESSION_ID}", _TIMEOUTS["status"])
        status = st_data.get("status", {}) if isinstance(st_data, dict) else {}
    except Exception as e:
        logger.warning(f"存储后获取状态失败（忽略）: {e}")

    result = {
        "status": "已存储",
        "turn_count": int(feed.get("turn_count", status.get("turn_count", 0)) or 0),
        "episodic_buffer_size": status.get("episodic_buffer_size", 0),
        "last_entropy": status.get("last_entropy"),
        "last_surprise": status.get("last_surprise"),
        "precision_mean": status.get("precision_mean"),
        "purpose_coherence": status.get("purpose_coherence"),
        "snapshot_saved": snapshot_saved,
        "snapshot_path": snapshot_path,
        # /feed 不返回 memory_context（服务端丢弃）；置空并说明，避免调用方误用
        "memory_context": "",
    }
    return _to_json(result)


def do_get_memory_status() -> str:
    """获取记忆系统的当前运行状态（转发 :8190 GET /status/{sid}）。"""
    try:
        data = _http_get(f"/status/{SESSION_ID}", _TIMEOUTS["status"])
        status = data.get("status", {}) if isinstance(data, dict) else {}
        return _to_json(status)
    except RuntimeError as e:
        # 404 = 会话尚未创建（API 惰性创建）：返回明确提示，不伪装成错误
        if "404" in str(e):
            return _to_json({
                "exists": False,
                "session_id": SESSION_ID,
                "message": f"会话 '{SESSION_ID}' 尚未创建，首次 store/chat 时自动创建",
            })
        raise


def do_dream_memory(steps: int = 20, full_cycle: bool = False) -> str:
    """触发记忆系统的"做梦"过程（转发 :8190 POST /dream/{sid}）。

    做梦在 API 进程内执行（单写者），快照由 API 统一保存；
    本进程不加载任何记忆状态。
    """
    data = _http_post(f"/dream/{SESSION_ID}", {
        "steps": int(steps),
        "full_cycle": bool(full_cycle),
    }, _TIMEOUTS["dream"])
    # /dream 返回 {"session_id": ..., "result": {...}}，解包 result 保持旧契约
    if isinstance(data, dict) and "result" in data:
        return _to_json(data["result"])
    return _to_json(data)


# ---------------------------------------------------------------------------
# 工具定义（契约不变：工具名/参数与阶段1-A 完全一致）
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "recall_memory",
        "description": (
            "检索与查询相关的历史对话记忆。当你需要回忆之前与用户讨论的内容时"
            "调用此工具。例如：用户问'你还记得我之前说的吗'，或当你需要之前的"
            "上下文信息时。建议在每次对话开始时主动调用。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索查询文本",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "store_memory",
        "description": (
            "将当前对话内容存储到记忆系统。在每次与用户完成一轮对话后调用此工具，"
            "保存用户的输入和你的回复。这使系统能在未来检索到本次对话内容。"
        ),
        "inputSchema": {
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
    },
    {
        "name": "get_memory_status",
        "description": (
            "获取记忆系统的当前状态（轮次数、熵、惊讶度、precision等）。"
            "用于了解记忆系统的运作情况。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "dream_memory",
        "description": (
            "触发记忆系统的'做梦'过程，在空闲时进行记忆巩固、遗忘和整合。"
            "建议在对话间歇时调用。做梦会让记忆系统在无输入时持续运转，"
            "精炼吸引子景观、衰减低价值记忆、演化目的层，使记忆从'冷工具'"
            "变为'活体'。"
        ),
        "inputSchema": {
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
    },
]


# ---------------------------------------------------------------------------
# stdio JSON-RPC 薄协议处理（与 lms_http_mcp.py 同款，不依赖 mcp SDK）
# ---------------------------------------------------------------------------

def handle_request(request: dict) -> dict | None:
    """处理单个 MCP JSON-RPC 请求（无 id 的通知消息返回 None）。"""
    method = request.get("method", "")
    params = request.get("params", {}) or {}
    req_id = request.get("id")

    # 跳过没有 id 的通知消息（如 notifications/initialized）
    if req_id is None:
        return None

    try:
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "living-memory", "version": "1.0.0-phase1b"},
                },
            }

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": _TOOLS},
            }

        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {}) or {}

            if tool_name == "recall_memory":
                result = do_recall_memory(arguments.get("query", ""))
            elif tool_name == "store_memory":
                result = do_store_memory(arguments.get("text", ""))
            elif tool_name == "get_memory_status":
                result = do_get_memory_status()
            elif tool_name == "dream_memory":
                result = do_dream_memory(
                    steps=arguments.get("steps", 20),
                    full_cycle=arguments.get("full_cycle", False),
                )
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(
                            {"success": False, "error": f"Unknown tool: {tool_name}"},
                            ensure_ascii=False)}],
                        "isError": True,
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": result}],
                    "isError": False,
                },
            }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown method: {method}"},
            }

    except Exception as e:
        # fail-open：错误显式暴露（含 LMS API 不可达/限流/4xx），绝不静默
        logger.error(f"请求处理出错: {e}", exc_info=True)
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": f"工具执行出错: {e}"}],
                "isError": True,
            },
        }


def main() -> None:
    """主循环：从 stdin 读取 JSON-RPC 请求，处理，输出到 stdout。"""
    logger.info(
        f"MCP 服务器 'living-memory' 正在启动 (stdio 薄桥, "
        f"LMS_API={LMS_API_URL}, session_id={SESSION_ID})...")
    buffer = ""
    for line in sys.stdin:
        buffer += line
        if buffer.strip():
            try:
                request = json.loads(buffer)
                response = handle_request(request)
                if response is not None:
                    print(json.dumps(response, ensure_ascii=False))
                    sys.stdout.flush()
            except json.JSONDecodeError:
                pass  # 半行 JSON：继续累积
            buffer = ""


if __name__ == "__main__":
    # 日志输出到 stderr（stdout 专用于 MCP 协议通信）
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()

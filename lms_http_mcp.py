#!/usr/bin/env python3
"""
LMS MCP 工具 - 通过HTTP API调用活体记忆系统

提供工具：
- lms_recall: 检索记忆context
- lms_store: 存储对话到记忆系统
- lms_status: 查询记忆系统状态
"""

import os
import sys
import json
import requests
from typing import Optional

# [2026-08-30 修复] V1 LMS :8190 已停用（四妹 V2 切换：agentos-v2/lms-api 监听 :8191）。
# 旧硬编码导致 DSH 的 MCP 工具调用死等（toolCallTimeoutMs 15s）→ 四妹“思考几秒就已停止”。
# 改为 env 可覆盖，默认指向当前在跑的 V2。
LMS_API_URL = os.environ.get("LMS_URL", "http://localhost:8191")
DEFAULT_SESSION = "main"


def lms_recall(user_input: str, sid: str = DEFAULT_SESSION) -> dict:
    """检索记忆context（纯只读）

    P0-9/T1.3：转发 :8190 /recall 只读端点——不 process_turn、不调 LLM、
    不写缓冲、不落盘，turn_count 零增量。

    2026-08-17 修复（recall 纯只读）：此前误 POST /chat（写路径，内部
    process_turn → turn_count += 1），每次检索 +1 turn，污染 allostatic
    统计与 turn 语义（754=750+4 实证）。现改走 /recall，并附加一次只读
    GET /status 组装 memory_state（与旧 /chat 返回形态同构）。

    Args:
        user_input: 用户输入
        sid: 会话ID（默认main）

    Returns:
        包含记忆context的字典
    """
    try:
        # 只读检索：query 走 /recall 的 query 字段（k=5 与服务端默认一致）。
        # 注意：/chat 是写路径（process_turn），检索禁用；/recall 才是只读口。
        response = requests.post(
            f"{LMS_API_URL}/recall",
            json={"session_id": sid, "query": user_input, "k": 5},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        # /recall 响应自带 turn_count 锚点（只读校验用，测试与可观测性）
        anchor_turn = int(data.get("turn_count", 0) or 0) \
            if isinstance(data, dict) else 0

        # 组装 memory_context（相关记忆文本；与旧 /chat 返回形态同构）
        results = data.get("results", []) if isinstance(data, dict) else []
        if results:
            lines = []
            for i, item in enumerate(results, 1):
                text = (item or {}).get("text", "")
                score = (item or {}).get("score", 0.0)
                lines.append(f"--- 记忆 {i}（相关度 {score:.3f}）---\n{text}")
            memory_context = "\n\n".join(lines)
        else:
            memory_context = ""

        # 附加当前会话状态（只读 GET；失败不影响检索结果本身）
        memory_state = {}
        turn_count = anchor_turn
        try:
            st = requests.get(f"{LMS_API_URL}/status/{sid}", timeout=5)
            if st.status_code == 200:
                status = st.json().get("status", {})
                if isinstance(status, dict):
                    memory_state = status
                    turn_count = int(status.get("turn_count", 0) or 0)
        except Exception:
            pass  # /recall 的 turn_count 锚点兜底

        return {
            "success": True,
            "memory_context": memory_context,
            "memory_state": memory_state,
            "turn_count": turn_count
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def lms_store(user_input: str, llm_output: str = "", sid: str = DEFAULT_SESSION) -> dict:
    """存储对话到记忆系统
    
    Args:
        user_input: 用户输入
        llm_output: AI回复（可选）
        sid: 会话ID
    
    Returns:
        存储结果
    """
    try:
        # P0-2 止血修复：旧实现拼接了 text 却从未发送（死代码），且请求体用 sid
        # 字段（服务端 pydantic 默认忽略未知字段）→ 所有 MCP 存储静默落进 default 脑。
        # 现在真正 POST {session_id, user_input, llm_output}，文本由服务端
        # process_turn(user_input, llm_output) 落库（session_id 为统一字段名）。
        response = requests.post(
            f"{LMS_API_URL}/chat",
            json={
                "session_id": sid,
                "user_input": user_input,
                "llm_output": llm_output,
            },
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "success": True,
            "stored": True,
            "turn_count": data.get("memory_state", {}).get("turn_count", 0)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def lms_status(sid: str = DEFAULT_SESSION) -> dict:
    """查询记忆系统状态
    
    Args:
        sid: 会话ID
    
    Returns:
        系统状态
    """
    try:
        response = requests.get(
            f"{LMS_API_URL}/status/{sid}",
            timeout=5
        )
        
        if response.status_code == 404:
            # 会话不存在，返回默认状态
            return {
                "success": True,
                "exists": False,
                "message": f"会话 '{sid}' 不存在，将自动创建"
            }
        
        response.raise_for_status()
        data = response.json()
        
        return {
            "success": True,
            "exists": True,
            "turn_count": data.get("turn_count", 0),
            "num_nodes": data.get("num_nodes", 256),
            "last_entropy": data.get("last_entropy"),
            "last_surprise": data.get("last_surprise"),
            "precision_mean": data.get("precision_mean"),
            "episodic_buffer_size": data.get("episodic_buffer_size", 0)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def lms_dream(sid: str = DEFAULT_SESSION) -> dict:
    """手动触发做梦
    
    Args:
        sid: 会话ID
    
    Returns:
        做梦结果
    """
    try:
        response = requests.post(
            f"{LMS_API_URL}/dream/{sid}",
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "success": True,
            "dream_completed": True,
            "details": data
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def lms_snapshot(sid: str = DEFAULT_SESSION) -> dict:
    """保存快照
    
    Args:
        sid: 会话ID
    
    Returns:
        快照结果
    """
    try:
        response = requests.post(
            f"{LMS_API_URL}/snapshot/{sid}",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        return {
            "success": True,
            "snapshot_saved": True,
            "path": data.get("path", "")
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# MCP协议处理
def handle_request(request: dict) -> dict:
    """处理MCP请求"""
    method = request.get("method", "")
    params = request.get("params", {})
    req_id = request.get("id")
    
    # 跳过没有id的通知消息
    if req_id is None:
        return None
    
    try:
        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "lms_recall",
                            "description": "从活体记忆系统检索记忆context，返回与当前输入相关的记忆状态",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "user_input": {"type": "string", "description": "用户输入"},
                                    "sid": {"type": "string", "description": "会话ID", "default": "main"}
                                },
                                "required": ["user_input"]
                            }
                        },
                        {
                            "name": "lms_store",
                            "description": "将对话存储到活体记忆系统",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "user_input": {"type": "string", "description": "用户输入"},
                                    "llm_output": {"type": "string", "description": "AI回复（可选）"},
                                    "sid": {"type": "string", "description": "会话ID", "default": "main"}
                                },
                                "required": ["user_input"]
                            }
                        },
                        {
                            "name": "lms_status",
                            "description": "查询活体记忆系统状态",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "sid": {"type": "string", "description": "会话ID", "default": "main"}
                                }
                            }
                        },
                        {
                            "name": "lms_dream",
                            "description": "手动触发记忆系统做梦（整合记忆）",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "sid": {"type": "string", "description": "会话ID", "default": "main"}
                                }
                            }
                        },
                        {
                            "name": "lms_snapshot",
                            "description": "保存记忆系统快照",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "sid": {"type": "string", "description": "会话ID", "default": "main"}
                                }
                            }
                        }
                    ]
                }
            }
        
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            
            if tool_name == "lms_recall":
                result = lms_recall(
                    arguments.get("user_input", ""),
                    arguments.get("sid", DEFAULT_SESSION)
                )
            elif tool_name == "lms_store":
                result = lms_store(
                    arguments.get("user_input", ""),
                    arguments.get("llm_output", ""),
                    arguments.get("sid", DEFAULT_SESSION)
                )
            elif tool_name == "lms_status":
                result = lms_status(arguments.get("sid", DEFAULT_SESSION))
            elif tool_name == "lms_dream":
                result = lms_dream(arguments.get("sid", DEFAULT_SESSION))
            elif tool_name == "lms_snapshot":
                result = lms_snapshot(arguments.get("sid", DEFAULT_SESSION))
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"success": False, "error": f"Unknown tool: {tool_name}"}, ensure_ascii=False)}],
                        "isError": True,
                    },
                }
            
            # MCP协议：tools/call 的 result 必须是 {"content":[{"type":"text","text":"..."}],"isError":...}
            is_error = isinstance(result, dict) and not result.get("success", True)
            try:
                text = json.dumps(result, ensure_ascii=False)
            except Exception:
                text = str(result)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": text}],
                    "isError": is_error,
                },
            }
        
        elif method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "lms-http-tools", "version": "1.0.0"}
                }
            }
        
        else:
            return {"jsonrpc": "2.0", "id": req_id, "error": f"Unknown method: {method}"}
    
    except Exception as e:
        return {"jsonrpc": "2.0", "id": req_id, "error": str(e)}


def main():
    """主循环：从stdin读取JSON请求，处理，输出到stdout"""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="[LMS-HTTP-MCP] %(message)s",
        stream=sys.stderr
    )
    logging.info("LMS HTTP MCP 工具启动")
    
    buffer = ""
    for line in sys.stdin:
        buffer += line
        if buffer.strip():
            try:
                request = json.loads(buffer)
                response = handle_request(request)
                if response:  # 只响应有id的请求
                    print(json.dumps(response, ensure_ascii=False))
                    sys.stdout.flush()
            except json.JSONDecodeError:
                pass
            buffer = ""


if __name__ == "__main__":
    main()
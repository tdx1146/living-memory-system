"""活体记忆系统 - FastAPI 服务

提供 HTTP 接口，使 TRAE/OpenClaw UI 在对话时后台自动生成记忆并适时注入。

端点概览:
    POST   /chat              完整对话（手动控制 LLM 注入时机）
    POST   /chat/simple       简化对话（自动处理记忆+LLM查询）
    GET    /status/{sid}      查询会话记忆状态
    POST   /snapshot/{sid}    保存快照
    POST   /restore/{sid}     从快照恢复
    GET    /sessions          列出所有会话
    DELETE /sessions/{sid}    删除指定会话
    GET    /health            健康检查

记忆处理与 LLM 调用均为同步执行（torch 非 async），FastAPI 端点使用
run_in_executor 将阻塞调用交给线程池，避免阻塞事件循环。
"""

import os
import time
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.session_manager import SessionManager
from api.config import get_api_config

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api.server")

# ---------------------------------------------------------------------------
# 全局会话管理器（惰性初始化，避免导入时即加载预训练模型）
# ---------------------------------------------------------------------------
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """获取全局 SessionManager 单例。"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
        logger.info("SessionManager 已初始化")
    return _session_manager


# ---------------------------------------------------------------------------
# Pydantic 请求/响应模型
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    user_input: str = Field(..., description="用户输入文本")
    session_id: str = Field("default", description="会话标识")
    llm_output: str = Field("", description="上一轮LLM输出（可选）")


class ChatResponse(BaseModel):
    response: str = Field(..., description="LLM回复（未配置LLM时为记忆context）")
    memory_context: str = Field(..., description="记忆context")
    memory_state: dict = Field(default_factory=dict, description="记忆系统状态")
    session_id: str = Field(..., description="会话标识")


class SimpleChatRequest(BaseModel):
    user_input: str = Field(..., description="用户输入文本")
    session_id: str = Field("default", description="会话标识")


class SimpleChatResponse(BaseModel):
    response: str = Field(..., description="LLM回复")
    memory_status: dict = Field(default_factory=dict, description="记忆系统状态")


class SnapshotRequest(BaseModel):
    path: Optional[str] = Field(
        None, description="自定义快照路径（默认 ./snapshots/{sid}_{timestamp}.pt）")


class RestoreRequest(BaseModel):
    path: str = Field(..., description="快照文件路径")


# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------
app = FastAPI(
    title="活体记忆系统 API",
    description=(
        "为 TRAE/OpenClaw UI 提供后台记忆服务的 HTTP 接口。"
        "对话时自动生成记忆并适时注入。"
    ),
    version="0.1.0",
)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _now_ts() -> str:
    """返回紧凑时间戳字符串（用于快照文件名）。"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _status_for(loop) -> dict:
    """从 LivingMemoryLoop 提取状态字典（含情景缓冲区与LLM标记）。"""
    status = loop.get_status()
    try:
        status['episodic_buffer_size'] = len(loop.memory._episodic_buffer)
    except Exception:
        status['episodic_buffer_size'] = 0
    status['llm_enabled'] = loop.bridge is not None
    return status


# ---------------------------------------------------------------------------
# 端点
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """健康检查。"""
    sm = get_session_manager()
    return {
        "status": "ok",
        "service": "living-memory-api",
        "version": app.version,
        "active_sessions": len(sm.list_sessions()),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """完整对话端点。

    流程:
        1. 获取或创建 session 的 LivingMemoryLoop
        2. process_turn(user_input, llm_output) 获取记忆 context
        3. 若配置了 LLM，调用 bridge.query(user_input, context) 获取回复
        4. 若未配置 LLM，返回记忆 context 作为 response
    LLM 调用失败时返回记忆 context 和错误信息，不中断服务。
    """
    if not req.user_input:
        raise HTTPException(status_code=400, detail="user_input 不能为空")

    sm = get_session_manager()
    loop = sm.get_or_create(req.session_id)

    # 1. 处理本轮，生成记忆 context（同步阻塞调用）
    memory_context = loop.process_turn(req.user_input, req.llm_output)

    # 2. 查询 LLM（若已配置）
    response_text = memory_context
    if loop.bridge is not None:
        try:
            response_text = loop.bridge.query(req.user_input, memory_context)
            logger.info(
                f"[{req.session_id}] LLM 回复长度: {len(response_text)}")
        except Exception as e:
            logger.error(f"[{req.session_id}] LLM 调用失败: {e}")
            response_text = (
                f"[记忆context已生成，但LLM调用失败: {e}]\n\n"
                f"---记忆context---\n{memory_context}"
            )
    else:
        logger.info(
            f"[{req.session_id}] 未配置 LLM，返回记忆 context")

    memory_state = _status_for(loop)
    return ChatResponse(
        response=response_text,
        memory_context=memory_context,
        memory_state=memory_state,
        session_id=req.session_id,
    )


@app.post("/chat/simple", response_model=SimpleChatResponse)
async def chat_simple(req: SimpleChatRequest):
    """简化对话端点。

    仅需 user_input，内部调用 loop.query_llm() 自动完成记忆+LLM查询。
    若未配置 LLM 会抛出 RuntimeError，此处转为 HTTP 503。
    """
    if not req.user_input:
        raise HTTPException(status_code=400, detail="user_input 不能为空")

    sm = get_session_manager()
    loop = sm.get_or_create(req.session_id)

    if loop.bridge is None:
        raise HTTPException(
            status_code=503,
            detail="未配置 LLM Bridge，无法使用 /chat/simple。"
                   "请配置 DEEPSEEK_API_KEY 后重试，或改用 /chat 端点。",
        )

    try:
        response_text = loop.query_llm(req.user_input)
        logger.info(
            f"[{req.session_id}] (simple) LLM 回复长度: {len(response_text)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{req.session_id}] (simple) 查询失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"LLM 查询失败: {e}",
        )

    memory_status = _status_for(loop)
    return SimpleChatResponse(
        response=response_text,
        memory_status=memory_status,
    )


@app.get("/status/{session_id}")
async def get_status(session_id: str):
    """返回指定 session 的记忆系统状态。"""
    sm = get_session_manager()
    status = sm.get_status(session_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"会话 '{session_id}' 不存在",
        )
    return {
        "session_id": session_id,
        "status": status,
    }


@app.post("/snapshot/{session_id}")
async def snapshot(session_id: str, req: SnapshotRequest):
    """保存指定 session 的快照。

    默认保存到 ./snapshots/{session_id}_{timestamp}.pt
    """
    sm = get_session_manager()
    loop = sm.get(session_id)
    if loop is None:
        raise HTTPException(
            status_code=404,
            detail=f"会话 '{session_id}' 不存在",
        )

    snapshot_dir = "./snapshots"
    os.makedirs(snapshot_dir, exist_ok=True)

    if req.path:
        path = req.path
    else:
        path = os.path.join(
            snapshot_dir, f"{session_id}_{_now_ts()}.pt")

    try:
        loop.save_state(path)
        abs_path = os.path.abspath(path)
        logger.info(f"[{session_id}] 快照已保存: {abs_path}")
        return {
            "session_id": session_id,
            "saved": True,
            "path": abs_path,
            "turn_count": loop.turn_count,
        }
    except Exception as e:
        logger.error(f"[{session_id}] 快照保存失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"快照保存失败: {e}",
        )


@app.post("/restore/{session_id}")
async def restore(session_id: str, req: RestoreRequest):
    """从快照恢复指定 session 的状态。"""
    if not req.path:
        raise HTTPException(status_code=400, detail="path 不能为空")

    sm = get_session_manager()
    loop = sm.get(session_id)
    if loop is None:
        raise HTTPException(
            status_code=404,
            detail=f"会话 '{session_id}' 不存在",
        )

    if not os.path.exists(req.path):
        raise HTTPException(
            status_code=404,
            detail=f"快照文件不存在: {req.path}",
        )

    try:
        loop.load_state(req.path)
        logger.info(f"[{session_id}] 已从 {req.path} 恢复状态")
        return {
            "session_id": session_id,
            "restored": True,
            "path": req.path,
            "status": _status_for(loop),
        }
    except Exception as e:
        logger.error(f"[{session_id}] 恢复失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"恢复失败: {e}",
        )


@app.get("/sessions")
async def list_sessions():
    """列出所有活跃的 session_id。"""
    sm = get_session_manager()
    sessions = sm.list_sessions()
    return {
        "sessions": sessions,
        "count": len(sessions),
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定 session。"""
    sm = get_session_manager()
    deleted = sm.remove(session_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"会话 '{session_id}' 不存在",
        )
    return {
        "session_id": session_id,
        "deleted": True,
    }


# ---------------------------------------------------------------------------
# 启动事件
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    """服务启动时记录日志。"""
    logger.info("活体记忆系统 API 服务启动")
    # 预热默认配置（不创建 session，仅打印配置摘要）
    try:
        cfg = get_api_config()
        llm_on = cfg.get('llm_api') is not None
        logger.info(
            f"默认配置: nodes={cfg['num_nodes']}, "
            f"dim={cfg['input_dim']}, "
            f"embedder={'Pretrained' if hasattr(cfg.get('embedder'), 'embed_text') else 'Simple'}, "
            f"llm={'ON' if llm_on else 'OFF'}"
        )
    except Exception as e:
        logger.warning(f"读取默认配置失败: {e}")


@app.on_event("shutdown")
async def on_shutdown():
    """服务关闭时清理。"""
    sm = get_session_manager()
    n = sm.clear()
    logger.info(f"服务关闭，已清理 {n} 个会话")


# 模块导入时确保项目根目录在 sys.path 中（支持 `python api/run.py` 启动）
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

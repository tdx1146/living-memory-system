"""活体记忆系统 - FastAPI 服务（含常驻做梦后台）

提供 HTTP 接口，使 TRAE/OpenClaw UI 在对话时后台自动生成记忆并适时注入。
内置 DreamScheduler 后台线程，在无对话时自动触发做梦引擎，
让记忆系统真正"活着"——即使没有人说话，系统也在持续进行记忆巩固。

端点概览:
    POST   /chat              完整对话（手动控制 LLM 注入时机）
    POST   /chat/simple       简化对话（自动处理记忆+LLM查询）
    GET    /status/{sid}      查询会话记忆状态
    POST   /snapshot/{sid}    保存快照
    POST   /restore/{sid}     从快照恢复
    GET    /sessions          列出所有会话
    DELETE /sessions/{sid}    删除指定会话
    POST   /dream/{sid}       手动触发做梦
    GET    /dream/status      查询做梦调度器状态
    GET    /health            健康检查

记忆处理与 LLM 调用均为同步执行（torch 非 async），FastAPI 端点使用
run_in_executor 将阻塞调用交给线程池，避免阻塞事件循环。
"""

import os
import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.session_manager import SessionManager
from api.config import get_api_config
from runtime.dream_scheduler import DreamScheduler

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("api.server")

# ---------------------------------------------------------------------------
# 全局会话管理器与做梦调度器（惰性初始化，避免导入时即加载预训练模型）
# ---------------------------------------------------------------------------
_session_manager: Optional[SessionManager] = None
_dream_scheduler: Optional[DreamScheduler] = None


def get_session_manager() -> SessionManager:
    """获取全局 SessionManager 单例。"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
        logger.info("SessionManager 已初始化")
    return _session_manager


def get_dream_scheduler() -> DreamScheduler:
    """获取全局 DreamScheduler 单例（惰性初始化）。

    DreamScheduler 通过 session_manager.get(session_id) 获取 loop 实例，
    在后台线程中自动触发空闲会话的做梦周期。
    """
    global _dream_scheduler
    if _dream_scheduler is None:
        sm = get_session_manager()

        def _get_loop(session_id: str):
            return sm.get(session_id)

        _dream_scheduler = DreamScheduler(
            get_loop_fn=_get_loop,
            idle_threshold=float(os.environ.get("DREAM_IDLE_THRESHOLD", "30")),
            dream_steps=int(os.environ.get("DREAM_STEPS", "20")),
            dream_full_cycle=os.environ.get(
                "DREAM_FULL_CYCLE", "false").lower() == "true",
            check_interval=float(os.environ.get("DREAM_CHECK_INTERVAL", "5")),
        )
        logger.info("DreamScheduler 已初始化")
    return _dream_scheduler


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


class DreamRequest(BaseModel):
    steps: int = Field(20, description="做梦步数")
    full_cycle: bool = Field(False, description="是否启用完整七阶段周期")


# ---------------------------------------------------------------------------
# Phase 4 / D0：总线事件喂入（只喂不指挥）
# ---------------------------------------------------------------------------
class FeedRequest(BaseModel):
    text: str = Field(..., description="总线事件文本摘要（喂入塑形输入侧）")
    session_id: str = Field("bus", description="会话标识（默认 bus，与总线隔离）")
    source: str = Field("event_bus", description="事件来源标记")


class FeedResponse(BaseModel):
    status: str = Field("ok", description="处理状态")
    entropy: float = Field(0.0, description="本轮回塑形后的激活熵")
    surprise: float = Field(0.0, description="本轮回塑形后的惊讶度")
    turn_count: int = Field(0, description="该会话累计轮次")
    session_id: str = Field(..., description="会话标识")


# /feed 限流状态（默认 ≤10 次/分钟，防总线风暴；LMS_FEED_RATE_LIMIT 可覆盖）
_feed_rate = {"window_start": 0.0, "count": 0}
_feed_rate_lock = asyncio.Lock()
_FEED_RATE_LIMIT = int(os.environ.get("LMS_FEED_RATE_LIMIT", "10"))
_FEED_RATE_WINDOW = 60.0


async def _feed_rate_limited() -> bool:
    """滑动窗口限流：窗口内超过上限返回 True（429）。"""
    global _feed_rate
    now = time.time()
    async with _feed_rate_lock:
        if now - _feed_rate["window_start"] >= _FEED_RATE_WINDOW:
            _feed_rate["window_start"] = now
            _feed_rate["count"] = 0
        _feed_rate["count"] += 1
        return _feed_rate["count"] > _FEED_RATE_LIMIT


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
        status['episodic_buffer_size'] = loop.memory.episodic_size()
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
        2. 通过 DreamScheduler 获取对话权限（等待做梦完成）
        3. process_turn(user_input, llm_output) 获取记忆 context
        4. 若配置了 LLM，调用 bridge.query(user_input, context) 获取回复
        5. 若未配置 LLM，返回记忆 context 作为 response
        6. 释放对话权限，更新会话活动时间（阻止后续做梦）
    LLM 调用失败时返回记忆 context 和错误信息，不中断服务。
    """
    if not req.user_input:
        raise HTTPException(status_code=400, detail="user_input 不能为空")

    sm = get_session_manager()
    scheduler = get_dream_scheduler()
    loop = sm.get_or_create(req.session_id)
    scheduler.register_session(req.session_id)

    # 获取对话权限（等待做梦完成），更新活动时间
    acquired = scheduler.acquire_conversation(req.session_id)
    scheduler.touch(req.session_id)
    if not acquired:
        # 做梦未结束且等待超时：拒绝请求以避免对话处理与后台做梦并发写
        # 同一份记忆状态（J矩阵/precision/latent）
        raise HTTPException(
            status_code=503,
            detail="系统正在做梦（记忆巩固中），请稍后重试。")
    try:
        # 1. 处理本轮，生成记忆 context（同步阻塞调用，交由线程池执行）
        memory_context = await asyncio.get_event_loop().run_in_executor(
            None, lambda: loop.process_turn(req.user_input, req.llm_output))

        # 2. 查询 LLM（若已配置）
        response_text = memory_context
        if loop.bridge is not None:
            try:
                response_text = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: loop.bridge.query(req.user_input, memory_context))
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
    finally:
        scheduler.release_conversation(req.session_id)

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
        response_text = await asyncio.get_event_loop().run_in_executor(
            None, lambda: loop.query_llm(req.user_input))
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


@app.post("/feed", response_model=FeedResponse)
async def feed(req: FeedRequest):
    """总线事件喂入端点（Phase 4 / D0 方向 1：只喂不指挥）。

    把总线事件文本摘要送入 LMS 塑形输入侧（process_turn），
    塑形但不产 LLM 回复（process_turn 返回的记忆 context 直接丢弃）。
    LMS 内部如何塑形（权重/吸引子演化）是它自己的事，总线不指挥。

    - 限流：默认 ≤10 次/分钟（LMS_FEED_RATE_LIMIT 可覆盖），超限 429
    - 与 /chat 等现有端点完全独立，互不影响
    - 做梦协调：与对话请求同等对待（等待做梦完成），防止并发写记忆状态
    """
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")

    if await _feed_rate_limited():
        raise HTTPException(
            status_code=429,
            detail=f"feed 限流：超过 {_FEED_RATE_LIMIT} 次/分钟，请稍后重试",
        )

    sm = get_session_manager()
    scheduler = get_dream_scheduler()
    loop = sm.get_or_create(req.session_id)
    scheduler.register_session(req.session_id)

    acquired = scheduler.acquire_conversation(req.session_id)
    scheduler.touch(req.session_id)
    if not acquired:
        raise HTTPException(
            status_code=503,
            detail="系统正在做梦（记忆巩固中），请稍后重试。")
    try:
        # 塑形但不产 LLM 回复：返回值是记忆 context，直接丢弃
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: loop.process_turn(req.text, llm_output=""))
        status = loop.get_status()
        logger.info(
            f"[{req.session_id}] /feed 塑形完成 "
            f"(source={req.source}, text_len={len(req.text)})")
    finally:
        scheduler.release_conversation(req.session_id)

    return FeedResponse(
        status="ok",
        entropy=float(status.get("last_entropy", 0.0) or 0.0),
        surprise=float(status.get("last_surprise", 0.0) or 0.0),
        turn_count=int(status.get("turn_count", 0) or 0),
        session_id=req.session_id,
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
    scheduler = get_dream_scheduler()
    deleted = sm.remove(session_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"会话 '{session_id}' 不存在",
        )
    scheduler.unregister_session(session_id)
    return {
        "session_id": session_id,
        "deleted": True,
    }


# ---------------------------------------------------------------------------
# 做梦端点
# ---------------------------------------------------------------------------

@app.post("/dream/{session_id}")
async def trigger_dream(session_id: str, req: Optional[DreamRequest] = None):
    """手动触发指定会话的做梦。

    若会话不存在则创建。做梦期间该会话的对话请求会等待。
    """
    sm = get_session_manager()
    scheduler = get_dream_scheduler()
    # 确保会话存在（get_or_create 的副作用：不存在时自动创建）
    sm.get_or_create(session_id)
    scheduler.register_session(session_id)

    steps = req.steps if req else 20
    full_cycle = req.full_cycle if req else False

    # 在线程池中执行（避免阻塞事件循环）
    loop_result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: scheduler.trigger_dream(
            session_id, steps=steps, full_cycle=full_cycle)
    )
    return {
        "session_id": session_id,
        "result": loop_result,
    }


@app.get("/dream/status")
async def dream_status():
    """查询做梦调度器状态。"""
    scheduler = get_dream_scheduler()
    return scheduler.get_status()


# ---------------------------------------------------------------------------
# 启动事件
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def on_startup():
    """服务启动时初始化并启动做梦调度器。"""
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

    # 启动做梦调度器后台线程
    scheduler = get_dream_scheduler()
    scheduler.start()
    logger.info(
        f"DreamScheduler 已启动 "
        f"(idle_threshold={scheduler.idle_threshold}s, "
        f"dream_steps={scheduler.dream_steps})")


@app.on_event("shutdown")
async def on_shutdown():
    """服务关闭时停止调度器并清理会话。"""
    scheduler = get_dream_scheduler()
    scheduler.stop()
    logger.info("DreamScheduler 已停止")

    sm = get_session_manager()
    n = sm.clear()
    logger.info(f"服务关闭，已清理 {n} 个会话")


# 模块导入时确保项目根目录在 sys.path 中（支持 `python api/run.py` 启动）
import sys  # noqa: E402
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

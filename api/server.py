"""活体记忆系统 - FastAPI 服务（含常驻做梦后台）

提供 HTTP 接口，使 TRAE/OpenClaw UI 在对话时后台自动生成记忆并适时注入。
内置 DreamScheduler 后台线程，在无对话时自动触发做梦引擎，
让记忆系统真正"活着"——即使没有人说话，系统也在持续进行记忆巩固。

端点概览:
    POST   /chat              完整对话（手动控制 LLM 注入时机）
    POST   /chat/simple       简化对话（自动处理记忆+LLM查询）
    POST   /recall            只读情景检索（T1.3/P0-9：不 process_turn、不调 LLM）
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
import re
import math
import uuid
import asyncio
import hashlib
import logging
import threading
import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from api.session_manager import SessionManager
from api.config import get_api_config
from runtime.dream_scheduler import DreamScheduler
from persistence.audit import audit

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
    # P0-3 止血：兼容字段 sid（旧版 lms-http MCP 客户端发送）。
    # pydantic 默认 extra="ignore" 会静默丢弃未知字段 → 旧客户端所有 /chat
    # 请求的 sid 被丢弃、session_id 恒为 default，全部流量静默落进 default 脑
    # （P0-2/P0-3 根因）。显式声明别名并在映射时记录告警，防静默丢弃重演。
    sid: Optional[str] = Field(None, description="session_id 兼容别名（旧客户端）")

    @model_validator(mode="after")
    def _apply_sid_alias(self) -> "ChatRequest":
        """P0-3：旧客户端 sid → session_id 映射（session_id 显式指定时以它为准）。"""
        if self.sid is not None:
            if self.session_id == "default":
                self.session_id = self.sid
                logger.warning(
                    f"[ChatRequest] 兼容字段 sid='{self.sid}' 已映射为 session_id"
                    "（P0-3 兼容；新客户端请直接使用 session_id）")
            elif self.session_id != self.sid:
                logger.warning(
                    f"[ChatRequest] 同时收到 session_id='{self.session_id}' 与 sid='{self.sid}'，"
                    "以 session_id 为准（P0-3 兼容告警）")
        return self


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


class RecallRequest(BaseModel):
    """T1.3/P0-9：/recall 只读检索请求。"""
    session_id: str = Field("default", description="会话标识")
    query: str = Field(..., description="检索查询文本")
    k: int = Field(5, description="返回条数（服务端钳制到 1-20）")
    # 兼容旧客户端 sid 字段（与 ChatRequest 同策略，防静默丢弃）
    sid: Optional[str] = Field(None, description="session_id 兼容别名（旧客户端）")

    @model_validator(mode="after")
    def _apply_sid_alias(self) -> "RecallRequest":
        """旧客户端 sid → session_id 映射（与 ChatRequest 的 P0-3 兼容同款）。"""
        if self.sid is not None:
            if self.session_id == "default":
                self.session_id = self.sid
                logger.warning(
                    f"[RecallRequest] 兼容字段 sid='{self.sid}' 已映射为 session_id"
                    "（P0-3 兼容；新客户端请直接使用 session_id）")
            elif self.session_id != self.sid:
                logger.warning(
                    f"[RecallRequest] 同时收到 session_id='{self.session_id}' 与 "
                    f"sid='{self.sid}'，以 session_id 为准（P0-3 兼容告警）")
        return self


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
    surprise: float = Field(0.0, description="本轮回塑形后的惊讶度（准确性项，恒≥0）")
    free_energy: float = Field(0.0, description="本轮回塑形后的自由能（未规范化变分能量，可负；仅供学习目标/诊断）")
    mse: float = Field(0.0, description="本轮回塑形后的均方预测误差（跨尺度可比）")
    turn_count: int = Field(0, description="该会话累计轮次")
    session_id: str = Field(..., description="会话标识")


# /feed 限流状态（默认 ≤10 次/分钟，防总线风暴；LMS_FEED_RATE_LIMIT 可覆盖）
_feed_rate = {"window_start": 0.0, "count": 0}
_feed_rate_lock = asyncio.Lock()
_FEED_RATE_LIMIT = int(os.environ.get("LMS_FEED_RATE_LIMIT", "10"))
_FEED_RATE_WINDOW = 60.0


async def _feed_rate_check() -> Optional[float]:
    """滑动窗口限流：窗口内超过上限返回剩余等待秒数（429 Retry-After），否则 None。

    T2.8/P2-4：限流时携带 Retry-After（RFC 7231），让客户端知道何时可重试。
    """
    global _feed_rate
    now = time.time()
    async with _feed_rate_lock:
        if now - _feed_rate["window_start"] >= _FEED_RATE_WINDOW:
            _feed_rate["window_start"] = now
            _feed_rate["count"] = 0
        _feed_rate["count"] += 1
        if _feed_rate["count"] > _FEED_RATE_LIMIT:
            # 剩余等待 = 窗口结束时刻 - 当前时刻（至少 1s，供 Retry-After 用）
            return max(1.0, _FEED_RATE_WINDOW - (now - _feed_rate["window_start"]))
        return None


async def _feed_rate_limited() -> bool:
    """兼容包装：仅返回是否限流（bool）。"""
    return await _feed_rate_check() is not None


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
# T2.8/P2-4：request_id 中间件（幂等/链路追踪）
# ---------------------------------------------------------------------------
# 为每个请求生成 X-Request-ID（客户端已带则沿用，否则生成 uuid4 hex），
# 存 request.state.request_id 并在响应头回传。纯观测性增量：
# 不改任何业务逻辑，不依赖开关（新增响应头不影响既有行为）。
@app.middleware("http")
async def _request_id_middleware(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
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
# P0-7 止血：快照路径钳制（/snapshot 与 /restore 共用）
# ---------------------------------------------------------------------------
# 快照文件名白名单：安全字符集 + .pt 后缀。覆盖新式 {session}_{YYYYMMDD_HHMMSS}.pt
# 与存量 latest.pt / snapshot_{n}.pt / snap_{session}_{ts}.pt 等命名规则文件；
# 拒绝分隔符、..、绝对路径及任何目录外路径（详见 _clamp_snapshot_path）。
_SNAPSHOT_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.pt$")
# T1.1/P0-5：新命名规范允许一层会话子目录
# （snapshots/{session}/snapshot_{session}_{turn}_{ts}.pt 等）；
# 两级组件均须命中安全字符集，拒绝更深层级。
_SNAPSHOT_SUBDIR_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*\.pt$")


def _clamp_snapshot_path(raw_path) -> Optional[str]:
    """P0-7 止血：把用户提供的快照路径钳制到 snapshots/ 目录内。

    规则：
      1. 拒绝空值/非字符串/绝对路径/含 .. 的路径（拒绝任何穿越）；
      2. 反斜杠统一归一为斜杠；
      3. 含斜杠 → 仅允许一层会话子目录（T1.1 新命名规范，命中
         _SNAPSHOT_SUBDIR_RE）；不含斜杠 → 平铺旧格式（命中
         _SNAPSHOT_FILENAME_RE）；其余一律拒绝；
      4. 最终路径 = snapshots/ + 规范化相对路径，绝对化后再次校验目录
         包含关系（防御纵深）。
    任何不满足 → 返回 None（调用方回 4xx），绝不触碰目录外文件。
    """
    if not raw_path or not isinstance(raw_path, str):
        return None
    path = raw_path.strip().replace("\\", "/")
    if path.startswith("/"):
        return None  # 绝对路径
    if ".." in path:
        return None  # 路径穿越
    if "/" in path:
        # T1.1：允许且仅允许一层会话子目录（snapshots/{session}/{file}.pt）
        if not _SNAPSHOT_SUBDIR_RE.match(path):
            return None
    else:
        # 平铺旧格式（存量 latest.pt / snapshot_{n}.pt / {sid}_{ts}.pt）
        if not _SNAPSHOT_FILENAME_RE.match(path):
            return None
    snapshot_dir = os.path.abspath("./snapshots")
    full = os.path.abspath(os.path.join(snapshot_dir, path))
    if not full.startswith(snapshot_dir + os.sep):
        return None  # 防御纵深：必须仍落在 snapshots/ 内
    return full


def _sha256_of(path: Optional[str]) -> Optional[str]:
    """计算文件 sha256（T2.6 审计用：记录被覆盖的 latest 快照旧哈希）。

    文件不存在/读取失败时返回 None（fail-open，不阻塞主流程）。
    """
    if not path or not os.path.isfile(path):
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:  # pylint: disable=broad-except
        return None


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
                # T2.6：关键错误审计（LLM 调用失败，不中断服务）
                audit("critical_error", component="llm",
                      session_id=req.session_id,
                      error=str(e)[:200])
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

    # T2.8/P2-4：限流检查（一次性调用，返回剩余等待秒数或 None）
    retry_after = await _feed_rate_check()
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=f"feed 限流：超过 {_FEED_RATE_LIMIT} 次/分钟，请稍后重试",
            headers={"Retry-After": str(int(math.ceil(retry_after)))},
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
        free_energy=float(status.get("last_free_energy", 0.0) or 0.0),
        mse=float(status.get("last_mse", 0.0) or 0.0),
        turn_count=int(status.get("turn_count", 0) or 0),
        session_id=req.session_id,
    )


@app.post("/recall")
async def recall(req: RecallRequest):
    """只读情景检索端点（T1.3/P0-9；T2.3 起为内存+归档合并检索）。

    编码 query + 合并检索：内存 200 条窗口（活体优先）∪ 归档 JSONL
    （窗口外补充，条目带 origin 标记），**不 process_turn、不调 LLM、
    不写缓冲、不落盘**——纯读取路径，目标耗时 ~1s 内。

    T2.3 变更（2026-08-10）：原只查内存，现改为 memory.recall_merged_readonly
    （内存 tier0 优先、归档 tier1 补充；归档检索超时 500ms 则跳过归档
    只回内存，fail-open，保持 <2s 目标；LMS_ARCHIVE_ENABLED=0 可一键
    关闭合并回退到纯内存路径）。响应条目新增 origin 字段
    （'memory' | 'archive'），旧客户端无感（多一个字段）。
    （/chat 7-10s 的根因是耦合了 LLM 对话与检索，本端点剥离。）

    会话不存在时惰性创建空脑（与 /self-ref/voice 一致）；创建本身
    不产生任何记忆写入。
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="query 不能为空")
    k = max(1, min(int(req.k), 20))  # 钳制返回条数

    sm = get_session_manager()
    loop = sm.get_or_create(req.session_id)

    t0 = time.time()
    try:
        # T2.3：内存+归档合并检索（内存优先；归档超时/异常内部 fail-open）
        results = await asyncio.get_event_loop().run_in_executor(
            None, lambda: loop.recall_merged_readonly(req.query, k=k))
    except Exception as e:
        # 只读路径异常：fail-open，返回空结果，绝不 500 拖垮调用方
        logger.error(f"[{req.session_id}] /recall 检索失败（返回空）: {e}")
        results = []
    duration_ms = round((time.time() - t0) * 1000, 1)

    return {
        "session_id": req.session_id,
        "query": req.query,
        "k": k,
        "count": len(results),
        "results": results,
        "duration_ms": duration_ms,
        # 只读校验锚点：检索不应改变轮次（测试与可观测性用）
        "turn_count": loop.turn_count,
    }


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
    """保存指定 session 的快照（T1.1/P0-5 新命名规范）。

    P0-7 止血延续：忽略用户传入的任意 path，路径完全由服务端生成。
    新命名：snapshots/{session}/snapshot_{session}_{turn}_{ts}.pt，
    并同步 snapshots/{session}/latest_{session}.pt（最新指针）。
    session_id 参与路径前清洗为安全字符集，杜绝任何路径穿越。
    """
    sm = get_session_manager()
    loop = sm.get(session_id)
    if loop is None:
        raise HTTPException(
            status_code=404,
            detail=f"会话 '{session_id}' 不存在",
        )

    try:
        # T2.6：保存前记录将被覆盖的 latest_{session}.pt 旧 sha256（审计对账用）
        latest_path = loop.latest_snapshot_path()
        old_sha256 = _sha256_of(latest_path)
        path = loop.save_session_state()
        if path is None:
            # 写锁超时被跳过（fail-open 策略的显式反馈，非静默）
            raise HTTPException(
                status_code=503,
                detail="快照写锁超时，保存被跳过，请稍后重试",
            )
        logger.info(f"[{session_id}] 快照已保存: {path}")
        # T2.6：快照保存审计（含被覆盖的 latest 旧 sha256）
        audit("snapshot_saved", session_id=session_id, path=path,
              latest_path=latest_path, old_latest_sha256=old_sha256,
              turn_count=loop.turn_count)
        return {
            "session_id": session_id,
            "saved": True,
            "path": path,
            "latest_path": loop.latest_snapshot_path(),
            "turn_count": loop.turn_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{session_id}] 快照保存失败: {e}")
        # T2.6：关键错误审计（快照保存失败）
        audit("critical_error", component="snapshot",
              session_id=session_id, error=str(e)[:200])
        raise HTTPException(
            status_code=500,
            detail=f"快照保存失败: {e}",
        )


@app.post("/restore/{session_id}")
async def restore(session_id: str, req: RestoreRequest):
    """从快照恢复指定 session 的状态。

    P0-7 止血：只允许 snapshots/ 内符合命名规则的 .pt 文件；
    任何 ../、绝对路径、目录外路径一律 4xx 拒绝（防任意文件 torch.load）。
    """
    if not req.path:
        raise HTTPException(status_code=400, detail="path 不能为空")

    # P0-7：路径钳制（非法/目录外路径直接拒绝）
    full_path = _clamp_snapshot_path(req.path)
    if full_path is None:
        raise HTTPException(
            status_code=400,
            detail=f"非法快照路径: {req.path!r}（仅允许 snapshots/ 内的 .pt 快照文件）",
        )

    sm = get_session_manager()
    loop = sm.get(session_id)
    if loop is None:
        raise HTTPException(
            status_code=404,
            detail=f"会话 '{session_id}' 不存在",
        )

    if not os.path.isfile(full_path):
        raise HTTPException(
            status_code=404,
            detail=f"快照文件不存在: {full_path}",
        )

    try:
        loop.load_state(full_path)
        logger.info(f"[{session_id}] 已从 {full_path} 恢复状态")
        # T2.6：加载/回退审计（手动 /restore）
        audit("state_restored", session_id=session_id, path=full_path,
              turn_count=loop.turn_count)
        return {
            "session_id": session_id,
            "restored": True,
            "path": full_path,
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


@app.get("/self-ref/voice")
async def self_ref_voice(session_id: str = "main", limit: int = 5):
    """返回指定 session 的自指回路最近自述（反思产物，供上层注入上下文）。

    设计（反思回流，2026-08-05）：
      - 数据源：SelfReferentialLoop.self_voice_history（内存，最近优先）；
      - 会话不存在时惰性创建（与 /chat 一致）：重启后首次读取也能工作；
      - 内存优先：读取前先 backfill_voice_history() 从持久层合并缺失条目
        （纯文本回填，绝不回注嵌入 → 无循环）；
      - 未启用/无历史 → 返回空列表，绝不报错（fail-open）；
      - limit 钳制到 [1, 20]。
    """
    sm = get_session_manager()
    loop = sm.get_or_create(session_id)
    self_ref = getattr(loop, "self_ref", None)
    if self_ref is None:
        return {"session_id": session_id, "enabled": False, "count": 0,
                "voices": []}
    # 持久层回填（内存优先：仅合并内存中缺失的条目；
    # 重启后内存为空时自动恢复历史）
    try:
        self_ref.backfill_voice_history()
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(f"[{session_id}] self_voice 回填失败（忽略）: {e}")
    history = list(getattr(self_ref, "self_voice_history", None) or [])
    limit = max(1, min(int(limit), 20))
    voices = history[-limit:][::-1]  # 最近在前
    return {
        "session_id": session_id,
        "enabled": True,
        "count": len(voices),
        "voices": voices,
        "persisted_count": int(getattr(
            self_ref, "persisted_voice_count", lambda: 0)()),
        "last_echo_similarity": getattr(self_ref, "last_echo_similarity", None),
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
    # T2.6：会话删除审计（session_manager.remove 内已 audit，此处补调度器侧信息）
    audit("session_deleted", session_id=session_id, scheduler_unregistered=True)
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
    # T2.6：服务启动审计
    audit("startup", pid=os.getpid(), version=app.version)
    # T1.4 补：启动时修剪快照（每会话只保留最近 N 个轮次快照，防目录无限增长）。
    # keep 由 LMS_SNAPSHOT_PRUNE_KEEP 环境变量覆盖，默认 20（见 snapshot.py）。
    try:
        from persistence.snapshot import prune_snapshots
        prune_keep = os.environ.get("LMS_SNAPSHOT_PRUNE_KEEP", "20")
        prune_stats = prune_snapshots(
            os.environ.get("LMS_SNAPSHOT_DIR", os.path.abspath("./snapshots")))
        total_removed = sum(v["removed"] for v in prune_stats.values())
        logger.info(
            f"启动快照修剪完成（keep={prune_keep}）: "
            f"扫描 {len(prune_stats)} 个会话目录，共删除 {total_removed} 个旧快照"
        )
    except Exception as e:
        logger.warning(f"启动快照修剪失败（不影响启动）: {e}")

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
        # T2.6：配置生效审计（含 T2.8 全部算法治理开关状态）
        audit("config_effective",
              nodes=cfg['num_nodes'], dim=cfg['input_dim'], llm_on=llm_on,
              norm_surprise=cfg.get('norm_surprise'),
              replay_surprise_cap=cfg.get('replay_surprise_cap'),
              norm_latent=cfg.get('norm_latent'),
              meta_lazy=cfg.get('meta_lazy'),
              self_ref_no_bus=cfg.get('self_ref_no_bus'),
              embed_circuit=cfg.get('embed_circuit'))
    except Exception as e:
        logger.warning(f"读取默认配置失败: {e}")

    # 启动做梦调度器后台线程
    scheduler = get_dream_scheduler()
    scheduler.start()
    # T2.6：调度器启动审计
    audit("scheduler_start",
          idle_threshold=getattr(scheduler, 'idle_threshold', None),
          dream_steps=getattr(scheduler, 'dream_steps', None))
    logger.info(
        f"DreamScheduler 已启动 "
        f"(idle_threshold={scheduler.idle_threshold}s, "
        f"dream_steps={scheduler.dream_steps})")


# ---------------------------------------------------------------------------
# T1.4/P0-13：优雅停机落盘
# ---------------------------------------------------------------------------
# SIGTERM/SIGINT 由 uvicorn 捕获并触发优雅停机（运行 on_shutdown）；
# 此处给出 30s 优雅窗口（LMS_SHUTDOWN_GRACE_SECONDS 可覆盖）：on_shutdown 内
# 启动看门狗线程，超时则 os._exit 强制退出，防止落盘卡死导致进程无限挂起。
SHUTDOWN_GRACE_SECONDS = float(os.environ.get("LMS_SHUTDOWN_GRACE_SECONDS", "30"))

# 停机看门狗取消事件（T1.4 修复：优雅停机成功后取消，防误杀同进程）
_SHUTDOWN_WATCHDOG = {"event": None}


def _start_shutdown_watchdog(grace_seconds: float = SHUTDOWN_GRACE_SECONDS) -> None:
    """启动停机看门狗：grace 秒后强制退出（兜底防挂死）。

    uvicorn 收到 SIGTERM/SIGINT 后进入优雅停机并触发 on_shutdown；
    若 on_shutdown 内落盘卡死，看门狗线程到点 os._exit 强制退出，
    保证进程不会无限挂起（T1.4/P0-13）。

    注意：看门狗必须在优雅停机**成功完成**后取消（_cancel_shutdown_watchdog），
    否则守护线程会在宽限期满后无条件 os._exit(1)——生产环境 uvicorn 随即退出
    无害，但同进程内继续存活的场景（如 pytest 全量套件的 TestClient）会被误杀。
    """
    cancel_event = threading.Event()

    def _force_exit() -> None:
        # Event.wait(timeout) 返回 True = 已收到取消信号 → 空转退出；
        # 返回 False = 宽限期满仍未取消 → 强制退出。
        if cancel_event.wait(grace_seconds):
            return
        logger.critical(
            f"优雅停机超时（>{grace_seconds}s），强制退出（可能未完成全部落盘）")
        os._exit(1)

    _SHUTDOWN_WATCHDOG["event"] = cancel_event
    threading.Thread(
        target=_force_exit, daemon=True, name="shutdown-watchdog").start()


def _cancel_shutdown_watchdog() -> None:
    """取消停机看门狗（优雅停机在宽限期内完成时调用）。

    设置取消 Event 后，看门狗线程的 wait 立即返回 True 并空转退出，
    不再触发 os._exit。
    """
    ev = _SHUTDOWN_WATCHDOG.get("event")
    if ev is not None:
        ev.set()
        _SHUTDOWN_WATCHDOG["event"] = None


@app.on_event("shutdown")
async def on_shutdown():
    """优雅停机（T1.4/P0-13）：先落盘再清理。

    顺序（比任务原文"先 save 再停调度器"更安全的一个偏差，理由见下）：
      1. 停止做梦调度器（join 等待，≤5s），杜绝落盘与后台做梦并发写同一脑；
      2. 对每个活跃会话 save_session_state()（新命名规范最终快照）；
      3. clear 会话。
    若做梦进行中直接落盘，会读到半巩固状态；先停调度器再落盘可保证
    快照一致性，且同样满足"停机前保存最终快照"的目标。

    T1.4 修复（2026-08-10）：优雅停机在宽限期内成功完成后，finally 中
    取消看门狗——守护线程若不被取消，会在 30s 后无条件 os._exit(1)，
    误杀同进程内继续存活的场景（如 pytest 全量套件中的 TestClient）。
    """
    logger.info("开始优雅停机：停止调度器 → 保存最终快照 → 清理会话")
    _start_shutdown_watchdog()  # 30s 兜底强退
    try:
        scheduler = get_dream_scheduler()
        scheduler.stop()
        # T2.6：调度器停止审计
        audit("scheduler_stop")
        logger.info("DreamScheduler 已停止")

        sm = get_session_manager()
        saved_paths = []
        for sid in sm.list_sessions():
            loop = sm.get(sid)
            if loop is None:
                continue
            try:
                path = loop.save_session_state()
                if path:
                    saved_paths.append(path)
                    logger.info(f"[{sid}] 停机最终快照已保存: {path}")
                else:
                    logger.warning(f"[{sid}] 停机最终快照因写锁超时被跳过")
            except Exception as e:
                logger.error(f"[{sid}] 停机最终快照保存失败: {e}")
        logger.info(f"优雅停机：已保存 {len(saved_paths)} 个会话的最终快照")

        n = sm.clear()
        logger.info(f"服务关闭，已清理 {n} 个会话")
        # T2.6：服务停机审计
        audit("shutdown", saved_count=len(saved_paths), sessions_cleared=n)
    finally:
        _cancel_shutdown_watchdog()


# 模块导入时确保项目根目录在 sys.path 中（支持 `python api/run.py` 启动）
import sys  # noqa: E402
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

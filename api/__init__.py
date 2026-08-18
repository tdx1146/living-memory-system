"""活体记忆系统 - FastAPI 服务层

为 TRAE/OpenClaw UI 提供 HTTP 后台服务，使对话时后台自动生成记忆并适时注入。

模块:
    config          -- API 配置（环境变量、预训练模型路径、默认 LLM 配置）
    session_manager -- 会话管理器（维护多 session 的 LivingMemoryLoop）
    server          -- FastAPI 应用与端点定义（数据面 :8190，唯一快照写者）
    control         -- FastAPI 应用与端点定义（管理面 :8191，独立 app）
    run             -- 启动脚本（uvicorn，数据面）
    scripts/run_control.py -- 启动脚本（uvicorn，管理面）

【阶段 2 变更（2026-08-10，.bak-phase2 可回滚）】
    原实现在此处急切导入 api.config / api.session_manager —— 二者会拉入
    runtime.loop → torch 全链路（~230MB RSS、高 CPU）。导致 `uvicorn
    api.control:app`（管理面 :8191）导入即被重型依赖拖垮。
    改为 PEP 562 惰性 __getattr__：
      * `from api.control import app` 只加载 stdlib + fastapi（轻量）；
      * `from api import get_api_config / SessionManager` 行为不变
        （首次属性访问时才加载重型模块，且只发生在数据面场景）。
"""

__all__ = ["get_api_config", "SessionManager"]


def __getattr__(name):
    """PEP 562 惰性属性：保持 `from api import get_api_config` 兼容，
    同时避免 api 包被导入时急切加载 torch 链路（管理面轻量化的关键）。"""
    if name == "get_api_config":
        from api.config import get_api_config
        return get_api_config
    if name == "SessionManager":
        from api.session_manager import SessionManager
        return SessionManager
    raise AttributeError(f"module 'api' has no attribute {name!r}")

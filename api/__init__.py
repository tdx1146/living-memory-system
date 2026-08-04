"""活体记忆系统 - FastAPI 服务层

为 TRAE/OpenClaw UI 提供 HTTP 后台服务，使对话时后台自动生成记忆并适时注入。

模块:
    config          -- API 配置（环境变量、预训练模型路径、默认 LLM 配置）
    session_manager -- 会话管理器（维护多 session 的 LivingMemoryLoop）
    server          -- FastAPI 应用与端点定义
    run             -- 启动脚本（uvicorn）
"""

from api.config import get_api_config
from api.session_manager import SessionManager

__all__ = ["get_api_config", "SessionManager"]

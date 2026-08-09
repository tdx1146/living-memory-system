# ============================================================================
# 活体记忆系统 (Living Memory System) - Dockerfile
# 对齐 2026-08-10 稳定化架构：数据面 :8190 + 管理面 :8191
# CPU-only 镜像，多阶段构建以减小体积
# 构建: docker build -t living-memory-system:latest .
# 运行: docker run -p 8190:8190 --env-file .env \
#         -v $(pwd)/snapshots:/app/snapshots -v $(pwd)/data:/app/data \
#         -v $(pwd)/logs:/app/logs living-memory-system:latest
#   （管理面: 另起容器 docker run -p 127.0.0.1:8191:8191 \
#         --env-file .env -e LMS_CTRL_HOST=0.0.0.0 \
#         living-memory-system:latest python scripts/run_control.py）
# ============================================================================

# ---------- Stage 1: 构建阶段（安装 Python 依赖） ----------
FROM python:3.11-slim AS builder

# 编译依赖（仅构建阶段需要，不进入最终镜像，从而减小运行镜像体积）
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# 创建独立虚拟环境，便于整目录拷贝到运行阶段
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 升级 pip 并安装打包工具
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# 1) 先单独安装 CPU 版 torch
#    指定 PyTorch 官方 CPU 索引，避免从 PyPI 拉取含 CUDA 的大体积 wheel（约 2GB -> 190MB）
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2) 安装其余 Python 依赖（torch 已满足 requirements 中的 torch>=2.0，不会重复下载）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ---------- Stage 2: 运行阶段 ----------
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="living-memory-system" \
      org.opencontainers.image.description="活体记忆系统 - 外挂于主LLM的海马体记忆层 (CPU-only)" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.license="MIT"

# 运行时系统依赖:
#   git  -- huggingface / sentence-transformers 模型拉取可能需要
#   curl -- HEALTHCHECK 健康检查（compose 与 Dockerfile 内均使用）
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 拷贝构建阶段装好的虚拟环境（含 torch / sentence-transformers / fastapi 等）
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# 复制项目源码（.dockerignore 已排除 tests/docs/.git/snapshots/data/logs/.env 等）
COPY . .

# 以可编辑模式安装本包
# 依赖已在 venv 中，--no-deps 仅注册包路径与 console_scripts 入口，无需重新解析/下载依赖
RUN pip install -e . --no-deps

# 运行时目录（由 docker-compose 挂载卷持久化；data/ 含 self_voice、archive、control）
RUN mkdir -p /app/snapshots /app/data /app/logs

# ----------------------------------------------------------------------------
# 环境变量默认值（参考 .env.example / api/config.py / runtime/config.py）
# ----------------------------------------------------------------------------

# 服务监听（容器内必须为 0.0.0.0 才能接受宿主机/外部访问）
ENV LMS_API_HOST=0.0.0.0
ENV LMS_API_PORT=8190

# 管理面（control.py，独立容器运行 scripts/run_control.py 时生效）
# 默认仅本机 127.0.0.1:8191；容器内须置 0.0.0.0 才能被端口映射暴露
ENV LMS_CTRL_HOST=0.0.0.0
ENV LMS_CTRL_PORT=8191
ENV LMS_CTRL_API_BASE=http://127.0.0.1:8190

# 嵌入器: Docker 默认 simple（纯 CPU、离线、秒级启动）
# 生产推荐 cloud（远端 Ollama bge-m3，见 .env.docker.example 的
# LMS_CLOUD_EMBED_URL：局域网 http://192.168.0.103:11435/v1/embeddings，
# 外网隧道 https://11435.tdx1146.cc/v1/embeddings —— 容器网络用隧道域名更稳）
ENV LMS_EMBEDDER=simple
ENV LMS_INPUT_DIM=64
ENV LMS_NUM_NODES=256

# 预训练模型路径（仅 LMS_EMBEDDER=pretrained 时生效，留空则使用代码内默认）
ENV LMS_PRETRAINED_MODEL=

# LLM 配置（默认 DeepSeek 兼容接口；API Key 留空则禁用 LLM，/chat 仅返回记忆 context）
ENV LMS_LLM_BASE_URL=https://api.deepseek.com/v1
ENV LMS_LLM_MODEL=deepseek-chat
ENV DEEPSEEK_API_KEY=
ENV LMS_LLM_API_KEY=

# 做梦调度器（DreamScheduler）
ENV DREAM_IDLE_THRESHOLD=30
ENV DREAM_STEPS=20
ENV DREAM_FULL_CYCLE=false
ENV DREAM_CHECK_INTERVAL=5

# Python 运行时
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 暴露 API 端口（数据面 8190 + 管理面 8191）
EXPOSE 8190 8191

# 健康检查：访问 /health 端点（管理面为 /control/health）
# start-period 60s 给 torch / 模型初始化留出预热时间
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8190/health || exit 1

# 启动 FastAPI 服务（数据面）
# run.py 会读取 LMS_API_HOST/LMS_API_PORT，并在 startup 事件中启动 DreamScheduler
# 管理面（同镜像）请用: python scripts/run_control.py
CMD ["python", "-m", "api.run"]

# LMS 容器化部署指南（Docker）

> 对齐 2026-08-10 稳定化架构：**数据面 :8190 + 管理面 :8191 + 配置中心 .env**
> 适用仓库：`living-memory-system-cloud/docker-compose.yml`（本文件）
> 相关：`Dockerfile`、`.env.docker.example`、`scripts/lms_backup.sh`、`scripts/lms_ctl.sh`

---

## 0. 架构速览（容器内）

```
┌─ 宿主机 ─────────────────────────────────────────────┐
│  MCP 薄桥 (mcp_memory_server.py)  ── 走 HTTP ──┐      │
│  stack_ctl / lms_ctl.sh（进程托管，非容器）        │      │
│  lms_backup.sh（备份，cron 调度）                  │      │
│  ┌─ docker compose（bridge 网络）────────────────┐  │      │
│  │  lms          :8190  数据面 FastAPI + 做梦调度  │  │      │
│  │  lms-control  :8191  管理面（control.py）       │  │      │
│  │    └─ 容器内互联 http://lms:8190                │  │      │
│  └───────────────────────────────────────────────┘  │      │
└─────────────────────────────────────────────────────┘      │
        ▲ 嵌入/LLM（外网隧道或 LAN）                          │
        └─ https://embed.example.com/v1/embeddings（bge-m3）  │
           （LAN: http://<LAN_IP>:11435/v1/embeddings） ┘
```

设计取舍（为什么是「容器 = 引擎，宿主机 = 控制面」）：

| 组件 | 部署位置 | 理由 |
|---|---|---|
| 数据面 api/server.py（:8190） | 容器 `lms` | 引擎本体，可移植、可重建 |
| 管理面 control.py（:8191） | 容器 `lms-control`（同镜像） | 独立 FastAPI app，与数据面零 import 冲突 |
| MCP 薄桥 | 宿主机 | 需要直连 OpenClaw MCP stdio 协议，容器化收益低 |
| stack_ctl / lms_ctl.sh | 宿主机 | 进程托管需要系统 PID/端口探活，容器内由 restart 策略替代 |
| lms_backup.sh | 宿主机（cron） | 备份目标在宿主机，且需 flock/rsync 系统能力 |
| 配置中心 .env | 宿主机（env_file 注入） | 密钥不进镜像；容器内同文件路径 /app |

---

## 1. 前置条件

- Docker Engine ≥ 24（含 compose v2）：`docker --version && docker compose version`
- 网络要求（embedder，手机 Ollama bge-m3）：
  - **外网/容器网络（推荐）**：`https://embed.example.com/v1/embeddings`（隧道 443，已验证 200）
    - ⚠️ 实测 `http://embed.example.com:11435`（域名+端口形式）**不通**，隧道只放行 443
  - **局域网直连**：`http://<LAN_IP>:11435/v1/embeddings`（与手机同网络时更快，可作 fallback）
  - LLM：DeepSeek `https://api.deepseek.com/v1`（或任意 OpenAI 兼容服务）

## 2. 首次部署

```bash
cd <LMS_ROOT>

# 1) 配置
cp .env.docker.example .env
$EDITOR .env          # 填 DEEPSEEK_API_KEY；生成并填 LMS_CONTROL_TOKEN:
                      #   openssl rand -hex 32
                      # 嵌入地址按网络环境选：
                      #   LMS_CLOUD_EMBED_URL=https://embed.example.com/v1/embeddings
                      #   LMS_CLOUD_EMBED_FALLBACK_URL=http://<LAN_IP>:11435/v1/embeddings

# 2) 构建并启动（首次构建约 5-15 分钟：torch CPU ~190MB + 依赖）
docker compose up -d --build

# 3) 验证
docker compose ps                       # 两服务均 healthy
curl http://127.0.0.1:8190/health       # 数据面 {"status":"ok",...}
curl http://127.0.0.1:8191/control/health  # 管理面（聚合数据面状态）
docker compose logs -f lms              # 查看启动日志

# 4) 日常运维
docker compose logs -f lms lms-control  # 日志
docker compose restart lms              # 重启数据面
docker compose down                     # 停止（数据已持久化在卷中）
```

## 3. 数据卷与备份恢复

容器持久化目录（compose 已挂载）：

| 宿主目录 | 容器路径 | 内容 |
|---|---|---|
| `./snapshots/` | `/app/snapshots` | 快照（唯一写者 = 数据面） |
| `./data/` | `/app/data` | `self_voice/`、`archive/`、`control/`（access.jsonl） |
| `./logs/` | `/app/logs` | 审计日志 audit-*.jsonl、control-audit.jsonl、alerts |
| `./lms_cache/`（可选） | `/root/.cache` | 模型缓存 |

> ⚠️ 容器内 `LMS_SNAPSHOT_DIR` 已被 compose 强制为 `/app/snapshots`，
> 不要用宿主机 .env 里的绝对路径（否则快照写进容器可写层，重建即丢）。

### 备份（宿主机，配合 lms_backup.sh）

容器与裸进程共用同一批数据目录，备份脚本**无需改动**：

```bash
cd <LMS_ROOT>
./scripts/lms_backup.sh inc     # 15 分钟级快照镜像（cron */15）
./scripts/lms_backup.sh daily   # 每日 02:30 全量 tar.zst（snapshots+data+logs）
./scripts/lms_backup.sh status  # 查看备份状态
```

### 恢复（三步）

```bash
docker compose down                      # 1. 停容器（避免写者竞争）
# 2. 恢复数据（示例：解每日全量；详见 docs/lms-restore-sop.md）
tar --zstd -xf <LMS_BACKUP_ROOT>/daily/lms-YYYYMMDD.tar.zst -C <LMS_ROOT>
docker compose up -d                     # 3. 起容器
```

## 4. 与宿主机裸进程部署的取舍

| 维度 | 容器化 | 裸进程（现状） |
|---|---|---|
| 引擎隔离/可移植 | ✅ 镜像即环境，换机秒级迁移 | ❌ 依赖本机 venv/python |
| 进程托管 | ✅ restart: unless-stopped 自愈 | systemd user 单元（lms-api/lms-control.service） |
| 管理面 | 独立容器 lms-control（127.0.0.1:8191） | 同机 `python scripts/run_control.py` |
| 备份/监控 | 卷挂载 + 宿主机 lms_backup.sh | 同左（目录不变） |
| MCP 薄桥 | 宿主机（不改） | 宿主机（现状） |
| 启动开销 | 冷启动多一层容器（可忽略） | 直接进程 |
| 适用场景 | 换机/多机/干净环境复现 | 当前生产（已稳定运行） |

**切换建议**：裸进程运行中时不要同时起容器（同端口/同数据目录写者冲突）。
切换流程：`lms_ctl.sh stop`（或 systemctl --user stop lms-api lms-control）
→ `docker compose up -d` → 观察 `docker compose ps` 与 `/health`。

## 5. 常见问题

- **`curl: (7) Failed to connect`**：容器内 healthcheck 用 curl（镜像已装），
  先查 `docker compose logs lms`；若 embed 相关报错，确认 11435 隧道/LAN 可达
  （`curl -X POST https://embed.example.com/v1/embeddings -d '{"model":"bge-m3","input":["hi"]}'`）。
- **管理面写端点 503**：`.env` 缺 `LMS_CONTROL_TOKEN` → 只读模式；
  补 token 后 `docker compose up -d lms-control` 重建容器。
- **快照不落盘**：检查 `docker compose exec lms printenv LMS_SNAPSHOT_DIR`，
  应为 `/app/snapshots`（不是宿主机路径）。
- **build 拉 torch 慢/失败**：Dockerfile 用官方 CPU 索引
  `https://download.pytorch.org/whl/cpu`；网络受限时先 `docker pull python:3.11-slim`
  并配置 Docker 镜像加速。

## 6. 参考

- 端口：数据面 8190 / 管理面 8191（与裸进程一致，可平滑互切）
- 环境变量全量说明：`.env.example`（算法开关 7 项 + 备份 + 审计）
- 备份恢复 SOP：`docs/lms-restore-sop.md`
- Agent OS 全系统容器化：`<AGENTOS_DOCKER>/README.md`

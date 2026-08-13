# 活体记忆系统（Living Memory System, LMS）

> 一个基于海马体模型的"活体"记忆痕迹维护器：通过自由能原理（FEP）学习规则，
> 让记忆在与对话的持续交互中**涌现**、**巩固**与**演化**，而不是静态的键值存取。

> 🔗 **系统定位**（2026-08-11）：本模块是「LMS 活体记忆」+ 体验层（实时反应/怀疑融入）。
> 上游依赖：无（被胶水层/self_pulse/OpenClaw 调用）｜ 下游消费者：胶水层(:19000)、self_pulse、OpenClaw 插件
> 外部接口：`:8190` /health、`/status/{sid}`、`/feed`、`/recall`、`/chat`、**`/react`（体验层A：infer-only 实时反应+解读段）**；`:8191` 控制口
> 仓库：`https://github.com/tdx1146/living-memory-system`（**main，公开**）
> 系统全图：**见 `tdx1146/agent-os` 仓库的 `TOPOLOGY.md`**（https://github.com/tdx1146/agent-os/blob/main/TOPOLOGY.md）｜ **部署中心/数据流/踩坑：`SYSTEM.md`**（https://github.com/tdx1146/agent-os/blob/main/SYSTEM.md）

## 目录

- [核心理念](#核心理念)
- [系统架构概览](#系统架构概览)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 端点文档](#api-端点文档)
- [MCP 工具说明](#mcp-工具说明)
- [测试](#测试)
- [项目结构](#项目结构)
- [技术栈](#技术栈)

---

## 核心理念

传统记忆系统是"冷工具"——写入即冻结，检索即复读。活体记忆系统试图回答一个问题：
**记忆能否像生物海马体一样，在无人说话时也在自我巩固、遗忘与重组？**

LMS 把记忆建模为一组相互竞争的**吸引子**（attractor），其连接矩阵 `J` 在每轮对话中
按自由能原理（FEP）持续更新。核心机制有三层：

1. **记忆痕迹的涌现** —— 每轮对话先经感官层编码为向量，吸引子网络通过 FEP 推断收敛到
   一个激活态，再以惊讶度（预测误差）驱动的学习规则更新 `J`。记忆不是被显式存储的，
   而是从网络动力学中涌现的。
2. **目的的自演化** —— 目的层（PurposeLayer）维护一组 `precision` 向量，描述系统当前
   "关注"哪些感官维度。`precision` 不是预设的，而是从自身活动中涌现，并在 `coherence`
   降至阈值时触发元目的翻转，使系统的"兴趣"持续漂移，避免僵化。
3. **睡眠中的活体巩固** —— 当对话空闲时，常驻的 `DreamScheduler` 后台线程自动触发
   做梦引擎，进行记忆回放、SHY 衰减、吸引子景观漂移与目的演化。这让系统即使无人说话，
   也在持续"做梦"，从冷工具变为**活体**。
4. **体验层（2026-08-11，v1.1 设计）** —— 让主 AI 每轮"看见大脑此刻的感受"，且这份感受是干净的、方向明确的、被质检过的：
   - **实时反应找回**：新端点 `POST /react`（infer-only 零持久化：不 learn/不落库/不写 sigma/turn_count 不变），
     输出 `reaction`（熵/惊讶/coherence 后验读出）+ `interpretation`（自然语言解读段，进 [回魂] 三段式）；
   - **子代理污染过滤**：`_GARBAGE_TEXT_RE` 纯增量 +6 条调度样板正则（`[Subagent Context]`/`[Subagent Task]`/`HEARTBEAT_OK` 等），
     main 脑不被调度噪音挤占 top-k；
   - **元目的翻转回初版**：coherence 低时**强化已关注维度**（`_meta_adjust` 改回 `avg_precision.argmax()`，越关注越专注）；
   - **怀疑融入（置信度场）**：记忆条目带 `confidence/rebuttal_count/labile/source_trust`，被证伪降权、做梦 `doubt_review` 阶段复核、
     反流畅项抑制虚假强化、唤起 salience 低置信配额相关性门控——**怀疑修正"信多少"，不改变"关注哪"**（confidence≠precision）。

简言之：**LMS 不存储记忆，而是维护一个能产生记忆的大脑状态。**

## 系统架构概览

系统采用四层架构，数据自底向上流转，核心层不依赖任何外部 IO。

```
┌──────────────────────────────────────────────────────────────┐
│  API / MCP 层   FastAPI(HTTP) + MCP(stdio) 暴露能力给外部     │
│                 api/server.py · mcp_memory_server.py          │
├──────────────────────────────────────────────────────────────┤
│  运行时层       LivingMemoryLoop 主循环 + DreamScheduler      │
│                 runtime/loop.py · runtime/dream_scheduler.py  │
├──────────────────────────────────────────────────────────────┤
│  桥接层         编码(对话→向量) / 解码(激活态→context) / LLM   │
│                 bridge/encoder.py · decoder.py · llm_bridge.py│
├──────────────────────────────────────────────────────────────┤
│  核心层         海马体模型（无外部依赖，纯 PyTorch 张量计算）   │
│                 core/hippocampus/* · core/sensory/* · core/meta│
└──────────────────────────────────────────────────────────────┘
```

### 主循环（每轮对话的记忆处理步骤）

`LivingMemoryLoop.process_turn(user_input, llm_output)` 对每轮对话执行以下步骤：

1. **编码输入** —— 用户输入 + LLM 输出 → 感官向量
2. **FEP 推断** —— 感官向量 → 激活态（含熵 entropy、惊讶度 surprise）
3. **FEP 学习** —— 更新连接矩阵 `J`
4. **在线熵管理** —— 熵过高则增强正交化，熵过低则放松正交化
5. **调整目的** —— 更新 `precision`，必要时触发元目的翻转
6. **记忆更新与巩固** —— 短时记忆更新 + 定期短时→长时迁移
7. **记忆检索** —— 用激活态检索长时记忆（关键修复：记忆不再"只写不读"）
8. **解码 context** —— 激活态 + 检索到的记忆 → LLM 可理解的 context
9. **自动快照** —— 可选，按间隔保存状态
10. **返回 context**

### 各层职责

| 层 | 目录 | 职责 |
|----|------|------|
| 核心层 | `core/` | 海马体吸引子网络、目的层、记忆管理器、元可塑性、感官嵌入器。纯张量计算，无 IO |
| 桥接层 | `bridge/` | 文本↔向量转换、LLM API 调用、指标转自然语言解释 |
| 运行时层 | `runtime/` | 主循环编排、做梦调度器、运行时配置 |
| API/MCP 层 | `api/`、`mcp_memory_server.py` | HTTP 端点（FastAPI）与 MCP 工具（stdio） |

## 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+（CPU 版即可运行）
- 可选：GPU（当前为 CPU 推理，GPU 管理在 P2 规划中）

### 1. 安装依赖

```bash
git clone <repo-url>
cd 活体记忆系统
# 创建虚拟环境（全部脚本都依赖 .venv，必须建）
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DeepSeek API Key（不填则 LLM 功能禁用，仅返回记忆 context）
```

⚠️ **关键**：`.env` 不会自动加载，启动前必须 source（否则得到静默降级实例：
embed 变 simple、LLM 不启用、/health 却显示正常——这是最常见的"部署了但没生效"坑）：

```bash
set -a; . ./.env; set +a
```

`.env` 关键项：

```ini
DEEPSEEK_API_KEY=sk-REPLACE_ME
# LMS_LLM_BASE_URL=https://api.deepseek.com/v1
# LMS_LLM_MODEL=deepseek-chat

# 嵌入器配置（三选一）
LMS_EMBEDDER=cloud
# cloud 模式（推荐，需远程 embed 服务）
LMS_CLOUD_EMBED_URL=https://embed.example.com/v1/embeddings
# 备用 embed 端点（主 URL 失败自动切换；本机 LAN→隧道 https://11435.tdx1146.cc/v1/embeddings）
# LMS_CLOUD_EMBED_FALLBACK_URL=
LMS_CLOUD_EMBED_MODEL=bge-m3
LMS_CLOUD_EMBED_DIM=1024
# pretrained 模式（本地加载，无需远程服务）
# LMS_EMBEDDER=pretrained
```

### 3a. 启动 HTTP API 服务

```bash
set -a; . ./.env; set +a   # 先加载配置（必做！）
python -m api.run
# 或: python api/run.py --host 0.0.0.0 --port 8190 --reload
```

服务默认监听 `http://127.0.0.1:8190`，交互式文档位于 `http://127.0.0.1:8190/docs`。
验证是否真的加载了配置：`curl http://127.0.0.1:8190/status/main` 应返回非空 turn_count，
且启动日志不应出现 "降级" 字样。

验证体验层（2026-08-11 新功能）是否生效：

```bash
curl -X POST http://127.0.0.1:8190/react -H 'Content-Type: application/json' \
  -d '{"user_input":"你好","k":0}'
# → 200，含 reaction（熵/惊讶/coherence）+ interpretation（解读段）；连续两次 turn_count 不变（零持久化）
curl http://127.0.0.1:8190/status/main | grep -o '"doubt"'   # 体验层D：怀疑融入字段存在
```

### 3b. 启动 MCP 服务器（供 TRAE IDE 调用）

```bash
python mcp_memory_server.py
```

MCP 服务器通过 stdio 传输与客户端通信，日志输出到 stderr。

### 4. 快速验证（可执行示例）

以下 Python 脚本仅使用标准库，可直接运行（需先启动 API 服务），验证服务是否正常：

```python
import json
import urllib.request

BASE = "http://127.0.0.1:8190"


def _request(method, path, body=None):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


# 健康检查
print(_request("GET", "/health"))

# 完整对话（未配置 DEEPSEEK_API_KEY 时，response 即为记忆 context）
print(_request("POST", "/chat",
               {"user_input": "你好，我叫小明", "session_id": "demo"}))

# 查询会话状态
print(_request("GET", "/status/demo"))

# 列出所有会话
print(_request("GET", "/sessions"))
```

## 配置说明

LMS 的配置分两套体系，均支持环境变量覆盖：

### CoreConfig（核心层参数）

定义于 `core/config.py`，使用 dataclass 集中管理所有 FEP 参数。关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_nodes` | 256 | 吸引子网络节点数（建议 256–1024） |
| `input_dim` | 64 | 感官输入维度（= embedding 维度，前 `input_dim` 个节点为感官节点） |
| `learning_rate` | 0.01 | FEP 学习率 η |
| `num_infer_steps` | 10 | FEP 推断迭代步数 K |
| `temperature` | 0.05 | Langevin 动力学温度（扩散项噪声强度） |
| `short_term_decay` | 0.8 | 短时记忆衰减系数（越小遗忘越快） |
| `long_term_decay` | 0.999 | 长时记忆衰减系数（越接近 1 保留越久） |
| `precision_min` / `precision_max` | 0.1 / 10.0 | precision 上下限，防发散 |
| `precision_lr` | 0.1 | 目的层 precision 调整学习率 |
| `coherence_threshold` | 0.5 | coherence 低于此值触发元目的翻转 |
| `coherence_direction_weight` | 0.5 | coherence 方向项权重 |
| `coherence_magnitude_weight` | 0.5 | coherence 幅度项权重 |
| `complexity_weight` | 0.01 | 自由能复杂性项权重（L2 正则） |
| `orth_weight` | 0.5 | 正交化压力权重 |
| `entropy_high_threshold` | 0.9 | 在线熵管理：熵超此值增强正交化 |
| `entropy_low_threshold` | 0.5 | 在线熵管理：熵低于此值放松正交化 |
| `meta_enabled` | True | 是否启用元可塑性 |
| `meta_interval` | 10 | 元更新频率（每 N 轮） |
| `seed` | 42 | 随机种子 |
| `buffer_capacity` | 100 | 记忆缓冲区容量 |
| `activation_threshold` | 0.3 | 习惯化激活阈值 |

`CoreConfig` 提供三个辅助方法：

- `validate()` —— 校验全部参数约束，失败抛出 `ValueError`（含字段名与无效值）
- `to_loop_config()` —— 转换为 `LivingMemoryLoop` 需要的 config dict（44 个标量键）
- `from_env(**overrides)` —— 从 `LMS_` 前缀环境变量读取覆盖值，支持 int/float/bool 类型转换

```python
from core.config import CoreConfig

cfg = CoreConfig.from_env(num_nodes=512)  # 环境变量优先，显式 overrides 最高
cfg.validate()                             # 可选校验
loop_config = cfg.to_loop_config()         # 转 loop 配置
```

### 环境变量

环境变量统一以 `LMS_` 前缀命名（API 服务相关除外）。完整列表见 `api/config.py`：

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `DEEPSEEK_API_KEY` | （空） | DeepSeek API 密钥，提供则启用 LLM |
| `LMS_LLM_API_KEY` | （空） | LLM API 密钥（与 `DEEPSEEK_API_KEY` 等效回退） |
| `LMS_LLM_BASE_URL` | `https://api.deepseek.com/v1` | LLM API 基础 URL |
| `LMS_LLM_MODEL` | `deepseek-chat` | LLM 模型名 |
| `LMS_EMBEDDER` | `pretrained` | 嵌入器类型：`cloud` / `pretrained` / `simple` |
| `LMS_CLOUD_EMBED_URL` | `https://embed.example.com/v1/embeddings` | 云端 embed 服务 URL（`cloud` 模式生效） |
| `LMS_CLOUD_EMBED_FALLBACK_URL` | （空） | 备用 embed 端点（主 URL 失败自动切换，如 LAN→隧道） |
| `LMS_CLOUD_EMBED_MODEL` | `bge-m3` | 云端 embed 模型名（`cloud` 模式生效） |
| `LMS_CLOUD_EMBED_DIM` | `1024` | 云端 embed 输出维度（`cloud` 模式生效） |
| `LMS_INPUT_DIM` | 64 | 输入维度 |
| `LMS_NUM_NODES` | 256 | 节点数 |
| `LMS_PRETRAINED_MODEL` | 本地缓存路径 | 预训练模型本地路径（覆盖默认缓存） |
| `LMS_API_HOST` | `127.0.0.1` | 服务监听地址 |
| `LMS_API_PORT` | `8190` | 服务监听端口 |
| `DREAM_IDLE_THRESHOLD` | `30` | 做梦空闲触发阈值（秒） |
| `DREAM_STEPS` | `20` | 自动做梦步数 |
| `DREAM_FULL_CYCLE` | `false` | 是否使用完整七阶段做梦周期 |
| `DREAM_CHECK_INTERVAL` | `5` | 调度器检查间隔（秒） |
| `LMS_FEED_RATE_LIMIT` | `10` | /feed 限流（次/分钟，超限 429，防总线风暴） |
| `LMS_SNAPSHOT_DIR` | `./snapshots` | 快照目录（.pt 状态文件） |
| `LMS_*`（其余） | — | 任意 CoreConfig 字段大写加 `LMS_` 前缀，如 `LMS_TEMPERATURE` |

> **嵌入器三模式**：
> - `cloud`（推荐）：通过 HTTP API 调用远程 embedding 服务（如 bge-m3，1024维），语义质量最高。
>   默认指向 `https://embed.example.com/v1/embeddings`，可通过 `LMS_CLOUD_EMBED_*` 环境变量配置。
> - `pretrained`：本地加载 `paraphrase-multilingual-MiniLM-L12-v2`（384维），无需远程服务，加载失败降级为 `simple`。
> - `simple`：随机冻结 embedding，无语义先验，仅用于测试。
>
> 三种模式均自动将 `activation_threshold` 适配为 0.02（`simple` 除外）。
> `cloud` 和 `pretrained` 模式均支持 `embed_text_raw()` 原始维度检索，提升语义召回精度。

## API 端点文档

API 服务由 `api/server.py` 提供，共 10 个端点。简要列表如下，详细文档见
[docs/API.md](docs/API.md)。

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/react` | 实时反应只读（体验层A：infer-only 零持久化，返回 reaction+interpretation+可选 recalled；k=0 只要反应+解读） |
| POST | `/chat` | 完整对话（手动控制 LLM 注入时机） |
| POST | `/chat/simple` | 简化对话（自动处理记忆 + LLM 查询） |
| GET | `/status/{sid}` | 查询会话记忆状态 |
| POST | `/snapshot/{sid}` | 保存快照 |
| POST | `/restore/{sid}` | 从快照恢复 |
| GET | `/sessions` | 列出所有会话 |
| DELETE | `/sessions/{sid}` | 删除指定会话 |
| POST | `/dream/{sid}` | 手动触发做梦 |
| GET | `/dream/status` | 查询做梦调度器状态 |
| GET | `/health` | 健康检查 |

## MCP 工具说明

MCP 服务器（`mcp_memory_server.py`）通过 stdio 协议向 TRAE IDE 暴露核心能力，
使 AI 助手能够检索记忆、存储对话、查询状态并触发做梦。详细文档见
[docs/MCP_TOOLS.md](docs/MCP_TOOLS.md)。

| 工具 | 用途 |
|------|------|
| `recall_memory` | 语义检索与查询相关的历史对话记忆（top 3） |
| `store_memory` | 将当前对话存储到记忆系统（执行完整记忆循环并保存快照） |
| `get_memory_status` | 获取记忆系统运行状态（轮次、熵、surprise[准确性项≥0]、free_energy、precision 等） |
| `dream_memory` | 触发记忆系统"做梦"（空闲态记忆巩固与整合，可选完整七阶段周期） |

## 测试

项目使用 pytest，测试位于 `tests/` 目录，根目录 `conftest.py` 已将项目根加入 `sys.path`。

```bash
# 运行全部测试
pytest

# 运行指定模块测试
pytest tests/test_api_server.py

# 显示详细输出
pytest -v

# 生成覆盖率报告（需安装 pytest-cov）
pytest --cov=. --cov-report=term-missing
```

API 层测试（`tests/test_api_server.py`）使用轻量级 config（`num_nodes=32`）和 MockLLMBridge
隔离外部 API，覆盖全部 10 个 HTTP 端点的正常流程、错误处理、并发 503 拒绝与快照往返恢复。

CI 配置位于 `.github/workflows/`：`ci.yml`（Python 3.10/3.11/3.12 矩阵 + CPU 版 PyTorch）
与 `lint.yml`（ruff 风格检查，line-length=120）。

## 项目结构

```
活体记忆系统/
├── api/                        # API 层（FastAPI 服务）
│   ├── __init__.py
│   ├── config.py               # API 配置（环境变量读取）
│   ├── run.py                  # 服务启动脚本
│   ├── server.py               # FastAPI 端点定义（10 个端点）
│   └── session_manager.py      # 会话管理器
├── bridge/                     # 桥接层（文本↔向量转换 + LLM）
│   ├── __init__.py
│   ├── decoder.py              # 解码器（指标转自然语言解释）
│   ├── encoder.py              # 编码器（对话→感官向量）
│   └── llm_bridge.py           # LLM 桥接（OpenAI 兼容，含重试策略）
├── core/                       # 核心层（海马体模型，纯张量计算）
│   ├── __init__.py
│   ├── config.py               # CoreConfig 统一配置（含 validate/from_env）
│   ├── paths.py                # 跨平台路径管理（pathlib，缓存目录探测）
│   ├── types.py                # 核心数据类型（SensoryInput/Activation/PurposeState；体验层D：EpisodicEntry 置信度场字段）
│   ├── doubt/                  # 体验层D：怀疑融入（置信度场/去稳定化/召回调度/doubt_ingest/gap_registry）
│   │   ├── __init__.py
│   │   ├── confidence_field.py # 置信度场（confidence = 1×(1−rebuttal_rate)×source_trust）
│   │   ├── reconsolidation.py  # 惊讶度双角色：去稳定化（mark_labile）
│   │   ├── recall_scheduler.py # 唤起 salience（相关性门控配额）
│   │   ├── doubt_ingest.py     # /feed 结构化 doubt 摄入（fail-open）
│   │   └── gap_registry.py     # /status doubt.gaps 登记（C 类仅诊断）
│   ├── hippocampus/            # 海马体子模块
│   │   ├── __init__.py
│   │   ├── attractor.py        # 吸引子网络（FEP 推断与学习）
│   │   ├── dream_engine.py     # 做梦引擎（MVP + 完整七阶段）
│   │   ├── memory.py           # 记忆管理器（短时/长时 + 向量化检索）
│   │   └── purpose.py          # 目的层（precision 演化 + coherence）
│   ├── meta/                   # 元可塑性
│   │   ├── __init__.py
│   │   └── meta_plasticity.py
│   └── sensory/                # 感官层
│       ├── __init__.py
│       ├── embedder.py         # 嵌入器（Pretrained / Simple）
│       ├── cloud_embedder.py   # 云端嵌入器（CloudEmbedder，HTTP API 调用远程 embed 服务）
│       └── tokenizer.py        # 分词器
├── docs/                       # 文档
│   ├── API.md                  # HTTP API 详细文档
│   ├── MCP_TOOLS.md            # MCP 工具文档
│   ├── ARCHITECTURE.md         # 架构文档
│   ├── EXECUTION_TRACKER.md    # 执行跟踪
│   └── 体验层实施-20260811.md   # 体验层 A-D 实施记录（提交/验证/遗留/部署待办）
├── persistence/                # 持久化层
│   ├── __init__.py
│   ├── recovery.py             # 恢复管理（异常链 + 回滚）
│   └── snapshot.py             # 快照管理（原子写入）
├── runtime/                    # 运行时层
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py               # 运行时配置（default_config）
│   ├── dream_scheduler.py      # 做梦调度器（常驻后台线程）
│   └── loop.py                 # 主循环 LivingMemoryLoop
├── tests/                      # 测试套件
│   ├── test_api_server.py      # API 层测试（27 个，覆盖全部端点）
│   ├── test_config.py          # 配置测试（88 个）
│   ├── test_react_readonly.py  # 体验层A：/react 零持久化断言（J/sigma/episodic/turn_count 四不变）
│   ├── test_garbage_filter.py  # 体验层B：子代理样板过滤
│   ├── test_purpose_meta_flip.py # 体验层C：元目的翻转（强化已关注）
│   ├── test_doubt_integration.py # 体验层D：怀疑链（conflict→证伪→做梦→doubt_review）
│   └── ...                     # 其余模块测试
├── snapshots/                  # 快照目录（.pt 状态文件）
├── conftest.py                 # pytest 根配置（sys.path）
├── mcp_memory_server.py        # MCP 服务器（stdio 传输）
├── memory_cli.py               # CLI 工具
├── e2e_test.py                 # 端到端测试
├── requirements.txt            # 依赖清单
├── setup.py                    # 安装配置
└── .env.example                # 环境变量示例
```

## 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| 语言 | Python 3.10+ | 主要开发语言 |
| 深度学习 | PyTorch 2.0+ | 吸引子网络张量计算、FEP 推断与学习 |
| 数值计算 | NumPy | 辅助数值运算 |
| Web 框架 | FastAPI + Uvicorn | HTTP API 服务层 |
| 语义嵌入 | sentence-transformers | 感官层多语言语义嵌入（`paraphrase-multilingual-MiniLM-L12-v2`，384维） |
| 语义嵌入（云端） | bge-m3 via HTTP API | 远程 embedding 服务（1024维，通过 `CloudEmbedder` 调用） |
| LLM 接口 | OpenAI SDK | LLM 调用（OpenAI 兼容格式，默认 DeepSeek） |
| 协议层 | MCP (Model Context Protocol) 2.0 | 向 TRAE IDE 暴露记忆能力（stdio 传输） |
| 测试 | pytest | 单元测试与集成测试 |
| 代码质量 | ruff | 风格检查（line-length=120） |

---

*任务编号 E-P2-3 · 文档随系统演进持续更新。*

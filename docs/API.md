# 活体记忆系统 - HTTP API 文档

> 基于 FastAPI 实现的 HTTP 接口，使 TRAE/OpenClaw UI 在对话时后台自动生成记忆并适时注入。
> 内置 `DreamScheduler` 后台线程，在无对话时自动触发做梦引擎，让记忆系统真正"活着"。

- **Base URL**：`http://127.0.0.1:8190`（默认，可由 `LMS_API_HOST` / `LMS_API_PORT` 覆盖）
- **交互式文档**：`http://127.0.0.1:8190/docs`（Swagger UI）
- **启动方式**：`python -m api.run` 或 `python api/run.py --host 0.0.0.0 --port 8190 --reload`
- **并发说明**：记忆处理与 LLM 调用均为同步阻塞（torch 非 async），端点通过
  `run_in_executor` 将阻塞调用交给线程池，避免阻塞事件循环。

## 目录

- [端点列表](#端点列表)
- [数据模型](#数据模型)
- [端点详情](#端点详情)
  - [POST /chat](#post-chat)
  - [POST /chat/simple](#post-chatsimple)
  - [GET /status/{session_id}](#get-statussession_id)
  - [POST /snapshot/{session_id}](#post-snapshotsession_id)
  - [POST /restore/{session_id}](#post-restoresession_id)
  - [GET /sessions](#get-sessions)
  - [DELETE /sessions/{session_id}](#delete-sessionsession_id)
  - [POST /dream/{session_id}](#post-dreamsession_id)
  - [GET /dream/status](#get-dreamstatus)
  - [GET /health](#get-health)
- [错误码说明](#错误码说明)

---

## 端点列表

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/chat` | 完整对话（手动控制 LLM 注入时机） |
| POST | `/chat/simple` | 简化对话（自动处理记忆 + LLM 查询） |
| GET | `/status/{session_id}` | 查询会话记忆状态 |
| POST | `/snapshot/{session_id}` | 保存快照 |
| POST | `/restore/{session_id}` | 从快照恢复 |
| GET | `/sessions` | 列出所有会话 |
| DELETE | `/sessions/{session_id}` | 删除指定会话 |
| POST | `/dream/{session_id}` | 手动触发做梦 |
| GET | `/dream/status` | 查询做梦调度器状态 |
| GET | `/health` | 健康检查 |

---

## 数据模型

### ChatRequest

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `user_input` | string | 是 | — | 用户输入文本 |
| `session_id` | string | 否 | `"default"` | 会话标识 |
| `llm_output` | string | 否 | `""` | 上一轮 LLM 输出（可选） |

### ChatResponse

| 字段 | 类型 | 说明 |
|------|------|------|
| `response` | string | LLM 回复（未配置 LLM 时为记忆 context） |
| `memory_context` | string | 记忆 context |
| `memory_state` | object | 记忆系统状态 |
| `session_id` | string | 会话标识 |

### SimpleChatRequest

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `user_input` | string | 是 | — | 用户输入文本 |
| `session_id` | string | 否 | `"default"` | 会话标识 |

### SimpleChatResponse

| 字段 | 类型 | 说明 |
|------|------|------|
| `response` | string | LLM 回复 |
| `memory_status` | object | 记忆系统状态 |

### SnapshotRequest

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `path` | string \| null | 否 | `null` | 自定义快照路径（默认 `./snapshots/{sid}_{timestamp}.pt`） |

### RestoreRequest

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `path` | string | 是 | 快照文件路径 |

### DreamRequest

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `steps` | integer | 否 | `20` | 做梦步数 |
| `full_cycle` | boolean | 否 | `false` | 是否启用完整七阶段周期 |

### 记忆系统状态（memory_state / memory_status）

`memory_state` / `memory_status` 为 `LivingMemoryLoop.get_status()` 的返回值，常见字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `turn_count` | integer | 当前对话轮次 |
| `num_nodes` | integer | 吸引子网络节点数 |
| `input_dim` | integer | 感官输入维度 |
| `last_entropy` | float | 上一轮激活熵 |
| `last_surprise` | float | 上一轮自由能（惊讶度） |
| `entropy_ratio` | float | 在线熵管理比例 |
| `entropy_high_threshold` | float | 熵上限阈值 |
| `entropy_low_threshold` | float | 熵下限阈值 |
| `purpose_coherence` | float | 目的层 coherence |
| `precision_mean` | float | precision 均值 |
| `precision_std` | float | precision 标准差 |
| `meta` | object \| null | 元可塑性状态（启用时） |
| `episodic_buffer_size` | integer | 情景记忆缓冲区大小（API 层补充） |
| `llm_enabled` | boolean | 是否配置了 LLM（API 层补充） |

---

## 端点详情

### POST /chat

完整对话端点，手动控制 LLM 注入时机。

**流程**：获取/创建会话 → 通过 DreamScheduler 获取对话权限 → `process_turn` 生成记忆 context
→ 若配置 LLM 则调用 `bridge.query` 获取回复，否则返回记忆 context → 释放对话权限。

**请求体**：`ChatRequest`

```json
{
  "user_input": "你好，我叫小明",
  "session_id": "demo",
  "llm_output": ""
}
```

**响应**：`200 OK`，`ChatResponse`

```json
{
  "response": "你好小明！很高兴认识你……",
  "memory_context": "【海马体记忆context】激活节点: ...",
  "memory_state": {
    "turn_count": 1,
    "num_nodes": 256,
    "input_dim": 64,
    "last_entropy": 0.72,
    "last_surprise": 1.34,
    "entropy_ratio": 0.8,
    "entropy_high_threshold": 0.9,
    "entropy_low_threshold": 0.5,
    "purpose_coherence": 0.95,
    "precision_mean": 1.02,
    "precision_std": 0.31,
    "episodic_buffer_size": 1,
    "llm_enabled": true
  },
  "session_id": "demo"
}
```

**状态码**

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | `user_input` 为空 |
| 503 | 系统正在做梦（记忆巩固中），请稍后重试；或 LLM 调用失败时返回记忆 context 与错误信息 |

> LLM 调用失败时不会中断服务：`response` 字段会包含错误说明并附上记忆 context，
> HTTP 状态仍为 200（错误在响应体内体现）。

---

### POST /chat/simple

简化对话端点，仅需 `user_input`，内部调用 `loop.query_llm()` 自动完成记忆 + LLM 查询。

**请求体**：`SimpleChatRequest`

```json
{
  "user_input": "你还记得我叫什么吗？",
  "session_id": "demo"
}
```

**响应**：`200 OK`，`SimpleChatResponse`

```json
{
  "response": "当然记得，你叫小明。",
  "memory_status": {
    "turn_count": 2,
    "num_nodes": 256,
    "input_dim": 64,
    "last_entropy": 0.65,
    "last_surprise": 0.88,
    "purpose_coherence": 0.93,
    "precision_mean": 1.05,
    "episodic_buffer_size": 2,
    "llm_enabled": true
  }
}
```

**状态码**

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | `user_input` 为空 |
| 503 | 未配置 LLM Bridge（需配置 `DEEPSEEK_API_KEY`，或改用 `/chat`） |
| 500 | LLM 查询失败 |

---

### GET /status/{session_id}

返回指定会话的记忆系统状态。

**路径参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话标识 |

**响应**：`200 OK`

```json
{
  "session_id": "demo",
  "status": {
    "turn_count": 2,
    "num_nodes": 256,
    "input_dim": 64,
    "last_entropy": 0.65,
    "last_surprise": 0.88,
    "entropy_ratio": 0.8,
    "entropy_high_threshold": 0.9,
    "entropy_low_threshold": 0.5,
    "purpose_coherence": 0.93,
    "precision_mean": 1.05,
    "precision_std": 0.30
  }
}
```

**状态码**

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 404 | 会话不存在 |

---

### POST /snapshot/{session_id}

保存指定会话的快照。默认保存到 `./snapshots/{session_id}_{timestamp}.pt`。

**路径参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话标识 |

**请求体**：`SnapshotRequest`（可为空对象 `{}`）

```json
{
  "path": null
}
```

**响应**：`200 OK`

```json
{
  "session_id": "demo",
  "saved": true,
  "path": "Z:\\QH\\AI专用\\活体记忆系统\\snapshots\\demo_20260731_120000.pt",
  "turn_count": 2
}
```

**状态码**

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 404 | 会话不存在 |
| 500 | 快照保存失败 |

---

### POST /restore/{session_id}

从快照恢复指定会话的状态。

**路径参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话标识 |

**请求体**：`RestoreRequest`

```json
{
  "path": "Z:\\QH\\AI专用\\活体记忆系统\\snapshots\\demo_20260731_120000.pt"
}
```

**响应**：`200 OK`

```json
{
  "session_id": "demo",
  "restored": true,
  "path": "Z:\\QH\\AI专用\\活体记忆系统\\snapshots\\demo_20260731_120000.pt",
  "status": {
    "turn_count": 2,
    "num_nodes": 256,
    "input_dim": 64,
    "last_entropy": 0.65,
    "last_surprise": 0.88,
    "purpose_coherence": 0.93,
    "precision_mean": 1.05,
    "episodic_buffer_size": 2,
    "llm_enabled": true
  }
}
```

**状态码**

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 400 | `path` 为空 |
| 404 | 会话不存在，或快照文件不存在 |
| 500 | 恢复失败 |

---

### GET /sessions

列出所有活跃的会话。

**响应**：`200 OK`

```json
{
  "sessions": ["default", "demo"],
  "count": 2
}
```

**状态码**：`200 OK`

---

### DELETE /sessions/{session_id}

删除指定会话，并从做梦调度器注销。

**路径参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话标识 |

**响应**：`200 OK`

```json
{
  "session_id": "demo",
  "deleted": true
}
```

**状态码**

| 状态码 | 说明 |
|--------|------|
| 200 | 成功 |
| 404 | 会话不存在 |

---

### POST /dream/{session_id}

手动触发指定会话的做梦。若会话不存在则创建；做梦期间该会话的对话请求会等待。

**路径参数**

| 参数 | 类型 | 说明 |
|------|------|------|
| `session_id` | string | 会话标识 |

**请求体**：`DreamRequest`（可选，可省略请求体使用默认值）

```json
{
  "steps": 30,
  "full_cycle": false
}
```

**响应**：`200 OK`

```json
{
  "session_id": "demo",
  "result": {
    "mode": "mvp",
    "steps": 30,
    "avg_surprise": 0.42,
    "collapse_count": 3,
    "snapshot_saved": true
  }
}
```

> `result` 为 `DreamEngine` 返回的做梦统计字典，额外包含 `snapshot_saved` 字段。
> 做梦失败时返回 `{"status": "error", "error": "..."}`。

**状态码**

| 状态码 | 说明 |
|--------|------|
| 200 | 成功（含做梦失败的情况，错误在响应体内体现） |

---

### GET /dream/status

查询做梦调度器（DreamScheduler）状态。

**响应**：`200 OK`

```json
{
  "running": true,
  "is_dreaming": false,
  "dreaming_session": null,
  "idle_threshold": 30.0,
  "dream_steps": 20,
  "dream_full_cycle": false,
  "check_interval": 5.0,
  "registered_sessions": 2,
  "sessions": [
    {
      "session_id": "default",
      "idle_seconds": 12.5,
      "dream_count": 3,
      "last_dream_ts": 1785500000.0,
      "last_dream_steps": 20
    },
    {
      "session_id": "demo",
      "idle_seconds": 45.2,
      "dream_count": 5,
      "last_dream_ts": 1785500030.0,
      "last_dream_steps": 20
    }
  ]
}
```

**字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| `running` | boolean | 调度器后台线程是否运行 |
| `is_dreaming` | boolean | 当前是否正在做梦 |
| `dreaming_session` | string \| null | 正在做梦的会话标识 |
| `idle_threshold` | float | 空闲触发阈值（秒） |
| `dream_steps` | integer | 每次自动做梦步数 |
| `dream_full_cycle` | boolean | 是否使用完整七阶段周期 |
| `check_interval` | float | 检查间隔（秒） |
| `registered_sessions` | integer | 已注册会话数 |
| `sessions` | array | 各会话活动状态明细 |

**状态码**：`200 OK`

---

### GET /health

健康检查。

**响应**：`200 OK`

```json
{
  "status": "ok",
  "service": "living-memory-api",
  "version": "0.1.0",
  "active_sessions": 2,
  "timestamp": "2026-07-31T12:00:00.000000"
}
```

**状态码**：`200 OK`

---

## 错误码说明

所有错误响应均为 FastAPI 标准 `HTTPException` 格式：

```json
{
  "detail": "错误描述信息"
}
```

| HTTP 状态码 | 触发场景 | 示例 detail |
|-------------|----------|-------------|
| 400 | 请求参数校验失败 | `"user_input 不能为空"` / `"path 不能为空"` |
| 404 | 资源不存在 | `"会话 'demo' 不存在"` / `"快照文件不存在: /path/to.pt"` |
| 500 | 服务端内部错误 | `"快照保存失败: ..."` / `"恢复失败: ..."` / `"LLM 查询失败: ..."` |
| 503 | 服务暂不可用 | `"系统正在做梦（记忆巩固中），请稍后重试。"` / `"未配置 LLM Bridge，无法使用 /chat/simple。请配置 DEEPSEEK_API_KEY 后重试，或改用 /chat 端点。"` |

### 503 并发保护机制

`/chat` 与 `/chat/simple` 通过 `DreamScheduler.acquire_conversation()` 协调对话与后台做梦的
并发写入。当系统正在做梦且等待超时时，返回 503 以避免对话处理与后台做梦并发写同一份
记忆状态（`J` 矩阵 / precision / latent）。客户端应稍后重试。

---

*端点定义源码：`api/server.py` · 配置源码：`api/config.py` · 启动脚本：`api/run.py`*

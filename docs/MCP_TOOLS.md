# 活体记忆系统 - MCP 工具文档

> 通过 MCP（Model Context Protocol）协议向 TRAE IDE 暴露活体记忆系统的核心能力，
> 使 AI 助手能够检索记忆、存储对话、查询状态。基于 mcp 2.0.0 的 Server 类
> （构造器注册 handler，非装饰器模式），通过 stdio 传输与客户端通信。

- **启动方式**：`python mcp_memory_server.py`
- **传输方式**：stdio（stdout 专用于 MCP 协议通信，日志输出到 stderr）
- **源码**：`mcp_memory_server.py`

## 目录

- [工具列表](#工具列表)
- [recall_memory](#recall_memory)
- [store_memory](#store_memory)
- [get_memory_status](#get_memory_status)
- [初始化与快照机制](#初始化与快照机制)
- [错误处理](#错误处理)

---

## 工具列表

| 工具 | 用途 | 参数 |
|------|------|------|
| `recall_memory` | 语义检索与查询相关的历史对话记忆（top 3） | `query`（必填） |
| `store_memory` | 将当前对话存储到记忆系统（执行完整记忆循环并保存快照） | `text`（必填） |
| `get_memory_status` | 获取记忆系统运行状态（轮次、熵、惊讶度、precision 等） | 无 |

---

## recall_memory

### 用途

检索与查询相关的历史对话记忆。当需要回忆之前与用户讨论的内容时调用此工具。
例如：用户问"你还记得我之前说的吗"，或需要之前的上下文信息时。建议在每次对话开始时主动调用。

### 工作原理

1. 使用预训练模型将查询编码为语义向量（优先 384 维原始向量，fallback 64 维投影向量）。
2. 在情景记忆缓冲区中检索最相关的 top 3 条记忆（`recall_episodic`，向量化矩阵运算）。
3. 返回记忆文本与系统状态。

> 若当前嵌入器为 `SimpleEmbedder`（不支持语义检索），则返回提示信息与缓冲区大小，
> 不执行检索。

### 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `query` | string | 是 | 检索查询文本 |

**inputSchema**

```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "检索查询文本"
    }
  },
  "required": ["query"]
}
```

### 调用示例

```json
{
  "name": "recall_memory",
  "arguments": {
    "query": "用户之前提到过的名字"
  }
}
```

### 返回值（CallToolResult.content[0].text）

返回格式化的纯文本字符串。命中记忆时示例：

```text
检索到 2 条相关记忆（查询: "用户之前提到过的名字"）:
情景记忆缓冲区大小: 5

--- 记忆 1（轮次 1，惊讶度 1.340）---
用户: 你好，我叫小明
助手: 你好小明！很高兴认识你。

--- 记忆 2（轮次 3，惊讶度 0.880）---
用户: 你还记得我叫什么吗？
助手: 当然记得，你叫小明。

记忆系统状态:
{
  "turn_count": 5,
  "num_nodes": 256,
  "input_dim": 64,
  "last_entropy": 0.65,
  "last_surprise": 0.88,
  "purpose_coherence": 0.93,
  "precision_mean": 1.05
}
```

未检索到记忆时示例：

```text
未检索到相关记忆（查询: "用户之前提到过的名字"）。
情景记忆缓冲区大小: 0

记忆系统状态:
{ ... }
```

嵌入器不支持语义检索时示例：

```text
当前嵌入器不支持语义检索（SimpleEmbedder），无法进行语义记忆检索。

情景记忆缓冲区大小: 5
记忆系统状态:
{ ... }
```

---

## store_memory

### 用途

将当前对话内容存储到记忆系统。在每次与用户完成一轮对话后调用此工具，
保存用户的输入和你的回复。这使系统能在未来检索到本次对话内容。

### 工作原理

1. 解析输入文本：若包含 `用户:` / `助手:` 标记，拆分为 `(user_input, llm_output)`
   两部分分别传入 `process_turn`；若为纯文本，则整段作为 `user_input`，
   `llm_output` 留空。（A-P1-1 修复：避免标记重复嵌套导致记忆编码失真。）
2. 调用 `LivingMemoryLoop.process_turn(user_input, llm_output)` 执行完整记忆循环
   （编码 → 推断 → 学习 → 在线熵管理 → 调整目的 → 记忆更新与巩固 → 检索 → 解码 context）。
3. 自动保存快照到 `snapshots/latest.pt`，确保持久化跨会话记忆。
4. 返回存储确认信息与记忆状态（JSON 格式）。

### 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `text` | string | 是 | 要存储的对话文本（建议格式：`用户: xxx\n助手: xxx`） |

**inputSchema**

```json
{
  "type": "object",
  "properties": {
    "text": {
      "type": "string",
      "description": "要存储的对话文本（可以是'用户: xxx\\n助手: xxx'格式）"
    }
  },
  "required": ["text"]
}
```

### 调用示例

```json
{
  "name": "store_memory",
  "arguments": {
    "text": "用户: 我喜欢用 Python 写代码\n助手: 很好，Python 是一门优雅的语言。"
  }
}
```

### 返回值（CallToolResult.content[0].text）

返回 JSON 格式字符串（经 `_to_json` 序列化，自动处理 torch/numpy 标量）：

```json
{
  "status": "已存储",
  "turn_count": 6,
  "episodic_buffer_size": 6,
  "last_entropy": 0.68,
  "last_surprise": 1.12,
  "precision_mean": 1.08,
  "purpose_coherence": 0.91,
  "snapshot_saved": true,
  "memory_context": "【海马体记忆context】激活节点: ..."
}
```

**字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | 固定为 `"已存储"` |
| `turn_count` | integer | 存储后的对话总轮次 |
| `episodic_buffer_size` | integer | 情景记忆缓冲区大小 |
| `last_entropy` | float \| null | 本轮激活熵 |
| `last_surprise` | float \| null | 本轮惊讶度（准确性项，恒≥0；2026-08-10 语义拆分，原为自由能） |
| `last_free_energy` | float \| null | 本轮自由能（未规范化变分能量，可负；仅供学习目标/诊断） |
| `precision_mean` | float \| null | precision 均值 |
| `purpose_coherence` | float \| null | 目的层 coherence |
| `snapshot_saved` | boolean | 是否成功保存快照 |
| `memory_context` | string | 本轮生成的记忆 context |

---

## get_memory_status

### 用途

获取记忆系统的当前运行状态（轮次数、熵、惊讶度、precision 等），用于了解记忆系统的运作情况。

### 工作原理

调用 `LivingMemoryLoop.get_status()` 获取状态字典，并补充情景记忆缓冲区大小与嵌入器类型信息。

### 参数

无参数。

**inputSchema**

```json
{
  "type": "object",
  "properties": {},
  "required": []
}
```

### 调用示例

```json
{
  "name": "get_memory_status",
  "arguments": {}
}
```

### 返回值（CallToolResult.content[0].text）

返回 JSON 格式字符串：

```json
{
  "turn_count": 6,
  "num_nodes": 256,
  "input_dim": 64,
  "last_entropy": 0.68,
  "last_surprise": 1.12,
  "entropy_ratio": 0.8,
  "entropy_high_threshold": 0.9,
  "entropy_low_threshold": 0.5,
  "purpose_coherence": 0.91,
  "precision_mean": 1.08,
  "precision_std": 0.32,
  "meta": {
    "surprise_trend": -0.05,
    "orth_factor": 1.02
  },
  "episodic_buffer_size": 6,
  "embedder_type": "PretrainedEmbedder",
  "embedder_raw_dim": 384
}
```

**字段说明**

| 字段 | 类型 | 说明 |
|------|------|------|
| `turn_count` | integer | 当前对话轮次 |
| `num_nodes` | integer | 吸引子网络节点数 |
| `input_dim` | integer | 感官输入维度 |
| `last_entropy` | float | 上一轮激活熵（无激活态时缺失） |
| `last_surprise` | float | 上一轮惊讶度（准确性项，恒≥0；2026-08-10 语义拆分，原为自由能；无激活态时缺失） |
| `last_free_energy` | float | 上一轮自由能（未规范化变分能量，可负；仅供学习目标/诊断；无激活态时缺失） |
| `entropy_ratio` | float | 在线熵管理比例 |
| `entropy_high_threshold` | float | 熵上限阈值 |
| `entropy_low_threshold` | float | 熵下限阈值 |
| `purpose_coherence` | float | 目的层 coherence |
| `precision_mean` | float | precision 均值 |
| `precision_std` | float | precision 标准差 |
| `meta` | object \| null | 元可塑性状态（启用 `meta_enabled` 时） |
| `episodic_buffer_size` | integer | 情景记忆缓冲区大小（工具补充） |
| `embedder_type` | string | 嵌入器类名（工具补充，如 `PretrainedEmbedder` / `SimpleEmbedder`） |
| `embedder_raw_dim` | integer | 嵌入器原始维度（仅 PretrainedEmbedder 存在，工具补充） |

---

## 初始化与快照机制

### 懒加载初始化

三个工具共享一个全局 `LivingMemoryLoop` 实例（`get_loop()`，懒加载）：

1. 首次调用任一工具时初始化记忆系统（加载预训练模型等）。
2. **自动加载最新快照**：扫描 `snapshots/` 目录下 `.pt` 文件，按修改时间取最新者
   通过 `load_state()` 恢复，实现跨会话记忆延续。
3. 若无快照，使用空白记忆启动。
4. 初始化失败时记录错误，后续调用直接抛出 `RuntimeError`，避免重复尝试。

### 嵌入器配置

- 默认使用 `PretrainedEmbedder`（`paraphrase-multilingual-MiniLM-L12-v2`，384→64 维投影），
  并将 `activation_threshold` 适配为 0.02。
- 加载失败时自动降级为 `SimpleEmbedder`（不支持语义检索，`recall_memory` 会返回提示）。
- 模型路径由环境变量 `LMS_PRETRAINED_MODEL` 覆盖，默认指向本地 modelscope 缓存。

### 快照保存

- `store_memory` 每次存储后自动保存到 `snapshots/latest.pt`。
- 快照保存失败不影响当前操作（仅记录 warning）。

---

## 错误处理

工具执行出错时返回 `is_error=true` 的 `CallToolResult`，`content[0].text` 为错误描述：

| 错误类型 | content[0].text | is_error |
|----------|-----------------|----------|
| 缺少必填参数 | `"错误：缺少必需参数 'query'"` / `"错误：缺少必需参数 'text'"` | true |
| 未知工具名 | `"错误：未知工具 'xxx'。可用工具: recall_memory, store_memory, get_memory_status, dream_memory"` | true |
| 执行异常 | `"工具执行出错: <异常信息>"` | true |
| 记忆系统初始化失败 | `"工具执行出错: 记忆系统初始化失败: <异常信息>"` | true |

> **注**：当前 `mcp_memory_server.py` 实际注册了 4 个工具，除上述 3 个外还包括
> `dream_memory`（触发记忆系统"做梦"，在空闲时进行记忆巩固与整合）。本文档按
> E-P2-3 任务要求重点记录核心的 3 个工具；`dream_memory` 的参数为
> `steps`（integer，默认 20）与 `full_cycle`（boolean，默认 false）。

---

*源码：`mcp_memory_server.py` · 核心循环：`runtime/loop.py` · 主循环步骤详见 README.md*

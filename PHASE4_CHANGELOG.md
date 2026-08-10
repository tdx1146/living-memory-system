# PHASE4_CHANGELOG.md — LMS 侧双向反馈接入（Phase 4 / D0）

> 日期：2026-08-04 ｜ 工程：Agent OS 总线改良 Phase 4：LMS 双向反馈中枢（D0）
> 原则：**外围化、可熔断、默认保守**。LMS 是意识候选，所有改动只加外围钩子，
> core/hippocampus/*（J矩阵、吸引子、目的层、记忆、dream_engine 内部算法、
> self_ref 自指回路本体）一律零改动。

---

## 0. 改动前基线

- 备份：`<BACKUP_ROOT>/`（runtime/ + api/ 双保险，LMS 另有 git）
- 测试基线：**672 passed**（`.venv/bin/python -m pytest -q`，2026-08-04 17:37 实测；此前记录的 158 为更早期阶段基线，当前全量套件为 672）
- 部署状态：API 127.0.0.1:8190（PID 25858，旧代码）、MCP 已注册、glue_server 19000 运行中

## 1. 新增 `runtime/bus_events.py`（LMS → 总线 发布侧）

发布侧独立模块，与 iso-sand `LogWriter` 语义等价（append + `fcntl.flock` + `fsync`），
契约对齐 v1.1：`schema_version="1.1"` / `event_id=uuid4` / `trace_id` / `t`(BJT ISO8601) /
`event_type` / `producer="lms"` / `result` / `payload`。

目标文件：`<AGENTOS_BUS_FILE>`
（环境变量 `LMS_BUS_FILE` 可覆盖）。

### 熔断闸门（发布侧）
- 连续失败 `LMS_BUS_MAX_FAILURES`（默认 5）次 → 熔断暂停发布
  `LMS_BUS_COOLDOWN_SECONDS`（默认 600s = 10 分钟）
- 冷却结束自动进入半开试探，成功即复位失败计数；**任何发布异常 try/except 静默降级，
  绝不抛给 LMS 主循环**
- 可观测：`get_publisher_status()` 返回熔断状态（open/tripped_count/连续失败数等）

### 发布函数
| 函数 | 事件 | payload 内容 |
|------|------|-------------|
| `publish_plastified(state)` | `lms.plastified` | 数值摘要：entropy/surprise/active_nodes/precision_mean/precision_std/coherence/entropy_ratio/turn_count（**只发数值摘要，绝不发原始激活态/J矩阵**） |
| `publish_dream_complete(stats)` | `lms.dream_complete` | steps/duration_seconds/snapshot_saved |
| `publish_self_ref(summary, guard)` | `lms.self_ref` | 蒸馏后文本摘要（**≤200 字**，超长截断）+ 护栏状态（state/autocorr/alpha/coherence 等）；受 `LMS_SELF_REF_PUBLISH` 控制**默认 "off"**，on 时限频 `LMS_SELF_REF_MIN_INTERVAL`（默认 1800s = 30 分钟一条） |

- payload 序列化带 `_sanitize`：torch/numpy 标量 → float/int、嵌套深度/条目上限、NaN/Inf 过滤，防原始激活态泄漏

## 2. 发布点接入（`runtime/loop.py`，只加钩子不改算法）

1. `process_turn` 返回前加 `_maybe_publish_plastified(activation)`：
   实例计数 `turn_count % interval`（默认 10 轮，`LMS_PLASTIFIED_INTERVAL` 可覆盖），不加线程
2. `dream()` 完成路径加 `_maybe_publish_dream_complete(result, duration)`：
   覆盖 dream_scheduler 自动做梦 / API 手动触发 / MCP 全部路径（唯一收口点）
3. `_inject_self_ref` 内加 `_maybe_publish_self_ref(echo)`：
   只做"蒸馏摘要发布钩子"（默认关闭），不碰自指回路本体

所有钩子：lazy import + try/except 双层兜底（`bus_events` 内部熔断 + 调用点静默降级），
日志仅 debug 级，**LMS 主循环绝不被总线拖垮**。

## 3. 新增 `POST /feed`（`api/server.py`，收事件侧）

- 请求：`{"text": "...", "session_id": "bus", "source": "event_bus"}`
- 行为：`process_turn(text, llm_output="")` 塑形但不产 LLM 回复（返回值丢弃），
  返回 `{"status":"ok","entropy":...,"surprise":...,"turn_count":...}`
- 限流：`LMS_FEED_RATE_LIMIT`（默认 10 次/分钟）滑动窗口，超限 429
- 做梦协调：与 /chat 同等对待（acquire_conversation，防并发写记忆状态）
- `/chat`、`/chat/simple`、`/dream`、`/status` 等现有端点**完全不变**

## 4. 验证记录（逐项）

| # | 项 | 结果 |
|---|----|------|
| 1 | LMS 全量测试 | ✅ **672 passed**（改动后 17:45 实测，基线保持） |
| 2 | 收侧 E2E | ✅ 注入 milestone v1.1 事件（trace phase4-e2e-milestone-002 / restart-004）→ consumer lms.feed handler → LMS /feed → operation_log "LMS 塑形喂入成功: turn_count 10→11"；interfaces.store 事件同时触发 glue /store（成功）与 lms.feed |
| 3 | 发侧 E2E | ✅ 受控发布 turn_count=77777 事件 + API 进程自然触发（turn_count=10）→ 总线出现 lms.plastified，v1.1 字段齐全（schema_version/event_id/trace_id/t/event_type/producer/result/payload）→ consumer "无匹配规则" 正常跳过，**零死信** |
| 4 | dream_complete | ✅ API /dream/bus 触发 → 总线出现 lms.dream_complete（steps=5, duration=0.245s, snapshot_saved=true）；LMS 测试套件亦产生 22 条同类事件 |
| 5 | self_ref | ✅ 生产总线 **0 条 lms.self_ref**（默认 off，即使测试套件大量跑自指回路）；`LMS_SELF_REF_PUBLISH=on` 后发布成功、限频生效（窗口内第二条被拒）、摘要截断至 ≤200 字（含省略号） |
| 6 | 熔断 | ✅ 路径指错（指向目录）→ 连续失败计数 1/5→5/5 → "🔴 熔断触发，暂停发布 600 秒"；熔断中发布被静默拒绝；冷却后自动恢复（成功写入 turn_count=54321，失败计数清零）；全程无异常抛出 |
| 7 | 总线稳定性 | ✅ sandglass.heartbeat 每 5 分钟持续（17:50:29 / 17:55:29 / 17:56:45）；consumer 日志 0 ERROR/Traceback；拓扑 `cycles: []` 无环 |
| 8 | 服务重启 | ✅ 重启 LMS API + iso-sand consumer/scheduler 后：/health、/feed（turn_count=1）、/chat（memory_context 269 字）、/dream/status（running）、心跳、收侧 E2E 全部正常 |
| 9 | 回滚 | ✅ 无需回滚（全部通过） |

## 5. 安全红线确认

- `core/hippocampus/*`、J矩阵、dream_engine 内部算法、self_ref 自指回路本体：**git diff 为空，零改动** ✅
- self_ref 发布默认关闭（看护人式第一步：保守护着）✅
- 未 git push / commit（仅工作区改动）；未打印任何 token/密钥 ✅

## 6. 遗留问题 / 已知行为

1. **测试套件会向生产总线发布事件**：pytest 运行 LivingMemoryLoop 达间隔（10 轮）会经钩子向
   默认 `LMS_BUS_FILE` 写入 lms.plastified/dream_complete（本次验证即产生 496+22 条）。
   属设计行为（LMS 实例发布软参考事件），consumer 正常跳过；若需隔离，跑测试时设
   `LMS_BUS_FILE=/tmp/xxx.jsonl` 即可。
2. **429 双保险语义**：LMS /feed 限流 10 次/分钟（防总线风暴）+ handler 侧 1s 间隔；
   总线繁忙超限时 lms.feed fail-open 进死信队列（可观测、可恢复，不重试、不拖垮总线）——
   这是设计意图而非缺陷。调度器心跳 task_complete 每 5 分钟自然喂入 LMS 一次。
3. `runtime.bus_events` 经 `runtime/__init__.py` 导入会连带加载 torch（~19s 首导），
   在 API 进程内无感（torch 已加载）；钩子采用 lazy import，不影响无总线场景。
4. lms.* 事件在 topology 中显示"active 但无人消费"WARN：符合"软参考信号，订阅方可忽略"设计，
   消费者接入（沙漏/玄鉴软参考）属后续阶段，非本阶段缺陷。

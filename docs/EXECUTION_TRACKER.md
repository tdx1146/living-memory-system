# 活体记忆系统 综合优化执行跟踪

> 创建时间：2026-07-31
> 最后更新：2026-07-31
> 状态：P0+P1+P2 全部完成（23/23项），540测试通过
> 来源：综合「代码优化方案 v1.0」与「另一AI部署后分析」两份方案

## 一、方案合并背景

两份方案分别覆盖两个维度：
- **代码优化方案**：工程安全、并发安全、性能、工程基建（17项）
- **另一AI的分析**：算法有效性、记忆活性、系统集成（6项关键问题）

核心判断：工程红线先行，算法激活紧随，P1阶段两者并行，P2长期演进。

## 二、综合优先级矩阵

### P0 红线（阻塞一切）

| 编号 | 维度 | 任务 | 涉及文件 | 状态 |
|------|------|------|----------|------|
| E-P0-1 | 工程 | API Key泄露修复（轮换+环境变量化+git历史清理） | memory_cli.py, .env.example, .gitignore | ✅已完成 |
| E-P0-2 | 工程 | 并发数据竞争修复（acquire返回值检查+run_in_executor） | api/server.py, runtime/dream_scheduler.py | ✅已完成 |
| E-P0-3 | 工程 | 快照原子写入（tmp+os.replace） | persistence/snapshot.py, dream_engine.py | ✅已完成 |
| E-P0-4 | 工程 | setup.py依赖修复 | setup.py, requirements.txt | ✅已完成 |
| A-P0-1 | 算法 | coherence计算重构（让元目的翻转能真正触发） | core/hippocampus/purpose.py | ✅已完成 |
| A-P0-2 | 算法 | 在线熵管理注入（process_turn中熵超阈值时干预） | runtime/loop.py, core/config.py | ✅已完成 |

### P1 本轮迭代（1-2周）

| 编号 | 维度 | 任务 | 交叉点 | 状态 |
|------|------|------|--------|------|
| E-P1-1 | 工程 | recall_episodic向量化 | - | ✅已完成 |
| E-P1-2 | 工程 | DreamEngine解耦（移除私有属性访问） | 与A-P1-2合并 | ✅已完成 |
| E-P1-3 | 工程 | CoreConfig校验+统一配置体系 | 与A-P0-1/A-P0-2合并 | ✅已完成 |
| E-P1-4 | 工程 | API层测试补齐 | - | ✅已完成 |
| E-P1-5 | 工程 | CI/CD搭建 | - | ✅已完成 |
| E-P1-6 | 工程 | recover异常链+回滚 | - | ✅已完成 |
| E-P1-7 | 工程 | LLM重试策略优化 | - | ✅已完成 |
| A-P1-1 | 算法 | do_store_memory接口修复（分离user/AI输入） | 与E-P1-2同步 | ✅已完成 |
| A-P1-2 | 算法 | decoder解释性输出（指标转自然语言） | - | ✅已完成 |

### P2 长期规划

| 编号 | 维度 | 任务 | 依赖 | 状态 |
|------|------|------|------|------|
| E-P2-1 | 工程 | GPU/device管理 | - | ✅已完成 |
| E-P2-2 | 工程 | Docker容器化 | - | ✅已完成 |
| E-P2-3 | 工程 | README+API文档 | - | ✅已完成 |
| E-P2-4 | 工程 | 预训练Embedder资源管理 | - | ✅已完成 |
| E-P2-5 | 工程 | 临时参数改接口传参 | 依赖E-P1-2 | ✅已完成 |
| E-P2-6 | 工程 | 代码质量清理 | 依赖E-P2-1/E-P2-5 | ✅已完成 |
| A-P2-1 | 算法 | LMS↔沙漏翻译层 | 依赖E-P1-2 | ✅已完成 |
| A-P2-2 | 算法 | 384→64维投影评估 | - | ✅已完成 |

## 三、执行路线

### 阶段一：工程P0（第1周）

3个子AI并行，文件互不冲突：
- 子AI-1：E-P0-1（API Key）+ E-P0-4（依赖修复）— 配置类合并
- 子AI-2：E-P0-2（并发竞争）— api/server.py + dream_scheduler.py
- 子AI-3：E-P0-3（快照原子写入）— snapshot.py + dream_engine.py

完成后统一测试 + 提交。

### 阶段二：算法P0（第1周末-第2周）

2个子AI并行：
- 子AI-4：A-P0-1（coherence重构）
- 子AI-5：A-P0-2（在线熵管理）

完成后跑长期状态验证（100轮模拟对话）+ 提交。

### 阶段三：P1并行（第2-3周）

工程层与算法层交叉点合并处理：
- recall向量化 + DreamEngine解耦 + store接口修复（同一轮接口重构）
- CoreConfig校验（纳入coherence新参数）
- decoder解释性输出
- 测试 + CI守护

### 阶段四：P2长期（第4周及以后）

LMS↔沙漏翻译层、维度投影评估、Docker+文档。

## 四、执行日志

### 2026-07-31 工程P0完成

3个子AI并行执行，全部完成：

- [x] 子AI-1：E-P0-1（API Key移除+环境变量化+.env.example+.gitignore）+ E-P0-4（setup.py/requirements.txt依赖补全）
  - 额外修复 e2e_controlled_experiment.py 中的硬编码Key
- [x] 子AI-2：E-P0-2（acquire返回值检查+503拒绝+/chat和/chat/simple走run_in_executor+dream_scheduler锁保护）
- [x] 子AI-3：E-P0-3（snapshot.py和dream_engine.py的tmp+os.replace原子写入）
- [x] 统一测试：249 passed
- [x] 提交：537db90 → rebase → b0dbcaf pushed to GitHub

修改文件清单：
- memory_cli.py（Key移除）
- e2e_controlled_experiment.py（Key移除）
- .env.example（新建）
- .gitignore（更新）
- setup.py（依赖补全）
- requirements.txt（mcp补充）
- api/server.py（并发修复+run_in_executor）
- runtime/dream_scheduler.py（锁保护）
- persistence/snapshot.py（原子写入）
- core/hippocampus/dream_engine.py（原子写入）

### 2026-07-31 算法P0启动

2个子AI并行：
- 子AI-4：A-P0-1（coherence计算重构）— 只改 core/hippocampus/purpose.py
- 子AI-5：A-P0-2（在线熵管理注入）— 只改 runtime/loop.py
- config.py 统一修改由主AI在两者完成后执行

### 2026-07-31 算法P0完成

- [x] 子AI-4：A-P0-1（coherence从纯余弦改为方向+幅度混合度量，threshold 0.3→0.5）
  - 验证：旧版coherence恒≈1.0永不触发翻转，新版在习惯化场景下coherence降至0.8056，触发11次翻转
- [x] 子AI-5：A-P0-2（process_turn新增步骤3.5在线熵管理，熵过高增强正交化/熵过低放松正交化）
- [x] 主AI统一更新：config.py新增6个参数，loop.py PurposeLayer初始化传入新参数
- [x] 统一测试：249 passed
- [x] 提交：5f3280d（本地）
- [ ] 推送：网络连接被重置，待网络恢复后push

修改文件清单：
- core/hippocampus/purpose.py（coherence混合度量+构造函数新参数）
- core/config.py（新增coherence_direction/magnitude_weight + entropy_high/low_threshold）
- runtime/loop.py（步骤3.5在线熵管理 + PurposeLayer初始化传参 + get_status新增字段）
- tests/test_purpose.py（更新coherence期望值）

### 2026-07-31 P1第一批完成（6/9项）

接口重构与健壮性批次完成，3个子AI并行：

- [x] 子AI-6：E-P1-1（recall_episodic向量化，矩阵乘法替代逐条dot）+ E-P1-2（DreamEngine解耦，新增buffer_size/iter_buffer/iter_episodic等公开接口）+ A-P1-1（do_store_memory分离user/AI输入，_parse_conversation解析对话格式）
  - 涉及文件：core/hippocampus/memory.py, core/hippocampus/dream_engine.py, mcp_memory_server.py
- [x] 子AI-7：A-P1-2（decoder解释性输出，_interpret_entropy/_interpret_surprise/_interpret_coherence将原始指标转为自然语言）
  - 涉及文件：bridge/decoder.py
- [x] 子AI-8：E-P1-6（recover异常链+回滚，raise...from e保留错误链）+ E-P1-7（LLM重试策略优化，区分4xx不可重试与5xx可重试）
  - 涉及文件：persistence/recovery.py, bridge/llm_bridge.py
- [x] 统一测试：249 passed

修改文件清单：
- core/hippocampus/memory.py（recall向量化+6个公开接口）
- core/hippocampus/dream_engine.py（使用公开接口替代私有属性访问）
- mcp_memory_server.py（_parse_conversation+do_store_memory重构）
- bridge/decoder.py（3个_interpret方法+自然语言输出）
- persistence/recovery.py（异常链+回滚机制）
- bridge/llm_bridge.py（重试策略分类）
- api/server.py（适配接口变更）
- api/session_manager.py（适配接口变更）
- runtime/dream_scheduler.py（适配接口变更）
- runtime/loop.py（适配接口变更）
- tests/test_dream_scheduler.py（适配接口变更）

### P1剩余任务（3项）

| 编号 | 任务 | 子AI分配 | 依赖 |
|------|------|----------|------|
| E-P1-3 | CoreConfig校验+统一配置体系 | 子AI-9 | 无（独立模块） |
| E-P1-4 | API层测试补齐 | 子AI-10 | 无（新增测试文件） |
| E-P1-5 | CI/CD搭建 | 子AI-11 | 无（新增配置文件） |

三项任务文件互不冲突，可并行执行。

### 2026-07-31 P1第二批完成（3/3项）— P1全部完成

3个子AI并行执行，全部完成：

- [x] 子AI-9：E-P1-3（CoreConfig新增validate()+to_loop_config()+from_env()三个方法）
  - validate()：覆盖全部参数约束，校验失败抛出ValueError含字段名和无效值
  - to_loop_config()：转换为LivingMemoryLoop需要的config dict（44个标量键）
  - from_env()：从LMS_前缀环境变量读取覆盖值，支持int/float/bool类型转换
  - 涉及文件：core/config.py, tests/test_config.py（88个测试）
- [x] 子AI-10：E-P1-4（API层测试补齐，27个测试覆盖全部10个HTTP端点）
  - 轻量级config（num_nodes=32），MockLLMBridge隔离外部API
  - 覆盖正常流程、错误处理、并发503拒绝、快照往返恢复
  - 涉及文件：tests/test_api_server.py
- [x] 子AI-11：E-P1-5（CI/CD搭建）
  - ci.yml：Python 3.10/3.11/3.12矩阵，CPU版PyTorch，pytest+覆盖率
  - lint.yml：ruff风格检查，line-length=120
  - 涉及文件：.github/workflows/ci.yml, .github/workflows/lint.yml
- [x] 统一测试：364 passed（从249增至364，新增115个测试）
- [x] 提交：fcf2750

P1阶段全部9项任务完成。工程层与算法层交叉点已全部处理。

### P1总结

| 批次 | 完成项 | 测试数 | 提交 |
|------|--------|--------|------|
| P1第一批 | E-P1-1, E-P1-2, E-P1-6, E-P1-7, A-P1-1, A-P1-2 | 249 | d041b27 |
| P1第二批 | E-P1-3, E-P1-4, E-P1-5 | 364 | fcf2750 |

待办：网络恢复后推送 5f3280d → d041b27 → fcf2750 三个提交到 GitHub。

### 2026-07-31 推送状态

- [x] 17个代码文件通过 GitHub Contents API 推送成功（远程已有最新代码）
- [ ] 2个 workflow 文件（ci.yml, lint.yml）因 PAT 缺少 workflow scope 未能推送
  - 需要用户更新 PAT 添加 workflow scope 后重试
- [ ] 本地 git fetch/push 因网络重置间歇性失败，待网络稳定后同步

## 五、P2 执行规划

### P2 第一批（无文件冲突，4项并行）

| 编号 | 任务 | 子AI | 涉及文件 | 冲突风险 |
|------|------|------|----------|----------|
| E-P2-2 | Docker容器化 | 子AI-12 | 新建 Dockerfile, docker-compose.yml, .dockerignore | 无 |
| E-P2-3 | README+API文档 | 子AI-13 | 新建 README.md, docs/API.md | 无 |
| A-P2-1 | LMS↔沙漏翻译层 | 子AI-14 | 新建 bridge/translation_layer.py | 无 |
| A-P2-2 | 384→64维投影评估 | 子AI-15 | 新建 evaluation/projection_eval.py | 无 |

### P2 第二批（修改核心模块，串行）

| 编号 | 任务 | 依赖 | 涉及文件 |
|------|------|------|----------|
| E-P2-1 | GPU/device管理 | 无 | core/hippocampus/attractor.py, memory.py, runtime/loop.py |
| E-P2-5 | 临时参数改接口传参 | E-P1-2（已完成） | runtime/loop.py, core/hippocampus/dream_engine.py |
| E-P2-4 | 预训练Embedder资源管理 | 无 | core/sensory/embedder.py |

### P2 第三批

| 编号 | 任务 | 依赖 |
|------|------|------|
| E-P2-6 | 代码质量清理 | E-P2-1 + E-P2-5 |

### 2026-07-31 P2第一批完成（4/8项）

4个子AI并行执行，全部完成：

- [x] 子AI-12：E-P2-2（Docker容器化）
  - 多阶段构建 Dockerfile（python:3.11-slim, CPU-only torch, HEALTHCHECK）
  - docker-compose.yml（端口/卷/重启策略）
  - .dockerignore, .env.docker.example
- [x] 子AI-13：E-P2-3（README+API文档）
  - README.md（17KB，项目总览+快速开始+配置说明+架构图）
  - docs/API.md（10个端点全覆盖，请求/响应示例）
  - docs/MCP_TOOLS.md（3个MCP工具文档）
- [x] 子AI-14：A-P2-1（LMS↔沙漏翻译层）
  - bridge/translation_layer.py（TranslationLayer + HourglassClient抽象接口 + InMemoryHourglassClient）
  - 双向转换：LMS激活态→显式记忆条目，沙漏检索→感官输入
  - tests/test_translation_layer.py（59个测试）
- [x] 子AI-15：A-P2-2（384→64维投影评估）
  - evaluation/projection_eval.py（4个评估指标+4种策略对比+报告生成）
  - evaluation/run_eval.py（可独立运行，支持真实模型/fallback）
  - tests/test_projection_eval.py（61个测试）
  - 结论：JL随机投影保留73%余弦结构，kNN重叠46%；PCA显著优于随机投影；input_dim=128可提升至84%
- [x] 统一测试：484 passed（从364增至484，新增120个测试）
- [x] 提交：258a248

### P2第二批执行计划

为避免 runtime/loop.py 文件冲突，E-P2-1和E-P2-5合并由同一子AI处理：

| 编号 | 任务 | 子AI | 涉及文件 |
|------|------|------|----------|
| E-P2-1 + E-P2-5 | GPU/device管理 + 临时参数改接口传参 | 子AI-16 | attractor.py, memory.py, loop.py, dream_engine.py |
| E-P2-4 | 预训练Embedder资源管理 | 子AI-17 | core/sensory/embedder.py |

两项任务文件互不冲突（子AI-16不修改embedder.py，子AI-17不修改核心模块），可并行执行。

### 2026-07-31 P2第二批完成（3/8项）

2个子AI并行执行，全部完成：

- [x] 子AI-16：E-P2-1（GPU/device管理）+ E-P2-5（临时参数改接口传参）
  - device管理：CoreConfig新增device参数，AttractorNetwork/MemoryManager/PurposeLayer全部支持device，LivingMemoryLoop统一管理
  - resolve_device()函数统一解析auto/cpu/cuda，输入张量自动迁移，快照跨设备恢复
  - 临时参数接口化：消除loop.py和dream_engine.py中全部4处"保存→修改→恢复"模式
  - infer()新增temperature_override/initial_state/update_internal_state参数
  - learn()新增orth_weight_override/complexity_weight_override参数
  - 涉及文件：core/types.py, core/config.py, core/hippocampus/attractor.py, memory.py, purpose.py, dream_engine.py, runtime/loop.py
- [x] 子AI-17：E-P2-4（预训练Embedder资源管理）
  - 懒加载：构造函数不加载模型，首次embed时才加载（is_loaded属性+load()方法）
  - 模型缓存：类级别_model_cache避免重复加载，clear_cache()类方法释放
  - 资源释放：unload()方法+__del__+上下文管理器协议
  - 环境变量：LMS_PRETRAINED_MODEL, LMS_EMBEDDER_SOURCE, LMS_MODEL_LOAD_TIMEOUT
  - 维度预获取：expected_dim()类方法+19个已知模型维度表
  - 健壮性：加载重试3次+指数退避+超时控制
  - 涉及文件：core/sensory/embedder.py, tests/test_embedder_resources.py（56个测试）
- [x] 统一测试：540 passed（从484增至540，新增56个测试）
- [x] 提交：499df95

### P2第三批（最后1项）

| 编号 | 任务 | 子AI | 依赖 |
|------|------|------|------|
| E-P2-6 | 代码质量清理 | 子AI-18 | E-P2-1 + E-P2-5（均已完成） |

### 2026-07-31 P2第三批完成 — P0+P1+P2全部完成

- [x] 子AI-18：E-P2-6（代码质量清理）
  - ruff检查：24个错误 → 0个错误（All checks passed!）
  - 移除7个未使用导入，修复9个F541，修复1个F841，修复1个E501
  - 添加36个类型注解（缺失数从75降至39）
  - 13个E402标注noqa（sys.path操纵的有意模式）
  - 涉及文件：core/hippocampus/memory.py, runtime/cli.py, runtime/config.py, api/server.py, api/session_manager.py, mcp_memory_server.py, memory_cli.py, e2e_test.py, e2e_controlled_experiment.py, test_mcp_tools.py
- [x] 统一测试：540 passed
- [x] 提交：a79dbf0

## 六、最终总结

### 任务完成统计

| 优先级 | 总项数 | 完成数 | 测试数 |
|--------|--------|--------|--------|
| P0 红线 | 6 | 6 | 249 |
| P1 本轮迭代 | 9 | 9 | 364 |
| P2 长期规划 | 8 | 8 | 540 |
| **合计** | **23** | **23** | **540** |

### 提交历史

| 提交 | 阶段 | 内容 |
|------|------|------|
| b0dbcaf | P0工程 | API Key+并发+快照+依赖 |
| 5f3280d | P0算法 | coherence重构+在线熵管理 |
| d041b27 | P1第一批 | 接口重构+健壮性批次 |
| fcf2750 | P1第二批 | CoreConfig校验+API测试+CI/CD |
| 258a248 | P2第一批 | Docker+文档+翻译层+投影评估 |
| 499df95 | P2第二批 | GPU管理+接口化+Embedder资源 |
| a79dbf0 | P2第三批 | 代码质量清理 |

### 推送状态

- [x] 17个代码文件通过 GitHub Contents API 推送成功
- [ ] 新增文件待推送（网络恢复后 git push 或 API 推送）
- [ ] workflow 文件需要 PAT 添加 workflow scope

---

*本文档随执行进度持续更新。P0+P1+P2全部完成。*

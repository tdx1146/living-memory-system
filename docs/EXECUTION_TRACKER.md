# 活体记忆系统 综合优化执行跟踪

> 创建时间：2026-07-31
> 最后更新：2026-07-31
> 状态：P1全部完成，P2待启动
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
| E-P2-1 | 工程 | GPU/device管理 | - | 待执行 |
| E-P2-2 | 工程 | Docker容器化 | - | 待执行 |
| E-P2-3 | 工程 | README+API文档 | - | 待执行 |
| E-P2-4 | 工程 | 预训练Embedder资源管理 | - | 待执行 |
| E-P2-5 | 工程 | 临时参数改接口传参 | 依赖E-P1-2 | 待执行 |
| E-P2-6 | 工程 | 代码质量清理 | 依赖E-P2-1/E-P2-5 | 待执行 |
| A-P2-1 | 算法 | LMS↔沙漏翻译层 | 依赖E-P1-2 | 待执行 |
| A-P2-2 | 算法 | 384→64维投影评估 | - | 待执行 |

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

---

*本文档随执行进度持续更新。每个阶段完成后在此追加日志。*

# 活体记忆系统（LMS）代码审计报告 V2

> 审计日期：2026-07-28
> 审计范围：修复后全量审计（`core/` `bridge/` `persistence/` `runtime/` `tests/` 全部源码）
> 审计基准：`AUDIT_REPORT.md` 中发现的 22 个问题（S1-S4, G1-G9, B1-B9）
> 测试状态：131 项测试全部通过，耗时 10.27 秒

---

## 一、总体评价

经过本轮修复，活体记忆系统的**集成层结构性缺口已全部闭合**。前一轮审计发现的 4 个严重问题（S1-S4）全部修复且修复质量优良，9 个一般问题（G1-G9）全部修复，9 个建议项（B1-B9）中 6 个已修复、3 个按审计要求标记为可接受。

### 测试增长

| 指标 | V1 审计时 | V2 审计时 |
|------|----------|----------|
| 测试总数 | 103 | 131 (+28) |
| 通过数 | 103 | 131 |
| 失败数 | 0 | 0 |
| 耗时 | — | 10.27s |
| 测试文件数 | 5 | 6 (+1: test_integration.py) |

新增的 `tests/test_integration.py` 包含 28 个测试用例，覆盖主循环 `process_turn()`、`save_state()`/`load_state()` 往返一致性、`query_llm()` 端到端流程、配置参数生效验证、长对话稳定性、自动快照功能。

### 修复质量总评

| 类别 | 总数 | 已修复 | 可接受 | 未修复 |
|------|------|--------|--------|--------|
| 严重 (S1-S4) | 4 | 4 | 0 | 0 |
| 一般 (G1-G9) | 9 | 9 | 0 | 0 |
| 建议 (B1-B9) | 9 | 6 | 3 | 0 |
| **合计** | **22** | **19** | **3** | **0** |

系统的核心架构在修复后依然健康：前后端分离彻底、模块依赖图为无环 DAG、FEP 推断与学习规则数学正确、目的层 precision 原生介入。修复过程中未引入架构退化或循环依赖。

---

## 二、修复验证表

### 严重问题 (S1-S4)

| 编号 | 原严重级别 | 修复状态 | 验证方式 | 备注 |
|------|-----------|---------|---------|------|
| S1 | 严重 | 已修复 | 代码审查 + 集成测试 | `runtime/loop.py` 第220行调用 `self.memory.recall(activation.state)`，第223行将结果传给 `self.decoder.decode(activation, recalled_memory=recalled)`。`query_llm()` 第271行同样接入 recall。记忆系统不再是"只写不读"。集成测试 `test_memory_recall_in_context` 验证多轮后记忆强度非零且递增。 |
| S2 | 严重 | 已修复 | 代码审查 + 集成测试 | `runtime/config.py` 的 `default_config()` 现包含全部 CoreConfig 字段（temperature, complexity_weight, orth_weight, precision_lr, max_history, habituation_rate, transfer_rate, replay_count 等）。`loop.py` 的 `__init__` 将这些参数传入 AttractorNetwork（seed, temperature）、PurposeLayer（全部9个参数）、MemoryManager（全部7个参数）。集成测试 `TestConfigWiring` 的4个用例验证参数生效。 |
| S3 | 严重 | 已修复 | 代码审查 + 集成测试 | `core/hippocampus/purpose.py` 第285-288行：`self.history.append(...)` 后执行 `if len(self.history) > self.max_history: self.history = self.history[-self.max_history:]`。`max_history` 作为构造参数暴露（默认100）。集成测试 `test_history_bounded_after_many_turns` 验证500轮后 history 恰好等于 max_history。 |
| S4 | 严重 | 已修复 | 代码审查 + 测试运行 | `tests/test_integration.py` 已存在，包含6个测试类、28个测试用例：TestProcessTurn(8)、TestSaveLoadState(5)、TestQueryLLM(4)、TestConfigWiring(4)、TestLongConversation(3)、TestAutoSnapshot(4)。全部通过。 |

### 一般问题 (G1-G9)

| 编号 | 原严重级别 | 修复状态 | 验证方式 | 备注 |
|------|-----------|---------|---------|------|
| G1 | 一般 | 已修复 | 代码审查 | `core/hippocampus/purpose.py` 第310-323行新增 `set_purpose(self, state: PurposeState)` 方法，封装 sensory_precision/history/coherence/attention 的恢复。`persistence/recovery.py` 的 `_restore_purpose()` 优先调用 `set_purpose()`，属性直写仅作兼容回退。 |
| G2 | 一般 | 已修复 | 代码审查 + 集成测试 | `purpose.py` 第94-97行新增 `encounter_count` 张量，第241-243行在 `adjust()` 中累积计数。第252-253行应用习惯化衰减：`habituation = 1.0 / (1.0 + encounter_count * habituation_rate)`，使常遇到维度的 per_dim_surprise 被抑制，实现"常遇到→低precision"。集成测试 `test_custom_habituation_rate_affects_purpose` 验证不同 habituation_rate 产生不同 precision。 |
| G3 | 一般 | 已修复 | 代码审查 | `core/hippocampus/memory.py` 第119-129行：`consolidate()` 现按 surprise 降序排序缓冲区（`sorted(self._buffer, key=lambda x: x[1], reverse=True)`），优先回放高 surprise 条目。回放权重也按 surprise 加权（`weight = replay_weight * max(surprise, 0.0)`）。不再是简单回放最近10条。 |
| G4 | 一般 | 已修复 | 代码审查 + 集成测试 | `runtime/loop.py` 第236-249行实现 auto_snapshot：当 `auto_snapshot=True` 且 `turn_count % interval == 0` 时自动调用 `save_state()`。异常被 try/except 捕获，不影响主循环。集成测试 `TestAutoSnapshot` 的4个用例验证文件创建、间隔控制、默认禁用、失败不崩溃。 |
| G5 | 一般 | 已修复 | 代码审查 | `persistence/snapshot.py` 第46-63行新增详细接口设计说明文档，解释 dict 接口是有意的架构决策（persistence 不依赖 core 具体类，符合 DAG 约束）。`recovery.py` 第8-21行同样补充了解耦说明。文档与实现对齐。 |
| G6 | 一般 | 已修复 | 代码审查 | `purpose.py` 第279-282行：`_meta_adjust()` 后重新计算 `self.attention = torch.softmax(...)` 和 `self.coherence = self._compute_coherence()`。`get_purpose()` 返回的 coherence 不再是翻转前的旧值。 |
| G7 | 一般 | 已修复 | 代码审查 + 集成测试 | `core/hippocampus/attractor.py` 第78行：`temperature` 作为构造参数 `__init__(self, num_nodes, input_dim, seed=42, temperature=0.05)`。`loop.py` 第105行从 config 读取并传入。集成测试 `test_custom_temperature_affects_attractor` 验证参数传递。 |
| G8 | 一般 | 已修复 | 代码审查 | `persistence/recovery.py` 第34-73行定义了两个 `@runtime_checkable` Protocol：`RestorableAttractor`（含 `set_landscape` 方法 + J/bias/sigma/num_nodes/input_dim 属性）和 `RestorablePurpose`（含 `set_purpose` 方法 + sensory_precision/history/coherence/attention 属性）。`recover()`、`_restore_attractor()`、`_restore_purpose()` 均使用 Protocol 类型标注。 |
| G9 | 一般 | 已修复 | 代码审查 | `core/hippocampus/memory.py` 的 `__init__` 现接受 `transfer_rate`、`replay_count`、`replay_weight`、`consolidation_decay`、`buffer_capacity` 五个参数（第37-41行）。`loop.py` 第132-141行从 config 读取并传入。魔法数字全部消除。 |

### 建议 (B1-B9)

| 编号 | 原严重级别 | 修复状态 | 验证方式 | 备注 |
|------|-----------|---------|---------|------|
| B1 | 建议 | 已修复 | 代码审查 | `purpose.py` 第192行：`_meta_adjust()` 现使用 `self.encounter_count.argmin()` 找最未被探索的维度，将其 precision 设为最大值（探索新方向），而非强化已有最高维度。实现了真正的"方向翻转"。 |
| B2 | 建议 | 可接受 | 代码审查 | `purpose.py` 第122-127行：`_compute_per_dim_surprise()` 仍使用 `|sigma_i|` 作为惊讶度代理。按审计要求标记为可接受——语义有偏差但实现简化合理，不影响系统核心功能。 |
| B3 | 建议 | 已修复 | 代码审查 | `runtime/loop.py` 第16行：`from core.types import Activation`——仅导入 `Activation`，不再导入未使用的 `PurposeState` 和 `SensoryInput`。未使用导入已清理。 |
| B4 | 建议 | 已修复 | 代码审查 | `core/hippocampus/memory.py` 第17行：`from collections import deque`，第71行：`self._buffer: deque = deque(maxlen=self._buffer_capacity)`。不再使用 `list.pop(0)`，O(1) 操作且自动处理容量限制。 |
| B5 | 建议 | 可接受 | 代码审查 | `recovery.py` 的 `import os` 已移至顶部（第23行），但 `snapshot.py`（`import os` 在 save() 内）、`config.py`（`import json` 在 load_config() 内）、`cli.py`（`import datetime` 在方法内）、`recovery.py`（`import torch.nn.functional` 在回退分支内）仍保留懒导入。按审计要求标记为可接受——风格不完全统一但不影响功能。 |
| B6 | 建议 | 可接受 | 代码审查 | `tests/test_attractor.py` 第144行：`change_short = (net.sigma - state_short).abs().mean().item()` 仍为死代码（永远为0），但该变量未被断言使用，不影响测试正确性。按审计要求标记为可接受。 |
| B7 | 建议 | 已修复 | 代码审查 | `core/hippocampus/attractor.py` 第157-164行：`infer()` 增加形状校验 `assert sensory_input.shape == (self.input_dim,)` 和 `assert precision.shape == (self.input_dim,)`，传入错误形状时给出清晰错误信息。 |
| B8 | 建议 | 已修复 | 代码审查 | `bridge/decoder.py` 第48-49行：`decode(self, activation, recalled_memory=None)` 接受可选的长时记忆检索结果。text 模式追加"长时记忆检索"段落（第118-123行），vector 模式追加 `[MEMORY:...]` 段（第180-183行）。为 None 时向后兼容。 |
| B9 | 建议 | 已修复 | 代码审查 + 全局搜索 | 全局搜索 `infer_steps` 确认所有引用统一为 `num_infer_steps`：`core/config.py`、`runtime/config.py`、`runtime/loop.py`（第200行 `config.get('num_infer_steps', 10)`）、`tests/conftest.py`、`tests/test_integration.py`。命名不一致已消除。 |

---

## 三、架构约束验证

### 1. core/ 无 IO 依赖 — 通过

对 `core/` 下全部 10 个 .py 文件执行 `import (os|json|logging|requests|sys|pathlib)` 模式搜索，结果为零匹配。

core/ 仅导入以下库：
- `torch`（张量计算）
- `dataclasses`、`typing`、`abc`（标准库类型/抽象基类）
- `re`（正则分词）
- `collections.deque`（记忆缓冲区）

无任何文件/网络/系统 IO 操作。纯计算层名副其实。

### 2. bridge/ 只依赖 core 抽象 — 通过

- `bridge/encoder.py`：导入 `core.types.SensoryInput`、`core.sensory.tokenizer.Tokenizer`（ABC）、`core.sensory.embedder.Embedder`（ABC），不导入 `SimpleTokenizer`/`SimpleEmbedder` 具体实现。依赖倒置正确。
- `bridge/decoder.py`：导入 `core.types.Activation`，不依赖 core 其他模块。
- `bridge/llm_bridge.py`：不导入 core 任何内容。`openai` 库为惰性导入（合理，因属可选依赖）。

### 3. persistence/ 通过 dict/Protocol 解耦 — 通过

- `persistence/snapshot.py`：完全不导入 core，仅操作 `dict`。`save()`/`load()` 接受/返回字典，调用方负责对象与字典的转换。
- `persistence/recovery.py`：定义 `RestorableAttractor` 和 `RestorablePurpose` 两个 Protocol（`@runtime_checkable`），通过协议接口操作 core 对象。仅导入 `core.types.PurposeState`（用于 Protocol 定义和兼容回退分支的类型构造），属类型层面的最小依赖，不构成耦合问题。

### 4. 无循环依赖 — 通过

依赖图仍为干净 DAG：
```
core（types, config, hippocampus.*, sensory.*）
  ↑
bridge（encoder, decoder, llm_bridge）—— 仅依赖 core
  ↑
persistence（snapshot, recovery）—— snapshot 不依赖 core；recovery 仅类型依赖 core.types
  ↑
runtime（loop, config, cli）—— 组合根，依赖以上全部
```

无反向依赖，无循环。

### 5. 高内聚低耦合 — 通过

- **core/ 内聚**：每个模块职责单一——attractor 负责推断与学习，purpose 负责 precision 调整，memory 负责多尺度记忆，tokenizer/embedder 负责感官编码。
- **接口清晰**：模块间通过 dataclass（`SensoryInput`、`Activation`、`PurposeState`）传递，类型一致，无隐式耦合。
- **配置统一**：`RuntimeConfig` 合并了 `CoreConfig` 全部字段，单一配置字典驱动所有组件，无配置断裂。
- **组合根唯一**：`runtime/loop.py` 是唯一"全知"模块，正确串联所有组件。

---

## 四、新发现问题

集成测试在验证修复过程中发现了以下 4 个新问题。这些问题不影响当前测试通过，但影响系统在跨会话场景下的行为正确性。

### N1. Memory 潜变量未持久化 — 一般

**位置**：`runtime/loop.py` `save_state()` / `load_state()`

**问题**：`save_state()` 仅保存 attractor 景观（J, bias, sigma）和 purpose 状态（precision, history, coherence），不保存 `MemoryManager` 的 `short_term_latent` 和 `long_term_latent`。`MemoryManager` 已实现 `get_state()`/`set_state()` 方法，但未被 `loop.py` 调用。

系统重启后，长时记忆潜变量归零，`recall()` 返回全零向量，长时记忆对输出的贡献为零——直到新会话重新积累。

**严重级别评估**：**一般**。这不影响系统的核心计算引擎（J 矩阵携带了学习到的吸引子结构，重启后推断仍正常），但影响"身份连续性"——长时记忆是系统"身份"的重要组成部分，跨会话丢失削弱了"活体记忆"的设计意图。

**建议**：应纳入快照格式。在 `save_state()` 中增加 `memory_state = self.memory.get_state()`，在快照中新增 `memory` 字段；`load_state()` 中调用 `self.memory.set_state()`。需同步更新 `Snapshot` 的格式版本号和 `Recovery` 的验证逻辑。建议作为下一迭代的高优先级修复项。

**当前状态**：集成测试 `test_memory_latents_not_persisted` 已明确记录此已知限制，以测试确认而非掩盖。

---

### N2. Tokenizer 词表未持久化 — 一般

**位置**：`core/sensory/tokenizer.py` `SimpleTokenizer` / `runtime/loop.py`

**问题**：`SimpleTokenizer` 动态构建词表（`_vocab` 字典），首次遇到的 token 自动分配新 ID。两个独立的 tokenizer 实例对相同文本会产生不同的 token IDs，进而通过 embedder 产生不同的感官向量。

系统重启后，新创建的 tokenizer 词表为空，相同文本的编码结果与重启前不同，导致感官输入不一致——即使 J 矩阵已恢复，相同的用户输入也会产生不同的激活态。

**严重级别评估**：**一般**。影响跨会话的输入一致性。在当前 `SimpleEmbedder`（固定种子初始化）下，不同 token IDs 选取 embedding 矩阵的不同行，感官向量完全不同。但此问题仅影响 `SimpleTokenizer`；若未来替换为 BPE/预训练分词器（词表固定），问题自动消失。

**建议**：
- 短期：将 tokenizer 词表纳入快照（`_vocab` 和 `_next_id`），或在 `save_state`/`load_state` 中共享 tokenizer 实例。
- 长期：替换为词表固定的预训练分词器，从根本上消除问题。
- 当前集成测试 `test_roundtrip_behavior_consistency` 通过共享 tokenizer 规避此问题，并在注释中明确记录。

---

### N3. encounter_count 未持久化 — 一般

**位置**：`core/hippocampus/purpose.py` / `runtime/loop.py` `save_state()`

**问题**：`PurposeLayer` 新增的 `encounter_count`（G2 修复引入的习惯化计数器）未被 `save_state()` 持久化。快照仅保存 precision、history、coherence。系统重启后 `encounter_count` 归零，习惯化历史丢失，目的层对"已熟悉内容"的关注度无法延续。

**严重级别评估**：**一般**。影响目的层的连续性。precision 和 history 已恢复，但习惯化机制重置后，已熟悉维度的 precision 会在后续对话中重新被推高（因 encounter_count 归零，habituation 衰减消失），与重启前的行为不一致。

**建议**：将 `encounter_count` 纳入 PurposeState 或快照的 purpose 字典中。可扩展 `PurposeState` dataclass 增加 `encounter_count` 字段，或在 `save_state()` 的 `purpose_dict` 中增加该键，并在 `set_purpose()` 中恢复。

**当前状态**：集成测试 `test_memory_latents_not_persisted` 已验证 encounter_count 重启后归零。

---

### N4. Habituation 阈值与 temperature 耦合 — 一般

**位置**：`core/hippocampus/purpose.py` `adjust()` 第242行

**问题**：习惯化计数器的更新条件是 `activation.state[:self.input_dim].abs() > 0.3`（硬编码阈值 0.3）。当 `temperature=0`（确定性推断）时，Langevin 扩散项被移除，激活值普遍偏小，可能无法超过 0.3 阈值，导致 `encounter_count` 始终为零，习惯化机制静默失效。

集成测试 `test_custom_habituation_rate_affects_purpose` 的注释明确记录了此耦合：需要 `temperature > 0` 才能使激活值超过阈值。

**严重级别评估**：**一般**。这是设计层面的耦合问题，非实现 bug。temperature=0 是合法配置（用于确定性测试/推理），但在此配置下习惯化机制完全失效且无任何警告。

**建议**：
- 方案A（解耦）：将习惯化阈值从硬编码 0.3 改为基于激活值分布的自适应阈值（如取当前激活值的均值或中位数），使其不依赖绝对幅度。
- 方案B（可配置）：将阈值暴露为构造参数 `habituation_threshold`，允许用户根据 temperature 调整。
- 方案C（文档化）：在文档中明确说明 temperature=0 时习惯化机制不生效，并在 `adjust()` 中增加日志警告。
- 推荐方案B，兼顾灵活性与简洁性。

---

## 五、新发现问题汇总表

| 编号 | 严重级别 | 模块 | 问题摘要 | 建议处理 |
|------|---------|------|---------|---------|
| N1 | 一般 | runtime/loop + memory | Memory 潜变量（short/long_term_latent）未持久化，重启后长时记忆归零 | 下一迭代高优先级修复 |
| N2 | 一般 | core/sensory/tokenizer | Tokenizer 词表未持久化，跨会话 token IDs 不一致 | 短期纳入快照，长期换预训练分词器 |
| N3 | 一般 | core/hippocampus/purpose | encounter_count 未持久化，习惯化历史重启后归零 | 纳入 PurposeState 或快照 |
| N4 | 一般 | core/hippocampus/purpose | 习惯化阈值(0.3)与 temperature 耦合，temperature=0 时静默失效 | 阈值可配置或自适应 |

---

## 六、结论

### 修复成果

活体记忆系统在本次修复中取得了**显著进展**：

1. **集成层缺口全部闭合**：S1（记忆只写不读）通过 recall 接入解码路径修复，S2（配置未接线）通过统一配置字典修复，S3（历史无界增长）通过 max_history 裁剪修复，S4（缺少集成测试）通过 28 个集成测试用例修复。

2. **设计意图得以实现**：G2（习惯化机制）使"常遇到→低precision"的设计意图落地，G3（重要性加权回放）使记忆巩固按 surprise 优先回放，B1（方向翻转）使元目的真正切换到新方向而非强化旧方向。

3. **代码质量提升**：G7/G9（配置提取）消除了硬编码，G8（Protocol 类型标注）提升了类型安全，B4（deque）优化了性能，B7（形状校验）改善了错误诊断。

4. **架构约束保持**：修复过程中未引入架构退化——core/ 仍无 IO 依赖、bridge/ 仍仅依赖 core 抽象、persistence/ 仍通过 dict/Protocol 解耦、依赖图仍为无环 DAG。

### 残留风险

修复后的残留风险集中在**跨会话状态持久化的不完整性**（N1-N3）。当前 `save_state()`/`load_state()` 仅保存 attractor 和 purpose 的核心状态，Memory 潜变量、tokenizer 词表、encounter_count 均未持久化。这意味着系统重启后：
- 推断能力保留（J 矩阵已保存）
- 短期/长期记忆丢失（潜变量未保存）
- 习惯化历史丢失（encounter_count 未保存）
- 输入编码可能不一致（tokenizer 词表未保存）

这些不影响单会话内的系统行为（131 项测试全部通过），但影响跨会话的"身份连续性"。建议作为下一迭代的优先修复项。

### 最终评级

| 维度 | V1 评级 | V2 评级 |
|------|--------|--------|
| 核心计算引擎 | 可靠 | 可靠（无变化） |
| 集成层完整性 | 有结构性缺口 | 缺口已闭合 |
| 配置体系 | 断裂 | 完整接线 |
| 测试覆盖 | 单元测试充分，集成测试缺失 | 单元+集成测试均充分 |
| 架构约束 | 健康 | 健康（无退化） |
| 跨会话连续性 | 未评估 | 存在已知限制（N1-N4） |

**系统已具备进入单会话生产使用的条件**。跨会话连续性问题（N1-N4）建议在进入多会话/长期运行场景前修复。

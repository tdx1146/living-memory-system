# 活体记忆系统（LMS）代码审计报告

> 审计日期：2026-07-28
> 审计范围：`core/` `bridge/` `persistence/` `runtime/` `tests/` 全部源码
> 审计基准：`docs/ARCHITECTURE.md` v0.1
> 测试状态：103 项测试全部通过

---

## 一、总体评价

系统的核心架构是**健康**的：前后端分离彻底、模块依赖图为无环 DAG、FEP 推断与学习规则的数学方向正确、目的层 precision 确实从推断和自由能两个层面原生介入（非后加）。103 项单元测试全部通过，代码风格统一，文档注释详尽。

但在**集成层面**存在若干结构性缺口：多尺度记忆的 `recall()` 从未被主循环调用、`CoreConfig` 完全未被接线（FEP 参数无法配置）、目的层历史无界增长、缺少集成测试。这些问题不影响当前单元测试通过，但会影响系统在真实长对话场景下的行为正确性。

| 严重程度 | 数量 |
|---------|------|
| 致命    | 0    |
| 严重    | 4    |
| 一般    | 9    |
| 建议    | 9    |

---

## 二、通过的检查项

### 1. 模块独立性 — 通过

- **core/ 完全无 IO 依赖**：`core/` 下所有文件仅导入 `torch`、`dataclasses`、`typing`、`abc`、`re` 等标准库，无任何文件/网络/系统 IO。纯计算层名副其实。
- **bridge/ 仅依赖 core 的抽象接口**：`encoder.py` 导入 `Tokenizer`（ABC）和 `Embedder`（ABC），不导入 `SimpleTokenizer`/`SimpleEmbedder` 具体实现；`decoder.py` 仅依赖 `Activation` 类型；`llm_bridge.py` 不依赖 core 任何内容。实现了真正的依赖倒置。
- **persistence/ 与 core 解耦**：`snapshot.py` 完全不导入 core，仅操作 `dict`；`recovery.py` 仅导入 `core.types.PurposeState`（用于兼容分支）。
- **runtime/ 正确串联所有模块**：`loop.py` 作为组合根，依赖 core/bridge/persistence，是唯一允许"全知"的模块。
- **无循环依赖**：依赖图为 `core ← bridge ← runtime`、`core ← persistence ← runtime` 的干净 DAG。

### 2. 接口咬合 — 基本通过

- **核心数据类型一致**：`SensoryInput`、`Activation`、`PurposeState` 三个 dataclass 在各模块间一致传递，字段定义与架构文档 5.1 节吻合。
- **数据流顺畅**：`loop.py` 的 `process_turn()` 忠实实现了 `编码→推断→学习→目的→巩固→解码` 的完整闭环，与架构文档第四节一致。
- **Activation 全链路贯通**：`infer()` 产出 → `learn()` 消费 → `adjust()` 消费 → `update()` 消费 → `decode()` 消费，类型一致，无中途转换损失。

### 3. FEP 规则忠实度 — 通过（含合理创新）

- **Langevin 激活函数正确**：`langevin(b) = 1/tanh(b) - 1/b = coth(b) - 1/b`，数学正确。使用 `1/tanh` 替代 `cosh/sinh` 避免大 b 溢出，小 b 区域用泰勒展开 `b/3` 处理除零——数值稳定性处理到位。
- **学习规则方向正确**：`ΔJ = Hebbian(σ) - complexity_grad`，其中 `Hebbian = -∂F_accuracy/∂J`（符号正确），`weight_decay = ∂F_complexity/∂J`（L2 正则梯度正确）。对称化、对角线置零均正确实现。
- **感官 clamping 与文档伪代码一致**：`b_q[:input_dim] += sensory_input * precision`，与架构文档第七节伪代码完全吻合。
- **未偷工减料**：实际比基础 FEP 多了 Langevin 扩散项（温度噪声），增强了打破对称性的能力，是合理的增强而非简化。

### 4. 目的层设计 — 通过（原生介入）

- **precision 真正传递给 `infer()`**：`loop.py` 中 `precision = self.purpose.get_precision()` → `attractor.infer(sensory_input.vector, precision, ...)`，precision 在 `infer()` 中既影响感官证据注入（`b_q += sensory * precision`），又影响自由能计算（`accuracy = 0.5 * Σ precision * error²`）。**目的层是原生的，不是后加的。**
- **adjust 基于 surprise**：全局 surprise 通过 sigmoid 缩放有效学习率，per-dimension surprise 通过激活强度提取。存在真实的 `infer → activation → adjust → precision → infer` 反馈闭环。
- **元目的翻转机制可实现**：coherence 低于阈值时触发，测试验证可触发且 flip_count 递增。

### 5. 前后端分离 — 通过

- **core/ 完全无 IO**：已验证，零文件/网络操作。
- **LLM 交互完全在 bridge/**：`llm_bridge.py` 封装所有 API 调用，含重试/超时/指数退避。
- **持久化完全在 persistence/**：`snapshot.py`/`recovery.py` 封装所有文件 IO，core 和 bridge 无文件操作。

### 6. 代码质量 — 基本通过

- **类型标注**：core 层和 bridge 层标注完整。
- **边界情况**：空输入（tokenizer/embedder/encoder/decoder 均处理）、零向量（entropy/coherence 有 `< 1e-8` 守卫）、数值溢出（langevin 用 1/tanh）均处理到位。
- **测试覆盖**：5 个测试文件、103 项测试，覆盖 attractor/purpose/memory/bridge/persistence 各模块的核心行为。

---

## 三、发现的问题

### 严重 (Serious)

#### S1. MemoryManager.recall() 从未被主循环调用 — 记忆系统"只写不读"

**位置**：`runtime/loop.py` `process_turn()` / `core/hippocampus/memory.py`

**问题**：`MemoryManager` 实现了 `update()`、`consolidate()`、`recall()` 三个方法，主循环 `process_turn()` 调用了 `update()` 和 `consolidate()`，但**从未调用 `recall()`**。解码器 `decoder.decode(activation)` 只看到当前激活态，看不到长时记忆潜变量。

这意味着：短时/长时记忆潜变量被持续更新和巩固，但这些信息**永远不会进入输出路径**。多尺度记忆系统实际上是一个"只写不读"的组件。

```python
# loop.py process_turn() 当前逻辑：
self.memory.update(activation)          # 写入
if ...:
    self.memory.consolidate()           # 巩固
memory_context = self.decoder.decode(activation)  # ← 只解码当前 activation，未 recall
# 缺失：recalled = self.memory.recall(activation.state) 并融入 context
```

**影响**：长时记忆对 LLM 的 context 贡献为零，"身份是吸引子景观"的设计意图在输出端无法体现。

**修复建议**：在解码前调用 `recall()`，将检索到的记忆与当前激活态融合后解码：
```python
recalled = self.memory.recall(activation.state)
# 方案A：让 decoder 接受额外的 memory 参数
memory_context = self.decoder.decode(activation, recalled)
# 方案B：将 recalled 注入 activation 的非感官节点后解码
```
同时需在 `Decoder` 中增加对记忆检索结果的处理逻辑。

---

#### S2. CoreConfig 完全未接线 — FEP 参数无法通过配置调整

**位置**：`core/config.py` / `core/hippocampus/attractor.py` / `runtime/loop.py`

**问题**：`CoreConfig` dataclass 定义了 16 个配置字段（`complexity_weight`、`orth_weight`、`precision_lr`、`coherence_threshold`、`num_infer_steps` 等），但**从未被任何代码实例化或使用**。全局搜索 `CoreConfig` 仅出现在 `core/config.py` 定义处和 `core/__init__.py` 导出处。

各模块实际使用硬编码默认值：
- `AttractorNetwork.__init__`：硬编码 `complexity_weight=0.01`、`orth_weight=0.5`、`temperature=0.05`
- `PurposeLayer.__init__`：参数仅来自构造函数直接传入，`loop.py` 调用时只传 `input_dim`
- `MemoryManager.__init__`：`loop.py` 调用时只传 `num_nodes`，`short_term_decay`/`long_term_decay` 用默认值

`runtime/config.py` 的 `default_config()` 也不包含任何 FEP 参数。唯一能调整 `orth_weight` 的方式是构造后直接修改属性（如测试中的 `net.orth_weight = 1.5`），这不是正式的配置路径。

**影响**：调参只能改代码，无法通过配置文件/环境变量调整 FEP 学习行为。`CoreConfig` 是误导性的死代码。

**修复建议**：
1. 让 `LivingMemoryLoop.__init__` 接受 `CoreConfig`（或合并后的统一配置），将其字段传入各核心组件构造函数。
2. 或删除 `CoreConfig`，将参数合并进 `RuntimeConfig`。
3. `AttractorNetwork.__init__` 应接受 `complexity_weight`、`orth_weight`、`temperature` 参数而非硬编码。

---

#### S3. PurposeLayer.history 无界增长 — 长对话内存泄漏

**位置**：`core/hippocampus/purpose.py` `adjust()`

**问题**：每次 `adjust()` 调用执行 `self.history.append(self.sensory_precision.clone())`，但**从未裁剪**。`_compute_coherence()` 和 `_meta_adjust()` 虽然用 `min(meta_window, len(history))` 限制读取窗口，但 history 列表本身只增不减。

```python
# adjust() 中：
self.history.append(self.sensory_precision.clone())  # 永远只 append，不 trim
```

在万轮级别的长对话中，history 会积累上万个 `[input_dim]` 张量，造成持续内存增长。且 `get_purpose()` 会深拷贝整个 history 列表返回，进一步放大开销。

**影响**：长时间运行的 session 内存持续增长，最终可能 OOM。快照保存也会因 history 过大而膨胀。

**修复建议**：在 append 后裁剪到合理上限（如 `max_history = meta_window * 5`）：
```python
MAX_HISTORY = 200  # 或从配置读取
self.history.append(self.sensory_precision.clone())
if len(self.history) > MAX_HISTORY:
    self.history = self.history[-MAX_HISTORY:]
```

---

#### S4. 缺少集成测试 — 主循环和 CLI 完全未测试

**位置**：`tests/`

**问题**：架构文档里程碑 6 列出 `test_integration.py`，但该文件不存在。当前测试全部是单元测试（针对单个模块），`runtime/loop.py`（核心组合根，串联所有模块的主循环）和 `runtime/cli.py` **没有任何测试覆盖**。

`process_turn()` 的完整数据流（编码→推断→学习→目的→巩固→解码）、`save_state()`/`load_state()` 的往返一致性、`query_llm()` 的 LLM 交互+记忆反馈——这些跨模块集成路径均未被验证。

**影响**：S1（recall 未调用）这类集成层问题正是单元测试无法发现、集成测试本应捕获的典型缺陷。

**修复建议**：新增 `tests/test_integration.py`，至少覆盖：
- `process_turn()` 完整循环的多轮执行（验证无异常、状态正确演化）
- `save_state()` → `load_state()` 往返一致性
- `query_llm()` with mock LLM 的端到端流程
- 长对话场景（验证 S3 的 history 增长问题）

---

### 一般 (General)

#### G1. PurposeLayer 缺少 set_purpose() — 恢复路径依赖脆弱的属性直写

**位置**：`core/hippocampus/purpose.py` / `persistence/recovery.py`

**问题**：`PurposeLayer` 有 `get_purpose()` 但没有对应的 `set_purpose()`。`recovery.py` 的 `_restore_purpose()` 检查 `hasattr(purpose, 'set_purpose')`，当前实现永远走 False 分支，直接操作内部属性：

```python
# recovery.py 实际执行路径（set_purpose 分支为死代码）：
purpose.sensory_precision = precision.clone()
purpose.history = [...]
purpose.coherence = coherence
purpose.attention = F.softmax(purpose.sensory_precision, dim=0)
```

这依赖 `sensory_precision`、`history`、`coherence`、`attention` 等内部属性名，一旦 PurposeLayer 内部重构，恢复静默失败。同时 `recovery.py` 因此导入 `PurposeState` 和 `torch.nn.functional`，增加了不必要的耦合。

**修复建议**：为 `PurposeLayer` 增加 `set_purpose(self, state: PurposeState)` 方法，封装内部属性恢复逻辑，使 recovery 通过公开接口操作。

---

#### G2. 目的层"常遇到→低 precision"设计意图与实现矛盾

**位置**：`core/hippocampus/purpose.py` `_compute_per_dim_surprise()` / 架构文档第六节

**问题**：架构文档 6.2 节明确表述：
> 常遇到的 → 低 precision（已熟悉），意外的 → 高 precision（值得关注）

但实现中 `per_dim_surprise = precision_min + |σ_i| * scale`，即**激活强度高的维度获得高 precision**。如果一个维度被频繁激活（常遇到），它的 `|σ_i|` 会持续偏高，precision 会被持续推高——与"常遇到→低 precision"的设计意图**相反**。

实现中没有频率跟踪或习惯化（habituation）机制。precision 只会向当前激活强度靠拢，不会因"熟悉"而下降（除非网络学会预测该维度使 `|σ_i|` 间接下降，但这不受保证）。

**影响**：目的层无法实现"对已熟悉内容减少关注"的行为，precision 会单调集中于高频维度。

**修复建议**：引入习惯化机制，例如维护 per-dim 累积激活计数，对高频维度施加 precision 衰减：
```python
# 在 adjust 中维护 encounter_count
self.encounter_count += (activation.state[:self.input_dim].abs() > threshold).float()
habituation = 1.0 / (1.0 + self.encounter_count * decay)
per_dim_surprise = per_dim_surprise * habituation
```

---

#### G3. memory.consolidate() 回放最近而非重要经验

**位置**：`core/hippocampus/memory.py` `consolidate()`

**问题**：文档注释写"回放重要经验"，但实现是 `self._buffer[-replay_count:]`——回放**最近的 10 条**，无重要性加权：

```python
replay_count = min(10, len(self._buffer))
for state in self._buffer[-replay_count:]:  # 按时间而非重要性
    self.long_term_latent += replay_weight * state
```

缓冲区也未记录各条目的 surprise/重要性，无法做加权回放。

**修复建议**：在 `_buffer` 中同时存储 `surprise` 值（来自 `Activation.surprise`），回放时按 surprise 加权或优先回放高 surprise 条目。

---

#### G4. auto_snapshot 配置已定义但从未实现

**位置**：`runtime/config.py` / `runtime/loop.py`

**问题**：`RuntimeConfig` 定义了 `auto_snapshot: bool` 和 `auto_snapshot_interval: int = 50`，`default_config()` 也输出了这两个键，但 `loop.py` 中**没有任何代码读取或使用它们**。自动快照功能未实现。

**修复建议**：在 `process_turn()` 末尾增加：
```python
if self.config.get('auto_snapshot') and \
   self.turn_count % self.config.get('auto_snapshot_interval', 50) == 0:
    self.save_state(self.config.get('snapshot_path'))
```
或若暂不实现，则从配置中移除以避免误导。

---

#### G5. Snapshot.save 接口与架构文档不匹配

**位置**：`persistence/snapshot.py` / 架构文档 5.5 节

**问题**：架构文档定义：
```python
def save(self, path: str, attractor: AttractorNetwork, purpose: PurposeLayer) -> None
```
实际实现：
```python
def save(self, path: str, attractor_landscape: dict, purpose_state: dict) -> None
```

实现接受 `dict` 而非对象。这实际上**更好地实现了解耦**（persistence 不依赖 core 类），但与文档不符，且迫使 `loop.py` 写胶水代码手动构造 dict：

```python
# loop.py save_state() 中的胶水代码：
purpose = self.purpose.get_purpose()
purpose_dict = {
    'precision': purpose.precision,
    'history': purpose.history,
    'coherence': purpose.coherence,
}
self.snapshot.save(path, landscape, purpose_dict)
```

**修复建议**：更新架构文档以反映实际接口（推荐保留当前 dict 接口），或在 PurposeState/landscape 上增加 `to_dict()` 方法消除胶水代码。

---

#### G6. 元目的翻转后 coherence 未重算

**位置**：`core/hippocampus/purpose.py` `adjust()`

**问题**：`adjust()` 中 Layer 2 计算 coherence 后，Layer 3 的 `_meta_adjust()` 大幅修改了 `sensory_precision`，但**未重新计算 coherence**。`get_purpose()` 返回的 coherence 是翻转前的旧值。

```python
self.coherence = self._compute_coherence()  # Layer 2
# ...
if self.coherence < threshold:
    self._meta_adjust()  # 修改了 precision，但未更新 coherence
    self.attention = torch.softmax(...)      # 只重算了 attention
# coherence 仍是旧值
```

**修复建议**：在 `_meta_adjust()` 后重新计算 coherence，或在 `get_purpose()` 时按需重算。

---

#### G7. AttractorNetwork.temperature 硬编码且不可配置

**位置**：`core/hippocampus/attractor.py` `__init__`

**问题**：`self.temperature: float = 0.05` 硬编码在构造函数中，不在 `CoreConfig` 中，也不作为构造参数暴露。温度是 Langevin 动力学的关键超参（控制探索 vs 收敛），无法通过配置调整。

```python
self.temperature: float = 0.05  # 硬编码，无法从外部配置
```

**修复建议**：将 `temperature` 加入构造参数和配置：`__init__(self, num_nodes, input_dim, seed=42, temperature=0.05)`。

---

#### G8. recovery.py 缺少类型标注

**位置**：`persistence/recovery.py`

**问题**：`recover(self, path: str, attractor, purpose) -> bool` 中 `attractor` 和 `purpose` 参数无类型标注。`_restore_attractor(self, attractor, landscape)` 和 `_restore_purpose(self, purpose, purpose_state)` 同样缺失。

**修复建议**：使用 `Protocol` 或直接标注为 `AttractorNetwork`/`PurposeLayer`（若接受耦合），或定义 `Restorable` Protocol。

---

#### G9. memory.py 大量魔法数字未提取为配置

**位置**：`core/hippocampus/memory.py`

**问题**：`consolidate()` 中有 5 个硬编码常量：

```python
transfer_rate = 0.1        # 短→长迁移率
replay_count = min(10, ...) # 回放条数
replay_weight = 0.01       # 回放权重
self.short_term_latent * 0.5  # 巩固后短时衰减
self._buffer_capacity = 100   # 缓冲区容量
```

这些均为魔法数字，不可配置，且与 `CoreConfig` 中已有的 `short_term_decay`/`long_term_decay` 风格不一致。

**修复建议**：将这些参数加入 `MemoryManager.__init__` 构造参数，并最终纳入配置体系。

---

### 建议 (Suggestions)

#### B1. "翻转"（flip）命名误导

**位置**：`core/hippocampus/purpose.py` `_meta_adjust()`

`_meta_adjust()` 在历史中找到平均 precision **最高**的维度并强化它。这是"加倍下注"而非"翻转"——真正的翻转应切换到一个**不同**的方向（如最未被探索的维度，或随机未饱和维度）。当前命名为"翻转"（meta-flip）但行为是"强化"（reinforce）。建议要么改名为 `_meta_reinforce`，要么实现真正的方向翻转逻辑。

#### B2. per-dim surprise 使用 |σ_i| 代理而非真实预测误差

**位置**：`core/hippocampus/purpose.py` `_compute_per_dim_surprise()`

当前用感官节点激活强度 `|σ_i|` 作为"惊讶度代理"。真正的 per-dim surprise 应为 `precision_i * (σ_i - sensory_i)²`（预测误差加权 precision）。使用代理简化了实现，但语义上有偏差。建议在 `Activation` 中携带 per-dim surprise 向量，由 `attractor.infer()` 计算后传递。

#### B3. runtime/loop.py 导入 PurposeState 但未使用

**位置**：`runtime/loop.py` 第 13 行

`from core.types import SensoryInput, Activation, PurposeState` 中 `PurposeState` 从未被 loop.py 直接引用。`SensoryInput` 同样仅用于类型注释中未出现的场景。建议清理未使用导入。

#### B4. memory.py 使用 list.pop(0) 而非 deque

**位置**：`core/hippocampus/memory.py` `update()`

`self._buffer.pop(0)` 是 O(n) 操作。虽然缓冲区上限 100 影响不大，但建议改用 `collections.deque(maxlen=100)`，自动处理容量限制且 O(1) 操作。

#### B5. 不一致的懒导入风格

**位置**：多处

部分文件在函数内部懒导入（`snapshot.py` 的 `import os`、`recovery.py` 的 `import torch.nn.functional as F`、`cli.py` 的 `import datetime`、`config.py` 的 `import json`），其他文件在顶部导入。风格不统一。建议统一为顶部导入（除非有明确的循环依赖或可选依赖原因，如 `llm_bridge.py` 对 `openai` 的懒导入是合理的）。

#### B6. 测试 test_infer_converges_with_more_steps 存在无效断言

**位置**：`tests/test_attractor.py` `test_infer_converges_with_more_steps`

```python
state_short = net.sigma.clone()
change_short = (net.sigma - state_short).abs().mean().item()  # 永远为 0
```

`change_short` 计算 `net.sigma` 与自身克隆的差，永远为 0，变量无意义。测试虽然通过（仅断言 `change_long < 0.5`），但 `change_short` 部分是死代码。建议修正为先保存状态、再跑更多步、再比较。

#### B7. attractor.infer() 缺少输入形状校验

**位置**：`core/hippocampus/attractor.py` `infer()`

`infer()` 不校验 `sensory_input` 和 `precision` 的形状是否为 `[input_dim]`。传入错误形状会产生晦涩的 PyTorch 广播错误。建议增加：
```python
assert sensory_input.shape == (self.input_dim,), ...
assert precision.shape == (self.input_dim,), ...
```

#### B8. Decoder 只看到当前激活态，看不到记忆检索结果

**位置**：`bridge/decoder.py` / 架构文档第四节数据流

这与 S1 相关但属于架构层面：架构文档的数据流图中，decoder 的输入是 `[激活态]`，未体现 memory.recall 的参与。即使修复 S1（在 loop 中调用 recall），decoder 的接口 `decode(self, activation: Activation) -> str` 也需要扩展以接受记忆检索结果。建议在架构层面明确记忆检索如何融入解码路径。

#### B9. 配置命名不一致：num_infer_steps vs infer_steps

**位置**：`core/config.py` (`num_infer_steps`) / `runtime/loop.py` (`config.get('infer_steps', 10)`)

`CoreConfig` 字段名为 `num_infer_steps`，但 `loop.py` 读取的配置键为 `infer_steps`。即使 CoreConfig 被接线，键名不匹配也会导致配置失效。建议统一命名。

---

## 四、问题汇总表

| 编号 | 严重程度 | 模块 | 问题摘要 |
|------|---------|------|---------|
| S1 | 严重 | runtime/loop + memory | recall() 从未被调用，记忆系统只写不读 |
| S2 | 严重 | core/config + 全局 | CoreConfig 完全未接线，FEP 参数不可配置 |
| S3 | 严重 | core/hippocampus/purpose | history 无界增长，长对话内存泄漏 |
| S4 | 严重 | tests | 缺少集成测试，主循环和 CLI 未测试 |
| G1 | 一般 | core/hippocampus/purpose + persistence | 缺少 set_purpose()，恢复依赖属性直写 |
| G2 | 一般 | core/hippocampus/purpose | "常遇到→低precision"意图与实现矛盾 |
| G3 | 一般 | core/hippocampus/memory | consolidate 回放最近而非重要经验 |
| G4 | 一般 | runtime/config + loop | auto_snapshot 配置已定义未实现 |
| G5 | 一般 | persistence/snapshot | save() 接口与架构文档不匹配 |
| G6 | 一般 | core/hippocampus/purpose | 翻转后 coherence 未重算 |
| G7 | 一般 | core/hippocampus/attractor | temperature 硬编码不可配置 |
| G8 | 一般 | persistence/recovery | recover() 缺少类型标注 |
| G9 | 一般 | core/hippocampus/memory | consolidate 中 5 个魔法数字 |
| B1 | 建议 | core/hippocampus/purpose | "翻转"命名误导，实为强化 |
| B2 | 建议 | core/hippocampus/purpose | per-dim surprise 用代理而非真实误差 |
| B3 | 建议 | runtime/loop | 导入 PurposeState 未使用 |
| B4 | 建议 | core/hippocampus/memory | list.pop(0) 应改用 deque |
| B5 | 建议 | 全局 | 懒导入风格不一致 |
| B6 | 建议 | tests/test_attractor | change_short 断言无效（永远为 0） |
| B7 | 建议 | core/hippocampus/attractor | infer() 缺少形状校验 |
| B8 | 建议 | bridge/decoder + 架构 | Decoder 接口需扩展以接受记忆检索 |
| B9 | 建议 | core/config + runtime/loop | num_infer_steps vs infer_steps 命名不一致 |

---

## 五、优先修复建议

**第一优先级（影响系统行为正确性）**：
1. **S1**：在 `process_turn()` 中接入 `memory.recall()`，让长时记忆参与解码
2. **S2**：接线 `CoreConfig`（或合并入 `RuntimeConfig`），让 FEP 参数可配置
3. **S3**：为 `PurposeLayer.history` 增加上限裁剪
4. **S4**：补充 `test_integration.py`，覆盖主循环

**第二优先级（影响设计意图实现）**：
5. **G1**：增加 `PurposeLayer.set_purpose()`
6. **G2**：引入习惯化机制实现"常遇到→低precision"
7. **G6**：翻转后重算 coherence

**第三优先级（代码质量）**：
8. **G7/G9**：提取 temperature 和 memory 魔法数字为配置
9. **G8**：补充类型标注
10. **B1-B9**：按建议逐项清理

---

## 六、结论

活体记忆系统的**核心计算引擎是可靠的**：FEP 推断规则数学正确、Langevin 函数数值稳定、学习规则方向无误、目的层 precision 原生介入推断与自由能、前后端分离彻底、无循环依赖。单元测试覆盖充分且全部通过。

主要风险集中在**集成层**：记忆检索未被接入输出路径（S1）、配置体系断裂（S2）、历史无界增长（S3）。这些问题在单元测试中不可见，但会在真实长对话场景中暴露。建议优先修复 S1-S4 后再进入生产使用。

# Phase 0 自指回路（Self-Referential Loop MVP）代码审计报告

> 审计日期：2026-08-01
> 审计范围：5 个文件（self_referential.py / memory.py / loop.py / snapshot.py / test_self_referential.py）
> 审计依据：SELF_REF_INTEGRATED_DESIGN.md v2.0 + ARCHITECTURE.md v0.1

---

## 一、审计总结

**结论：通过（PASS）**

Phase 0 MVP 代码质量优良，架构约束严格遵循，设计符合性高，测试覆盖完整，回归安全有保障。发现 0 个 Critical 级、0 个 High 级问题，6 个 Low 级问题及 3 个 Info 级观察项，均不影响 Phase 0 验收标准。

---

## 二、逐项检查结果

### 1. 架构约束

| 检查项 | 结果 | 说明 |
|--------|------|------|
| core/ 层无 IO 依赖 | ✅ | `self_referential.py` 仅导入 `logging`、`typing`、`torch`、`core.types`（Activation, resolve_device）。无 `os`/`pathlib`/`open()`/`requests`/`socket` 等任何文件或网络 IO。完全符合架构文档"核心层（纯计算，无IO依赖）"约束。 |
| 模块独立性（依赖注入） | ✅ | `SelfReferentialLoop.__init__` 接收 `encoder`、`tokenizer`、`embedder` 三个组件引用，不 `new` 任何新实例。`loop.py` 第 210-214 行接线时传入已初始化的 `self.encoder`/`self.tokenizer`/`self.embedder`。 |
| 高内聚低耦合 | ✅ | 自指逻辑全部集中在 `self_referential.py`（468 行）。`loop.py` 仅有 2 个薄插入点：插入点 A（第 254-266 行，6 行核心逻辑）和插入点 B（第 388-389 行，1 行调用）。`__init__` 接线 7 行，`save_state`/`load_state` 各 4 行。总计 loop.py 改动约 22 行，均为守卫包裹的薄层调用。 |
| 无循环依赖（DAG） | ✅ | `self_referential.py` 导入链：`core.types` → `torch` → `logging`。不反向导入 `runtime.loop` 或 `bridge`。Grep 确认无 `from runtime` / `import loop` 语句。依赖方向为 `runtime → core`，符合 DAG 约束 `core ← persistence ← runtime`。 |
| 默认关闭 | ✅ | `loop.py` 第 208 行：`config.get('self_ref_enabled', False)`，默认 False。第 206 行 `self.self_ref = None`。所有自指代码块均在 `if self.self_ref is not None:` 守卫内（第 254、388、513、564、605 行共 5 处守卫）。 |

### 2. 代码质量

| 检查项 | 结果 | 说明 |
|--------|------|------|
| type hints 完整 | ✅ | 所有公开方法均有完整类型标注：`distill(memory_context: str) -> str`、`generate_echo(...) -> Optional[dict]`、`observe(memory_context: str, activation: Activation) -> None`、`get_state() -> dict`、`set_state(state: dict) -> None`、`get_status() -> dict`。内部方法 `_trim(history: list) -> None`、`_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float` 亦有标注。`__init__` 参数标注完整。 |
| docstring 覆盖所有公开方法 | ✅ | 模块级 docstring（22 行）、类级 docstring（`SelfVoiceDistiller` 16 行，`SelfReferentialLoop` 28 行）、所有公开方法均有详细 docstring，包含参数说明、返回值说明、行为描述。内部方法 `_extract_activation_nodes`、`_trim`、`_cosine_similarity` 也有 docstring。 |
| 使用 logging 而非 print | ✅ | `logger = logging.getLogger(__name__)`，使用 `logger.debug(...)` 输出调试信息（第 308、359 行）。无任何 `print` 调用。 |
| 无魔法数字未注释 | ✅ | `alpha_base=0.15` 在 docstring 和设计文档中明确解释。`1e-8`（余弦相似度 epsilon）是标准数值稳定性常数。`0.95`/`0.80`（echo_threshold/echo_decay）标注为"Phase 1 使用"。`20`（history_cap 默认值）在 docstring 中说明。 |
| Tensor 操作 device 一致性 | ✅ | `observe()`：`current_emb = sensory_input.vector.detach().to(self.device).float()`，`prev_emb = self.sensory_self_prev.to(self.device).float()`——双方迁移到 `self.device` 后计算。`generate_echo()`：`sensory_self = self.sensory_self_prev.to(ext_sensory.device).detach()`——回注向量对齐到外部感官向量设备。`set_state()`：`prev.clone().to(self.device)`——恢复时迁移到当前设备。 |
| 异常处理：边界情况 | ✅ | `distill()` 处理空字符串、纯空白、无结构标记的纯文本、缺失段落、嵌套标记、超长文本——均返回字符串不崩溃。`generate_echo()` 处理首轮无缓存（返回 None）。`_cosine_similarity()` 处理零向量（返回 0.0）。`set_state()` 处理缺失字段（`state.get(key, default)`）。 |
| 向后兼容 | ✅ | `EpisodicEntry.source` 带默认值 `'external'`（memory.py 第 48 行），现有代码无需修改。`snapshot.py` 的 `self_ref` 为可选字段（`Optional[dict] = None`），旧快照无此字段时 `load_state` 走 else 分支打印兼容日志。`SNAPSHOT_VERSION` 升级至 `"0.4.0"`。 |

### 3. 设计符合性

| 检查项 | 结果 | 说明 |
|--------|------|------|
| distill() 正确提取自述、排除外部记忆 | ✅ | 蒸馏器三段处理：`interpret` 段取 `- ` 开头条目（自述）；`recall` 段整段 `continue` 丢弃（外部记忆回声）；`detail` 段提取"激活节点"部分。与 `bridge/decoder.py` 第 140-153 行的输出格式完全匹配。 |
| generate_echo() 首轮返回 None | ✅ | 第 290 行：`if self.turn_count == 0 or self.sensory_self_prev is None: return None`。`turn_count` 初始为 0，首次调用 `generate_echo`（在首轮 `process_turn` 的插入点 A）时返回 None。 |
| 1 轮延迟 | ✅ | `observe()` 在轮尾缓存 `sensory_self_prev = current_emb`（第 350 行）并递增 `turn_count`（第 357 行）。`generate_echo()` 在轮首使用 `self.sensory_self_prev`（第 298 行）。时序：第 t-1 轮 observe → 缓存 → 第 t 轮 generate_echo 使用。测试 `test_one_turn_delay` 通过 FakeEncoder 编码向量验证延迟内容来源正确。 |
| observe() 正确计算 echo_similarity | ✅ | 第 342-346 行：`echo_sim = self._cosine_similarity(current_emb, prev_emb)`，使用余弦相似度。首次 observe 无 prev 时 `echo_sim = None`（第 341 行）。结果存入 `echo_similarity_history` 并裁剪。`_cosine_similarity` 实现正确（第 465-468 行），处理零向量返回 0.0。 |
| get_state()/set_state() 序列化完整 | ✅ | `get_state()` 返回 8 个字段：`self_voice_history`、`sensory_self_prev`（clone 副本）、`gain_history`、`echo_similarity_history`、`turn_count`、`alpha_base`、`last_alpha`、`last_echo_similarity`。`set_state()` 逐一恢复，缺失字段用默认值，张量迁移到当前 device。 |
| process_turn 插入点位置正确 | ✅ | 插入点 A（第 250-266 行）：在 `encoder.encode` 之后（第 248 行）、`attractor.infer` 之前（第 299 行）——编码后推断前，符合设计。插入点 B（第 385-389 行）：在 `decoder.decode` 之后（第 380 行）、`return memory_context` 之前（第 428 行）——解码后返回前，符合设计。 |
| save_state/load_state 正确处理 self_ref_state | ✅ | `save_state`（第 511-514 行）：`if self.self_ref is not None: self_ref_state = self.self_ref.get_state()`，传入 `snapshot.save(..., self_ref_state=self_ref_state)`。`load_state`（第 562-570 行）：三种情况全覆盖——快照有+启用→恢复；快照有+未启用→跳过日志；快照无→兼容日志。 |

### 4. 测试质量

| 检查项 | 结果 | 说明 |
|--------|------|------|
| 28 个测试覆盖完整 | ✅ | Grep 确认 `def test_` 出现 28 次。分布：蒸馏 7 + 回注/延迟 5 + 相似度/裁剪 4 + 状态往返 4 + 来源标记 3 + 回归保护 5 = 28。覆盖蒸馏/回注/延迟/相似度/状态往返/来源标记/回归保护全部维度。 |
| 无 skip 的测试（实际执行） | ⚠️ | 28 个测试中无 `@pytest.mark.skip` 装饰器。但存在 5 处条件性 `pytest.skip()` 调用（第 67、637、649、662、718、744 行），用于并行开发时的容错。由于模块已实现（`_SELF_REF_AVAILABLE=True`）且 `EpisodicEntry.source` 已存在（`_episodic_has_source()=True`），所有 skip 分支不会触发。**功能上所有测试实际执行**，但残留的 skip 守卫属于应清理的死代码。 |
| mock 隔离外部依赖 | ✅ | 使用 `FakeEncoder`（md5 种子确定性向量）、`FakeTokenizer`、`FakeEmbedder` 隔离真实编码器/分词器/嵌入器。回归保护测试使用 `make_test_config()` 构造小规模配置（num_nodes=32, input_dim=16）。无网络/文件依赖（除 LivingMemoryLoop 集成测试内部可能涉及的快照目录，但未触发自动快照）。 |
| 边界用例覆盖 | ✅ | 蒸馏：空 context、缺失段落、纯文本无标记、仅标记无内容、嵌套标记、超长 context（6 个边界用例）。回注：首轮无回注、自定义 alpha。状态：未 observe 时 get_status。来源：默认值、自定义值、不影响既有字段。 |

### 5. 回归安全

| 检查项 | 结果 | 说明 |
|--------|------|------|
| self_ref_enabled=False 时行为不变 | ✅ | `self.self_ref = None`，5 处 `if self.self_ref is not None:` 守卫全部跳过。`process_turn` 执行路径与引入前完全一致（`alpha_t = 0.0`，sensory_input 不被修改）。`save_state`/`load_state`/`get_status` 中的自指分支同样跳过。测试 `test_default_disabled`、`test_explicit_disabled`、`test_disabled_identical_to_baseline` 验证。 |
| 快照向后兼容 | ✅ | `snapshot.py` 的 `self_ref` 为可选字段（`if self_ref_state is not None: data['self_ref'] = self_ref_state`）。`load_state` 使用 `raw_data.get('self_ref')`，旧快照返回 None 走 else 分支。`set_state` 使用 `state.get(key, default)` 处理所有字段。 |
| EpisodicEntry.source 不影响现有代码 | ✅ | `source: str = 'external'` 带默认值，位于 dataclass 字段末尾。现有 `store_episodic` 调用（loop.py 第 397-399 行）不传 source，使用默认值。现有 `EpisodicEntry(text=, semantic_vector=, surprise=, turn=)` 构造方式完全兼容。 |

---

## 三、发现的问题清单

按严重度排序：

### Critical（0 项）

无。

### High（0 项）

无。

### Medium（0 项）

无。

### Low（6 项）

**L1. 测试文件残留并行开发 skip 守卫（死代码）**

- 文件：`tests/test_self_referential.py`
- 位置：第 42-53 行（`_SELF_REF_AVAILABLE` 机制）、第 64-70 行（`_require_self_ref`）、第 73-79 行（`_episodic_has_source`）、第 636-637/648-649/661-662/717-718/743-744 行（条件 skip）
- 描述：模块已实现，所有 skip 分支不会触发，但守卫代码仍残留。审计项"无 skip 的测试"严格来说存在 skip 调用语句。
- 影响：无功能影响。代码可读性略受影响。
- 建议：Phase 0 验收后清理 skip 守卫，将条件 skip 改为直接断言（`assert SelfVoiceDistiller is not None`）。

**L2. `_cosine_similarity` 中 `float(denom)` 强制 GPU-CPU 同步**

- 文件：`core/hippocampus/self_referential.py`
- 位置：第 466 行 `if float(denom) < 1e-8:`
- 描述：`denom = a.norm() * b.norm()` 在 GPU 上产生 GPU 张量，`float(denom)` 触发 GPU→CPU 同步。在 GPU 设备上高频调用时有性能开销。
- 影响：Phase 0 MVP 规模小（history_cap=20，每轮 1 次），性能影响可忽略。
- 建议：Phase 1 若引入高频调用，可改为 `if denom.item() < 1e-8` 或使用 `torch.where` 避免 sync。

**L3. `test_distill_unstructured_text` 断言过弱**

- 文件：`tests/test_self_referential.py`
- 位置：第 288-294 行
- 描述：仅断言 `isinstance(result, str)`，未验证返回内容。注释称"优雅处理：返回字符串即可（可为原文或空）"。
- 影响：无法捕获蒸馏器对异常格式的处理回归。
- 建议：补充断言，如 `assert result == ""` 或 `assert "[记忆context]" not in result`。

**L4. `loop.py` 中 `hasattr(self, 'last_entropy_ratio')` 永远为 True**

- 文件：`runtime/loop.py`
- 位置：第 256 行 `entropy_ratio=self.last_entropy_ratio if hasattr(self, 'last_entropy_ratio') else 0.5`
- 描述：`last_entropy_ratio` 在 `__init__` 第 183 行已初始化为 `0.0`，`hasattr` 检查永远为 True，else 分支（0.5）永远不会执行。
- 影响：无功能影响。过度防御。
- 建议：简化为 `entropy_ratio=self.last_entropy_ratio`。

**L5. `history_cap=0` 导致历史不裁剪（无限增长）**

- 文件：`core/hippocampus/self_referential.py`
- 位置：第 451 行 `if cap > 0 and len(history) > cap:`
- 描述：当 `history_capacity=0` 时，条件 `cap > 0` 为 False，`_trim` 不执行，历史列表无限增长。
- 影响：正常使用不会传 `history_cap=0`（默认 20）。但若误配置可能导致内存泄漏。
- 建议：在 docstring 中注明 `history_cap` 应为正整数，或在 `__init__` 中 assert `history_capacity > 0`。

**L6. `get_status()` 返回冗余别名键**

- 文件：`core/hippocampus/self_referential.py`
- 位置：第 424-438 行
- 描述：`echo_similarity` 与 `last_echo_similarity` 同值；`history_cap` 与 `history_capacity` 同值；`history_size` 与 `self_voice_history_size` 同值。共 3 组冗余别名。
- 影响：无功能影响。API 表面积略大。
- 建议：Phase 1 统一键名后移除冗余别名，或在 docstring 中说明别名用途。

### Info（3 项，观察性记录）

**I1. 无文件级 save/load 往返测试**

- `test_state_roundtrip` 和 `test_state_roundtrip_echo_consistent` 测试 `get_state()`/`set_state()` 内存级往返，但未测试 `save_state(path)` → 文件 → `load_state(path)` 完整磁盘往返。
- 设计文档将"save→load 往返后 generate_echo 产出一致"列为 Phase 2 验证标准，Phase 0 不要求。

**I2. `echo_threshold`/`echo_decay` 已存储但 Phase 0 未使用**

- `self_referential.py` 第 238-243 行存储了 `echo_threshold=0.95` 和 `echo_decay=0.80`，docstring 标注"Phase 1 使用"。
- 这是为 Phase 1 硬守卫预留的配置参数，符合渐进式设计。

**I3. `_prev_activation` 已存储但 Phase 0 未使用**

- `loop.py` 第 207 行初始化 `self._prev_activation = None`，第 392 行更新，传入 `generate_echo(activation_prev=...)`，但 `generate_echo` 在 Phase 0 不使用该参数。
- docstring 和注释均标注"Phase 0 预留"，符合设计。

---

## 四、改进建议

### 短期（Phase 0 验收前可选）

1. **清理测试 skip 守卫**（L1）：移除 `_require_self_ref()`、`_SELF_REF_AVAILABLE`、`_episodic_has_source()` 及相关条件 skip，改为直接导入断言。这使测试意图更清晰，也使审计项"无 skip 的测试"完全满足。

2. **增强 `test_distill_unstructured_text` 断言**（L3）：补充 `assert result == ""` 验证无结构标记的纯文本蒸馏结果为空。

3. **简化 `hasattr` 检查**（L4）：移除 `loop.py` 第 256 行的 `hasattr` 守卫。

### 中期（Phase 1 实施时）

4. **文档化 `history_cap` 约束**（L5）：在 `__init__` docstring 中注明 `history_cap` 应为正整数。

5. **评估 `float(denom)` 性能影响**（L2）：若 Phase 1 引入更高频的相似度计算，考虑优化 GPU sync。

6. **统一 `get_status()` 键名**（L6）：确定标准键名后移除冗余别名。

---

## 五、架构符合性验证总结

```
依赖方向（DAG）验证：
  runtime/loop.py
    → core.hippocampus.self_referential  (新)
    → core.hippocampus.memory            (改)
    → persistence.snapshot               (改)
  
  core.hippocampus.self_referential
    → core.types (Activation, resolve_device)
    → torch, logging
    ✗ 不导入 runtime/loop  (无循环依赖)
    ✗ 不导入 persistence   (无跨层依赖)
    ✗ 不导入 os/pathlib    (无 IO 依赖)

信号通路验证：
  [编码] → ★插入点A(generate_echo回注) → [推断] → [学习] → [熵管理] → [目的] → [记忆] → [检索] → [解码] → ★插入点B(observe蒸馏) → [返回]
           ↑ 第t-1轮缓存                        ↑ 无自指干扰                                          ↑ 缓存第t轮供t+1使用
```

**最终评定：Phase 0 MVP 代码通过审计，满足全部验收标准，可进入 Phase 1 开发。**

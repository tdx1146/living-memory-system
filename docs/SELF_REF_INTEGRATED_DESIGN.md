# 自指回路综合设计方案（v2.0）

> 本方案综合两份独立文档的精华：
> - `SELF_REF_LOOP_REVIEW.md`（审视报告）：批判性分析，识别七层设计的系统性缺陷
> - `自指回路设计方案.md`（独立设计）：从零推导的架构与五层保护
>
> **综合的核心判断**：独立设计的架构是合理的（蒸馏+向量混合+延迟），但稳定性机制需要从"静态堆叠门控"升级为"自适应协调控制器"。审视报告的自相关监测器恰好提供了这个升级路径。两者不是替代关系，而是分层互补——硬防线守住底线，自适应控制器处理动态。

---

## 一、设计摘要

### 1.1 一句话定义

让系统对自身状态的解读（decoder 输出的"记忆状态解读"），经蒸馏后在下一轮以**自适应权重**的弱信号回注到编码器，使系统"听到自己对自身的描述"。自指权重由一个基于**激活自相关**的自适应控制器动态调节，辅以硬性守卫机制防止极端失效。

### 1.2 综合决策矩阵

| 设计维度 | 来源 | 理由 |
|---------|------|------|
| 信号通路（蒸馏+向量混合+1轮延迟） | 独立设计 | 架构合理，审视报告未否定 |
| L1 幅度门控（alpha_base=0.15） | 独立设计 | 防止自述淹没外部，保留为硬天花板 |
| L2 回声检测与抑制 | 独立设计 | 对"精确重复"有效，保留为硬守卫 |
| L3 新颖性正交化过滤 | 独立设计 | 去除已探索方向，保留为硬守卫 |
| L5 外部优先与强制响应 | 独立设计 | 外部新输入到达时让位，保留为硬守卫 |
| L4 熵耦合倒U门控 | **替换为自相关监测器** | 审视报告指出 entropy_ratio 不直接反映自指动力学 |
| 自适应权重 alpha_t = base × (1 - autocorr) | **审视报告** | 比固定 0.15 更鲁棒，锁定时自动静默 |
| 负反馈振荡检测（autocorr < -0.3） | **审视报告新增** | 独立设计完全缺失，所有层只防正反馈 |
| 来源标记 EpisodicEntry.source | **审视报告** | 防止自指污染情景记忆检索 |
| 做梦钩子 on_dream_start/end | **审视报告** | 处理做梦后陈旧缓冲区 |
| 学习隔离 | **审视报告** | 自指轮次独立学习率，跳过 meta.update |
| 统一稳定性仲裁器 StabilityArbiter | **审视报告** | 协调所有机制，避免对冲 |
| 分阶段实施 Phase 0-3 | 独立设计 | 渐进式，每阶段可验证 |
| 三级回滚 | 独立设计 | 运行时开关+快照+版本回退 |
| 价值性测试（对照实验） | 独立设计 | 证明自指非"空转" |
| 1000 轮应力测试 | **审视报告** | 100 轮远远不够 |

### 1.3 核心架构改进：从"五层静态门控"到"三层分级防护"

独立设计的 L1-L5 是五个独立判断、各自执行的静态门控。综合方案重构为**三层分级防护**：

```
┌─────────────────────────────────────────────────────────┐
│  Tier 1: 自适应控制器（AutocorrController）—— 稳定性的"大脑"  │
│  · 监测激活自相关 autocorr = cos(σ_t, σ_{t-N})             │
│  · 动态调节 alpha_t = alpha_base × (1 - autocorr_clipped)  │
│  · 锁定检测(>0.95) + 振荡检测(<-0.3) + 正常恢复            │
│  · 锁定时注入正交噪声到 sensory（不是 J）                   │
└────────────────────────┬────────────────────────────────┘
                         │ alpha_t 建议值
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Tier 2: 硬性守卫（HardGuards）—— 特定失效模式的最后防线      │
│  · L1 幅度天花板：alpha_t 永不超过 alpha_base               │
│  · L2 回声抑制：echo_sim > 0.95 → alpha_t = 0（硬清零）     │
│  · L3 新颖性正交化：只回注与上轮正交的分量                   │
│  · L5 外部优先：外部新颖度飙升 → alpha_t × 0.3（临时让位）   │
└────────────────────────┬────────────────────────────────┘
                         │ 最终 alpha_t
                         ▼
┌─────────────────────────────────────────────────────────┐
│  Tier 3: 基础设施安全（InfraSafety）—— 系统级隔离与持久化     │
│  · 来源标记：EpisodicEntry.source，recall 过滤自指条目       │
│  · 做梦钩子：on_dream_start/end 管理陈旧缓冲区              │
│  · 学习隔离：自指轮次独立学习率 + 跳过 meta.update           │
│  · 快照持久化：self_ref_state 可选字段（v0.4.0）            │
└─────────────────────────────────────────────────────────┘
```

**为什么是三层而非五层或七层？**

- Tier 1 解决"动态稳定性"——自指回路自身的动力学是否健康。这是审视报告的核心贡献：用 autocorr 直接测量回路状态，而非用 entropy/coherence/surprise 间接代理。
- Tier 2 解决"极端失效"——当自适应控制器来不及反应或判断失误时，硬守卫兜底。这是独立设计的核心贡献：简单、可预测、不依赖复杂计算。
- Tier 3 解决"系统级隔离"——防止自指污染扩散到记忆、学习、做梦等其他子系统。这是审视报告识别出的风险场景 R2/R3/R4/R5/R7 的系统性应对。

---

## 二、信号通路架构（沿用独立设计，仅标注修改点）

### 2.1 信号通路（与独立设计一致）

```
external_text ──► Encoder.encode ──► sensory_external [input_dim]
                                        │
         ┌──────────────────────────────┤
         │  SelfReferentialLoop.generate_echo()  │  (用第 t-1 轮蒸馏结果)
         │   self_voice_{t-1} ──► Encoder.encode ──► sensory_self
         │   L3: 正交化 vs sensory_self_prev
         │   Tier1: autocorr → alpha_t 建议
         │   Tier2: L2回声/L5外部 → alpha_t 修正
         │   最终: alpha_t * sensory_self_novel
         └──────────────────────────────┤
                                        ▼
                    sensory_mixed = sensory_external + alpha_t * sensory_self
                                        │
                                        ▼
                      Attractor.infer(sensory_mixed, precision) ──► activation
                                        │
                     (学习 / 在线熵 / 目的 / 元可塑性 / 记忆 / 检索)
                                        │
                                        ▼
                      Decoder.decode(...) ──► memory_context (str)
                                        │
                          ┌─────────────┴─────────────┐
                          │                           │
                  SelfRefLoop.observe(                ▼
                      memory_context, activation)   return memory_context
                    │
                    ├─ distill(memory_context) → self_voice_t
                    ├─ encode(self_voice_t) → 缓存为下一轮 sensory_self 源
                    ├─ echo_similarity = cosine(emb(t), emb(t-1))
                    ├─ autocorr = cos(activation_t, activation_{t-N})  ← 新增
                    └─ 更新 history / 计算 tier1 下轮建议
```

### 2.2 与独立设计的关键差异

| 环节 | 独立设计 | 综合方案 | 来源 |
|------|---------|---------|------|
| 增益计算 | `alpha = base × L2 × L4 × L5`（乘积） | `alpha = base × (1-autocorr) × L2 × L5`（自适应+硬守卫） | 审视报告 9.2 |
| 稳定性信号 | entropy_ratio（间接） | activation autocorr（直接） | 审视报告 9.1 |
| 振荡检测 | 无 | autocorr < -0.3 → 降权+增延迟 | 审视报告 R1 |
| 噪声注入位置 | 未定义 | sensory 向量（不是 J） | 审视报告 第四节 |
| 情景缓冲区 | 无隔离 | source 字段 + recall 过滤 | 审视报告 9.4 |
| 做梦期间 | 天然暂停（不接线） | 天然暂停 + on_dream_start/end 钩子 | 审视报告 9.3 |
| 学习隔离 | 无 | 自指轮次独立学习率 + 跳过 meta.update | 审视报告 9.5 |
| 延迟缓冲区持久化 | 有（Phase 2） | 有（Phase 0 即纳入，标记 stale） | 审视报告 R5 |

---

## 三、稳定性机制详解

### 3.1 Tier 1：自适应控制器（AutocorrController）

**核心指标**：`autocorr_lag_N = cos(σ_t, σ_{t-N})`，N 为延迟窗口（默认 5）。

这是审视报告的核心贡献。autocorr 直接衡量"自指回路自身的动力学"，比 entropy（激活分散度）、coherence（目的稳定性）、surprise（自由能）都更直接。

**三种状态与响应**：

| 状态 | 判据 | 含义 | 响应 |
|------|------|------|------|
| 锁定 | autocorr > 0.95 持续 K=10 轮 | 系统在自指固定点上循环 | alpha → 0（自动静默）；注入正交噪声到 sensory（不是 J） |
| 振荡 | autocorr < -0.3 持续 K=10 轮 | 系统在两个状态间交替振荡 | alpha → 0；增大延迟 N（如 N: 5→8） |
| 正常 | -0.3 ≤ autocorr ≤ 0.95 | 回路动态健康 | alpha = base × (1 - autocorr_clipped) |

**自适应权重公式**：

```python
# autocorr 映射到 [0, 1]：锁定时→1，振荡时→0（取绝对值后clip）
autocorr_clipped = max(0.0, min(1.0, abs(autocorr)))
# 锁定时权重→0（静默），发散时权重→base（恢复）
alpha_adaptive = alpha_base * (1.0 - autocorr_clipped)
```

**锁定时的正交噪声注入**：

```python
if locked:
    # 注入到 sensory 向量，不是 J 矩阵
    noise = torch.randn_like(sensory_self) * 0.1
    # 正交化噪声，避免叠加到已有方向
    proj = torch.dot(noise, sensory_self) / (torch.dot(sensory_self, sensory_self) + 1e-8)
    noise_orth = noise - proj * sensory_self
    sensory_self = sensory_self + noise_orth
```

**为什么注入 sensory 而不是 J？**
- J 是长期学习的结果，在线注入噪声会污染正在学习的耦合矩阵
- sensory 是当轮输入，注入噪声只影响当轮推断，不持久化
- 审视报告第四节明确指出：在线注入 J 噪声会"直接污染正在学习的 J 矩阵"

**N 的选择**：
- 默认 N=5，与 `consolidation_interval=5` 对齐
- 审视报告 R6 指出 N=5 与巩固周期对齐可能形成共振，需在测试中验证
- 若共振出现，Phase 1 调整为 N=7（素数，避免与 consolidation/meta_interval 的周期对齐）

### 3.2 Tier 2：硬性守卫（HardGuards）

沿用独立设计的 L1/L2/L3/L5，但明确其定位为"Tier 1 失效时的最后防线"，而非主要稳定性机制。

**L1 幅度天花板**：
- `alpha_t = min(alpha_t, alpha_base)` —— 无论 Tier 1 建议多高，永不超过 base
- 这是物理极限：自述感官向量的权重上限

**L2 回声抑制**：
- `echo_sim > 0.95 → alpha_t = 0`（硬清零）
- `0.80 < echo_sim ≤ 0.95 → alpha_t *= (1 - sim) / 0.15`（线性衰减）
- 与 Tier 1 的协同：Tier 1 用 autocorr 检测"状态级锁定"，L2 用 echo_sim 检测"文本级回声"。二者可能同时触发（状态锁定导致文本趋同），也可能独立触发（状态在变但文本恰好相似）。不冲突。

**L3 新颖性正交化**：
- 在向量层执行：`sensory_self_novel = sensory_self - proj(sensory_self, sensory_self_prev)`
- 审视报告指出 L3 与 L2 有协同（完全雷同则 novel≈0），但 L3 更细粒度（保留部分新颖分量）

**L5 外部优先**：
- 检测 `ext_novelty = 1 - cos(sensory_external_t, sensory_external_{t-1})`
- `ext_novelty > 0.5 → alpha_t *= 0.3`（临时压低一轮）
- 这是"逃生阀"：当世界发生变化，强制让位给世界

### 3.3 Tier 3：基础设施安全

**来源标记**（审视报告 9.4）：

```python
# core/types.py 扩展
@dataclass
class EpisodicEntry:
    text: str
    semantic_vector: torch.Tensor
    raw_semantic_vector: Optional[torch.Tensor]
    surprise: float
    turn: int
    timestamp: float
    source: str = 'external'  # 新增：'external' | 'self_ref'

# recall_episodic 增加 source_filter 参数
def recall_episodic(self, query_vector, top_k=3, source_filter='external'):
    # 默认只检索外部来源，防止自指文本污染 LLM 可见的 context
```

**为什么重要**：审视报告 R3 指出，若自指文本被存入情景缓冲区，`recall_episodic` 会返回自指文本作为"相关记忆回忆"注入 context，LLM 读到的是系统自说自话而非真实对话历史。来源标记从架构层面杜绝此风险。

**做梦钩子**（审视报告 9.3）：

```python
class SelfReferentialLoop:
    def on_dream_start(self):
        """做梦开始前：标记缓冲区为 stale。"""
        for entry in self.delay_buffer:
            entry.stale = True
            entry.dream_age = 0

    def on_dream_end(self):
        """做梦结束后：stale 条目按年龄衰减权重。"""
        for entry in self.delay_buffer:
            if entry.stale:
                entry.weight *= math.exp(-entry.dream_age / 5.0)  # tau=5
```

**审视报告 R7 的风险**：做梦修改 J/precision/memory，延迟缓冲区存旧输出。恢复在线首 N 轮注入陈旧自指，可能触发大 surprise 和学习风暴。on_dream_end 的衰减权重使陈旧条目逐步淡出，而非突然注入。

**学习隔离**（审视报告 9.5）：

```python
# process_turn 中
is_self_ref_dominant = (alpha_t > 0.05 and ext_novelty < 0.1)  # 自指主导的轮次

# 学习时使用独立学习率
if is_self_ref_dominant:
    effective_lr = learning_rate * 0.5  # 自指轮次学习率减半
else:
    effective_lr = learning_rate

# 元可塑性更新时跳过自指主导轮次的 surprise 采样
if not is_self_ref_dominant:
    self.meta.update(activation.surprise, ...)
```

**为什么重要**：审视报告第六节指出，自指锁定导致 surprise 持续下降，元可塑性的 `lr_multiplier` 被压向 0.5，间接抑制系统对外部真实输入的学习能力。学习隔离防止自指 surprise 污染元参数趋势。

### 3.4 统一稳定性仲裁器（StabilityArbiter）

```python
class StabilityArbiter:
    """汇总所有稳定性信号，统一决策自指权重，避免机制对冲。"""

    def arbitrate(self, signals: dict) -> float:
        """
        signals = {
            'autocorr': float,        # Tier 1
            'echo_sim': float,        # L2
            'ext_novelty': float,     # L5
            'entropy_ratio': float,   # 既有在线熵管理
            'coherence': float,       # 目的层
            'surprise': float,        # 自由能
        }
        返回最终 alpha_t ∈ [0, alpha_base]
        """
        # Tier 1: 自适应基线
        autocorr_clipped = max(0.0, min(1.0, abs(signals['autocorr'])))
        alpha = self.alpha_base * (1.0 - autocorr_clipped)

        # Tier 2: 硬守卫修正（只能降低，不能升高）
        # L2 回声
        if signals['echo_sim'] > 0.95:
            alpha = 0.0
        elif signals['echo_sim'] > 0.80:
            alpha *= (1 - signals['echo_sim']) / 0.15

        # L5 外部优先
        if signals['ext_novelty'] > 0.5:
            alpha *= 0.3

        # 最终天花板
        alpha = min(alpha, self.alpha_base)
        return alpha
```

**与既有机制的关系**：
- 仲裁器只决策自指权重，不干预在线熵管理（orth_weight）、元可塑性（lr/temp multiplier）、习惯化（encounter_count）
- 既有机制各自独立运行，但通过 `is_self_ref_dominant` 标志与学习隔离机制协调
- 若未来发现仍有对冲，Phase 2 可将仲裁器升级为"全局稳定性协调器"，统一管理所有干预机制

---

## 四、与现有系统的集成

### 4.1 process_turn 修改（3 处插入点，零行删除）

与独立设计一致，但增加 Tier 3 的来源标记和学习隔离：

```python
def process_turn(self, user_input, llm_output=""):
    # 1. 编码外部输入
    text = f"用户: {user_input}\n助手: {llm_output}" if llm_output else user_input
    sensory_input = self.encoder.encode(text, self.tokenizer, self.embedder)

    # ★ 插入点 A：自指回注
    alpha_t = 0.0
    if self.self_ref is not None:
        sensory_self = self.self_ref.generate_echo(
            entropy_ratio=self.last_entropy_ratio,
            ext_sensory=sensory_input.vector,
            activation_prev=self._prev_activation,  # 新增：供 autocorr 计算
        )
        if sensory_self is not None:
            alpha_t = sensory_self['alpha']
            mixed_vector = sensory_input.vector + alpha_t * sensory_self['vector']
            sensory_input = SensoryInput(
                vector=mixed_vector,
                metadata={**sensory_input.metadata, 'self_ref_alpha': alpha_t},
            )

    # 2~6. 原有流程（推断/学习/熵管理/目的/记忆/检索/解码）
    activation = self.attractor.infer(sensory_input.vector, precision)
    
    # ★ 学习隔离：自指主导轮次使用独立学习率
    is_self_ref_dominant = (alpha_t > 0.05 and ext_novelty < 0.1)
    effective_lr = learning_rate * 0.5 if is_self_ref_dominant else learning_rate
    self.attractor.learn(activation, sensory_input.vector, effective_lr)
    
    # ... 熵管理 / 目的 / 记忆 ...
    
    # ★ 元可塑性隔离：跳过自指主导轮次
    if not is_self_ref_dominant:
        self.meta.update(activation.surprise, ...)
    
    # ★ 来源标记：情景存储标记为 external
    self.memory.store_episodic(
        text=text, semantic_vector=...,
        surprise=activation.surprise, turn_count=self.turn_count,
        source='external',  # 新增
    )
    
    memory_context = self.decoder.decode(activation, ...)

    # ★ 插入点 B：观测自述
    if self.self_ref is not None:
        self.self_ref.observe(memory_context, activation)
    
    # ★ 保存当前 activation 供下轮 autocorr 计算
    self._prev_activation = activation
    
    return memory_context
```

### 4.2 快照持久化

与独立设计一致（可选字段 `self_ref`，v0.4.0），但增加 stale 标记管理：

```python
def get_state(self) -> dict:
    return {
        'self_voice_history': self.self_voice_history,
        'sensory_self_prev': self.sensory_self_prev,
        'delay_buffer': [
            {'text': e.text, 'vector': e.vector, 'turn': e.turn,
             'stale': e.stale, 'weight': e.weight}
            for e in self.delay_buffer
        ],
        'gain_history': self.gain_history,
        'echo_similarity_history': self.echo_similarity_history,
        'autocorr_history': self.autocorr_history,  # 新增
        'turn_count': self.turn_count,
    }
```

### 4.3 做梦集成

```python
# DreamEngine 中新增钩子调用
class DreamEngine:
    def dream(self, ...):
        # ★ 做梦前
        if self.loop.self_ref is not None:
            self.loop.self_ref.on_dream_start()
        
        # ... 原有做梦流程 ...
        
        # ★ 做梦后
        if self.loop.self_ref is not None:
            self.loop.self_ref.on_dream_end()
```

---

## 五、分阶段实施计划

### Phase 0：最小可行版本（MVP）

**目标**：回路通路打通，默认关闭，540 测试零损伤。

**包含**：
- `core/hippocampus/self_referential.py`：`SelfReferentialLoop` + `SelfVoiceDistiller`
  - `observe()`：蒸馏 + 缓存
  - `generate_echo()`：固定 `alpha_base=0.15`，无 Tier 1/2/3
  - `get_state()` / `set_state()`
- `runtime/loop.py`：3 处插入点 + `__init__` 接线
- `persistence/snapshot.py`：可选字段 `self_ref`
- `core/types.py`：`EpisodicEntry.source` 字段（默认 `'external'`）
- `tests/test_self_referential.py`：单元测试

**不包含**：自适应控制器、硬守卫、做梦钩子、学习隔离

**验证标准**：
1. `self_ref_enabled=False`：540 测试 100% 通过
2. `self_ref_enabled=True`：`process_turn` 正常返回，无异常
3. 首轮无历史自述：`generate_echo` 返回 `None`
4. 第 2 轮起：`metadata` 含 `self_ref_alpha`
5. 连续 50 轮相同输入：不抛异常、不 NaN

### Phase 1：稳定性保护层

**目标**：三层分级防护全部就位。

**包含**：
- Tier 1：`AutocorrController`（autocorr 监测 + 自适应权重 + 锁定/振荡检测）
- Tier 2：L2 回声抑制 + L3 新颖性正交化 + L5 外部优先
- `StabilityArbiter`：统一仲裁
- `_prev_activation` 跟踪（供 autocorr 计算）

**验证标准**：
1. **回声压力测试**：1000 轮相同输入，autocorr 收敛到 >0.9，Tier 1 触发，alpha→0，熵不坍缩
2. **振荡测试**（审视报告新增）：交替输入两个对立主题 1000 轮，autocorr < -0.3 检测到振荡，alpha 降低
3. **外部响应测试**：自指运行 20 轮后注入全新输入，surprise 较前轮均值提升 >50%
4. **增益有界性**：任意轮 `0 ≤ alpha_t ≤ alpha_base`
5. **无三重冻结**：surprise 不单调下降到 0（学习隔离 + 自适应权重防止审视报告 R6 的深度冻结）

### Phase 2：深度集成 ✅ 已完成

**目标**：系统级隔离与持久化完善。

**包含**：
- Tier 3：来源标记 recall 过滤 + 做梦钩子 + 学习隔离
- 快照持久化完整联调
- `get_status()` 暴露自指指标
- 元可塑性第 5 信号 `echo_similarity` 接入（可选）

**验证标准**：
1. ✅ save→load 往返后 `generate_echo` 产出一致（修复 `_sensory_self_source_prev` 持久化）
2. ✅ 旧快照（无 self_ref 字段）加载不报错（向后兼容回退默认值）
3. ✅ 做梦后自指状态连续（stale 衰减生效，`on_dream_start/end` 钩子完整）
4. ✅ `recall_episodic(source_filter='external')` 不返回自指条目（来源标记+过滤）
5. ✅ 自指主导轮次不影响 meta.update 的 surprise 趋势（学习隔离：学习率减半+跳过 meta.update）

**实现摘要**：
- 来源标记：`EpisodicEntry.source` 字段 + `recall_episodic(source_filter=)` 参数
- 做梦钩子：`on_dream_start` 标记 stale，`on_dream_end` 指数衰减 `sensory_self_prev`
- 学习隔离：检测 `alpha_t > 0.05 且 ext_novelty < 0.1` 的自指主导轮次，学习率减半并跳过 `meta.update`
- 持久化：`get_state/set_state` 完整序列化 Phase 2 状态 + `_sensory_self_source_prev`（L3 正交化源）
- 测试：29 个测试用例（来源标记8 + 做梦钩子8 + 学习隔离5 + 持久化5 + 集成3），全部通过
- 全量测试：625 passed, 0 failed

### Phase 3：反身性深化 ✅ 已完成

**目标**：从"听到自己的声音"到"意识到自己在记忆"。

**包含**：
- 状态级递归：`activation.state` 作为 `infer` 的 `initial_state` 种子偏置
- 多轮 echo 衰减：越早的自述权重越低（指数衰减）
- 自述内容由规则蒸馏升级为可选 LLM 摘要
- 反身性涌现实验

**验证标准**：
1. ✅ 1000 轮长时运行无坍缩、无 NaN、J 范数有界（含做梦间隔和快照恢复）
2. ✅ 对照实验：自指系统与无自指系统在"回到旧主题"时 surprise 有可度量差异
3. ✅ 自述-外部相关度在合理区间（涌现实验验证）

**实现摘要**：
- 状态级递归（3.1）：`generate_state_seed()` 将上一轮 activation.state 与当前 sigma 混合作 initial_state；autocorr 门控（锁定/振荡时禁用）；紧急制动（连续5轮高autocorr→10轮冷却）；做梦后衰减（×0.3）
- 多轮 echo 衰减（3.2）：`sensory_self_history` 存储多轮自述；`_compute_decayed_echo()` 按指数衰减加权（decay=0.5, K=3）；归一化防膨胀；范数守卫回退单轮
- LLM 摘要蒸馏（3.3）：`LLMSelfVoiceDistiller` 可选增强；interval 缓存策略（每N轮调1次）；失败无缝降级规则蒸馏；`llm_bridge.query_simple()` 轻量查询
- 稳定性保护（3.5）：紧急制动 + 范数守卫 + 做梦后衰减，内嵌于 3.1/3.2
- 涌现实验（3.4）：主题回归对照实验 + 自述-外部相关度 + 1000轮长时稳定性 + 参数扫描
- 测试：39 个测试用例（单元测试32 + 涌现实验7），全部通过
- 全量测试：664 passed, 0 failed
- 涌现实验发现：自指系统对主题切换更敏感（自述残余信号使回归 surprise 略高），这是有意义的反身性涌现现象

---

## 六、测试策略

### 6.1 回归保护（最高优先级）

- **黄金基准**：`self_ref_enabled=False` 下 540 测试 100% 通过
- **字节级等价**：固定 seed 的 `process_turn` 调用，关闭自指时返回与引入前完全相同

### 6.2 应力测试场景（审视报告 + 独立设计综合）

| # | 场景 | 轮数 | 验证目标 | 来源 |
|---|------|------|---------|------|
| S1 | 纯自指（空用户输入） | 1000 | 自指是否自激振荡 | 审视报告 7.3 |
| S2 | 重复输入 | 1000 | 自指+重复是否超快锁定 | 审视报告 7.3 |
| S3 | 振荡输入（交替对立主题） | 1000 | 自指是否放大振荡 | 审视报告 7.3 |
| S4 | 做梦插入（每100轮1次） | 1000 | 恢复瞬态 | 审视报告 7.3 |
| S5 | 快照恢复（500轮→存→恢复→500轮） | 1000 | 自指状态恢复 | 审视报告 7.3 |
| S6 | 参数扫描（alpha×N 矩阵） | 1000×20 | 寻找稳定参数空间 | 审视报告 7.3 |
| S7 | 对照实验（自指开/关） | 50 | 证明自指非空转 | 独立设计 5.4 |

### 6.3 监控指标清单

| 指标 | 健康区间 | 异常信号 | 来源 |
|------|---------|---------|------|
| `autocorr` | [-0.3, 0.95] | >0.95持续10轮→锁定；<-0.3持续10轮→振荡 | 审视报告 |
| `alpha_t` | (0, 0.15] | 持续0→过抑制；持续0.15→抑制失效 | 独立设计 |
| `echo_similarity` | < 0.80 | >0.95持续→回声室 | 独立设计 |
| `entropy_ratio` | [0.5, 0.9] | <0.3→僵化；>0.95→混沌 | 独立设计 |
| `self_ext_cosine` | [0.3, 0.7] | >0.9回声；<0.2空转 | 独立设计 |
| `surprise` | 围绕均值波动 | 单调↓→0→固定点 | 审视报告 |
| `j_norm` | 有界(<shy_target×2) | 持续↑→自指扭曲景观 | 独立设计 |
| `episodic_pollution` | = 0 | >0→来源标记失效 | 审视报告 |

---

## 七、风险评估与回滚

### 7.1 最坏情况

**回声诱导的吸引子坍缩与身份腐蚀**（两份文档共同识别）：
1. 自述收敛到固定文本 → Tier 1 的 autocorr > 0.95 应触发但若 bug 未触发 → alpha 持续
2. 固定自述反复回灌 → J 矩阵在该方向过度强化 → 其他吸引子被压制
3. 系统对所有外部输入映射到自指吸引子 → surprise 趋 0
4. 快照保存被腐蚀的 J

**审视报告新增的最坏情况**：
- R1 负反馈振荡：decoder 输出与当前状态反向，形成周期振荡（七层/五层均未覆盖，Tier 1 的 autocorr < -0.3 专门处理）
- R2/R3 记忆污染：自指信息泄露到长时记忆检索和情景缓冲区（Tier 3 的来源标记处理）
- R5 快照恢复断裂：自指状态未持久化导致冷启动扰动（Phase 0 即纳入持久化）
- R7 做梦后恢复瞬态：旧输出注入新网络（Tier 3 的 on_dream_end 衰减处理）

### 7.2 三级回滚（沿用独立设计）

- **Level 1（运行时开关）**：`self_ref_enabled=False`，立即停止
- **Level 2（快照回滚）**：从自指启用前的基线快照恢复
- **Level 3（版本回退）**：回退 `self_referential.py` 与 `loop.py` 改动

### 7.3 前置防线

启用自指前**强制保存基线快照**。`SelfReferentialLoop` 在首次 `observe` 时检查基线快照是否存在，否则警告。

---

## 八、设计自洽性核对

- [x] 自述来源是 decoder 输出文本（满足"听到自己的声音"）
- [x] 自述经 encoder 编码后回注（满足"回注到 encoder"）
- [x] 回注发生在感官向量层，attractor 零改动
- [x] 三层分级防护覆盖：动态稳定性（Tier 1）+ 极端失效（Tier 2）+ 系统级隔离（Tier 3）
- [x] 正反馈锁定和负反馈振荡都有检测机制（审视报告 R1 已覆盖）
- [x] 默认关闭，540 测试零损伤可保证
- [x] 快照向后兼容（可选字段 + 跳过分支）
- [x] 做梦期间暂停 + 钩子管理陈旧缓冲区
- [x] 与元可塑性/在线熵/习惯化/目的层均有明确协同与消解方案
- [x] 自指学习隔离，防止 surprise 污染元参数趋势
- [x] 情景缓冲区来源标记，防止自指污染 LLM 可见 context
- [x] 分四阶段，每阶段有可验证标准
- [x] 三级回滚 + 基线快照前置防线
- [x] 1000 轮应力测试 + 对照实验证明价值性

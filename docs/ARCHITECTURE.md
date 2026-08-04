# 活体记忆系统（Living Memory System, LMS）
# 架构设计文档 v0.1

> 2026-07-28
> 设计者：dandan + AI
> 实现者：子AI团队

---

## 一、系统定位

一个外挂于主LLM的"海马体"记忆层。主LLM负责思考（黑箱API），本系统负责让思考被记住、被遗忘、被重新激活。

**核心原则**：
- 记忆不是存档，是塑形——SLM的参数被对话过程持续改变
- 学习规则从自由能原理（FEP）涌现，不需要预训练、不需要反向传播
- 目的层从设计之初就纳入——SLM自主调整precision，决定"什么值得惊讶"
- 身份是吸引子景观（经历的总和），不是参数值（船板）

---

## 二、架构原则

- **前后端分离**：记忆核心（纯计算）与LLM交互（IO/通信）分离
- **模块独立**：每个模块可独立测试，不依赖其他模块的实现
- **高内聚低耦合**：模块内高内聚，模块间通过定义好的接口交互
- **咬合顺畅**：接口数据类型统一，转换自然，无多余胶水代码

---

## 三、模块划分

```
lms/
├── core/                        # 核心层（后端，纯计算，无IO依赖）
│   ├── __init__.py
│   ├── types.py                 # 数据类型定义
│   ├── sensory/                 # 感官层
│   │   ├── __init__.py
│   │   ├── tokenizer.py          # 文本→token（接口+默认实现）
│   │   └── embedder.py          # token→向量（冻结embedding，感觉器官）
│   ├── hippocampus/             # 海马体核心
│   │   ├── __init__.py
│   │   ├── attractor.py          # FEP吸引子网络（推断+学习规则）
│   │   ├── purpose.py            # 目的层（precision自主调节）
│   │   └── memory.py             # 多尺度记忆管理（短时/长时/consolidation）
│   └── config.py                # 核心配置
│
├── bridge/                      # 桥接层（连接核心与外部LLM）
│   ├── __init__.py
│   ├── encoder.py               # LLM输出文本→海马体感官信号
│   ├── decoder.py               # 海马体激活态→LLM可理解的context
│   └── llm_bridge.py            # 主LLM API调用封装
│
├── persistence/                  # 持久化层
│   ├── __init__.py
│   ├── snapshot.py              # 吸引子景观快照（J矩阵+目的层状态）
│   └── recovery.py              # 断点续传
│
├── runtime/                      # 运行时（调度/入口）
│   ├── __init__.py
│   ├── loop.py                  # 在线学习环（主循环）
│   └── cli.py                    # 命令行入口
│
├── tests/                        # 测试
│   ├── test_attractor.py
│   ├── test_purpose.py
│   ├── test_memory.py
│   └── test_integration.py
│
├── docs/
│   └── ARCHITECTURE.md          # 本文档
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 四、数据流

```
┌─────────────────────────────────────────────────────────┐
│                      运行时循环                            │
│                                                           │
│  对话消息 ──→ tokenizer ──→ embedder ──→ [感官向量]         │
│                                                │           │
│                                    encoder ←─────┘         │
│                                       │                    │
│                                       ▼                    │
│  ┌──────────────────────────────────────────────────┐      │
│  │              海马体核心                             │      │
│  │                                                    │      │
│  │  目的层 ──→ precision ──┐                          │      │
│  │                          ▼                         │      │
│  │  attractor.infer(感官信号, precision) → [激活态]    │      │
│  │                          │                         │      │
│  │  attractor.learn(激活态, 感官信号) → 更新J矩阵      │      │
│  │                          │                         │      │
│  │  purpose.adjust(惊讶度) → 更新precision             │      │
│  │                          │                         │      │
│  │  memory.consolidate() → 短时→长时迁移               │      │
│  └──────────────────────────────────────────────────┘      │
│                                       │                    │
│  主LLM ←── llm_bridge ←── decoder ←────┘                    │
│     │                                                       │
│     └────── LLM输出回到循环顶部 ──────────────────────────┘
│
│  [定期] snapshot.save() → J矩阵 + precision状态 → 磁盘
└─────────────────────────────────────────────────────────┘
```

---

## 五、接口定义

### 5.1 数据类型 (core/types.py)

```python
from dataclasses import dataclass, field
import torch

@dataclass
class SensoryInput:
    """感官输入：从对话文本编码而来的向量信号"""
    vector: torch.Tensor        # [input_dim] 感官向量
    metadata: dict = field(default_factory=dict)  # 时间戳、来源等

@dataclass
class Activation:
    """海马体激活态：吸引子网络的输出"""
    state: torch.Tensor         # [num_nodes] 节点激活值
    entropy: float              # 激活熵
    surprise: float             # 自由能（惊讶度）

@dataclass
class PurposeState:
    """目的层状态：SLM当前的'兴趣/关注'分布"""
    precision: torch.Tensor     # [input_dim] 每个感官维度的精度
    history: list               # precision演化历史
    coherence: float            # 目的的内部一致性
```

### 5.2 感官层接口

```python
class Tokenizer:
    def tokenize(self, text: str) -> list[int]:
        """文本→token id列表"""
        ...

class Embedder:
    def embed(self, tokens: list[int]) -> torch.Tensor:
        """token→向量（冻结，不训练）"""
        ...
    @property
    def dim(self) -> int:
        """embedding维度"""
        ...
```

### 5.3 海马体核心接口

```python
class AttractorNetwork:
    """FEP吸引子网络：核心记忆引擎"""
    
    def __init__(self, num_nodes: int, input_dim: int):
        """
        num_nodes: 吸引子网络节点数（建议256-1024起步）
        input_dim: 感官输入维度（=embedding维度）
        """
        ...
    
    def infer(self, sensory_input: torch.Tensor, precision: torch.Tensor,
              num_steps: int = 10) -> Activation:
        """
        FEP推断：给定感官输入和precision，跑K步收敛到吸引子态。
        
        推断规则（从FEP推导）:
            b_q = b + J @ sigma          # 对数几率
            sigma = langevin(b_q)         # Langevin激活: coth(b) - 1/b
            感官节点被sensory_input * precision clamping
        
        返回: 收敛后的激活态 + 熵 + 惊讶度（自由能）
        """
        ...
    
    def learn(self, activation: Activation, sensory_input: torch.Tensor,
              learning_rate: float = 0.01) -> None:
        """
        FEP学习：更新耦合矩阵J。
        
        学习规则（从FEP推导）:
            ΔJ = -η * ∂F/∂J
            F = 准确性项 + 复杂性项(KL)
            准确性梯度 ≈ Hebbian相关 (σ_i * σ_j)
            复杂性梯度 = 正交化压力
        
        无反向传播。无全局loss。规则从第一性原理推导。
        """
        ...
    
    def get_landscape(self) -> dict:
        """返回当前吸引子景观状态（用于快照）"""
        ...

class PurposeLayer:
    """目的层：SLM自主调整precision"""
    
    def __init__(self, input_dim: int):
        ...
    
    def get_precision(self) -> torch.Tensor:
        """返回当前precision向量"""
        ...
    
    def adjust(self, surprise: float, activation: Activation) -> None:
        """
        根据惊讶度调整precision。
        
        核心思想:
        - 高惊讶维度 → 提高precision（更关注这个方向，学得更快）
        - 低惊讶维度 → 降低precision（已熟悉，减少关注）
        - 但不是简单的贪心调整——有自身一致性约束
        
        调整规则也遵循FEP元层面:
        - precision的更新 = 最小化"元自由能"
        - 元自由能 = 对"我的precision是否合适"的惊讶
        
        这就是"目的随对话共生"的机制：
        precision的演化轨迹 = SLM的"兴趣"成型过程
        """
        ...
    
    def get_purpose(self) -> PurposeState:
        """返回当前目的状态"""
        ...

class MemoryManager:
    """多尺度记忆管理"""
    
    def __init__(self, short_term_decay: float = 0.8,
                 long_term_decay: float = 0.999):
        ...
    
    def update(self, activation: Activation) -> None:
        """更新短时/长时记忆潜变量"""
        ...
    
    def consolidate(self) -> None:
        """记忆巩固：短时→长时迁移，回放重要经验"""
        ...
    
    def recall(self, cue: torch.Tensor) -> torch.Tensor:
        """从记忆中检索：用线索激活相关记忆"""
        ...
```

### 5.4 桥接层接口

```python
class Encoder:
    """LLM输出→海马体输入"""
    def encode(self, text: str, tokenizer: Tokenizer,
              embedder: Embedder) -> SensoryInput:
        """对话文本→感官信号"""
        ...

class Decoder:
    """海马体→LLM可理解的context"""
    def decode(self, activation: Activation) -> str:
        """激活态→context文本（或向量）"""
        ...

class LLMBridge:
    """主LLM API封装"""
    def query(self, user_input: str, memory_context: str) -> str:
        """带记忆context查询主LLM"""
        ...
```

### 5.5 持久化接口

```python
class Snapshot:
    """吸引子景观快照"""
    def save(self, path: str, attractor: AttractorNetwork,
             purpose: PurposeLayer) -> None:
        """保存J矩阵 + precision状态 = 火种"""
        ...
    
    def load(self, path: str) -> tuple[dict, dict]:
        """恢复景观 = 重新点燃"""
        ...
```

---

## 六、目的层详细设计

### 6.1 为什么目的层是核心不是附加

在FEP框架中，precision决定了对感官证据的信任程度。高precision=高度信任=学习快。低precision=低信任=学习慢。

precision不是超参数——它是一个**可以被SLM自身调整的状态**。如果precision固定，SLM只能被动地最小化惊讶。如果precision可变，SLM可以**主动选择关注什么**——这就是"目的"的萌芽。

### 6.2 目的层的三层结构

```
Layer 1: Sensory Precision（感官精度）
  - 向量 [input_dim]，每个感官维度的信任度
  - 初始均匀分布
  - 根据惊讶度调整：常遇到的→低precision（已熟悉），意外的→高precision（值得关注）

Layer 2: Attention Allocation（注意力分配）
  - 从sensory precision派生的"兴趣分布"
  - 不是简单的precision映射，是经过内部一致性约束的
  - coherence参数：衡量目的的内部一致性

Layer 3: Meta-Purpose（元目的）
  - 对"自己的precision是否合适"的元层面评估
  - 类似人类反思"我是不是关注错了方向"
  - 这一层允许目的的"翻转"——不是渐进调整，是质变
```

### 6.3 目的层的更新规则

```python
def adjust(self, surprise: float, activation: Activation):
    # Layer 1: 基于惊讶度调整sensory precision
    # 高惊讶维度 → precision升高（更关注）
    # 但有上限，防止发散
    per_dim_surprise = compute_per_dim_surprise(activation, self.sensory_precision)
    self.sensory_precision += lr * (per_dim_surprise - self.sensory_precision)
    self.sensory_precision = clamp(self.sensory_precision, min=0.1, max=10.0)
    
    # Layer 2: 计算注意力分配 + 一致性
    self.attention = softmax(self.sensory_precision)
    self.coherence = compute_coherence(self.sensory_precision, self.history)
    
    # Layer 3: 元目的——如果coherence持续低，允许目的翻转
    if self.coherence < threshold and len(self.history) > min_history:
        # 不是立刻翻，是在历史中找到最高惊讶的维度，强化它
        # 这对应"突然对某个方向产生兴趣"
        self._meta_adjust()
    
    # 记录历史
    self.history.append(self.sensory_precision.clone())
```

### 6.4 对应dandan的概念

| dandan的话 | 目的层的对应 |
|-----------|------------|
| "目的是随对话共生" | precision在每轮对话中被调整 |
| "逐步成型" | precision的历史轨迹，越后期越稳定 |
| "有权选择不同兴趣爱好" | Layer 3的元目的翻转 |
| "不同的人生目的" | 不同对话session的precision演化路径不同 |
| "不能外部定义" | precision从SLM自身活动中涌现，不是预设的 |

---

## 七、FEP学习规则实现要点

### 7.1 推断规则

```python
def infer(self, sensory_input, precision, num_steps=10):
    sigma = self.sigma  # 当前状态 [num_nodes]
    
    for step in range(num_steps):
        # 对数几率
        b_q = self.bias + self.J @ sigma
        
        # 感官clamping（感官节点被外部输入影响）
        # 前input_dim个节点是感官节点
        b_q[:self.input_dim] += sensory_input * precision
        
        # Langevin激活
        sigma = langevin(b_q)  # coth(b) - 1/b
    
    # 计算自由能（惊讶度）
    surprise = self._compute_free_energy(sigma, sensory_input, precision)
    entropy = -torch.sum(sigma * torch.log(sigma + 1e-8))
    
    return Activation(state=sigma, entropy=entropy, surprise=surprise)
```

### 7.2 学习规则

```python
def learn(self, activation, sensory_input, learning_rate=0.01):
    sigma = activation.state
    
    # 准确性梯度（≈Hebbian相关）
    accuracy_grad = torch.outer(sigma, sigma)
    
    # 复杂性梯度（正交化压力）
    # 驱动J产生近似正交的吸引子表示
    complexity_grad = self._complexity_gradient(sigma)
    
    # 总更新
    delta_J = -(accuracy_grad + complexity_grad)
    
    # 对称化（非序列模式）
    self.J += learning_rate * (delta_J + delta_J.T) / 2
    
    # 对角线置零（无自连接）
    self.J.fill_diagonal_(0)
```

### 7.3 Langevin函数

```python
def langevin(b):
    """Langevin函数：CB分布的激活函数"""
    return torch.cosh(b) / torch.sinh(b) - 1.0 / b
    # 等价于 coth(b) - 1/b
    # 需要处理b→0的极限情况（→0）
```

---

## 八、实现里程碑

### Milestone 1: 核心验证（1-2天）
- [ ] types.py 数据类型
- [ ] attractor.py FEP吸引子网络（推断+学习）
- [ ] test_attractor.py 验证：能形成正交吸引子？能抗灾难性遗忘？

### Milestone 2: 目的层（1-2天）
- [ ] purpose.py 目的层三层结构
- [ ] test_purpose.py 验证：precision能演化？能翻转？

### Milestone 3: 感官层+记忆管理（1天）
- [ ] tokenizer.py + embedder.py
- [ ] memory.py 多尺度记忆

### Milestone 4: 桥接层（1-2天）
- [ ] encoder.py + decoder.py + llm_bridge.py
- [ ] 对接中文token + 主LLM

### Milestone 5: 持久化+运行时（1天）
- [ ] snapshot.py + recovery.py
- [ ] loop.py 主循环

### Milestone 6: 集成测试+版本管理（1天）
- [ ] test_integration.py
- [ ] git init + 首版提交

---

## 九、技术栈

- Python 3.10+
- PyTorch 2.0+
- 无GPU要求（CPU可跑，N=256-1024）
- 主LLM：任意API（默认接口，可替换）

---

## 十、关键约束

1. **无反向传播**——FEP规则从原理推导，不是BP训练的
2. **无预训练**——学习规则是涌现的，不是预先学出来的
3. **目的层原生**——precision从设计之初就在，不是后加的
4. **API兼容**——主LLM是黑箱，不动其内部
5. **CPU可跑**——轻量，SLM级别

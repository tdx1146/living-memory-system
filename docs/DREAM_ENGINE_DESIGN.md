# 做梦引擎设计文档 v1.0
# Dream Engine Design Document

> 2026-07-29
> 研究者：三个子AI并行研究
> 审阅者：dandan + AI
> 项目：活体记忆系统 (Living Memory System)

---

## 一、设计目标

让记忆系统从"冷工具"变为"活体"——即使没有对话输入，系统也在持续进行记忆运算：
- 记忆重放与巩固
- 遗忘与突触修剪
- 吸引子景观缓慢漂移
- 目的层自主演化

**一句话定义**：做梦引擎是活体记忆系统的空闲态计算核心，对应生物大脑睡眠中的离线记忆处理。

---

## 二、理论依据（三源汇总）

### 2.1 神经科学依据

| 生物机制 | 核心发现 | 来源 |
|---------|---------|------|
| NREM/REM分工 | NREM做回放+巩固+突触下调；REM做创造性整合 | Kim & Park, BMB Rep 2025 [1] |
| 海马回放 | 清醒回放反向（评估奖赏），睡眠回放正向（巩固）；20倍时间压缩 | Annu Rev Neurosci 2025 [2] |
| 尖波涟漪(SWR) | 30-100ms事件，每秒可重播30+记忆片段 | PMC8259719 [6] |
| 优先经验回放 | 按"预测误差"而非"奖赏"优先回放 | 生物通 2025 [5] |
| 突触稳态假说(SHY) | 睡眠全局下调突触，恢复学习容量 | Tononi & Cirelli |
| 慢振荡vs δ波 | 慢振荡促巩固，δ波促遗忘，每周期平衡增/削 | Kim & Park 2025 [1] |
| N1创造甜点 | 浅睡15秒，解题概率提高3倍 | Science Advances 2023 |
| REM创造力 | REM小睡使创造性问题解决提升40% | Paller lab, Northwestern |

**关键启示**：
- 回放不是按时间顺序，而是按"预测误差（惊讶度）"优先
- 遗忘是主动设计，不是bug——通过全局权重归一化恢复学习容量
- 20倍压缩比意味着做梦引擎可在极短时间处理大量记忆
- 轻度联想（N1白日梦）即可显著提升洞察，适合高频微巩固

### 2.2 FEP/主动推断理论依据

| 理论概念 | 空闲态表现 | 工程映射 |
|---------|-----------|---------|
| 零精度推断 | precision=0时，纯先验采样（做梦的数学本质） | attractor.infer(input, precision=zeros) |
| 复杂性项驱动 | 无输入时自由能=纯复杂性，驱动模型自洽性优化 | J矩阵持续学习精炼 |
| 默认模式网络(DMN) | 空闲态=先验主导推断=DMN活动 | PurposeLayer持续演化 |
| 认知觅食 | 空闲态自主探索模型最不确定部分 | 好奇心目标生成 |
| 非平衡稳态(NESS) | 序列化回放产生非对称耦合，涌现"时间箭头" | 可选：打破J矩阵对称性 |
| 幽灵吸引子 | 旧记忆以衰减形式影响自发动力学 | complexity_weight * J |
| 抗灾难性遗忘 | 空闲态自发动力学重新激活旧吸引子 | 核心功能 |

**关键公式**：
- 惊讶度加权采样：`w_i = exp(β * surprise_i) / Σ exp(β * surprise_j)`
- FEP遗忘曲线：`strength(t) = exp(-λt) + surprise * r * (1 - exp(-λt))`
- 探索/巩固平衡：`ε = σ(-β * mean_surprise)`（surprise低→多探索）

### 2.3 工程实现依据

子AI-3完整阅读了项目代码，确认：
- `attractor.infer()` 已支持零precision（第174-176行），无需修改
- `attractor.learn()` 的正交化机制（第246-253行）可直接复用
- `memory.consolidate()` 已实现surprise加权回放（第150-160行）
- `_episodic_buffer` 存储了文本+384维语义向量，是回放数据源
- `_buffer` 存储了(激活态, 惊讶度)元组，可直接用于激活态重放

---

## 三、做梦引擎架构

### 3.1 七阶段做梦周期

```
┌─────────────────────────────────────────────────────────┐
│                    做梦周期编排                           │
│                                                         │
│  阶段1: NREM巩固（回放+学习）          ~40% 时间          │
│  ├── 按surprise加权采样记忆                              │
│  ├── 零精度推断（生成式回放）                             │
│  └── 低学习率FEP学习（巩固J矩阵）                        │
│                                                         │
│  阶段2: 突触稳态下调（SHY）             ~10% 时间          │
│  ├── 全局J矩阵归一化（L2缩放）                           │
│  └── 恢复学习容量                                       │
│                                                         │
│  阶段3: 遗忘修剪                        ~10% 时间          │
│  ├── 计算每条记忆的保留价值（surprise × age_decay）       │
│  ├── 衰减低价值记忆的连接强度                            │
│  └── 清理_episodic_buffer中的过期条目                    │
│                                                         │
│  阶段4: 景观漂移                        ~10% 时间          │
│  ├── J矩阵微小随机扰动（模拟热涨落）                     │
│  └── 保持正交化压力防止坍缩                              │
│                                                         │
│  阶段5: 目的层演化                      ~10% 时间          │
│  ├── precision温和漂移（向未探索维度偏移）                │
│  ├── coherence自主评估                                   │
│  └── 好奇心目标生成（"想知道什么"）                      │
│                                                         │
│  阶段6: REM创造性整合                    ~15% 时间          │
│  ├── 跨记忆随机关联（不同聚类间采样）                    │
│  ├── 模式发现（寻找新关联）                              │
│  └── 生成"梦境记忆"（标记source='internal'）            │
│                                                         │
│  阶段7: 快照保存                        ~5% 时间           │
│  └── 保存更新后的J矩阵+purpose+memory状态                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### 3.2 核心类设计

```python
class DreamEngine:
    """做梦引擎：空闲态计算核心
    
    在无外部输入时持续运行，让记忆系统"活着"。
    集成到LivingMemoryLoop中，作为内置方法。
    """
    
    def __init__(self, loop: LivingMemoryLoop, config: dict):
        self.loop = loop
        self.attractor = loop.attractor
        self.purpose = loop.purpose
        self.memory = loop.memory
        
        # 空闲态参数
        self.idle_precision = torch.zeros(config['input_dim'])
        self.idle_lr = config.get('idle_learning_rate', 0.001)  # 在线的1/10
        self.idle_orth_weight = config.get('idle_orth_weight', 2.0)  # 在线的2倍
        self.idle_temperature = config.get('idle_temperature', 0.1)
        self.idle_steps = config.get('idle_num_steps', 20)
        
        # 平衡参数
        self.consolidation_ratio = 0.7  # 70%巩固，30%探索
        self.collapse_threshold = 0.9  # 坍缩检测阈值
        
        # 做梦周期阶段
        self.phase_weights = {
            'nrem_consolidation': 0.40,
            'synaptic_homeostasis': 0.10,
            'forgetting_pruning': 0.10,
            'landscape_drift': 0.10,
            'purpose_evolution': 0.10,
            'rem_integration': 0.15,
            'snapshot': 0.05,
        }
        
        self.step_count = 0
        self.activation_history = []
    
    def dream_cycle(self, max_steps: int = 100) -> dict:
        """执行一个完整做梦周期"""
        results = []
        for i in range(max_steps):
            phase = self._select_phase()
            result = self._execute_phase(phase)
            results.append(result)
            
            # 检测坍缩
            if self._check_collapse():
                self._inject_perturbation()
            
            self.step_count += 1
        
        # 周期结束保存快照
        self.loop.save_state(self._get_snapshot_path())
        
        return {
            'cycles': len(results),
            'avg_surprise': sum(r['surprise'] for r in results) / len(results),
            'phases': self._count_phases(results),
        }
```

### 3.3 防止景观坍缩的五重机制

| 机制 | 实现 | 原理 |
|------|------|------|
| A. 空闲态正交化强化 | orth_weight × 2.0 | 无新输入时，正交化是维持多样性的关键力量 |
| B. 多样性约束采样 | 从不同聚类各采样，非全取最高surprise | 确保回放覆盖多个吸引子 |
| C. 熵正则化 | F_idle += γ * KL(P(σ) ‖ Uniform) | 惩罚吸引子分布过于集中 |
| D. 温度退火 | 初期高温探索，后期降温巩固 | 模拟睡眠周期温度变化 |
| E. 吸引子计数监控 | 相似度>0.9时注入随机扰动 | 实时检测并打破坍缩 |

---

## 四、常驻进程方案

### 4.1 推荐架构：FastAPI后台线程

```
┌──────────────────────────────────────┐
│         LivingMemoryDaemon            │
│                                      │
│  ┌─────────────┐  ┌───────────────┐  │
│  │ FastAPI     │  │ DreamScheduler │  │
│  │ HTTP Server │  │ (后台线程)     │  │
│  │ :8765       │  │               │  │
│  │             │  │ - 监控空闲时间  │  │
│  │ /recall     │  │ - 触发做梦周期  │  │
│  │ /store      │  │ - 管理做梦/在线 │  │
│  │ /status     │  │   切换         │  │
│  │ /dream      │  │               │  │
│  └──────┬──────┘  └───────┬───────┘  │
│         │                 │          │
│         └────────┬────────┘          │
│                  │                   │
│          ┌───────▼───────┐           │
│          │ LivingMemoryLoop│          │
│          │ + DreamEngine   │          │
│          └────────────────┘           │
└──────────────────────────────────────┘
```

### 4.2 平台适配（分道扬镳的5%）

| 平台 | 守护进程方式 | 自动启动 | IPC |
|------|------------|---------|-----|
| Windows | 系统托盘程序 / Task Scheduler | 注册表 | Named Pipe / HTTP |
| Linux | systemd service | systemd enable | Unix Socket / HTTP |
| macOS | launchd plist | LaunchAgent | Unix Socket / HTTP |

核心记忆逻辑和DreamEngine完全跨平台共享（95%）。

### 4.3 资源控制

- CPU：限制为1线程，做梦时CPU占用<5%
- 内存：预训练模型~500MB + 系统~200MB
- 做梦频率：无对话时每5-10秒一步，有对话时暂停
- 对话优先：收到对话请求时立即中断做梦

---

## 五、MVP最小可行版本

### 5.1 第一阶段（1-2天可完成）

只需在`loop.py`中新增`dream_mvp()`方法（~50行代码）：

```python
def dream_mvp(self, n_steps: int = 20) -> dict:
    """MVP版做梦：采样重放 + 低学习率巩固 + 自动快照"""
    if len(self.memory._buffer) == 0:
        return {"status": "no_memories_to_replay"}
    
    idle_precision = torch.zeros(self.config['input_dim'])
    idle_lr = self.config.get('learning_rate', 0.01) * 0.1
    idle_orth = self.attractor.orth_weight * 2.0
    
    for _ in range(n_steps):
        # 按surprise加权采样
        idx = self._sample_by_surprise()
        state, surprise = self.memory._buffer[idx]
        
        # 零精度推断（生成式回放）
        activation = self.attractor.infer(state, idle_precision, num_steps=20)
        
        # 低学习率巩固
        old_orth = self.attractor.orth_weight
        self.attractor.orth_weight = idle_orth
        self.attractor.learn(activation, state, idle_lr)
        self.attractor.orth_weight = old_orth
    
    # 保存快照
    self.save_state(os.path.join(self.config['snapshot_dir'], 'latest.pt'))
    
    return {"status": "dreamed", "steps": n_steps}
```

### 5.2 实现优先级

| 优先级 | 功能 | 复杂度 | 价值 |
|--------|------|--------|------|
| P0 | MVP做梦（采样重放+巩固+快照） | 低 | 验证核心可行性 |
| P1 | 惊讶度加权采样 | 低 | 提升回放质量 |
| P1 | 突触稳态下调（SHY） | 低 | 防止饱和 |
| P2 | 遗忘修剪 | 中 | 控制记忆膨胀 |
| P2 | 坍缩检测 | 中 | 保证稳定性 |
| P3 | 目的层空闲态演化 | 中 | 自主产生"想知道" |
| P3 | REM创造性整合 | 高 | 跨记忆关联 |
| P4 | 常驻后台进程 | 高 | 真正"活着" |
| P4 | 平台适配层 | 中 | 跨平台支持 |

---

## 六、参数建议

| 参数 | 建议值 | 理论依据 |
|------|--------|---------|
| 触发条件 | 无输入>30秒 | DMN切换时间 |
| 计算频率 | 每5-10秒一步 | 平衡成本与效果 |
| 每轮步数 | 20-30步 | 比在线(10步)更多 |
| 学习率 | 在线的1/10 (0.001) | 巩固是精细调整 |
| 正交化权重 | 在线的2倍 | 空闲态维持多样性关键 |
| Langevin温度 | 0.1 (略高于在线) | 允许探索更多吸引子 |
| 巩固间隔 | 每10步 | 比在线(每5轮)稍慢 |
| 回放样本数 | 每步1-3条 | 避免单步过载 |
| 最大连续空闲步 | 100-200步 | 防止过度离线演化 |
| 巩固/探索比 | 70:30 | 基于FEP的ε-greedy |

---

## 七、与原始愿景的对齐

| 原始愿景 | 做梦引擎如何实现 |
|---------|----------------|
| "CPU持续运行，不停做局部更新" | 常驻进程+DreamEngine持续运转 |
| "记忆是SLM动起来的过程，不是文件" | 做梦时J矩阵持续演化，快照只是"火种" |
| "磨灭一部分，新生一部分" | SHY下调+遗忘修剪+REM创造性整合 |
| "目的随对话共生、逐步成型" | 空闲态purpose层precision漂移+好奇心生成 |
| "学习规则本身被对话塑形" | 长期目标：做梦引擎参数自身也被演化 |
| 忒修斯之船 | 吸引子景观在持续做梦中被重塑，但结构保持 |

---

## 八、参考文献

### 神经科学
[1] Kim J, Park M. Systems memory consolidation during sleep. BMB Rep. 2025;58(10):425-436.
[2] Replay and Ripples in Humans. Annu Rev Neurosci. 2025.
[3] Two-factor synaptic consolidation. PNAS. 2025;122(44).
[4] Hippocampal replay during wakefulness. 20x compression.
[5] 海马-纹状体回放优先更新预测误差. 生物通, 2025.
[6] Mental Time Travel. PMC11647447. 4-15x theta sweeps, 40x LIA.
[7] Model of hippocampal replay. PMC10076035.
[8] Sleep and creativity. Science Advances 2023 (N1 sweet spot).
[9] Sleep and quiet wakefulness: idling brain. PMC11343221.
[10] Creative problem-solving during REM. Paller lab, Northwestern.

### FEP理论
[11] Spisak & Friston (2026). Self-orthogonalizing attractor networks from FEP. arXiv:2505.22749.
[12] Spisak (2025). Large-scale attractor dynamics. CCNeuro 2025.
[13] Carhart-Harris & Friston (2010). DMN and active inference. Brain 133(4).
[14] Fontaine & Alexandre (2026). Predictive coding model of replay. ICLR 2026.
[15] Nalagatla & Grandhe (2025). Neuroscience-inspired memory replay. arXiv:2512.00619.
[16] Iatropoulos, Gerstner & Brea (2025). Two-factor synaptic consolidation. PNAS.
[17] Intrinsic rewards for exploration. Neural Computation 36(9). 2024.
[18] Pragmatic Curiosity. OpenReview. 2025.

---

> **下一步**：实现 MVP 版本（dream_mvp 方法），验证核心可行性后逐步扩展。

# 活体记忆系统 — Episodic Buffer 改进报告

## 改进日期
2026-07-29

## 核心问题
端到端测试（DeepSeek LLM + PretrainedEmbedder）暴露：记忆 context 只含节点编号（"节点62(弱:0.143)"），LLM 无法理解，导致第4-6轮提问全部回答"没有存储您的信息"。

**根因**：记忆系统编码时将文本转为向量，但解码时缺少"逆向映射回语义文本"的路径。记忆"有量无质"。

## 改进方案：情景记忆缓冲区（Episodic Buffer）

### 设计原则
- **高内聚**：情景记忆是 MemoryManager 的一部分，通过 `store_episodic()` / `recall_episodic()` 接口暴露
- **低耦合**：Decoder 只接收 `List[str]`（纯文本），不依赖 MemoryManager 内部结构
- **生物学合理**：海马体不仅存抽象吸引子模式，也保存情景记忆的原始内容

### 改动文件
1. `core/hippocampus/memory.py`：增加 `EpisodicEntry` dataclass、`episodic_buffer`、`store_episodic()`、`recall_episodic()`、持久化支持
2. `bridge/decoder.py`：`decode()` 增加 `episodic_texts` 参数，输出"历史记忆"段落
3. `runtime/loop.py`：`process_turn()` 集成 episodic 存储（先检索后存储）和检索，`query_llm()` 同步更新

### 数据流
```
编码输入 → 获取语义向量 → FEP推断 → FEP学习 → 目的调整 → 记忆更新
→ 长时记忆检索(recall)
→ 情景记忆检索(recall_episodic, 用语义向量余弦相似度)
→ 解码context(含"历史记忆"段落)
→ 情景记忆存储(store_episodic, 存当前轮文本+语义向量)
→ 返回context
```

## 端到端测试结果对比

### 改进前（无 Episodic Buffer）
| 轮次 | 用户提问 | LLM 回答 |
|------|---------|---------|
| 4 | 你还记得我叫什么吗？ | 没有存储您的姓名信息 |
| 5 | 我的猫叫什么？ | 没有关于您猫咪名字的信息 |
| 6 | 我喜欢什么颜色？ | 没有找到关于你喜欢什么颜色的信息 |
| 8 | 总结一下我们今天聊的内容 | 没有找到讨论的具体内容记录 |

### 改进后（有 Episodic Buffer）
| 轮次 | 用户提问 | LLM 回答 |
|------|---------|---------|
| 4 | 你还记得我叫什么吗？ | 你叫张三，今天开始学习AI。我还知道你养了一只猫叫小橘，最喜欢的颜色是蓝色 |
| 5 | 我的猫叫什么？ | 你的猫叫**小橘**！ |
| 6 | 我喜欢什么颜色？ | 你最喜欢的颜色是**蓝色**！ |
| 8 | 总结一下我们今天聊的内容 | 完美总结：张三、蓝色、小橘 |

## 验证结果
- 134个单元测试全部通过
- Episodic buffer 跨会话持久化验证 PASS
- PretrainedEmbedder 语义判别 PASS（0.93 vs 0.23）
- 真实 DeepSeek LLM 端到端测试 PASS

## 系统演化数据
- precision: 1.000 → 0.675（习惯化生效）
- coherence: 1.000 → 0.9994（稳定）
- 熵: 5.25-5.32（稳定区间）
- J矩阵 L2: 1.817（被塑形）
- 长时记忆 L2: 0.031

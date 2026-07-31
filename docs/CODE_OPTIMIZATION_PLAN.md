# 活体记忆系统 代码优化方案

> 版本：v1.0 ｜ 对应代码状态：249 测试全通过，核心功能已闭环
> 适用范围：`core/`、`api/`、`runtime/`、`persistence/`、`bridge/`、工程基建

## 项目现状概述

活体记忆系统是一个基于自由能原理（FEP）的吸引子神经网络记忆系统，外挂于主 LLM。当前已实现 FEP 推断/学习、目的层、多尺度记忆管理、做梦引擎（七阶段周期）、元可塑性、MCP server 与 FastAPI 服务，249 个测试全部通过，功能层面已形成闭环。

三个探索子 AI 对全量代码做了系统性审查，共归纳出 57 项发现，覆盖性能、架构耦合、并发安全、错误恢复、测试缺口、依赖管理、CI/CD、安全与部署九个维度。其中并发数据竞争与 API Key 泄露属于必须在合入主干前处理的高危项，性能与解耦类问题适合在本轮迭代集中收口，容器化与文档类工作可后续规划。

本方案按优先级将优化项分为 P0（立即修复，阻塞发布）、P1（本轮迭代，1-2 周内完成）、P2（后续规划）三档，每项给出问题描述、影响分析、可执行的优化方案与工作量预估，并在末尾给出分阶段执行路线图。工作量档位：XS (<1h)、S (1-4h)、M (4-16h)、L (16-40h)。

---

## P0 立即修复（阻塞发布）

P0 为安全与数据完整性红线问题，必须在任何新功能合入前完成，且修复后需立即轮换已泄露的凭据。

### P0-1 API Key 泄露修复（轮换 + 环境变量化 + git 历史清理）

**问题描述**

`memory_cli.py` 第 34-35 行通过 `os.environ.setdefault` 硬编码了 DeepSeek API Key 明文：

```python
os.environ.setdefault(
    "DEEPSEEK_API_KEY", "sk-d91a6339112040a98c6f0617e6142307")
```

同一 Key 还出现在 `mcp.json` 中，且已经进入 git 提交历史。

**影响分析**

该 Key 已随 git 历史永久暴露，任何能访问仓库的人可直接盗用 DeepSeek 配额，造成计费损失与潜在数据风险。即使删除当前文件内容，历史 commit 仍可被检索，单纯删除文件无法消除风险。`setdefault` 的默认值语义意味着即使运维未配置环境变量，系统也会用泄露的 Key 继续运行，掩盖配置缺失。

**优化方案**

分三步执行，顺序不可颠倒：

1. 先在 DeepSeek 控制台吊销 `sk-d91a...2307` 并生成新 Key，新 Key 不落盘到任何代码文件。
2. 清理代码中的硬编码，改为强制从环境变量读取，缺失时直接报错而非静默用默认值：

```python
# memory_cli.py 顶部，替换原 setdefault 逻辑
import os
import sys

# 不再提供默认值；未配置环境变量时直接退出，避免静默使用空 Key
_DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not _DEEPSEEK_API_KEY:
    # CLI 场景下打印到 stderr 后以非零码退出
    sys.stderr.write(
        "[memory_cli] 未配置 DEEPSEEK_API_KEY 环境变量，"
        "无法初始化 LLM Bridge。\n"
        "请复制 .env.example 为 .env 并填入有效 Key。\n")
    # 仅在直接执行 CLI 时退出；被 import 时不阻塞（保留 _INIT_OK 机制判断）
    if __name__ == "__main__":
        sys.exit(2)
```

3. 清理 git 历史，新增 `.env.example` 模板与 `.gitignore`：

```bash
# 使用 git filter-repo 清理历史中的 Key（比 filter-branch 更安全高效）
pip install git-filter-repo
git filter-repo --replace-text <(echo "sk-d91a6339112040a98c6f0617e6142307==>***REDACTED***")
git push origin --force --all
```

```ini
# .env.example（提交到仓库作为模板，真实 .env 被 .gitignore 忽略）
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
```

```gitignore
# .gitignore 追加
.env
*.key
snapshots/
```

4. 在 pre-commit 钩子中加入密钥扫描，防止再次泄露（见 P1-5 CI/CD）：

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/Yelp/detect-secrets
    rev: v1.4.0
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

**预估工作量**：M（git 历史重写需在备份分支上验证，含强制推送与协作者同步通知）

---

### P0-2 并发数据竞争修复（acquire 返回值检查 + async 端点 run_in_executor）

**问题描述**

存在两个独立但叠加后会导致数据损坏的并发问题：

1. `api/server.py` 第 196 行调用 `scheduler.acquire_conversation(req.session_id)` 后不检查返回值。`dream_scheduler.py` 第 168-174 行显示该方法在超时后返回 `False` 并打印"强制继续"日志，但调用方无视该返回值，直接执行 `loop.process_turn()`，导致对话处理与后台做梦并发写同一份 J 矩阵 / precision / latent。
2. `/chat`（第 200 行 `process_turn`）与 `/chat/simple`（第 251 行 `query_llm`）是 `async def`，却直接调用含 torch 推断 + LLM 网络请求的同步阻塞方法，未包裹 `run_in_executor`，阻塞 FastAPI 事件循环。对比 `/dream/{session_id}`（第 413 行）已正确使用 `run_in_executor`。

**影响分析**

问题 1 在做梦超时（默认 10s）后必然触发，对话与做梦同时修改 `attractor.J`、`purpose.sensory_precision`、`memory.long_term_latent`，产生写写竞争，记忆状态可能进入不可恢复的半损坏态。问题 2 在单条对话耗时超过百毫秒时就会饿死后台任务（如 `/dream/status` 轮询、健康检查），并发吞吐降为串行。

**优化方案**

问题 1：检查返回值，超时则返回 503，绝不进入 process_turn：

```python
# api/server.py  chat() 内，替换原 196 行附近
acquired = scheduler.acquire_conversation(req.session_id)
scheduler.touch(req.session_id)
if not acquired:
    # 做梦未结束且等待超时，拒绝执行以避免数据竞争
    raise HTTPException(
        status_code=503,
        detail="系统正在做梦（记忆巩固中），请稍后重试。",
    )
try:
    memory_context = await asyncio.get_event_loop().run_in_executor(
        None, lambda: loop.process_turn(req.user_input, req.llm_output))
    # ... 后续 LLM 查询同样放线程池 ...
finally:
    scheduler.release_conversation(req.session_id)
```

问题 2：所有阻塞调用统一走 `run_in_executor`，与 `/dream/{session_id}` 对齐：

```python
# /chat/simple 端点
import asyncio
loop_exec = asyncio.get_event_loop()
try:
    response_text = await loop_exec.run_in_executor(
        None, lambda: loop.query_llm(req.user_input))
except HTTPException:
    raise
except Exception as e:
    logger.error(f"[{req.session_id}] (simple) 查询失败: {e}")
    raise HTTPException(status_code=500, detail="LLM 查询失败")
```

同时修复 `dream_scheduler.py` 中 `_is_dreaming` / `_dreaming_session` 无锁读取问题（第 401-403 行 `get_status` 读，第 296-297 行写）：用 `self._lock` 保护读写，或改为 `threading.Event` + 原子赋值。

**预估工作量**：M（需回归测试对话与做梦并发场景，验证 503 路径与事件循环不再阻塞）

---

### P0-3 快照原子写入

**问题描述**

`persistence/snapshot.py` 第 139 行 `torch.save(data, path)` 直接写目标文件，`core/hippocampus/dream_engine.py` 第 915 行 `_save_snapshot` 同样直接写 `latest.pt`。写入过程中若进程崩溃（OOM、kill、断电），目标文件会被截断，原有有效快照被覆盖为半写文件，下次 `load` 抛 `EOFError`。

**影响分析**

快照是系统"火种"（J 矩阵 + precision + 记忆潜变量），损坏意味着积累的身份记忆丢失。做梦引擎每个周期末（阶段7）都会写 `latest.pt`，写入频率高，崩溃窗口客观存在。`recovery.validate` 虽能检测损坏，但损坏后无可回退的有效快照。

**优化方案**

采用 tmp 文件 + `os.replace()` 原子替换。`os.replace` 在同一文件系统上是原子的 POSIX/Windows 系统调用，要么完整可见要么不可见：

```python
# persistence/snapshot.py  save() 末尾，替换 torch.save(data, path)
import os
import tempfile

dir_path = os.path.dirname(os.path.abspath(path))
if dir_path:
    os.makedirs(dir_path, exist_ok=True)

# 写到同目录临时文件（必须同目录，否则 os.replace 跨文件系统会退化为非原子拷贝）
fd, tmp_path = tempfile.mkstemp(
    prefix=".snap_", suffix=".tmp", dir=dir_path or ".")
try:
    with os.fdopen(fd, "wb") as f:
        torch.save(data, f)  # 写到文件句柄而非路径
    os.replace(tmp_path, path)  # 原子替换
except Exception:
    # 写失败时清理临时文件，原快照保持完整
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    raise
logger.info(f"快照已原子保存到 {path}")
```

`dream_engine.py` 的 `_save_snapshot` 应复用同一逻辑，或直接调用 `persistence.snapshot.Snapshot.save`（需先解开 core 不反向依赖 persistence 的约束，可在 runtime 层注入保存函数）。短期方案：在 dream_engine 内复制上述原子写入逻辑。

**预估工作量**：S

---

### P0-4 setup.py 依赖修复

**问题描述**

`setup.py` 第 23-27 行 `install_requires` 仅声明 `torch`、`numpy`、`openai`，遗漏 `fastapi`、`uvicorn`、`sentence-transformers`、`mcp`。`pip install -e .` 后运行 `api/server.py` 会因 `ModuleNotFoundError: No module named 'fastapi'` 启动失败。

**影响分析**

新环境部署必现失败，且 `requirements.txt` 全部用 `>=` 无版本锁定，CI 与不同机器可能装到不兼容版本，复现性差。`mcp` 依赖在 `requirements.txt` 和 `setup.py` 中均未声明，MCP server 无法开箱运行。

**优化方案**

补全 `install_requires`，并分离生产/开发依赖；同时新增 `requirements.lock`（用 `pip-compile` 生成）锁定可复现版本：

```python
# setup.py
install_requires=[
    "torch>=2.0",
    "numpy>=1.21",
    "openai>=1.0",
    "fastapi>=0.100",
    "uvicorn[standard]>=0.20",
    "sentence-transformers>=2.2",
    "mcp>=1.0",
],
extras_require={
    "dev": [
        "pytest>=7.0",
        "pytest-asyncio>=0.21",
        "httpx>=0.24",        # TestClient 依赖
        "pytest-cov>=4.0",
        "detect-secrets>=1.4",
    ],
},
```

```bash
# 生成锁文件，提交到仓库保证 CI 复现
pip install pip-tools
pip-compile requirements.in -o requirements.lock
pip install -r requirements.lock
```

`requirements.txt` 同步补上 `mcp>=1.0`、`fastapi>=0.100`、`uvicorn[standard]>=0.20`、`sentence-transformers>=2.2`。

**预估工作量**：S

---

## P1 本轮迭代（1-2 周）

P1 聚焦性能、架构解耦与工程基建补齐，是本轮迭代的主体工作量。

### P1-1 recall_episodic 向量化优化

**问题描述**

`core/hippocampus/memory.py` 第 269-293 行 `recall_episodic` 逐条遍历缓冲区，每条做一次 `torch.dot` + `.item()` 同步。缓冲区满载 200 条时，单次检索产生 200 次 Python 循环 + 200 次 `.item()` GPU/CPU 同步调用。

**影响分析**

`.item()` 每次都强制同步，是性能杀手。在 GPU 上 200 次同步开销远超计算本身；即使 CPU，Python 循环与张量逐次创建也有可观开销。该方法是每轮对话检索情景记忆的热路径，直接决定对话首 token 延迟。审查预估向量化后可提升 10-100 倍。

**优化方案**

按维度分组（兼容 384 维 + 64 维混合缓冲区），每组 `torch.stack` 成矩阵后一次矩阵乘法算完所有相似度，消除逐条 `.item()`：

```python
def recall_episodic(self, query_vector: torch.Tensor,
                    top_k: int = 3,
                    fallback_query: Optional[torch.Tensor] = None
                    ) -> List[EpisodicEntry]:
    if len(self._episodic_buffer) == 0:
        return []

    # 按维度分组（支持混合维度缓冲区：384 维 + 旧 64 维共存）
    entries = list(self._episodic_buffer)
    dim_groups: dict = {}  # {dim: [(global_idx, vec), ...]}
    for idx, entry in enumerate(entries):
        v = entry.semantic_vector.detach().cpu().float()
        if v.dim() > 1:
            v = v.squeeze()
        dim_groups.setdefault(v.shape[-1], []).append((idx, v))

    # 准备归一化查询向量
    query = query_vector.detach().cpu().float()
    if query.dim() == 1:
        query = query.unsqueeze(0)
    qd = query.shape[-1]
    query_norm = torch.nn.functional.normalize(query, dim=-1).squeeze(0)

    fb_norm = None
    fd = None
    if fallback_query is not None:
        fb = fallback_query.detach().cpu().float()
        if fb.dim() == 1:
            fb = fb.unsqueeze(0)
        fd = fb.shape[-1]
        fb_norm = torch.nn.functional.normalize(fb, dim=-1).squeeze(0)

    # 每组一次矩阵乘法，替代逐条 dot + .item()
    scored: List[Tuple[float, EpisodicEntry]] = []
    for vd, group in dim_groups.items():
        if vd == qd:
            q = query_norm
        elif fb_norm is not None and vd == fd:
            q = fb_norm
        else:
            continue  # 维度均不匹配，整组跳过
        mat = torch.stack([v for (_, v) in group])          # [n, vd]
        mat_norm = torch.nn.functional.normalize(mat, dim=-1)
        sims = (mat_norm @ q).tolist()                       # 一次算完 [n]
        for sim, (gidx, _) in zip(sims, group):
            scored.append((sim, entries[gidx]))

    if not scored:
        return []
    k = min(top_k, len(scored))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in scored[:k]]
```

200 条同维度时从 200 次 matmul 调用降为 1 次 `[200, 384] @ [384]`，`.item()` 同步从 200 次降为 1 次 `tolist()`。需补充基准测试用例断言向量化结果与逐条版本一致（top_k 排序不变）。

**预估工作量**：M（含等价性回归测试与基准对比脚本）

---

### P1-2 DreamEngine 解耦（移除私有属性访问）

**问题描述**

`DreamEngine` 大量直接访问 `MemoryManager` 私有属性：`self.memory._buffer`（`dream_engine.py` 第 183、256、280、340、362、402、588 行）、`self.memory._episodic_buffer`（第 363、606、610、617 行）。`api/server.py` 第 151 行 `_status_for`、`api/session_manager.py` 第 119 行 `get_status`、`runtime/dream_scheduler.py` 第 281 行同样访问 `loop.memory._buffer` / `_episodic_buffer`。此外 `dream_engine.py` 第 672-683 行 `_purpose_evolve` 直接改写 `self.purpose.sensory_precision` 绕过封装。

**影响分析**

私有属性访问破坏封装边界，`MemoryManager` 内部数据结构（deque）一旦调整，DreamEngine、server、scheduler 三处全部需同步修改，耦合面被放大到整个 runtime 层。`_purpose_evolve` 直接改 precision 后还需手动重算 attention（第 682 行），逻辑散落在调用方，目的层一旦增加后处理约束就会被绕过。

**优化方案**

在 `MemoryManager` 上补齐只读访问接口，DreamEngine 与上层全部改用公开接口：

```python
# core/hippocampus/memory.py 新增公开接口
def buffer_size(self) -> int:
    """回放缓冲区当前条目数。"""
    return len(self._buffer)

def episodic_size(self) -> int:
    """情景记忆缓冲区当前条目数。"""
    return len(self._episodic_buffer)

def iter_buffer(self):
    """迭代回放缓冲区 (state, surprise) 的只读视图。"""
    return iter(self._buffer)  # deque 迭代器是只读的，调用方不可原地修改

def iter_episodic(self):
    """迭代情景记忆条目的只读视图。"""
    return iter(self._episodic_buffer)

def get_episodic_maxlen(self) -> int:
    return self._episodic_buffer.maxlen or 200

def replace_episodic_buffer(self, entries) -> int:
    """用过滤后的条目重建情景缓冲区（供遗忘修剪使用），返回剔除数。"""
    maxlen = self._episodic_buffer.maxlen or 200
    old_len = len(self._episodic_buffer)
    self._episodic_buffer = deque(entries, maxlen=maxlen)
    return old_len - len(self._episodic_buffer)
```

DreamEngine 内 `len(self.memory._buffer)` 改 `self.memory.buffer_size()`，`list(self.memory._buffer)` 改 `list(self.memory.iter_buffer())`，`_forgetting_pruning` 重建缓冲区改调 `replace_episodic_buffer`。`server.py` / `session_manager.py` / `dream_scheduler.py` 同步改用 `episodic_size()` / `buffer_size()`。

`_purpose_evolve` 的越权改写改为在 `PurposeLayer` 上新增公开方法：

```python
# core/hippocampus/purpose.py 新增
def nudge_low_encounter_dim(self, nudge: float) -> None:
    """向 encounter_count 最低的维度偏移 precision（好奇心萌芽），封装内部 clamp 与 attention 重算。"""
    if self.encounter_count.numel() == 0:
        return
    target_dim = int(self.encounter_count.argmin().item())
    self.sensory_precision[target_dim] += nudge
    self.sensory_precision = torch.clamp(
        self.sensory_precision, self.precision_min, self.precision_max)
    self.attention = torch.softmax(self.sensory_precision, dim=0)
```

DreamEngine `_purpose_evolve` 改为 `self.purpose.nudge_low_encounter_dim(self.purpose_evolve_nudge)`。

**预估工作量**：M（涉及 4 个文件机械替换 + 全量回归测试确认做梦与状态查询行为不变）

---

### P1-3 CoreConfig 校验 + 统一配置体系

**问题描述**

`core/config.py` 的 `CoreConfig` dataclass 缺少 `__post_init__` 校验，错误配置（如 `short_term_decay=1.5`、`precision_min > precision_max`）会延迟到运行时产生晦涩异常。同时存在两套配置体系：`CoreConfig`（dataclass）与 DreamEngine 接收的 `config: dict`（第 75 行），`phase_weights`（第 139-147 行）硬编码不从 config 读取，DreamEngine 字段未纳入 `CoreConfig`。

**影响分析**

缺校验导致配置错误难定位，例如 `precision_min=10, precision_max=0.1` 会让 `torch.clamp` 行为反常但无报错。两套配置体系使参数来源不一致，`phase_weights` 调整需改代码重发版。

**优化方案**

为 `CoreConfig` 加 `__post_init__`，并将 DreamEngine 配置纳入同一 dataclass：

```python
from dataclasses import dataclass, field

@dataclass
class DreamEngineConfig:
    """做梦引擎配置（统一纳入 CoreConfig，消除 dict 双轨制）。"""
    idle_learning_rate: float = 0.001
    idle_orth_weight: float = 1.0
    idle_temperature: float = 0.1
    idle_num_steps: int = 20
    consolidation_ratio: float = 0.7
    collapse_threshold: float = 0.9
    max_idle_steps: int = 200
    snapshot_dir: str = "snapshots"
    surprise_beta: float = 5.0
    shy_target_norm: float = 10.0
    drift_scale: float = 0.001
    collapse_window: int = 5
    forget_prune_rate: float = 0.005
    forget_max_age: int = 50
    purpose_evolve_nudge: float = 0.05
    # 七阶段权重（可调，运行期通过 config 修改）
    phase_weights: dict = field(default_factory=lambda: {
        "nrem_consolidation": 0.40,
        "synaptic_homeostasis": 0.10,
        "forgetting_pruning": 0.10,
        "landscape_drift": 0.10,
        "purpose_evolution": 0.10,
        "rem_integration": 0.15,
        "snapshot": 0.05,
    })

@dataclass
class CoreConfig:
    # ... 原有字段 ...
    dream: DreamEngineConfig = field(default_factory=DreamEngineConfig)

    def __post_init__(self):
        # 区间校验：失败时抛 ValueError，定位到具体字段
        if not 0.0 < self.short_term_decay < 1.0:
            raise ValueError(
                f"short_term_decay 须在 (0,1)，当前 {self.short_term_decay}")
        if not 0.0 < self.long_term_decay <= 1.0:
            raise ValueError(
                f"long_term_decay 须在 (0,1]，当前 {self.long_term_decay}")
        if self.precision_min >= self.precision_max:
            raise ValueError(
                f"precision_min({self.precision_min}) 须 < precision_max({self.precision_max})")
        if self.input_dim > self.num_nodes:
            raise ValueError(
                f"input_dim({self.input_dim}) 不能大于 num_nodes({self.num_nodes})")
        if not 0.0 < self.temperature < 1.0:
            raise ValueError(
                f"temperature 须在 (0,1)，当前 {self.temperature}")
        # phase_weights 之和应接近 1.0
        pw_sum = sum(self.dream.phase_weights.values())
        if abs(pw_sum - 1.0) > 1e-6:
            raise ValueError(f"phase_weights 之和须为 1.0，当前 {pw_sum}")
```

DreamEngine 构造函数改为接收 `DreamEngineConfig` 实例（或从 `CoreConfig.dream` 取），移除 `config: dict` 入参与硬编码 `phase_weights`。runtime 层负责 `CoreConfig -> dict` 的单向转换，保留 dict 兼容旧入口，但单一真相源是 dataclass。

**预估工作量**：M

---

### P1-4 API 层测试补齐（TestClient 方案）

**问题描述**

`api/server.py` 含 9 个 HTTP 端点与 startup/shutdown 生命周期，完全无测试覆盖。`mcp_memory_server.py` 仅 8 个手动 `print` 测试无 pytest 断言，`memory_cli.py`、`runtime/cli.py` 同样无测试。

**影响分析**

P0-2 修复并发问题后必须有用例守护"acquire 超时返回 503"与"run_in_executor 不阻塞"行为，否则回归无门。startup 生命周期（调度器启动/停止）无测试，关闭时序错误可能导致线程泄漏。

**优化方案**

用 `fastapi.testclient.TestClient` 同步测试（TestClient 内部跑事件循环，无需 async 测试框架），并通过依赖注入隔离全局单例：

```python
# tests/test_api_server.py
import pytest
from fastapi.testclient import TestClient
import api.server as srv

@pytest.fixture
def client(monkeypatch):
    """隔离全局单例，每个用例独立 SessionManager。"""
    srv._session_manager = None
    srv._dream_scheduler = None
    # 关闭真实做梦后台线程，避免测试期间触发做梦
    monkeypatch.setattr(
        "runtime.dream_scheduler.DreamScheduler.start", lambda self: None)
    with TestClient(srv.app) as c:
        yield c
    srv._session_manager = None
    srv._dream_scheduler = None

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_chat_empty_input_rejected(client):
    r = client.post("/chat", json={"user_input": ""})
    assert r.status_code == 400

def test_chat_acquire_timeout_returns_503(client, monkeypatch):
    # 模拟做梦占用锁导致 acquire 超时
    sm = srv.get_session_manager()
    sched = srv.get_dream_scheduler()
    sched._busy_lock.acquire()  # 占住锁模拟做梦中
    monkeypatch.setattr(sched, "acquire_conversation",
                        lambda sid, timeout=10: False)
    r = client.post("/chat", json={"user_input": "hi"})
    assert r.status_code == 503
    sched._busy_lock.release()

def test_sessions_lifecycle(client):
    client.post("/chat", json={"user_input": "hello", "session_id": "s1"})
    r = client.get("/sessions")
    assert "s1" in r.json()["sessions"]
    r = client.delete("/sessions/s1")
    assert r.json()["deleted"] is True
```

`mcp_memory_server.py` 的 8 个 print 测试改造为 pytest 断言函数，`memory_cli.py` 用 `subprocess` 跑 CLI 并断言 JSON 输出 `status` 字段。目标覆盖率达到 server.py 行覆盖 ≥80%。

**预估工作量**：M

---

### P1-5 CI/CD 搭建（GitHub Actions）

**问题描述**

项目完全无 CI/CD：无 `.github/workflows/`，无自动化测试、代码质量检查、pre-commit。无 `mypy.ini` / `.flake8` 类型与风格检查。

**影响分析**

249 个测试只能靠本地手动运行，回归依赖个人自觉；类型错误、风格漂移、密钥泄露无自动化拦截。P0-1 的密钥扫描若没有 CI 守护，仍可能再次合入。

**优化方案**

新增 GitHub Actions 工作流，覆盖测试、lint、类型检查、密钥扫描：

```yaml
# .github/workflows/ci.yml
name: CI
on:
  push:
    branches: [master, main]
  pull_request:
    branches: [master, main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r requirements.lock
      - run: pip install -e ".[dev]"
      - name: Run tests
        run: pytest tests/ -q --cov=. --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        if: matrix.python-version == '3.11'

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install flake8 mypy detect-secrets
      - run: flake8 core api runtime persistence bridge --max-line-length=100
      - run: mypy core/api --ignore-missing-imports || true  # 渐进式启用
      - name: Secret scan
        run: detect-secrets scan --baseline .secrets.baseline
```

配套 `.flake8`、`mypy.ini`、`.pre-commit-config.yaml`（含 `flake8`、`detect-secrets`、`end-of-file-fixer`）。CI 失败禁止合并（分支保护规则）。

**预估工作量**：M（含 mypy 渐进式启用与现有类型问题修复）

---

### P1-6 recover 异常链保留 + 部分恢复回滚

**问题描述**

`persistence/recovery.py` 第 160-162 行 `recover()` 用 `except Exception as e: ... return False` 吞掉所有异常，丢失根因堆栈。且恢复过程无回滚：若 `_restore_attractor` 成功但 `_restore_purpose` 失败（第 145 行），系统停留在 attractor 已更新、purpose 未更新的半恢复不一致态。`validate`（第 277 行）、`load`（第 139 行）、`load_raw`（第 149 行）对同一快照文件重复读取 3 次。`_is_version_compatible`（第 327-335 行）解析了 minor 但未比较，只要 major 匹配即通过。

**影响分析**

吞异常导致恢复失败时只能看到 `恢复失败: <str>`，无法定位是反序列化失败还是字段缺失，排障成本高。半恢复态比完全失败更危险——系统以为恢复成功继续运行，但 precision 与 J 矩阵来自不同版本快照，推断结果不可预测。

**优化方案**

1. 保留异常链，用 `raise ... from e`：
2. 恢复前快照当前状态，失败时回滚；
3. 单次读取快照复用：

```python
def recover(self, path, attractor, purpose, memory=None) -> bool:
    # 单次读取，复用给 validate/load/restore，避免 3 次磁盘 IO
    try:
        raw_data = _torch_load(path)
    except Exception as e:
        raise RuntimeError(f"快照加载失败: {path}") from e

    if not self._validate_data(raw_data):
        raise ValueError(f"快照校验失败: {path}")

    # 快照当前状态用于回滚（深拷贝景观/目的/记忆）
    backup = {
        "attractor": attractor.get_landscape() if hasattr(attractor, "get_landscape") else None,
        "purpose": self._snapshot_purpose(purpose),
    }
    if memory is not None and hasattr(memory, "get_state"):
        backup["memory"] = memory.get_state()

    try:
        self._restore_attractor(attractor, raw_data["attractor"])
        self._restore_purpose(purpose, raw_data["purpose"])
        if memory is not None:
            mem_state = raw_data.get("memory")
            if mem_state is not None:
                self._restore_memory(memory, mem_state)
    except Exception as e:
        # 任一步失败则回滚到恢复前状态，避免半恢复不一致
        logger.error(f"恢复中途失败，回滚到恢复前状态: {e}")
        self._rollback(attractor, purpose, memory, backup)
        raise RuntimeError(f"恢复失败已回滚: {path}") from e

    logger.info(f"从 {path} 恢复成功")
    return True
```

`_is_version_compatible` 补 minor 比较：当前版本 minor 须 >= 快照 minor（向后兼容），major 必须相等。validate 改为对内存中 `raw_data` 校验而非重新 `_torch_load`。

**预估工作量**：M

---

### P1-7 LLM 重试策略优化（区分 4xx/5xx）

**问题描述**

`bridge/llm_bridge.py` 第 111-131 行 `query` 对所有异常一视同仁重试 3 次，含指数退避。4xx 错误（如 401 鉴权失败、400 参数错误）重试必然失败，却仍阻塞 `3 * 30s timeout + 退避 7s = 最长 97s`。OpenAI client 无显式 `close()`，httpx 连接池可能泄漏。

**影响分析**

4xx 重试纯属浪费，且 97s 阻塞会让 `/chat/simple` 端点长时间不响应（P0-2 修复 run_in_executor 后虽不阻塞事件循环，但仍占用线程池 worker）。鉴权错误本应立即报错让运维介入，重试反而掩盖问题。

**优化方案**

利用 openai SDK 的异常类型分层重试，仅对 5xx 与网络错误重试：

```python
def query(self, user_input: str, memory_context: str) -> str:
    messages = self._build_messages(user_input, memory_context)
    last_error = None
    for attempt in range(self.max_retries):
        try:
            client = self._get_client()
            return client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=self.max_tokens, temperature=self.temperature,
            ).choices[0].message.content
        except Exception as e:
            last_error = e
            if not self._should_retry(e):
                # 4xx / 鉴权错误：立即抛出，不重试不退避
                raise RuntimeError(f"LLM 调用失败（不可重试）: {e}") from e
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2 ** attempt)
                time.sleep(delay)
    raise RuntimeError(
        f"LLM API 调用失败，已重试 {self.max_retries} 次。最后错误: {last_error}")

def _should_retry(self, exc: Exception) -> bool:
    """仅对 5xx 与网络/超时错误重试；4xx 立即失败。"""
    try:
        import openai
    except ImportError:
        return True  # 非 openai SDK，保守重试
    # 4xx 客户端错误：鉴权、参数、配额，重试无意义
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code >= 500
    # 连接/超时错误：重试
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError,
                        openai.RateLimitError)):
        return True
    return False
```

补充 `close()` 方法释放 client，并在 runtime 层 shutdown 时调用：

```python
def close(self) -> None:
    if self._client is not None:
        # openai>=1.0 的 OpenAI client 持有 httpx client，需显式关闭
        try:
            self._client.close()
        except Exception:
            pass
        self._client = None
```

**预估工作量**：S

---

## P2 后续规划

P2 为非阻塞的长期改进，可在 P1 完成后按需排期。

### P2-1 GPU / device 管理策略

**问题描述**

全系统张量未指定 device/dtype，`torch.zeros(num_nodes)` 等调用默认 CPU、float32。`recall_episodic` 第 252 行 `query.detach().cpu()` 假定输入在 CPU。无法利用 GPU 加速，也无法在显存受限环境显式约束 CPU。

**影响分析**

当前 num_nodes 默认 256、input_dim 64，CPU 足够；但若扩展到 1024 节点 + 大缓冲区，Langevin 推断的 `J @ sigma`（O(n²)）与 `recall_episodic` 矩阵乘法在 CPU 上会成为瓶颈。无 device 策略时迁移 GPU 需全系统改写。

**优化方案**

引入 `DeviceManager` 单例，启动时探测 `torch.cuda.is_available()` 决定默认 device，所有张量创建走统一工厂：

```python
# core/device.py
import torch

class DeviceManager:
    _device: torch.device = torch.device("cpu")
    _dtype: torch.dtype = torch.float32

    @classmethod
    def configure(cls, device: str = "auto", dtype: str = "float32"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        cls._device = torch.device(device)
        cls._dtype = getattr(torch, dtype)

    @classmethod
    def device(cls) -> torch.device:
        return cls._device

    @classmethod
    def tensor(cls, *args, **kwargs):
        kwargs.setdefault("device", cls._device)
        kwargs.setdefault("dtype", cls._dtype)
        return torch.tensor(*args, **kwargs)
```

`AttractorNetwork.__init__`、`MemoryManager`、`PurposeLayer` 中所有 `torch.zeros/randn` 改用 `DeviceManager.tensor`。`CoreConfig` 新增 `device: str = "auto"`。注意 Langevin 推断中 `torch.randn_like(sigma)` 会继承 sigma 的 device，无需额外处理，但跨模块张量运算（如 DreamEngine 注入 `idle_precision`）需保证 device 一致，构造时统一绑定。

**预估工作量**：M（机械替换 + 跨模块 device 一致性测试）

---

### P2-2 Docker 容器化

**问题描述**

无 `Dockerfile`、`docker-compose.yml`，硬编码本地路径 `C:\Users\dandan\...`（`memory_cli.py` 等）导致跨机器/容器部署失败。无 `.env.example`（P0-1 已补）。

**影响分析**

部署强依赖开发机环境，无法一键拉起，CI 也无法在干净容器中验证可部署性。

**优化方案**

多阶段构建 Dockerfile，基础镜像含 CUDA（torch GPU）或 CPU-only 两套：

```dockerfile
# Dockerfile
FROM python:3.11-slim AS base
WORKDIR /app
COPY requirements.lock setup.py ./
COPY core api runtime persistence bridge mcp_memory_server.py memory_cli.py ./
RUN pip install --no-cache-dir -r requirements.lock && pip install -e .
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
services:
  memory-api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    volumes:
      - ./snapshots:/app/snapshots   # 快照持久化到宿主
    restart: unless-stopped
```

清理所有 `C:\Users\dandan\` 硬编码，改用 `os.path` 相对路径或环境变量 `LMS_HOME`。

**预估工作量**：M

---

### P2-3 README + API 文档

**问题描述**

`docs/ARCHITECTURE.md` 声明存在 README 但实际不存在，缺 API 文档、部署指南、开发指南、CHANGELOG。

**影响分析**

新成员上手成本高，外部使用者无法快速集成。FastAPI 虽自动生成 `/docs`，但缺少架构级说明与部署约束。

**优化方案**

补 `README.md`（项目简介、快速开始、安装、CLI/API 用法、配置项），`docs/API.md`（9 个端点的请求/响应示例与错误码），`docs/DEPLOYMENT.md`（Docker 部署、systemd/supervisor 进程管理、环境变量清单）。FastAPI 端点补充 `response_model` 与 `responses` 字段以增强自动文档。新增 `CHANGELOG.md` 按 Keep a Changelog 规范记录版本变更。

**预估工作量**：M

---

### P2-4 预训练 Embedder 资源管理

**问题描述**

`core/sensory/embedder.py` 的 `PretrainedEmbedder` 模型（约 90MB）只加载不释放，无 `close()` / `__del__`。DreamEngine 懒加载后无释放机制，长期运行内存只增不减。`tokenizer.tokenize()` 动态扩展词表无锁，`PretrainedEmbedder` 非线程安全。

**影响分析**

多 session 共享单例 embedder 时模型常驻内存合理，但 `SessionManager.clear()` 删除会话时不释放关联资源，长期运行内存泄漏。tokenizer 词表动态扩展无锁在并发请求下可能丢词或索引错乱。

**优化方案**

`PretrainedEmbedder` 增加 `close()` 释放模型与 tokenizer，`SimpleTokenizer` 用 `threading.Lock` 保护词表扩展。`SessionManager` 增加 `release(session_id)` 在删除会话时清理 loop 持有的 embedder 引用（若该 embedder 不再被其他会话引用）。引入引用计数或弱引用管理 embedder 生命周期。

**预估工作量**：M

---

### P2-5 临时参数修改改为接口传参

**问题描述**

`DreamEngine._idle_infer`（第 446-478 行）与 `_idle_learn`（第 498-523 行）用 try/finally 临时修改 `attractor.temperature/orth_weight/complexity_weight/sigma` 后恢复。脆弱：若恢复前抛出非 finally 可捕获的中断（如信号、KeyboardInterrupt 在某些场景），在线态参数被污染。

**影响分析**

try/finally 模式在正常异常下安全，但参数散落在调用方，attractor 的 `infer`/`learn` 接口无法感知"当前是空闲态"，无法对空闲态做差异化处理（如不同 clamp 策略）。每次调用都要保存/恢复 4 个属性，易遗漏。

**优化方案**

为 `AttractorNetwork.infer` / `learn` 增加可选参数，调用方显式传入空闲态参数而非改全局属性：

```python
# core/hippocampus/attractor.py
def infer(self, sensory_input, precision, num_steps=10,
          temperature=None, sigma_seed=None) -> Activation:
    temp = self.temperature if temperature is None else temperature
    sigma = sigma_seed.clone() if sigma_seed is not None else self.sigma.clone()
    for _ in range(num_steps):
        b_q = self.bias + self.J @ sigma
        b_q[:self.input_dim] = b_q[:self.input_dim] + sensory_input * precision
        sigma = langevin(b_q)
        if temp > 0:
            sigma = sigma + torch.randn_like(sigma) * temp
            sigma = torch.clamp(sigma, -0.999, 0.999)
    # 注意：空闲态不应污染 self.sigma，仅在线态更新
    if sigma_seed is None:
        self.sigma = sigma.clone()
    # ... 计算 surprise/entropy ...
```

`_idle_infer` 改为 `self.attractor.infer(zero_input, self.idle_precision, num_steps=..., temperature=effective_temp, sigma_seed=state)`，移除 try/finally 与属性保存/恢复。`learn` 同理增加 `orth_weight`/`complexity_weight`/`learning_rate` 参数（`learning_rate` 已有，补另两个）。

**预估工作量**：M（需保证在线路径行为不变，sigma 更新时序需仔细对齐）

---

### P2-6 代码质量清理（魔术数字、命名债务、重复代码）

**问题描述**

多处魔术数字散落：`clamp(-0.999, 0.999)` 在 `attractor.py` 第 188、255 行、`dream_engine.py` 第 745、849 行重复；`0.3` 衰减系数（`purpose.py` 第 207 行）、`1e-10` 阈值多处。`purpose.py` 的 `_meta_adjust` 方法名与语义不符（实际是方向翻转）。`MetaState.surprise_history` 与 `_surprise_deque` 数据冗余（`meta_plasticity.py` 第 252、265、291 行同步两份副本）。`meta_plasticity.py` 的 `update()` 约 80 行过长。`dream_engine.py` 的 `dream_mvp` 与 `dream_cycle` 重复采样/坍缩检测/快照逻辑。返回字典含重复别名键 `cycles/steps`、`phases/phase_counts`（第 334-336 行，测试驱动技术债）。`attractor.py` 用 `assert` 做运行时校验（第 89、157、161 行），`python -O` 会被剥离。`_SNAPSHOT_VERSION` 硬编码（`dream_engine.py` 第 45 行）与 `persistence.snapshot.SNAPSHOT_VERSION`（第 25 行）双写。

**影响分析**

魔术数字重复使调参需多处修改；命名债务（`_meta_adjust` 实为翻转）误导维护者；冗余副本易不一致；`assert` 在生产 `-O` 模式失效导致校验消失；版本号双写需人工同步，遗漏则做梦快照与 persistence 版本不一致。

**优化方案**

集中处理：

1. 魔术数字提取为模块常量：`SIGMA_CLAMP = 0.999`、`DECAY_FLIP = 0.3`、`EPSILON = 1e-10`，在 `core/constants.py` 统一管理。
2. `_meta_adjust` 重命名为 `_meta_flip`，保留 `_meta_adjust` 为 deprecated 别名指向新方法，过渡期后删除。
3. `MetaState` 移除 `surprise_history` 字段，持久化时由 `_surprise_deque` 现场导出 `list(self._surprise_deque)`，消除冗余副本。
4. `update()` 拆分为 `_collect_signals`、`_compute_targets`、`_smooth_apply` 三个子方法。
5. `dream_mvp` 与 `dream_cycle` 提取公共的 `_run_dream_loop`（采样→推断→学习→坍缩检测→统计），两者只差异在阶段调度策略。
6. 返回字典删除别名键，测试改用规范键名（`steps`/`phase_counts`），消除别名技术债。
7. `attractor.py` 的 `assert` 改为 `if ...: raise ValueError(...)`。
8. `_SNAPSHOT_VERSION` 改为从 `persistence.snapshot` 导入，或在 runtime 层注入版本号，消除 core 层硬编码；若坚持 core 不依赖 persistence，则在 `core/types.py` 定义 `SNAPSHOT_VERSION` 常量，persistence 与 dream_engine 都从该处导入。

**预估工作量**：M（机械重构为主，需全量回归确认 249 测试不退化）

---

## 优化路线图（分阶段执行计划）

### 阶段一：发布阻塞修复（第 1 周，P0）

| 顺序 | 项目 | 工作量 | 关键产出 |
|------|------|--------|----------|
| 1 | P0-1 API Key 轮换 + 历史清理 | M | 新 Key 生效、git 历史无明文、`.env.example` 入库 |
| 2 | P0-2 并发数据竞争修复 | M | acquire 返回值检查、`/chat` `/chat/simple` 走 run_in_executor |
| 3 | P0-3 快照原子写入 | S | tmp + os.replace，崩溃不损坏旧快照 |
| 4 | P0-4 setup.py 依赖修复 | S | `pip install -e .` 可直接启动 API |

阶段一验收：并发压测下无数据竞争、崩溃后快照可恢复、新环境一键安装可运行。阶段一完成前禁止合入新功能。

### 阶段二：性能与解耦（第 2-3 周，P1）

| 顺序 | 项目 | 工作量 | 关键产出 |
|------|------|--------|----------|
| 5 | P1-1 recall_episodic 向量化 | M | 检索延迟降一个量级，等价性回归通过 |
| 6 | P1-2 DreamEngine 解耦 | M | 私有属性访问清零，公开接口就位 |
| 7 | P1-3 CoreConfig 校验 + 统一配置 | M | 配置错误在构造期暴露，消除 dict 双轨制 |
| 8 | P1-6 recover 异常链 + 回滚 | M | 失败可定位根因，半恢复自动回滚 |
| 9 | P1-7 LLM 重试策略优化 | S | 4xx 立即失败，连接池可释放 |
| 10 | P1-4 API 层测试补齐 | M | server.py 行覆盖 ≥80%，守护 P0-2 行为 |
| 11 | P1-5 CI/CD 搭建 | M | push 自动跑测试/lint/密钥扫描 |

阶段二验收：249 测试全绿且新增 API 测试通过、CI 在 PR 上强制门禁、recall 性能基准达标。建议 P1-4 与 P1-5 在 P1-1/P1-2 之后做，让测试与 CI 守护重构成果。

### 阶段三：工程化收尾（第 4 周及以后，P2）

| 顺序 | 项目 | 工作量 | 关键产出 |
|------|------|--------|----------|
| 12 | P2-1 GPU/device 管理 | M | num_nodes 可扩到 1024，device 可切换 |
| 13 | P2-5 临时参数改接口传参 | M | 移除 try/finally 属性篡改 |
| 14 | P2-4 Embedder 资源管理 | M | 长期运行内存稳定，tokenizer 线程安全 |
| 15 | P2-6 代码质量清理 | M | 魔术数字/命名/重复代码/版本号双写清理 |
| 16 | P2-2 Docker 容器化 | M | 一键 docker-compose up 部署 |
| 17 | P2-3 README + API 文档 | M | 上手与集成文档齐全 |

阶段三为非阻塞改进，可按实际需求排期。P2-6 建议在 P2-1/P2-5 重构后做，避免二次返工。

### 执行约束

- 每个 P0/P1 项合入前必须伴随测试（新增或回归），P1-4 测试基建应尽早落地以守护后续重构。
- 阶段一与阶段二的每一步都应在独立分支上完成，经 CI 绿后 squash merge，保留可回溯的提交粒度。
- 跨阶段依赖：P1-2（解耦）为 P2-5（接口传参）的前置；P1-5（CI）为所有后续重构的安全网；P0-1（密钥扫描）依赖 P1-5 才能长效守护。

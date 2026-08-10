# 活体记忆系统（LMS）部署说明 — NAS 公网环境版

> 更新时间：2026-08-07
> 适用：独立 NAS（公网环境，无内网直连 embed 服务）
> 基于：上游部署方案的适配 + 本机适配经验

## 环境差异（重要）

本机设计是 **LAN 直连优先 + 隧道兜底**：
```ini
LMS_CLOUD_EMBED_URL=http://<LAN_IP>:11435/v1/embeddings   # 局域网内 embed 服务
LMS_CLOUD_EMBED_FALLBACK_URL=https://embed.example.com/v1/embeddings  # 公网隧道
```

**部署于另一网络的 NAS 无法访问本 LAN**，只能用公网隧道：
```ini
LMS_CLOUD_EMBED_URL=https://embed.example.com/v1/embeddings   # 唯一可达
```

## 公网环境的关键适配（本次踩坑修复）

| 参数 | 本机默认（内网） | 公网需改 | 原因 |
|------|----------------|---------|------|
| `timeout` | 5.0s | **30.0s** | 公网 11435 延迟波动 0.4~2.5s，5s 太紧 |
| `retries` | 1 | **3** | 公网间歇 `ConnectionResetError(104)`，需重试 |

修改位置：`api/config.py` `build_embedder()` 里 CloudEmbedder 构造参数。

**现象**：`/feed` 端点报 `RuntimeError: Embed API 调用失败（已重试1次）：ConnectionResetError(104)`。
**根因**：内网假设的 timeout=5/retries=1 在公网不稳定连接下不够健壮。

## 总线路径适配

`runtime/bus_events.py` 默认 `_DEFAULT_BUS_FILE` 指向 `/vol2/...`（本机），
本机改为 `<AGENTOS_BUS_FILE>`。
（`LMS_BUS_FILE` 环境变量可覆盖，推荐用它而非改代码。）

## 完整启动命令（公网环境，新版功能全开）

```bash
cd /path/to/living-memory-system
export LMS_EMBEDDER=cloud
export LMS_CLOUD_EMBED_URL=https://embed.example.com/v1/embeddings
export LMS_CLOUD_EMBED_MODEL=bge-m3
export LMS_CLOUD_EMBED_DIM=1024
export LMS_SELF_REF_ENABLED=1          # 开启自指（默认 off）
export LMS_SELF_REF_PUBLISH=on         # 开启自指发布（默认 off）
export LMS_BUS_FILE="<AGENTOS_BUS_FILE>"
nohup .venv/bin/python -u api/run.py --port 8190 > lms.log 2>&1 &
```

## 验证清单

```bash
# 1. 健康
curl http://localhost:8190/health
# 2. 总线喂入塑形（双向反馈方向1）
curl -X POST http://localhost:8190/feed -H "Content-Type: application/json" \
  -d '{"session_id":"main","text":"测试"}'
# 应返回 {"status":"ok","entropy":...,"surprise":...（≥0 准确性项）,"free_energy":...,"mse":...}（2026-08-10 语义拆分后）
# 3. 总线发布（方向2：LMS → 总线）——等做梦周期后检查事件总线
grep "producer.*lms" <AGENTOS_BUS_FILE>
# 应有 lms.dream_complete 等事件
# 4. 自指读取
curl "http://localhost:8190/self-ref/voice?session_id=main"
```

## 已知修复（本次提交）

1. **api/server.py 语法修复**：此前提交时把 3 行 commit message 误写入源码，
   导致 `IndentationError` 启动失败。已删除垃圾行（备份 `api/server.py.bak-20260807`）。
2. **api/config.py 公网适配**：timeout 5→30、retries 1→3。
3. **runtime/bus_events.py 路径**：/vol2 → /vol1（或用 LMS_BUS_FILE 覆盖）。

## 核心概念（来自原设计）

这不是普通的"记忆存储"，是**模拟海马体的活体系统**：
- 记忆涌现：从神经网络动力学中"长"出来
- 做梦机制：空闲时自动做梦（记忆巩固、遗忘、重组）
- 目的演化：系统的"关注点"会自己漂移

**简单说：LMS 不存储记忆，而是维护一个能产生记忆的大脑状态。**

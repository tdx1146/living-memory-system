# LMS 快照损坏检测与恢复 SOP（lms-restore-sop）

- 版本：v1.0（2026-08-10 运维收尾产出，T2.4 配套）
- 适用范围：`<LMS_ROOT>`（数据面 :8190，唯一快照写者）
- 目标：任何「快照损坏 / 误删 / 状态回退」场景下，有明确、可演练、可校验的恢复路径
- 频率：**季度恢复演练**（每 3 个月一次，步骤见 §6）；故障时随时按 §5 执行

---

## 1. 快照布局与回退源（先认清家底）

```
snapshots/
├── {session}/                          # 每会话独立子目录（T1.4）
│   ├── latest_{session}.pt             # 「最新状态指针」，永不参与 prune
│   ├── snapshot_{session}_{turn}_{ts}.pt   # 轮次快照（prune 每会话保留最近 20 个）
│   └── *.lock                          # fcntl 伴生锁（零字节，非数据）
├── rescue-backup-20260810/             # 历史抢救备份（旧版扁平命名，只读封存）
├── latest.pt / snapshot_NNN.pt ...     # 旧版扁平文件（历史遗留，兼容可读，不再写入）
```

**回退源链（优先级从高到低）**：
| 层级 | 位置 | 粒度 | 说明 |
|---|---|---|---|
| L1 轮次快照 | `snapshots/{session}/snapshot_{sid}_{turn}_{ts}.pt` | 会话级 | prune keep=20，同会话最多保留 20 个回退点 |
| L2 最新指针 | `snapshots/{session}/latest_{session}.pt` | 会话级 | 可能被损坏的对象本体 |
| L3 15min 镜像 | `backups/lms/snapshots-15min/`（cron */15 rsync 镜像） | 全会话 | 与线上 snapshots/ 严格一致（--delete），mtime 保留 |
| L4 每小时归档 | `backups/lms/hourly/lms-hourly-*.tar.zst`（保留 168h） | 全会话 | 只含 snapshots/，zstd 压缩 |
| L5 每日全量 | `backups/lms/daily/lms-YYYYMMDD.tar.zst`（保留 30 天） | 全会话 | snapshots/ + data/ + logs/ |
| L6 跨机 | `REMOTE_BACKUP`（若配置） | 全会话 | rsync 镜像备份根 |

> 备份脚本：`scripts/lms_backup.sh`（--quick / --hourly / --daily / status），
> MANIFEST：`backups/lms/MANIFEST.jsonl`（追加式，含 sha256）。
> **「.prev 双副本」**：当前代码未实现 latest 旁的 .prev 副本；回退语义由
> 「L1 轮次快照 + L3–L5 备份层」承担，效果等价且更完整。

---

## 2. 损坏检测（三层，按序执行）

### 2.1 自动检测（已内置）
- **L3 深度健康检查**（`scripts/lms_ops_monitor.py` + :8190 `/health/deep`）：
  对最新 3 个 .pt 执行 `torch.load + validate`，失败即告警（`logs/lms_alerts.jsonl`）。
- **启动加载**：API 启动时若 `latest_{session}.pt` 损坏，加载路径记录 WARNING/CRIT，
  但**不会自动回退**（避免静默换脑）——需按 §4 人工/半自动回退。

### 2.2 手动检测命令（怀疑损坏时立即执行）
```bash
cd <LMS_ROOT>
# 快速元数据探测（不加载完整状态；损坏/截断会抛异常）
.venv/bin/python - <<'EOF'
from persistence.snapshot import Snapshot
s = Snapshot()
for p in ["snapshots/main/latest_main.pt",
          "snapshots/main/snapshot_main_111_20260810_010546.pt"]:
    try:
        print("OK  ", s.get_metadata(p))
    except Exception as e:
        print("BAD ", p, "->", type(e).__name__, e)
EOF

# 完整校验（字段 + 版本兼容）
.venv/bin/python - <<'EOF'
from persistence.recovery import RecoveryManager
r = RecoveryManager()
print("latest valid:", r.validate("snapshots/main/latest_main.pt"))
EOF
```

### 2.3 症状对照
| 症状 | 判定 |
|---|---|
| `/health/deep` 报 snapshot validate 失败 / alerts 出现 CRIT | 快照损坏（或写入中截断） |
| 会话 turn_count 重启后回退、记忆内容丢失 | 加载了旧快照 / 白板起步 |
| `torch.load` 抛 `EOFError` / `zipfile.BadZipFile` / `pickle.UnpicklingError` | 文件截断或损坏 |
| 文件大小为 0 / 比同会话其他快照小一个数量级 | 疑似损坏，走 §4 回退 |

---

## 3. 恢复前置：冻结写入（避免与回退竞争）

```bash
# 1) 停止做梦调度器（可选的温和方式）：
curl -s -X POST http://127.0.0.1:8190/dream/main -H 'Content-Type: application/json' -d '{"steps":0}' >/dev/null 2>&1 || true

# 2) 若需完全冻结（推荐用于演练/重大回退）：停 API（优雅停机先落盘再退出）
bash <LMS_ROOT>/scripts/lms_ctl.sh stop

# 3) 冻结后先做一次只读快照备份（回退前的现场）
bash <LMS_ROOT>/scripts/lms_backup.sh --quick
```

---

## 4. 回退流程（按损坏范围选择）

### 4.1 仅 latest_{session}.pt 损坏（最常见）
```bash
# 1) 列出该会话可用回退点（按时间倒序）
ls -t snapshots/main/snapshot_main_*.pt | head -5

# 2) 用次新轮次快照手工覆盖 latest（原子替换 + fcntl 锁由 save_copy 保障）
.venv/bin/python - <<'EOF'
from persistence.snapshot import Snapshot
s = Snapshot()
# 选「损坏前最近一次成功快照」；先 validate 再覆盖
src = "snapshots/main/snapshot_main_111_20260810_010546.pt"
dst = "snapshots/main/latest_main.pt"
print("src valid:", s.get_metadata(src))
print("copied:", s.save_copy(src, dst))
EOF

# 3) 重启 API 加载修复后的 latest（或直接 /restore，见 §5）
bash scripts/lms_ctl.sh start
```

### 4.2 会话目录整体损坏 / 误删
```bash
# 从 15min 镜像恢复（最近状态）：
rsync -a --exclude='*.lock' <LMS_BACKUP_ROOT>/snapshots-15min/main/ snapshots/main/
# 或从 hourly 归档解出单会话：
tar --zstd -xf <LMS_BACKUP_ROOT>/hourly/lms-hourly-*.tar.zst -C /tmp/lms-restore snapshots/main
# 或从 daily 全量解出（含 data/ 一起恢复时用这个）：
tar --zstd -xf <LMS_BACKUP_ROOT>/daily/lms-YYYYMMDD.tar.zst -C /tmp/lms-restore
# 解出后先 validate 再放回 snapshots/（见 §2.2），最后重启 API。
```

### 4.3 全库灾难恢复（含 data/ 与配置）
```bash
# 1) 解包最近 daily 全量到临时区
mkdir -p /tmp/lms-restore && tar --zstd -xf \
  <LMS_BACKUP_ROOT>/daily/lms-$(date +%Y%m%d).tar.zst -C /tmp/lms-restore
# 2) 校验清单：sha256sum -c 对比 MANIFEST.jsonl 中对应条目
# 3) 停止 API → 备份现有坏库 → 放回
bash scripts/lms_ctl.sh stop
mv snapshots snapshots.corrupt-$(date +%Y%m%d_%H%M%S)
mv data data.corrupt-$(date +%Y%m%d_%H%M%S)
cp -a /tmp/lms-restore/snapshots ./snapshots
cp -a /tmp/lms-restore/data ./data
# 4) 启动 + 校验（§5）
bash scripts/lms_ctl.sh start
```

---

## 5. /restore 端点恢复（在线，单会话）

`POST /restore/{session_id}`：从 snapshots/ 内任意合法 .pt 恢复指定会话。
路径钳制：仅允许 `snapshots/` 内、一层会话子目录、`[A-Za-z0-9._-]+\.pt` 命名的文件；
`../`、绝对路径、目录外路径一律 400。

```bash
# 在线恢复到指定轮次快照（API 运行中即可执行）
curl -s -X POST http://127.0.0.1:8190/restore/main \
  -H 'Content-Type: application/json' \
  -d '{"path":"main/snapshot_main_100_20260810_010105.pt"}'
# 期望返回：{"session_id":"main","restored":true,"path":"...","status":{...}}

# 恢复后确认会话状态（turn/熵/惊讶度/快照时间）
curl -s http://127.0.0.1:8190/status/main
```

**注意**：
- `/restore` 走 `loop.load_state()`（恢复 attractor/purpose/memory/tokenizer/meta/self_ref），
  失败抛 500 且**不改变**当前内存态（加载失败即回滚，不会半恢复）。
- 恢复后立即做一次快照固化：`POST /snapshot/main`（或 `curl -X POST http://127.0.0.1:8190/snapshot/main`），
  让恢复态成为新的 `latest_main.pt`。

---

## 6. 季度恢复演练（每 3 个月一次，全程 ~30 分钟）

**目标**：证明「损坏 → 回退 → 恢复 → 校验」全链路可用，演练后无残留。

### 6.1 演练前准备（10 分钟）
```bash
# 1) 记录基线
curl -s http://127.0.0.1:8190/status/main | python3 -m json.tool > /tmp/drill-baseline-main.json
curl -s http://127.0.0.1:8190/sessions
# 2) 确保备份链新鲜
bash scripts/lms_backup.sh --quick && bash scripts/lms_backup.sh status
# 3) 准备演练专用会话（不污染生产会话）：用 rescue 快照造一个 test 会话目录
mkdir -p snapshots/drill-test
cp snapshots/rescue-backup-20260810/snapshot_200.pt snapshots/drill-test/snapshot_drill-test_0_$(date +%Y%m%d_%H%M%S).pt
```

### 6.2 损坏注入（5 分钟）
```bash
# 1) 在 drill-test 会话执行 /restore 使其进入内存
curl -s -X POST http://127.0.0.1:8190/restore/drill-test -H 'Content-Type: application/json' \
  -d '{"path":"drill-test/snapshot_drill-test_0_*.pt"}' || true
# 2) 人为损坏其 latest（截断文件）
cp snapshots/drill-test/latest_drill-test.pt /tmp/drill-latest-good.pt 2>/dev/null || true
head -c 1000 snapshots/drill-test/latest_drill-test.pt > /tmp/trunc.pt && \
  cp /tmp/trunc.pt snapshots/drill-test/latest_drill-test.pt
# 3) 验证检测层能发现（期望 validate=False / 抛异常）
.venv/bin/python -c "
from persistence.recovery import RecoveryManager
print('损坏检测:', RecoveryManager().validate('snapshots/drill-test/latest_drill-test.pt'))"
```

### 6.3 回退 + 恢复（10 分钟）
```bash
# 1) 回退：用次新轮次快照修复 latest（§4.1 步骤）
# 2) /restore 在线恢复 + 固化快照（§5）
# 3) 校验恢复结果（§6.4）
```

### 6.4 校验（5 分钟）
```bash
# 1) /restore 返回 restored:true 且 status 字段完整
# 2) 轮次/记忆连续性：GET /status/drill-test 的 turn_count 与基线一致
# 3) 检索探针：POST /recall 能命中恢复内容
curl -s -X POST http://127.0.0.1:8190/recall -H 'Content-Type: application/json' \
  -d '{"session_id":"drill-test","query":"测试","k":3}' | head -c 300
# 4) 备份链完好：MANIFEST 最新 3 条均为 ok
tail -3 <LMS_BACKUP_ROOT>/MANIFEST.jsonl
```

### 6.5 清理（5 分钟）
```bash
# 删除演练会话与临时文件（保留损坏现场供复盘则移入 /tmp）
curl -s -X DELETE http://127.0.0.1:8190/sessions/drill-test || rm -rf snapshots/drill-test
rm -rf /tmp/lms-restore /tmp/trunc.pt /tmp/drill-latest-good.pt
# 演练结果记录到 memory/（季度条目）：日期 / 注入类型 / 检测耗时 / 恢复耗时 / 是否通过
```

**演练验收标准**：检测 ≤5min 发现；恢复 ≤10min 完成；校验 4 项全过；生产会话零影响。

---

## 7. 故障速查表

| 场景 | 处置 | 关键命令/文件 |
|---|---|---|
| latest 损坏 | §4.1 回退次新 | `ls -t snapshots/{sid}/snapshot_*` |
| 会话目录误删 | §4.2 镜像恢复 | `rsync backups/lms/snapshots-15min/{sid}/` |
| 全库灾难 | §4.3 daily 全量 | `tar --zstd -xf backups/lms/daily/...` |
| 需要在线回退单会话 | §5 /restore | `POST /restore/{sid} {"path":"..."}` |
| 不确定是否损坏 | §2.2 手动检测 | `RecoveryManager().validate(path)` |
| 恢复后状态错乱 | 固化快照 + 重启 | `POST /snapshot/{sid}` + `lms_ctl.sh restart` |
| 备份缺失 | 检查 cron + MANIFEST | `crontab -l`、`lms_backup.sh status` |

> 维护提示：本 SOP 所述回退源（L3–L5）由 crontab 备份条目驱动（2026-08-10 安装：
> `*/15 --quick`、`0 * * * * --hourly`、`30 2 * * * --daily`）；若备份 cron 停用，
> 恢复能力退化到 L1/L2 单机单点，请保持备份链常开。

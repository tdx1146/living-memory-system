# LMS 代码审计问题清单（2026-08-23）

> 三代理并行审计产物（核心引擎 / API与外围 / 运维脚本），共 56 项。
> 本文件是修复工作流的**单一事实来源**。格式：`[严重度] 文件:行 | 问题 | 建议修复`
>
> **红线纪律（所有修复代理必须遵守）**：
> 1. 血管不换：endpoint signatures（/chat /feed /store /recall /status /landscape /snapshot /restore /dream /e3/review /sessions 等现有签名）、six-layer injection、沙漏 path、wake chain —— **一律不动**；需要新增端点或改签名的修复 → 标记 `[需批准]`，设计进方案但不实施。
> 2. `.env` 红线参数不动：LMS_J_TARGET_NORM=12、LMS_BIAS_SCALE=0.9、LMS_BIAS_ADAPT=1 等（dandan 拍板的参数）。
> 3. 每修改一个文件：先备份 `cp file file.bak-$(date +%Y%m%d%H%M)`，改完 `python -m py_compile` 验证。
> 4. 修复必须带注释：注明对应 issue 编号 + 为什么这么修。
> 5. 全部改完：`cd /vol2/1000/AI专用/living-memory-system-cloud && git add -A && git commit -m "fix(模块): ..."`。
> 6. 不重启服务、不做破坏性操作；只读代码用 read，改代码用 edit/write。
> 7. 修复后可以跑相关单测（tests/ 目录），但不要动测试文件本身（除非是新增断言）。
>
> 严重度：5=正在造成数据/记忆损失或服务不可用；4=潜在数据损失或频繁误触发；3=逻辑错误但影响有限；2=健壮性；1=建议优化。

---

## 模块 A：核心引擎（core/hippocampus/ + runtime/loop.py + runtime/bus_events.py + core/doubt/）

### P0
- [A1][5] `core/hippocampus/attractor.py:262,283-288` | **allostatic J 饱和/崩塌重锚 100% 抛 UnboundLocalError 被 fail-open 吞掉**：`z` 只在 surprise 统计带 else 分支赋值，`sat`/`col` 分支走 `logger.info(...z...)` 时 `z` 未定义必抛异常（`if abs(new_target-j_target)>1e-9` 在 step=0.5 时必成立→必进）；loop.py:766-768 broad-except 吞掉，attractor.py:299 `self.j_target = new_target` 在异常点之后从不执行 → **σ 饱和/崩塌时设定点永不更新**，这正是 σ 饱和长期无法自愈的根因候选。修法：`if sat:` 之前初始化 `z = 0.0`（或 sat/col 分支前也算 `z = self._surprise_z(surprise)`）；加单测注入 `SigmaStats(frac_gt0_9=0.95, act05=200)` 断言 j_target 递减。
- [A2][4] `runtime/bus_events.py:209-219` | `CircuitBreaker.status()` 持非重入 `threading.Lock` 期间调用 `is_open()` 二次 acquire 同一把锁 → **自死锁**（任何状态观测接线或 quick_test 即挂线程）。修法：status() 内不调 is_open()，内联 `self._open_until > 0.0 and time.time() < self._open_until`；或改 `threading.RLock()`。
- [A3][4] `core/hippocampus/memory.py:683-695` | `_recall_episodic_scored` 在 top-k 截取**之前**对全部相似命中条目执行 `record_reference` + 刷新 `last_reinforced_turn` → 每轮给整批相似条目（含从未进 LLM context 的）虚增引用/召回/置信度/磨损计时，置信度趋近 1.0、E3/labile 指标失真。修法：先排序截取 top_k，再对 `scored[:k]` 执行 record_reference/刷新。

### 严重度 3
- [A4][3] `core/doubt/gap_registry.py:226-231 vs 148-155` | `mark_resolved` 用精确相等移除 A/B 类，而 `register_fok_unresolved` 用 `_fok_core`+`_fok_similar`（前缀≥12字/Jaccard≥0.4）模糊去重 → 同悬案换包装登记后**永远无法被消解移除**，E3 反复选中已闭环主题。修法：mark_resolved/is_resolved 改用与登记侧同构的归一化判定（_fok_core+_fok_similar）。
- [A5][3] `core/hippocampus/attractor.py:338-341` | `AllostaticJController.snapshot()` 浅拷贝 `events=list(self.events)` 后原地改写 `ev['ts_turn']` → 每次 /status 把**全部历史事件**的 ts_turn 覆写为当前 turn，真实发生轮次永久丢失。修法：`events=[dict(ev) for ev in self.events]` 深拷贝后再写。
- [A6][3] `core/hippocampus/attractor.py:1336-1342,1121` | `set_landscape` 恢复 allostatic.j_target 但**不回写 `self.j_target_norm`** → 恢复后 learn() 范数钳制用 stale 设定点（可能永远停 40），饱和态下恢复的低设定点不生效。修法：恢复 j_target 后同步 `self.j_target_norm = self.allostatic.j_target`。
- [A7][3] `runtime/loop.py:1815-1830` | `_manage_entropy` 在同一轮第二次调用 `attractor.learn`（同样 activation）→ 极端熵轮次 J 被更新两次、Hebbian 重复强化，与"每轮 learn 一次"契约不符。修法：改为只对正交化/复杂性梯度做增量校正（或并入主 learn 的参数调制），避免同轮二次完整 learn。

### 严重度 2
- [A8][2] `core/hippocampus/dream_engine.py:452-562` | `dream_cycle`（七阶段）从不调用 `memory.consolidate()`（dream_mvp 调用）→ full_cycle 模式短时→长时迁移缺失，与 docstring"阶段1 NREM巩固"不符。修法：收尾补 consolidate() 或 docstring 声明设计依据。
- [A9][2] `core/hippocampus/memory.py:55,472-476` | `_GARBAGE_FILTERED` 计数器声明"可被 status 读取"但从不自增 → 垃圾过滤次数不可观测。修法：命中处 `global _GARBAGE_FILTERED; _GARBAGE_FILTERED += 1`。
- [A10][2] `runtime/loop.py:2704-2771` | `load_state` 不恢复 `last_activation` → 重启/恢复后 query_llm 首轮返回 "[无记忆]"，LLM context 丢失最近激活态。修法：save/load 双侧补 last_activation 状态。
- [A11][2] `persistence/recovery.py:302-311,345-377` | 恢复失败回滚快照不含 `allostatic` 字段 → 回滚无法复原已被 set_landscape 改写的 allostatic.j_target。修法：_snapshot_current_state 追加 'allostatic' 键，回滚一并恢复。
- [A12][2] `runtime/loop.py:1421` | E3 路径 B（`_e3_reactivate_path_b`）把重激活线索文本作为普通 external 条目写入 episodic（无 [doubt] 前缀）→ 探针文本污染记忆、可被后续检索注入、自喂风险。修法：走 `source='doubt'` 或加 `[doubt-supersedes]` 前缀或存 gray。
- [A13][2] `core/hippocampus/attractor.py:857,941` | `infer` docstring 声称扩散项 `sqrt(2*T)*noise`，实现用 `T`（`randn_like * temperature`）→ 注释与实现不符，温度标定偏离。修法：统一（实现改 `*math.sqrt(2.0*temperature)` 或注释改 T）。
- [A14][2] `core/hippocampus/dream_engine.py:918-924,1025-1027` | `_value_replay` 的 `score_map`/`replay_set_ids` 以 `_entry_id`（无 id 回退 turn）为键，supersedes 记录复用原条目 turn → 多条目共享同键分数错配/ID 歧义。修法：_entry_id 优先 id/文本哈希，turn 仅兜底。

### 严重度 1
- [A15][1] `core/hippocampus/purpose.py:307-308` | `_meta_adjust` 注释与语义矛盾（注释"设为最大值（探索新方向）"实际"强化已关注方向"）。修法：删/改注释。
- [A16][1] `runtime/loop.py:616,1062,2549-2555` | 存储文本格式不一致（llm_output 非空才加"用户:"前缀）；快照命名 turn 比内容 +1。修法：统一前缀构造；_auto_snapshot 用增量前 turn。

---

## 模块 B：API 与外围（api/ + core/doubt/ + tools/ + bridge/）

### P0
- [B1][4] `api/server.py:716-754 + runtime/loop.py:2560-2593` | **/chat/simple 绕过做梦互斥锁**：无 acquire/release_conversation，query_llm 内部 process_turn 是写路径 → 可与后台做梦并发写同一脑（J/precision/episodic 撕裂）。修法：补 register_session + acquire_conversation（失败 503）+ finally release，与 /chat 一致。
- [B2][4] `api/session_manager.py:107-110` | 启动自动恢复把全局旧格式 `snapshots/latest.pt` 当任意会话候选（load_state 对 session_id 不一致仅告警）→ **跨会话污染/静默回退过期脑**。修法：对全局 latest.pt 候选加归属校验（顶层 session_id 非空且≠当前 sid 则跳过）或移除该候选。

### 严重度 3
- [B3][3] `api/server.py:1354-1403` | `/restore` 不校验快照归属会话、不加做梦锁 → 可跨会话载入整脑 + 梦期恢复竞态。修法：校验快照顶层 session_id（不符→400）+ acquire/release。
- [B4][3] `api/server.py:1301-1351` | `/snapshot` 不加做梦锁 → 梦期落盘半巩固状态，自动恢复会载入坏状态。修法：snapshot 前 acquire（失败 503），保存后 release。
- [B5][3] `api/server.py:669/786/877/1171 + runtime/dream_scheduler.py:194-214` | async 端点内同步阻塞锁 + 全局单锁 → 做梦期间整个事件循环冻结 10s、多会话写串行化。修法：acquire 移 run_in_executor，或每会话锁+全局做梦锁两级。
- [B6][3] `api/session_manager.py:66-67` | 空 session_id 静默兜底到 "default"（无日志）→ 拼错字段的客户端污染 default 脑。修法：空值记 WARNING 或 400。
- [B7][3] `core/doubt/gap_registry.py:218-238` | 消解/判重键（原始 topic 精确匹配）与登记去重键（_fok_core+_fok_similar）不一致 → 已消解悬案可复活。修法：统一归一化键。
- [B8][3] `api/control.py:507-557` | /control/config 热配置只写 _overrides，数据面直读 os.environ → **假生效**（显示已应用实际零变化）。修法：同步写 os.environ 或标注"对数据面无效"+审计告警。
- [B9][3] `api/control.py:604-659,200-213` | 注册下发的 client token 无消费方校验（_check_auth 只比对 CONTROL_TOKEN）→ "只发不收"半成品鉴权。修法：二选一（校验注册表 token_hash / 移除注册流程）。
- [B10][3] `tools/archive_job.py:139-169 + core/archive/archive_store.py:393-433` | 归档重建整体替换 → 永久丢失不在保留快照中的旧归档条目（窗口外记忆被工具重新引入丢失）。修法：rebuild 语义改为"按 (turn,text_hash) 去重合并、只增不删"。
- [B11][3] `api/server.py:1455-1472 + api/session_manager.py:91-143` | `DELETE /sessions/{sid}` 不删快照/归档 → 会话下次访问即"复活"，隐私清理不彻底。修法：DELETE 同步删 snapshots/{sid}/ 与 archive，或加 purge=true。
- [B12][3] `api/config.py:187-188/223-224/250-271` | env 值为空串时 int()/float() 直接抛异常 → 配置构建崩溃 500，且不指向具体键。修法：全部包 try/except+默认值回退+logger.warning 指明键名。

### 严重度 2
- [B13][2] `api/wiring.py` | M1 接线模块是死代码（无生产 import），与 server.py 内联 writer 双实现漂移 → 若切到 wiring 会静默丢字段。修法：删除或让 server.py 真正走 wiring。
- [B14][2] `api/server.py:927-934` | /store 新条目 meta 用 `entries[-1]` 假设"本轮只追加一条" → 多条时 meta 只落最后一条。修法：用增量条目列表逐个打 meta。
- [B15][2] `api/server.py:152-160,231-238` | SimpleChatRequest/FeedRequest 缺 sid 兼容别名（其余请求模型都有）→ 旧客户端 sid 被静默丢弃。修法：补同款 sid alias + 告警。
- [B16][2] `bridge/translation_layer.py:561-564` | recall_from_hourglass 的 embed 调用绕过熔断器 → 故障裸抛/挂起。修法：统一走 get_default_embed_circuit().call。
- [B17][2] `core/doubt/gap_registry.py` | 无锁，跨线程读写竞态（做梦线程 vs API 线程）→ json.dump 可能抛 RuntimeError 被吞、列表推导期间增删丢更新。修法：加 threading.RLock，统一持锁。
- [B18][2] `tools/clean_episodic.py:129-132` | 无 fcntl 锁直写快照文件，服务运行中清洗不生效（被内存覆盖）。修法：探测 :8190 运行则拒绝，或走 Snapshot.save 持锁。
- [B19][2] `api/session_manager.py:69-88` | get_or_create 持全局 RLock 执行 torch.load 自动恢复 → 首次访问阻塞所有会话。修法：恢复移出锁外/会话级锁。
- [B20][2] `bridge/mcp_server.py:211-217` | save_snapshot 接受客户端任意路径写文件（无路径钳制）。修法：钳制到 snapshots/ 内（复用 _clamp_snapshot_path）。

### 严重度 1
- [B21][1] `api/server.py:357-419,490-492` | 旧幂等/限流死代码残留（无调用点，误导维护者）。修法：删除或注释指向新实现。
- [B22][1] `bridge/llm_bridge.py:169` | LLM 返回 content=None 被误判为调用失败（server.py:690 len(None) 抛 TypeError）→ 包装成 degraded 错误。修法：content or "" 归一化。
- [B23][1] `api/control.py:696-780` | /control/diagnose 只读探针有建会话副作用（POST /recall 会 get_or_create）。修法：改打 /health + /landscape/main 纯只读。

---

## 模块 C：运维脚本（scripts/ + *.sh + tools/ 迁移工具）

### P0
- [C1][5] `lms_backup.sh:162-180,217-235` | **hourly/daily tar 归档连续多日 100% 失败**：live 每 ~34s 写一次快照（snapshots/main/snapshot_main_1810_*.pt 实测 34s 间隔），tar 报 "file changed as we read it" 退出码 1 → `if !(tar)` 判失败 → `rm -f $tmp_archive`。hourly 只剩 08-13/08-19，daily 缺 08-11..21 —— **灾备只剩 15min rsync 镜像一层**。修法：① `tar ... 2>$err; rc=$?; grep -q 'file changed' $err && rc=0`；② 或先 rsync 到暂存再打包；③ cron 错开整点竞争。
- [C2][4] `lms_entropy_watchdog.py:126-129,199-206` | **换蛋恢复的是 latest_main.pt（近当前态，每 34s 覆盖）而非真健康蛋**：last_healthy 仅在 σ<0.9/熵<0.995 瞬间记录最新 mtime 快照（刚恢复后的状态）→ 恢复后 ~35 分钟再度饱和 → 换蛋循环（实测 01:50/02:30/03:10），每次回滚 ~35 分钟记忆，dsh_feed 水位不回退 → **被回滚记忆永久丢失**。修法：按 lms-field-restore 技能选蛋（J 范数≈7、快照年龄下限 ≥2h、灰池修复后）；换蛋后隔离期（2h 内不记录 last_healthy）。
- [C3][4] `lms_entropy_watchdog.py:84-120` | **σ 重置对 live 进程无效**：独立进程 get_or_create 加载磁盘副本 reset_state + save，live :8190 内存 σ 依然饱和，且 live 每 34s 覆盖磁盘（重置仅存活 ~34s），两进程并发写同一 latest_main.pt 有撕裂风险。修法：走 :8191 控制面加 `POST /control/reset-sigma` 进程内端点（`[需批准]` 新增端点）；在获批前，watchdog 至少**不写磁盘**（避免并发写撕裂），仅告警。
- [C4][4] `lms_entropy_watchdog.py:199-206` | 健康分支 `if sigma_max < 0.9 or entropy < 0.995` 吞掉"σ 饱和但熵<0.995"的 onset 阶段：σ 重置分支（要求熵≥0.995）**永不触发**，且把饱和快照记为 last_healthy（毒蛋）。修法：重构判定顺序——先判 σ 饱和/熵烧焦，再判健康；健康记录要求 σ<0.9 **且** 熵<0.995 同时成立，σmax>0.9 时绝不写 last_healthy。
- [C5][4] `lms_entropy_watchdog.py:158-174` | dispose_burnt 恢复脚本 returncode 不检查、重启后无验证（docstring 承诺"等 20s 验证"未实现）→ **假成功换蛋** + 30 分钟盲区。修法：returncode!=0 → CRIT 不重启；重启后 sleep 20s 复读 /status /landscape 复核，仍烧焦则交人工且不置 last_dispose。

### 严重度 3
- [C6][3] `lms_ops_monitor.py:157,345-348` | 孤儿进程误报风暴：api.run/run_control 是 setsid 守护（PPID=1 是设计常态，crontab 注释实锤）→ 每 5 分钟 WARN 一条，告警通道被噪音淹没。修法：排除正在服务 /health 的 API 主进程与 control plane，只对 MCP/glue 判孤儿。
- [C7][3] `lms_dsh_feed.py:156-161` | 水位跳到未配对 user 消息的 seq（posted==0 时推进）→ 该轮对话（含后续 assistant 回复）永久不喂入 LMS。修法：水位只推进到最后一个已配对 assistant seq；未配对 user seq 存 pending_user_seq 下轮续配。
- [C8][3] `lms_dsh_feed.py:134-146` | 首次回填上限（20 轮）自破：达到上限后 backfilled=True 反而解除上限，同轮全量冲刷。修法：达上限时推进水位但**不**置 backfilled=True（下一轮才继续），或把 backfilled 语义改为"本轮已执行回填上限"。
- [C9][3] `lms_feed_docs_chunked.py:24-36,57-60` | 无句边界的超长块被 `chunk[:CHUNK]` 截断并标记完成 → 文档尾部永久丢失；失败时 exit 0 运维无感知。修法：超长 parts 按硬长度二次切分；失败返回非零。
- [C10][2] `lms_http_mcp.py:396-408` | stdin 多行 JSON 请求解析失败即清空 buffer → 半截请求被丢弃、调用方超时。修法：JSONDecodeError 时不清 buffer，成功解析才清。
- [C11][2] `lms_entropy_watchdog.py:39-46 + crontab` | 双重日志（log() 写文件+print，cron 又重定向同文件）→ 每条两行；SIGMA_RESET_COOLDOWN_S=300 恰等于 cron 周期 → 实际无冷却。修法：cron 去重定向改 /dev/null；冷却取周期 2 倍以上或记录连续重置次数升级处置。
- [C12][2] `lms_ops_monitor.py:259-260,373` | L2 探针 POST /recall 制造 default 幻影会话；mcp_orphan_count 字段名不副实。修法：探针复用 main 或只读端点；字段改名。
- [C13][2] `lms_ctl.sh:246-251` | start 失败仍触发 doubt 部署钩子（manual_start 返回值被丢弃）→ 部署失败记成成功事件。修法：`start) manual_start || exit 1`；PID 解析限定 cmdline 含 api.run。
- [C14][2] `tools/migrate_gray_visible.py:7-8,37` | "前置：服务已停止"未强制，live 运行时执行双写竞态。修法：入口先 GET /health，可达即报错退出（fail-closed）。
- [C15][2] `tools/calibrate_per_dim_scale.py:63` | torch.cat 空样本崩溃（全部 per_dim_surprise=None 时）。修法：空列表守卫。

---

## 汇总

- 模块 A：16 项（P0: A1-A3）
- 模块 B：23 项（P0: B1-B2）
- 模块 C：15 项（P0: C1-C5）
- **合计 54 项**（另有 2 项已在 08-22 修复确认：lms_status 嵌套读取、fok 去重——不计入待修）

**修复优先级**：C1（备份止血）→ A1（内在自愈根因）→ C2/C4/C5（watchdog 循环与毒蛋）→ B1/B2（并发/跨会话）→ A2/A3 → 其余按严重度。
**需 dandan 批准**：C3（新增 /control/reset-sigma 端点，血管级）。

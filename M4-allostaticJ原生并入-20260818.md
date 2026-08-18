# M4 allostatic J 原生并入 attractor（2026-08-18）

规格：四妹-LMS核心重写规格v2-20260817.md §1.3（attractor.py：allostatic J 原生并入）
前置：M3-2 已落盘（104 测试过）；原外挂 `runtime/allostatic_j.py`（开关默认 0、与核心分离）

## 任务完成对照

| 任务 | 状态 |
| --- | --- |
| 1. allostatic J 并入 attractor（j_target_norm 滑动设定点原生：Mehra 1970 innovation 法——σ 饱和降/崩塌升/动态稳——step 增量/[3,40] 护栏/persist 防抖/冷启动 30 轮） | ✅ core/hippocampus/attractor.py |
| 2. 删外挂 runtime/allostatic_j.py（loop 不再 import 外挂） | ✅ 已删除（含 pycache） |
| 3. /status allostatic_j 观测块保留（并入后仍可观测） | ✅ get_status → attractor.allostatic_snapshot |
| 4. 单测（饱和→降/崩塌→升/动态→稳/冷启动不滑动） | ✅ test_allostatic_j.py（控制器级 + 原生路径双覆盖） |
| 5. 自验：rewrite-ws 跑相关测试全绿（生产 venv） | ✅ 见下 |
| 6. 输出 rewrite-ws/ | ✅ 本文件 + 改动清单 |

## 落盘文件清单

| 文件 | 内容 |
| --- | --- |
| core/hippocampus/attractor.py | ① 原生并入 `SigmaStats` / `compute_sigma_stats` / `allostatic_j_enabled` / `AllostaticJController`（原外挂模块整体迁入，行为零改动）；② `AttractorNetwork.__init__` 按 env（LMS_J_ALLOSTATIC，默认 0=关）自建 `self.allostatic` 并让 `j_target_norm` 立即由滑动设定点接管；③ 原生方法 `update_allostatic(surprise, sigma_state)`（j_target_norm 唯一写者，fail-open）与 `allostatic_snapshot(turn_count)`（观测块） |
| runtime/loop.py | ① 删除外挂 import（`from runtime.allostatic_j import ...`）与控制器构造块；② `process_turn` 2.5 步改走 `attractor.update_allostatic(activation.surprise, activation.state)`（fail-open）；③ `get_status` 观测块改走 `attractor.allostatic_snapshot(turn_count)`（开关关 → `{'enabled': False}`，§4.2 独立追加语义）；④ 兼容引用 `loop.allostatic_j = getattr(attractor, 'allostatic', None)`（glue 集成点保留） |
| runtime/allostatic_j.py | **已删除**（外挂文件 + __pycache__ 残留） |
| tests/test_allostatic_j.py | ① 导入路径迁移 `runtime.allostatic_j` → `core.hippocampus.attractor`；② 新增 `TestNativeIntegration` 8 例：默认关零参与 / env 开自建 / **冷启动不滑动**（29 轮 σ 饱和在场仍稳）/ **饱和降** / **崩塌升** / **动态稳** / surprise 带原生全链路 / 观测块结构；③ `TestAttractorIntegration` 改走原生 `update_allostatic` → learn 按动态设定点钳制 |
| api/config.py | 仅文档注释指向更新（机制参数详见 → core/hippocampus/attractor.py）；env 解析键保留（env 为唯一事实源） |

## 设计要点（对照规格 §1.3）

- **J_target 不再是常量**：surprise 序列（= Kalman innovation，Mehra 1970）running window 统计重估 + σ 越界硬信号（饱和降/崩塌升/动态稳），与双快照复测判据（frac_gt0.9 / act05）逐项对应；`LMS_J_TARGET_NORM` 仅作初始设定点（= 快照 j_initial，保留 norm-7 参考工作点作诊断基准，不再硬编码钳制）。
- **滑动带学习率**：step 增量（默认 0.5）+ [j_min=3, j_max=40] 护栏 + persist 轮确认（默认 5，防单轮抖动），参数全部 env 化（LMS_J_ALLOSTATIC_*，与 api/config.py 头部参数表一致）。
- **冷启动**：窗口样本 < min_samples（默认 30）不动作（保持初始设定点），σ 饱和在场也不重锚（任务书核心单测）。
- **原生写者单一**：`update_allostatic` 是 j_target_norm 的唯一写者（state_update 后 emit 步）；`learn()` 范数钳制 `min(1, target/‖J‖_F)` 按动态值执行（attractor.py learn 尾部，零改动）。
- **治理/回滚**：LMS_J_ALLOSTATIC=0（默认）→ 控制器不建、全部路径零参与、固定 J 行为完全不变；纯进程内存状态（同 precision_adapt 先例：重启即失、快照不落盘——get_landscape/set_landscape 只保存 J/bias/sigma，恢复后设定点回到 env 初值）。
- **fail-open**：控制器/统计/观测所有异常静默降级，不阻断主路径。

## 自验结果

生产 venv：`/tmp/repro3-1786556208/living-memory-system-cloud/.venv`（Python 3.11.2，torch 2.13.0+cu130，pytest 9.1.1）

- `pytest tests/test_allostatic_j.py tests/test_attractor.py -q` → **60 passed**（基线 52 + 新增原生 8）
- `pytest tests/test_doubt_m3.py tests/test_doubt_integration.py tests/test_api_server.py tests/test_bridge.py tests/test_config.py -q` → **243 passed**（loop 接线回归）
- 全套 `pytest tests/ -q` → **1060 passed / 0 failed**（173.81s）

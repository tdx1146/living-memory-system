"""M5 loop 重组：turn 生命周期 9 步固化 + 状态管理清单（§1.2 / §7.1 M5）。

规格：四妹-LMS核心重写规格v2-20260817.md
  - §1.2  loop.py：turn 生命周期（process_turn 唯一增量点；生命周期固定
          9 步，禁止旁路）+ 状态管理清单（状态归属模块、由谁更新、由谁只读）
  - §7.1  M5 loop 重组：9 步生命周期固化、状态管理清单落地（前置 M2/M3/M4）

本模块是 M5 的机器可读交付（纯 stdlib，不 import 本仓库运行时——与
core/recall 同款防循环依赖策略）：

  TURN_LIFECYCLE_9   9 步固化清单（规格 §1.2 步名/语义 + 代码锚点 +
                     写者/读者——状态管理清单交叉引用）
  STATE_OWNERSHIP    状态管理清单（规格 §1.2 表格：六项状态 ×
                     归属模块 / 唯一写者 / 只读读者 / 验证锚点）
  LifecycleTrace     每轮生命周期记录器（纯内存，重启即失——同
                     react_surprise_history / gap_registry 先例：
                     不落盘、不进快照；状态汇总 = 观测数据）

运行契约（由 runtime/loop.py + tests/test_loop_m5.py 双验证）：
  - process_turn 每轮必须完整走过 9 步：记录器断言 9 步各恰一次、
    无重复、无未知步（missing/duplicated/unknown → 汇总 broken=True
    并在日志告警——G 模式：以日志可见，不以静默吞掉）；
  - turn 计数增量只能发生在 emit 步（loop._increment_turn 唯一写者；
    快照恢复是赋值恢复不是增量——文档登记于 STATE_OWNERSHIP）；
  - 记录器零副作用：只读观测、绝不 setattr 条目、绝不写持久层。

执行序说明：TURN_LIFECYCLE_9 按规格 §1.2 的规范序登记；process_turn 的
实际执行序是 FEP 感知-行动循环的算法序（state_update 的锚点分散在管线：
learn/π/σ 更新须在检索前完成——算法依赖，不能挪动）。LifecycleTrace 按
**执行序**记录，summary 按**规格序**呈现，并同时暴露 execution_order 与
spec_order 供审计。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# 一、9 步固化清单（规格 §1.2：生命周期固定为 9 步，禁止旁路）
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class LifecycleStep:
    """turn 生命周期的一步（固化契约，机器可读）。"""

    idx: int                          # 规格步号（1..9）
    name: str                         # 规格步名（§1.2）
    spec: str                         # 规格语义要点
    anchor: str                       # runtime/loop.py 实现锚点
    writes: Tuple[str, ...]           # 本步写出的状态（STATE_OWNERSHIP 键）
    reads: Tuple[str, ...]            # 本步读取的状态（STATE_OWNERSHIP 键）


TURN_LIFECYCLE_9: Tuple[LifecycleStep, ...] = (
    LifecycleStep(
        idx=1,
        name="ingest",
        spec="写侧入口（feed/store）→ 注入时怀疑（§2.2）",
        anchor="text 组装 + 降级标记复位 + [doubt] 事件结构化摄入（doubt_ingest，写侧注入怀疑前置）",
        writes=("_last_turn_degraded", "_degraded_events"),
        reads=("user_input", "llm_output"),
    ),
    LifecycleStep(
        idx=2,
        name="encode",
        spec="计算 surprise = 0.5·Σπᵢ(σᵢ−sᵢ)²（mse 线性）；precision π 逐通道调制",
        anchor="自指回注 + encoder.encode（感官向量）+ 语义向量提取 + 元参数调整 + attractor.infer（surprise/熵 计算，π 读取调制）",
        writes=("_lifecycle_trace",),
        reads=("attractor", "purpose.precision", "embedder", "encoder"),
    ),
    LifecycleStep(
        idx=3,
        name="query",
        spec="构造检索 query（与注入内容同源但独立构造——防伪独立 query 维）",
        anchor="长时线索 = activation.state；情景语义 query 由 _retrieve_episodic 内部经 _encode_query_vector 独立构造（同源不同构）",
        writes=(),
        reads=("activation.state",),
    ),
    LifecycleStep(
        idx=4,
        name="retrieve",
        spec="recall 只读检索 → 检索时怀疑（§2.2，labile 窗口，仅内存态标记）",
        anchor="memory.recall(activation.state) + _retrieve_episodic(text)（只读；检索时怀疑投影入口——M3-1）",
        writes=(),
        reads=("memory", "attractor"),
    ),
    LifecycleStep(
        idx=5,
        name="integrate",
        spec="结果并入工作记忆；反流畅性：来源与内容分开评估（Sperber 2010），高流畅≠真（Hasher 1977），来源可信度进 π",
        anchor="decoder.decode（激活态 + 检索记忆 → context）+ self_ref.observe（自指观测）",
        writes=(),
        reads=("purpose.coherence", "decoder"),
    ),
    LifecycleStep(
        idx=6,
        name="doubt_check",
        spec="验证链判定（§2.3）：矛盾三选一 / 元数据排除 / 幂等——此处只登记，不改条目",
        anchor="injection_check（注入时怀疑登记 suspect→验证链）+ _apply_verification_conflicts（验证结果 CONFLICT 写侧应用，幂等记账）",
        writes=("entry 怀疑态", "fok_unresolved"),
        reads=("doubt_state", "verification_chain"),
    ),
    LifecycleStep(
        idx=7,
        name="commit",
        spec="写侧提交（store 提取层统一出口；写侧默认保守——宁可拦错不轻信）",
        anchor="store_episodic（memory 提取层统一出口；store_gray 口径保留；[doubt] 系统事件不入库）",
        writes=("episodic 条目集", "entry.confidence"),
        reads=("memory", "semantic_vector"),
    ),
    LifecycleStep(
        idx=8,
        name="state_update",
        spec="allostatic J 滑动设定点（§1.3）+ σ 更新 + π 更新",
        anchor="update_allostatic（滑动设定点）+ attractor.learn（J）+ 在线熵管理 + purpose.adjust（π）+ meta.update + memory.update/consolidate + precision_adapt.observe（锚点分散在 FEP 管线，观测在管线末端聚合）",
        writes=("J 内容/工作点/σ", "π"),
        reads=("activation", "purpose"),
    ),
    LifecycleStep(
        idx=9,
        name="emit",
        spec="观测：sandglass 事件流 + 状态快照（绝不可静默失败，G 模式）",
        anchor="_increment_turn（turn 唯一增量点）+ precision 观测 + _auto_snapshot + _maybe_publish_plastified + 生命周期状态汇总",
        writes=("turn 计数",),
        reads=("turn_count", "memory", "attractor"),
    ),
)

#: 规格步名 → 步定义（测试/审计查表用）
_LIFECYCLE_BY_NAME: Dict[str, LifecycleStep] = {
    s.name: s for s in TURN_LIFECYCLE_9
}


def lifecycle_step(name: str) -> LifecycleStep:
    """按规格步名取步定义（未知步名抛 KeyError——固化红线）。"""
    return _LIFECYCLE_BY_NAME[name]


# ---------------------------------------------------------------------- #
# 二、状态管理清单（规格 §1.2 表格落地：状态 / 归属模块 / 写者 / 读者）
# ---------------------------------------------------------------------- #


@dataclass(frozen=True)
class StateOwnership:
    """状态管理清单一行（机器可读；写者唯一性由 tests/test_loop_m5.py 断言）。"""

    state: str                        # 状态名（§1.2 表第一列）
    module: str                       # 归属模块（§1.2 表第二列）
    writer: str                       # 唯一写者（§1.2 表第三列）
    readers: Tuple[str, ...]          # 只读读者（§1.2 表第四列）
    spec_ref: str                     # 规格引用
    verified_by: str                  # M5 验证锚点（测试 + 代码位置）


STATE_OWNERSHIP: Tuple[StateOwnership, ...] = (
    StateOwnership(
        state="turn 计数",
        module="loop",
        writer="process_turn 唯一（loop._increment_turn —— 全库唯一 `+= 1` 站点；快照恢复为赋值恢复，非增量）",
        readers=("全局只读",),
        spec_ref="§1.2",
        verified_by="runtime/loop.py::_increment_turn；tests/test_loop_m5.py::TestUniqueTurnIncrement",
    ),
    StateOwnership(
        state="J 内容/工作点/σ 估计",
        module="attractor",
        writer="state_update 唯一（attractor.learn + update_allostatic 滑动设定点）",
        readers=("recall 只读投影",),
        spec_ref="§1.2 / §1.3",
        verified_by="core/hippocampus/attractor.py；tests/test_loop_m5.py::TestStateOwnershipRegistry",
    ),
    StateOwnership(
        state="π（精度，按通道/来源）",
        module="doubt",
        writer="ingest/consolidation（写侧时相；process_turn 内 purpose.adjust 为逐轮观测调制）",
        readers=("encode/integrate",),
        spec_ref="§1.2 / §2.3",
        verified_by="core/doubt/；tests/test_loop_m5.py::TestStateOwnershipRegistry",
    ),
    StateOwnership(
        state="entry.confidence",
        module="store",
        writer="注入/巩固（写侧；store 提取层统一出口）",
        readers=("recall 只读",),
        spec_ref="§1.2 / §1.5",
        verified_by="core/store/；tests/test_loop_m5.py::TestStateOwnershipRegistry",
    ),
    StateOwnership(
        state="entry 怀疑态",
        module="doubt",
        writer="三时相状态机（写侧时相：injection_check / consolidation_resolve）",
        readers=("recall 只读投影",),
        spec_ref="§1.2 / §2.1",
        verified_by="core/doubt/state_machine.py；tests/test_doubt_m3.py::TestThreePhaseStateMachine",
    ),
    StateOwnership(
        state="fok_unresolved（gap 登记）",
        module="doubt",
        writer="验证链（只登记，不改条目）",
        readers=("巩固期",),
        spec_ref="§1.2 / §2.2",
        verified_by="core/doubt/gap_registry.py；tests/test_loop_m5.py::TestStateOwnershipRegistry",
    ),
)

#: 状态名 → 归属（测试/审计查表用）
_OWNERSHIP_BY_STATE: Dict[str, StateOwnership] = {
    o.state: o for o in STATE_OWNERSHIP
}


def state_ownership(state: str) -> StateOwnership:
    """按状态名取归属行（未知状态名抛 KeyError——清单固化红线）。"""
    return _OWNERSHIP_BY_STATE[state]


# ---------------------------------------------------------------------- #
# 三、每轮生命周期记录器（纯内存观测；零副作用）
# ---------------------------------------------------------------------- #


@dataclass
class LifecycleTrace:
    """一轮 process_turn 的生命周期记录（纯内存，重启即失）。

    - record(step_name, **obs)：按**执行序**记录一步的观测；同一步重复
      记录即抛（固化红线：禁止旁路/重复执行），由 loop._record_lifecycle
      fail-open 接住（日志可见，不阻断主循环）；
    - summary(turn_count)：状态汇总——按规格序呈现每步观测，并给出
      broken 判定（missing / duplicated / unknown）；get_status 的
      lifecycle 观测块数据源。
    """

    turn_start: float = field(default_factory=time.time)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    _seen: set = field(default_factory=set)
    _rejected: List[str] = field(default_factory=list)

    def record(self, step_name: str, **obs: Any) -> None:
        """记录一步观测（执行序；同一步恰一次）。

        未知步名 / 同一步重复 → 记入 ``_rejected`` 并抛异常（固化红线；
        由 loop._record_lifecycle fail-open 接住，汇总 broken=True）。
        """
        if step_name not in _LIFECYCLE_BY_NAME:
            self._rejected.append(step_name)
            raise KeyError(
                f"未知生命周期步骤 {step_name!r}"
                f"（固化清单仅允许：{list(_LIFECYCLE_BY_NAME)}）")
        if step_name in self._seen:
            self._rejected.append(step_name)
            raise RuntimeError(
                f"生命周期步骤 {step_name!r} 重复记录——禁止旁路/重复执行")
        self._seen.add(step_name)
        self.steps.append({
            "step": step_name,
            "t_seconds": round(time.time() - self.turn_start, 4),
            "obs": dict(obs),
        })

    def recorded_names(self) -> List[str]:
        """已记录步名（执行序）。"""
        return [s["step"] for s in self.steps]

    def summary(self, turn_count: int) -> Dict[str, Any]:
        """状态汇总：按规格序呈现 9 步观测 + 完整性判定（broken 红线）。"""
        recorded = self.recorded_names()
        expected = [s.name for s in TURN_LIFECYCLE_9]
        missing = [n for n in expected if n not in recorded]
        # duplicated：被拒绝的重复/未知步（record 抛异常前已记入 _rejected）
        duplicated = sorted(set(self._rejected))
        unknown = [n for n in self._rejected if n not in expected]
        broken = bool(missing or duplicated or unknown)
        if broken:
            logger.warning(
                "M5 生命周期不完整（broken=True）：missing=%s duplicated=%s "
                "unknown=%s（9 步固化红线——禁止旁路）",
                missing, duplicated, unknown)
        steps_by_name = {s["step"]: s for s in self.steps}
        return {
            "turns": int(turn_count),
            "broken": broken,
            "missing": missing,
            "duplicated": duplicated,
            "unknown": unknown,
            "execution_order": recorded,       # 实际执行序（FEP 算法序）
            "spec_order": expected,            # 规格 §1.2 规范序
            "steps": {
                name: steps_by_name[name]["obs"]
                for name in expected if name in steps_by_name
            },
            "duration_seconds": round(time.time() - self.turn_start, 4),
        }

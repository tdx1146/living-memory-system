"""在线学习环（主循环）

活体记忆系统的主循环，对每轮对话执行完整的记忆循环：
编码输入 -> 推断 -> 学习 -> 调整目的 -> 巩固记忆 -> 检索记忆 -> 解码context -> 返回

其中"检索记忆"步骤（S1 修复）是关键：长时记忆通过 recall() 进入解码路径，
不再"只写不读"。

遵循架构文档第四节数据流定义。
"""

import os
import math
import time
import logging
from collections import deque
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import torch

from core.types import Activation, SensoryInput
from core.paths import get_snapshot_dir
from core.hippocampus.attractor import AttractorNetwork
from core.hippocampus.purpose import PurposeLayer
from core.hippocampus.memory import MemoryManager
from core.sensory.tokenizer import SimpleTokenizer
from core.sensory.embedder import SimpleEmbedder

from bridge.encoder import Encoder
from bridge.decoder import Decoder
from bridge.llm_bridge import LLMBridge

from persistence.snapshot import (
    Snapshot, snapshot_path_for, latest_path_for, sanitize_session_id,
)
from persistence.recovery import Recovery

# M2（recall 只读化）：只读四不变守卫 + 怀疑信号投影。
# core/recall 为纯 stdlib 模块（不 import 本仓库运行时），无循环依赖。
from core.recall.guard import (
    FourInvariantGuard, episodic_fingerprint)
from core.recall.suspicion import empty_suspicion

# M5（loop 重组，§7.1）：turn 生命周期 9 步固化 + 状态管理清单。
# runtime/lifecycle 为纯 stdlib（不 import 本仓库运行时），无循环依赖——
# 同 core/recall 防循环依赖策略。
from runtime.lifecycle import (
    LifecycleTrace, STATE_OWNERSHIP, TURN_LIFECYCLE_9)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
#  M5（规格 §5.2：claim 登记，machine-readable；实现与 runtime/claims.json
#  同源）。loop 是 turn 生命周期的编排者——claim 覆盖：唯一增量点 / 9 步
#  无旁路 / 状态管理清单单一写者 / 记录器零副作用 / 四不变衔接。
# ---------------------------------------------------------------------- #
MODULE_CLAIMS: dict = {
    "module": "runtime/loop",
    "milestone": "M5",
    "rewrite_spec": "四妹-LMS核心重写规格v2-20260817.md §1.2/§7.1 M5",
    "claims": {
        "turn_increment_unique": {
            "statement": "turn 计数唯一增量点：全库唯一的 `turn_count += 1` "
                         "只发生在 process_turn 内（loop._increment_turn，"
                         "emit 步）；快照恢复是赋值恢复不是增量；/recall、"
                         "/react、/dream 零增量",
            "verified_by": "tests/test_loop_m5.py::TestUniqueTurnIncrement",
        },
        "lifecycle_nine_steps_no_bypass": {
            "statement": "turn 生命周期固定 9 步（§1.2：ingest/encode/query/"
                         "retrieve/integrate/doubt_check/commit/state_update/"
                         "emit），process_turn 每轮 9 步各恰一次、无重复、"
                         "无未知步（broken=False 才合法——禁止旁路）",
            "verified_by": "tests/test_loop_m5.py::TestNineStepLifecycle",
        },
        "state_ownership_single_writer": {
            "statement": "状态管理清单（§1.2）六项状态各登记唯一写者：turn "
                         "计数=process_turn、J/工作点/σ=state_update、π=doubt "
                         "ingest/consolidation、entry.confidence=store 注入/"
                         "巩固、entry 怀疑态=doubt 三时相状态机（写侧）、"
                         "fok_unresolved=验证链；检索路径零写",
            "verified_by": "tests/test_loop_m5.py::TestStateOwnershipRegistry",
        },
        "lifecycle_trace_side_effect_free": {
            "statement": "生命周期记录器零副作用：只读观测、绝不 setattr "
                         "条目、绝不写持久层、不进快照；/recall 与 /react "
                         "只读口不产生生命周期记录（不 process_turn 则无记录）",
            "verified_by": "tests/test_loop_m5.py::"
                           "TestLifecycleTraceSideEffectFree",
        },
        "readonly_four_invariants_hold_after_m5": {
            "statement": "M5 重组后只读四不变仍成立：一次 recall（/recall 与 "
                         "/react 检索段）执行前后 turn/episodic 条目集/J/σ "
                         "零增量；四不变守卫强制开启",
            "verified_by": "tests/test_loop_m5.py::"
                           "TestReadonlyFourInvariantsAfterM5",
        },
        "purpose_drift_gate_in_doubt_check": {
            "statement": "目的时相（总任务书 §二.5）并入 9 步生命周期 "
                         "doubt_check 步（不加步——9 步固化红线不变）：每轮 "
                         "purpose_drift_check 消费本轮活动信号 → 闸门信号"
                         "「是否偏离+理由」（Pan 警示：无符合度分数）；"
                         "verdict=drifted 时登记 [doubt] purpose-drift（gap "
                         "A 类登记，可审计）；uncertain=未判（只进观测待补判，"
                         "不登记）；P2-B 口径修正：只读轮（无写侧活动）Q1 "
                         "豁免不判 drifted；开关关/异常 → 中性/None 零参与"
                         "（fail-open）",
            "verified_by": "tests/test_purpose_drift.py + "
                           "tests/test_loop_m5.py::TestNineStepLifecycle",
        },
        "reconsolidation_wired_in_consolidation_phase": {
            "statement": "R3（C1 接线，语义决策 D-2026-08-18-01 运行时落地）："
                         "loop 巩固期（doubt_check 后、state_update 前——9 "
                         "步固化不加步）调用 reconsolidation_queue.maybe_rewrite"
                         "（三闸门机器防线 G1 候选在队/G2 巩固期时相/G3 写侧"
                         "委托 state_machine 巩固时相入口）；入队只允许写侧"
                         "时相（注入 suspect 标记后 / 去稳定化 labile 登记，"
                         "写侧入队即落盘）；检索路径零改写保持（只读四不变"
                         "不破坏——/recall 与 /react 只读口不经过巩固期）",
            "verified_by": "tests/test_loop_reconsolidation_labile.py",
        },
        "e3_switch_off_bitwise_equivalent": {
            "statement": "E3（自我怀疑驱动的主动调节，dandan 拍板 2026-08-20）："
                         "总开关 LMS_E3_ENABLED 默认 0=关——关时选择器/重激活/"
                         "satiety/min-age 全部新路径零参与，行为与开关引入前"
                         "逐位一致（e3_reactivate 返回 enabled:false，min-age "
                         "闸门不生效）；开时受 LMS_E3_MIN_AGE_TURNS（默认 1）/ "
                         "LMS_E3_SATIETY_COOLDOWN_H（默认 12）/ "
                         "LMS_E3_REACTIVATE_MAX（默认 2）治理",
            "verified_by": "tests/test_e3_reactivation.py",
        },
    },
}


# ---------------------------------------------------------------------- #
#  E3 env 开关（自我怀疑驱动的主动调节，dandan 拍板 2026-08-20 22:14）。
#  总开关默认 0=关 → 全部新路径零参与（行为与开关引入前逐位一致）。
#  子开关在总开关开时才生效（§5.3 写侧默认保守先例）。
# ---------------------------------------------------------------------- #

_ENV_E3_ENABLED = "LMS_E3_ENABLED"
_ENV_E3_MIN_AGE_TURNS = "LMS_E3_MIN_AGE_TURNS"
_ENV_E3_REACTIVATE_MAX = "LMS_E3_REACTIVATE_MAX"
_ENV_E3_TOPIC_LEN = "LMS_E3_REACTIVATE_TOPIC_LEN"


def _e3_enabled() -> bool:
    """E3 总开关（LMS_E3_ENABLED，默认 0=关）。关 → 全部新路径零参与。"""
    raw = os.environ.get(_ENV_E3_ENABLED, "0")
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _e3_min_age_turns() -> int:
    """min-age 闸门轮数（LMS_E3_MIN_AGE_TURNS，默认 1）。

    E3 总开关关 → 0（不闸——行为与 E3 引入前逐位一致：同轮即消保持）。
    """
    if not _e3_enabled():
        return 0
    try:
        return max(0, int(os.environ.get(_ENV_E3_MIN_AGE_TURNS, "1") or 1))
    except (TypeError, ValueError):
        return 1


def _e3_reactivate_max() -> int:
    """单次闭环最多重激活条数（LMS_E3_REACTIVATE_MAX，dandan 拍板放宽=2）。"""
    try:
        return max(1, int(os.environ.get(_ENV_E3_REACTIVATE_MAX, "2") or 2))
    except (TypeError, ValueError):
        return 2


def _e3_topic_len() -> int:
    """线索/证据摘要长度上限（LMS_E3_REACTIVATE_TOPIC_LEN，默认 120）。"""
    try:
        return max(40, int(os.environ.get(_ENV_E3_TOPIC_LEN, "120") or 120))
    except (TypeError, ValueError):
        return 120


# ---------------------------------------------------------------------- #
#  T2.3 检索扩容：归档检索线程池（单 worker，只读路径）
# ---------------------------------------------------------------------- #
# /recall 的归档补充检索放进独立线程，由 future.result(timeout=...) 控制超时：
# 超时即跳过归档只回内存（fail-open），绝不拖垮响应。单 worker 串行化归档
# 扫描，避免多请求并发读放大；懒创建避免 fork/导入副作用。
_ARCHIVE_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _get_archive_executor() -> ThreadPoolExecutor:
    """懒创建归档检索线程池（线程名 lms-archive）。"""
    global _ARCHIVE_EXECUTOR
    if _ARCHIVE_EXECUTOR is None:
        _ARCHIVE_EXECUTOR = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='lms-archive')
    return _ARCHIVE_EXECUTOR


class LivingMemoryLoop:
    """在线学习环：活体记忆系统的主循环。

    对每轮对话执行完整的记忆循环：
    1. 编码输入（用户输入+LLM输出 -> 感官向量）
    2. FEP推断（感官向量 -> 激活态）
    3. FEP学习（更新J矩阵）
    3.5 在线熵管理（熵过高增强正交化 / 熵过低放松正交化）
    4. 调整目的（更新precision）
    5. 记忆更新与巩固（短时更新 + 定期短时->长时迁移）
    5.5 记忆检索（用激活态检索长时记忆——S1 修复，记忆不再只写不读）
    6. 解码context（激活态 + 检索到的记忆 -> LLM可理解的context）
    7. 自动快照（可选，按间隔保存状态——G4 修复）
    8. 返回context

    属性:
        attractor: 吸引子网络
        purpose: 目的层
        memory: 记忆管理器
        encoder: 编码器
        decoder: 解码器
        bridge: LLM桥接器（可选）
        snapshot: 快照管理器
        recovery: 恢复管理器
        turn_count: 当前对话轮次
        last_activation: 上一轮的激活态
    """

    def __init__(self, config: dict):
        """初始化所有组件。

        参数:
            config: 配置字典，支持以下键:
                - num_nodes: 节点数（默认256）
                - input_dim: 输入维度（默认64）
                - decoder_mode: 解码模式（默认'text'）
                - consolidation_interval: 巩固间隔（默认5）
                - llm_api: LLM API配置（可选）
                - attractor: 自定义吸引子网络（可选，覆盖默认）
                - purpose: 自定义目的层（可选，覆盖默认）
                - memory: 自定义记忆管理器（可选，覆盖默认）
                - tokenizer: 自定义分词器（可选）
                - embedder: 自定义嵌入器（可选）
                - encoder: 自定义编码器（可选）
                - decoder: 自定义解码器（可选）
                - llm_bridge: 自定义LLM桥接器（可选）

                FEP 参数（S2 修复：自建组件时生效，注入实例时不覆盖）:
                - seed, temperature: 吸引子网络随机种子与 Langevin 温度
                - complexity_weight, orth_weight: 自由能复杂性与正交化权重
                - precision_lr, precision_min, precision_max,
                  coherence_threshold, min_history_length, meta_window,
                  max_history, habituation_rate, activation_threshold:
                  目的层参数（N4: activation_threshold 解耦习惯化阈值与 temperature）
                - short_term_decay, long_term_decay, transfer_rate,
                  replay_count, replay_weight, consolidation_decay,
                  buffer_capacity: 记忆管理器参数
                - num_infer_steps: FEP 推断步数（默认10）
                - learning_rate: FEP 学习率（默认0.01）

                运行时参数:
                - auto_snapshot: 是否自动快照（默认False）
                - auto_snapshot_interval: 自动快照间隔（默认50）
                - snapshot_dir: 快照保存目录
        """
        self.config = config

        # 会话标识（T1.1/P0-5：快照按会话命名与元数据持久化需要）。
        # SessionManager 注入 config['session_id']；未注入时默认 'default'。
        self.session_id = str(config.get('session_id', 'default'))

        # 核心参数
        num_nodes = config.get('num_nodes', 256)
        input_dim = config.get('input_dim', 64)
        consolidation_interval = config.get('consolidation_interval', 5)

        # FEP 参数（S2 修复：从 config 读取，接线 CoreConfig 字段）
        seed = config.get('seed', 42)
        temperature = config.get('temperature', 0.05)

        # E-P2-1: 设备管理——从 config 读取 device 标识，传递给所有核心组件
        device = config.get('device', 'auto')

        # 初始化核心组件（允许外部注入自定义实现）
        # 注入时不覆盖其内部参数；自建时用 config 驱动全部 FEP 参数
        # E-P2-1: 自建时传入 device，组件构造函数内部解析为 torch.device
        self.attractor = config.get('attractor') or AttractorNetwork(
            num_nodes, input_dim,
            seed=seed,
            temperature=temperature,
            device=device,
            # T2.8/P2-1：惊讶度归一化开关（None → 读 LMS_NORM_SURPRISE）
            norm_surprise=config.get('norm_surprise'),
        )
        if not config.get('attractor'):
            # 仅在自建时设置（不覆盖外部注入的实例）
            self.attractor.complexity_weight = config.get(
                'complexity_weight', 0.01)
            self.attractor.orth_weight = config.get('orth_weight', 0.5)

        self.purpose = config.get('purpose') or PurposeLayer(
            input_dim,
            precision_lr=config.get('precision_lr', 0.1),
            precision_min=config.get('precision_min', 0.1),
            precision_max=config.get('precision_max', 10.0),
            coherence_threshold=config.get('coherence_threshold', 0.5),
            coherence_direction_weight=config.get(
                'coherence_direction_weight', 0.5),
            coherence_magnitude_weight=config.get(
                'coherence_magnitude_weight', 0.5),
            min_history_length=config.get('min_history_length', 5),
            meta_window=config.get('meta_window', 10),
            max_history=config.get('max_history', 100),
            habituation_rate=config.get('habituation_rate', 0.05),
            activation_threshold=config.get('activation_threshold', 0.3),
            device=device,
        )

        self.memory = config.get('memory') or MemoryManager(
            num_nodes,
            short_term_decay=config.get('short_term_decay', 0.8),
            long_term_decay=config.get('long_term_decay', 0.999),
            transfer_rate=config.get('transfer_rate', 0.1),
            replay_count=config.get('replay_count', 10),
            replay_weight=config.get('replay_weight', 0.01),
            consolidation_decay=config.get('consolidation_decay', 0.5),
            buffer_capacity=config.get('buffer_capacity', 100),
            device=device,
            # T2.8/P2-1：回放权重钳制 + 潜变量归一化（None → 读环境变量）
            replay_surprise_cap=config.get('replay_surprise_cap'),
            norm_latent=config.get('norm_latent'),
        )
        self.tokenizer = config.get('tokenizer') or SimpleTokenizer()
        self.embedder = config.get('embedder') or SimpleEmbedder(dim=input_dim)

        # 初始化桥接组件
        self.encoder = config.get('encoder') or Encoder()
        decoder_mode = config.get('decoder_mode', 'text')
        self.decoder = config.get('decoder') or Decoder(mode=decoder_mode)

        # LLM桥接器（可选）
        self.bridge: Optional[LLMBridge] = None
        if config.get('llm_bridge'):
            self.bridge = config['llm_bridge']
        elif config.get('llm_api'):
            self.bridge = LLMBridge(config['llm_api'])

        # 持久化
        self.snapshot = Snapshot()
        self.recovery = Recovery()

        # 运行状态
        self.turn_count = 0
        self.last_activation: Optional[Activation] = None
        self.consolidation_interval = consolidation_interval
        # 在线熵管理：记录最后一轮的 entropy_ratio（entropy / max_entropy）
        self.last_entropy_ratio = 0.0

        # 体验层 A（设计 v1.1 §3.2）：/react 实时反应的惊讶度自有窗口。
        # 纯内存 deque（maxlen=200），不落盘、不进快照、重启即失；
        # 与 decoder 共享窗口（self.surprise_history，/chat 路径在用）隔离，
        # /react 解读零污染 /chat 路径（惊讶度修复 P0-12 零回归）。
        self.react_surprise_history: deque = deque(maxlen=200)

        # M2（recall 只读化）：最近一次只读检索的怀疑信号投影。
        # 纯内存态（同 react_surprise_history 先例：不落盘、不进快照、
        # 重启即失）；/recall 与 /react 响应 suspicion 区段数据源。
        self.last_recall_suspicion: dict = empty_suspicion()

        # M3-1（核心重建规格 v2 §二，自我怀疑原生机制）：三时相怀疑状态机
        # + 验证链。开关：注入时怀疑 LMS_DOUBT_INJECTION_ENABLED（默认 1=
        # 开——机制本体）；验证链 LMS_VERIFICATION_CHAIN_ENABLED（默认 0=
        # 关——§5.3 写侧默认保守，假阳性演练通过才 env 开启）。纯进程内存
        # 状态（同 precision_adapt 先例：重启即失、快照不落盘）。
        from core.doubt.state_machine import DoubtStateMachine
        from core.doubt.verification_chain import VerificationChain
        self.verification_chain = VerificationChain()
        self.doubt_state = DoubtStateMachine(
            verification_chain=self.verification_chain)

        # 体验层 D（设计 v1.1 §6.2）：去稳定化的近 200 轮 surprise 窗口
        # （G1 思路，LMS 内自有 deque；与 /react 窗口独立）
        self.destab_surprise_window: deque = deque(maxlen=200)
        # 体验层 D：信息缺口登记（怀疑灯数据源，A/B 类进灯、C 类仅诊断）
        from core.doubt.gap_registry import GapRegistry
        self.gap_registry = GapRegistry()

        # R3（C1 接线，语义决策 D-2026-08-18-01 运行时落地）：再巩固候选队列
        # （跨重启持久化）+ 巩固期受控改写（三闸门：候选在队/巩固期触发/
        # 改写受控——机器防线）。loop 在 turn 生命周期巩固期（doubt_check
        # 后、state_update 前）调用 maybe_rewrite；检索路径零改写保持
        # （队列只读接口绝不 setattr 条目——只读四不变不破坏）。
        # 测试可注入 reconsolidation_queue 实例或 reconsolidation_queue_path
        # （隔离路径）；缺省 → env LMS_DOUBT_RECONSOLIDATION_QUEUE_PATH >
        # 数据目录。开关 LMS_DOUBT_RECONSOLIDATION_ENABLED（默认 1）关 →
        # 队列全部路径零参与（行为与开关引入前完全一致）。
        from core.doubt.reconsolidation_queue import ReconsolidationQueue
        self.reconsolidation_queue = (
            config.get('reconsolidation_queue')
            or ReconsolidationQueue(path=config.get('reconsolidation_queue_path')))
        # E3（自我怀疑驱动的主动调节，dandan 拍板 2026-08-20 22:14）：
        #   - satiety 状态机（冷却/计数/武装-待机——终止信号防 OCD 化，
        #     设计 §3.2 ⑤）；纯进程内存态（同 gap_registry 先例）。
        #   - min-age 闸门滚动快照（候选至少活过当轮，根因 4 修复）。
        #   - 闭环观测（最近一次重激活 / 闭环计数）。
        # 总开关 LMS_E3_ENABLED 默认 0=关 → 全部新路径零参与（逐位等价）。
        from core.doubt.satiety import SatietyGate
        self.satiety_gate = SatietyGate()
        self._e3_turn_start_keys = None          # 懒创建 deque（min-age 数据源）
        self._e3_pending_targets: list = []      # 上轮重激活目标（satiety 消解扫）
        self._e3_last_reactivation = None        # 观测：最近一次闭环结果
        self._e3_closed_loops: int = 0           # 观测：闭环计数
        # 上一轮巩固期再巩固是否真实发生（目的时相 reconsolidated 信号——
        # 目的检查在 doubt_check 步、再巩固在其后，用上一轮结果保持诚实；
        # 纯进程内存态，重启即失，同 gap_registry 先例）。
        self._last_turn_reconsolidated: bool = False

        # 阶段 3（precision 三层动态化，质疑自动校准）：全局怀疑强度状态机。
        # 治理开关 LMS_PRECISION_ADAPT（默认 1=开；0=关 → None，全部路径
        # 零参与，行为与开关引入前完全一致——8/10 治理开关先例风格）。
        # 纯进程内存状态（同 gap_registry 先例：重启即失、快照不落盘）。
        from core.doubt.precision_adapt import (
            PrecisionAdaptState, precision_adapt_enabled)
        if precision_adapt_enabled(config.get('precision_adapt')):
            self.precision_adapt = PrecisionAdaptState()
        else:
            self.precision_adapt = None
        # 负性证据标记：本轮发生过证伪（conflict/去稳定化）→ observe_surprise
        # 时 is_negative=True（对称性约束：坏消息 PE 不被系统性低估）
        self._pending_negative_evidence: bool = False

        # 论文机制 A（M4 原生并入，2026-08-18）：allostatic J 滑动设定点已原生
        # 并入 core/hippocampus/attractor.py——attractor 在 __init__ 按 env
        # （LMS_J_ALLOSTATIC，默认 0=关 → 固定 J 行为完全不变，回滚干净）自建
        # AllostaticJController 并写回 j_target_norm（learn 钳制按动态值执行）。
        # 原外挂 runtime/allostatic_j.py 已删除——loop 不再 import 外挂，
        # process_turn/get_status 均走 attractor 原生方法。
        # 兼容引用：loop.allostatic_j = attractor.allostatic（None = 关；
        # getattr 防御外部注入的 attractor 缺该属性）。
        self.allostatic_j = getattr(self.attractor, 'allostatic', None)

        # M5（loop 重组，§7.1）：turn 生命周期 9 步固化 + 状态管理清单。
        # lifecycle_trace：当前轮进行中的生命周期记录器（process_turn 内
        # 创建，结束时转存 last_turn_lifecycle——纯内存，不落盘、不进快照，
        # 同 react_surprise_history / gap_registry 先例：重启即失）。
        # last_turn_lifecycle：最近一轮的**状态汇总**（9 步观测 + broken
        # 判定；get_status lifecycle 观测块数据源）。/recall 与 /react
        # 只读口不创建记录器（不 process_turn 则无记录——只读四不变）。
        self.lifecycle_trace: Optional[LifecycleTrace] = None
        self.last_turn_lifecycle: Optional[dict] = None

        # 元可塑性控制器（可选，根据 config 决定是否启用）
        self.meta = None
        if config.get('meta_enabled', True):
            from core.meta.meta_plasticity import MetaPlasticityController
            meta_config = {
                'meta_interval': config.get('meta_interval', 10),
                'meta_lr': config.get('meta_lr', 0.01),
                'bounds_min': config.get('meta_bounds_min', 0.5),
                'bounds_max': config.get('meta_bounds_max', 2.0),
                'surprise_window': config.get('meta_surprise_window', 20),
                'orth_alpha': config.get('meta_orth_alpha', 1.0),
                'temp_beta': config.get('meta_temp_beta', 5.0),
                'cw_gamma': config.get('meta_cw_gamma', 1.0),
                'lr_delta': config.get('meta_lr_delta', 2.0),
                'shy_target_norm': config.get('meta_shy_target_norm', 10.0),
                # T2.8/P2-2：惰性规则开关（None → 读 LMS_META_LAZY）
                'lazy': config.get('meta_lazy'),
            }
            self.meta = MetaPlasticityController(meta_config)

        # 自指回路（可选，根据 config 决定是否启用）
        # Phase 0：默认关闭，self_ref_enabled=False 时 self_ref 保持 None，
        # 所有自指代码块在 `if self.self_ref is not None:` 守卫内不执行
        self.self_ref = None
        self._prev_activation = None  # 供 autocorr 计算（Phase 0 预留）
        # 供 L5 外部新颖度计算（Phase 1 预留）。
        # 保守策略：generate_echo 内部自行管理 prev_ext_sensory 跟踪，
        # loop 不向 generate_echo 新增参数；此字段保留供未来 loop 层直接
        # 计算 ext_novelty 时使用。
        self._prev_ext_sensory = None
        if config.get('self_ref_enabled', False):
            from core.hippocampus.self_referential import SelfReferentialLoop
            self.self_ref = SelfReferentialLoop(
                encoder=self.encoder, tokenizer=self.tokenizer,
                embedder=self.embedder, config=config,
                device=self.attractor.device,
            )

        # Phase 3.3: 可选 LLM 增强自述蒸馏
        # 当 self_ref_llm_distill_enabled=True 且 bridge 可用时，向 SelfReferentialLoop
        # 注入 LLMSelfVoiceDistiller 实例。LLM 不可用/失败时自动降级为规则蒸馏。
        if (config.get('self_ref_llm_distill_enabled', False)
                and self.self_ref is not None
                and self.bridge is not None):
            from core.hippocampus.self_referential import LLMSelfVoiceDistiller
            self.self_ref.llm_distiller = LLMSelfVoiceDistiller(
                llm_bridge=self.bridge,
                interval=config.get('self_ref_llm_distill_interval', 5),
            )

        # 做梦引擎（懒加载，首次调用 get_dream_engine() 时创建）
        self.dream_engine = None

        # T2.8/P2-6：embed 熔断器（LMS_EMBED_CIRCUIT=1 启用，默认 0）。
        # 保护 _encode_query_vector（/recall 只读检索路径）：embed 服务
        # 连续失败 3 次 → 熔断 5 分钟，期间 /recall 快速返回空而非卡 10s。
        from core.sensory.circuit_breaker import EmbedCircuitBreaker
        self._embed_circuit = EmbedCircuitBreaker(
            enabled=config.get('embed_circuit'))

        # 提取层 v1.4（S1-7/P3）：写侧熔断降级观测——
        # degraded_events：写侧语义向量 embed 熔断降级累计次数（本轮不落
        #   episodic）；last_turn_degraded：最近一轮 process_turn 是否降级
        #   （/store 据此返回 503；/chat 塑形降级但响应照常）。
        self.degraded_events: int = 0
        self.last_turn_degraded: bool = False

        logger.info(
            f"LivingMemoryLoop已初始化 "
            f"(nodes={self.attractor.num_nodes}, dim={self.attractor.input_dim})"
        )

    def _increment_turn(self) -> int:
        """turn 计数**唯一增量点**（§1.2：process_turn 唯一；全库唯一
        ``+= 1`` 站点——M5 固化，grep 可证）。

        - emit 步调用；/recall、/react、/dream、save/load 均零增量；
        - ``load_state`` 的快照恢复是**赋值恢复**（不是增量），不经过本
          方法——已在 STATE_OWNERSHIP（turn 计数）登记说明。
        """
        self.turn_count += 1
        return self.turn_count

    def _record_lifecycle(self, step: str, **obs) -> None:
        """M5：记录当前轮生命周期一步的观测（fail-open——G 模式以日志
        可见，不以静默吞掉；记录异常绝不阻断主循环）。

        固化红线：同一步重复记录/未知步名由 LifecycleTrace.record 抛
        RuntimeError/KeyError → 本方法告警 → 状态汇总 broken=True
        （missing/duplicated/unknown）——机器可验，测试断言。
        """
        trace = self.lifecycle_trace
        if trace is None:
            return
        try:
            trace.record(step, **obs)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "M5 生命周期步骤 %r 记录失败（fail-open，汇总将 broken=True）:"
                " %s", step, e)

    def process_turn(self, user_input: str, llm_output: str = "") -> str:
        """处理一轮对话，执行完整的记忆循环。

        流程（M5 loop 重组——turn 生命周期**固定 9 步，禁止旁路**；规格
        四妹-LMS核心重写规格v2-20260817.md §1.2；固化清单见
        runtime/lifecycle.py ``TURN_LIFECYCLE_9``，每步观测进状态汇总）：
        step 1  ingest        写侧入口：输入进入 + [doubt] 事件结构化摄入
                              （注入时怀疑前置，§2.2）
        step 2  encode        编码输入：自指回注 + 感官向量 + 语义向量 +
                              FEP infer（surprise/熵 计算，π 读取调制）
        step 3  query         构造检索 query（长时线索 activation.state +
                              情景语义 query 独立构造）
        step 4  retrieve      只读检索（memory.recall + _retrieve_episodic；
                              检索时怀疑投影入口）
        step 5  integrate     结果并入工作记忆（decoder.decode + 自指观测；
                              反流畅性：来源与内容分开评估）
        step 6  doubt_check   验证链判定登记（注入时怀疑 + conflict 应用，
                              只登记不改条目）+ 目的时相判定（每轮
                              [doubt] purpose-drift 判定——闸门信号「是否
                              偏离+理由」，drifted 登记 gap；§二.5）
        step 7  commit        写侧提交（store_episodic——store 提取层统一
                              出口；写侧默认保守）
        step 8  state_update  allostatic J 滑动设定点 + J/σ 更新（learn）+
                              π 更新（purpose.adjust）+ 记忆更新/巩固
        step 9  emit          观测：turn 唯一增量 + 自动快照 + plastified
                              事件 + 状态汇总

        参数:
            user_input: 用户输入文本
            llm_output: LLM输出文本（上一轮的，首轮可为空）

        返回:
            记忆context文本（供LLM查询使用）
        """
        # M5（§7.1）：本轮生命周期记录器（9 步固化——每步观测进状态汇总；
        # 纯内存，重启即失；/recall 与 /react 只读口不创建——只读四不变）。
        self.lifecycle_trace = LifecycleTrace()
        # 本轮尚未完成（正常路径在 emit 步落状态汇总；异常路径 last_turn_
        # lifecycle 保持 None——get_status 观测块给出"未完成"提示）。
        self.last_turn_lifecycle = None
        # M5（§1.2 状态管理清单）：本轮状态增量基准（turn 前的 J/σ/π
        # 观测值——state_update 步据此汇总"期望增量/零增量"断言；与 M4
        # 规格 §1.3 emit 落观测同源）。
        _j_norm_before = float(torch.norm(self.attractor.J, p='fro').item())
        _sigma_norm_before = float(torch.norm(self.attractor.sigma).item())
        _precision_mean_before = float(self.purpose.get_precision().mean())

        # 目的时相（总任务书 §二.5）：本轮怀疑计数基准——doubt_state 的
        # injection_checks / injection_suspect_marked 是**累计**计数，
        # purpose-drift 判定用差分取本轮增量（含本轮 [doubt] 摄入事件），
        # 与 commit 步条目数基准同款口径（观测 = 本轮增量）。
        _injection_checks_before = self.doubt_state.injection_checks
        _suspect_marked_before = self.doubt_state.injection_suspect_marked

        # ────────────────────────────────────────────────────────── #
        # M5 step 1/9 ingest —— 写侧入口（输入进入 + 注入时怀疑前置）
        # ────────────────────────────────────────────────────────── #
        # 1. 编码输入
        text = f"用户: {user_input}\n助手: {llm_output}" if llm_output else user_input
        # E3（min-age 闸门数据源，根因 4 修复）：记录本 turn 起始的再巩固
        # 候选键快照——候选入队后至少隔 LMS_E3_MIN_AGE_TURNS（默认 1）轮才
        # 允许巩固期消费（根治"同轮即消"：候选至少活过当轮，做梦 M6 能扫到
        # labile/suspect）。总开关关 → 零参与（_e3_record_turn_start_keys
        # 内部判定，行为与 E3 引入前逐位一致）。
        self._e3_record_turn_start_keys()
        # 提取层 v1.4（S1-7/P3）：每轮开始重置降级标记（观测粒度=单轮）
        self.last_turn_degraded = False
        sensory_input = self.encoder.encode(text, self.tokenizer, self.embedder)
        # 编码降级（embed 熔断 → 零向量 SensoryInput）：FEP 照常（零向量
        # Hebbian 学习 outer(0,0)=0，J 不变，安全）；本轮不写 episodic
        # （语义向量同样会熔断失败 → None）。
        if sensory_input.metadata.get('degraded'):
            self.last_turn_degraded = True
            self.degraded_events += 1

        # 体验层 D（设计 v1.1 §6.6）：结构化怀疑摄入（[doubt] 前缀解析，
        # fail-open）。无前缀/解析失败 = 普通塑形，逐字节不变（红线）。
        # 阶段 3：conflict 证伪事件 → conformal 校准集 + 负性证据标记
        # （对称性约束：坏消息进入全局怀疑基线，不被系统性低估）。
        _doubt_action = None
        try:
            from core.doubt.doubt_ingest import ingest as doubt_ingest
            ev = doubt_ingest(self, text)
            if ev:
                _doubt_action = ev.get('action')
                # E3（satiety 恢复路径，设计 §3.2 ⑤）：新 [doubt] 事件 →
                # 重新武装（清待机 + 重置闭环计数——系统重新开始怀疑）。
                # fail-open：重激活异常绝不影响怀疑摄入结果。
                _gate = getattr(self, 'satiety_gate', None)
                if _gate is not None:
                    try:
                        _gate.rearm()
                    except Exception:  # pylint: disable=broad-except
                        pass
            if (self.precision_adapt is not None and ev
                    and ev.get('action') == 'rebutted'):
                self._pending_negative_evidence = True
                hit = ev.get('entry')
                if hit is not None:
                    try:
                        self.precision_adapt.record_rebuttal(hit)
                    except Exception:
                        pass
        except Exception:
            pass

        # M5 step 1/9 ingest 观测（写侧入口：输入进入 + [doubt] 结构化摄入；
        # 无前缀 = 普通塑形，逐字节不变——红线）。
        self._record_lifecycle(
            "ingest",
            text_len=len(text),
            degraded=self.last_turn_degraded,
            doubt_action=_doubt_action,
        )

        # ★ 插入点 A：自指回注（提取为 _inject_self_ref）
        # 在编码后、推断前，将上一轮蒸馏的自述以自适应权重回注到感官向量。
        # 默认关闭（self_ref is None）时此块完全跳过，sensory_input 保持原样。
        sensory_input, alpha_t, is_self_ref_dominant = self._inject_self_ref(
            sensory_input)

        # 1.5 获取语义向量（用于情景记忆存储与检索）
        # PretrainedEmbedder 提供 embed_text；SimpleEmbedder 无此方法时跳过
        # 提取层 v1.4（S1-7/P3 ②③）：写侧两处 embed 包熔断——
        #   ② semantic_vector 失败 → None＋degraded（本轮不写 episodic，
        #     sem 是必需向量；不落僵尸：无向量条目=检索不可达）；
        #   ③ raw_semantic_vector 失败 → None（退化为投影向量，与读侧
        #     同语义；②成功时照常落库）。
        semantic_vector = None
        raw_semantic_vector = None
        if hasattr(self.embedder, 'embed_text'):
            # 写侧与 encode 共用默认熔断器单例（S1-7：同一 embed 服务、同一
            # 熔断状态——否则 encode 熔断后此处仍 CLOSED 会裸抛原始异常）。
            from core.sensory.circuit_breaker import (
                CircuitOpenError, get_default_embed_circuit,
            )
            try:
                semantic_vector = get_default_embed_circuit().call(
                    self.embedder.embed_text, text)
            except CircuitOpenError:
                logger.warning(
                    "embed 熔断中：写侧语义向量降级为 None"
                    "（本轮不写 episodic，/store 将 503）")
                semantic_vector = None
                self.last_turn_degraded = True
                self.degraded_events += 1
            # 384 维原始语义向量（投影前），用于 episodic buffer 高精度检索
            # PretrainedEmbedder 提供 embed_text_raw；无此方法时退化为 None
            # （如自定义 embedder 仅有 embed_text），此时退化为用投影向量
            if semantic_vector is not None and hasattr(
                    self.embedder, 'embed_text_raw'):
                try:
                    raw_semantic_vector = get_default_embed_circuit().call(
                        self.embedder.embed_text_raw, text)
                except CircuitOpenError:
                    logger.warning(
                        "embed 熔断中：raw 语义向量降级为 None"
                        "（退化为投影向量）")
                    raw_semantic_vector = None

        # --- 元可塑性：计算调整后的参数（E-P2-5: 不修改实例属性）---
        _effective_lr, _meta_temp, _meta_orth, _meta_cw = \
            self._get_meta_adjusted_params()

        # ★ Phase 2: 学习隔离——自指主导轮次学习率减半
        # 防止自指回路的 Hebbian 相关过度强化 J 矩阵中的自指方向
        if is_self_ref_dominant:
            _effective_lr = _effective_lr * 0.5
            logger.info(
                "Phase 2 学习隔离: 自指主导轮次, "
                "学习率减半 → %.6f", _effective_lr)

        # 2. FEP推断（E-P2-5: 通过 temperature_override 传递元调整后的温度）
        precision = self.purpose.get_precision()
        # Phase 3.1: 状态级递归——将上一轮 activation.state 作为 infer 种子偏置
        initial_state = None
        if self.self_ref is not None:
            initial_state = self.self_ref.generate_state_seed(
                self.attractor.sigma)
        activation = self.attractor.infer(
            sensory_input.vector, precision,
            num_steps=self.config.get('num_infer_steps', 10),
            temperature_override=_meta_temp,
            initial_state=initial_state,
        )

        # M5 step 2/9 encode 观测（编码输入 + surprise 计算落点——规格
        # 步骤 2：surprise=0.5·Σπᵢ(σᵢ−sᵢ)²，π 读取逐通道调制）。
        self._record_lifecycle(
            "encode",
            surprise=round(float(activation.surprise), 6),
            entropy=round(float(activation.entropy), 6),
            precision_mean=round(float(precision.mean()), 6),
            semantic_available=bool(semantic_vector is not None),
            raw_semantic_available=bool(raw_semantic_vector is not None),
            is_self_ref_dominant=bool(is_self_ref_dominant),
            self_ref_alpha=round(float(alpha_t), 4),
        )

        # 2.5 论文机制 A（默认关）：allostatic J 滑动设定点（M4 原生）
        # 用本轮 surprise + σ 激活态更新 J_target，并在 learn 前写回
        # attractor.j_target_norm——learn() 的范数钳制按动态设定点执行
        # （attractor 原生持有控制器；开关关时零参与）。fail-open。
        try:
            self.attractor.update_allostatic(
                activation.surprise, activation.state)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("allostatic J 更新失败（fail-open）: %s", e)

        # 3. FEP学习（E-P2-5: 通过 override 参数传递元调整后的权重）
        self.attractor.learn(
            activation, sensory_input.vector, _effective_lr,
            orth_weight_override=_meta_orth,
            complexity_weight_override=_meta_cw,
        )

        # 3.5 在线熵管理（提取为 _manage_entropy）
        # FEP学习之后、调整目的之前，根据当前激活熵主动干预：
        # 熵过高（混沌饱和）则增强正交化压力驱散相似表示；
        # 熵过低（僵化）则放松正交化压力允许更多激活。
        self._manage_entropy(activation, sensory_input, _effective_lr)

        # 4. 调整目的
        self.purpose.adjust(activation.surprise, activation)

        # --- 元可塑性：收集信号并更新 ---
        # ★ Phase 2: 学习隔离——跳过自指主导轮次的 meta.update
        # 自指锁定导致 surprise 持续下降，若馈入 meta 会将 lr_multiplier
        # 压向 0.5，间接抑制系统对外部真实输入的学习能力（审视报告 9.5）
        if self.meta is not None and not is_self_ref_dominant:
            j_norm = float(torch.norm(self.attractor.J, p='fro').item())
            collapse_occurred = self.purpose.flipped  # 元目的翻转视为坍缩信号
            coherence = self.purpose.coherence
            self.meta.update(activation.surprise, coherence, collapse_occurred, j_norm)
        elif self.meta is not None and is_self_ref_dominant:
            logger.debug(
                "Phase 2 学习隔离: 跳过 meta.update "
                "(自指主导轮次, surprise=%.4f)", activation.surprise)

        # 5. 记忆更新与巩固
        self.memory.update(activation, activation.surprise,
                           turn=self.turn_count)
        if self.turn_count > 0 and \
           self.turn_count % self.consolidation_interval == 0:
            self.memory.consolidate()
            logger.debug(f"第{self.turn_count}轮：执行记忆巩固")

        # M5 step 3/9 query —— 构造检索 query（规格步骤 3：与注入内容同源
        # 但独立构造——防伪独立 query 维；长时线索 = activation.state，
        # 情景语义 query 由 _retrieve_episodic 内部经 _encode_query_vector
        # 独立构造）。
        self._record_lifecycle(
            "query",
            long_term_cue="activation.state",
            episodic_query_source="text → _encode_query_vector",
        )

        # 5.5 长时记忆检索（S1 修复：用当前激活态作为线索，检索长时记忆）
        # recall() 返回 [num_nodes] 维向量，与 activation.state 同维。
        # 这是"记忆只写不读"缺陷的核心修复点：长时记忆通过此步进入输出路径。
        recalled = self.memory.recall(activation.state)

        # 5.6 情景记忆检索（用语义向量找最相关的历史文本）
        # 先检索后存储：避免当前轮文本出现在检索结果中
        episodic_texts = self._retrieve_episodic(text)

        # M5 step 4/9 retrieve 观测（只读检索；检索时怀疑投影入口——
        # _retrieve_episodic 内部写侧引用匹配 record_reference 属写侧时相
        # 的引用加固，非条目改写——四不变守卫口径不变）。
        self._record_lifecycle(
            "retrieve",
            long_term_recalled=bool(recalled is not None),
            episodic_hits=len(episodic_texts) if episodic_texts else 0,
        )

        # 5.7 体验层 D（设计 v1.1 §6.2）：惊讶度双角色-角色2 去稳定化
        # 高惊讶（z>2）→ 标记被当前输入违反的旧记忆为 labile
        # （已关注方向内的证伪，Schiller 2010 B2）；插在 store_episodic 前。
        # 角色 1（学习信号）不动（红线）。fail-open。
        self._destabilize_if_high_surprise(
            activation, semantic_vector, raw_semantic_vector, text)

        # 6. 解码context（传入检索到的记忆和情景文本，使长期记忆+语义文本参与输出）
        # A-P1-2: 传入 purpose.coherence，使解码器能输出"关注方向"解读
        memory_context = self.decoder.decode(
            activation, recalled_memory=recalled,
            episodic_texts=episodic_texts,
            coherence=self.purpose.coherence)

        # M5 step 5/9 integrate 观测（结果并入工作记忆；反流畅性：来源与
        # 内容分开评估——来源可信度进 π，高流畅≠真）。
        self._record_lifecycle(
            "integrate",
            context_len=len(memory_context),
            coherence=round(float(self.purpose.coherence), 6),
        )

        # ★ 插入点 B：自指观测
        # 观测自述：将 decoder 输出和当前激活态送入自指回路进行蒸馏缓存，
        # 供下一轮 generate_echo 使用。默认关闭时此块完全跳过。
        if self.self_ref is not None:
            self.self_ref.observe(memory_context, activation)

        # 保存当前 activation 供下轮 autocorr 计算（Phase 0 预留）
        self._prev_activation = activation

        # 6.5 情景记忆存储（当前轮文本存入缓冲区，供后续检索）
        # 优先存 384 维 raw 向量；无 raw 向量时退化为投影向量（向后兼容）
        # P0 污染处置（2026-08-17）：系统事件不是对话——[doubt 前缀事件
        # 已由 doubt_ingest 结构化摄入（gap_registry 登记 / 证伪标记），
        # 不再作为普通对话入库（对齐 doubt_ingest 模块 docstring 已声明
        # 未执行的语义，注释与实现的根因修复；防复发）。fail-open：
        # 判定异常不阻断主循环。
        try:
            from core.doubt.doubt_ingest import is_doubt_event
            _is_sys_event = is_doubt_event(text)
        except Exception:
            _is_sys_event = False
        # M5 step 7/9 commit —— 写侧提交前置：条目数基准（store_episodic
        # 可能被垃圾过滤跳过——用条目数变化判定注入时怀疑；基准外提，
        # 观测口径与旧判定逐字节一致）。
        _epi_before = self.memory.episodic_size()
        if semantic_vector is not None and not _is_sys_event:
            # M3-1 注入时怀疑（§2.1 写侧时相）：仅当本轮确实新增了条目
            # （store_episodic 可能被垃圾过滤跳过——用条目数变化判定）。
            self.memory.store_episodic(
                text, semantic_vector, activation.surprise, self.turn_count,
                raw_semantic_vector=raw_semantic_vector,
                source='external')  # Phase 2: 显式来源标记
            if self.memory.episodic_size() > _epi_before:
                try:
                    # 注入时怀疑判定：高 surprise（>factor×J_target）或
                    # rebuttal 命中 → 标 suspect + 登记验证链（写侧）。
                    # fail-open：怀疑逻辑异常绝不阻断主循环（G 模式以日志
                    # 可见，不以静默吞掉）。
                    entry = list(self.memory.iter_episodic())[-1]
                    from core.doubt.state_machine import (
                        DoubtPhase,
                        compute_rebuttal_hit,
                    )
                    rebuttal_hit = compute_rebuttal_hit(
                        entry, self.memory.iter_episodic())
                    self.doubt_state.injection_check(
                        entry,
                        surprise=activation.surprise,
                        j_target=getattr(
                            self.attractor, 'j_target_norm', None),
                        rebuttal_hit=rebuttal_hit,
                        verification_chain=self.verification_chain,
                    )
                    # R3（C1 接线）：注入 suspect 标记后登记再巩固候选
                    # （reconsolidation_queue 契约："入队只允许写侧时相——
                    # 注入 suspect 标记后登记"；写侧入队即落盘，跨重启不失）。
                    # fail-open：入队异常绝不阻断写侧提交（G 模式以日志可见）。
                    if (self.doubt_state.injection_suspect_marked
                            > _suspect_marked_before):
                        try:
                            self.reconsolidation_queue.enqueue(
                                entry, reason="injection_suspect",
                                score=float(activation.surprise),
                                phase=DoubtPhase.INJECTION.value)
                        except Exception as e:  # pylint: disable=broad-except
                            logger.warning(
                                "再巩固候选入队失败（fail-open）: %s", e)
                except Exception as e:  # pylint: disable=broad-except
                    logger.warning(
                        "M3-1 注入时怀疑检查失败（fail-open）: %s", e)

        # M5 step 7/9 commit 观测（写侧提交：store_episodic——store 提取层
        # 统一出口；写侧默认保守；[doubt] 系统事件不入库——P0 污染处置）。
        _epi_delta = self.memory.episodic_size() - _epi_before
        self._record_lifecycle(
            "commit",
            episodic_delta=_epi_delta,
            entry_added=bool(_epi_delta > 0),
            suspect_marked=self.doubt_state.injection_suspect_marked,
            sys_event_skipped=bool(_is_sys_event),
        )

        # M3-2 验证链全链写侧（§2.2）：验证结果 CONFLICT → [doubt] conflict
        # 事件 → 目标旧记忆 labile（写侧时相）。fail-open；开关默认关
        # （LMS_VERIFICATION_CHAIN_ENABLED=0）→ 零参与。
        self._apply_verification_conflicts()

        # 目的时相（总任务书 §二.5）：每轮 [doubt] purpose-drift 判定——
        # 与质疑系统同处 doubt_check 步（9 步生命周期不加步——"目的时相与
        # 质疑系统融合"）。消费本轮活动信号 → 闸门信号「是否偏离+理由」
        # （Pan 警示：只输出是否偏离+理由，不输出可优化分数）；verdict=
        # drifted 时登记 [doubt] purpose-drift（gap A 类登记，可审计）。
        # 开关关/异常 → None（fail-open，绝不阻断主循环）。
        _purpose_gate = self._purpose_drift_check(
            episodic_added=bool(_epi_delta > 0),
            doubt_events=(self.doubt_state.injection_checks
                          - _injection_checks_before
                          + (1 if _doubt_action is not None else 0)),
            suspect_marked=bool(self.doubt_state.injection_suspect_marked
                                > _suspect_marked_before),
        )

        # M5 step 6/9 doubt_check 观测（验证链判定：矛盾三选一/元数据排除/
        # 幂等——只登记，不改条目；应用由写侧时相驱动。目的时相判定与验证
        # 链同处本步——观测含 purpose 闸门信号（verdict/是否偏离/理由）。
        try:
            _chain_pending = len(self.verification_chain.pending_conflicts())
        except Exception:
            _chain_pending = 0
        _purpose_obs: dict = {}
        if _purpose_gate is not None:
            # 观测值须为标量（M5 生命周期观测契约：int/float/str/bool/None
            # ——原因列表 join 为字符串，可回放可审计）。
            _purpose_obs = {
                "purpose_verdict": _purpose_gate.get("verdict"),
                "purpose_drift_gate": bool(
                    _purpose_gate.get("purpose_drift")),
                "purpose_reasons": " | ".join(
                    _purpose_gate.get("reasons", []) or [])[:500],
                "purpose_text": str(_purpose_gate.get("purpose", "") or ""),
            }
        self._record_lifecycle(
            "doubt_check",
            chain_enabled=bool(self.verification_chain.enabled),
            chain_pending_conflicts=_chain_pending,
            injection_checks=self.doubt_state.injection_checks,
            injection_suspect_marked=self.doubt_state.injection_suspect_marked,
            **_purpose_obs,
        )

        # 阶段 3：precision 动态校准观测（HGF 波动性 → 全局怀疑基线；
        # 逐试次更新，Mathys 2011 / Behrens 2007）。is_negative 由本轮
        # 证伪事件（conflict/去稳定化）标记，对称性约束补偿坏消息 PE。
        # fail-open：观测异常绝不阻断主循环。
        if self.precision_adapt is not None:
            try:
                self.precision_adapt.observe_surprise(
                    activation.surprise,
                    is_negative=self._pending_negative_evidence)
            except Exception:
                pass
        self._pending_negative_evidence = False

        # R3（C1 接线）：巩固期受控改写——turn 生命周期巩固期（doubt_check
        # 后、state_update 前）调用 reconsolidation_queue.maybe_rewrite
        # （三闸门机器防线：G1 候选在队 / G2 巩固期时相 / G3 写侧委托——
        # 经 state_machine 巩固时相写侧入口 consolidation_resolve 受控
        # 改写；写侧默认保守）。检索路径零改写保持（只读四不变不破坏：
        # 本段只走写侧时相，/recall 与 /react 只读口不经过 process_turn）。
        # 结果进 state_update 观测 + 下一轮目的时相 reconsolidated 信号。
        try:
            _reconsolidated_count = self._reconsolidation_consolidate()
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("巩固期再巩固失败（fail-open）: %s", e)
            _reconsolidated_count = 0
        self._last_turn_reconsolidated = bool(_reconsolidated_count > 0)

        # M5 step 8/9 state_update 观测（状态更新：allostatic J 滑动设定点
        # + J/σ 更新（learn）+ π 更新（purpose.adjust）+ 记忆更新/巩固——
        # 锚点分散在 FEP 管线，观测在此聚合；增量 = 本轮期望增量，供
        # 回归断言零增量/期望增量——M4 规格 §1.3 emit 同源）。
        try:
            _j_norm_after = float(torch.norm(self.attractor.J, p='fro').item())
        except Exception:  # pylint: disable=broad-except
            _j_norm_after = _j_norm_before
        try:
            _sigma_norm_after = float(
                torch.norm(self.attractor.sigma).item())
        except Exception:  # pylint: disable=broad-except
            _sigma_norm_after = _sigma_norm_before
        try:
            _precision_mean_after = float(
                self.purpose.get_precision().mean())
        except Exception:  # pylint: disable=broad-except
            _precision_mean_after = _precision_mean_before
        _allostatic_events = 0
        try:
            _allostatic_ctl = getattr(self.attractor, 'allostatic', None)
            if _allostatic_ctl is not None:
                _allostatic_events = len(
                    list(getattr(_allostatic_ctl, 'events', []) or []))
        except Exception:  # pylint: disable=broad-except
            _allostatic_events = 0
        self._record_lifecycle(
            "state_update",
            j_norm=round(float(_j_norm_after), 6),
            j_norm_delta=round(
                float(_j_norm_after - _j_norm_before), 6),
            sigma_norm_delta=round(
                float(_sigma_norm_after - _sigma_norm_before), 6),
            precision_mean_delta=round(
                float(_precision_mean_after - _precision_mean_before), 6),
            consolidation_ran=bool(
                self.turn_count > 0
                and self.turn_count % self.consolidation_interval == 0),
            allostatic_events=_allostatic_events,
            reconsolidated=int(_reconsolidated_count),
        )

        # 更新状态
        self.last_activation = activation
        # M5 step 9/9 emit —— turn 计数**唯一增量点**（§1.2：process_turn
        # 唯一；全库唯一 `+= 1` 站点，见 _increment_turn）。
        self._increment_turn()

        # M7 双写回滚期（规格 §三.3.2-5 / M7-数据迁移 doc）：每轮登记旧/新
        # 存储增量到 dual-write journal（LMS_M7_DUAL_WRITE_ROUNDS>0 才参与；
        # 默认 0=关 → 零参与、零 IO、行为与开关引入前完全一致）。round_no
        # = 本轮 turn（emit 步已增量，≥1）；new_inc = 本轮 new 侧条目增量
        # （_epi_delta——store_episodic 可能被垃圾过滤跳过，条目数差分是
        # 唯一可靠口径，与目的时相同款）；old_inc = 部署侧旧存储本轮写入
        # 计数（双写期旧/新接收同一批写入；rewrite-ws 内无 live 旧存储 →
        # 以同批口径登记，切单写闸门由部署侧 check_rounds 把关）。fail-open：
        # 登记失败仅告警（该轮不可验证 = 闸门自然不放行，语义安全）。
        try:
            from runtime.m7_dual_write import dual_write_enabled, record_round
            if dual_write_enabled():
                record_round(
                    round_no=self.turn_count,
                    old_inc=_epi_delta,
                    new_inc=_epi_delta,
                )
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("M7 双写登记失败（fail-open）: %s", e)

        logger.debug(
            f"第{self.turn_count}轮: "
            f"熵={activation.entropy:.3f}, "
            f"惊讶度={activation.surprise:.3f}"
        )

        # 7. 自动快照（G4 修复：按间隔自动保存状态）—— 提取为 _auto_snapshot
        self._auto_snapshot()

        # ★ Phase 4 钩子：塑形状态反哺总线（外围、静默降级，绝不影响主循环）
        self._maybe_publish_plastified(activation)

        # M5 step 9/9 emit 观测（观测：turn 唯一增量 + 快照 + plastified 事件
        # + **状态汇总**——规格步骤 9：绝不可静默失败，G 模式以日志可见，
        # 不阻断返回）。
        try:
            _plastified_interval = int(os.environ.get(
                "LMS_PLASTIFIED_INTERVAL",
                str(self.config.get("lms_plastified_interval", 10))))
        except (TypeError, ValueError):
            _plastified_interval = 10
        self._record_lifecycle(
            "emit",
            turn_count=self.turn_count,
            auto_snapshot_enabled=bool(
                self.config.get('auto_snapshot', False)),
            auto_snapshot_interval=self.config.get(
                'auto_snapshot_interval', 50),
            plastified_interval=_plastified_interval,
        )
        # 状态汇总（9 步固化判定：broken=False 才合法；fail-open 不阻断返回）。
        try:
            self.last_turn_lifecycle = self.lifecycle_trace.summary(
                self.turn_count)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "M5 状态汇总失败（fail-open，不阻断返回）: %s", e)
        # 本轮记录结束：trace 清空——lifecycle_trace 仅在 process_turn 执行
        # 期间非 None；/recall 与 /react 只读口恒为 None（只读四不变）。
        self.lifecycle_trace = None

        # 8. 返回context
        return memory_context

    def _apply_verification_conflicts(self) -> None:
        """验证链 conflict 结果写侧应用（M3-2 §2.2 全链收尾）。

        验证结果 CONFLICT → 以 ``[doubt] conflict: <目标旧记忆文本>`` 事件
        走 doubt_ingest 结构化摄入（写侧：目标条目 mark_labile + gap 登记）
        ——与人工 [doubt] conflict 同一条证伪路径，不另造机制。幂等：
        应用后 ``chain.mark_conflict_applied`` 记账，同键不再重复应用。
        开关默认关（LMS_VERIFICATION_CHAIN_ENABLED=0）→ 零参与；fail-open
        （G 模式：异常以日志可见，不以静默吞掉）。
        """
        chain = getattr(self, "verification_chain", None)
        if chain is None or not chain.enabled:
            return
        try:
            from core.doubt.doubt_ingest import ingest as doubt_ingest
            for payload in chain.pending_conflicts():
                target = (payload.get("target_text") or "").strip()
                if not target:
                    chain.mark_conflict_applied(payload["request_key"])
                    continue
                doubt_ingest(self, f"[doubt] conflict: {target}")
                chain.mark_conflict_applied(payload["request_key"])
            # M6（规格 v2 §4.3）：验证链事件发布（verify_requested/result/
            # resolved + 计数——落沙必发事件，坑 7 根治；静默降级）
            self._maybe_publish_verification_events()
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("验证链 conflict 写侧应用失败（fail-open）: %s", e)

    def _purpose_drift_check(
        self,
        *,
        episodic_added: bool,
        doubt_events: int,
        suspect_marked: bool,
    ) -> Optional[dict]:
        """目的时相（总任务书 §二.5）：每轮 [doubt] purpose-drift 判定。

        消费本轮活动信号（写侧活动/怀疑事件/验证链参与/熵参与决策/生命周期
        记录）→ 目的五问（Q1-Q5，core/doubt/purpose_drift.py 判据）→ 闸门
        信号「是否偏离 + 理由」——**只输出是否偏离+理由，不输出任何可被优化
        的分数**（Pan 警示硬约束；无分数保证由 tests/test_purpose_drift.py
        递归断言）。

        - verdict == "drifted"（任一问判否）→ 登记 ``[doubt] purpose-drift``
          （gap_registry A 类登记——可审计；与人工 [doubt] event 同款登记面）；
        - verdict == "uncertain"（"不确定"=未判，不是通过）→ 闸门亮但**不
          登记**（只进生命周期观测 + /status purpose 块，暴露待补判信号——
          防 gap 泛滥）；
        - 开关关（LMS_PURPOSE_DRIFT_ENABLED=0）→ 中性闸门零参与；任何异常
          → fail-open 返回 None（G 模式以日志可见，绝不阻断主循环）。
        """
        try:
            chain = getattr(self, "verification_chain", None)
            chain_active = bool(
                chain is not None and chain.enabled
                and (suspect_marked or doubt_events > 0))
            # P2-B 目的检查口径修正（C3）：本轮是否只读轮（无任何写侧活动
            # ——如 /recall 查询）→ Q1 豁免（产出未流入活结构是只读语义，
            # 不是偏离，不判 drifted）。写侧活动口径与 Q1 同源：episodic
            # 写入 / suspect 标记 / 再巩固（上一轮巩固期结果——目的检查在
            # doubt_check 步、本轮再巩固在其后，用上一轮保持诚实）。
            _recon_last = bool(getattr(self, "_last_turn_reconsolidated", False))
            _write_activity = bool(episodic_added) or bool(suspect_marked) \
                or _recon_last
            round_signals = {
                "episodic_added": bool(episodic_added),
                "doubt_events": int(doubt_events),
                "verification_chain_active": chain_active,
                #: VERIFY-* provenance 等价于验证链本轮活跃（docstring 同源）
                "provenance": chain_active,
                "suspect_marked": bool(suspect_marked),
                # 本轮熵/惊讶真实参与决策：allostatic J 滑动设定点 + 注入时
                # 怀疑（surprise>factor）+ π 调整 + 元可塑性/在线熵管理——
                # 不是只作展示（诚实信号，非凑分）。
                "surprise_in_decisions": True,
                "lifecycle_trace": bool(self.lifecycle_trace is not None),
                "reconsolidated": _recon_last,
                "dream_consolidated": False,
                # P2-B（C3）：只读轮豁免 Q1（不判"产出未流入活结构"）
                "readonly_round": not _write_activity,
                "conclusion_only": False,
                "memory_idle": False,
                "surprise_display_only": False,
                "not_replayable": False,
            }
            gate = self.doubt_state.purpose_drift_check(
                round_signals,
                purpose_text=str(self.config.get("purpose_text", "") or ""),
            )
            # [doubt] purpose-drift 登记：仅真实偏离（drifted）——uncertain
            # 是"未判"（记观测待补判），不是偏离（不登记，防 gap 泛滥）。
            if gate.get("verdict") == "drifted":
                try:
                    self.gap_registry.register_fok_unresolved(
                        topic="purpose-drift: drifted",
                        detail=" | ".join(
                            gate.get("reasons", []) or [])[:300],
                    )
                except Exception as e:  # pylint: disable=broad-except
                    logger.warning(
                        "目的偏离 gap 登记失败（fail-open）: %s", e)
            return gate
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("目的时相判定失败（fail-open）: %s", e)
            return None

    def _reconsolidation_consolidate(self) -> int:
        """R3（C1 接线）：巩固期受控改写（三闸门 maybe_rewrite）。

        turn 生命周期巩固期（doubt_check 后、state_update 前——9 步固化
        不加步，再巩固观测并入 state_update 步）调用：遍历 episodic 缓冲，
        对在再巩固候选队列中的条目执行 ``maybe_rewrite``：

          - G1 候选在队（candidate_in_queue）：条目键在持久化队列中；
          - G2 巩固期触发（consolidation_triggered）：本段以巩固期写侧
            时相（DoubtPhase.CONSOLIDATION）调用——改写只由巩固调度方
            触发；
          - G3 改写受控（rewrite_controlled）：改写动作经 ``rewrite_fn``
            委托 state_machine 巩固时相写侧入口（consolidation_resolve
            ——证据裁决：conflict → superseded / confirm → kept / 超时 →
            downgraded；写侧默认保守）。任一不过 → 不改写（机器防线）。

        **检索路径零改写保持**：本段只走写侧时相（process_turn 内）；
        /recall、/react 只读口不经过本方法（只读四不变不破坏）。

        返回本轮实际改写（三闸门全过）的条目数；任何异常 fail-open
        （G 模式以日志可见，绝不阻断主循环）。
        """
        q = getattr(self, "reconsolidation_queue", None)
        if q is None or not q.enabled:
            return 0
        from core.doubt.state_machine import DoubtPhase
        # E3（根因 4 修复）：min-age 闸门——只消费入队 ≥N 轮的候选
        # （N = LMS_E3_MIN_AGE_TURNS，默认 1）。E3 总开关关 → 返回 None
        # （不闸，行为与 E3 引入前逐位一致：同轮即消保持）。
        eligible = self._e3_eligible_keys()
        rewritten = 0
        try:
            for entry in self.memory.iter_episodic():
                if not q.contains(entry):
                    continue
                if eligible is not None:
                    # 候选键不在可消费集（入队不足 N 轮）→ 本轮跳过，
                    # 至少活过当轮留给做梦 M6 扫 labile/suspect。
                    from core.doubt.reconsolidation_queue import (
                        entry_key as _queue_entry_key)
                    k = _queue_entry_key(entry)
                    if not k or k not in eligible:
                        continue
                res = q.maybe_rewrite(
                    entry,
                    phase=DoubtPhase.CONSOLIDATION.value,
                    rewrite_fn=self._reconsolidation_write_side,
                    detail="巩固期受控改写（loop 接线）",
                )
                if res.get("passed"):
                    rewritten += 1
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("再巩固候选处理失败（fail-open）: %s", e)
        return rewritten

    def _reconsolidation_write_side(self, entry) -> None:
        """R3（C1 接线）：G3 写侧委托——state_machine 巩固时相写侧入口。

        由 ``reconsolidation_queue.maybe_rewrite`` 在**三闸门全过**后调用
        （队列本体只做裁决与登记，不直接改条目）——本方法经
        ``doubt_state.consolidation_resolve``（写侧唯一转移入口之一）执行
        真正的状态转移：证伪证据（violated_by）→ superseded；窗口内无
        证据 → kept（stable，confidence 重巩固）；窗口超时 → downgraded。
        与 M6 梦期巩固时相同语义（写侧默认保守）。任何异常 fail-open
        （maybe_rewrite 已接住，队列不动）。
        """
        self.doubt_state.consolidation_resolve(entry)

    # ================================================================== #
    #  E3（自我怀疑驱动的主动调节，dandan 拍板 2026-08-20 22:14）：
    #  选择 → 重激活 → satiety 消解；min-age 闸门数据源；/status e3 观测。
    #  总开关 LMS_E3_ENABLED 默认 0=关 → 全部新路径零参与（逐位等价）。
    #  全部走既有刹车（sleep_check 由触发侧 self_pulse 把关；本方法只做
    #  记忆内容调节，不调任何阈值参数——不动 B 级修复成果）。
    # ================================================================== #

    def _e3_record_turn_start_keys(self) -> None:
        """E3：记录本 turn 起始的再巩固候选键快照（min-age 闸门数据源）。

        滚动保留最近 N 份（N = LMS_E3_MIN_AGE_TURNS）turn 起始快照；
        入队于第 T' 轮的候选自第 T'+1 轮起始出现在快照。总开关关 →
        零参与。fail-open：任何异常以日志可见，绝不阻断主循环。
        """
        n = _e3_min_age_turns()
        if n <= 0:
            return
        q = getattr(self, "reconsolidation_queue", None)
        if q is None or not q.enabled:
            return
        try:
            keys = frozenset(
                r.get("entry_key") for r in q.peek(max_items=4096)
                if r.get("entry_key"))
            snaps = self._e3_turn_start_keys
            if snaps is None or snaps.maxlen != max(1, n):
                snaps = deque(maxlen=max(1, n))
                self._e3_turn_start_keys = snaps
            snaps.append(keys)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("E3 turn 起始键快照记录失败（fail-open）: %s", e)

    def _e3_eligible_keys(self):
        """E3 min-age 闸门的可消费键集（None = 开关关，不闸）。

        取滚动快照中**最旧**一份 s_{T-N+1}（第 T 轮消费时年龄 ≥N 轮 ⟺
        键 ∈ s_{T-N+1}）；运行不足 N 轮 → 空集（无候选年龄达标，保守）；
        无快照（防御）→ None（fail-open 不拦，回到无闸门行为）。
        """
        n = _e3_min_age_turns()
        if n <= 0:
            return None
        snaps = getattr(self, "_e3_turn_start_keys", None)
        if not snaps:
            return None
        if len(snaps) < n:
            return frozenset()
        return snaps[0]

    def _e3_build_clue(self, cand: dict) -> str:
        """生成重激活线索（无 LLM——dandan 拍板 3：免费版自动生成）。

        复用 dream_engine._build_incomplete_clue 思路（前缀截断 + 省略号 +
        质疑问句——B3 绝不复述全文，完全重复反而稳定旧记忆），叠加新上下文/
        新证据（gap 登记的 detail/证据摘要承载——预测误差源，不是温和"重新
        审视"）。长度上限 LMS_E3_REACTIVATE_TOPIC_LEN（默认 120）。
        纯函数，fail-open（异常回退通用疑问句）。
        """
        try:
            target = (cand.get("target_text") or cand.get("topic") or "").strip()
            if not target:
                return "这条记忆还成立吗？"
            cut = max(8, min(int(len(target) * 0.6), 40))
            fragment = target[:cut].strip()
            if len(fragment) < len(target):
                fragment += "……"
            clue = f"{fragment}？这条还成立吗？"
            detail = (cand.get("detail") or "").strip()
            if detail:
                clue += f" 新证据/疑点：{detail[:60]}"
            return clue[:max(40, _e3_topic_len())]
        except Exception:  # pylint: disable=broad-except
            return "这条记忆还成立吗？"

    def _e3_find_target_by_key(self, entry_key: str):
        """按稳定键找回目标条目（选择器只回键/文本，不回条目对象）。"""
        if not entry_key:
            return None
        from core.doubt.reconsolidation_queue import entry_key as _queue_entry_key
        for entry in self.memory.iter_episodic():
            try:
                if _queue_entry_key(entry) == entry_key:
                    return entry
            except Exception:  # pylint: disable=broad-except
                continue
        return None

    def _e3_reactivate_path_b(self, clue: str) -> bool:
        """路径 B 兜底：线索文本走普通 store → FEP surprise → 既有去稳定化。

        走 ``process_turn``（与 /store 同路径）：新条目写入 → FEP 预测误差
        → ``_destabilize_if_high_surprise``（既有 L1514 起，z>2 → mark_labile
        + enqueue）。成功后候选队列尺寸增大（含注入 suspect 入队）→ True。
        fail-open：任何异常返回 False（兜底失败不阻断主流程）。
        """
        try:
            q = getattr(self, "reconsolidation_queue", None)
            size_before = q.size() if q is not None else 0
            self.process_turn(clue)
            size_after = q.size() if q is not None else 0
            return bool(size_after > size_before)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("E3 路径 B 重激活失败（fail-open）: %s", e)
            return False

    def e3_reactivate(self, *, dry_run: bool = False,
                      limit: Optional[int] = None) -> dict:
        """E3 重激活动作（选择 → 重激活 → satiety 记账 → 返回结果 dict）。

        总开关 LMS_E3_ENABLED=0 → {"enabled": False} 零参与（A6 逐位等价）。

        流程（设计 §3.2 ②，dandan 拍板 3：无 LLM）：
          0. satiety 消解扫（上轮重激活目标：superseded/kept≥N → 判 resolved
             → gap_registry.mark_resolved + satiety 记账——A5 落点）；
          1. satiety 武装检查（待机中 → 直接返回待机态）；
          2. 选择器选悬案（fok_unresolved ∪ low_confidence ∪ suspect 未决；
             score = α·epistemic + β·progress + γ·reachable，纯函数无 LLM）；
          3. 对 top-N（N = limit 或 LMS_E3_REACTIVATE_MAX，默认 2）重激活：
             路径 A（主）：feed "[doubt] conflict: <线索+新证据摘要>" →
               doubt_ingest（target_entry 显式传入——选择器已定位目标）→
               mark_labile(violated_by=线索) → 补 enqueue（doubt_ingest 已接）
               → 候选带证据进队列；
             路径 B（兜底）：线索走普通 store → FEP surprise → 既有去稳定化；
          4. satiety 记账：record_reactivation(topic)（冷却内不重选同一悬案）
             + close_cycle（累计 3 次闭环 → 待机 24h）；
          5. 观测：surprise_before（A2 观测字段，不设阈值判定——dandan 拍板
             4：到验收时由 dandan 判断）。

        任何异常 fail-open（G 模式以日志可见，绝不抛给调用方）。
        """
        if not _e3_enabled():
            return {"enabled": False,
                    "note": "LMS_E3_ENABLED=0（总开关关，零参与）"}
        from core.doubt.doubt_ingest import ingest as doubt_ingest
        from core.doubt.epistemic_selector import select_cases
        from core.doubt.satiety import judge_resolved
        gate = getattr(self, "satiety_gate", None)
        result = {
            "enabled": True,
            "armed": False,
            "session_id": self.session_id,
            "surprise_before": self._e3_last_surprise(),
            "selected": 0, "activated": 0, "rejected": 0,
            "candidates": [], "activated_items": [], "rejected_items": [],
            "satiety": gate.snapshot() if gate is not None else None,
            "note": "",
        }
        try:
            # 0. satiety 消解扫（上轮重激活目标的闭环收口）
            if gate is not None:
                self._e3_satiety_sweep(judge_resolved=judge_resolved)
            # 1. 武装检查
            if gate is None or not gate.is_armed():
                result.update({
                    "armed": False,
                    "note": "satiety 待机中（累计闭环达上限；新 [doubt] 事件或"
                            "新 plateau 触发后 rearm 恢复）",
                    "satiety": gate.snapshot() if gate is not None else None,
                })
                return result
            result["armed"] = True
            # 2. 选料（纯函数）
            snap = {}
            try:
                snap = self.gap_registry.snapshot()
            except Exception:  # pylint: disable=broad-except
                pass
            fok_list = list(snap.get("fok_unresolved", []) or [])
            lowconf_list = list(
                snap.get("low_confidence_unreviewed", []) or [])
            episodic = list(self.memory.iter_episodic())
            # 补充候选源：suspect 未决条目（fok 形态混入，选择器只消费列表）
            try:
                for e in episodic:
                    if str(getattr(e, "doubt_state", "stable") or "stable") \
                            == "suspect":
                        fok_list.append({
                            "topic": (getattr(e, "text", "") or "")[:120],
                            "detail": "suspect 未决条目（doubt_state=suspect）",
                        })
            except Exception:  # pylint: disable=broad-except
                pass
            excluded = set()
            try:
                excluded.update(
                    r.get("topic") for r in self.gap_registry.resolved_list())
            except Exception:  # pylint: disable=broad-except
                pass
            if gate is not None:
                try:
                    excluded.update(gate.cooldown_topics())
                except Exception:  # pylint: disable=broad-except
                    pass
            max_items = max(1, min(
                int(limit if limit is not None else _e3_reactivate_max()), 8))
            candidates = select_cases(
                fok_list, lowconf_list, episodic=episodic,
                surprise_window=list(
                    getattr(self, "react_surprise_history", None) or []),
                excluded_topics=excluded, max_candidates=max_items * 2)
            selected = candidates[:max_items]
            result["selected"] = len(selected)
            result["candidates"] = [
                {k: c.get(k) for k in
                 ("kind", "topic", "score", "epistemic", "progress",
                  "reachable", "target_found", "entry_key")
                 if k in c}
                for c in selected]
            if dry_run:
                result["dry_run"] = True
                result["note"] = "dry_run：只选择不激活（A1 观测）"
                return result
            # 3. 重激活（路径 A 主 / 路径 B 兜底）
            pending = []
            for cand in selected:
                clue = self._e3_build_clue(cand)
                brief = {
                    "kind": cand.get("kind"),
                    "topic": cand.get("topic"),
                    "clue": clue,
                }
                # 路径 A：显式目标条目 → doubt_ingest conflict → labile+入队
                target = None
                if cand.get("entry_key"):
                    target = self._e3_find_target_by_key(cand.get("entry_key"))
                ev = None
                try:
                    ev = doubt_ingest(
                        self, f"[doubt] conflict: {clue}",
                        target_entry=target)
                except Exception:  # pylint: disable=broad-except
                    ev = None
                if ev is not None and ev.get("action") == "rebutted":
                    result["activated_items"].append({**brief, "path": "A"})
                    result["activated"] += 1
                    pending.append(cand)
                    if gate is not None:
                        gate.record_reactivation(cand.get("topic"))
                    continue
                # 路径 B（兜底）：普通 store → FEP surprise → 去稳定化
                if self._e3_reactivate_path_b(clue):
                    result["activated_items"].append({**brief, "path": "B"})
                    result["activated"] += 1
                    pending.append(cand)
                    if gate is not None:
                        gate.record_reactivation(cand.get("topic"))
                    continue
                result["rejected_items"].append({
                    **brief, "reason": "路径 A 未命中 + 路径 B 无去稳定化"})
                result["rejected"] += 1
            # 4. satiety 闭环记账（下次触发做消解扫；累计 3 次 → 待机）
            if result["activated"] > 0:
                self._e3_pending_targets = [
                    {"topic": c.get("topic"), "entry_key": c.get("entry_key")}
                    for c in pending]
                self._e3_closed_loops += 1
                self._e3_last_reactivation = {
                    "ts": time.time(),
                    "closed_loop": self._e3_closed_loops,
                    "activated": result["activated"],
                    "topics": [c.get("topic") for c in pending],
                }
                if gate is not None:
                    try:
                        gate.close_cycle(
                            outcome="activated",
                            detail=", ".join(
                                str(c.get("topic")) for c in pending)[:300])
                    except Exception:  # pylint: disable=broad-except
                        pass
            result["satiety"] = gate.snapshot() if gate is not None else None
            return result
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("E3 重激活失败（fail-open）: %s", e)
            result["note"] = f"e3_reactivate fail-open: {e}"
            return result

    def _e3_satiety_sweep(self, *, judge_resolved) -> None:
        """satiety 消解扫：上轮重激活目标的闭环收口（A5 落点）。

        对 ``_e3_pending_targets`` 逐条：目标条目 superseded（重巩固改写
        落库）或复核报告 kept ≥ kept_threshold（N 轮无新证据）→ 判
        resolved → ``gap_registry.mark_resolved`` + satiety 记账（冷却）。
        尽力而为不无限怀疑（Szechtman & Woody 2004 终止信号）。fail-open。
        """
        pending = list(getattr(self, "_e3_pending_targets", []) or [])
        if not pending:
            return
        try:
            stats = self.gap_registry.review_stats()
        except Exception:  # pylint: disable=broad-except
            stats = {}
        still_pending = []
        for item in pending:
            topic = item.get("topic")
            verdict = "pending"
            try:
                entry = self._e3_find_target_by_key(item.get("entry_key"))
                verdict = judge_resolved(entry, outcomes=stats)
                if verdict == "resolved":
                    try:
                        self.gap_registry.mark_resolved(
                            topic, detail="E3 satiety 消解判定")
                    except Exception:  # pylint: disable=broad-except
                        pass
                    gate = getattr(self, "satiety_gate", None)
                    if gate is not None:
                        try:
                            gate.record_resolved(topic)
                            gate.close_cycle(outcome="resolved", detail=str(topic))
                        except Exception:  # pylint: disable=broad-except
                            pass
                    continue
            except Exception as e:  # pylint: disable=broad-except
                logger.warning("E3 satiety 消解扫条目失败（fail-open）: %s", e)
            if verdict != "resolved":
                still_pending.append(item)
        self._e3_pending_targets = still_pending

    def _e3_last_surprise(self) -> Optional[float]:
        """最近一次激活惊讶度（A2 观测字段——不设阈值判定，dandan 拍板 4）。"""
        try:
            if self.last_activation is not None:
                return round(float(self.last_activation.surprise), 4)
        except Exception:  # pylint: disable=broad-except
            pass
        return None

    def e3_observation(self) -> dict:
        """E3 观测块（/status 数据源：闭环计数/待机态/最近悬案/冷却倒计时）。

        总开关关 → {"enabled": False}（A6 回归可观测）；fail-open：任何
        异常只缺省该字段（纯增量字段，旧客户端忽略）。
        """
        if not _e3_enabled():
            return {"enabled": False}
        obs: dict = {
            "enabled": True,
            "satiety": None,
            "closed_loops": int(getattr(self, "_e3_closed_loops", 0) or 0),
            "last_reactivation": getattr(self, "_e3_last_reactivation", None),
            "pending_targets": len(
                getattr(self, "_e3_pending_targets", []) or []),
            "last_surprise": self._e3_last_surprise(),
            "fok_resolved_count": 0,
            "queue_size": 0,
        }
        try:
            gate = getattr(self, "satiety_gate", None)
            if gate is not None:
                obs["satiety"] = gate.snapshot()
        except Exception:  # pylint: disable=broad-except
            pass
        try:
            obs["fok_resolved_count"] = self.gap_registry.resolved_count()
        except Exception:  # pylint: disable=broad-except
            pass
        try:
            q = getattr(self, "reconsolidation_queue", None)
            obs["queue_size"] = q.size() if q is not None else 0
        except Exception:  # pylint: disable=broad-except
            pass
        return obs

    # ================================================================== #
    #  私有辅助方法（由 process_turn / query_llm 拆分而来，保持行为不变）
    # ================================================================== #

    def _inject_self_ref(
        self, sensory_input: SensoryInput
    ) -> tuple[SensoryInput, float, bool]:
        """自指回注（插入点 A）。

        在编码后、推断前，将上一轮蒸馏的自述以自适应权重回注到感官向量。
        默认关闭（self_ref is None）时此块完全跳过，sensory_input 保持原样。

        参数:
            sensory_input: 原始感官输入
        返回:
            (修改后的 sensory_input, alpha_t, is_self_ref_dominant)
        """
        # ★ 插入点 A：自指回注
        alpha_t = 0.0
        echo = None
        if self.self_ref is not None:
            echo = self.self_ref.generate_echo(
                entropy_ratio=self.last_entropy_ratio if hasattr(self, 'last_entropy_ratio') else 0.5,
                ext_sensory=sensory_input.vector,
                activation_prev=self._prev_activation,
            )
            if echo is not None:
                alpha_t = echo['alpha']
                mixed_vector = sensory_input.vector + alpha_t * echo['vector']
                # ★ 自指回注（Phase 1 增强）：用 .get() 安全访问新增字段，
                # 即使 self_referential.py 的增强尚未完成也不崩溃。
                sensory_input = SensoryInput(
                    vector=mixed_vector,
                    metadata={
                        **sensory_input.metadata,
                        'self_ref_alpha': alpha_t,
                        'self_ref_state': echo.get('state', 'normal'),
                        'self_ref_autocorr': echo.get('autocorr'),
                    },
                )

        # ★ Phase 2: 学习隔离——检测自指主导轮次
        # 自指主导 = 自指权重较高且外部输入新颖度极低（系统在"自说自话"）
        # 此类轮次使用减半学习率，且跳过 meta.update，防止自指 surprise
        # 污染元参数趋势（审视报告 9.5）
        _self_ref_ext_novelty = None
        if self.self_ref is not None and echo is not None:
            _self_ref_ext_novelty = echo.get('ext_novelty')
        is_self_ref_dominant = (
            alpha_t > 0.05
            and _self_ref_ext_novelty is not None
            and _self_ref_ext_novelty < 0.1
        )
        if self.self_ref is not None:
            self.self_ref.last_is_self_ref_dominant = is_self_ref_dominant

        # ★ Phase 4 钩子：self_ref 蒸馏摘要发布（默认关闭，受开关+限频控制）
        # 看护人式：只观察+有限回应，不内部改写自指回路本体；
        # 发布内容为"蒸馏后的可发布摘要"（≤200字）+ 护栏状态，软参考信号。
        if self.self_ref is not None and echo is not None:
            self._maybe_publish_self_ref(echo)

        return sensory_input, alpha_t, is_self_ref_dominant

    def _get_meta_adjusted_params(
        self
    ) -> tuple[float, float | None, float | None, float | None]:
        """元可塑性：计算调整后的参数（E-P2-5: 不修改实例属性）。

        返回:
            (effective_lr, meta_temp, meta_orth, meta_cw)
            其中 meta_* 为 None 表示未启用元可塑性（无调整）。
        """
        # --- 元可塑性：计算调整后的参数（E-P2-5: 不修改实例属性）---
        effective_lr = self.config.get('learning_rate', 0.01)
        meta_temp = None
        meta_orth = None
        meta_cw = None
        if self.meta is not None:
            adjusted = self.meta.get_adjusted_params(
                base_lr=effective_lr,
                base_orth=self.attractor.orth_weight,
                base_temp=self.attractor.temperature,
                base_cw=self.attractor.complexity_weight,
            )
            meta_temp = adjusted['temperature']
            meta_orth = adjusted['orth_weight']
            meta_cw = adjusted['complexity_weight']
            effective_lr = adjusted['learning_rate']
        return effective_lr, meta_temp, meta_orth, meta_cw

    def _manage_entropy(
        self, activation, sensory_input, effective_lr
    ) -> float:
        """在线熵管理（3.5）。

        FEP学习之后、调整目的之前，根据当前激活熵主动干预：
        熵过高（混沌饱和）则增强正交化压力驱散相似表示；
        熵过低（僵化）则放松正交化压力允许更多激活。

        参数:
            activation: 当前激活态
            sensory_input: 感官输入
            effective_lr: 有效学习率
        返回:
            entropy_ratio（熵 / 最大熵）
        """
        # 3.5 在线熵管理
        entropy = activation.entropy
        max_entropy = math.log(self.attractor.num_nodes)  # ln(256) ≈ 5.55
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0.0
        self.last_entropy_ratio = entropy_ratio  # 供 get_status() 读取

        # 从config读取阈值（带默认值）
        entropy_high_threshold = self.config.get('entropy_high_threshold', 0.9)
        entropy_low_threshold = self.config.get('entropy_low_threshold', 0.5)

        if entropy_ratio > entropy_high_threshold:
            # 熵过高：系统混沌，增强正交化压力驱散相似表示
            # E-P2-5: 通过 orth_weight_override 传递临时权重，不修改实例属性
            self.attractor.learn(
                activation, sensory_input.vector, effective_lr * 0.3,
                orth_weight_override=self.attractor.orth_weight * 1.5,
            )
            logger.info(f"在线熵管理: 熵={entropy:.3f}(ratio={entropy_ratio:.2f}) 过高，增强正交化")
        elif entropy_ratio < entropy_low_threshold:
            # 熵过低：系统僵化，降低正交化压力允许更多激活
            # E-P2-5: 通过 orth_weight_override 传递临时权重，不修改实例属性
            self.attractor.learn(
                activation, sensory_input.vector, effective_lr * 0.2,
                orth_weight_override=self.attractor.orth_weight * 0.7,
            )
            logger.info(f"在线熵管理: 熵={entropy:.3f}(ratio={entropy_ratio:.2f}) 过低，放松正交化")

        return entropy_ratio

    def _maybe_publish_plastified(self, activation) -> None:
        """Phase 4 钩子：按间隔发布 lms.plastified（只发数值摘要，不发原始激活态）。

        用实例计数（turn_count % interval）控制，不加线程；
        间隔默认 10 轮，可用环境变量 LMS_PLASTIFIED_INTERVAL 覆盖。
        任何异常静默降级，绝不影响主循环（熔断由 bus_events 内部管理）。
        """
        try:
            interval = int(os.environ.get(
                "LMS_PLASTIFIED_INTERVAL",
                str(self.config.get("lms_plastified_interval", 10))))
        except (TypeError, ValueError):
            interval = 10
        if interval <= 0 or self.turn_count % interval != 0:
            return
        try:
            from runtime.bus_events import publish_plastified
            precision = self.purpose.get_purpose().precision
            active_nodes = 0
            if hasattr(activation, "state") and activation.state is not None:
                try:
                    active_nodes = int((activation.state > 0.0).sum().item())
                except Exception:
                    active_nodes = 0
            state = {
                "turn_count": self.turn_count,
                "entropy": float(getattr(activation, "entropy", 0.0) or 0.0),
                "surprise": float(getattr(activation, "surprise", 0.0) or 0.0),
                "entropy_ratio": float(
                    getattr(self, "last_entropy_ratio", 0.0) or 0.0),
                "active_nodes": active_nodes,
                "precision_mean": float(precision.mean()),
                "precision_std": float(precision.std()),
                "coherence": float(self.purpose.coherence),
            }
            publish_plastified(state)
        except Exception as e:
            # 外围钩子最后一道防线：绝不影响主循环
            logger.debug("Phase 4 plastified 发布跳过（静默降级）: %s", e)

    def _maybe_publish_self_ref(self, echo: dict) -> None:
        """Phase 4 钩子：self_ref 蒸馏摘要发布（最敏感，默认关闭）。

        只发"蒸馏后的可发布摘要"：取自自指回路已蒸馏的 self_voice 文本
        （≤200 字，由 bus_events 裁剪）+ 护栏状态数值摘要。
        开关 LMS_SELF_REF_PUBLISH 默认 off；开启后限频 ≥30 分钟一条。
        任何异常静默降级，绝不影响自指回路本体。
        """
        try:
            from runtime.bus_events import publish_self_ref
            summary = ""
            history = getattr(self.self_ref, "self_voice_history", None)
            if history:
                summary = history[-1]
            guard = {
                "state": echo.get("state", "normal"),
                "alpha": echo.get("alpha", 0.0),
                "autocorr": echo.get("autocorr"),
                "ext_novelty": echo.get("ext_novelty"),
                "echo_similarity": echo.get("echo_similarity"),
                "coherence": float(self.purpose.coherence),
                "entropy_ratio": float(
                    getattr(self, "last_entropy_ratio", 0.0) or 0.0),
            }
            publish_self_ref(summary, guard)
        except Exception as e:
            logger.debug("Phase 4 self_ref 发布跳过（静默降级）: %s", e)

    def _maybe_publish_dream_complete(self, result: dict, duration: float) -> None:
        """Phase 4 钩子：发布 lms.dream_complete（步数/耗时/结果/梦质量指标，可观测性信号）。

        梦醒回路阶段1-A（断点 A① 修复）：把 dream_mvp/dream_cycle 已返回的梦质量指标
        （avg_surprise/max_surprise/avg_entropy/collapse_count/j_change/buffer_size）
        透传进 payload——此前这些信号从未上总线；status/mode 用于区分
        'dreamed'（梦了）与 'no_memories_to_replay'（空缓冲没梦）。
        纯透传、零算法改动；缺失字段留 null（如 no_memories 分支无 j_change）。
        任何异常静默降级，绝不影响做梦结果返回（熔断由 bus_events 内部管理）。
        """
        try:
            from runtime.bus_events import publish_dream_complete
            publish_dream_complete({
                "status": result.get("status") or "dreamed",
                "mode": result.get("mode") or "mvp",
                "steps": int(result.get("steps", 0) or 0),
                "duration_seconds": round(float(duration), 3),
                "snapshot_saved": bool(result.get("snapshot_saved", False)),
                "avg_surprise": result.get("avg_surprise"),
                "max_surprise": result.get("max_surprise"),
                "avg_entropy": result.get("avg_entropy"),
                "collapse_count": result.get("collapse_count"),
                "j_change": result.get("j_change"),
                "buffer_size": result.get("buffer_size"),
                # 体验层 D（设计 v1.1 §6.3）：怀疑复核报告透传
                # （reviewed/downgraded/rewritten/kept/flagged，[梦醒] 数据源）
                "doubt_review": result.get("doubt_review"),
            })
        except Exception as e:
            logger.debug("Phase 4 dream_complete 发布跳过（静默降级）: %s", e)

    def _maybe_publish_doubt_consolidation(self, result: dict) -> None:
        """M6（规格 v2 §2.4/§4.3）钩子：发布 lms.doubt_consolidation。

        梦期巩固时相（resolve_labile confirm/supersedes 两分支）结果事件：
        复核报告（reviewed/rewritten/kept/downgraded）+ supersedes 记录
        明细 + 验证链快照——§4.3 落沙必发事件（坑 7 根治：梦期完成/状态
        更新一律发事件；此前落沙路径不发布事件 → 事件流断流 5 天）。
        只发数值摘要与少量改写记录；任何异常静默降级，绝不影响做梦结果
        返回（熔断由 bus_events 内部管理，断流以日志 + 熔断状态告警）。
        """
        try:
            from runtime.bus_events import publish_doubt_consolidation
            publish_doubt_consolidation({
                "status": result.get("status") or "dreamed",
                "mode": result.get("mode") or "mvp",
                "steps": int(result.get("steps", 0) or 0),
                "doubt_review": result.get("doubt_review"),
                "consolidation": result.get("consolidation"),
            })
        except Exception as e:
            logger.debug(
                "M6 doubt_consolidation 发布跳过（静默降级）: %s", e)

    def _maybe_publish_verification_events(self) -> None:
        """M6（规格 v2 §4.3）钩子：发布 lms.verification（验证链事件）。

        验证链事件（verify_requested / verify_result / verify_resolved）
        一律发事件（坑 7 根治）+ 待应用冲突/已登记计数。开关默认关
        （LMS_VERIFICATION_CHAIN_ENABLED=0）→ 链零参与，本钩子直接返回；
        任何异常静默降级，绝不影响写侧主流程（熔断由 bus_events 管理）。
        """
        try:
            chain = getattr(self, "verification_chain", None)
            if chain is None or not chain.enabled:
                return
            from runtime.bus_events import publish_verification_events
            publish_verification_events({
                "events": chain.events(last_n=10),
                "snapshot": chain.snapshot(),
            })
        except Exception as e:
            logger.debug("M6 verification 事件发布跳过（静默降级）: %s", e)

    def _encode_query_vector(
        self, text: str
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """编码查询文本为语义向量（检索共用，T1.3/P0-9 提取）。

        与 process_turn 内的检索编码完全一致：优先取 384 维原始向量
        （embed_text_raw，高精度），退化为投影向量（embed_text）。

        返回:
            (raw_vec, sem_vec)：embedder 不支持语义编码时返回 (None, None)。
        """
        if not hasattr(self.embedder, 'embed_text'):
            return None, None

        # T2.8/P2-6：embed 调用走熔断器（开关 LMS_EMBED_CIRCUIT=1 时生效）。
        # 熔断 OPEN 期间快速失败（不触网）→ 调用方（/recall）返回空结果；
        # 关闭状态行为与原来完全一致（embed 异常照常上抛）。
        from core.sensory.circuit_breaker import CircuitOpenError
        try:
            sem_vec = self._embed_circuit.call(
                self.embedder.embed_text, text)
        except CircuitOpenError:
            logger.warning(
                "embed 熔断中：快速失败，跳过语义编码（/recall 将返回空）")
            return None, None
        raw_vec = None
        if hasattr(self.embedder, 'embed_text_raw'):
            try:
                raw_vec = self._embed_circuit.call(
                    self.embedder.embed_text_raw, text)
            except CircuitOpenError:
                logger.warning(
                    "embed 熔断中：跳过 raw 编码（退化为投影向量）")
                return None, sem_vec
        return raw_vec, sem_vec

    def _destabilize_if_high_surprise(
        self, activation, semantic_vector, raw_semantic_vector, text
    ) -> None:
        """体验层 D（设计 v1.1 §6.2）：高惊讶去稳定化（角色 2）。

        近 200 轮 surprise 窗口，surprise > mean + 2·std → 定位本轮检索
        命中条目中**置信度最高**者（最 established 的旧记忆，工程近似：
        语义矛盾检测无 LLM 依赖）→ mark_labile(violated_by=当前输入)。

        专注化强调：被标记的是"被当前输入违反的旧记忆"（已关注方向里的
        证伪内容），不是"值得探索的新方向"。任何异常静默（fail-open）。
        """
        try:
            self.destab_surprise_window.append(activation.surprise)
            window = list(self.destab_surprise_window)
            if len(window) < 20:
                return
            mean = sum(window) / len(window)
            std = (sum((v - mean) ** 2 for v in window) / len(window)) ** 0.5
            if std < 1e-8:
                return
            z = (activation.surprise - mean) / std
            if z <= 2.0:
                return
            query_vec = (raw_semantic_vector if raw_semantic_vector is not None
                         else semantic_vector)
            if query_vec is None:
                return
            from core.doubt.reconsolidation import (
                find_violated_entry, mark_labile)
            scored = self.memory.recall_episodic_scored(
                query_vec, top_k=3, fallback_query=semantic_vector,
                source_filter='external', count_reference=False)
            if not scored:
                return
            entry = find_violated_entry(scored)
            if entry is not None:
                snippet = (text or '')[:80]
                if mark_labile(entry, violated_by=snippet):
                    logger.info(
                        f"体验层D: 高惊讶 z={z:.2f} → 去稳定化旧记忆 "
                        f"(turn={getattr(entry, 'turn', '?')}, "
                        f"rebuttal={getattr(entry, 'rebuttal_count', 0)})")
                    # R3（C1 接线）：去稳定化（labile 标记）条目登记再巩固
                    # 候选（写侧时相——mark_labile 是写侧动作；队列契约
                    # "入队只允许写侧时相"；巩固期 maybe_rewrite 消化）。
                    # fail-open：入队异常绝不影响去稳定化结果。
                    try:
                        from core.doubt.state_machine import DoubtPhase
                        self.reconsolidation_queue.enqueue(
                            entry, reason="destabilized_labile",
                            score=float(activation.surprise),
                            phase=DoubtPhase.INJECTION.value)
                    except Exception:  # pylint: disable=broad-except
                        pass
                    # 阶段 3：证伪 → conformal 校准集 + 负性证据标记
                    # （对称性约束：坏消息 PE 不被系统性低估）
                    if self.precision_adapt is not None:
                        try:
                            self._pending_negative_evidence = True
                            self.precision_adapt.record_rebuttal(entry)
                        except Exception:
                            pass
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("去稳定化标记失败（fail-open）: %s", e)

    # ================================================================== #
    #  阶段 3（precision 三层动态化）：召回/观测钩子
    # ================================================================== #

    def _project_consistency_readonly(self, scored) -> dict:
        """只读一致性投影（M2，替换旧 _attach_consistency）。

        旧 ``_attach_consistency`` 在 recall 内改写 ``entry.consistency``
        并随快照持久化（``memory.get_state()`` 直接返回条目对象）= **只读
        泄漏（P1，未修）**——M2 已整体删除该方法。

        本方法只计算 + 更新**进程内观测缓存**（consistency_cache / 召回簇
        统计），**绝不 setattr 条目**；一致性值通过返回的 ``cons_map``
        （{id(entry): cons}）供调用方做响应注解与怀疑投影，绝不落库
        （进程内观测同 react_surprise_history 先例：重启即失、快照不落盘）。

        开关关/异常 → 返回 {}（调用方 fail-open 降级回旧行为）。
        """
        if self.precision_adapt is None or not scored:
            return {}
        try:
            from core.doubt.precision_adapt import compute_consistency
            cons = compute_consistency(scored)
            # 进程内观测缓存（供 /recall 注解与 doubt 观测块；零持久化）
            self.precision_adapt.consistency_cache.update(cons)
            try:
                self.precision_adapt.record_recall_cohort(scored)
            except Exception:  # pylint: disable=broad-except
                pass
            return cons
        except Exception:  # pylint: disable=broad-except
            return {}

    def _readonly_state_snapshot(self) -> dict:
        """只读四不变状态快照（§5.1：turn / episodic 条目集 / J / σ）。

        供 core.recall.guard 比较；张量转 list（纯 stdlib 可比形态）。
        episodic 条目集指纹含全部可变标量字段（含 consistency）——既能
        抓条目增删，也能抓字段改写（旧 _attach_consistency 泄漏形态）。
        """
        return {
            "turn": self.turn_count,
            "episodic": episodic_fingerprint(self.memory.iter_episodic()),
            "J": self.attractor.J.detach().cpu().tolist(),
            "sigma": self.attractor.sigma.detach().cpu().tolist(),
        }

    def _recall_guard(self, scope: str) -> FourInvariantGuard:
        """只读四不变守卫（M2 机器防线，默认强制开启——§5.1 违反即抛）。"""
        return FourInvariantGuard(
            state_reader=self._readonly_state_snapshot, scope=scope)

    def doubt_status_block(self) -> dict:
        """阶段 3：precision 动态化观测块（/status precision_adapt、
        /react reaction.doubt、/recall doubt 共用）。

        开关关/异常 → {}（调用方 fail-open 降级回旧行为）。
        """
        if self.precision_adapt is None:
            return {}
        try:
            return self.precision_adapt.snapshot()
        except Exception:
            return {}

    def _retrieve_episodic(self, text: str) -> list[str] | None:
        """情景记忆检索：用语义向量找最相关的历史文本。

        优先用 384 维 raw 向量查询（高精度）；fallback 用 64 维投影向量
        （向后兼容旧快照中 64 维条目）；无 raw 向量时退化为投影向量查询。

        参数:
            text: 查询文本
        返回:
            检索到的情景文本列表，或 None（embedder 无 embed_text 方法时）
        """
        raw_vec, sem_vec = self._encode_query_vector(text)
        if raw_vec is None and sem_vec is None:
            return None
        episodic_query = raw_vec if raw_vec is not None else sem_vec
        # 提取层 v1.4（P2-B 写侧引用加固）：内部检索即"写侧引用匹配"——
        # 用本轮新条目向量（零额外 embed）对 episodic 引用匹配 top-k=3，
        # count_reference=True 默认路径（record_reference，复用 reference_count
        # 为加固计数唯一权威），并传入 reinforce_turn 刷新命中条目
        # last_reinforced_turn（wear 重新计时，丰碑"引用自动加固"）；
        # source_filter='external' 天然跳过 gray（source='store_gray'）——
        # 灰度三重冻结③。
        entries = self.memory.recall_episodic(
            episodic_query, top_k=3, fallback_query=sem_vec,
            reinforce_turn=self.turn_count)
        if entries:
            return [e.text for e in entries]
        return None

    def recall_episodic_readonly(self, query: str, k: int = 5) -> list[dict]:
        """只读情景检索（T1.3/P0-9：/recall 端点专用）。

        编码 query + 检索 episodic 缓冲区，**不做任何状态更新**：
          - 不 process_turn（不推断/不学习/不更新 turn_count）
          - 不调 LLM
          - 不写缓冲（不 store_episodic）
          - 不落盘（不保存快照）
          - record_reference=False（体验层 D：外部只读探针不计数引用，
            P0-12 零持久化保持）

        与 process_turn 内的检索共用同一套编码/检索逻辑
        （_encode_query_vector + memory.recall_episodic_scored）。

        体验层 D（设计 v1.1 §6.5）：检索结果应用**低置信复核配额**
        （relevance-gated：只有 cue_sim ≥ top1×0.5 的待复核条目才占
        1 个配额席位，否则宁缺毋滥——怀疑不变成跑偏源）。

        参数:
            query: 查询文本
            k: 返回条数（调用方已钳制到 [1,20]）

        返回:
            [{'text': str, 'score': float, ...}, ...]（按相关度降序）；
            embedder 不支持语义编码或缓冲区为空时返回空列表（fail-open）。
        """
        if not query or not query.strip():
            return []
        # M2：只读四不变守卫（§5.1 机器防线——违反即抛 ReadOnlyViolation）
        guard = self._recall_guard("recall_episodic_readonly")
        with guard:
            raw_vec, sem_vec = self._encode_query_vector(query)
            if raw_vec is None and sem_vec is None:
                return []  # embedder 无 embed_text：无可检索的语义空间（fail-open）
            query_vec = raw_vec if raw_vec is not None else sem_vec
            scored = self.memory.recall_episodic_scored(
                query_vec, top_k=k, fallback_query=sem_vec,
                count_reference=False)
            # M2：只读一致性投影——计算但不写回条目（旧 _attach_consistency
            # 的 entry.consistency 改写 = 只读泄漏 P1，已整体删除；条目指纹
            # 含 consistency 字段，任何改写都会被四不变守卫当场抓住）。
            cons_map = self._project_consistency_readonly(scored)
            # 体验层 D：低置信复核配额（relevance-gated，只读选择）
            try:
                from core.doubt.recall_scheduler import (
                    select_with_low_confidence_quota)
                scored = select_with_low_confidence_quota(scored, k)
            except Exception:
                pass
            results = []
            for score, entry in scored:
                text = getattr(entry, 'text', None)
                if text:
                    item = {'text': text, 'score': float(score)}
                    # 体验层 D（设计 v1.1 §8.1/§8.3）：置信度场字段注解
                    item['confidence'] = round(
                        float(getattr(entry, 'confidence', 1.0) or 1.0), 3)
                    item['rebuttal_count'] = int(
                        getattr(entry, 'rebuttal_count', 0) or 0)
                    item['labile'] = bool(getattr(entry, 'labile', False))
                    item['source_trust'] = round(
                        float(getattr(entry, 'source_trust', 1.0) or 1.0), 3)
                    item['last_recalled_at'] = getattr(
                        entry, 'last_recalled_at', None)
                    # 阶段 3：真实 precision 数据源（质疑层注入用）——
                    # adaptive_confidence（Koriat 混合）/ consistency（自一致性
                    # 投影）/ doubt_verdict（conformal 分位怀疑线判定）。
                    # M2：consistency 来自只读投影 cons_map（绝不写回条目）。
                    # 开关关时保持 None/False（零参与，旧客户端无感）。
                    if self.precision_adapt is not None:
                        try:
                            cons = cons_map.get(id(entry))
                            vconf = self.precision_adapt.verdict_confidence(
                                entry, cons)
                            item['adaptive_confidence'] = round(vconf, 3)
                            item['consistency'] = (
                                round(cons, 4) if cons is not None else None)
                            item['doubt_verdict'] = bool(
                                vconf < self.precision_adapt.doubt_threshold())
                        except Exception:
                            pass
                    results.append(item)
            # M2：检索时怀疑投影（§2.1——labile 内存态投影，只读不落库；
            # /recall 响应 suspicion 区段数据源）。
            # M3-1 衔接：经 doubt_state.retrieval_projection 执行——命中
            # 条目进 labile 窗口（内存态累积，M6 做梦期待核查清单）+ 投影
            # 结构逐键同构（M2 形态冻结，§4.1）。
            self.last_recall_suspicion = self.doubt_state.retrieval_projection(
                scored, precision_adapt=self.precision_adapt,
                consistency_provider=(
                    (lambda e: cons_map.get(id(e))) if cons_map else None))
            return results

    # ================================================================== #
    #  体验层 A（设计 v1.1 §3）：/react 实时反应只读接口（纯 infer 不 store）
    # ================================================================== #

    def react_readonly(self, user_input: str, k: int = 3) -> dict:
        """实时反应只读接口（/react 端点后端，infer-only）。

        编码 → FEP 推断 → 解读 → 返回。零持久化（设计 v1.1 §3.2 定案，
        P0-12 防查询回声零回归）：
          - 不 learn（不调 attractor.learn，不碰 J）
          - 不 purpose.adjust（只读 get_precision() + 元调整只读应用）
          - 不 memory.update / 不落 episodic（不调 store_episodic）
          - 不写回 sigma（update_internal_state=False，E-P2-5 现成机制）
          - turn_count 不变（只读锚点，对齐 /recall 惯例）
          - 不做长时潜变量 recall / 不 acquire_conversation（对齐 /recall
            无锁只读先例；做梦写 J 期间的瞬时读不一致可接受）
        唯一可变状态：self.react_surprise_history（内存 deque，不落盘
        不进快照）与 M2 的 self.last_recall_suspicion（检索时怀疑投影，
        同为内存态）——/react 是"读"，反应不是大脑轮次。

        参数:
            user_input: 当前用户输入文本。
            k: 检索条数 [0,10]；0 = 只要反应+解读（插件用，轻量）。

        返回:
            {turn_count, reaction, interpretation, recalled, detail} 字典。
        """
        text = user_input
        # M2：只读四不变守卫（§5.1——/react 是 infer-only 读口，执行前后
        # turn / episodic 条目集 / J / σ 零增量；违反抛 ReadOnlyViolation
        # → 端点 500 + 告警，绝不静默）。
        _guard = self._recall_guard("react_readonly")
        _before_readonly = _guard.snapshot()
        # 1. 编码（与 process_turn 相同：只对当前输入反应，不带 llm_output）
        sensory_input = self.encoder.encode(
            text, self.tokenizer, self.embedder)

        # 1.5 语义向量（用于检索，同 process_turn 1.5）
        semantic_vector = None
        raw_semantic_vector = None
        if hasattr(self.embedder, 'embed_text'):
            semantic_vector = self.embedder.embed_text(text)
            if hasattr(self.embedder, 'embed_text_raw'):
                raw_semantic_vector = self.embedder.embed_text_raw(text)

        # --- 元可塑性：调整后参数只读应用（不更新 meta）---
        _effective_lr, _meta_temp, _meta_orth, _meta_cw = \
            self._get_meta_adjusted_params()

        # 2. FEP 推断（★ update_internal_state=False：不写回 self.sigma）
        precision = self.purpose.get_precision()
        activation = self.attractor.infer(
            sensory_input.vector, precision,
            num_steps=self.config.get('num_infer_steps', 10),
            temperature_override=_meta_temp,
            update_internal_state=False,
        )

        # 5. 只读情景检索（k>0 时；体验层 A 阶段无 record_reference 钩子，
        #    不产生任何引用计数副作用）
        recalled = []
        self.last_recall_suspicion = empty_suspicion()
        if k > 0:
            query_vec = (raw_semantic_vector if raw_semantic_vector is not None
                         else semantic_vector)
            if query_vec is not None:
                try:
                    scored = self.memory.recall_episodic_scored(
                        query_vec, top_k=k, fallback_query=semantic_vector,
                        source_filter='external', count_reference=False)
                    # M2：只读一致性投影——计算但不写回条目（旧
                    # _attach_consistency 的 entry.consistency 改写 = 只读
                    # 泄漏 P1，已整体删除）。
                    self._project_consistency_readonly(scored)
                    for score, entry in scored:
                        if getattr(entry, 'text', None):
                            recalled.append({
                                'text': entry.text,
                                'score': float(score),
                                'origin': 'memory',
                            })
                    # M2：检索时怀疑投影（§2.1——labile 内存态投影，只读
                    # 不落库；/react 响应 suspicion 区段数据源）。
                    # M3-1 衔接：同 recall 路径——进 labile 窗口 + 投影
                    # 逐键同构（形态冻结）。
                    self.last_recall_suspicion = self.doubt_state.retrieval_projection(
                        scored, precision_adapt=self.precision_adapt)
                except Exception as e:  # pylint: disable=broad-except
                    # fail-open：只读检索异常不阻塞反应返回
                    logger.warning(
                        "react_readonly 只读检索失败（fail-open）: %s", e)

        # 6. 记忆状态解读（decoder 组装函数：复用熵/coherence 模板，
        #    惊讶解读用 /react 自有窗口的等价实现，_interpret_surprise 本体不动）
        surprise_window = list(self.react_surprise_history)
        interpretation = self.decoder.build_react_interpretation(
            activation, self.purpose.coherence, surprise_window)

        # 7. 唯一可变状态：记录惊讶度到 /react 自有窗口
        #    （在解读之后记录，避免当前值参与自身均值比较）
        self.react_surprise_history.append(activation.surprise)

        # 详细数据（向后兼容格式，复用 decoder._build_detail）
        detail = self.decoder._build_detail(activation)

        # 反应指标（surprise/free_energy 语义一字不动：准确性项恒≥0）
        max_entropy = (math.log(self.attractor.num_nodes)
                       if self.attractor.num_nodes > 1 else 1.0)
        entropy_ratio = (activation.entropy / max_entropy
                         if max_entropy > 0 else 0.0)
        surprise_z = None
        if len(surprise_window) >= 3:
            mean_s = sum(surprise_window) / len(surprise_window)
            std_s = (sum((v - mean_s) ** 2 for v in surprise_window)
                     / len(surprise_window)) ** 0.5
            if std_s > 1e-8:
                surprise_z = round(
                    (activation.surprise - mean_s) / std_s, 3)

        reaction = {
            'entropy': round(float(activation.entropy), 4),
            'entropy_ratio': round(float(entropy_ratio), 4),
            'surprise': round(float(activation.surprise), 4),
            'free_energy': round(float(activation.free_energy), 4),
            'mse': (round(float(activation.mse), 4)
                    if activation.mse is not None else None),
            'coherence': round(float(self.purpose.coherence), 4),
            'precision_mean': round(float(precision.mean()), 4),
            'surprise_z': surprise_z,
            # 阶段 3：precision 动态化观测块（质疑层数据源——全局怀疑
            # 基线/conformal 分位怀疑线/域怀疑；经 glue /react 薄代理
            # 原样透传到注入插件。开关关 → {}（插件回退旧行为））。
            'doubt': self.doubt_status_block(),
        }
        # M2：只读四不变断言（§5.1——/react 执行前后四量零增量；
        # 违反 → ReadOnlyViolation → 端点 500，绝不静默）。
        _guard.assert_unchanged(_before_readonly)

        return {
            'turn_count': self.turn_count,
            'reaction': reaction,
            'interpretation': interpretation,
            'recalled': recalled,
            'detail': detail,
        }

    # ================================================================== #
    #  T2.3 检索扩容：归档导出 + 合并检索（内存活体优先、归档补充）
    # ================================================================== #

    def _export_episodic_to_archive(self) -> int:
        """把当前情景缓冲区的条目追加导出到 data/archive/{session}.jsonl。

        （T2.3：快照落盘时调用，按 (turn, text_hash) 去重；
        供窗口外记忆的归档补充检索使用。任何异常由调用方兜底，fail-open。）

        返回:
            本次新增的归档条目数。
        """
        from core.archive.archive_store import export_episodic
        # 提取层 v1.4（S1-1，灰度三重冻结闭环）：store_gray 条目不进归档——
        # 内存路径由 source_filter='external' 过滤，归档若导出 gray 则 /recall
        # 的归档补充检索会绕过内存过滤泄漏 gray（L1 不可见被破坏）。灰度
        # 观察通道 = status/episodic 尾部直接抽查（非 /recall）。
        entries = [e for e in self.memory.iter_episodic()
                   if getattr(e, 'source', 'external') != 'store_gray']
        if not entries:
            return 0
        return export_episodic(
            self.session_id, entries,
            archive_dir=self.config.get('archive_dir'))

    def recall_merged_readonly(self, query: str, k: int = 5) -> list[dict]:
        """合并情景检索（T2.3）：内存 200 条 ∪ 归档，内存优先、归档带来源标记。

        **护栏（设计原理核对报告 2026-08-10 §5 R1）**：
          - 内存（活体）结果 tier0 优先展示，归档结果仅作 tier1 补充——
            绝不让 SQLite/JSONL 冷检索成为主路径；
          - 归档条目显式携带 ``origin='archive'`` 来源标记；
          - 归档检索带超时（默认 500ms，LMS_ARCHIVE_TIMEOUT_MS 可覆盖）：
            超时/异常一律跳过归档只回内存（fail-open），/recall 响应 <2s 目标不变；
          - 合并开关 ``LMS_ARCHIVE_ENABLED=0`` 可一键关闭（回滚路径），
            关闭时行为与旧版 recall_episodic_readonly 完全一致。

        与 process_turn 内检索（_retrieve_episodic）的关系：进程内每轮检索仍只走
        内存窗口（快路径），本入口只供 /recall 等**外部只读查询**做归档扩容——
        活体检索（attractor 驱动的潜变量检索）路径完全未动。

        参数:
            query: 查询文本。
            k: 返回条数（调用方已钳制到 [1,20]）。

        返回:
            [{'text', 'score', 'origin'}, ...]（内存条目在前，归档条目在后，
            各自按相似度降序；按 text 去重，内存版本优先；最多 k 条）。
        """
        # M2：只读四不变守卫（§5.1——/recall 全路径含归档补充检索段；
        # 违反抛 ReadOnlyViolation → 端点 500 + 告警，绝不静默）。
        guard = self._recall_guard("recall_merged_readonly")
        with guard:
            return self._recall_merged_readonly_impl(query, k=k)

    def _recall_merged_readonly_impl(self, query: str, k: int = 5) -> list[dict]:
        """合并检索内部实现（由 recall_merged_readonly 守卫包装内执行）。

        与旧 recall_merged_readonly 行为逐字节一致（血管不换），仅被
        只读四不变守卫包裹。
        """
        # 0. 合并开关（LMS_ARCHIVE_ENABLED=0 关闭合并，回滚路径）
        archive_enabled = str(self.config.get(
            'archive_enabled',
            os.environ.get('LMS_ARCHIVE_ENABLED', '1'))).strip().lower()
        archive_enabled = archive_enabled not in ('0', 'false', 'no', 'off')
        if not archive_enabled:
            return self.recall_episodic_readonly(query, k=k)

        # 1. 内存路径（活体优先；行为与旧版一致，仅补充 origin 标记）
        results = self.recall_episodic_readonly(query, k=k)
        for r in results:
            r['origin'] = 'memory'

        # 2. 归档补充检索（带超时，fail-open）
        if not query or not query.strip():
            return results
        raw_vec, sem_vec = self._encode_query_vector(query)
        if raw_vec is None and sem_vec is None:
            return results  # embedder 无语义编码能力：只回内存
        query_vec = raw_vec if raw_vec is not None else sem_vec
        archive_dir = self.config.get('archive_dir')
        timeout_ms = int(self.config.get(
            'archive_timeout_ms',
            os.environ.get('LMS_ARCHIVE_TIMEOUT_MS', '500')))

        try:
            from core.archive.archive_store import query_archive
            fut = _get_archive_executor().submit(
                query_archive, self.session_id, query_vec, k, archive_dir)
            try:
                archive_results = fut.result(timeout=max(0.05, timeout_ms / 1000.0))
            except TimeoutError:
                # 归档扫描超时：跳过归档只回内存（fail-open），
                # 后台只读扫描自然结束，不影响后续请求
                logger.warning(
                    f"[{self.session_id}] 归档检索超时"
                    f"（>{timeout_ms}ms），本次跳过归档（fail-open）")
                archive_results = []
        except Exception as e:
            # 归档 IO/解析异常：跳过归档只回内存（fail-open）
            logger.warning(
                f"[{self.session_id}] 归档检索失败，本次跳过归档"
                f"（fail-open）: {e}")
            archive_results = []

        # 3. 融合：内存优先展示（tier0），归档补充（tier1）
        #    按 text 去重（同文本出现两次时内存版本先到、保留内存版）
        seen: set = set()
        merged: list = []
        for r in results + archive_results:
            t = r.get('text')
            if not t or t in seen:
                continue
            seen.add(t)
            merged.append(r)
        return merged[:k]

    def _snapshot_dir_path(self) -> Path:
        """快照根目录：优先 config['snapshot_dir']，否则 core.paths.get_snapshot_dir()。

        （T1.1/P0-5：从 _auto_snapshot / dream 中提取的公共逻辑，
        避免两处重复且口径不一致。）
        """
        snapshot_dir_cfg = self.config.get('snapshot_dir')
        if snapshot_dir_cfg:
            return Path(snapshot_dir_cfg).expanduser()
        return get_snapshot_dir()

    def _auto_snapshot(self) -> None:
        """自动快照（G4 修复：按间隔自动保存状态）。

        根据配置的间隔，在特定轮次自动保存状态快照。
        0.5.0/T1.1：改用会话级命名规范 save_session_state()——
        `snapshots/{session}/snapshot_{session}_{turn}_{ts}.pt` +
        同步 `snapshots/{session}/latest_{session}.pt`。
        """
        # 7. 自动快照（G4 修复：按间隔自动保存状态）
        if self.config.get('auto_snapshot', False):
            interval = self.config.get('auto_snapshot_interval', 50)
            if self.turn_count > 0 and self.turn_count % interval == 0:
                try:
                    path = self.save_session_state()
                    if path:
                        logger.info(f"自动快照已保存: {path}")
                except Exception as e:
                    logger.warning(f"自动快照失败: {e}")

    def query_llm(self, user_input: str) -> str:
        """使用记忆context查询主LLM。

        参数:
            user_input: 用户输入文本

        返回:
            LLM的响应文本

        异常:
            RuntimeError: 未配置LLM Bridge时抛出
        """
        if self.bridge is None:
            raise RuntimeError("未配置LLM Bridge，无法查询LLM")

        # 获取记忆context（同样接入长时记忆检索和情景记忆，保持读取路径一致）
        if self.last_activation is not None:
            recalled = self.memory.recall(self.last_activation.state)
            # 情景记忆检索：优先用 384 维 raw 向量，fallback 用投影向量
            episodic_texts = self._retrieve_episodic(user_input)
            memory_context = self.decoder.decode(
                self.last_activation, recalled_memory=recalled,
                episodic_texts=episodic_texts,
                coherence=self.purpose.coherence)
        else:
            memory_context = "[无记忆]"

        # 查询LLM
        response = self.bridge.query(user_input, memory_context)

        # 将LLM输出也送入记忆系统
        self.process_turn(user_input, response)

        return response

    def save_state(self, path: str) -> bool:
        """保存当前状态到快照文件。

        保存吸引子景观（J矩阵、bias、sigma）、目的层状态（precision、history、
        coherence、encounter_count）、记忆潜变量（short/long_term_latent）
        和分词器词表。

        0.5.0/T1.1：快照顶层额外写入元数据 session_id / turn_count /
        last_entropy_ratio（供重启后恢复轮次连续与归属校验）。

        参数:
            path: 快照文件路径

        返回:
            True 表示已保存；False 表示因写锁超时被跳过（fail-open，见
            persistence.snapshot.save）。
        """
        # 获取吸引子景观
        landscape = self.attractor.get_landscape()

        # 获取目的层状态并转为字典
        # N3: encounter_count 纳入持久化
        purpose = self.purpose.get_purpose()
        purpose_dict = {
            'precision': purpose.precision,
            'history': purpose.history,
            'coherence': purpose.coherence,
            'encounter_count': purpose.encounter_count,
        }

        # N1: 获取记忆潜变量状态
        memory_state = self.memory.get_state()

        # N2: 获取 tokenizer 词表（tokenizer 是 runtime 层依赖）
        tokenizer_state = None
        if hasattr(self.tokenizer, 'get_vocab'):
            tokenizer_state = self.tokenizer.get_vocab()

        # 元可塑性状态
        meta_state = None
        if self.meta is not None:
            meta_state = self.meta.get_state()

        # 自指回路状态（可选）
        self_ref_state = None
        if self.self_ref is not None:
            self_ref_state = self.self_ref.get_state()

        saved = self.snapshot.save(path, landscape, purpose_dict,
                           memory_state=memory_state,
                           tokenizer_state=tokenizer_state,
                           meta_state=meta_state,
                           self_ref_state=self_ref_state,
                           session_id=self.session_id,
                           turn_count=self.turn_count,
                           last_entropy_ratio=self.last_entropy_ratio)
        if not saved:
            # P1-1 修复：传播写锁超时的真实结果，禁止"声称已保存、实际未落盘"。
            # （调用方依赖该返回值触发 503/跳过提示/优雅停机重试。）
            logger.warning(f"状态保存被跳过（写锁超时，fail-open）: {path}")
            return False
        logger.info(f"状态已保存到 {path}")
        return True

    def save_session_state(self) -> Optional[str]:
        """按新命名规范保存会话快照，并同步更新 latest_{session}.pt（T1.1/P0-5）。

        - 轮次快照：snapshots/{session}/snapshot_{session}_{turn}_{ts}.pt
          （按轮次归档，天然隔离会话，杜绝跨会话撞名）；
        - 最新指针：snapshots/{session}/latest_{session}.pt（写最新时同步更新，
          供加载方快速定位该会话最新状态）。

        返回:
            轮次快照路径；保存被写锁超时跳过时返回 None（fail-open）。
        """
        snap_dir = self._snapshot_dir_path()
        snap_dir.mkdir(parents=True, exist_ok=True)
        turn_path = snapshot_path_for(
            str(snap_dir), self.session_id, self.turn_count)
        saved = self.save_state(turn_path)
        if not saved:
            return None
        latest_path = latest_path_for(str(snap_dir), self.session_id)
        try:
            self.snapshot.save_copy(turn_path, latest_path)
        except Exception as e:
            logger.warning(
                f"同步 latest_{self.session_id}.pt 失败（不影响主快照）: {e}")

        # T2.3 检索扩容：快照落盘后把 episodic 追加导出到归档
        # （data/archive/{session}.jsonl，按 (turn,text_hash) 去重）。
        # fail-open：归档导出失败绝不回滚/中断快照主流程，仅告警。
        try:
            added = self._export_episodic_to_archive()
            logger.debug(
                f"[{self.session_id}] episodic 归档导出 +{added} 条")
        except Exception as e:
            logger.warning(
                f"[{self.session_id}] episodic 归档导出失败"
                f"（fail-open，不影响快照）: {e}")
        return turn_path

    def latest_snapshot_path(self) -> str:
        """返回当前会话最新快照路径（snapshots/{session}/latest_{session}.pt）。

        （供 API 层在 /snapshot 响应中回传；不校验文件是否存在。）
        """
        return latest_path_for(str(self._snapshot_dir_path()), self.session_id)

    def load_state(self, path: str) -> None:
        """从快照文件恢复状态。

        恢复吸引子景观、目的层状态（含 encounter_count）、记忆潜变量
        和分词器词表。向后兼容：旧版快照无 memory/tokenizer 字段时优雅降级。

        参数:
            path: 快照文件路径

        异常:
            RuntimeError: 恢复失败时抛出
        """
        # N1: 传入 memory 对象，recovery 会自动检测并恢复 memory 字段
        success = self.recovery.recover(
            path, self.attractor, self.purpose, memory=self.memory
        )
        if not success:
            raise RuntimeError(f"无法从 {path} 恢复状态")

        # N2: 恢复 tokenizer 词表（tokenizer 是 runtime 层依赖，
        # 不经过 persistence 层的 Protocol，在 loop.py 中直接处理）
        raw_data = self.snapshot.load_raw(path)
        tokenizer_state = raw_data.get('tokenizer')
        if tokenizer_state is not None and hasattr(self.tokenizer, 'set_vocab'):
            self.tokenizer.set_vocab(tokenizer_state)
            logger.info("tokenizer 词表已恢复")
        else:
            logger.info("快照不含 tokenizer 字段，跳过词表恢复（向后兼容）")

        # 元可塑性状态恢复
        meta_state = raw_data.get('meta')
        if meta_state is not None and self.meta is not None:
            self.meta.set_state(meta_state)
            logger.info("元可塑性状态已恢复")
        elif meta_state is not None and self.meta is None:
            logger.info("快照含 meta 字段但元学习未启用，跳过恢复")
        else:
            logger.info("快照不含 meta 字段，跳过元状态恢复（向后兼容）")

        # 自指回路状态恢复
        self_ref_state = raw_data.get('self_ref')
        if self_ref_state is not None and self.self_ref is not None:
            self.self_ref.set_state(self_ref_state)
            logger.info("自指回路状态已恢复")
        elif self_ref_state is not None and self.self_ref is None:
            logger.info("快照含 self_ref 字段但自指回路未启用，跳过恢复")
        else:
            logger.info("快照不含 self_ref 字段，跳过自指回路恢复（向后兼容）")

        # 0.5.0/T1.1: 恢复 turn_count / last_entropy_ratio 元数据（向后兼容：
        # 旧快照无这些字段时 turn_count 从 0 重数，并记录 WARNING）
        snap_turn = raw_data.get('turn_count')
        if snap_turn is None:
            logger.warning(
                f"快照 {path} 无 turn_count 字段（旧版快照），turn_count 从 0 重数")
            self.turn_count = 0
        else:
            self.turn_count = int(snap_turn)
        snap_entropy = raw_data.get('last_entropy_ratio')
        if snap_entropy is not None:
            self.last_entropy_ratio = float(snap_entropy)
        snap_session = raw_data.get('session_id')
        if snap_session is not None and str(snap_session) != self.session_id:
            logger.warning(
                f"快照 session_id={snap_session} 与当前会话 "
                f"{self.session_id} 不一致（以当前会话为准）")

        logger.info(f"已从 {path} 恢复状态")

    def get_status(self) -> dict:
        """获取当前运行状态摘要。

        返回:
            状态字典，包含轮次、激活熵、惊讶度等
        """
        status = {
            'turn_count': self.turn_count,
            'num_nodes': self.attractor.num_nodes,
            'input_dim': self.attractor.input_dim,
        }

        if self.last_activation is not None:
            status['last_entropy'] = self.last_activation.entropy
            status['last_surprise'] = self.last_activation.surprise
            # 2026-08-10 惊讶度语义拆分（设计v1.1 §3.5/C9）：增量暴露自由能
            # 与 MSE（纯增量字段，旧客户端忽略）。surprise 为准确性项恒≥0；
            # free_energy 为未规范化变分能量可负，仅供学习目标/诊断。
            status['last_free_energy'] = self.last_activation.free_energy
            status['last_mse'] = self.last_activation.mse

        # 在线熵管理状态
        status['entropy_ratio'] = self.last_entropy_ratio
        status['entropy_high_threshold'] = self.config.get('entropy_high_threshold', 0.9)
        status['entropy_low_threshold'] = self.config.get('entropy_low_threshold', 0.5)

        # 提取层 v1.4（S1-7/S1-9/S1-14）：熔断降级与容量观测（纯增量字段，
        # 旧客户端忽略；进灰度仪表与 dream_state.json）
        status['degraded_events'] = int(getattr(self, 'degraded_events', 0) or 0)
        status['last_turn_degraded'] = bool(
            getattr(self, 'last_turn_degraded', False))
        try:
            status['capacity_usage'] = self.memory.episodic_size()
            status['capacity_soft_limit'] = self.memory.get_episodic_maxlen()
            status['capacity_full_events'] = int(getattr(
                self.memory, 'capacity_full_events', 0) or 0)
            status['capacity_hard_drops'] = int(getattr(
                self.memory, 'capacity_hard_drops', 0) or 0)
            status['capacity_warning_events'] = int(getattr(
                self.memory, 'capacity_warning_events', 0) or 0)
        except Exception:
            pass

        purpose = self.purpose.get_purpose()
        status['purpose_coherence'] = purpose.coherence
        status['precision_mean'] = float(purpose.precision.mean())
        status['precision_std'] = float(purpose.precision.std())

        # 元可塑性状态（可选）
        if self.meta is not None:
            status['meta'] = self.meta.get_status()

        # 体验层 D（设计 v1.1 §6.6/§8.1）：doubt 增量字段
        #   gaps: 信息缺口（A/B 类进怀疑灯，C 类仅诊断——专注化修订）
        #   labile_count / low_confidence_count: 置信度场体检
        #   last_review: 最近一次做梦怀疑复核时间
        # 纯增量字段：C-01 required（entropy_ratio/purpose_coherence/
        # turn_count）不变 → 绿。fail-open：任何异常只缺省该字段。
        try:
            labile_count = 0
            low_conf_count = 0
            for e in self.memory.iter_episodic():
                if bool(getattr(e, 'labile', False)):
                    labile_count += 1
                try:
                    if float(getattr(e, 'confidence', 1.0) or 1.0) < 0.3:
                        low_conf_count += 1
                except (TypeError, ValueError):
                    pass
            status['doubt'] = {
                'gaps': self.gap_registry.snapshot(),
                'labile_count': labile_count,
                'low_confidence_count': low_conf_count,
                'last_review': self.gap_registry.last_review(),
            }
        except Exception as e:  # pylint: disable=broad-except
            logger.debug("get_status doubt 字段组装失败（fail-open）: %s", e)

        # 阶段 3：precision 三层动态化观测（纯增量字段；开关关 → {}）
        try:
            status['precision_adapt'] = self.doubt_status_block()
        except Exception as e:  # pylint: disable=broad-except
            logger.debug(
                "get_status precision_adapt 字段组装失败（fail-open）: %s", e)

        # M3-1（规格 v2 §二）：三时相怀疑状态机 + 验证链观测（纯增量字段；
        # 开关关 → {'enabled': False}；旧客户端忽略——§4.2 独立追加语义）。
        try:
            status['doubt_native'] = self.doubt_state.snapshot()
        except Exception as e:  # pylint: disable=broad-except
            logger.debug(
                "get_status doubt_native 字段组装失败（fail-open）: %s", e)

        # E3（自我怀疑驱动的主动调节，dandan 拍板 2026-08-20）：观测块
        # （闭环计数/待机态/最近悬案/冷却倒计时——A5/A6 数据源；纯增量
        # 字段；总开关关 → {'enabled': False}，旧客户端忽略）。
        try:
            status['e3'] = self.e3_observation()
        except Exception as e:  # pylint: disable=broad-except
            logger.debug(
                "get_status e3 字段组装失败（fail-open）: %s", e)

        # 论文机制 A：allostatic J 滑动设定点观测（纯增量字段；开关关 →
        # {'enabled': False}——§4.2 独立追加语义，旧客户端忽略）。
        # 灵魂指标②③：j_history 序列动态（非固定）+ events 越界触发可观测。
        # M4：原生并入 attractor 后由 attractor.allostatic_snapshot 提供。
        try:
            status['allostatic_j'] = self.attractor.allostatic_snapshot(
                turn_count=self.turn_count)
        except Exception as e:  # pylint: disable=broad-except
            logger.debug(
                "get_status allostatic_j 字段组装失败（fail-open）: %s", e)

        # M5（§7.1）：turn 生命周期状态汇总观测（纯增量字段；未跑过
        # process_turn 时 last_turn_lifecycle 为 None → 提供空汇总提示——
        # §4.2 独立追加语义，旧客户端忽略）。
        try:
            status['lifecycle'] = self.last_turn_lifecycle or {
                'turns': self.turn_count,
                'broken': None,
                'note': '尚未执行 process_turn（无生命周期记录；'
                        '/recall 与 /react 只读口不产生记录）',
            }
        except Exception as e:  # pylint: disable=broad-except
            logger.debug(
                "get_status lifecycle 字段组装失败（fail-open）: %s", e)

        # 自指回路状态（可选）
        if self.self_ref is not None:
            status['self_ref_enabled'] = True
            status['self_ref_alpha'] = self.config.get('self_ref_alpha_base', 0.15)
            # ★ Phase 1 增强：暴露更多自指监控字段。
            # 用 .get() 安全访问，self_referential.py 增强未就绪时为 None。
            sr_status = self.self_ref.get_status()
            status['self_ref_autocorr'] = sr_status.get('autocorr')
            status['self_ref_state'] = sr_status.get('state', 'normal')
            status['self_ref_ext_novelty'] = sr_status.get('ext_novelty')
            # Phase 2 新增
            status['self_ref_dream_stale'] = sr_status.get('dream_stale', False)
            status['self_ref_dream_age'] = sr_status.get('dream_age', 0)
            status['self_ref_is_dominant'] = sr_status.get(
                'is_self_ref_dominant', False)
        else:
            status['self_ref_enabled'] = False

        return status

    # ================================================================== #
    #  做梦引擎集成
    # ================================================================== #

    def get_dream_engine(self):
        """获取做梦引擎实例（懒加载）。

        首次调用时创建 DreamEngine 实例，传入当前组件引用和配置。
        后续调用直接返回已创建的实例。

        返回:
            DreamEngine 实例。
        """
        if self.dream_engine is None:
            from core.hippocampus.dream_engine import DreamEngine
            # 合并做梦相关配置：从 self.config 读取，补充默认值
            dream_config = dict(self.config)
            # 确保快照目录指向系统快照目录
            dream_config.setdefault(
                'snapshot_dir',
                self.config.get('snapshot_dir', 'snapshots'))
            # 体验层 D：怀疑复核联动 gap_registry（B 类清空 + 复核报告）
            dream_config['gap_registry'] = self.gap_registry
            # 阶段 3：precision 动态化状态传入做梦引擎（doubt_review 的
            # 低置信复核阈值改用 conformal 分位怀疑线；None=开关关，回退 0.3）
            dream_config['precision_adapt'] = self.precision_adapt
            # M6（规格 v2 §2.4）：三时相怀疑状态机 + 验证链注入做梦引擎
            # ——梦期巩固时相（_consolidation_phase）对 labile/suspect 条目
            # 走 consolidation_resolve（confirm/supersedes 两分支，写侧唯一
            # 转移入口）。缺省 None=独立使用，回退纯函数 resolve_labile。
            dream_config['doubt_state'] = self.doubt_state
            dream_config['verification_chain'] = self.verification_chain
            self.dream_engine = DreamEngine(
                attractor=self.attractor,
                purpose=self.purpose,
                memory=self.memory,
                embedder=self.embedder,
                config=dream_config,
                meta=self.meta,  # 新增
            )
            logger.info("DreamEngine 已懒加载创建")
        return self.dream_engine

    def dream(self, n_steps: int = 20, full_cycle: bool = False) -> dict:
        """触发记忆系统的"做梦"过程。

        在空闲时进行记忆巩固、遗忘和整合，让记忆系统在无对话输入时
        持续运转。做梦后自动保存快照。

        参数:
            n_steps: 做梦步数（默认 20）。
            full_cycle: False 时执行 MVP 做梦（dream_mvp），
                True 时执行完整做梦周期（dream_cycle）。

        返回:
            做梦统计字典（由 DreamEngine 返回），额外包含
            'snapshot_saved' 字段表示是否成功保存快照。
        """
        dream_engine = self.get_dream_engine()

        # ★ Phase 4: 记录做梦开始时间（仅用于外围发布钩子，不影响算法）
        _dream_t0 = time.time()

        # ★ Phase 2: 做梦前钩子——标记自述状态为陈旧
        if self.self_ref is not None:
            self.self_ref.on_dream_start()

        if full_cycle:
            result = dream_engine.dream_cycle(max_steps=n_steps)
        else:
            result = dream_engine.dream_mvp(n_steps=n_steps)
        _dream_duration = time.time() - _dream_t0

        # ★ Phase 2: 做梦后钩子——衰减陈旧自述，清除 stale 标记
        if self.self_ref is not None:
            self.self_ref.on_dream_end()

        # 做梦后自动保存快照（完整状态，含 tokenizer）
        # 0.5.0/T1.1：会话级命名规范 save_session_state()——
        # `snapshots/{session}/snapshot_{session}_{turn}_{ts}.pt` +
        # 同步 `snapshots/{session}/latest_{session}.pt`（替换原平铺 latest.pt）
        snapshot_saved = False
        try:
            path = self.save_session_state()
            if path:
                snapshot_saved = True
                logger.info(f"做梦后快照已保存: {path}")
        except Exception as e:
            logger.warning(f"做梦后快照保存失败（不影响做梦结果）: {e}")

        result['snapshot_saved'] = snapshot_saved

        # ★ Phase 4 钩子：做梦完成反哺总线（外围、静默降级，绝不影响主循环）
        self._maybe_publish_dream_complete(result, _dream_duration)

        # ★ M6（规格 v2 §2.4/§4.3）钩子：梦期巩固时相结果发布
        # （lms.doubt_consolidation——confirm/supersedes 结果 + supersedes
        # 记录 + 验证链快照；落沙必发事件，坑 7 根治。静默降级，绝不影响
        # 做梦结果返回）
        self._maybe_publish_doubt_consolidation(result)

        return result

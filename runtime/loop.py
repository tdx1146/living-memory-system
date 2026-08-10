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

logger = logging.getLogger(__name__)


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

        logger.info(
            f"LivingMemoryLoop已初始化 "
            f"(nodes={self.attractor.num_nodes}, dim={self.attractor.input_dim})"
        )

    def process_turn(self, user_input: str, llm_output: str = "") -> str:
        """处理一轮对话，执行完整的记忆循环。

        流程：
        1. 编码输入（用户输入+LLM输出 -> 感官向量）
        2. FEP推断（感官向量 + precision -> 激活态）
        3. FEP学习（更新J矩阵）
        3.5 在线熵管理（熵过高增强正交化 / 熵过低放松正交化）
        4. 调整目的（更新precision）
        5. 记忆更新与巩固（短时更新 + 定期短时->长时迁移）
        5.5 记忆检索（用当前激活态作为线索，检索长时记忆）—— S1 修复
        6. 解码context（激活态 + 检索到的记忆 -> context文本）
        7. 自动快照（可选，按间隔保存状态）—— G4 修复
        8. 返回context

        参数:
            user_input: 用户输入文本
            llm_output: LLM输出文本（上一轮的，首轮可为空）

        返回:
            记忆context文本（供LLM查询使用）
        """
        # 1. 编码输入
        text = f"用户: {user_input}\n助手: {llm_output}" if llm_output else user_input
        sensory_input = self.encoder.encode(text, self.tokenizer, self.embedder)

        # ★ 插入点 A：自指回注（提取为 _inject_self_ref）
        # 在编码后、推断前，将上一轮蒸馏的自述以自适应权重回注到感官向量。
        # 默认关闭（self_ref is None）时此块完全跳过，sensory_input 保持原样。
        sensory_input, alpha_t, is_self_ref_dominant = self._inject_self_ref(
            sensory_input)

        # 1.5 获取语义向量（用于情景记忆存储与检索）
        # PretrainedEmbedder 提供 embed_text；SimpleEmbedder 无此方法时跳过
        semantic_vector = None
        raw_semantic_vector = None
        if hasattr(self.embedder, 'embed_text'):
            semantic_vector = self.embedder.embed_text(text)
            # 384 维原始语义向量（投影前），用于 episodic buffer 高精度检索
            # PretrainedEmbedder 提供 embed_text_raw；无此方法时退化为 None
            # （如自定义 embedder 仅有 embed_text），此时退化为用投影向量
            if hasattr(self.embedder, 'embed_text_raw'):
                raw_semantic_vector = self.embedder.embed_text_raw(text)

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
        self.memory.update(activation, activation.surprise)
        if self.turn_count > 0 and \
           self.turn_count % self.consolidation_interval == 0:
            self.memory.consolidate()
            logger.debug(f"第{self.turn_count}轮：执行记忆巩固")

        # 5.5 长时记忆检索（S1 修复：用当前激活态作为线索，检索长时记忆）
        # recall() 返回 [num_nodes] 维向量，与 activation.state 同维。
        # 这是"记忆只写不读"缺陷的核心修复点：长时记忆通过此步进入输出路径。
        recalled = self.memory.recall(activation.state)

        # 5.6 情景记忆检索（用语义向量找最相关的历史文本）
        # 先检索后存储：避免当前轮文本出现在检索结果中
        episodic_texts = self._retrieve_episodic(text)

        # 6. 解码context（传入检索到的记忆和情景文本，使长期记忆+语义文本参与输出）
        # A-P1-2: 传入 purpose.coherence，使解码器能输出"关注方向"解读
        memory_context = self.decoder.decode(
            activation, recalled_memory=recalled,
            episodic_texts=episodic_texts,
            coherence=self.purpose.coherence)

        # ★ 插入点 B：自指观测
        # 观测自述：将 decoder 输出和当前激活态送入自指回路进行蒸馏缓存，
        # 供下一轮 generate_echo 使用。默认关闭时此块完全跳过。
        if self.self_ref is not None:
            self.self_ref.observe(memory_context, activation)

        # 保存当前 activation 供下轮 autocorr 计算（Phase 0 预留）
        self._prev_activation = activation

        # 6.5 情景记忆存储（当前轮文本存入缓冲区，供后续检索）
        # 优先存 384 维 raw 向量；无 raw 向量时退化为投影向量（向后兼容）
        if semantic_vector is not None:
            self.memory.store_episodic(
                text, semantic_vector, activation.surprise, self.turn_count,
                raw_semantic_vector=raw_semantic_vector,
                source='external')  # Phase 2: 显式来源标记

        # 更新状态
        self.last_activation = activation
        self.turn_count += 1

        logger.debug(
            f"第{self.turn_count}轮: "
            f"熵={activation.entropy:.3f}, "
            f"惊讶度={activation.surprise:.3f}"
        )

        # 7. 自动快照（G4 修复：按间隔自动保存状态）—— 提取为 _auto_snapshot
        self._auto_snapshot()

        # ★ Phase 4 钩子：塑形状态反哺总线（外围、静默降级，绝不影响主循环）
        self._maybe_publish_plastified(activation)

        # 8. 返回context
        return memory_context

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
            })
        except Exception as e:
            logger.debug("Phase 4 dream_complete 发布跳过（静默降级）: %s", e)

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
        entries = self.memory.recall_episodic(
            episodic_query, top_k=3, fallback_query=sem_vec)
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

        与 process_turn 内的检索共用同一套编码/检索逻辑
        （_encode_query_vector + memory.recall_episodic_scored）。

        参数:
            query: 查询文本
            k: 返回条数（调用方已钳制到 [1,20]）

        返回:
            [{'text': str, 'score': float}, ...]（按相关度降序）；
            embedder 不支持语义编码或缓冲区为空时返回空列表（fail-open）。
        """
        if not query or not query.strip():
            return []
        raw_vec, sem_vec = self._encode_query_vector(query)
        if raw_vec is None and sem_vec is None:
            return []  # embedder 无 embed_text：无可检索的语义空间（fail-open）
        query_vec = raw_vec if raw_vec is not None else sem_vec
        scored = self.memory.recall_episodic_scored(
            query_vec, top_k=k, fallback_query=sem_vec)
        results = []
        for score, entry in scored:
            text = getattr(entry, 'text', None)
            if text:
                results.append({'text': text, 'score': float(score)})
        return results

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
        entries = list(self.memory.iter_episodic())
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

        purpose = self.purpose.get_purpose()
        status['purpose_coherence'] = purpose.coherence
        status['precision_mean'] = float(purpose.precision.mean())
        status['precision_std'] = float(purpose.precision.std())

        # 元可塑性状态（可选）
        if self.meta is not None:
            status['meta'] = self.meta.get_status()

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

        return result

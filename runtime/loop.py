"""在线学习环（主循环）

活体记忆系统的主循环，对每轮对话执行完整的记忆循环：
编码输入 -> 推断 -> 学习 -> 调整目的 -> 巩固记忆 -> 检索记忆 -> 解码context -> 返回

其中"检索记忆"步骤（S1 修复）是关键：长时记忆通过 recall() 进入解码路径，
不再"只写不读"。

遵循架构文档第四节的数据流定义。
"""

import os
import math
import logging
from typing import Optional

import torch

from core.types import Activation
from core.hippocampus.attractor import AttractorNetwork
from core.hippocampus.purpose import PurposeLayer
from core.hippocampus.memory import MemoryManager
from core.sensory.tokenizer import SimpleTokenizer
from core.sensory.embedder import SimpleEmbedder

from bridge.encoder import Encoder
from bridge.decoder import Decoder
from bridge.llm_bridge import LLMBridge

from persistence.snapshot import Snapshot
from persistence.recovery import Recovery

logger = logging.getLogger(__name__)


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
            }
            self.meta = MetaPlasticityController(meta_config)

        # 做梦引擎（懒加载，首次调用 get_dream_engine() 时创建）
        self.dream_engine = None

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
        _effective_lr = self.config.get('learning_rate', 0.01)
        _meta_temp = None
        _meta_orth = None
        _meta_cw = None
        if self.meta is not None:
            adjusted = self.meta.get_adjusted_params(
                base_lr=_effective_lr,
                base_orth=self.attractor.orth_weight,
                base_temp=self.attractor.temperature,
                base_cw=self.attractor.complexity_weight,
            )
            _meta_temp = adjusted['temperature']
            _meta_orth = adjusted['orth_weight']
            _meta_cw = adjusted['complexity_weight']
            _effective_lr = adjusted['learning_rate']

        # 2. FEP推断（E-P2-5: 通过 temperature_override 传递元调整后的温度）
        precision = self.purpose.get_precision()
        activation = self.attractor.infer(
            sensory_input.vector, precision,
            num_steps=self.config.get('num_infer_steps', 10),
            temperature_override=_meta_temp,
        )

        # 3. FEP学习（E-P2-5: 通过 override 参数传递元调整后的权重）
        self.attractor.learn(
            activation, sensory_input.vector, _effective_lr,
            orth_weight_override=_meta_orth,
            complexity_weight_override=_meta_cw,
        )

        # 3.5 在线熵管理
        # FEP学习之后、调整目的之前，根据当前激活熵主动干预：
        # 熵过高（混沌饱和）则增强正交化压力驱散相似表示；
        # 熵过低（僵化）则放松正交化压力允许更多激活。
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
                activation, sensory_input.vector, _effective_lr * 0.3,
                orth_weight_override=self.attractor.orth_weight * 1.5,
            )
            logger.info(f"在线熵管理: 熵={entropy:.3f}(ratio={entropy_ratio:.2f}) 过高，增强正交化")
        elif entropy_ratio < entropy_low_threshold:
            # 熵过低：系统僵化，降低正交化压力允许更多激活
            # E-P2-5: 通过 orth_weight_override 传递临时权重，不修改实例属性
            self.attractor.learn(
                activation, sensory_input.vector, _effective_lr * 0.2,
                orth_weight_override=self.attractor.orth_weight * 0.7,
            )
            logger.info(f"在线熵管理: 熵={entropy:.3f}(ratio={entropy_ratio:.2f}) 过低，放松正交化")

        # 4. 调整目的
        self.purpose.adjust(activation.surprise, activation)

        # --- 元可塑性：收集信号并更新 ---
        if self.meta is not None:
            j_norm = float(torch.norm(self.attractor.J, p='fro').item())
            collapse_occurred = self.purpose.flipped  # 元目的翻转视为坍缩信号
            coherence = self.purpose.coherence
            self.meta.update(activation.surprise, coherence, collapse_occurred, j_norm)

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
        # 优先用 384 维 raw 向量查询（高精度）；fallback 用 64 维投影向量
        # （向后兼容旧快照中 64 维条目）；无 raw 向量时退化为投影向量查询
        episodic_texts = None
        if semantic_vector is not None:
            episodic_query = (raw_semantic_vector
                              if raw_semantic_vector is not None
                              else semantic_vector)
            entries = self.memory.recall_episodic(
                episodic_query, top_k=3, fallback_query=semantic_vector)
            if entries:
                episodic_texts = [e.text for e in entries]

        # 6. 解码context（传入检索到的记忆和情景文本，使长期记忆+语义文本参与输出）
        # A-P1-2: 传入 purpose.coherence，使解码器能输出"关注方向"解读
        memory_context = self.decoder.decode(
            activation, recalled_memory=recalled,
            episodic_texts=episodic_texts,
            coherence=self.purpose.coherence)

        # 6.5 情景记忆存储（当前轮文本存入缓冲区，供后续检索）
        # 优先存 384 维 raw 向量；无 raw 向量时退化为投影向量（向后兼容）
        if semantic_vector is not None:
            self.memory.store_episodic(
                text, semantic_vector, activation.surprise, self.turn_count,
                raw_semantic_vector=raw_semantic_vector)

        # 更新状态
        self.last_activation = activation
        self.turn_count += 1

        logger.debug(
            f"第{self.turn_count}轮: "
            f"熵={activation.entropy:.3f}, "
            f"惊讶度={activation.surprise:.3f}"
        )

        # 7. 自动快照（G4 修复：按间隔自动保存状态）
        if self.config.get('auto_snapshot', False):
            interval = self.config.get('auto_snapshot_interval', 50)
            if self.turn_count > 0 and self.turn_count % interval == 0:
                snapshot_dir = self.config.get(
                    'snapshot_dir', '~/.lms/snapshots')
                snapshot_path = os.path.join(
                    os.path.expanduser(snapshot_dir),
                    f'snapshot_{self.turn_count}.pt'
                )
                try:
                    self.save_state(snapshot_path)
                    logger.info(f"自动快照已保存: {snapshot_path}")
                except Exception as e:
                    logger.warning(f"自动快照失败: {e}")

        # 8. 返回context
        return memory_context

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
            episodic_texts = None
            if hasattr(self.embedder, 'embed_text'):
                sem_vec = self.embedder.embed_text(user_input)
                raw_vec = None
                if hasattr(self.embedder, 'embed_text_raw'):
                    raw_vec = self.embedder.embed_text_raw(user_input)
                episodic_query = raw_vec if raw_vec is not None else sem_vec
                entries = self.memory.recall_episodic(
                    episodic_query, top_k=3, fallback_query=sem_vec)
                if entries:
                    episodic_texts = [e.text for e in entries]
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

    def save_state(self, path: str) -> None:
        """保存当前状态到快照文件。

        保存吸引子景观（J矩阵、bias、sigma）、目的层状态（precision、history、
        coherence、encounter_count）、记忆潜变量（short/long_term_latent）
        和分词器词表。

        参数:
            path: 快照文件路径
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

        self.snapshot.save(path, landscape, purpose_dict,
                           memory_state=memory_state,
                           tokenizer_state=tokenizer_state,
                           meta_state=meta_state)
        logger.info(f"状态已保存到 {path}")

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

        if full_cycle:
            result = dream_engine.dream_cycle(max_steps=n_steps)
        else:
            result = dream_engine.dream_mvp(n_steps=n_steps)

        # 做梦后自动保存快照（完整状态，含 tokenizer）
        snapshot_saved = False
        try:
            snapshot_dir = self.config.get(
                'snapshot_dir', os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    '..', 'snapshots'))
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_path = os.path.join(snapshot_dir, 'latest.pt')
            self.save_state(snapshot_path)
            snapshot_saved = True
            logger.info(f"做梦后快照已保存: {snapshot_path}")
        except Exception as e:
            logger.warning(f"做梦后快照保存失败（不影响做梦结果）: {e}")

        result['snapshot_saved'] = snapshot_saved
        return result

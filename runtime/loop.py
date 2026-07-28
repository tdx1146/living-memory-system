"""在线学习环（主循环）

活体记忆系统的主循环，对每轮对话执行完整的记忆循环：
编码输入 -> 推断 -> 学习 -> 调整目的 -> 巩固记忆 -> 检索记忆 -> 解码context -> 返回

其中"检索记忆"步骤（S1 修复）是关键：长时记忆通过 recall() 进入解码路径，
不再"只写不读"。

遵循架构文档第四节的数据流定义。
"""

import os
import logging
from typing import Optional

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
                  max_history, habituation_rate: 目的层参数
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

        # 初始化核心组件（允许外部注入自定义实现）
        # 注入时不覆盖其内部参数；自建时用 config 驱动全部 FEP 参数
        self.attractor = config.get('attractor') or AttractorNetwork(
            num_nodes, input_dim,
            seed=seed,
            temperature=temperature,
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
            coherence_threshold=config.get('coherence_threshold', 0.3),
            min_history_length=config.get('min_history_length', 5),
            meta_window=config.get('meta_window', 10),
            max_history=config.get('max_history', 100),
            habituation_rate=config.get('habituation_rate', 0.05),
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

        # 2. FEP推断
        precision = self.purpose.get_precision()
        activation = self.attractor.infer(
            sensory_input.vector, precision,
            num_steps=self.config.get('num_infer_steps', 10)
        )

        # 3. FEP学习
        learning_rate = self.config.get('learning_rate', 0.01)
        self.attractor.learn(activation, sensory_input.vector, learning_rate)

        # 4. 调整目的
        self.purpose.adjust(activation.surprise, activation)

        # 5. 记忆更新与巩固
        self.memory.update(activation, activation.surprise)
        if self.turn_count > 0 and \
           self.turn_count % self.consolidation_interval == 0:
            self.memory.consolidate()
            logger.debug(f"第{self.turn_count}轮：执行记忆巩固")

        # 5.5 记忆检索（S1 修复：用当前激活态作为线索，检索长时记忆）
        # recall() 返回 [num_nodes] 维向量，与 activation.state 同维。
        # 这是"记忆只写不读"缺陷的核心修复点：长时记忆通过此步进入输出路径。
        recalled = self.memory.recall(activation.state)

        # 6. 解码context（传入检索到的记忆，使长期记忆参与输出）
        memory_context = self.decoder.decode(activation, recalled_memory=recalled)

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

        # 获取记忆context（同样接入长时记忆检索，保持读取路径一致）
        if self.last_activation is not None:
            recalled = self.memory.recall(self.last_activation.state)
            memory_context = self.decoder.decode(
                self.last_activation, recalled_memory=recalled)
        else:
            memory_context = "[无记忆]"

        # 查询LLM
        response = self.bridge.query(user_input, memory_context)

        # 将LLM输出也送入记忆系统
        self.process_turn(user_input, response)

        return response

    def save_state(self, path: str) -> None:
        """保存当前状态到快照文件。

        保存吸引子景观（J矩阵、bias、sigma）和目的层状态（precision、history）。

        参数:
            path: 快照文件路径
        """
        # 获取吸引子景观
        landscape = self.attractor.get_landscape()

        # 获取目的层状态并转为字典
        purpose = self.purpose.get_purpose()
        purpose_dict = {
            'precision': purpose.precision,
            'history': purpose.history,
            'coherence': purpose.coherence,
        }

        self.snapshot.save(path, landscape, purpose_dict)
        logger.info(f"状态已保存到 {path}")

    def load_state(self, path: str) -> None:
        """从快照文件恢复状态。

        参数:
            path: 快照文件路径

        异常:
            RuntimeError: 恢复失败时抛出
        """
        success = self.recovery.recover(path, self.attractor, self.purpose)
        if not success:
            raise RuntimeError(f"无法从 {path} 恢复状态")
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

        purpose = self.purpose.get_purpose()
        status['purpose_coherence'] = purpose.coherence
        status['precision_mean'] = float(purpose.precision.mean())
        status['precision_std'] = float(purpose.precision.std())

        return status

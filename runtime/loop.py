"""在线学习环（主循环）

活体记忆系统的主循环，对每轮对话执行完整的记忆循环：
编码输入 -> 推断 -> 学习 -> 调整目的 -> 巩固记忆 -> 解码context -> 返回

遵循架构文档第四节的数据流定义。
"""

import os
import logging
from typing import Optional

from core.types import SensoryInput, Activation, PurposeState
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
    5. 巩固记忆（短时->长时迁移）
    6. 解码context（激活态 -> LLM可理解的context）
    7. 返回context

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
        """
        self.config = config

        # 核心参数
        num_nodes = config.get('num_nodes', 256)
        input_dim = config.get('input_dim', 64)
        consolidation_interval = config.get('consolidation_interval', 5)

        # 初始化核心组件（允许外部注入自定义实现）
        self.attractor = config.get('attractor') or AttractorNetwork(num_nodes, input_dim)
        self.purpose = config.get('purpose') or PurposeLayer(input_dim)
        self.memory = config.get('memory') or MemoryManager(num_nodes)
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
        5. 巩固记忆（定期短时->长时迁移）
        6. 解码context（激活态 -> context文本）
        7. 返回context

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
            num_steps=self.config.get('infer_steps', 10)
        )

        # 3. FEP学习
        learning_rate = self.config.get('learning_rate', 0.01)
        self.attractor.learn(activation, sensory_input.vector, learning_rate)

        # 4. 调整目的
        self.purpose.adjust(activation.surprise, activation)

        # 5. 记忆更新与巩固
        self.memory.update(activation)
        if self.turn_count > 0 and \
           self.turn_count % self.consolidation_interval == 0:
            self.memory.consolidate()
            logger.debug(f"第{self.turn_count}轮：执行记忆巩固")

        # 6. 解码context
        memory_context = self.decoder.decode(activation)

        # 更新状态
        self.last_activation = activation
        self.turn_count += 1

        logger.debug(
            f"第{self.turn_count}轮: "
            f"熵={activation.entropy:.3f}, "
            f"惊讶度={activation.surprise:.3f}"
        )

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

        # 获取记忆context
        if self.last_activation is not None:
            memory_context = self.decoder.decode(self.last_activation)
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

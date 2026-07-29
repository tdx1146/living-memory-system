"""吸引子景观快照

保存和加载吸引子网络状态（J矩阵、bias、sigma）和目的层状态（precision、history）。
格式：PyTorch的.pt文件（torch.save/load）。

遵循架构文档 5.5 节的接口定义。
保存内容：J矩阵、bias、sigma状态、precision向量、目的层历史。
可选保存：memory潜变量、tokenizer词表（N1+N2 修复）、meta元可塑性状态（0.3.0）。
包含版本号和时间戳。

版本历史:
    - 0.2.0: 增加 memory 和 tokenizer 可选字段（N1+N2 修复）。
    - 0.3.0: 增加 meta 元可塑性可选字段（向后兼容：旧快照无此字段时跳过）。
"""

import time
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 快照格式版本号（N1+N2: 0.2.0 增加 memory 和 tokenizer 可选字段；
#                  0.3.0 增加 meta 元可塑性可选字段）
SNAPSHOT_VERSION = "0.3.0"


class Snapshot:
    """吸引子景观快照管理器。

    保存"火种"（J矩阵 + precision状态），恢复时"重新点燃"。

    快照格式（v0.3.0）：
        {
            'version': str,           # 快照版本号
            'timestamp': float,       # 保存时间戳
            'attractor': dict,        # 吸引子景观状态
            'purpose': dict,          # 目的层状态
            'memory': dict,           # 记忆潜变量（可选，N1）
            'tokenizer': dict,        # 分词器词表（可选，N2）
            'meta': dict,             # 元可塑性状态（可选，0.3.0）
        }

    attractor字典包含：
        - J: 耦合矩阵 [num_nodes, num_nodes]
        - bias: 偏置向量 [num_nodes]
        - sigma: 当前激活状态 [num_nodes]
        - num_nodes: 节点数
        - input_dim: 输入维度

    purpose字典包含：
        - precision: precision向量 [input_dim]
        - history: precision历史列表
        - coherence: 一致性值
        - encounter_count: 习惯化计数器 [input_dim]（N3，可选）

    memory字典包含（N1）：
        - short_term_latent: 短时潜变量 [num_nodes]
        - long_term_latent: 长时潜变量 [num_nodes]
        - num_nodes: 节点数

    tokenizer字典包含（N2）：
        - vocab: 词表字典 {token_str: token_id}

    meta字典包含（0.3.0，可选）：
        - 元可塑性层的自适应状态（如各学习规则参数的倍率、surprise历史等）。
          具体结构由 core.meta 层定义；persistence 层仅做透传，不解释其内容。
          旧版快照（v0.2.0）无此字段，加载时优雅跳过（向后兼容）。

    接口设计说明（G5）:
        save()/load() 采用 **dict 接口** 而非直接接受 core 对象。
        这是有意的架构决策，目的是彻底解耦 persistence 与 core 层
        （架构约束：依赖图为无环DAG core <- persistence <- runtime，
        persistence 只负责序列化/反序列化，不做业务逻辑）。

        - 使用 dict 接口：persistence 无需在运行时导入 core 的具体类
          （如 AttractorNetwork / PurposeLayer），避免循环依赖与层间耦合。
          符合"高内聚 + 低耦合"原则。
        - 调用方负责对象 <-> dict 的转换：
            * attractor_landscape 由调用方通过
              AttractorNetwork.get_landscape() 获取后传入；
            * purpose_state 由调用方构造为包含 precision / history /
              coherence 键的 dict（可由 PurposeLayer.get_purpose()
              返回的 PurposeState 拆解得到）。
        - 架构文档 5.5 节虽描述为"接受对象"的接口，但 dict 实现是更优的
          解耦设计，本实现以 dict 为准，并在 persistence 层内部保持一致。
    """

    def save(self, path: str, attractor_landscape: dict,
             purpose_state: dict,
             memory_state: Optional[dict] = None,
             tokenizer_state: Optional[dict] = None,
             meta_state: Optional[dict] = None) -> None:
        """保存吸引子景观和目的层状态到文件。

        参数:
            path: 保存路径（.pt文件）
            attractor_landscape: 吸引子景观状态字典，由调用方通过
                AttractorNetwork.get_landscape() 获取后传入。应包含键：
                J / bias / sigma（torch.Tensor）与 num_nodes / input_dim（int）。
            purpose_state: 目的层状态字典，由调用方构造。应包含键：
                precision（torch.Tensor）/ history（list[torch.Tensor]）/
                coherence（float）。可由 PurposeLayer.get_purpose() 返回的
                PurposeState 拆解得到。
            memory_state: 记忆潜变量状态字典（可选，N1）。由 MemoryManager
                的 get_state() 获取后传入。包含 short_term_latent /
                long_term_latent / num_nodes 键。为 None 时不保存此字段。
            tokenizer_state: 分词器词表字典（可选，N2）。由 SimpleTokenizer
                的 get_vocab() 获取后传入。为 None 时不保存此字段。
            meta_state: 元可塑性状态字典（可选，0.3.0）。由 core.meta 层的
                get_state() 获取后传入。persistence 层仅透传，不解释其内容。
                为 None 时不保存此字段。
        """
        data = {
            'version': SNAPSHOT_VERSION,
            'timestamp': time.time(),
            'attractor': attractor_landscape,
            'purpose': purpose_state,
        }

        # N1: 可选保存 memory 潜变量
        if memory_state is not None:
            data['memory'] = memory_state

        # N2: 可选保存 tokenizer 词表
        if tokenizer_state is not None:
            data['tokenizer'] = tokenizer_state

        # 0.3.0: 可选保存 meta 元可塑性状态
        if meta_state is not None:
            data['meta'] = meta_state

        # 确保目录存在
        import os
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        torch.save(data, path)
        logger.info(f"快照已保存到 {path}")

    def load(self, path: str) -> tuple[dict, dict]:
        """从文件加载吸引子景观和目的层状态。

        仅返回必需的 attractor / purpose 两个状态；可选字段
        （memory / tokenizer / meta）不在返回值中，需通过 load_raw()
        获取（与 N1/N2 既定模式一致；保持本方法二元组返回以兼容
        recovery.py 及现有测试的解包调用）。

        向后兼容：旧版快照（v0.2.0，无 meta 字段）可正常加载——
        本方法仅读取 attractor / purpose，不依赖任何可选字段。

        参数:
            path: 快照文件路径（.pt文件）

        返回:
            元组 (attractor_landscape, purpose_state)

        异常:
            FileNotFoundError: 文件不存在
            KeyError: 文件格式不正确
        """
        data = _torch_load(path)

        if 'attractor' not in data or 'purpose' not in data:
            raise KeyError("快照文件格式不正确：缺少attractor或purpose字段")

        version = data.get('version', 'unknown')
        logger.info(
            f"从 {path} 加载快照 "
            f"(版本: {version}, 时间: {data.get('timestamp', 'unknown')})"
        )

        # 可选字段（memory/tokenizer/meta）不在此返回，由调用方通过
        # load_raw() 按需取用；旧快照无 meta 字段时自动跳过（向后兼容）。
        return data['attractor'], data['purpose']

    def load_raw(self, path: str) -> dict:
        """加载完整快照数据（包括可选字段 memory/tokenizer/meta）。

        供 runtime 层获取 persistence 层未直接恢复的可选状态
        （如 tokenizer 词表——N2: tokenizer 在 runtime 层直接处理；
        meta 元可塑性状态——0.3.0: 由 core.meta 层通过
        raw_data.get('meta') 取用，旧快照无此字段时返回 None）。

        参数:
            path: 快照文件路径

        返回:
            完整的快照数据字典
        """
        return _torch_load(path)

    def get_metadata(self, path: str) -> dict:
        """获取快照元数据（不加载完整状态）。

        参数:
            path: 快照文件路径

        返回:
            元数据字典，包含version和timestamp
        """
        data = _torch_load(path)
        return {
            'version': data.get('version', 'unknown'),
            'timestamp': data.get('timestamp', 0),
        }


def _torch_load(path: str) -> dict:
    """兼容不同PyTorch版本的torch.load封装。

    PyTorch 2.6+ 默认 weights_only=True，需要显式设置为False
    以加载包含Python对象的字典。

    参数:
        path: 文件路径

    返回:
        加载的数据字典
    """
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        # 旧版本PyTorch不支持weights_only参数
        return torch.load(path)

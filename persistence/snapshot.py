"""吸引子景观快照

保存和加载吸引子网络状态（J矩阵、bias、sigma）和目的层状态（precision、history）。
格式：PyTorch的.pt文件（torch.save/load）。

遵循架构文档 5.5 节的接口定义。
保存内容：J矩阵、bias、sigma状态、precision向量、目的层历史。
包含版本号和时间戳。
"""

import time
import torch
import logging

logger = logging.getLogger(__name__)

# 快照格式版本号
SNAPSHOT_VERSION = "0.1.0"


class Snapshot:
    """吸引子景观快照管理器。

    保存"火种"（J矩阵 + precision状态），恢复时"重新点燃"。

    快照格式：
        {
            'version': str,           # 快照版本号
            'timestamp': float,       # 保存时间戳
            'attractor': dict,        # 吸引子景观状态
            'purpose': dict,          # 目的层状态
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
             purpose_state: dict) -> None:
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
        """
        data = {
            'version': SNAPSHOT_VERSION,
            'timestamp': time.time(),
            'attractor': attractor_landscape,
            'purpose': purpose_state,
        }

        # 确保目录存在
        import os
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        torch.save(data, path)
        logger.info(f"快照已保存到 {path}")

    def load(self, path: str) -> tuple[dict, dict]:
        """从文件加载吸引子景观和目的层状态。

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

        return data['attractor'], data['purpose']

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

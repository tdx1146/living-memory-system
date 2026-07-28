"""断点续传

从快照恢复系统状态，验证快照完整性，处理版本兼容性。

遵循架构文档 5.5 节的接口定义。
兼容不同实现版本的core层（如PurposeLayer有无set_purpose方法）。

接口设计说明（G5）:
    本模块通过 Protocol 接口操作 core 层对象，不直接依赖 core 的具体类，
    以实现 persistence 与 core 的彻底解耦
    （架构约束：依赖图为无环DAG core <- persistence <- runtime）。
    recover() / _restore_attractor() / _restore_purpose() 接受的是 Protocol
    类型，而非具体的 AttractorNetwork / PurposeLayer。

    快照数据以 dict 形式在 persistence 层流转（而非 core 对象），
    persistence 只负责序列化/反序列化，不做业务逻辑（高内聚 + 低耦合）：
    - attractor_landscape: 由调用方通过 AttractorNetwork.get_landscape()
      获取后传入，包含 J / bias / sigma / num_nodes / input_dim 键。
    - purpose_state: 由调用方构造为包含 precision / history / coherence
      键的 dict。
"""

import os
import logging
import torch
from typing import Protocol, runtime_checkable

from persistence.snapshot import Snapshot, SNAPSHOT_VERSION, _torch_load
from core.types import PurposeState

logger = logging.getLogger(__name__)


@runtime_checkable
class RestorableAttractor(Protocol):
    """可恢复的吸引子网络接口。

    定义 persistence 层恢复吸引子状态所需的最小接口契约。
    core 层的 AttractorNetwork 满足此 Protocol（已实现 set_landscape）。
    """

    def set_landscape(self, landscape: dict) -> None:
        """从景观字典恢复吸引子状态。"""
        ...

    J: torch.Tensor
    bias: torch.Tensor
    sigma: torch.Tensor
    num_nodes: int
    input_dim: int


@runtime_checkable
class RestorablePurpose(Protocol):
    """可恢复的目的层接口。

    定义 persistence 层恢复目的层状态所需的最小接口契约。

    注意: set_purpose 为 Wave 1A 计划新增的接口。在 core 层 PurposeLayer
    添加 set_purpose() 之前，_restore_purpose() 会通过 hasattr 检查回退到
    直接设置属性的方式，保证向后兼容。一旦 Wave 1A 完成，该分支将变为
    可用路径。
    """

    def set_purpose(self, state: PurposeState) -> None:
        """从 PurposeState 恢复目的层状态。"""
        ...

    sensory_precision: torch.Tensor
    history: list
    coherence: float
    attention: torch.Tensor


class Recovery:
    """断点续传管理器。

    提供快照验证和状态恢复功能。
    处理版本兼容性，确保快照可以安全地加载。

    属性:
        snapshot: Snapshot实例
    """

    def __init__(self):
        """初始化恢复管理器。"""
        self.snapshot = Snapshot()

    def recover(self, path: str, attractor: RestorableAttractor,
                purpose: RestorablePurpose) -> bool:
        """从快照恢复吸引子和目的层状态。

        兼容不同core实现：
        - AttractorNetwork: 优先调用set_landscape，否则直接设置属性
        - PurposeLayer: 优先调用set_purpose，否则直接设置属性

        参数:
            path: 快照文件路径
            attractor: 满足 RestorableAttractor 协议的对象
                （如 core 层的 AttractorNetwork 实例）
            purpose: 满足 RestorablePurpose 协议的对象
                （如 core 层的 PurposeLayer 实例）

        返回:
            True表示恢复成功，False表示恢复失败

        说明:
            快照内部以 dict 形式存储状态（见 Snapshot），本方法不直接操作
            core 对象的业务方法，仅做反序列化与状态回填，实现 persistence
            与 core 的解耦。
        """
        try:
            if not self.validate(path):
                logger.error(f"快照验证失败: {path}")
                return False

            # 加载快照数据
            landscape, purpose_state = self.snapshot.load(path)

            # 恢复吸引子网络
            self._restore_attractor(attractor, landscape)

            # 恢复目的层
            self._restore_purpose(purpose, purpose_state)

            logger.info(f"从 {path} 恢复成功")
            return True

        except Exception as e:
            logger.error(f"恢复失败: {e}")
            return False

    def _restore_attractor(self, attractor: RestorableAttractor,
                           landscape: dict) -> None:
        """恢复吸引子网络状态。

        优先调用set_landscape方法，否则直接设置属性。

        参数:
            attractor: 满足 RestorableAttractor 协议的对象
            landscape: 景观状态字典（由 AttractorNetwork.get_landscape()
                获取，包含 J / bias / sigma / num_nodes / input_dim 键）
        """
        if hasattr(attractor, 'set_landscape'):
            attractor.set_landscape(landscape)
            logger.info("吸引子网络状态已恢复（via set_landscape）")
        else:
            # 直接设置属性（兼容不同实现）
            if 'J' in landscape:
                attractor.J = landscape['J'].clone()
            if 'bias' in landscape:
                attractor.bias = landscape['bias'].clone()
            if 'sigma' in landscape:
                attractor.sigma = landscape['sigma'].clone()
            if 'num_nodes' in landscape:
                attractor.num_nodes = landscape['num_nodes']
            if 'input_dim' in landscape:
                attractor.input_dim = landscape['input_dim']
            logger.info("吸引子网络状态已恢复（via direct attributes）")

    def _restore_purpose(self, purpose: RestorablePurpose,
                         purpose_state: dict) -> None:
        """恢复目的层状态。

        优先调用set_purpose方法，否则直接设置属性。
        兼容没有set_purpose方法的PurposeLayer实现。

        参数:
            purpose: 满足 RestorablePurpose 协议的对象
            purpose_state: 目的层状态字典（由调用方构造，包含
                precision / history / coherence 键）
        """
        precision = purpose_state.get('precision')
        history = purpose_state.get('history', [])
        coherence = purpose_state.get('coherence', 1.0)

        if hasattr(purpose, 'set_purpose'):
            # 有set_purpose方法的实现
            purpose_state_obj = PurposeState(
                precision=precision,
                history=history,
                coherence=coherence,
            )
            purpose.set_purpose(purpose_state_obj)
            logger.info("目的层状态已恢复（via set_purpose）")
        else:
            # 直接设置属性（兼容没有set_purpose的实现）
            if precision is not None and hasattr(purpose, 'sensory_precision'):
                purpose.sensory_precision = precision.clone()
            if hasattr(purpose, 'history'):
                purpose.history = [h.clone() if isinstance(h, torch.Tensor) else h
                                   for h in history]
            if hasattr(purpose, 'coherence'):
                purpose.coherence = coherence
            # 重新计算attention
            if hasattr(purpose, 'sensory_precision') and hasattr(purpose, 'attention'):
                import torch.nn.functional as F
                purpose.attention = F.softmax(purpose.sensory_precision, dim=0)
            logger.info("目的层状态已恢复（via direct attributes）")

    def validate(self, path: str) -> bool:
        """验证快照完整性。

        检查项：
        1. 文件存在
        2. 文件可加载
        3. 包含必需字段（version, attractor, purpose）
        4. 版本兼容

        参数:
            path: 快照文件路径

        返回:
            True表示快照有效，False表示无效
        """
        try:
            # 检查文件存在
            if not os.path.exists(path):
                logger.error(f"快照文件不存在: {path}")
                return False

            # 尝试加载
            data = _torch_load(path)

            # 检查必需字段
            required_fields = ['version', 'attractor', 'purpose']
            for field in required_fields:
                if field not in data:
                    logger.error(f"快照缺少必需字段: {field}")
                    return False

            # 版本兼容性检查
            version = data['version']
            if not self._is_version_compatible(version):
                logger.error(f"快照版本不兼容: {version} (当前版本: {SNAPSHOT_VERSION})")
                return False

            # 检查attractor字段
            attractor = data['attractor']
            attractor_required = ['J', 'bias', 'sigma']
            for field in attractor_required:
                if field not in attractor:
                    logger.error(f"attractor缺少必需字段: {field}")
                    return False

            # 检查purpose字段
            purpose = data['purpose']
            if 'precision' not in purpose:
                logger.error("purpose缺少必需字段: precision")
                return False

            return True

        except Exception as e:
            logger.error(f"快照验证异常: {e}")
            return False

    def _is_version_compatible(self, version: str) -> bool:
        """检查版本兼容性。

        当前策略：主版本号必须匹配，次版本号向后兼容。
        未来版本可以在这里添加迁移逻辑。

        参数:
            version: 快照版本号字符串

        返回:
            True表示兼容
        """
        if version == SNAPSHOT_VERSION:
            return True

        # 解析版本号
        try:
            major, minor, _ = version.split('.')
            current_major, current_minor, _ = SNAPSHOT_VERSION.split('.')
            # 主版本号必须匹配
            if major != current_major:
                return False
            # 次版本号向后兼容
            return True
        except (ValueError, AttributeError):
            return False

"""断点续传

从快照恢复系统状态，验证快照完整性，处理版本兼容性。

遵循架构文档 5.5 节的接口定义。
兼容不同实现版本的core层（如PurposeLayer有无set_purpose方法）。

接口设计说明（G5）:
    本模块通过 Protocol 接口操作 core 层对象，不直接依赖 core 的具体类，
    以实现 persistence 与 core 的彻底解耦
    （架构约束：依赖图为无环DAG core <- persistence <- runtime）。
    recover() / _restore_attractor() / _restore_purpose() / _restore_memory()
    接受的是 Protocol 类型，而非具体的 AttractorNetwork / PurposeLayer /
    MemoryManager。

    快照数据以 dict 形式在 persistence 层流转（而非 core 对象），
    persistence 只负责序列化/反序列化，不做业务逻辑（高内聚 + 低耦合）：
    - attractor_landscape: 由调用方通过 AttractorNetwork.get_landscape()
      获取后传入，包含 J / bias / sigma / num_nodes / input_dim 键。
    - purpose_state: 由调用方构造为包含 precision / history / coherence
      键的 dict。
    - memory_state: 由调用方通过 MemoryManager.get_state() 获取后传入，
      包含 short_term_latent / long_term_latent / num_nodes 键（N1）。
"""

import os
import logging
import torch
from typing import Protocol, runtime_checkable, Optional

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


@runtime_checkable
class RestorableMemory(Protocol):
    """可恢复的记忆管理器接口（N1）。

    定义 persistence 层恢复记忆潜变量所需的最小接口契约。
    core 层的 MemoryManager 满足此 Protocol（已实现 set_state）。
    """

    def set_state(self, state: dict) -> None:
        """从状态字典恢复记忆潜变量。"""
        ...


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
                purpose: RestorablePurpose,
                memory: Optional[RestorableMemory] = None) -> bool:
        """从快照恢复吸引子和目的层状态。

        兼容不同core实现：
        - AttractorNetwork: 优先调用set_landscape，否则直接设置属性
        - PurposeLayer: 优先调用set_purpose，否则直接设置属性
        - MemoryManager: 优先调用set_state（N1），否则跳过

        参数:
            path: 快照文件路径
            attractor: 满足 RestorableAttractor 协议的对象
                （如 core 层的 AttractorNetwork 实例）
            purpose: 满足 RestorablePurpose 协议的对象
                （如 core 层的 PurposeLayer 实例）
            memory: 满足 RestorableMemory 协议的对象（可选，N1）
                （如 core 层的 MemoryManager 实例）。为 None 时跳过
                memory 恢复（向后兼容旧版调用方）。

        返回:
            True表示恢复成功，False表示恢复失败

        说明:
            快照内部以 dict 形式存储状态（见 Snapshot），本方法不直接操作
            core 对象的业务方法，仅做反序列化与状态回填，实现 persistence
            与 core 的解耦。memory 字段为可选——旧版快照无此字段时优雅降级。

            E-P1-6 改进：
            - 单次读取优化：validate/load/load_raw 原先对同一文件读取3次，
              现改为单次 _torch_load 后复用 raw_data。
            - 异常链保留：失败时用 logger.exception() 记录完整堆栈（含根因），
              而非仅记录异常字符串（不再吞掉根因堆栈）。
            - 回滚：恢复前快照当前状态，任一恢复步骤失败时回滚到恢复前状态，
              避免留下半恢复的不一致状态。
        """
        rollback_snapshot = None
        try:
            # 文件存在性检查（避免对不存在的文件触发 noisy 堆栈）
            if not os.path.exists(path):
                logger.error(f"快照文件不存在: {path}")
                return False

            # E-P1-6: 单次读取优化——原先 validate/load/load_raw 对同一文件
            # 读取3次，现改为单次读取后复用 raw_data
            raw_data = _torch_load(path)

            # 验证（复用已读取的 raw_data，避免重复读取文件）
            if not self._validate_data(raw_data):
                logger.error(f"快照验证失败: {path}")
                return False

            # E-P1-6: 恢复前快照当前状态，任一步骤失败时回滚
            # 在所有修改操作之前捕获，确保回滚基准正确
            rollback_snapshot = self._snapshot_current_state(
                attractor, purpose, memory)

            landscape = raw_data['attractor']
            purpose_state = raw_data['purpose']

            # 恢复吸引子网络
            self._restore_attractor(attractor, landscape)

            # 恢复目的层
            self._restore_purpose(purpose, purpose_state)

            # N1: 恢复记忆潜变量（如果提供了 memory 对象且快照包含 memory 字段）
            if memory is not None:
                memory_state = raw_data.get('memory')
                if memory_state is not None:
                    self._restore_memory(memory, memory_state)
                    logger.info("记忆潜变量已恢复")
                else:
                    logger.info("快照不含 memory 字段，跳过记忆恢复（向后兼容）")

            logger.info(f"从 {path} 恢复成功")
            return True

        except Exception as e:
            # E-P1-6: 保留异常链——logger.exception 记录完整堆栈（含根因），
            # 而非仅记录异常字符串，避免吞掉根因堆栈
            logger.exception(f"恢复失败: {e}")
            # E-P1-6: 尝试回滚到恢复前状态，避免留下半恢复的不一致状态
            if rollback_snapshot is not None:
                self._rollback(rollback_snapshot, attractor, purpose, memory)
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
        N3: 同时恢复 encounter_count（如果快照中包含此字段）。

        参数:
            purpose: 满足 RestorablePurpose 协议的对象
            purpose_state: 目的层状态字典（由调用方构造，包含
                precision / history / coherence / encounter_count 键）
        """
        precision = purpose_state.get('precision')
        history = purpose_state.get('history', [])
        coherence = purpose_state.get('coherence', 1.0)
        encounter_count = purpose_state.get('encounter_count')  # N3

        if hasattr(purpose, 'set_purpose'):
            # 有set_purpose方法的实现
            purpose_state_obj = PurposeState(
                precision=precision,
                history=history,
                coherence=coherence,
                encounter_count=encounter_count,
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
            # N3: 恢复 encounter_count
            if encounter_count is not None and hasattr(purpose, 'encounter_count'):
                purpose.encounter_count = encounter_count.clone()
            # 重新计算attention
            if hasattr(purpose, 'sensory_precision') and hasattr(purpose, 'attention'):
                import torch.nn.functional as F
                purpose.attention = F.softmax(purpose.sensory_precision, dim=0)
            logger.info("目的层状态已恢复（via direct attributes）")

    def _restore_memory(self, memory: RestorableMemory,
                        memory_state: dict) -> None:
        """恢复记忆潜变量状态（N1）。

        优先调用set_state方法，否则跳过（向后兼容）。

        参数:
            memory: 满足 RestorableMemory 协议的对象
            memory_state: 记忆状态字典（由 MemoryManager.get_state() 获取，
                包含 short_term_latent / long_term_latent / num_nodes 键）
        """
        if hasattr(memory, 'set_state'):
            memory.set_state(memory_state)
            logger.info("记忆潜变量已恢复（via set_state）")
        else:
            logger.warning("memory 对象不支持 set_state，跳过记忆恢复")

    def _snapshot_current_state(self, attractor: RestorableAttractor,
                                purpose: RestorablePurpose,
                                memory: Optional[RestorableMemory]) -> dict:
        """快照当前状态（用于恢复失败时回滚）。

        对各组件的关键状态做深拷贝（张量 clone、列表重建），避免回滚时
        与已被恢复操作修改的现场共享引用。回滚时直接复用 _restore_attractor
        / _restore_purpose / _restore_memory 将保存的状态写回。

        参数:
            attractor: 满足 RestorableAttractor 协议的对象
            purpose: 满足 RestorablePurpose 协议的对象
            memory: 满足 RestorableMemory 协议的对象（可为 None）

        返回:
            状态快照字典，结构对齐 _restore_* 的入参：
            {'attractor': landscape_dict, 'purpose': purpose_state_dict,
             'memory': memory_state_dict or None}
        """
        snapshot = {}

        # attractor 景观（对齐 set_landscape / get_landscape 的 dict 结构）
        attr_state = {}
        for key in ('J', 'bias', 'sigma'):
            val = getattr(attractor, key, None)
            attr_state[key] = val.clone() if isinstance(val, torch.Tensor) else val
        for key in ('num_nodes', 'input_dim'):
            attr_state[key] = getattr(attractor, key, None)
        snapshot['attractor'] = attr_state

        # purpose 状态（对齐 _restore_purpose 的 purpose_state 字典结构）
        precision = getattr(purpose, 'sensory_precision', None)
        purp_state = {
            'precision': precision.clone()
            if isinstance(precision, torch.Tensor) else precision,
            'history': [h.clone() if isinstance(h, torch.Tensor) else h
                        for h in getattr(purpose, 'history', [])],
            'coherence': getattr(purpose, 'coherence', None),
        }
        # N3: encounter_count（可选字段）
        encounter_count = getattr(purpose, 'encounter_count', None)
        if encounter_count is not None:
            purp_state['encounter_count'] = (
                encounter_count.clone()
                if isinstance(encounter_count, torch.Tensor)
                else encounter_count)
        snapshot['purpose'] = purp_state

        # memory 状态（可选）：通过 get_state 获取后深拷贝
        snapshot['memory'] = None
        if memory is not None:
            get_state = getattr(memory, 'get_state', None)
            if callable(get_state):
                try:
                    snapshot['memory'] = _clone_state_dict(get_state())
                except Exception as e:
                    logger.warning(
                        f"快照 memory 状态失败，回滚将不覆盖 memory: {e}")
                    snapshot['memory'] = None

        return snapshot

    def _rollback(self, snapshot: dict, attractor: RestorableAttractor,
                  purpose: RestorablePurpose,
                  memory: Optional[RestorableMemory]) -> None:
        """回滚到恢复前状态（尽力而为，失败仅记录日志）。

        复用 _restore_attractor / _restore_purpose / _restore_memory 将
        _snapshot_current_state 保存的状态写回，确保各组件恢复到恢复操作
        之前的取值。回滚本身的异常不会向上抛出，仅记录日志（避免掩盖
        原始恢复失败的原因）。

        参数:
            snapshot: _snapshot_current_state 返回的状态快照字典
            attractor: 满足 RestorableAttractor 协议的对象
            purpose: 满足 RestorablePurpose 协议的对象
            memory: 满足 RestorableMemory 协议的对象（可为 None）
        """
        try:
            attr_state = snapshot.get('attractor')
            if attr_state:
                self._restore_attractor(attractor, attr_state)

            purp_state = snapshot.get('purpose')
            if purp_state:
                self._restore_purpose(purpose, purp_state)

            mem_state = snapshot.get('memory')
            if mem_state is not None and memory is not None:
                self._restore_memory(memory, mem_state)

            logger.info("已回滚到恢复前状态")
        except Exception as rb_err:
            # 回滚失败不应掩盖原始恢复失败，仅记录
            logger.error(f"回滚失败（状态可能不一致）: {rb_err}")

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

            # 复用 _validate_data 完成字段与版本校验（与 recover 共用同一份逻辑）
            return self._validate_data(data)

        except Exception as e:
            logger.error(f"快照验证异常: {e}")
            return False

    def _validate_data(self, data: dict) -> bool:
        """验证已加载的快照数据完整性（不重复读取文件）。

        将原 validate 中的字段/版本校验逻辑抽取为独立方法，供 validate
        与 recover 复用，避免 recover 中 validate/load/load_raw 三次读取
        同一文件（E-P1-6 单次读取优化）。

        检查项：
        1. 包含必需字段（version, attractor, purpose）
        2. 版本兼容
        3. attractor 子字段完整（J, bias, sigma）
        4. purpose 包含 precision 字段

        参数:
            data: 已通过 _torch_load 加载的快照数据字典

        返回:
            True表示数据有效，False表示无效
        """
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

    def _is_version_compatible(self, version: str) -> bool:
        """检查版本兼容性。

        当前策略：主版本号必须匹配，次版本号向后/向前兼容
        （跨次版本新增字段均为可选字段，加载时优雅降级，故双向往返兼容）。
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
            # E-P1-6: 补 minor 比较——原先 minor 被解析但未参与判定，
            # 现显式比较 minor 并记录版本关系，便于诊断兼容性。
            # 策略：minor 差异视为兼容（新增字段均为可选，向后/向前兼容），
            # 但通过日志区分快照来自旧版还是新版。
            snap_minor = int(minor)
            cur_minor = int(current_minor)
            if snap_minor < cur_minor:
                logger.info(
                    f"加载旧版次版本快照(snapshot minor={snap_minor} < "
                    f"current={cur_minor})，依赖可选字段向后兼容")
            elif snap_minor > cur_minor:
                logger.info(
                    f"加载新版次版本快照(snapshot minor={snap_minor} > "
                    f"current={cur_minor})，未知可选字段将被忽略（向前兼容）")
            return True
        except (ValueError, AttributeError):
            return False


def _clone_state_dict(state):
    """深拷贝状态字典中的张量与列表，用于安全回滚。

    对 dict / list 递归复制结构，对 torch.Tensor 调用 clone 产生独立副本，
    其他不可变类型（int/float/str/None 等）原样返回。

    用途：_snapshot_current_state 捕获 memory.get_state() 返回值时，
    需要与现场解耦——若 set_state 以原地修改方式写入张量，未深拷贝的
    引用会被同步污染，导致回滚基准失效。

    参数:
        state: 待深拷贝的状态对象

    返回:
        深拷贝后的状态对象
    """
    if isinstance(state, dict):
        return {k: _clone_state_dict(v) for k, v in state.items()}
    if isinstance(state, list):
        return [_clone_state_dict(v) for v in state]
    if isinstance(state, torch.Tensor):
        return state.clone()
    return state

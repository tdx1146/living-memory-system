"""
活体记忆系统 - 海马体核心：做梦引擎（Dream Engine）
=====================================================

做梦引擎是活体记忆系统的空闲态计算核心，对应生物大脑睡眠中的离线记忆处理。
它让记忆系统从"冷工具"变为"活体"——即使没有对话输入，系统也在持续进行
记忆运算：记忆重放与巩固、遗忘与突触修剪、吸引子景观缓慢漂移、目的层自主演化。

理论依据（三源汇总）：
  1. 神经科学：NREM 做回放+巩固+突触下调；REM 做创造性整合；按"预测误差"
     而非"奖赏"优先回放；突触稳态假说（SHY）通过全局下调恢复学习容量。
  2. FEP/主动推断：precision=0 时为纯先验采样（做梦的数学本质）；无输入时
     自由能=纯复杂性，驱动模型自洽性优化；默认模式网络（DMN）=空闲态先验主导。
  3. 工程：attractor.infer() 已支持零 precision（生成式回放）；
     memory 缓冲区存储 (激活态, 惊讶度) 元组，可直接用于激活态重放。

七阶段做梦周期：
  阶段1 NREM巩固(40%): 采样回放 + 零精度推断 + 低学习率学习
  阶段2 SHY突触下调(10%): J 矩阵全局 L2 归一化
  阶段3 遗忘修剪(10%): 衰减低 surprise 记忆（forgetting_pruning）
  阶段4 景观漂移(10%): J 矩阵微小随机扰动
  阶段5 目的演化(10%): precision 温和漂移
  阶段6 REM整合(15%): 跨记忆随机关联
  阶段7 快照(5%): 保存状态

参考：docs/DREAM_ENGINE_DESIGN.md
"""

import os
import re
import time
import random
import logging
import tempfile
from typing import Optional, Tuple, List

import torch

# fcntl 伴生锁（T1.2/P0-4 过渡期兜底；与 persistence/snapshot.py 同款模式，
# core 层保持自包含不反向依赖 persistence——阶段 1-B 单写者收口后此落盘
# 将统一走 API，届时可删去本处重复代码）
try:
    import fcntl
    _DREAM_HAVE_FCNTL = True
except ImportError:  # pragma: no cover - 仅非 POSIX 平台
    fcntl = None  # type: ignore[assignment]
    _DREAM_HAVE_FCNTL = False

_DREAM_LOCK_TIMEOUT = 5.0
_DREAM_LOCK_RETRY = 0.05


def _dream_lock_path_for(path: str) -> str:
    """伴生锁文件路径（与 persistence.snapshot._lock_path_for 同规则）。"""
    return path + ".lock"


def _dream_acquire_write_lock(lock_path: str) -> Optional[int]:
    """排他锁（非阻塞 + 重试至超时）；超时返回 None（调用方跳过保存，fail-open）。"""
    if not _DREAM_HAVE_FCNTL:
        return None
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    deadline = time.time() + _DREAM_LOCK_TIMEOUT
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except (BlockingIOError, OSError):
            if time.time() >= deadline:
                os.close(fd)
                return None
            time.sleep(_DREAM_LOCK_RETRY)


def _dream_release_lock(fd: Optional[int]) -> None:
    """释放锁并关闭 fd（尽力而为）。"""
    if fd is None:
        return
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except OSError:
        pass
    os.close(fd)

from core.types import Activation

logger = logging.getLogger("dream_engine")

# 快照格式版本（与 persistence.snapshot.SNAPSHOT_VERSION 保持一致，
# 此处硬编码以避免 core 层反向依赖 persistence 层，保持架构 DAG 无环）
# 0.3.0: 新增可选 meta 字段（元可塑性状态）
_SNAPSHOT_VERSION = "0.5.0"


class DreamEngine:
    """做梦引擎：空闲态计算核心，让记忆系统在无输入时持续运转。

    接收现有组件引用（不创建新实例），在空闲态对记忆系统进行离线运算。
    空闲态参数（如 temperature、orth_weight、complexity_weight）通过
    AttractorNetwork 的 override 接口传递（E-P2-5），不再"保存→修改→
    恢复"实例属性，避免影响在线模式；而真正属于"做梦成果"的状态变更
    （J 矩阵巩固、突触下调、景观漂移、目的演化等）则持久保留。

    属性:
        attractor: 吸引子网络引用（AttractorNetwork）。
        purpose: 目的层引用（PurposeLayer）。
        memory: 记忆管理器引用（MemoryManager）。
        embedder: 嵌入器引用（用于状态报告，可选参与 REM 整合）。
        idle_lr: 空闲态学习率（在线的 1/10，默认 0.001）。
        idle_orth_weight: 空闲态正交化权重（默认 1.0）。
        idle_temperature: 空闲态 Langevin 温度（默认 0.1，略高于在线）。
        idle_num_steps: 空闲态推断步数（默认 20，比在线的 10 更多）。
        surprise_beta: 惊讶度加权采样的 softmax 温度（默认 5.0）。
        collapse_threshold: 吸引子坍缩检测阈值（默认 0.9）。
        phase_weights: 七阶段做梦周期的权重分配。
        step_count: 累计做梦步数。
        activation_history: 近期激活态历史（用于坍缩检测）。
        meta: 元可塑性控制器引用（MetaPlasticityController），可选。
            为 None 时做梦引擎行为与之前完全一致。
    """

    def __init__(self, attractor, purpose, memory, embedder,
                 config: dict, meta=None) -> None:
        """初始化做梦引擎。

        接收现有组件引用（不创建新实例），从 config 读取空闲态参数。

        参数:
            attractor: 吸引子网络实例（AttractorNetwork）。
            purpose: 目的层实例（PurposeLayer）。
            memory: 记忆管理器实例（MemoryManager）。
            embedder: 嵌入器实例（Embedder），用于状态报告与可选的 REM 整合。
            config: 配置字典，支持以下键:
                - idle_learning_rate: 空闲态学习率（默认 0.001，在线的 1/10）
                - idle_orth_weight: 空闲态正交化权重（默认 1.0）
                - idle_temperature: 空闲态 Langevin 温度（默认 0.1）
                - idle_num_steps: 空闲态推断步数（默认 20）
                - consolidation_ratio: 巩固/探索比（默认 0.7）
                - collapse_threshold: 坍缩检测阈值（默认 0.9）
                - max_idle_steps: 最大连续空闲步数（默认 200）
                - snapshot_dir: 快照保存目录（默认 'snapshots'）
                - surprise_beta: 惊讶度采样 softmax 温度（默认 5.0）
                - shy_target_norm: SHY 目标 Frobenius 范数（默认 10.0）
                - drift_scale: 景观漂移扰动幅度（默认 0.001）
                - collapse_window: 坍缩检测窗口大小（默认 5）
                - forget_prune_rate: 遗忘修剪衰减率（默认 0.005）
                - forget_max_age: 情景记忆最大保留年龄（默认 50 轮）
                - purpose_evolve_nudge: 目的演化偏移幅度（默认 0.05）
            meta: 元可塑性控制器（MetaPlasticityController）实例，可选。
                当提供时，空闲态计算将使用元调整后的学习参数（学习率、
                正交化权重、温度、复杂度权重）。为 None 时行为与之前完全一致。
        """
        self.attractor = attractor
        self.purpose = purpose
        self.memory = memory
        self.embedder = embedder
        self.meta = meta
        # 保存 config 引用（T1.1/P0-5：_save_snapshot 读取 session_id 用）
        self.config = config

        # --- 便捷属性（转发到 attractor，便于测试与外部访问） ---
        self.num_nodes = attractor.num_nodes
        self.input_dim = attractor.input_dim

        # --- 空闲态参数 ---
        self.idle_lr = config.get('idle_learning_rate', 0.001)
        self.idle_orth_weight = config.get('idle_orth_weight', 1.0)
        self.idle_temperature = config.get('idle_temperature', 0.1)
        self.idle_num_steps = config.get('idle_num_steps', 20)
        # 零精度向量（存储为属性，便于检查与测试）
        # E-P2-1: 创建在 attractor 所在 device 上
        self.idle_precision = torch.zeros(
            self.input_dim, device=self.attractor.device)

        # --- 平衡与控制参数 ---
        self.consolidation_ratio = config.get('consolidation_ratio', 0.7)
        self.collapse_threshold = config.get('collapse_threshold', 0.9)
        self.max_idle_steps = config.get('max_idle_steps', 200)
        self.snapshot_dir = config.get('snapshot_dir', 'snapshots')

        # --- 采样与各阶段参数 ---
        # surprise_beta：惊讶度加权采样 softmax 温度。
        # 2026-08-10 惊讶度语义拆分：采样输入改为缓冲区内 z-score 标准化 +
        # z 截断（clamp ±3），β 默认从 5.0 降至 1.0（"强但不退化"第二道保险，
        # 防准确性项重尾离群下 exp(βz) 退化为 argmax）。
        self.surprise_beta = config.get('surprise_beta', 1.0)
        self.shy_target_norm = config.get('shy_target_norm', 10.0)
        self.drift_scale = config.get('drift_scale', 0.001)
        self.collapse_window = config.get('collapse_window', 5)
        self.forget_prune_rate = config.get('forget_prune_rate', 0.005)
        self.forget_max_age = config.get('forget_max_age', 50)
        self.purpose_evolve_nudge = config.get('purpose_evolve_nudge', 0.05)

        # --- 七阶段做梦周期权重（体验层 D：nrem 让渡 0.10 给 doubt_review）---
        self.phase_weights = {
            'nrem_consolidation': 0.30,      # 阶段1: NREM 巩固（0.40→0.30）
            'synaptic_homeostasis': 0.10,    # 阶段2: SHY 突触下调
            'forgetting_pruning': 0.10,      # 阶段3: 遗忘修剪
            'landscape_drift': 0.10,         # 阶段4: 景观漂移
            'purpose_evolution': 0.10,       # 阶段5: 目的演化
            'rem_integration': 0.15,         # 阶段6: REM 整合
            'snapshot': 0.05,               # 阶段7: 快照
            'doubt_review': 0.10,            # 阶段8: 怀疑复核（体验层 D，
                                             #   睡眠=整合/复核窗口）
        }

        # --- 运行状态 ---
        self.step_count: int = 0
        self.activation_history: List[torch.Tensor] = []

        # 体验层 D（设计 v1.1 §6.3）：最近一次怀疑复核报告
        # （reviewed/downgraded/rewritten/kept/flagged，进 dream 结果字典）
        self.last_doubt_review: dict = {
            'reviewed': 0, 'downgraded': 0, 'rewritten': 0,
            'kept': 0, 'flagged': 0,
        }
        # gap_registry 注入（loop.get_dream_engine 经 config 传入；缺省 None）
        self.gap_registry = config.get('gap_registry')

        logger.info(
            f"DreamEngine 已初始化 "
            f"(idle_lr={self.idle_lr}, idle_temp={self.idle_temperature}, "
            f"idle_steps={self.idle_num_steps}, "
            f"meta={'启用' if self.meta is not None else '未启用'})"
        )

    # ================================================================== #
    #  公开接口
    # ================================================================== #

    def dream_mvp(self, n_steps: int = 20) -> dict:
        """MVP 版做梦：采样重放 + 低学习率巩固 + 自动快照。

        最简版本，验证核心可行性。流程：
          1. 从 memory 缓冲区按 surprise 加权采样
          2. 用零精度推断做生成式回放
          3. 低学习率学习（巩固 J 矩阵）
          4. 更新 memory 潜变量
          5. 自动快照

        每步执行坍缩检测，发现坍缩立即注入扰动。

        参数:
            n_steps: 做梦步数（默认 20）。

        返回:
            做梦统计字典，包含步数、平均惊讶度、坍缩次数、快照路径等。
            若缓冲区为空，返回 status='no_memories_to_replay'。
        """
        if self.memory.buffer_size() == 0:
            logger.warning("记忆缓冲区为空，无记忆可回放")
            return {
                'status': 'no_memories_to_replay',
                'mode': 'mvp',
                'steps': 0,
                'avg_surprise': 0.0,
                'collapse_count': 0,
                'snapshot_path': None,
            }

        surprises: List[float] = []
        collapse_count = 0
        entropies: List[float] = []
        # 体验层 D：一轮做梦开始重置复核统计（防跨梦膨胀）
        self.last_doubt_review = {'reviewed': 0, 'downgraded': 0,
                                  'rewritten': 0, 'kept': 0, 'flagged': 0}

        # 保存初始 J 矩阵（用于计算 j_change）
        J_initial = self.attractor.J.clone()

        for step in range(n_steps):
            # 1. 按 surprise 加权采样
            sampled = self._sample_by_surprise()
            if sampled is None:
                break
            state, _surprise = sampled

            # 2. 零精度推断（生成式回放）
            activation = self._idle_infer(state)

            # 3. 低学习率 FEP 学习（巩固 J 矩阵）
            self._idle_learn(activation, state)

            # 4. 更新 memory 潜变量
            self.memory.update(activation, activation.surprise)

            # 记录统计
            surprises.append(activation.surprise)
            entropies.append(activation.entropy)

            # 坍缩检测与扰动注入
            if self._check_collapse(activation):
                self._inject_perturbation()
                collapse_count += 1
                logger.debug(f"MVP 做梦第 {step} 步：检测到坍缩，已注入扰动")

            self.step_count += 1

        # 提取层 v1.4（S1-13）：价值重放——PER 概率采样 → 加固 → 聚类
        # （fail-open）→ 代表幸存加固（§2.6 流程；观测进 dream 结果）
        value_replay = self._value_replay()

        # 记忆巩固（短时 -> 长时迁移）
        self.memory.consolidate()

        # 体验层 D（设计 v1.1 §6.3）：MVP 路径也跑一轮轻量怀疑复核
        # （工程决策：生产默认 dream_full_cycle=false 走 MVP，doubt_review
        # 只在七阶段周期内概率执行无法保证复核可达；此处兜底保证默认
        # 做梦路径的 [梦醒] 摘要含复核计数。fail-open，异常不阻断。）
        self._doubt_review()

        # 5. 自动快照
        snapshot_path = self._save_snapshot()

        avg_surprise = sum(surprises) / len(surprises) if surprises else 0.0
        avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
        max_surprise = max(surprises) if surprises else 0.0
        j_change = float(
            torch.linalg.norm(self.attractor.J - J_initial).item())

        logger.info(
            f"MVP 做梦完成: {n_steps} 步, "
            f"平均惊讶度={avg_surprise:.4f}, 坍缩={collapse_count} 次"
        )

        return {
            'status': 'dreamed',
            'mode': 'mvp',
            'steps': n_steps,
            'avg_surprise': float(avg_surprise),
            'max_surprise': float(max_surprise),
            'avg_entropy': float(avg_entropy),
            'collapse_count': collapse_count,
            'j_change': j_change,
            'snapshot_path': snapshot_path,
            'buffer_size': self.memory.buffer_size(),
            'doubt_review': dict(self.last_doubt_review),
            # 提取层 v1.4（S1-13）：价值重放观测（S1-9 汇总进 dream_state.json）
            'value_replay': value_replay,
        }

    def dream_cycle(self, max_steps: int = 100) -> dict:
        """完整做梦周期：七阶段循环。

        按阶段权重随机选择阶段执行，模拟生物睡眠的 NREM/REM 周期：
          阶段1 NREM巩固(40%): 采样回放 + 零精度推断 + 低学习率学习
          阶段2 SHY突触下调(10%): J 矩阵全局 L2 归一化
          阶段3 遗忘修剪(10%): 衰减低 surprise 记忆
          阶段4 景观漂移(10%): J 矩阵微小随机扰动
          阶段5 目的演化(10%): precision 温和漂移
          阶段6 REM整合(15%): 跨记忆随机关联
          阶段7 快照(5%): 保存状态

        坍缩检测在每步后执行，发现坍缩立即注入扰动。

        参数:
            max_steps: 做梦周期最大步数（默认 100）。

        返回:
            做梦统计字典，包含步数、各阶段执行次数、平均惊讶度、
            坍缩次数、快照路径等。若缓冲区为空，返回相应状态。
        """
        if self.memory.buffer_size() == 0:
            logger.warning("记忆缓冲区为空，无记忆可回放")
            return {
                'status': 'no_memories_to_replay',
                'mode': 'full_cycle',
                'steps': 0,
                'phase_counts': {},
                'avg_surprise': 0.0,
                'collapse_count': 0,
                'snapshot_path': None,
            }

        # 限制最大步数
        max_steps = min(max_steps, self.max_idle_steps)

        phase_counts = {phase: 0 for phase in self.phase_weights}
        surprises: List[float] = []
        collapse_count = 0
        # 体验层 D：一轮做梦开始重置复核统计（跨阶段累积，防跨梦膨胀）
        self.last_doubt_review = {'reviewed': 0, 'downgraded': 0,
                                  'rewritten': 0, 'kept': 0, 'flagged': 0}

        # 提取层 v1.4（S1-13）：价值重放——PER 概率采样 → 加固 → 聚类
        # （fail-open）→ 代表幸存加固（每周期一次，§2.6）
        value_replay = self._value_replay()

        for step in range(max_steps):
            # 按权重随机选择阶段
            phase = self._select_phase()
            phase_counts[phase] += 1

            # 执行阶段
            activation = self._execute_phase(phase)

            # 坍缩检测（仅对产生激活态的阶段）
            if activation is not None:
                surprises.append(activation.surprise)
                if self._check_collapse(activation):
                    self._inject_perturbation()
                    collapse_count += 1
                    logger.debug(
                        f"做梦周期第 {step} 步（{phase}）："
                        f"检测到坍缩，已注入扰动")

            self.step_count += 1

        # 周期结束：兜底跑一轮怀疑复核（保证 [梦醒] 摘要含复核计数，
        # 体验层 D 工程决策；fail-open）
        self._doubt_review()

        # 周期结束保存快照
        snapshot_path = self._save_snapshot()

        avg_surprise = sum(surprises) / len(surprises) if surprises else 0.0

        logger.info(
            f"完整做梦周期完成: {max_steps} 步, "
            f"阶段分布={phase_counts}, "
            f"平均惊讶度={avg_surprise:.4f}, 坍缩={collapse_count} 次"
        )

        return {
            'status': 'dreamed',
            'mode': 'full_cycle',
            'steps': max_steps,
            'cycles': max_steps,  # 别名（兼容测试期望）
            'phase_counts': phase_counts,
            'phases': phase_counts,  # 别名（兼容测试期望）
            'avg_surprise': float(avg_surprise),
            'collapse_count': collapse_count,
            'snapshot_path': snapshot_path,
            'buffer_size': self.memory.buffer_size(),
            'doubt_review': dict(self.last_doubt_review),
            # 提取层 v1.4（S1-13）：价值重放观测（S1-9 汇总进 dream_state.json）
            'value_replay': value_replay,
        }

    def get_status(self) -> dict:
        """返回做梦引擎当前状态。

        返回:
            状态字典，包含累计步数、各参数配置、缓冲区大小、
            J 矩阵范数、precision 摘要等信息。
        """
        purpose_state = self.purpose.get_purpose()
        j_norm = float(torch.norm(self.attractor.J, p='fro').item())
        status = {
            'step_count': self.step_count,
            'num_nodes': self.num_nodes,
            'input_dim': self.input_dim,
            'idle_learning_rate': self.idle_lr,
            'idle_orth_weight': self.idle_orth_weight,
            'idle_temperature': self.idle_temperature,
            'idle_num_steps': self.idle_num_steps,
            'consolidation_ratio': self.consolidation_ratio,
            'collapse_threshold': self.collapse_threshold,
            'buffer_size': self.memory.buffer_size(),
            'episodic_buffer_size': self.memory.episodic_size(),
            'j_norm': j_norm,
            'j_frobenius_norm': j_norm,  # 别名
            'precision_mean': float(purpose_state.precision.mean().item()),
            'precision_std': float(purpose_state.precision.std().item()),
            'purpose_coherence': purpose_state.coherence,
            'phase_weights': dict(self.phase_weights),
        }
        # 元可塑性状态
        if self.meta is not None:
            status['meta_enabled'] = True
            status['meta'] = self.meta.get_status()
        else:
            status['meta_enabled'] = False
        return status

    # ================================================================== #
    #  采样
    # ================================================================== #

    def _sample_by_surprise(self, buffer=None) -> Optional[Tuple[torch.Tensor, float]]:
        """按 surprise softmax 加权采样记忆。

        高 surprise（意外/重要）的记忆被更频繁地回放，对应生物海马体
        按"预测误差"而非"奖赏"优先回放的机制。

        权重公式（2026-08-10 惊讶度语义拆分后）：
            z_i = (surprise_i − mean) / std          # 缓冲区内 z-score
            z_i = clamp(z_i, −3, 3)                   # z 截断防重尾离群退化
            w_i = exp(beta * z_i) / sum(exp(beta * z_j))

        标准化使采样对 surprise 任意量级稳健（准确性项 O(1~100) 或重尾
        分布下不再退化为 argmax）；β 默认 1.0，可配置 surprise_beta。

        参数:
            buffer: 可选的自定义缓冲区（deque of (state, surprise)）。
                默认使用 memory.iter_buffer()。

        返回:
            (state, surprise) 元组，state 为 [num_nodes] 激活态副本。
            缓冲区为空时返回 None。
        """
        if buffer is not None:
            entries = list(buffer)
        else:
            entries = list(self.memory.iter_buffer())
        if len(entries) == 0:
            return None

        # 提取 surprise 列表（体验层 D：缓冲条目向后兼容 2/3-tuple）
        surprises = torch.tensor(
            [float(item[1]) for item in entries], dtype=torch.float32)

        # softmax 加权（2026-08-10 起：z-score 标准化 + z 截断 + β=1.0）
        # 标准化使采样尺度无关（surprise 现为准确性项 O(1~100)，且重尾分布
        # 下原始值 softmax 会退化为近似 argmax）；z 截断 clamp(±3) 防单条
        # 离群记忆（z≈10）垄断采样；β 默认降至 1.0 作为第二道保险。
        beta = self.surprise_beta
        mean_s = surprises.mean()
        # 单元素缓冲区无标准差可言（避免自由度警告），按退化处理
        std_s = (surprises.std() if surprises.numel() > 1
                 else torch.tensor(0.0, dtype=surprises.dtype))
        if std_s > 1e-8:
            z = (surprises - mean_s) / std_s
        else:
            z = torch.zeros_like(surprises)
        z = z.clamp(-3.0, 3.0)
        shifted = beta * z
        weights = torch.exp(shifted)
        weights = weights / weights.sum()

        # 按权重采样一个索引
        idx = int(torch.multinomial(weights, 1).item())
        item = entries[idx]
        state, surprise = item[0], item[1]
        return state.clone(), float(surprise)

    # ================================================================== #
    #  空闲态推断与学习
    # ================================================================== #

    def _idle_infer(self, sensory_input: torch.Tensor) -> Activation:
        """零精度推断（生成式回放）。

        precision 全零时，感官 clamping 项被乘以零，网络纯粹依靠内部
        模型（先验）运行 Langevin 动力学——这就是"做梦"的数学本质。

        为了让回放"重访"特定记忆，将传入的记忆状态作为网络初始 sigma
        的种子，使动力学从该记忆附近出发并放松到吸引子。

        惊讶度语义（2026-08-10 修复零精度陷阱）：零 precision 下准确性项
        恒为 0，做梦 surprise 会退化为 0，使 avg_surprise/回填/遗忘修剪
        全部失真。因此做梦时的惊讶度改为**对种子记忆的重构误差**
        （π=1）：surprise = 0.5·Σ(σ − seed)²，语义 = "回放漂移越大的记忆
        越值得巩固"，与神经科学（高预测误差优先回放）一致。
        （_rem_integration 的 blended 是人工凸组合，重构误差以种子记忆为
        参照，混合态下为近似度量。）

        E-P2-1: 零精度输入向量创建在 attractor 所在 device 上。
        E-P2-5: 不再"保存→修改→恢复"实例属性，改为通过 infer() 的
            override 参数传递空闲态值：
            - temperature_override: 临时使用空闲态温度
            - initial_state: 用记忆状态作为推断种子（不修改 self.sigma）
            - update_internal_state=False: 不污染在线 sigma

        参数:
            sensory_input: 记忆激活态 [num_nodes]，用作生成式回放的种子。
                若维度与 num_nodes 不匹配，则不设种子（从当前 sigma 出发）。

        返回:
            生成式回放得到的 Activation（surprise = 对种子的重构误差）。
        """
        # 计算有效温度（若 meta 可用，使用元调整后的温度）
        effective_temp = self.idle_temperature
        if self.meta is not None:
            adjusted = self.meta.get_adjusted_params(
                base_lr=self.idle_lr,
                base_orth=self.idle_orth_weight,
                base_temp=self.idle_temperature,
                base_cw=self.attractor.complexity_weight,
            )
            effective_temp = adjusted['temperature']

        # E-P2-1: 零精度输入创建在 attractor device 上
        zero_input = torch.zeros(
            self.attractor.input_dim, device=self.attractor.device)

        # E-P2-5: 通过接口参数传递空闲态值，不修改实例属性
        # 用记忆状态作为生成式回放的种子（initial_state 不修改 self.sigma）
        seed = None
        if sensory_input.shape[0] == self.attractor.num_nodes:
            seed = sensory_input

        # 零精度推断：precision 全零 → 纯先验采样（无感官 clamping）
        # update_internal_state=False: 推断不写回 self.sigma，避免污染在线状态
        activation = self.attractor.infer(
            zero_input, self.idle_precision,
            num_steps=self.idle_num_steps,
            temperature_override=effective_temp,
            initial_state=seed,
            update_internal_state=False,
        )

        # 做梦时的惊讶度 = 生成式回放对种子记忆的重构误差（π=1）
        # 零精度陷阱修复（2026-08-10）：零 precision 下 infer 的准确性项恒为
        # 0（surprise≡0），导致 avg_surprise/buffer 回填/遗忘修剪全部退化。
        # 改以被回放的种子记忆为参照计算重构误差，语义="回放漂移越大的记忆
        # 越值得巩固"（与神经科学高预测误差优先回放一致）。
        # 注意：sensory_input 是 [num_nodes] 记忆态，种子感官部分取前
        # input_dim 个节点（与在线感官节点定义一致）。
        if seed is not None:
            seed_sensory = sensory_input[:self.attractor.input_dim]
            replay_error = (
                activation.state[:self.attractor.input_dim] - seed_sensory)
            activation.surprise = float(
                0.5 * torch.sum(replay_error ** 2).item())
            activation.per_dim_surprise = (
                replay_error ** 2).detach().clone()

        return activation

    def _idle_learn(self, activation: Activation,
                    sensory_input: torch.Tensor) -> None:
        """低学习率 FEP 学习（巩固 J 矩阵）。

        使用空闲态学习率（在线的 1/10）和空闲态正交化权重进行 FEP 学习，
        精细调整 J 矩阵以巩固回放的记忆。低学习率确保巩固是温和的微调，
        而非剧烈重塑。

        E-P2-5: 不再"保存→修改→恢复"实例属性，改为通过 learn() 的
            override 参数传递空闲态值：
            - orth_weight_override: 临时使用空闲态正交化权重
            - complexity_weight_override: 临时使用空闲态复杂度权重
            覆盖值仅在本次 learn() 调用内生效，不修改 self.orth_weight
            / self.complexity_weight。若 meta 可用，使用元调整后的
            学习率、正交化权重和复杂度权重。

        参数:
            activation: 生成式回放得到的激活态。
            sensory_input: 产生该激活态的记忆状态（传入 learn 接口，
                当前实现中 learn 不直接使用此参数，但保留以兼容接口）。
        """
        # 计算有效的学习参数
        effective_lr = self.idle_lr
        effective_orth = self.idle_orth_weight
        effective_cw = self.attractor.complexity_weight

        # 若 meta 可用，使用元调整后的 lr / orth / cw（base * multiplier）
        if self.meta is not None:
            adjusted = self.meta.get_adjusted_params(
                base_lr=self.idle_lr,
                base_orth=self.idle_orth_weight,
                base_temp=self.idle_temperature,
                base_cw=self.attractor.complexity_weight,
            )
            effective_lr = adjusted['learning_rate']
            effective_orth = adjusted['orth_weight']
            effective_cw = adjusted['complexity_weight']

        # E-P2-5: 通过 override 参数传递空闲态权重，不修改实例属性
        self.attractor.learn(
            activation, sensory_input,
            learning_rate=effective_lr,
            orth_weight_override=effective_orth,
            complexity_weight_override=effective_cw,
        )

    # ================================================================== #
    #  阶段2: 突触稳态下调（SHY）
    # ================================================================== #

    def _synaptic_homeostasis(self) -> None:
        """SHY：J 矩阵全局 L2 归一化。

        突触稳态假说（Synaptic Homeostasis Hypothesis）：睡眠期间全局下调
        突触强度，恢复学习容量，防止突触饱和。本方法通过 Frobenius 范数
        归一化实现：当 J 的范数超过目标值时，按比例缩放回目标值。

        公式:
            norm = ||J||_F
            J = J * (target_norm / max(norm, target_norm))

        当 norm <= target_norm 时不做缩放（因子为 1），仅下调不过调。

        归一化后保持对称性与零对角线。

        若 meta 可用，通过 cw_multiplier 缩放目标范数：J 矩阵饱和度越高
        （cw_multiplier > 1），目标越小，归一化越积极；饱和度越低
        （cw_multiplier < 1），目标越大，归一化越温和。
        """
        # 计算有效目标范数
        effective_target = self.shy_target_norm
        if self.meta is not None:
            # 元可塑性通过 cw_multiplier 影响 shy_target_norm
            # cw_multiplier > 1 表示 J 饱和度高 → 应该更积极地下调
            # 因此用 cw_multiplier 缩放 target_norm（缩小目标 = 更积极归一化）
            effective_target = self.shy_target_norm / self.meta.state.cw_multiplier

        norm = float(torch.norm(self.attractor.J, p='fro').item())
        target = effective_target
        denom = max(norm, target)
        if denom < 1e-10:
            return  # J 近似为零，无需归一化

        scale = target / denom
        if scale < 1.0:  # 仅下调
            self.attractor.J = self.attractor.J * scale
            # 保持对称性与零对角线
            self.attractor.J = (self.attractor.J + self.attractor.J.T) / 2
            self.attractor.J.fill_diagonal_(0)
            logger.debug(
                f"SHY 突触下调: ||J||_F {norm:.4f} -> "
                f"{float(torch.norm(self.attractor.J, p='fro').item()):.4f}")

    # ================================================================== #
    #  阶段3: 遗忘修剪
    # ================================================================== #

    def _forgetting_pruning(self) -> None:
        """遗忘修剪：衰减低 surprise 记忆。

        遗忘是主动设计，不是 bug。本方法：
          1. 对 memory 缓冲区中 surprise 低于均值的条目，从 long_term_latent
             中减去其状态的小比例，弱化"无聊"记忆的痕迹。
          2. 清理情景记忆缓冲区中 turn 过旧的条目（age > max_age），
             控制情景记忆膨胀。

        这模拟了生物睡眠中 δ 波促遗忘的机制——与慢振荡促巩固形成平衡。
        """
        # --- 衰减低 surprise 记忆在 long_term_latent 中的痕迹 ---
        buf = list(self.memory.iter_buffer())
        if len(buf) > 0:
            surprises = [float(item[1]) for item in buf]
            mean_surprise = sum(surprises) / len(surprises)
            prune_rate = self.forget_prune_rate

            for item in buf:
                state, surprise = item[0], item[1]
                if surprise < mean_surprise:
                    # 弱化低 surprise 记忆的长期痕迹
                    self.memory.long_term_latent = (
                        self.memory.long_term_latent
                        - prune_rate * state
                    )
            logger.debug(
                f"遗忘修剪: 均值 surprise={mean_surprise:.4f}, "
                f"衰减率={prune_rate}")

        # --- 清理过旧的情景记忆条目 ---
        epi = list(self.memory.iter_episodic())
        if len(epi) > 0:
            current_turn = max(e.turn for e in epi)
            # 提取层 v1.4（S1-13，D1 丰碑哲学）：surviving 判据从"按轮数"
            # 改为"按磨损量"——effective_wear(e,t) < wear_threshold，初期
            # wear_threshold=∞（永不触发删除）；forget_max_age 保留为远期
            # 参数（承载力到上限后调小）。
            # 注：latent 弱化（本方法前半段，L707-717）与 SHY 保留为系统固有
            # 磨损、不改（S1-13 申报标注，理由见设计 §P1-B：作用对象=经验缓冲
            # 区 activation 状态无条目 ID 映射；SHY 是 J 矩阵稳定性归一化；
            # 条目文本/语义向量不被触及，D1"不删除"在条目层成立）。
            wear_threshold = float('inf')
            surviving = [
                e for e in epi
                if self._effective_wear(e, current_turn) < wear_threshold
            ]
            pruned = len(epi) - len(surviving)
            if pruned > 0:
                self.memory.replace_episodic_buffer(surviving)
                logger.debug(
                    f"遗忘修剪: 清理 {pruned} 条过期情景记忆 "
                    f"(wear_threshold={wear_threshold})")

    # ------------------------------------------------------------------ #
    #  提取层 v1.4（S1-13）：磨损观测与价值重放
    # ------------------------------------------------------------------ #

    def _effective_wear(self, entry, current_turn: int) -> float:
        """条目磨损量（D1 §2.3 公式；v1.4：只观测与排序，不用于删除）。

        effective_wear(i, t) = base_decay × (1 − min(0.9, 0.1×reference_count(i)))
                               × (t − last_reinforced_turn(i))

        链接（reference_count）越多磨损越慢；被加固（last_reinforced_turn
        刷新）后重新计时。base_decay=1.0（相对磨损量，无绝对时间尺度）。
        旧快照条目缺 last_reinforced_turn 时回退到 turn（写入即计时起点）。
        """
        base_decay = 1.0
        ref = int(getattr(entry, 'reference_count', 0) or 0)
        reinforced = getattr(entry, 'last_reinforced_turn', None)
        if reinforced is None:
            reinforced = getattr(entry, 'turn', current_turn)
        elapsed = max(0, int(current_turn) - int(reinforced or current_turn))
        return base_decay * (1.0 - min(0.9, 0.1 * ref)) * elapsed

    @staticmethod
    def _entry_id(entry):
        """条目标识（id 优先，无 id 时用 turn 兜底；观测用）。"""
        eid = getattr(entry, 'id', None)
        if eid is None:
            eid = getattr(entry, 'turn', None)
        return eid

    def _reinforce_entry(self, entry, current_turn: int) -> None:
        """加固写点（P2-B 事件 1/3）：reference_count+1＋last_reinforced_turn
        刷新（复用 record_reference，P2-A：reference_count 为加固计数唯一
        权威，不新增 link_count）。fail-open：异常静默不阻断做梦。
        """
        try:
            from core.doubt.confidence_field import record_reference
            record_reference(entry)
            entry.last_reinforced_turn = int(current_turn)
        except Exception:  # pylint: disable=broad-except
            pass

    def _wear_stats(self, entries) -> dict:
        """wear 分布观测（D1 §2.2：只观测不触发动作；远期淘汰分级预埋）。"""
        if not entries:
            return {'count': 0, 'mean': 0.0, 'max': 0.0, 'min': 0.0}
        current_turn = max(
            int(getattr(e, 'turn', 0) or 0) for e in entries)
        wears = [self._effective_wear(e, current_turn) for e in entries]
        return {
            'count': len(wears),
            'mean': round(sum(wears) / len(wears), 4),
            'max': round(max(wears), 4),
            'min': round(min(wears), 4),
        }

    def _cluster_replay_set(self, entries) -> list:
        """对采样集做聚类归组（H-6：fail-open）。

        工程常规（设计附录 B 标注：K-means/层次聚类为经典工程常规，无独立
        论文）：贪心分组——两两余弦相似度 ≥0.7 归一组。直接用条目已存
        语义向量（零额外 embed、无网络调用，与写侧引用匹配同哲学）；
        向量缺失/异常 → fail-open 每条目自成一组＋告警（做梦不瘫）。
        """
        self._last_cluster_failed = False
        try:
            if len(entries) <= 1:
                return [[e] for e in entries]
            vectors = []
            for e in entries:
                vec = getattr(e, 'semantic_vector', None)
                if vec is None:
                    raise RuntimeError("条目无语义向量")
                vectors.append(vec.detach().cpu().float())
            norms = torch.nn.functional.normalize(
                torch.stack(vectors), dim=-1)  # [n, d]
            sims = norms @ norms.T  # 余弦相似度矩阵
            used = [False] * len(entries)
            groups = []
            for i in range(len(entries)):
                if used[i]:
                    continue
                group = [entries[i]]
                used[i] = True
                for j in range(i + 1, len(entries)):
                    if used[j]:
                        continue
                    if float(sims[i, j].item()) >= 0.7:
                        group.append(entries[j])
                        used[j] = True
                groups.append(group)
            return groups
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "价值重放聚类失败（fail-open，跳过聚类直接整合）: %s", e)
            self._last_cluster_failed = True
            return [[e] for e in entries]

    def _value_replay(self, k=None) -> dict:
        """价值重放（S1-13 写点；§2.6 流程）：PER 概率采样 → 加固 →
        聚类归组（fail-open）→ 代表幸存加固。

        PER 采样调用 dream_replay.sample_replay_set（S1-6 纯函数采样子模块，
        §附 #5 调用关系定案）；候选集排除 gray（三重冻结①）；被采中条目
        reference_count+1＋last_reinforced_turn 刷新（加固事件 1）；聚类后
        代表幸存条目再加固（加固事件 3）。

        返回:
            观测字典（replay_set_ids/scores/reinforced_ids/merged_count/
            superseded_count/wear_stats/cluster_failed，进 dream 结果与
            dream_state.json）。异常 fail-open 返回部分统计，不阻断做梦。
        """
        result = {
            'replay_set_ids': [], 'scores': [], 'sampled_via_prob': True,
            'reinforced_ids': [], 'merged_count': 0, 'superseded_count': 0,
            'wear_stats': self._wear_stats([]), 'cluster_failed': False,
        }
        try:
            from runtime.dream_replay import (
                sample_replay_set, replay_k, replay_alpha,
            )
            entries = list(self.memory.iter_episodic())
            if not entries:
                return result
            result['wear_stats'] = self._wear_stats(entries)
            replay_set = sample_replay_set(
                entries, k=(k or replay_k()), alpha=replay_alpha())
            if not replay_set:
                return result
            current_turn = max(
                int(getattr(e, 'turn', 0) or 0) for e in entries)
            score_map = {self._entry_id(e): s for e, s, _p in replay_set}
            # 加固事件 1：被采中条目 reference_count+1＋last_reinforced_turn 刷新
            for entry, score, _prob in replay_set:
                self._reinforce_entry(entry, current_turn)
                result['scores'].append(round(float(score), 6))
                result['reinforced_ids'].append(self._entry_id(entry))
                result['replay_set_ids'].append(self._entry_id(entry))
            # 聚类归组（fail-open）：embed 不可达/异常 → 每条目自成一组
            groups = self._cluster_replay_set([e for e, _s, _p in replay_set])
            result['cluster_failed'] = bool(
                getattr(self, '_last_cluster_failed', False))
            # 加固事件 3：≥2 条的组选代表（分数最高）幸存 → 加固
            reps = 0
            for group in groups:
                if len(group) >= 2:
                    rep = max(
                        group, key=lambda e: score_map.get(
                            self._entry_id(e), 0.0))
                    self._reinforce_entry(rep, current_turn)
                    result['reinforced_ids'].append(self._entry_id(rep))
                    reps += 1
            result['merged_count'] = reps
            # 被 superseded 的旧条目不加固（本阶段不做文本合并，标记 0）
            result['superseded_count'] = 0
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("价值重放异常（fail-open）: %s", e)
        return result

    # ================================================================== #
    #  阶段4: 景观漂移
    # ================================================================== #

    def _landscape_drift(self) -> None:
        """景观漂移：J 矩阵微小随机扰动。

        模拟热涨落，让吸引子景观缓慢漂移，防止景观僵化。这对应生物大脑
        中突触连接的持续微小变动，保持系统的"活性"与可塑性。

        扰动后保持 J 的对称性与零对角线，避免引入非物理的非对称耦合。

        公式:
            J += random_noise * drift_scale
            J = (J + J.T) / 2   （对称化）
            J.fill_diagonal_(0)  （无自连接）
        """
        noise = torch.randn_like(self.attractor.J) * self.drift_scale
        self.attractor.J = self.attractor.J + noise
        # 保持对称性
        self.attractor.J = (self.attractor.J + self.attractor.J.T) / 2
        # 无自连接
        self.attractor.J.fill_diagonal_(0)

    # ================================================================== #
    #  阶段5: 目的演化
    # ================================================================== #

    def _purpose_evolve(self, activation: Activation) -> None:
        """目的层空闲态演化。

        用内部生成的 surprise 驱动 precision 调整（调用 purpose.adjust），
        使目的层在无外部输入时也能自主演化。此外，向 encounter_count
        最低的维度缓慢偏移 precision，鼓励探索尚未被关注的方向——
        这对应"好奇心"的空闲态萌芽。

        与在线模式的 adjust 不同，此处额外加入温和的低探索维度偏移，
        实现空闲态特有的"想知道什么"。

        参数:
            activation: 内部生成的激活态（来自回放或 REM 整合）。
        """
        # 用内部 surprise 驱动 precision 调整
        self.purpose.adjust(activation.surprise, activation)

        # 向 encounter_count 最低的维度缓慢偏移 precision（好奇心萌芽）
        self.purpose.nudge_low_encounter_dim(self.purpose_evolve_nudge)

    # ================================================================== #
    #  坍缩检测与扰动
    # ================================================================== #

    def _check_collapse(self, activation: Activation) -> bool:
        """检测吸引子坍缩。

        检查最近 N 个激活态之间的平均余弦相似度。如果超过阈值，说明
        网络陷入单一吸引子（坍缩），返回 True。

        实现方式：取最近 collapse_window 个激活态，计算两两余弦相似度
        上三角的平均值。高相似度 = 所有激活态趋于相同 = 坍缩。

        参数:
            activation: 当前激活态（会被记录到历史中）。

        返回:
            True 表示检测到坍缩，False 表示正常。
        """
        # 记录当前激活态
        self.activation_history.append(activation.state.detach().clone())
        # 限制历史长度，防止无界增长
        if len(self.activation_history) > 100:
            self.activation_history = self.activation_history[-100:]

        window = min(self.collapse_window, len(self.activation_history))
        if window < 2:
            return False

        # 取最近 window 个激活态
        recent = torch.stack(self.activation_history[-window:])  # [window, dim]
        # L2 归一化
        recent_norm = torch.nn.functional.normalize(recent, dim=-1)
        # 余弦相似度矩阵
        sim_matrix = recent_norm @ recent_norm.T  # [window, window]
        # 上三角（不含对角线）的平均相似度
        mask = torch.triu(
            torch.ones(window, window, dtype=torch.bool), diagonal=1)
        avg_sim = float(sim_matrix[mask].mean().item())

        return avg_sim > self.collapse_threshold

    def _inject_perturbation(self) -> None:
        """注入扰动打破坍缩。

        当检测到吸引子坍缩时，向 sigma 和 J 注入随机扰动，打破网络
        陷入单一吸引子的状态，恢复多样性。这对应五重防坍缩机制中的
        "吸引子计数监控"——实时检测并打破坍缩。

        扰动:
            sigma += randn * 0.3   （较大扰动打破状态坍缩）
            J += randn * 0.01      （小扰动打破景观坍缩）
        扰动后保持 J 的对称性与零对角线。
        """
        # sigma 扰动：打破状态坍缩
        self.attractor.sigma = (
            self.attractor.sigma
            + torch.randn_like(self.attractor.sigma) * 0.3
        )
        self.attractor.sigma = torch.clamp(
            self.attractor.sigma, -0.999, 0.999)

        # J 扰动：打破景观坍缩
        self.attractor.J = (
            self.attractor.J
            + torch.randn_like(self.attractor.J) * 0.01
        )
        # 保持对称性与零对角线
        self.attractor.J = (self.attractor.J + self.attractor.J.T) / 2
        self.attractor.J.fill_diagonal_(0)

        # 清空激活态历史，重新开始坍缩监测
        self.activation_history.clear()
        logger.debug("已注入扰动打破坍缩")

    # ================================================================== #
    #  做梦周期阶段调度
    # ================================================================== #

    def _select_phase(self) -> str:
        """按阶段权重随机选择一个做梦阶段。

        使用 random.choices 按权重抽样，对应生物睡眠中各阶段的
        时间分配比例。

        返回:
            选中的阶段名称（phase_weights 的键之一）。
        """
        phases = list(self.phase_weights.keys())
        weights = list(self.phase_weights.values())
        return random.choices(phases, weights=weights, k=1)[0]

    def _execute_phase(self, phase: str) -> Optional[Activation]:
        """执行指定的做梦阶段。

        参数:
            phase: 阶段名称。

        返回:
            该阶段产生的激活态（用于坍缩检测），无激活态的阶段返回 None。
        """
        if phase == 'nrem_consolidation':
            return self._nrem_consolidation()
        elif phase == 'synaptic_homeostasis':
            self._synaptic_homeostasis()
            return None
        elif phase == 'forgetting_pruning':
            self._forgetting_pruning()
            return None
        elif phase == 'landscape_drift':
            self._landscape_drift()
            return None
        elif phase == 'purpose_evolution':
            return self._purpose_evolution_phase()
        elif phase == 'rem_integration':
            return self._rem_integration()
        elif phase == 'doubt_review':
            # 体验层 D：怀疑复核（不产生激活态，无坍缩检测）
            self._doubt_review()
            return None
        elif phase == 'snapshot':
            self._save_snapshot()
            return None
        else:
            logger.warning(f"未知做梦阶段: {phase}")
            return None

    def _nrem_consolidation(self) -> Optional[Activation]:
        """阶段1 NREM 巩固：采样回放 + 零精度推断 + 低学习率学习。

        对应 NREM 睡眠的记忆巩固：按 surprise 加权采样一条记忆，
        用零精度推断做生成式回放，再以低学习率进行 FEP 学习巩固 J 矩阵，
        最后更新 memory 潜变量。

        返回:
            回放得到的激活态（用于坍缩检测），无记忆可回放时返回 None。
        """
        sampled = self._sample_by_surprise()
        if sampled is None:
            return None
        state, _surprise = sampled

        activation = self._idle_infer(state)
        self._idle_learn(activation, state)
        self.memory.update(activation, activation.surprise)
        return activation

    def _rem_integration(self) -> Optional[Activation]:
        """阶段6 REM 整合：跨记忆随机关联。

        对应 REM 睡眠的创造性整合：采样两条不同的记忆，将它们的激活态
        随机混合（凸组合），从混合态出发做生成式回放并学习。这使网络
        在不同吸引子之间建立新的关联——"梦境"中记忆的创造性重组。

        返回:
            整合回放得到的激活态（用于坍缩检测），记忆不足时返回 None。
        """
        s1 = self._sample_by_surprise()
        s2 = self._sample_by_surprise()
        if s1 is None or s2 is None:
            return None

        state1, _ = s1
        state2, _ = s2

        # 随机凸组合混合两条记忆
        alpha = float(torch.rand(1).item())
        blended = alpha * state1 + (1.0 - alpha) * state2
        blended = torch.clamp(blended, -0.999, 0.999)

        # 从混合态出发做生成式回放（创造新关联）
        activation = self._idle_infer(blended)
        self._idle_learn(activation, blended)
        self.memory.update(activation, activation.surprise)
        return activation

    def _purpose_evolution_phase(self) -> Optional[Activation]:
        """阶段5 目的演化：用回放激活态驱动 precision 演化。

        先采样一条记忆做生成式回放得到激活态，再用该激活态驱动
        目的层演化。返回激活态用于坍缩检测。

        返回:
            回放得到的激活态，无记忆时返回 None。
        """
        sampled = self._sample_by_surprise()
        if sampled is None:
            return None
        state, _ = sampled
        activation = self._idle_infer(state)
        self._purpose_evolve(activation)
        return activation

    # ================================================================== #
    #  快照
    # ================================================================== #

    # ------------------------------------------------------------------ #
    #  体验层 D（设计 v1.1 §6.3）：怀疑复核阶段（doubt_review）
    # ------------------------------------------------------------------ #

    def _doubt_review(self) -> None:
        """怀疑复核（睡眠=整合/复核窗口，Walker & Stickgold 2009）。

        内容（P1 §2.3 原样；显式裁决流程为"睡眠=整合/复核窗口"的工程
        翻译，溯源 §4.2 标注——论文支持睡眠做重放/整合/遗忘/去虚假吸引子，
        不支持逐条打分-裁决流程，故裁决逻辑做成纯函数、乘子保守、独立
        可回滚）：
          ① labile 窗口裁决：改写=新增 source='doubt' supersedes 条目 /
             无证据 confidence×1.02 重巩固 / 超时 ×0.98 折损
             （reconsolidation.resolve_labile 纯函数）
          ② 低置信（<0.3）复核：用不完全线索生成式回放（B3，Sinclair &
             Barense 2019：完全重复反而稳定旧记忆，绝不复述原文）——
             工程翻译：仅做保守重评估（×1.02 回稳）并标记 reviewed，
             生成式回放由既有 surprise 加权采样机制自然覆盖
          ③ 反教条抽查 top-N 高频记忆（reference/recall_count 高者——
             正是已关注方向里被 illusory truth 强化的内容，D2 语义：
             优先怀疑高频，Sharot 2011 C2）

        输出复核报告（reviewed/downgraded/rewritten/kept/flagged）进
        self.last_doubt_review（dream 结果字典 + [梦醒] 透传数据源）。
        任何异常只记日志不阻断做梦（fail-open）。
        """
        stats = dict(self.last_doubt_review)  # 跨调用累积（一轮做梦内多次复核合并）
        try:
            entries = list(self.memory.iter_episodic())
            if not entries:
                self.last_doubt_review = stats
                return
            from core.doubt.reconsolidation import resolve_labile
            from core.doubt.confidence_field import is_low_confidence

            # ① labile 窗口裁决
            for e in entries:
                if getattr(e, 'labile', False):
                    outcome = resolve_labile(e)
                    stats[outcome] = stats.get(outcome, 0) + 1
                    stats['reviewed'] += 1
                    if outcome == 'rewritten':
                        # 改写 = 新增 source='doubt' supersedes 条目
                        # （更新而非抹除，Schiller 2010 B2）
                        self._add_doubt_supersede(e)

            # ② 低置信（<0.3）复核（不完全线索原则：绝不复述原文，
            #    只做保守重评估并标记 reviewed）
            for e in entries:
                if is_low_confidence(e):
                    try:
                        e.confidence = max(0.0, min(
                            1.0, float(e.confidence or 1.0) * 1.02))
                    except (TypeError, ValueError):
                        pass
                    stats['reviewed'] += 1

            # ③ 反教条抽查 top-N 高频记忆（优先怀疑高频——已关注方向内
            #    被 illusory truth 强化的内容；只 flag 不抹除）
            high_freq = sorted(
                [e for e in entries
                 if int(getattr(e, 'reference_count', 0) or 0) > 0],
                key=lambda x: (int(getattr(x, 'reference_count', 0) or 0),
                               int(getattr(x, 'recall_count', 0) or 0)),
                reverse=True)[:3]
            stats['flagged'] += len(high_freq)

            # 联动 gap_registry（B 类清空 + 复核报告 + last_review）
            if self.gap_registry is not None:
                try:
                    self.gap_registry.mark_review(stats)
                except Exception:
                    pass
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("doubt_review 复核异常（fail-open）: %s", e)
        self.last_doubt_review = stats

    def _add_doubt_supersede(self, entry) -> None:
        """改写 = 新增 source='doubt' supersedes 条目（不抹除原文）。

        fail-open：任何异常静默（不阻断做梦）。
        """
        try:
            vec = getattr(entry, 'semantic_vector', None)
            if vec is None:
                return
            text = getattr(entry, 'text', '') or ''
            violated_by = getattr(entry, 'violated_by', None)
            new_text = (
                "[doubt-supersedes] 原记忆被证伪: "
                f"{violated_by or '（无证据记录）'} —— 原: {text[:80]}")
            self.memory.store_episodic(
                new_text, vec, surprise=float(getattr(entry, 'surprise', 0.0) or 0.0),
                turn=int(getattr(entry, 'turn', 0) or 0),
                source='doubt')
        except Exception:  # pylint: disable=broad-except
            pass

    def _save_snapshot(self) -> Optional[str]:
        """保存做梦后的状态快照。

        使用 torch.save 直接序列化为与 persistence.snapshot 兼容的格式
        （避免 core 层反向依赖 persistence 层，保持架构 DAG 无环）。
        保存吸引子景观（J/bias/sigma）、目的层状态（precision/history/
        coherence/encounter_count）和记忆潜变量。当 meta 可用时，
        额外保存元可塑性状态（meta 字段）。

        0.5.0/T1.1：目标路径改为会话级 `snapshots/{session}/latest_{session}.pt`
        （session_id 取自 config，默认 default）；写入加 fcntl 排他锁
        （T1.2 过渡期兜底，超时告警跳过，fail-open）。

        返回:
            快照文件路径，保存失败时返回 None。
        """
        try:
            landscape = self.attractor.get_landscape()
            purpose = self.purpose.get_purpose()
            purpose_dict = {
                'precision': purpose.precision,
                'history': purpose.history,
                'coherence': purpose.coherence,
                'encounter_count': purpose.encounter_count,
            }
            memory_state = self.memory.get_state()

            session_id = str(self.config.get('session_id', 'default'))
            sid = re.sub(r"[^A-Za-z0-9_-]", "_", session_id) or "session"

            data = {
                'version': _SNAPSHOT_VERSION,
                'timestamp': time.time(),
                'attractor': landscape,
                'purpose': purpose_dict,
                'memory': memory_state,
                # 0.5.0：会话归属元数据（与 persistence.snapshot 顶层一致）
                'session_id': sid,
            }
            # 元可塑性状态（可选）
            if self.meta is not None:
                data['meta'] = self.meta.get_state()

            snap_dir = self.snapshot_dir
            path = os.path.join(snap_dir, sid, f'latest_{sid}.pt')

            # 先确保会话子目录存在（锁文件与快照同目录，目录缺失时无法建锁）
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # T1.2：fcntl 排他锁（写路径串行化）
            lock_fd = _dream_acquire_write_lock(_dream_lock_path_for(path))
            if lock_fd is None and _DREAM_HAVE_FCNTL:
                logger.warning(
                    f"做梦快照写锁超时（{_DREAM_LOCK_TIMEOUT}s），本次跳过: {path}")
                return None

            try:
                # 原子写入：先写临时文件，再原子替换，避免崩溃时截断原有快照
                fd, tmp_path = tempfile.mkstemp(
                    prefix=".snap_", suffix=".tmp",
                    dir=os.path.dirname(path))
                try:
                    with os.fdopen(fd, "wb") as f:
                        torch.save(data, f)
                    os.replace(tmp_path, path)
                except Exception:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    raise
            finally:
                _dream_release_lock(lock_fd)
            logger.info(f"做梦快照已保存: {path}")
            return path
        except Exception as e:
            logger.warning(f"做梦快照保存失败（不影响做梦结果）: {e}")
            return None

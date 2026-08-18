"""
活体记忆系统 - 海马体核心：自指回路（Self-Referential Loop）
============================================================

让系统"听到自己对自身的描述"：将 decoder 输出的"记忆状态解读"
经蒸馏后，在下一轮以弱信号回注到编码器，形成自指回路。

核心机制:
  1. 蒸馏（distill）: 从 decoder 输出的 memory_context 中提取"自述"文本，
     丢弃外部内容回声（相关记忆回忆）与结构化标记。
  2. 编码回注（generate_echo）: 将上一轮蒸馏的自述编码为感官向量，
     以自适应权重 alpha 回注到感官输入
     （sensory_mixed = sensory_external + alpha * sensory_self）。
  3. 观测（observe）: 每轮结束后蒸馏当轮 memory_context，计算回声相似度，
     更新历史，为下一轮回注做准备。

Phase 0 MVP:
  - 固定 alpha_base=0.15，无 Tier 1/2/3 门控
  - 首轮无历史自述时 generate_echo 返回 None
  - 默认不接线（由 runtime/loop.py 按 self_ref_enabled 开关启用，540 测试零损伤）

Phase 1 稳定性保护层:
  - Tier 1: AutocorrController —— 激活自相关监测 + 自适应权重 + 锁定/振荡检测
  - Tier 2: L2 回声抑制 + L3 新颖性正交化 + L5 外部优先（硬守卫）
  - StabilityArbiter: 统一仲裁所有稳定性信号，决策最终自指权重
  - 锁定时注入正交噪声到 sensory（不是 J），避免污染正在学习的耦合矩阵
  - 历史不足（首轮 / activation_prev 为 None）时优雅降级为基线，不崩溃

参考：docs/SELF_REF_INTEGRATED_DESIGN.md
"""

import json
import logging
import math
import os
import time
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from core.types import Activation, resolve_device

logger = logging.getLogger(__name__)


class SelfVoiceDistiller:
    """自述蒸馏器：从 decoder 输出中提取系统对自身的描述。

    decoder（text 模式）输出的 memory_context 包含三部分：
      1. "记忆状态解读"——熵/惊讶度/coherence 的自然语言解释（自述，保留）
      2. "相关记忆回忆"——检索到的情景记忆文本（外部内容回声，丢弃）
      3. "详细数据"——原始指标，其中"激活节点: ..."部分为自述（保留）

    蒸馏规则：
      - 必取：``记忆状态解读:`` 段下以 ``- `` 开头的条目
      - 必取：``详细数据:`` 段中的"激活节点: ..."部分
      - 必弃：``相关记忆回忆:`` 段（外部内容回声）
      - 必弃：结构化标记行（``[记忆context]``、段标题）

    该类为静态工具类，所有方法均为 staticmethod，无实例状态。
    """

    # 段落标题与结构标记（与 bridge/decoder.py 的输出格式保持一致）
    _HEADER_MARKER = "[记忆context]"
    _INTERPRET_HEADER = "记忆状态解读:"
    _RECALL_HEADER = "相关记忆回忆:"
    _DETAIL_PREFIX = "详细数据:"

    @staticmethod
    def distill(memory_context: str) -> str:
        """从 memory_context 中蒸馏出紧凑的自述字符串。

        参数:
            memory_context: decoder（text 模式）输出的完整 context 文本。

        返回:
            紧凑的自述字符串，各片段以 ``" | "`` 连接。
            空输入或无法识别格式（如 vector 模式输出）时返回空字符串。
        """
        if not memory_context or not memory_context.strip():
            return ""

        lines = memory_context.split("\n")
        fragments: List[str] = []
        # 当前所处段落：'interpret' | 'recall' | 'detail' | None
        section: Optional[str] = None

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue

            # 丢弃结构化标记行
            if line == SelfVoiceDistiller._HEADER_MARKER:
                continue

            # 段落标题识别
            if line == SelfVoiceDistiller._INTERPRET_HEADER:
                section = "interpret"
                continue
            if line == SelfVoiceDistiller._RECALL_HEADER:
                section = "recall"
                continue
            if line.startswith(SelfVoiceDistiller._DETAIL_PREFIX):
                section = "detail"
                # 详细数据为单行：立即提取"激活节点"部分
                detail_content = line[len(
                    SelfVoiceDistiller._DETAIL_PREFIX):].strip()
                nodes = SelfVoiceDistiller._extract_activation_nodes(
                    detail_content)
                if nodes:
                    fragments.append(nodes)
                continue

            # 内容行按所在段落处理
            if section == "interpret":
                # 取 "- ..." 条目（熵/惊讶度/coherence 解读）
                if line.startswith("- "):
                    fragments.append(line[2:].strip())
                elif line.startswith("-"):
                    fragments.append(line[1:].strip())
                # 非列表行（异常格式）忽略，保证鲁棒
            elif section == "recall":
                # 外部内容回声：整段丢弃
                continue
            # section == 'detail' 的多行情况（当前 decoder 不会产生，
            # 防御性处理：忽略后续行，激活节点已在上面提取）

        return " | ".join(fragments)

    @staticmethod
    def _extract_activation_nodes(detail_content: str) -> str:
        """从"详细数据"内容中提取"激活节点: ..."部分。

        详细数据形如::

            熵:0.123, 惊讶度:0.456 | 激活节点: 节点0(强:0.789), 节点1(中:0.234)

        或无显著激活时::

            无显著激活节点 (熵:0.123, 惊讶度:0.456)

        参数:
            detail_content: "详细数据:" 之后的原始文本。

        返回:
            "激活节点: ..." 字符串，或"无显著激活节点"；
            无激活节点信息时返回空字符串。
        """
        if not detail_content:
            return ""

        # 按 "|" 分段，定位含"激活节点"的段
        for seg in detail_content.split("|"):
            seg = seg.strip()
            if not seg:
                continue
            if "激活节点" not in seg:
                continue
            # 无显著激活节点：仅保留该描述，去除附加的熵/惊讶度
            if seg.startswith("无显著激活节点"):
                return "无显著激活节点"
            # 激活节点: ... —— 从"激活节点:"起截取
            idx = seg.find("激活节点:")
            if idx != -1:
                return seg[idx:].strip()
            # 兜底：返回该段原文
            return seg
        return ""


class LLMSelfVoiceDistiller:
    """LLM 增强自述蒸馏器：在规则蒸馏基础上可选调用 LLM 生成更丰富自述。

    降级策略：
      1. LLM 不可用（无 bridge / API 错误）→ 回退规则蒸馏
      2. LLM 可用但距上次调用不足 interval 轮 → 使用上次缓存的 LLM 摘要
      3. LLM 可用且到达 interval → 调用 LLM 生成新摘要

    成本控制：
      - interval 参数控制调用频率（如每 5 轮调用一次）
      - 非调用轮次使用缓存结果 + 规则蒸馏补充
    """

    def __init__(self, llm_bridge=None, interval: int = 5,
                 max_tokens: int = 100, timeout: float = 5.0):
        """初始化 LLM 增强自述蒸馏器。

        参数:
            llm_bridge: LLMBridge 实例（可选）。为 None 时完全降级为规则蒸馏。
            interval: LLM 调用间隔轮数（默认 5）。每 interval 轮调用一次 LLM，
                其余轮次使用缓存结果。
            max_tokens: LLM 生成的最大 token 数（默认 100）。
            timeout: LLM 请求超时秒数（默认 5.0）。
        """
        self.llm_bridge = llm_bridge
        self.interval = interval
        self.max_tokens = max_tokens
        self.timeout = timeout
        # 缓存的 LLM 摘要文本
        self._cached_summary: Optional[str] = None
        # 上次成功调用 LLM 的轮次
        self._cache_turn: int = -1
        # 内部轮次计数器（每次 distill 递增）
        self._turn_counter: int = 0

    def distill(self, memory_context: str,
                activation: Optional[Activation] = None) -> str:
        """蒸馏自述，可选调用 LLM。

        始终先执行规则蒸馏作为基线和降级方案，然后根据 interval 和 LLM 可用性
        决定是否调用 LLM 生成增强摘要。

        参数:
            memory_context: decoder 输出的完整 context 文本。
            activation: 当轮激活态（可选，提供时将熵/惊讶度指标传入 LLM prompt）。

        返回:
            蒸馏后的自述字符串。LLM 可用时返回 LLM 摘要，否则返回规则蒸馏结果。
        """
        # 1. 始终先做规则蒸馏（作为基线和降级方案）
        rule_distilled = SelfVoiceDistiller.distill(memory_context)

        # 2. LLM 不可用：直接返回规则蒸馏
        if self.llm_bridge is None:
            return rule_distilled

        # 3. 到达 interval：调用 LLM
        self._turn_counter += 1
        if (self._turn_counter - self._cache_turn >= self.interval
                or self._cached_summary is None):
            try:
                llm_summary = self._call_llm(
                    rule_distilled, memory_context, activation)
                if llm_summary:
                    # 长度限制：截断到 100 字符
                    if len(llm_summary) > 100:
                        llm_summary = llm_summary[:100]
                    self._cached_summary = llm_summary
                    self._cache_turn = self._turn_counter
            except Exception:
                logger.warning(
                    "LLM distill failed, falling back to rule distillation",
                    exc_info=True)
                return rule_distilled

        # 4. 返回 LLM 摘要（如有缓存），否则规则蒸馏
        return self._cached_summary or rule_distilled

    def _call_llm(self, rule_distilled: str, memory_context: str,
                  activation: Optional[Activation]) -> Optional[str]:
        """调用 LLM 生成自述摘要。

        构造包含规则蒸馏结果和状态指标的 prompt，通过 llm_bridge.query_simple
        发送轻量查询。

        参数:
            rule_distilled: 规则蒸馏得到的自述文本。
            memory_context: 原始 memory_context（当前未直接使用，保留供未来扩展）。
            activation: 当轮激活态（可选，用于提取熵/惊讶度指标）。

        返回:
            LLM 生成的自述文本，或 None（LLM 无响应时）。
        """
        if self.llm_bridge is None:
            return None

        metrics = ""
        if activation is not None:
            metrics = (f"熵={activation.entropy:.3f}, "
                       f"惊讶度={activation.surprise:.3f}")

        prompt = (
            f"以下是一个记忆系统的自监测数据。\n"
            f"规则蒸馏: {rule_distilled}\n"
            f"状态指标: {metrics}\n"
            f"请用一句话（不超过30字）描述系统当前的记忆状态，"
            f"以第一人称'我'开头。只输出自述，不要解释。"
        )

        response = self.llm_bridge.query_simple(
            prompt, max_tokens=self.max_tokens,
            timeout=self.timeout)
        if response and response.strip():
            logger.info(
                f"LLM distill OK: {response.strip()[:60]}")
        return response.strip() if response else None


class AutocorrController:
    """Tier 1: 自适应控制器 - 通过激活自相关监测自指回路的动力学健康度。

    核心指标: ``autocorr = cos(σ_t, σ_{t-N})``，N 为延迟窗口（默认 5）。
    autocorr 直接衡量"自指回路自身的动力学"，比 entropy/coherence/surprise
    更直接反映回路状态。

    三种状态与响应:

    ============  ===================================  ====================================
    状态          判据                                 响应
    ============  ===================================  ====================================
    locked        autocorr > lock_threshold            alpha→0（自动静默）；
                 持续 persistence 轮                   注入正交噪声到 sensory
    oscillating   autocorr < oscillation_threshold     alpha→0（降权）
                 持续 persistence 轮
    normal        -threshold ≤ autocorr ≤ lock_thresh  alpha = base × (1 - autocorr_clipped)
    ============  ===================================  ====================================

    参考: docs/SELF_REF_INTEGRATED_DESIGN.md 第三节 3.1
    """

    def __init__(self, lag: int = 5, lock_threshold: float = 0.95,
                 oscillation_threshold: float = -0.3, persistence: int = 10,
                 noise_strength: float = 0.1, **kwargs) -> None:
        """初始化自适应控制器。

        Args:
            lag: 自相关计算的延迟窗口（默认5轮）。
            lock_threshold: 锁定检测阈值
                （autocorr > 此值持续 persistence 轮 = 锁定）。
            oscillation_threshold: 振荡检测阈值
                （autocorr < 此值持续 persistence 轮 = 振荡）。
            persistence: 锁定/振荡需持续的轮数才触发。
            noise_strength: 锁定时注入正交噪声的强度。

        兼容别名（并行测试使用，通过 **kwargs 传入）:
            lag_N: 等同于 lag。
            lock_persistence: 等同于 persistence。
            alpha_base: 存储为 self.alpha_base，供 compute_adaptive_alpha
                默认使用。
        """
        # 兼容别名
        if 'lag_N' in kwargs:
            lag = kwargs['lag_N']
        if 'lock_persistence' in kwargs:
            persistence = kwargs['lock_persistence']
        self.alpha_base: Optional[float] = kwargs.get('alpha_base', None)

        self.lag = lag
        self.lock_threshold = lock_threshold
        self.oscillation_threshold = oscillation_threshold
        self.persistence = persistence
        self.noise_strength = noise_strength

        # 连续计数器（实例状态，跨轮累积）
        self.lock_count: int = 0
        self.oscillation_count: int = 0

        # 最近一次自相关值（监控用）
        self.last_autocorr: Optional[float] = None

        # 内部激活态历史（当 update() 未传 prev_activations 时使用）
        self._internal_history: List[Activation] = []

        # 最近一次 update() 结果（供 compute_adaptive_alpha 和 state 属性使用）
        self._last_result: dict = {
            'autocorr': None, 'state': 'normal',
            'lock_count': 0, 'oscillation_count': 0,
            'should_inject_noise': False,
        }

    def update(self, current_activation: Optional[Activation],
               prev_activations: Optional[list] = None) -> dict:
        """更新自相关监测，返回状态评估。

        Args:
            current_activation: 当前轮的激活态。为 None 时（Phase 0 兼容、
                单元测试未传 activation_prev）返回基线结果，不修改计数器。
            prev_activations: 之前各轮的激活态列表（按时间顺序，最近的在最后）。
                需要至少有 lag 个元素才能计算 autocorr。
                为 None 时使用控制器内部维护的历史（单元测试单参数调用）。

        Returns:
            包含以下键的字典::

                {
                    'autocorr': float or None,       # 自相关值，历史不足时为 None
                    'state': 'normal'|'locked'|'oscillating',
                    'lock_count': int,               # 连续锁定轮数
                    'oscillation_count': int,        # 连续振荡轮数
                    'should_inject_noise': bool,     # 是否需要注入正交噪声
                }
        """
        # 无当前激活态（Phase 0 兼容）：返回基线，不修改计数器
        if current_activation is None:
            result = {
                'autocorr': None,
                'state': 'normal',
                'lock_count': self.lock_count,
                'oscillation_count': self.oscillation_count,
                'should_inject_noise': False,
            }
            self._last_result = result
            return result

        # 使用内部历史（单元测试单参数调用）或外部历史（loop 传入）
        use_internal = prev_activations is None
        if use_internal:
            prev_activations = self._internal_history

        # 计算自相关：需要足够的历史
        autocorr: Optional[float] = None
        if len(prev_activations) >= self.lag:
            lag_activation = prev_activations[-self.lag]
            try:
                autocorr = self._compute_autocorr(
                    current_activation.state, lag_activation.state)
            except Exception:
                logger.warning(
                    "autocorr computation failed, treating as None",
                    exc_info=True)
                autocorr = None

        # 更新锁定/振荡计数器（仅当有有效 autocorr 时）
        if autocorr is not None:
            if autocorr > self.lock_threshold:
                self.lock_count += 1
                self.oscillation_count = 0
            elif autocorr < self.oscillation_threshold:
                self.oscillation_count += 1
                self.lock_count = 0
            else:
                # 正常区间：重置两个计数器
                self.lock_count = 0
                self.oscillation_count = 0

        self.last_autocorr = autocorr

        # 判定状态
        state = 'normal'
        should_inject_noise = False
        if self.lock_count >= self.persistence:
            state = 'locked'
            should_inject_noise = True
        elif self.oscillation_count >= self.persistence:
            state = 'oscillating'

        # 更新内部历史（仅单参数调用模式，避免与 loop 的 activation_history 重复）
        if use_internal:
            self._internal_history.append(current_activation)
            max_hist = self.lag + 5
            if len(self._internal_history) > max_hist:
                del self._internal_history[
                    :len(self._internal_history) - max_hist]

        logger.debug(
            "AutocorrController.update: autocorr=%s state=%s "
            "lock_count=%d osc_count=%d",
            autocorr, state, self.lock_count, self.oscillation_count)

        result = {
            'autocorr': autocorr,
            'state': state,
            'lock_count': self.lock_count,
            'oscillation_count': self.oscillation_count,
            'should_inject_noise': should_inject_noise,
        }
        self._last_result = result
        return result

    @property
    def state(self) -> str:
        """当前稳定性状态（'normal' / 'locked' / 'oscillating'）。"""
        return self._last_result.get('state', 'normal')

    def compute_adaptive_alpha(self, base_alpha: Optional[float] = None,
                                autocorr_result: Optional[dict] = None) -> float:
        """根据自相关状态计算自适应增益。

        - locked/oscillating → 0
        - normal + autocorr is not None → ``base_alpha * (1 - autocorr_clipped)``
        - normal + autocorr is None → base_alpha（历史不足，使用基线）

        Args:
            base_alpha: 自指增益基线上限。为 None 时使用构造时存储的
                ``self.alpha_base``（若也为 None 则回退 0.15）。
            autocorr_result: :meth:`update` 的返回字典。为 None 时使用
                最近一次 :meth:`update` 的结果（``self._last_result``）。

        Returns:
            自适应增益值 ∈ [0, base_alpha]。
        """
        if base_alpha is None:
            base_alpha = self.alpha_base if self.alpha_base is not None else 0.15
        if autocorr_result is None:
            autocorr_result = self._last_result

        state = autocorr_result.get('state', 'normal')
        autocorr = autocorr_result.get('autocorr')

        # 锁定或振荡：增益归零
        if state in ('locked', 'oscillating'):
            return 0.0

        # 正常状态 + 无 autocorr：使用基线
        if autocorr is None:
            return base_alpha

        # 正常状态 + 有效 autocorr：自适应
        autocorr_clipped = max(0.0, min(1.0, abs(autocorr)))
        return base_alpha * (1.0 - autocorr_clipped)

    def generate_orthogonal_noise(self,
                                   sensory_self: torch.Tensor) -> torch.Tensor:
        """生成与 sensory_self 正交的噪声向量（锁定时使用）。

        噪声注入到 sensory 而不是 J，避免污染正在学习的耦合矩阵。
        先随机生成噪声，再减去在 sensory_self 方向上的投影分量，
        确保噪声只影响当轮推断，不持久化。

        Args:
            sensory_self: 当前自述感官向量。

        Returns:
            与 sensory_self 正交的噪声向量，同形状同设备。
        """
        noise = torch.randn_like(sensory_self) * self.noise_strength
        # 正交化：减去在 sensory_self 方向上的投影
        ss_norm_sq = torch.dot(sensory_self, sensory_self)
        if float(ss_norm_sq) > 1e-8:
            proj_coef = torch.dot(noise, sensory_self) / ss_norm_sq
            noise = noise - proj_coef * sensory_self
        return noise

    @staticmethod
    def _compute_autocorr(a: torch.Tensor, b: torch.Tensor) -> float:
        """计算两个激活态向量的余弦相似度（自相关）。

        使用 ``torch.nn.functional.cosine_similarity``。

        Args:
            a: 激活态向量 A。
            b: 激活态向量 B。

        Returns:
            余弦相似度标量 ∈ [-1, 1]。零向量时返回 0.0。
        """
        # 确保同设备
        b = b.to(a.device)
        # cosine_similarity 需要 batch 维度
        sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)
        val = float(sim.item())
        if val != val:  # NaN check
            return 0.0
        return val

    def get_state(self) -> dict:
        """返回控制器状态（用于快照序列化）。"""
        return {
            'lock_count': self.lock_count,
            'oscillation_count': self.oscillation_count,
            'last_autocorr': self.last_autocorr,
        }

    def set_state(self, state: dict) -> None:
        """从快照恢复控制器状态。

        Args:
            state: :meth:`get_state` 返回的字典。缺失字段回退默认值。
        """
        self.lock_count = int(state.get('lock_count', 0))
        self.oscillation_count = int(state.get('oscillation_count', 0))
        la = state.get('last_autocorr')
        self.last_autocorr = float(la) if la is not None else None


class StabilityArbiter:
    """统一稳定性仲裁器 - 收集所有信号，统一决策自指权重。

    汇总 Tier 1（自适应控制器）、L2（回声抑制）、L5（外部优先）
    的信号，统一计算最终自指增益 alpha，避免机制对冲。

    仲裁逻辑::

        1. Tier 1: alpha = autocorr_controller.compute_adaptive_alpha(base, result)
        2. L2 回声: echo_sim > 0.95 → alpha = 0;
                    0.80 < echo_sim ≤ 0.95 → alpha *= (1-sim)/0.15
        3. L5 外部: ext_novelty > 0.5 → alpha *= 0.3
        4. 天花板: alpha = min(alpha, alpha_base)
        5. 噪声注入: locked → inject_noise = True

    所有输入信号为 None 时优雅降级为基线 alpha_base，不崩溃。

    参考: docs/SELF_REF_INTEGRATED_DESIGN.md 第三节 3.4
    """

    def __init__(self, alpha_base: float = 0.15,
                 echo_suppress_threshold: float = 0.95,
                 echo_decay_threshold: float = 0.80,
                 ext_priority_threshold: float = 0.5,
                 ext_priority_factor: float = 0.3) -> None:
        """初始化仲裁器。

        Args:
            alpha_base: 自指增益基线上限。
            echo_suppress_threshold: L2 完全抑制阈值。
            echo_decay_threshold: L2 线性衰减起点。
            ext_priority_threshold: L5 外部优先触发阈值。
            ext_priority_factor: L5 触发时的衰减因子。
        """
        self.alpha_base = alpha_base
        self.echo_suppress_threshold = echo_suppress_threshold
        self.echo_decay_threshold = echo_decay_threshold
        self.ext_priority_threshold = ext_priority_threshold
        self.ext_priority_factor = ext_priority_factor

    def arbitrate(self, signals: dict) -> dict:
        """汇总所有稳定性信号，统一决策自指权重。

        支持两种信号格式：

        **结构化格式**（SelfReferentialLoop.generate_echo 调用）::

            {
                'autocorr_result': dict,      # AutocorrController.update() 返回
                'echo_similarity': float,     # L2 回声相似度
                'ext_novelty': float,         # L5 外部输入新颖度
            }

        **扁平格式**（单元测试调用）::

            {
                'autocorr': float,           # 自相关值
                'state': str,                # 'normal'|'locked'|'oscillating'
                'echo_sim': float,           # 回声相似度
                'ext_novelty': float,        # 外部新颖度
                'inject_noise': bool,        # （可选）直接指定噪声注入
            }

        Args:
            signals: 信号字典（见上述两种格式）。

        Returns:
            包含以下键的字典::

                {
                    'alpha': float,           # 最终自指增益 ∈ [0, alpha_base]
                    'tier1_alpha': float,    # Tier 1 建议值
                    'l2_factor': float,      # L2 衰减因子
                    'l5_factor': float,      # L5 衰减因子
                    'inject_noise': bool,    # 是否需要注入正交噪声
                    'state': str,            # 'locked'|'oscillating'|'normal'
                    'reasoning': str,        # 决策理由（用于日志/监控）
                }
        """
        # --- 信号格式兼容：统一为结构化格式 ---
        if 'autocorr_result' in signals:
            autocorr_result = signals['autocorr_result']
        else:
            # 扁平格式：从 autocorr + state 构造
            autocorr_val = signals.get('autocorr')
            state_val = signals.get('state', 'normal')
            should_inject = signals.get(
                'inject_noise', state_val == 'locked')
            autocorr_result = {
                'autocorr': autocorr_val,
                'state': state_val,
                'should_inject_noise': should_inject,
                'lock_count': 0,
                'oscillation_count': 0,
            }

        echo_sim = signals.get(
            'echo_similarity', signals.get('echo_sim'))
        ext_novelty = signals.get('ext_novelty')

        state = autocorr_result.get('state', 'normal')
        autocorr = autocorr_result.get('autocorr')

        # --- Tier 1: 自适应基线 ---
        tier1_alpha = self._compute_tier1_alpha(autocorr_result)
        alpha = tier1_alpha

        # --- L2 回声抑制 ---
        l2_factor = 1.0
        if echo_sim is not None:
            if echo_sim > self.echo_suppress_threshold:
                # 完全抑制
                alpha = 0.0
                l2_factor = 0.0
            elif echo_sim > self.echo_decay_threshold:
                # 线性衰减：从 echo_decay_threshold（l2=1.0）到
                # echo_suppress_threshold（l2=0.0）线性递减
                decay_range = (
                    self.echo_suppress_threshold - self.echo_decay_threshold)
                if decay_range > 1e-8:
                    l2_factor = (
                        self.echo_suppress_threshold - echo_sim) / decay_range
                    alpha = alpha * l2_factor

        # --- L5 外部优先 ---
        l5_factor = 1.0
        if ext_novelty is not None and ext_novelty > self.ext_priority_threshold:
            l5_factor = self.ext_priority_factor
            alpha = alpha * l5_factor

        # --- 天花板 ---
        alpha = min(alpha, self.alpha_base)
        alpha = max(0.0, alpha)  # 确保非负

        # --- 噪声注入 ---
        inject_noise = autocorr_result.get('should_inject_noise', False)

        # --- 决策理由 ---
        reasoning = self._build_reasoning(
            tier1_alpha, autocorr, state, echo_sim, l2_factor,
            ext_novelty, l5_factor, alpha, inject_noise)

        logger.debug("StabilityArbiter.arbitrate: %s", reasoning)

        return {
            'alpha': alpha,
            'tier1_alpha': tier1_alpha,
            'l2_factor': l2_factor,
            'l5_factor': l5_factor,
            'inject_noise': inject_noise,
            'state': state,
            'reasoning': reasoning,
        }

    def _compute_tier1_alpha(self, autocorr_result: dict) -> float:
        """计算 Tier 1 建议的 alpha 值。

        与 :meth:`AutocorrController.compute_adaptive_alpha` 逻辑一致，
        但由仲裁器独立实现以保持仲裁逻辑的完整性。
        """
        state = autocorr_result.get('state', 'normal')
        autocorr = autocorr_result.get('autocorr')

        # 锁定或振荡：增益归零
        if state in ('locked', 'oscillating'):
            return 0.0

        # 无 autocorr：使用基线
        if autocorr is None:
            return self.alpha_base

        # 自适应：alpha = base * (1 - |autocorr|_clipped)
        autocorr_clipped = max(0.0, min(1.0, abs(autocorr)))
        return self.alpha_base * (1.0 - autocorr_clipped)

    @staticmethod
    def _build_reasoning(tier1_alpha: float, autocorr: Optional[float],
                         state: str, echo_sim: Optional[float],
                         l2_factor: float, ext_novelty: Optional[float],
                         l5_factor: float, alpha: float,
                         inject_noise: bool) -> str:
        """构建决策理由字符串（用于日志/监控）。"""
        ac_str = f"{autocorr:.4f}" if autocorr is not None else "N/A"
        es_str = f"{echo_sim:.4f}" if echo_sim is not None else "N/A"
        en_str = f"{ext_novelty:.4f}" if ext_novelty is not None else "N/A"
        parts = [
            f"tier1={tier1_alpha:.4f}",
            f"autocorr={ac_str}",
            f"state={state}",
            f"l2={l2_factor:.4f}(sim={es_str})",
            f"l5={l5_factor:.4f}(novelty={en_str})",
            f"alpha={alpha:.4f}",
        ]
        if inject_noise:
            parts.append("inject_noise=True")
        return "; ".join(parts)


class SelfReferentialLoop:
    """自指回路：将系统对自身的描述在下一轮回注到感官输入。

    信号通路（与综合设计方案一致）::

        第 t-1 轮: observe(memory_context_{t-1})
                     └─ distill → self_voice_{t-1}
                        └─ encode → 缓存为 sensory_self 源
        第 t 轮:   generate_echo(...)
                     └─ 取 sensory_self 源 → {'alpha', 'vector'}
                        └─ sensory_mixed = ext + alpha * sensory_self

    Phase 0 MVP：固定 alpha_base，无自适应门控；首轮返回 None。
    Phase 1：Tier 1 自适应控制器 + Tier 2 硬守卫 + StabilityArbiter 仲裁；
    activation_prev 为 None 时优雅降级为基线（Phase 0 兼容）。

    构造方式兼容两种调用约定：
      - 直接传参（单元测试）：``SelfReferentialLoop(enc, tok, emb,
        alpha_base=0.15, history_cap=20)``
      - 传 config 字典（runtime/loop.py 接线）：
        ``SelfReferentialLoop(encoder=, tokenizer=, embedder=,
        config=config, device=...)``

    属性:
        encoder: 编码器实例（透传给 encoder.encode）。
        tokenizer: 分词器实例（透传给 encoder）。
        embedder: 嵌入器实例（透传给 encoder）。
        device: 计算设备。
        alpha_base: 自述回注基础权重。
        echo_threshold: 回声抑制阈值（L2 完全抑制阈值）。
        echo_decay: 回声衰减区间下限（L2 线性衰减起点）。
        history_capacity: 历史记录容量上限。
        autocorr_controller: Tier 1 自适应控制器实例。
        arbiter: 统一稳定性仲裁器实例。
        activation_history: 激活态历史列表（供 autocorr 计算）。
        prev_ext_sensory: 上一轮外部感官向量（供 L5 新颖度计算）。
    """

    def __init__(self, encoder, tokenizer, embedder,
                 alpha_base: Optional[float] = None,
                 history_cap: Optional[int] = None,
                 config: Optional[dict] = None,
                 device: Union[str, torch.device, None] = "auto",
                 echo_threshold: Optional[float] = None,
                 echo_decay: Optional[float] = None) -> None:
        """初始化自指回路。

        参数优先级：显式关键字参数 > config 字典 > 内置默认值。
        这样既支持单元测试直接传 ``alpha_base`` / ``history_cap``，
        也支持 runtime/loop.py 传 ``config`` 字典统一配置。

        参数:
            encoder: Encoder 实例，提供
                ``encode(text, tokenizer, embedder) -> SensoryInput`` 接口。
            tokenizer: 分词器实例（透传给 encoder）。
            embedder: 嵌入器实例（透传给 encoder）。
            alpha_base: 自述回注基础权重。None 时从 config 读取
                ``self_ref_alpha_base``（默认 0.15）。
            history_cap: 历史记录容量上限。None 时从 config 读取
                ``self_ref_history_cap``（默认 20）。
            config: 配置字典（可选）。读取以下键：

                - ``self_ref_alpha_base``（默认 0.15）
                - ``self_ref_echo_threshold``（默认 0.95，L2 完全抑制阈值）
                - ``self_ref_echo_decay``（默认 0.80，L2 线性衰减起点）
                - ``self_ref_history_cap``（默认 20）
                - ``self_ref_autocorr_lag``（默认 5，Tier 1 延迟窗口）
                - ``self_ref_lock_threshold``（默认 0.95，锁定检测阈值）
                - ``self_ref_oscillation_threshold``（默认 -0.3，振荡检测阈值）
                - ``self_ref_persistence``（默认 10，锁定/振荡持续轮数）
                - ``self_ref_noise_strength``（默认 0.1，正交噪声强度）
                - ``self_ref_ext_priority_threshold``（默认 0.5，L5 触发阈值）
                - ``self_ref_ext_priority_factor``（默认 0.3，L5 衰减因子）
                - ``session_id``（默认 "main"，自述持久化按会话隔离）
                - ``self_ref_voice_persist_path``（自述 JSONL 持久化文件路径；
                  显式 ``False``/空串 可禁用；缺省回退环境变量
                  ``LMS_SELF_VOICE_PATH`` / ``LMS_SELF_VOICE_DIR``，
                  再回退仓库默认 ``data/self_voice/``，见
                  :meth:`_resolve_voice_persist_path`）

            device: 计算设备（E-P2-1）。支持 ``"auto"``/``"cpu"``/``"cuda"``
                /``"cuda:0"`` 或 ``torch.device``。缓存的自述向量将创建在
                该设备上。
            echo_threshold: 回声抑制阈值（可选，覆盖 config）。
            echo_decay: 回声衰减区间下限（可选，覆盖 config）。
        """
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.device: torch.device = resolve_device(device)

        # 配置参数：显式参数优先，其次 config，最后内置默认值（保证向后兼容）
        cfg = config or {}
        self.alpha_base: float = float(
            alpha_base if alpha_base is not None
            else cfg.get("self_ref_alpha_base", 0.15))
        self.echo_threshold: float = float(
            echo_threshold if echo_threshold is not None
            else cfg.get("self_ref_echo_threshold", 0.95))
        self.echo_decay: float = float(
            echo_decay if echo_decay is not None
            else cfg.get("self_ref_echo_decay", 0.80))
        self.history_capacity: int = int(
            history_cap if history_cap is not None
            else cfg.get("self_ref_history_cap", 20))

        # 自述持久化（Phase 3.4 反思回流收尾，2026-08-05）
        # 自述文本跨重启持久到 JSONL（只写不读回 → 无循环）；
        # 路径解析优先级：cfg 显式 > 环境变量 > 仓库默认（详见
        # _resolve_voice_persist_path）。None 表示禁用持久化。
        self._session_id: str = str(cfg.get("session_id", "main") or "main")
        self.voice_persist_path: Optional[str] = (
            self._resolve_voice_persist_path(cfg))

        # T2.8/P2-3：bus 禁 LLM 自述开关（LMS_SELF_REF_NO_BUS=1 启用，默认 0）。
        # bus 会话 1456 条 LLM 蒸馏几乎全是「我正高度唤醒…」模板套话（信息量≈0）；
        # 总线已禁做梦（P1-4），此开关为兜底：bus 会话跳过 LLM 增强蒸馏，
        # 只用规则蒸馏（避免套话继续污染 self_voice 历史）。
        # 优先级：cfg 显式键 > 环境变量 LMS_SELF_REF_NO_BUS。
        self.self_ref_no_bus: bool = bool(cfg.get(
            "self_ref_no_bus",
            os.environ.get("LMS_SELF_REF_NO_BUS", "0") == "1"))

        # Phase 1 配置参数
        self_ref_autocorr_lag = int(
            cfg.get("self_ref_autocorr_lag", 5))
        self_ref_lock_threshold = float(
            cfg.get("self_ref_lock_threshold", 0.95))
        self_ref_oscillation_threshold = float(
            cfg.get("self_ref_oscillation_threshold", -0.3))
        self_ref_persistence = int(
            cfg.get("self_ref_persistence", 10))
        self_ref_noise_strength = float(
            cfg.get("self_ref_noise_strength", 0.1))
        self_ref_ext_priority_threshold = float(
            cfg.get("self_ref_ext_priority_threshold", 0.5))
        self_ref_ext_priority_factor = float(
            cfg.get("self_ref_ext_priority_factor", 0.3))

        # Tier 1: 自适应控制器（激活自相关监测）
        self.autocorr_controller = AutocorrController(
            lag=self_ref_autocorr_lag,
            lock_threshold=self_ref_lock_threshold,
            oscillation_threshold=self_ref_oscillation_threshold,
            persistence=self_ref_persistence,
            noise_strength=self_ref_noise_strength,
        )

        # 统一稳定性仲裁器
        self.arbiter = StabilityArbiter(
            alpha_base=self.alpha_base,
            echo_suppress_threshold=self.echo_threshold,
            echo_decay_threshold=self.echo_decay,
            ext_priority_threshold=self_ref_ext_priority_threshold,
            ext_priority_factor=self_ref_ext_priority_factor,
        )

        # 动态状态
        self.self_voice_history: List[str] = []
        # 上一轮自述嵌入（下一轮 generate_echo 的回注源）
        self.sensory_self_prev: Optional[torch.Tensor] = None
        self.gain_history: List[float] = []
        self.echo_similarity_history: List[float] = []
        self.turn_count: int = 0
        # 监控指标（最近一次取值）
        self.last_alpha: float = 0.0
        self.last_echo_similarity: Optional[float] = None

        # Phase 1 动态状态
        # 激活态历史（供 autocorr 计算，按时间顺序，最近的在最后）
        self.activation_history: List[Activation] = []
        # 上一轮外部感官向量（供 L5 外部新颖度计算）
        self.prev_ext_sensory: Optional[torch.Tensor] = None
        # 上上轮自述源嵌入（供 L3 正交化使用，observe 中更新）
        self._sensory_self_source_prev: Optional[torch.Tensor] = None
        # Phase 1 监控指标（最近一次取值）
        self.last_autocorr: Optional[float] = None
        self.last_ext_novelty: Optional[float] = None
        self.last_arbiter_state: str = 'normal'

        # Phase 2: 做梦钩子状态
        # dream_stale: 做梦后标记当前缓冲区为陈旧（审视报告 R7）
        # dream_age: 陈旧条目年龄（on_dream_end 中按指数衰减权重）
        self.dream_stale: bool = False
        self.dream_age: int = 0
        # Phase 2: 学习隔离标志（loop 中设置，generate_echo 中读取）
        self.last_is_self_ref_dominant: bool = False

        # Phase 3.1: 状态级递归
        # 将上一轮 activation.state 作为下一轮 infer() 的 initial_state 种子偏置。
        # 默认关闭，开启后通过自适应门控（autocorr 越高偏置越弱）防止锁定加剧。
        self.state_recursion_enabled: bool = bool(
            cfg.get("self_ref_state_recursion_enabled", False))
        self.state_recursion_strength: float = float(
            cfg.get("self_ref_state_recursion_strength", 0.3))
        # 上一轮激活态 [num_nodes] 维（observe 中缓存，generate_state_seed 中读取）
        self.prev_activation_state: Optional[torch.Tensor] = None
        # 最近一次实际偏置强度（generate_state_seed 中设置，默认 0）
        self.state_recursion_alpha: float = 0.0
        # Phase 3.5.1: 冷却剩余轮数（连续高 autocorr 触发后逐轮递减）
        self._state_recursion_cooldown: int = 0
        # 连续高 autocorr（>0.8）计数，达到 5 轮触发 10 轮冷却期
        self._high_autocorr_streak: int = 0

        # Phase 3.2: 多轮 echo 衰减
        # 从 1 轮延迟扩展到多轮，越早的自述权重越低（指数衰减）。
        # 默认 echo_max_rounds=1（向后兼容，等同 Phase 2 单轮模式）。
        self.sensory_self_history: List[torch.Tensor] = []
        self.echo_decay_rate: float = float(
            cfg.get("self_ref_echo_decay_rate", 0.5))
        self.echo_max_rounds: int = int(
            cfg.get("self_ref_echo_max_rounds", 1))

        # Phase 3.3: LLM 增强自述蒸馏（默认关闭，由 loop.py 注入 distiller 实例）
        self.llm_distill_enabled: bool = bool(
            cfg.get("self_ref_llm_distill_enabled", False))
        self.llm_distill_interval: int = int(
            cfg.get("self_ref_llm_distill_interval", 5))
        self.llm_distiller: Optional[LLMSelfVoiceDistiller] = None

    # ------------------------------------------------------------------ #
    #  回注生成
    # ------------------------------------------------------------------ #

    def generate_state_seed(self,
                            current_sigma: torch.Tensor
                            ) -> Optional[torch.Tensor]:
        """Phase 3.1: 生成状态级递归种子偏置。

        将上一轮 activation.state 作为下一轮 infer() 的 initial_state 种子，
        使推断从"上一轮收敛态"附近出发，形成状态级递归。通过自适应门控
        （autocorr 越高偏置越弱）和紧急制动（连续高 autocorr 触发冷却期）
        防止锁定加剧。

        逻辑：
          1. 功能关闭或无上一轮激活态 → 返回 None
          2. 当前处于 locked/oscillating 状态 → 返回 None（alpha=0）
          3. 冷却期内 → 递减 cooldown，返回 None
          4. autocorr > 0.8 连续 5 轮 → 触发 10 轮冷却期，返回 None
          5. 自适应门控：gate = 1 - |autocorr|（autocorr 越高偏置越弱）
          6. 做梦后衰减：dream_stale=True 时 effective_strength *= 0.3
          7. 过弱偏置 → 返回 None
          8. seed = (1-strength)*sigma + strength*prev_state

        参数:
            current_sigma: 当前吸引子网络的 sigma（推断起始状态）。

        返回:
            偏置后的种子张量，或 None（不启用状态递归时）。
        """
        # 1. 功能关闭或无上一轮激活态：不偏置
        if (not self.state_recursion_enabled
                or self.prev_activation_state is None):
            self.state_recursion_alpha = 0.0
            return None

        # 2. 读取 AutocorrController 最近一次结果
        autocorr_result = self.autocorr_controller._last_result
        autocorr = autocorr_result.get('autocorr')
        state = autocorr_result.get('state', 'normal')

        # locked/oscillating 状态：状态递归会加剧锁定/振荡，直接关闭
        if state in ('locked', 'oscillating'):
            self.state_recursion_alpha = 0.0
            # 处于异常状态时重置高 autocorr 连续计数
            self._high_autocorr_streak = 0
            return None

        # 3. 冷却期内：递减 cooldown，不偏置
        if self._state_recursion_cooldown > 0:
            self._state_recursion_cooldown -= 1
            self.state_recursion_alpha = 0.0
            logger.debug(
                "Phase 3.1 状态递归冷却中, 剩余 %d 轮",
                self._state_recursion_cooldown)
            return None

        # 4. 连续高 autocorr 紧急制动（Phase 3.5.1）
        if autocorr is not None and autocorr > 0.8:
            self._high_autocorr_streak += 1
            if self._high_autocorr_streak >= 5:
                # 触发 10 轮冷却期
                self._state_recursion_cooldown = 10
                self._high_autocorr_streak = 0
                self.state_recursion_alpha = 0.0
                logger.warning(
                    "Phase 3.5.1 状态递归紧急制动: 连续 %d 轮 "
                    "autocorr>0.8, 触发 10 轮冷却期",
                    5)
                return None
        else:
            # autocorr 回落或不可用：重置连续计数
            self._high_autocorr_streak = 0

        # 5. 自适应门控：autocorr 越高偏置越弱
        if autocorr is not None:
            gate = 1.0 - max(0.0, min(1.0, abs(autocorr)))
        else:
            # 无 autocorr（历史不足）：全强度门控
            gate = 1.0

        # 6. 计算有效偏置强度
        effective_strength = self.state_recursion_strength * gate

        # 7. 做梦后衰减（Phase 3.5.3）
        if self.dream_stale:
            effective_strength *= 0.3
            logger.debug(
                "Phase 3.5.3 状态递归做梦衰减: "
                "effective_strength *= 0.3 → %.6f",
                effective_strength)

        # 8. 过弱偏置：不偏置
        if effective_strength < 1e-6:
            self.state_recursion_alpha = 0.0
            return None

        # 9. 线性插值生成种子
        prev_state = self.prev_activation_state.to(current_sigma.device)
        seed = ((1.0 - effective_strength) * current_sigma
                + effective_strength * prev_state)

        self.state_recursion_alpha = effective_strength
        logger.debug(
            "Phase 3.1 状态递归种子: strength=%.4f gate=%.4f "
            "autocorr=%s", effective_strength, gate, autocorr)
        return seed

    def generate_echo(self,
                      entropy_ratio: float = 0.0,
                      ext_sensory: Optional[torch.Tensor] = None,
                      activation_prev: Optional[Activation] = None
                      ) -> Optional[dict]:
        """生成自指回注信号（Phase 1：自适应权重 + 三层分级防护）。

        取上一轮 observe 缓存的自述嵌入，经 Tier 1/2 仲裁后以自适应权重
        alpha 回注。首轮（``turn_count == 0`` 或无缓存自述）返回 None。

        Phase 1 增强流程::

            1. 取 sensory_self_prev → sensory_self（设备对齐）
            2. 更新 activation_history，调用 AutocorrController.update()
            3. 取 echo_similarity（observe 中已计算）
            4. 计算 ext_novelty = 1 - cos(ext_t, ext_{t-1})（L5）
            5. 调用 StabilityArbiter.arbitrate() → 最终 alpha
            6. L3 正交化（仅 echo_sim > echo_decay 时执行）
            7. 锁定时注入正交噪声到 sensory_self
            8. 更新 prev_ext_sensory
            9. 返回 {'alpha', 'vector', 'state', 'autocorr', ...}

        所有参数均有默认值，既支持 ``generate_echo()`` 无参调用（单元测试，
        activation_prev 为 None 时退化为基线 alpha_base），也支持
        ``generate_echo(entropy_ratio=, ext_sensory=, activation_prev=)``
        完整调用（runtime/loop.py 接线）。

        参数:
            entropy_ratio: 当前在线熵比例（保留接口，Phase 1 未直接使用）。
            ext_sensory: 当前轮外部感官向量，用于对齐回注向量设备与
                计算 L5 外部新颖度。为 None 时回注向量保持在缓存设备上。
            activation_prev: 上一轮激活态（保留接口兼容性，Phase 1 中
                控制器在 observe() 中更新，此处不再直接调用 update）。

        返回:
            ``None``（首轮无缓存自述），或包含以下键的字典::

                {
                    'alpha': float,           # 最终自指增益 ∈ [0, alpha_base]
                    'vector': torch.Tensor,   # 回注向量
                    'state': str,             # 'normal'|'locked'|'oscillating'
                    'autocorr': float|None,   # 激活自相关值
                    'reasoning': str,         # 仲裁决策理由
                    'tier1_alpha': float,     # Tier 1 建议值
                    'l2_factor': float,       # L2 衰减因子
                    'l5_factor': float,       # L5 衰减因子
                    'inject_noise': bool,     # 是否注入了噪声
                    'echo_similarity': float|None,
                    'ext_novelty': float|None,
                }

            当 ``ext_sensory`` 非 None 时，``vector`` 与其位于同一设备。
        """
        # 首轮或无历史自述：不回注
        if self.turn_count == 0 or self.sensory_self_prev is None:
            return None

        # 设备对齐：提供 ext_sensory 时回注向量与之同设备，保证后续混合
        # 运算不出错；未提供时（如单元测试）保持在缓存设备上
        # Phase 3.2: 多轮 echo 衰减——越早的自述权重越低（指数衰减）
        if self.echo_max_rounds > 1 and len(self.sensory_self_history) > 1:
            sensory_self = self._compute_decayed_echo(
                ext_sensory if ext_sensory is not None else None)
            if ext_sensory is not None:
                sensory_self = sensory_self.to(ext_sensory.device)
        else:
            # 向后兼容：单轮模式（echo_max_rounds=1 或历史不足）
            if ext_sensory is not None:
                sensory_self = self.sensory_self_prev.to(
                    ext_sensory.device).detach()
            else:
                sensory_self = self.sensory_self_prev.detach()

        # --- Phase 1: Tier 1 激活自相关 ---
        # 控制器在 observe() 中已更新（使用当前轮 activation），
        # 此处直接读取最近一次结果。activation_prev 参数保留接口兼容性。
        autocorr_result = self.autocorr_controller._last_result
        self.last_autocorr = autocorr_result.get('autocorr')

        # --- Phase 1: L2 回声相似度（observe 中已计算）---
        echo_sim = self.last_echo_similarity

        # --- Phase 1: L5 外部新颖度 ---
        ext_novelty = self._compute_ext_novelty(
            ext_sensory, self.prev_ext_sensory)
        self.last_ext_novelty = ext_novelty

        # --- Phase 1: 统一稳定性仲裁 ---
        signals = {
            'autocorr_result': autocorr_result,
            'echo_similarity': echo_sim,
            'ext_novelty': ext_novelty,
        }
        arbiter_result = self.arbiter.arbitrate(signals)
        alpha = arbiter_result['alpha']

        # --- Phase 2: 做梦后陈旧衰减 ---
        # 做梦修改了 J/precision/memory，sensory_self_prev 基于旧网络状态。
        # 标记为 stale 时临时压低自指权重，避免陈旧自述引起大 surprise。
        if self.dream_stale:
            alpha = alpha * 0.3
            logger.info(
                "self_ref generate_echo: dream_stale=True, "
                "alpha reduced to %.4f (stale decay)", alpha)

        # --- Phase 1: L3 新颖性正交化 ---
        # 仅当回声相似度超过衰减起点时执行（检测到回声才需要正交化）。
        # 不同文本的自述嵌入相似度通常远低于 echo_decay，
        # 因此 L3 在正常情况下为 no-op，不修改向量。
        # 锁定状态时跳过 L3：噪声注入是锁定时的首要机制，
        # L3 会将近零化向量，导致噪声无法正确正交化。
        sensory_self_original = sensory_self
        if (not arbiter_result['inject_noise']
                and echo_sim is not None
                and echo_sim > self.echo_decay
                and self._sensory_self_source_prev is not None):
            prev_source = self._sensory_self_source_prev.to(
                sensory_self.device).float()
            ss_norm_sq = torch.dot(prev_source, prev_source)
            if float(ss_norm_sq) > 1e-8:
                proj_coef = torch.dot(
                    sensory_self, prev_source) / ss_norm_sq
                sensory_self = sensory_self - proj_coef * prev_source

        # --- Phase 1: 锁定时注入正交噪声 ---
        # 噪声正交于原始 sensory_self（L3 之前的向量），
        # 并加到原始向量上（而非 L3 修改后的向量），
        # 确保 echo_vec - original = noise 与 original 正交。
        if arbiter_result['inject_noise']:
            noise = self.autocorr_controller.generate_orthogonal_noise(
                sensory_self_original)
            sensory_self = sensory_self_original + noise
            logger.info(
                "self_ref generate_echo: locked state detected, "
                "injecting orthogonal noise (strength=%.4f)",
                self.autocorr_controller.noise_strength)

        # --- Phase 1: 更新 prev_ext_sensory（供下轮 L5 计算）---
        if ext_sensory is not None:
            self.prev_ext_sensory = ext_sensory.detach().clone()

        # --- 记录增益历史 ---
        self.last_alpha = alpha
        self.gain_history.append(alpha)
        self._trim(self.gain_history)

        self.last_arbiter_state = arbiter_result['state']

        logger.debug(
            "self_ref generate_echo: turn=%d alpha=%.4f state=%s "
            "autocorr=%s ext_novelty=%s",
            self.turn_count, alpha, arbiter_result['state'],
            autocorr_result['autocorr'], ext_novelty)

        return {
            "alpha": alpha,
            "vector": sensory_self,
            "state": arbiter_result['state'],
            "autocorr": autocorr_result['autocorr'],
            "reasoning": arbiter_result['reasoning'],
            "tier1_alpha": arbiter_result['tier1_alpha'],
            "l2_factor": arbiter_result['l2_factor'],
            "l5_factor": arbiter_result['l5_factor'],
            "inject_noise": arbiter_result['inject_noise'],
            "echo_similarity": echo_sim,
            "ext_novelty": ext_novelty,
        }

    # ------------------------------------------------------------------ #
    #  自述持久化（Phase 3.4 反思回流收尾，2026-08-05）
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_voice_persist_path(cfg: dict) -> Optional[str]:
        """解析自述持久化文件路径（JSONL）。

        优先级（显式 > 环境变量 > 仓库默认）：
          1. ``cfg['self_ref_voice_persist_path']``（显式文件路径；
             显式 ``False`` / 空串 可禁用持久化）
          2. 环境变量 ``LMS_SELF_VOICE_PATH``（显式文件路径）
          3. 环境变量 ``LMS_SELF_VOICE_DIR``（目录，文件名为
             ``self_voice_{session_id}.jsonl``）
          4. 默认 ``<仓库根>/data/self_voice/self_voice_{session_id}.jsonl``

        返回:
            文件路径字符串；解析失败/显式禁用时返回 None（不持久化）。
        """
        explicit = cfg.get("self_ref_voice_persist_path")
        if explicit is False or explicit == "":
            return None
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()

        env_path = os.environ.get("LMS_SELF_VOICE_PATH", "").strip()
        if env_path:
            return env_path

        session_id = str(cfg.get("session_id", "main") or "main")
        env_dir = os.environ.get("LMS_SELF_VOICE_DIR", "").strip()
        if env_dir:
            return os.path.join(env_dir, f"self_voice_{session_id}.jsonl")

        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(
            repo_root, "data", "self_voice",
            f"self_voice_{session_id}.jsonl")

    def _persist_voice(self, text: str) -> None:
        """把一条自述追加到持久化 JSONL（只写不读回 → 无循环）。

        记录结构: ``{"ts": 秒, "session_id": ..., "text": ...}``。
        任何异常静默降级（fail-open），绝不影响 observe 主流程。
        """
        if not self.voice_persist_path or not text or not text.strip():
            return
        try:
            os.makedirs(os.path.dirname(self.voice_persist_path),
                        exist_ok=True)
            record = {
                "ts": time.time(),
                "session_id": self._session_id,
                "text": text,
            }
            with open(self.voice_persist_path, "a",
                      encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("self_voice 持久化失败（静默降级）: %s", e)

    def persisted_voice_count(self) -> int:
        """返回持久化文件中的自述条目数（fail-open 返回 0）。"""
        if not self.voice_persist_path or not os.path.exists(
                self.voice_persist_path):
            return 0
        try:
            with open(self.voice_persist_path, "r",
                      encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("self_voice 计数失败（静默降级）: %s", e)
            return 0

    def backfill_voice_history(self) -> int:
        """从持久化文件回填自述历史（纯文本，不回注嵌入 → 无循环）。

        语义:
          - 只读文件中的 ``text`` 字段；
          - 内存优先：与内存现有历史按文本去重，缺失条目按文件顺序追加；
          - 追加后裁剪到 ``history_capacity``；
          - 绝不触碰 ``sensory_self_prev`` / 嵌入缓存 / 编码器——
            回填只恢复"读"侧历史，不会触发新一轮反思或回注。

        返回:
            本次回填的条目数（fail-open 返回 0）。
        """
        if not self.voice_persist_path or not os.path.exists(
                self.voice_persist_path):
            return 0
        try:
            memory_seen = set(self.self_voice_history)
            added: List[str] = []
            with open(self.voice_persist_path, "r",
                      encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:  # pylint: disable=broad-except
                        continue  # 跳过损坏行，不阻塞回填
                    text = (record.get("text")
                            if isinstance(record, dict) else None)
                    if not isinstance(text, str) or not text.strip():
                        continue
                    if text in memory_seen:
                        continue  # 内存已有（内存优先）
                    added.append(text)
            if added:
                before = len(self.self_voice_history)
                self.self_voice_history.extend(added)
                self._trim(self.self_voice_history)
                # 返回实际留在内存中的新增条数（裁剪后可能少于 added）
                return max(0, len(self.self_voice_history) - before)
            return 0
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("self_voice 回填失败（静默降级）: %s", e)
            return 0

    # ------------------------------------------------------------------ #
    #  观测与状态更新
    # ------------------------------------------------------------------ #

    def observe(self, memory_context: str, activation: Activation) -> None:
        """观测当轮 memory_context，蒸馏并缓存为下一轮回注源。

        步骤:
          1. ``distill(memory_context)`` 提取自述文本
          2. ``encode(自述文本)`` 得到自述嵌入
          3. 计算 ``echo_similarity = cosine(当前嵌入, 上一轮嵌入)``
          4. 缓存当前嵌入为 ``sensory_self_prev``（供下一轮 generate_echo）
          5. 更新历史（裁剪到 history_capacity）
          6. Phase 1: 更新 activation_history，馈入 AutocorrController
          7. 递增 turn_count

        参数:
            memory_context: decoder（text 模式）输出的完整 context。
            activation: 当轮海马体激活态。Phase 1 中馈入 Tier 1 控制器
                计算激活自相关（autocorr），用于锁定/振荡检测。
        """
        # 1. 蒸馏自述
        # Phase 3.3: 若 LLM 蒸馏器已注入，使用 LLM 增强蒸馏；否则回退规则蒸馏
        # T2.8/P2-3: bus 会话跳过 LLM 蒸馏（LMS_SELF_REF_NO_BUS=1 时），
        # 只做规则蒸馏——总线事件套话不进 self_voice 历史。
        if (self.llm_distiller is not None
                and not (self.self_ref_no_bus and self._session_id == 'bus')):
            self_voice_text = self.llm_distiller.distill(
                memory_context, activation)
        else:
            self_voice_text = SelfVoiceDistiller.distill(memory_context)

        # 2. 编码为向量（空自述退化为零向量，由 encoder 处理）
        sensory_input = self.encoder.encode(
            self_voice_text, self.tokenizer, self.embedder)
        current_emb = sensory_input.vector.detach().to(self.device).float()

        # 3. 回声相似度（与上一轮自述嵌入比；首轮无 prev 时为 None）
        echo_sim: Optional[float] = None
        if self.sensory_self_prev is not None:
            prev_emb = self.sensory_self_prev.to(self.device).float()
            echo_sim = self._cosine_similarity(current_emb, prev_emb)
            self.echo_similarity_history.append(echo_sim)
            self._trim(self.echo_similarity_history)
        self.last_echo_similarity = echo_sim

        # 4. 缓存为下一轮回注源
        # Phase 1: 保存上上轮源嵌入（供 generate_echo 中 L3 正交化使用）
        self._sensory_self_source_prev = self.sensory_self_prev
        self.sensory_self_prev = current_emb

        # Phase 3.2: 多轮自述嵌入历史（供 _compute_decayed_echo 使用）
        # 越早的条目权重越低（指数衰减），保留 echo_max_rounds+2 个条目
        self.sensory_self_history.append(current_emb.clone())
        max_hist = max(self.echo_max_rounds + 2, 3)
        if len(self.sensory_self_history) > max_hist:
            del self.sensory_self_history[
                :len(self.sensory_self_history) - max_hist]

        # 5. 自述文本历史
        self.self_voice_history.append(self_voice_text)
        self._trim(self.self_voice_history)

        # 5.1 自述持久化（Phase 3.4：只写不读回，防循环；
        # 空自述不落盘；失败静默降级，绝不影响主流程）
        if self_voice_text and self_voice_text.strip():
            self._persist_voice(self_voice_text)

        # Phase 1: 更新 activation_history 并馈入 Tier 1 控制器
        # activation 是当前轮的激活态。先追加到历史，再用不含当前元素
        # 的历史计算 autocorr（避免自指比较）。
        self.activation_history.append(activation)
        max_hist = self.autocorr_controller.lag + 5
        if len(self.activation_history) > max_hist:
            del self.activation_history[
                :len(self.activation_history) - max_hist]

        prev_hist = (self.activation_history[:-1]
                     if len(self.activation_history) > 1 else [])
        autocorr_result = self.autocorr_controller.update(
            activation, prev_hist)
        self.last_autocorr = autocorr_result.get('autocorr')

        # Phase 3: 缓存当前 activation.state 供下轮状态递归
        self.prev_activation_state = (
            activation.state.detach().clone().to(self.device))

        # 6. 递增轮次
        self.turn_count += 1

        logger.debug(
            "self_ref observe: turn=%d echo_sim=%s voice_len=%d",
            self.turn_count, echo_sim, len(self_voice_text))

    # ------------------------------------------------------------------ #
    #  做梦钩子（Phase 2: Tier 3）
    # ------------------------------------------------------------------ #

    def on_dream_start(self) -> None:
        """做梦开始前钩子：标记当前自述状态为陈旧。

        做梦会修改 J 矩阵、precision、memory 潜变量等，使当前缓存的
        ``sensory_self_prev``（基于做梦前的网络状态蒸馏）变得陈旧。
        标记为 stale 后，``generate_echo`` 会临时降低自指权重，
        避免陈旧自述注入做梦后已变化的网络引起大 surprise（审视报告 R7）。

        此方法应在做梦流程开始前由 ``LivingMemoryLoop.dream()`` 调用。
        """
        self.dream_stale = True
        self.dream_age = 0
        logger.info(
            "self_ref on_dream_start: marking sensory_self_prev as stale "
            "(turn=%d)", self.turn_count)

    def on_dream_end(self) -> None:
        """做梦结束后钩子：按指数衰减陈旧自述，清除 stale 标记。

        做梦后 ``sensory_self_prev`` 基于旧网络状态，直接回注可能触发
        大 surprise 和学习风暴。此方法对 ``sensory_self_prev`` 施加
        指数衰减（``weight = exp(-1/tau)``，``tau=5``），使陈旧自述
        逐步淡出而非突然注入。

        衰减后清除 ``dream_stale`` 标志，恢复正常自指权重计算。
        ``dream_age`` 保留为 1 供诊断查询（表示经历过一次做梦衰减）。

        此方法应在做梦流程结束后由 ``LivingMemoryLoop.dream()`` 调用。
        """
        tau = 5.0
        decay_factor = math.exp(-1.0 / tau)

        if self.sensory_self_prev is not None:
            self.sensory_self_prev = self.sensory_self_prev * decay_factor
            logger.info(
                "self_ref on_dream_end: decayed sensory_self_prev by "
                "factor=%.4f (tau=%.1f), stale cleared",
                decay_factor, tau)
        else:
            logger.info(
                "self_ref on_dream_end: no sensory_self_prev to decay "
                "(turn=%d), stale cleared", self.turn_count)

        self.dream_stale = False
        self.dream_age = 1

    # ------------------------------------------------------------------ #
    #  序列化
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict:
        """返回自指回路当前状态（用于快照）。

        张量以 clone 副本形式返回，避免外部修改污染内部状态。
        encoder/tokenizer/embedder 不参与序列化（由调用方在重建时注入）。

        Phase 1 新增: activation_history（仅保留最近 lag 个元素的 state 张量）、
        autocorr_controller 状态、prev_ext_sensory 等监控字段。

        返回:
            包含全部动态状态与配置基线的字典。
        """
        # Phase 1: activation_history 序列化（仅保留最近 lag+5 个 state 张量）
        lag = self.autocorr_controller.lag
        max_serialize = lag + 5
        hist_tensors = []
        for act in self.activation_history[-max_serialize:]:
            hist_tensors.append(act.state.clone().cpu())

        # Phase 1: prev_ext_sensory
        prev_ext = None
        if self.prev_ext_sensory is not None:
            prev_ext = self.prev_ext_sensory.clone()

        return {
            "self_voice_history": list(self.self_voice_history),
            "sensory_self_prev": (self.sensory_self_prev.clone()
                                  if self.sensory_self_prev is not None
                                  else None),
            "gain_history": list(self.gain_history),
            "echo_similarity_history": list(self.echo_similarity_history),
            "turn_count": self.turn_count,
            "alpha_base": self.alpha_base,
            "last_alpha": self.last_alpha,
            "last_echo_similarity": self.last_echo_similarity,
            # Phase 1 新增
            "activation_history": hist_tensors,
            "autocorr_state": self.autocorr_controller.get_state(),
            "prev_ext_sensory": prev_ext,
            "last_autocorr": self.last_autocorr,
            "last_ext_novelty": self.last_ext_novelty,
            "last_arbiter_state": self.last_arbiter_state,
            # Phase 1: L3 正交化源嵌入（observe 中更新，generate_echo 中读取）
            "_sensory_self_source_prev": (
                self._sensory_self_source_prev.clone()
                if self._sensory_self_source_prev is not None
                else None),
            # Phase 1 顶层别名（供测试断言与序列化往返）
            "lock_count": self.autocorr_controller.lock_count,
            "osc_count": self.autocorr_controller.oscillation_count,
            # Phase 2: 做梦钩子状态
            "dream_stale": self.dream_stale,
            "dream_age": self.dream_age,
            "last_is_self_ref_dominant": self.last_is_self_ref_dominant,
            # Phase 3.1: 状态级递归
            "prev_activation_state": (
                self.prev_activation_state.clone()
                if self.prev_activation_state is not None
                else None),
            "state_recursion_enabled": self.state_recursion_enabled,
            "state_recursion_strength": self.state_recursion_strength,
            "state_recursion_alpha": self.state_recursion_alpha,
            "_state_recursion_cooldown": self._state_recursion_cooldown,
            "_high_autocorr_streak": self._high_autocorr_streak,
            # Phase 3.2: 多轮 echo 衰减
            "sensory_self_history": [
                t.clone() for t in self.sensory_self_history],
            "echo_decay_rate": self.echo_decay_rate,
            "echo_max_rounds": self.echo_max_rounds,
        }

    def set_state(self, state: dict) -> None:
        """从快照恢复自指回路状态。

        E-P2-1: 恢复的张量自动迁移到当前 device。
        向后兼容：缺失字段回退到默认值，旧快照（无自指状态或无 Phase 1
        字段）不报错。

        参数:
            state: :meth:`get_state` 返回的字典。
        """
        self.self_voice_history = list(state.get("self_voice_history", []))
        prev = state.get("sensory_self_prev")
        self.sensory_self_prev = (prev.clone().to(self.device)
                                  if prev is not None else None)
        self.gain_history = list(state.get("gain_history", []))
        self.echo_similarity_history = list(
            state.get("echo_similarity_history", []))
        self.turn_count = int(state.get("turn_count", 0))
        # alpha_base 为配置基线，快照中存在则恢复，否则保留构造时取值
        if "alpha_base" in state:
            self.alpha_base = float(state["alpha_base"])
        self.last_alpha = float(state.get("last_alpha", 0.0))
        les = state.get("last_echo_similarity")
        self.last_echo_similarity = (float(les) if les is not None else None)

        # Phase 1: activation_history 恢复（从 state 张量重建 Activation）
        self.activation_history = []
        for tensor in state.get("activation_history", []):
            self.activation_history.append(Activation(
                state=tensor.clone().to(self.device),
                entropy=0.0,
                surprise=0.0,
            ))

        # Phase 1: AutocorrController 状态恢复
        autocorr_state = state.get("autocorr_state")
        if autocorr_state is not None:
            self.autocorr_controller.set_state(autocorr_state)

        # Phase 1: 顶层 lock_count / osc_count 恢复（优先于 autocorr_state）
        if "lock_count" in state:
            self.autocorr_controller.lock_count = int(state["lock_count"])
        if "osc_count" in state:
            self.autocorr_controller.oscillation_count = int(
                state["osc_count"])

        # Phase 1: prev_ext_sensory 恢复
        prev_ext = state.get("prev_ext_sensory")
        self.prev_ext_sensory = (prev_ext.clone().to(self.device)
                                 if prev_ext is not None else None)

        # Phase 1: L3 正交化源嵌入恢复（向后兼容：旧快照无此字段时为 None）
        source_prev = state.get("_sensory_self_source_prev")
        self._sensory_self_source_prev = (
            source_prev.clone().to(self.device)
            if source_prev is not None else None)

        # Phase 1: 监控指标恢复
        la = state.get("last_autocorr")
        self.last_autocorr = float(la) if la is not None else None
        ln = state.get("last_ext_novelty")
        self.last_ext_novelty = float(ln) if ln is not None else None
        self.last_arbiter_state = state.get("last_arbiter_state", "normal")

        # Phase 1: arbiter 的 alpha_base 与当前 self.alpha_base 保持同步
        self.arbiter.alpha_base = self.alpha_base

        # Phase 2: 做梦钩子状态恢复（向后兼容：旧快照无此字段时回退默认值）
        self.dream_stale = bool(state.get("dream_stale", False))
        self.dream_age = int(state.get("dream_age", 0))
        self.last_is_self_ref_dominant = bool(
            state.get("last_is_self_ref_dominant", False))

        # Phase 3.1: 状态级递归恢复（向后兼容：旧快照无此字段时回退默认值）
        pas = state.get("prev_activation_state")
        self.prev_activation_state = (
            pas.clone().to(self.device) if pas is not None else None)
        self.state_recursion_enabled = bool(
            state.get("state_recursion_enabled", False))
        self.state_recursion_strength = float(
            state.get("state_recursion_strength", 0.3))
        self.state_recursion_alpha = float(
            state.get("state_recursion_alpha", 0.0))
        self._state_recursion_cooldown = int(
            state.get("_state_recursion_cooldown", 0))
        self._high_autocorr_streak = int(
            state.get("_high_autocorr_streak", 0))

        # Phase 3.2: 多轮 echo 衰减恢复（向后兼容：旧快照无此字段时空列表）
        self.sensory_self_history = [
            t.clone().to(self.device)
            for t in state.get("sensory_self_history", [])]
        self.echo_decay_rate = float(
            state.get("echo_decay_rate", 0.5))
        self.echo_max_rounds = int(
            state.get("echo_max_rounds", 1))

    # ------------------------------------------------------------------ #
    #  监控
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        """返回自指回路的监控指标快照（只读）。

        返回:
            包含 turn_count、last_alpha、echo_similarity、history 等
            监控指标的字典。键名兼顾 runtime 监控与单元测试断言。

            Phase 1 新增: autocorr、state（arbiter_state）、ext_novelty、
            lock_count、oscillation_count、autocorr_lag 等。
        """
        return {
            "turn_count": self.turn_count,
            "last_alpha": self.last_alpha,
            "last_echo_similarity": self.last_echo_similarity,
            "echo_similarity": self.last_echo_similarity,
            "alpha_base": self.alpha_base,
            "echo_threshold": self.echo_threshold,
            "history_capacity": self.history_capacity,
            "history_cap": self.history_capacity,
            "history_size": len(self.self_voice_history),
            "self_voice_history_size": len(self.self_voice_history),
            "gain_history_size": len(self.gain_history),
            "echo_similarity_history_size": len(self.echo_similarity_history),
            "has_sensory_self_prev": self.sensory_self_prev is not None,
            # Phase 1 新增
            "autocorr": self.last_autocorr,
            "state": self.last_arbiter_state,
            "arbiter_state": self.last_arbiter_state,
            "ext_novelty": self.last_ext_novelty,
            "lock_count": self.autocorr_controller.lock_count,
            "oscillation_count": self.autocorr_controller.oscillation_count,
            "autocorr_lag": self.autocorr_controller.lag,
            "activation_history_size": len(self.activation_history),
            "has_prev_ext_sensory": self.prev_ext_sensory is not None,
            # Phase 2 新增
            "dream_stale": self.dream_stale,
            "dream_age": self.dream_age,
            "is_self_ref_dominant": self.last_is_self_ref_dominant,
            # Phase 3.1 新增：状态级递归
            "state_recursion_enabled": self.state_recursion_enabled,
            "state_recursion_strength": self.state_recursion_strength,
            "state_recursion_alpha": self.state_recursion_alpha,
            # Phase 3.2 新增：多轮 echo 衰减
            "echo_max_rounds": self.echo_max_rounds,
            "echo_decay_rate": self.echo_decay_rate,
            "sensory_self_history_size": len(self.sensory_self_history),
            # Phase 3.3 新增：LLM 增强自述蒸馏
            "llm_distill_enabled": self.llm_distill_enabled,
        }

    # ------------------------------------------------------------------ #
    #  内部辅助
    # ------------------------------------------------------------------ #

    def _trim(self, history: list) -> None:
        """将历史列表原地裁剪到 history_capacity（保留最近条目）。

        参数:
            history: 待裁剪的列表（原地修改）。
        """
        cap = self.history_capacity
        if cap > 0 and len(history) > cap:
            del history[:len(history) - cap]

    @staticmethod
    def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """计算两个 1-D 向量的余弦相似度。

        参数:
            a: 向量 A。
            b: 向量 B（与 A 同形状、同设备）。

        返回:
            余弦相似度标量 ∈ [-1, 1]。零向量时返回 0.0。
        """
        denom = a.norm() * b.norm()
        if float(denom) < 1e-8:
            return 0.0
        return float(torch.dot(a, b) / denom)

    def _compute_ext_novelty(self,
                             ext_sensory: Optional[torch.Tensor],
                             prev_ext_sensory: Optional[torch.Tensor]
                             ) -> Optional[float]:
        """计算外部输入新颖度: ``ext_novelty = 1 - cos(ext_t, ext_{t-1})``。

        用于 L5 外部优先守卫：当外部输入发生显著变化时（novelty > 阈值），
        临时压低自指权重，让位给外部世界。

        优雅降级:
          - 任一输入为 None → 返回 None（Arbiter 视为无信号，不触发 L5）
          - 任一为零向量 → 返回 0.0（零向量 = 无外部输入 = 无新颖度）

        参数:
            ext_sensory: 当前轮外部感官向量。
            prev_ext_sensory: 上一轮外部感官向量。

        返回:
            外部新颖度标量 ∈ [0, 2]（通常 ∈ [0, 1]），或 None。
        """
        if ext_sensory is None or prev_ext_sensory is None:
            return None
        # 零向量无法有意义地计算余弦相似度
        if (float(ext_sensory.norm()) < 1e-8
                or float(prev_ext_sensory.norm()) < 1e-8):
            return 0.0
        cos_sim = self._cosine_similarity(ext_sensory, prev_ext_sensory)
        novelty = 1.0 - cos_sim
        # 限制到非负（cos_sim > 1 时理论上不应发生，但防御性处理）
        return max(0.0, novelty)

    def _compute_decayed_echo(self,
                              ext_sensory: Optional[torch.Tensor]
                              ) -> torch.Tensor:
        """Phase 3.2: 计算多轮指数衰减的感官自述嵌入。

        将 sensory_self_history 中最近 K 轮的自述嵌入按指数衰减加权平均，
        越早的自述权重越低（decay^i），使回注信号融合近期多轮自述而非
        仅依赖上一轮。

        加权公式::

            weighted_sum = Σ_{i=0}^{K-1} decay^i * history[-(i+1)]
            sensory_self = weighted_sum / Σ_{i=0}^{K-1} decay^i

        其中 i=0 对应最近一轮（权重 1.0），i=K-1 对应最早一轮
        （权重 decay^{K-1}）。

        Phase 3.5.2 范数守卫：多轮叠加后范数不应膨胀超过单轮的 1.5 倍，
        否则回退单轮模式，防止数值不稳定。

        参数:
            ext_sensory: 当前轮外部感官向量（仅用于设备对齐，可为 None）。

        返回:
            衰减加权后的感官自述嵌入（已 detach）。
        """
        K = min(self.echo_max_rounds, len(self.sensory_self_history))

        # 历史不足或配置为单轮：回退单轮模式
        if K <= 1 or self.echo_max_rounds <= 1:
            if ext_sensory is not None:
                return self.sensory_self_prev.to(
                    ext_sensory.device).detach()
            return self.sensory_self_prev.detach()

        # 设备对齐
        device = (ext_sensory.device if ext_sensory is not None
                  else self.sensory_self_prev.device)

        decay = self.echo_decay_rate
        weighted_sum = None
        total_weight = 0.0
        for i in range(K):
            # i=0 → history[-1]（最近一轮），i=K-1 → history[-K]（最早一轮）
            hist_vec = self.sensory_self_history[-(i + 1)].to(device).float()
            w = decay ** i
            if weighted_sum is None:
                weighted_sum = w * hist_vec
            else:
                weighted_sum = weighted_sum + w * hist_vec
            total_weight += w

        # 归一化
        if total_weight > 1e-8:
            sensory_self = weighted_sum / total_weight
        else:
            sensory_self = weighted_sum

        # Phase 3.5.2: 范数守卫——多轮叠加后范数不应膨胀超过单轮的 1.5 倍
        single_norm = float(self.sensory_self_prev.norm())
        multi_norm = float(sensory_self.norm())
        if single_norm > 1e-8 and multi_norm > single_norm * 1.5:
            logger.warning(
                "Phase 3.5 范数守卫: 多轮echo范数 %.4f 超过单轮 %.4f × 1.5, "
                "回退单轮模式", multi_norm, single_norm)
            sensory_self = self.sensory_self_prev.to(device).detach()

        return sensory_self.detach()

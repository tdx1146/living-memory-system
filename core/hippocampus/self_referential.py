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

参考：docs/SELF_REF_INTEGRATED_DESIGN.md
"""

import logging
from typing import List, Optional, Union

import torch

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
        echo_threshold: 回声抑制阈值（Phase 1 使用）。
        echo_decay: 回声衰减区间下限（Phase 1 使用）。
        history_capacity: 历史记录容量上限。
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
                - ``self_ref_echo_threshold``（默认 0.95，Phase 1 使用）
                - ``self_ref_echo_decay``（默认 0.80，Phase 1 使用）
                - ``self_ref_history_cap``（默认 20）

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

    # ------------------------------------------------------------------ #
    #  回注生成
    # ------------------------------------------------------------------ #

    def generate_echo(self,
                      entropy_ratio: float = 0.0,
                      ext_sensory: Optional[torch.Tensor] = None,
                      activation_prev: Optional[Activation] = None
                      ) -> Optional[dict]:
        """生成自指回注信号（Phase 0 MVP：固定权重，无门控）。

        取上一轮 observe 缓存的自述嵌入，以固定 ``alpha_base`` 回注。
        首轮（``turn_count == 0`` 或无缓存自述）返回 None，不回注。

        Phase 0 不使用 ``entropy_ratio`` / ``activation_prev``（保留给
        Phase 1 的自适应控制器与 autocorr 计算）。所有参数均有默认值，
        既支持 ``generate_echo()`` 无参调用（单元测试），也支持
        ``generate_echo(entropy_ratio=, ext_sensory=, activation_prev=)``
        完整调用（runtime/loop.py 接线）。

        参数:
            entropy_ratio: 当前在线熵比例（Phase 0 未使用）。
            ext_sensory: 当前轮外部感官向量，用于对齐回注向量设备。
                为 None 时回注向量保持在缓存设备上。
            activation_prev: 上一轮激活态（Phase 0 未使用）。

        返回:
            ``{'alpha': float, 'vector': torch.Tensor}`` 或 ``None``。
            当 ``ext_sensory`` 非 None 时，``vector`` 与其位于同一设备。
        """
        # 首轮或无历史自述：不回注
        if self.turn_count == 0 or self.sensory_self_prev is None:
            return None

        alpha = self.alpha_base

        # 设备对齐：提供 ext_sensory 时回注向量与之同设备，保证后续混合
        # 运算不出错；未提供时（如单元测试）保持在缓存设备上
        if ext_sensory is not None:
            sensory_self = self.sensory_self_prev.to(
                ext_sensory.device).detach()
        else:
            sensory_self = self.sensory_self_prev.detach()

        # 记录增益历史
        self.last_alpha = alpha
        self.gain_history.append(alpha)
        self._trim(self.gain_history)

        logger.debug("self_ref generate_echo: turn=%d alpha=%.4f",
                     self.turn_count, alpha)
        return {"alpha": alpha, "vector": sensory_self}

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
          6. 递增 turn_count

        参数:
            memory_context: decoder（text 模式）输出的完整 context。
            activation: 当轮海马体激活态（Phase 0 未使用，保留给 Phase 1
                的 autocorr 计算）。
        """
        # 1. 蒸馏自述
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
        self.sensory_self_prev = current_emb

        # 5. 自述文本历史
        self.self_voice_history.append(self_voice_text)
        self._trim(self.self_voice_history)

        # 6. 递增轮次
        self.turn_count += 1

        logger.debug(
            "self_ref observe: turn=%d echo_sim=%s voice_len=%d",
            self.turn_count, echo_sim, len(self_voice_text))

    # ------------------------------------------------------------------ #
    #  序列化
    # ------------------------------------------------------------------ #

    def get_state(self) -> dict:
        """返回自指回路当前状态（用于快照）。

        张量以 clone 副本形式返回，避免外部修改污染内部状态。
        encoder/tokenizer/embedder 不参与序列化（由调用方在重建时注入）。

        返回:
            包含全部动态状态与配置基线的字典。
        """
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
        }

    def set_state(self, state: dict) -> None:
        """从快照恢复自指回路状态。

        E-P2-1: 恢复的张量自动迁移到当前 device。
        向后兼容：缺失字段回退到默认值，旧快照（无自指状态）不报错。

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

    # ------------------------------------------------------------------ #
    #  监控
    # ------------------------------------------------------------------ #

    def get_status(self) -> dict:
        """返回自指回路的监控指标快照（只读）。

        返回:
            包含 turn_count、last_alpha、echo_similarity、history 等
            监控指标的字典。键名兼顾 runtime 监控与单元测试断言。
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

"""
活体记忆系统 - 桥接层：LMS↔沙漏翻译层
==========================================

在活体记忆系统（LMS）与外部"沙漏"显式记忆系统之间双向转换记忆：
  - LMS→沙漏：将 LMS 的激活态/记忆痕迹提取为显式记忆条目，供沙漏存储
  - 沙漏→LMS：将沙漏检索结果转换为 LMS 可用的感官输入，注入记忆循环

LMS 是基于海马体模型的隐式记忆系统（吸引子网络 + FEP 学习规则），
"沙漏"是外部显式记忆系统（向量数据库、知识图谱等），提供精确事实检索。
翻译层解耦两者的数据格式，使 LMS 能按需"外化"记忆到精确存储，
也能在需要时从精确存储"内化"事实到记忆循环。

设计要点：
  1. HourglassClient 为抽象接口，不绑定具体后端（向量库/图谱/内存）。
  2. 翻译层是无状态转换器（除持有的默认 hourglass_client 外），
     可安全地在多轮对话中复用。
  3. 所有输出字典均使用 JSON 可序列化的原生类型（list/float/int/str），
     便于直接存入任意后端。
  4. 维度兼容：检索时自动跳过维度不匹配的条目（与
     MemoryManager.recall_episodic 的优雅降级策略一致）。

任务编号：A-P2-1
"""

import math
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch

from core.types import Activation, SensoryInput
# [B16] embed 熔断器：与 bridge/encoder.py:67 同款（写侧 embed 三处共用同一
# 熔断状态）；recall_from_hourglass 的 embed 调用统一走熔断，故障不裸抛/挂起
from core.sensory.circuit_breaker import (  # noqa: E402
    CircuitOpenError,
    get_default_embed_circuit,
)


# ================================================================== #
#  沙漏客户端抽象接口
# ================================================================== #

class HourglassClient(ABC):
    """沙漏显式记忆系统客户端抽象基类。

    定义显式记忆存储系统的统一接口，不绑定具体后端实现。
    具体后端可以是向量数据库（FAISS / Milvus / Pinecone）、
    知识图谱、关系型数据库，或简单的内存实现
   （``InMemoryHourglassClient``）。

    所有方法均以 ``dict`` 作为记忆记录的载体。记录格式由调用方
    （``TranslationLayer``）与具体后端协商，但用于检索的记录至少
    应包含 ``semantic_vector`` 字段（list 或 torch.Tensor）。

    约束：
      - ``store`` 返回的 ID 在同一客户端实例内唯一。
      - ``search`` 返回的记录按相似度降序排列。
      - ``delete`` 对不存在的 ID 返回 ``False`` 而非抛异常。
    """

    @abstractmethod
    def store(self, record: dict) -> str:
        """存储一条记忆记录。

        参数:
            record: 记忆记录字典。应包含 ``semantic_vector`` 字段
                （list 或 torch.Tensor）供后续向量检索使用。
                其他字段（text、surprise 等）由后端原样保存。

        返回:
            记录的唯一标识符（ID字符串）。
        """
        ...

    @abstractmethod
    def search(self, query_vector: torch.Tensor,
               top_k: int = 3) -> List[dict]:
        """根据查询向量检索最相关的记忆。

        参数:
            query_vector: 查询语义向量（1-D Tensor）。
            top_k: 返回的最大条目数。

        返回:
            最相关的记忆记录列表（按相似度降序）。
            无匹配时返回空列表。
        """
        ...

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """删除指定 ID 的记忆记录。

        参数:
            record_id: ``store`` 返回的记录唯一标识符。

        返回:
            是否成功删除。记录不存在时返回 ``False``。
        """
        ...


class InMemoryHourglassClient(HourglassClient):
    """内存版沙漏客户端：用于测试与原型验证。

    所有记忆记录保存在内存列表中，检索基于余弦相似度。
    不依赖任何外部服务，适合单元测试与快速原型。

    存储策略:
      - ``store`` 时预提取 ``semantic_vector`` 并归一化缓存，
        加速后续检索（避免每次 search 重复转换）。
      - 内部字段以 ``_`` 前缀标记（``_id``、``_vector``），
        ``search`` 返回结果时自动剥离这些内部字段。

    检索策略:
      - 余弦相似度（归一化向量的点积）。
      - 维度不匹配的条目被跳过（优雅降级），与
        ``MemoryManager.recall_episodic`` 的混合维度策略一致。
    """

    def __init__(self) -> None:
        """初始化空的内存存储。"""
        self._records: List[dict] = []

    def store(self, record: dict) -> str:
        """存储一条记忆记录到内存。

        参数:
            record: 记忆记录字典。

        返回:
            生成的记录 UUID。
        """
        record_id = str(uuid.uuid4())
        # 浅拷贝，避免外部修改影响已存储记录
        stored = dict(record)
        stored['_id'] = record_id
        stored['_stored_at'] = time.time()
        # 预提取并归一化向量，加速后续检索
        vec = record.get('semantic_vector')
        if vec is not None:
            stored['_vector'] = self._to_tensor(vec)
        self._records.append(stored)
        return record_id

    def search(self, query_vector: torch.Tensor,
               top_k: int = 3) -> List[dict]:
        """基于余弦相似度检索最相关的记忆。

        参数:
            query_vector: 查询语义向量。
            top_k: 返回的最大条目数。

        返回:
            匹配的记忆记录列表（按相似度降序），每条附加
            ``score``（相似度）与 ``record_id`` 字段。
        """
        if not self._records:
            return []

        query = self._to_tensor(query_vector)
        if query.dim() == 1:
            query = query.unsqueeze(0)  # [1, qd]
        query_norm = torch.nn.functional.normalize(query, dim=-1).squeeze(0)
        qd = query_norm.shape[-1]

        scored: List[tuple] = []
        for rec in self._records:
            vec = rec.get('_vector')
            if vec is None:
                continue
            # 维度不匹配则跳过（优雅降级）
            if vec.shape[-1] != qd:
                continue
            v = vec if vec.dim() == 1 else vec.squeeze()
            v_norm = torch.nn.functional.normalize(
                v.unsqueeze(0), dim=-1).squeeze(0)
            sim = float(torch.dot(v_norm, query_norm).item())
            scored.append((sim, rec))

        if not scored:
            return []

        scored.sort(key=lambda x: x[0], reverse=True)
        k = min(top_k, len(scored))

        # 返回不含内部字段的记录副本，附加检索分数与 ID
        results: List[dict] = []
        for sim, rec in scored[:k]:
            out = {key: val for key, val in rec.items()
                   if not key.startswith('_')}
            out['record_id'] = rec.get('_id')
            out['score'] = sim
            results.append(out)
        return results

    def delete(self, record_id: str) -> bool:
        """删除指定 ID 的记忆记录。

        参数:
            record_id: 记录唯一标识符。

        返回:
            是否成功删除。
        """
        for i, rec in enumerate(self._records):
            if rec.get('_id') == record_id:
                del self._records[i]
                return True
        return False

    def size(self) -> int:
        """返回当前存储的记录数。"""
        return len(self._records)

    @staticmethod
    def _to_tensor(vec: Any) -> torch.Tensor:
        """将 list 或 Tensor 统一为 CPU float 1-D Tensor。"""
        if isinstance(vec, torch.Tensor):
            t = vec.detach().cpu().float()
        else:
            t = torch.tensor(vec, dtype=torch.float32)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        return t


# ================================================================== #
#  翻译层
# ================================================================== #

class TranslationLayer:
    """LMS↔沙漏翻译层。

    在活体记忆系统（LMS）与沙漏显式记忆系统之间双向转换记忆。

    LMS→沙漏方向:
      - :meth:`extract_explicit_memories`: 从情景记忆缓冲区提取显式记忆条目
      - :meth:`extract_activation_pattern`: 将激活态序列化为模式描述

    沙漏→LMS方向:
      - :meth:`inject_explicit_memory`: 将沙漏检索文本转换为 LMS 感官输入
      - :meth:`create_context_injection`: 将显式记忆格式化为 LLM 可读上下文

    双向同步:
      - :meth:`sync_to_hourglass`: 批量同步 LMS 记忆到沙漏
      - :meth:`recall_from_hourglass`: 从沙漏检索记忆

    属性:
        hourglass_client: 默认沙漏客户端，供 :meth:`recall_from_hourglass`
            使用。为 None 时调用该方法会抛出 RuntimeError。
            :meth:`sync_to_hourglass` 接受独立的 client 参数，不受此属性约束。
    """

    def __init__(self,
                 hourglass_client: Optional[HourglassClient] = None) -> None:
        """初始化翻译层。

        参数:
            hourglass_client: 可选的默认沙漏客户端实例。供
                :meth:`recall_from_hourglass` 使用。
        """
        self.hourglass_client = hourglass_client

    # ------------------------------------------------------------------ #
    #  LMS → 沙漏
    # ------------------------------------------------------------------ #

    def extract_explicit_memories(self, memory_manager,
                                  top_k: int = 5) -> List[dict]:
        """从 LMS 情景记忆缓冲区提取显式记忆条目。

        遍历 ``memory_manager.iter_episodic()`` 返回的情景记忆条目
       （``EpisodicEntry``），按惊讶度（surprise）降序选取最重要的
        ``top_k`` 条，转换为 JSON 可序列化的显式记忆格式。

        每个条目包含：
          - ``text``: 原始对话文本
          - ``semantic_vector``: 语义向量（转换为 list，便于序列化）
          - ``vector_dim``: 语义向量维度
          - ``timestamp``: 时间戳（用对话轮次 turn 作为时序标记）
          - ``surprise``: 惊讶度（准确性项，恒≥0），衡量该记忆的"意外/重要"程度
          - ``activation_strength``: 激活强度（语义向量 L2 范数，
            作为记忆痕迹强度的代理度量）
          - ``source``: 来源标记（``'lms_episodic'``）
          - ``extracted_at``: 提取时的墙钟时间戳

        参数:
            memory_manager: MemoryManager 实例，需提供 ``iter_episodic()``
                与 ``episodic_size()`` 接口。
            top_k: 提取的最大条目数（按惊讶度降序）。
                ``top_k <= 0`` 时提取全部条目。

        返回:
            显式记忆条目列表（按惊讶度降序）。缓冲区为空时返回空列表。
        """
        episodic_entries = list(memory_manager.iter_episodic())
        if not episodic_entries:
            return []

        # 按惊讶度降序排列（高 surprise = 重要记忆，与 consolidation 策略一致）
        sorted_entries = sorted(
            episodic_entries, key=lambda e: e.surprise, reverse=True)

        # top_k <= 0 表示提取全部
        if top_k is not None and top_k > 0:
            k = min(top_k, len(sorted_entries))
            selected = sorted_entries[:k]
        else:
            selected = sorted_entries

        results: List[dict] = []
        for entry in selected:
            vec = entry.semantic_vector.detach().cpu().float()
            # 统一为 1-D
            if vec.dim() == 0:
                vec = vec.unsqueeze(0)
            results.append({
                'text': entry.text,
                'semantic_vector': vec.tolist(),
                'vector_dim': int(vec.shape[-1]),
                'timestamp': int(entry.turn),
                'surprise': float(entry.surprise),
                'activation_strength': float(vec.norm().item()),
                'source': 'lms_episodic',
                'extracted_at': time.time(),
            })
        return results

    def extract_activation_pattern(self, activation: Activation,
                                   top_k: int = 10,
                                   threshold: float = 0.01) -> dict:
        """将激活态转换为可序列化的模式描述。

        提取高度激活的节点（按激活绝对值排序）、熵、惊讶度等，
        生成一个 JSON 可序列化的字典，供沙漏系统存储与分析。

        输出字段：
          - ``active_nodes``: 高度激活节点列表，每项含
            ``node_id`` / ``activation``（带符号值）/ ``strength``（绝对值）
          - ``num_active``: 显著激活节点数
          - ``entropy``: 激活熵
          - ``entropy_ratio``: 熵占最大熵（ln(num_nodes)）的比例
          - ``surprise``: 惊讶度（准确性项，恒≥0）
          - ``num_nodes``: 节点总数
          - ``mean_activation``: 平均激活绝对值
          - ``max_activation``: 最大激活绝对值
          - ``activation_norm``: 激活态 L2 范数

        参数:
            activation: 海马体激活态（``Activation`` 实例）。
            top_k: 提取的强激活节点数上限。
            threshold: 激活阈值，低于此值的节点不计入 active_nodes。

        返回:
            可序列化的激活模式描述字典。
        """
        state = activation.state.detach().cpu().float()
        if state.dim() == 0:
            state = state.unsqueeze(0)
        num_nodes = int(state.shape[-1])

        # 提取 top_k 强激活节点
        active_nodes: List[dict] = []
        if num_nodes > 0:
            k = min(top_k, num_nodes)
            if k > 0:
                top_values, top_indices = torch.topk(state.abs(), k=k)
                mask = top_values > threshold
                top_values = top_values[mask]
                top_indices = top_indices[mask]

                for val, idx in zip(top_values, top_indices):
                    node_id = int(idx)
                    active_nodes.append({
                        'node_id': node_id,
                        'activation': float(state[node_id]),
                        'strength': float(val),
                    })

        # 熵归一化比例
        max_entropy = math.log(num_nodes) if num_nodes > 1 else 1.0
        entropy_ratio = (float(activation.entropy) / max_entropy
                         if max_entropy > 0 else 0.0)

        return {
            'active_nodes': active_nodes,
            'num_active': len(active_nodes),
            'entropy': float(activation.entropy),
            'entropy_ratio': entropy_ratio,
            'surprise': float(activation.surprise),
            'num_nodes': num_nodes,
            'mean_activation': float(state.abs().mean().item()),
            'max_activation': float(state.abs().max().item()),
            'activation_norm': float(state.norm().item()),
        }

    # ------------------------------------------------------------------ #
    #  沙漏 → LMS
    # ------------------------------------------------------------------ #

    def inject_explicit_memory(self, text: str, encoder,
                               tokenizer, embedder) -> SensoryInput:
        """将沙漏检索到的文本转换为 LMS 可用的感官输入。

        复用 Encoder 的编码逻辑（自动适配文本路径 / token id 路径），
        将文本编码为 ``SensoryInput``，并在 metadata 中标注来源为
        ``'hourglass_injection'``，使下游模块能区分该输入来自沙漏
        而非普通对话。

        参数:
            text: 沙漏检索到的文本（事实/记忆内容）。
            encoder: Encoder 实例。
            tokenizer: 分词器实例（文本路径下不参与编码）。
            embedder: 嵌入器实例。

        返回:
            SensoryInput 对象，metadata 中 ``source`` 为
            ``'hourglass_injection'``，``injected_text`` 为截断后的原始文本。
        """
        sensory_input = encoder.encode(text, tokenizer, embedder)
        # 标注来源为沙漏注入，使下游能区分注入记忆与普通对话输入
        sensory_input.metadata['source'] = 'hourglass_injection'
        sensory_input.metadata['injected_text'] = text[:200]
        return sensory_input

    def create_context_injection(self, memories: List[dict],
                                 max_tokens: int = 500) -> str:
        """将显式记忆条目格式化为 LLM 可读的上下文文本。

        将 ``extract_explicit_memories`` 或 ``recall_from_hourglass``
        返回的记忆列表格式化为结构化文本，供 LLM 作为上下文使用。

        输出格式::

            [显式记忆回忆]
            1. (时间:0, 惊讶度:0.50, 强度:1.23) "记忆文本..."
            2. (时间:1, 惊讶度:1.00, 强度:0.80) "另一段记忆..."

        参数:
            memories: 显式记忆条目列表（dict），每项应包含
                ``text`` / ``surprise`` / ``timestamp`` / ``activation_strength``
                字段（缺失字段以默认值填充）。
            max_tokens: 上下文的近似 token 上限。由于无分词器，
                以字符数近似（1 token ≈ 1 字符的保守估计），
                超出时截断后续条目。单条文本最长 200 字符。

        返回:
            格式化的上下文文本。输入为空时返回空字符串。
        """
        if not memories:
            return ""

        lines: List[str] = ["[显式记忆回忆]"]
        used = len(lines[0])
        budget = max_tokens

        for i, mem in enumerate(memories, 1):
            # 安全提取字段：显式 None 与缺失键均回退到默认值
            text_raw = mem.get('text')
            text = str(text_raw) if text_raw is not None else ''

            surprise_raw = mem.get('surprise')
            surprise = float(surprise_raw) if surprise_raw is not None else 0.0

            timestamp = mem.get('timestamp')
            if timestamp is None:
                timestamp = '?'

            strength_raw = mem.get('activation_strength')
            strength = (float(strength_raw)
                        if strength_raw is not None else 0.0)

            # 单条文本截断（避免单条记忆撑爆上下文）
            if len(text) > 200:
                text = text[:200] + "..."

            line = (f'{i}. (时间:{timestamp}, '
                    f'惊讶度:{surprise:.2f}, 强度:{strength:.2f}) '
                    f'"{text}"')
            line_len = len(line) + 1  # +1 for newline

            # 超出 token 预算则停止
            if used + line_len > budget:
                break

            lines.append(line)
            used += line_len

        if len(lines) <= 1:
            # 只有标题行，说明所有记忆都被预算截断
            return ""
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  双向同步
    # ------------------------------------------------------------------ #

    def sync_to_hourglass(self, memory_manager,
                          hourglass_client: HourglassClient) -> int:
        """批量同步 LMS 记忆到沙漏系统。

        从 ``memory_manager`` 提取全部情景记忆条目（不限 top_k），
        逐条存入 ``hourglass_client``。单条存储失败不影响其他条目
        （容错策略），返回实际成功同步的条目数。

        参数:
            memory_manager: MemoryManager 实例。
            hourglass_client: 目标沙漏客户端实例。

        返回:
            成功同步的条目数。缓冲区为空时返回 0。
        """
        count = memory_manager.episodic_size()
        if count == 0:
            return 0

        # 提取全部记忆（top_k=0 表示不限数量）
        memories = self.extract_explicit_memories(
            memory_manager, top_k=0)

        synced = 0
        for mem in memories:
            try:
                hourglass_client.store(mem)
                synced += 1
            except Exception:
                # 单条存储失败不影响整体同步（容错）
                continue
        return synced

    def recall_from_hourglass(self, query_text: str, embedder,
                              top_k: int = 3) -> List[dict]:
        """从沙漏系统检索记忆。

        将查询文本编码为语义向量，在 ``self.hourglass_client`` 中
        检索最相关的 ``top_k`` 条记忆。

        embedder 需提供 ``embed_text(text)`` 或 ``embed_text_raw(text)``
        方法（PretrainedEmbedder 同时提供两者）。SimpleEmbedder 无此
        方法时不支持文本检索——调用方需使用具备文本编码能力的 embedder。

        参数:
            query_text: 查询文本。
            embedder: 嵌入器实例（需提供 embed_text 或 embed_text_raw）。
            top_k: 返回的最大条目数。

        返回:
            匹配的记忆记录列表（按相似度降序），每条附加
            ``score`` 与 ``record_id`` 字段。

        异常:
            RuntimeError: 未配置 ``self.hourglass_client`` 时抛出。
            ValueError: embedder 不支持文本编码时抛出。
        """
        if self.hourglass_client is None:
            raise RuntimeError(
                "未配置 hourglass_client，无法从沙漏检索。"
                " 请在 TranslationLayer 构造时传入 hourglass_client，"
                " 或使用 sync_to_hourglass + client.search 组合。")

        # 获取查询语义向量
        # [B16] 统一走熔断器（与 encoder.py:67 同款）：embed 服务故障时快速抛
        # CircuitOpenError（不触网死等），调用方按熔断降级处理——旧实现裸调
        # embed_text 会重试链 5-10s 挂起或裸抛 embed 异常
        cb = get_default_embed_circuit()
        if hasattr(embedder, 'embed_text'):
            query_vector = cb.call(lambda: embedder.embed_text(query_text))
        elif hasattr(embedder, 'embed_text_raw'):
            query_vector = cb.call(lambda: embedder.embed_text_raw(query_text))
        else:
            raise ValueError(
                "embedder 必须提供 embed_text 或 embed_text_raw 方法"
                " 以支持文本检索。SimpleEmbedder 不支持，"
                " 请使用 PretrainedEmbedder 或自定义文本嵌入器。")

        return self.hourglass_client.search(query_vector, top_k=top_k)

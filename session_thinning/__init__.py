"""会话层薄化 · 会话层薄化工具包（session_thinning）

总任务书 §二.7（A 会话层薄化）前两步落地 + 后两步接口定义：

- measure.py      步骤0 测量拆分（A0）——上下文构成分析（纯 stdlib）
- compression.py  步骤1 K 轮参数化压缩策略（保留最近 K 轮原文 + 更早历史折叠为
                  单一 checkpoint；K 灰度 50→30→20，env 配置化）
- interfaces.py   步骤2 指针化 + 步骤3 任务状态对象——接口定义与集成点标记
                  （接线到 OpenClaw 会话层内核 = 待 OpenClaw 集成点，
                   需主代理确认，不擅改环境）
- runbook.md      OpenClaw 侧如何采集真实上下文构成样本（采样需 OpenClaw 侧执行）

依据：《四妹-会话层薄化-20260817.md》§三（步骤0-3 原文）；《总任务书-重编一口气
-四妹-20260818.md》§二.7（前两步可落地代码；指针化/任务状态对象出实现方案 + 标记
待集成点）。
"""

from .measure import (
    ContextSample,
    ContextAnalysis,
    SegmentStat,
    SEGMENT_KEYS,
    analyze_context,
    sample_from_mapping,
    make_sample_skeleton,
    analysis_to_dict,
    sample_to_dict,
)
from .compression import (
    CompressionConfig,
    CompressionResult,
    TurnRecord,
    K_GRAYSCALE_BANDS,
    compress_rounds,
    resolve_config,
    k_from_env,
    mode_from_env,
    should_enable,
    validate_k_band,
)
from .interfaces import (
    ArtifactPointer,
    TaskState,
    POINTER_KINDS,
    expand_pointer,
    load_task_state,
    save_task_state,
)

__all__ = [
    # measure
    "ContextSample",
    "ContextAnalysis",
    "SegmentStat",
    "SEGMENT_KEYS",
    "analyze_context",
    "sample_from_mapping",
    "make_sample_skeleton",
    "analysis_to_dict",
    "sample_to_dict",
    # compression
    "CompressionConfig",
    "CompressionResult",
    "TurnRecord",
    "K_GRAYSCALE_BANDS",
    "compress_rounds",
    "resolve_config",
    "k_from_env",
    "mode_from_env",
    "should_enable",
    "validate_k_band",
    # interfaces
    "ArtifactPointer",
    "TaskState",
    "POINTER_KINDS",
    "expand_pointer",
    "load_task_state",
    "save_task_state",
]

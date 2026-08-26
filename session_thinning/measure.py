"""会话层薄化 · 步骤0：上下文构成测量拆分（A0 measure）

依据：《四妹-会话层薄化-20260817.md》§三 步骤0（只读，1 天内）：

    /context detail + /status 拆 982k 构成：transcript vs LMS 注入 vs 工具
    schema vs system prompt——产出「上下文构成表」，决定砍谁（预期 transcript
    占大头，但必须实测——这是全案第一份硬数据）

本模块 = 测量拆分工具：输入一份上下文构成样本（JSON 结构见 ContextSample，
四段 + 模型窗口），输出构成分析（各段占比 / 占用率 / 超窗诊断 / 建议裁剪
优先级）。纯 stdlib、纯函数，可单测。

核心洞察（§二.1）：薄化 = 换排序函数（时间序 → 相关度序），不是删历史。
本工具只回答「窗口里各来源占多少、超窗多少、先砍谁」——砍与不砍的决策在
方案文档 / runbook 中给出，本模块不做任何写入。

采样边界（重要）：本模块只在 rewrite-ws 内做「构成分析」；真实样本
（/context detail、/status、OpenClaw 日志）需在 OpenClaw 侧采集——
见同目录 runbook.md。
待 OpenClaw 集成点：/context detail 采样（会话层检视工具，环境在 OpenClaw
侧）——需主代理确认，不擅改环境。
"""

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Mapping, Optional, Tuple

#: 构成拆分的四段（与 OpenClaw 会话层上下文来源一一对应）
#: - transcript     ：会话历史重放（时间序；方案 §三 步骤1 的压缩对象）
#: - lms_injection ：LMS 六层注入（相关度序；占窗口大头，§二.2 B 段）
#: - tool_schema   ：工具 schema 常驻（模型无关可剪枝，§一.2 工具结果剪枝）
#: - system_prompt ：系统提示常驻（静态锚不变量，§二.2 A 段——最后考虑裁剪）
SEGMENT_KEYS: Tuple[str, ...] = (
    "transcript",
    "lms_injection",
    "tool_schema",
    "system_prompt",
)

#: 压力阈值（对齐 harness BasicCompactionEngine thresholdRatio=0.8，§一.2）：
#: 窗口占用率 > NEAR_THRESHOLD 视为接近压缩触发点；> 1.0 为超窗。
NEAR_THRESHOLD: float = 0.8

#: 裁剪策略说明（写进 diagnostics，便于审计「为什么先砍它」）
_TRIM_POLICY_NOTES = {
    "transcript": "时间序占满窗口的病根（§二.1）：优先压缩尾部为最近 K 轮原文"
                  "（§三 步骤1，K 灰度 50→30→20）",
    "lms_injection": "相关度序注入（§二.2 B 段）：收紧检索预算，不删历史",
    "tool_schema": "工具 schema 可模型无关剪枝（§一.2：保持 tool call/result "
                   "配对边界）",
    "system_prompt": "静态锚不变量（§二.2 A 段）——最后考虑，不做常规裁剪",
}


@dataclass(frozen=True)
class ContextSample:
    """上下文构成样本（步骤0 的输入数据 schema）。

    字段与 OpenClaw 侧采集值的对应关系见 runbook.md（采样需 OpenClaw 侧执行）。

    属性:
        transcript_chars: 会话历史（transcript 重放/压缩）字符数。
        lms_injection_chars: LMS 六层注入字符数。
        tool_schema_chars: 工具 schema 字符数。
        system_prompt_chars: 系统提示字符数。
        model_window: 路由模型上下文窗口（字符口径；OpenClaw 侧按 4 字符/token
            启发式换算，见 runbook）。
        total_chars: 实际窗口占用总字符数。缺省（None）时取四段之和；若显式给
            出且与四段之和不等，多出的部分记为 other（窗口 overhead）。
    """

    transcript_chars: int
    lms_injection_chars: int
    tool_schema_chars: int
    system_prompt_chars: int
    model_window: int
    total_chars: Optional[int] = None


@dataclass(frozen=True)
class SegmentStat:
    """单段的构成统计。"""

    key: str
    chars: int
    share: float  # 占总占用比例（0..1）
    window_occupancy: float  # 占模型窗口比例（0..1+，可 >1）


@dataclass(frozen=True)
class ContextAnalysis:
    """上下文构成分析（步骤0 的输出）。

    属性:
        sample: 输入样本。
        segments: 四段的统计（按 SEGMENT_KEYS 序）。
        computed_sum: 四段字符之和。
        total_chars: 实际总占用（= sample.total_chars 或四段之和）。
        other_chars: total_chars - computed_sum（可为负 → 诊断警告）。
        window_usage_ratio: 总占用 / 模型窗口。
        over_window: 是否超窗（window_usage_ratio > 1.0）。
        overflow_chars: 超出窗口的字符数（未超窗为 0）。
        pressure_level: "ok" | "near_threshold" | "over_window"。
        trim_priority: 建议裁剪优先级（段 key 序：按占用降序，
            system_prompt 恒为最后——静态锚不变量）。
        required_cut_chars: 超窗时需砍掉的字符数（未超窗为 0）。
        diagnostics: 人类可读诊断行（含裁剪建议与依据）。
    """

    sample: ContextSample
    segments: Tuple[SegmentStat, ...]
    computed_sum: int
    total_chars: int
    other_chars: int
    window_usage_ratio: float
    over_window: bool
    overflow_chars: int
    pressure_level: str
    trim_priority: Tuple[str, ...]
    required_cut_chars: int
    diagnostics: Tuple[str, ...] = field(default_factory=tuple)


def _validate_chars(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} 必须为 int，got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} 不能为负，got {value}")


def sample_from_mapping(mapping: Mapping) -> ContextSample:
    """从映射（JSON 兼容 dict）构造 ContextSample。

    接受的键（缺一不可，total_chars 可缺省）：
        transcript_chars / lms_injection_chars / tool_schema_chars /
        system_prompt_chars / model_window / [total_chars]

    参数:
        mapping: 样本映射，可直接来自 JSON 文件。

    返回:
        ContextSample 实例。

    异常:
        TypeError: 类型不合法（如字符串字符数）。
        ValueError: 数值不合法（负数、非正窗口）。
        KeyError: 缺少必需键。
    """
    missing = [k for k in (
        "transcript_chars", "lms_injection_chars", "tool_schema_chars",
        "system_prompt_chars", "model_window") if k not in mapping]
    if missing:
        raise KeyError(f"缺少必需字段: {missing}")
    for k in ("transcript_chars", "lms_injection_chars", "tool_schema_chars",
              "system_prompt_chars"):
        _validate_chars(mapping[k], k)
    _validate_chars(mapping["model_window"], "model_window")
    if mapping["model_window"] <= 0:
        raise ValueError(f"model_window 必须为正，got {mapping['model_window']}")
    total = mapping.get("total_chars")
    if total is not None:
        _validate_chars(total, "total_chars")
    return ContextSample(
        transcript_chars=mapping["transcript_chars"],
        lms_injection_chars=mapping["lms_injection_chars"],
        tool_schema_chars=mapping["tool_schema_chars"],
        system_prompt_chars=mapping["system_prompt_chars"],
        model_window=mapping["model_window"],
        total_chars=total,
    )


def make_sample_skeleton() -> Dict[str, object]:
    """生成一份可填写的样本骨架（模板）。

    供 OpenClaw 侧采集时填写（runbook.md 有逐字段说明）；
    采样需 OpenClaw 侧执行（待主代理确认）。
    """
    return {
        "transcript_chars": 0,
        "lms_injection_chars": 0,
        "tool_schema_chars": 0,
        "system_prompt_chars": 0,
        "total_chars": 0,  # 可省略：缺省 = 四段之和
        "model_window": 1000000,  # 路由模型窗口（字符口径）
        "_注释": "见 session_thinning/runbook.md：各字段在 OpenClaw 侧的采集来源",
    }


def analyze_context(sample_or_mapping) -> ContextAnalysis:
    """对样本做构成分析（步骤0 主入口）。

    参数:
        sample_or_mapping: ContextSample 实例或 JSON 兼容映射
            （自动经 sample_from_mapping 归一）。

    返回:
        ContextAnalysis：各段占比 / 占用率 / 超窗诊断 / 建议裁剪优先级。

    异常:
        TypeError / ValueError / KeyError: 样本不合法（见 sample_from_mapping）。
    """
    sample = (sample_or_mapping if isinstance(sample_or_mapping, ContextSample)
              else sample_from_mapping(sample_or_mapping))

    computed_sum = (sample.transcript_chars + sample.lms_injection_chars
                    + sample.tool_schema_chars + sample.system_prompt_chars)
    total = sample.total_chars if sample.total_chars is not None else computed_sum
    other = total - computed_sum
    window = sample.model_window

    if total > 0:
        shares = {k: chars / total for k, chars in _chars_map(sample).items()}
    else:
        shares = {k: 0.0 for k in SEGMENT_KEYS}
    occupancies = {k: chars / window for k, chars in _chars_map(sample).items()}

    segments = tuple(
        SegmentStat(key=k,
                    chars=_chars_map(sample)[k],
                    share=round(shares[k], 6),
                    window_occupancy=round(occupancies[k], 6))
        for k in SEGMENT_KEYS
    )

    usage = total / window
    over_window = usage > 1.0
    overflow = max(0, total - window)
    if over_window:
        pressure = "over_window"
    elif usage > NEAR_THRESHOLD:
        pressure = "near_threshold"
    else:
        pressure = "ok"

    # 建议裁剪优先级：按占用降序；system_prompt（静态锚不变量）恒为最后。
    ranked = [s.key for s in sorted(segments, key=lambda s: s.chars,
                                    reverse=True)]
    priority = tuple(k for k in ranked if k != "system_prompt") + ("system_prompt",)

    # 超窗时给出「至少砍到哪一段、砍多少」的可执行建议。
    required = overflow
    cuts: List[str] = []
    remaining = required
    for key in priority:
        if remaining <= 0:
            break
        seg_chars = _chars_map(sample)[key]
        take = min(seg_chars, remaining)
        cuts.append(f"    - {key}: 至少砍 {take} chars（≈{(take + 3) // 4} tokens，"
                    f"{_TRIM_POLICY_NOTES[key]}）")
        remaining -= take

    diagnostics = [
        f"[measure] 上下文构成：transcript={sample.transcript_chars} "
        f"lms_injection={sample.lms_injection_chars} "
        f"tool_schema={sample.tool_schema_chars} "
        f"system_prompt={sample.system_prompt_chars}（四段和 {computed_sum}，"
        f"总占用 {total}）",
        f"[measure] 窗口占用率 {usage:.1%}（窗口 {window} chars）"
        f"——{'超窗 ' + str(overflow) + ' chars' if over_window else '未超窗'}",
    ]
    if other != 0:
        diagnostics.append(
            f"[measure] 注意：total_chars 与四段之和相差 {other} chars"
            f"（窗口 overhead/未拆分来源）——采样口径需核实")
    if pressure == "over_window":
        diagnostics.append(
            f"[measure] 超窗诊断：需至少砍 {required} chars"
            f"（≈{(required + 3) // 4} tokens）回到窗口内；建议顺序：")
        diagnostics.extend(cuts)
    elif pressure == "near_threshold":
        diagnostics.append(
            f"[measure] 接近压缩阈值（>{NEAR_THRESHOLD:.0%}，对齐 dsh "
            f"thresholdRatio）：建议先做步骤1 K 轮压缩腾出余量")
    else:
        diagnostics.append("[measure] 窗口压力正常，暂无需裁剪")
    diagnostics.append(
        f"[measure] 裁剪优先级：{' > '.join(priority)}"
        f"（依据：占用降序 + 系统提示为静态锚不变量恒最后；"
        f"预期 transcript 占大头，但必须实测——§三 步骤0）")

    return ContextAnalysis(
        sample=sample,
        segments=segments,
        computed_sum=computed_sum,
        total_chars=total,
        other_chars=other,
        window_usage_ratio=round(usage, 6),
        over_window=over_window,
        overflow_chars=overflow,
        pressure_level=pressure,
        trim_priority=priority,
        required_cut_chars=required,
        diagnostics=tuple(diagnostics),
    )


def _chars_map(sample: ContextSample) -> Dict[str, int]:
    return {
        "transcript": sample.transcript_chars,
        "lms_injection": sample.lms_injection_chars,
        "tool_schema": sample.tool_schema_chars,
        "system_prompt": sample.system_prompt_chars,
    }


def sample_to_dict(sample: ContextSample) -> Dict[str, object]:
    """ContextSample → JSON 兼容 dict。"""
    return asdict(sample)


def analysis_to_dict(analysis: ContextAnalysis) -> Dict[str, object]:
    """ContextAnalysis → JSON 兼容 dict（步骤0 输出落盘格式）。"""
    return {
        "computed_sum": analysis.computed_sum,
        "total_chars": analysis.total_chars,
        "other_chars": analysis.other_chars,
        "window_usage_ratio": analysis.window_usage_ratio,
        "over_window": analysis.over_window,
        "overflow_chars": analysis.overflow_chars,
        "pressure_level": analysis.pressure_level,
        "trim_priority": list(analysis.trim_priority),
        "required_cut_chars": analysis.required_cut_chars,
        "segments": [
            {"key": s.key, "chars": s.chars, "share": s.share,
             "window_occupancy": s.window_occupancy}
            for s in analysis.segments
        ],
        "diagnostics": list(analysis.diagnostics),
    }


def _cli(argv: Optional[List[str]] = None) -> int:
    """命令行入口：python -m session_thinning.measure [--file sample.json] [--skeleton]。

    样例:
        python -m session_thinning.measure --skeleton          # 输出可填模板
        python -m session_thinning.measure --file sample.json  # 输出构成分析
    """
    parser = argparse.ArgumentParser(
        prog="python -m session_thinning.measure",
        description="会话层薄化步骤0：上下文构成分析（输入样本 JSON → 输出分析 JSON）")
    parser.add_argument("--file", help="样本 JSON 文件路径")
    parser.add_argument("--skeleton", action="store_true",
                        help="输出可填写的样本骨架（供 OpenClaw 侧采集）")
    args = parser.parse_args(argv)

    if args.skeleton:
        print(json.dumps(make_sample_skeleton(), ensure_ascii=False, indent=2))
        return 0
    if args.file is None:
        parser.error("需要 --file 或 --skeleton")
    with open(args.file, "r", encoding="utf-8") as fh:
        mapping = json.load(fh)
    sample = sample_from_mapping(mapping)
    analysis = analyze_context(sample)
    print(json.dumps(analysis_to_dict(analysis), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
活体记忆系统 - 对照实验脚本 (Controlled Experiment)
=====================================================

目的
----
证明 LLM 正确回忆用户信息（张三、蓝色、小橘）是因为我们的记忆系统
（episodic buffer / 情景记忆缓冲区），而不是 LLM 自己的上下文能力。

背景
----
LLM Bridge (bridge/llm_bridge.py) 的 query() 每次调用只发送 system + user
两条消息，不发送对话历史。因此理论上 LLM 无法通过自己的上下文窗口回忆
之前的内容——它只能依赖系统在 system prompt 中注入的"记忆 context"。

记忆 context 由 Decoder 生成，其中包含三段：
  1. [记忆context]      —— 激活态的抽象描述（节点编号/强度，无语义内容）
  2. 长时记忆检索       —— EMA 潜变量的抽象维度信息（无语义文本）
  3. 历史记忆           —— episodic buffer 检索到的原始对话文本（语义内容！）

只有第 3 段（"历史记忆"）包含 "张三/蓝色/小橘" 这样的语义文本。
因此，禁用 episodic buffer 后，LLM 的 context 中将不包含这些信息，
LLM 应当无法正确回忆。

实验设计
--------
  实验A（对照组）：禁用 episodic buffer
    - 用 PretrainedEmbedder（预训练语义嵌入）
    - 每轮 process_turn 后清空 loop.memory._episodic_buffer
    - 重新解码 context 时显式不传 episodic_texts
    - 确保 LLM 查询时的 context 不含"历史记忆"段落

  实验B（实验组）：正常启用 episodic buffer
    - 同样的配置和对话
    - 正常启用 episodic buffer，"历史记忆"段落正常注入

预期结果
--------
  - 实验A：第 4/5/6/8 轮 LLM 无法回忆张三/蓝色/小橘（回答"不知道/没有存储"）
  - 实验B：第 4/5/6/8 轮 LLM 正确回忆
  => 证明是记忆系统（episodic buffer）在起作用，而非 LLM 自身上下文

用法
----
    python e2e_controlled_experiment.py
    （工作目录：z:\\QH\\AI专用\\活体记忆系统）
"""

import os
import sys

# ---------------------------------------------------------------------------
# sys.path 处理：将项目根目录（本脚本所在目录）加入 Python 路径，
# 确保 core / bridge / runtime 等包可被正确导入。
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 修复 Windows 控制台中文输出
import io
try:
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import torch
from runtime.loop import LivingMemoryLoop
from runtime.config import default_config
from core.sensory.embedder import PretrainedEmbedder


# ===========================================================================
# 常量配置
# ===========================================================================

# 预训练模型本地路径（modelscope 下载缓存）
_PRETRAINED_MODEL_PATH = (
    r"C:\Users\dandan\.cache\modelscope\models"
    r"\sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
    r"\snapshots\master"
)

# DeepSeek API 配置
DEEPSEEK_API_KEY = "sk-d91a6339112040a98c6f0617e6142307"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

# 8 轮对话内容
#   1-3 轮：引入个人信息（记忆形成）
#   4-6 轮：检索之前的信息（记忆检索 —— 重点对比）
#   7-8 轮：开放对话与总结（体现记忆 context）
CONVERSATIONS = [
    "我叫张三，今天开始学习AI",
    "我最喜欢的颜色是蓝色",
    "我养了一只猫叫小橘",
    "你还记得我叫什么吗？",
    "我的猫叫什么？",
    "我喜欢什么颜色？",
    "告诉我一些关于AI的知识",
    "总结一下我们今天聊的内容",
]

# 重点对比轮次（检索与总结轮）
FOCUS_TURNS = [4, 5, 6, 8]

# 分隔线宽度
_SEP = 60


# ===========================================================================
# 配置构建
# ===========================================================================
def build_config() -> dict:
    """构建实验配置：基于 default_config 覆盖指定参数。

    使用 PretrainedEmbedder（预训练语义嵌入）+ DeepSeek API。
    两个实验共用同一份配置，确保唯一变量是 episodic buffer 的开关。

    返回:
        完整配置字典，可直接传给 LivingMemoryLoop。
    """
    config = default_config()

    # --- 系统配置 ---
    config['num_nodes'] = 256
    config['input_dim'] = 64
    # 预训练 embedder 输出幅度约 0.42，阈值适配
    config['activation_threshold'] = 0.02
    # 关闭自动快照（实验无需持久化）
    config['auto_snapshot'] = False

    # --- Embedder：预训练语义嵌入 ---
    config['embedder'] = PretrainedEmbedder(
        dim=64, model_name=_PRETRAINED_MODEL_PATH)

    # --- DeepSeek API 配置 ---
    # system_prompt 明确告知 LLM：只能依赖记忆 context，看不到对话历史。
    # 若 context 中没有相关信息，应如实说明"没有存储"。
    config['llm_api'] = {
        'base_url': DEEPSEEK_BASE_URL,
        'api_key': DEEPSEEK_API_KEY,
        'model': DEEPSEEK_MODEL,
        'temperature': 0.7,
        'max_tokens': 500,
        'timeout': 30,
        'max_retries': 3,
        'system_prompt': (
            '你是一个有记忆能力的AI助手。系统会为你提供海马体的记忆context，'
            '其中可能包含激活节点信息、长时记忆检索结果和历史记忆段落。'
            '请结合记忆context和当前用户输入来回答问题。'
            '重要规则：你无法看到此前的对话历史，只能依赖系统提供的记忆context。'
            '如果记忆context的"历史记忆"段落中有相关信息，请利用它来回答；'
            '如果记忆context中没有相关信息，请如实回答"我没有存储相关信息"，'
            '不要猜测或编造。'
        ),
    }

    return config


# ===========================================================================
# LLM 查询（带容错）
# ===========================================================================
def query_llm_safe(loop: LivingMemoryLoop, user_input: str,
                  context: str) -> str:
    """安全查询 LLM，处理调用失败的情况。

    LLMBridge 内部已有重试（指数退避）和超时机制，
    此函数在其之上捕获最终的 RuntimeError，保证实验不中断。

    参数:
        loop: LivingMemoryLoop 实例。
        user_input: 用户输入文本。
        context: 记忆 context 文本（由 Decoder 生成）。

    返回:
        LLM 回答文本；失败时返回错误提示文本。
    """
    if loop.bridge is None:
        return "[错误: 未配置 LLM Bridge]"
    try:
        response = loop.bridge.query(user_input, context)
        return response
    except Exception as e:
        return f"[LLM调用失败: {e}]"


# ===========================================================================
# 回忆正确性判定
# ===========================================================================
def check_recall(turn: int, response: str):
    """检查 LLM 回答是否包含预期的用户信息。

    参数:
        turn: 对话轮次（1-8）。
        response: LLM 回答文本。

    返回:
        (expected, found, label) 三元组：
          - expected: 预期回忆的关键词
          - found: 是否在回答中找到
          - label: 判定标签（"正确回忆" / "未能回忆" / "N/A"）
    """
    if turn == 4:  # 名字
        keyword = "张三"
        found = keyword in response
        return (keyword, found,
                "正确回忆" if found else "未能回忆")
    elif turn == 5:  # 猫名
        keyword = "小橘"
        found = keyword in response
        return (keyword, found,
                "正确回忆" if found else "未能回忆")
    elif turn == 6:  # 颜色
        keyword = "蓝色"
        found = keyword in response
        return (keyword, found,
                "正确回忆" if found else "未能回忆")
    elif turn == 8:  # 总结（应包含全部三项）
        keywords = ["张三", "小橘", "蓝色"]
        found_all = all(k in response for k in keywords)
        return ("张三+小橘+蓝色", found_all,
                "正确回忆" if found_all else "未能回忆")
    return ("N/A", False, "N/A")


# ===========================================================================
# 运行单组实验
# ===========================================================================
def run_experiment(label: str, disable_episodic: bool):
    """运行一组实验，返回每轮的结果记录。

    参数:
        label: 实验标签（用于打印）。
        disable_episodic: True=实验A（禁用 episodic buffer），
                         False=实验B（正常启用）。

    返回:
        results: 列表，每个元素是一个字典，包含：
            - turn: 轮次编号
            - user: 用户输入
            - context: 发送给 LLM 的记忆 context
            - response: LLM 回答
            - has_episodic: context 中是否包含"历史记忆"段落
    """
    print()
    print('=' * _SEP)
    print(f"  {label}")
    print('=' * _SEP)

    # 每组实验使用全新的 LivingMemoryLoop 实例（独立状态）
    config = build_config()
    loop = LivingMemoryLoop(config)

    results = []
    prev_response = ""

    for i, user_input in enumerate(CONVERSATIONS, 1):
        # ---------------------------------------------------------------
        # 1. process_turn：编码输入 -> FEP推断/学习 -> 调整目的
        #    -> 记忆更新/巩固 -> 检索长时记忆 -> 检索情景记忆 -> 解码 context
        #    记忆系统正常执行（学习、长时记忆均不受影响）
        # ---------------------------------------------------------------
        context = loop.process_turn(user_input, prev_response)

        # ---------------------------------------------------------------
        # 2. 实验A关键操作：禁用 episodic buffer
        # ---------------------------------------------------------------
        if disable_episodic:
            # (a) 清空情景记忆缓冲区 —— 禁用 episodic buffer
            #     这使得后续轮次的 recall_episodic 检索不到任何历史文本
            loop.memory._episodic_buffer.clear()

            # (b) 重新解码 context，显式不传 episodic_texts
            #     确保 LLM 查询时的 context 不含"历史记忆"段落
            #     （长时记忆 recalled 保留，隔离的唯一变量是 episodic buffer）
            recalled = loop.memory.recall(loop.last_activation.state)
            context_for_llm = loop.decoder.decode(
                loop.last_activation,
                recalled_memory=recalled,
                episodic_texts=None,  # 关键：不注入情景文本
            )
        else:
            # 实验B：直接使用 process_turn 返回的 context（含"历史记忆"）
            context_for_llm = context

        # ---------------------------------------------------------------
        # 3. 查询 LLM（只发送 system + user 两条消息，不含对话历史）
        # ---------------------------------------------------------------
        llm_response = query_llm_safe(loop, user_input, context_for_llm)

        # 记录结果
        has_episodic = "历史记忆" in context_for_llm
        results.append({
            'turn': i,
            'user': user_input,
            'context': context_for_llm,
            'response': llm_response,
            'has_episodic': has_episodic,
        })

        # 打印本轮详情
        print()
        print('-' * _SEP)
        print(f"第 {i} 轮")
        print(f"  用户: {user_input}")
        print(f"  [context 含历史记忆段落]: {has_episodic}")
        # 打印 context（截断显示）
        ctx_display = context_for_llm if len(context_for_llm) <= 200 \
            else context_for_llm[:200] + " ..."
        print(f"  [记忆context]: {ctx_display}")
        print(f"  [助手回答]: {llm_response}")
        print('-' * _SEP)

        # 记录 LLM 输出供下一轮编码使用（失败时不送入记忆系统）
        if not llm_response.startswith('[LLM') and \
           not llm_response.startswith('[错误'):
            prev_response = llm_response
        else:
            prev_response = ""

    return results


# ===========================================================================
# 对比表格输出
# ===========================================================================
def print_comparison(results_a, results_b):
    """输出两组实验的对比表格，重点对比第 4/5/6/8 轮。

    参数:
        results_a: 实验A（无记忆）的结果列表。
        results_b: 实验B（有记忆）的结果列表。
    """
    print()
    print('=' * _SEP)
    print("  对比结果（重点：第 4 / 5 / 6 / 8 轮）")
    print('=' * _SEP)

    # --- 1. context 段落验证：证明实验操纵有效 ---
    print()
    print('【1. 实验操纵验证：context 是否含"历史记忆"段落】')
    print("-" * _SEP)
    header = (f"{'轮次':<4} | {'用户提问':<28} | "
              f"{'实验A(无记忆)':<16} | {'实验B(有记忆)':<16}")
    print(header)
    print("-" * _SEP)
    for turn in FOCUS_TURNS:
        ra = results_a[turn - 1]
        rb = results_b[turn - 1]
        a_mark = "有历史记忆" if ra['has_episodic'] else "无历史记忆"
        b_mark = "有历史记忆" if rb['has_episodic'] else "无历史记忆"
        user_short = ra['user'][:26] + ".." if len(ra['user']) > 28 \
            else ra['user']
        print(f"{turn:<4} | {user_short:<28} | "
              f"{a_mark:<16} | {b_mark:<16}")
    print("-" * _SEP)

    # --- 2. LLM 回答对比 ---
    print()
    print("【2. LLM 回答对比（是否正确回忆用户信息）】")
    print("-" * _SEP)
    for turn in FOCUS_TURNS:
        ra = results_a[turn - 1]
        rb = results_b[turn - 1]

        expected, found_a, label_a = check_recall(turn, ra['response'])
        _, found_b, label_b = check_recall(turn, rb['response'])

        print()
        print(f"  第 {turn} 轮: {ra['user']}")
        print(f"  预期回忆: {expected}")
        print(f"  --- 实验A（无记忆 / 禁用 episodic buffer）---")
        print(f"      判定: [{label_a}]")
        resp_a = ra['response'].replace('\n', ' ')
        resp_a = resp_a if len(resp_a) <= 120 else resp_a[:120] + "..."
        print(f"      回答: {resp_a}")
        print(f"  --- 实验B（有记忆 / 启用 episodic buffer）---")
        print(f"      判定: [{label_b}]")
        resp_b = rb['response'].replace('\n', ' ')
        resp_b = resp_b if len(resp_b) <= 120 else resp_b[:120] + "..."
        print(f"      回答: {resp_b}")
        print(f"  {'=' * 50}")

    # --- 3. 汇总表格 ---
    print()
    print("【3. 汇总表格】")
    print("-" * _SEP)
    header = (f"{'轮次':<4} | {'提问':<26} | {'预期':<18} | "
              f"{'实验A(无记忆)':<14} | {'实验B(有记忆)':<14}")
    print(header)
    print("-" * _SEP)
    for turn in FOCUS_TURNS:
        ra = results_a[turn - 1]
        rb = results_b[turn - 1]
        expected, _, _ = check_recall(turn, "")
        _, found_a, label_a = check_recall(turn, ra['response'])
        _, found_b, label_b = check_recall(turn, rb['response'])

        user_short = ra['user'][:24] + ".." if len(ra['user']) > 26 \
            else ra['user']
        a_flag = "Y 正确回忆" if found_a else "X 未能回忆"
        b_flag = "Y 正确回忆" if found_b else "X 未能回忆"
        print(f"{turn:<4} | {user_short:<26} | {expected:<18} | "
              f"{a_flag:<14} | {b_flag:<14}")
    print("-" * _SEP)


# ===========================================================================
# 结论输出
# ===========================================================================
def print_conclusion(results_a, results_b):
    """根据实验结果输出结论。

    参数:
        results_a: 实验A（无记忆）的结果列表。
        results_b: 实验B（有记忆）的结果列表。
    """
    print()
    print('=' * _SEP)
    print("  实验结论")
    print('=' * _SEP)

    # 统计各重点轮次的回忆情况
    a_success = 0
    b_success = 0
    total = len(FOCUS_TURNS)

    for turn in FOCUS_TURNS:
        _, found_a, _ = check_recall(turn, results_a[turn - 1]['response'])
        _, found_b, _ = check_recall(turn, results_b[turn - 1]['response'])
        if found_a:
            a_success += 1
        if found_b:
            b_success += 1

    print()
    print(f"  实验A（禁用 episodic buffer）正确回忆轮次: "
          f"{a_success}/{total}")
    print(f"  实验B（启用 episodic buffer）正确回忆轮次: "
          f"{b_success}/{total}")
    print()

    if a_success == 0 and b_success >= 2:
        print("  ==> 结论：LLM 正确回忆用户信息（张三、蓝色、小橘）")
        print("      是因为我们的记忆系统（episodic buffer）在起作用，")
        print("      而非 LLM 自身的上下文窗口能力。")
        print()
        print("      证据链：")
        print("      1. LLM Bridge 每次只发送 system + user 两条消息，")
        print("         不发送对话历史 -> LLM 无法依赖自身上下文回忆。")
        print("      2. 实验A 禁用 episodic buffer 后，context 不含'历史记忆'段落，")
        print("         LLM 回答'没有存储相关信息' -> 证明 LLM 自身无法回忆。")
        print("      3. 实验B 启用 episodic buffer 后，context 含'历史记忆'段落，")
        print("         LLM 正确回忆 -> 证明记忆系统提供了回忆能力。")
        print("      4. 唯一变量是 episodic buffer 的开关 -> 因果关系成立。")
    elif b_success > a_success:
        print("  ==> 结论：实验B（有记忆）的回忆表现明显优于实验A（无记忆），")
        print("      证明记忆系统（episodic buffer）对 LLM 回忆起关键作用。")
    else:
        print("  ==> 结论：两组实验表现接近，记忆系统的作用不明显。")
        print("      （可能原因：LLM 推测性回答、context 泄漏等，需进一步排查）")

    print()
    print('=' * _SEP)


# ===========================================================================
# 主函数
# ===========================================================================
def main() -> None:
    """对照实验主流程。"""
    # --- 检查依赖 ---
    try:
        import openai  # noqa: F401
    except ImportError:
        print("错误: 未安装 openai 库。请运行: pip install openai")
        sys.exit(1)

    # --- 打印实验头部 ---
    print('=' * _SEP)
    print("  活体记忆系统 - 对照实验 (Controlled Experiment)")
    print("  证明 LLM 回忆用户信息依赖 episodic buffer，而非自身上下文")
    print('=' * _SEP)
    print(f"  LLM 模型:     {DEEPSEEK_MODEL}")
    print(f"  API 地址:     {DEEPSEEK_BASE_URL}")
    print(f"  Embedder:     PretrainedEmbedder (paraphrase-multilingual-MiniLM)")
    print(f"  对话轮数:     {len(CONVERSATIONS)}")
    print(f"  重点对比轮次: {FOCUS_TURNS}")
    print(f"  LLM消息机制:  每次只发 system+user，不发对话历史")
    print('=' * _SEP)

    # --- 运行实验A（对照组：禁用 episodic buffer）---
    results_a = run_experiment(
        "实验A（对照组）：禁用 episodic buffer",
        disable_episodic=True)

    # --- 运行实验B（实验组：正常启用 episodic buffer）---
    results_b = run_experiment(
        "实验B（实验组）：正常启用 episodic buffer",
        disable_episodic=False)

    # --- 输出对比表格 ---
    print_comparison(results_a, results_b)

    # --- 输出结论 ---
    print_conclusion(results_a, results_b)

    print()
    print("对照实验完成。")


if __name__ == '__main__':
    main()

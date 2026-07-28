#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
活体记忆系统 - 端到端测试脚本 (E2E Test)
=========================================

接入真实 LLM (DeepSeek API) 进行端到端测试，展示活体记忆系统的
完整工作流程：记忆形成 -> 记忆检索 -> LLM 回答体现记忆 context。

测试流程：
  1. 使用 default_config 初始化 LivingMemoryLoop（覆盖指定参数）
  2. 进行 8 轮对话，前 3 轮引入信息（记忆形成），后续轮次检索记忆
  3. 每轮展示记忆系统状态（熵/惊讶/coherence/precision/翻转）和 context
  4. 全部结束后输出系统演化总结并保存快照

用法:
    python e2e_test.py --api-key sk-xxx
    或
    $env:DEEPSEEK_API_KEY="sk-xxx"; python e2e_test.py
"""

import os
import sys
import argparse

# ---------------------------------------------------------------------------
# sys.path 处理：将项目根目录（本脚本所在目录）加入 Python 路径，
# 确保 core / bridge / runtime 等包可被正确导入。
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from runtime.loop import LivingMemoryLoop
from runtime.config import default_config


# ===========================================================================
# 对话设计
# ===========================================================================
# 前 3 轮：引入个人信息（记忆形成）
# 第 4-6 轮：检索之前的信息（记忆检索）
# 第 7-8 轮：开放对话与总结（体现记忆 context）
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

# 分隔线长度（方便截图）
_SEP_WIDTH = 40


# ===========================================================================
# 配置构建
# ===========================================================================
def build_config(api_key: str) -> dict:
    """构建测试配置：基于 default_config 覆盖指定参数。

    参数:
        api_key: DeepSeek API 密钥。

    返回:
        完整配置字典，可直接传给 LivingMemoryLoop。
    """
    config = default_config()

    # --- 系统配置覆盖 ---
    config['num_nodes'] = 256
    config['input_dim'] = 64
    # N4 解耦后的参数：适配 SimpleEmbedder 的小信号（embedding 方差 0.1）
    config['activation_threshold'] = 0.05
    config['auto_snapshot'] = True
    config['auto_snapshot_interval'] = 10
    config['snapshot_dir'] = './snapshots'

    # --- DeepSeek API 配置 ---
    config['llm_api'] = {
        'base_url': 'https://api.deepseek.com/v1',
        'api_key': api_key,
        'model': 'deepseek-chat',
        'temperature': 0.7,
        'max_tokens': 500,
        'timeout': 30,
        'max_retries': 3,
        'system_prompt': (
            '你是一个有记忆能力的AI助手。系统会为你提供海马体的记忆context，'
            '其中包含激活节点信息和长时记忆检索结果，'
            '这代表了你对此前对话的记忆状态。'
            '请结合记忆context和当前用户输入来回答问题。'
            '如果记忆context中有相关信息，请尝试利用它来回答；'
            '同时保持自然流畅的对话。'
        ),
    }

    return config


# ===========================================================================
# 输出格式化
# ===========================================================================
def print_turn(n: int, user_input: str, context: str,
               llm_response: str, loop: LivingMemoryLoop) -> None:
    """格式化输出一轮对话（方便截图）。

    参数:
        n: 轮次编号（从 1 开始）。
        user_input: 用户输入文本。
        context: decoder 输出的记忆 context 文本。
        llm_response: LLM 回答文本。
        loop: LivingMemoryLoop 实例（用于读取当前状态）。
    """
    activation = loop.last_activation
    purpose = loop.purpose

    print()
    print('=' * _SEP_WIDTH)
    print(f"第 {n} 轮对话")
    print('-' * _SEP_WIDTH)
    print(f"用户: {user_input}")
    print("[记忆系统状态]")
    print(f"  熵: {activation.entropy:.4f} | "
          f"惊讶: {activation.surprise:.4f} | "
          f"coherence: {purpose.coherence:.4f}")
    print(f"  precision均值: {float(purpose.sensory_precision.mean()):.4f} | "
          f"翻转次数: {purpose.flip_count}")
    print("[记忆context]")
    print(f"  {context}")
    print('-' * _SEP_WIDTH)
    print(f"助手(LLM): {llm_response}")
    print('=' * _SEP_WIDTH)


# ===========================================================================
# LLM 查询（带容错）
# ===========================================================================
def query_llm_safe(loop: LivingMemoryLoop, user_input: str,
                   context: str) -> str:
    """安全查询 LLM，处理调用失败的情况。

    LLMBridge 内部已有重试（指数退避）和超时机制，
    此函数在其之上捕获最终的 RuntimeError，保证测试不中断。

    参数:
        loop: LivingMemoryLoop 实例。
        user_input: 用户输入文本。
        context: 记忆 context 文本。

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
# 测试总结
# ===========================================================================
def print_summary(loop: LivingMemoryLoop,
                  precision_history: list,
                  coherence_history: list,
                  entropy_history: list,
                  surprise_history: list,
                  initial_precision_mean: float) -> None:
    """输出全部对话结束后的系统演化总结。

    参数:
        loop: LivingMemoryLoop 实例。
        precision_history: 每轮 precision 均值列表。
        coherence_history: 每轮 coherence 列表。
        entropy_history: 每轮熵列表。
        surprise_history: 每轮惊讶度列表。
        initial_precision_mean: 初始 precision 均值。
    """
    print()
    print('=' * 50)
    print("测试总结")
    print('=' * 50)

    # J 矩阵 L2 范数（被塑形的程度）
    j_norm = float(loop.attractor.J.norm().item())
    print(f"\nJ矩阵 L2 范数（被塑形的程度）: {j_norm:.6f}")

    # 长时记忆 L2 范数
    long_term_norm = float(loop.memory.long_term_latent.norm().item())
    print(f"长时记忆 L2 范数: {long_term_norm:.6f}")

    # precision 演化情况
    final_precision_mean = (precision_history[-1]
                            if precision_history else 0.0)
    print(f"\nprecision 演化情况:")
    print(f"  初始均值: {initial_precision_mean:.6f}")
    print(f"  最终均值: {final_precision_mean:.6f}")
    print(f"  变化量:   {final_precision_mean - initial_precision_mean:+.6f}")
    print(f"  逐轮轨迹: {' -> '.join(f'{p:.4f}' for p in precision_history)}")

    # coherence 变化
    print(f"\ncoherence 变化:")
    print(f"  初始: 1.0000")
    if coherence_history:
        print(f"  最终: {coherence_history[-1]:.4f}")
    print(f"  逐轮轨迹: {' -> '.join(f'{c:.4f}' for c in coherence_history)}")

    # 元目的翻转次数
    print(f"\n元目的翻转次数: {loop.purpose.flip_count}")

    # 熵与惊讶度演化
    print(f"\n熵 演化轨迹: {' -> '.join(f'{e:.4f}' for e in entropy_history)}")
    print(f"惊讶度演化轨迹: {' -> '.join(f'{s:.4f}' for s in surprise_history)}")

    # 短时记忆 L2 范数（补充信息）
    short_term_norm = float(loop.memory.short_term_latent.norm().item())
    print(f"\n短时记忆 L2 范数: {short_term_norm:.6f}")
    print(f"对话总轮数: {loop.turn_count}")


# ===========================================================================
# 快照保存
# ===========================================================================
def save_snapshot(loop: LivingMemoryLoop) -> str:
    """保存快照到 './snapshots/e2e_test.pt'。

    参数:
        loop: LivingMemoryLoop 实例。

    返回:
        快照文件的绝对路径。
    """
    snapshot_dir = './snapshots'
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_path = os.path.join(snapshot_dir, 'e2e_test.pt')
    try:
        loop.save_state(snapshot_path)
        abs_path = os.path.abspath(snapshot_path)
        print(f"\n快照已保存: {abs_path}")
        return abs_path
    except Exception as e:
        print(f"\n快照保存失败: {e}")
        return ''


# ===========================================================================
# 主函数
# ===========================================================================
def main() -> None:
    """端到端测试主流程。"""
    # --- 解析命令行参数 ---
    parser = argparse.ArgumentParser(
        description='活体记忆系统端到端测试 (DeepSeek API)')
    parser.add_argument(
        '--api-key', type=str, default=None,
        help='DeepSeek API Key（也可通过环境变量 DEEPSEEK_API_KEY 提供）')
    args = parser.parse_args()

    # --- 获取 API Key ---
    api_key = args.api_key or os.environ.get('DEEPSEEK_API_KEY', '')
    if not api_key:
        print("错误: 未提供 DeepSeek API Key。")
        print("请通过 --api-key 参数或 DEEPSEEK_API_KEY 环境变量提供。")
        print("示例:")
        print("  python e2e_test.py --api-key sk-xxx")
        print("  $env:DEEPSEEK_API_KEY='sk-xxx'; python e2e_test.py")
        sys.exit(1)

    # --- 检查 openai 库是否安装 ---
    try:
        import openai  # noqa: F401
    except ImportError:
        print("错误: 未安装 openai 库。")
        print("请运行: pip install openai")
        sys.exit(1)

    # --- 构建配置 ---
    config = build_config(api_key)

    # --- 打印测试头部 ---
    print('=' * 50)
    print("活体记忆系统 - 端到端测试 (DeepSeek API)")
    print('=' * 50)
    print(f"LLM 模型:    {config['llm_api']['model']}")
    print(f"API 地址:    {config['llm_api']['base_url']}")
    print(f"节点数:      {config['num_nodes']}")
    print(f"输入维度:    {config['input_dim']}")
    print(f"激活阈值:    {config['activation_threshold']}")
    print(f"自动快照:    每 {config['auto_snapshot_interval']} 轮")
    print(f"对话轮数:    {len(CONVERSATIONS)}")
    print('=' * 50)

    # --- 确保快照目录存在（auto_snapshot 可能需要）---
    os.makedirs(config['snapshot_dir'], exist_ok=True)

    # --- 初始化记忆系统 ---
    loop = LivingMemoryLoop(config)

    # --- 记录初始状态与演化轨迹 ---
    initial_precision_mean = float(loop.purpose.get_precision().mean())
    precision_history: list = []
    coherence_history: list = []
    entropy_history: list = []
    surprise_history: list = []

    # --- 多轮对话 ---
    prev_response = ""
    for i, user_input in enumerate(CONVERSATIONS, 1):
        # 1. 处理本轮：编码输入（含上一轮 LLM 输出）-> 推断 -> 学习
        #    -> 调整目的 -> 记忆更新/巩固 -> 检索长时记忆 -> 解码 context
        context = loop.process_turn(user_input, prev_response)

        # 2. 捕获当前状态（用于输出与轨迹记录）
        activation = loop.last_activation
        precision_history.append(float(loop.purpose.sensory_precision.mean()))
        coherence_history.append(loop.purpose.coherence)
        entropy_history.append(activation.entropy)
        surprise_history.append(activation.surprise)

        # 3. 查询 LLM（使用当前轮的 memory context）
        llm_response = query_llm_safe(loop, user_input, context)

        # 4. 格式化输出本轮
        print_turn(i, user_input, context, llm_response, loop)

        # 5. 记录 LLM 输出供下一轮编码使用
        #    （LLM 失败时不将错误信息送入记忆系统）
        if not llm_response.startswith('[LLM调用失败') and \
           not llm_response.startswith('[错误'):
            prev_response = llm_response
        else:
            prev_response = ""

    # --- 输出总结 ---
    print_summary(loop, precision_history, coherence_history,
                  entropy_history, surprise_history, initial_precision_mean)

    # --- 保存快照 ---
    save_snapshot(loop)

    print()
    print('=' * 50)
    print("端到端测试完成")
    print('=' * 50)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""活体记忆系统 — 记忆对话效果测试

用 PretrainedEmbedder + DeepSeek API 进行8轮对话，
展示记忆形成 -> 检索 -> 回忆的完整链路。
"""
import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ.setdefault("DEEPSEEK_API_KEY", "sk-d91a6339112040a98c6f0617e6142307")

from api.config import get_api_config
from runtime.loop import LivingMemoryLoop

config = get_api_config()
config['num_nodes'] = 128
config['input_dim'] = 64
config['auto_snapshot'] = True
config['auto_snapshot_interval'] = 10
config['snapshot_dir'] = os.path.join(_PROJECT_ROOT, 'snapshots')
# 启用元可塑性
config['meta_enabled'] = True

loop = LivingMemoryLoop(config)

print("=" * 70)
print("  活体记忆系统 — 记忆对话效果测试")
print("=" * 70)
print()
print("系统配置:")
print("  网络: {}节点 / {}维".format(loop.attractor.num_nodes, loop.attractor.input_dim))
print("  设备: {}".format(loop.attractor.device))
print("  Embedder: {}".format(type(loop.embedder).__name__))
print("  LLM: {}".format(config['llm_api']['model'] if config.get('llm_api') else '未配置'))
print("  元可塑性: {}".format('启用' if loop.meta else '关闭'))
print()

# 8轮对话：前3轮引入信息，4-6轮检索，7-8轮开放
conversations = [
    ("我叫张三，今天开始学习AI", "你好张三！AI是个很有趣的领域，你打算从哪里开始学？"),
    ("我最喜欢的颜色是蓝色", "蓝色确实很宁静，像大海和天空的颜色。"),
    ("我养了一只猫叫小橘", "小橘是个可爱的名字！它是什么品种的猫？"),
    ("你还记得我叫什么名字吗？", ""),
    ("我的猫叫什么名字？", ""),
    ("我喜欢什么颜色？", ""),
    ("结合我们刚才聊的内容，给我总结一下", ""),
    ("你觉得AI会有记忆吗？就像你记住我的信息一样", ""),
]

print("-" * 70)
print("对话开始")
print("-" * 70)

for i, (user_input, llm_output) in enumerate(conversations):
    print()
    print("[第{}轮]".format(i + 1))
    print("  用户: {}".format(user_input))

    # 处理对话
    memory_context = loop.process_turn(user_input, llm_output)

    # 系统状态
    status = loop.get_status()
    entropy = status.get('last_entropy', 0)
    surprise = status.get('last_surprise', 0)
    coherence = status.get('purpose_coherence', 0)
    epi_count = len(loop.memory.episodic_buffer) if hasattr(loop.memory, 'episodic_buffer') else 0

    print("  [记忆状态] 熵={:.3f} 惊讶={:.4f} coherence={:.3f} 情景记忆={}条".format(
        entropy, surprise, coherence, epi_count))

    # 如果有LLM输出，打印
    if llm_output:
        print("  助手(上轮): {}".format(llm_output))

    # 打印记忆context（截取关键部分）
    print("  [记忆context]")
    # 逐行打印context，缩进
    for line in memory_context.split('\n'):
        if line.strip():
            print("    {}".format(line.strip()))

    # 第4-6轮（检索轮）特别标注
    if i in [3, 4, 5]:
        print("  >>> 这是记忆检索轮，系统应该能从情景记忆中找到相关信息 <<<")

# 最终统计
print()
print("=" * 70)
print("  最终统计")
print("=" * 70)
status = loop.get_status()
print("  总轮次: {}".format(status.get('turn_count')))
print("  熵比: {:.3f}".format(status.get('entropy_ratio', 0)))
print("  coherence: {:.3f}".format(status.get('purpose_coherence', 0)))
print("  precision: 均值={:.4f} 标准差={:.4f}".format(
    status.get('precision_mean', 0), status.get('precision_std', 0)))

if hasattr(loop.memory, 'episodic_buffer'):
    buf = loop.memory.episodic_buffer
    print()
    print("  情景记忆缓冲区 ({}条):".format(len(buf)))
    for e in buf:
        src = getattr(e, 'source', '?')
        txt = e.text[:70]
        print("    [turn={}] surprise={:.4f} src={} | {}".format(
            e.turn, e.surprise, src, txt))

# 做梦
print()
print("=" * 70)
print("  做梦测试 (10步)")
print("=" * 70)
dream_result = loop.dream(n_steps=10)
print("  做梦步数: {}".format(dream_result.get('steps', 0)))
print("  快照保存: {}".format(dream_result.get('snapshot_saved', False)))

print()
print("  做梦后状态:")
status = loop.get_status()
print("  熵: {:.3f}".format(status.get('last_entropy', 0)))
print("  coherence: {:.3f}".format(status.get('purpose_coherence', 0)))

print()
print("=" * 70)
print("  测试完成")
print("=" * 70)

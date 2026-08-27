"""C2 必做验证：purpose_coherence 演化曲线对比（C2 前后各 ≥500 轮）。

输出 500 轮的 purpose_coherence 均值/波动、precision 均值/std，
用于确认 C2 后目的层无异常漂移。
"""
import statistics

from runtime.loop import LivingMemoryLoop

N_TURNS = 500
TOPICS = [
    "讨论机器学习的反向传播算法",
    "分析中国古诗的意境表达",
    "研究量子力学的测不准原理",
    "探讨气候变化对生态的影响",
    "介绍区块链技术的共识机制",
    "回顾文艺复兴时期的艺术成就",
    "思考人工智能的伦理问题",
    "描述深海生物的适应性进化",
    "阐述经济学中的供需理论",
    "解析音乐和声的基本规则",
]


def main() -> None:
    loop = LivingMemoryLoop({
        'num_nodes': 256,
        'input_dim': 64,
        'meta_enabled': False,
        'auto_snapshot': False,
    })
    cohs = []
    prec_means = []
    prec_stds = []
    for i in range(N_TURNS):
        text = f"{TOPICS[i % len(TOPICS)]} 第{i // len(TOPICS) + 1}轮"
        loop.process_turn(text)
        cohs.append(loop.purpose.coherence)
        prec = loop.purpose.get_precision()
        prec_means.append(float(prec.mean()))
        prec_stds.append(float(prec.std()))
    # 只统计后半段（前 100 轮为预热）
    warm = 100
    coh = cohs[warm:]
    print(f"轮次={N_TURNS} (预热{warm})")
    print(f"purpose_coherence: mean={statistics.mean(coh):.4f} "
          f"std={statistics.stdev(coh):.4f} "
          f"min={min(coh):.4f} max={max(coh):.4f}")
    print(f"precision mean: avg={statistics.mean(prec_means):.4f} "
          f"std_avg={statistics.mean(prec_stds):.4f}")


if __name__ == "__main__":
    main()

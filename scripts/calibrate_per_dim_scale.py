"""C2 k 值实测校准——生产量级仿真（bge-m3 投影尺度）。

CloudEmbedder: 1024 维 bge-m3（L2 归一化）→ randn(1024,64)/sqrt(1024) 投影。
输出分量 std ≈ sqrt(1024 * (1/1024) * E[v_j²]) = 1/sqrt(1024) ≈ 0.031。
用同量级 FakeEmbedder 仿真生产输入尺度，跑 240 轮统计 per_dim_surprise 分布，
校准 per_dim_surprise_scale（k）使典型误差落在 tanh 灵敏区。
"""
import math
import statistics

import torch

from runtime.loop import LivingMemoryLoop

NUM_NODES = 256
INPUT_DIM = 64
N_TURNS = 240

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


class FakeCloudEmbedder:
    """仿真 CloudEmbedder 输出尺度：每分量 N(0, 0.031²)，L2 归一化。"""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self._std = 1.0 / math.sqrt(1024)

    def embed_text(self, text: str) -> torch.Tensor:
        v = torch.randn(self.dim) * self._std
        return v / (v.norm() + 1e-8) * 0.3  # 归一化后放大到投影典型幅度


def main() -> None:
    loop = LivingMemoryLoop({
        'num_nodes': NUM_NODES,
        'input_dim': INPUT_DIM,
        'meta_enabled': False,
        'auto_snapshot': False,
        'embedder': FakeCloudEmbedder(INPUT_DIM),
    })
    samples = []
    surprises = []
    for i in range(N_TURNS):
        text = f"{TOPICS[i % len(TOPICS)]} 第{i // len(TOPICS) + 1}轮"
        loop.process_turn(text)
        act = loop.last_activation
        if act.per_dim_surprise is not None:
            samples.append(act.per_dim_surprise.detach().cpu().flatten())
        surprises.append(act.surprise)

    all_vals = torch.cat(samples).numpy()
    per_turn_means = [float(s.mean()) for s in samples]
    sorted_vals = sorted(all_vals.tolist())
    n = len(sorted_vals)
    p10 = sorted_vals[int(n * 0.10)]
    p90 = sorted_vals[int(n * 0.90)]
    med = statistics.median(sorted_vals)
    mean = statistics.mean(sorted_vals)
    print(f"轮次: {N_TURNS}")
    print(f"per_dim_surprise 分布: mean={mean:.6f} median={med:.6f} "
          f"P10={p10:.6f} P90={p90:.6f}")
    print(f"每轮 per_dim 均值: mean={statistics.mean(per_turn_means):.6f} "
          f"min={min(per_turn_means):.6f} max={max(per_turn_means):.6f}")
    print(f"全局 surprise: mean={statistics.mean(surprises):.4f} "
          f"min={min(surprises):.4f} max={max(surprises):.4f}")
    print("--- tanh 灵敏区扫描（目标 raw/k ∈ [0.2, 2]）---")
    for k in (1.0, 0.1, 0.03, 0.01, 0.003, 0.001):
        z_med, z_p90 = med / k, p90 / k
        t_med = torch.tanh(torch.tensor(z_med)).item()
        t_p90 = torch.tanh(torch.tensor(z_p90)).item()
        mapped_med = 0.1 + 9.9 * t_med
        mapped_p90 = 0.1 + 9.9 * t_p90
        print(f"k={k}: median tanh({z_med:.2f})={t_med:.3f}→mapped {mapped_med:.2f} | "
              f"P90 tanh({z_p90:.2f})={t_p90:.3f}→mapped {mapped_p90:.2f}")


if __name__ == "__main__":
    main()

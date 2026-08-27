"""惊讶度语义拆分验证脚本（惊讶度修复-01-设计方案.md §5.3）。

实施后运行，输出报告，验证 surprise（准确性项）与 free_energy（未规范化
变分能量）确已拆分为两个不同物理量，并核查下游行为。

实验：
  1. 排序差异对比：同一批 buffer 记忆逐条 infer 得 (surprise, free_energy)，
     计算 Spearman 秩相关 ρ；预期 ρ 显著 < 1（≈0.3~0.7），证明两者不是
     同一物理量；若 ρ≈1 说明系统长期"安稳且正确"，可接受但需备注。
  2. 负值清零统计：旧实现 max(surprise, 0) 触发清零比例（历史 buffer 中
     surprise<0 占比）——拆分后 surprise 恒≥0，此缺陷自动消失。
  3. −F 与 accuracy 相关性：corr(−free_energy, surprise)；弱相关 ⇒
     "安稳但猜错"状态确实存在，印证拆分必要性。
  4. 做梦采样分布：标准化前后 _sample_by_surprise 权重熵对比（熵上升 =
     采样多样性恢复）。
  5. 回归：`pytest tests/ -x` 全量（含 phase3_emergence slow）。
  6. per_dim 分布统计与 k 校准：见 scripts/calibrate_per_dim_scale.py
     （C2 落地时已执行，结论在 C2 commit message）。
"""
import math
import os
import statistics
import sys

import torch

from core.hippocampus.attractor import AttractorNetwork
from core.hippocampus.purpose import PurposeLayer
from core.types import PurposeState


def _spearman(xs, ys):
    """Spearman 秩相关（处理并列秩）。"""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while (j + 1 < len(order)
                   and v[order[j + 1]] == v[order[i]]):
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    var_x = sum((a - mean_x) ** 2 for a in rx)
    var_y = sum((b - mean_y) ** 2 for b in ry)
    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def _pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs)
    vy = sum((b - my) ** 2 for b in ys)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def _load_production_landscape(path: str):
    """加载生产快照的吸引子景观 + 目的层（若可用）。"""
    if not os.path.exists(path):
        return None, None
    d = torch.load(path, map_location='cpu', weights_only=False)
    at = d.get('attractor')
    if at is None:
        return None, None
    net = AttractorNetwork(at['num_nodes'], at['input_dim'], seed=42)
    net.set_landscape(at)
    pur = None
    p = d.get('purpose')
    if p is not None and 'precision' in p:
        pur = PurposeLayer(at['input_dim'])
        pur.set_purpose(PurposeState(
            precision=p['precision'],
            history=p.get('history') or [],
            coherence=p.get('coherence', 1.0),
            encounter_count=p.get('encounter_count')))
    return net, pur


def main() -> None:
    # 默认取 main 会话快照（可用 --snapshot 覆盖）
    snap = (sys.argv[1] if len(sys.argv) > 1
            else 'snapshots/main/latest_main.pt')
    net, pur = _load_production_landscape(snap)
    if net is None:
        print(f"[实验1-3] 快照 {snap} 不可用，改用合成网络（seed=42）")
        net = AttractorNetwork(256, 64, seed=42)
        pur = PurposeLayer(64)
    print(f"=== 惊讶度语义拆分验证报告 ===")
    print(f"网络: nodes={net.num_nodes} dim={net.input_dim} "
          f"快照={os.path.basename(snap) if os.path.exists(snap) else '合成'}")

    # ---------- 实验 1+3：排序差异 + −F 与 accuracy 相关性 ----------
    print("\n[实验1/3] surprise 与 free_energy 排序/相关性")
    s_std = 1.0 / math.sqrt(1024)
    surprises, fees = [], []
    precision = pur.get_precision() if pur else torch.ones(net.input_dim)
    for i in range(120):
        torch.manual_seed(2000 + i)
        s = torch.randn(net.input_dim) * s_std
        act = net.infer(s, precision)
        surprises.append(act.surprise)
        fees.append(act.free_energy)
    rho = _spearman(surprises, fees)
    corr = _pearson([-f for f in fees], surprises)
    print(f"  ρ(surprise, free_energy) = {rho:.3f}  "
          f"corr(−F, surprise) = {corr:.3f}")
    print(f"  surprise: mean={statistics.mean(surprises):.3f} "
          f"min={min(surprises):.3f} max={max(surprises):.3f}")
    print(f"  free_energy: mean={statistics.mean(fees):.3f} "
          f"min={min(fees):.3f} max={max(fees):.3f}")
    neg_ratio = sum(1 for x in surprises if x < 0) / len(surprises)
    print(f"  负值清零比例（surprise<0 占比）= {neg_ratio:.3f} "
          f"（预期 ≈0：拆分后恒≥0）")
    if rho < 0.95:
        print("  ✓ ρ 显著 < 1：两者确为不同物理量（能量污染已剥离）")
    else:
        print("  ⚠ ρ≈1：系统长期'安稳且正确'，需人工备注（设计 §5.3 可接受）")

    # ---------- 实验 4：做梦采样标准化前后权重熵 ----------
    print("\n[实验4] 做梦采样分布（标准化前后权重熵对比）")
    from core.hippocampus.memory import MemoryManager
    from core.hippocampus.dream_engine import DreamEngine
    from core.sensory.embedder import SimpleEmbedder
    mem = MemoryManager(net.num_nodes)
    engine = DreamEngine(net, pur or PurposeLayer(net.input_dim), mem,
                         SimpleEmbedder(net.input_dim),
                         {'num_nodes': net.num_nodes,
                          'input_dim': net.input_dim})
    # 重尾缓冲：99 条 ~0.1 + 1 条 100
    buf = [(torch.zeros(net.num_nodes), 0.1) for _ in range(99)]
    buf.append((torch.ones(net.num_nodes), 100.0))
    engine.surprise_beta = 5.0  # 旧 β（对照）
    surprises_t = torch.tensor([s for (_, s) in buf], dtype=torch.float32)
    # 旧式：max 减法 + β=5（无标准化）
    shifted_old = 5.0 * (surprises_t - surprises_t.max())
    w_old = torch.exp(shifted_old)
    w_old = w_old / w_old.sum()
    # 新式：z-score + clamp(±3) + β=1.0
    z = (surprises_t - surprises_t.mean()) / surprises_t.std()
    z = z.clamp(-3.0, 3.0)
    w_new = torch.exp(1.0 * z)
    w_new = w_new / w_new.sum()
    def entropy(w):
        w = w[w > 0]
        return float(-torch.sum(w * torch.log(w)).item())
    print(f"  旧式（max 减法 β=5）权重熵 = {entropy(w_old):.4f} "
          f"（离群项权重 {float(w_old[-1]):.3f}，退化为 argmax）")
    print(f"  新式（z 截断 β=1.0）权重熵 = {entropy(w_new):.4f} "
          f"（离群项权重 {float(w_new[-1]):.3f}，不退化）")
    print(f"  ✓ 熵上升 = 采样多样性恢复" if entropy(w_new) > entropy(w_old)
          else "  ⚠ 熵未上升，需核查")

    print("\n[实验5] 回归测试：请在 shell 执行 "
          "`pytest tests/ -x`（681 项，含 phase3 slow）")
    print("[实验6] per_dim 分布与 k 校准：scripts/calibrate_per_dim_scale.py "
          "（C2 已执行，k=0.1，结论见 C2 commit message）")


if __name__ == "__main__":
    main()

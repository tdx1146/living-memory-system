"""
活体记忆系统 - 投影信息损失评估工具 (任务 A-P2-2)
==================================================

量化 384维 -> 64维 投影前后的信息损失，并对比不同降维策略。

背景
----
PretrainedEmbedder 使用预训练模型 paraphrase-multilingual-MiniLM-L12-v2
（384 维）编码文本语义，再通过一个冻结的随机投影矩阵（Johnson-Lindenstrauss
缩放，即 randn(d,k)/sqrt(d)，种子 42）降到系统配置的 input_dim（默认 64）。
本模块评估该投影对语义几何结构（余弦相似度、欧氏距离、k 近邻）的保持程度，
并对比四种降维策略：

  1. random_projection —— 随机投影（当前方案，JL 缩放 randn/sqrt(d)）
  2. pca               —— PCA 降维（捕捉主方差方向，基于 SVD，无 sklearn 依赖）
  3. truncation        —— 均匀截断（取前 k 维）
  4. random_selection  —— 随机选择 k 维

设计要点
--------
- 全部指标以 numpy 计算；输入接受 torch.Tensor / np.ndarray / list，内部
  统一转为 float64 numpy 数组。
- 不依赖预训练模型：compare_projection_strategies 在未注入 encoder 时使用
  种子化的结构化随机向量（低秩信号 + 噪声），可在无 sentence-transformers
  的环境下运行与测试。需要真实语义评估时，通过 encoder 参数注入（见
  run_eval.py）。
- 评估的是“几何结构保持”而非“逐维数值保持”：余弦/距离/kNN 指标均基于
  样本两两关系，与具体维度无关，因此可直接对比 384 维与 64 维空间。

参考
----
- Johnson-Lindenstrauss 引理：k >= 8·ln(n)/eps^2 即可保证距离畸变 <= eps。
- embedder.py 中 PretrainedEmbedder._projection 的实现。
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Union

import numpy as np

# torch 为可选依赖：可用时支持把 Tensor 输入自动转为 numpy
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - 运行环境决定
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

# 接受的向量输入类型
ArrayLike = Union[np.ndarray, "torch.Tensor", list]


# ================================================================== #
#  内部工具函数
# ================================================================== #

def _to_numpy(x: ArrayLike) -> np.ndarray:
    """将输入统一转为 float64 的 numpy 数组。

    支持 torch.Tensor（detach/cpu/float）、np.ndarray 与可被 np.asarray
    解析的嵌套列表。
    """
    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64)


def _as_matrix(x: ArrayLike) -> np.ndarray:
    """将输入规整为 2D 矩阵 [n_samples, dim]。

    单条向量 [d] 会被视作 [1, d]。空输入返回 shape (0, 0)。
    """
    arr = _to_numpy(x)
    if arr.size == 0:
        return arr.reshape(0, 0)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"期望 1D 或 2D 输入，得到 {arr.ndim}D")
    return arr


def _validate_pair(original_vecs: ArrayLike,
                   projected_vecs: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """校验原始/投影向量对，返回 (original, projected) 2D 数组。

    要求两者样本数一致（投影前后的同一批样本）。
    """
    o = _as_matrix(original_vecs)
    p = _as_matrix(projected_vecs)
    if o.shape[0] != p.shape[0]:
        raise ValueError(
            f"原始向量与投影向量样本数不一致: {o.shape[0]} vs {p.shape[0]}"
        )
    return o, p


def _pairwise_cosine(X: np.ndarray) -> np.ndarray:
    """计算成对余弦相似度矩阵 [n, n]。"""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.clip(norms, 1e-12, None)
    return Xn @ Xn.T


def _pairwise_dist(X: np.ndarray) -> np.ndarray:
    """计算成对欧氏距离矩阵 [n, n]。"""
    sq = np.sum(X * X, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    d2 = np.maximum(d2, 0.0)  # 抵消浮点误差导致的负值
    return np.sqrt(d2)


def _upper_triangle(mat: np.ndarray) -> np.ndarray:
    """取对称矩阵的严格上三角（k=1），返回一维数组（不含对角自相似）。"""
    n = mat.shape[0]
    if n < 2:
        return np.array([], dtype=np.float64)
    iu = np.triu_indices(n, k=1)
    return mat[iu]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson 相关系数。<2 个点或零方差时返回 NaN。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(np.sum(a * a)) * float(np.sum(b * b)))
    if denom < 1e-15:
        return float("nan")
    return float(np.sum(a * b) / denom)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """平均秩（average rank，处理并列值），与 scipy.stats.rankdata 默认一致。"""
    a = np.asarray(a, dtype=np.float64)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size)
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关系数（基于平均秩的 Pearson）。<2 个点返回 NaN。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    return _pearson(_rankdata(a), _rankdata(b))


def _knn_indices(X: np.ndarray, k: int) -> np.ndarray:
    """返回每个样本的 k 近邻索引矩阵 [n, k]（不含自身，按距离升序）。"""
    n = X.shape[0]
    kk = min(k, n - 1) if n > 1 else 0
    if kk < 1:
        return np.empty((n, 0), dtype=np.intp)
    D = _pairwise_dist(X)
    np.fill_diagonal(D, np.inf)  # 排除自身
    # argpartition 取最近 kk 个，再按距离排序
    part = np.argpartition(D, kth=kk - 1, axis=1)[:, :kk]
    row = np.arange(n)[:, None]
    order = np.argsort(D[row, part], axis=1)
    return part[row, order]


# ================================================================== #
#  评估指标
# ================================================================== #

def evaluate_cosine_preservation(original_vecs: ArrayLike,
                                 projected_vecs: ArrayLike) -> dict:
    """评估投影前后成对余弦相似度的保持程度。

    分别在原始空间与投影空间计算 n×n 的成对余弦相似度矩阵，取严格上三角
    得到两列成对相似度，衡量其一致性：

      - cosine_pearson_correlation: 线性保持程度（越接近 1 越好）。
      - cosine_spearman_correlation: 秩保持程度（对近邻排序更敏感）。
      - cosine_mae: 成对余弦相似度的平均绝对误差。
      - cosine_mean_original / cosine_mean_projected: 两空间相似度均值。

    参数:
        original_vecs: 原始向量 [n, d_orig]。
        projected_vecs: 投影后向量 [n, d_proj]。

    返回:
        指标字典。样本数 < 2 时成对指标为 NaN。
    """
    o, p = _validate_pair(original_vecs, projected_vecs)
    n = o.shape[0]
    if n < 2:
        return {
            "n_samples": n,
            "n_pairs": 0,
            "cosine_pearson_correlation": float("nan"),
            "cosine_spearman_correlation": float("nan"),
            "cosine_mae": float("nan"),
            "cosine_mean_original": float("nan"),
            "cosine_mean_projected": float("nan"),
        }
    a = _upper_triangle(_pairwise_cosine(o))
    b = _upper_triangle(_pairwise_cosine(p))
    return {
        "n_samples": n,
        "n_pairs": int(a.size),
        "cosine_pearson_correlation": _pearson(a, b),
        "cosine_spearman_correlation": _spearman(a, b),
        "cosine_mae": float(np.mean(np.abs(a - b))),
        "cosine_mean_original": float(np.mean(a)),
        "cosine_mean_projected": float(np.mean(b)),
    }


def evaluate_distance_preservation(original_vecs: ArrayLike,
                                   projected_vecs: ArrayLike) -> dict:
    """评估投影前后成对欧氏距离的保持程度。

    先用最小二乘求得最优全局缩放 c* = Σ(d_proj·d_orig)/Σ(d_proj²)，
    使投影距离对齐到原始距离的尺度，再计算每个点对的相对畸变
    |c*·d_proj/d_orig − 1|。这是 Johnson-Lindenstrauss 保证的直接度量。

      - distance_pearson_correlation / distance_spearman_correlation:
        距离的线性/秩保持（尺度无关）。
      - distance_mean_relative_error: 平均相对畸变（越小越好）。
      - distance_max_relative_error: 最大相对畸变（对应 JL 的 eps）。
      - distance_scale_factor: 最优对齐缩放 c*；对 JL 随机投影理论上
        约为 sqrt(d_orig/d_proj)。

    参数:
        original_vecs: 原始向量 [n, d_orig]。
        projected_vecs: 投影后向量 [n, d_proj]。

    返回:
        指标字典。样本数 < 2 时成对指标为 NaN。
    """
    o, p = _validate_pair(original_vecs, projected_vecs)
    n = o.shape[0]
    if n < 2:
        return {
            "n_samples": n,
            "n_pairs": 0,
            "distance_pearson_correlation": float("nan"),
            "distance_spearman_correlation": float("nan"),
            "distance_mean_relative_error": float("nan"),
            "distance_max_relative_error": float("nan"),
            "distance_scale_factor": float("nan"),
            "distance_mean_original": float("nan"),
            "distance_mean_projected": float("nan"),
        }
    a = _upper_triangle(_pairwise_dist(o))  # 原始距离
    b = _upper_triangle(_pairwise_dist(p))  # 投影距离
    # 最优全局缩放（最小二乘）
    denom = float(np.sum(b * b))
    scale = float(np.sum(b * a) / denom) if denom > 1e-15 else float("nan")
    # 相对畸变：避免除零
    safe_a = np.where(a > 1e-12, a, np.nan)
    ratio = scale * b / safe_a  # c*·d_proj / d_orig
    rel_err = np.abs(ratio - 1.0)
    rel_err = rel_err[~np.isnan(rel_err)]
    return {
        "n_samples": n,
        "n_pairs": int(a.size),
        "distance_pearson_correlation": _pearson(a, b),
        "distance_spearman_correlation": _spearman(a, b),
        "distance_mean_relative_error": float(np.mean(rel_err)) if rel_err.size else float("nan"),
        "distance_max_relative_error": float(np.max(rel_err)) if rel_err.size else float("nan"),
        "distance_scale_factor": scale,
        "distance_mean_original": float(np.mean(a)),
        "distance_mean_projected": float(np.mean(b)),
    }


def evaluate_nearest_neighbor_overlap(original_vecs: ArrayLike,
                                      projected_vecs: ArrayLike,
                                      k: int = 5) -> dict:
    """评估投影前后 k 近邻的重叠率。

    对每个样本分别在原始空间与投影空间找其 k 近邻（不含自身），统计两者
    重叠程度：

      - mean_overlap: 平均重叠比例 |交集|/k（1.0 表示完全保持）。
      - mean_jaccard: 平均 Jaccard 相似度 |交集|/|并集|。
      - recall_at_k: 等同 mean_overlap，便于检索语境理解。

    参数:
        original_vecs: 原始向量 [n, d_orig]。
        projected_vecs: 投影后向量 [n, d_proj]。
        k: 近邻数，默认 5。当 n-1 < k 时自动收紧为 n-1。

    返回:
        指标字典。样本数 < 2 时成对指标为 NaN。
    """
    o, p = _validate_pair(original_vecs, projected_vecs)
    n = o.shape[0]
    kk = min(k, n - 1) if n > 1 else 0
    if n < 2 or kk < 1:
        return {
            "n_samples": n,
            "k_requested": k,
            "k_used": 0,
            "mean_overlap": float("nan"),
            "mean_jaccard": float("nan"),
            "recall_at_k": float("nan"),
        }
    nn_o = _knn_indices(o, kk)
    nn_p = _knn_indices(p, kk)
    overlaps = []
    jaccards = []
    for i in range(n):
        so = set(nn_o[i].tolist())
        sp = set(nn_p[i].tolist())
        inter = len(so & sp)
        union = len(so | sp)
        overlaps.append(inter / kk)
        jaccards.append(inter / union if union > 0 else 0.0)
    mean_overlap = float(np.mean(overlaps))
    return {
        "n_samples": n,
        "k_requested": k,
        "k_used": kk,
        "mean_overlap": mean_overlap,
        "mean_jaccard": float(np.mean(jaccards)),
        "recall_at_k": mean_overlap,
    }


def evaluate_information_capacity(original_dim: int,
                                  projected_dim: int) -> dict:
    """理论信息容量分析（基于 Johnson-Lindenstrauss 引理）。

    JL 引理：将 n 个点嵌入 k 维并保证任意点对距离畸变 <= eps，只需
    k >= 8·ln(n)/eps²。据此分析给定维度配置的理论保证。

    返回指标:
      - compression_ratio: 压缩比 projected_dim/original_dim。
      - dimensionality_reduction_pct: 降维百分比。
      - jl_epsilon_for_n: 给定若干 n，所需的理论畸变上界 eps。
      - jl_max_n_for_epsilon: 给定若干 eps，最大可忠实嵌入的点数 n。
      - jl_min_dims_for_epsilon: 给定若干 eps，保 eps 所需最小维度
        （用于与 projected_dim 对比判断是否充分）。
    """
    if original_dim <= 0 or projected_dim <= 0:
        raise ValueError("维度必须为正整数")
    k = projected_dim
    compression = k / original_dim
    reduction = (1.0 - compression) * 100.0

    eps_for_n: dict[int, float] = {}
    for n in (10, 100, 1000, 10000, 100000):
        eps_for_n[n] = math.sqrt(8.0 * math.log(n) / k)

    max_n_for_eps: dict[float, float] = {}
    for eps in (0.1, 0.2, 0.3, 0.5):
        max_n_for_eps[eps] = math.exp(k * eps * eps / 8.0)

    min_dims_for_eps: dict[float, float] = {}
    for eps in (0.1, 0.2, 0.3, 0.5):
        # 取 n=10000 时的所需维度作为参考
        min_dims_for_eps[eps] = math.ceil(8.0 * math.log(10000) / (eps * eps))

    return {
        "original_dim": int(original_dim),
        "projected_dim": int(projected_dim),
        "compression_ratio": float(compression),
        "dimensionality_reduction_pct": float(reduction),
        "jl_epsilon_for_n": eps_for_n,
        "jl_max_n_for_epsilon": max_n_for_eps,
        "jl_min_dims_for_epsilon": min_dims_for_eps,
    }


# ================================================================== #
#  投影策略
# ================================================================== #

def _random_projection_project(X: np.ndarray, target_dim: int,
                               seed: int = 42) -> np.ndarray:
    """随机投影（当前系统方案）：P = randn(d,k)/sqrt(d)，y = X·P。

    与 PretrainedEmbedder._projection 的 JL 缩放一致（统计等价；
    此处用 numpy RandomState(seed) 复现，与 torch RNG 的具体数值不同
    但分布性质相同）。
    """
    d = X.shape[1]
    k = min(target_dim, d)
    rng = np.random.RandomState(seed)
    P = rng.randn(d, k) / math.sqrt(d)
    return X @ P


def _pca_project(X: np.ndarray, target_dim: int) -> np.ndarray:
    """PCA 降维：数据中心化后取前 target_dim 个主成分（基于 SVD，无 sklearn）。"""
    n, d = X.shape
    if n == 0:
        # 空输入：直接返回空矩阵，避免 mean of empty slice 警告
        return np.zeros((0, min(target_dim, d)))
    k = min(target_dim, d, max(n - 1, 1)) if n > 1 else min(target_dim, d)
    mu = X.mean(axis=0)
    Xc = X - mu
    if n < 2 or not np.any(Xc):
        # 无方差：退化为取前 k 维（结果为零向量矩阵）
        return np.zeros((n, k))
    # 经济型 SVD: Xc = U S V^T
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(k, Vt.shape[0])
    return Xc @ Vt[:k].T  # [n, k]


def _truncation_project(X: np.ndarray, target_dim: int) -> np.ndarray:
    """均匀截断：取前 target_dim 维。"""
    k = min(target_dim, X.shape[1])
    return X[:, :k].copy()


def _random_selection_project(X: np.ndarray, target_dim: int,
                              seed: int = 42) -> np.ndarray:
    """随机选择 target_dim 个维度（无放回，固定种子）。"""
    d = X.shape[1]
    k = min(target_dim, d)
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(d, size=k, replace=False))
    return X[:, idx].copy()


# ================================================================== #
#  策略对比
# ================================================================== #

def _random_vectors(n: int, dim: int, seed: int = 42) -> np.ndarray:
    """生成结构化随机向量（低秩信号 + 噪声），供无预训练模型时使用。

    有效秩约为 dim/6，使数据具备低维结构——PCA 能捕捉主方差方向从而
    优于随机投影/截断，使评估对比更具区分度。纯各向同性高斯数据下四种
    策略期望表现相近，故加入低秩信号以反映真实嵌入的“可降维性”。
    """
    rng = np.random.RandomState(seed)
    eff_rank = max(dim // 6, 2)
    basis = rng.randn(eff_rank, dim) * 0.5
    codes = rng.randn(n, eff_rank)
    noise = rng.randn(n, dim) * 0.2
    return codes @ basis + noise


def _auto_encode(texts: List[str], original_dim: int,
                 seed: int = 42) -> tuple[np.ndarray, str]:
    """未注入 encoder 时的默认向量生成：结构化随机向量。

    返回 (vectors, source_label)。不触碰预训练模型，保证测试快速、
    确定且无网络下载。真实语义评估请通过 encoder 参数注入（见 run_eval.py）。
    """
    n = len(texts)
    if n == 0:
        return np.zeros((0, original_dim)), "empty"
    return _random_vectors(n, original_dim, seed), "random(low-rank)"


def _full_metrics(original: np.ndarray, projected: np.ndarray,
                  original_dim: int, target_dim: int, k: int = 5) -> dict:
    """对一对 (原始, 投影) 向量计算全部指标。"""
    return {
        "cosine": evaluate_cosine_preservation(original, projected),
        "distance": evaluate_distance_preservation(original, projected),
        "nn_overlap": evaluate_nearest_neighbor_overlap(original, projected, k=k),
        "info_capacity": evaluate_information_capacity(original_dim, target_dim),
    }


def compare_projection_strategies(
    texts: List[str],
    original_dim: int = 384,
    target_dim: int = 64,
    encoder: Optional[Callable[[List[str]], ArrayLike]] = None,
    seed: int = 42,
    source_label: Optional[str] = None,
) -> dict:
    """对比四种降维策略在给定文本/向量上的信息保持效果。

    策略：
      1. random_projection —— 随机投影（当前方案，JL 缩放）
      2. pca               —— PCA 降维
      3. truncation        —— 均匀截断
      4. random_selection  —— 随机选择维度

    每种策略计算余弦保持、距离保持、k 近邻重叠与理论信息容量。

    参数:
        texts: 文本列表（仅用于决定样本数与可选编码）。
        original_dim: 原始向量维度（预训练模型为 384）。
        target_dim: 目标投影维度（系统默认 64）。
        encoder: 可选的文本->向量编码函数，签名为 encoder(texts)->ArrayLike。
            为 None 时使用结构化随机向量（不依赖预训练模型，便于测试）。
            传入真实编码器可评估真实语义（见 run_eval.py）。
        seed: 随机策略的种子，保证可复现。
        source_label: 可选的向量来源标签，覆盖报告中默认的 "encoder(injected)"
            / "random(low-rank)"，便于 run_eval.py 标注真实来源（如预训练模型名）。

    返回:
        dict，结构::

            {
              "meta": {
                  "original_dim": int, "target_dim": int,
                  "n_samples": int, "encoder": str  # 向量来源标签
              },
              "strategies": {
                  "random_projection": {cosine, distance, nn_overlap, info_capacity},
                  "pca": {...}, "truncation": {...}, "random_selection": {...},
              }
            }
    """
    texts = list(texts)
    n = len(texts)
    if encoder is not None:
        original = _as_matrix(encoder(texts))
        source = "encoder(injected)"
    else:
        original, source = _auto_encode(texts, original_dim, seed)
    if source_label is not None:
        source = source_label

    # 以实际向量维度为准（编码器可能返回与 original_dim 不同的维度）
    actual_dim = original.shape[1] if original.size > 0 else original_dim
    if actual_dim != original_dim:
        original_dim = actual_dim

    strategies: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "random_projection": lambda X: _random_projection_project(X, target_dim, seed),
        "pca": lambda X: _pca_project(X, target_dim),
        "truncation": lambda X: _truncation_project(X, target_dim),
        "random_selection": lambda X: _random_selection_project(X, target_dim, seed),
    }

    out: dict = {
        "meta": {
            "original_dim": int(original_dim),
            "target_dim": int(target_dim),
            "n_samples": int(n),
            "encoder": source,
        },
        "strategies": {},
    }
    for name, fn in strategies.items():
        projected = fn(original)
        out["strategies"][name] = _full_metrics(
            original, projected, original_dim, target_dim
        )
    return out


# ================================================================== #
#  报告生成
# ================================================================== #

def _disp_width(s: str) -> int:
    """显示宽度：CJK/全角字符计 2，其余计 1。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def _pad_right(s: str, width: int) -> str:
    return s + " " * max(0, width - _disp_width(s))


def _pad_left(s: str, width: int) -> str:
    return " " * max(0, width - _disp_width(s)) + s


def _fmt(x, ndigits: int = 4) -> str:
    """格式化数值：NaN 显示为 '  nan'，否则定宽浮点。"""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "nan"
    return f"{float(x):.{ndigits}f}"


def generate_report(results: dict) -> str:
    """由 compare_projection_strategies 的结果生成文本评估报告。

    报告包含：元信息、策略对比表（余弦/距离/kNN 指标）、Johnson-Lindenstrauss
    理论保证、以及自动结论（按余弦 Pearson 选出最佳策略）。

    参数:
        results: compare_projection_strategies 的返回值。

    返回:
        多行文本报告字符串。
    """
    meta = results.get("meta", {})
    strategies = results.get("strategies", {})
    lines: list[str] = []

    lines.append("=" * 76)
    lines.append("投影信息损失评估报告")
    lines.append("=" * 76)
    lines.append(
        f"原始维度: {meta.get('original_dim', '?')}  "
        f"目标维度: {meta.get('target_dim', '?')}  "
        f"样本数: {meta.get('n_samples', '?')}  "
        f"向量来源: {meta.get('encoder', '?')}"
    )
    lines.append("")

    if not strategies:
        lines.append("（无策略结果）")
        return "\n".join(lines)

    # ---- 策略对比表 ----
    col_name = 20
    lines.append("--- 策略对比（几何结构保持，越接近 1 越好；误差越小越好）---")
    header = (
        _pad_right("策略", col_name) + "| "
        + _pad_left("余弦Pearson", 12) + " | "
        + _pad_left("余弦Spearman", 13) + " | "
        + _pad_left("距离Pearson", 12) + " | "
        + _pad_left("距离Spearman", 13) + " | "
        + _pad_left("kNN重叠@5", 10) + " | "
        + _pad_left("距离相对误差", 12)
    )
    lines.append(header)
    lines.append("-" * len(header))

    best_name = None
    best_score = -float("inf")
    for name, m in strategies.items():
        cos_p = m.get("cosine", {}).get("cosine_pearson_correlation", float("nan"))
        cos_s = m.get("cosine", {}).get("cosine_spearman_correlation", float("nan"))
        dist_p = m.get("distance", {}).get("distance_pearson_correlation", float("nan"))
        dist_s = m.get("distance", {}).get("distance_spearman_correlation", float("nan"))
        nn = m.get("nn_overlap", {}).get("mean_overlap", float("nan"))
        dist_err = m.get("distance", {}).get("distance_mean_relative_error", float("nan"))
        row = (
            _pad_right(name, col_name) + "| "
            + _pad_left(_fmt(cos_p), 12) + " | "
            + _pad_left(_fmt(cos_s), 13) + " | "
            + _pad_left(_fmt(dist_p), 12) + " | "
            + _pad_left(_fmt(dist_s), 13) + " | "
            + _pad_left(_fmt(nn), 10) + " | "
            + _pad_left(_fmt(dist_err), 12)
        )
        lines.append(row)
        if not math.isnan(cos_p) and cos_p > best_score:
            best_score = cos_p
            best_name = name
    lines.append("")

    # ---- JL 理论保证 ----
    # info_capacity 仅依赖维度，各策略一致，取首个
    any_m = next(iter(strategies.values()))
    info = any_m.get("info_capacity", {})
    if info:
        k = info.get("projected_dim", "?")
        lines.append("--- Johnson-Lindenstrauss 理论保证 ---")
        lines.append(
            f"压缩比: {_fmt(info.get('compression_ratio', float('nan')))} "
            f"（降维 {info.get('dimensionality_reduction_pct', 0):.1f}%）"
        )
        eps_map = info.get("jl_epsilon_for_n", {})
        if eps_map:
            lines.append("给定点数 n 的理论距离畸变上界 eps = sqrt(8·ln(n)/k):")
            for n_val in sorted(eps_map):
                lines.append(f"  n={n_val:<7d} -> eps ≈ {_fmt(eps_map[n_val])}")
        maxn = info.get("jl_max_n_for_epsilon", {})
        if maxn:
            lines.append(f"目标维度 k={k} 下，给定 eps 的最大可忠实嵌入点数 n:")
            for eps_val in sorted(maxn):
                lines.append(
                    f"  eps={eps_val:<4.2f} -> n ≤ {_fmt(maxn[eps_val], 1)}"
                )
        lines.append("")

    # ---- 自动结论 ----
    lines.append("--- 结论 ---")
    if best_name is not None:
        lines.append(
            f"在当前样本上，按余弦 Pearson 衡量，最佳策略为 '{best_name}'"
            f"（r={_fmt(best_score)}）。"
        )
        if best_name == "pca":
            lines.append(
                "数据具备可被主成分捕捉的低秩结构，PCA 优于随机投影——"
                "若实际语义嵌入存在显著主方差方向，可考虑用 PCA 替代随机投影。"
            )
        elif best_name == "random_projection":
            lines.append(
                "随机投影（当前方案）表现最佳或并列最佳，说明 JL 随机投影对当前"
                "数据分布已是合理选择；其无需训练、可复现，适合作为感官层默认降维。"
            )
        else:
            lines.append(
                f"注意：'{best_name}' 表现最佳，但其在各向同性数据上通常不优于随机"
                "投影，建议结合真实语义向量复核。"
            )
    else:
        lines.append("样本不足，无法判定最佳策略。")
    lines.append("=" * 76)
    return "\n".join(lines)


# ================================================================== #
#  主入口
# ================================================================== #

def run_evaluation(
    texts: List[str],
    target_dims: List[int] = (32, 64, 128),
    original_dim: int = 384,
    encoder: Optional[Callable[[List[str]], ArrayLike]] = None,
    seed: int = 42,
    source_label: Optional[str] = None,
) -> str:
    """对多个目标维度运行完整评估并生成综合报告。

    对 target_dims 中每个目标维度调用 compare_projection_strategies，
    生成各自的对比报告，并在头部汇总各维度下最佳策略。

    参数:
        texts: 文本列表。
        target_dims: 待评估的目标维度列表，默认 (32, 64, 128)。
        original_dim: 原始维度，默认 384。
        encoder: 可选文本编码器；None 时用结构化随机向量。
        seed: 随机种子。
        source_label: 可选向量来源标签，透传给 compare_projection_strategies，
            用于在报告中标注真实来源。

    返回:
        综合文本报告字符串。
    """
    sections: list[str] = []
    summary: list[str] = ["各目标维度最佳策略（按余弦 Pearson）:"]

    for td in target_dims:
        res = compare_projection_strategies(
            texts, original_dim=original_dim, target_dim=td,
            encoder=encoder, seed=seed, source_label=source_label,
        )
        sections.append(generate_report(res))
        # 汇总最佳
        best_name, best_score = None, -float("inf")
        for name, m in res["strategies"].items():
            sc = m.get("cosine", {}).get("cosine_pearson_correlation", float("nan"))
            if not math.isnan(sc) and sc > best_score:
                best_score, best_name = sc, name
        if best_name is not None:
            summary.append(
                f"  k={td:<4d}: {best_name} (r={_fmt(best_score)})"
            )
        else:
            summary.append(f"  k={td:<4d}: 样本不足")

    header = [
        "#" * 76,
        "# 活体记忆系统 投影信息损失综合评估 (384维 -> 多目标维度)",
        "# " + " | ".join(summary[1:]) if len(summary) > 1 else "# ",
        "#" * 76,
        "",
    ]
    return "\n".join(header) + "\n\n".join(sections)

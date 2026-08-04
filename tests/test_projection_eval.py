"""投影信息损失评估测试 (任务 A-P2-2)

测试 evaluation/projection_eval.py 中的评估指标、策略对比与报告生成。
全部使用随机向量，不依赖预训练模型，保证测试快速、确定、可离线运行。
"""

import math
import numpy as np
import pytest
import torch

from evaluation.projection_eval import (
    evaluate_cosine_preservation,
    evaluate_distance_preservation,
    evaluate_nearest_neighbor_overlap,
    evaluate_information_capacity,
    compare_projection_strategies,
    generate_report,
    run_evaluation,
    _to_numpy,
    _as_matrix,
    _random_vectors,
    _pairwise_cosine,
    _pairwise_dist,
    _pearson,
    _spearman,
    _rankdata,
)


# ============================================================
# 辅助夹具
# ============================================================

@pytest.fixture
def rng():
    """固定种子的 numpy 随机数生成器。"""
    return np.random.RandomState(0)


@pytest.fixture
def random_vecs(rng):
    """200 条 384 维随机向量。"""
    return rng.randn(200, 384)


@pytest.fixture
def projected_vecs(random_vecs):
    """对 random_vecs 做 JL 随机投影到 64 维。"""
    d = random_vecs.shape[1]
    P = np.random.RandomState(42).randn(d, 64) / math.sqrt(d)
    return random_vecs @ P


@pytest.fixture
def structured_vecs():
    """结构化（低秩）随机向量，使 PCA 有可捕捉的主成分。"""
    return _random_vectors(120, 384, seed=7)


# ============================================================
# 内部工具函数测试
# ============================================================

class TestInternalHelpers:
    """内部辅助函数测试。"""

    def test_to_numpy_from_ndarray(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = _to_numpy(arr)
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float64
        np.testing.assert_array_equal(out, arr)

    def test_to_numpy_from_torch(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = _to_numpy(t)
        assert isinstance(out, np.ndarray)
        np.testing.assert_array_equal(out, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_to_numpy_from_list(self):
        out = _to_numpy([[1, 2], [3, 4]])
        assert out.shape == (2, 2)
        assert out.dtype == np.float64

    def test_as_matrix_1d_to_2d(self):
        out = _as_matrix([1.0, 2.0, 3.0])
        assert out.shape == (1, 3)

    def test_as_matrix_empty(self):
        out = _as_matrix([])
        assert out.shape == (0, 0)

    def test_pairwise_cosine_diagonal_one(self, random_vecs):
        sub = random_vecs[:10]
        cos = _pairwise_cosine(sub)
        # 对角线自相似度为 1
        np.testing.assert_allclose(np.diag(cos), 1.0, atol=1e-10)
        # 对称
        np.testing.assert_allclose(cos, cos.T, atol=1e-10)

    def test_pairwise_dist_zero_diagonal(self, random_vecs):
        sub = random_vecs[:10]
        dist = _pairwise_dist(sub)
        # 对角线自距离为 0（浮点抵消误差约 1e-7，放宽到 1e-6）
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-6)
        np.testing.assert_allclose(dist, dist.T, atol=1e-6)
        assert (dist >= -1e-9).all()

    def test_pearson_identity(self):
        a = np.arange(20.0)
        assert _pearson(a, a) == pytest.approx(1.0)

    def test_pearson_negated(self):
        a = np.arange(20.0)
        assert _pearson(a, -a) == pytest.approx(-1.0)

    def test_pearson_too_few(self):
        assert math.isnan(_pearson(np.array([1.0]), np.array([2.0])))

    def test_spearman_monotonic(self):
        # 完美单调关系 -> spearman = 1
        a = np.arange(20.0)
        b = a * 3 + 5
        assert _spearman(a, b) == pytest.approx(1.0)

    def test_spearman_too_few(self):
        assert math.isnan(_spearman(np.array([1.0]), np.array([2.0])))

    def test_rankdata_average_ties(self):
        # 两个并列值应得到平均秩
        r = _rankdata(np.array([1.0, 2.0, 2.0, 3.0]))
        # 秩应为 [1, 2.5, 2.5, 4]
        np.testing.assert_allclose(r, [1.0, 2.5, 2.5, 4.0])


# ============================================================
# 余弦保持度测试
# ============================================================

class TestCosinePreservation:
    """evaluate_cosine_preservation 测试。"""

    def test_returns_expected_keys(self, random_vecs, projected_vecs):
        res = evaluate_cosine_preservation(random_vecs, projected_vecs)
        for key in ("n_samples", "n_pairs", "cosine_pearson_correlation",
                    "cosine_spearman_correlation", "cosine_mae",
                    "cosine_mean_original", "cosine_mean_projected"):
            assert key in res, f"缺少键 {key}"

    def test_n_samples_and_pairs(self, random_vecs, projected_vecs):
        n = 200
        res = evaluate_cosine_preservation(random_vecs, projected_vecs)
        assert res["n_samples"] == n
        assert res["n_pairs"] == n * (n - 1) // 2

    def test_correlation_in_range(self, random_vecs, projected_vecs):
        res = evaluate_cosine_preservation(random_vecs, projected_vecs)
        for key in ("cosine_pearson_correlation", "cosine_spearman_correlation"):
            v = res[key]
            assert -1.0001 <= v <= 1.0001

    def test_identity_perfect_preservation(self, random_vecs):
        """投影=原始时，余弦应完全保持。"""
        res = evaluate_cosine_preservation(random_vecs, random_vecs)
        assert res["cosine_pearson_correlation"] == pytest.approx(1.0)
        assert res["cosine_spearman_correlation"] == pytest.approx(1.0)
        assert res["cosine_mae"] == pytest.approx(0.0, abs=1e-12)

    def test_accepts_torch_tensors(self):
        """接受 torch.Tensor 输入。"""
        o = torch.randn(50, 384)
        p = torch.randn(50, 64)
        res = evaluate_cosine_preservation(o, p)
        assert res["n_samples"] == 50
        assert not math.isnan(res["cosine_pearson_correlation"])


# ============================================================
# 距离保持度测试
# ============================================================

class TestDistancePreservation:
    """evaluate_distance_preservation 测试。"""

    def test_returns_expected_keys(self, random_vecs, projected_vecs):
        res = evaluate_distance_preservation(random_vecs, projected_vecs)
        for key in ("distance_pearson_correlation", "distance_spearman_correlation",
                    "distance_mean_relative_error", "distance_max_relative_error",
                    "distance_scale_factor"):
            assert key in res

    def test_identity_perfect_preservation(self, random_vecs):
        res = evaluate_distance_preservation(random_vecs, random_vecs)
        assert res["distance_pearson_correlation"] == pytest.approx(1.0)
        assert res["distance_spearman_correlation"] == pytest.approx(1.0)
        assert res["distance_mean_relative_error"] == pytest.approx(0.0, abs=1e-9)
        assert res["distance_max_relative_error"] == pytest.approx(0.0, abs=1e-9)
        # 自身对齐缩放应为 1
        assert res["distance_scale_factor"] == pytest.approx(1.0)

    def test_correlation_in_range(self, random_vecs, projected_vecs):
        res = evaluate_distance_preservation(random_vecs, projected_vecs)
        assert -1.0001 <= res["distance_pearson_correlation"] <= 1.0001

    def test_jl_random_projection_preserves_distance(self, random_vecs,
                                                     projected_vecs):
        """JL 随机投影应正保持距离（pearson > 0）。"""
        res = evaluate_distance_preservation(random_vecs, projected_vecs)
        assert res["distance_pearson_correlation"] > 0.0


# ============================================================
# k 近邻重叠测试
# ============================================================

class TestNearestNeighborOverlap:
    """evaluate_nearest_neighbor_overlap 测试。"""

    def test_returns_expected_keys(self, random_vecs, projected_vecs):
        res = evaluate_nearest_neighbor_overlap(random_vecs, projected_vecs, k=5)
        for key in ("k_requested", "k_used", "mean_overlap", "mean_jaccard",
                    "recall_at_k"):
            assert key in res

    def test_identity_full_overlap(self, random_vecs):
        """投影=原始时，kNN 完全重叠。"""
        res = evaluate_nearest_neighbor_overlap(random_vecs, random_vecs, k=5)
        assert res["mean_overlap"] == pytest.approx(1.0)
        assert res["mean_jaccard"] == pytest.approx(1.0)
        assert res["recall_at_k"] == pytest.approx(1.0)
        assert res["k_used"] == 5

    def test_overlap_in_range(self, random_vecs, projected_vecs):
        res = evaluate_nearest_neighbor_overlap(random_vecs, projected_vecs, k=5)
        assert 0.0 <= res["mean_overlap"] <= 1.0
        assert 0.0 <= res["mean_jaccard"] <= 1.0

    def test_k_larger_than_n(self):
        """k > n-1 时应自动收紧到 n-1。"""
        X = np.random.RandomState(1).randn(4, 64)
        res = evaluate_nearest_neighbor_overlap(X, X, k=10)
        assert res["k_requested"] == 10
        assert res["k_used"] == 3  # n-1 = 3
        assert res["mean_overlap"] == pytest.approx(1.0)

    def test_custom_k(self, random_vecs, projected_vecs):
        res = evaluate_nearest_neighbor_overlap(random_vecs, projected_vecs, k=10)
        assert res["k_requested"] == 10
        assert res["k_used"] == 10


# ============================================================
# 理论信息容量测试
# ============================================================

class TestInformationCapacity:
    """evaluate_information_capacity 测试。"""

    def test_compression_ratio(self):
        res = evaluate_information_capacity(384, 64)
        assert res["compression_ratio"] == pytest.approx(64 / 384)
        assert res["dimensionality_reduction_pct"] == pytest.approx(
            (1 - 64 / 384) * 100)

    def test_dims_recorded(self):
        res = evaluate_information_capacity(384, 64)
        assert res["original_dim"] == 384
        assert res["projected_dim"] == 64

    def test_jl_epsilon_decreases_with_dim(self):
        """维度越大，理论畸变 eps 越小。"""
        small = evaluate_information_capacity(384, 32)
        large = evaluate_information_capacity(384, 128)
        eps_small = small["jl_epsilon_for_n"][1000]
        eps_large = large["jl_epsilon_for_n"][1000]
        assert eps_small > eps_large

    def test_jl_max_n_increases_with_dim(self):
        """维度越大，可嵌入点数越多。"""
        small = evaluate_information_capacity(384, 32)
        large = evaluate_information_capacity(384, 128)
        n_small = small["jl_max_n_for_epsilon"][0.2]
        n_large = large["jl_max_n_for_epsilon"][0.2]
        assert n_small < n_large

    def test_jl_epsilon_for_n_known_value(self):
        """校验一个已知值：k=64, n=1000 -> eps=sqrt(8*ln(1000)/64)。"""
        res = evaluate_information_capacity(384, 64)
        expected = math.sqrt(8 * math.log(1000) / 64)
        assert res["jl_epsilon_for_n"][1000] == pytest.approx(expected)

    def test_invalid_dim_raises(self):
        with pytest.raises(ValueError):
            evaluate_information_capacity(0, 64)
        with pytest.raises(ValueError):
            evaluate_information_capacity(384, -1)


# ============================================================
# 策略对比测试
# ============================================================

class TestCompareProjectionStrategies:
    """compare_projection_strategies 测试（使用随机向量，不依赖预训练模型）。"""

    def test_all_strategies_present(self):
        texts = [f"sample {i}" for i in range(80)]
        res = compare_projection_strategies(texts, original_dim=384,
                                            target_dim=64)
        for name in ("random_projection", "pca", "truncation",
                     "random_selection"):
            assert name in res["strategies"]

    def test_meta_fields(self):
        texts = [f"t{i}" for i in range(50)]
        res = compare_projection_strategies(texts, original_dim=384,
                                            target_dim=64)
        assert res["meta"]["original_dim"] == 384
        assert res["meta"]["target_dim"] == 64
        assert res["meta"]["n_samples"] == 50
        assert res["meta"]["encoder"] == "random(low-rank)"

    def test_each_strategy_has_metric_groups(self):
        texts = [f"t{i}" for i in range(80)]
        res = compare_projection_strategies(texts, original_dim=384,
                                            target_dim=64)
        for name, m in res["strategies"].items():
            for group in ("cosine", "distance", "nn_overlap", "info_capacity"):
                assert group in m, f"{name} 缺少指标组 {group}"

    def test_deterministic(self):
        """相同输入应产生相同结果。"""
        texts = [f"t{i}" for i in range(60)]
        r1 = compare_projection_strategies(texts, original_dim=384,
                                           target_dim=64, seed=42)
        r2 = compare_projection_strategies(texts, original_dim=384,
                                           target_dim=64, seed=42)
        a = r1["strategies"]["random_projection"]["cosine"]["cosine_pearson_correlation"]
        b = r2["strategies"]["random_projection"]["cosine"]["cosine_pearson_correlation"]
        assert a == b

    def test_pca_better_than_truncation_on_structured_data(self, structured_vecs):
        """结构化数据上 PCA 应优于截断（验证指标区分度）。"""
        texts = ["x"] * structured_vecs.shape[0]
        encoder = lambda ts: structured_vecs  # noqa: E731 注入预生成向量
        res = compare_projection_strategies(texts, original_dim=384,
                                            target_dim=64, encoder=encoder)
        pca_cos = res["strategies"]["pca"]["cosine"]["cosine_pearson_correlation"]
        trunc_cos = res["strategies"]["truncation"]["cosine"]["cosine_pearson_correlation"]
        assert pca_cos > trunc_cos

    def test_pca_better_than_random_projection_on_structured_data(
            self, structured_vecs):
        """结构化数据上 PCA 也应优于随机投影。"""
        texts = ["x"] * structured_vecs.shape[0]
        encoder = lambda ts: structured_vecs  # noqa: E731
        res = compare_projection_strategies(texts, original_dim=384,
                                            target_dim=64, encoder=encoder)
        pca_cos = res["strategies"]["pca"]["cosine"]["cosine_pearson_correlation"]
        rp_cos = res["strategies"]["random_projection"]["cosine"]["cosine_pearson_correlation"]
        assert pca_cos > rp_cos

    def test_injected_encoder_used(self):
        """注入的 encoder 应被使用，source 标签相应改变。"""
        calls = {"n": 0}

        def enc(texts):
            calls["n"] += 1
            return np.random.RandomState(123).randn(len(texts), 384)

        texts = [f"t{i}" for i in range(40)]
        res = compare_projection_strategies(texts, original_dim=384,
                                            target_dim=64, encoder=enc)
        assert calls["n"] == 1
        assert res["meta"]["encoder"] == "encoder(injected)"

    def test_random_projection_matches_jl_formula(self, random_vecs):
        """随机投影策略的输出应与 JL 公式一致。"""
        texts = ["x"] * random_vecs.shape[0]
        encoder = lambda ts: random_vecs  # noqa: E731
        res = compare_projection_strategies(texts, original_dim=384,
                                            target_dim=64, encoder=encoder,
                                            seed=42)
        # 直接用同样公式重构投影向量，验证策略输出与之一致
        P = np.random.RandomState(42).randn(384, 64) / math.sqrt(384)
        expected = random_vecs @ P
        from evaluation.projection_eval import evaluate_distance_preservation
        direct = evaluate_distance_preservation(random_vecs, expected)
        strat = res["strategies"]["random_projection"]["distance"]
        assert strat["distance_pearson_correlation"] == pytest.approx(
            direct["distance_pearson_correlation"])


# ============================================================
# 报告生成测试
# ============================================================

class TestGenerateReport:
    """generate_report 测试。"""

    def test_returns_string(self):
        texts = [f"t{i}" for i in range(60)]
        res = compare_projection_strategies(texts, original_dim=384,
                                            target_dim=64)
        report = generate_report(res)
        assert isinstance(report, str)
        assert len(report) > 0

    def test_contains_title(self):
        texts = [f"t{i}" for i in range(60)]
        res = compare_projection_strategies(texts, original_dim=384,
                                            target_dim=64)
        report = generate_report(res)
        assert "投影信息损失评估报告" in report

    def test_contains_strategy_names(self):
        texts = [f"t{i}" for i in range(60)]
        res = compare_projection_strategies(texts, original_dim=384,
                                            target_dim=64)
        report = generate_report(res)
        for name in ("random_projection", "pca", "truncation",
                     "random_selection"):
            assert name in report

    def test_contains_jl_section(self):
        texts = [f"t{i}" for i in range(60)]
        res = compare_projection_strategies(texts, original_dim=384,
                                            target_dim=64)
        report = generate_report(res)
        assert "Johnson-Lindenstrauss" in report
        assert "压缩比" in report

    def test_contains_conclusion(self):
        texts = [f"t{i}" for i in range(60)]
        res = compare_projection_strategies(texts, original_dim=384,
                                            target_dim=64)
        report = generate_report(res)
        assert "结论" in report

    def test_contains_numeric_values(self):
        """报告应包含格式化的数值。"""
        texts = [f"t{i}" for i in range(60)]
        res = compare_projection_strategies(texts, original_dim=384,
                                            target_dim=64)
        report = generate_report(res)
        # 数值以 4 位小数出现，如 0.0000 形式
        import re
        assert re.search(r"\d\.\d{4}", report) is not None

    def test_empty_strategies(self):
        """无策略结果时报告应优雅处理。"""
        report = generate_report({"meta": {}, "strategies": {}})
        assert "投影信息损失评估报告" in report
        assert "无策略结果" in report


# ============================================================
# run_evaluation 测试
# ============================================================

class TestRunEvaluation:
    """run_evaluation 测试。"""

    def test_returns_string(self):
        texts = [f"t{i}" for i in range(60)]
        report = run_evaluation(texts, target_dims=[32, 64], original_dim=384)
        assert isinstance(report, str)

    def test_contains_all_target_dims(self):
        texts = [f"t{i}" for i in range(60)]
        report = run_evaluation(texts, target_dims=[32, 64, 128],
                                original_dim=384)
        for td in (32, 64, 128):
            assert f"目标维度: {td}" in report

    def test_contains_summary(self):
        texts = [f"t{i}" for i in range(60)]
        report = run_evaluation(texts, target_dims=[64], original_dim=384)
        assert "最佳策略" in report


# ============================================================
# 边界情况测试
# ============================================================

class TestEdgeCases:
    """边界情况：空输入、单向量、双向量。"""

    def test_empty_input_cosine(self):
        res = evaluate_cosine_preservation(np.zeros((0, 64)), np.zeros((0, 32)))
        assert res["n_samples"] == 0
        assert res["n_pairs"] == 0
        assert math.isnan(res["cosine_pearson_correlation"])

    def test_empty_input_distance(self):
        res = evaluate_distance_preservation(np.zeros((0, 64)), np.zeros((0, 32)))
        assert res["n_samples"] == 0
        assert math.isnan(res["distance_pearson_correlation"])

    def test_empty_input_nn(self):
        res = evaluate_nearest_neighbor_overlap(np.zeros((0, 64)),
                                                np.zeros((0, 32)), k=5)
        assert res["n_samples"] == 0
        assert res["k_used"] == 0
        assert math.isnan(res["mean_overlap"])

    def test_single_vector(self):
        o = np.random.RandomState(0).randn(1, 64)
        p = np.random.RandomState(1).randn(1, 32)
        res = evaluate_cosine_preservation(o, p)
        assert res["n_samples"] == 1
        assert res["n_pairs"] == 0
        assert math.isnan(res["cosine_pearson_correlation"])
        # kNN 无邻居
        nn = evaluate_nearest_neighbor_overlap(o, p, k=5)
        assert nn["k_used"] == 0
        assert math.isnan(nn["mean_overlap"])

    def test_two_vectors(self):
        """n=2：仅 1 个点对，Pearson 不可定义（NaN），但 MAE 可算。"""
        o = np.random.RandomState(0).randn(2, 64)
        p = np.random.RandomState(1).randn(2, 32)
        res = evaluate_cosine_preservation(o, p)
        assert res["n_pairs"] == 1
        assert math.isnan(res["cosine_pearson_correlation"])
        assert not math.isnan(res["cosine_mae"])
        # kNN：k 收紧为 1，若邻居一致则重叠为 1
        nn = evaluate_nearest_neighbor_overlap(o, p, k=5)
        assert nn["k_used"] == 1

    def test_two_vectors_identity_nn(self):
        """n=2 且投影=原始：唯一邻居必一致，重叠为 1。"""
        o = np.random.RandomState(0).randn(2, 64)
        nn = evaluate_nearest_neighbor_overlap(o, o, k=5)
        assert nn["k_used"] == 1
        assert nn["mean_overlap"] == pytest.approx(1.0)

    def test_mismatched_sample_count_raises(self):
        o = np.zeros((10, 64))
        p = np.zeros((5, 32))
        with pytest.raises(ValueError):
            evaluate_cosine_preservation(o, p)

    def test_compare_with_empty_texts(self):
        """空文本列表不应崩溃。"""
        res = compare_projection_strategies([], original_dim=384,
                                            target_dim=64)
        assert res["meta"]["n_samples"] == 0
        assert "strategies" in res

    def test_compare_with_few_samples(self):
        """样本数少于目标维度（PCA 需降级处理）。"""
        texts = ["a", "b", "c"]
        res = compare_projection_strategies(texts, original_dim=384,
                                            target_dim=64)
        # PCA 仅能产生 n-1=2 个主成分，但不应崩溃
        assert "pca" in res["strategies"]
        for name, m in res["strategies"].items():
            assert "cosine" in m

    def test_target_dim_larger_than_original(self):
        """目标维度大于原始维度时不应崩溃（k 被 clamp）。"""
        texts = [f"t{i}" for i in range(20)]
        res = compare_projection_strategies(texts, original_dim=32,
                                            target_dim=64)
        assert "strategies" in res
        # info_capacity 仍记录请求的 target_dim
        assert res["meta"]["target_dim"] == 64

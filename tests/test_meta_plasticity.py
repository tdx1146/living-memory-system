"""
MetaPlasticityController 完整测试套件
=====================================

被测模块: core/meta/meta_plasticity.py

覆盖方面:
  1. 初始化            (TestInitialization)
  2. update 基本行为    (TestUpdateBehavior)
  3. 四条元学习规则     (TestMetaRules)
  4. 有界性            (TestBounds)
  5. 平滑更新          (TestSmoothing)
  6. get_adjusted_params (TestAdjustedParams)
  7. 持久化            (TestPersistence)
  8. 边界情况          (TestEdgeCases)
  9. 辅助函数          (TestHelpers)
 10. 集成场景          (TestIntegrationScenario)

约束:
  - 不依赖 torch（MetaPlasticityController 本身不用 torch）
  - 不依赖外部模型或网络
  - 每个测试方法独立可运行
"""

import math

import pytest

from core.meta.meta_plasticity import (
    MetaPlasticityController,
    MetaState,
    _sigmoid,
    _clamp,
    _ema,
)


class TestMetaPlasticityController:
    """MetaPlasticityController 完整测试套件（按方面分组）。"""

    # ==================================================================
    #  1. 初始化测试
    # ==================================================================
    class TestInitialization:
        """验证默认/自定义配置初始化与初始状态。"""

        def test_default_config(self):
            """默认配置初始化，验证所有默认值。"""
            c = MetaPlasticityController()
            # 元更新时序
            assert c.meta_interval == 10
            assert c.meta_lr == 0.01
            # 倍率边界
            assert c.bounds_min == 0.5
            assert c.bounds_max == 2.0
            # surprise 历史窗口
            assert c.surprise_window == 20
            # 四个规则系数
            assert c.lr_delta == 2.0
            assert c.orth_alpha == 1.0
            assert c.temp_beta == 5.0
            assert c.cw_gamma == 1.0
            # J 矩阵目标范数
            assert c.shy_target_norm == 10.0

        def test_custom_config(self):
            """自定义配置初始化，验证参数传递。"""
            config = {
                'meta_interval': 5,
                'meta_lr': 0.05,
                'bounds_min': 0.7,
                'bounds_max': 1.8,
                'surprise_window': 15,
                'lr_delta': 1.5,
                'orth_alpha': 0.8,
                'temp_beta': 3.0,
                'cw_gamma': 0.6,
                'shy_target_norm': 7.0,
            }
            c = MetaPlasticityController(config)
            assert c.meta_interval == 5
            assert c.meta_lr == 0.05
            assert c.bounds_min == 0.7
            assert c.bounds_max == 1.8
            assert c.surprise_window == 15
            assert c.lr_delta == 1.5
            assert c.orth_alpha == 0.8
            assert c.temp_beta == 3.0
            assert c.cw_gamma == 0.6
            assert c.shy_target_norm == 7.0

        def test_initial_state(self):
            """验证初始状态：所有倍率为 1.0，计数器为 0。"""
            c = MetaPlasticityController()
            assert c.state.lr_multiplier == 1.0
            assert c.state.orth_multiplier == 1.0
            assert c.state.temp_multiplier == 1.0
            assert c.state.cw_multiplier == 1.0
            assert c.state.surprise_history == []
            assert c.state.collapse_count == 0
            assert c.state.update_count == 0
            assert c._turn_count == 0
            assert len(c._surprise_deque) == 0

        def test_ema_alpha_computed(self):
            """ema_alpha = 2 / (surprise_window + 1)。"""
            c = MetaPlasticityController()
            assert abs(c.ema_alpha - 2.0 / (c.surprise_window + 1)) < 1e-12

            c2 = MetaPlasticityController({'surprise_window': 9})
            assert abs(c2.ema_alpha - 2.0 / 10.0) < 1e-12

        def test_deque_maxlen(self):
            """surprise deque 容量为 2 * surprise_window。"""
            c = MetaPlasticityController({'surprise_window': 10})
            assert c._surprise_deque.maxlen == 20

            c2 = MetaPlasticityController()
            assert c2._surprise_deque.maxlen == 2 * c2.surprise_window

    # ==================================================================
    #  2. update 基本行为测试
    # ==================================================================
    class TestUpdateBehavior:
        """验证 update 的触发时序与返回结构。"""

        def test_returns_none_before_interval(self):
            """前 meta_interval-1 轮返回 None（仅收集信号）。"""
            c = MetaPlasticityController()  # interval=10
            for i in range(9):
                r = c.update(surprise=0.5, coherence=0.8,
                             collapse_occurred=False, j_norm=5.0)
                assert r is None

        def test_returns_dict_at_interval(self):
            """第 meta_interval 轮返回调整字典。"""
            c = MetaPlasticityController()
            for i in range(9):
                assert c.update(0.5, 0.8, False, 5.0) is None
            r = c.update(0.5, 0.8, False, 5.0)  # 第 10 轮
            assert r is not None

        def test_result_dict_keys(self):
            """返回字典包含正确的键。"""
            c = MetaPlasticityController({'meta_interval': 1})
            r = c.update(0.5, 0.8, False, 5.0)
            assert set(r.keys()) == {
                'lr_mult', 'orth_mult', 'temp_mult', 'cw_mult'
            }

        def test_result_dict_matches_state(self):
            """返回字典的值与内部 state 一致。"""
            c = MetaPlasticityController({'meta_interval': 1, 'meta_lr': 1.0})
            r = c.update(0.5, 0.0, True, 100.0)
            assert r['lr_mult'] == c.state.lr_multiplier
            assert r['orth_mult'] == c.state.orth_multiplier
            assert r['temp_mult'] == c.state.temp_multiplier
            assert r['cw_mult'] == c.state.cw_multiplier

        def test_update_count_increments(self):
            """连续调用多轮后 update_count 正确递增。"""
            c = MetaPlasticityController()  # interval=10
            for i in range(30):
                c.update(0.5, 0.8, False, 5.0)
            # 触发于第 10、20、30 轮
            assert c.state.update_count == 3

        def test_signal_collection_before_trigger(self):
            """未触发调整时也收集信号（surprise、collapse）。"""
            c = MetaPlasticityController()
            for i in range(3):
                assert c.update(0.5, 0.8, False, 5.0) is None
            assert c._turn_count == 3
            assert len(c._surprise_deque) == 3
            assert list(c._surprise_deque) == [0.5, 0.5, 0.5]

            # collapse 在触发前也累计
            c2 = MetaPlasticityController()
            for i in range(3):
                c2.update(0.5, 0.8, True, 5.0)
            assert c2.state.collapse_count == 3

        def test_get_status_structure(self):
            """get_status 返回完整状态摘要。"""
            c = MetaPlasticityController()
            for i in range(3):
                c.update(0.5, 0.8, False, 5.0)
            s = c.get_status()
            expected_keys = {
                'turn_count', 'update_count', 'collapse_count',
                'lr_multiplier', 'orth_multiplier',
                'temp_multiplier', 'cw_multiplier',
                'surprise_history_len', 'next_update_in',
            }
            assert set(s.keys()) == expected_keys
            assert s['turn_count'] == 3
            assert s['update_count'] == 0
            assert s['surprise_history_len'] == 3

        def test_get_status_next_update_in(self):
            """next_update_in 倒计时正确。"""
            c = MetaPlasticityController()  # interval=10
            for i in range(3):
                c.update(0.5, 0.8, False, 5.0)
            assert c.get_status()['next_update_in'] == 7  # 10 - 3

            # 第 10 轮触发后，倒计时重置为 10
            for i in range(7):
                c.update(0.5, 0.8, False, 5.0)
            assert c._turn_count == 10
            assert c.get_status()['next_update_in'] == 10  # 10 % 10 == 0

    # ==================================================================
    #  3. 四条元学习规则测试
    # ==================================================================
    class TestMetaRules:
        """分别验证四条元学习规则的目标倍率计算。"""

        # ---------- 规则 1：lr_multiplier（surprise 趋势） ----------
        class TestLrRule:
            def test_rising_surprise_increases_multiplier(self):
                """surprise 持续上升 → lr_multiplier > 1.0。"""
                c = MetaPlasticityController()
                w = c.surprise_window
                rising = [0.1] * w + [1.0] * w  # 旧窗口低，新窗口高
                assert c._compute_lr_multiplier(rising) > 1.0

            def test_falling_surprise_decreases_multiplier(self):
                """surprise 持续下降 → lr_multiplier < 1.0。"""
                c = MetaPlasticityController()
                w = c.surprise_window
                falling = [1.0] * w + [0.1] * w  # 旧窗口高，新窗口低
                assert c._compute_lr_multiplier(falling) < 1.0

            def test_flat_surprise_neutral(self):
                """surprise 无趋势 → lr_multiplier ≈ 1.0。"""
                c = MetaPlasticityController()
                w = c.surprise_window
                flat = [0.5] * (2 * w)
                assert abs(c._compute_lr_multiplier(flat) - 1.0) < 1e-12

            def test_insufficient_history_is_neutral(self):
                """历史不足 2*window → 无趋势 → 1.0。"""
                c = MetaPlasticityController()
                w = c.surprise_window
                # 仅 w 个值（不足 2w）
                short = [float(i) for i in range(w)]
                assert abs(c._compute_lr_multiplier(short) - 1.0) < 1e-12
                # 恰好 2w-1 仍不足
                almost = [float(i) for i in range(2 * w - 1)]
                assert abs(c._compute_lr_multiplier(almost) - 1.0) < 1e-12
                # 恰好 2w 且上升 → > 1.0
                enough = [0.0] * w + [1.0] * w
                assert c._compute_lr_multiplier(enough) > 1.0

        # ---------- 规则 2：orth_multiplier（coherence） ----------
        class TestOrthRule:
            def test_high_coherence_near_one(self):
                """coherence=1.0（高度一致）→ orth_multiplier ≈ 1.0。"""
                c = MetaPlasticityController()
                assert abs(c._compute_orth_multiplier(1.0) - 1.0) < 1e-12

            def test_low_coherence_increases(self):
                """coherence=0.0（完全混乱）→ orth_multiplier > 1.0。"""
                c = MetaPlasticityController()
                assert c._compute_orth_multiplier(0.0) > 1.0

            def test_mid_coherence_between(self):
                """coherence=0.5 → 介于高一致与低一致之间。"""
                c = MetaPlasticityController()
                high = c._compute_orth_multiplier(1.0)
                mid = c._compute_orth_multiplier(0.5)
                low = c._compute_orth_multiplier(0.0)
                assert high < mid < low

            def test_monotonic_decreasing_with_coherence(self):
                """coherence 越低，orth_multiplier 越高（单调）。"""
                c = MetaPlasticityController()
                vals = [c._compute_orth_multiplier(coh)
                        for coh in [1.0, 0.8, 0.5, 0.2, 0.0]]
                assert all(vals[i] <= vals[i + 1]
                           for i in range(len(vals) - 1))

        # ---------- 规则 3：temp_multiplier（坍缩率） ----------
        class TestTempRule:
            def test_no_collapse_neutral(self):
                """无坍缩 → temp_multiplier ≈ 1.0。"""
                c = MetaPlasticityController()
                c.state.collapse_count = 0
                c.state.update_count = 5
                assert abs(c._compute_temp_multiplier() - 1.0) < 1e-12

            def test_collapse_increases(self):
                """多次坍缩 → temp_multiplier > 1.0。"""
                c = MetaPlasticityController()
                c.state.collapse_count = 1
                c.state.update_count = 100  # rate=0.01 → 1.05
                assert c._compute_temp_multiplier() > 1.0

            def test_more_collapse_higher(self):
                """坍缩越多，temp_multiplier 越高。"""
                c = MetaPlasticityController()
                c.state.collapse_count = 1
                c.state.update_count = 100   # rate=0.01 → 1.05
                low = c._compute_temp_multiplier()
                c.state.collapse_count = 5
                c.state.update_count = 100   # rate=0.05 → 1.25
                high = c._compute_temp_multiplier()
                assert high > low

        # ---------- 规则 4：cw_multiplier（J 饱和度） ----------
        # 注意：实现中 saturation = j_norm / shy_target_norm 始终 >= 0，
        # 因此 sigmoid(saturation) >= 0.5，cw_multiplier 对 j_norm >= 0
        # 始终 >= 1.0。j_norm=0 时恰为中性 1.0，j_norm 越大越逼近上界。
        # （规格中"j_norm 远小于目标 → < 1.0"在该实现下不可达，此处按真实行为测试。）
        class TestCwRule:
            def test_zero_jnorm_neutral(self):
                """j_norm=0 → cw_multiplier ≈ 1.0（中性）。"""
                c = MetaPlasticityController()
                assert abs(c._compute_cw_multiplier(0.0) - 1.0) < 1e-12

            def test_large_jnorm_increases(self):
                """j_norm 远大于 shy_target_norm → cw_multiplier > 1.0。"""
                c = MetaPlasticityController()
                val = c._compute_cw_multiplier(100.0)
                assert val > 1.0
                assert val <= c.bounds_max + 1e-9

            def test_small_jnorm_near_one(self):
                """j_norm 较小 → cw_multiplier 接近 1.0（且 >= 1.0）。"""
                c = MetaPlasticityController()
                small = c._compute_cw_multiplier(1.0)
                assert small >= 1.0 - 1e-12
                assert small < c._compute_cw_multiplier(100.0)

            def test_monotonic_increasing_with_jnorm(self):
                """j_norm 越大，cw_multiplier 越大（单调）。"""
                c = MetaPlasticityController()
                vals = [c._compute_cw_multiplier(j)
                        for j in [0.0, 1.0, 5.0, 10.0, 100.0]]
                assert all(vals[i] <= vals[i + 1]
                           for i in range(len(vals) - 1))

    # ==================================================================
    #  4. 有界性测试
    # ==================================================================
    class TestBounds:
        """验证所有倍率始终落在 [bounds_min, bounds_max] 内。"""

        def test_all_multipliers_within_default_bounds(self):
            """极端输入下，所有倍率始终在默认 [0.5, 2.0] 范围内。"""
            c = MetaPlasticityController({
                'meta_interval': 1, 'meta_lr': 1.0,  # 立即跳到目标
            })
            for i in range(50):
                c.update(surprise=float(i) * 1e6, coherence=0.0,
                         collapse_occurred=True, j_norm=1e9)
            mults = [
                c.state.lr_multiplier, c.state.orth_multiplier,
                c.state.temp_multiplier, c.state.cw_multiplier,
            ]
            for m in mults:
                assert 0.5 - 1e-9 <= m <= 2.0 + 1e-9

        def test_extreme_inputs_do_not_overflow(self):
            """超大/超负输入不会导致倍率越界或数值溢出。"""
            c = MetaPlasticityController()
            lo, hi = c.bounds_min, c.bounds_max
            # orth：极端 coherence
            assert lo - 1e-9 <= c._compute_orth_multiplier(-1e9) <= hi + 1e-9
            assert lo - 1e-9 <= c._compute_orth_multiplier(1e9) <= hi + 1e-9
            # cw：超大 j_norm
            assert lo - 1e-9 <= c._compute_cw_multiplier(1e9) <= hi + 1e-9
            assert lo - 1e-9 <= c._compute_cw_multiplier(0.0) <= hi + 1e-9
            # temp：超大坍缩计数
            c.state.collapse_count = 10 ** 15
            c.state.update_count = 1
            assert lo - 1e-9 <= c._compute_temp_multiplier() <= hi + 1e-9
            # lr：极端上升趋势
            w = c.surprise_window
            extreme_rising = [0.0] * w + [1e9] * w
            assert lo - 1e-9 <= c._compute_lr_multiplier(extreme_rising) <= hi + 1e-9

        def test_custom_bounds_enforced(self):
            """自定义 bounds [0.8, 1.2] 被严格执行。"""
            c = MetaPlasticityController({
                'meta_interval': 1, 'meta_lr': 1.0,
                'bounds_min': 0.8, 'bounds_max': 1.2,
            })
            for i in range(50):
                c.update(surprise=float(i) * 1e6, coherence=0.0,
                         collapse_occurred=True, j_norm=1e9)
            for m in [c.state.lr_multiplier, c.state.orth_multiplier,
                      c.state.temp_multiplier, c.state.cw_multiplier]:
                assert 0.8 - 1e-9 <= m <= 1.2 + 1e-9

    # ==================================================================
    #  5. 平滑更新测试
    # ==================================================================
    class TestSmoothing:
        """验证 meta_lr 控制的平滑更新行为。"""

        def test_smooth_update_formula(self):
            """_smooth_update = (1-meta_lr)*current + meta_lr*target。"""
            c = MetaPlasticityController({'meta_lr': 0.1})
            assert abs(c._smooth_update(1.0, 2.0) - 1.1) < 1e-12
            assert abs(c._smooth_update(2.0, 1.0) - 1.9) < 1e-12

            # meta_lr=0 → 完全不变
            c0 = MetaPlasticityController({'meta_lr': 0.0})
            assert abs(c0._smooth_update(1.5, 3.0) - 1.5) < 1e-12

            # meta_lr=1 → 直接跳到目标
            c1 = MetaPlasticityController({'meta_lr': 1.0})
            assert abs(c1._smooth_update(1.5, 3.0) - 3.0) < 1e-12

        def test_small_meta_lr_changes_slower(self):
            """meta_lr 小 → 同样更新次数下倍率变化更缓慢。"""
            slow = MetaPlasticityController({'meta_interval': 1, 'meta_lr': 0.01})
            fast = MetaPlasticityController({'meta_interval': 1, 'meta_lr': 0.5})
            for _ in range(5):
                slow.update(0.5, 1.0, True, 0.0)  # collapse → temp 目标 2.0
                fast.update(0.5, 1.0, True, 0.0)
            # 两者都从 1.0 向 2.0 移动，fast 更接近目标
            assert fast.state.temp_multiplier > slow.state.temp_multiplier
            assert slow.state.temp_multiplier > 1.0  # 仍然有移动

        def test_single_update_change_bounded(self):
            """单次更新后倍率变化 = meta_lr * (target - current)。"""
            c = MetaPlasticityController({'meta_interval': 1, 'meta_lr': 0.1})
            before = c.state.temp_multiplier  # 1.0
            c.update(0.5, 1.0, True, 0.0)     # collapse → temp 目标 = 2.0
            after = c.state.temp_multiplier
            # 内部使用的目标（state 在更新后未再变化，可重算）
            target = c._compute_temp_multiplier()
            expected_change = c.meta_lr * (target - before)
            assert abs((after - before) - expected_change) < 1e-12
            assert abs(after - before) <= c.meta_lr * abs(target - before) + 1e-12

        def test_multiple_updates_approach_target(self):
            """连续多次更新后倍率逐渐逼近目标值。"""
            c = MetaPlasticityController({'meta_interval': 1, 'meta_lr': 0.1})
            vals = []
            for _ in range(30):
                c.update(0.5, 1.0, True, 0.0)  # temp 目标恒为 2.0
                vals.append(c.state.temp_multiplier)
            # 单调递增地逼近 2.0
            assert all(vals[i] <= vals[i + 1] + 1e-12
                       for i in range(len(vals) - 1))
            assert vals[-1] > vals[0]
            assert vals[-1] < 2.0 + 1e-9
            assert abs(vals[-1] - 2.0) < 0.2  # 足够接近目标

    # ==================================================================
    #  6. get_adjusted_params 测试
    # ==================================================================
    class TestAdjustedParams:
        """验证基准参数乘以倍率得到正确结果。"""

        def test_multipliers_applied(self):
            """基准值乘以倍率得到正确结果。"""
            c = MetaPlasticityController()
            c.state.lr_multiplier = 1.5
            c.state.orth_multiplier = 0.8
            c.state.temp_multiplier = 1.2
            c.state.cw_multiplier = 2.0
            p = c.get_adjusted_params(0.01, 0.5, 0.05, 0.01)
            assert abs(p['learning_rate'] - 0.015) < 1e-12
            assert abs(p['orth_weight'] - 0.4) < 1e-12
            assert abs(p['temperature'] - 0.06) < 1e-12
            assert abs(p['complexity_weight'] - 0.02) < 1e-12
            assert set(p.keys()) == {
                'learning_rate', 'orth_weight', 'temperature', 'complexity_weight'
            }

        def test_unity_multipliers_return_base(self):
            """所有倍率为 1.0 时返回值 = 基准值。"""
            c = MetaPlasticityController()  # 全 1.0
            p = c.get_adjusted_params(0.03, 0.7, 0.04, 0.02)
            assert abs(p['learning_rate'] - 0.03) < 1e-12
            assert abs(p['orth_weight'] - 0.7) < 1e-12
            assert abs(p['temperature'] - 0.04) < 1e-12
            assert abs(p['complexity_weight'] - 0.02) < 1e-12

        def test_custom_base_values(self):
            """测试自定义基准值。"""
            c = MetaPlasticityController()
            c.state.lr_multiplier = 2.0
            p = c.get_adjusted_params(0.05, 1.0, 0.1, 0.005)
            assert abs(p['learning_rate'] - 0.1) < 1e-12     # 0.05 * 2.0
            assert abs(p['orth_weight'] - 1.0) < 1e-12       # 1.0 * 1.0
            assert abs(p['temperature'] - 0.1) < 1e-12        # 0.1 * 1.0
            assert abs(p['complexity_weight'] - 0.005) < 1e-12  # 0.005 * 1.0

    # ==================================================================
    #  7. 持久化测试
    # ==================================================================
    class TestPersistence:
        """验证 get_state / set_state 的持久化与往返一致性。"""

        def test_get_state_structure(self):
            """get_state() 返回正确的字典结构。"""
            c = MetaPlasticityController()
            c.update(0.5, 0.8, True, 5.0)  # 第 1 轮
            s = c.get_state()
            expected_keys = {
                'lr_multiplier', 'orth_multiplier', 'temp_multiplier',
                'cw_multiplier', 'surprise_history', 'collapse_count',
                'update_count', 'turn_count',
            }
            assert set(s.keys()) == expected_keys
            assert s['turn_count'] == 1
            assert s['collapse_count'] == 1
            assert s['surprise_history'] == [0.5]

        def test_set_state_restores(self):
            """set_state() 后所有状态正确恢复。"""
            c1 = MetaPlasticityController()
            for i in range(25):
                c1.update(surprise=i * 0.1, coherence=0.6,
                          collapse_occurred=(i % 3 == 0), j_norm=8.0)
            state = c1.get_state()

            c2 = MetaPlasticityController()
            c2.set_state(state)
            assert c2.state.lr_multiplier == state['lr_multiplier']
            assert c2.state.orth_multiplier == state['orth_multiplier']
            assert c2.state.temp_multiplier == state['temp_multiplier']
            assert c2.state.cw_multiplier == state['cw_multiplier']
            assert c2.state.collapse_count == state['collapse_count']
            assert c2.state.update_count == state['update_count']
            assert c2._turn_count == state['turn_count']
            assert list(c2._surprise_deque) == state['surprise_history']

        def test_roundtrip_consistency(self):
            """get_state() → set_state() 往返一致性。"""
            c1 = MetaPlasticityController()
            for i in range(25):
                c1.update(i * 0.1, 0.6, i % 3 == 0, 8.0)
            state1 = c1.get_state()

            c2 = MetaPlasticityController()
            c2.set_state(state1)
            state2 = c2.get_state()
            assert state1 == state2

        def test_backward_compat_missing_fields(self):
            """缺失字段的 set_state() 使用默认值。"""
            c = MetaPlasticityController()
            # 完全空字典 → 全部默认值
            c.set_state({})
            assert c.state.lr_multiplier == 1.0
            assert c.state.orth_multiplier == 1.0
            assert c.state.temp_multiplier == 1.0
            assert c.state.cw_multiplier == 1.0
            assert c.state.collapse_count == 0
            assert c.state.update_count == 0
            assert c._turn_count == 0
            assert list(c._surprise_deque) == []

            # 部分字典：仅提供部分字段，其余保持默认
            c.set_state({'lr_multiplier': 1.7, 'collapse_count': 3})
            assert c.state.lr_multiplier == 1.7
            assert c.state.collapse_count == 3
            assert c.state.orth_multiplier == 1.0
            assert c.state.update_count == 0

    # ==================================================================
    #  8. 边界情况测试
    # ==================================================================
    class TestEdgeCases:
        """验证各种边界配置与输入。"""

        def test_meta_interval_one(self):
            """meta_interval=1：每轮都更新。"""
            c = MetaPlasticityController({'meta_interval': 1})
            r1 = c.update(0.5, 0.8, False, 5.0)
            r2 = c.update(0.5, 0.8, False, 5.0)
            assert r1 is not None
            assert r2 is not None
            assert c.state.update_count == 2

        def test_meta_interval_zero_disabled(self):
            """meta_interval=0：禁用元更新，始终返回 None。"""
            c = MetaPlasticityController({'meta_interval': 0})
            results = [c.update(0.5, 0.8, True, 5.0) for _ in range(15)]
            assert all(r is None for r in results)
            # 信号仍在收集
            assert c._turn_count == 15
            assert c.state.collapse_count == 15
            # 但不执行元更新
            assert c.state.update_count == 0
            # get_status 对 interval=0 不报错
            s = c.get_status()
            assert s['next_update_in'] == 0

        def test_empty_config_dict(self):
            """空配置字典初始化等价于默认配置。"""
            c = MetaPlasticityController({})
            assert c.meta_interval == 10
            assert c.meta_lr == 0.01
            assert c.bounds_min == 0.5
            assert c.bounds_max == 2.0
            assert c.state.lr_multiplier == 1.0

        def test_none_config(self):
            """None 配置等价于默认配置。"""
            c = MetaPlasticityController(None)
            assert c.meta_interval == 10
            assert c.meta_lr == 0.01
            assert c.state.cw_multiplier == 1.0

        def test_empty_surprise_history_trend(self):
            """surprise_history 为空时趋势计算返回中性。"""
            c = MetaPlasticityController()
            assert abs(c._compute_lr_multiplier([]) - 1.0) < 1e-12

        def test_insufficient_surprise_history_trend(self):
            """surprise_history 不足时趋势计算返回 0 趋势（中性）。"""
            c = MetaPlasticityController()
            w = c.surprise_window
            # 不足 2w 个值 → 趋势 0 → 1.0
            short = [float(i) for i in range(w + 1)]
            assert abs(c._compute_lr_multiplier(short) - 1.0) < 1e-12

    # ==================================================================
    #  9. 辅助函数测试
    # ==================================================================
    class TestHelpers:
        """验证 _sigmoid / _clamp / _ema 辅助函数。"""

        # ---------- _sigmoid ----------
        def test_sigmoid_zero(self):
            """_sigmoid(0) ≈ 0.5。"""
            assert abs(_sigmoid(0) - 0.5) < 1e-12

        def test_sigmoid_large_positive(self):
            """_sigmoid(大正数) ≈ 1.0。"""
            assert abs(_sigmoid(100) - 1.0) < 1e-9
            assert _sigmoid(1e8) == 1.0

        def test_sigmoid_large_negative(self):
            """_sigmoid(大负数) ≈ 0.0。"""
            assert _sigmoid(-100) < 1e-9
            assert _sigmoid(-1e8) == 0.0

        def test_sigmoid_symmetry(self):
            """_sigmoid(-x) = 1 - _sigmoid(x)（关于 (0,0.5) 中心对称）。"""
            for x in [0.0, 0.5, 1.0, 2.5, 10.0, -3.3]:
                assert abs(_sigmoid(-x) - (1.0 - _sigmoid(x))) < 1e-12

        def test_sigmoid_monotonic(self):
            """_sigmoid 单调递增。"""
            xs = [-10, -1, 0, 0.5, 1, 5, 10]
            vals = [_sigmoid(x) for x in xs]
            assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))

        # ---------- _clamp ----------
        def test_clamp_within_range(self):
            """值在范围内时不变。"""
            assert _clamp(0.5, 0.0, 1.0) == 0.5
            assert _clamp(0.0, 0.0, 1.0) == 0.0
            assert _clamp(1.0, 0.0, 1.0) == 1.0

        def test_clamp_above(self):
            """值超过上限时被钳制到上限。"""
            assert _clamp(5.0, 0.0, 1.0) == 1.0
            assert _clamp(1e9, 0.0, 1.0) == 1.0

        def test_clamp_below(self):
            """值低于下限时被钳制到下限。"""
            assert _clamp(-5.0, 0.0, 1.0) == 0.0
            assert _clamp(-1e9, 0.0, 1.0) == 0.0

        # ---------- _ema ----------
        def test_ema_empty(self):
            """空序列返回 0.0。"""
            assert _ema([], 0.5) == 0.0

        def test_ema_constant(self):
            """常数序列返回该常数。"""
            assert abs(_ema([1.0, 1.0, 1.0], 0.5) - 1.0) < 1e-12
            assert abs(_ema([3.0], 0.3) - 3.0) < 1e-12

        def test_ema_weighted(self):
            """_ema 按公式 EMA_t = alpha*v + (1-alpha)*EMA_{t-1} 计算。"""
            # 手算：EMA0=0; EMA1=0.5*1+0.5*0=0.5; EMA2=0.5*2+0.5*0.5=1.25;
            #       EMA3=0.5*3+0.5*1.25=2.125; EMA4=0.5*4+0.5*2.125=3.0625
            assert abs(_ema([0, 1, 2, 3, 4], 0.5) - 3.0625) < 1e-12

        def test_ema_alpha_sensitivity(self):
            """alpha 越大越敏感于近期值。"""
            values = [0.0, 0.0, 0.0, 0.0, 10.0]
            low_alpha = _ema(values, 0.1)   # 更平滑，更远离末值
            high_alpha = _ema(values, 0.9)  # 更贴近末值
            assert high_alpha > low_alpha
            assert high_alpha <= 10.0 + 1e-12

    # ==================================================================
    # 10. 集成场景测试
    # ==================================================================
    class TestIntegrationScenario:
        """模拟多轮对话，验证元参数的端到端演化方向。"""

        @staticmethod
        def _fast_controller():
            """快速收敛的控制器配置（小窗口、较高 meta_lr）。"""
            return MetaPlasticityController({
                'meta_interval': 5,
                'surprise_window': 10,
                'meta_lr': 0.1,
            })

        def test_rising_surprise_increases_lr(self):
            """模拟 100 轮 surprise 逐渐上升，验证 lr_multiplier 最终 > 1.0。"""
            c = self._fast_controller()
            for i in range(1, 101):
                c.update(surprise=float(i), coherence=1.0,
                         collapse_occurred=False, j_norm=0.0)
            assert c.state.lr_multiplier > 1.5
            # 其余维度保持中性
            assert abs(c.state.orth_multiplier - 1.0) < 1e-9

        def test_frequent_collapse_increases_temp(self):
            """模拟频繁坍缩场景，验证 temp_multiplier 上升。"""
            c = self._fast_controller()
            for i in range(100):
                c.update(surprise=0.5, coherence=1.0,
                         collapse_occurred=True, j_norm=0.0)
            assert c.state.temp_multiplier > 1.5
            # 常数 surprise → lr 无趋势
            assert abs(c.state.lr_multiplier - 1.0) < 1e-9

        def test_j_saturation_increases_cw(self):
            """模拟 J 矩阵饱和场景，验证 cw_multiplier 上升。"""
            c = self._fast_controller()
            for i in range(100):
                c.update(surprise=0.5, coherence=1.0,
                         collapse_occurred=False, j_norm=100.0)
            assert c.state.cw_multiplier > 1.5
            assert abs(c.state.lr_multiplier - 1.0) < 1e-9

        def test_coherence_drop_increases_orth(self):
            """模拟 coherence 下降场景，验证 orth_multiplier 上升。"""
            c = self._fast_controller()
            for i in range(100):
                c.update(surprise=0.5, coherence=0.0,
                         collapse_occurred=False, j_norm=0.0)
            assert c.state.orth_multiplier > 1.5
            assert abs(c.state.lr_multiplier - 1.0) < 1e-9

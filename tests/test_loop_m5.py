"""M5 loop 重组：turn 生命周期 9 步固化 + 状态管理清单（§1.2 / §7.1 M5）。

被测模块:
  - runtime/lifecycle.py —— 9 步固化清单（TURN_LIFECYCLE_9）+
    状态管理清单（STATE_OWNERSHIP）+ 每轮生命周期记录器（LifecycleTrace）
  - runtime/loop.py —— process_turn 9 步固化 + _increment_turn 唯一增量点
    + get_status lifecycle 观测块

覆盖（规格 §6.2 新增单测的 M5 部分 + §7.1 验收）:
  1. 固化清单：9 步、规格步名/序；状态管理清单六项单一写者      (TestManifest)
  2. 唯一增量点：每轮恰 +1；/recall /react /merged 只读口零增量；
     load_state 是赋值恢复不是增量                               (TestUniqueTurnIncrement)
  3. 9 步无旁路：process_turn 每轮 broken=False、9 步各恰一次；
     未知步/重复步 → 汇总 broken=True（机器可验红线）            (TestNineStepLifecycle)
  4. 状态汇总：结构（steps/execution_order/spec_order/turns/
     duration_seconds）；观测全为可序列化标量                    (TestLifecycleSummary)
  5. 记录器零副作用：只读口不产生生命周期记录；观测不泄漏条目对象
     （可 json 序列化）                                         (TestLifecycleTraceSideEffectFree)
  6. 只读四不变在 M5 重组后仍成立（/recall 与 /react 检索段）    (TestReadonlyFourInvariantsAfterM5)
  7. get_status lifecycle 观测块 + C-01 必需字段不变             (TestStatusLifecycleBlock)
  8. M1-M4 集成收口：commit 步写侧统一入口真实落库（episodic_delta
     ≥1，source='external'）；怀疑状态机计数进 doubt_check 观测  (TestM1M4IntegrationConsolidation)

设计约束:
  - 不改动其他源文件；仅新增本测试文件（与 M1-M4 同款"一次一变量"纪律）
  - 与既有风格一致：小规模配置 + 固定 seed + 纯逻辑断言
"""

import json
from pathlib import Path

import pytest
import torch

from runtime.lifecycle import (
    LifecycleTrace,
    STATE_OWNERSHIP,
    TURN_LIFECYCLE_9,
    lifecycle_step,
    state_ownership,
)
from runtime.loop import LivingMemoryLoop, MODULE_CLAIMS
from runtime.config import default_config
from core.sensory.embedder import Embedder


# ================================================================== #
#  辅助
# ================================================================== #

class FakeEmbedder(Embedder):
    """带 embed_text 的确定性假嵌入器（触发写侧语义向量路径）。

    embed_text / embed_text_raw 同源同维（dim=input_dim），使 encode、
    语义存储、检索全链路维度一致；向量由文本确定性生成（测试可复现）。
    """

    def __init__(self, dim: int = 32):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, tokens: list) -> torch.Tensor:
        return torch.zeros(self._dim)

    def embed_text(self, text: str) -> torch.Tensor:
        vec = torch.zeros(self._dim)
        for i, ch in enumerate(text):
            vec[i % self._dim] += ord(ch) / 1000.0
        return vec

    def embed_text_raw(self, text: str) -> torch.Tensor:
        return self.embed_text(text)


def make_loop(**overrides) -> LivingMemoryLoop:
    """小规模测试 loop（带语义嵌入路径，32 维）。"""
    torch.manual_seed(42)
    config = default_config()
    config.update({
        'num_nodes': 32,
        'input_dim': 32,
        'num_infer_steps': 5,
        'consolidation_interval': 3,
        'seed': 42,
        'embedder': FakeEmbedder(dim=32),
    })
    config.update(overrides)
    return LivingMemoryLoop(config)


SPEC_STEP_NAMES = [
    'ingest', 'encode', 'query', 'retrieve', 'integrate',
    'doubt_check', 'commit', 'state_update', 'emit',
]


# ================================================================== #
#  1. 固化清单（TURN_LIFECYCLE_9 / STATE_OWNERSHIP）
# ================================================================== #

class TestManifest:
    """9 步固化清单 + 状态管理清单（规格 §1.2 机器可读落地）。"""

    def test_nine_steps_spec_names_and_order(self):
        names = [s.name for s in TURN_LIFECYCLE_9]
        assert names == SPEC_STEP_NAMES  # 规格 §1.2 规范步名与序
        assert [s.idx for s in TURN_LIFECYCLE_9] == list(range(1, 10))
        assert len(set(names)) == 9  # 步名唯一

    def test_each_step_has_spec_and_anchor(self):
        for s in TURN_LIFECYCLE_9:
            assert s.spec.strip()
            assert s.anchor.strip()

    def test_lifecycle_step_lookup(self):
        assert lifecycle_step('ingest').idx == 1
        assert lifecycle_step('emit').idx == 9
        with pytest.raises(KeyError):
            lifecycle_step('不存在步')

    def test_state_ownership_six_entries(self):
        states = [o.state for o in STATE_OWNERSHIP]
        assert states == [
            'turn 计数',
            'J 内容/工作点/σ 估计',
            'π（精度，按通道/来源）',
            'entry.confidence',
            'entry 怀疑态',
            'fok_unresolved（gap 登记）',
        ]
        assert len(set(states)) == 6

    def test_state_ownership_single_writer(self):
        # 每个状态唯一写者（规格 §1.2 表第三列）——写侧不打架
        writers = [o.writer for o in STATE_OWNERSHIP]
        assert len(set(writers)) == len(writers)

    def test_state_ownership_lookup(self):
        assert 'process_turn 唯一' in state_ownership('turn 计数').writer
        with pytest.raises(KeyError):
            state_ownership('不存在状态')

    def test_claims_json_consistent_with_module(self):
        claims_path = Path(__file__).resolve().parent.parent / 'runtime' / 'claims.json'
        data = json.loads(claims_path.read_text(encoding='utf-8'))
        assert data['module'] == 'runtime/loop'
        assert data['milestone'] == 'M5'
        assert set(data['claims'].keys()) == set(MODULE_CLAIMS['claims'].keys())
        for key, claim in MODULE_CLAIMS['claims'].items():
            assert claim['statement'].strip()
            assert 'verified_by' in claim


# ================================================================== #
#  2. turn 计数唯一增量点（§1.2：process_turn 唯一）
# ================================================================== #

class TestUniqueTurnIncrement:
    """turn 计数唯一增量点：每轮恰 +1；只读口零增量。"""

    def test_turn_increments_exactly_once_per_turn(self):
        loop = make_loop()
        for expected in (1, 2, 3):
            loop.process_turn(f'第{expected}轮输入内容。', '助手回复。')
            assert loop.turn_count == expected
            assert loop.last_turn_lifecycle['turns'] == expected
            assert loop.last_turn_lifecycle['broken'] is False
            assert loop.last_turn_lifecycle['steps']['emit']['turn_count'] == expected

    def test_increment_turn_is_sole_increment_site(self):
        # 全库唯一 `self.turn_count += 1` 必须位于 _increment_turn 方法体内
        # （机器可验 claim：turn_increment_unique）。
        src_path = Path(__file__).resolve().parent.parent / 'runtime' / 'loop.py'
        lines = src_path.read_text(encoding='utf-8').splitlines()
        hits = [i for i, ln in enumerate(lines)
                if 'self.turn_count += 1' in ln]
        assert len(hits) == 1, f'期望唯一增量点，实际 {len(hits)} 处'
        idx = hits[0]
        for i in range(idx, -1, -1):
            if lines[i].lstrip().startswith('def '):
                assert 'def _increment_turn' in lines[i]
                break
        else:
            pytest.fail('self.turn_count += 1 不在任何方法内')

    def test_readonly_endpoints_zero_increment(self):
        loop = make_loop()
        loop.process_turn('先跑一轮。', '回复。')
        before = loop.turn_count
        before_lifecycle = loop.last_turn_lifecycle
        loop.recall_episodic_readonly('只读查询', k=3)
        loop.react_readonly('只读反应')
        loop.recall_merged_readonly('合并只读查询', k=3)
        assert loop.turn_count == before  # /recall /react /merged 零增量
        # 只读口不产生新生命周期记录（不 process_turn 则无记录）
        assert loop.last_turn_lifecycle is before_lifecycle
        assert loop.lifecycle_trace is None

    def test_load_state_restores_not_increments(self, tmp_path):
        loop = make_loop()
        loop.process_turn('第一轮。', '回复。')
        loop.process_turn('第二轮。', '回复。')
        assert loop.turn_count == 2
        path = str(tmp_path / 'snap.pt')
        assert loop.save_state(path) is True
        loaded = make_loop()
        loaded.load_state(path)
        # 快照恢复 = 赋值恢复（不是增量）：turn_count 恢复为 2，不再 +1
        assert loaded.turn_count == 2
        assert loaded.last_turn_lifecycle is None  # 恢复不产生生命周期记录


# ================================================================== #
#  3. 9 步无旁路（broken=False 才合法）
# ================================================================== #

class TestNineStepLifecycle:
    """process_turn 每轮 9 步各恰一次；未知/重复步 → broken=True。"""

    def test_turn_records_all_nine_steps_once(self):
        loop = make_loop()
        loop.process_turn('完整生命周期验证。', '回复。')
        lc = loop.last_turn_lifecycle
        assert lc['broken'] is False
        assert lc['missing'] == []
        assert lc['duplicated'] == []
        assert lc['unknown'] == []
        recorded = lc['execution_order']
        assert len(recorded) == 9
        assert len(set(recorded)) == 9
        assert set(recorded) == set(SPEC_STEP_NAMES)
        # 每步都有观测
        assert set(lc['steps'].keys()) == set(SPEC_STEP_NAMES)

    def test_trace_rejects_unknown_step(self):
        trace = LifecycleTrace()
        with pytest.raises(KeyError):
            trace.record('bogus_step')

    def test_trace_rejects_duplicate_step(self):
        trace = LifecycleTrace()
        trace.record('ingest', x=1)
        with pytest.raises(RuntimeError):
            trace.record('ingest', x=2)  # 同一步重复 = 旁路红线

    def test_partial_trace_marks_broken(self):
        trace = LifecycleTrace()
        trace.record('ingest', x=1)
        trace.record('encode', x=2)
        summary = trace.summary(turn_count=0)
        assert summary['broken'] is True
        assert set(summary['missing']) == (
            set(SPEC_STEP_NAMES) - {'ingest', 'encode'})

    def test_duplicate_step_marks_broken(self):
        trace = LifecycleTrace()
        trace.record('ingest', x=1)
        with pytest.raises(RuntimeError):
            trace.record('ingest', x=1)  # 同一步重复 = 旁路红线（记录时拒绝）
        summary = trace.summary(turn_count=0)
        assert summary['broken'] is True
        assert 'ingest' in summary['duplicated']  # 拒绝记录可观测（G 模式）

    def test_multiple_turns_each_broken_false(self):
        loop = make_loop()
        for i in range(5):
            loop.process_turn(f'第{i}轮：持续运行验证。', '回复。')
            assert loop.last_turn_lifecycle['broken'] is False, i


# ================================================================== #
#  4. 状态汇总（LifecycleTrace.summary 结构）
# ================================================================== #

class TestLifecycleSummary:
    """状态汇总结构 + 观测键。"""

    def test_summary_structure(self):
        loop = make_loop()
        loop.process_turn('状态汇总结构验证。', '回复。')
        s = loop.last_turn_lifecycle
        for key in ('turns', 'broken', 'missing', 'duplicated', 'unknown',
                    'execution_order', 'spec_order', 'steps',
                    'duration_seconds'):
            assert key in s, key
        assert s['spec_order'] == SPEC_STEP_NAMES  # 按规格序呈现
        assert s['duration_seconds'] >= 0.0

    def test_step_observation_keys(self):
        loop = make_loop()
        loop.process_turn('观测键验证。', '回复。')
        steps = loop.last_turn_lifecycle['steps']
        assert 'text_len' in steps['ingest']
        assert 'degraded' in steps['ingest']
        assert 'surprise' in steps['encode']
        assert 'precision_mean' in steps['encode']
        assert 'episodic_query_source' in steps['query']
        assert 'episodic_hits' in steps['retrieve']
        assert 'context_len' in steps['integrate']
        assert 'coherence' in steps['integrate']
        assert 'chain_enabled' in steps['doubt_check']
        assert 'episodic_delta' in steps['commit']
        assert 'j_norm_delta' in steps['state_update']
        assert 'sigma_norm_delta' in steps['state_update']
        assert 'precision_mean_delta' in steps['state_update']
        assert 'turn_count' in steps['emit']

    def test_state_update_observes_expected_increments(self):
        # 状态更新增量落观测（M4 规格 §1.3 emit 同源：变更必须可见）
        loop = make_loop()
        loop.process_turn('增量观测。', '回复。')
        su = loop.last_turn_lifecycle['steps']['state_update']
        assert isinstance(su['j_norm'], float)
        assert isinstance(su['j_norm_delta'], float)
        assert isinstance(su['sigma_norm_delta'], float)
        assert isinstance(su['precision_mean_delta'], float)

    def test_summary_json_serializable(self):
        # 观测全部可序列化标量——绝不泄漏条目对象/张量（记录器零副作用）
        loop = make_loop()
        loop.process_turn('可序列化验证。', '回复。')
        json.dumps(loop.last_turn_lifecycle)


# ================================================================== #
#  5. 记录器零副作用（只读口无记录 / 不泄漏对象）
# ================================================================== #

class TestLifecycleTraceSideEffectFree:
    """生命周期记录器纯观测：只读口不产生记录；观测无对象泄漏。"""

    def test_fresh_loop_no_lifecycle(self):
        loop = make_loop()
        assert loop.lifecycle_trace is None
        assert loop.last_turn_lifecycle is None

    def test_readonly_paths_do_not_touch_lifecycle(self, tmp_path):
        # 做梦/dream 不是 process_turn：不产生 turn 生命周期记录；
        # snapshot_dir 指向临时目录，测试不写脏仓库。
        loop = make_loop(snapshot_dir=str(tmp_path / 'snaps'))
        loop.process_turn('一轮。', '回复。')
        lc_before = loop.last_turn_lifecycle
        trace_before = loop.lifecycle_trace
        loop.recall_episodic_readonly('查', k=3)
        loop.react_readonly('应')
        loop.dream(n_steps=1)  # 做梦不是 process_turn：不产生 turn 生命周期
        assert loop.lifecycle_trace is trace_before
        assert loop.last_turn_lifecycle is lc_before
        assert loop.last_turn_lifecycle['broken'] is False

    def test_obs_values_are_scalars(self):
        loop = make_loop()
        loop.process_turn('标量验证。', '回复。')
        scalar = (int, float, str, bool, type(None))
        for name, obs in loop.last_turn_lifecycle['steps'].items():
            for key, val in obs.items():
                assert isinstance(val, scalar), (name, key, type(val))


# ================================================================== #
#  6. 只读四不变在 M5 重组后仍成立（§5.1 机器防线）
# ================================================================== #

class TestReadonlyFourInvariantsAfterM5:
    """M5 重组后 /recall 与 /react 检索段四不变零增量（回归红线）。"""

    def test_recall_readonly_four_invariants_after_turns(self):
        loop = make_loop()
        loop.process_turn('写入记忆一：苹果是水果。', '回复一。')
        loop.process_turn('写入记忆二：香蕉也是水果。', '回复二。')
        before = loop._readonly_state_snapshot()
        results = loop.recall_episodic_readonly('水果有哪些？', k=3)
        after = loop._readonly_state_snapshot()
        assert before == after  # turn / episodic 条目集 / J / σ 零增量
        assert isinstance(results, list)

    def test_react_readonly_four_invariants_after_turns(self):
        loop = make_loop()
        loop.process_turn('写入记忆。', '回复。')
        before = loop._readonly_state_snapshot()
        loop.react_readonly('实时反应输入。')
        after = loop._readonly_state_snapshot()
        assert before == after

    def test_recall_guard_still_enforced(self):
        # 四不变守卫强制开启：/recall 执行前后由守卫断言，违反即抛
        loop = make_loop()
        loop.process_turn('守卫验证。', '回复。')
        guard = loop._recall_guard('test_loop_m5')
        with guard:
            snap = guard.snapshot()
            loop.recall_episodic_readonly('查询', k=3)
            guard.assert_unchanged(snap)


# ================================================================== #
#  7. get_status lifecycle 观测块（§4.2 独立追加语义）
# ================================================================== #

class TestStatusLifecycleBlock:
    """get_status 新增 lifecycle 块（纯增量，C-01 必需字段不变）。"""

    def test_status_lifecycle_fresh(self):
        loop = make_loop()
        st = loop.get_status()
        assert st['lifecycle']['broken'] is None
        assert 'note' in st['lifecycle']
        assert st['lifecycle']['turns'] == 0

    def test_status_lifecycle_after_turn(self):
        loop = make_loop()
        loop.process_turn('状态观测。', '回复。')
        st = loop.get_status()
        assert st['lifecycle'] == loop.last_turn_lifecycle
        assert st['lifecycle']['broken'] is False

    def test_c01_required_keys_unchanged(self):
        # C-01 契约必需键（turn_count/entropy_ratio/purpose_coherence）仍存在
        loop = make_loop()
        loop.process_turn('契约验证。', '回复。')
        st = loop.get_status()
        for key in ('turn_count', 'entropy_ratio', 'purpose_coherence',
                    'precision_mean', 'last_surprise'):
            assert key in st, key
        assert st['turn_count'] == 1


# ================================================================== #
#  8. M1-M4 集成收口：写侧统一入口真实落库 + 怀疑状态机衔接
# ================================================================== #

class TestM1M4IntegrationConsolidation:
    """M5 是 M1-M4 的集成收口：commit 步走写侧统一入口落库；
    doubt_check 步观测怀疑状态机计数（M3 衔接）。"""

    def test_commit_step_writes_episodic_via_store_entry(self):
        loop = make_loop()
        loop.process_turn('这是一条会被写入情景记忆的测试内容。', '系统回复。')
        lc = loop.last_turn_lifecycle
        assert lc['broken'] is False
        # 写侧统一入口（store_episodic）真实落库：条目数 +1（source=external）
        assert lc['steps']['commit']['episodic_delta'] >= 1
        assert lc['steps']['commit']['entry_added'] is True
        assert loop.memory.episodic_size() >= 1
        last = list(loop.memory.iter_episodic())[-1]
        assert getattr(last, 'source', 'external') == 'external'

    def test_commit_obs_sys_event_skipped(self):
        # [doubt] 系统事件不入库（P0 污染处置）：commit 步登记 skip
        loop = make_loop()
        loop.process_turn('[doubt] gap: 这是一个系统怀疑事件', '')
        lc = loop.last_turn_lifecycle
        assert lc['broken'] is False
        assert lc['steps']['commit']['sys_event_skipped'] is True
        assert lc['steps']['commit']['entry_added'] is False

    def test_doubt_check_observes_state_machine(self):
        # 怀疑状态机衔接（M3-1）：doubt_check 步观测 injection 计数
        loop = make_loop()
        loop.process_turn('怀疑状态机衔接验证。', '回复。')
        dc = loop.last_turn_lifecycle['steps']['doubt_check']
        assert 'chain_enabled' in dc
        assert 'injection_checks' in dc
        assert 'injection_suspect_marked' in dc
        assert isinstance(dc['chain_enabled'], bool)

    def test_allostatic_observation_visible(self):
        # M4 衔接：state_update 步的 allostatic_events 观测键存在
        # （开关默认关 → 0；开关开时 >0 由 test_allostatic_j.py 覆盖）
        loop = make_loop()
        loop.process_turn('allostatic 观测验证。', '回复。')
        assert 'allostatic_events' in loop.last_turn_lifecycle[
            'steps']['state_update']

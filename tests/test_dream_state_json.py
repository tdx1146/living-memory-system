# -*- coding: utf-8 -*-
"""dream_state.json 观测测试（提取层 v1.4 S1-9，M8 定案）

覆盖：
  1. 做梦后 LMS 仓 runtime/dream_state.json 生成
  2. 字段：value_replay 观测（replay_set_ids/scores/reinforced_ids/
     wear_stats/merged_count）+ 容量观测（capacity.usage/soft_limit/
     full_events/hard_drops）+ 熔断降级观测（degraded.events）
  3. fail-open：result 缺 value_replay 字段不崩溃
"""

import json
import os
import time

import pytest
import torch

from runtime.dream_scheduler import DreamScheduler
from runtime.loop import LivingMemoryLoop
from runtime.config import default_config


def make_loop(tmp_path) -> LivingMemoryLoop:
    config = default_config()
    config['num_nodes'] = 32
    config['input_dim'] = 16
    config['num_infer_steps'] = 5
    config['consolidation_interval'] = 3
    config['seed'] = 42
    config['snapshot_dir'] = str(tmp_path)
    loop = LivingMemoryLoop(config)
    # 填充记忆使做梦可跑（buffer + episodic）
    from core.types import Activation
    torch.manual_seed(7)
    for i in range(3):
        state = torch.randn(32) * 0.5
        loop.memory.update(Activation(state=state, entropy=1.0,
                                      surprise=0.5), surprise=0.5)
    from core.hippocampus.memory import EpisodicEntry
    e = EpisodicEntry(text="测试记忆", semantic_vector=torch.zeros(16),
                      surprise=0.5, turn=1)
    e.info_value = 0.5
    loop.memory._episodic_buffer.append(e)
    return loop


@pytest.fixture
def scheduler(tmp_path, monkeypatch):
    """DreamScheduler（不启动后台线程）。"""
    # 让 runtime/dream_state.json 落在 tmp_path（避免测试写脏仓库）
    monkeypatch.setenv("LMS_DREAM_STATE_PATH",
                       os.path.join(str(tmp_path), "dream_state.json"))
    loops = {}

    def get_loop(sid):
        return loops.get(sid)

    sched = DreamScheduler(get_loop_fn=get_loop, idle_threshold=3600)
    sched._loops = loops
    return sched


class TestDreamStateJson:
    def test_trigger_dream_writes_dream_state(self, scheduler, tmp_path):
        loop = make_loop(tmp_path)
        scheduler._loops['main'] = loop
        scheduler.register_session('main')

        result = scheduler.trigger_dream('main', steps=3, timeout=60)
        assert result.get('status') == 'dreamed'

        state_path = os.path.join(str(tmp_path), 'dream_state.json')
        assert os.path.isfile(state_path), "dream_state.json 未生成"
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        # value_replay 观测字段
        latest = state.get('latest', {})
        assert 'replay_set_ids' in latest
        assert 'scores' in latest
        assert 'reinforced_ids' in latest
        assert 'wear_stats' in latest
        assert 'sampled_via_prob' in latest
        assert 'session_id' in latest and latest['session_id'] == 'main'
        # 重放集含测试记忆（turn=1）
        assert 1 in latest['replay_set_ids']

        # 容量观测
        cap = state.get('capacity', {})
        assert cap.get('usage') == 1
        assert cap.get('soft_limit') == 2000
        assert 'full_events' in cap
        assert 'hard_drops' in cap

        # 熔断降级观测
        deg = state.get('degraded', {})
        assert 'events' in deg
        assert 'last_turn_degraded' in deg

    def test_missing_value_replay_fail_open(self, scheduler, tmp_path,
                                            monkeypatch):
        """result 缺 value_replay（如空缓冲做梦 early return）不崩溃。"""
        loop = make_loop(tmp_path)
        scheduler._loops['main'] = loop
        scheduler.register_session('main')
        # 清空记忆 → dream 返回 no_memories_to_replay（无 value_replay 键）
        loop.memory._buffer.clear()
        loop.memory._episodic_buffer.clear()

        result = scheduler.trigger_dream('main', steps=3, timeout=60)
        assert result.get('status') == 'no_memories_to_replay'

        state_path = os.path.join(str(tmp_path), 'dream_state.json')
        assert os.path.isfile(state_path)
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        assert 'latest' in state
        assert 'capacity' in state
        assert state['latest'].get('replay_set_ids') == []

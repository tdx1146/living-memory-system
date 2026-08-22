"""pytest共享测试夹具

为所有测试提供公共的夹具和路径配置。
"""

import sys
import os
import pytest
import torch

# 确保项目根目录在Python路径中
# tests/ 的父目录就是项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def pytest_configure(config):
    """注册自定义 pytest marker。"""
    config.addinivalue_line(
        "markers", "slow: 反身性涌现长时实验（运行较慢）")
    # 阶段 3（precision 三层动态化，2026-08-14）：测试套件默认关闭自适应
    # 开关——全部既有测试验证开关引入前的行为（LMS_PRECISION_ADAPT=0 即
    # 回滚路径）；自适应路径由 test_precision_adapt.py 显式 monkeypatch
    # 启用。setdefault：CI 显式设 1 时尊重（此时旧行为测试需自行适配）。
    os.environ.setdefault("LMS_PRECISION_ADAPT", "0")


@pytest.fixture(scope="session", autouse=True)
def _self_voice_persist_dir(tmp_path_factory):
    """把 self_voice 持久化目录重定向到临时目录。

    自指回路默认将自述持久化到仓库 data/self_voice/；测试环境统一
    重定向到 pytest 临时目录，避免测试运行写脏仓库（行为零影响：
    显式传入 cfg 路径的测试优先于环境变量）。
    """
    tmp = tmp_path_factory.mktemp("self_voice")
    os.environ["LMS_SELF_VOICE_DIR"] = str(tmp)
    yield tmp


@pytest.fixture(autouse=True)
def _gap_registry_persist_isolated(tmp_path, monkeypatch):
    """[F3] GapRegistry 默认持久化路径按**每个测试**隔离到 tmp_path。

    收尾修复（审计 F3：测试隔离缺陷）：GapRegistry 无参构造直读共享
    data/gap_registry.json（gitignored）——test_mark_resolved_lifecycle 等
    写侧测试把悬案A/悬案B 等测试数据落进共享文件，后续测试（loop 构造的
    GapRegistry 也会读默认路径）被污染 → 4 例 doubt/e3 假失败（clean 状态
    4/4 通过实证）。此夹具把类默认路径重定向为每测试独立的临时文件：
    - 函数级作用域 → 跨测试零残留（写侧测试的落盘不外泄）；
    - monkeypatch 自动还原 → 不改变显式传 persist_path 的测试；
    - 行为零影响：仅隔离路径，读写语义不变。
    """
    from core.doubt.gap_registry import GapRegistry
    monkeypatch.setattr(
        GapRegistry, "_PERSIST_PATH",
        str(tmp_path / "gap_registry.json"))

from core.types import SensoryInput, Activation, PurposeState
from core.sensory.tokenizer import SimpleTokenizer
from core.sensory.embedder import SimpleEmbedder
from core.hippocampus.attractor import AttractorNetwork
from core.hippocampus.purpose import PurposeLayer
from core.hippocampus.memory import MemoryManager


# ============================================================
# 感官层夹具
# ============================================================

@pytest.fixture
def tokenizer():
    """提供简单分词器。"""
    return SimpleTokenizer()


@pytest.fixture
def embedder():
    """提供简单嵌入器（32维）。"""
    return SimpleEmbedder(dim=32)


@pytest.fixture
def small_embedder():
    """提供小维度嵌入器（16维），用于快速测试。"""
    return SimpleEmbedder(dim=16)


# ============================================================
# 核心组件夹具
# ============================================================

@pytest.fixture
def attractor():
    """提供小规模吸引子网络（64节点，32维输入）。"""
    return AttractorNetwork(num_nodes=64, input_dim=32)


@pytest.fixture
def purpose():
    """提供目的层（32维）。"""
    return PurposeLayer(input_dim=32)


@pytest.fixture
def memory():
    """提供记忆管理器（64节点）。"""
    return MemoryManager(num_nodes=64)


# ============================================================
# 数据夹具
# ============================================================

@pytest.fixture
def sample_text():
    """提供测试文本。"""
    return "你好，这是一段测试文本。Hello world."


@pytest.fixture
def sample_sensory_input():
    """提供示例感官输入。"""
    return SensoryInput(
        vector=torch.randn(32),
        metadata={'timestamp': 1234567890, 'source': 'test'}
    )


@pytest.fixture
def sample_activation():
    """提供示例激活态。"""
    state = torch.randn(64) * 0.5
    return Activation(state=state, entropy=1.5, surprise=0.8)


@pytest.fixture
def zero_activation():
    """提供零激活态。"""
    return Activation(
        state=torch.zeros(64),
        entropy=0.0,
        surprise=0.0
    )


@pytest.fixture
def sample_landscape():
    """提供示例吸引子景观字典。"""
    return {
        'J': torch.randn(64, 64) * 0.01,
        'bias': torch.zeros(64),
        'sigma': torch.zeros(64),
        'num_nodes': 64,
        'input_dim': 32,
    }


@pytest.fixture
def sample_purpose_state():
    """提供示例目的层状态字典。"""
    return {
        'precision': torch.ones(32),
        'history': [torch.ones(32), torch.ones(32) * 1.5],
        'coherence': 0.85,
    }


# ============================================================
# 循环配置夹具
# ============================================================

@pytest.fixture
def loop_config():
    """提供循环配置（小规模，用于快速测试）。"""
    return {
        'num_nodes': 64,
        'input_dim': 32,
        'decoder_mode': 'text',
        'consolidation_interval': 3,
        'num_infer_steps': 5,
        'learning_rate': 0.01,
    }

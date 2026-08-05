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

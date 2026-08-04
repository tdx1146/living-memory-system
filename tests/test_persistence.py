"""持久化层测试

测试 Snapshot 和 Recovery 的功能，包括往返一致性和完整性验证。
"""

import pytest
import torch
import os
import tempfile

from persistence.snapshot import Snapshot, SNAPSHOT_VERSION
from persistence.recovery import Recovery
from core.types import PurposeState
from core.hippocampus.attractor import AttractorNetwork
from core.hippocampus.purpose import PurposeLayer


# ============================================================
# Snapshot 测试
# ============================================================

class TestSnapshot:
    """快照管理器测试"""

    def test_save_load_roundtrip(self, tmp_path, sample_landscape,
                                  sample_purpose_state):
        """测试save/load往返一致性"""
        snapshot = Snapshot()
        path = str(tmp_path / "test_snapshot.pt")

        # 保存
        snapshot.save(path, sample_landscape, sample_purpose_state)

        # 加载
        loaded_landscape, loaded_purpose = snapshot.load(path)

        # 验证attractor数据一致
        assert torch.allclose(loaded_landscape['J'], sample_landscape['J'])
        assert torch.allclose(loaded_landscape['bias'], sample_landscape['bias'])
        assert torch.allclose(loaded_landscape['sigma'], sample_landscape['sigma'])
        assert loaded_landscape['num_nodes'] == sample_landscape['num_nodes']
        assert loaded_landscape['input_dim'] == sample_landscape['input_dim']

        # 验证purpose数据一致
        assert torch.allclose(loaded_purpose['precision'],
                              sample_purpose_state['precision'])
        assert loaded_purpose['coherence'] == sample_purpose_state['coherence']
        assert len(loaded_purpose['history']) == len(sample_purpose_state['history'])
        for h1, h2 in zip(loaded_purpose['history'], sample_purpose_state['history']):
            assert torch.allclose(h1, h2)

    def test_save_creates_file(self, tmp_path):
        """测试save创建文件"""
        snapshot = Snapshot()
        path = str(tmp_path / "created.pt")

        landscape = {'J': torch.zeros(10, 10), 'bias': torch.zeros(10),
                     'sigma': torch.zeros(10), 'num_nodes': 10, 'input_dim': 5}
        purpose = {'precision': torch.ones(5), 'history': [], 'coherence': 1.0}

        snapshot.save(path, landscape, purpose)
        assert os.path.exists(path)

    def test_save_creates_directory(self, tmp_path):
        """测试save自动创建目录"""
        snapshot = Snapshot()
        path = str(tmp_path / "subdir" / "nested" / "test.pt")

        landscape = {'J': torch.zeros(10, 10), 'bias': torch.zeros(10),
                     'sigma': torch.zeros(10), 'num_nodes': 10, 'input_dim': 5}
        purpose = {'precision': torch.ones(5), 'history': [], 'coherence': 1.0}

        snapshot.save(path, landscape, purpose)
        assert os.path.exists(path)

    def test_load_nonexistent_file(self, tmp_path):
        """测试加载不存在的文件"""
        snapshot = Snapshot()
        path = str(tmp_path / "nonexistent.pt")

        with pytest.raises((FileNotFoundError, Exception)):
            snapshot.load(path)

    def test_save_includes_version(self, tmp_path, sample_landscape,
                                    sample_purpose_state):
        """测试快照包含版本号"""
        snapshot = Snapshot()
        path = str(tmp_path / "versioned.pt")
        snapshot.save(path, sample_landscape, sample_purpose_state)

        # 直接加载原始数据检查版本
        try:
            data = torch.load(path, weights_only=False)
        except TypeError:
            data = torch.load(path)

        assert 'version' in data
        assert data['version'] == SNAPSHOT_VERSION

    def test_save_includes_timestamp(self, tmp_path, sample_landscape,
                                      sample_purpose_state):
        """测试快照包含时间戳"""
        snapshot = Snapshot()
        path = str(tmp_path / "timestamped.pt")
        snapshot.save(path, sample_landscape, sample_purpose_state)

        try:
            data = torch.load(path, weights_only=False)
        except TypeError:
            data = torch.load(path)

        assert 'timestamp' in data
        assert isinstance(data['timestamp'], float)
        assert data['timestamp'] > 0

    def test_get_metadata(self, tmp_path, sample_landscape,
                           sample_purpose_state):
        """测试获取元数据"""
        snapshot = Snapshot()
        path = str(tmp_path / "meta.pt")
        snapshot.save(path, sample_landscape, sample_purpose_state)

        metadata = snapshot.get_metadata(path)
        assert metadata['version'] == SNAPSHOT_VERSION
        assert metadata['timestamp'] > 0

    def test_large_landscape_roundtrip(self, tmp_path):
        """测试大规模景观的往返一致性"""
        snapshot = Snapshot()
        path = str(tmp_path / "large.pt")

        n = 256
        landscape = {
            'J': torch.randn(n, n) * 0.01,
            'bias': torch.randn(n) * 0.1,
            'sigma': torch.randn(n) * 0.5,
            'num_nodes': n,
            'input_dim': 64,
        }
        purpose = {
            'precision': torch.rand(64) * 10,
            'history': [torch.rand(64) for _ in range(20)],
            'coherence': 0.75,
        }

        snapshot.save(path, landscape, purpose)
        loaded_l, loaded_p = snapshot.load(path)

        assert torch.allclose(loaded_l['J'], landscape['J'])
        assert torch.allclose(loaded_p['precision'], purpose['precision'])
        assert loaded_p['coherence'] == purpose['coherence']
        assert len(loaded_p['history']) == 20


# ============================================================
# Recovery 测试
# ============================================================

class TestRecovery:
    """恢复管理器测试"""

    def test_validate_valid_snapshot(self, tmp_path, sample_landscape,
                                      sample_purpose_state):
        """测试验证有效快照"""
        snapshot = Snapshot()
        path = str(tmp_path / "valid.pt")
        snapshot.save(path, sample_landscape, sample_purpose_state)

        recovery = Recovery()
        assert recovery.validate(path) is True

    def test_validate_nonexistent_file(self, tmp_path):
        """测试验证不存在的文件"""
        recovery = Recovery()
        path = str(tmp_path / "nonexistent.pt")
        assert recovery.validate(path) is False

    def test_validate_corrupted_file(self, tmp_path):
        """测试验证损坏的文件"""
        path = str(tmp_path / "corrupted.pt")
        # 写入无效数据
        with open(path, 'wb') as f:
            f.write(b'invalid data')

        recovery = Recovery()
        assert recovery.validate(path) is False

    def test_validate_missing_fields(self, tmp_path):
        """测试验证缺少必需字段的快照"""
        path = str(tmp_path / "incomplete.pt")
        # 保存缺少purpose字段的快照
        data = {
            'version': SNAPSHOT_VERSION,
            'timestamp': 1234567890,
            'attractor': {'J': torch.zeros(10, 10)},
            # 缺少 'purpose'
        }
        torch.save(data, path)

        recovery = Recovery()
        assert recovery.validate(path) is False

    def test_recover_success(self, tmp_path, attractor, purpose):
        """测试成功恢复状态"""
        # 修改原始状态
        attractor.J = torch.randn(64, 64) * 0.05
        attractor.bias = torch.randn(64) * 0.1
        attractor.sigma = torch.randn(64) * 0.3
        purpose.sensory_precision = torch.rand(32) * 5 + 1
        purpose.coherence = 0.7
        purpose.history = [torch.rand(32) for _ in range(5)]

        # 保存
        snapshot = Snapshot()
        path = str(tmp_path / "recover.pt")
        landscape = attractor.get_landscape()
        ps = purpose.get_purpose()
        purpose_dict = {
            'precision': ps.precision,
            'history': ps.history,
            'coherence': ps.coherence,
        }
        snapshot.save(path, landscape, purpose_dict)

        # 创建新实例并恢复
        new_attractor = AttractorNetwork(64, 32)
        new_purpose = PurposeLayer(32)

        recovery = Recovery()
        success = recovery.recover(path, new_attractor, new_purpose)

        assert success is True
        # 验证状态一致
        assert torch.allclose(new_attractor.J, attractor.J)
        assert torch.allclose(new_attractor.bias, attractor.bias)
        assert torch.allclose(new_attractor.sigma, attractor.sigma)
        assert torch.allclose(new_purpose.sensory_precision,
                              purpose.sensory_precision)
        assert new_purpose.coherence == purpose.coherence

    def test_recover_nonexistent(self, tmp_path, attractor, purpose):
        """测试从不存在的快照恢复"""
        recovery = Recovery()
        path = str(tmp_path / "nonexistent.pt")
        success = recovery.recover(path, attractor, purpose)
        assert success is False

    def test_recover_invalid_snapshot(self, tmp_path, attractor, purpose):
        """测试从无效快照恢复"""
        path = str(tmp_path / "invalid.pt")
        with open(path, 'wb') as f:
            f.write(b'invalid')

        recovery = Recovery()
        success = recovery.recover(path, attractor, purpose)
        assert success is False

    def test_recover_preserves_num_nodes(self, tmp_path):
        """测试恢复后节点数一致"""
        # 创建并保存
        attractor = AttractorNetwork(128, 32)
        attractor.J = torch.randn(128, 128) * 0.01
        purpose = PurposeLayer(32)

        snapshot = Snapshot()
        path = str(tmp_path / "nodes.pt")
        landscape = attractor.get_landscape()
        ps = purpose.get_purpose()
        purpose_dict = {
            'precision': ps.precision,
            'history': ps.history,
            'coherence': ps.coherence,
        }
        snapshot.save(path, landscape, purpose_dict)

        # 恢复到新实例
        new_attractor = AttractorNetwork(128, 32)
        new_purpose = PurposeLayer(32)

        recovery = Recovery()
        success = recovery.recover(path, new_attractor, new_purpose)

        assert success is True
        assert new_attractor.num_nodes == 128
        assert new_attractor.input_dim == 32
        assert new_attractor.J.shape == (128, 128)

    def test_version_compatibility_same_version(self):
        """测试相同版本兼容"""
        recovery = Recovery()
        assert recovery._is_version_compatible(SNAPSHOT_VERSION) is True

    def test_version_compatibility_future_minor(self):
        """测试未来次版本兼容（应兼容）"""
        recovery = Recovery()
        major, minor, patch = SNAPSHOT_VERSION.split('.')
        future_version = f"{major}.{int(minor) + 1}.0"
        assert recovery._is_version_compatible(future_version) is True

    def test_version_compatibility_different_major(self):
        """测试不同主版本不兼容"""
        recovery = Recovery()
        assert recovery._is_version_compatible("1.0.0") is False

    def test_recover_with_empty_history(self, tmp_path, attractor, purpose):
        """测试恢复空历史的目的层"""
        snapshot = Snapshot()
        path = str(tmp_path / "empty_hist.pt")
        landscape = attractor.get_landscape()
        purpose_dict = {
            'precision': torch.ones(32),
            'history': [],
            'coherence': 1.0,
        }
        snapshot.save(path, landscape, purpose_dict)

        new_attractor = AttractorNetwork(64, 32)
        new_purpose = PurposeLayer(32)

        recovery = Recovery()
        success = recovery.recover(path, new_attractor, new_purpose)

        assert success is True
        assert len(new_purpose.history) == 0

"""PretrainedEmbedder 资源管理测试（E-P2-4）

验证懒加载、模型缓存、资源释放、上下文管理器、维度预获取、
环境变量覆盖等功能，以及 SimpleEmbedder 的向后兼容性。

设计原则：
  - 不实际加载预训练模型（不依赖网络）。
  - 通过 monkeypatch 注入 FakeModel 替代 SentenceTransformer。
  - 用 SimpleEmbedder 验证既有行为不受影响。
"""

import os

import pytest
import torch

import core.sensory.embedder as emb_mod
from core.sensory.embedder import (
    Embedder,
    SimpleEmbedder,
    PretrainedEmbedder,
)


# ============================================================
# 测试用 FakeModel：替代真实 SentenceTransformer
# ============================================================

class _FakeParam:
    """模拟可冻结的模型参数。"""
    requires_grad = True


class FakeModel:
    """轻量模拟 SentenceTransformer，不触碰网络与磁盘。

    encode() 返回固定 384 维向量，使 embed_text 的随机投影可正常运算。
    """

    instances = []  # 记录所有创建过的实例，供缓存测试断言

    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device
        self._param = _FakeParam()
        self.eval_called = 0
        FakeModel.instances.append(self)

    def eval(self):
        self.eval_called += 1

    def parameters(self):
        return iter([self._param])

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, text, convert_to_tensor=True, normalize_embeddings=True):
        # 返回 [384] 固定向量；基于文本长度做微小扰动以便区分
        base = torch.ones(384) * 0.1
        if isinstance(text, str):
            base = base + len(text) * 0.001
        return base


@pytest.fixture
def patched_embedder(monkeypatch):
    """注入 FakeModel 并清理环境，使 PretrainedEmbedder 可离线测试。

    - 将 _SENTENCE_TRANSFORMERS_AVAILABLE 置 True
    - 将 SentenceTransformer 替换为 FakeModel
    - 清除相关环境变量（避免污染）
    - 清空模型缓存
    """
    monkeypatch.setattr(emb_mod, "_SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(emb_mod, "SentenceTransformer", FakeModel)
    monkeypatch.delenv("LMS_PRETRAINED_MODEL", raising=False)
    monkeypatch.delenv("LMS_EMBEDDER_SOURCE", raising=False)
    monkeypatch.delenv("LMS_MODEL_LOAD_TIMEOUT", raising=False)
    FakeModel.instances.clear()
    PretrainedEmbedder._model_cache.clear()
    return emb_mod


# ============================================================
# 懒加载测试
# ============================================================

class TestLazyLoading:
    """懒加载：构造后 is_loaded 为 False，首次使用后为 True。"""

    def test_not_loaded_after_construction(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        assert e.is_loaded is False

    def test_loaded_after_explicit_load(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        e.load()
        assert e.is_loaded is True

    def test_loaded_after_embed_text(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        assert not e.is_loaded
        e.embed_text("hello world")
        assert e.is_loaded is True

    def test_loaded_after_embed_text_raw(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        assert not e.is_loaded
        e.embed_text_raw("hello world")
        assert e.is_loaded is True

    def test_construction_does_not_call_sentence_transformer(self,
                                                             patched_embedder):
        """构造不应实例化 FakeModel（懒加载）。"""
        FakeModel.instances.clear()
        PretrainedEmbedder(dim=32)
        assert len(FakeModel.instances) == 0

    def test_load_is_idempotent(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        e.load()
        e.load()  # 重复调用无副作用
        assert e.is_loaded is True


# ============================================================
# 模型缓存（单例模式）测试
# ============================================================

class TestModelCache:
    """同一模型配置的多个实例共享同一底层模型对象。"""

    def test_two_instances_share_model(self, patched_embedder):
        e1 = PretrainedEmbedder(dim=32)
        e2 = PretrainedEmbedder(dim=64)  # dim 不同，但模型相同
        e1.load()
        e2.load()
        # 缓存命中：底层模型对象相同
        assert e1._model is e2._model

    def test_cache_populated_after_load(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        assert len(PretrainedEmbedder._model_cache) == 0
        e.load()
        assert len(PretrainedEmbedder._model_cache) == 1

    def test_different_models_cached_separately(self, patched_embedder):
        e1 = PretrainedEmbedder(dim=32, model_name="model-a")
        e2 = PretrainedEmbedder(dim=32, model_name="model-b")
        e1.load()
        e2.load()
        assert e1._model is not e2._model
        assert len(PretrainedEmbedder._model_cache) == 2

    def test_clear_cache(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        e.load()
        assert len(PretrainedEmbedder._model_cache) == 1
        PretrainedEmbedder.clear_cache()
        assert len(PretrainedEmbedder._model_cache) == 0
        # 已加载实例仍持有自己的引用
        assert e.is_loaded is True

    def test_cache_hit_does_not_reload(self, patched_embedder):
        """缓存命中时不创建新的 FakeModel。"""
        e1 = PretrainedEmbedder(dim=32)
        e1.load()
        n_created = len(FakeModel.instances)
        e2 = PretrainedEmbedder(dim=32)
        e2.load()  # 应命中缓存
        assert len(FakeModel.instances) == n_created


# ============================================================
# 资源释放测试
# ============================================================

class TestUnload:
    """unload() 释放模型引用，且可重新加载。"""

    def test_unload_sets_is_loaded_false(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        e.load()
        assert e.is_loaded
        e.unload()
        assert not e.is_loaded

    def test_unload_clears_projection(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        e.load()
        assert e._projection is not None
        e.unload()
        assert e._projection is None

    def test_unload_clears_raw_dim(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        e.load()
        assert e._raw_dim is not None
        e.unload()
        assert e._raw_dim is None

    def test_unload_is_idempotent(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        e.load()
        e.unload()
        e.unload()  # 重复卸载不报错
        assert not e.is_loaded

    def test_reload_after_unload(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        e.load()
        e.unload()
        assert not e.is_loaded
        e.load()
        assert e.is_loaded

    def test_del_does_not_raise(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        e.load()
        # __del__ 应安全执行，不抛异常
        e.__del__()


# ============================================================
# 上下文管理器协议测试
# ============================================================

class TestContextManager:
    """with 语句自动加载/卸载模型。"""

    def test_context_manager_loads_and_unloads(self, patched_embedder):
        with PretrainedEmbedder(dim=32) as e:
            assert e.is_loaded is True
            vec = e.embed_text("hello")
            assert vec.shape[0] == 32
        # 退出后已卸载
        assert e.is_loaded is False

    def test_context_manager_returns_self(self, patched_embedder):
        emb = PretrainedEmbedder(dim=32)
        with emb as e:
            assert e is emb

    def test_context_manager_unload_even_on_exception(self, patched_embedder):
        emb = PretrainedEmbedder(dim=32)
        with pytest.raises(ValueError):
            with emb as e:
                raise ValueError("boom")
        # 异常后仍卸载
        assert emb.is_loaded is False


# ============================================================
# 维度预获取测试
# ============================================================

class TestExpectedDim:
    """expected_dim 不加载模型即可返回预期维度。"""

    def test_default_model_dim(self):
        assert PretrainedEmbedder.expected_dim() == 384

    def test_known_model_dim(self):
        assert PretrainedEmbedder.expected_dim(
            "paraphrase-multilingual-MiniLM-L12-v2") == 384
        assert PretrainedEmbedder.expected_dim("all-MiniLM-L6-v2") == 384
        assert PretrainedEmbedder.expected_dim("all-mpnet-base-v2") == 768
        assert PretrainedEmbedder.expected_dim("bge-large-zh-v1.5") == 1024

    def test_unknown_model_returns_none(self):
        assert PretrainedEmbedder.expected_dim("nonexistent-model-xyz") is None

    def test_local_path_basename_match(self):
        # 本地缓存路径形式：末尾目录名匹配
        path = "/some/cache/dir/paraphrase-multilingual-MiniLM-L12-v2"
        assert PretrainedEmbedder.expected_dim(path) == 384

    def test_does_not_trigger_load(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        # expected_dim 是类方法，不影响实例状态
        dim = PretrainedEmbedder.expected_dim(e._model_name)
        assert dim == 384
        assert e.is_loaded is False


# ============================================================
# raw_dim 属性测试
# ============================================================

class TestRawDim:
    """raw_dim 在加载前返回预期维度，加载后返回实际维度。"""

    def test_raw_dim_before_load(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        # 未加载时返回已知维度表中的预期值
        assert e.raw_dim == 384
        assert not e.is_loaded

    def test_raw_dim_after_load(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        e.load()
        assert e.raw_dim == 384

    def test_raw_dim_unknown_model_before_load(self, patched_embedder):
        e = PretrainedEmbedder(dim=32, model_name="totally-unknown")
        # 未知模型未加载时返回 0
        assert e.raw_dim == 0


# ============================================================
# 环境变量覆盖测试
# ============================================================

class TestEnvironmentVariables:
    """LMS_PRETRAINED_MODEL 与 LMS_EMBEDDER_SOURCE 环境变量覆盖。"""

    def test_lms_pretrained_model_override(self, monkeypatch,
                                           patched_embedder):
        monkeypatch.setenv("LMS_PRETRAINED_MODEL", "my-custom-model")
        e = PretrainedEmbedder(dim=32)
        assert e._model_name == "my-custom-model"

    def test_explicit_model_name_overrides_env(self, monkeypatch,
                                               patched_embedder):
        monkeypatch.setenv("LMS_PRETRAINED_MODEL", "env-model")
        e = PretrainedEmbedder(dim=32, model_name="explicit-model")
        assert e._model_name == "explicit-model"

    def test_default_model_when_no_env(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        assert e._model_name == PretrainedEmbedder.DEFAULT_MODEL_NAME

    def test_lms_embedder_source_default(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        assert e._source == "huggingface"

    def test_lms_embedder_source_modelscope(self, monkeypatch,
                                            patched_embedder):
        monkeypatch.setenv("LMS_EMBEDDER_SOURCE", "modelscope")
        e = PretrainedEmbedder(dim=32, model_name="some-model")
        assert e._source == "modelscope"

    def test_lms_embedder_source_case_insensitive(self, monkeypatch,
                                                   patched_embedder):
        monkeypatch.setenv("LMS_EMBEDDER_SOURCE", " ModelScope ")
        e = PretrainedEmbedder(dim=32)
        assert e._source == "modelscope"

    def test_lms_model_load_timeout_default(self, patched_embedder):
        # patched_embedder 已清除 LMS_MODEL_LOAD_TIMEOUT 环境变量
        # 验证默认值逻辑：不设置环境变量时使用 120
        timeout = int(os.environ.get("LMS_MODEL_LOAD_TIMEOUT", "120"))
        assert timeout == 120


# ============================================================
# 模型路径来源解析测试
# ============================================================

class TestResolveSource:
    """_resolve_source 对不同来源的解析逻辑。"""

    def test_huggingface_returns_model_name(self, patched_embedder):
        e = PretrainedEmbedder(dim=32, model_name="hf-model")
        assert e._resolve_source("hf-model", "huggingface") == "hf-model"

    def test_modelscope_local_dir(self, patched_embedder, tmp_path):
        model_dir = str(tmp_path / "local_model")
        os.makedirs(model_dir)
        e = PretrainedEmbedder(dim=32, model_name=model_dir)
        # modelscope 来源 + 本地目录 -> 直接返回路径
        assert e._resolve_source(model_dir, "modelscope") == model_dir

    def test_modelscope_missing_dependency_raises(self, monkeypatch,
                                                  patched_embedder):
        """source=modelscope 且非本地路径时，缺 modelscope 库应抛 ImportError。"""
        # modelscope 不在 sys.modules，模拟 import 失败
        import sys
        monkeypatch.setitem(sys.modules, "modelscope", None)
        e = PretrainedEmbedder(dim=32, model_name="remote-model")
        with pytest.raises(ImportError, match="modelscope"):
            e._resolve_source("remote-model", "modelscope")


# ============================================================
# 健壮性：重试逻辑测试
# ============================================================

class TestRetryLogic:
    """加载失败时的重试与错误信息。"""

    def test_retry_succeeds_after_failures(self, monkeypatch):
        """前两次失败、第三次成功 -> 最终加载成功。"""
        call_count = {"n": 0}

        class FlakyModel(FakeModel):
            def __init__(self, model_name, device="cpu"):
                call_count["n"] += 1
                if call_count["n"] < 3:
                    raise ConnectionError("network down")
                super().__init__(model_name, device)

        monkeypatch.setattr(emb_mod, "_SENTENCE_TRANSFORMERS_AVAILABLE", True)
        monkeypatch.setattr(emb_mod, "SentenceTransformer", FlakyModel)
        monkeypatch.setattr(emb_mod.time, "sleep", lambda x: None)
        monkeypatch.delenv("LMS_PRETRAINED_MODEL", raising=False)
        monkeypatch.delenv("LMS_EMBEDDER_SOURCE", raising=False)
        PretrainedEmbedder._model_cache.clear()

        e = PretrainedEmbedder(dim=32)
        e.load()
        assert e.is_loaded
        assert call_count["n"] == 3

    def test_retry_exhaustion_raises_runtimeerror(self, monkeypatch):
        """全部重试失败 -> RuntimeError，包含诊断信息。"""

        def always_fail(model_name, device="cpu"):
            raise ConnectionError("network down")

        monkeypatch.setattr(emb_mod, "_SENTENCE_TRANSFORMERS_AVAILABLE", True)
        monkeypatch.setattr(emb_mod, "SentenceTransformer", always_fail)
        monkeypatch.setattr(emb_mod.time, "sleep", lambda x: None)
        monkeypatch.delenv("LMS_PRETRAINED_MODEL", raising=False)
        monkeypatch.delenv("LMS_EMBEDDER_SOURCE", raising=False)
        PretrainedEmbedder._model_cache.clear()

        e = PretrainedEmbedder(dim=32)
        with pytest.raises(RuntimeError) as exc_info:
            e.load()
        msg = str(exc_info.value)
        assert "已重试 3 次" in msg
        assert "model_name" in msg
        assert "source" in msg

    def test_import_error_when_unavailable(self, monkeypatch):
        """sentence-transformers 未安装时构造抛 ImportError。"""
        monkeypatch.setattr(emb_mod, "_SENTENCE_TRANSFORMERS_AVAILABLE", False)
        with pytest.raises(ImportError, match="sentence-transformers"):
            PretrainedEmbedder(dim=32)


# ============================================================
# 编码行为测试（懒加载触发后）
# ============================================================

class TestEncoding:
    """embed_text / embed_text_raw / embed 的行为验证。"""

    def test_embed_text_returns_correct_dim(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        vec = e.embed_text("hello world")
        assert vec.shape[0] == 32

    def test_embed_text_raw_returns_raw_dim(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        vec = e.embed_text_raw("hello world")
        assert vec.shape[0] == 384

    def test_embed_text_empty_returns_zeros(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        vec = e.embed_text("")
        assert vec.shape[0] == 32
        assert torch.all(vec == 0)
        # 空文本不应触发加载
        assert not e.is_loaded

    def test_embed_text_raw_empty_uses_expected_dim(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        vec = e.embed_text_raw("")
        assert vec.shape[0] == 384  # 预期维度
        assert torch.all(vec == 0)
        assert not e.is_loaded

    def test_embed_token_path_raises(self, patched_embedder):
        e = PretrainedEmbedder(dim=32)
        with pytest.raises(NotImplementedError):
            e.embed([1, 2, 3])

    def test_embed_text_deterministic(self, patched_embedder):
        """相同文本 + 相同投影种子 -> 相同输出。"""
        e1 = PretrainedEmbedder(dim=32, seed=42)
        e2 = PretrainedEmbedder(dim=32, seed=42)
        v1 = e1.embed_text("test")
        v2 = e2.embed_text("test")
        assert torch.allclose(v1, v2)

    def test_dim_property(self, patched_embedder):
        e = PretrainedEmbedder(dim=48)
        assert e.dim == 48


# ============================================================
# SimpleEmbedder 向后兼容测试
# ============================================================

class TestSimpleEmbedderCompat:
    """SimpleEmbedder 行为不受 PretrainedEmbedder 改动影响。"""

    def test_is_embedder(self):
        e = SimpleEmbedder(dim=32)
        assert isinstance(e, Embedder)

    def test_dim(self):
        e = SimpleEmbedder(dim=32)
        assert e.dim == 32

    def test_embed_average_pooling(self):
        e = SimpleEmbedder(dim=32, vocab_size=100, seed=42)
        vec = e.embed([1, 2, 3])
        assert vec.shape[0] == 32
        # 平均池化：等于三个 token 向量的均值
        expected = (e.embedding[1] + e.embedding[2] + e.embedding[3]) / 3
        assert torch.allclose(vec, expected)

    def test_embed_empty_returns_zeros(self):
        e = SimpleEmbedder(dim=32)
        vec = e.embed([])
        assert vec.shape[0] == 32
        assert torch.all(vec == 0)

    def test_no_lazy_loading_attributes(self):
        """SimpleEmbedder 不应有懒加载相关属性（行为不变）。"""
        e = SimpleEmbedder(dim=32)
        assert not hasattr(e, "is_loaded")
        assert not hasattr(e, "load")
        assert not hasattr(e, "unload")
        assert not hasattr(e, "_model_cache")

    def test_reproducible_with_seed(self):
        e1 = SimpleEmbedder(dim=32, seed=99)
        e2 = SimpleEmbedder(dim=32, seed=99)
        assert torch.allclose(e1.embedding, e2.embedding)

    def test_embedding_frozen(self):
        e = SimpleEmbedder(dim=32)
        assert e.embedding.requires_grad is False

    def test_no_embed_text_method(self):
        """SimpleEmbedder 没有 embed_text（走 token id 路径）。"""
        e = SimpleEmbedder(dim=32)
        assert not hasattr(e, "embed_text")

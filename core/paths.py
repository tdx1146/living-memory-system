"""跨平台路径管理
=================

集中管理活体记忆系统中所有路径常量，确保 Windows / Linux / macOS 通用。

设计原则:
  - 所有路径使用 ``pathlib.Path``，自动处理分隔符差异
  - 预训练模型路径优先从环境变量读取，回退到跨平台缓存目录探测
  - 快照目录默认在用户主目录下，跨平台一致

环境变量:
    LMS_PRETRAINED_MODEL  -- 预训练模型路径（覆盖自动探测）
    LMS_SNAPSHOT_DIR      -- 快照目录（覆盖默认 ~/.lms/snapshots）
    LMS_DATA_DIR          -- 数据根目录（覆盖默认 ~/.lms）
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ======================================================================
# 项目根目录
# ======================================================================

# core/paths.py -> core/ -> 项目根
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


# ======================================================================
# 预训练模型路径
# ======================================================================

# 默认模型名（HuggingFace / ModelScope 通用标识）
DEFAULT_MODEL_NAME: str = "paraphrase-multilingual-MiniLM-L12-v2"


def _detect_model_cache_dirs() -> list[Path]:
    """探测预训练模型可能的缓存目录（跨平台）。

    返回按优先级排列的候选列表：
      - Windows: %USERPROFILE%\\.cache\\modelscope  /  %HF_HOME%
      - Linux/macOS: ~/.cache/modelscope  /  $HF_HOME  /  ~/.cache/huggingface
    """
    home = Path.home()
    candidates: list[Path] = []

    # 1. HuggingFace 缓存
    hf_home = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
    if hf_home:
        candidates.append(Path(hf_home))

    # 2. ModelScope 缓存
    ms_cache = os.environ.get("MODELSCOPE_CACHE")
    if ms_cache:
        candidates.append(Path(ms_cache))
    else:
        candidates.append(home / ".cache" / "modelscope")

    # 3. HuggingFace 默认缓存（Linux/macOS 风格）
    candidates.append(home / ".cache" / "huggingface")

    return candidates


def resolve_pretrained_model_path() -> str | None:
    """解析预训练模型路径。

    优先级:
      1. 环境变量 ``LMS_PRETRAINED_MODEL``（直接返回，不做存在性检查）
      2. 自动探测各平台缓存目录中的模型快照
      3. 返回模型名（让 sentence-transformers 自动下载）

    返回:
        模型路径字符串，或模型名（触发自动下载），或 None。
    """
    # 1. 环境变量显式指定
    env_path = os.environ.get("LMS_PRETRAINED_MODEL", "").strip()
    if env_path:
        return env_path

    # 2. 自动探测缓存目录
    model_slug = DEFAULT_MODEL_NAME
    # ModelScope 目录格式: models/sentence-transformers--{model}/snapshots/master
    ms_subdir = f"sentence-transformers--{model_slug}"

    for cache_dir in _detect_model_cache_dirs():
        # ModelScope 风格
        ms_path = cache_dir / "models" / ms_subdir / "snapshots" / "master"
        if ms_path.is_dir():
            return str(ms_path)

        # HuggingFace 风格: hub/models--sentence-transformers--{model}/snapshots/*
        hf_path = cache_dir / "hub" / f"models--sentence-transformers--{model_slug}"
        if hf_path.is_dir():
            snapshots = hf_path / "snapshots"
            if snapshots.is_dir():
                # 取第一个快照
                for snap in sorted(snapshots.iterdir()):
                    if snap.is_dir():
                        return str(snap)

        # 直接目录风格（手动放置）
        direct_path = cache_dir / ms_subdir
        if direct_path.is_dir():
            return str(direct_path)

    # 3. 回退到模型名，让 sentence-transformers 自动下载
    return model_slug


# ======================================================================
# 快照与数据目录
# ======================================================================

def get_data_dir() -> Path:
    """获取数据根目录。

    优先级: ``LMS_DATA_DIR`` 环境变量 > ``~/.lms``
    """
    env_dir = os.environ.get("LMS_DATA_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    return Path.home() / ".lms"


def get_snapshot_dir() -> Path:
    """获取快照目录。

    优先级: ``LMS_SNAPSHOT_DIR`` 环境变量 > ``LMS_DATA_DIR/snapshots`` > ``~/.lms/snapshots``
    """
    env_dir = os.environ.get("LMS_SNAPSHOT_DIR", "").strip()
    if env_dir:
        return Path(env_dir)
    return get_data_dir() / "snapshots"


def ensure_dir(path: Path | str) -> Path:
    """确保目录存在，不存在则创建。

    参数:
        path: 目录路径。

    返回:
        Path 对象。
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ======================================================================
# 便捷函数（向后兼容 os.path 风格调用）
# ======================================================================

def project_root() -> str:
    """返回项目根目录字符串。"""
    return str(PROJECT_ROOT)


def snapshot_dir() -> str:
    """返回快照目录字符串。"""
    return str(get_snapshot_dir())

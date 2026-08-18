"""
活体记忆系统 - 投影信息损失评估脚本 (任务 A-P2-2)
==================================================

可独立运行的评估入口：

  - 若 sentence-transformers 可用：用真实预训练模型
    paraphrase-multilingual-MiniLM-L12-v2（384 维）编码内置语料，
    评估真实语义向量经投影后的信息损失。
  - 若不可用：退化为结构化随机向量（低秩信号 + 噪声）作为 fallback，
    仍可完整跑通评估流程并输出报告。

用法::

    python -m evaluation.run_eval
    python evaluation/run_eval.py

可通过命令行参数自定义::

    python evaluation/run_eval.py --n-samples 300 --target-dims 32 64 128
    python evaluation/run_eval.py --no-real-model   # 强制使用随机 fallback
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, List

import numpy as np

# 确保项目根目录在 sys.path 中（便于直接 `python evaluation/run_eval.py` 运行）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from evaluation.projection_eval import (  # noqa: E402
    run_evaluation,
    compare_projection_strategies,
    generate_report,
    _random_vectors,
)

# 预训练模型为可选依赖
try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except ImportError:  # pragma: no cover - 取决于运行环境
    _ST_AVAILABLE = False
    SentenceTransformer = None  # type: ignore[assignment,misc]


# 内置评估语料：覆盖中文日常对话、知识、情感等多类语义，
# 使 384 维向量空间具备真实的低维流形结构（便于检验 PCA 等策略）。
SAMPLE_TEXTS: List[str] = [
    # 问候 / 日常
    "你好", "早上好", "嗨，最近怎么样？", "好久不见", "很高兴见到你",
    "晚安", "明天见", "回聊", "保重", "一路平安",
    # 饮食
    "今天午饭吃什么好？", "推荐一家附近的餐厅", "我喜欢吃辣", "这菜太咸了",
    "咖啡还是茶？", "素食主义", "夜宵", "烘焙面包", "寿司很好吃", "火锅聚会",
    # 工作 / 学习
    "这个项目的截止日期是下周", "会议改到下午三点", "请帮我审阅这份报告",
    "学习机器学习需要数学基础", "代码 review 发现几个 bug", "加班到很晚",
    "演示文稿准备好了吗？", "数据库查询优化", "部署到生产环境", "敏捷开发",
    # 科技 / AI
    "大语言模型正在改变搜索方式", "神经网络的注意力机制", "向量数据库用于语义检索",
    "自由能原理与主动推断", "强化学习的奖励设计", "图神经网络", "联邦学习",
    "模型压缩与量化", "多模态融合", "因果推断",
    # 情感 / 生活
    "今天心情很好", "最近压力有点大", "看了一部感人的电影", "下雨天想睡觉",
    "周末去爬山", "养了一只猫", "读书让人平静", "音乐治愈心灵",
    "旅行计划", "想念家乡的菜",
    # 知识 / 事实
    "地球绕太阳公转一周约 365 天", "水的化学式是 H2O", "中国首都是北京",
    "光合作用把光能转化为化学能", "DNA 是双螺旋结构", "光速约为每秒 30 万公里",
    "二战结束于 1945 年", "珠穆朗玛峰是世界最高峰", "圆周率约 3.14159",
    "人体约有 206 块骨头",
    # 相似语义对（用于检验余弦保持）
    "如何学习 Python 编程？", "Python 入门教程推荐",
    "这个 bug 怎么修复？", "如何解决这个程序错误？",
    "今天天气不错", "今日气候宜人",
    "我想去旅游", "计划一次旅行",
    # 英文
    "Hello, how are you?", "Good morning!",
    "Machine learning is fascinating.", "Deep neural networks.",
    "What's for dinner tonight?", "Recommend a restaurant nearby.",
    "The project deadline is next week.", "Please review my code.",
    "I love spicy food.", "Coffee or tea?",
]


def _make_pretrained_encoder(model_name: str) -> Callable[[List[str]], np.ndarray]:
    """构建基于真实预训练模型的文本编码器（返回 384 维原始向量）。

    使用 embed_text_raw 等价路径（L2 归一化的原始语义向量，未经投影），
    以评估“投影前”的完整 384 维信息。
    """
    model = SentenceTransformer(model_name)
    model.eval()
    raw_dim = int(model.get_sentence_embedding_dimension())

    def encode(texts: List[str]) -> np.ndarray:
        # 过滤空文本，sentence-transformers 批量编码
        cleaned = [t.strip() if t and t.strip() else " " for t in texts]
        with __import__("torch").no_grad():
            vecs = model.encode(
                cleaned,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        arr = vecs.detach().cpu().float().numpy()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # 若模型原始维度与 384 不符，以实际为准（动态适配）
        if arr.shape[1] != raw_dim:
            raw_dim = arr.shape[1]
        return arr

    # 把实际维度挂到函数属性上，供 run_eval 读取
    encode.raw_dim = raw_dim  # type: ignore[attr-defined]
    return encode


def _make_random_encoder(n_total: int, dim: int,
                         seed: int = 42) -> Callable[[List[str]], np.ndarray]:
    """构建结构化随机向量编码器（fallback）。"""
    # 预生成固定向量，保证 texts 长度变化时仍确定
    cache = {"vecs": None}

    def encode(texts: List[str]) -> np.ndarray:
        n = len(texts)
        if cache["vecs"] is None or cache["vecs"].shape[0] < n:
            cache["vecs"] = _random_vectors(max(n, n_total), dim, seed=seed)
        return cache["vecs"][:n]

    encode.raw_dim = dim  # type: ignore[attr-defined]
    return encode


def build_encoder(use_real_model: bool, n_samples: int,
                  seed: int = 42) -> tuple[Callable[[List[str]], np.ndarray], str, int]:
    """构建编码器，返回 (encoder, source_label, raw_dim)。

    优先使用真实预训练模型；不可用或被强制禁用时使用随机 fallback。
    """
    if use_real_model and _ST_AVAILABLE:
        try:
            enc = _make_pretrained_encoder(
                "paraphrase-multilingual-MiniLM-L12-v2"
            )
            return enc, "pretrained(MiniLM-L12-v2, raw)", enc.raw_dim
        except Exception as exc:  # pragma: no cover - 下载/加载失败
            print(f"[警告] 预训练模型加载失败 ({exc})，回退到随机向量。")
    elif use_real_model and not _ST_AVAILABLE:
        print("[提示] sentence-transformers 未安装，回退到随机向量。"
              " 可 `pip install sentence-transformers` 启用真实语义评估。")

    enc = _make_random_encoder(n_samples, 384, seed=seed)
    return enc, "random(low-rank, fallback)", enc.raw_dim


def run_eval(n_samples: int, target_dims: List[int],
             use_real_model: bool, seed: int = 42) -> str:
    """运行完整评估并返回报告字符串。"""
    # 选语料：按需循环采样到 n_samples 条
    texts = [SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)] for i in range(n_samples)]

    encoder, source, raw_dim = build_encoder(use_real_model, n_samples, seed)
    print(f"[信息] 向量来源: {source} (raw_dim={raw_dim})")
    print(f"[信息] 评估目标维度: {target_dims}")
    print(f"[信息] 样本数: {len(texts)}")
    print()

    report = run_evaluation(
        texts,
        target_dims=target_dims,
        original_dim=raw_dim,
        encoder=encoder,
        seed=seed,
        source_label=source,
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="活体记忆系统 投影信息损失评估 (384维->降维)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=120,
        help="评估用样本数（默认 120）",
    )
    parser.add_argument(
        "--target-dims", type=int, nargs="+", default=[32, 64, 128],
        help="目标维度列表（默认 32 64 128）",
    )
    parser.add_argument(
        "--no-real-model", action="store_true",
        help="强制不使用预训练模型（使用随机 fallback）",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子（默认 42）",
    )
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = _parse_args()
    report = run_eval(
        n_samples=args.n_samples,
        target_dims=args.target_dims,
        use_real_model=not args.no_real_model,
        seed=args.seed,
    )
    print(report)


if __name__ == "__main__":
    main()

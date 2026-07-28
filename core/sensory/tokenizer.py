"""
活体记忆系统 - 感官层：文本分词
================================

将对话文本转换为 token id 序列。
提供抽象基类 Tokenizer 定义接口，以及默认实现 SimpleTokenizer。

SimpleTokenizer 策略：
  - 英文/数字：按空格与标点分割为词
  - 中文：按单字分词
  - 自动构建词表，支持动态扩展

参考：架构文档 第五节《接口定义》5.2 感官层接口
"""

import re
from abc import ABC, abstractmethod


class Tokenizer(ABC):
    """分词器抽象基类。

    定义文本到 token id 的统一接口。
    未来可替换为 BPE / SentencePiece / 预训练 SLM 的分词器。
    """

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """文本 -> token id 列表。

        参数:
            text: 待分词的文本字符串。

        返回:
            token id 列表，每个 id 对应词表中的一个 token。
        """
        ...

    @property
    def vocab_size(self) -> int:
        """当前词表大小。"""
        return len(getattr(self, "_vocab", {}))


class SimpleTokenizer(Tokenizer):
    """简单分词器：基于空格/标点分割，支持中文按字分词。

    分词规则:
      1. 连续的中文字符（U+4E00-U+9FFF）逐字拆分。
      2. 连续的字母/数字作为一个 token。
      3. 标点与空白作为分隔符，不产出 token。
      4. 首次遇到的 token 自动分配新 id。

    适用于测试与原型验证，无需外部依赖。
    """

    # 匹配中文 CJK 基本区汉字
    _CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")

    def __init__(self) -> None:
        """初始化空词表。"""
        self._vocab: dict[str, int] = {}
        self._next_id: int = 0

    def _split(self, text: str) -> list[str]:
        """将文本切分为 token 字符串列表。

        中文字符逐字拆分；英文/数字连续序列作为整体。
        """
        tokens: list[str] = []
        current: list[str] = []

        for char in text:
            if self._CJK_PATTERN.match(char):
                # 遇到中文：先Flush当前累积的英文/数字 token
                if current:
                    tokens.append("".join(current))
                    current = []
                tokens.append(char)
            elif char.isalnum():
                current.append(char)
            else:
                # 标点/空白：分隔符
                if current:
                    tokens.append("".join(current))
                    current = []
        if current:
            tokens.append("".join(current))
        return tokens

    def tokenize(self, text: str) -> list[int]:
        """文本 -> token id 列表。

        参数:
            text: 待分词的文本，可中英混合。

        返回:
            token id 列表。未知 token 自动加入词表。
        """
        tokens = self._split(text)
        ids: list[int] = []
        for tok in tokens:
            if tok not in self._vocab:
                self._vocab[tok] = self._next_id
                self._next_id += 1
            ids.append(self._vocab[tok])
        return ids

    @property
    def vocab_size(self) -> int:
        """当前词表大小。"""
        return len(self._vocab)

    def get_vocab(self) -> dict[str, int]:
        """返回当前词表（用于持久化）。

        N2 修复：支持词表跨会话持久化，确保两个独立 tokenizer 实例
        对相同文本产生相同的 token IDs。

        返回:
            词表字典的副本，键为 token 字符串，值为 token id。
        """
        return dict(self._vocab)

    def set_vocab(self, vocab: dict[str, int]) -> None:
        """从持久化数据恢复词表。

        N2 修复：跨会话恢复词表，保证感官输入可复现。

        参数:
            vocab: 由 get_vocab() 返回的词表字典。
        """
        self._vocab = dict(vocab)
        self._next_id = max(vocab.values()) + 1 if vocab else 0

"""桥接层

连接核心层与外部LLM，包含：
- Encoder: LLM输出文本 -> 海马体感官信号
- Decoder: 海马体激活态 -> LLM可理解的context
- LLMBridge: 主LLM API调用封装
"""

from bridge.encoder import Encoder
from bridge.decoder import Decoder
from bridge.llm_bridge import LLMBridge

__all__ = ["Encoder", "Decoder", "LLMBridge"]

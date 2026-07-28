"""主LLM API封装

封装主LLM的API调用，提供带记忆context的查询接口。
默认使用OpenAI兼容API格式，可配置base_url、api_key、model。
接口设计便于替换为其他LLM（如本地模型）。

遵循架构文档 5.4 节的接口定义。
包含重试、超时、错误处理。
"""

import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMBridge:
    """主LLM API封装。

    使用OpenAI兼容API格式，可配置为任意兼容的LLM服务。
    包含重试、超时和错误处理机制。

    属性:
        base_url: API基础URL
        api_key: API密钥
        model: 模型名称
        max_tokens: 最大生成token数
        temperature: 采样温度
        timeout: 请求超时时间（秒）
        max_retries: 最大重试次数
        retry_delay: 重试延迟基数（秒，指数退避）
        system_prompt: 系统提示词
    """

    def __init__(self, api_config: dict):
        """初始化LLM桥接器。

        参数:
            api_config: API配置字典，包含以下可选键:
                - base_url: API基础URL（默认OpenAI）
                - api_key: API密钥
                - model: 模型名称
                - max_tokens: 最大生成token数（默认1000）
                - temperature: 采样温度（默认0.7）
                - timeout: 请求超时秒数（默认30）
                - max_retries: 最大重试次数（默认3）
                - retry_delay: 重试延迟基数秒（默认1.0）
                - system_prompt: 系统提示词
        """
        self.base_url = api_config.get('base_url', 'https://api.openai.com/v1')
        self.api_key = api_config.get('api_key', '')
        self.model = api_config.get('model', 'gpt-3.5-turbo')
        self.max_tokens = api_config.get('max_tokens', 1000)
        self.temperature = api_config.get('temperature', 0.7)
        self.timeout = api_config.get('timeout', 30)
        self.max_retries = api_config.get('max_retries', 3)
        self.retry_delay = api_config.get('retry_delay', 1.0)
        self.system_prompt = api_config.get(
            'system_prompt',
            '你是一个有记忆能力的AI助手。以下是你海马体的记忆context，'
            '请参考它来回答用户的问题。'
        )
        # 缓存client实例
        self._client = None

    def _get_client(self):
        """获取或创建OpenAI客户端实例（惰性初始化）。"""
        if self._client is None:
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "需要安装openai库: pip install openai"
                )
            self._client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        return self._client

    def query(self, user_input: str, memory_context: str) -> str:
        """带记忆context查询主LLM。

        将记忆context注入系统提示，然后发送用户输入给LLM。
        包含重试机制和错误处理。

        参数:
            user_input: 用户输入文本
            memory_context: 记忆context文本（由Decoder生成）

        返回:
            LLM的响应文本

        异常:
            RuntimeError: 当所有重试都失败时抛出
        """
        messages = [
            {
                "role": "system",
                "content": f"{self.system_prompt}\n\n记忆context:\n{memory_context}"
            },
            {
                "role": "user",
                "content": user_input
            },
        ]

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                client = self._get_client()
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                return response.choices[0].message.content

            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM API调用失败 "
                    f"(尝试 {attempt + 1}/{self.max_retries}): {e}"
                )
                if attempt < self.max_retries - 1:
                    # 指数退避
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)

        raise RuntimeError(
            f"LLM API调用失败，已重试{self.max_retries}次。最后错误: {last_error}"
        )

    def set_client(self, client) -> None:
        """设置自定义客户端（用于测试或使用非OpenAI SDK）。

        参数:
            client: 客户端对象，需实现chat.completions.create接口
        """
        self._client = client

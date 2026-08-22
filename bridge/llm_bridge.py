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

    def _should_retry(self, exc: Exception) -> bool:
        """判断异常是否值得重试。

        区分可重试与不可重试异常，避免对必然失败的请求（如 401 鉴权失败）
        进行无意义的重试与退避，浪费资源并阻塞调用方。

        判定规则：
            - openai.APIStatusError 且 status_code >= 500 → 可重试（服务端瞬时故障）
            - openai.APIConnectionError / APITimeoutError / RateLimitError → 可重试
              （网络抖动、超时、限流均为瞬时问题）
            - 4xx 客户端错误（如 401/403/404）→ 不可重试，立即失败
            - 非 openai SDK 异常 → 保守重试（兼容自定义 client / 测试 mock）
            - openai SDK 未安装 → 保守重试

        参数:
            exc: 捕获到的异常对象

        返回:
            True 表示可重试，False 表示应立即失败
        """
        try:
            import openai
        except ImportError:
            # openai SDK 未安装，无法精细判定，保守重试
            return True

        # 连接 / 超时 / 限流：瞬时问题，可重试
        # 注意：APITimeoutError 是 APIConnectionError 的子类，RateLimitError
        # 是 APIStatusError 的子类（status_code=429），需在这些 4xx 规则之前判定
        if isinstance(exc, (openai.APIConnectionError,
                            openai.APITimeoutError,
                            openai.RateLimitError)):
            return True

        # HTTP 状态错误：按状态码判定
        if isinstance(exc, openai.APIStatusError):
            status_code = getattr(exc, 'status_code', None)
            if status_code is not None and status_code >= 500:
                # 5xx 服务端错误：可重试
                return True
            # 4xx 客户端错误（含 400/401/403/404 等）：重试必然失败，不重试
            return False

        # 非 openai SDK 异常（如自定义 client 抛出的异常、测试 mock 异常）：
        # 保守重试，保持向后兼容
        return True

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
            RuntimeError: 当所有重试都失败、或遇到不可重试异常（如 4xx
                鉴权失败）时抛出。不可重试异常会立即抛出（不退避），
                其原始异常通过 raise ... from 保留在异常链中。
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
                # [B22] content=None 归一化为空串：LLM 返回 content=None 会被
                # 误判为调用失败（server.py 侧 len(None) 抛 TypeError → 被包装
                # 成 degraded 错误）；归一化后 /chat 的 degraded 分支只对真异常触发
                content = response.choices[0].message.content
                return (content or "")

            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM API调用失败 "
                    f"(尝试 {attempt + 1}/{self.max_retries}): {e}"
                )
                # 区分可重试与不可重试异常：
                # 4xx 客户端错误（如 401 鉴权失败）重试必然失败，立即抛出不退避
                if not self._should_retry(e):
                    raise RuntimeError(
                        f"LLM API调用失败（不可重试异常，已跳过重试）: {e}"
                    ) from e
                if attempt < self.max_retries - 1:
                    # 指数退避
                    delay = self.retry_delay * (2 ** attempt)
                    time.sleep(delay)

        raise RuntimeError(
            f"LLM API调用失败，已重试{self.max_retries}次。最后错误: {last_error}"
        )

    def query_simple(self, prompt: str, max_tokens: int = 100,
                     timeout: float = 5.0) -> str:
        """轻量 LLM 查询（不注入 memory_context，用于自述蒸馏等内部用途）。

        与 query() 的区别：
        - 不注入记忆 context（避免循环依赖）
        - 不更新记忆系统（纯查询）
        - 更短的超时和 token 限制
        - 无重试（失败时返回空字符串，由调用方处理降级）

        参数:
            prompt: 查询文本（直接作为 user 消息发送，不注入系统提示）。
            max_tokens: 最大生成 token 数（默认 100）。
            timeout: 请求超时秒数（默认 5.0）。覆盖 client 初始化时的超时。

        返回:
            LLM 的响应文本。失败时返回空字符串（由调用方处理降级）。
        """
        messages = [
            {
                "role": "user",
                "content": prompt,
            },
        ]

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature,
                timeout=timeout,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning(f"query_simple 调用失败: {e}")
            return ""

    def set_client(self, client) -> None:
        """设置自定义客户端（用于测试或使用非OpenAI SDK）。

        参数:
            client: 客户端对象，需实现chat.completions.create接口
        """
        self._client = client

    def close(self) -> None:
        """释放 OpenAI client（httpx 连接池）。

        关闭底层 httpx 客户端，释放连接池资源，避免句柄/连接泄漏。
        对未初始化或自定义 client 安全调用（幂等）：若 client 不提供
        close() 方法则仅置空引用。
        """
        if self._client is not None:
            try:
                close_method = getattr(self._client, 'close', None)
                if callable(close_method):
                    close_method()
            except Exception as e:
                logger.warning(f"关闭 LLM client 时发生异常: {e}")
            finally:
                self._client = None

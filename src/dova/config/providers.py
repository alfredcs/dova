"""
LLM Provider Configuration and Router.

Supports multiple LLM providers with automatic fallback and task-specific routing.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

import structlog

logger = structlog.get_logger(__name__)


class TaskType(Enum):
    """Types of LLM tasks with different model requirements."""

    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    CODE_GENERATION = "code_generation"
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"


class RoutingStrategy(Enum):
    """Provider selection strategies."""

    PRIORITY = "priority"
    COST = "cost"
    LATENCY = "latency"
    ROUND_ROBIN = "round_robin"
    FIXED = "fixed"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""

    model_id: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""

    name: str
    enabled: bool = True
    priority: int = 10
    models: dict[TaskType, ModelConfig] = field(default_factory=dict)
    rate_limit_rpm: int = 60
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


@dataclass
class LLMRequest:
    """Request to an LLM provider."""

    task_type: TaskType
    messages: list[dict[str, str]]
    system_prompt: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    user_id: str | None = None


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        self.name = config.name

    @abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate a completion."""
        pass

    @abstractmethod
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream a completion."""
        pass

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for texts."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy."""
        pass

    def get_model_config(self, task_type: TaskType) -> ModelConfig:
        """Get model configuration for a task type."""
        if task_type not in self.config.models:
            raise ValueError(f"No model configured for task type: {task_type}")
        return self.config.models[task_type]


class BedrockProvider(LLMProvider):
    """AWS Bedrock LLM provider."""

    def __init__(self, config: ProviderConfig, region: str = "us-east-1"):
        super().__init__(config)
        self.region = region
        self._client = None

    @property
    def client(self) -> Any:
        """Lazy initialization of Bedrock client."""
        if self._client is None:
            import boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using Bedrock."""
        import json
        import time

        model_config = self.get_model_config(request.task_type)

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens or model_config.max_tokens,
            "temperature": request.temperature or model_config.temperature,
            "messages": request.messages,
        }

        # System prompt must be a top-level parameter, not a message
        if request.system_prompt:
            body["system"] = request.system_prompt

        start_time = time.time()
        response = self.client.invoke_model(
            modelId=model_config.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        latency_ms = (time.time() - start_time) * 1000

        result = json.loads(response["body"].read())
        content = result["content"][0]["text"]

        return LLMResponse(
            content=content,
            provider=self.name,
            model=model_config.model_id,
            input_tokens=result.get("usage", {}).get("input_tokens", 0),
            output_tokens=result.get("usage", {}).get("output_tokens", 0),
            latency_ms=latency_ms,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream completion using Bedrock."""
        import json

        model_config = self.get_model_config(request.task_type)

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens or model_config.max_tokens,
            "temperature": request.temperature or model_config.temperature,
            "messages": request.messages,
        }

        # System prompt must be a top-level parameter, not a message
        if request.system_prompt:
            body["system"] = request.system_prompt

        response = self.client.invoke_model_with_response_stream(
            modelId=model_config.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )

        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                yield chunk["delta"].get("text", "")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using Bedrock Titan."""
        import json

        model_config = self.get_model_config(TaskType.EMBEDDING)
        embeddings = []

        for text in texts:
            body = {"inputText": text}
            response = self.client.invoke_model(
                modelId=model_config.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(response["body"].read())
            embeddings.append(result["embedding"])

        return embeddings

    async def health_check(self) -> bool:
        """Check Bedrock connectivity."""
        try:
            self.client.list_foundation_models()
            return True
        except Exception as e:
            logger.warning("bedrock_health_check_failed", error=str(e))
            return False


class AnthropicProvider(LLMProvider):
    """Anthropic direct API provider."""

    def __init__(self, config: ProviderConfig, api_key: str):
        super().__init__(config)
        self.api_key = api_key
        self._client = None

    @property
    def client(self) -> Any:
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using Anthropic API."""
        import time

        model_config = self.get_model_config(request.task_type)

        start_time = time.time()
        response = await self.client.messages.create(
            model=model_config.model_id,
            max_tokens=request.max_tokens or model_config.max_tokens,
            temperature=request.temperature or model_config.temperature,
            system=request.system_prompt or "",
            messages=request.messages,
        )
        latency_ms = (time.time() - start_time) * 1000

        return LLMResponse(
            content=response.content[0].text,
            provider=self.name,
            model=model_config.model_id,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            latency_ms=latency_ms,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream completion using Anthropic API."""
        model_config = self.get_model_config(request.task_type)

        async with self.client.messages.stream(
            model=model_config.model_id,
            max_tokens=request.max_tokens or model_config.max_tokens,
            temperature=request.temperature or model_config.temperature,
            system=request.system_prompt or "",
            messages=request.messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Anthropic doesn't support embeddings - raise error."""
        raise NotImplementedError("Anthropic does not provide embedding models")

    async def health_check(self) -> bool:
        """Check Anthropic API connectivity."""
        try:
            # Make a minimal request
            await self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            logger.warning("anthropic_health_check_failed", error=str(e))
            return False


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""

    def __init__(self, config: ProviderConfig, api_key: str):
        super().__init__(config)
        self.api_key = api_key
        self._client = None

    @property
    def client(self) -> Any:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using OpenAI API."""
        import time

        model_config = self.get_model_config(request.task_type)

        messages = request.messages.copy()
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})

        start_time = time.time()
        response = await self.client.chat.completions.create(
            model=model_config.model_id,
            max_tokens=request.max_tokens or model_config.max_tokens,
            temperature=request.temperature or model_config.temperature,
            messages=messages,
        )
        latency_ms = (time.time() - start_time) * 1000

        return LLMResponse(
            content=response.choices[0].message.content or "",
            provider=self.name,
            model=model_config.model_id,
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
            latency_ms=latency_ms,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream completion using OpenAI API."""
        model_config = self.get_model_config(request.task_type)

        messages = request.messages.copy()
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})

        stream = await self.client.chat.completions.create(
            model=model_config.model_id,
            max_tokens=request.max_tokens or model_config.max_tokens,
            temperature=request.temperature or model_config.temperature,
            messages=messages,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using OpenAI."""
        model_config = self.get_model_config(TaskType.EMBEDDING)
        response = await self.client.embeddings.create(
            model=model_config.model_id,
            input=texts,
        )
        return [item.embedding for item in response.data]

    async def health_check(self) -> bool:
        """Check OpenAI API connectivity."""
        try:
            await self.client.models.list()
            return True
        except Exception as e:
            logger.warning("openai_health_check_failed", error=str(e))
            return False


class LLMRouter:
    """Routes LLM requests to appropriate providers with fallback."""

    def __init__(
        self,
        providers: dict[str, LLMProvider],
        default_strategy: RoutingStrategy = RoutingStrategy.PRIORITY,
    ):
        self.providers = providers
        self.default_strategy = default_strategy
        self._circuit_breakers: dict[str, bool] = {}

    def _get_sorted_providers(
        self, strategy: RoutingStrategy
    ) -> list[tuple[str, LLMProvider]]:
        """Get providers sorted by strategy."""
        enabled = [
            (name, p) for name, p in self.providers.items() if p.config.enabled
        ]

        if strategy == RoutingStrategy.PRIORITY:
            return sorted(enabled, key=lambda x: x[1].config.priority)
        elif strategy == RoutingStrategy.COST:
            return sorted(
                enabled,
                key=lambda x: x[1].config.cost_per_1k_input
                + x[1].config.cost_per_1k_output,
            )
        return enabled

    async def complete(
        self,
        request: LLMRequest,
        strategy: RoutingStrategy | None = None,
        preferred_provider: str | None = None,
    ) -> LLMResponse:
        """Route completion request to providers with fallback."""
        strategy = strategy or self.default_strategy

        # Try preferred provider first if specified
        if preferred_provider and preferred_provider in self.providers:
            provider = self.providers[preferred_provider]
            try:
                return await provider.complete(request)
            except Exception as e:
                logger.warning(
                    "preferred_provider_failed",
                    provider=preferred_provider,
                    error=str(e),
                )

        # Try providers in order
        sorted_providers = self._get_sorted_providers(strategy)
        last_error: Exception | None = None

        for name, provider in sorted_providers:
            if name == preferred_provider:
                continue  # Already tried
            try:
                return await provider.complete(request)
            except Exception as e:
                logger.warning("provider_failed", provider=name, error=str(e))
                last_error = e
                continue

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

    async def embed(
        self, texts: list[str], preferred_provider: str | None = None
    ) -> list[list[float]]:
        """Route embedding request to providers."""
        # Embeddings require consistency - use fixed provider
        providers_with_embedding = [
            (name, p)
            for name, p in self.providers.items()
            if TaskType.EMBEDDING in p.config.models and p.config.enabled
        ]

        if preferred_provider and preferred_provider in self.providers:
            provider = self.providers[preferred_provider]
            if TaskType.EMBEDDING in provider.config.models:
                return await provider.embed(texts)

        if not providers_with_embedding:
            raise RuntimeError("No providers configured for embeddings")

        return await providers_with_embedding[0][1].embed(texts)


def create_llm_router_from_settings() -> LLMRouter:
    """
    Create an LLMRouter with providers configured from settings.

    Returns:
        LLMRouter with configured providers
    """
    import os

    from dotenv import load_dotenv

    from dova.config.settings import get_settings

    # Load .env file to ensure env vars are available
    load_dotenv()

    settings = get_settings()
    providers: dict[str, LLMProvider] = {}

    # Configure Bedrock provider if AWS credentials are available
    if settings.llm.default_provider == "bedrock" or os.environ.get("AWS_ACCESS_KEY_ID"):
        # Check both BEDROCK_MODEL_ID and AWS_BEDROCK_MODEL_ID for flexibility
        bedrock_model_id = (
            os.environ.get("BEDROCK_MODEL_ID")
            or os.environ.get("AWS_BEDROCK_MODEL_ID")
            or settings.aws.bedrock_model_id
        )
        logger.info("bedrock_model_selected", model_id=bedrock_model_id)
        bedrock_config = ProviderConfig(
            name="bedrock",
            enabled=True,
            priority=1,
            models={
                TaskType.REASONING: ModelConfig(
                    model_id=bedrock_model_id,
                    max_tokens=4096,
                    temperature=0.7,
                ),
                TaskType.SUMMARIZATION: ModelConfig(
                    model_id=bedrock_model_id,
                    max_tokens=2048,
                    temperature=0.3,
                ),
                TaskType.CODE_GENERATION: ModelConfig(
                    model_id=bedrock_model_id,
                    max_tokens=8192,
                    temperature=0.2,
                ),
                TaskType.CLASSIFICATION: ModelConfig(
                    model_id=bedrock_model_id,
                    max_tokens=10240,
                    temperature=0.0,
                ),
            },
        )
        try:
            providers["bedrock"] = BedrockProvider(
                bedrock_config, region=settings.aws.region
            )
            logger.info("bedrock_provider_configured", region=settings.aws.region)
        except Exception as e:
            logger.warning("bedrock_provider_init_failed", error=str(e))

    # Configure Anthropic provider if API key is available and package is installed
    anthropic_key = settings.llm.anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
    anthropic_available = False
    try:
        import anthropic  # noqa: F401
        anthropic_available = True
    except ImportError:
        pass

    if anthropic_key and not anthropic_key.startswith("${") and anthropic_available:
        anthropic_config = ProviderConfig(
            name="anthropic",
            enabled=True,
            priority=2,
            models={
                TaskType.REASONING: ModelConfig(
                    model_id="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    temperature=0.7,
                ),
                TaskType.SUMMARIZATION: ModelConfig(
                    model_id="claude-sonnet-4-20250514",
                    max_tokens=2048,
                    temperature=0.3,
                ),
                TaskType.CODE_GENERATION: ModelConfig(
                    model_id="claude-sonnet-4-20250514",
                    max_tokens=8192,
                    temperature=0.2,
                ),
                TaskType.CLASSIFICATION: ModelConfig(
                    model_id="claude-haiku-3-5-20241022",
                    max_tokens=10240,
                    temperature=0.0,
                ),
            },
        )
        try:
            providers["anthropic"] = AnthropicProvider(anthropic_config, api_key=anthropic_key)
            logger.info("anthropic_provider_configured")
        except Exception as e:
            logger.warning("anthropic_provider_init_failed", error=str(e))

    # Configure OpenAI provider if API key is available and package is installed
    openai_key = settings.llm.openai_api_key or os.environ.get("OPENAI_API_KEY")
    openai_available = False
    try:
        import openai  # noqa: F401
        openai_available = True
    except ImportError:
        pass

    if openai_key and not openai_key.startswith("${") and openai_available:
        openai_config = ProviderConfig(
            name="openai",
            enabled=True,
            priority=3,
            models={
                TaskType.REASONING: ModelConfig(
                    model_id="gpt-4o",
                    max_tokens=4096,
                    temperature=0.7,
                ),
                TaskType.SUMMARIZATION: ModelConfig(
                    model_id="gpt-4o-mini",
                    max_tokens=2048,
                    temperature=0.3,
                ),
                TaskType.CODE_GENERATION: ModelConfig(
                    model_id="gpt-4o",
                    max_tokens=8192,
                    temperature=0.2,
                ),
                TaskType.CLASSIFICATION: ModelConfig(
                    model_id="gpt-4o-mini",
                    max_tokens=10240,
                    temperature=0.0,
                ),
                TaskType.EMBEDDING: ModelConfig(
                    model_id="text-embedding-3-small",
                    max_tokens=8192,
                ),
            },
        )
        try:
            providers["openai"] = OpenAIProvider(openai_config, api_key=openai_key)
            logger.info("openai_provider_configured")
        except Exception as e:
            logger.warning("openai_provider_init_failed", error=str(e))

    if not providers:
        logger.error(
            "no_llm_providers_configured",
            hint="Set AWS credentials for Bedrock, ANTHROPIC_API_KEY, or OPENAI_API_KEY",
        )

    return LLMRouter(providers=providers)

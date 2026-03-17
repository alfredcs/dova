"""
LLM Provider Configuration and Router.

Supports multiple LLM providers with automatic fallback and task-specific routing.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
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
    EXTRACTION = "extraction"  # Entity extraction, parsing
    CHAT = "chat"  # General conversation


class ModelTier(Enum):
    """Model capability tiers for intelligent routing."""

    BASIC = "basic"  # Fast, cheap - for simple tasks
    STANDARD = "standard"  # Balanced - for moderate tasks
    ADVANCED = "advanced"  # Powerful - for complex tasks
    REASONING = "reasoning"  # Extended thinking - for deep reasoning


# Map task types to required model tiers
TASK_TIER_MAPPING: dict[TaskType, ModelTier] = {
    TaskType.CLASSIFICATION: ModelTier.BASIC,
    TaskType.EXTRACTION: ModelTier.BASIC,
    TaskType.SUMMARIZATION: ModelTier.STANDARD,
    TaskType.CHAT: ModelTier.STANDARD,
    TaskType.CODE_GENERATION: ModelTier.ADVANCED,
    TaskType.REASONING: ModelTier.ADVANCED,
    TaskType.EMBEDDING: ModelTier.BASIC,
}


# Default models for each provider and tier
# - BASIC: Fast, low-cost tasks (classification, extraction, summarization)
# - STANDARD: General tasks (chat, research)
# - ADVANCED: Complex tasks (coding, deep reasoning)
# - REASONING: Extended thinking with budget tokens
DEFAULT_BEDROCK_MODELS: dict[ModelTier, str] = {
    ModelTier.BASIC: os.environ.get("BEDROCK_MODEL_BASIC", "global.anthropic.claude-haiku-4-5-20251001-v1:0"),
    ModelTier.STANDARD: os.environ.get("BEDROCK_MODEL_STANDARD", "global.anthropic.claude-sonnet-4-6"),
    ModelTier.ADVANCED: os.environ.get("BEDROCK_MODEL_ADVANCED", "global.anthropic.claude-opus-4-6-v1"),
    ModelTier.REASONING: os.environ.get("BEDROCK_MODEL_REASONING", "global.anthropic.claude-opus-4-6-v1"),
}

DEFAULT_ANTHROPIC_MODELS: dict[ModelTier, str] = {
    ModelTier.BASIC: "claude-haiku-4-5-20251022",
    ModelTier.STANDARD: "claude-sonnet-4-20250514",
    ModelTier.ADVANCED: "claude-opus-4-5-20251101",
    ModelTier.REASONING: "claude-opus-4-5-20251101",
}

DEFAULT_OPENAI_MODELS: dict[ModelTier, str] = {
    ModelTier.BASIC: "gpt-5.4-mini",
    ModelTier.STANDARD: "gpt-5.4",
    ModelTier.ADVANCED: "gpt-5.4",
    ModelTier.REASONING: "gpt-5.4",
}

# Max completion tokens per OpenAI model (to avoid invalid_request_error)
OPENAI_MAX_COMPLETION_TOKENS: dict[str, int] = {
    "gpt-5.4": 16384,
    "gpt-5.4-mini": 16384,
}
OPENAI_DEFAULT_MAX_TOKENS = 16384


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
    max_tokens: int = 40960
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
class ThinkingConfig:
    """Configuration for extended thinking/reasoning."""

    enabled: bool = False
    budget_tokens: int = 0

    def to_api_params(self) -> dict:
        """Convert to API parameters for LLM request."""
        if not self.enabled or self.budget_tokens <= 0:
            return {}
        return {
            "thinking": {
                "type": "enabled",
                "budget_tokens": self.budget_tokens,
            }
        }


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
    thinking: ThinkingConfig | None = None


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

    def __init__(self, config: ProviderConfig, region: str = "us-west-2"):
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
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            partial(
                self.client.invoke_model,
                modelId=model_config.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            ),
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
        loop = asyncio.get_event_loop()

        for text in texts:
            body = {"inputText": text}
            response = await loop.run_in_executor(
                None,
                partial(
                    self.client.invoke_model,
                    modelId=model_config.model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                    accept="application/json",
                ),
            )
            result = json.loads(response["body"].read())
            embeddings.append(result["embedding"])

        return embeddings

    async def health_check(self) -> bool:
        """Check Bedrock connectivity."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.client.list_foundation_models)
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

    def _clamp_max_tokens(self, model_id: str, max_tokens: int) -> int:
        """Clamp max_tokens to the model's supported limit."""
        model_limit = OPENAI_MAX_COMPLETION_TOKENS.get(model_id, OPENAI_DEFAULT_MAX_TOKENS)
        return min(max_tokens, model_limit)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using OpenAI API."""
        import time

        model_config = self.get_model_config(request.task_type)

        messages = request.messages.copy()
        if request.system_prompt:
            messages.insert(0, {"role": "system", "content": request.system_prompt})

        max_tokens = self._clamp_max_tokens(
            model_config.model_id,
            request.max_tokens or model_config.max_tokens,
        )

        start_time = time.time()
        response = await self.client.chat.completions.create(
            model=model_config.model_id,
            max_completion_tokens=max_tokens,
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

        max_tokens = self._clamp_max_tokens(
            model_config.model_id,
            request.max_tokens or model_config.max_tokens,
        )

        stream = await self.client.chat.completions.create(
            model=model_config.model_id,
            max_completion_tokens=max_tokens,
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
        errors: list[tuple[str, Exception]] = []

        for name, provider in sorted_providers:
            if name == preferred_provider:
                continue  # Already tried
            try:
                return await provider.complete(request)
            except Exception as e:
                logger.warning("provider_failed", provider=name, error=str(e))
                errors.append((name, e))
                continue

        error_details = "; ".join(f"[{name}] {e}" for name, e in errors)
        raise RuntimeError(f"All providers failed: {error_details}")

    async def stream(
        self,
        request: LLMRequest,
        strategy: RoutingStrategy | None = None,
        preferred_provider: str | None = None,
    ) -> AsyncIterator[str]:
        """Route streaming request to providers with fallback."""
        strategy = strategy or self.default_strategy

        if preferred_provider and preferred_provider in self.providers:
            provider = self.providers[preferred_provider]
            try:
                async for token in provider.stream(request):
                    yield token
                return
            except Exception as e:
                logger.warning(
                    "preferred_provider_stream_failed",
                    provider=preferred_provider,
                    error=str(e),
                )

        sorted_providers = self._get_sorted_providers(strategy)
        errors: list[tuple[str, Exception]] = []

        for name, provider in sorted_providers:
            if name == preferred_provider:
                continue
            try:
                async for token in provider.stream(request):
                    yield token
                return
            except Exception as e:
                logger.warning("provider_stream_failed", provider=name, error=str(e))
                errors.append((name, e))
                continue

        error_details = "; ".join(f"[{name}] {e}" for name, e in errors)
        raise RuntimeError(f"All providers failed: {error_details}")

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
        # Get tiered model configuration from environment or defaults
        bedrock_models = {
            ModelTier.BASIC: os.environ.get("BEDROCK_MODEL_BASIC", DEFAULT_BEDROCK_MODELS[ModelTier.BASIC]),
            ModelTier.STANDARD: os.environ.get("BEDROCK_MODEL_STANDARD", DEFAULT_BEDROCK_MODELS[ModelTier.STANDARD]),
            ModelTier.ADVANCED: os.environ.get("BEDROCK_MODEL_ADVANCED", DEFAULT_BEDROCK_MODELS[ModelTier.ADVANCED]),
            ModelTier.REASONING: os.environ.get("BEDROCK_MODEL_REASONING", DEFAULT_BEDROCK_MODELS[ModelTier.REASONING]),
        }

        # Legacy: Allow single model override for all tiers
        legacy_model = os.environ.get("BEDROCK_MODEL_ID") or os.environ.get("AWS_BEDROCK_MODEL_ID")
        if legacy_model:
            bedrock_models = {tier: legacy_model for tier in ModelTier}

        logger.info(
            "bedrock_models_configured",
            basic=bedrock_models[ModelTier.BASIC],
            standard=bedrock_models[ModelTier.STANDARD],
            advanced=bedrock_models[ModelTier.ADVANCED],
        )

        # Build task-to-model mapping using tier system
        bedrock_task_models = {}
        for task_type in TaskType:
            tier = TASK_TIER_MAPPING.get(task_type, ModelTier.STANDARD)
            model_id = bedrock_models[tier]

            # Configure appropriate parameters based on task type (10x max_tokens)
            if task_type == TaskType.CLASSIFICATION:
                bedrock_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=10240, temperature=0.0)
            elif task_type == TaskType.EXTRACTION:
                bedrock_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=20480, temperature=0.0)
            elif task_type == TaskType.SUMMARIZATION:
                bedrock_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=20480, temperature=0.3)
            elif task_type == TaskType.CODE_GENERATION:
                bedrock_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=81920, temperature=0.2)
            elif task_type == TaskType.REASONING:
                bedrock_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=40960, temperature=0.7)
            elif task_type == TaskType.CHAT:
                bedrock_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=40960, temperature=0.7)
            elif task_type == TaskType.EMBEDDING:
                # Use Amazon Titan for embeddings (Claude doesn't support embeddings)
                embedding_model = os.environ.get("BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0")
                bedrock_task_models[task_type] = ModelConfig(model_id=embedding_model, max_tokens=81920)
            else:
                bedrock_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=40960, temperature=0.7)

        bedrock_config = ProviderConfig(
            name="bedrock",
            enabled=True,
            priority=1,
            models=bedrock_task_models,
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
        # Get tiered model configuration from environment or defaults
        anthropic_models = {
            ModelTier.BASIC: os.environ.get("ANTHROPIC_MODEL_BASIC", DEFAULT_ANTHROPIC_MODELS[ModelTier.BASIC]),
            ModelTier.STANDARD: os.environ.get("ANTHROPIC_MODEL_STANDARD", DEFAULT_ANTHROPIC_MODELS[ModelTier.STANDARD]),
            ModelTier.ADVANCED: os.environ.get("ANTHROPIC_MODEL_ADVANCED", DEFAULT_ANTHROPIC_MODELS[ModelTier.ADVANCED]),
            ModelTier.REASONING: os.environ.get("ANTHROPIC_MODEL_REASONING", DEFAULT_ANTHROPIC_MODELS[ModelTier.REASONING]),
        }

        # Build task-to-model mapping using tier system
        anthropic_task_models = {}
        for task_type in TaskType:
            if task_type == TaskType.EMBEDDING:
                continue  # Anthropic doesn't have embeddings
            tier = TASK_TIER_MAPPING.get(task_type, ModelTier.STANDARD)
            model_id = anthropic_models[tier]

            if task_type == TaskType.CLASSIFICATION:
                anthropic_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=10240, temperature=0.0)
            elif task_type == TaskType.EXTRACTION:
                anthropic_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=20480, temperature=0.0)
            elif task_type == TaskType.SUMMARIZATION:
                anthropic_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=20480, temperature=0.3)
            elif task_type == TaskType.CODE_GENERATION:
                anthropic_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=81920, temperature=0.2)
            elif task_type == TaskType.REASONING:
                anthropic_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=40960, temperature=0.7)
            else:
                anthropic_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=40960, temperature=0.7)

        anthropic_config = ProviderConfig(
            name="anthropic",
            enabled=True,
            priority=2,
            models=anthropic_task_models,
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
        # Get tiered model configuration from environment or defaults
        openai_models = {
            ModelTier.BASIC: os.environ.get("OPENAI_MODEL_BASIC", DEFAULT_OPENAI_MODELS[ModelTier.BASIC]),
            ModelTier.STANDARD: os.environ.get("OPENAI_MODEL_STANDARD", DEFAULT_OPENAI_MODELS[ModelTier.STANDARD]),
            ModelTier.ADVANCED: os.environ.get("OPENAI_MODEL_ADVANCED", DEFAULT_OPENAI_MODELS[ModelTier.ADVANCED]),
            ModelTier.REASONING: os.environ.get("OPENAI_MODEL_REASONING", DEFAULT_OPENAI_MODELS[ModelTier.REASONING]),
        }

        # Build task-to-model mapping using tier system
        openai_task_models = {}
        for task_type in TaskType:
            tier = TASK_TIER_MAPPING.get(task_type, ModelTier.STANDARD)
            model_id = openai_models[tier]

            if task_type == TaskType.CLASSIFICATION:
                openai_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=10240, temperature=0.0)
            elif task_type == TaskType.EXTRACTION:
                openai_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=20480, temperature=0.0)
            elif task_type == TaskType.SUMMARIZATION:
                openai_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=20480, temperature=0.3)
            elif task_type == TaskType.CODE_GENERATION:
                openai_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=81920, temperature=0.2)
            elif task_type == TaskType.REASONING:
                openai_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=40960, temperature=0.7)
            elif task_type == TaskType.EMBEDDING:
                openai_task_models[task_type] = ModelConfig(model_id="text-embedding-3-small", max_tokens=81920)
            else:
                openai_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=40960, temperature=0.7)

        openai_config = ProviderConfig(
            name="openai",
            enabled=True,
            priority=3,
            models=openai_task_models,
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

    # Log provider summary at startup
    if providers:
        sorted_providers = sorted(providers.values(), key=lambda p: p.config.priority)
        order = [f"{p.name} (priority={p.config.priority})" for p in sorted_providers]
        logger.info("llm_provider_order", providers=order)
        for p in sorted_providers:
            models_by_tier: dict[str, str] = {}
            for task_type, model_cfg in p.config.models.items():
                tier = TASK_TIER_MAPPING.get(task_type, ModelTier.STANDARD).value
                if tier not in models_by_tier:
                    models_by_tier[tier] = model_cfg.model_id
            logger.info(f"llm_models_{p.name}", **models_by_tier)

    return LLMRouter(providers=providers)


def get_model_for_task(provider_name: str, task_type: TaskType) -> tuple[str, ModelTier]:
    """
    Get the model ID and tier for a specific task type and provider.

    Args:
        provider_name: Name of the provider (bedrock, anthropic, openai)
        task_type: Type of task to perform

    Returns:
        Tuple of (model_id, model_tier)
    """
    tier = TASK_TIER_MAPPING.get(task_type, ModelTier.STANDARD)

    model_defaults = {
        "bedrock": DEFAULT_BEDROCK_MODELS,
        "anthropic": DEFAULT_ANTHROPIC_MODELS,
        "openai": DEFAULT_OPENAI_MODELS,
    }

    if provider_name in model_defaults:
        model_id = model_defaults[provider_name].get(tier, model_defaults[provider_name][ModelTier.STANDARD])
        return model_id, tier

    return "unknown", tier


def print_model_configuration() -> None:
    """Print the current model configuration for debugging."""
    print("\n=== DOVA Model Configuration ===\n")

    print("Task → Tier Mapping:")
    print("-" * 40)
    for task_type, tier in TASK_TIER_MAPPING.items():
        print(f"  {task_type.value:20s} → {tier.value}")

    print("\n\nDefault Models by Provider and Tier:")
    print("-" * 60)

    providers = [
        ("Bedrock", DEFAULT_BEDROCK_MODELS),
        ("Anthropic", DEFAULT_ANTHROPIC_MODELS),
        ("OpenAI", DEFAULT_OPENAI_MODELS),
    ]

    for provider_name, models in providers:
        print(f"\n{provider_name}:")
        for tier, model_id in models.items():
            print(f"  {tier.value:12s} → {model_id}")

    print("\n\nEnvironment Variables for Custom Configuration:")
    print("-" * 60)
    print("""
  BEDROCK_MODEL_BASIC=<model-id>      # For classification, extraction, summarization
  BEDROCK_MODEL_STANDARD=<model-id>   # For chat, general tasks
  BEDROCK_MODEL_ADVANCED=<model-id>   # For coding, reasoning
  BEDROCK_MODEL_REASONING=<model-id>  # For deep reasoning with extended thinking

  ANTHROPIC_MODEL_BASIC=<model-id>
  ANTHROPIC_MODEL_STANDARD=<model-id>
  ANTHROPIC_MODEL_ADVANCED=<model-id>
  ANTHROPIC_MODEL_REASONING=<model-id>

  OPENAI_MODEL_BASIC=<model-id>
  OPENAI_MODEL_STANDARD=<model-id>
  OPENAI_MODEL_ADVANCED=<model-id>
  OPENAI_MODEL_REASONING=<model-id>
    """)

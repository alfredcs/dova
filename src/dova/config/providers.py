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
from dotenv import load_dotenv

# Load .env before module-level defaults are evaluated so BEDROCK_MODEL_*
# env vars win over the hard-coded fallbacks below.
load_dotenv()

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


# Default models for each provider and tier. All values are read from
# environment variables first (set via .env) and only fall back to a literal
# if the env var is unset — this keeps all model-ID choices in .env so a
# single config file controls every LLM call dova makes.
#
# Tiers:
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

# Anthropic tier runs through AWS Bedrock Mantle (Messages API at /v1, bearer
# auth). Model IDs are env-driven; adjust ANTHROPIC_MODEL_* in .env to whatever
# naming Mantle's Anthropic endpoint expects.
DEFAULT_ANTHROPIC_MODELS: dict[ModelTier, str] = {
    ModelTier.BASIC: os.environ.get("ANTHROPIC_MODEL_BASIC", "global.anthropic.claude-haiku-4-5-20251001-v1:0"),
    ModelTier.STANDARD: os.environ.get("ANTHROPIC_MODEL_STANDARD", "global.anthropic.claude-opus-4-8"),
    ModelTier.ADVANCED: os.environ.get("ANTHROPIC_MODEL_ADVANCED", "global.anthropic.claude-opus-4-8"),
    ModelTier.REASONING: os.environ.get("ANTHROPIC_MODEL_REASONING", "global.anthropic.claude-opus-4-8"),
}

# OpenAI tier runs through AWS Bedrock Mantle (Responses API), which expects
# provider-prefixed model IDs (e.g. "openai.gpt-5.4").
DEFAULT_OPENAI_MODELS: dict[ModelTier, str] = {
    ModelTier.BASIC: os.environ.get("OPENAI_MODEL_BASIC", "openai.gpt-5.4-mini"),
    ModelTier.STANDARD: os.environ.get("OPENAI_MODEL_STANDARD", "openai.gpt-5.4"),
    ModelTier.ADVANCED: os.environ.get("OPENAI_MODEL_ADVANCED", "openai.gpt-5.4"),
    ModelTier.REASONING: os.environ.get("OPENAI_MODEL_REASONING", "openai.gpt-5.4"),
}

# Embedding models — also env-driven.
DEFAULT_BEDROCK_EMBEDDING_MODEL: str = os.environ.get(
    "BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"
)
DEFAULT_OPENAI_EMBEDDING_MODEL: str = os.environ.get(
    "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
)

# Max completion tokens per OpenAI model (to avoid invalid_request_error).
# Defaults are loaded from the env so operators can tune per-model caps
# without touching source. Format: OPENAI_MAX_TOKENS_<model_id_upper>.
OPENAI_DEFAULT_MAX_TOKENS: int = int(os.environ.get("OPENAI_DEFAULT_MAX_TOKENS", "16384"))
OPENAI_MAX_COMPLETION_TOKENS: dict[str, int] = {
    "openai.gpt-5.4": int(os.environ.get("OPENAI_MAX_TOKENS_GPT_5_4", str(OPENAI_DEFAULT_MAX_TOKENS))),
    "openai.gpt-5.4-mini": int(os.environ.get("OPENAI_MAX_TOKENS_GPT_5_4_MINI", str(OPENAI_DEFAULT_MAX_TOKENS))),
}

# Provider priority order (primary, secondary, tertiary). Lower number = higher
# priority. Comma-separated list drives fallback order in LLMRouter when the
# primary provider fails. Accepts any subset of {bedrock, anthropic, openai}.
DEFAULT_PROVIDER_ORDER: list[str] = [
    p.strip() for p in os.environ.get(
        "LLM_PROVIDER_ORDER", "bedrock,anthropic,openai"
    ).split(",") if p.strip()
]


def _provider_priority(name: str) -> int:
    """Return 1-based priority for a provider per ``LLM_PROVIDER_ORDER``.

    Unknown providers get priority 99 so configured ones always win.
    """
    try:
        return DEFAULT_PROVIDER_ORDER.index(name) + 1
    except ValueError:
        return 99


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


def _model_rejects_temperature(model_id: str) -> bool:
    """Anthropic reasoning models (Opus 4.6+) reject the `temperature` field."""
    # Normalize for substring checks like "global.anthropic.claude-opus-4-6-v1".
    normalized = model_id.lower()
    return "opus-4-6" in normalized or "opus-4-7" in normalized or "opus-4-8" in normalized


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

    def _build_body(self, request: LLMRequest, model_config: ModelConfig) -> dict:
        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens or model_config.max_tokens,
            "messages": request.messages,
        }
        if not _model_rejects_temperature(model_config.model_id):
            body["temperature"] = request.temperature or model_config.temperature
        if request.system_prompt:
            body["system"] = request.system_prompt
        return body

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using Bedrock."""
        import json
        import time

        model_config = self.get_model_config(request.task_type)
        body = self._build_body(request, model_config)

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
        body = self._build_body(request, model_config)

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
    """Anthropic provider via AWS Bedrock Mantle (Messages API).

    Uses ``AsyncAnthropic`` pointed at the Mantle gateway (``base_url`` +
    ``bearer_token``). Auth is a Bedrock Mantle bearer token
    (``Authorization: Bearer``) rather than SigV4 or a direct Anthropic API key.
    The SDK appends ``/v1/messages`` to ``base_url``, so ``base_url`` is the
    gateway's ``/anthropic`` endpoint root (e.g. ``https://…api.aws/anthropic``).
    """

    def __init__(self, config: ProviderConfig, base_url: str, bearer_token: str):
        super().__init__(config)
        self.base_url = base_url
        self.bearer_token = bearer_token
        self._client = None

    @property
    def client(self) -> Any:
        """Lazy initialization of the Mantle-backed Anthropic client."""
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(
                base_url=self.base_url,
                auth_token=self.bearer_token,
            )
        return self._client

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using Anthropic API.

        Uses server-side streaming under the hood to avoid the SDK's pre-flight
        ``ValueError`` for long non-streaming requests (triggered when
        ``max_tokens`` implies >10 min processing). The caller still gets a
        single accumulated ``LLMResponse``.
        """
        import time

        model_config = self.get_model_config(request.task_type)
        kwargs: dict[str, Any] = {
            "model": model_config.model_id,
            "max_tokens": request.max_tokens or model_config.max_tokens,
            "system": request.system_prompt or "",
            "messages": request.messages,
        }
        if not _model_rejects_temperature(model_config.model_id):
            kwargs["temperature"] = request.temperature or model_config.temperature

        start_time = time.time()
        async with self.client.messages.stream(**kwargs) as stream:
            final_message = await stream.get_final_message()
        latency_ms = (time.time() - start_time) * 1000

        return LLMResponse(
            content=final_message.content[0].text,
            provider=self.name,
            model=model_config.model_id,
            input_tokens=final_message.usage.input_tokens,
            output_tokens=final_message.usage.output_tokens,
            latency_ms=latency_ms,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream completion using Anthropic API."""
        model_config = self.get_model_config(request.task_type)
        kwargs: dict[str, Any] = {
            "model": model_config.model_id,
            "max_tokens": request.max_tokens or model_config.max_tokens,
            "system": request.system_prompt or "",
            "messages": request.messages,
        }
        if not _model_rejects_temperature(model_config.model_id):
            kwargs["temperature"] = request.temperature or model_config.temperature

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Anthropic doesn't support embeddings - raise error."""
        raise NotImplementedError("Anthropic does not provide embedding models")

    async def health_check(self) -> bool:
        """Check Bedrock (Anthropic) connectivity."""
        try:
            # Make a minimal request against the configured basic-tier model.
            model_id = self.get_model_config(TaskType.CLASSIFICATION).model_id
            await self.client.messages.create(
                model=model_id,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True
        except Exception as e:
            logger.warning("anthropic_health_check_failed", error=str(e))
            return False


class OpenAIProvider(LLMProvider):
    """OpenAI provider via AWS Bedrock Mantle (Responses API).

    Completions are routed through the Mantle gateway (``base_url`` +
    ``bearer_token``) using the Responses API. Embeddings stay on the direct
    OpenAI API (``api_key``), since Mantle is a chat/responses gateway.
    """

    def __init__(
        self,
        config: ProviderConfig,
        api_key: str,
        base_url: str | None = None,
        bearer_token: str | None = None,
    ):
        super().__init__(config)
        self.api_key = api_key  # direct OpenAI key — used for embeddings
        self.base_url = base_url  # Bedrock Mantle gateway — used for completions
        self.bearer_token = bearer_token  # Mantle auth — used for completions
        self._client = None
        self._embed_client = None

    @property
    def client(self) -> Any:
        """Completion client. Routes to Bedrock Mantle when ``base_url`` is set."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                api_key=self.bearer_token or self.api_key,
                base_url=self.base_url,
            )
        return self._client

    @property
    def embed_client(self) -> Any:
        """Embedding client — direct OpenAI, independent of the Mantle gateway."""
        if self._embed_client is None:
            from openai import AsyncOpenAI

            self._embed_client = AsyncOpenAI(api_key=self.api_key)
        return self._embed_client

    def _clamp_max_tokens(self, model_id: str, max_tokens: int) -> int:
        """Clamp max_tokens to the model's supported limit."""
        model_limit = OPENAI_MAX_COMPLETION_TOKENS.get(model_id, OPENAI_DEFAULT_MAX_TOKENS)
        return min(max_tokens, model_limit)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Generate completion using the Bedrock Mantle Responses API."""
        import time

        model_config = self.get_model_config(request.task_type)

        max_tokens = self._clamp_max_tokens(
            model_config.model_id,
            request.max_tokens or model_config.max_tokens,
        )

        # The Responses API takes the chat history as ``input`` and the system
        # prompt as ``instructions``. Temperature is omitted: gpt-5.x reasoning
        # models reject non-default temperature values.
        kwargs: dict[str, Any] = {
            "model": model_config.model_id,
            "input": request.messages,
            "max_output_tokens": max_tokens,
        }
        if request.system_prompt:
            kwargs["instructions"] = request.system_prompt

        start_time = time.time()
        response = await self.client.responses.create(**kwargs)
        latency_ms = (time.time() - start_time) * 1000

        return LLMResponse(
            content=response.output_text or "",
            provider=self.name,
            model=model_config.model_id,
            input_tokens=response.usage.input_tokens if response.usage else 0,
            output_tokens=response.usage.output_tokens if response.usage else 0,
            latency_ms=latency_ms,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """Stream completion using the Bedrock Mantle Responses API.

        Server-side ``error`` and ``response.failed`` events are raised as
        exceptions so ``LLMRouter.stream`` can fall back to the next provider
        instead of ending the stream silently with truncated output. Refusal
        deltas are forwarded as text, since a refusal is a valid model response
        rather than a transport error.
        """
        model_config = self.get_model_config(request.task_type)

        max_tokens = self._clamp_max_tokens(
            model_config.model_id,
            request.max_tokens or model_config.max_tokens,
        )

        kwargs: dict[str, Any] = {
            "model": model_config.model_id,
            "input": request.messages,
            "max_output_tokens": max_tokens,
            "stream": True,
        }
        if request.system_prompt:
            kwargs["instructions"] = request.system_prompt

        stream = await self.client.responses.create(**kwargs)
        async for event in stream:
            event_type = event.type
            if event_type in ("response.output_text.delta", "response.refusal.delta"):
                yield event.delta
            elif event_type == "error":
                logger.warning(
                    "openai_stream_error", code=event.code, message=event.message
                )
                raise RuntimeError(f"OpenAI stream error: {event.message}")
            elif event_type == "response.failed":
                err = getattr(event.response, "error", None)
                detail = err.message if err else "unknown error"
                logger.warning("openai_stream_failed", detail=detail)
                raise RuntimeError(f"OpenAI stream failed: {detail}")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings using the direct OpenAI API."""
        model_config = self.get_model_config(TaskType.EMBEDDING)
        response = await self.embed_client.embeddings.create(
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
                bedrock_task_models[task_type] = ModelConfig(
                    model_id=DEFAULT_BEDROCK_EMBEDDING_MODEL, max_tokens=81920
                )
            else:
                bedrock_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=40960, temperature=0.7)

        bedrock_config = ProviderConfig(
            name="bedrock",
            enabled=True,
            priority=_provider_priority("bedrock"),
            models=bedrock_task_models,
        )
        try:
            providers["bedrock"] = BedrockProvider(
                bedrock_config, region=settings.aws.region
            )
            logger.info("bedrock_provider_configured", region=settings.aws.region)
        except Exception as e:
            logger.warning("bedrock_provider_init_failed", error=str(e))

    # Bedrock Mantle gateway, shared by the Anthropic (Messages API at /anthropic)
    # and OpenAI (Responses API at /openai/v1) fallback tiers. Both authenticate
    # with the Mantle bearer token (Authorization: Bearer) rather than SigV4. The
    # URL is the gateway host root; each provider appends its own path.
    # Normalize to the gateway host root so each tier can append its own path
    # (/anthropic for Anthropic, /openai/v1 for OpenAI). Tolerates BEDROCK_MANTLE_URL
    # being the root or already carrying a /v1 or /openai/v1 suffix.
    mantle_url = (os.environ.get("BEDROCK_MANTLE_URL") or "").rstrip("/")
    for _suffix in ("/openai/v1", "/v1"):
        if mantle_url.endswith(_suffix):
            mantle_url = mantle_url[: -len(_suffix)]
            break
    mantle_url = mantle_url or None
    mantle_token = os.environ.get("BEDROCK_MANTLE_TOKEN") or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")

    # Configure Anthropic provider via the Mantle gateway (Messages API). Enabled
    # whenever the SDK is installed and the Mantle URL + bearer token are set.
    anthropic_available = False
    try:
        import anthropic  # noqa: F401
        anthropic_available = True
    except ImportError:
        pass

    if anthropic_available and mantle_url and mantle_token and "anthropic" in DEFAULT_PROVIDER_ORDER:
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
            priority=_provider_priority("anthropic"),
            models=anthropic_task_models,
        )
        try:
            anthropic_base_url = f"{mantle_url}/anthropic"
            providers["anthropic"] = AnthropicProvider(
                anthropic_config,
                base_url=anthropic_base_url,
                bearer_token=mantle_token,
            )
            logger.info("anthropic_provider_configured", backend="bedrock-mantle", base_url=anthropic_base_url)
        except Exception as e:
            logger.warning("anthropic_provider_init_failed", error=str(e))

    # Configure OpenAI provider via AWS Bedrock Mantle (Responses API).
    # Completions need the Mantle gateway URL + bearer token; the direct OpenAI
    # key is only used for the embedding path.
    openai_key = settings.llm.openai_api_key or os.environ.get("OPENAI_API_KEY")
    openai_available = False
    try:
        import openai  # noqa: F401
        openai_available = True
    except ImportError:
        pass

    if openai_available and mantle_url and mantle_token:
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
                openai_task_models[task_type] = ModelConfig(model_id=DEFAULT_OPENAI_EMBEDDING_MODEL, max_tokens=81920)
            else:
                openai_task_models[task_type] = ModelConfig(model_id=model_id, max_tokens=40960, temperature=0.7)

        openai_config = ProviderConfig(
            name="openai",
            enabled=True,
            priority=_provider_priority("openai"),
            models=openai_task_models,
        )
        try:
            openai_base_url = f"{mantle_url}/openai/v1"
            providers["openai"] = OpenAIProvider(
                openai_config,
                api_key=openai_key or "",
                base_url=openai_base_url,
                bearer_token=mantle_token,
            )
            logger.info("openai_provider_configured", backend="bedrock-mantle", base_url=openai_base_url)
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
                if task_type == TaskType.EMBEDDING:
                    continue  # Embedding uses a dedicated model, not a tier LLM
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

"""
Self-evaluation and error diagnosis service.

Provides response quality assessment and intelligent error recovery strategies.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ErrorType(Enum):
    """Categories of errors for diagnosis."""

    TRANSIENT = "transient"  # Temporary failures (network, rate limits)
    CONFIGURATION = "configuration"  # Config/setup issues
    CAPABILITY = "capability"  # Model/provider can't handle request
    UNKNOWN = "unknown"  # Unclassified errors


class RecoveryAction(Enum):
    """Recommended recovery actions."""

    RETRY_WITH_BACKOFF = "retry_with_backoff"
    ALERT_USER = "alert_user"
    FALLBACK = "fallback"
    LOG_AND_CONTINUE = "log_and_continue"


@dataclass
class EvaluationResult:
    """Result of self-evaluation on a response."""

    confidence: float  # 0.0 to 1.0
    scores: dict[str, float] = field(default_factory=dict)
    should_retry: bool = False
    caveats: list[str] = field(default_factory=list)

    @property
    def is_acceptable(self) -> bool:
        """Check if response meets quality threshold."""
        return self.confidence >= 0.6 and not self.should_retry


@dataclass
class ErrorDiagnosis:
    """Diagnosis of an error with recovery recommendation."""

    error_type: ErrorType
    action: RecoveryAction
    retry_after_seconds: float | None = None
    fallback_model: str | None = None
    message: str = ""


# Error patterns for diagnosis
ERROR_PATTERNS: dict[str, tuple[ErrorType, RecoveryAction]] = {
    "rate limit": (ErrorType.TRANSIENT, RecoveryAction.RETRY_WITH_BACKOFF),
    "throttl": (ErrorType.TRANSIENT, RecoveryAction.RETRY_WITH_BACKOFF),
    "timeout": (ErrorType.TRANSIENT, RecoveryAction.RETRY_WITH_BACKOFF),
    "connection": (ErrorType.TRANSIENT, RecoveryAction.RETRY_WITH_BACKOFF),
    "503": (ErrorType.TRANSIENT, RecoveryAction.RETRY_WITH_BACKOFF),
    "502": (ErrorType.TRANSIENT, RecoveryAction.RETRY_WITH_BACKOFF),
    "api key": (ErrorType.CONFIGURATION, RecoveryAction.ALERT_USER),
    "auth": (ErrorType.CONFIGURATION, RecoveryAction.ALERT_USER),
    "credential": (ErrorType.CONFIGURATION, RecoveryAction.ALERT_USER),
    "permission": (ErrorType.CONFIGURATION, RecoveryAction.ALERT_USER),
    "not found": (ErrorType.CONFIGURATION, RecoveryAction.ALERT_USER),
    "model not": (ErrorType.CAPABILITY, RecoveryAction.FALLBACK),
    "context length": (ErrorType.CAPABILITY, RecoveryAction.FALLBACK),
    "too long": (ErrorType.CAPABILITY, RecoveryAction.FALLBACK),
    "unsupported": (ErrorType.CAPABILITY, RecoveryAction.FALLBACK),
}

# Backoff times based on error type
BACKOFF_SECONDS: dict[ErrorType, float] = {
    ErrorType.TRANSIENT: 2.0,
    ErrorType.CONFIGURATION: 0.0,
    ErrorType.CAPABILITY: 0.0,
    ErrorType.UNKNOWN: 1.0,
}


class SelfEvaluator:
    """
    Evaluates LLM responses and diagnoses errors.

    Provides automated quality assessment and intelligent error recovery.
    """

    def __init__(
        self,
        min_confidence: float = 0.6,
        fallback_models: list[str] | None = None,
    ):
        self.min_confidence = min_confidence
        self.fallback_models = fallback_models or []
        self._logger = logger.bind(service="evaluation")

    async def evaluate(
        self,
        response: str,
        prompt: str,
        expected_format: str | None = None,
    ) -> EvaluationResult:
        """
        Evaluate an LLM response for quality.

        Args:
            response: The LLM response to evaluate
            prompt: Original prompt for context
            expected_format: Expected response format (json, markdown, etc.)

        Returns:
            EvaluationResult with confidence scores and caveats
        """
        scores: dict[str, float] = {}
        caveats: list[str] = []

        # Check response length
        response_len = len(response)
        if response_len < 10:
            scores["length"] = 0.2
            caveats.append("Response too short")
        elif response_len < 50:
            scores["length"] = 0.5
        else:
            scores["length"] = 1.0

        # Check for refusal patterns
        refusal_indicators = [
            "i cannot",
            "i can't",
            "i'm unable",
            "i am unable",
            "as an ai",
        ]
        has_refusal = any(ind in response.lower() for ind in refusal_indicators)
        if has_refusal:
            scores["refusal"] = 0.3
            caveats.append("Response contains refusal language")
        else:
            scores["refusal"] = 1.0

        # Check format compliance if expected
        if expected_format:
            format_score = self._check_format(response, expected_format)
            scores["format"] = format_score
            if format_score < 0.5:
                caveats.append(f"Response doesn't match expected format: {expected_format}")

        # Check relevance (simple keyword overlap)
        relevance = self._estimate_relevance(response, prompt)
        scores["relevance"] = relevance
        if relevance < 0.3:
            caveats.append("Response may not be relevant to prompt")

        # Calculate overall confidence
        confidence = sum(scores.values()) / len(scores) if scores else 0.5

        # Determine if retry is needed
        should_retry = confidence < self.min_confidence and not has_refusal

        result = EvaluationResult(
            confidence=confidence,
            scores=scores,
            should_retry=should_retry,
            caveats=caveats,
        )

        self._logger.debug(
            "evaluation_complete",
            confidence=confidence,
            scores=scores,
            should_retry=should_retry,
        )

        return result

    def diagnose_error(
        self,
        error: Exception | str,
        context: dict[str, Any] | None = None,
    ) -> ErrorDiagnosis:
        """
        Diagnose an error and recommend recovery action.

        Args:
            error: The error to diagnose
            context: Additional context (provider, model, etc.)

        Returns:
            ErrorDiagnosis with recommended recovery action
        """
        error_str = str(error).lower()
        context = context or {}

        # Match error patterns
        error_type = ErrorType.UNKNOWN
        action = RecoveryAction.LOG_AND_CONTINUE

        for pattern, (etype, eaction) in ERROR_PATTERNS.items():
            if pattern in error_str:
                error_type = etype
                action = eaction
                break

        # Calculate backoff time
        retry_after = BACKOFF_SECONDS.get(error_type, 1.0)

        # Suggest fallback model if capability error
        fallback_model = None
        if error_type == ErrorType.CAPABILITY and self.fallback_models:
            current_model = context.get("model", "")
            for model in self.fallback_models:
                if model != current_model:
                    fallback_model = model
                    break

        diagnosis = ErrorDiagnosis(
            error_type=error_type,
            action=action,
            retry_after_seconds=retry_after if action == RecoveryAction.RETRY_WITH_BACKOFF else None,
            fallback_model=fallback_model,
            message=f"Error classified as {error_type.value}: {action.value}",
        )

        self._logger.info(
            "error_diagnosed",
            error_type=error_type.value,
            action=action.value,
            fallback_model=fallback_model,
        )

        return diagnosis

    def _check_format(self, response: str, expected_format: str) -> float:
        """Check if response matches expected format."""
        format_lower = expected_format.lower()

        if format_lower == "json":
            try:
                import json
                json.loads(response)
                return 1.0
            except json.JSONDecodeError:
                # Check if it contains JSON-like structure
                if "{" in response and "}" in response:
                    return 0.5
                return 0.2

        if format_lower == "markdown":
            # Check for markdown indicators
            md_indicators = ["#", "**", "- ", "* ", "```", "|"]
            found = sum(1 for ind in md_indicators if ind in response)
            return min(1.0, found / 3)

        if format_lower == "list":
            # Check for list-like structure
            lines = response.strip().split("\n")
            list_lines = sum(
                1 for line in lines
                if line.strip().startswith(("-", "*", "•")) or
                (len(line) > 0 and line[0].isdigit() and "." in line[:3])
            )
            return min(1.0, list_lines / max(1, len(lines) * 0.3))

        return 1.0  # No format check for unknown formats

    def _estimate_relevance(self, response: str, prompt: str) -> float:
        """Estimate response relevance to prompt using keyword overlap."""
        # Extract significant words from prompt
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "to", "of", "and", "or", "in", "on", "at", "for", "with",
            "it", "this", "that", "what", "how", "why", "when", "where",
        }

        prompt_words = set(
            word.lower().strip(".,!?\"'")
            for word in prompt.split()
            if len(word) > 2 and word.lower() not in stop_words
        )

        if not prompt_words:
            return 0.5

        response_lower = response.lower()
        matches = sum(1 for word in prompt_words if word in response_lower)

        return min(1.0, matches / (len(prompt_words) * 0.3))

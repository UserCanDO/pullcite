"""
Extractor configuration and hooks.

This module defines configuration options and lifecycle hooks
for customizing extraction behavior.

Key types:
- ExtractorConfig: Settings for extraction behavior
- Hooks: Lifecycle callbacks for custom processing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .document import Document
    from .evidence import VerificationResult
    from .fields import VerifierPolicy


@dataclass(frozen=True)
class ExtractorConfig:
    """
    Configuration for Extractor behavior.

    Controls thresholds, retry limits, and verification settings.
    Immutable - use with_* methods to create modified copies.

    Attributes:
        confidence_threshold: Minimum confidence for VERIFIED status.
        low_confidence_threshold: Below this, flag as LOW_CONFIDENCE.
        max_extraction_retries: Retries for extraction on parse errors.
        max_correction_attempts: How many correction loops to run.
        max_tool_rounds: Max search tool calls per verification batch.
        temperature: LLM temperature (0 = deterministic).
        verification_batch_size: Fields per verification call. None = all at once.
        skip_verification: If True, skip verification phase entirely.
        policies: Custom verification policies.
    """

    # Confidence thresholds
    confidence_threshold: float = 0.80
    """Minimum confidence for VERIFIED status."""

    low_confidence_threshold: float = 0.50
    """Below this, flag as LOW_CONFIDENCE."""

    # Retry limits
    max_extraction_retries: int = 2
    """Retries for extraction phase on parse errors."""

    max_correction_attempts: int = 2
    """How many correction loops to run."""

    max_tool_rounds: int = 25
    """Max search tool calls per verification batch."""

    # LLM settings
    temperature: float = 0.0
    """LLM temperature (0 = deterministic)."""

    # Verification settings
    verification_batch_size: int | None = None
    """Fields per verification call. None = all at once."""

    skip_verification: bool = False
    """If True, skip verification phase entirely (for testing)."""

    # Policies
    policies: tuple["VerifierPolicy", ...] = ()
    """Custom verification policies."""

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be 0.0-1.0, got {self.confidence_threshold}"
            )

        if not 0.0 <= self.low_confidence_threshold <= 1.0:
            raise ValueError(
                f"low_confidence_threshold must be 0.0-1.0, got {self.low_confidence_threshold}"
            )

        if self.low_confidence_threshold > self.confidence_threshold:
            raise ValueError(
                f"low_confidence_threshold ({self.low_confidence_threshold}) "
                f"cannot exceed confidence_threshold ({self.confidence_threshold})"
            )

        if self.max_extraction_retries < 0:
            raise ValueError("max_extraction_retries must be >= 0")

        if self.max_correction_attempts < 0:
            raise ValueError("max_correction_attempts must be >= 0")

        if self.max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be >= 1")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be 0.0-2.0, got {self.temperature}")

        if (
            self.verification_batch_size is not None
            and self.verification_batch_size < 1
        ):
            raise ValueError("verification_batch_size must be >= 1 or None")

    def with_policies(self, *policies: "VerifierPolicy") -> "ExtractorConfig":
        """
        Return new config with additional policies.

        Args:
            *policies: Policies to add.

        Returns:
            New ExtractorConfig with merged policies.
        """
        return ExtractorConfig(
            confidence_threshold=self.confidence_threshold,
            low_confidence_threshold=self.low_confidence_threshold,
            max_extraction_retries=self.max_extraction_retries,
            max_correction_attempts=self.max_correction_attempts,
            max_tool_rounds=self.max_tool_rounds,
            temperature=self.temperature,
            verification_batch_size=self.verification_batch_size,
            skip_verification=self.skip_verification,
            policies=self.policies + tuple(policies),
        )

    def with_thresholds(
        self,
        confidence: float | None = None,
        low_confidence: float | None = None,
    ) -> "ExtractorConfig":
        """
        Return new config with updated thresholds.

        Args:
            confidence: New confidence threshold.
            low_confidence: New low confidence threshold.

        Returns:
            New ExtractorConfig with updated thresholds.
        """
        return ExtractorConfig(
            confidence_threshold=(
                confidence if confidence is not None else self.confidence_threshold
            ),
            low_confidence_threshold=(
                low_confidence
                if low_confidence is not None
                else self.low_confidence_threshold
            ),
            max_extraction_retries=self.max_extraction_retries,
            max_correction_attempts=self.max_correction_attempts,
            max_tool_rounds=self.max_tool_rounds,
            temperature=self.temperature,
            verification_batch_size=self.verification_batch_size,
            skip_verification=self.skip_verification,
            policies=self.policies,
        )

    def with_retries(
        self,
        extraction: int | None = None,
        correction: int | None = None,
        tool_rounds: int | None = None,
    ) -> "ExtractorConfig":
        """
        Return new config with updated retry limits.

        Args:
            extraction: Max extraction retries.
            correction: Max correction attempts.
            tool_rounds: Max tool calls per verification.

        Returns:
            New ExtractorConfig with updated limits.
        """
        return ExtractorConfig(
            confidence_threshold=self.confidence_threshold,
            low_confidence_threshold=self.low_confidence_threshold,
            max_extraction_retries=(
                extraction if extraction is not None else self.max_extraction_retries
            ),
            max_correction_attempts=(
                correction if correction is not None else self.max_correction_attempts
            ),
            max_tool_rounds=(
                tool_rounds if tool_rounds is not None else self.max_tool_rounds
            ),
            temperature=self.temperature,
            verification_batch_size=self.verification_batch_size,
            skip_verification=self.skip_verification,
            policies=self.policies,
        )


@dataclass(frozen=True)
class Hooks:
    """
    Lifecycle hooks for custom processing.

    Hooks are called at specific points in the extraction pipeline,
    allowing custom logic without modifying the core flow.

    All hooks receive the current data and can return modified data.
    Return the input unchanged if no modification is needed.
    """

    post_extract: Callable[[dict, "Document"], dict] | None = None
    """
    Called after extraction, before verification.

    Args:
        data: Extracted dict (not yet Pydantic model)
        document: Source document

    Returns:
        Modified data dict

    Use for: Computing derived fields, normalizing structure.
    """

    post_verify: (
        Callable[[dict, list["VerificationResult"], "Document"], dict] | None
    ) = None
    """
    Called after verification, before correction.

    Args:
        data: Extracted dict
        results: Verification results
        document: Source document

    Returns:
        Modified data dict

    Use for: Custom handling of verification failures.
    """

    post_correct: (
        Callable[[dict, list["VerificationResult"], "Document"], dict] | None
    ) = None
    """
    Called after correction, before final result.

    Args:
        data: Corrected dict
        results: Final verification results
        document: Source document

    Returns:
        Modified data dict

    Use for: Final cleanup, pruning empty values.
    """

    on_llm_call: Callable[[str, list[dict], dict | None], None] | None = None
    """
    Called before each LLM call (for logging/debugging).

    Args:
        role: "extractor", "verifier", or "corrector"
        messages: Messages being sent
        tools: Tool definitions (if any)

    Use for: Logging, cost tracking, debugging.
    """

    on_tool_call: Callable[[str, dict, Any], None] | None = None
    """
    Called after each tool execution.

    Args:
        tool_name: Name of tool called
        input: Tool input
        output: Tool output

    Use for: Logging, debugging.
    """

    on_retry: Callable[[str, int, Exception], None] | None = None
    """
    Called when a retry occurs.

    Args:
        phase: "extraction", "verification", or "correction"
        attempt: Current attempt number (1-indexed)
        error: The exception that caused the retry

    Use for: Logging, alerting.
    """

    def with_hook(
        self,
        hook_name: str,
        hook_fn: Callable,
    ) -> "Hooks":
        """
        Return new Hooks with one hook updated.

        Args:
            hook_name: Name of hook to update.
            hook_fn: New hook function.

        Returns:
            New Hooks with updated hook.

        Raises:
            ValueError: If hook_name is invalid.
        """
        valid_hooks = {
            "post_extract",
            "post_verify",
            "post_correct",
            "on_llm_call",
            "on_tool_call",
            "on_retry",
        }
        if hook_name not in valid_hooks:
            raise ValueError(
                f"Invalid hook name: {hook_name}. Must be one of {valid_hooks}"
            )

        return Hooks(
            post_extract=hook_fn if hook_name == "post_extract" else self.post_extract,
            post_verify=hook_fn if hook_name == "post_verify" else self.post_verify,
            post_correct=hook_fn if hook_name == "post_correct" else self.post_correct,
            on_llm_call=hook_fn if hook_name == "on_llm_call" else self.on_llm_call,
            on_tool_call=hook_fn if hook_name == "on_tool_call" else self.on_tool_call,
            on_retry=hook_fn if hook_name == "on_retry" else self.on_retry,
        )


# Default configuration
DEFAULT_CONFIG = ExtractorConfig()

# Strict configuration (higher thresholds, more retries)
STRICT_CONFIG = ExtractorConfig(
    confidence_threshold=0.90,
    low_confidence_threshold=0.70,
    max_extraction_retries=3,
    max_correction_attempts=3,
    max_tool_rounds=50,
)

# Fast configuration (fewer retries, skip verification option)
FAST_CONFIG = ExtractorConfig(
    confidence_threshold=0.70,
    max_extraction_retries=1,
    max_correction_attempts=1,
    max_tool_rounds=10,
)


__all__ = [
    "ExtractorConfig",
    "Hooks",
    "DEFAULT_CONFIG",
    "STRICT_CONFIG",
    "FAST_CONFIG",
]

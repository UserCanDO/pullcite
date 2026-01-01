"""
Tests for config.py - ExtractorConfig and Hooks.
"""

import pytest
from pullcite.core.config import (
    ExtractorConfig,
    Hooks,
    DEFAULT_CONFIG,
    STRICT_CONFIG,
    FAST_CONFIG,
)
from pullcite.core.fields import VerifierPolicy, Parsers


class TestExtractorConfig:
    """Test ExtractorConfig dataclass."""

    def test_defaults(self):
        config = ExtractorConfig()
        assert config.confidence_threshold == 0.80
        assert config.low_confidence_threshold == 0.50
        assert config.max_extraction_retries == 2
        assert config.max_correction_attempts == 2
        assert config.max_tool_rounds == 25
        assert config.temperature == 0.0
        assert config.verification_batch_size is None
        assert config.skip_verification is False
        assert config.policies == ()

    def test_custom_values(self):
        config = ExtractorConfig(
            confidence_threshold=0.90,
            max_extraction_retries=5,
            temperature=0.5,
            skip_verification=True,
        )
        assert config.confidence_threshold == 0.90
        assert config.max_extraction_retries == 5
        assert config.temperature == 0.5
        assert config.skip_verification is True

    def test_confidence_threshold_validation(self):
        with pytest.raises(ValueError) as exc:
            ExtractorConfig(confidence_threshold=1.5)
        assert "confidence_threshold must be 0.0-1.0" in str(exc.value)

        with pytest.raises(ValueError):
            ExtractorConfig(confidence_threshold=-0.1)

    def test_low_confidence_threshold_validation(self):
        with pytest.raises(ValueError) as exc:
            ExtractorConfig(low_confidence_threshold=1.5)
        assert "low_confidence_threshold must be 0.0-1.0" in str(exc.value)

    def test_threshold_ordering_validation(self):
        with pytest.raises(ValueError) as exc:
            ExtractorConfig(
                confidence_threshold=0.50,
                low_confidence_threshold=0.80,
            )
        assert "cannot exceed confidence_threshold" in str(exc.value)

    def test_max_extraction_retries_validation(self):
        with pytest.raises(ValueError):
            ExtractorConfig(max_extraction_retries=-1)

        # Zero is valid (no retries)
        config = ExtractorConfig(max_extraction_retries=0)
        assert config.max_extraction_retries == 0

    def test_max_correction_attempts_validation(self):
        with pytest.raises(ValueError):
            ExtractorConfig(max_correction_attempts=-1)

    def test_max_tool_rounds_validation(self):
        with pytest.raises(ValueError):
            ExtractorConfig(max_tool_rounds=0)

    def test_temperature_validation(self):
        with pytest.raises(ValueError):
            ExtractorConfig(temperature=3.0)

        with pytest.raises(ValueError):
            ExtractorConfig(temperature=-0.1)

        # Edge cases valid
        config = ExtractorConfig(temperature=0.0)
        assert config.temperature == 0.0

        config = ExtractorConfig(temperature=2.0)
        assert config.temperature == 2.0

    def test_verification_batch_size_validation(self):
        with pytest.raises(ValueError):
            ExtractorConfig(verification_batch_size=0)

        # None is valid
        config = ExtractorConfig(verification_batch_size=None)
        assert config.verification_batch_size is None

        # Positive is valid
        config = ExtractorConfig(verification_batch_size=10)
        assert config.verification_batch_size == 10

    def test_with_policies(self):
        config = ExtractorConfig()
        policy = VerifierPolicy(id="test", field_pattern="*.amount")

        new_config = config.with_policies(policy)

        # Original unchanged
        assert config.policies == ()

        # New has policy
        assert len(new_config.policies) == 1
        assert new_config.policies[0].id == "test"

    def test_with_policies_appends(self):
        policy1 = VerifierPolicy(id="p1", field_pattern="*")
        policy2 = VerifierPolicy(id="p2", field_pattern="*.copay")

        config = ExtractorConfig(policies=(policy1,))
        new_config = config.with_policies(policy2)

        assert len(new_config.policies) == 2
        assert new_config.policies[0].id == "p1"
        assert new_config.policies[1].id == "p2"

    def test_with_thresholds(self):
        config = ExtractorConfig()
        new_config = config.with_thresholds(confidence=0.95, low_confidence=0.60)

        assert config.confidence_threshold == 0.80  # Original unchanged
        assert new_config.confidence_threshold == 0.95
        assert new_config.low_confidence_threshold == 0.60

    def test_with_thresholds_partial(self):
        config = ExtractorConfig(confidence_threshold=0.85)
        new_config = config.with_thresholds(low_confidence=0.55)

        assert new_config.confidence_threshold == 0.85  # Preserved
        assert new_config.low_confidence_threshold == 0.55

    def test_with_retries(self):
        config = ExtractorConfig()
        new_config = config.with_retries(extraction=5, correction=3, tool_rounds=50)

        assert config.max_extraction_retries == 2  # Original unchanged
        assert new_config.max_extraction_retries == 5
        assert new_config.max_correction_attempts == 3
        assert new_config.max_tool_rounds == 50

    def test_with_retries_partial(self):
        config = ExtractorConfig(max_extraction_retries=10)
        new_config = config.with_retries(correction=5)

        assert new_config.max_extraction_retries == 10  # Preserved
        assert new_config.max_correction_attempts == 5

    def test_immutability(self):
        config = ExtractorConfig()
        with pytest.raises(AttributeError):
            config.confidence_threshold = 0.99


class TestHooks:
    """Test Hooks dataclass."""

    def test_defaults(self):
        hooks = Hooks()
        assert hooks.post_extract is None
        assert hooks.post_verify is None
        assert hooks.post_correct is None
        assert hooks.on_llm_call is None
        assert hooks.on_tool_call is None
        assert hooks.on_retry is None

    def test_with_callbacks(self):
        def my_extract_hook(data, doc):
            return data

        def my_llm_hook(role, messages, tools):
            pass

        hooks = Hooks(
            post_extract=my_extract_hook,
            on_llm_call=my_llm_hook,
        )

        assert hooks.post_extract is my_extract_hook
        assert hooks.on_llm_call is my_llm_hook

    def test_with_hook(self):
        hooks = Hooks()

        def new_hook(data, doc):
            return data

        new_hooks = hooks.with_hook("post_extract", new_hook)

        assert hooks.post_extract is None  # Original unchanged
        assert new_hooks.post_extract is new_hook

    def test_with_hook_invalid_name(self):
        hooks = Hooks()

        with pytest.raises(ValueError) as exc:
            hooks.with_hook("invalid_hook", lambda: None)
        assert "Invalid hook name" in str(exc.value)

    def test_with_hook_preserves_others(self):
        def hook1(data, doc):
            return data

        def hook2(data, results, doc):
            return data

        hooks = Hooks(post_extract=hook1)
        new_hooks = hooks.with_hook("post_verify", hook2)

        assert new_hooks.post_extract is hook1  # Preserved
        assert new_hooks.post_verify is hook2

    def test_immutability(self):
        hooks = Hooks()
        with pytest.raises(AttributeError):
            hooks.post_extract = lambda d, doc: d


class TestPresetConfigs:
    """Test preset configuration constants."""

    def test_default_config(self):
        assert DEFAULT_CONFIG.confidence_threshold == 0.80
        assert DEFAULT_CONFIG.max_extraction_retries == 2

    def test_strict_config(self):
        assert STRICT_CONFIG.confidence_threshold == 0.90
        assert STRICT_CONFIG.low_confidence_threshold == 0.70
        assert STRICT_CONFIG.max_extraction_retries == 3
        assert STRICT_CONFIG.max_tool_rounds == 50

    def test_fast_config(self):
        assert FAST_CONFIG.confidence_threshold == 0.70
        assert FAST_CONFIG.max_extraction_retries == 1
        assert FAST_CONFIG.max_tool_rounds == 10


class TestConfigChaining:
    """Test chaining config modification methods."""

    def test_chain_methods(self):
        config = (
            ExtractorConfig()
            .with_thresholds(confidence=0.95)
            .with_retries(extraction=5)
            .with_policies(VerifierPolicy(id="p", field_pattern="*"))
        )

        assert config.confidence_threshold == 0.95
        assert config.max_extraction_retries == 5
        assert len(config.policies) == 1

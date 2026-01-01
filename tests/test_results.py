"""
Tests for result.py - ExtractionResult, ExtractionStats, ExtractionFlag.
"""

import pytest
from pullcite.core.result import (
    ExtractionResult,
    ExtractionStats,
    ExtractionFlag,
    ExtractionFlagType,
    ExtractionStatus,
    compute_status,
    compute_confidence,
)
from pullcite.core.evidence import (
    Evidence,
    VerificationResult,
    VerificationStatus,
)


class TestExtractionStatus:
    """Test ExtractionStatus enum."""

    def test_values(self):
        assert ExtractionStatus.VERIFIED.value == "VERIFIED"
        assert ExtractionStatus.PARTIAL.value == "PARTIAL"
        assert ExtractionStatus.LOW_CONFIDENCE.value == "LOW_CONFIDENCE"
        assert ExtractionStatus.FAILED.value == "FAILED"


class TestExtractionFlagType:
    """Test ExtractionFlagType enum."""

    def test_values(self):
        assert ExtractionFlagType.LOW_CONFIDENCE.value == "LOW_CONFIDENCE"
        assert ExtractionFlagType.AMBIGUOUS.value == "AMBIGUOUS"
        assert ExtractionFlagType.NOT_FOUND.value == "NOT_FOUND"
        assert ExtractionFlagType.MISMATCH_CORRECTED.value == "MISMATCH_CORRECTED"


class TestExtractionFlag:
    """Test ExtractionFlag dataclass."""

    def test_basic_creation(self):
        flag = ExtractionFlag(
            type=ExtractionFlagType.LOW_CONFIDENCE,
            message="Confidence below threshold",
            path="deductible.individual",
        )
        assert flag.type == ExtractionFlagType.LOW_CONFIDENCE
        assert flag.message == "Confidence below threshold"
        assert flag.path == "deductible.individual"
        assert flag.details is None

    def test_without_path(self):
        flag = ExtractionFlag(
            type=ExtractionFlagType.SCHEMA_ERROR,
            message="Invalid JSON output",
        )
        assert flag.path is None

    def test_with_details(self):
        flag = ExtractionFlag(
            type=ExtractionFlagType.AMBIGUOUS,
            message="Multiple values found",
            path="copay",
            details={"candidates": [100, 200]},
        )
        assert flag.details == {"candidates": [100, 200]}

    def test_str_with_path(self):
        flag = ExtractionFlag(
            type=ExtractionFlagType.NOT_FOUND,
            message="Value not in document",
            path="missing.field",
        )
        s = str(flag)
        assert "NOT_FOUND" in s
        assert "missing.field" in s
        assert "Value not in document" in s

    def test_str_without_path(self):
        flag = ExtractionFlag(
            type=ExtractionFlagType.TOOL_ERROR,
            message="Search failed",
        )
        s = str(flag)
        assert "TOOL_ERROR" in s
        assert "Search failed" in s

    def test_immutability(self):
        flag = ExtractionFlag(
            type=ExtractionFlagType.LOW_CONFIDENCE,
            message="test",
        )
        with pytest.raises(AttributeError):
            flag.message = "other"


class TestExtractionStats:
    """Test ExtractionStats dataclass."""

    def test_defaults(self):
        stats = ExtractionStats()
        assert stats.total_duration_ms == 0
        assert stats.extraction_llm_calls == 0
        assert stats.fields_verified == 0

    def test_with_values(self):
        stats = ExtractionStats(
            total_duration_ms=5000,
            extraction_duration_ms=2000,
            verification_duration_ms=2500,
            correction_duration_ms=500,
            extraction_input_tokens=1000,
            extraction_output_tokens=500,
            extraction_llm_calls=1,
            verification_input_tokens=2000,
            verification_output_tokens=300,
            verification_llm_calls=1,
            verification_tool_calls=5,
            correction_input_tokens=500,
            correction_output_tokens=100,
            correction_llm_calls=1,
            fields_verified=10,
            fields_passed=8,
            fields_corrected=1,
            fields_failed=1,
        )
        assert stats.total_duration_ms == 5000
        assert stats.verification_tool_calls == 5

    def test_total_input_tokens(self):
        stats = ExtractionStats(
            extraction_input_tokens=100,
            verification_input_tokens=200,
            correction_input_tokens=50,
        )
        assert stats.total_input_tokens == 350

    def test_total_output_tokens(self):
        stats = ExtractionStats(
            extraction_output_tokens=100,
            verification_output_tokens=200,
            correction_output_tokens=50,
        )
        assert stats.total_output_tokens == 350

    def test_total_tokens(self):
        stats = ExtractionStats(
            extraction_input_tokens=100,
            extraction_output_tokens=50,
            verification_input_tokens=200,
            verification_output_tokens=100,
        )
        assert stats.total_tokens == 450

    def test_total_llm_calls(self):
        stats = ExtractionStats(
            extraction_llm_calls=1,
            verification_llm_calls=2,
            correction_llm_calls=1,
        )
        assert stats.total_llm_calls == 4

    def test_verification_pass_rate(self):
        stats = ExtractionStats(
            fields_verified=10,
            fields_passed=8,
        )
        assert stats.verification_pass_rate == 0.8

    def test_verification_pass_rate_zero_verified(self):
        stats = ExtractionStats(fields_verified=0)
        assert stats.verification_pass_rate == 0.0

    def test_to_dict(self):
        stats = ExtractionStats(
            total_duration_ms=1000,
            fields_verified=5,
            fields_passed=4,
        )
        d = stats.to_dict()
        assert d["total_duration_ms"] == 1000
        assert d["fields_verified"] == 5
        assert d["verification_pass_rate"] == 0.8
        assert "total_tokens" in d


class TestExtractionResult:
    """Test ExtractionResult dataclass."""

    def test_basic_creation(self):
        result = ExtractionResult(
            data={"test": "data"},
            status=ExtractionStatus.VERIFIED,
            confidence=0.95,
            document_id="doc123",
        )
        assert result.data == {"test": "data"}
        assert result.status == ExtractionStatus.VERIFIED
        assert result.confidence == 0.95
        assert result.document_id == "doc123"

    def test_confidence_validation(self):
        with pytest.raises(ValueError):
            ExtractionResult(
                data={},
                status=ExtractionStatus.VERIFIED,
                confidence=1.5,
                document_id="doc",
            )

    def test_evidence_lookup(self):
        ev = Evidence(
            value=1500,
            quote="$1,500",
            page=1,
            bbox=None,
            chunk_index=0,
            confidence=0.95,
            verified=True,
        )
        result = ExtractionResult(
            data={},
            status=ExtractionStatus.VERIFIED,
            confidence=0.9,
            document_id="doc",
            evidence_map={"deductible": ev},
        )

        assert result.evidence("deductible") is ev
        assert result.evidence("missing") is None

    def test_is_verified(self):
        ev = Evidence(
            value=100,
            quote="$100",
            page=1,
            bbox=None,
            chunk_index=0,
            confidence=0.9,
            verified=True,
        )
        result = ExtractionResult(
            data={},
            status=ExtractionStatus.VERIFIED,
            confidence=0.9,
            document_id="doc",
            evidence_map={"verified_field": ev},
        )

        assert result.is_verified("verified_field") is True
        assert result.is_verified("missing_field") is False

    def test_is_verified_false_when_not_verified(self):
        ev = Evidence(
            value=100,
            quote=None,
            page=None,
            bbox=None,
            chunk_index=None,
            confidence=0.3,
            verified=False,
        )
        result = ExtractionResult(
            data={},
            status=ExtractionStatus.PARTIAL,
            confidence=0.5,
            document_id="doc",
            evidence_map={"field": ev},
        )

        assert result.is_verified("field") is False

    def test_verified_fields(self):
        vr1 = VerificationResult(
            path="field1",
            status=VerificationStatus.MATCH,
            extracted_value=1,
        )
        vr2 = VerificationResult(
            path="field2",
            status=VerificationStatus.MISMATCH,
            extracted_value=2,
        )
        vr3 = VerificationResult(
            path="field3",
            status=VerificationStatus.MATCH,
            extracted_value=3,
        )

        result = ExtractionResult(
            data={},
            status=ExtractionStatus.PARTIAL,
            confidence=0.8,
            document_id="doc",
            verification_results=(vr1, vr2, vr3),
        )

        assert result.verified_fields == ["field1", "field3"]

    def test_unverified_fields(self):
        vr1 = VerificationResult(
            path="good",
            status=VerificationStatus.MATCH,
            extracted_value=1,
        )
        vr2 = VerificationResult(
            path="bad",
            status=VerificationStatus.NOT_FOUND,
            extracted_value=2,
        )
        vr3 = VerificationResult(
            path="skipped",
            status=VerificationStatus.SKIPPED,
            extracted_value=3,
        )

        result = ExtractionResult(
            data={},
            status=ExtractionStatus.PARTIAL,
            confidence=0.7,
            document_id="doc",
            verification_results=(vr1, vr2, vr3),
        )

        assert result.unverified_fields == ["bad"]

    def test_corrected_fields(self):
        flags = (
            ExtractionFlag(
                type=ExtractionFlagType.MISMATCH_CORRECTED,
                message="Fixed",
                path="field1",
            ),
            ExtractionFlag(
                type=ExtractionFlagType.LOW_CONFIDENCE,
                message="Low conf",
                path="field2",
            ),
            ExtractionFlag(
                type=ExtractionFlagType.MISMATCH_CORRECTED,
                message="Fixed",
                path="field3",
            ),
        )

        result = ExtractionResult(
            data={},
            status=ExtractionStatus.VERIFIED,
            confidence=0.9,
            document_id="doc",
            flags=flags,
        )

        assert result.corrected_fields == ["field1", "field3"]

    def test_is_success(self):
        verified = ExtractionResult(
            data={},
            status=ExtractionStatus.VERIFIED,
            confidence=0.9,
            document_id="doc",
        )
        assert verified.is_success is True

        partial = ExtractionResult(
            data={},
            status=ExtractionStatus.PARTIAL,
            confidence=0.7,
            document_id="doc",
        )
        assert partial.is_success is True

        failed = ExtractionResult(
            data={},
            status=ExtractionStatus.FAILED,
            confidence=0.0,
            document_id="doc",
        )
        assert failed.is_success is False

        low_conf = ExtractionResult(
            data={},
            status=ExtractionStatus.LOW_CONFIDENCE,
            confidence=0.4,
            document_id="doc",
        )
        assert low_conf.is_success is False

    def test_has_warnings(self):
        no_warnings = ExtractionResult(
            data={},
            status=ExtractionStatus.VERIFIED,
            confidence=0.9,
            document_id="doc",
        )
        assert no_warnings.has_warnings is False

        with_warnings = ExtractionResult(
            data={},
            status=ExtractionStatus.VERIFIED,
            confidence=0.9,
            document_id="doc",
            flags=(
                ExtractionFlag(type=ExtractionFlagType.LOW_CONFIDENCE, message="x"),
            ),
        )
        assert with_warnings.has_warnings is True

    def test_get_flags_for_path(self):
        flags = (
            ExtractionFlag(
                type=ExtractionFlagType.LOW_CONFIDENCE, message="a", path="field1"
            ),
            ExtractionFlag(
                type=ExtractionFlagType.NOT_FOUND, message="b", path="field2"
            ),
            ExtractionFlag(
                type=ExtractionFlagType.AMBIGUOUS, message="c", path="field1"
            ),
        )

        result = ExtractionResult(
            data={},
            status=ExtractionStatus.PARTIAL,
            confidence=0.7,
            document_id="doc",
            flags=flags,
        )

        field1_flags = result.get_flags_for_path("field1")
        assert len(field1_flags) == 2

        field2_flags = result.get_flags_for_path("field2")
        assert len(field2_flags) == 1

        missing_flags = result.get_flags_for_path("field3")
        assert len(missing_flags) == 0

    def test_get_flags_by_type(self):
        flags = (
            ExtractionFlag(type=ExtractionFlagType.LOW_CONFIDENCE, message="a"),
            ExtractionFlag(type=ExtractionFlagType.LOW_CONFIDENCE, message="b"),
            ExtractionFlag(type=ExtractionFlagType.NOT_FOUND, message="c"),
        )

        result = ExtractionResult(
            data={},
            status=ExtractionStatus.PARTIAL,
            confidence=0.7,
            document_id="doc",
            flags=flags,
        )

        low_conf = result.get_flags_by_type(ExtractionFlagType.LOW_CONFIDENCE)
        assert len(low_conf) == 2

        not_found = result.get_flags_by_type(ExtractionFlagType.NOT_FOUND)
        assert len(not_found) == 1

    def test_to_dict(self):
        result = ExtractionResult(
            data={"key": "value"},
            status=ExtractionStatus.VERIFIED,
            confidence=0.95,
            document_id="doc123",
        )

        d = result.to_dict()
        assert d["status"] == "VERIFIED"
        assert d["confidence"] == 0.95
        assert d["document_id"] == "doc123"
        assert "stats" in d


class TestComputeStatus:
    """Test compute_status function."""

    def test_no_verification_high_confidence(self):
        status = compute_status(
            confidence=0.9,
            confidence_threshold=0.8,
            verification_results=[],
            critical_required=[],
        )
        assert status == ExtractionStatus.VERIFIED

    def test_no_verification_low_confidence(self):
        status = compute_status(
            confidence=0.5,
            confidence_threshold=0.8,
            verification_results=[],
            critical_required=[],
        )
        assert status == ExtractionStatus.LOW_CONFIDENCE

    def test_all_required_passed(self):
        vr = VerificationResult(
            path="required_field",
            status=VerificationStatus.MATCH,
            extracted_value=1,
        )
        status = compute_status(
            confidence=0.9,
            confidence_threshold=0.8,
            verification_results=[vr],
            critical_required=["required_field"],
        )
        assert status == ExtractionStatus.VERIFIED

    def test_required_failed_partial(self):
        vr1 = VerificationResult(
            path="field1",
            status=VerificationStatus.MATCH,
            extracted_value=1,
        )
        vr2 = VerificationResult(
            path="field2",
            status=VerificationStatus.NOT_FOUND,
            extracted_value=2,
        )
        status = compute_status(
            confidence=0.7,
            confidence_threshold=0.8,
            verification_results=[vr1, vr2],
            critical_required=["field1", "field2"],
        )
        assert status == ExtractionStatus.PARTIAL

    def test_all_required_failed(self):
        vr = VerificationResult(
            path="required",
            status=VerificationStatus.NOT_FOUND,
            extracted_value=1,
        )
        status = compute_status(
            confidence=0.5,
            confidence_threshold=0.8,
            verification_results=[vr],
            critical_required=["required"],
        )
        assert status == ExtractionStatus.FAILED

    def test_all_passed_but_low_confidence(self):
        vr = VerificationResult(
            path="field",
            status=VerificationStatus.MATCH,
            extracted_value=1,
        )
        status = compute_status(
            confidence=0.6,
            confidence_threshold=0.8,
            verification_results=[vr],
            critical_required=["field"],
        )
        assert status == ExtractionStatus.LOW_CONFIDENCE


class TestComputeConfidence:
    """Test compute_confidence function."""

    def test_no_results(self):
        conf = compute_confidence([])
        assert conf == 1.0

    def test_custom_base(self):
        conf = compute_confidence([], base_confidence=0.5)
        assert conf == 0.5

    def test_all_match(self):
        ev = Evidence(
            value=1,
            quote="1",
            page=1,
            bbox=None,
            chunk_index=0,
            confidence=0.9,
            verified=True,
        )
        vr = VerificationResult(
            path="field",
            status=VerificationStatus.MATCH,
            extracted_value=1,
            evidence=ev,
        )
        conf = compute_confidence([vr])
        assert conf == 0.9

    def test_mismatch_partial_credit(self):
        vr = VerificationResult(
            path="field",
            status=VerificationStatus.MISMATCH,
            extracted_value=1,
            found_value=2,
        )
        conf = compute_confidence([vr])
        assert conf == 0.3

    def test_not_found_zero_credit(self):
        vr = VerificationResult(
            path="field",
            status=VerificationStatus.NOT_FOUND,
            extracted_value=1,
        )
        conf = compute_confidence([vr])
        assert conf == 0.0

    def test_skipped_ignored(self):
        vr = VerificationResult(
            path="field",
            status=VerificationStatus.SKIPPED,
            extracted_value=1,
        )
        conf = compute_confidence([vr])
        assert conf == 1.0  # Base confidence, skipped doesn't count

    def test_mixed_results(self):
        ev = Evidence(
            value=1,
            quote="1",
            page=1,
            bbox=None,
            chunk_index=0,
            confidence=0.9,
            verified=True,
        )
        results = [
            VerificationResult(
                path="good",
                status=VerificationStatus.MATCH,
                extracted_value=1,
                evidence=ev,
            ),
            VerificationResult(
                path="bad",
                status=VerificationStatus.NOT_FOUND,
                extracted_value=2,
            ),
        ]
        conf = compute_confidence(results)
        # (0.9 + 0.0) / 2 = 0.45
        assert conf == 0.45

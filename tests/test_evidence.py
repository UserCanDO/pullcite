"""
Tests for evidence.py - Evidence types for verification.
"""

import pytest
from pullcite.core.evidence import (
    Evidence,
    EvidenceCandidate,
    VerificationResult,
    VerificationStatus,
    select_best_candidate,
    create_match_result,
    create_mismatch_result,
    create_not_found_result,
    create_ambiguous_result,
)


class TestEvidence:
    """Test Evidence dataclass."""

    def test_basic_creation(self):
        ev = Evidence(
            value=1500,
            quote="Deductible: $1,500",
            page=3,
            bbox=(100.0, 200.0, 300.0, 220.0),
            chunk_index=5,
            confidence=0.95,
            verified=True,
        )
        assert ev.value == 1500
        assert ev.quote == "Deductible: $1,500"
        assert ev.page == 3
        assert ev.bbox == (100.0, 200.0, 300.0, 220.0)
        assert ev.chunk_index == 5
        assert ev.confidence == 0.95
        assert ev.verified is True
        assert ev.verification_note is None

    def test_minimal_creation(self):
        ev = Evidence(
            value="test",
            quote=None,
            page=None,
            bbox=None,
            chunk_index=None,
            confidence=0.5,
            verified=False,
            verification_note="Could not find in document",
        )
        assert ev.value == "test"
        assert ev.quote is None
        assert ev.verification_note == "Could not find in document"

    def test_confidence_validation_too_low(self):
        with pytest.raises(ValueError) as exc:
            Evidence(
                value=1,
                quote=None,
                page=None,
                bbox=None,
                chunk_index=None,
                confidence=-0.1,
                verified=False,
            )
        assert "confidence must be between 0.0 and 1.0" in str(exc.value)

    def test_confidence_validation_too_high(self):
        with pytest.raises(ValueError) as exc:
            Evidence(
                value=1,
                quote=None,
                page=None,
                bbox=None,
                chunk_index=None,
                confidence=1.5,
                verified=False,
            )
        assert "confidence must be between 0.0 and 1.0" in str(exc.value)

    def test_confidence_boundary_values(self):
        # 0.0 and 1.0 should be valid
        ev_zero = Evidence(
            value=1,
            quote=None,
            page=None,
            bbox=None,
            chunk_index=None,
            confidence=0.0,
            verified=False,
        )
        assert ev_zero.confidence == 0.0

        ev_one = Evidence(
            value=1,
            quote=None,
            page=None,
            bbox=None,
            chunk_index=None,
            confidence=1.0,
            verified=True,
        )
        assert ev_one.confidence == 1.0

    def test_page_validation(self):
        with pytest.raises(ValueError) as exc:
            Evidence(
                value=1,
                quote=None,
                page=0,
                bbox=None,
                chunk_index=None,
                confidence=0.5,
                verified=False,
            )
        assert "page must be >= 1" in str(exc.value)

    def test_page_negative(self):
        with pytest.raises(ValueError):
            Evidence(
                value=1,
                quote=None,
                page=-1,
                bbox=None,
                chunk_index=None,
                confidence=0.5,
                verified=False,
            )

    def test_bbox_validation(self):
        with pytest.raises(ValueError) as exc:
            Evidence(
                value=1,
                quote=None,
                page=None,
                bbox=(1.0, 2.0, 3.0),  # Only 3 values
                chunk_index=None,
                confidence=0.5,
                verified=False,
            )
        assert "bbox must have 4 values" in str(exc.value)

    def test_immutability(self):
        ev = Evidence(
            value=1,
            quote="test",
            page=1,
            bbox=None,
            chunk_index=0,
            confidence=0.9,
            verified=True,
        )
        with pytest.raises(AttributeError):
            ev.value = 2

    def test_with_value(self):
        ev = Evidence(
            value=1500,
            quote="$1,500",
            page=3,
            bbox=(0, 0, 100, 50),
            chunk_index=5,
            confidence=0.9,
            verified=True,
        )
        new_ev = ev.with_value(2000)

        # Original unchanged
        assert ev.value == 1500

        # New has updated value, same everything else
        assert new_ev.value == 2000
        assert new_ev.quote == "$1,500"
        assert new_ev.page == 3
        assert new_ev.confidence == 0.9

    def test_with_confidence(self):
        ev = Evidence(
            value=1500,
            quote="$1,500",
            page=3,
            bbox=None,
            chunk_index=5,
            confidence=0.9,
            verified=True,
        )
        new_ev = ev.with_confidence(0.75)

        assert ev.confidence == 0.9  # Original unchanged
        assert new_ev.confidence == 0.75
        assert new_ev.value == 1500  # Other fields preserved


class TestEvidenceCandidate:
    """Test EvidenceCandidate dataclass."""

    def test_basic_creation(self):
        cand = EvidenceCandidate(
            quote="Deductible: $1,500",
            chunk_index=5,
            score=0.92,
            page=3,
            bbox=(100.0, 200.0, 300.0, 220.0),
            parsed_value=1500,
        )
        assert cand.quote == "Deductible: $1,500"
        assert cand.chunk_index == 5
        assert cand.score == 0.92
        assert cand.page == 3
        assert cand.parsed_value == 1500

    def test_minimal_creation(self):
        cand = EvidenceCandidate(
            quote="Some text",
            chunk_index=0,
            score=0.5,
        )
        assert cand.quote == "Some text"
        assert cand.page is None
        assert cand.bbox is None
        assert cand.parsed_value is None

    def test_page_validation(self):
        with pytest.raises(ValueError):
            EvidenceCandidate(
                quote="test",
                chunk_index=0,
                score=0.5,
                page=0,
            )

    def test_bbox_validation(self):
        with pytest.raises(ValueError):
            EvidenceCandidate(
                quote="test",
                chunk_index=0,
                score=0.5,
                bbox=(1.0, 2.0),  # Only 2 values
            )

    def test_to_evidence(self):
        cand = EvidenceCandidate(
            quote="Copay: $25",
            chunk_index=3,
            score=0.88,
            page=2,
            bbox=(10.0, 20.0, 100.0, 40.0),
            parsed_value=25,
        )

        ev = cand.to_evidence(
            value=25,
            verified=True,
            confidence=0.88,
            note="Exact match",
        )

        assert ev.value == 25
        assert ev.quote == "Copay: $25"
        assert ev.page == 2
        assert ev.bbox == (10.0, 20.0, 100.0, 40.0)
        assert ev.chunk_index == 3
        assert ev.confidence == 0.88
        assert ev.verified is True
        assert ev.verification_note == "Exact match"

    def test_to_evidence_without_note(self):
        cand = EvidenceCandidate(quote="test", chunk_index=0, score=0.5)
        ev = cand.to_evidence(value="test", verified=False, confidence=0.5)

        assert ev.verification_note is None

    def test_immutability(self):
        cand = EvidenceCandidate(quote="test", chunk_index=0, score=0.5)
        with pytest.raises(AttributeError):
            cand.score = 0.9


class TestVerificationStatus:
    """Test VerificationStatus enum."""

    def test_values(self):
        assert VerificationStatus.MATCH.value == "MATCH"
        assert VerificationStatus.MISMATCH.value == "MISMATCH"
        assert VerificationStatus.NOT_FOUND.value == "NOT_FOUND"
        assert VerificationStatus.AMBIGUOUS.value == "AMBIGUOUS"
        assert VerificationStatus.SKIPPED.value == "SKIPPED"

    def test_string_enum(self):
        # .value gives the string
        assert VerificationStatus.MATCH.value == "MATCH"
        # Can compare directly with string (str, Enum inheritance)
        assert VerificationStatus.MATCH == "MATCH"
        # Can use in string formatting
        assert f"{VerificationStatus.MATCH.value}" == "MATCH"


class TestVerificationResult:
    """Test VerificationResult dataclass."""

    def test_match_result(self):
        ev = Evidence(
            value=1500,
            quote="$1,500",
            page=1,
            bbox=None,
            chunk_index=0,
            confidence=0.95,
            verified=True,
        )
        result = VerificationResult(
            path="deductible.individual",
            status=VerificationStatus.MATCH,
            extracted_value=1500,
            found_value=1500,
            evidence=ev,
        )

        assert result.is_match is True
        assert result.is_mismatch is False
        assert result.needs_correction is False
        assert result.is_failure is False

    def test_mismatch_result(self):
        ev = Evidence(
            value=2000,
            quote="$2,000",
            page=1,
            bbox=None,
            chunk_index=0,
            confidence=0.90,
            verified=False,
        )
        result = VerificationResult(
            path="deductible.individual",
            status=VerificationStatus.MISMATCH,
            extracted_value=1500,
            found_value=2000,
            evidence=ev,
        )

        assert result.is_match is False
        assert result.is_mismatch is True
        assert result.needs_correction is True
        assert result.is_failure is True

    def test_not_found_result(self):
        result = VerificationResult(
            path="some.field",
            status=VerificationStatus.NOT_FOUND,
            extracted_value="test",
        )

        assert result.is_match is False
        assert result.needs_correction is False
        assert result.is_failure is True

    def test_ambiguous_result(self):
        cand1 = EvidenceCandidate(
            quote="$100", chunk_index=0, score=0.9, parsed_value=100
        )
        cand2 = EvidenceCandidate(
            quote="$200", chunk_index=1, score=0.9, parsed_value=200
        )

        result = VerificationResult(
            path="copay",
            status=VerificationStatus.AMBIGUOUS,
            extracted_value=100,
            candidates=(cand1, cand2),
        )

        assert result.needs_correction is True
        assert result.is_failure is True
        assert len(result.candidates) == 2

    def test_skipped_result(self):
        result = VerificationResult(
            path="optional.field",
            status=VerificationStatus.SKIPPED,
            extracted_value=None,
        )

        assert result.is_match is False
        assert result.is_failure is False
        assert result.needs_correction is False

    def test_with_search_queries(self):
        result = VerificationResult(
            path="test.path",
            status=VerificationStatus.MATCH,
            extracted_value=100,
            search_queries=("deductible amount", "individual deductible"),
        )

        assert result.search_queries == ("deductible amount", "individual deductible")

    def test_immutability(self):
        result = VerificationResult(
            path="test",
            status=VerificationStatus.MATCH,
            extracted_value=1,
        )
        with pytest.raises(AttributeError):
            result.status = VerificationStatus.MISMATCH


class TestSelectBestCandidate:
    """Test select_best_candidate function."""

    def test_empty_list(self):
        assert select_best_candidate([]) is None

    def test_single_candidate(self):
        cand = EvidenceCandidate(quote="test", chunk_index=0, score=0.8)
        assert select_best_candidate([cand]) is cand

    def test_selects_highest_score(self):
        cand1 = EvidenceCandidate(quote="low", chunk_index=0, score=0.5, page=1)
        cand2 = EvidenceCandidate(quote="high", chunk_index=1, score=0.9, page=1)
        cand3 = EvidenceCandidate(quote="mid", chunk_index=2, score=0.7, page=1)

        best = select_best_candidate([cand1, cand2, cand3])
        assert best is cand2
        assert best.score == 0.9

    def test_tie_break_by_page(self):
        # Same score, different pages - prefer earlier page
        cand1 = EvidenceCandidate(quote="page2", chunk_index=0, score=0.9, page=2)
        cand2 = EvidenceCandidate(quote="page1", chunk_index=1, score=0.9, page=1)

        best = select_best_candidate([cand1, cand2])
        assert best is cand2
        assert best.page == 1

    def test_tie_break_by_chunk_index(self):
        # Same score, same page - prefer earlier chunk
        cand1 = EvidenceCandidate(quote="chunk5", chunk_index=5, score=0.9, page=1)
        cand2 = EvidenceCandidate(quote="chunk2", chunk_index=2, score=0.9, page=1)

        best = select_best_candidate([cand1, cand2])
        assert best is cand2
        assert best.chunk_index == 2

    def test_none_page_sorts_last(self):
        cand1 = EvidenceCandidate(quote="no page", chunk_index=0, score=0.9, page=None)
        cand2 = EvidenceCandidate(quote="page 5", chunk_index=1, score=0.9, page=5)

        best = select_best_candidate([cand1, cand2])
        assert best is cand2
        assert best.page == 5

    def test_full_tie_break_chain(self):
        # Test complete priority: score > page > chunk_index
        candidates = [
            EvidenceCandidate(quote="a", chunk_index=5, score=0.8, page=1),
            EvidenceCandidate(
                quote="b", chunk_index=2, score=0.9, page=2
            ),  # High score but later page
            EvidenceCandidate(
                quote="c", chunk_index=3, score=0.9, page=1
            ),  # High score, early page
            EvidenceCandidate(
                quote="d", chunk_index=1, score=0.9, page=1
            ),  # High score, early page, early chunk - WINNER
        ]

        best = select_best_candidate(candidates)
        assert best.quote == "d"


class TestHelperFunctions:
    """Test helper functions for creating VerificationResults."""

    def test_create_match_result(self):
        ev = Evidence(
            value=100,
            quote="$100",
            page=1,
            bbox=None,
            chunk_index=0,
            confidence=0.95,
            verified=True,
        )
        result = create_match_result(
            path="test.path",
            value=100,
            evidence=ev,
            search_queries=["test query"],
        )

        assert result.status == VerificationStatus.MATCH
        assert result.extracted_value == 100
        assert result.found_value == 100
        assert result.evidence is ev
        assert result.search_queries == ("test query",)

    def test_create_mismatch_result(self):
        ev = Evidence(
            value=200,
            quote="$200",
            page=1,
            bbox=None,
            chunk_index=0,
            confidence=0.90,
            verified=False,
        )
        result = create_mismatch_result(
            path="test.path",
            extracted_value=100,
            found_value=200,
            evidence=ev,
        )

        assert result.status == VerificationStatus.MISMATCH
        assert result.extracted_value == 100
        assert result.found_value == 200

    def test_create_not_found_result(self):
        result = create_not_found_result(
            path="missing.field",
            extracted_value="something",
            search_queries=["search1", "search2"],
        )

        assert result.status == VerificationStatus.NOT_FOUND
        assert result.extracted_value == "something"
        assert result.found_value is None
        assert result.evidence is None
        assert len(result.search_queries) == 2

    def test_create_ambiguous_result(self):
        cand1 = EvidenceCandidate(quote="$100", chunk_index=0, score=0.9)
        cand2 = EvidenceCandidate(quote="$200", chunk_index=1, score=0.9)

        result = create_ambiguous_result(
            path="ambiguous.field",
            extracted_value=100,
            candidates=[cand1, cand2],
        )

        assert result.status == VerificationStatus.AMBIGUOUS
        assert len(result.candidates) == 2

    def test_create_match_result_defaults(self):
        ev = Evidence(
            value=1,
            quote="1",
            page=None,
            bbox=None,
            chunk_index=None,
            confidence=0.8,
            verified=True,
        )
        result = create_match_result(path="p", value=1, evidence=ev)

        assert result.candidates == ()
        assert result.search_queries == ()


class TestEvidenceEquality:
    """Test equality comparison for evidence types."""

    def test_evidence_equality(self):
        ev1 = Evidence(
            value=100,
            quote="$100",
            page=1,
            bbox=None,
            chunk_index=0,
            confidence=0.9,
            verified=True,
        )
        ev2 = Evidence(
            value=100,
            quote="$100",
            page=1,
            bbox=None,
            chunk_index=0,
            confidence=0.9,
            verified=True,
        )
        assert ev1 == ev2

    def test_evidence_inequality(self):
        ev1 = Evidence(
            value=100,
            quote="$100",
            page=1,
            bbox=None,
            chunk_index=0,
            confidence=0.9,
            verified=True,
        )
        ev2 = Evidence(
            value=200,
            quote="$200",
            page=1,
            bbox=None,
            chunk_index=0,
            confidence=0.9,
            verified=True,
        )
        assert ev1 != ev2

    def test_candidate_equality(self):
        cand1 = EvidenceCandidate(quote="test", chunk_index=0, score=0.5)
        cand2 = EvidenceCandidate(quote="test", chunk_index=0, score=0.5)
        assert cand1 == cand2


class TestEvidenceHashability:
    """Test that evidence types can be used in sets."""

    def test_evidence_hashable(self):
        ev = Evidence(
            value=100,
            quote="$100",
            page=1,
            bbox=None,
            chunk_index=0,
            confidence=0.9,
            verified=True,
        )
        # Should not raise
        hash(ev)
        s = {ev}
        assert ev in s

    def test_candidate_hashable(self):
        cand = EvidenceCandidate(quote="test", chunk_index=0, score=0.5)
        hash(cand)
        s = {cand}
        assert cand in s

    def test_result_hashable(self):
        result = VerificationResult(
            path="test",
            status=VerificationStatus.MATCH,
            extracted_value=1,
        )
        hash(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

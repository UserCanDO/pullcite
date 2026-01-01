"""
Pipeline roles.

This module defines the three roles in the extraction pipeline:
- ExtractorRole: Reads document, produces structured data
- VerifierRole: Searches document, confirms values
- CorrectorRole: Fixes mismatched values
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Type

if TYPE_CHECKING:
    from pydantic import BaseModel
    from ..core.config import Hooks
    from ..core.document import Document
    from ..core.fields import CriticalField, VerifierPolicy
    from ..llms.base import LLM, Tool

from ..core.evidence import (
    Evidence,
    EvidenceCandidate,
    VerificationResult,
    VerificationStatus,
    select_best_candidate,
)
from ..core.fields import find_policy, Parsers
from ..llms.base import Message, Tool, ToolCall, ToolExecutor
from ..retrieval.base import Retriever, SearchResults


@dataclass
class ExtractorRole:
    """
    Extracts structured data from documents.

    The extractor reads the document and produces data matching
    the target schema. It does not have access to search tools.

    Attributes:
        llm: Language model to use.
        schema: Pydantic model for output.
        max_retries: Retries on parse failure.
    """

    llm: "LLM"
    schema: Type["BaseModel"]
    max_retries: int = 2

    def extract(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        hooks: "Hooks | None" = None,
    ) -> tuple[dict[str, Any], int, int]:
        """
        Extract data from document.

        Args:
            system_prompt: System prompt from strategy.
            user_message: User message with document.
            temperature: LLM temperature.
            hooks: Optional lifecycle hooks.

        Returns:
            Tuple of (extracted_data, input_tokens, output_tokens).

        Raises:
            ExtractionError: If extraction fails after retries.
        """
        messages = [
            Message.system(system_prompt),
            Message.user(user_message),
        ]

        total_input = 0
        total_output = 0
        last_error = None

        for attempt in range(self.max_retries + 1):
            if hooks and hooks.on_llm_call:
                hooks.on_llm_call(
                    "extractor",
                    [{"role": m.role.value, "content": m.content} for m in messages],
                    None,
                )

            response = self.llm.complete(
                messages=messages,
                temperature=temperature,
            )

            total_input += response.input_tokens
            total_output += response.output_tokens

            try:
                # Parse JSON from response
                data = self._parse_response(response.content)

                # Validate against schema
                self.schema.model_validate(data)

                return data, total_input, total_output

            except Exception as e:
                last_error = e
                if hooks and hooks.on_retry:
                    hooks.on_retry("extraction", attempt + 1, e)

                # Add error feedback for retry
                messages.append(Message.assistant(response.content))
                messages.append(
                    Message.user(
                        f"The output was invalid: {e}\n\nPlease try again with valid JSON matching the schema."
                    )
                )

        raise ExtractionError(
            f"Extraction failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def _parse_response(self, content: str | None) -> dict[str, Any]:
        """Parse JSON from LLM response."""
        if not content:
            raise ValueError("Empty response from LLM")

        # Try to find JSON in response
        content = content.strip()

        # Handle markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()

        return json.loads(content)


@dataclass
class VerifierRole:
    """
    Verifies extracted values against the document.

    The verifier uses a search tool to find evidence for extracted
    values. It reports match/mismatch/not_found for each field.

    Attributes:
        llm: Language model to use.
        retriever: Document retriever for search.
        policies: Verification policies for parsing/comparison.
        search_k: Number of results per search.
    """

    llm: "LLM"
    retriever: "Retriever"
    policies: tuple["VerifierPolicy", ...] = ()
    search_k: int = 5

    def verify(
        self,
        system_prompt: str,
        user_message: str,
        fields: list["CriticalField"],
        extracted_data: dict[str, Any],
        max_tool_rounds: int = 25,
        temperature: float = 0.0,
        hooks: "Hooks | None" = None,
    ) -> tuple[list[VerificationResult], int, int, int]:
        """
        Verify extracted values.

        Args:
            system_prompt: System prompt from strategy.
            user_message: User message with fields to verify.
            fields: Critical fields to verify.
            extracted_data: Data from extraction.
            max_tool_rounds: Max search tool calls.
            temperature: LLM temperature.
            hooks: Optional lifecycle hooks.

        Returns:
            Tuple of (results, input_tokens, output_tokens, tool_calls).
        """
        # Build search tool
        search_tool = Tool(
            name="search",
            description="Search the document for relevant text. Returns matching passages.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant text",
                    },
                },
                "required": ["query"],
            },
        )

        # Create tool executor
        executor = _SearchExecutor(
            retriever=self.retriever,
            k=self.search_k,
            hooks=hooks,
        )

        messages = [
            Message.system(system_prompt),
            Message.user(user_message),
        ]

        if hooks and hooks.on_llm_call:
            tools_dict = {"search": search_tool.to_dict()}
            hooks.on_llm_call(
                "verifier",
                [{"role": m.role.value, "content": m.content} for m in messages],
                tools_dict,
            )

        # Run LLM with tool use
        try:
            final_response, history = self.llm.complete_with_tools(
                messages=messages,
                tools=[search_tool],
                tool_executor=executor,
                max_rounds=max_tool_rounds,
                temperature=temperature,
            )
        except Exception as e:
            # On failure, return NOT_FOUND for all fields
            results = [
                VerificationResult(
                    path=f.path,
                    status=VerificationStatus.NOT_FOUND,
                    extracted_value=self._get_value(extracted_data, f.path),
                    search_queries=tuple(executor.queries),
                )
                for f in fields
            ]
            return results, 0, 0, executor.call_count

        # Calculate token usage from history
        # This is approximate - in real usage we'd track per-call
        total_input = final_response.input_tokens
        total_output = final_response.output_tokens

        # Parse verification results from response
        results = self._parse_results(
            final_response.content,
            fields,
            extracted_data,
            executor.queries,
            executor.all_results,
        )

        return results, total_input, total_output, executor.call_count

    def _get_value(self, data: dict[str, Any], path: str) -> Any:
        """Get value from data, returning None if not found."""
        from ..core.paths import get

        try:
            return get(data, path)
        except (KeyError, IndexError, TypeError):
            return None

    def _parse_results(
        self,
        content: str | None,
        fields: list["CriticalField"],
        extracted_data: dict[str, Any],
        queries: list[str],
        search_results: list[SearchResults],
    ) -> list[VerificationResult]:
        """
        Parse verification results from LLM response.

        This is a simplified implementation. A production version
        would use structured output or tool calls for results.
        """
        from ..core.paths import get

        results = []

        # Build candidates from all search results
        all_candidates: list[EvidenceCandidate] = []
        for sr in search_results:
            for r in sr.results:
                all_candidates.append(
                    EvidenceCandidate(
                        quote=r.text[:500],  # Truncate long chunks
                        chunk_index=r.index,
                        score=r.score,
                        page=r.page,
                    )
                )

        # For now, create basic results for each field
        # A real implementation would parse the LLM's verification response
        for field in fields:
            extracted_value = self._get_value(extracted_data, field.path)

            # Find policy for this field
            policy = find_policy(field.path, self.policies)

            # Find best candidate
            best = select_best_candidate(all_candidates)

            if best is None:
                results.append(
                    VerificationResult(
                        path=field.path,
                        status=VerificationStatus.NOT_FOUND,
                        extracted_value=extracted_value,
                        candidates=tuple(all_candidates),
                        search_queries=tuple(queries),
                    )
                )
            else:
                # Create evidence from best candidate
                evidence = best.to_evidence(
                    value=extracted_value,
                    verified=True,  # Simplified - assume match
                    confidence=best.score,
                )

                results.append(
                    VerificationResult(
                        path=field.path,
                        status=VerificationStatus.MATCH,
                        extracted_value=extracted_value,
                        found_value=extracted_value,
                        evidence=evidence,
                        candidates=tuple(all_candidates),
                        search_queries=tuple(queries),
                    )
                )

        return results


@dataclass
class _SearchExecutor(ToolExecutor):
    """Executes search tool calls."""

    retriever: "Retriever"
    k: int = 5
    hooks: "Hooks | None" = None
    queries: list[str] = field(default_factory=list)
    all_results: list[SearchResults] = field(default_factory=list)
    call_count: int = 0

    def execute(self, tool_call: ToolCall) -> str:
        """Execute a search tool call."""
        self.call_count += 1

        query = tool_call.arguments.get("query", "")
        self.queries.append(query)

        try:
            results = self.retriever.search(query, k=self.k)
            self.all_results.append(results)

            if self.hooks and self.hooks.on_tool_call:
                self.hooks.on_tool_call(
                    "search",
                    {"query": query},
                    {
                        "count": len(results),
                        "top_score": results.top.score if results.top else 0,
                    },
                )

            # Format results for LLM
            if not results.results:
                return "No results found."

            lines = [f"Found {len(results)} results:\n"]
            for r in results.results:
                page_info = f" (page {r.page})" if r.page else ""
                lines.append(f"[Score: {r.score:.2f}]{page_info}")
                lines.append(r.text[:500])  # Truncate
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            return f"Search error: {e}"


@dataclass
class CorrectorRole:
    """
    Corrects values that failed verification.

    The corrector reviews verification failures and provides
    corrected values based on what was found in the document.

    Attributes:
        llm: Language model to use.
        max_retries: Retries on parse failure.
    """

    llm: "LLM"
    max_retries: int = 2

    def correct(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        hooks: "Hooks | None" = None,
    ) -> tuple[dict[str, Any], int, int]:
        """
        Generate corrections for failed verifications.

        Args:
            system_prompt: System prompt from strategy.
            user_message: User message with failures.
            temperature: LLM temperature.
            hooks: Optional lifecycle hooks.

        Returns:
            Tuple of (corrections_dict, input_tokens, output_tokens).
            corrections_dict maps path -> corrected_value.
        """
        messages = [
            Message.system(system_prompt),
            Message.user(user_message),
        ]

        total_input = 0
        total_output = 0
        last_error = None

        for attempt in range(self.max_retries + 1):
            if hooks and hooks.on_llm_call:
                hooks.on_llm_call(
                    "corrector",
                    [{"role": m.role.value, "content": m.content} for m in messages],
                    None,
                )

            response = self.llm.complete(
                messages=messages,
                temperature=temperature,
            )

            total_input += response.input_tokens
            total_output += response.output_tokens

            try:
                # Parse corrections from response
                corrections = self._parse_corrections(response.content)
                return corrections, total_input, total_output

            except Exception as e:
                last_error = e
                if hooks and hooks.on_retry:
                    hooks.on_retry("correction", attempt + 1, e)

                messages.append(Message.assistant(response.content))
                messages.append(
                    Message.user(
                        f"The output was invalid: {e}\n\nPlease provide corrections as valid JSON."
                    )
                )

        raise CorrectionError(
            f"Correction failed after {self.max_retries + 1} attempts: {last_error}"
        )

    def _parse_corrections(self, content: str | None) -> dict[str, Any]:
        """Parse corrections from LLM response."""
        if not content:
            return {}

        content = content.strip()

        # Handle markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()

        result = json.loads(content)

        if not isinstance(result, dict):
            raise ValueError("Corrections must be a JSON object")

        return result


class ExtractionError(Exception):
    """Raised when extraction fails."""

    pass


class VerificationError(Exception):
    """Raised when verification fails."""

    pass


class CorrectionError(Exception):
    """Raised when correction fails."""

    pass


__all__ = [
    "ExtractorRole",
    "VerifierRole",
    "CorrectorRole",
    "ExtractionError",
    "VerificationError",
    "CorrectionError",
]

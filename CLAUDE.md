# CLAUDE.md

## Project Overview

**Pullcite** — Evidence-backed structured extraction from documents.

Extracts structured data from PDFs/DOCX/text into Pydantic schemas, then **verifies critical fields with evidence** (quote + page + bbox) and applies **minimal patches** when verification fails.

Core promise: Every important value traces back to where it came from in the source document.

## Commands
```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Testing
pytest tests/ -v                          # Run all tests
pytest tests/test_paths.py -v             # Run specific test file
pytest tests/ -v --cov=pullcite             # With coverage
pytest tests/ -v -k "test_ambiguous"      # Run tests matching pattern

# Type checking
mypy pullcite/

# Linting
ruff check pullcite/
ruff format pullcite/
```

## Project Structure
```
pullcite/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── pullcite/
│   ├── __init__.py           # Public API exports
│   ├── core/
│   │   ├── __init__.py
│   │   ├── paths.py          # ✅ Path parsing, get/set/expand
│   │   ├── chunk.py          # ✅ Immutable document chunks
│   │   ├── document.py       # ✅ Document loading + chunking
│   │   ├── evidence.py       # 🔲 Evidence, VerificationResult
│   │   ├── fields.py         # 🔲 CriticalField, VerifierPolicy, Parsers
│   │   ├── result.py         # 🔲 ExtractionResult, ExtractionStats
│   │   ├── config.py         # 🔲 ExtractorConfig, Hooks
│   │   └── extractor.py      # 🔲 Main Extractor orchestrator
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py           # 🔲 Embedder ABC
│   │   ├── openai.py         # 🔲 OpenAI embeddings
│   │   ├── voyage.py         # 🔲 Voyage embeddings
│   │   └── local.py          # 🔲 Sentence Transformers
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── base.py           # 🔲 Retriever ABC, SearchResult
│   │   ├── memory.py         # 🔲 In-memory NumPy retriever
│   │   ├── chroma.py         # 🔲 ChromaDB
│   │   └── pgvector.py       # 🔲 PostgreSQL + pgvector
│   ├── llms/
│   │   ├── __init__.py
│   │   ├── base.py           # 🔲 LLM ABC, Tool, ToolCall
│   │   ├── anthropic.py      # 🔲 Claude
│   │   └── openai.py         # 🔲 GPT-4
│   └── pipeline/
│       ├── __init__.py
│       ├── strategy.py       # 🔲 ExtractionStrategy, ScaffoldStrategy
│       ├── patcher.py        # 🔲 Patch application
│       └── roles.py          # 🔲 ExtractorRole, VerifierRole, CorrectorRole
└── tests/
    ├── test_paths.py         # ✅ 111 tests
    ├── test_chunk.py         # ✅ 52 tests
    └── test_document.py      # ✅ 40 tests
```

## Implementation Status

### ✅ Phase 1 — Foundations (COMPLETE)
1. `core/paths.py` — Path grammar, parse/get/set/delete/expand, ambiguity errors
2. `core/chunk.py` — Immutable chunks, sorted metadata, validation
3. `core/document.py` — Document loaders, chunking, deterministic IDs

### 🔲 Phase 2 — Evidence + Policies
4. `core/evidence.py` — Evidence, VerificationResult, EvidenceCandidate
5. `core/fields.py` — CriticalField, VerifierPolicy, Parsers, Comparators
6. `core/result.py` — ExtractionResult, ExtractionStats, ExtractionFlag
7. `core/config.py` — ExtractorConfig, Hooks

### 🔲 Phase 3 — Retrieval
8. `embeddings/base.py` — Embedder ABC
9. `embeddings/openai.py` — OpenAI embeddings
10. `retrieval/base.py` — Retriever ABC, SearchResult
11. `retrieval/memory.py` — In-memory NumPy retriever

### 🔲 Phase 4 — Pipeline
12. `llms/base.py` — LLM ABC, Tool, ToolCall
13. `llms/anthropic.py` — Claude integration
14. `pipeline/strategy.py` — ExtractionStrategy, DefaultStrategy
15. `pipeline/patcher.py` — Patch application with validation
16. `pipeline/roles.py` — ExtractorRole, VerifierRole, CorrectorRole

### 🔲 Phase 5 — Orchestrator
17. `core/extractor.py` — Main Extractor class
18. `__init__.py` — Public API exports

## Key Design Decisions

### Path Grammar
```
path          = segment ("." segment)*
segment       = identifier selector?
identifier    = [a-zA-Z_][a-zA-Z0-9_]*
selector      = "[" selector_key "]"
selector_key  = index | wildcard | key
index         = [0-9]+           # Pure digits = index
wildcard      = "*"
key           = [a-zA-Z0-9_]+    # Anything else = key lookup
```

**Examples:**
- `vendor.name` — nested field
- `items[0].price` — index access (0-based)
- `services[PCP_VISIT].copay` — key lookup
- `services[*].copay` — wildcard (expand only)

### Key Lookup Priority

When resolving `[KEY]` in a list of dicts, check fields in order:
1. `service_code`
2. `code`
3. `id`
4. `key`
5. `name`

### Ambiguity Handling

| Situation | Policy |
|-----------|--------|
| `[KEY]` matches multiple items | Raise `AmbiguousPathError` |
| `[KEY]` matches zero items | Raise `PathNotFoundError` |
| Index out of bounds | Raise `PathNotFoundError` (strict) or return default (soft) |
| Wildcard in `set()`/`delete()` | Raise `PathError` |

### get() vs get_strict()

- `get(data, path, default=None)` — Soft, returns default on missing, never raises `PathNotFoundError`
- `get_strict(data, path)` — Raises `PathNotFoundError` if path doesn't exist
- Both raise `AmbiguousPathError` and `InvalidPathError`

### Chunk Immutability

- `Chunk` is a frozen dataclass
- Metadata stored as `tuple[tuple[str, Any], ...]` (sorted by key)
- Use `chunk.with_metadata(key=value)` to create modified copy
- Validation: `index >= 0`, `page >= 1` (if set), `bbox` has 4 floats

### Document IDs

- Generated from content hash (SHA-256, first 16 chars)
- Same content = same ID (deterministic for caching)
- Filename stored separately, not part of ID

### Evidence Selection Tie-break

When multiple evidence candidates exist:
1. Highest similarity score
2. Lowest page number
3. Lowest chunk index

### Patching Safety

- Corrections are patch-only (no full re-extraction)
- `set()` creates intermediate dicts but refuses to grow lists by default
- Use `allow_list_growth=True` explicitly if needed

## Code Patterns

### Creating a new module

1. Create `pullcite/core/mymodule.py` (or appropriate subpackage)
2. Create `tests/test_mymodule.py`
3. Write tests first (TDD encouraged)
4. Implement until tests pass
5. Add exports to `__init__.py` if public API

### Test file template
```python
"""
Tests for {module_name}.
"""

import pytest
from pullcite.core.{module_name} import {ClassOrFunction}


class TestClassName:
    """Test {description}."""
    
    def test_basic_case(self):
        # Arrange
        # Act
        # Assert
        pass
    
    def test_edge_case(self):
        pass
    
    def test_error_case(self):
        with pytest.raises(SomeError):
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Dataclass pattern (immutable)
```python
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class MyClass:
    required_field: str
    optional_field: int | None = None
    
    def __post_init__(self) -> None:
        """Validate invariants."""
        if not self.required_field:
            raise ValueError("required_field cannot be empty")
    
    def with_field(self, **kwargs) -> "MyClass":
        """Return copy with modified fields."""
        return MyClass(
            required_field=kwargs.get("required_field", self.required_field),
            optional_field=kwargs.get("optional_field", self.optional_field),
        )
```

### ABC pattern
```python
from abc import ABC, abstractmethod

class MyBase(ABC):
    """Abstract base for {purpose}."""
    
    @abstractmethod
    def required_method(self, arg: str) -> int:
        """Do something.
        
        Args:
            arg: Description.
            
        Returns:
            Description.
        """
        ...
    
    def optional_method(self) -> None:
        """Default implementation."""
        pass
```

## Common Gotchas

1. **Path selectors with pure digits are ALWAYS index access**
   - `items[123]` = index 123, not key "123"
   - If you need key "123", you can't (by design, no escaping)

2. **Chunks don't contain embeddings**
   - Embeddings are managed by Retriever
   - Chunk is just text + location

3. **Metadata must be JSON-serializable**
   - `chunk.with_metadata(bad=object())` raises TypeError

4. **Wildcard expansion uses natural keys when available**
   - `expand(data, "items[*].price")` returns `["items[PCP].price", ...]` not `["items[0].price", ...]`
   - Falls back to index if no key field found

5. **Tests shadow built-in `set()`**
   - Don't use `set(paths)` in tests, use `sorted(paths)` or rename variable

## Dependencies

### Required
- `pydantic>=2.0` — Schema definitions

### Optional (extras)
- `anthropic` — Claude LLM
- `openai` — GPT-4 + embeddings
- `chromadb` — ChromaDB vector store
- `psycopg2-binary`, `pgvector` — PostgreSQL vector store
- `sentence-transformers` — Local embeddings
- `pdfminer.six` — PDF loading
- `python-docx` — DOCX loading

### Dev
- `pytest`, `pytest-cov` — Testing
- `mypy` — Type checking
- `ruff` — Linting/formatting

## Git Workflow
```bash
# Before committing
pytest tests/ -v
mypy pullcite/
ruff check pullcite/

# Commit message format
# feat: Add evidence.py with Evidence and VerificationResult
# fix: Handle empty metadata in Chunk serialization
# test: Add edge case tests for path expansion
# docs: Update CLAUDE.md with Phase 2 status
```

## Links

- Design discussion: See chat transcript for full architecture decisions
- Path grammar: `pullcite/core/paths.py` module docstring
- README: Implementation order and API examples

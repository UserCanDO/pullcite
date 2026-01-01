## Pullcite Implementation Roadmap

### Phase 1 — Foundations ✅ COMPLETE

Core primitives with no internal dependencies.

| Step | File | Description | Status |
|------|------|-------------|--------|
| 1.1 | `core/paths.py` | Path grammar parser, get/set/delete/expand, ambiguity errors, wildcard expansion | ✅ Done |
| 1.2 | `core/chunk.py` | Immutable document chunk with page/bbox, sorted metadata, validation | ✅ Done |
| 1.3 | `core/document.py` | Document loaders (text, Docling), chunking with overlap, deterministic ID from content hash | ✅ Done |

**Tests:** 203 passing

---

### Phase 2 — Evidence + Policies ← YOU ARE HERE

Types that define what verification means.

| Step | File | Description | Depends on |
|------|------|-------------|------------|
| 2.1 | `core/evidence.py` | `Evidence` (proof a value came from doc), `EvidenceCandidate` (potential match before selection), `VerificationResult` (outcome of verifying one field) | None |
| 2.2 | `core/fields.py` | `CriticalField` (what to verify + search hints), `VerifierPolicy` (how to parse/compare values), `Parsers` (money, percent, yes_no, integer), `Comparators` (exact, normalized, numeric_tolerance) | `paths` (for path validation) |
| 2.3 | `core/result.py` | `ExtractionResult` (final output with data + evidence), `ExtractionStats` (token counts, timing per phase), `ExtractionFlag` (warnings like AMBIGUOUS, LOW_CONFIDENCE) | `evidence` |
| 2.4 | `core/config.py` | `ExtractorConfig` (thresholds, retry limits, temperature), `Hooks` (post_extract, post_verify, post_correct callbacks) | `fields`, `result` |

**Milestone:** Can define what fields to verify and how to judge correctness.

---

### Phase 3 — Retrieval

Search infrastructure the Verifier uses to find evidence.

| Step | File | Description | Depends on |
|------|------|-------------|------------|
| 3.1 | `embeddings/base.py` | `Embedder` ABC with `embed(text) -> list[float]` and `embed_batch(texts)` | None |
| 3.2 | `embeddings/openai.py` | OpenAI embeddings implementation (text-embedding-3-small) | `embeddings/base` |
| 3.3 | `retrieval/base.py` | `Retriever` ABC with `index(doc)`, `search(query, doc_id)`, `SearchResult` dataclass with chunk + score + provenance | `chunk` |
| 3.4 | `retrieval/memory.py` | In-memory NumPy retriever with cosine similarity, document-scoped indexing | `retrieval/base`, `embeddings/base` |

**Milestone:** Can index a document and search for relevant chunks by query.

---

### Phase 4 — Pipeline

The three LLM roles and their coordination.

| Step | File | Description | Depends on |
|------|------|-------------|------------|
| 4.1 | `llms/base.py` | `LLM` ABC with `complete(messages, tools)`, `Tool` schema definition, `ToolCall` dataclass, `Message` types | None |
| 4.2 | `llms/anthropic.py` | Claude implementation with tool use support | `llms/base` |
| 4.3 | `pipeline/strategy.py` | `ExtractionStrategy` ABC with `build_prompt()` and `build_scaffold()`, `DefaultStrategy` for simple schemas | `document` |
| 4.4 | `pipeline/patcher.py` | `Patch` dataclass (path + value), `apply_patches()` using paths.set, validation against schema, rollback on failure | `paths` |
| 4.5 | `pipeline/roles.py` | `ExtractorRole` (fill schema from doc), `VerifierRole` (tool loop to find evidence), `CorrectorRole` (generate patches from failures) | `llms/base`, `retrieval/base`, `evidence`, `fields`, `patcher`, `strategy` |

**Milestone:** Can run extract → verify → correct pipeline with real LLM.

---

### Phase 5 — Orchestrator

Wire everything together into the public API.

| Step | File | Description | Depends on |
|------|------|-------------|------------|
| 5.1 | `core/extractor.py` | `Extractor` class that orchestrates roles, manages retries, computes confidence, attaches evidence map, runs hooks | Everything |
| 5.2 | `__init__.py` | Public API exports: `Extractor`, `Document`, `CriticalField`, `Evidence`, `ExtractionResult`, `ExtractorConfig` | Everything |

**Milestone:** Library is usable end-to-end.

---

### Phase 6 — Hardening (post-MVP)

Production readiness, additional backends.

| Step | File | Description | Priority |
|------|------|-------------|----------|
| 6.1 | `embeddings/voyage.py` | Voyage embeddings (better retrieval quality) | Medium |
| 6.2 | `embeddings/local.py` | Sentence Transformers (no API costs) | Low |
| 6.3 | `llms/openai.py` | GPT-4 support | Medium |
| 6.4 | `retrieval/chroma.py` | ChromaDB for persistence | Medium |
| 6.5 | `retrieval/pgvector.py` | PostgreSQL + pgvector for production | High |
| 6.6 | Async support | `async def extract()` option | Medium |
| 6.7 | Batch extraction | Multiple documents in parallel | Low |
| 6.8 | Caching | Cache embeddings by content hash | Medium |

---

## Dependency Graph

```
                    ┌─────────────────────────────────────────┐
                    │           Phase 1 (DONE)                │
                    │  paths ← chunk ← document               │
                    └─────────────────┬───────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────────┐
                    │           Phase 2 (NOW)                 │
                    │  evidence → fields → result → config    │
                    └─────────────────┬───────────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────┐       ┌─────────────────────┐      ┌──────────────────┐
│ embeddings/base │       │     llms/base       │      │ pipeline/strategy│
└────────┬────────┘       └──────────┬──────────┘      └────────┬─────────┘
         │                           │                          │
         ▼                           ▼                          │
┌─────────────────┐       ┌─────────────────────┐               │
│embeddings/openai│       │  llms/anthropic     │               │
└────────┬────────┘       └──────────┬──────────┘               │
         │                           │                          │
         ▼                           │                          │
┌─────────────────┐                  │                          │
│ retrieval/base  │                  │                          │
└────────┬────────┘                  │                          │
         │                           │                          │
         ▼                           │                          │
┌─────────────────┐                  │        ┌─────────────────┘
│retrieval/memory │                  │        │
└────────┬────────┘                  │        │
         │                           │        │
         │         ┌─────────────────┘        │
         │         │    ┌────────────────────┬┘
         │         │    │                    │
         │         ▼    ▼                    ▼
         │    ┌──────────────┐      ┌─────────────────┐
         │    │pipeline/roles│◄─────│pipeline/patcher │
         │    └──────┬───────┘      └─────────────────┘
         │           │
         └─────┬─────┘
               │
               ▼
       ┌───────────────┐
       │   extractor   │
       └───────┬───────┘
               │
               ▼
       ┌───────────────┐
       │  __init__.py  │
       └───────────────┘
```

---

## Time Estimates (rough)

| Phase | Effort | Cumulative |
|-------|--------|------------|
| 1 ✅ | Done | Done |
| 2 | 2-3 hours | 2-3 hours |
| 3 | 2-3 hours | 4-6 hours |
| 4 | 4-6 hours | 8-12 hours |
| 5 | 2-3 hours | 10-15 hours |
| 6 | As needed | Ongoing |

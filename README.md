# Pullcite

**Evidence-backed structured extraction from documents.**

Pullcite loads a document (PDF, DOCX, text), breaks it into location-aware chunks, uses an LLM to extract structured data into a Pydantic schema, then **verifies critical fields with evidence** and applies **minimal patches** when verification fails.

The core promise:

> Every important value can be traced back to *where it came from* in the source document (quote + page + bbox), and corrections are *surgical*, not a full re-extraction.

---

## Installation

```bash
# Core package
pip install pullcite

# With specific providers
pip install pullcite[anthropic]      # Anthropic Claude LLM
pip install pullcite[openai]         # OpenAI GPT LLM + embeddings
pip install pullcite[voyage]         # Voyage AI embeddings
pip install pullcite[local]          # Local sentence-transformers
pip install pullcite[chroma]         # ChromaDB vector store
pip install pullcite[pgvector]       # PostgreSQL pgvector
pip install pullcite[docling]        # Document parsing (PDF, DOCX)

# Everything
pip install pullcite[all]

# Development
pip install pullcite[dev]
```

---

## Quick Start

```python
from pydantic import BaseModel
from pullcite.core.document import Document
from pullcite.core.chunk import Chunk
from pullcite.embeddings import OpenAIEmbedder
from pullcite.retrieval import MemoryRetriever
from pullcite.llms import AnthropicLLM, Message

# 1. Define your schema
class Invoice(BaseModel):
    vendor_name: str | None = None
    invoice_number: str | None = None
    total_amount: float | None = None
    due_date: str | None = None

# 2. Load a document
doc = Document.from_file("invoice.pdf")
print(f"Loaded {len(doc.chunks)} chunks from {doc.filename}")

# 3. Set up retrieval for evidence search
embedder = OpenAIEmbedder()  # Uses OPENAI_API_KEY env var
retriever = MemoryRetriever(embedder)
retriever.index(doc)

# 4. Search for relevant context
results = retriever.search("total amount due", k=5)
for r in results:
    print(f"Score {r.score:.2f} (page {r.page}): {r.text[:100]}...")

# 5. Extract with an LLM
llm = AnthropicLLM()  # Uses ANTHROPIC_API_KEY env var
context = "\n\n".join(results.texts)

response = llm.complete([
    Message.system("Extract invoice data from the provided context."),
    Message.user(f"Context:\n{context}\n\nExtract: vendor_name, invoice_number, total_amount, due_date"),
])
print(response.content)
```

---

## Usage Examples

### Document Loading

```python
from pullcite.core.document import Document

# From file path (auto-detects format)
doc = Document.from_file("contract.pdf")
doc = Document.from_file("report.docx")
doc = Document.from_file("notes.md")

# From bytes
with open("file.pdf", "rb") as f:
    doc = Document.from_bytes(f.read(), filename="file.pdf")

# From text string
doc = Document.from_text("""
    Insurance Policy Summary

    Deductible: $500 Individual / $1,000 Family
    Out-of-pocket Maximum: $5,000 Individual
""")

# Access chunks
for chunk in doc.chunks:
    print(f"Chunk {chunk.index} (page {chunk.page}): {chunk.text[:50]}...")

# Custom chunking
doc = Document.from_file(
    "large_doc.pdf",
    chunk_size=1000,      # Characters per chunk
    chunk_overlap=200,    # Overlap between chunks
)
```

### Embeddings

```python
from pullcite.embeddings import (
    OpenAIEmbedder,
    VoyageEmbedder,
    LocalEmbedder,
    CachedEmbedder,
)

# OpenAI embeddings
embedder = OpenAIEmbedder(
    api_key="sk-...",  # Or use OPENAI_API_KEY env var
    model="text-embedding-3-small",  # Default
)
result = embedder.embed("What is the deductible?")
print(f"Vector dimensions: {len(result.vector)}")  # 1536
print(f"Tokens used: {result.token_count}")

# Batch embedding
texts = ["deductible", "copay", "out-of-pocket maximum"]
batch_result = embedder.embed_batch(texts)
print(f"Embedded {len(batch_result.vectors)} texts")

# Voyage AI embeddings (optimized for retrieval)
voyage = VoyageEmbedder(
    api_key="...",  # Or use VOYAGE_API_KEY env var
    model="voyage-3",
)
# Use specialized methods for better retrieval
query_embedding = voyage.embed_query("What is covered?")
doc_embeddings = voyage.embed_documents(["Policy covers...", "Exclusions..."])

# Local embeddings (no API needed)
local = LocalEmbedder(
    model="all-MiniLM-L6-v2",  # Fast, 384 dimensions
    device="cpu",  # Or "cuda", "mps"
)
result = local.embed("Hello world")
print(f"Dimensions: {result.dimensions}")  # 384

# Cached embeddings (avoid redundant API calls)
cached = CachedEmbedder(
    OpenAIEmbedder(),
    cache_type="both",  # "memory", "disk", or "both"
    cache_path=".cache/embeddings.db",
    memory_size=10000,
)

# First call hits API
result1 = cached.embed("What is the deductible?")
# Second call uses cache (no API call)
result2 = cached.embed("What is the deductible?")

print(cached.cache_stats)
# {'hits': 1, 'misses': 1, 'hit_rate': 0.5, ...}
```

### Retrieval

```python
from pullcite.retrieval import (
    MemoryRetriever,
    ChromaRetriever,
    PgVectorRetriever,
)
from pullcite.embeddings import OpenAIEmbedder

embedder = OpenAIEmbedder()

# In-memory retriever (fast, no persistence)
retriever = MemoryRetriever(embedder)
retriever.index(document)

results = retriever.search("individual deductible", k=5)
print(f"Found {len(results)} results")

for r in results:
    print(f"  [{r.rank}] Score: {r.score:.3f}")
    print(f"      Page: {r.page}, Index: {r.index}")
    print(f"      Text: {r.text[:100]}...")

# Filter by score threshold
high_quality = results.above_threshold(0.7)

# Get just the text
context = "\n".join(results.texts)

# ChromaDB retriever (persistent, scalable)
chroma = ChromaRetriever(
    embedder,
    collection_name="insurance_docs",
    persist_directory="./chroma_db",
    distance_metric="cosine",  # "cosine", "l2", or "ip"
)
chroma.index(document)
results = chroma.search("copay amounts", k=10)

# PostgreSQL pgvector (production-ready)
pg = PgVectorRetriever(
    embedder,
    connection_string="postgresql://user:pass@localhost/pullcite",
    table_name="document_chunks",
    distance_metric="cosine",
)
pg.index(document)
results = pg.search("coverage limits", k=5)
pg.close()  # Clean up connection
```

### LLMs

```python
from pullcite.llms import (
    AnthropicLLM,
    OpenAILLM,
    Message,
    Tool,
    ToolCall,
    ToolExecutor,
)

# Anthropic Claude
llm = AnthropicLLM(
    api_key="...",  # Or use ANTHROPIC_API_KEY env var
    model="claude-sonnet-4-20250514",  # Default
)

# Simple completion
response = llm.complete([
    Message.system("You are a helpful assistant."),
    Message.user("What is 2 + 2?"),
])
print(response.content)  # "4"
print(f"Tokens: {response.total_tokens}")

# OpenAI GPT
gpt = OpenAILLM(
    api_key="...",  # Or use OPENAI_API_KEY env var
    model="gpt-4o",  # Default
)

response = gpt.complete(
    messages=[Message.user("Explain quantum computing in one sentence.")],
    temperature=0.7,
    max_tokens=100,
)

# Multi-turn conversation
messages = [
    Message.system("You are a math tutor."),
    Message.user("What is the derivative of x^2?"),
]
response = llm.complete(messages)

messages.append(Message.assistant(content=response.content))
messages.append(Message.user("What about x^3?"))
response = llm.complete(messages)

# Tool use
search_tool = Tool(
    name="search_document",
    description="Search the document for relevant information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    },
)

response = llm.complete(
    messages=[Message.user("Find information about the deductible")],
    tools=[search_tool],
)

if response.has_tool_calls:
    for tc in response.tool_calls:
        print(f"Tool: {tc.name}, Args: {tc.arguments}")

# Automatic tool execution loop
class DocumentSearchExecutor(ToolExecutor):
    def __init__(self, retriever):
        self.retriever = retriever

    def execute(self, tool_call: ToolCall) -> str:
        if tool_call.name == "search_document":
            results = self.retriever.search(tool_call.arguments["query"], k=3)
            return "\n".join(results.texts)
        return "Unknown tool"

executor = DocumentSearchExecutor(retriever)
final_response, history = llm.complete_with_tools(
    messages=[Message.user("What is the deductible amount?")],
    tools=[search_tool],
    tool_executor=executor,
    max_rounds=5,
)
print(final_response.content)
```

### Path Grammar

```python
from pullcite.core.paths import get, set_value, expand, exists

data = {
    "policy": {
        "holder": {"name": "John Doe", "age": 35},
        "coverage": {
            "deductible": {"individual": 500, "family": 1000},
        },
    },
    "services": [
        {"code": "PCP", "copay": 25},
        {"code": "SPECIALIST", "copay": 50},
        {"code": "ER", "copay": 150},
    ],
}

# Simple dot notation
name = get(data, "policy.holder.name")  # "John Doe"

# Nested access
deductible = get(data, "policy.coverage.deductible.individual")  # 500

# List index access
first_service = get(data, "services[0].copay")  # 25

# Key selector (finds item by code/id/name)
pcp_copay = get(data, "services[PCP].copay")  # 25
er_copay = get(data, "services[ER].copay")  # 150

# Check existence
exists(data, "policy.holder.name")  # True
exists(data, "policy.holder.email")  # False

# Wildcard expansion
all_copays = expand(data, "services[*].copay")
# ["services[0].copay", "services[1].copay", "services[2].copay"]

# Set values (returns new dict, original unchanged)
updated = set_value(data, "policy.holder.age", 36)
updated = set_value(data, "services[PCP].copay", 30)

# Create nested paths
new_data = set_value({}, "a.b.c.d", "value")
# {"a": {"b": {"c": {"d": "value"}}}}
```

### Evidence and Verification

```python
from pullcite.core.evidence import Evidence, VerificationResult, VerificationStatus
from pullcite.core.fields import (
    CriticalField,
    VerifierPolicy,
    Parsers,
    Comparators,
    INTEGER_POLICY,
    CURRENCY_POLICY,
)

# Define critical fields to verify
critical_fields = [
    CriticalField(
        path="deductible.individual",
        label="Individual Deductible",
        search_query="individual deductible amount",
        policy=CURRENCY_POLICY,  # Handles "$500" vs "500"
    ),
    CriticalField(
        path="deductible.family",
        label="Family Deductible",
        search_query="family deductible",
        policy=INTEGER_POLICY,
    ),
    CriticalField(
        path="copay.pcp",
        label="PCP Copay",
        search_query="primary care copay",
        required=True,  # Extraction fails if not verified
    ),
]

# Custom verification policy
custom_policy = VerifierPolicy(
    parser=Parsers.currency(),  # Parse "$1,500" -> 1500.0
    comparator=Comparators.numeric_tolerance(0.01),  # 1% tolerance
    patterns=[r"\$[\d,]+"],  # Regex patterns to match
)

# Evidence objects track verification
evidence = Evidence(
    path="deductible.individual",
    extracted_value=500,
    verified_value=500,
    quote="Individual Deductible: $500 per person",
    page=3,
    chunk_index=12,
    confidence=0.95,
)

print(evidence.is_match)  # True
print(evidence.quote)     # "Individual Deductible: $500 per person"
```

### Patching

```python
from pydantic import BaseModel
from pullcite.pipeline.patcher import Patcher, Patch, create_patches

class InsuranceData(BaseModel):
    deductible: int
    copay: int
    name: str

# Apply patches to fix extraction errors
patcher = Patcher(schema=InsuranceData, strict=True)

data = {"deductible": 500, "copay": 25, "name": "Basic Plan"}
patches = [
    Patch(
        path="deductible",
        old_value=500,
        new_value=750,
        reason="Verified value from page 2",
    ),
]

result = patcher.apply(data, patches)
print(result)  # {"deductible": 750, "copay": 25, "name": "Basic Plan"}

# Check results
print(patcher.successful_patches)  # [Patch(...)]
print(patcher.failed_patches)      # []

# Create patches from corrections dict
corrections = {"deductible": 750, "copay": 30}
original = {"deductible": 500, "copay": 25, "name": "Basic Plan"}
patches = create_patches(corrections, original, reason="Verification correction")
```

### Custom Extraction Strategies

The `DefaultStrategy` provides sensible default prompts, but you can customize them for your domain:

```python
from pullcite.pipeline.strategy import DefaultStrategy, StrategyContext

# Option 1: Full prompt override
# Use when you need complete control over the prompt
strategy = DefaultStrategy(
    extractor_prompt="""You are an expert at extracting health insurance data from SBC documents.

Your task is to extract structured data matching the provided schema.

IMPORTANT:
- Pay special attention to deductible and copay amounts
- Distinguish between in-network and out-of-network values
- Look for coverage tiers (Individual, Family, etc.)

Output valid JSON only."""
)

# Option 2: Append extra instructions to defaults
# Use when the default prompts are good but need domain context
strategy = DefaultStrategy(
    extra_instructions="""
Focus on:
- This is a PPO plan, so in-network/out-of-network distinction matters
- Deductibles are typically listed in the "Costs" section
- Mental health coverage may be under "Behavioral Health"
"""
)

# Option 3: Mix - override some prompts, augment others
strategy = DefaultStrategy(
    extractor_prompt="Custom extraction prompt...",  # Full override
    extra_instructions="Extra context for verifier and corrector",  # Appends to defaults
)

# Option 4: Override individual prompts
strategy = DefaultStrategy(
    verifier_prompt="""You are verifying extracted insurance values.

Search the document carefully. Insurance documents often have:
- Summary tables at the beginning
- Detailed coverage grids in the middle
- Exclusions and limitations at the end

For each value, find the exact quote that supports it.""",
)

# Use the strategy with StrategyContext
from pullcite.core.document import Document
from pydantic import BaseModel

class HealthPlan(BaseModel):
    deductible: float
    copay: float

doc = Document.from_file("sbc.pdf")
context = StrategyContext(
    document=doc,
    schema=HealthPlan,
    critical_fields=[],
)

# Get the customized prompts
extractor_prompt = strategy.build_extractor_prompt(context)
verifier_prompt = strategy.build_verifier_prompt(context)
corrector_prompt = strategy.build_corrector_prompt(context)
```

**When to use each option:**

| Scenario | Approach |
|----------|----------|
| Domain-specific extraction (insurance, legal, medical) | `extractor_prompt` override |
| Adding context without changing prompt structure | `extra_instructions` |
| Custom verification logic | `verifier_prompt` override |
| Tweaking correction behavior | `corrector_prompt` override |
| Production with known document types | Full custom strategy class |

### Full Extraction Pipeline

```python
from pydantic import BaseModel
from pullcite.core.document import Document
from pullcite.core.evidence import CriticalField
from pullcite.embeddings import OpenAIEmbedder, CachedEmbedder
from pullcite.retrieval import MemoryRetriever
from pullcite.llms import AnthropicLLM, Message
from pullcite.core.fields import CURRENCY_POLICY, YES_NO_POLICY

# 1. Define schema
class HealthPlan(BaseModel):
    plan_name: str | None = None
    deductible_individual: float | None = None
    deductible_family: float | None = None
    out_of_pocket_max: float | None = None
    pcp_copay: float | None = None
    specialist_copay: float | None = None
    er_copay: float | None = None
    covers_mental_health: bool | None = None

# 2. Define critical fields for verification
critical_fields = [
    CriticalField(
        path="deductible_individual",
        label="Individual Deductible",
        search_query="individual deductible annual",
        policy=CURRENCY_POLICY,
        required=True,
    ),
    CriticalField(
        path="deductible_family",
        label="Family Deductible",
        search_query="family deductible",
        policy=CURRENCY_POLICY,
    ),
    CriticalField(
        path="pcp_copay",
        label="PCP Visit Copay",
        search_query="primary care physician copay office visit",
        policy=CURRENCY_POLICY,
    ),
    CriticalField(
        path="covers_mental_health",
        label="Mental Health Coverage",
        search_query="mental health behavioral health coverage",
        policy=YES_NO_POLICY,
    ),
]

# 3. Set up components
embedder = CachedEmbedder(OpenAIEmbedder(), cache_type="disk")
retriever = MemoryRetriever(embedder)
llm = AnthropicLLM()

# 4. Load and index document
doc = Document.from_file("health_plan_sbc.pdf")
retriever.index(doc)

# 5. Extract initial data
extract_prompt = f"""
Extract health plan information from the document.
Return JSON matching this schema: {HealthPlan.model_json_schema()}

Document content:
{chr(10).join(chunk.text for chunk in doc.chunks[:20])}
"""

response = llm.complete([
    Message.system("Extract structured data from insurance documents. Return valid JSON only."),
    Message.user(extract_prompt),
])

import json
extracted = json.loads(response.content)
print("Extracted:", extracted)

# 6. Verify critical fields
verifications = []
for field in critical_fields:
    # Search for evidence
    results = retriever.search(field.search_query, k=5)
    context = "\n".join(results.texts)

    verify_prompt = f"""
    Verify the extracted value for "{field.label}".
    Extracted value: {extracted.get(field.path.replace('.', '_'))}

    Search results:
    {context}

    Return JSON: {{"verified_value": <value or null>, "quote": "<exact quote>", "confidence": <0-1>}}
    """

    verify_response = llm.complete([
        Message.system("Verify extracted values against source documents."),
        Message.user(verify_prompt),
    ])

    verification = json.loads(verify_response.content)
    verifications.append({
        "field": field.path,
        **verification,
    })
    print(f"Verified {field.label}: {verification}")

# 7. Apply corrections if needed
from pullcite.pipeline.patcher import Patcher, Patch

patches = []
for v in verifications:
    field_key = v["field"].replace(".", "_")
    if v["verified_value"] is not None and v["verified_value"] != extracted.get(field_key):
        patches.append(Patch(
            path=field_key,
            old_value=extracted.get(field_key),
            new_value=v["verified_value"],
            reason=f"Verified from: {v['quote'][:50]}...",
        ))

if patches:
    patcher = Patcher(schema=HealthPlan, strict=False)
    corrected = patcher.apply(extracted, patches)
    print("Corrected:", corrected)
else:
    print("No corrections needed")
```

---

## Why Pullcite exists

LLM extraction alone is not reliable enough for compliance-sensitive workflows (like insurance SBCs, underwriting docs, contracts, etc.).

Pullcite adds:
- **Determinism** where it matters (verification + tie-breakers)
- **Auditability** (evidence objects per field)
- **Safety** (patch-only correction; no "re-imagining" the full output)
- **Pluggability** (LLM providers, embedding providers, vector stores)

---

## Concepts (Mental Model)

### Document → Chunks
A `Document` is loaded from bytes/text and split into `Chunk`s.

A `Chunk` is immutable and carries:
- `text`
- `index` (monotonic)
- optional `page`, `bbox`
- stable `metadata`

Chunks do **not** contain embeddings.

### Embedder → Vectors
An `Embedder` converts text into vectors (`list[float]`). Providers:
- OpenAI (`text-embedding-3-small`, `text-embedding-3-large`)
- Voyage (`voyage-3`, `voyage-code-3`, `voyage-finance-2`)
- Local (sentence-transformers: `all-MiniLM-L6-v2`, `all-mpnet-base-v2`)

### Retriever → Search
A `Retriever` indexes chunk embeddings and searches them by similarity.

Search results return:
- matching `Chunk`
- similarity score
- rank
- provenance (document_id, query, distance metric)

Providers:
- `MemoryRetriever` - In-memory NumPy (fast, no persistence)
- `ChromaRetriever` - ChromaDB (persistent, scalable)
- `PgVectorRetriever` - PostgreSQL pgvector (production-ready)

### Extract → Verify → Correct (3-role pipeline)
1. **ExtractorRole**: LLM fills the target schema.
2. **VerifierRole**: LLM searches the document for each critical field and returns evidence.
3. **CorrectorRole**: LLM generates minimal `Patch` objects for mismatches (no re-extraction).

### Patching (safety mechanism)
A patch is: **(path, value)**.

Instead of regenerating the whole JSON (which can break other fields), Pullcite updates only the specific fields that were proven wrong, using deterministic paths.

---

## Evidence objects

Each verified field can return an `Evidence`:

* `value`: verified value
* `quote`: exact supporting text
* `page`: 1-indexed page number (PDF)
* `bbox`: bounding box for the quote (PDF)
* `confidence`: verifier confidence
* `verified`: True/False
* `verification_note`: why it failed (if it did)

This enables:

* compliance reporting
* UI highlights (click-to-highlight quote)
* auditable logs

---

## Path Grammar (critical to patching)

Pullcite identifies fields using dot paths with selectors.

Examples:

* `vendor.name`
* `items[0].price` (index selector, 0-based)
* `services[PCP_VISIT].coverage_by_tier.INN.copay` (key selector)
* `services[*].copay` (wildcard; reads/expand only)

Rules:

* `[0]` → index access (all digits)
* `[*]` → wildcard expansion
* `[PCP_VISIT]` / `[123ABC]` → key selector

Key selectors match list-of-dict items using priority fields:

1. `service_code`
2. `code`
3. `id`
4. `key`
5. `name`

Ambiguity:

* multiple matches → `AmbiguousPathError`
* zero matches → `PathNotFoundError`

Wildcard use:

* allowed only in `expand()` and `expand_with_values()`
* forbidden in `set()` and `delete()`

---

## Project Structure

```
pullcite/
├── __init__.py
├── core/
│   ├── chunk.py           # Chunk dataclass (immutable)
│   ├── document.py        # Document loaders + chunking
│   ├── extractor.py       # Extractor orchestrator
│   ├── config.py          # ExtractorConfig, Hooks
│   ├── paths.py           # Path parsing/get/set/expand
│   ├── evidence.py        # Evidence, VerificationResult
│   ├── fields.py          # CriticalField, policies, parsers
│   └── result.py          # ExtractionResult, ExtractionStats
├── embeddings/
│   ├── base.py            # Embedder ABC
│   ├── openai.py          # OpenAI text-embedding-3
│   ├── voyage.py          # Voyage AI embeddings
│   ├── local.py           # Sentence Transformers
│   └── cache.py           # CachedEmbedder, MemoryCache, DiskCache
├── retrieval/
│   ├── base.py            # Retriever ABC, SearchResult
│   ├── memory.py          # InMemoryRetriever (NumPy)
│   ├── chroma.py          # ChromaDB retriever
│   └── pgvector.py        # PostgreSQL pgvector retriever
├── llms/
│   ├── base.py            # LLM ABC + tool definitions
│   ├── anthropic.py       # Anthropic Claude
│   └── openai.py          # OpenAI GPT-4
└── pipeline/
    ├── strategy.py        # Prompt strategy interface
    ├── roles.py           # Extract / Verify / Correct roles
    └── patcher.py         # Patch application + validation
```

---

## Configuration

### Environment Variables

```bash
# LLM providers
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."

# Embedding providers
export VOYAGE_API_KEY="..."

# Database (for pgvector)
export DATABASE_URL="postgresql://user:pass@localhost/pullcite"
```

### Supported Models

**LLMs:**
- Anthropic: `claude-opus-4-5-20251101`, `claude-sonnet-4-20250514`, `claude-3-5-sonnet-20241022`, `claude-3-opus-20240229`
- OpenAI: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`

**Embeddings:**
- OpenAI: `text-embedding-3-small` (1536d), `text-embedding-3-large` (3072d)
- Voyage: `voyage-3` (1024d), `voyage-3-lite` (512d), `voyage-code-3`, `voyage-finance-2`
- Local: `all-MiniLM-L6-v2` (384d), `all-mpnet-base-v2` (768d)

---

## Development

```bash
# Clone and install
git clone https://github.com/usercando/pullcite
cd pullcite
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=pullcite --cov-report=term-missing

# Format code
black pullcite/ tests/

# Type checking
mypy pullcite/
```

### Development Notes

**Determinism:**
Pullcite aims to be reproducible:
* strict ambiguity errors for paths
* deterministic evidence selection tie-breakers
* default `temperature=0.0`

**Safety principles:**
* never invent missing values
* verification must cite evidence
* correction is patch-only (no re-extract)

**Intended use cases:**
* insurance SBC extraction + comparison
* compliance document extraction
* underwriting / contracts
* any "structured output + must-prove-it" workflow

---

## License

MIT

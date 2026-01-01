# Pullcite

> **Alpha** — API may change. Expect breaking changes until v1.0.

**Evidence-backed structured extraction from documents.**

Extract structured data from documents using LLMs while providing **proof of where each value came from** (quote + page + bounding box).

```python
from pullcite import Document, Extractor, ExtractionSchema, StringField, DecimalField, BM25Searcher
from pullcite.llms.anthropic import AnthropicLLM

class Invoice(ExtractionSchema):
    vendor = StringField(query="vendor company name", description="Company that issued the invoice")
    total = DecimalField(query="total amount due", description="Final amount due")

extractor = Extractor(schema=Invoice, llm=AnthropicLLM(), searcher=BM25Searcher())
result = extractor.extract(Document.from_file("invoice.pdf"))

print(result.data.vendor)                # "Acme Corp"
print(result.data.total)                 # Decimal("1500.00")
print(result.evidence_map["total"].quote) # "Grand Total: $1,500.00"
print(result.evidence_map["total"].page)  # 1
```

---

## Installation

```bash
pip install pullcite                 # Core
pip install pullcite[anthropic]      # + Claude
pip install pullcite[openai]         # + GPT
pip install pullcite[docling]        # + PDF/DOCX parsing
pip install pullcite[all]            # Everything
```

---

## Defining Schemas

Each field specifies a search query and description:

```python
from pullcite import ExtractionSchema, StringField, DecimalField, PercentField, BooleanField

class HealthPlan(ExtractionSchema):
    plan_name = StringField(
        query="plan name health plan title",
        description="Official name of the health insurance plan",
    )
    individual_deductible = DecimalField(
        query="individual deductible annual",
        description="Annual deductible for individual coverage (in-network)",
    )
    coinsurance = PercentField(
        query="coinsurance percentage member pays",
        description="Percentage the member pays after deductible (not plan's share)",
    )
    preventive_covered = BooleanField(
        query="preventive care covered",
        description="Whether preventive care is covered at no cost",
        required=False,
    )
```

### Field Types

| Type | Python | Parses |
|------|--------|--------|
| `StringField` | `str` | Text |
| `IntegerField` | `int` | `100`, `"100 days"` |
| `DecimalField` | `Decimal` | `"$1,500.00"` → `1500.00` |
| `PercentField` | `float` | `"30%"`, `0.30` → `30.0` |
| `BooleanField` | `bool` | `"yes"`, `"true"`, `1` → `True` |
| `DateField` | `str` | `"2024-01-15"` |
| `ListField` | `list` | `["a", "b"]` |
| `EnumField` | `str` | Must match choices |

### Descriptions Matter

Descriptions tell the LLM what each field means. Without them, extraction is less accurate:

```python
# Good - LLM understands context
total = DecimalField(query="total", description="Final invoice total including tax")

# Works but less accurate - LLM only sees field name "total"
total = DecimalField(query="total")
```

---

## Chunking

Chunking controls how documents are split for search. **Critical for extraction quality.**

```python
from pullcite import Document, SlidingWindowChunker, SentenceChunker

# Explicit chunking (recommended)
doc = Document.from_file("report.pdf", chunker=SlidingWindowChunker(size=500, stride=250))

# Sentence-aware chunking (better for prose)
doc = Document.from_file("contract.pdf", chunker=SentenceChunker(target_size=1000, overlap=200))

# Default: SentenceChunker(target_size=1200, overlap=200)
doc = Document.from_file("invoice.pdf")
```

| Chunker | Use Case |
|---------|----------|
| `SlidingWindowChunker(size, stride)` | Predictable, good default |
| `SentenceChunker(target_size, overlap)` | Prose documents |
| `ParagraphChunker(target_size, overlap_paragraphs)` | Well-structured docs |

---

## Extraction

```python
from pullcite import Extractor, BM25Searcher
from pullcite.llms.anthropic import AnthropicLLM

extractor = Extractor(
    schema=HealthPlan,
    llm=AnthropicLLM(),
    searcher=BM25Searcher(),
    top_k=5,         # Chunks per field (tune for your docs)
    verify=True,     # Verify against source (default)
)

result = extractor.extract(doc)
```

### top_k Parameter

Controls chunks retrieved per field. **Critical for quality.**

- `top_k=5` (default) - Good starting point
- Higher = more context, better accuracy, more tokens
- Lower = faster, cheaper, may miss context

### Async Support

```python
import asyncio

results = await asyncio.gather(*[
    extractor.extract_async(doc) for doc in documents
])
```

---

## Verification

Pullcite verifies extracted values against the source document:

1. **Search** - Retrieve chunks using field's query
2. **Parse** - Find values in chunk text via `field.parse_from_text()`
3. **Compare** - Check match via `field.compare()` (type-aware: tolerances for decimals, case-insensitive for strings)

```python
print(result.status)  # VERIFIED, PARTIAL, or FAILED

for vr in result.verification_results:
    print(f"{vr.path}: {vr.status.value}")
    # MATCH - Value verified in source
    # MISMATCH - Found different value
    # NOT_FOUND - Required field missing
    # SKIPPED - No context to verify
```

### Evidence

Every verified field has traceable evidence:

```python
evidence = result.evidence_map["total"]
print(evidence.quote)       # "Grand Total: $1,500.00"
print(evidence.page)        # 1
print(evidence.bbox)        # (72.0, 540.2, 200.5, 555.8)
print(evidence.confidence)  # 0.95
print(evidence.verified)    # True
```

---

## Complete Example

```python
from pullcite import (
    Document, Extractor, ExtractionSchema, BM25Searcher,
    StringField, DecimalField, PercentField, BooleanField,
    SentenceChunker,
)
from pullcite.llms.anthropic import AnthropicLLM


class HealthPlan(ExtractionSchema):
    plan_name = StringField(
        query="plan name health plan title",
        description="Official name of the health insurance plan",
    )
    plan_type = StringField(
        query="plan type HMO PPO EPO",
        description="Type of plan: HMO, PPO, EPO, or POS",
    )
    individual_deductible = DecimalField(
        query="individual deductible annual",
        description="Annual deductible for individual coverage (in-network)",
    )
    family_deductible = DecimalField(
        query="family deductible annual",
        description="Annual deductible for family coverage (in-network)",
    )
    coinsurance = PercentField(
        query="coinsurance percentage member pays",
        description="Percentage the member pays after deductible",
    )
    pcp_copay = DecimalField(
        query="primary care physician copay PCP",
        description="Copay for primary care visits",
    )
    preventive_covered = BooleanField(
        query="preventive care covered no cost",
        description="Whether preventive care is covered at 100%",
        required=False,
    )


# Load with explicit chunking
doc = Document.from_file(
    "summary_of_benefits.pdf",
    chunker=SentenceChunker(target_size=1000, overlap=200),
)

# Extract with custom instructions
extractor = Extractor(
    schema=HealthPlan,
    llm=AnthropicLLM(model="claude-sonnet-4-20250514"),
    searcher=BM25Searcher(),
    top_k=5,
    extra_instructions="""
    - Extract IN-NETWORK values when both in/out-of-network are shown
    - Coinsurance is what the MEMBER pays, not the plan
    - "No charge" or "Covered in full" means $0
    """,
)

result = extractor.extract(doc)

# Results
print(f"Plan: {result.data.plan_name} ({result.data.plan_type})")
print(f"Deductible: ${result.data.individual_deductible}")
print(f"Coinsurance: {result.data.coinsurance}%")
print(f"Status: {result.status}")

# Evidence
for field, evidence in result.evidence_map.items():
    print(f"{field}: \"{evidence.quote[:50]}...\" (page {evidence.page})")
```

---

## Search Types

Fields can use different search strategies:

```python
class MySchema(ExtractionSchema):
    # BM25: Keyword search (fast, no embeddings)
    invoice_number = StringField(
        query="invoice number invoice #",
        search_type=SearchType.BM25,
    )

    # Semantic: Vector similarity (requires embeddings)
    description = StringField(
        query="product service description",
        search_type=SearchType.SEMANTIC,
    )

    # Hybrid: BM25 + semantic with rank fusion
    vendor = StringField(
        query="vendor company supplier",
        search_type=SearchType.HYBRID,
    )
```

For semantic/hybrid, provide a retriever:

```python
from pullcite.embeddings.openai import OpenAIEmbedder
from pullcite.retrieval.memory import MemoryRetriever

extractor = Extractor(
    schema=MySchema,
    llm=my_llm,
    searcher=BM25Searcher(),
    retriever=MemoryRetriever(OpenAIEmbedder()),
)
```

---

## Custom Prompts

```python
# Append instructions to default prompt
extractor = Extractor(
    schema=Invoice,
    llm=my_llm,
    searcher=BM25Searcher(),
    extra_instructions="All amounts are in USD. Use Grand Total, not subtotals.",
)

# Replace entire system prompt
extractor = Extractor(
    schema=Invoice,
    llm=my_llm,
    searcher=BM25Searcher(),
    system_prompt="You are an expert invoice parser. Extract precisely.",
)

# Full control with custom builder
def my_prompt_builder(schema, field_contexts):
    lines = ["Extract these fields:"]
    for name, field in schema.get_fields().items():
        contexts = field_contexts.get(name, [])
        lines.append(f"\n## {name}: {field.description or ''}")
        if contexts:
            lines.append(f"Found in: {contexts[0].text[:200]}")
    return "\n".join(lines)

extractor = Extractor(..., prompt_builder=my_prompt_builder)
```

---

## Large Documents

For schemas with many fields or large documents:

```python
extractor = Extractor(
    schema=LargeSchema,
    llm=my_llm,
    searcher=BM25Searcher(),
    max_fields_per_batch=10,      # Split into multiple LLM calls
    max_context_chars=50000,      # Limit context per batch
    include_document_text=False,  # Use only retrieved chunks
    top_k=10,                     # More chunks per field
)
```

---

## LLM Providers

```python
from pullcite.llms.anthropic import AnthropicLLM
from pullcite.llms.openai import OpenAILLM

llm = AnthropicLLM(model="claude-sonnet-4-20250514")  # Uses ANTHROPIC_API_KEY
llm = OpenAILLM(model="gpt-4o")                       # Uses OPENAI_API_KEY
```

---

## Environment Variables

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export VOYAGE_API_KEY="..."
```

---

## Project Structure

```
pullcite/
├── core/
│   ├── document.py      # Document loading
│   ├── chunk.py         # Chunk dataclass
│   ├── chunker.py       # Chunking strategies
│   ├── evidence.py      # Evidence types
│   └── result.py        # ExtractionResult
├── schema/
│   ├── base.py          # ExtractionSchema, Field
│   ├── fields.py        # Field types
│   └── extractor.py     # Extractor
├── search/
│   ├── bm25.py          # BM25Searcher
│   └── hybrid.py        # HybridSearcher
├── embeddings/          # OpenAI, Voyage, local
├── retrieval/           # Memory, Chroma, pgvector
└── llms/                # Anthropic, OpenAI
```

---

## License

MIT

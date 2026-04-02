# D.R.I.V.E. Evaluator — Multi-Agent Instruction Manual

## Project Overview

The D.R.I.V.E. Evaluator is an NLP backend that scores candidate text (interview answers, essays) against five company values using a 4-stage **Semantic Funnel** pipeline. Raw text enters, and structured JSON evaluation results exit.

The five values:

| Code | Value | Description |
|------|-------|-------------|
| **D** | Disciplined Resilience | Emotional self-regulation, healthy habits, and determination |
| **R** | Responsible Innovation | Creative problem-solving, ethical use of technology, hypothesis testing, data-driven decision-making |
| **I** | Insightful Vision | Systems thinking, foresight, analysis, well-balanced judgment |
| **V** | Values-Driven Leadership | Dignity, inclusion, dialogue, learning through service |
| **E** | Entrepreneurial Execution | Opportunity seeking, partnerships, financial literacy, storytelling |

---

## System Architecture — The Semantic Funnel

```
Input Text
    │
    ▼
┌──────────────────────────────────────────────┐
│  Stage 1: CHUNKING                           │
│  Split raw text into sentences (spaCy).      │
│  Output: list[str]                           │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Stage 2: SEMANTIC FILTERING                 │
│  Embed each sentence + each value description│
│  via sentence-transformers. Compute cosine   │
│  similarity. Keep sentences above threshold. │
│  Output: list[ScoredChunk]                   │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Stage 3: LLM EVALUATION                    │
│  Send top-matching sentences + their matched │
│  value to an LLM API. The LLM confirms or   │
│  rejects the match and returns structured    │
│  reasoning (Pydantic model).                 │
│  Output: list[LLMVerdict]                    │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│  Stage 4: HIGHLIGHTING                       │
│  Run POS tagging (spaCy) on confirmed        │
│  sentences. Flag tokens by category:         │
│    - Action Verbs  (e.g. "led", "built")     │
│    - Adverbs       (e.g. "consistently")     │
│    - Assertive Nouns (e.g. "strategy")       │
│  Output: list[HighlightedSentence]           │
└──────────────────────────────────────────────┘
    │
    ▼
Final JSON Response
```

### Data Flow Types (defined in `app/schemas/`)

```
str  →  list[str]  →  list[ScoredChunk]  →  list[LLMVerdict]  →  list[HighlightedSentence]
       (Stage 1)       (Stage 2)              (Stage 3)              (Stage 4)
```

---

## Tech Stack & Dependencies

**Runtime:** Python 3.11+

**Core dependencies** (install via `pip install`):

| Package | Purpose |
|---------|---------|
| `fastapi` | HTTP API framework |
| `uvicorn[standard]` | ASGI server |
| `pydantic>=2.0` | Request/response validation, structured LLM output |
| `sentence-transformers` | Embedding model for semantic similarity |
| `spacy>=3.7` | Sentence segmentation and POS tagging |
| `httpx` | Async HTTP client for LLM API calls |
| `python-dotenv` | Environment variable loading |

**spaCy model** (download after install):

```bash
python -m spacy download en_core_web_sm
```

**Dev dependencies:**

| Package | Purpose |
|---------|---------|
| `pytest` | Testing |
| `pytest-asyncio` | Async test support |
| `ruff` | Linting and formatting |

### requirements.txt

```
fastapi>=0.110
uvicorn[standard]>=0.29
pydantic>=2.0
sentence-transformers>=2.6
spacy>=3.7
httpx>=0.27
python-dotenv>=1.0
pytest>=8.0
pytest-asyncio>=0.23
ruff>=0.4
```

---

## Directory Structure

```
decentra5.0/
├── CLAUDE.md                  # This file — project instructions
├── requirements.txt           # Pinned dependencies
├── .env                       # LLM API key (never commit)
├── .env.example               # Template for .env
├── app/
│   ├── __init__.py
│   ├── main.py                # FastAPI app factory, lifespan, middleware
│   ├── config.py              # Settings via pydantic-settings / dotenv
│   ├── values.py              # D.R.I.V.E. value definitions (code, name, description)
│   ├── routers/
│   │   ├── __init__.py
│   │   └── evaluate.py        # POST /evaluate endpoint
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requests.py        # EvaluateRequest
│   │   └── responses.py       # ScoredChunk, LLMVerdict, HighlightedSentence, EvaluateResponse
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── chunker.py         # Stage 1 — sentence splitting
│   │   ├── embedder.py        # Stage 2 — embedding + cosine similarity
│   │   ├── llm_evaluator.py   # Stage 3 — LLM confirmation
│   │   └── highlighter.py     # Stage 4 — POS tagging & token flagging
│   └── services/
│       ├── __init__.py
│       └── funnel.py          # Orchestrator — chains Stages 1-4
└── tests/
    ├── __init__.py
    ├── conftest.py             # Shared fixtures (test client, sample texts)
    ├── test_chunker.py
    ├── test_embedder.py
    ├── test_llm_evaluator.py
    ├── test_highlighter.py
    └── test_api.py
```

---

## Coding Standards

These rules are **mandatory** for every file produced by every agent.

### Python Style

- **Type hints on all function signatures** — parameters and return types. No `Any` unless unavoidable.
- **Pydantic models for all data boundaries** — requests, responses, inter-stage data. Never pass raw dicts between stages.
- **Async by default** — all I/O-bound functions (LLM calls, embedding inference) must be `async def`. CPU-bound work (POS tagging) can be sync but must be run via `asyncio.to_thread` if called from an async context.
- **Docstrings** — every public function and class gets a one-line summary + Args/Returns section in Google style.
- **No global mutable state** — models and clients are initialized in the FastAPI lifespan and passed via dependency injection or app state.

### Output Format

- **All API responses are JSON.** Use Pydantic's `.model_dump(mode="json")` for serialization.
- **No print statements.** Use `logging` with the module logger: `logger = logging.getLogger(__name__)`.

### Error Handling

- Raise `fastapi.HTTPException` with appropriate status codes for user-facing errors.
- Catch and log exceptions from external services (LLM API, model loading) — never let raw tracebacks leak to the client.

### Configuration

- All secrets and tunable parameters go in `.env` and are read through `app/config.py`.
- Key config values: `LLM_API_KEY`, `LLM_API_URL`, `LLM_MODEL`, `EMBEDDING_MODEL_NAME` (default `all-MiniLM-L6-v2`), `SIMILARITY_THRESHOLD` (default `0.45`), `SPACY_MODEL` (default `en_core_web_sm`).

### Testing

- Every pipeline stage gets its own test file with at least 2 test cases.
- Tests must be runnable with `pytest` from the project root with no extra setup beyond installing dependencies.
- Use fixtures in `conftest.py` for shared data (sample candidate texts, mock LLM responses).

### Formatting & Linting

- Format and lint with `ruff`. Run: `ruff check . && ruff format .`

---

## Parallel Agent Execution Plan

### Dependency Graph

```
                 ┌─────────────────────┐
                 │   PHASE 1 (serial)  │
                 │   Agent: Foundation  │
                 └────────┬────────────┘
                          │
            ┌─────────────┼─────────────────────┐
            │             │                     │
            ▼             ▼                     ▼
   ┌────────────┐  ┌─────────────┐  ┌────────────────────┐
   │  PHASE 2a  │  │  PHASE 2b   │  │     PHASE 2c       │
   │  Agent:    │  │  Agent:     │  │     Agent:          │
   │  Chunker   │  │  Embedder   │  │     LLM Evaluator   │
   └─────┬──────┘  └──────┬──────┘  └──────────┬─────────┘
         │                │                     │
         │         ┌──────┘                     │
         │         │  ┌─────────────────────────┘
         ▼         ▼  ▼
   ┌────────────────────┐
   │    PHASE 2d        │
   │    Agent:          │
   │    Highlighter     │
   │  (can also run in  │
   │   parallel — see   │
   │   note below)      │
   └────────┬───────────┘
            │
            ▼
   ┌────────────────────┐
   │   PHASE 3 (serial) │
   │   Agent: Integrator │
   └────────────────────┘
```

**Parallelism rules:**
- **Phase 1** must complete before any Phase 2 agent starts (they depend on schemas and config).
- **Phase 2a, 2b, 2c, 2d** are fully independent — they touch separate files and import only from `app/schemas/` and `app/values.py` (created in Phase 1). Run all four agents simultaneously.
- **Phase 3** must wait for all Phase 2 agents to finish (it imports and wires all pipeline stages).

---

### PHASE 1 — Agent: Foundation

> **Must complete before all other agents start.**

**Owns these files (creates all of them):**

| File | Purpose |
|------|---------|
| `requirements.txt` | Dependencies |
| `.env.example` | Config template |
| `app/__init__.py` | Package init |
| `app/config.py` | Pydantic `BaseSettings` |
| `app/values.py` | D.R.I.V.E. value definitions |
| `app/main.py` | FastAPI app with lifespan (skeleton — no routers yet) |
| `app/schemas/__init__.py` | Package init |
| `app/schemas/requests.py` | `EvaluateRequest` model |
| `app/schemas/responses.py` | All response/inter-stage models |
| `app/pipeline/__init__.py` | Package init (empty) |
| `app/routers/__init__.py` | Package init (empty) |
| `app/services/__init__.py` | Package init (empty) |
| `tests/__init__.py` | Package init |
| `tests/conftest.py` | Shared fixtures |

**Instructions:**

1. Create `requirements.txt` with the dependencies listed in the Tech Stack section.

2. Create `.env.example`:
   ```
   LLM_API_KEY=your-key-here
   LLM_API_URL=https://api.openai.com/v1/chat/completions
   LLM_MODEL=gpt-4o-mini
   EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
   SIMILARITY_THRESHOLD=0.45
   SPACY_MODEL=en_core_web_sm
   ```

3. Create `app/config.py` — a Pydantic `BaseSettings` class that reads all keys from `.env`. Fields:
   - `LLM_API_KEY: str`
   - `LLM_API_URL: str` (default `"https://api.openai.com/v1/chat/completions"`)
   - `LLM_MODEL: str` (default `"gpt-4o-mini"`)
   - `EMBEDDING_MODEL_NAME: str` (default `"all-MiniLM-L6-v2"`)
   - `SIMILARITY_THRESHOLD: float` (default `0.45`)
   - `SPACY_MODEL: str` (default `"en_core_web_sm"`)

4. Create `app/values.py` — define a `Value` Pydantic model with fields `code: str`, `name: str`, `description: str`. Create a module-level list `DRIVE_VALUES: list[Value]` containing all 5 values with their full descriptions.

5. Create `app/schemas/requests.py`:
   - `EvaluateRequest` — field: `text: str` (the candidate's full answer).

6. Create `app/schemas/responses.py` — all inter-stage and final response models:
   - `ScoredChunk` — `text: str`, `value_code: str`, `value_name: str`, `similarity_score: float`
   - `LLMVerdict` — `text: str`, `value_code: str`, `value_name: str`, `confirmed: bool`, `reasoning: str`
   - `HighlightedToken` — `token: str`, `pos_category: str` (one of `"ACTION_VERB"`, `"ADVERB"`, `"ASSERTIVE_NOUN"`)
   - `HighlightedSentence` — `text: str`, `value_code: str`, `value_name: str`, `reasoning: str`, `highlights: list[HighlightedToken]`
   - `EvaluateResponse` — `results: list[HighlightedSentence]`, `summary: dict[str, int]` (count of confirmed sentences per value code)

7. Create `app/main.py` — FastAPI app with:
   - An `asynccontextmanager` lifespan that loads the spaCy model and sentence-transformer model into `app.state.nlp` and `app.state.embedding_model`.
   - A `GET /health` endpoint returning `{"status": "ok"}`.
   - No routers wired yet (the Integrator agent will add them).

8. Create `tests/conftest.py` with shared fixtures:
   - `sample_text` — a multi-sentence paragraph that touches multiple D.R.I.V.E. values.
   - `nlp` — a loaded spaCy `en_core_web_sm` model.
   - `embedding_model` — a loaded `SentenceTransformer("all-MiniLM-L6-v2")`.
   - `test_client` — a FastAPI `TestClient` wrapping the app.

9. Create all empty `__init__.py` files for the packages.

**Verify:** `uvicorn app.main:app --reload` → `GET /health` returns `{"status": "ok"}`.

---

### PHASE 2a — Agent: Chunker

> **Runs in parallel with Agents 2b, 2c, 2d. Depends only on Phase 1.**

**Owns these files:**

| File | Purpose |
|------|---------|
| `app/pipeline/chunker.py` | Stage 1 implementation |
| `tests/test_chunker.py` | Stage 1 tests |

**Reads (do not modify):** `app/schemas/responses.py`, `app/values.py`, `tests/conftest.py`

**Instructions:**

1. In `app/pipeline/chunker.py`, write:
   ```python
   def chunk_text(text: str, nlp: spacy.Language) -> list[str]:
   ```
   - Use spaCy's sentence segmenter (`doc.sents`).
   - Strip whitespace from each sentence.
   - Drop empty strings.
   - Drop sentences with fewer than 5 words (noise filter).
   - Return the list of clean sentence strings.

2. In `tests/test_chunker.py`, write at least 2 tests:
   - **Test 1:** Pass a multi-sentence paragraph (3+ sentences). Assert the correct number of sentences are returned and content matches.
   - **Test 2:** Pass text containing a very short fragment (e.g. "Yes. I agree. I consistently regulated my emotions under extreme pressure."). Assert the short fragments are filtered out and only the valid sentence remains.

**Verify:** `pytest tests/test_chunker.py -v` — all tests pass.

---

### PHASE 2b — Agent: Embedder

> **Runs in parallel with Agents 2a, 2c, 2d. Depends only on Phase 1.**

**Owns these files:**

| File | Purpose |
|------|---------|
| `app/pipeline/embedder.py` | Stage 2 implementation |
| `tests/test_embedder.py` | Stage 2 tests |

**Reads (do not modify):** `app/schemas/responses.py`, `app/values.py`, `tests/conftest.py`

**Instructions:**

1. In `app/pipeline/embedder.py`, write:
   ```python
   async def filter_by_similarity(
       sentences: list[str],
       model: SentenceTransformer,
       threshold: float,
   ) -> list[ScoredChunk]:
   ```
   - Embed all input sentences in one batch call.
   - Embed the 5 D.R.I.V.E. value descriptions (cache these embeddings on first call using a module-level variable or `functools.lru_cache`).
   - Compute cosine similarity of each sentence against each value description.
   - For each sentence, keep only the highest-scoring value match.
   - Discard any pair below `threshold`.
   - Sort results by `similarity_score` descending.
   - Return `list[ScoredChunk]`.

2. In `tests/test_embedder.py`, write at least 2 tests:
   - **Test 1:** Pass the sentence `"I consistently regulated my emotions under pressure"`. Assert it maps to value code `"D"` (Disciplined Resilience).
   - **Test 2:** Pass a clearly irrelevant sentence like `"The weather is nice today"`. Assert it is filtered out (returns empty list or falls below threshold).

**Verify:** `pytest tests/test_embedder.py -v` — all tests pass.

---

### PHASE 2c — Agent: LLM Evaluator

> **Runs in parallel with Agents 2a, 2b, 2d. Depends only on Phase 1.**

**Owns these files:**

| File | Purpose |
|------|---------|
| `app/pipeline/llm_evaluator.py` | Stage 3 implementation |
| `tests/test_llm_evaluator.py` | Stage 3 tests |

**Reads (do not modify):** `app/schemas/responses.py`, `app/values.py`, `app/config.py`, `tests/conftest.py`

**Instructions:**

1. In `app/pipeline/llm_evaluator.py`, write:
   ```python
   async def evaluate_with_llm(
       chunks: list[ScoredChunk],
       config: Settings,
   ) -> list[LLMVerdict]:
   ```
   - For each `ScoredChunk`, build this prompt:
     ```
     You are evaluating whether a candidate's sentence demonstrates the company value "{value_name}: {value_description}".

     Sentence: "{text}"

     Does this sentence demonstrate the value? Respond with JSON only:
     {"confirmed": true/false, "reasoning": "one sentence explanation"}
     ```
   - Look up the full value description from `DRIVE_VALUES` using `chunk.value_code`.
   - Call the LLM API via `httpx.AsyncClient`:
     - POST to `config.LLM_API_URL`
     - Headers: `Authorization: Bearer {config.LLM_API_KEY}`, `Content-Type: application/json`
     - Body: standard chat completions format with `model: config.LLM_MODEL`
   - Parse the JSON from the LLM response content into `confirmed` and `reasoning`.
   - Construct `LLMVerdict` for each chunk.
   - Return only verdicts where `confirmed is True`.

2. In `tests/test_llm_evaluator.py`, write at least 2 tests:
   - **Test 1 (confirmed):** Mock the HTTP call to return `{"confirmed": true, "reasoning": "Shows emotional regulation"}`. Assert the verdict is included in results.
   - **Test 2 (rejected):** Mock the HTTP call to return `{"confirmed": false, "reasoning": "Unrelated to value"}`. Assert the verdict is excluded from results.
   - Use `httpx.MockTransport` or `unittest.mock.patch` to mock the API call. Do NOT make real HTTP requests in tests.

**Verify:** `pytest tests/test_llm_evaluator.py -v` — all tests pass.

---

### PHASE 2d — Agent: Highlighter

> **Runs in parallel with Agents 2a, 2b, 2c. Depends only on Phase 1.**

**Owns these files:**

| File | Purpose |
|------|---------|
| `app/pipeline/highlighter.py` | Stage 4 implementation |
| `tests/test_highlighter.py` | Stage 4 tests |

**Reads (do not modify):** `app/schemas/responses.py`, `tests/conftest.py`

**Instructions:**

1. In `app/pipeline/highlighter.py`, define:
   ```python
   ASSERTIVE_NOUNS: set[str] = {
       "strategy", "leadership", "initiative", "impact", "goal",
       "vision", "growth", "innovation", "resilience", "partnership",
       "decision", "opportunity", "integrity", "discipline", "insight",
       "execution", "accountability", "service", "collaboration",
       "determination",
   }
   ```

2. Write the main function:
   ```python
   def highlight_sentence(
       text: str,
       verdict: LLMVerdict,
       nlp: spacy.Language,
   ) -> HighlightedSentence:
   ```
   - Run spaCy POS tagging on `text`.
   - For each token, check:
     - `token.pos_ == "VERB"` → create `HighlightedToken(token=token.text, pos_category="ACTION_VERB")`
     - `token.pos_ == "ADV"` → create `HighlightedToken(token=token.text, pos_category="ADVERB")`
     - `token.pos_ == "NOUN" and token.lemma_.lower() in ASSERTIVE_NOUNS` → create `HighlightedToken(token=token.text, pos_category="ASSERTIVE_NOUN")`
   - Return `HighlightedSentence` with the collected highlights, plus `value_code`, `value_name`, and `reasoning` from the verdict.

3. In `tests/test_highlighter.py`, write at least 2 tests:
   - **Test 1:** Pass `"I consistently led the strategic initiative"`. Assert:
     - `"consistently"` is tagged `ADVERB`
     - `"led"` is tagged `ACTION_VERB`
     - `"initiative"` is tagged `ASSERTIVE_NOUN`
   - **Test 2:** Pass `"The cat sat on the mat"`. Assert no `ASSERTIVE_NOUN` highlights are returned (common nouns not in the curated set).

**Verify:** `pytest tests/test_highlighter.py -v` — all tests pass.

---

### PHASE 3 — Agent: Integrator

> **Must wait for ALL Phase 2 agents to complete.**

**Owns these files:**

| File | Purpose |
|------|---------|
| `app/services/funnel.py` | Pipeline orchestrator |
| `app/routers/evaluate.py` | API endpoint |
| `tests/test_api.py` | Integration tests |

**Modifies:** `app/main.py` (adds router import and wiring)

**Reads:** All `app/pipeline/*.py`, `app/schemas/*.py`, `app/config.py`, `app/values.py`

**Instructions:**

1. Create `app/services/funnel.py`:
   ```python
   async def run_funnel(text: str, app_state) -> EvaluateResponse:
   ```
   - **Stage 1:** Call `chunk_text(text, app_state.nlp)` → `list[str]`
   - **Stage 2:** Call `filter_by_similarity(sentences, app_state.embedding_model, config.SIMILARITY_THRESHOLD)` → `list[ScoredChunk]`
   - **Stage 3:** Call `evaluate_with_llm(scored_chunks, config)` → `list[LLMVerdict]`
   - **Stage 4:** For each confirmed verdict, call `highlight_sentence(verdict.text, verdict, app_state.nlp)` → `list[HighlightedSentence]`
   - Build `summary`: count confirmed sentences per `value_code` → `dict[str, int]`
   - Return `EvaluateResponse(results=highlighted_sentences, summary=summary)`

2. Create `app/routers/evaluate.py`:
   - Register `POST /evaluate` endpoint.
   - Accept `EvaluateRequest` as the request body.
   - Call `run_funnel(request.text, request.app.state)`.
   - Return `EvaluateResponse`.

3. Modify `app/main.py`:
   - Import the evaluate router: `from app.routers.evaluate import router as evaluate_router`
   - Add `app.include_router(evaluate_router)` after app creation.

4. Create `tests/test_api.py`:
   - **Test 1:** POST `/evaluate` with a sample paragraph. Assert response status is 200 and body matches `EvaluateResponse` structure (has `results` list and `summary` dict).
   - **Test 2:** POST `/evaluate` with empty or very short text. Assert graceful handling (empty results, not a 500 error).
   - Mock the LLM API call so tests don't require a real API key.

**Verify:**
1. `pytest -v` — all tests pass (all agents' tests + integration tests).
2. `ruff check . && ruff format .` — no lint issues.
3. Manual smoke test:
   ```bash
   curl -X POST http://localhost:8000/evaluate \
     -H "Content-Type: application/json" \
     -d '{"text": "I consistently pushed through obstacles and regulated my emotions to lead the team toward our strategic goals."}'
   ```
4. Verify the response contains highlighted sentences mapped to the correct D.R.I.V.E. values.

---

## Agent File Ownership Matrix

This matrix ensures no two agents write to the same file, preventing merge conflicts.

| File | Foundation | Chunker | Embedder | LLM Eval | Highlighter | Integrator |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| `requirements.txt` | **W** | | | | | |
| `.env.example` | **W** | | | | | |
| `app/config.py` | **W** | | | R | | R |
| `app/values.py` | **W** | R | R | R | | R |
| `app/main.py` | **W** | | | | | **M** |
| `app/schemas/requests.py` | **W** | | | | | R |
| `app/schemas/responses.py` | **W** | R | R | R | R | R |
| `app/pipeline/__init__.py` | **W** | | | | | |
| `app/pipeline/chunker.py` | | **W** | | | | R |
| `app/pipeline/embedder.py` | | | **W** | | | R |
| `app/pipeline/llm_evaluator.py` | | | | **W** | | R |
| `app/pipeline/highlighter.py` | | | | | **W** | R |
| `app/services/funnel.py` | | | | | | **W** |
| `app/routers/evaluate.py` | | | | | | **W** |
| `tests/conftest.py` | **W** | R | R | R | R | R |
| `tests/test_chunker.py` | | **W** | | | | |
| `tests/test_embedder.py` | | | **W** | | | |
| `tests/test_llm_evaluator.py` | | | | **W** | | |
| `tests/test_highlighter.py` | | | | | **W** | |
| `tests/test_api.py` | | | | | | **W** |

**Legend:** **W** = writes/creates, **M** = modifies, **R** = reads only

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `pip install -r requirements.txt` | Install dependencies |
| `python -m spacy download en_core_web_sm` | Download spaCy model |
| `uvicorn app.main:app --reload` | Run dev server |
| `pytest -v` | Run all tests |
| `pytest tests/test_chunker.py -v` | Run single agent's tests |
| `ruff check . && ruff format .` | Lint and format |

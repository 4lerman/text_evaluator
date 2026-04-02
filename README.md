# D.R.I.V.E. Evaluator

An NLP backend that scores candidate text (interview answers, essays) against five company values using a 4-stage **Semantic Funnel** pipeline.

## D.R.I.V.E. Values

| Code | Value | Description |
|------|-------|-------------|
| **D** | Disciplined Resilience | Emotional self-regulation, healthy habits, determination |
| **R** | Responsible Innovation | Creative problem-solving, ethical technology use |
| **I** | Insightful Vision | Systems thinking, foresight, analysis |
| **V** | Values-Driven Leadership | Dignity, inclusion, dialogue, service |
| **E** | Entrepreneurial Execution | Opportunity seeking, partnerships, storytelling |

## Supported Languages

🇷🇺 Russian · 🇰🇿 Kazakh · 🇬🇧 English

## Setup

### 1. Clone & create virtualenv

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download spaCy multilingual model

```bash
python -m spacy download xx_sent_ud_sm
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and set your LLM_API_KEY
```

### 5. Run the server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## API

### `POST /evaluate`

Evaluates candidate text against the D.R.I.V.E. values.

**Request:**
```json
{
  "text": "Мен қиын жағдайда өз эмоцияларымды реттей отырып, команданы стратегиялық мақсатқа жеткіздім."
}
```

**Response:**
```json
{
  "results": [
    {
      "text": "...",
      "value_code": "D",
      "value_name": "Disciplined Resilience / ...",
      "reasoning": "Shows emotional self-regulation under pressure",
      "highlights": [
        { "token": "реттей", "pos_category": "ACTION_VERB" }
      ]
    }
  ],
  "summary": { "D": 1 }
}
```

### `GET /health`

```json
{ "status": "ok" }
```

## Architecture — Semantic Funnel

```
Input Text
    │
    ▼
Stage 1: CHUNKING         — spaCy sentence segmentation (xx_sent_ud_sm)
    │
    ▼
Stage 2: SEMANTIC FILTER  — paraphrase-multilingual-MiniLM-L12-v2 embeddings
    │                        + cosine similarity vs value descriptions (EN+RU+KZ)
    ▼
Stage 3: LLM EVALUATION   — GPT-4o-mini confirms/rejects matched sentences
    │
    ▼
Stage 4: HIGHLIGHTING     — spaCy POS tagging, assertive noun detection
    │
    ▼
JSON Response
```

## Development

```bash
# Run all tests
pytest -v

# Lint and format
ruff check . && ruff format .
```

## Configuration

All settings are controlled via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | — | Your OpenAI (or compatible) API key |
| `LLM_API_URL` | OpenAI chat completions | LLM endpoint |
| `LLM_MODEL` | `gpt-4o-mini` | Model name |
| `EMBEDDING_MODEL_NAME` | `paraphrase-multilingual-MiniLM-L12-v2` | Sentence embedding model |
| `SIMILARITY_THRESHOLD` | `0.45` | Min cosine similarity to pass Stage 2 |
| `SPACY_MODEL` | `xx_sent_ud_sm` | spaCy model (multilingual: RU + KZ) |

from pydantic import BaseModel

class ScoredChunk(BaseModel):
    """A sentence chunk scored against a D.R.I.V.E. value."""
    text: str
    value_code: str
    value_name: str
    similarity_score: float

class LLMVerdict(BaseModel):
    """The LLM's evaluation of a matched sentence."""
    text: str
    value_code: str
    value_name: str
    confirmed: bool
    reasoning: str
    score: int = 0               # ← new: 1–5 from Prometheus
    evidence_quote: str | None = None

class HighlightedToken(BaseModel):
    """A specific token highlighted within a confirmed sentence."""
    token: str
    pos_category: str
    start: int  # character offset of token start within the sentence
    end: int    # character offset of token end (exclusive)

class HighlightedSentence(BaseModel):
    """A fully processed sentence ready to be returned to the client."""
    text: str
    value_code: str
    value_name: str
    reasoning: str
    score: int = 0
    highlights: list[HighlightedToken]

class Metrics(BaseModel):
    """Aggregate metrics across all confirmed sentences."""
    overall_score: float
    coverage: int           # number of distinct values with score >= 3
    balance_score: float    # 1 - (stdev / 2.19), range 0–1
    strongest: list[str]    # value codes with highest mean score
    weakest: list[str]      # value codes with lowest mean score

class EvaluateResponse(BaseModel):
    """The final API response with all evaluations and a summary."""
    results: list[HighlightedSentence]
    summary: dict[str, int]
    metrics: Metrics | None = None
    lang: str = "en"

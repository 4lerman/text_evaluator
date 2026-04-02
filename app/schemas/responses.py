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
    highlights: list[HighlightedToken]

class EvaluateResponse(BaseModel):
    """The final API response with all evaluations and a summary."""
    results: list[HighlightedSentence]
    summary: dict[str, int]

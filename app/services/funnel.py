import asyncio
import logging
import statistics
from collections import defaultdict

from langdetect import detect, LangDetectException

from app.config import config
from app.schemas.responses import EvaluateResponse, HighlightedSentence, Metrics
from app.pipeline.chunker import chunk_text
from app.pipeline.embedder import filter_by_similarity
from app.pipeline.llm_evaluator import evaluate_with_llm
from app.pipeline.highlighter import highlight_sentence
from app.values import CYRILLIC_LANGS

logger = logging.getLogger(__name__)


def _compute_metrics(verdicts) -> Metrics:
    """Compute aggregate DRIVE metrics from confirmed verdicts.

    Args:
        verdicts: List of confirmed LLMVerdict objects with score fields.

    Returns:
        Metrics with overall_score, coverage, balance_score, strongest, weakest.
    """
    # Aggregate scores per value code (take max if same value appears multiple times)
    value_scores: dict[str, list[int]] = defaultdict(list)
    for v in verdicts:
        value_scores[v.value_code].append(v.score)

    mean_per_value = {code: statistics.mean(scores) for code, scores in value_scores.items()}
    values = list(mean_per_value.values())

    overall = round(statistics.mean(values), 2)
    coverage = sum(1 for s in values if s >= 3)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    balance = round(max(0.0, 1.0 - (std / 2.19)), 2)

    max_score = max(values)
    min_score = min(values)
    strongest = [c for c, s in mean_per_value.items() if s == max_score]
    weakest = [c for c, s in mean_per_value.items() if s == min_score]

    return Metrics(
        overall_score=overall,
        coverage=coverage,
        balance_score=balance,
        strongest=strongest,
        weakest=weakest,
    )


def _detect_lang(text: str) -> str:
    """Detect ISO 639-1 language code from *text*, defaulting to 'en'.

    Args:
        text: Raw input text.

    Returns:
        Two-letter language code string.
    """
    try:
        return detect(text)
    except LangDetectException:
        return "en"


async def run_funnel(text: str, app_state) -> EvaluateResponse:
    """Orchestrate the 4-stage semantic evaluation pipeline.

    Args:
        text: Candidate text to evaluate.
        app_state: FastAPI app state containing loaded NLP models.

    Returns:
        EvaluateResponse with highlighted sentences and per-value summary.
    """
    # Detect language once — used for description routing (Stage 2) and
    # POS model selection (Stage 4).
    lang = _detect_lang(text)
    logger.info(f"Detected language: {lang!r}")

    # Stage 1: Chunking
    sentences = chunk_text(text, app_state.nlp)
    if not sentences:
        return EvaluateResponse(results=[], summary={}, lang=lang)

    # Stage 2: Semantic Filtering with language-routed descriptions + z-score
    scored_chunks = await asyncio.wait_for(
        filter_by_similarity(
            sentences,
            app_state.embedding_model,
            lang=lang,
            raw_floor=config.SIMILARITY_THRESHOLD,
            z_threshold=config.Z_SCORE_THRESHOLD,
            competitor_margin=config.COMPETITOR_MARGIN,
        ),
        timeout=config.EMBEDDING_TIMEOUT_SECONDS,
    )
    if not scored_chunks:
        return EvaluateResponse(results=[], summary={}, lang=lang)

    # Stage 3: LLM Evaluation
    verdicts = await evaluate_with_llm(scored_chunks, config)
    if not verdicts:
        return EvaluateResponse(results=[], summary={}, lang=lang)

    # Stage 4: POS Highlighting — route to Russian model for Cyrillic input.
    # All sentences are highlighted in a single background thread to avoid
    # concurrent spaCy access across many threads under load.
    pos_nlp = app_state.ru_pos_nlp if lang in CYRILLIC_LANGS else app_state.pos_nlp

    def _highlight_all() -> list:
        return [highlight_sentence(v.text, v, pos_nlp) for v in verdicts]

    try:
        highlighted_sentences = await asyncio.wait_for(
            asyncio.to_thread(_highlight_all),
            timeout=config.HIGHLIGHT_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("Stage 4 highlighting timed out; returning results without highlights")
        highlighted_sentences = [
            HighlightedSentence(
                text=v.text,
                value_code=v.value_code,
                value_name=v.value_name,
                reasoning=v.reasoning,
                score=v.score,
                highlights=[],
            )
            for v in verdicts
        ]
    summary_counts: dict[str, int] = defaultdict(int)
    for verdict in verdicts:
        summary_counts[verdict.value_code] += 1

    return EvaluateResponse(
        results=highlighted_sentences,
        summary=dict(summary_counts),
        metrics=_compute_metrics(verdicts),
        lang=lang,
    )

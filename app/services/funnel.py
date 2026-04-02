import asyncio
import logging
from collections import defaultdict

from langdetect import detect, LangDetectException

from app.config import config
from app.schemas.responses import EvaluateResponse
from app.pipeline.chunker import chunk_text
from app.pipeline.embedder import filter_by_similarity
from app.pipeline.llm_evaluator import evaluate_with_llm
from app.pipeline.highlighter import highlight_sentence
from app.values import CYRILLIC_LANGS

logger = logging.getLogger(__name__)


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
        return EvaluateResponse(results=[], summary={})

    # Stage 2: Semantic Filtering with language-routed descriptions + z-score
    scored_chunks = await filter_by_similarity(
        sentences,
        app_state.embedding_model,
        lang=lang,
        raw_floor=config.SIMILARITY_THRESHOLD,
        z_threshold=config.Z_SCORE_THRESHOLD,
        competitor_margin=config.COMPETITOR_MARGIN,
    )
    if not scored_chunks:
        return EvaluateResponse(results=[], summary={})

    # Stage 3: LLM Evaluation
    verdicts = await evaluate_with_llm(scored_chunks, config)
    if not verdicts:
        return EvaluateResponse(results=[], summary={})

    # Stage 4: POS Highlighting — route to Russian model for Cyrillic input
    pos_nlp = app_state.ru_pos_nlp if lang in CYRILLIC_LANGS else app_state.pos_nlp

    highlighted_sentences = []
    summary_counts = defaultdict(int)
    for verdict in verdicts:
        hs = await asyncio.to_thread(highlight_sentence, verdict.text, verdict, pos_nlp)
        highlighted_sentences.append(hs)
        summary_counts[verdict.value_code] += 1

    return EvaluateResponse(
        results=highlighted_sentences,
        summary=dict(summary_counts),
    )

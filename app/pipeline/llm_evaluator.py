import asyncio
import logging
from prometheus_eval import PrometheusEval
from prometheus_eval.litellm import LiteLLM

from app.schemas.responses import ScoredChunk, LLMVerdict
from app.config import Settings
from app.values import DRIVE_VALUES

logger = logging.getLogger(__name__)

# Global evaluator instance, initialized on first use or module load.
# We use LiteLLM to interface with Prometheus (running via Ollama).
_evaluator = None


def get_evaluator(model_name: str) -> PrometheusEval:
    """Lazy initialization of the Prometheus evaluator."""
    global _evaluator
    if _evaluator is None:
        logger.info(f"Initializing Prometheus evaluator with model: {model_name}")
        model = LiteLLM(name=model_name)
        _evaluator = PrometheusEval(model=model)
    return _evaluator


async def _evaluate_chunk(
    chunk: ScoredChunk,
    evaluator: PrometheusEval,
    config: Settings,
) -> LLMVerdict | None:
    """Evaluate a single chunk with Prometheus. Returns None on failure or rejection.

    Args:
        chunk: The scored chunk to evaluate.
        evaluator: Initialized PrometheusEval instance.
        config: The application config.

    Returns:
        LLMVerdict if confirmed, None otherwise.
    """
    value_def = next((v for v in DRIVE_VALUES if v.code == chunk.value_code), None)
    if not value_def:
        logger.warning("Value code %s not found in DRIVE_VALUES.", chunk.value_code)
        return None

    try:
        feedback, score = await asyncio.to_thread(
            evaluator.single_absolute_grade,
            instruction=value_def.instruction,
            response=chunk.text,
            rubric=value_def.rubric,
            reference_answer=value_def.reference_answer,
        )
        if score >= config.CONFIRMATION_THRESHOLD:
            logger.info(
                "Confirmed match for %s (score: %d): %s...",
                chunk.value_code, score, chunk.text[:50],
            )
            return LLMVerdict(
                text=chunk.text,
                value_code=chunk.value_code,
                value_name=chunk.value_name,
                confirmed=True,
                reasoning=feedback.strip(),
                score=score,
                evidence_quote=None,
            )
        logger.debug(
            "Rejected match for %s (score: %d): %s...",
            chunk.value_code, score, chunk.text[:50],
        )
    except Exception as exc:
        logger.error("Error during Prometheus evaluation for chunk: %s", exc)

    return None


async def evaluate_with_llm(
    chunks: list[ScoredChunk],
    config: Settings,
) -> list[LLMVerdict]:
    """Evaluate chunks using Prometheus concurrently with a 1–5 rubric.

    Args:
        chunks: The chunks that passed semantic filtering.
        config: The application config.

    Returns:
        Only the confirmed LLMVerdicts (score >= threshold), sorted by score descending.
    """
    if not chunks:
        return []

    evaluator = get_evaluator(config.PROMETHEUS_MODEL)
    results = await asyncio.gather(
        *[_evaluate_chunk(chunk, evaluator, config) for chunk in chunks]
    )
    return sorted(
        [v for v in results if v is not None],
        key=lambda v: v.score,
        reverse=True,
    )

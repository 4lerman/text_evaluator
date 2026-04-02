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


async def evaluate_with_llm(
    chunks: list[ScoredChunk],
    config: Settings,
) -> list[LLMVerdict]:
    """Evaluate chunks using Prometheus to confirm the match with a 1–5 rubric.

    Args:
        chunks: The chunks that passed semantic filtering.
        config: The application config.

    Returns:
        Only the confirmed LLMVerdicts (score >= threshold).
    """
    if not chunks:
        return []

    evaluator = get_evaluator(config.PROMETHEUS_MODEL)
    verdicts = []

    for chunk in chunks:
        value_def = next(
            (v for v in DRIVE_VALUES if v.code == chunk.value_code), None
        )
        if not value_def:
            logger.warning(
                "Value code %s not found in DRIVE_VALUES.", chunk.value_code
            )
            continue

        try:
            # single_absolute_grade is synchronous, so we run it in a thread to avoid blocking.
            # It returns a tuple of (feedback, score).
            result = await asyncio.to_thread(
                evaluator.single_absolute_grade,
                instruction=value_def.instruction,
                response=chunk.text,
                rubric=value_def.rubric,
                reference_answer=value_def.reference_answer,
            )
            
            feedback, score = result
            confirmed = score >= config.CONFIRMATION_THRESHOLD

            verdict = LLMVerdict(
                text=chunk.text,
                value_code=chunk.value_code,
                value_name=chunk.value_name,
                confirmed=confirmed,
                reasoning=feedback.strip(),
                score=score,
                evidence_quote=None,  # Prometheus doesn't return exact substrings by default
            )

            if verdict.confirmed:
                verdicts.append(verdict)
                logger.info(
                    f"Confirmed match for {chunk.value_code} (score: {score}): {chunk.text[:50]}..."
                )
            else:
                logger.debug(
                    f"Rejected match for {chunk.value_code} (score: {score}): {chunk.text[:50]}..."
                )

        except Exception as exc:
            logger.error(f"Error during Prometheus evaluation for chunk: {exc}")
            continue

    return verdicts

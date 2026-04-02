import asyncio
import logging
from typing import Optional

from prometheus_eval import PrometheusEval
from prometheus_eval.litellm import LiteLLM

from app.schemas.responses import ScoredChunk, LLMVerdict
from app.config import Settings
from app.values import DRIVE_VALUES

logger = logging.getLogger(__name__)

# Global evaluator instance, initialized on first use or module load.
# We use LiteLLM to interface with Prometheus (running via Ollama).
_evaluator: Optional[PrometheusEval] = None

# Global semaphore: serializes all Prometheus/Ollama calls across concurrent
# HTTP requests. Ollama queues internally but concurrent requests cause
# timeouts; one-at-a-time is the correct model.
_prometheus_sem: Optional[asyncio.Semaphore] = None


def _get_semaphore(concurrency: int) -> asyncio.Semaphore:
    """Return (creating once) the global Prometheus semaphore."""
    global _prometheus_sem
    if _prometheus_sem is None:
        _prometheus_sem = asyncio.Semaphore(concurrency)
    return _prometheus_sem


def get_evaluator(model_name: str) -> PrometheusEval:
    """Lazy initialization of the Prometheus evaluator."""
    global _evaluator
    if _evaluator is None:
        logger.info(f"Initializing Prometheus evaluator with model: {model_name}")
        model = LiteLLM(name=model_name)
        _evaluator = PrometheusEval(model=model)
    return _evaluator


_MAX_RETRIES = 2
_RETRY_BASE_DELAY = 2.0  # seconds; doubles each attempt


async def _evaluate_chunk(
    chunk: ScoredChunk,
    evaluator: PrometheusEval,
    config: Settings,
) -> Optional[LLMVerdict]:
    """Evaluate a single chunk with Prometheus, retrying on transient failures.

    Args:
        chunk: The scored chunk to evaluate.
        evaluator: Initialized PrometheusEval instance.
        config: The application config.

    Returns:
        LLMVerdict if confirmed, None if rejected or all retries exhausted.
    """
    value_def = next((v for v in DRIVE_VALUES if v.code == chunk.value_code), None)
    if not value_def:
        logger.warning("Value code %s not found in DRIVE_VALUES.", chunk.value_code)
        return None

    sem = _get_semaphore(config.PROMETHEUS_GLOBAL_CONCURRENCY)

    for attempt in range(_MAX_RETRIES + 1):
        try:
            async with sem:
                feedback, score = await asyncio.wait_for(
                    asyncio.to_thread(
                        evaluator.single_absolute_grade,
                        instruction=value_def.instruction,
                        response=chunk.text,
                        rubric=value_def.rubric,
                        reference_answer=value_def.reference_answer,
                    ),
                    timeout=config.PROMETHEUS_TIMEOUT_SECONDS,
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
            return None  # deliberate rejection — no retry needed

        except asyncio.TimeoutError:
            exc_msg = f"timed out after {config.PROMETHEUS_TIMEOUT_SECONDS}s"
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Prometheus call %s (attempt %d/%d), retrying in %.0fs",
                    exc_msg, attempt + 1, _MAX_RETRIES + 1, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "Prometheus call %s after %d attempts for chunk [%s]",
                    exc_msg, _MAX_RETRIES + 1, chunk.value_code,
                )
        except Exception as exc:
            if attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "Prometheus call failed (attempt %d/%d), retrying in %.0fs: %s",
                    attempt + 1, _MAX_RETRIES + 1, delay, exc,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "Prometheus call failed after %d attempts for chunk [%s]: %s",
                    _MAX_RETRIES + 1, chunk.value_code, exc,
                )

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
    # Sequential — Ollama processes one inference at a time regardless of concurrency.
    # Concurrent requests just queue and time out; sequential avoids that failure mode.
    verdicts = []
    for chunk in chunks:
        result = await _evaluate_chunk(chunk, evaluator, config)
        if result is not None:
            verdicts.append(result)

    if not verdicts:
        logger.warning(
            "Stage 3: all %d chunk(s) were rejected or failed. "
            "Check Ollama is running and CONFIRMATION_THRESHOLD (%d) is not too high.",
            len(chunks), config.CONFIRMATION_THRESHOLD,
        )

    return sorted(verdicts, key=lambda v: v.score, reverse=True)

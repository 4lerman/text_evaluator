import asyncio
import logging
from collections import defaultdict
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
    sem: asyncio.Semaphore,
) -> Optional[LLMVerdict]:
    """Evaluate a single chunk with Prometheus, retrying on transient failures.

    Args:
        chunk: The scored chunk to evaluate.
        evaluator: Initialized PrometheusEval instance.
        config: The application config.
        sem: Semaphore that serializes Ollama calls within this request.

    Returns:
        LLMVerdict if confirmed, None if rejected or all retries exhausted.
    """
    value_def = next((v for v in DRIVE_VALUES if v.code == chunk.value_code), None)
    if not value_def:
        logger.warning("Value code %s not found in DRIVE_VALUES.", chunk.value_code)
        return None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            # Use context window (sentence + neighbors) so Prometheus can see the
            # surrounding narrative. Falls back to the bare sentence if context
            # was not populated (e.g. single-sentence input).
            response_text = chunk.context if chunk.context else chunk.text
            async with sem:
                feedback, score = await asyncio.wait_for(
                    asyncio.to_thread(
                        evaluator.single_absolute_grade,
                        instruction=value_def.instruction,
                        response=response_text,
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
            logger.info(
                "Rejected match for %s (score: %d, threshold: %d): %s...",
                chunk.value_code, score, config.CONFIRMATION_THRESHOLD, chunk.text[:50],
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
    total_sentences: Optional[int] = None,
) -> list[LLMVerdict]:
    """Evaluate chunks using Prometheus concurrently with a 1–5 rubric.

    Args:
        chunks: The chunks that passed semantic filtering.
        config: The application config.
        total_sentences: Total count of sentences in the original document.
            Allows position-aware selection to correctly split the document.

    Returns:
        Only the confirmed LLMVerdicts (score >= threshold), sorted by score descending.
    """
    if not chunks:
        return []

    # Chunks arrive sorted by similarity_score (best first). Before capping,
    # distribute slots using position-aware selection: for each value, pick
    # the best chunk from the first half AND second half of the text. This
    # prevents later, more explicitly-worded sentences from monopolizing all
    # slots while earlier sentences (which may demonstrate values in context)
    # get zero Prometheus coverage.
    if len(chunks) > config.PROMETHEUS_MAX_CHUNKS:
        per_value: dict[str, list[ScoredChunk]] = defaultdict(list)
        for chunk in chunks:
            per_value[chunk.value_code].append(chunk)

        # Find the midpoint of sentence positions. We prefer the actual
        # document midpoint if provided; otherwise, we fall back to the
        # median of the filtered indices.
        if total_sentences is not None:
            mid_index = total_sentences // 2
        else:
            all_indices = sorted({c.sentence_index for c in chunks})
            mid_index = all_indices[len(all_indices) // 2] if all_indices else 0

        selected: list[ScoredChunk] = []
        seen_keys: set[tuple[str, str]] = set()  # (text, value_code)

        # Phase 1: best chunk per value from EACH text half
        for code, value_chunks in per_value.items():
            first_half = sorted(
                [c for c in value_chunks if c.sentence_index < mid_index],
                key=lambda x: x.similarity_score, reverse=True,
            )
            second_half = sorted(
                [c for c in value_chunks if c.sentence_index >= mid_index],
                key=lambda x: x.similarity_score, reverse=True,
            )
            for pool in (second_half, first_half):
                if pool:
                    c = pool[0]
                    key = (c.text, c.value_code)
                    if key not in seen_keys:
                        selected.append(c)
                        seen_keys.add(key)

        # Phase 2: fill remaining slots with highest-similarity unchosen chunks
        remaining = sorted(
            [c for c in chunks if (c.text, c.value_code) not in seen_keys],
            key=lambda x: x.similarity_score, reverse=True,
        )
        for c in remaining:
            if len(selected) >= config.PROMETHEUS_MAX_CHUNKS:
                break
            selected.append(c)

        n_before = len(chunks)
        chunks = sorted(selected, key=lambda x: x.similarity_score, reverse=True)[
            : config.PROMETHEUS_MAX_CHUNKS
        ]
        first_half_count = sum(1 for c in chunks if c.sentence_index < mid_index)
        logger.info(
            "Stage 3: selected %d chunk(s) from %d candidates "
            "(first-half=%d, second-half=%d, PROMETHEUS_MAX_CHUNKS=%d)",
            len(chunks), n_before,
            first_half_count, len(chunks) - first_half_count,
            config.PROMETHEUS_MAX_CHUNKS,
        )

    evaluator = get_evaluator(config.PROMETHEUS_MODEL)
    # One semaphore per top-level evaluate_with_llm call, bound to the current
    # event loop. This avoids stale-loop issues with module-level semaphores
    # while still serializing Ollama calls within a single request.
    sem = asyncio.Semaphore(config.PROMETHEUS_GLOBAL_CONCURRENCY)

    verdicts = []
    for chunk in chunks:
        result = await _evaluate_chunk(chunk, evaluator, config, sem)
        if result is not None:
            verdicts.append(result)

    if not verdicts:
        logger.warning(
            "Stage 3: all %d chunk(s) were rejected or failed. "
            "Check Ollama is running and CONFIRMATION_THRESHOLD (%d) is not too high.",
            len(chunks), config.CONFIRMATION_THRESHOLD,
        )

    return sorted(verdicts, key=lambda v: v.score, reverse=True)

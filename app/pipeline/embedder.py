import asyncio
import logging

from sentence_transformers import SentenceTransformer, util

from app.schemas.responses import ScoredChunk
from app.values import DRIVE_VALUES

logger = logging.getLogger(__name__)

# Cache: (model object id, lang) → value-description embeddings tensor.
# Keyed by object identity so each model instance computes embeddings once
# per language across the process lifespan.
_value_embedding_cache: dict[tuple[int, str], object] = {}


def _get_value_embeddings(model: SentenceTransformer, lang: str):
    """Return cached passage embeddings for value descriptions in *lang*.

    Args:
        model: Loaded SentenceTransformer instance.
        lang: ISO 639-1 language code; routes to the matching description field.

    Returns:
        Tensor of shape (len(DRIVE_VALUES), embedding_dim).
    """
    key = (id(model), lang)
    if key not in _value_embedding_cache:
        # "passage: " prefix activates the instruction-tuned alignment in e5 models.
        descriptions = [
            "passage: " + val.description_for(lang) for val in DRIVE_VALUES
        ]
        _value_embedding_cache[key] = model.encode(descriptions, convert_to_tensor=True)
    return _value_embedding_cache[key]


async def filter_by_similarity(
    sentences: list[str],
    model: SentenceTransformer,
    lang: str,
    raw_floor: float,
    z_threshold: float,
    competitor_margin: float = 0.02,
) -> list[ScoredChunk]:
    """Filter sentences by semantic similarity to D.R.I.V.E. value descriptions.

    Uses same-language descriptions (Step 1) with e5 query/passage prefixes
    (Step 2) and z-score normalization (Step 3) to produce language-invariant
    scores.

    Args:
        sentences: Candidate sentences to score.
        model: Loaded SentenceTransformer.
        lang: Detected language of the input text.
        raw_floor: Minimum raw cosine similarity — guards against all-noise input.
        z_threshold: Minimum z-score (standard deviations above the per-sentence
            mean) required to keep a match.

    Returns:
        Filtered and sorted list of ScoredChunk, best match first.
    """
    if not sentences:
        return []

    value_embeddings = _get_value_embeddings(model, lang)

    # "query: " prefix pairs with "passage: " on the value side for e5 models.
    prefixed = ["query: " + s for s in sentences]
    sentence_embeddings = await asyncio.to_thread(
        model.encode, prefixed, convert_to_tensor=True
    )

    # Shape: (n_sentences, n_values)
    cosine_scores = await asyncio.to_thread(
        util.cos_sim, sentence_embeddings, value_embeddings
    )

    # How close a secondary value score must be to the best raw score to also be forwarded.
    # Reads from config (default 0.02): if best=0.85, also forward any value scoring >= 0.83.
    _margin = competitor_margin

    results = []
    for i, sentence in enumerate(sentences):
        scores = cosine_scores[i].cpu().numpy()  # shape (n_values,)

        # Z-score normalization across the 5 value scores for this sentence.
        mean = scores.mean()
        std = scores.std()
        z_scores = (scores - mean) / (std + 1e-8)

        best_raw = float(scores.max())

        # Forward the best value AND any runner-up within COMPETITOR_MARGIN of best.
        # This prevents argmax ties from silently dropping the correct value.
        for j, value in enumerate(DRIVE_VALUES):
            raw = float(scores[j])
            z = float(z_scores[j])

            if raw >= raw_floor and z >= z_threshold and raw >= (best_raw - _margin):
                results.append(
                    ScoredChunk(
                        text=sentence,
                        value_code=value.code,
                        value_name=value.name,
                        similarity_score=raw,
                    )
                )

    results.sort(key=lambda x: x.similarity_score, reverse=True)
    return results

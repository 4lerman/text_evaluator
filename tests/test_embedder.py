import pytest
from app.pipeline.embedder import filter_by_similarity


@pytest.mark.asyncio
async def test_filter_by_similarity_matches_value(embedding_model):
    sentences = ["I consistently regulated my emotions under pressure"]

    results = await filter_by_similarity(
        sentences, embedding_model, lang="en", raw_floor=0.80, z_threshold=1.5
    )

    assert len(results) > 0
    assert results[0].value_code == "D"
    assert results[0].text == sentences[0]


@pytest.mark.asyncio
async def test_filter_by_similarity_filters_irrelevant(embedding_model):
    # e5-large scores noise at ~0.74; raw_floor=0.80 filters it out.
    sentences = ["The weather is exceptionally nice today"]

    results = await filter_by_similarity(
        sentences, embedding_model, lang="en", raw_floor=0.80, z_threshold=1.5
    )

    assert len(results) == 0

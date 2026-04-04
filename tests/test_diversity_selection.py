import pytest
from app.pipeline.llm_evaluator import evaluate_with_llm
from app.schemas.responses import ScoredChunk
from app.config import Settings
from unittest.mock import MagicMock, patch

@pytest.fixture
def mock_config():
    # Set PROMETHEUS_MAX_CHUNKS to a small number to trigger selection logic
    return Settings(
        PROMETHEUS_MAX_CHUNKS=2,
        CONFIRMATION_THRESHOLD=3,
        PROMETHEUS_MODEL="mock-model"
    )

def create_chunk(text, value_code, score, index):
    return ScoredChunk(
        text=text,
        value_code=value_code,
        value_name="Test Value",
        similarity_score=score,
        sentence_index=index
    )

@pytest.mark.asyncio
@patch("app.pipeline.llm_evaluator.get_evaluator")
async def test_position_aware_selection(mock_get_evaluator, mock_config):
    """
    Test that selection picks one from the first half AND one from the second half
    when PROMETHEUS_MAX_CHUNKS is 2 and we provide total_sentences.
    """
    mock_evaluator = MagicMock()
    mock_get_evaluator.return_value = mock_evaluator
    # Mock evaluation to always return a confirmed score
    mock_evaluator.single_absolute_grade.return_value = ("Confirmed", 5)

    # Chunks: 
    # 1. Index 10 (First half if total=100)
    # 2. Index 80 (Second half if total=100)
    # 3. Index 90 (Second half if total=100, but higher score than index 80)
    #
    # Globally best are Index 90 (0.95) and Index 80 (0.90).
    # But if it's position-aware, it should pick Index 10 and Index 90.
    
    chunks = [
        create_chunk("Chunk 1", "D", 0.80, 10),
        create_chunk("Chunk 2", "D", 0.90, 80),
        create_chunk("Chunk 3", "D", 0.95, 90),
    ]
    
    # Run with total_sentences = 100 (midpoint = 50)
    results = await evaluate_with_llm(chunks, mock_config, total_sentences=100)
    
    # We expect 2 results (MAX_CHUNKS=2)
    assert len(results) == 2
    
    selected_texts = {r.text for r in results}
    # Should include Chunk 1 (best in first half) and Chunk 3 (best in second half)
    assert "Chunk 1" in selected_texts
    assert "Chunk 3" in selected_texts
    assert "Chunk 2" not in selected_texts

@pytest.mark.asyncio
@patch("app.pipeline.llm_evaluator.get_evaluator")
async def test_fallback_selection(mock_get_evaluator, mock_config):
    """
    Test fallback behavior when total_sentences is NOT provided.
    It should use the median of filtered indices.
    """
    mock_evaluator = MagicMock()
    mock_get_evaluator.return_value = mock_evaluator
    mock_evaluator.single_absolute_grade.return_value = ("Confirmed", 5)

    # Filtered chunks at indices [80, 85, 90].
    # Median is 85.
    # First half (< 85): [80]
    # Second half (>= 85): [85, 90]
    chunks = [
        create_chunk("Chunk 80", "D", 0.80, 80),
        create_chunk("Chunk 85", "D", 0.90, 85),
        create_chunk("Chunk 90", "D", 0.95, 90),
    ]
    
    # Run without total_sentences
    results = await evaluate_with_llm(chunks, mock_config)
    
    assert len(results) == 2
    selected_texts = {r.text for r in results}
    # Should pick "Chunk 80" (best in < 85) and "Chunk 90" (best in >= 85)
    assert "Chunk 80" in selected_texts
    assert "Chunk 90" in selected_texts

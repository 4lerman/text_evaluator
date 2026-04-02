import pytest
from unittest.mock import MagicMock, patch

from app.config import config
from app.pipeline.llm_evaluator import evaluate_with_llm
from app.schemas.responses import ScoredChunk


@pytest.fixture
def mock_chunk():
    return ScoredChunk(
        text="I stayed calm and handled the crisis.",
        value_code="D",
        value_name="Disciplined Resilience",
        similarity_score=0.85,
    )


@pytest.mark.asyncio
@patch("app.pipeline.llm_evaluator.get_evaluator")
async def test_evaluate_with_llm_confirmed(mock_get_evaluator, mock_chunk):
    mock_evaluator = MagicMock()
    mock_get_evaluator.return_value = mock_evaluator
    
    # Mock return value of single_absolute_grade: (feedback, score)
    mock_evaluator.single_absolute_grade.return_value = ("Shows deliberate emotional regulation.", 5)

    test_config = config.model_copy(update={"CONFIRMATION_THRESHOLD": 3})
    results = await evaluate_with_llm([mock_chunk], test_config)

    assert len(results) == 1
    assert results[0].confirmed is True
    assert results[0].score == 5
    assert results[0].reasoning == "Shows deliberate emotional regulation."


@pytest.mark.asyncio
@patch("app.pipeline.llm_evaluator.get_evaluator")
async def test_evaluate_with_llm_rejected(mock_get_evaluator, mock_chunk):
    mock_evaluator = MagicMock()
    mock_get_evaluator.return_value = mock_evaluator
    
    # Mock return value: score 2 is below threshold 3
    mock_evaluator.single_absolute_grade.return_value = ("No concrete behavior described.", 2)

    test_config = config.model_copy(update={"CONFIRMATION_THRESHOLD": 3})
    results = await evaluate_with_llm([mock_chunk], test_config)

    assert len(results) == 0

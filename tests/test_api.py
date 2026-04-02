from unittest.mock import patch, AsyncMock

from app.schemas.responses import (
    EvaluateResponse,
    HighlightedSentence,
    HighlightedToken,
    Metrics,
)


@patch("app.routers.evaluate.run_funnel", new_callable=AsyncMock)
def test_evaluate_endpoint_success(mock_funnel, test_client):
    mock_funnel.return_value = EvaluateResponse(
        results=[
            HighlightedSentence(
                text="I consistently regulated my emotions under extreme pressure.",
                value_code="D",
                value_name="Disciplined Resilience",
                reasoning="Shows emotional regulation",
                score=4,
                highlights=[
                    HighlightedToken(token="consistently", pos_category="ADVERB", start=2, end=14),
                ],
            )
        ],
        summary={"D": 1},
        metrics=Metrics(
            overall_score=4.0,
            coverage=1,
            balance_score=1.0,
            strongest=["D"],
            weakest=["D"],
        ),
        lang="en",
    )

    response = test_client.post(
        "/evaluate",
        json={"text": "I consistently regulated my emotions under extreme pressure."},
    )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "summary" in data
    assert len(data["results"]) >= 1
    assert data["summary"].get("D", 0) >= 1


@patch("app.routers.evaluate.run_funnel", new_callable=AsyncMock)
def test_evaluate_endpoint_empty_text(mock_funnel, test_client):
    mock_funnel.return_value = EvaluateResponse(results=[], summary={}, lang="en")

    response = test_client.post("/evaluate", json={"text": "No."})

    assert response.status_code == 200
    data = response.json()
    assert data["results"] == []
    assert data["summary"] == {}

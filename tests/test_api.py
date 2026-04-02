from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient
import json

@patch('app.pipeline.llm_evaluator.httpx.AsyncClient')
def test_evaluate_endpoint_success(mock_client, test_client):
    # Mock LLM API response to simulate a confirmed evaluation
    mock_instance = AsyncMock()
    mock_client.return_value.__aenter__.return_value = mock_instance
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": json.dumps({
                            "confirmed": True,
                            "reasoning": "Simulated reasoning logic for resilience"
                        })}
                    ]
                }
            }
        ]
    }
    mock_instance.post.return_value = mock_response
    
    # Request body
    payload = {
        "text": "I consistently regulated my emotions under extreme pressure."
    }
    
    response = test_client.post("/evaluate", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "summary" in data
    
    # Check if a result mapped back
    # With a high threshold, and this sentence matching "D", it should return a result
    # Assuming threshold allows it to pass and LLM confirms
    assert len(data["results"]) >= 1
    assert data["summary"].get("D", 0) >= 1

@patch('app.pipeline.llm_evaluator.httpx.AsyncClient')
def test_evaluate_endpoint_empty_text(mock_client, test_client):
    # Tests short text graceful handling (should return empty results)
    payload = {"text": "No."}
    
    response = test_client.post("/evaluate", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["results"] == []
    assert data["summary"] == {}
    
    # Ensure no HTTP calls were made because chunker should have blocked it
    assert not mock_client.called

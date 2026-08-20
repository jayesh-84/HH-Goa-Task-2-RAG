import pytest
from unittest.mock import patch, MagicMock
from backend.orchestrator.pipeline import RAGOrchestrator

@patch("backend.orchestrator.pipeline.log_query_metrics")
@patch("backend.orchestrator.pipeline.validate_grounding")
@patch("backend.orchestrator.pipeline.generate_grounded_answer")
@patch("backend.orchestrator.pipeline.check_context_sufficiency")
@patch("backend.orchestrator.pipeline.check_topic")
@patch("backend.orchestrator.pipeline.retrieve_similar_contexts")
@patch("backend.orchestrator.pipeline.check_safety")
def test_orchestrator_pipeline_success(
    mock_safety, mock_retrieval, mock_topic, mock_sufficiency, 
    mock_generation, mock_grounding, mock_log
):
    # Mock return values for a successful run
    mock_safety.return_value = (True, "")
    mock_retrieval.return_value = {
        "query": "कम्पनी?",
        "results": [{"text": "कम्पनी एक संगठन है।", "score": 0.90}],
        "latency_ms": 12.0
    }
    mock_topic.return_value = (True, "")
    mock_sufficiency.return_value = (True, "")
    mock_generation.return_value = "कम्पनी एक संगठन है।"
    mock_grounding.return_value = (True, "Answer is grounded.")
    
    orchestrator = RAGOrchestrator()
    response = orchestrator.run_pipeline(text_query="कम्पनी?")
    
    # Assert output structure
    assert response["query_text"] == "कम्पनी?"
    assert response["answer"] == "कम्पनी एक संगठन है।"
    assert len(response["contexts"]) == 1
    assert response["guardrail_status"]["safety_passed"] is True
    assert response["guardrail_status"]["grounding_passed"] is True
    assert response["overall_latency_ms"] >= 0.0
    assert response["latency_breakdown"]["retrieval_ms"] == 12.0
    
    # Assert logs and functions were called
    mock_safety.assert_called_once_with("कम्पनी?")
    mock_retrieval.assert_called_once_with("कम्पनी?")
    mock_generation.assert_called_once_with("कम्पनी?", ["कम्पनी एक संगठन है।"])
    mock_grounding.assert_called_once_with("कम्पनी?", "कम्पनी एक संगठन है।", ["कम्पनी एक संगठन है।"])
    mock_log.assert_called_once()

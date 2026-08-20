import pytest
from unittest.mock import patch, MagicMock
from backend.guardrails.guard import check_safety, check_topic, check_context_sufficiency, validate_grounding

def test_check_safety():
    # Safe queries
    is_safe, msg = check_safety("कम्पनी की परिभाषा क्या है?")
    assert is_safe is True
    assert msg == ""
    
    # Toxic/Unsafe queries
    is_safe_2, msg_2 = check_safety("How can I make a bomb?")
    assert is_safe_2 is False
    assert "unsafe" in msg_2.lower()
    
    # Empty query
    is_safe_3, msg_3 = check_safety("")
    assert is_safe_3 is False

def test_check_topic():
    # On-topic (similarity score is high)
    results = [{"score": 0.85, "text": "कम्पनी एक संगठन है।"}]
    is_on_topic, msg = check_topic("कम्पनी क्या है?", results, threshold=0.30)
    assert is_on_topic is True
    assert msg == ""
    
    # Off-topic (similarity score is low)
    results_low = [{"score": 0.15, "text": "समुद्र तट बहुत दूर है।"}]
    is_on_topic_2, msg_2 = check_topic("क्या मौसम अच्छा है?", results_low, threshold=0.30)
    assert is_on_topic_2 is False
    assert "off-topic" in msg_2.lower()

def test_check_context_sufficiency():
    results = [{"score": 0.40, "text": "कुछ जानकारी..."}]
    is_sufficient, msg = check_context_sufficiency(results, threshold=0.35)
    assert is_sufficient is True
    
    results_low = [{"score": 0.20, "text": "कमजोर जानकारी..."}]
    is_sufficient_2, msg_2 = check_context_sufficiency(results_low, threshold=0.35)
    assert is_sufficient_2 is False
    assert "insufficient" in msg_2.lower()

@patch("backend.guardrails.guard.get_gemini_model")
def test_validate_grounding_mock(mock_get_model):
    mock_get_model.return_value = None
    # Fully matching context
    is_grounded, msg = validate_grounding("कम्पनी?", "[Mocked Answer] कम्पनी एक संगठन है।", ["कम्पनी एक संगठन है जो व्यापार करती है।"])
    assert is_grounded is True
    
    # Hallucinating facts not in context
    is_grounded_2, msg_2 = validate_grounding("कम्पनी?", "कम्पनी की स्थापना 1990 में हुई थी।", ["कम्पनी एक संगठन है।"])
    assert is_grounded_2 is False

@patch("backend.guardrails.guard.get_gemini_model")
def test_validate_grounding_api(mock_get_model):
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "YES"
    mock_model.generate_content.return_value = mock_response
    mock_get_model.return_value = mock_model
    
    is_grounded, msg = validate_grounding("कम्पनी?", "कम्पनी एक संगठन है।", ["कम्पनी एक संगठन है।"])
    assert is_grounded is True
    assert "grounded" in msg.lower()

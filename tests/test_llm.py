import pytest
import os
from unittest.mock import patch, MagicMock
from google.genai.errors import APIError
from backend.llm.generator import (
    generate_grounded_answer,
    generate_mock_answer,
    get_gemini_model,
    generation_mode_var,
    GeminiModelWrapper
)

def test_generate_mock_answer():
    # Sufficiency test
    ans = generate_mock_answer("कम्पनी क्या है?", ["कम्पनी एक संगठन है जो व्यापार करता है।"])
    assert "Mocked Answer" in ans
    assert "कम्पनी एक संगठन है" in ans
    
    # Fallback response
    ans_fallback = generate_mock_answer("असंबंधित प्रश्न?", [])
    assert ans_fallback == "I cannot answer this question based on the provided dataset."

@patch("backend.llm.generator.get_gemini_model")
def test_generate_grounded_answer_api(mock_get_model):
    # Mock model
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "  कम्पनी एक संगठन है।  "
    mock_model.generate_content.return_value = mock_response
    mock_get_model.return_value = mock_model
    
    contexts = ["कम्पनी एक व्यावसायिक संगठन है।"]
    query = "कम्पनी क्या है?"
    
    ans = generate_grounded_answer(query, contexts)
    
    assert ans == "कम्पनी एक संगठन है।"
    assert generation_mode_var.get() == "REAL GEMINI + GROUNDED"
    mock_model.generate_content.assert_called_once()

@patch("backend.llm.generator.genai.Client")
def test_gemini_initialization_success(mock_client):
    # Clear cache
    import backend.llm.generator as generator
    generator._client_instance = None
    generator._model_cache = {}
    
    with patch.dict(os.environ, {"GEMINI_API_KEY": "test_valid_key"}):
        model = get_gemini_model()
        assert model is not None
        assert isinstance(model, GeminiModelWrapper)
        
    # Reset cache
    generator._client_instance = None
    generator._model_cache = {}

def test_gemini_initialization_missing_key():
    # Clear cache
    import backend.llm.generator as generator
    generator._client_instance = None
    generator._model_cache = {}
    
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        model = get_gemini_model()
        assert model is None
        
    with patch.dict(os.environ, {"GEMINI_API_KEY": "PASTE_MY_KEY_HERE"}):
        model = get_gemini_model()
        assert model is None

@patch("backend.llm.generator.get_gemini_model")
def test_generate_grounded_answer_api_failure(mock_get_model):
    # Mock model to raise APIError
    mock_model = MagicMock()
    
    # APIError constructor takes code and response_json
    mock_api_error = APIError(429, {"error": {"message": "Quota exceeded"}})
    mock_model.generate_content.side_effect = mock_api_error
    mock_get_model.return_value = mock_model
    
    contexts = ["कम्पनी एक व्यावसायिक संगठन है।"]
    query = "कम्पनी क्या है?"
    
    ans = generate_grounded_answer(query, contexts)
    assert "Gemini API Error" in ans
    assert "Quota exceeded" in ans
    assert generation_mode_var.get() == "ERROR"

@patch("backend.llm.generator.get_gemini_model")
def test_generate_grounded_answer_empty_response(mock_get_model):
    # Mock model returning empty text
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = ""
    mock_model.generate_content.return_value = mock_response
    mock_get_model.return_value = mock_model
    
    contexts = ["कम्पनी एक व्यावसायिक संगठन है।"]
    query = "कम्पनी क्या है?"
    
    ans = generate_grounded_answer(query, contexts)
    assert "empty or malformed" in ans.lower()
    assert generation_mode_var.get() == "ERROR"

@patch("backend.llm.generator.get_gemini_model")
def test_generate_grounded_answer_fallback_refusal(mock_get_model):
    # Mock model returning refusal
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "I cannot answer this question based on the provided dataset."
    mock_model.generate_content.return_value = mock_response
    mock_get_model.return_value = mock_model
    
    contexts = ["कम्पनी एक व्यावसायिक संगठन है।"]
    query = "कम्पनी क्या है?"
    
    ans = generate_grounded_answer(query, contexts)
    assert ans == "I cannot answer this question based on the provided dataset."
    assert generation_mode_var.get() == "REAL GEMINI + GROUNDED"

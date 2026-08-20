import pytest
from unittest.mock import patch, MagicMock
from backend.stt.sarvam_stt import speech_to_text

@patch("backend.stt.sarvam_stt.SARVAM_API_KEY", "")
def test_speech_to_text_mock_fallback():
    # If API key is empty, it should fallback to mock response
    transcript = speech_to_text("dummy_path.wav")
    assert transcript == "कम्पनी क्या है?"

@patch("backend.stt.sarvam_stt.requests.post")
@patch("backend.stt.sarvam_stt.os.path.exists")
@patch("backend.stt.sarvam_stt.SARVAM_API_KEY", "dummy_key")
def test_speech_to_text_api_call(mock_exists, mock_post):
    mock_exists.return_value = True
    
    # Mock successful response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"transcript": "हेलो वर्ल्ड"}
    mock_post.return_value = mock_response
    
    # Run transcription
    # We patch open to return a mock file handle so we don't open a real file
    with patch("builtins.open", MagicMock()):
        transcript = speech_to_text("real_path.wav", language_code="hi-IN")
        
    assert transcript == "हेलो वर्ल्ड"
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert kwargs["headers"]["api-subscription-key"] == "dummy_key"
    assert kwargs["data"]["language-code"] == "hi-IN"

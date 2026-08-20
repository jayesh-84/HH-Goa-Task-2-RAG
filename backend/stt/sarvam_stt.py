import os
import requests
from backend.config import SARVAM_API_KEY

def speech_to_text(audio_file_path: str, language_code: str = "hi-IN") -> str:
    """
    Transcribes audio file to text using Sarvam AI Speech-to-Text API.
    Supports modular fallback if API key is missing.
    """
    if not SARVAM_API_KEY:
        print("Warning: SARVAM_API_KEY is missing. Running Speech-to-Text in mock fallback mode.")
        # Return a mock query to trigger the rest of the RAG pipeline
        return "कम्पनी क्या है?"
        
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
        
    url = "https://api.sarvam.ai/speech-to-text"
    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }
    
    # We pass file as binary multipart form data
    # and optional parameters as form data
    data = {
        "model": "saaras:v1",
        "language-code": language_code,
        "mode": "transcribe"
    }
    
    try:
        with open(audio_file_path, "rb") as f:
            files = {
                "file": (os.path.basename(audio_file_path), f, "audio/wav")
            }
            response = requests.post(url, headers=headers, files=files, data=data)
            
        if response.status_code == 200:
            result = response.json()
            transcript = result.get("transcript", "").strip()
            print(f"STT Transcript: '{transcript}'")
            return transcript
        else:
            print(f"STT API Error (status {response.status_code}): {response.text}")
            return "कम्पनी क्या है?"  # Fallback to mock text on error to allow pipeline progress
    except Exception as e:
        print(f"STT Exception: {e}")
        return "कम्पनी क्या है?"  # Fallback to mock text on exception

import pytest
from unittest.mock import patch, MagicMock
from backend.ingestion.pipeline import clean_text, ingest_dataset

def test_clean_text():
    assert clean_text("  hello   world  ") == "hello world"
    assert clean_text("\nhello\tworld\n\n") == "hello world"
    assert clean_text("") == ""
    assert clean_text(None) == ""

@patch("backend.ingestion.pipeline.load_dataset")
def test_ingest_dataset(mock_load_dataset):
    # Mock MSMARCO-XI data item
    mock_data = [
        {
            "query_id": 1,
            "query": "कम्पनी क्या है?",
            "Answer": "कम्पनी एक संगठन है।",
            "target_lang": "hin_Deva",
            "passages": {
                "Translated_passages": ["कम्पनी की परिभाषा...", "अन्य तथ्य..."],
                "English_passages": ["Company definition...", "Other facts..."],
                "is_selected": [1, 0]
            }
        },
        {
            "query_id": 2,
            "query": "অন্য প্রশ্ন?",
            "Answer": "উত্তর...",
            "target_lang": "asm_Beng",
            "passages": {
                "Translated_passages": ["অসমীয়া অনুচ্ছেদ..."],
                "English_passages": ["Assamese paragraph..."],
                "is_selected": [1]
            }
        }
    ]
    
    mock_load_dataset.return_value = mock_data
    
    # Ingest for Hindi
    documents = ingest_dataset(limit=1, target_lang="hin_Deva")
    
    # We should have 2 passages for Hindi (since Hindi item has 2 translated passages)
    assert len(documents) == 2
    
    # Check document properties
    assert documents[0]["text"] == "कम्पनी की परिभाषा..."
    assert documents[0]["metadata"]["query_id"] == 1
    assert documents[0]["metadata"]["passage_index"] == 0
    assert documents[0]["metadata"]["is_selected"] is True
    assert documents[0]["metadata"]["english_text"] == "Company definition..."
    assert documents[0]["metadata"]["query_text"] == "कम्पनी क्या है?"
    assert documents[0]["metadata"]["answer_text"] == "कम्पनी एक संगठन है।"
    assert documents[0]["metadata"]["target_lang"] == "hin_Deva"
    
    assert documents[1]["text"] == "अन्य तथ्य..."
    assert documents[1]["metadata"]["is_selected"] is False

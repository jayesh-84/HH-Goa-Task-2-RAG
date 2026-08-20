import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from backend.retrieval.retriever import retrieve_similar_contexts

@patch("backend.retrieval.retriever.get_vector_store")
@patch("backend.retrieval.retriever.embed_query")
def test_retrieve_similar_contexts(mock_embed_query, mock_get_vector_store):
    # Mock query embedding
    mock_embed_query.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    
    # Mock vector store
    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [
        {
            "text": "समुद्र तट बहुत सुंदर हैं।",
            "metadata": {"query_id": 10},
            "score": 0.95
        }
    ]
    mock_get_vector_store.return_value = mock_store
    
    # Run retrieval
    response = retrieve_similar_contexts("समुद्र", k=2)
    
    # Verify response structure
    assert response["query"] == "समुद्र"
    assert response["latency_ms"] >= 0.0
    assert len(response["results"]) == 1
    assert response["results"][0]["text"] == "समुद्र तट बहुत सुंदर हैं।"
    assert response["results"][0]["score"] == 0.95
    assert response["results"][0]["metadata"]["query_id"] == 10
    
    # Verify mock interactions
    mock_embed_query.assert_called_once_with("समुद्र")
    mock_store.similarity_search.assert_called_once_with(mock_embed_query.return_value, k=2)

import pytest
from unittest.mock import patch
import numpy as np
from backend.chunking.strategies import FixedSizeChunker, SemanticChunker, MetadataChunker, get_chunker

def test_fixed_size_chunker_basic():
    doc = {
        "text": "abcdefghij",  # 10 chars
        "metadata": {"test": "val"}
    }
    chunker = FixedSizeChunker(chunk_size=4, chunk_overlap=2)
    chunks = chunker.chunk_document(doc)
    
    assert len(chunks) == 5
    assert chunks[0]["text"] == "abcd"
    assert chunks[0]["metadata"]["chunk_index"] == 0
    assert chunks[0]["metadata"]["chunk_start"] == 0
    assert chunks[0]["metadata"]["chunk_end"] == 4
    assert chunks[0]["metadata"]["test"] == "val"
    
    assert chunks[1]["text"] == "cdef"
    assert chunks[2]["text"] == "efgh"
    assert chunks[3]["text"] == "ghij"
    assert chunks[4]["text"] == "ij"

def test_fixed_size_chunker_short_text():
    doc = {
        "text": "abc",
        "metadata": {}
    }
    chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
    chunks = chunker.chunk_document(doc)
    
    assert len(chunks) == 1
    assert chunks[0]["text"] == "abc"
    assert chunks[0]["metadata"]["chunk_index"] == 0

def test_fixed_size_chunker_invalid():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=0, chunk_overlap=2)
        
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=10, chunk_overlap=12)

@patch("backend.chunking.strategies.embed_texts")
def test_semantic_chunker_splitting(mock_embed_texts):
    # Mock embeddings: sentence 1 and sentence 2 are similar, sentence 3 is different
    # Sentence 1: "This is sentence one."
    # Sentence 2: "This is sentence two."
    # Sentence 3: "A completely different topic here."
    mock_embed_texts.return_value = np.array([
        [1.0, 0.0],  # Sentence 1
        [0.99, 0.01], # Sentence 2 (cosine similarity ~ 0.99 with Sent 1)
        [0.0, 1.0]   # Sentence 3 (cosine similarity ~ 0.0 with Sent 2)
    ], dtype=np.float32)
    
    doc = {
        "text": "This is sentence one. This is sentence two. A completely different topic here.",
        "metadata": {"source": "doc1"}
    }
    
    chunker = SemanticChunker(similarity_threshold=0.5)
    chunks = chunker.chunk_document(doc)
    
    # We expect 2 chunks:
    # Chunk 1: "This is sentence one. This is sentence two."
    # Chunk 2: "A completely different topic here."
    assert len(chunks) == 2
    assert chunks[0]["text"] == "This is sentence one. This is sentence two."
    assert chunks[0]["metadata"]["chunk_index"] == 0
    assert chunks[1]["text"] == "A completely different topic here."
    assert chunks[1]["metadata"]["chunk_index"] == 1

def test_metadata_chunker():
    doc = {
        "text": "This is sample text.",
        "metadata": {
            "query_text": "What is the query?",
            "target_lang": "hin_Deva"
        }
    }
    
    chunker = MetadataChunker(chunk_size=100, chunk_overlap=10)
    chunks = chunker.chunk_document(doc)
    
    assert len(chunks) == 1
    # Expect metadata headers prepended
    assert chunks[0]["text"].startswith("[Query: What is the query? | Lang: hin_Deva]")
    assert "This is sample text." in chunks[0]["text"]
    assert chunks[0]["metadata"]["clean_text"] == "This is sample text."

def test_get_chunker_factory():
    c1 = get_chunker("fixed_size", chunk_size=300, chunk_overlap=30)
    assert isinstance(c1, FixedSizeChunker)
    
    c2 = get_chunker("semantic", similarity_threshold=0.6)
    assert isinstance(c2, SemanticChunker)
    assert c2.similarity_threshold == 0.6
    
    c3 = get_chunker("metadata", chunk_size=200, chunk_overlap=20)
    assert isinstance(c3, MetadataChunker)

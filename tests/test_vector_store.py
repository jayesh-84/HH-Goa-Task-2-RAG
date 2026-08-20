import os
import tempfile
import numpy as np
import pytest
from unittest.mock import patch
from backend.vector_store.store import FAISSVectorStore

def test_faiss_vector_store_basic():
    # Bypass saving to actual config paths by pointing files to a temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        with patch("backend.vector_store.store.FAISS_INDEX_PATH", tmp_dir):
            # Use dimension 3 for simple testing
            store = FAISSVectorStore(dimension=3)
            
            store.index_file = os.path.join(tmp_dir, "index.faiss")
            store.metadata_file = os.path.join(tmp_dir, "metadata.pkl")
            
            docs = [
                {"text": "docA", "metadata": {"id": "A"}},
                {"text": "docB", "metadata": {"id": "B"}},
                {"text": "docC", "metadata": {"id": "C"}},
            ]
            # Unit vectors
            embeddings = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
            
            store.add_documents(docs, embeddings)
            assert store.index is not None
            assert store.index.ntotal == 3
            assert len(store.documents) == 3
            
            # Query matching docA [1.0, 0.0, 0.0]
            query_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            results = store.similarity_search(query_emb, k=2)
            
            assert len(results) == 2
            assert results[0]["text"] == "docA"
            assert abs(results[0]["score"] - 1.0) < 1e-5
            
            # Test saving and reloading
            store.save()
            assert os.path.exists(store.index_file)
            assert os.path.exists(store.metadata_file)
            
            # Load into a new store (it will also see the patched temporary directory during __init__)
            new_store = FAISSVectorStore(dimension=3)
            new_store.index_file = store.index_file
            new_store.metadata_file = store.metadata_file
            loaded = new_store.load()
            
            assert loaded is True
            assert new_store.index.ntotal == 3
            assert len(new_store.documents) == 3
            
            # Query reload
            results2 = new_store.similarity_search(query_emb, k=1)
            assert results2[0]["text"] == "docA"

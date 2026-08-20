import os
import faiss
import pickle
import numpy as np
from backend.config import FAISS_INDEX_PATH

class FAISSVectorStore:
    """
    Modular Vector Store implementation wrapping FAISS.
    Uses cosine similarity (Inner Product on L2-normalized embeddings).
    Stores document metadata in a pickle file alongside the FAISS index.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None
        self.documents = []  # List of dicts: {"text": str, "metadata": dict}
        
        # Determine file paths
        self.index_file = f"{FAISS_INDEX_PATH}/index.faiss"
        self.metadata_file = f"{FAISS_INDEX_PATH}/metadata.pkl"
        
        # Load from disk if index exists
        self.load()

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalizes vectors to unit length so Inner Product equals Cosine Similarity.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-8)

    def add_documents(self, documents: list, embeddings: np.ndarray):
        """
        Adds documents and their corresponding pre-computed embeddings to the index.
        """
        if len(documents) == 0:
            return
            
        assert len(documents) == len(embeddings), "Mismatch between number of documents and embeddings"
        assert embeddings.shape[1] == self.dimension, f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
        
        # Standardize type to float32
        embeddings = embeddings.astype(np.float32)
        # Normalize for cosine similarity
        norm_embeddings = self._normalize(embeddings)
        
        # Lazy initialization of FAISS Index
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)
            
        # Add to index
        self.index.add(norm_embeddings)
        # Append documents metadata
        self.documents.extend(documents)

    def similarity_search(self, query_embedding: np.ndarray, k: int = 3) -> list:
        """
        Performs cosine similarity search using the query embedding.
        Returns a list of dictionaries with document text, metadata, and similarity score.
        """
        if self.index is None or self.index.ntotal == 0:
            return []
            
        # Reshape query embedding if 1D
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        query_embedding = query_embedding.astype(np.float32)
        # Normalize query embedding
        query_embedding = self._normalize(query_embedding)
        
        # Query FAISS
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.documents[idx]
            results.append({
                "text": doc["text"],
                "metadata": doc["metadata"],
                "score": float(score)  # Cosine similarity score
            })
            
        return results

    def save(self):
        """
        Persists the FAISS index and documents metadata to disk.
        """
        if self.index is None:
            return
            
        os.makedirs(os.path.dirname(self.index_file), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_file)
        
        # Save document metadata
        with open(self.metadata_file, "wb") as f:
            pickle.dump(self.documents, f)
            
        print(f"Saved FAISS index ({self.index.ntotal} vectors) to '{self.index_file}'")

    def load(self) -> bool:
        """
        Loads the FAISS index and documents metadata from disk if they exist.
        Returns True if loaded successfully, False otherwise.
        """
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.metadata_file, "rb") as f:
                    self.documents = pickle.load(f)
                print(f"Loaded existing FAISS index from disk ({self.index.ntotal} vectors).")
                return True
            except Exception as e:
                print(f"Failed to load FAISS index from disk: {e}")
                self.index = None
                self.documents = []
                
        return False
        
    def clear(self):
        """
        Clears the index and deletes files from disk.
        """
        self.index = None
        self.documents = []
        if os.path.exists(self.index_file):
            os.remove(self.index_file)
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
        print("Vector store cleared.")

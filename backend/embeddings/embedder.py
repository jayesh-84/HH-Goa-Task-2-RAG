from sentence_transformers import SentenceTransformer
from backend.config import EMBEDDING_MODEL_NAME
import numpy as np

# Singleton instance of the model
_model_instance = None

def get_embedding_model() -> SentenceTransformer:
    """
    Loads and caches the SentenceTransformer model lazily.
    """
    global _model_instance
    if _model_instance is None:
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        _model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Model loaded successfully.")
    return _model_instance

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Generates embeddings for a list of text strings.
    Returns a numpy array of shape (num_texts, embedding_dimension).
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings.astype(np.float32)

def embed_query(query: str) -> np.ndarray:
    """
    Generates embedding for a single query string.
    Returns a 1D numpy array of shape (embedding_dimension,).
    """
    model = get_embedding_model()
    embedding = model.encode(query, show_progress_bar=False, convert_to_numpy=True)
    return embedding.astype(np.float32)

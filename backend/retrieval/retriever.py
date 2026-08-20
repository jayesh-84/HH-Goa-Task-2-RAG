import time
from backend.embeddings.embedder import embed_query
from backend.vector_store.store import FAISSVectorStore
from backend.config import RETRIEVAL_TOP_K

_vector_store_instance = None

def get_vector_store() -> FAISSVectorStore:
    """
    Returns a singleton instance of FAISSVectorStore.
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = FAISSVectorStore()
    return _vector_store_instance

def retrieve_similar_contexts(query: str, k: int = None) -> dict:
    """
    Performs top-k similarity search on the query.
    Measures retrieval latency in milliseconds.
    
    Returns a dict with:
      - "query": original query
      - "results": list of matches with {"text", "metadata", "score"}
      - "latency_ms": search latency in milliseconds
    """
    if k is None:
        k = RETRIEVAL_TOP_K
        
    start_time = time.perf_counter()
    
    # 1. Embed query
    query_emb = embed_query(query)
    
    # 2. Search vector store
    store = get_vector_store()
    results = store.similarity_search(query_emb, k=k)
    
    end_time = time.perf_counter()
    latency_ms = (end_time - start_time) * 1000.0
    
    return {
        "query": query,
        "results": results,
        "latency_ms": round(latency_ms, 2)
    }

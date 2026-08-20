from abc import ABC, abstractmethod
from backend.embeddings.embedder import embed_texts

class BaseChunker(ABC):
    """
    Abstract Base Class for all chunking strategies.
    Decoupled interface to allow comparing and swapping strategies easily.
    """
    @abstractmethod
    def chunk_document(self, doc: dict) -> list:
        """
        Chunks a document dictionary: {"text": str, "metadata": dict}
        Returns a list of chunk dictionaries: [{"text": str, "metadata": dict}]
        """
        pass

class FixedSizeChunker(BaseChunker):
    """
    Implements fixed-size character-based chunking with configurable overlap.
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and less than chunk_size")
            
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: dict) -> list:
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        
        chunks = []
        text_len = len(text)
        
        # If the text is shorter than or equal to chunk_size, return it as a single chunk
        if text_len <= self.chunk_size:
            return [{
                "text": text,
                "metadata": {
                    **metadata,
                    "chunk_index": 0,
                    "chunk_start": 0,
                    "chunk_end": text_len,
                    "chunking_strategy": "fixed_size"
                }
            }]
            
        start = 0
        chunk_idx = 0
        
        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_text = text[start:end]
            
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": chunk_idx,
                    "chunk_start": start,
                    "chunk_end": end,
                    "chunking_strategy": "fixed_size"
                }
            })
            
            chunk_idx += 1
            # Move start forward by chunk_size - chunk_overlap
            start += (self.chunk_size - self.chunk_overlap)
            
            # Prevent infinite loop if calculation gets stuck
            if self.chunk_size == self.chunk_overlap:
                break
                
        return chunks

class SemanticChunker(BaseChunker):
    """
    Implements semantic-aware chunking.
    Splits text into sentences, computes sentence embeddings,
    and groups sentences that are semantically close using cosine similarity.
    """
    def __init__(self, similarity_threshold: float = 0.5):
        self.similarity_threshold = similarity_threshold

    def chunk_document(self, doc: dict) -> list:
        import re
        import numpy as np
        
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        
        if not text.strip():
            return []
            
        # Split by sentences (handles English periods and Hindi dandas, as well as ? and !)
        sentence_delimiters = re.compile(r'(?<=[।\.!\?])\s+')
        sentences = [s.strip() for s in sentence_delimiters.split(text) if s.strip()]
        
        if len(sentences) <= 1:
            return [{
                "text": text,
                "metadata": {
                    **metadata,
                    "chunk_index": 0,
                    "chunk_start": 0,
                    "chunk_end": len(text),
                    "chunking_strategy": "semantic"
                }
            }]
            
        # Compute sentence embeddings
        try:
            embeddings = embed_texts(sentences)
            # Normalize embeddings for cosine similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norm_embeddings = embeddings / (norms + 1e-8)
            
            # Compute cosine similarities between adjacent sentences
            similarities = []
            for idx in range(len(sentences) - 1):
                sim = float(np.dot(norm_embeddings[idx], norm_embeddings[idx+1]))
                similarities.append(sim)
        except Exception as e:
            # Fallback to naive splitting if embedding fails
            print(f"Warning: Semantic chunking failed to embed, falling back to sentence splitting. Error: {e}")
            similarities = [0.0] * (len(sentences) - 1)
            
        chunks = []
        current_sentences = [sentences[0]]
        current_start = text.find(sentences[0])
        
        for idx, sim in enumerate(similarities):
            next_sentence = sentences[idx+1]
            if sim < self.similarity_threshold:
                # Similarity is below threshold; split chunk here
                chunk_text = " ".join(current_sentences)
                current_end = text.find(current_sentences[-1], current_start) + len(current_sentences[-1])
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": len(chunks),
                        "chunk_start": current_start,
                        "chunk_end": current_end,
                        "chunking_strategy": "semantic"
                    }
                })
                current_sentences = [next_sentence]
                current_start = text.find(next_sentence, current_end)
                if current_start == -1:
                    current_start = text.find(next_sentence)
            else:
                current_sentences.append(next_sentence)
                
        # Append final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            # Try to find final position
            current_end = text.find(current_sentences[-1], current_start)
            if current_end == -1:
                current_end = len(text)
            else:
                current_end += len(current_sentences[-1])
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": len(chunks),
                    "chunk_start": current_start,
                    "chunk_end": current_end,
                    "chunking_strategy": "semantic"
                }
            })
            
        return chunks

class MetadataChunker(BaseChunker):
    """
    Implements metadata-aware chunking.
    Splits the text using fixed-size chunks, but prepends structured metadata
    (e.g., associated query_text, target_lang) to enrich the chunk content for vector search.
    """
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 40):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: dict) -> list:
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})
        
        # Split using fixed size
        from backend.chunking.strategies import FixedSizeChunker
        base_chunker = FixedSizeChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        base_chunks = base_chunker.chunk_document(doc)
        
        enriched_chunks = []
        q_text = metadata.get("query_text", "")
        lang = metadata.get("target_lang", "")
        
        prefix = ""
        if q_text:
            prefix += f"Query: {q_text} | "
        if lang:
            prefix += f"Lang: {lang} | "
        if prefix:
            prefix = f"[{prefix.strip(' | ')}] "
            
        for chunk in base_chunks:
            enriched_text = f"{prefix}{chunk['text']}"
            enriched_chunks.append({
                "text": enriched_text,
                "metadata": {
                    **chunk["metadata"],
                    "clean_text": chunk["text"],
                    "chunking_strategy": "metadata"
                }
            })
            
        return enriched_chunks

def get_chunker(strategy: str = "fixed_size", **kwargs) -> BaseChunker:
    """
    Factory function to retrieve the configured chunker strategy.
    """
    if strategy == "fixed_size":
        size = kwargs.get("chunk_size", 500)
        overlap = kwargs.get("chunk_overlap", 50)
        return FixedSizeChunker(chunk_size=size, chunk_overlap=overlap)
    elif strategy == "semantic":
        threshold = kwargs.get("similarity_threshold", 0.5)
        return SemanticChunker(similarity_threshold=threshold)
    elif strategy == "metadata":
        size = kwargs.get("chunk_size", 400)
        overlap = kwargs.get("chunk_overlap", 40)
        return MetadataChunker(chunk_size=size, chunk_overlap=overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

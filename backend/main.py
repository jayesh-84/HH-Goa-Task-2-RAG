import sys
import argparse
from backend.config import DATASET_LANGUAGE, MAX_DATASET_ITEMS, CHUNKER_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_TOP_K
from backend.ingestion.pipeline import ingest_dataset
from backend.chunking.strategies import get_chunker
from backend.embeddings.embedder import embed_texts
from backend.vector_store.store import FAISSVectorStore
from backend.retrieval.retriever import retrieve_similar_contexts

def build_index(rebuild: bool = False):
    """
    Ingests dataset, chunks, computes embeddings, and builds FAISS index.
    Saves the index to disk.
    """
    store = FAISSVectorStore()
    if store.index is not None and not rebuild:
        print(f"FAISS index already exists with {len(store.documents)} documents. Skipping rebuild.")
        return store
        
    print("Initializing FAISS index building process...")
    # 1. Ingest dataset
    raw_docs = ingest_dataset(limit=MAX_DATASET_ITEMS, target_lang=DATASET_LANGUAGE)
    if not raw_docs:
        print("No documents were ingested. Check dataset configuration.")
        return None
        
    # 2. Chunk documents
    chunker = get_chunker(strategy=CHUNKER_STRATEGY, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Chunking {len(raw_docs)} passages using '{CHUNKER_STRATEGY}' strategy (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    
    all_chunks = []
    for doc in raw_docs:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
        
    print(f"Generated {len(all_chunks)} chunks from {len(raw_docs)} raw documents.")
    
    # 3. Generate embeddings
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    print(f"Generating embeddings for {len(chunk_texts)} chunks...")
    embeddings = embed_texts(chunk_texts)
    
    # 4. Add to vector store and save
    store.clear()
    store.add_documents(all_chunks, embeddings)
    store.save()
    print("FAISS index built and saved successfully.")
    return store

def run_search(query: str, k: int = None):
    """
    Performs similarity search for query and prints results.
    """
    if k is None:
        k = RETRIEVAL_TOP_K
    print(f"\nSearching for query: '{query}' (top-k: {k})")
    response = retrieve_similar_contexts(query, k=k)
    
    print(f"Retrieval Latency: {response['latency_ms']} ms")
    print(f"Found {len(response['results'])} matches:")
    for i, res in enumerate(response["results"]):
        print(f"\n[{i+1}] Cosine Similarity: {res['score']:.4f}")
        print(f"    Text: {res['text']}")
        meta = res["metadata"]
        print(f"    Query ID: {meta.get('query_id')}")
        print(f"    Passage Index: {meta.get('passage_index')} (Selected: {meta.get('is_selected')})")
        if meta.get("english_text"):
            print(f"    Original English: {meta.get('english_text')}")
            
    return response

def interactive_mode():
    """
    Runs an interactive loop for querying the index.
    """
    build_index(rebuild=False)
    print("\n=== RAGInGoa Retrieval Interface ===")
    print("Type your query to search or 'exit' to quit.")
    print("Default indexing language is set to:", DATASET_LANGUAGE)
    
    while True:
        try:
            query = input("\nEnter query: ").strip()
            if not query:
                continue
            if query.lower() in ('exit', 'quit'):
                print("Exiting interactive search mode.")
                break
            run_search(query)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive search mode.")
            break

def main():
    parser = argparse.ArgumentParser(description="RAGInGoa CLI Backend Runner")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build the FAISS vector index")
    build_parser.add_argument("--rebuild", action="store_true", help="Force rebuild the index even if it exists")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Query the FAISS vector index")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("-k", type=int, default=None, help="Number of results to return")
    
    # Start server command
    start_parser = subparsers.add_parser("start", help="Start the FastAPI backend server")
    start_parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host")
    start_parser.add_argument("--port", type=int, default=8000, help="Server port")
    start_parser.add_argument("--reload", action="store_true", help="Enable hot reload")
    
    args = parser.parse_args()
    
    if args.command == "build":
        build_index(rebuild=args.rebuild)
    elif args.command == "search":
        # Ensure index exists before searching
        store = FAISSVectorStore()
        if store.index is None:
            print("Index not found. Building it first...")
            build_index(rebuild=False)
        run_search(args.query, k=args.k)
    elif args.command == "start":
        import uvicorn
        # Ensure index is loaded or built before serving
        store = FAISSVectorStore()
        if store.index is None:
            print("FAISS Index is empty. Building index from MSMARCO-XI dataset first...")
            build_index(rebuild=False)
            
        print(f"Starting FastAPI Web Server at http://{args.host}:{args.port}")
        uvicorn.run("backend.server:app", host=args.host, port=args.port, reload=args.reload)
    else:
        # Default to interactive mode if no arguments are provided
        interactive_mode()

if __name__ == "__main__":
    main()

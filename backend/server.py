import os
import shutil
import tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import json

from backend.config import DATASET_LANGUAGE, EMBEDDING_MODEL_NAME, CHUNKER_STRATEGY, CHUNK_SIZE, CHUNK_OVERLAP, RETRIEVAL_TOP_K
from backend.chunking.strategies import get_chunker
from backend.orchestrator.pipeline import RAGOrchestrator, LOG_FILE_PATH

app = FastAPI(title="RAGInGoa – Backend API Server")

# Allow CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "null",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

orchestrator = RAGOrchestrator()

@app.get("/api/config")
def get_config():
    """
    Returns active backend configuration settings.
    """
    return {
        "dataset_language": DATASET_LANGUAGE,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunking_strategy": CHUNKER_STRATEGY,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "retrieval_top_k": RETRIEVAL_TOP_K
    }

@app.post("/api/rag")
async def post_rag(
    query: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    language: str = Form("hi-IN")
):
    """
    RAG interface accepting text queries or audio files.
    """
    if not query and not file:
        raise HTTPException(status_code=400, detail="Must provide either text 'query' or audio 'file'.")
        
    temp_file_path = None
    try:
        # If audio file is provided, write it to a temporary file
        if file:
            suffix = os.path.splitext(file.filename)[1] or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_file_path = temp_file.name
                
        # Run pipeline
        response = orchestrator.run_pipeline(
            text_query=query,
            audio_file_path=temp_file_path,
            language_code=language
        )
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        # Ensure temporary file is cleaned up
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as e:
                print(f"Error removing temp audio file {temp_file_path}: {e}")

@app.post("/api/chunking/compare")
def compare_chunking(
    text: str = Form(...),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    similarity_threshold: float = Form(0.5)
):
    """
    Receives raw text and runs it through all three chunking strategies for side-by-side comparison.
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    doc = {
        "text": text,
        "metadata": {
            "query_text": "Sample Query Context",
            "target_lang": "hin_Deva"
        }
    }
    
    try:
        # Get chunkers
        fixed_chunker = get_chunker("fixed_size", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        semantic_chunker = get_chunker("semantic", similarity_threshold=similarity_threshold)
        metadata_chunker = get_chunker("metadata", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Split
        fixed_chunks = fixed_chunker.chunk_document(doc)
        semantic_chunks = semantic_chunker.chunk_document(doc)
        metadata_chunks = metadata_chunker.chunk_document(doc)
        
        return {
            "fixed_size": [{
                "text": c["text"],
                "index": c["metadata"]["chunk_index"],
                "length": len(c["text"])
            } for c in fixed_chunks],
            "semantic": [{
                "text": c["text"],
                "index": c["metadata"]["chunk_index"],
                "length": len(c["text"])
            } for c in semantic_chunks],
            "metadata": [{
                "text": c["text"],
                "index": c["metadata"]["chunk_index"],
                "length": len(c["text"])
            } for c in metadata_chunks]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunking comparison error: {str(e)}")

@app.get("/api/analytics")
def get_analytics():
    """
    Parses logged latency measurements and computes P50, P70, P100, and averages.
    """
    if not os.path.exists(LOG_FILE_PATH):
        return {
            "p50": 0.0,
            "p70": 0.0,
            "p100": 0.0,
            "avg": 0.0,
            "count": 0,
            "details": []
        }
        
    latencies = []
    stt_latencies = []
    retrieval_latencies = []
    generation_latencies = []
    guardrails_latencies = []
    
    recent_runs = []
    
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                
                latencies.append(data.get("overall_latency_ms", 0.0))
                stt_latencies.append(data.get("stt_latency_ms", 0.0))
                retrieval_latencies.append(data.get("retrieval_latency_ms", 0.0))
                generation_latencies.append(data.get("generation_latency_ms", 0.0))
                guardrails_latencies.append(data.get("guardrails_latency_ms", 0.0))
                
                recent_runs.append({
                    "timestamp": data.get("timestamp", 0),
                    "query": data.get("query", ""),
                    "latency_ms": data.get("overall_latency_ms", 0.0),
                    "stt_ms": data.get("stt_latency_ms", 0.0),
                    "retrieval_ms": data.get("retrieval_latency_ms", 0.0),
                    "generation_ms": data.get("generation_latency_ms", 0.0),
                    "guardrails_ms": data.get("guardrails_latency_ms", 0.0),
                    "safety": data.get("safety_passed", True),
                    "grounding": data.get("grounding_passed", True)
                })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading latency logs: {str(e)}")
        
    if not latencies:
        return {
            "p50": 0.0,
            "p70": 0.0,
            "p100": 0.0,
            "avg": 0.0,
            "count": 0,
            "details": []
        }
        
    def get_percentile(arr, p):
        sorted_arr = sorted(arr)
        idx = int(round(p * (len(sorted_arr) - 1)))
        return sorted_arr[idx]
        
    return {
        "p50": round(get_percentile(latencies, 0.50), 2),
        "p70": round(get_percentile(latencies, 0.70), 2),
        "p100": round(get_percentile(latencies, 1.00), 2),
        "avg": round(sum(latencies) / len(latencies), 2),
        "count": len(latencies),
        
        "stt_p50": round(get_percentile(stt_latencies, 0.50), 2) if stt_latencies else 0.0,
        "retrieval_p50": round(get_percentile(retrieval_latencies, 0.50), 2) if retrieval_latencies else 0.0,
        "generation_p50": round(get_percentile(generation_latencies, 0.50), 2) if generation_latencies else 0.0,
        "guardrails_p50": round(get_percentile(guardrails_latencies, 0.50), 2) if guardrails_latencies else 0.0,
        
        "details": recent_runs[-10:] # Return last 10 queries
    }

# Check if frontend folder exists, and mount static folder
# We'll build the frontend shortly, so we ensure no crash if it's created next
os.makedirs("frontend", exist_ok=True)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

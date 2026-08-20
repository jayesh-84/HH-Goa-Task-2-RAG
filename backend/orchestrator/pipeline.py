import time
import os
import json
from backend.stt.sarvam_stt import speech_to_text
from backend.retrieval.retriever import retrieve_similar_contexts
from backend.llm.generator import generate_grounded_answer, get_gemini_model, generation_mode_var
from backend.guardrails.guard import check_safety, check_topic, check_context_sufficiency, validate_grounding

LOG_FILE_PATH = "./data/query_logs.jsonl"

def log_query_metrics(metrics: dict):
    """
    Appends execution metrics of a query to a local JSONL log file.
    These logs are used to compute P50, P70, P100 latencies.
    """
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    try:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error logging metrics: {e}")

class RAGOrchestrator:
    """
    Structured RAG orchestrator coordinating:
    Input Safety -> STT (optional) -> Topic Check -> Retrieval ->
    Sufficiency Check -> Generation -> Grounding Check -> Metrics Logging.
    """
    def run_pipeline(self, text_query: str = None, audio_file_path: str = None, language_code: str = "hi-IN") -> dict:
        start_time = time.perf_counter()
        
        # Initialize generation mode status
        model = get_gemini_model()
        default_mode = "REAL GEMINI + GROUNDED" if model is not None else "MOCK FALLBACK"
        generation_mode_var.set(default_mode)
        
        # Initialize latency breakdown
        latency = {
            "stt_ms": 0.0,
            "retrieval_ms": 0.0,
            "generation_ms": 0.0,
            "guardrails_ms": 0.0
        }
        
        guard_status = {
            "safety_passed": True,
            "topic_passed": True,
            "context_sufficient": True,
            "grounding_passed": True,
            "message": ""
        }
        
        # 1. Speech-to-Text (if audio is provided)
        query = text_query
        stt_transcript = None
        if audio_file_path:
            stt_start = time.perf_counter()
            try:
                stt_transcript = speech_to_text(audio_file_path, language_code=language_code)
                query = stt_transcript
            except Exception as e:
                return {
                    "query_text": "",
                    "answer": f"Speech-to-Text failed: {e}",
                    "contexts": [],
                    "generation_mode": "ERROR",
                    "guardrail_status": {**guard_status, "safety_passed": False, "message": f"STT failed: {e}"},
                    "latency_breakdown": latency,
                    "overall_latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2)
                }
            latency["stt_ms"] = round((time.perf_counter() - stt_start) * 1000.0, 2)
            
        if not query or not query.strip():
            return {
                "query_text": "",
                "answer": "I cannot answer an empty query.",
                "contexts": [],
                "generation_mode": "ERROR",
                "guardrail_status": {**guard_status, "safety_passed": False, "message": "Empty query"},
                "latency_breakdown": latency,
                "overall_latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2)
            }
            
        # 2. Safety Check (Input Guardrail)
        guard_start = time.perf_counter()
        is_safe, safety_msg = check_safety(query)
        if not is_safe:
            latency["guardrails_ms"] = round((time.perf_counter() - guard_start) * 1000.0, 2)
            return {
                "query_text": query,
                "stt_transcript": stt_transcript,
                "answer": "Your query contains inappropriate language or topics and was flagged by our safety filters.",
                "contexts": [],
                "generation_mode": generation_mode_var.get(),
                "guardrail_status": {
                    "safety_passed": False,
                    "topic_passed": False,
                    "context_sufficient": False,
                    "grounding_passed": False,
                    "message": safety_msg
                },
                "latency_breakdown": latency,
                "overall_latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2)
            }
            
        # 3. Vector Database Retrieval
        retrieval_start = time.perf_counter()
        retrieval_res = retrieve_similar_contexts(query)
        latency["retrieval_ms"] = retrieval_res["latency_ms"]
        contexts = retrieval_res["results"]
        
        # 4. Domain / Topic Check (based on retrieval scores)
        is_on_topic, topic_msg = check_topic(query, contexts)
        if not is_on_topic:
            guard_duration = (time.perf_counter() - guard_start) - (time.perf_counter() - retrieval_start)
            latency["guardrails_ms"] = round(max(guard_duration * 1000.0, 0.0), 2)
            return {
                "query_text": query,
                "stt_transcript": stt_transcript,
                "answer": "I cannot answer this question based on the provided dataset.",
                "contexts": contexts,
                "generation_mode": generation_mode_var.get(),
                "guardrail_status": {
                    "safety_passed": True,
                    "topic_passed": False,
                    "context_sufficient": False,
                    "grounding_passed": True,
                    "message": topic_msg
                },
                "latency_breakdown": latency,
                "overall_latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2)
            }
            
        # 5. Context Sufficiency Check
        is_sufficient, sufficiency_msg = check_context_sufficiency(contexts)
        if not is_sufficient:
            guard_duration = (time.perf_counter() - guard_start) - (time.perf_counter() - retrieval_start)
            latency["guardrails_ms"] = round(max(guard_duration * 1000.0, 0.0), 2)
            return {
                "query_text": query,
                "stt_transcript": stt_transcript,
                "answer": "I cannot answer this question based on the provided dataset.",
                "contexts": contexts,
                "generation_mode": generation_mode_var.get(),
                "guardrail_status": {
                    "safety_passed": True,
                    "topic_passed": True,
                    "context_sufficient": False,
                    "grounding_passed": True,
                    "message": sufficiency_msg
                },
                "latency_breakdown": latency,
                "overall_latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2)
            }
            
        # 6. LLM Grounded Answer Generation
        gen_start = time.perf_counter()
        context_texts = [ctx["text"] for ctx in contexts]
        answer = generate_grounded_answer(query, context_texts)
        latency["generation_ms"] = round((time.perf_counter() - gen_start) * 1000.0, 2)
        
        # 7. Grounding Check / Hallucination Validation
        validation_start = time.perf_counter()
        is_grounded, grounding_msg = validate_grounding(query, answer, context_texts)
        validation_time_ms = (time.perf_counter() - validation_start) * 1000.0
        
        # Combine all guardrails latencies
        total_guard_time = (time.perf_counter() - guard_start) - (time.perf_counter() - retrieval_start) - latency["generation_ms"]
        latency["guardrails_ms"] = round(max(total_guard_time, 0.0), 2)
        
        if not is_grounded:
            # Retry mechanism: regenerate answer once if validation fails
            print("Grounding check failed. Regenerating answer...")
            gen_start_retry = time.perf_counter()
            answer = generate_grounded_answer(query, context_texts)
            latency["generation_ms"] += round((time.perf_counter() - gen_start_retry) * 1000.0, 2)
            
            # Recheck
            validation_start_retry = time.perf_counter()
            is_grounded, grounding_msg = validate_grounding(query, answer, context_texts)
            latency["guardrails_ms"] += round((time.perf_counter() - validation_start_retry) * 1000.0, 2)
            
            if not is_grounded:
                # If still not grounded, fallback to safe refusal
                print("Grounding check failed on second attempt. Forcing safe refusal.")
                answer = "I cannot answer this question based on the provided dataset."
                guard_status["grounding_passed"] = False
                guard_status["message"] = grounding_msg
                
        # Overall end-to-end processing latency
        end_to_end_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        
        # Format response
        result = {
            "query_text": query,
            "stt_transcript": stt_transcript,
            "answer": answer,
            "contexts": contexts,
            "generation_mode": generation_mode_var.get(),
            "guardrail_status": {
                "safety_passed": True,
                "topic_passed": True,
                "context_sufficient": True,
                "grounding_passed": guard_status["grounding_passed"],
                "message": guard_status["message"] if not guard_status["grounding_passed"] else "All guardrails passed successfully."
            },
            "latency_breakdown": latency,
            "overall_latency_ms": end_to_end_ms
        }
        
        # Log query metrics to file for latency analysis
        log_metrics = {
            "timestamp": time.time(),
            "query": query,
            "stt_latency_ms": latency["stt_ms"],
            "retrieval_latency_ms": latency["retrieval_ms"],
            "generation_latency_ms": latency["generation_ms"],
            "guardrails_latency_ms": latency["guardrails_ms"],
            "overall_latency_ms": end_to_end_ms,
            "answer_length": len(answer),
            "safety_passed": result["guardrail_status"]["safety_passed"],
            "grounding_passed": result["guardrail_status"]["grounding_passed"]
        }
        log_query_metrics(log_metrics)
        
        return result

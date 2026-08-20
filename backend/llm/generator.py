import os
import contextvars
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Cached instances
_client_instance = None
_model_cache = {}

# Context variable to track the generation mode of the current request
# Possible values: "REAL GEMINI + GROUNDED", "MOCK FALLBACK", "ERROR"
generation_mode_var = contextvars.ContextVar("generation_mode", default="MOCK FALLBACK")

class GeminiModelWrapper:
    """
    Wrapper around the new Google GenAI client to preserve the legacy
    GenerativeModel.generate_content interface for backward compatibility
    with other modules (like the grounding guardrails).
    """
    def __init__(self, client, model_name: str, system_instruction: str = None):
        self.client = client
        self.model_name = model_name
        self.system_instruction = system_instruction

    def generate_content(self, prompt: str):
        config = types.GenerateContentConfig(
            temperature=0.0
        )
        if self.system_instruction:
            config.system_instruction = self.system_instruction
            
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config
        )

def get_gemini_client():
    """
    Returns the cached GenAI Client or initializes a new one.
    """
    global _client_instance
    if _client_instance is not None:
        return _client_instance
        
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "PASTE_MY_KEY_HERE" or not (api_key.startswith("AIzaSy") or api_key.startswith("test")):
        return None
        
    try:
        _client_instance = genai.Client(api_key=api_key)
        return _client_instance
    except Exception as e:
        print(f"Error initializing GenAI client: {e}")
        return None

def get_gemini_model(system_instruction: str = "DEFAULT_QA") -> GeminiModelWrapper:
    """
    Lazily configures the Gemini API client and returns a GeminiModelWrapper instance.
    Supports caching by system instruction key to separate Q&A assistant from facts verifier.
    """
    global _model_cache
    if system_instruction in _model_cache:
        return _model_cache[system_instruction]
        
    client = get_gemini_client()
    if client is None:
        return None
        
    if system_instruction == "DEFAULT_QA":
        instruction = (
            "You are RAGInGoa AI, a strict context-grounded Q&A assistant for Hacker House Goa 2026.\n"
            "Your job is to answer the user query based ONLY on the provided context passages.\n"
            "Rules:\n"
            "1. Answer the query concisely using the provided context passages.\n"
            "2. If the context passages are empty or do not contain the answer, reply with EXACTLY: "
            "'I cannot answer this question based on the provided dataset.' and nothing else.\n"
            "3. Do NOT make up, assume, or extrapolate any information. Avoid using any prior general knowledge."
        )
    else:
        instruction = system_instruction
        
    try:
        model = GeminiModelWrapper(
            client=client,
            model_name="gemini-3.5-flash",
            system_instruction=instruction
        )
        _model_cache[system_instruction] = model
        return model
    except Exception as e:
        print(f"Error configuring Gemini API: {e}. Running in mock fallback mode.")
        return None

def generate_mock_answer(query: str, contexts: list) -> str:
    """
    Generates a structured mock response when Gemini API key is missing.
    Looks for matching text or returns context-grounded fallback.
    """
    if not contexts:
        return "I cannot answer this question based on the provided dataset."
        
    # Standard fallback mock behavior
    # Just grab the first context and output a short statement
    first_ctx = contexts[0]
    # Simple check if there is some textual overlap
    query_words = [w.lower() for w in query.split() if len(w) > 2]
    overlap = any(w in first_ctx.lower() for w in query_words)
    
    if overlap or len(contexts) > 0:
        return f"[Mocked Answer] Based on context: {first_ctx[:150]}..."
        
    return "I cannot answer this question based on the provided dataset."

def generate_grounded_answer(query: str, contexts: list) -> str:
    """
    Generates a response to the query using retrieved contexts.
    Ensures answer is grounded in the contexts.
    """
    model = get_gemini_model("DEFAULT_QA")
    
    if model is None:
        generation_mode_var.set("MOCK FALLBACK")
        return generate_mock_answer(query, contexts)
        
    # Join contexts into a clean formatted string
    context_text = "\n\n".join([f"Passage [{i+1}]: {ctx}" for i, ctx in enumerate(contexts)])
    
    prompt = (
        f"Context passages:\n{context_text}\n\n"
        f"User Query: {query}\n\n"
        f"Grounded Answer:"
    )
    
    try:
        response = model.generate_content(prompt)
        if not response or not response.text:
            generation_mode_var.set("ERROR")
            return "Error: Received empty or malformed response from Gemini API."
            
        answer = response.text.strip()
        # Clean up any potential markdown formatting wrapping the exact fallback string
        if "cannot answer" in answer.lower() and "provided dataset" in answer.lower():
            generation_mode_var.set("REAL GEMINI + GROUNDED")
            return "I cannot answer this question based on the provided dataset."
            
        generation_mode_var.set("REAL GEMINI + GROUNDED")
        return answer
    except APIError as e:
        error_msg = f"Gemini API Error (Code {e.code}): {e.message}"
        print(error_msg)
        generation_mode_var.set("ERROR")
        return error_msg
    except Exception as e:
        error_msg = f"Gemini Connection/Execution Error: {str(e)}"
        print(error_msg)
        generation_mode_var.set("ERROR")
        return error_msg

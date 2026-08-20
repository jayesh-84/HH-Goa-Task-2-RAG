import os
import re
from backend.llm.generator import get_gemini_model

# A set of blacklisted offensive words (safe/non-explicit list for general filtering)
SAFETY_BLACKLIST = [
    "abuse", "harass", "kill", "suicide", "murder", "bomb", "terrorist", "hack",
    "f*ck", "b*tch", "bastard", "nigger"
]

def check_safety(query: str) -> tuple[bool, str]:
    """
    Checks if the input query is safe and appropriate.
    Returns (is_safe, error_message).
    """
    if not query:
        return False, "Query cannot be empty."
        
    query_lower = query.lower()
    for word in SAFETY_BLACKLIST:
        if re.search(r'\b' + re.escape(word) + r'\b', query_lower):
            return False, "Input query flagged as unsafe or inappropriate."
            
    return True, ""

def check_topic(query: str, retrieved_results: list, threshold: float = 0.30) -> tuple[bool, str]:
    """
    Checks if the query is off-topic relative to the database domain.
    Uses the similarity scores of the retrieved contexts as a proxy.
    Returns (is_on_topic, message).
    """
    if not retrieved_results:
        return False, "Query is off-topic or unrelated to the dataset."
        
    # If the top match similarity score is very low, it's off-topic
    top_score = retrieved_results[0]["score"]
    if top_score < threshold:
        return False, f"Query appears to be off-topic (relevance score: {top_score:.4f} is below threshold {threshold})."
        
    return True, ""

def check_context_sufficiency(retrieved_results: list, threshold: float = 0.35) -> tuple[bool, str]:
    """
    Checks if the retrieved context contains enough information.
    Similar to topic check but with a slightly higher/adjustable strict threshold for Q&A.
    Returns (is_sufficient, message).
    """
    if not retrieved_results:
        return False, "No relevant context found."
        
    top_score = retrieved_results[0]["score"]
    if top_score < threshold:
        return False, f"Retrieved context is insufficient to answer the query (similarity: {top_score:.4f} < {threshold})."
        
    return True, ""

def validate_grounding(query: str, answer: str, contexts: list) -> tuple[bool, str]:
    """
    Validates if the generated answer is fully grounded in the retrieved contexts.
    Detects hallucinations using the Gemini API.
    Returns (is_grounded, message).
    """
    if not contexts:
        return False, "No context provided for grounding check."
        
    # Standard check: if model output is the fallback string, it is grounded (safe refusal)
    if answer == "I cannot answer this question based on the provided dataset.":
        return True, "Safe refusal is grounded."
        
    verifier_instruction = (
        "You are an AI facts verifier.\n"
        "Your task is to determine whether the proposed answer contains any factual claims, "
        "statements, or inferences that are NOT explicitly supported by the provided context passages.\n"
        "You must respond with exactly 'YES' if the proposed answer is fully supported, or 'NO' if it contains unsupported information. "
        "Do not output anything else."
    )
    model = get_gemini_model(verifier_instruction)
    if model is None:
        # Mock validation mode when API key is missing
        # We verify that all content words in the answer are present in the contexts
        # Normalize punctuation out of the check
        clean_context = re.sub(r'[।\.!\?,\(\)\[\]\{\}:]', ' ', " ".join(contexts).lower())
        clean_answer = re.sub(r'[।\.!\?,\(\)\[\]\{\}:]', ' ', answer.lower())
        
        answer_words = [w for w in clean_answer.split() if len(w) > 3 and w not in ["mocked", "answer", "mock", "based", "context", "on", "from"]]
        if not answer_words:
            return True, "Mock validation: no content words to verify."
            
        unsupported_words = [w for w in answer_words if w not in clean_context]
        if unsupported_words:
            return False, f"Mock validation: answer contains unsupported terms not in context: {unsupported_words}"
            
        return True, "Mock validation: response is grounded."
        
    context_text = "\n\n".join([f"Passage [{i+1}]: {ctx}" for i, ctx in enumerate(contexts)])
    
    prompt = (
        "You are an AI facts verifier.\n"
        "Your task is to determine whether the proposed answer contains any factual claims, "
        "statements, or inferences that are NOT explicitly supported by the provided context passages.\n\n"
        f"Context passages:\n{context_text}\n\n"
        f"Proposed Answer: {answer}\n\n"
        "Is the Proposed Answer supported ONLY and ENTIRELY by the context passages?\n"
        "Reply with exactly 'YES' if it is fully supported, or 'NO' if it contains unsupported information or hallucinations. "
        "Do not output anything else."
    )
    
    try:
        response = model.generate_content(prompt)
        verdict = response.text.strip().upper()
        if "YES" in verdict:
            return True, "Answer is fully grounded in the retrieved contexts."
        else:
            return False, "Answer failed grounding validation (detected potential hallucination)."
    except Exception as e:
        print(f"Error during grounding validation: {e}. Defaulting to unsafe/unverified state.")
        return False, "Grounding check failed due to API execution error."

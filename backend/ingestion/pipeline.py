import re
from datasets import load_dataset
from backend.config import DATASET_NAME, DATASET_SPLIT

def clean_text(text: str) -> str:
    """
    Cleans and normalizes text by stripping whitespace, merging multiple spaces,
    and removing common artifacts.
    """
    if not text:
        return ""
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def ingest_dataset(limit: int = 1000, target_lang: str = "hin_Deva") -> list:
    """
    Ingests and processes the MSMARCO-XI dataset validation split.
    Streams items to find records with target_lang, extracts passages,
    and returns a structured list of documents ready for chunking.
    """
    print(f"Starting dataset ingestion for language '{target_lang}' (limit: {limit} queries)...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)
    
    documents = []
    queries_processed = 0
    
    for i, item in enumerate(dataset):
        item_lang = item.get("target_lang", "")
        if item_lang != target_lang:
            continue
            
        query_id = item.get("query_id")
        query_text = clean_text(item.get("query", ""))
        answer_text = clean_text(item.get("Answer", ""))
        
        passages = item.get("passages", {})
        translated_passages = passages.get("Translated_passages", [])
        english_passages = passages.get("English_passages", [])
        is_selected = passages.get("is_selected", [])
        
        # Ensure we have clean match between translated and English passages
        num_passages = len(translated_passages)
        for idx in range(num_passages):
            trans_txt = clean_text(translated_passages[idx])
            eng_txt = clean_text(english_passages[idx]) if idx < len(english_passages) else ""
            selected = bool(is_selected[idx]) if idx < len(is_selected) else False
            
            if not trans_txt:
                continue
                
            documents.append({
                "text": trans_txt,
                "metadata": {
                    "query_id": query_id,
                    "passage_index": idx,
                    "is_selected": selected,
                    "english_text": eng_txt,
                    "query_text": query_text,
                    "answer_text": answer_text,
                    "target_lang": target_lang
                }
            })
            
        queries_processed += 1
        if queries_processed >= limit:
            print(f"Reached ingestion limit of {limit} queries.")
            break
            
    print(f"Ingestion complete. Extracted {len(documents)} raw passages from {queries_processed} queries.")
    return documents

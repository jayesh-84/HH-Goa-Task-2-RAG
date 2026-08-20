import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")

# Dataset Configuration
DATASET_NAME = os.getenv("DATASET_NAME", "ai4bharat/MSMARCO-XI")
DATASET_SPLIT = os.getenv("DATASET_SPLIT", "validation")
DATASET_LANGUAGE = os.getenv("DATASET_LANGUAGE", "hin_Deva")
try:
    MAX_DATASET_ITEMS = int(os.getenv("MAX_DATASET_ITEMS", "1000"))
except ValueError:
    MAX_DATASET_ITEMS = 1000

# Embedding Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Vector Store Configuration
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")

# Chunking Configuration
CHUNKER_STRATEGY = os.getenv("CHUNKER_STRATEGY", "fixed_size")
try:
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
except ValueError:
    CHUNK_SIZE = 500

try:
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
except ValueError:
    CHUNK_OVERLAP = 50

# Retrieval Configuration
try:
    RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
except ValueError:
    RETRIEVAL_TOP_K = 3

# Ensure data directory exists
os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

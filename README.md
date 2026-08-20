# 🎙️ RAGInGoa — Voice-Enabled Multilingual RAG System

<p align="center">

  <strong>Ask. Retrieve. Ground.</strong>

  <br><br>

  A multilingual Voice-Enabled Retrieval-Augmented Generation (RAG)
  system built for <strong>Hacker House Goa 2026 — Task 2</strong>.

  <br><br>

  <img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/RAG-Retrieval%20Augmented%20Generation-purple?style=for-the-badge" />
  <img src="https://img.shields.io/badge/FAISS-Vector%20Search-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Gemini-Grounded%20Generation-4285F4?style=for-the-badge&logo=google" />
  <img src="https://img.shields.io/badge/Multilingual-Hindi%20%7C%20Hinglish-green?style=for-the-badge" />

</p>

---

## 🌟 Overview

**RAGInGoa** is a voice-enabled multilingual Retrieval-Augmented Generation system designed to answer questions using information retrieved from a provided knowledge dataset.

Instead of relying only on the language model's internal knowledge, the system first retrieves relevant context and then generates an answer grounded in that retrieved information.

### Core Pipeline

```text
🎤 Voice Input
      │
      ▼
🗣️ Speech-to-Text
      │
      ▼
🧩 Query Processing
      │
      ▼
🔢 Multilingual Embedding
      │
      ▼
📦 FAISS Vector Retrieval
      │
      ▼
🛡️ Guardrails
      │
      ▼
🧠 Gemini Grounded Generation
      │
      ▼
💬 Final Answer
```

The system is designed to know **when to answer and when not to answer**.

---

# 🎯 Problem Statement

The goal of Hacker House Goa Task 2 is to build a voice-enabled RAG system where:

> A user speaks a question → the system transcribes it → retrieves relevant information → generates a grounded answer.

The project uses the provided **AI4Bharat MSMARCO-XI** dataset as the knowledge source.

---

# ✨ Key Features

### 🎤 Voice Input

- Voice-based question input
- Speech-to-text processing
- Hindi voice interaction
- Support for multilingual queries

### 🌐 Multilingual RAG

Supports queries such as:

```text
निगम क्या है?
```

and:

```text
corporation kya hai?
```

The system processes multilingual queries and retrieves relevant context.

### 🔎 Semantic Retrieval

The system converts text into vector embeddings and searches the FAISS index for the most relevant passages.

### 🧠 Grounded Generation

Gemini generates responses using retrieved context instead of answering purely from general model knowledge.

### 🛡️ Guardrails

The system includes checks for:

- Unsafe queries
- Off-topic queries
- Insufficient context
- Low-similarity retrieval results
- Ungrounded answers

If the required context is not available, the system refuses to fabricate an answer.

Example:

```text
I cannot answer this question based on the provided dataset.
```

### 📊 Retrieval Information

The system can track:

- Retrieved passages
- Similarity scores
- Top-K retrieval
- Grounding status
- Safety status
- Query processing information
- Latency

---

# 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │     User Voice      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Speech-to-Text    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Query Processing  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Multilingual        │
                    │ Embedding Model     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    FAISS Index      │
                    │  Vector Retrieval   │
                    └──────────┬──────────┘
                               │
                         Top-K Context
                               │
                               ▼
                    ┌─────────────────────┐
                    │     Guardrails      │
                    │ Safety + Relevance  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Gemini Generation   │
                    │ Grounded Response   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Final Answer     │
                    └─────────────────────┘
```

---

# 🧩 Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| Frontend | HTML, CSS, JavaScript |
| Backend | Python |
| Speech-to-Text | Sarvam / configured STT provider |
| LLM | Google Gemini |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| Vector Database | FAISS |
| Dataset | AI4Bharat MSMARCO-XI |
| API | REST |
| Testing | Pytest |
| Environment | Python `.env` configuration |

---

# 📂 Project Structure

```text
HH-Goa-Task-2-RAG/
│
├── backend/
│   ├── ...
│   └── ...
│
├── frontend/
│   ├── index.html
│   ├── index.css
│   └── index.js
│
├── data/
│   ├── faiss_index/
│   └── query_logs.jsonl
│
├── tests/
│   ├── test_chunking.py
│   ├── test_guardrails.py
│   ├── test_ingestion.py
│   ├── test_llm.py
│   ├── test_orchestrator.py
│   ├── test_retrieval.py
│   ├── test_stt.py
│   └── test_vector_store.py
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

# ⚙️ Configuration

Create a `.env` file from `.env.example`.

```env
SARVAM_API_KEY=your_sarvam_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

DATASET_NAME=ai4bharat/MSMARCO-XI
DATASET_SPLIT=validation
DATASET_LANGUAGE=hin_Deva
MAX_DATASET_ITEMS=1000

EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

FAISS_INDEX_PATH=./data/faiss_index

CHUNKER_STRATEGY=fixed_size
CHUNK_SIZE=500
CHUNK_OVERLAP=50

RETRIEVAL_TOP_K=3
```

> ⚠️ **Never commit `.env` or expose API keys.**
>
> Only `.env.example` should be included in the repository.

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/jayesh-84/HH-Goa-Task-2-RAG.git
cd HH-Goa-Task-2-RAG
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure environment variables

```bash
copy .env.example .env
```

Then add your API keys to `.env`.

---

# ▶️ Running the Application

## Start Backend

```bash
python -m backend.main start
```

Backend:

```text
http://127.0.0.1:8000
```

## Start Frontend

Open another terminal:

```bash
python -m http.server 5500 --directory frontend
```

Frontend:

```text
http://localhost:5500
```

Open the frontend in your browser:

```text
http://localhost:5500
```

---

# 🧪 Example Queries

### Hindi

```text
निगम क्या है?
```

### Hinglish

```text
corporation kya hai?
```

### English

```text
What is Broadcom Corporation?
```

### Out-of-domain query

```text
Who is the Prime Minister of India?
```

The system should reject questions that cannot be answered from the available dataset.

---

# 🛡️ Guardrail Strategy

RAGInGoa uses multiple checks before returning an answer.

```text
User Query
    │
    ▼
Safety Check
    │
    ├── Unsafe → Reject
    │
    ▼
Retrieval
    │
    ▼
Similarity Check
    │
    ├── Insufficient Context → Reject
    │
    ▼
Context Sufficiency
    │
    ├── Insufficient → Grounded Refusal
    │
    ▼
Gemini Generation
    │
    ▼
Grounding Validation
    │
    ├── Failed → Reject / Regenerate
    │
    ▼
Final Answer
```

This prevents the system from blindly generating answers when relevant evidence is unavailable.

---

# 📈 Retrieval

The current system uses:

```text
Embedding Model:
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Vector search:

```text
FAISS
```

Current retrieval configuration:

```text
Top-K = 3
Chunk Size = 500
Chunk Overlap = 50
```

Retrieved passages are passed to the generation layer as grounding context.

---

# ⚡ Performance

The system records pipeline latency and retrieval information for evaluation.

During verification, the initial end-to-end query showed approximately:

```text
23.71 seconds
```

This initial measurement included lazy loading of the embedding model and model weights. Subsequent queries are expected to be faster after initialization.

> Performance benchmarking should be measured across multiple queries rather than relying on a single best-case result.

---

# 🧪 Testing

The repository contains automated tests covering major RAG components.

```text
tests/
├── test_chunking.py
├── test_guardrails.py
├── test_ingestion.py
├── test_llm.py
├── test_orchestrator.py
├── test_retrieval.py
├── test_stt.py
└── test_vector_store.py
```

Run the test suite:

```bash
pytest
```

For coverage:

```bash
pytest --cov
```

---

# 🔌 API

The backend exposes REST endpoints for communication between the frontend and RAG pipeline.

Example:

```text
GET /api/config
POST /api/rag
```

The `/api/rag` endpoint handles the RAG query flow:

```text
Query
  ↓
Embedding
  ↓
FAISS Retrieval
  ↓
Guardrails
  ↓
Gemini
  ↓
Grounded Response
```

---

# 🔐 Security

API credentials are stored using environment variables.

```text
.env
```

is excluded from Git using:

```gitignore
.env
```

Never place API keys directly inside:

- JavaScript
- HTML
- README
- GitHub commits
- Screenshots
- Public configuration files

---

# 📸 Demo

### RAG Lab

The application provides a dedicated interface for:

- Voice input
- Speech transcription
- Retrieved context
- Grounded answers
- Retrieval information
- RAG system status

### Example Flow

```text
🎤 Speak
   ↓
📝 Transcription
   ↓
🔎 Retrieve
   ↓
🛡️ Validate
   ↓
🧠 Generate
   ↓
💬 Grounded Answer
```

---

# 🎥 Hackathon Submission

### Project

**RAGInGoa — Voice-Enabled Multilingual RAG System**

### Event

**Hacker House Goa 2026**

### Task

**Task 2 — Voice-Enabled RAG Model**

### Repository

https://github.com/jayesh-84/HH-Goa-Task-2-RAG

### Live Demo

> Add your deployed/live URL here.

### Demo Video

> Add your demo video link here.

### Team/Process Video

> Add your 90-second team/process video link here.

---

# 🎯 Design Principles

RAGInGoa is built around four principles:

### 1. Retrieve Before Generate

The model should use retrieved evidence instead of relying only on parametric knowledge.

### 2. Ground Every Answer

Responses should be supported by retrieved context.

### 3. Know When Not to Answer

If the system cannot find sufficient evidence, it should refuse instead of hallucinating.

### 4. Multilingual by Design

The system is designed to handle multilingual and Hinglish queries.

---

# 🏆 Why RAGInGoa?

Traditional LLM applications can confidently answer questions even when they do not have reliable information.

RAGInGoa takes a different approach:

```text
                 Traditional LLM
                       │
                 User Question
                       │
                       ▼
                 Model Answer
                       │
                 ❓ Hallucination Risk


                    RAGInGoa
                       │
                 User Question
                       │
                       ▼
                  Retrieval
                       │
                       ▼
                Relevant Context
                       │
                       ▼
                 Guardrails
                       │
                       ▼
              Grounded Generation
                       │
                       ▼
                Reliable Answer
```

---

# 👨‍💻 Developer

**Jayesh Patil**

GitHub:  
https://github.com/jayesh-84

---

# 📜 License

This project was developed as part of the **Hacker House Goa 2026 Task 2** evaluation.

---

<p align="center">

### 🎙️ RAGInGoa

<strong>Ask. Retrieve. Ground.</strong>

Built for Hacker House Goa 2026 🚀

</p>
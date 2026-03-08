# RAG Q&A System

A production-ready **Retrieval-Augmented Generation (RAG)** API that lets you upload documents and ask questions against them — with cited, AI-powered answers.

Built with FastAPI, LangChain, Qdrant Cloud, and OpenAI.

---

## Features

- Upload **PDF**, **TXT**, and **CSV** documents
- Ask natural language questions and receive grounded answers with source references
- Streaming responses for real-time feedback
- Built-in **RAGAS evaluation** to measure answer quality (faithfulness, answer relevancy)
- LangSmith tracing for full observability
- Dockerized with a multi-stage build (`python:3.13-slim`) and non-root user for production safety
- CI/CD pipeline via GitHub Actions

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| RAG Orchestration | LangChain (LCEL) |
| Vector Database | Qdrant Cloud |
| Embeddings | OpenAI `text-embedding-ada-002` (configurable) |
| LLM | OpenAI `gpt-4o-mini` (configurable) |
| Evaluation | RAGAS |
| Observability | LangSmith |
| Containerization | Docker (multi-stage, `python:3.13-slim`) |

---

## Project Structure

```
Basic-RAG/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── documents.py   # Upload, info, delete collection
│   │   │   ├── health.py      # Health + readiness checks
│   │   │   └── query.py       # Query, streaming, document search
│   │   └── schemas.py         # Pydantic request/response models
│   ├── core/
│   │   ├── document_processor.py  # PDF/TXT/CSV chunking
│   │   ├── embeddings.py          # Embedding service
│   │   ├── rag_chain.py           # RAG chain (sync + async)
│   │   ├── ragas_evaluator.py     # RAGAS evaluation (lazy-loaded)
│   │   └── vector_store.py        # Qdrant vector store service
│   ├── utils/
│   │   └── logger.py          # Structured logging (structlog)
│   ├── config.py              # Pydantic settings (loaded from .env)
│   └── main.py                # App entry point, lifespan, middleware
├── .github/
│   └── workflows/             # GitHub Actions CI/CD
├── Dockerfile                 # Multi-stage production Dockerfile
├── requirements.txt
├── .env.example               # Template — copy to .env and fill in values
└── .gitignore
```

---

## Getting Started

### Prerequisites

- Python 3.13+
- A [Qdrant Cloud](https://cloud.qdrant.io/) account (free tier works)
- An [OpenAI](https://platform.openai.com/) API key
- (Optional) A [LangSmith](https://smith.langchain.com/) account for tracing

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Basic-RAG
```

### 2. Set up environment variables

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
```

### 3. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the application

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

---

## Running with Docker

### Build the image

```bash
docker build -t rag-qa-system .
```

### Run the container

```bash
docker run -p 8000:8000 --env-file .env rag-qa-system
```

---

## API Endpoints

### Health

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Basic health check |
| `GET` | `/health/ready` | Readiness check — verifies Qdrant connectivity |

### Documents

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents/upload` | Upload a document (PDF, TXT, or CSV) |
| `GET` | `/documents/info` | Get collection info (document count, status) |
| `DELETE` | `/documents/collection` | Delete the entire vector store collection |

### Query

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/query` | Ask a question; optionally include sources and RAGAS evaluation |
| `POST` | `/query/stream` | Ask a question with streaming text response |
| `POST` | `/query/search` | Retrieve relevant document chunks without generating an answer |

### Docs & Root

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | App info |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc UI |

---

## Configuration Reference

All settings are loaded from `.env` via Pydantic Settings.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key |
| `QDRANT_URL` | — | **Required.** Qdrant cluster URL |
| `QDRANT_API_KEY` | — | **Required.** Qdrant API key |
| `COLLECTION_NAME` | `rag_documents` | Qdrant collection name |
| `EMBEDDING_MODEL` | `text-embedding-ada-002` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI LLM model |
| `LLM_TEMPERATURE` | `0.0` | LLM temperature |
| `CHUNK_SIZE` | `1000` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Chunk overlap (characters) |
| `RETRIEVAL_K` | `3` | Number of chunks retrieved per query |
| `ENABLE_RAGAS_EVALUATION` | `true` | Enable RAGAS answer evaluation |
| `RAGAS_TIMEOUT_SECONDS` | `30.0` | Timeout for RAGAS evaluation |
| `RAGAS_LOG_RESULTS` | `true` | Log RAGAS scores to stdout |
| `RAGAS_LLM_MODEL` | *(uses `LLM_MODEL`)* | Override LLM for RAGAS |
| `RAGAS_EMBEDDING_MODEL` | *(uses `EMBEDDING_MODEL`)* | Override embeddings for RAGAS |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith tracing |
| `LANGSMITH_ENDPOINT` | `https://api.smith.langchain.com` | LangSmith endpoint |
| `LANGSMITH_API_KEY` | — | LangSmith API key |
| `LANGSMITH_PROJECT` | `rag-qa-production` | LangSmith project name |
| `API_HOST` | `0.0.0.0` | Host to bind |
| `API_PORT` | `8000` | Port to bind |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins (comma-separated) |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## RAGAS Evaluation

When `ENABLE_RAGAS_EVALUATION=true`, each `/query` request (with evaluation enabled in the body) is automatically scored using [RAGAS](https://docs.ragas.io/) metrics:

- **Faithfulness** — is the answer grounded in the retrieved context?
- **Answer Relevancy** — does the answer actually address the question?

The evaluator is **lazy-loaded** on first use. Results are logged via `structlog` and optionally traced in LangSmith. Evaluation failures are caught gracefully and returned as `null` scores — they won't break your query response.

---

## `.env.example` Template

```env
# Required
OPENAI_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=

# Collection
COLLECTION_NAME=rag_documents

# Models
EMBEDDING_MODEL=text-embedding-ada-002
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0

# Chunking
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval
RETRIEVAL_K=3

# Logging
LOG_LEVEL=INFO

# API
API_HOST=0.0.0.0
API_PORT=8000
ALLOWED_ORIGINS=*

# RAGAS
ENABLE_RAGAS_EVALUATION=true
RAGAS_TIMEOUT_SECONDS=30.0
RAGAS_LOG_RESULTS=true

# LangSmith (optional)
LANGSMITH_TRACING=false
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=rag-qa-production
```

---

## License

MIT

# RAG Q&A System

A production-ready **Retrieval-Augmented Generation (RAG)** API that lets you upload documents and ask questions against them — with cited, AI-powered answers.

Built with FastAPI, LangChain, Qdrant Cloud, and OpenAI.

---

## Features

- Upload **PDF**, **TXT**, and **CSV** documents
- Ask natural language questions and receive grounded answers with source references
- Streaming responses for real-time feedback
- Built-in **RAGAS evaluation** to measure answer quality
- LangSmith tracing for full observability
- Dockerized with a multi-stage build and non-root user for production safety
- CI/CD pipeline via GitHub Actions

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| RAG Orchestration | LangChain |
| Vector Database | Qdrant Cloud |
| Embeddings | OpenAI `text-embedding-3-small` |
| LLM | OpenAI `gpt-4o-mini` |
| Evaluation | RAGAS |
| Observability | LangSmith |
| Containerization | Docker (multi-stage) |

---

## Project Structure

```
Basic-RAG/
├── app/
│   ├── api/
│   │   └── routes/         # FastAPI route handlers (documents, query, health)
│   ├── core/
│   │   ├── rag_chain.py    # RAG chain logic
│   │   └── vector_store.py # Qdrant vector store service
│   ├── utils/
│   │   └── logger.py       # Structured logging setup
│   ├── config.py           # Pydantic settings (loaded from .env)
│   └── main.py             # App entry point, lifespan, middleware
├── .github/
│   └── workflows/          # GitHub Actions CI/CD
├── Dockerfile              # Multi-stage production Dockerfile
├── requirements.txt
└── .env.example            # Template for required environment variables
```

---

## Getting Started

### Prerequisites

- Python 3.12+
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

Open `.env` and set:

```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
LANGSMITH_API_KEY=your_langsmith_api_key   # optional
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

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root — app info |
| `GET` | `/health` | Health check |
| `POST` | `/documents/upload` | Upload a document (PDF, TXT, CSV) |
| `POST` | `/query` | Ask a question against uploaded documents |
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
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI LLM model |
| `LLM_TEMPERATURE` | `0.0` | LLM temperature |
| `CHUNK_SIZE` | `1000` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Chunk overlap (characters) |
| `RETRIEVAL_K` | `4` | Number of chunks retrieved per query |
| `ENABLE_RAGAS_EVALUATION` | `true` | Enable RAGAS answer evaluation |
| `LANGSMITH_TRACING` | `false` | Enable LangSmith tracing |
| `API_HOST` | `0.0.0.0` | Host to bind |
| `API_PORT` | `8000` | Port to bind |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins (comma-separated) |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## RAGAS Evaluation

When `ENABLE_RAGAS_EVALUATION=true`, each query response is automatically evaluated using [RAGAS](https://docs.ragas.io/) metrics including faithfulness, answer relevancy, and context recall. Results are logged to stdout and optionally to LangSmith.

---

## Environment Variable Template

Create a `.env.example` in the project root with this content (no real values):

```env
OPENAI_API_KEY=
QDRANT_URL=
QDRANT_API_KEY=
COLLECTION_NAME=rag_documents
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
RETRIEVAL_K=4
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
ALLOWED_ORIGINS=*
ENABLE_RAGAS_EVALUATION=true
RAGAS_TIMEOUT_SECONDS=30.0
LANGSMITH_TRACING=false
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=rag-qa-production
```

---

## License

MIT

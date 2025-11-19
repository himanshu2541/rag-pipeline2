# Modular RAG System with FastAPI, LangChain & Pinecone

A robust, class-based Retrieval Augmented Generation (RAG) system built with FastAPI and LangChain. This system features a hybrid search architecture using BM25 (sparse/keyword) and Pinecone (dense/semantic) retrieval.

It is designed to be modular, supporting both OpenAI's embeddings and local HuggingFace models (like all-MiniLM-L6-v2).

---

## Installation

### Option 1: Using uv (Recommended)

This project uses **uv** for extremely fast package management.

1. **Install uv (if not already installed):**

```bash
# On macOS/Linux
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
# On Windows
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

2. **Sync dependencies from uv.lock:**
This will create the virtual environment and install the exact locked versions.

```bash
uv sync
```

3. Activate the environment:

```bash
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate   # On Windows
```

### Option 2: Using Standard pip

If you prefer standard Python tools or don't use uv:

1. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

## Pinecone Initialization (Critical Step)

Before running the app, you must create an index in the [Pinecone Console](https://app.pinecone.io/). The settings must match your chosen embedding model.

### Configuration for Local Model (**all-MiniLM-L6-v2**)

If you are using the local model configured in this project:

Index Name: **rag-local-384** (or similar)

Dimensions: **384** (Critical! Do not use 1536)

Metric: **cosine**

Capacity Mode: **Serverless** (Recommended)

### Configuration for OpenAI Model (text-embedding-3-small)

If you switch config to use OpenAI embeddings:

Index Name: **rag-openai-1536**

Dimensions: **1536**

Metric: **cosine**

### Configuration

1. Copy the example environment file:

```bash
cp .env.example .env
```

2. Open **.env** and configure your keys:

```bash
OPENAI_API_KEY="sk-..."
PINECONE_API_KEY="..."
PINECONE_INDEX_NAME="your-index-name"

# To use the local model (Free embeddings):
EMBEDDING_PROVIDER="local"
EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

## Running the App

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at http://127.0.0.1:8000.
Interactive Docs: http://127.0.0.1:8000/docs.

📡 Usage Examples

1. Upload a Document (/upload)

Upload a text file to be ingested (split, embedded, and stored in Pinecone).

```bash
curl -X POST "[http://127.0.0.1:8000/upload](http://127.0.0.1:8000/upload)" \
     -F "file=@/path/to/your/document.txt"
```

2. Chat with Your Data (/chat)

Ask questions based on the uploaded content.

```bash
curl -X POST "[http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat)" \
     -H "Content-Type: application/json" \
     -d '{"query": "What are the key points in the document?"}'
```

## Architecture

- Ingestion: Text Loader -> Recursive Splitter -> HuggingFace/OpenAI Embeddings -> Pinecone.

- Retrieval: Hybrid Ensemble (BM25 for keywords + Pinecone for semantics).

- Generation: GPT-4o-mini (via LangChain).

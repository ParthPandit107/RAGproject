# Cost-Efficient RAG Application

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system capable of answering questions from PDF, HTML and Markdown documents.

## Features

- PDF, HTML and Markdown ingestion
- Token-aware chunking
- SentenceTransformer embeddings
- LanceDB vector database
- Google Gemini integration
- FastAPI REST API
- Idempotent re-ingestion
- Source attribution

## Tech Stack

- Python
- FastAPI
- LanceDB
- SentenceTransformers
- Google Gemini
- BeautifulSoup
- PyMuPDF

## Run


### Install dependencies

```bash
pip install -r requirements.txt
```

### Ingest documents

```bash
python ingest.py
```

### Start the API

```bash
uvicorn app:app --reload
```

### Open Swagger UI

```
http://127.0.0.1:8000/docs
```

### Run Evaluation

```bash
python eval/evaluate.py
```


## Cost Comparison

| Component | Selected | Why |
|-----------|----------|-----|
| LLM | Gemini Flash | Fast inference with a generous free tier |
| Vector DB | LanceDB | Local vector database, no hosting cost |
| Embeddings | all-MiniLM-L6-v2 |384-dimensional embeddings with good retrieval quality |
| Framework | FastAPI | Lightweight REST API with automatic OpenAPI documentation |
## Architecture

```text
Documents
    │
    ▼
Loader
    │
    ▼
Token Chunking
    │
    ▼
SentenceTransformer
    │
    ▼
LanceDB
    │
    ▼
User Query
    │
    ▼
Query Embedding
    │
    ▼
Semantic Search
    │
    ▼
Gemini
    │
    ▼
Answer + Sources
```
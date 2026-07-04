# Cost-Efficient RAG Application

A simple RAG system to ingest PDF, HTML, and Markdown files and ask questions using LanceDB and Google Gemini.

## Features

- Ingests PDF, HTML, and Markdown
- Token-aware chunking (with overlap check)
- Local embeddings using SentenceTransformers (`all-MiniLM-L6-v2`)
- Local vector storage using LanceDB (no hosting cost)
- Google Gemini integration
- FastAPI with `/ask` and `/retrieve` endpoints
- Robust evaluation script that caches Gemini responses to avoid rate limits

## Tech Stack

- Python 3.12
- FastAPI
- LanceDB
- SentenceTransformers
- Google Gemini
- BeautifulSoup4
- PyMuPDF

## How to Run

### 1. Setup Environment
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Ingest Documents
Put your documents in the `data` folder, then run:
```bash
python ingest.py
```

### 4. Start the API
```bash
uvicorn app:app --reload
```
The API will run at `http://127.0.0.1:8000`. You can open `http://127.0.0.1:8000/docs` to test the endpoints in Swagger UI.

- `POST /ask`: Retrieves context and generates an answer using Gemini.
- `POST /retrieve`: Performs a local semantic search and returns the matched text chunks and sources (doesn't hit Gemini).

### 5. Run Evaluation
```bash
python eval/evaluate.py
```
*Note: To avoid hitting the Gemini free-tier daily quota and rate limits, the evaluation script caches LLM responses in `eval/cache.json`. If Gemini is rate-limited or the daily limit is reached, it will skip LLM calls for remaining questions but still evaluate retrieval performance locally.*

## Cost Comparison

| Component | Selected | Why |
|-----------|----------|-----|
| LLM | Gemini Flash | Fast and has a generous free tier |
| Vector DB | LanceDB | Local database, zero hosting cost |
| Embeddings | all-MiniLM-L6-v2 | 384-dimensional local embeddings with good retrieval quality |
| Framework | FastAPI | Lightweight, fast, and generates Swagger docs |
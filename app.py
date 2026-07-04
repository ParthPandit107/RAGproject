from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import os

import lancedb
import google.generativeai as genai

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Gemini API key not found.")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

embedding_model = SentenceTransformer(EMBEDDING_MODEL)

db = lancedb.connect("db")

llm = genai.GenerativeModel("gemini-2.5-flash")


class Question(BaseModel):
    question: str


def get_table():

    try:
        return db.open_table("documents")

    except Exception as e:
        raise RuntimeError("LanceDB table 'documents' not found. Please run ingest.py first.") from e


def retrieve(query, k=3):

    table = get_table()

    embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = (
        table.search(embedding)
        .limit(k)
        .to_list()
    )

    return results


def build_prompt(question, contexts):

    context_text = ""

    for item in contexts:
        context_text += item["text"] + "\n\n"

    prompt = f"""
You are a helpful assistant.

Answer ONLY using the provided context.

If the answer is not in the context,
say that you do not know.

Context:
{context_text}

Question:
{question}

Answer:
"""

    return prompt


def ask_gemini(prompt):

    response = llm.generate_content(prompt)
    return response.text


@app.post("/retrieve")
def retrieve_chunks(question: Question):

    try:
        contexts = retrieve(question.question)

        results = []

        for item in contexts:
            results.append(
                {
                    "source": item["source"],
                    "chunk_index": item["chunk_index"],
                    "text": item["text"]
                }
            )

        return results

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask(question: Question):

    try:
        contexts = retrieve(question.question)

    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    prompt = build_prompt(
        question.question,
        contexts
    )

    try:
        answer = ask_gemini(prompt)

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {str(e)}")

    sources = []

    for item in contexts:
        sources.append(
            {
                "source": item["source"],
                "chunk_index": item["chunk_index"]
            }
        )

    return {
        "question": question.question,
        "answer": answer,
        "sources": sources
    }
from fastapi import FastAPI
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

model = SentenceTransformer(EMBEDDING_MODEL)

db = lancedb.connect("db")
table = db.open_table("documents")

llm = genai.GenerativeModel("gemini-2.5-flash")


class Question(BaseModel):
    question: str

def retrieve(query, k=3):

    embedding = model.encode(
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


@app.post("/ask")
def ask(question: Question):

    contexts = retrieve(question.question)

    prompt = build_prompt(
        question.question,
        contexts
    )

    answer = ask_gemini(prompt)

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
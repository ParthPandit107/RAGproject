from pathlib import Path
import hashlib
import fitz
from bs4 import BeautifulSoup
import lancedb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
DB_DIR = "db"


CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBEDDING_MODEL)
tokenizer = model.tokenizer
db = lancedb.connect(DB_DIR)

def load_pdf(file_path):
    doc = fitz.open(str(file_path))

    text = []

    for page in doc:
        text.append(page.get_text())

    doc.close()

    return "\n".join(text)


def load_html(file_path):

    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    return soup.get_text(separator="\n")


def load_markdown(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    
def load_document(file_path):

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path)

    elif suffix in [".html", ".htm"]:
        return load_html(file_path)

    elif suffix == ".md":
        return load_markdown(file_path)

    else:
        raise ValueError(f"Unsupported file type: {suffix}")   

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    chunks = []

    start = 0

    while start < len(token_ids):

        end = start + chunk_size

        chunk = tokenizer.decode(token_ids[start:end])

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

def create_embedding(text):
    embedding = model.encode(
        text,
        normalize_embeddings=True
    )

    return embedding.tolist()

def generate_hash(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()




def ingest_documents():
    rows = []

    for file_path in DATA_DIR.iterdir():

        if not file_path.is_file():
            continue

        print(f"Processing {file_path.name}")

        text = load_document(file_path)

        chunks = chunk_text(text)

        embeddings = model.encode(
            chunks,
            normalize_embeddings=True
        ).tolist()

        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

            rows.append(
                {
                    "id": generate_hash(chunk),
                    "text": chunk,
                    "embedding": embedding,
                    "source": file_path.name,
                    "chunk_index": index,
                }
            )

    if not rows:
        print("No documents found.")
        return
    try:
        table = db.open_table("documents")
        table_exists = True
    except Exception:
        table_exists = False
    
    existing_ids = set()

    if table_exists:

        existing = table.to_pandas()

        if "id" in existing.columns:
            existing_ids = set(existing["id"])

    new_rows = [
        row
        for row in rows
        if row["id"] not in existing_ids
    ]

    if new_rows:

        if table_exists:

            table.add(new_rows)

        else:

            table = db.create_table(
                "documents",
                data=new_rows
            )

        print(f"Inserted {len(new_rows)} new chunks.")

    else:

        print("Database already up to date.")

if __name__ == "__main__":
    ingest_documents()
import pickle

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

PDF_FILE = "python.pdf"
INDEX_FILE = "rag.index"
CHUNKS_FILE = "chunks.pkl"

CHUNK_SIZE = 500


def clean_text(text: str) -> str:
    return text.encode("utf-8", "ignore").decode("utf-8")


def extract_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)

    pages = []

    for page_no, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()

            if page_text:
                page_text = clean_text(page_text)
                pages.append(page_text)

        except Exception as e:
            print(f"Skipping page {page_no}: {e}")

    return "\n".join(pages)


def create_chunks(text: str, chunk_size: int):
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunk = text[i : i + chunk_size]

        if chunk:
            chunk = str(chunk).strip()

            if chunk:
                chunks.append(chunk)

    return chunks


def main():
    print("Reading PDF...")

    text = extract_text(PDF_FILE)

    print(f"Total text length: {len(text)}")

    chunks = create_chunks(text, CHUNK_SIZE)

    print(f"Total chunks: {len(chunks)}")

    if not chunks:
        raise Exception("No chunks created.")

    print("\nValidating chunks...")

    bad_chunks = [
        (i, type(chunk)) for i, chunk in enumerate(chunks) if not isinstance(chunk, str)
    ]

    if bad_chunks:
        print("Bad chunks found:")
        print(bad_chunks[:10])
        raise Exception("Chunk validation failed.")

    print("Chunk validation passed")

    print("\nLoading embedding model...")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Generating embeddings...")

    embeddings = model.encode(
        chunks, batch_size=32, show_progress_bar=True, convert_to_numpy=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    print("Embedding shape:", embeddings.shape)

    dimension = embeddings.shape[1]

    print("\nCreating FAISS index...")

    index = faiss.IndexFlatL2(dimension)  # Eucledian Distance

    index.add(embeddings)

    print("Vectors stored:", index.ntotal)

    print("\nSaving index...")

    faiss.write_index(index, INDEX_FILE)

    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print("Done!")
    print(f"Index: {INDEX_FILE}")
    print(f"Chunks: {CHUNKS_FILE}")


if __name__ == "__main__":
    main()

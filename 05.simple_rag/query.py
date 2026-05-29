import pickle

import faiss
from sentence_transformers import SentenceTransformer

# Load
index = faiss.read_index("rag.index")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


while True:
    query = input("\nQuestion: ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, k=3)

    print("\nRelevant Context:\n")

    for idx in indices[0]:
        print("-" * 60)
        print(chunks[idx])

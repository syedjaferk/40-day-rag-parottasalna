import ollama
import chromadb

# FLOW 1
def ollama_embed(texts):
    embeddings = []
    
    for t in texts:
        res = ollama.embeddings(
            model="nomic-embed-text",
            prompt=t
        )
        embeddings.append(res["embedding"])
    
    return embeddings

docs = [
    "Python is a programming language",
    "FastAPI is great for building APIs",
    "ChromaDB is a vector database",
    "I love machine learning",
    "PostgreSQL is a relational Database",

]

client = chromadb.PersistentClient(path="./chroma_storage")
collection = client.get_or_create_collection("ollama_demo")

# Create embeddings via Ollama
emb = ollama_embed(docs)

collection.add(
    documents=docs,
    embeddings=emb,
    ids=[str(i) for i in range(len(docs))]
)

print("Added using Ollama embeddings!")


# FLOW 2

query = "tell me about vector database"

q_emb = ollama_embed([query])

results = collection.query(
    query_embeddings=q_emb,
    n_results=2
)

print(results["documents"])
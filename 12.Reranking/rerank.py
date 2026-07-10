import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# STEP 1 : Load Models

print("Loading Embedding Model...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading ReRanker...")

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

# STEP 2 : Create Chroma Client

client = chromadb.Client()

collection_name = "knowledge_base"

# Delete existing collection if it exists
try:
    client.delete_collection(collection_name)
except:
    pass

collection = client.create_collection(collection_name)

# STEP 3 : Documents

documents = [

    "Python is an interpreted programming language used for web development, AI and scripting.",

    "FastAPI is a modern Python web framework that supports asynchronous programming.",

    "MongoDB transactions provide ACID guarantees and work only on replica sets.",

    "Java introduced Records in Java 16 for immutable data classes.",

    "MongoDB transactions can span multiple collections and multiple databases.",

    "Docker containers package applications together with their dependencies.",

    "Redis is an in-memory key-value database commonly used for caching.",

    "PostgreSQL supports MVCC and ACID transactions.",

    "MongoDB replica sets provide high availability by maintaining multiple copies of data.",

    "Flask is a lightweight Python web framework."

]

ids = [str(i) for i in range(len(documents))]

# STEP 4 : Generate Embeddings

print("\nGenerating embeddings...\n")

embeddings = embedding_model.encode(documents).tolist()

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print("Documents inserted into ChromaDB")

# STEP 5 : User Query

query = "How do MongoDB transactions work?"

print("\nUser Query:")
print(query)

query_embedding = embedding_model.encode(query).tolist()

# STEP 6 : Retrieve Similar Documents

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

retrieved_documents = results["documents"][0]
retrieved_distances = results["distances"][0]

print("\n")
print("=" * 60)
print("VECTOR SEARCH RESULTS")
print("=" * 60)



for i, (doc, distance) in enumerate(zip(retrieved_documents, retrieved_distances), start=1):
    print(f"\nRank {i}")
    print("Distance :", round(distance, 4))
    print(doc)

input("Press enter to re rank")


# STEP 7 : Prepare Query-Document Pairs

pairs = []

for doc in retrieved_documents:
    pairs.append([query, doc])

# STEP 8 : Re-Rank

scores = reranker.predict(pairs)

# STEP 9 : Sort by Score

reranked_results = sorted(
    zip(retrieved_documents, scores),
    key=lambda x: x[1],
    reverse=True
)

print("\n")
print("=" * 60)
print("AFTER RE-RANKING")
print("=" * 60)

for i, (doc, score) in enumerate(reranked_results, start=1):

    print(f"\nRank {i}")
    print("Score :", round(float(score), 4))
    print(doc)

# STEP 10 : Select Top Documents

top_k = 3

top_documents = [
    doc for doc, score in reranked_results[:top_k]
]

# STEP 11 : Build Prompt

context = "\n\n".join(top_documents)

prompt = f"""
You are a helpful assistant.

Answer only using the provided context.

------------------------

Context

{context}

------------------------

Question

{query}
"""

print("\n")
print("=" * 60)
print("FINAL CONTEXT SENT TO LLM")
print("=" * 60)

print(context)

print("\n")
print("=" * 60)
print("PROMPT")
print("=" * 60)

print(prompt)

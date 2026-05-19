import os

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from visualizer import visualize_embeddings_tsne

embedding_model = SentenceTransformer("BAAI/bge-small-en")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="rag_collection")


documents = [
    "FastAPI is a modern Python framework",
    "Redis is used for caching",
    "Docker helps package applications",
    "PostgreSQL is a relational database",
]

ids = ["doc1", "doc2", "doc3", "doc4"]
embeddings = embedding_model.encode(documents)
collection.upsert(documents=documents, embeddings=embeddings.tolist(), ids=ids)
print("Documents inserted successfully.")


query = "What is FastAPI?"
query_embedding = embedding_model.encode(query)
results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=2)
retrieved_docs = results["documents"][0]
print("\nRetrieved Documents:\n")

for doc in retrieved_docs:
    print(doc)

context = "\n".join(retrieved_docs)

visualize_embeddings_tsne(
    document_embeddings=embeddings,
    documents=documents,
    query_embedding=query_embedding,
    query_text=query,
)


# Query the LLM
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

prompt = f"""
Answer the question using the context below.

Context:
{context}

Question:
{query}
"""
response = groq_client.chat.completions.create(
    model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}]
)
print("\nFinal Answer:\n")
print(response.choices[0].message.content)

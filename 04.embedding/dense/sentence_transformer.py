import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from visualizer import visualize_embeddings_tsne

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Redis is an in-memory database",
    "FastAPI is used for building APIs",
    "Docker helps create containers",
    "Kubernetes manages containers",
    "PostgreSQL is a relational database",
]

document_embeddings = model.encode(documents)

query = "Which database works in memory?"

query_embedding = model.encode([query])

scores = cosine_similarity(query_embedding, document_embeddings)

scores = scores[0]

ranked_results = np.argsort(scores)[::-1]

print("\nQUERY:")
print(query)

print("\nTOP MATCHES:\n")

for index in ranked_results:
    print("=" * 60)
    print("Document:", documents[index])
    print("Similarity Score:", round(scores[index], 4))


visualize_embeddings_tsne(
    document_embeddings=document_embeddings,
    documents=documents,
    query_embedding=query_embedding,
    query_text=query,
)

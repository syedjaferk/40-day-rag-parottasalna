import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "FastAPI Redis PubSub system",
    "Docker Kubernetes deployment",
    "Python backend engineering",
]

# Sparse Retrieval - Text based

tokenized_docs = [doc.lower().split() for doc in documents]

bm25 = BM25Okapi(tokenized_docs)

# Dense Retrieval - Semantic


model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(documents)


# Query


query = "Redis async backend"

# Sparse Score
bm25_scores = bm25.get_scores(query.lower().split())

# Dense Score
query_embedding = model.encode([query])

dense_scores = cosine_similarity(query_embedding, doc_embeddings)[0]


# Hybrid Score


hybrid_scores = 0.5 * np.array(bm25_scores) + 0.5 * np.array(dense_scores)

ranked = sorted(zip(documents, hybrid_scores), key=lambda x: x[1], reverse=True)

for doc, score in ranked:
    print(score, "->", doc)

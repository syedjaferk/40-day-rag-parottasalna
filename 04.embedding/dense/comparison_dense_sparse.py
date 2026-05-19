from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = ["Tips for fat loss", "Python programming tutorial", "Healthy diet plans"]

query = "How to reduce body weight?"

print("SPARSE EMBEDDINGS (TF-IDF)")

tfidf = TfidfVectorizer()

# Query + docs together
tfidf_vectors = tfidf.fit_transform([query] + documents)

# Sparse matrix
print("Sparse Vector Shape:")
print(tfidf_vectors.shape)

print("\nSparse Vector Example:\n")

print(tfidf_vectors.toarray()[0])

# Cosine similarity
sparse_scores = cosine_similarity(tfidf_vectors[0:1], tfidf_vectors[1:])

print("\nSparse Similarities:\n")

for doc, score in zip(documents, sparse_scores[0]):
    print(f"{score:.4f} -> {doc}")


print("DENSE EMBEDDINGS")


dense_model = SentenceTransformer("BAAI/bge-small-en")

dense_vectors = dense_model.encode([query] + documents)

print("Dense Vector Shape:")
print(dense_vectors.shape)

print("\nDense Vector Example:\n")

print(dense_vectors[0][:10])  # first 10 dims

dense_scores = cosine_similarity([dense_vectors[0]], dense_vectors[1:])

print("\nDense Similarities:\n")

for doc, score in zip(documents, dense_scores[0]):
    print(f"{score:.4f} -> {doc}")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("BAAI/bge-small-en")

sentences = [
    "I love machine learning",
    "Artificial Intelligence is fascinating",
    "I enjoy eating pizza",
]

embeddings = model.encode(sentences)

similarity_matrix = cosine_similarity(embeddings)

print(similarity_matrix)

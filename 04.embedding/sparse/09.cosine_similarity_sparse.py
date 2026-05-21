from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "redis fastapi caching",
    "python redis backend",
    "docker kubernetes deployment",
]

query = ["redis backend"]

vectorizer = TfidfVectorizer()

doc_vectors = vectorizer.fit_transform(documents)

query_vector = vectorizer.transform(query)

scores = cosine_similarity(query_vector, doc_vectors)

print(scores)

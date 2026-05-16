from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "Redis is an in-memory database",
    "PostgreSQL supports SQL databases",
    "FastAPI is a Python framework",
    "RabbitMQ is a message broker",
]


vectorizer = TfidfVectorizer()

document_vectors = vectorizer.fit_transform(documents)


def search(query):
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(query_vector, document_vectors)[0]

    results = []

    for index, score in enumerate(similarities):
        results.append({"document": documents[index], "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)

    return results


results = search("database")


for result in results:
    print(f"""
Document:
{result["document"]}

Score:
{result["score"]:.4f}
""")

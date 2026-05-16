from rank_bm25 import BM25Okapi

documents = [
    {
        "id": 1,
        "title": "Redis Introduction",
        "content": """
        Redis is an in-memory database used
        for caching pub sub queues and streams.
        """,
    },
    {
        "id": 2,
        "title": "PostgreSQL Guide",
        "content": """
        PostgreSQL is a relational database
        supporting SQL transactions indexing.
        """,
    },
    {
        "id": 3,
        "title": "FastAPI Tutorial",
        "content": """
        FastAPI is a modern Python web framework
        for APIs and async applications.
        """,
    },
    {
        "id": 4,
        "title": "RabbitMQ Messaging",
        "content": """
        RabbitMQ is a message broker implementing
        the AMQP protocol.
        """,
    },
    {
        "id": 5,
        "title": "Vector Databases",
        "content": """
        Vector databases are used for semantic
        search and embedding retrieval.
        """,
    },
]


tokenized_documents = []

for document in documents:
    tokens = document["content"].lower().split()

    tokenized_documents.append(tokens)


bm25 = BM25Okapi(tokenized_documents)


def search(query, top_k=3):
    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    results = []

    for index, score in enumerate(scores):
        results.append({"document": documents[index], "score": float(score)})

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]


query = "database for caching"

results = search(query)


print("=" * 60)
print(f"QUERY: {query}")
print("=" * 60)

for result in results:
    document = result["document"]

    print(f"""
Title:
{document["title"]}

Content:
{document["content"].strip()}

BM25 Score:
{result["score"]:.4f}
""")

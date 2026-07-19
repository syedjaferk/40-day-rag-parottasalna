import json
import redis
import numpy as np

from embeddings import get_embedding

redis_client = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

CACHE_PREFIX = "semantic:"
SIMILARITY_THRESHOLD = 0.90


def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    return np.dot(v1, v2) / (
        np.linalg.norm(v1) * np.linalg.norm(v2)
    )


def search_cache(question):

    query_embedding = get_embedding(question)

    for key in redis_client.scan_iter(f"{CACHE_PREFIX}*"):

        data = json.loads(redis_client.get(key))

        similarity = cosine_similarity(
            query_embedding,
            data["embedding"]
        )

        print(f"{data['question']} -> {similarity:.3f}")

        if similarity >= SIMILARITY_THRESHOLD:

            print("CACHE HIT")

            return data["answer"]

    print("CACHE MISS")

    return None


def store_cache(question, answer):

    embedding = get_embedding(question)

    value = {
        "question": question,
        "embedding": embedding,
        "answer": answer
    }

    redis_client.set(
        CACHE_PREFIX + str(abs(hash(question))),
        json.dumps(value)
    )

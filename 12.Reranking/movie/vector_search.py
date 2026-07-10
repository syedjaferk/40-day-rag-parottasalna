import requests
import pandas as pd
import uuid

import chromadb


client = chromadb.PersistentClient(path="./chroma_imdb")

collection = client.get_or_create_collection(
    name="movies_1"
)

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "nomic-embed-text"

df = pd.read_csv("imdb_processed.csv")


def get_embedding(text):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": text
        }
    )

    response.raise_for_status()

    return response.json()["embedding"]


if collection.count() == 0:

    embeddings = []
    documents = []
    metadatas = []
    ids = []

    for _, row in df.iterrows():

        description = str(row["Description"])

        embedding = get_embedding(description)

        embeddings.append(embedding)
        documents.append(description)
        metadatas.append(row.to_dict())
        ids.append(str(uuid.uuid4()))

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )


def search_movies(query, n_results=10):

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results


if __name__ == "__main__":

    query = input("Enter Genre : ")

    results = search_movies(query)

    for movie in results["metadatas"][0]:

        print("=" * 50)
        print(movie["Movie Name"])

"""
docker run -d \
  --name opensearch \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "plugins.security.disabled=true" \
  opensearchproject/opensearch:latest
"""

import pandas as pd
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# CONFIGURATION

INDEX_NAME = "imdb_movies"
CSV_FILE = "imdb_processed.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 5

# LOAD DATASET

print("\nLoading IMDb Dataset...")

df = pd.read_csv(CSV_FILE)

print(df.head())

# PREPARE DOCUMENTS

"""
Expected columns:
-----------------
title
description

Modify below if your dataset contains different columns.
"""

documents = []

for _, row in df.iterrows():
    title = str(row["title"])

    description = str(row["description"])

    document = {"title": title, "description": description}

    documents.append(document)

print(f"\nTotal Documents: {len(documents)}")

# LOAD EMBEDDING MODEL

print("\nLoading Embedding Model...")

model = SentenceTransformer(EMBEDDING_MODEL)

# GENERATE EMBEDDINGS

print("\nGenerating Embeddings...")

texts = [doc["description"] for doc in documents]

embeddings = model.encode(texts, show_progress_bar=True)

print("\nEmbedding Shape:")
print(embeddings.shape)

VECTOR_DIMENSION = embeddings.shape[1]

# CONNECT TO OPENSEARCH

print("\nConnecting to OpenSearch...")

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
)

print("\nConnected Successfully")

# DELETE OLD INDEX (OPTIONAL)

if client.indices.exists(INDEX_NAME):
    print("\nDeleting Existing Index...")

    client.indices.delete(index=INDEX_NAME)

# CREATE INDEX

print("\nCreating Index...")

mapping = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "title": {"type": "text"},
            "description": {"type": "text"},
            "embedding": {"type": "knn_vector", "dimension": VECTOR_DIMENSION},
        }
    },
}

client.indices.create(index=INDEX_NAME, body=mapping)

print("\nIndex Created Successfully")

# INDEX DOCUMENTS

print("\nIndexing Documents...")

for doc, embedding in tqdm(zip(documents, embeddings), total=len(documents)):
    body = {
        "title": doc["title"],
        "description": doc["description"],
        "embedding": embedding.tolist(),
    }

    client.index(index=INDEX_NAME, body=body)

client.indices.refresh(index=INDEX_NAME)

print("\nDocuments Indexed Successfully")

# SPARSE SEARCH (BM25)


def sparse_search(query_text):
    print("SPARSE SEARCH (BM25)")

    query = {"size": TOP_K, "query": {"match": {"description": query_text}}}

    response = client.search(index=INDEX_NAME, body=query)

    hits = response["hits"]["hits"]

    for i, hit in enumerate(hits, start=1):
        print(f"\nRank #{i}")

        print("Score:", round(hit["_score"], 4))

        print("Title:", hit["_source"]["title"])

        print("Description:", hit["_source"]["description"][:300])


# DENSE VECTOR SEARCH


def dense_search(query_text):
    print("DENSE VECTOR SEARCH")

    query_embedding = model.encode(query_text)

    query = {
        "size": TOP_K,
        "query": {
            "knn": {"embedding": {"vector": query_embedding.tolist(), "k": TOP_K}}
        },
    }

    response = client.search(index=INDEX_NAME, body=query)

    hits = response["hits"]["hits"]

    for i, hit in enumerate(hits, start=1):
        print(f"\nRank #{i}")

        print("Score:", round(hit["_score"], 4))

        print("Title:", hit["_source"]["title"])

        print("Description:", hit["_source"]["description"][:300])


# HYBRID SEARCH


def hybrid_search(query_text):
    print("HYBRID SEARCH")

    query_embedding = model.encode(query_text)

    query = {
        "size": TOP_K,
        "query": {
            "hybrid": {
                "queries": [
                    {"match": {"description": query_text}},
                    {
                        "knn": {
                            "embedding": {
                                "vector": query_embedding.tolist(),
                                "k": TOP_K,
                            }
                        }
                    },
                ]
            }
        },
    }

    response = client.search(index=INDEX_NAME, body=query)

    hits = response["hits"]["hits"]

    for i, hit in enumerate(hits, start=1):
        print(f"\nRank #{i}")

        print("Score:", round(hit["_score"], 4))

        print("Title:", hit["_source"]["title"])

        print("Description:", hit["_source"]["description"][:300])


print("IMDb Hybrid Search System Ready")


while True:
    print("\n")
    print("1 -> Sparse Search")
    print("2 -> Dense Search")
    print("3 -> Hybrid Search")
    print("4 -> Exit")

    choice = input("\nEnter Choice: ").strip()

    if choice == "4":
        break

    query = input("\nEnter Query: ")

    if choice == "1":
        sparse_search(query)

    elif choice == "2":
        dense_search(query)

    elif choice == "3":
        hybrid_search(query)

    else:
        print("\nInvalid Choice")

print("\nGoodbye")

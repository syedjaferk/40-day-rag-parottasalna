"""
START OPENSEARCH

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

INDEX_NAME = "imdb_movies"
CSV_FILE = "imdb_processed.csv"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5

print("\nLoading IMDb Dataset...")

df = pd.read_csv(CSV_FILE)

print("\nDataset Loaded Successfully")

print("\nFirst 5 Rows:")
print(df.head())

print("\nAvailable Columns:")
print(df.columns.tolist())

df["Movie Name"] = df["Movie Name"].fillna("")
df["Description"] = df["Description"].fillna("")
df["Duration"] = df["Duration"].fillna("")
df["IMdb rating"] = df["IMdb rating"].fillna("")
df["Rating"] = df["Rating"].fillna("")
df["Votes"] = df["Votes"].fillna("")
df["Poster"] = df["Poster"].fillna("")
df["Month of Release"] = df["Month of Release"].fillna("")

documents = []

for _, row in df.iterrows():
    movie_name = str(row["Movie Name"])

    description = str(row["Description"])

    duration = str(row["Duration"])

    imdb_rating = str(row["IMdb rating"])

    rating = str(row["Rating"])

    votes = str(row["Votes"])

    poster = str(row["Poster"])

    month_of_release = str(row["Month of Release"])

    combined_text = f"""
    Movie Name: {movie_name}

    Description: {description}

    Duration: {duration}

    IMDb Rating: {imdb_rating}

    Rating: {rating}

    Votes: {votes}

    Month of Release: {month_of_release}
    """

    document = {
        "movie_name": movie_name,
        "description": description,
        "duration": duration,
        "imdb_rating": imdb_rating,
        "rating": rating,
        "votes": votes,
        "poster": poster,
        "month_of_release": month_of_release,
        "combined_text": combined_text.strip(),
    }

    documents.append(document)

print(f"\nTotal Documents Prepared: {len(documents)}")

print("\nLoading Embedding Model...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("\nEmbedding Model Loaded Successfully")

print("\nGenerating Embeddings...")

texts = [doc["combined_text"] for doc in documents]

embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=64,
)

print("\nEmbeddings Generated Successfully")

print("\nEmbedding Shape:")
print(embeddings.shape)

VECTOR_DIMENSION = embeddings.shape[1]

print("\nConnecting to OpenSearch...")

client = OpenSearch(
    hosts=[{"host": "localhost", "port": 9200}],
    http_compress=True,
    use_ssl=False,
    verify_certs=False,
)

print("\nConnected Successfully")

if client.indices.exists(index=INDEX_NAME):
    print("\nDeleting Existing Index...")

    client.indices.delete(index=INDEX_NAME)

    print("\nOld Index Deleted")

print("\nCreating Index...")

mapping = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            "movie_name": {"type": "text"},
            "description": {"type": "text"},
            "duration": {"type": "keyword"},
            "imdb_rating": {"type": "float"},
            "rating": {"type": "keyword"},
            "votes": {"type": "keyword"},
            "poster": {"type": "keyword"},
            "month_of_release": {"type": "keyword"},
            "embedding": {"type": "knn_vector", "dimension": VECTOR_DIMENSION},
        }
    },
}

client.indices.create(index=INDEX_NAME, body=mapping)

print("\nIndex Created Successfully")

print("\nIndexing Documents...")

for doc, embedding in tqdm(zip(documents, embeddings), total=len(documents)):
    body = {
        "movie_name": doc["movie_name"],
        "description": doc["description"],
        "duration": doc["duration"],
        "imdb_rating": float(doc["imdb_rating"]) if doc["imdb_rating"] else 0.0,
        "rating": doc["rating"],
        "votes": doc["votes"],
        "poster": doc["poster"],
        "month_of_release": doc["month_of_release"],
        "embedding": embedding.tolist(),
    }

    client.index(
        index=INDEX_NAME,
        body=body,
    )

client.indices.refresh(index=INDEX_NAME)

print("\nDocuments Indexed Successfully")


def sparse_search(query_text):
    print("\n==============================")
    print("SPARSE SEARCH (BM25)")
    print("==============================")

    query = {
        "size": TOP_K,
        "query": {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "movie_name^3",
                    "description",
                    "month_of_release",
                ],
            }
        },
    }

    response = client.search(
        index=INDEX_NAME,
        body=query,
    )

    hits = response["hits"]["hits"]

    for i, hit in enumerate(hits, start=1):
        source = hit["_source"]

        print(f"\nRank #{i}")

        print("Score:", round(hit["_score"], 4))

        print("Movie Name:", source["movie_name"])

        print("IMDb Rating:", source["imdb_rating"])

        print("Duration:", source["duration"])

        print("Votes:", source["votes"])

        print("Month:", source["month_of_release"])

        print("Description:", source["description"][:300])


def dense_search(query_text):
    print("\n==============================")
    print("DENSE VECTOR SEARCH")
    print("==============================")

    query_embedding = model.encode(query_text)

    query = {
        "size": TOP_K,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding.tolist(),
                    "k": TOP_K,
                }
            }
        },
    }

    response = client.search(
        index=INDEX_NAME,
        body=query,
    )

    hits = response["hits"]["hits"]

    for i, hit in enumerate(hits, start=1):
        source = hit["_source"]

        print(f"\nRank #{i}")

        print("Score:", round(hit["_score"], 4))

        print("Movie Name:", source["movie_name"])

        print("IMDb Rating:", source["imdb_rating"])

        print("Duration:", source["duration"])

        print("Votes:", source["votes"])

        print("Month:", source["month_of_release"])

        print("Description:", source["description"][:300])


def hybrid_search(query_text):
    print("\n==============================")
    print("HYBRID SEARCH")
    print("==============================")

    query_embedding = model.encode(query_text)

    query = {
        "size": TOP_K,
        "query": {
            "hybrid": {
                "queries": [
                    {
                        "multi_match": {
                            "query": query_text,
                            "fields": [
                                "movie_name^3",
                                "description",
                                "month_of_release",
                            ],
                        }
                    },
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

    response = client.search(
        index=INDEX_NAME,
        body=query,
    )

    hits = response["hits"]["hits"]

    for i, hit in enumerate(hits, start=1):
        source = hit["_source"]

        print(f"\nRank #{i}")

        print("Score:", round(hit["_score"], 4))

        print("Movie Name:", source["movie_name"])

        print("IMDb Rating:", source["imdb_rating"])

        print("Duration:", source["duration"])

        print("Votes:", source["votes"])

        print("Month:", source["month_of_release"])

        print("Description:", source["description"][:300])


print("\nIMDb Hybrid Search System Ready")

while True:
    print("\n")
    print("1 -> Sparse Search")
    print("2 -> Dense Search")
    print("3 -> Hybrid Search")
    print("4 -> Exit")

    choice = input("\nEnter Choice: ").strip()

    if choice == "4":
        print("\nGoodbye")

        break

    query = input("\nEnter Query: ").strip()

    if not query:
        print("\nQuery cannot be empty")

        continue

    if choice == "1":
        sparse_search(query)

    elif choice == "2":
        dense_search(query)

    elif choice == "3":
        hybrid_search(query)

    else:
        print("\nInvalid Choice")

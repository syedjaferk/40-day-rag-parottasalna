import chromadb
import matplotlib.pyplot as plt
import numpy as np
import ollama
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def ollama_embed(texts):
    embeddings = []

    for t in texts:
        res = ollama.embeddings(model="nomic-embed-text", prompt=t)
        embeddings.append(res["embedding"])

        # print(f"Text: {t} , Embedding: {res['embedding'][:10]}")
        # input("Press enter to continue...")
    return embeddings


docs = [
    "Python is a programming language",
    "FastAPI is great for building APIs and mainly used in vector based solutions",
    "ChromaDB is a vector database",
    "I love machine learning",
    "PostgreSQL is a relational Database",
]

client = chromadb.Client()
collection = client.create_collection("ollama_demo")

# Create embeddings via Ollama
emb = ollama_embed(docs)

collection.add(documents=docs, embeddings=emb, ids=[str(i) for i in range(len(docs))])

# print("Added using Ollama embeddings!")


query = "tell me about vector database"  # actual point
q_emb = ollama_embed([query])

results = collection.query(query_embeddings=q_emb, n_results=3)

print(results["documents"])


data = collection.get(include=["embeddings", "documents"])

embeddings = np.array(data["embeddings"])
documents = data["documents"]


def plot_embeddings(embeddings, labels, title):
    reducer = TSNE(n_components=2, perplexity=3, random_state=42)
    reduced_data = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    plt.scatter(
        reduced_data[:, 0], reduced_data[:, 1], c="blue", edgecolors="k", alpha=0.6
    )

    for i, txt in enumerate(labels):
        plt.annotate(
            txt,
            (reduced_data[i, 0], reduced_data[i, 1]),
            fontsize=9,
            xytext=(5, 2),
            textcoords="offset points",
        )

    plt.title(f"Embedding Visualization using {title}")
    plt.grid(True)
    plt.show()


plot_embeddings(embeddings, documents, "t-SNE")

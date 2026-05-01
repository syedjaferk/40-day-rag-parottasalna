import numpy as np
import chromadb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

client = chromadb.PersistentClient(path="./chroma_storage")
collection = client.get_or_create_collection("ollama_demo")

data = collection.get(include=['embeddings', 'documents'])
embeddings = np.array(data['embeddings'])
documents = data['documents']

def plot_embeddings(embeddings, labels, title):

    reducer = TSNE(n_components=2, perplexity=len(labels)-1, random_state=42)
    reduced_data = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='blue', edgecolors='k', alpha=0.6)

    for i, txt in enumerate(labels):
        plt.annotate(txt, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=9, xytext=(5, 2), 
                     textcoords='offset points')

    plt.title(f"Embedding Visualization using {title}")
    plt.grid(True)
    plt.show()

plot_embeddings(embeddings, documents, "t-SNE")
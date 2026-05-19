import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def visualize_embeddings_tsne(
    document_embeddings, documents, query_embedding=None, query_text=None
):
    vectors = list(document_embeddings)
    labels = list(documents)

    # Add query embedding if provided
    if query_embedding is not None:
        vectors.append(query_embedding)
        labels.append(f"QUERY: {query_text}")

    vectors = np.array(vectors)

    # Reduce dimensions
    tsne = TSNE(n_components=2, perplexity=3, random_state=42)

    reduced_vectors = tsne.fit_transform(vectors)

    # Plot
    plt.figure(figsize=(12, 8))

    for i, label in enumerate(labels):
        x = reduced_vectors[i][0]
        y = reduced_vectors[i][1]

        # Highlight query
        if label.startswith("QUERY:"):
            plt.scatter(x, y, s=300, marker="X")
        else:
            plt.scatter(x, y)

        plt.text(x + 0.5, y + 0.5, label, fontsize=9)

    plt.title("Dense Embedding Visualization using t-SNE")

    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")

    plt.grid(True)

    plt.show()

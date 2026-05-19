from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("BAAI/bge-small-en")

texts = ["How to learn Python?", "Best way to study Python", "I love pizza"]

embeddings = model.encode(texts)

embeddings = cosine_similarity(embeddings)
for i, text in enumerate(texts):
    print(f"Similarity Scores Text: {text}")
    for index, item in enumerate(embeddings[i]):
        print(f"Similarity Score:{text} {texts[index]} - {item}")
    # print(f"Similarity Scores: {embeddings[i]}")
    print("=" * 60)

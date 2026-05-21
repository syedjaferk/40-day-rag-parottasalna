import re
from collections import Counter

import pandas as pd

# Load data
df = pd.read_csv("imdb_processed.csv")

# Clean columns
df["Description"] = df["Description"].fillna("")
df["Movie Name"] = df["Movie Name"].fillna("")
df["Movie Name Clean"] = df["Movie Name"].str.replace(r"^\s*\d+\.\s*", "", regex=True)

# Small stopword list for better search quality
stopwords = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "from",
    "their",
    "his",
    "her",
    "its",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "as",
    "at",
    "that",
    "this",
    "these",
    "those",
    "it",
    "they",
    "them",
    "he",
    "she",
    "you",
    "your",
    "our",
    "us",
    "after",
    "before",
    "when",
    "who",
    "whom",
    "which",
    "while",
    "into",
    "than",
    "then",
    "there",
    "here",
    "about",
    "against",
    "over",
    "under",
    "through",
    "during",
    "within",
    "without",
    "between",
    "two",
    "one",
}


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", str(text).lower())
    return [w for w in words if w not in stopwords]


# Build TF vectors for every description
doc_tokens = df["Description"].map(tokenize)
doc_lengths = doc_tokens.map(len).replace(0, 1)

tf_docs = []
for tokens, length in zip(doc_tokens, doc_lengths):
    counts = Counter(tokens)
    tf = {term: count / length for term, count in counts.items()}
    tf_docs.append(tf)


def search_movies(query, top_k=5):
    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    q_counts = Counter(q_tokens)
    q_len = len(q_tokens)
    q_tf = {term: count / q_len for term, count in q_counts.items()}

    scores = []
    for idx, doc_tf in enumerate(tf_docs):
        score = 0.0
        for term, q_weight in q_tf.items():
            score += q_weight * doc_tf.get(term, 0.0)
        if score > 0:
            scores.append((score, idx))

    scores.sort(reverse=True)
    results = []
    for score, idx in scores[:top_k]:
        results.append(
            {
                "movie": df.loc[idx, "Movie Name Clean"],
                "score": round(score, 4),
                "description": df.loc[idx, "Description"],
            }
        )
    return results


# Example searches
queries = [
    "bank heist money",
    "love family",
    "murder investigation",
    "autistic teenager woman abducted",
    "rights injustice murder",
]

for q in queries:
    print(f"\nQUERY: {q}")
    for row in search_movies(q, top_k=10):
        print(row["score"], "-", row["movie"])
    input()

import math
import re
from collections import Counter

import pandas as pd

df = pd.read_csv("imdb_processed.csv")


df["Description"] = df["Description"].fillna("")
df["Movie Name"] = df["Movie Name"].fillna("")


df["Movie Name Clean"] = df["Movie Name"].str.replace(r"^\s*\d+\.\s*", "", regex=True)

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
}


def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", str(text).lower())

    cleaned_words = [w for w in words if w not in stopwords]

    return cleaned_words


doc_tokens = df["Description"].map(tokenize)

doc_lengths = doc_tokens.map(len).replace(0, 1)


tf_docs = []

for tokens, length in zip(doc_tokens, doc_lengths):
    counts = Counter(tokens)

    tf = {}

    for term, count in counts.items():
        tf[term] = count / length

    tf_docs.append(tf)


df_counts = Counter()

for tokens in doc_tokens:
    unique_terms = set(tokens)

    for term in unique_terms:
        df_counts[term] += 1


N = len(df)

idf_scores = {}

for term, df_value in df_counts.items():
    idf_scores[term] = math.log(N / df_value)


tfidf_docs = []

for tf in tf_docs:
    tfidf = {}

    for term, tf_value in tf.items():
        tfidf[term] = tf_value * idf_scores[term]

    tfidf_docs.append(tfidf)


def search_movies(query, top_k=5):
    q_tokens = tokenize(query)

    if not q_tokens:
        return []

    q_counts = Counter(q_tokens)

    q_len = len(q_tokens)

    q_tf = {}

    for term, count in q_counts.items():
        q_tf[term] = count / q_len

    q_tfidf = {}

    for term, tf_value in q_tf.items():
        idf = idf_scores.get(term, 0)

        q_tfidf[term] = tf_value * idf

    scores = []

    for idx, doc_vector in enumerate(tfidf_docs):
        score = 0.0

        for term, q_weight in q_tfidf.items():
            doc_weight = doc_vector.get(term, 0.0)

            score += q_weight * doc_weight

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


queries = [
    "bank heist money",
    "murder investigation",
    "love family",
    "alien invasion",
    "crime gangster",
]


for q in queries:
    print("\n" + "=" * 60)

    print("QUERY:", q)

    print("=" * 60)

    results = search_movies(q, top_k=3)

    for row in results:
        print("\nMovie:", row["movie"])
        print("Score:", row["score"])
        print("Description:")
        print(row["description"])

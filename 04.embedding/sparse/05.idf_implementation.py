import math
from collections import Counter

documents = [
    "python redis fastapi",
    "python docker kubernetes",
    "python redis caching",
    "java spring backend",
    "python machine learning",
]

# Total number of documents
N = len(documents)

# Count document frequency
df_counts = Counter()

for doc in documents:
    # Unique words only
    unique_terms = set(doc.split())
    print(unique_terms)
    for term in unique_terms:
        df_counts[term] += 1
    print(df_counts)
    print("#" * 10)

# Calculate IDF
idf_scores = {}

for term, df in df_counts.items():
    idf_scores[term] = math.log(N / df)

# Print IDF values
for term, score in idf_scores.items():
    print(term, "->", round(score, 3))

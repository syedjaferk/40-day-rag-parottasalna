from collections import Counter

document = "Redis is an in-memory database database database"


tokens = document.split()

total_terms = len(tokens)

counter = Counter(tokens)

tf_scores = {}

for term, count in counter.items():
    tf_scores[term] = count / total_terms

print(tf_scores)

# TF(t,d)= Number of times term t appears in document d / Total number of terms in document d

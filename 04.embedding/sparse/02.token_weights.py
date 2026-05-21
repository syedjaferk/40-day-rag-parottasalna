from collections import Counter

sentence = "redis is fast and redis supports caching"

tokens = sentence.lower().split()

token_counts = Counter(tokens)

print(token_counts)

vocabulary = {"python": 0, "redis": 1, "fastapi": 2, "docker": 3, "database": 4}

sentence = "python fastapi application. fastapi is a nice framework. "

tokens = sentence.lower().split()

vector = [0] * len(vocabulary)

for token in tokens:
    if token in vocabulary:
        vector[vocabulary[token]] = 1

print("Sparse Vector:")
print(vector)


# One hot encoding, Binary Encoding.

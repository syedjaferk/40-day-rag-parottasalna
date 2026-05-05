text = """
Redis is an in-memory data structure store that is commonly used as a database, cache, and message broker. 
It supports various data structures such as strings, lists, sets, sorted sets, hashes, bitmaps, and streams. 
Because Redis stores data in memory, it is extremely fast and is often used in scenarios requiring low latency.

Redis also provides persistence mechanisms like RDB (snapshotting) and AOF (append-only file) to ensure data durability. 
These persistence strategies allow Redis to recover data after restarts, making it reliable for production use cases.

In addition to persistence, Redis supports replication, allowing data to be copied across multiple nodes for high availability. 
It also provides clustering capabilities, enabling horizontal scaling across multiple machines.

Redis is widely used in real-time analytics, caching layers, session storage, and pub/sub messaging systems. 
Its simplicity and performance make it a popular choice among developers building scalable applications.
"""

def overlap_chunking(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += chunk_size - overlap

    return chunks

overlap_chunks = overlap_chunking(text)

print("\n=== Overlap Chunking ===")
for i, c in enumerate(overlap_chunks):
    print(f"\nChunk {i+1}:\n{c}")
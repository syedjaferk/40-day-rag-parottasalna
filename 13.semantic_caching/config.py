
GROQ_API_KEY = "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"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"

CHROMA_DB = "./chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_INDEX = "semantic_cache"
REDIS_PREFIX = "cache:"
VECTOR_DIM = 384

SIMILARITY_THRESHOLD = 0.90

CACHE_TTL = 60 * 60 * 24 * 7

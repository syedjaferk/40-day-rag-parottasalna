from langchain_community.embeddings import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)

def get_embedding(text: str):
    return embedding_model.embed_query(text)

def get_embedding_model():
    return embedding_model

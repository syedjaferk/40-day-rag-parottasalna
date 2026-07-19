from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from embeddings import get_embedding_model

embeddings = get_embedding_model()
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

def retrieval(query):
    docs = db.similarity_search(query, k=3)
    context = "\n".join(
        [doc.page_content for doc in docs]
    )
    return context

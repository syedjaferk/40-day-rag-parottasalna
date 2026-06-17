from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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

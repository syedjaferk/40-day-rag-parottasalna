import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 300
)

# Load Documents
loader = PyPDFLoader("./python.pdf")
documents = loader.load()

def clean_text(text):
    text = text.encode("utf-8", "ignore").decode("utf-8")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Chunk Documents
clean_docs = []
for doc in documents:
    doc.page_content = clean_text(doc.page_content)
    clean_docs.append(doc)

chunks = splitter.split_documents(clean_docs)

# Embedding
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
db.persist()

print("Ingestion Completed...")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableLambda

from sentence_transformers import CrossEncoder

from groq_client import call_groq

import os


PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_collection"



# Load Cross Encoder only once


reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)



# Load PDF


def load_pdf(path):

    loader = PyPDFLoader(path)

    return loader.load()



# Clean text


def clean_text(text):

    return text.encode(
        "utf-8",
        "ignore"
    ).decode("utf-8")



# Split


def split_docs(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.page_content = clean_text(
            chunk.page_content
        )

    return chunks



# Vector DB


def create_vector_db(chunks):

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    if os.path.exists(PERSIST_DIR):

        print("Loading Existing DB...")

        vector_db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR
        )

    else:

        print("Creating DB...")

        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR
        )

        vector_db.persist()

    return vector_db



# Groq Runnable


def groq_runnable(prompt_value):

    role_map = {
        "human": "user",
        "system": "system",
        "ai": "assistant"
    }

    messages = []

    for msg in prompt_value.messages:

        messages.append(
            {
                "role": role_map.get(
                    msg.type,
                    msg.type
                ),
                "content": msg.content
            }
        )

    return call_groq(messages)



# Reranker


def rerank_documents(query, docs, top_k=4):

    pairs = []

    for doc in docs:
        pairs.append(
            (
                query,
                doc.page_content
            )
        )

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True
    )

    print("\n============= After Reranking =============")

    for score, doc in ranked:

        print(f"Score : {score:.4f}")
        print(doc.page_content[:120])
        print()

    return [
        doc
        for score, doc in ranked[:top_k]
    ]



# Build Chain


def build_chain(vector_db):

    custom_llm = RunnableLambda(
        groq_runnable
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant.

Answer ONLY from the given context.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    retriever = vector_db.as_retriever(
        search_kwargs={
            "k": 15
        }
    )

    def retrieve_and_rerank(query):

        print("\nRetrieving documents...\n")

        docs = retriever.invoke(query)

        print(f"Retrieved {len(docs)} documents")

        docs = rerank_documents(
            query,
            docs,
            top_k=4
        )

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        return context

    chain = (

        {
            "context": RunnableLambda(
                retrieve_and_rerank
            ),
            "question": RunnablePassthrough()
        }

        | prompt

        | custom_llm
    )

    return chain



# Pipeline


def create_chatbot(pdf_path):

    docs = load_pdf(pdf_path)

    chunks = split_docs(docs)

    vector_db = create_vector_db(chunks)

    chain = build_chain(
        vector_db
    )

    return chain



# Main


chain = create_chatbot("python.pdf")

while True:

    question = input("\nAsk Question : ")

    if question.lower() == "exit":
        break

    response = chain.invoke(question)

    print("\n================ Answer ================\n")

    print(response)

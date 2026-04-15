# ingest.py — loads PDF, creates embeddings, stores in ChromaDB
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

def ingest_pdf(pdf_path: str):
    print(f"Loading PDF: {pdf_path}")

    # Step 1 — Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    # Step 2 — Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Step 3 — Create embeddings (runs locally, free)
    print("Creating embeddings — this takes ~1 minute first time...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Step 4 — Store in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")
    print("Ingestion complete!")
    return vectorstore

if __name__ == "__main__":
    ingest_pdf("sample.pdf")
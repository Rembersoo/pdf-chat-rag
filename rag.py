# rag.py
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
print(f"GROQ KEY FOUND: {groq_key is not None}")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def ingest_uploaded_pdf(pdf_path: str, session_id: str):
    print(f"Ingesting uploaded PDF: {pdf_path}")

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = get_embeddings()

    # Each session gets its own ChromaDB collection
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=f"chroma_sessions/{session_id}"
    )
    print(f"Stored {len(chunks)} chunks for session {session_id}")
    return vectorstore

def load_rag_chain(session_id: str = "default"):
    embeddings = get_embeddings()

    persist_dir = f"chroma_sessions/{session_id}"

    # Fall back to default chroma_db if session doesn't exist
    if not os.path.exists(persist_dir):
        persist_dir = "chroma_db"

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    prompt_template = """You are a helpful assistant answering questions
about a document. Use ONLY the context below to answer.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return chain

def ask_question(chain, question: str):
    result = chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]
    pages = list(set([
        doc.metadata.get("page", 0) + 1
        for doc in sources
    ]))
    return answer, pages
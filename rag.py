# rag.py — retrieves chunks and generates answer via Groq
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

load_dotenv()

def load_rag_chain():
    # Load the same embeddings used during ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # Load ChromaDB with stored chunks
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    # Connect to Groq LLM
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

    # Prompt template — tells LLM to only use the retrieved context
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

    # Build the RAG chain
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

    # Show which pages were used
    pages = list(set([
        doc.metadata.get("page", "?") + 1
        for doc in sources
    ]))

    return answer, pages
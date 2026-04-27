# PDF Chat — RAG Chatbot

An AI-powered chatbot that lets you upload ANY PDF and have a 
natural conversation with it. Ask questions, get summaries, 
extract key information — all powered by LLaMA 3.

Built with a completely free stack — no paid APIs required.

## Live Demo
https://pdf-chat-rag-production.up.railway.app/

## What's New
- Upload any PDF directly in the app
- Each user gets their own session — no overlap between users
- Instant re-indexing when a new PDF is uploaded
- Works with research papers, contracts, textbooks, reports — any PDF

## How it works
1. User uploads a PDF via the sidebar
2. PDF is split into 500-character overlapping chunks
3. Each chunk is converted into a vector embedding (all-MiniLM-L6-v2)
4. Embeddings stored in ChromaDB (session-specific collection)
5. User asks a question → question gets embedded
6. ChromaDB finds the 4 most semantically similar chunks
7. LLaMA 3 generates an answer using only those chunks as context
8. Answer shown with source page numbers

## Tech Stack
| Component        | Tool                           | Cost  |
|------------------|-------------------------------|-------|
| PDF loading      | LangChain + PyPDF              | Free  |
| Text splitting   | RecursiveCharacterTextSplitter | Free  |
| Embeddings       | sentence-transformers (local)  | Free  |
| Vector database  | ChromaDB (session-based)       | Free  |
| LLM              | LLaMA 3.1 via Groq             | Free  |
| UI               | Streamlit                      | Free  |
| Deployment       | Railway + Docker               | Free  |

## Run Locally

### 1. Clone the repo
git clone https://github.com/Rembersoo/pdf-chat-rag
cd pdf-chat-rag

### 2. Create conda environment
conda create -n ragenv python=3.11
conda activate ragenv

### 3. Install dependencies
pip install -r requirements.txt

### 4. Add your Groq API key
Create a .env file in the project root:
GROQ_API_KEY=your_groq_key_here

Get a free key at: https://console.groq.com

### 5. Run the app
streamlit run app.py

Open http://localhost:8501 — upload a PDF and start chatting!

## Run with Docker
docker build -t pdf-chat-rag .
docker run -p 8501:8501 \
  -e PORT=8501 \
  -e GROQ_API_KEY=your_key \
  pdf-chat-rag

## Project Structure
pdf-chat-rag/
├── app.py            # Streamlit UI with PDF upload
├── rag.py            # RAG chain — ingestion, retrieval, generation
├── ingest.py         # One-time ingestion for default PDF
├── sample.pdf        # Default PDF (used at build time)
├── chroma_db/        # Default vector store (built during Docker build)
├── Dockerfile        # Container config
├── requirements.txt  # Python dependencies
└── .env              # API keys (never committed)

## Key Features
- Session isolation — each user gets their own ChromaDB collection
  so multiple users can upload different PDFs simultaneously
- Source citations — every answer shows which pages were used
- Re-indexing on upload — switching PDFs clears chat history
  and rebuilds the vector store automatically
- Fallback to default PDF — if no PDF is uploaded, the app
  uses the built-in sample.pdf

## What I Learned
- End-to-end RAG pipeline from scratch
- Vector embeddings and semantic search with ChromaDB
- Session management in Streamlit
- Handling file uploads and temp files in Python
- Building LLM chains with LangChain
- Connecting to free LLM APIs (Groq + LLaMA 3.1)
- Reducing Docker image size (6.5GB → 2GB)
- Debugging production deployments on Railway
- CI/CD with GitHub → Railway auto-deploy

## Troubleshooting
**"I couldn't find that in the document"**
The question may not match the PDF content closely enough.
Try rephrasing or asking for a summary first.

**App slow on first question**
The embedding model loads on first use — subsequent 
questions are much faster.

**Upload not working**
Make sure your file is a valid PDF under 200MB.

##

# PDF Chat — RAG Chatbot

Chat with any PDF using AI. Upload a document and ask questions in 
natural language — the app finds the most relevant sections and 
generates accurate answers using LLaMA 3.

Built with a fully free stack — no paid APIs required.

## Live Demo
🔗 https://your-app.up.railway.app

## How it works
1. PDF is loaded and split into 500-character chunks
2. Each chunk is converted into a vector embedding (all-MiniLM-L6-v2)
3. Embeddings are stored in ChromaDB (local vector database)
4. User asks a question → question gets embedded
5. ChromaDB finds the 4 most semantically similar chunks
6. LLaMA 3 generates an answer using only those chunks as context

## Tech Stack
| Component        | Tool                        | Cost  |
|------------------|-----------------------------|-------|
| PDF loading      | LangChain + PyPDF           | Free  |
| Text splitting   | RecursiveCharacterTextSplitter | Free |
| Embeddings       | sentence-transformers (local) | Free |
| Vector database  | ChromaDB                    | Free  |
| LLM              | LLaMA 3.1 via Groq          | Free  |
| UI               | Streamlit                   | Free  |
| Deployment       | Railway                     | Free  |

## Run Locally

### 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/pdf-chat-rag
cd pdf-chat-rag

### 2. Create conda environment
conda create -n ragenv python=3.11
conda activate ragenv

### 3. Install dependencies
pip install -r requirements.txt

### 4. Add your Groq API key
Create a .env file:
GROQ_API_KEY=your_groq_key_here

Get a free key at: https://console.groq.com

### 5. Ingest your PDF
Add your PDF to the project folder, rename it sample.pdf, then:
python ingest.py

### 6. Run the app
streamlit run app.py

Open http://localhost:8501 in your browser.

## Run with Docker
docker build -t pdf-chat-rag .
docker run -p 8501:8501 -e PORT=8501 -e GROQ_API_KEY=your_key pdf-chat-rag

## Project Structure
pdf-chat-rag/
├── app.py          # Streamlit chat UI
├── rag.py          # RAG chain — retrieval + Groq generation  
├── ingest.py       # PDF ingestion — chunking + embeddings
├── chroma_db/      # Vector database (auto-generated)
├── sample.pdf      # PDF to chat with
├── Dockerfile      # Container config
└── requirements.txt

## What I Learned
- How RAG (Retrieval Augmented Generation) works end to end
- Vector embeddings and semantic search with ChromaDB
- Building LLM-powered chains with LangChain
- Connecting to free LLM APIs (Groq + LLaMA 3)
- Building chat UIs with Streamlit
- Deploying AI apps with Docker and Railway

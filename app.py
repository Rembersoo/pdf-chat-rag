# app.py
import streamlit as st
import tempfile
import os
import uuid
from rag import load_rag_chain, ask_question, ingest_uploaded_pdf

st.set_page_config(
    page_title="PDF Chat",
    page_icon="📄",
    layout="centered"
)

st.title("Chat with your PDF")
st.caption("Upload any PDF and ask questions about it")

# Generate unique session ID for this user
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Sidebar — PDF upload
with st.sidebar:
    st.header("Upload your PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload any PDF to chat with it"
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.pdf_name:
            with st.spinner("Reading and indexing your PDF..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                # Ingest the PDF
                ingest_uploaded_pdf(tmp_path, st.session_state.session_id)

                # Load new chain for this PDF
                st.session_state.chain = load_rag_chain(
                    st.session_state.session_id
                )
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.messages = []

                # Clean up temp file
                os.unlink(tmp_path)

            st.success(f"Ready! Ask me anything about {uploaded_file.name}")

    if st.session_state.pdf_name:
        st.info(f"Current PDF: {st.session_state.pdf_name}")

    st.divider()
    st.caption("Powered by LLaMA 3 + Groq + LangChain")

# Main chat area
if st.session_state.chain is None:
    st.info("Upload a PDF in the sidebar to get started!")
    st.stop()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if question := st.chat_input("Ask a question about your PDF..."):
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, pages = ask_question(
                st.session_state.chain, question
            )
            st.write(answer)
            st.caption(f"Sources: pages {pages}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
# app.py — Streamlit chat UI
import streamlit as st
from rag import load_rag_chain, ask_question

st.set_page_config(
    page_title="PDF Chat",
    page_icon="📄",
    layout="centered"
)

st.title("Chat with your PDF")
st.caption("Powered by LLaMA 3 via Groq + RAG")

# Load RAG chain once and cache it
@st.cache_resource
def get_chain():
    return load_rag_chain()

chain = get_chain()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if question := st.chat_input("Ask a question about your PDF..."):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, pages = ask_question(chain, question)
            st.write(answer)
            st.caption(f"Sources: pages {pages}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
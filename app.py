import streamlit as st
import os

from utils.pdf_loader import load_pdf
from utils.text_splitter import split_text
from scripts.ingest import ingest
from models.llm_model import load_llm
from utils.prompt_template import get_prompt
from utils.retriever import Retriever

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(
    page_title="Course-Aware AI Teaching Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Course-Aware AI Teaching Assistant")
st.caption("Upload course material and ask doubts (Open-Source LLM)")

# -----------------------------
# Directories Setup
# -----------------------------
UPLOAD_DIR = "data/uploads"
VECTOR_DIR = "embeddings/vector_store"

# Ensure upload and vector directories exist
for directory in [UPLOAD_DIR, VECTOR_DIR]:
    if os.path.exists(directory) and not os.path.isdir(directory):
        st.error(f"❌ '{directory}' exists but is a FILE. Please delete or rename it.")
        st.stop()
    os.makedirs(directory, exist_ok=True)

# -----------------------------
# File Upload Section
# -----------------------------
st.subheader("📤 Upload Course Files")

uploaded_files = st.file_uploader(
    "Upload PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    all_chunks = []

    with st.spinner("Processing and indexing documents..."):
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

            # Save file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Load text
            if uploaded_file.name.lower().endswith(".pdf"):
                text = load_pdf(file_path)
            else:
                text = uploaded_file.read().decode("utf-8")

            # Split text into chunks
            chunks = split_text(text)
            all_chunks.extend(chunks)

        # Index chunks in FAISS
        ingest(all_chunks)

    st.success("✅ Documents indexed successfully!")

# -----------------------------
# Question Answering Section
# -----------------------------
st.divider()
st.subheader("Hello! May I help you with your course material?")

question = st.text_input("Enter your question")

if question:
    with st.spinner("Thinking..."):
        llm = load_llm()
        retriever = Retriever()
        # Retrieve top relevant chunks from FAISS
        context = retriever.retrieve(question, top_k=3)

        prompt = get_prompt(context, question)
        response = llm(prompt)

    st.markdown("### 📘 Answer")
    st.write(response[0]["generated_text"])

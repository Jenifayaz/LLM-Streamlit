
import os
from pathlib import Path

import streamlit as st

from utils.pdf_loader import load_pdf
from utils.ppt_loader import load_ppt
from utils.text_splitter import split_text
from scripts.ingest import ingest
from models.llm_model import load_llm
from utils.prompt_template import get_prompt
from utils.retriever import Retriever


# ---------------------------------
# App Configuration
# ---------------------------------
st.set_page_config(
    page_title="Course-Aware AI Teaching Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Course-Aware AI Teaching Assistant")

st.caption(
    "Upload PDF, PowerPoint or text files and ask "
    "questions based on your course materials."
)


# ---------------------------------
# Directories Setup
# ---------------------------------
UPLOAD_DIR = Path("data/uploads")
VECTOR_DIR = Path("embeddings/vector_store")

for directory in [UPLOAD_DIR, VECTOR_DIR]:
    if directory.exists() and not directory.is_dir():
        st.error(
            f"❌ '{directory}' exists as a file. "
            "Please delete or rename it."
        )
        st.stop()

    directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------
# Session State
# ---------------------------------
if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# ---------------------------------
# File Upload Section
# ---------------------------------
st.subheader("📤 Upload Course Materials")

uploaded_files = st.file_uploader(
    "Upload PDF, TXT or PowerPoint files",
    type=["pdf", "txt", "pptx"],
    accept_multiple_files=True,
    help="Supported formats: PDF, TXT and PPTX"
)

if uploaded_files:
    st.info(
        f"📁 {len(uploaded_files)} file(s) selected."
    )

    for uploaded_file in uploaded_files:
        st.write(f"📄 {uploaded_file.name}")


# ---------------------------------
# Process Documents
# ---------------------------------
if st.button(
    "🚀 Process Documents",
    type="primary",
    disabled=not uploaded_files
):

    all_chunks = []
    processed_files = []

    with st.spinner(
        "Reading, splitting and indexing documents..."
    ):

        for uploaded_file in uploaded_files:

            # Identify supported extension
            file_extension = (
                Path(uploaded_file.name).suffix.lower()
            )

            if file_extension not in [
                ".pdf", ".txt", ".pptx"
            ]:
                st.warning(
                    f"Unsupported file: {uploaded_file.name}"
                )
                continue

            # Use only the filename, not directory paths
            safe_name = Path(uploaded_file.name).name
            file_path = UPLOAD_DIR / safe_name

            try:
                # Save uploaded file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Extract text by file type
                if file_extension == ".pdf":
                    text = load_pdf(str(file_path))

                elif file_extension == ".pptx":
                    text = load_ppt(str(file_path))

                elif file_extension == ".txt":
                    text = file_path.read_text(
                        encoding="utf-8-sig"
                    )

                # Check extracted text
                if not text or not text.strip():
                    st.warning(
                        f"No readable text found in "
                        f"{safe_name}"
                    )
                    continue

                # Split text into chunks
                chunks = split_text(text)

                if not chunks:
                    st.warning(
                        f"No text chunks generated for "
                        f"{safe_name}"
                    )
                    continue

                all_chunks.extend(chunks)
                processed_files.append(safe_name)

                st.success(
                    f"Processed: {safe_name} "
                    f"({len(chunks)} chunks)"
                )

            except Exception as e:
                st.error(
                    f"Error processing {safe_name}: {e}"
                )

        # ---------------------------------
        # Index documents in FAISS
        # ---------------------------------
        if all_chunks:
            try:
                ingest(all_chunks)

                st.session_state.documents_processed = True
                st.session_state.processed_files = (
                    processed_files
                )

                st.success(
                    "✅ Documents indexed successfully!"
                )

            except Exception as e:
                st.session_state.documents_processed = False
                st.error(
                    f"Error indexing documents: {e}"
                )

        else:
            st.session_state.documents_processed = False
            st.warning(
                "No valid text content was available "
                "for indexing."
            )


# ---------------------------------
# Indexed Documents Status
# ---------------------------------
st.divider()

st.subheader("📚 Indexed Course Materials")

if st.session_state.documents_processed:

    st.success(
        "Your course materials are ready "
        "for questions."
    )

    for filename in st.session_state.processed_files:
        st.write(f"✅ {filename}")

else:
    st.info(
        "Upload and process course materials "
        "to begin asking questions."
    )


# ---------------------------------
# Question Answering Section
# ---------------------------------
st.divider()

st.subheader(
    "💬 Hello! How can I help you "
    "with your course material?"
)

question = st.text_input(
    "Enter your question",
    placeholder=(
        "e.g. Explain machine learning "
        "using my uploaded lecture slides"
    )
)

ask_button = st.button(
    "🔍 Ask Question",
    type="primary"
)

if ask_button:

    if not question.strip():
        st.warning(
            "Please enter a question."
        )

    elif not st.session_state.documents_processed:
        st.warning(
            "Please upload and process "
            "course materials first."
        )

    else:
        with st.spinner(
            "Searching course materials "
            "and generating an answer..."
        ):

            try:
                # Load open-source language model
                llm = load_llm()

                # Initialise FAISS retriever
                retriever = Retriever()

                # Retrieve relevant document chunks
                context = retriever.retrieve(
                    question,
                    top_k=3
                )

                if not context:
                    st.warning(
                        "No relevant information was "
                        "found in the uploaded materials."
                    )

                else:
                    # Create RAG prompt
                    prompt = get_prompt(
                        context,
                        question
                    )

                    # Generate answer
                    response = llm(prompt)

                    # Display answer
                    st.markdown("### 📘 Answer")

                    if (
                        isinstance(response, list)
                        and response
                        and isinstance(response[0], dict)
                    ):
                        answer = response[0].get(
                            "generated_text",
                            str(response)
                        )
                    else:
                        answer = str(response)

                    st.write(answer)

                    st.caption(
                        "AI-generated answers may "
                        "contain errors. Verify important "
                        "information against your "
                        "course materials."
                    )

            except Exception as e:
                st.error(
                    f"Unable to generate an answer: {e}"
                )


# ---------------------------------
# Footer
# ---------------------------------
st.divider()

st.caption(
    "📚 Course-Aware AI Teaching Assistant | "
    "Powered by Streamlit, Hugging Face and FAISS"
)

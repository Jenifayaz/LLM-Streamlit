import faiss
import os
import pickle
from models.embedding_model import load_embedding_model

VECTOR_DIR = "embeddings/vector_store"

def ingest(chunks):
    # Ensure vector store directory exists
    os.makedirs(VECTOR_DIR, exist_ok=True)

    model = load_embedding_model()
    embeddings = model.encode(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, os.path.join(VECTOR_DIR, "index.faiss"))

    # Save text chunks
    with open(os.path.join(VECTOR_DIR, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

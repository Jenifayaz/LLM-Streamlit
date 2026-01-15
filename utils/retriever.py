import faiss
import pickle
from models.embedding_model import load_embedding_model
import numpy as np

VECTOR_DIR = "embeddings/vector_store"

class Retriever:
    def __init__(self):
        # Load FAISS index
        self.index = faiss.read_index(f"{VECTOR_DIR}/index.faiss")
        # Load text chunks
        with open(f"{VECTOR_DIR}/chunks.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        # Load embedding model
        self.embed_model = load_embedding_model()

    def retrieve(self, query, top_k=3):
        q_emb = self.embed_model.encode([query])
        D, I = self.index.search(np.array(q_emb).astype("float32"), top_k)
        results = [self.chunks[i] for i in I[0]]
        return "\n".join(results)

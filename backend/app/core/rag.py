import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# DevOps: Load paths from Environment or default to Docker volume
DATA_PATH = os.getenv("DATA_PATH", "/app/data")
ARTIFACTS_DIR = os.path.join(DATA_PATH, "artifacts")

class RAGService:
    def __init__(self):
        self.index = None
        self.chunks = None
        self.bm25 = None
        self.model = None
        self.is_ready = False

    def load_artifacts(self):
        """Loads the FAISS index, Metadata, and BM25 from disk."""
        print(f"[INFO] Loading RAG Artifacts from {ARTIFACTS_DIR}...")
        try:
            # 1. Load Embedding Model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 2. Load FAISS Index
            index_path = os.path.join(ARTIFACTS_DIR, "vector_index.faiss")
            self.index = faiss.read_index(index_path)
            
            # 3. Load Metadata (Text Chunks)
            with open(os.path.join(ARTIFACTS_DIR, "metadata.pkl"), "rb") as f:
                self.chunks = pickle.load(f)

            # 4. Load BM25
            with open(os.path.join(ARTIFACTS_DIR, "bm25.pkl"), "rb") as f:
                self.bm25 = pickle.load(f)
                
            self.is_ready = True
            print("[SUCCESS] RAG Service Loaded and Ready.")
        except Exception as e:
            print(f"[ERROR] Failed to load artifacts: {e}")
            print("Did you run ingest.py? The API will start but cannot answer questions.")
            self.is_ready = False

    def search(self, query: str, k: int = 3):
        """
        Retrieves top K chunks using Vector Search (Dense).
        (Future: Hybrid search can be enabled here using self.bm25)
        """
        if not self.is_ready:
            return []

        query_vector = self.model.encode([query])
        
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                results.append(self.chunks[idx])
        
        return results

    def generate_answer(self, query: str, retrieved_chunks: list):
        """
        Synthesizes a readable answer from the retrieved chunks.
        """
        if not retrieved_chunks:
            return {
                "answer": "I'm sorry, I couldn't find any information about that in the official documents.",
                "sources": []
            }

        # 1. Extract Sources
        sources_list = []
        context_text = ""
        
        for chunk in retrieved_chunks:
            source = chunk['metadata']['source']
            page = chunk['metadata']['page']
            text = chunk['text'].replace("\n", " ").strip()
            
            context_text += f"- {text}\n"
            sources_list.append(f"{source} (Page {page})")

        # 2. Simple Synthesis (Template based for CPU speed)
        # If you add an LLM later, this is where you call OpenAI/Llama
        answer = (
            f"Based on your query '{query}', here is the relevant information:\n\n"
            f"{context_text}\n"
        )

        return {
            "answer": answer,
            "sources": list(set(sources_list)) # Remove duplicates
        }

# Singleton Instance
rag_service = RAGService()
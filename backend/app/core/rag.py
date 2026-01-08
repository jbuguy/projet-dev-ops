import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Paths matching your Docker Volume
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
        """Loads the ML models and indices into memory."""
        print("Loading RAG Artifacts...")
        try:
            # 1. Load Embedding Model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 2. Load FAISS Index
            self.index = faiss.read_index(os.path.join(ARTIFACTS_DIR, "vector_index.faiss"))
            
            # 3. Load Metadata (Chunks)
            with open(os.path.join(ARTIFACTS_DIR, "metadata.pkl"), "rb") as f:
                self.chunks = pickle.load(f)

            # 4. Load BM25
            with open(os.path.join(ARTIFACTS_DIR, "bm25.pkl"), "rb") as f:
                self.bm25 = pickle.load(f)
                
            self.is_ready = True
            print("✅ RAG Service Loaded Successfully.")
        except Exception as e:
            print(f"⚠️ Warning: Could not load artifacts: {e}")
            print("Server will start, but RAG features will fail until data is ingested.")
            self.is_ready = False

    def search(self, query: str, k: int = 3):
        """
        Performs Hybrid Search:
        1. Dense Retrieval (FAISS)
        2. Sparse Retrieval (BM25) - Optional, simplified here to just FAISS for MVP stability
        """
        if not self.is_ready:
            return []

        # 1. Vector Search (Semantic)
        query_vector = self.model.encode([query])
        # Search FAISS
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
        
        return results

    def generate_answer(self, query: str, retrieved_chunks: list):
        """
        Synthesizes an answer based on retrieved docs.
        For CPU optimization, we use a structured 'Extractive' approach 
        instead of a heavy Seq2Seq model for now.
        """
        if not retrieved_chunks:
            return "I couldn't find any information relevant to your query in the documents."

        # Construct a digestible response
        answer = "Based on the official documents, here is what I found:\n\n"
        
        sources = set()
        for i, chunk in enumerate(retrieved_chunks):
            # Clean up newlines for display
            text = chunk['text'].replace("\n", " ").strip()
            source = chunk['metadata']['source']
            page = chunk['metadata']['page']
            
            answer += f"• {text}...\n"
            sources.add(f"{source} (Page {page})")

        return {
            "answer": answer,
            "sources": list(sources),
            "raw_chunks": retrieved_chunks
        }

# Singleton instance
rag_service = RAGService()
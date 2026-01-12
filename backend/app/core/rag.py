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
            self.model = SentenceTransformer('all-mpnet-base-v2')

            index_path = os.path.join(ARTIFACTS_DIR, "vector_index.faiss")
            self.index = faiss.read_index(index_path)

            with open(os.path.join(ARTIFACTS_DIR, "metadata.pkl"), "rb") as f:
                self.chunks = pickle.load(f)

            with open(os.path.join(ARTIFACTS_DIR, "bm25.pkl"), "rb") as f:
                self.bm25 = pickle.load(f)

            self.is_ready = True
            print("[SUCCESS] RAG Service Loaded and Ready.")
        except Exception as e:
            print(f"[ERROR] Failed to load artifacts: {e}")
            self.is_ready = False

    def _contextualize_query(self, query: str, history: list) -> str:
        """
        Rewrites the query to include context from history.
        NOTE: Ideally, an LLM does this. Without an LLM, we use a heuristic:
        We prepend the *previous user query* to the current one.

        Example:
        1. User: "What is Docker?"
        2. User: "How do I install it?" 
        -> Rewritten: "What is Docker? How do I install it?" (Helps vector search find 'Docker')
        """
        if not history:
            return query

        # Get the last message sent by the user
        last_user_msg = next((m['content'] for m in reversed(
            history) if m['role'] == 'user'), None)

        if last_user_msg:
            print(
                f"[DEBUG] Rewriting query with context: {last_user_msg[:50]}...")
            return f"{last_user_msg} {query}"

        return query

    def search(self, query: str, history: list = [], k: int = 3):
        """
        Retrieves chunks based on a context-aware query.
        """
        if not self.is_ready:
            return [], query

        # 1. Rewrite Query for Context
        contextual_query = self._contextualize_query(query, history)

        # 2. Vector Search using the rewritten query
        query_vector = self.model.encode([contextual_query])
        distances, indices = self.index.search(
            np.array(query_vector).astype('float32'), k)

        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                results.append(self.chunks[idx])

        return results, contextual_query

    def generate_answer(self, query: str, retrieved_chunks: list):
        if not retrieved_chunks:
            return {
                "answer": "I'm sorry, I couldn't find any information about that in the official documents.",
                "sources": []
            }

        sources_list = []
        context_text = ""

        for chunk in retrieved_chunks:
            source = chunk['metadata']['source']
            page = chunk['metadata']['page']
            text = chunk['text'].replace("\n", " ").strip()

            context_text += f"- {text}\n"
            sources_list.append(f"{source} (Page {page})")

        # In a real LLM setup, you would also pass the 'history' here so the LLM knows the conversation flow.
        answer = (
            f"Based on your query '{query}', here is the relevant information:\n\n"
            f"{context_text}\n"
        )

        return {
            "answer": answer,
            "sources": list(set(sources_list))
        }


rag_service = RAGService()

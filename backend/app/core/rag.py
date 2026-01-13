import os
import pickle
import numpy as np
import faiss
import time

from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

DATA_PATH = os.getenv("DATA_PATH", "/app/data")
ARTIFACTS_DIR = os.path.join(DATA_PATH, "artifacts")

# LLM Configuration
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct:novita"
HF_TOKEN = os.getenv("HF_TOKEN")


class RAGService:
    def __init__(self):
        self.index = None
        self.chunks = None
        self.bm25 = None
        self.model = None       # Embedding model
        self.llm = None         # InferenceClient
        self.is_ready = False

    def load_artifacts(self):
        """Loads FAISS index, metadata, BM25, embedding model, and LLM client."""
        print(f"[INFO] Loading RAG artifacts from {ARTIFACTS_DIR}...")
        start_total = time.time()

        try:
            # 1. Load embedding model
            start = time.time()
            print("[INFO] Loading embedding model...")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            print(f"[INFO] ✓ Embedding model loaded in {time.time() - start:.2f}s")

            # 2. Load FAISS index
            start = time.time()
            index_path = os.path.join(ARTIFACTS_DIR, "vector_index.faiss")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"FAISS index not found at {index_path}")
            self.index = faiss.read_index(index_path)
            print(f"[INFO] ✓ FAISS index loaded in {time.time() - start:.2f}s")

            # 3. Load metadata (chunks)
            start = time.time()
            metadata_path = os.path.join(ARTIFACTS_DIR, "metadata.pkl")
            with open(metadata_path, "rb") as f:
                self.chunks = pickle.load(f)
            print(f"[INFO] ✓ Metadata loaded in {time.time() - start:.2f}s")

            # 4. Load BM25 (optional, hybrid search support)
            start = time.time()
            bm25_path = os.path.join(ARTIFACTS_DIR, "bm25.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    self.bm25 = pickle.load(f)
                print(f"[INFO] ✓ BM25 loaded in {time.time() - start:.2f}s")

            # 5. Initialize Hugging Face Inference Client
            if HF_TOKEN:
                start = time.time()
                print("[INFO] Initializing Hugging Face InferenceClient...")
                self.llm = InferenceClient(api_key=HF_TOKEN)
                print(f"[INFO] ✓ InferenceClient ready in {time.time() - start:.2f}s")
            else:
                print("[WARNING] HF_TOKEN not set. Running in retrieval-only mode.")
                self.llm = None

            self.is_ready = True
            print(f"[SUCCESS] RAG Service ready in {time.time() - start_total:.2f}s")

        except Exception as e:
            print(f"[ERROR] Failed to load RAG artifacts: {e}")
            self.is_ready = False

    def _contextualize_query(self, query: str, history: list) -> str:
        """
        Rewrites the query using conversation history (last user message).
        """
        if not history:
            return query

        last_user_msg = next(
            (m["content"] for m in reversed(history) if m["role"] == "user"),
            None,
        )

        if last_user_msg:
            print("[DEBUG] Contextualizing query...")
            return f"{last_user_msg} {query}"

        return query

    def search(self, query: str, history: list = [], k: int = 3):
        """
        Dense vector search using FAISS.
        """
        if not self.is_ready:
            return [], query

        contextual_query = self._contextualize_query(query, history)

        query_vector = self.model.encode([contextual_query])
        distances, indices = self.index.search(
            np.array(query_vector).astype("float32"), k
        )

        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                results.append(self.chunks[idx])

        return results, contextual_query

    def generate_answer(self, query: str, retrieved_chunks: list):
        """
        Generate answer using Llama-3.1-8B-Instruct via HF Inference API.
        """
        if not retrieved_chunks:
            return {
                "answer": "I'm sorry, I couldn't find any information about that in the official documents.",
                "sources": [],
            }

        # Prepare context and sources
        context_text = ""
        sources = set()

        for chunk in retrieved_chunks:
            text = chunk["text"].replace("\n", " ").strip()
            source = chunk["metadata"]["source"]
            page = chunk["metadata"].get("page", "N/A")

            context_text += f"- {text}\n"
            sources.add(f"{source} (Page {page})")

        if self.llm:
            try:
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful university assistant. "
                            "Answer ONLY using the provided context. "
                            "If the answer is not present, say: "
                            "'I don't know based on the documents.'"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"""
Context:
{context_text}

Question:
{query}
""",
                    },
                ]

                completion = self.llm.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.1,
                )

                answer = completion.choices[0].message.content.strip()

            except Exception as e:
                print(f"[ERROR] LLM generation failed: {e}")
                answer = (
                    "I found relevant documents, but I couldn't generate a summary right now."
                )
        else:
            answer = (
                f"Based on your query '{query}', here are the relevant excerpts:\n\n"
                f"{context_text}"
            )

        return {
            "answer": answer,
            "sources": list(sources),
        }


# Singleton instance
rag_service = RAGService()

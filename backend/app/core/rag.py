import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

DATA_PATH = os.getenv("DATA_PATH", "/app/data")
ARTIFACTS_DIR = os.path.join(DATA_PATH, "artifacts")

# Configuration for the LLM (From main branch)
REPO_ID = "HuggingFaceH4/zephyr-7b-beta"
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

class RAGService:
    def __init__(self):
        self.index = None
        self.chunks = None
        self.bm25 = None
        self.model = None  # The Embedding Model (Search)
        self.llm = None    # The Generation Model (Chat)
        self.is_ready = False

    def load_artifacts(self):
        """Loads the FAISS index, Metadata, BM25, and initializes the LLM."""
        print(f"[INFO] Loading RAG Artifacts from {ARTIFACTS_DIR}...")
        try:
            # 1. Load Embedding Model
            # Using 'paraphrase-multilingual' from main (better for mixed languages)
            self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            
            # 2. Load FAISS Index
            index_path = os.path.join(ARTIFACTS_DIR, "vector_index.faiss")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Index not found at {index_path}")
            self.index = faiss.read_index(index_path)

            # 3. Load Metadata (Chunks)
            with open(os.path.join(ARTIFACTS_DIR, "metadata.pkl"), "rb") as f:
                self.chunks = pickle.load(f)

            # 4. Load BM25 (From yassinenew - keeps compatibility for hybrid search)
            bm25_path = os.path.join(ARTIFACTS_DIR, "bm25.pkl")
            if os.path.exists(bm25_path):
                with open(bm25_path, "rb") as f:
                    self.bm25 = pickle.load(f)

            # 5. Initialize the LLM (Hugging Face API from main)
            if HF_TOKEN:
                print("[INFO] Connecting to Hugging Face API...")
                self.llm = HuggingFaceEndpoint(
                    repo_id=REPO_ID,
                    task="text-generation",
                    max_new_tokens=512,
                    top_k=30,
                    temperature=0.1,  # Keep it factual
                    huggingfacehub_api_token=HF_TOKEN
                )
            else:
                print("[WARNING] No HUGGINGFACEHUB_API_TOKEN found. Using simple fallback mode.")
                self.llm = None

            self.is_ready = True
            print("[SUCCESS] RAG Service Loaded and Ready.")
        except Exception as e:
            print(f"[ERROR] Failed to load artifacts: {e}")
            self.is_ready = False

    def _contextualize_query(self, query: str, history: list) -> str:
        """
        Rewrites the query to include context from history.
        (Feature from yassinenew)
        """
        if not history:
            return query

        # Get the last message sent by the user
        last_user_msg = next((m['content'] for m in reversed(history) if m['role'] == 'user'), None)

        if last_user_msg:
            print(f"[DEBUG] Rewriting query with context: {last_user_msg[:50]}...")
            return f"{last_user_msg} {query}"

        return query

    def search(self, query: str, history: list = [], k: int = 3):
        """
        Retrieves top K chunks using Vector Search (Dense).
        """
        if not self.is_ready:
            return [], query

        # 1. Rewrite Query for Context (Using logic from yassinenew)
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
        """
        Synthesizes a readable answer using Zephyr-7B (from main).
        """
        if not retrieved_chunks:
            return {
                "answer": "I'm sorry, I couldn't find any information about that in the official documents.",
                "sources": []
            }

        # 1. Prepare Context & Sources
        sources_list = []
        context_text = ""
        for chunk in retrieved_chunks:
            source_name = chunk['metadata']['source']
            page_num = chunk['metadata'].get('page', 'N/A')
            text = chunk['text'].replace("\n", " ").strip()

            context_text += f"- {text}\n"
            sources_list.append(f"{source_name} (Page {page_num})")

        sources_list = list(set(sources_list))

        # 2. Generate Answer (LLM vs Simple Fallback)
        if self.llm:
            try:
                # The Prompt Template
                template = """
                You are a helpful university assistant. Answer the question based ONLY on the context provided below.
                If the answer is not in the context, say "I don't know based on the documents."
                
                Context:
                {context}
                
                Question:
                {question}
                
                Answer:
                """
                prompt = PromptTemplate(template=template, input_variables=["context", "question"])
                
                # Create chain and invoke
                chain = prompt | self.llm
                answer = chain.invoke({"context": context_text, "question": query})
                
                # Cleanup: Sometimes models leave trailing text
                answer = answer.strip()
            except Exception as e:
                print(f"[ERROR] LLM Generation failed: {e}")
                answer = "I found relevant documents, but I couldn't generate a summary right now. Please check the sources below."
        else:
            # Fallback (Simple Template if no API Key)
            answer = (
                f"Based on your query '{query}', here is the relevant text from the documents:\n\n"
                f"{context_text}\n"
            )

        return {
            "answer": answer,
            "sources": sources_list
        }

rag_service = RAGService()
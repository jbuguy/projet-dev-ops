import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

DATA_PATH = os.getenv("DATA_PATH", "/app/data")
ARTIFACTS_DIR = os.path.join(DATA_PATH, "artifacts")

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
        """Loads the FAISS index, Metadata, and initializes the LLM."""
        print(f"[INFO] Loading RAG Artifacts from {ARTIFACTS_DIR}...")
        try:
            # 1. Load Embedding Model (MUST match ingest.py!)
            # We switched to the multilingual model (768 dimensions)
            self.model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            
            # 2. Load FAISS Index
            index_path = os.path.join(ARTIFACTS_DIR, "vector_index.faiss")
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Index not found at {index_path}")
            self.index = faiss.read_index(index_path)
            
            # 3. Load Metadata (Text Chunks)
            with open(os.path.join(ARTIFACTS_DIR, "metadata.pkl"), "rb") as f:
                self.chunks = pickle.load(f)

            # 4. Initialize the LLM (Hugging Face API)
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
            print("Did you run ingest.py? The API will start but cannot answer questions.")
            self.is_ready = False

    def search(self, query: str, k: int = 3):
        """
        Retrieves top K chunks using Vector Search (Dense).
        """
        if not self.is_ready:
            return []

        # Convert query to vector (768 dimensions)
        query_vector = self.model.encode([query])
        
        # Search FAISS
        distances, indices = self.index.search(np.array(query_vector).astype('float32'), k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.chunks):
                results.append(self.chunks[idx])
        
        return results

    def generate_answer(self, query: str, retrieved_chunks: list):
        """
        Synthesizes a readable answer using Zephyr-7B (or fallback).
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

        # 2. Generate Answer (LLM vs Simple)
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
                # Note: Newer LangChain uses the pipe | syntax
                chain = prompt | self.llm
                answer = chain.invoke({"context": context_text, "question": query})
                
                # Cleanup: Sometimes models leave trailing text
                answer = answer.strip()

            except Exception as e:
                print(f"[ERROR] LLM Generation failed: {e}")
                answer = "I found relevant documents, but I couldn't generate a summary right now. Please check the sources below."
        else:
            # Fallback (Simple Template)
            answer = (
                f"Based on your query '{query}', here is the relevant text from the documents:\n\n"
                f"{context_text}\n"
            )

        return {
            "answer": answer,
            "sources": sources_list
        }

# Singleton Instance
rag_service = RAGService()
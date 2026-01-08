import os
import pickle
import numpy as np
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi

# --- DevOps Configuration ---
# We use environment variables or defaults matching the Docker volume
DATA_PATH = os.getenv("DATA_PATH", "/app/data")
RAW_DATA_DIR = os.path.join(DATA_PATH, "raw")
ARTIFACTS_DIR = os.path.join(DATA_PATH, "artifacts")

# Ensure output directory exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def main():
    print(f"--- Starting Ingestion Pipeline ---")
    print(f"Reading from: {RAW_DATA_DIR}")
    print(f"Saving to:   {ARTIFACTS_DIR}")

    # ---------------------------------------------------------
    # STEP 1: LOAD (Your Logic)
    # ---------------------------------------------------------
    if not os.path.exists(RAW_DATA_DIR) or not os.listdir(RAW_DATA_DIR):
        print("ERROR: No files found in raw directory. Did DVC pull work?")
        return

    loader = DirectoryLoader(
        RAW_DATA_DIR, 
        glob="./*.pdf", 
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    # ---------------------------------------------------------
    # STEP 2: CHUNK (Your Logic - Excellent Choice)
    # ---------------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Reduced slightly for better retrieval precision
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # LangChain returns "Document" objects
    split_docs = text_splitter.split_documents(documents)
    print(f"Generated {len(split_docs)} chunks.")

    # Convert to a simple dictionary format for our internal use
    chunks = []
    for doc in split_docs:
        chunks.append({
            "text": doc.page_content,
            "metadata": {
                "source": os.path.basename(doc.metadata.get("source", "unknown")),
                "page": doc.metadata.get("page", 0) + 1
            }
        })

    # ---------------------------------------------------------
    # STEP 3: EMBED (New Logic needed for Phase 2)
    # ---------------------------------------------------------
    print("Loading Embedding Model (all-MiniLM-L6-v2)...")
    # This runs on CPU efficiently
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create vector representations of the text
    text_content = [c["text"] for c in chunks]
    embeddings = model.encode(text_content)
    print(f"Embeddings shape: {embeddings.shape}")

    # ---------------------------------------------------------
    # STEP 4: BUILD INDICES (FAISS + BM25)
    # ---------------------------------------------------------
    
    # A. FAISS (Dense Index)
    print("Building FAISS Index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    # B. BM25 (Sparse Index - for keyword matching)
    print("Building BM25 Index...")
    tokenized_corpus = [doc.split(" ") for doc in text_content]
    bm25 = BM25Okapi(tokenized_corpus)

    # ---------------------------------------------------------
    # STEP 5: SAVE ARTIFACTS
    # ---------------------------------------------------------
    print("Saving artifacts...")
    
    # Save the FAISS index
    faiss.write_index(index, os.path.join(ARTIFACTS_DIR, "vector_index.faiss"))
    
    # Save the Data (Chunks) and BM25 object via Pickle
    # We save 'chunks' so we can retrieve the actual text later when the user asks a Q
    with open(os.path.join(ARTIFACTS_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(chunks, f)
        
    with open(os.path.join(ARTIFACTS_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    print("--- Pipeline Finished Successfully ---")

if __name__ == "__main__":
    main()
import os
import glob
import pickle
import numpy as np
import faiss
import mlflow
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DATA_PATH = os.getenv("DATA_PATH", "/app/data")
RAW_DATA_DIR = os.path.join(DATA_PATH, "raw")
ARTIFACTS_DIR = os.path.join(DATA_PATH, "artifacts")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_existing_artifacts():
    """
    Loads existing metadata and index to avoid re-processing old files.
    """
    meta_path = os.path.join(ARTIFACTS_DIR, "metadata.pkl")
    faiss_path = os.path.join(ARTIFACTS_DIR, "vector_index.faiss")
    bm25_path = os.path.join(ARTIFACTS_DIR, "bm25.pkl")

    if os.path.exists(meta_path) and os.path.exists(faiss_path) and os.path.exists(bm25_path):
        print("[INFO] Loading existing artifacts for incremental update...")
        with open(meta_path, "rb") as f:
            existing_chunks = pickle.load(f)
        
        index = faiss.read_index(faiss_path)
        
        processed_files = set(c['metadata']['source'] for c in existing_chunks)
        
        return existing_chunks, index, processed_files
    else:
        print("[INFO] No existing artifacts found. Starting fresh.")
        return [], None, set()


def load_new_documents(directory, processed_files):
    """
    Scans directory and loads ONLY files that haven't been processed yet.
    """
    documents = []
    
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    all_files = pdf_files + txt_files
    
    new_files_count = 0
    
    for filepath in all_files:
        filename = os.path.basename(filepath)
        if filename in processed_files:
            print(f"   [SKIP] Already processed: {filename}")
            continue
        try:
            if filepath.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filepath.endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            documents.extend(docs)
            new_files_count += 1
            print(f"   [NEW] Loaded: {filename}")
        except Exception as e:
            print(f"   [ERR] Error loading {filename}: {e}")
    return documents, new_files_count


def main():
    print(f"--- Starting Incremental Ingestion Pipeline ---")
    
    # MLflow Setup
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Ensure this experiment name matches what you use elsewhere
    mlflow.set_experiment("DevOps_RAG_Context")
    
    with mlflow.start_run():
        # 1. Configuration
        CHUNK_SIZE = 500
        CHUNK_OVERLAP = 50
        
        # CRITICAL: This MUST match the model used in rag_service.py
        # Using the fast, efficient MiniLM model (80MB vs 500MB for multilingual)
        EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
        
        mlflow.log_param("chunk_size", CHUNK_SIZE)
        mlflow.log_param("chunk_overlap", CHUNK_OVERLAP)
        mlflow.log_param("embedding_model", EMBEDDING_MODEL)

        # 2. Load State
        existing_chunks, index, processed_files = load_existing_artifacts()

        # 3. Load Only New Docs
        new_docs, new_count = load_new_documents(RAW_DATA_DIR, processed_files)
        
        mlflow.log_metric("new_files_count", new_count)

        if new_count == 0:
            print("[INFO] No new files to process. Index is up to date.")
            return

        print(f"[INFO] Processing {len(new_docs)} new pages/documents...")

        # 4. Chunk New Data
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        split_docs = text_splitter.split_documents(new_docs)
        print(f"[INFO] Generated {len(split_docs)} new chunks.")
        
        mlflow.log_metric("new_chunks_generated", len(split_docs)) 

        new_chunks = []
        for doc in split_docs:
            source_name = os.path.basename(doc.metadata.get("source", "unknown"))
            page_num = doc.metadata.get("page", 0) + 1
            
            new_chunks.append({
                "text": doc.page_content,
                "metadata": {
                    "source": source_name,
                    "page": page_num
                }
            })

        # 5. Embed New Chunks
        print("[INFO] Generating embeddings for new data...")
        model = SentenceTransformer(EMBEDDING_MODEL)
        new_embeddings = model.encode([c["text"] for c in new_chunks], show_progress_bar=True)

        # 6. Update Indices
        dimension = new_embeddings.shape[1]
        if index is None:
            print("[INFO] Creating new FAISS index...")
            index = faiss.IndexFlatL2(dimension)
        
        print(f"[INFO] Adding {len(new_embeddings)} vectors to FAISS...")
        index.add(np.array(new_embeddings).astype('float32'))

        # Merge Metadata
        all_chunks = existing_chunks + new_chunks
        
        # Log total corpus size
        mlflow.log_metric("total_corpus_chunks", len(all_chunks))

        # Rebuild BM25
        print("[INFO] Rebuilding BM25 Index (Full Corpus)...")
        tokenized_corpus = [c["text"].split(" ") for c in all_chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        # 7. Save Everything
        print("[INFO] Saving updated artifacts...")
        faiss.write_index(index, os.path.join(ARTIFACTS_DIR, "vector_index.faiss"))
        
        with open(os.path.join(ARTIFACTS_DIR, "metadata.pkl"), "wb") as f:
            pickle.dump(all_chunks, f)
            
        with open(os.path.join(ARTIFACTS_DIR, "bm25.pkl"), "wb") as f:
            pickle.dump(bm25, f)
        print(f"--- Success! Total documents in index: {len(processed_files) + new_count} ---")
        
        
if __name__ == "__main__":
    main()
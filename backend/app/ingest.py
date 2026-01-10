import os
import glob
import pickle
import numpy as np
import faiss
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# --- DevOps Configuration ---
DATA_PATH = os.getenv("DATA_PATH", "/app/data")
RAW_DATA_DIR = os.path.join(DATA_PATH, "raw")
ARTIFACTS_DIR = os.path.join(DATA_PATH, "artifacts")

# Ensure output directory exists
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
        
        # Create a set of already processed filenames
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
        
        # SKIP if already processed
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
    
    # 1. Load State
    existing_chunks, index, processed_files = load_existing_artifacts()

    # 2. Load Only New Docs
    new_docs, new_count = load_new_documents(RAW_DATA_DIR, processed_files)
    
    if new_count == 0:
        print("[INFO] No new files to process. Index is up to date.")
        return

    print(f"[INFO] Processing {len(new_docs)} new pages/documents...")

    # 3. Chunk New Data
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(new_docs)
    print(f"[INFO] Generated {len(split_docs)} new chunks.")

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

    # 4. Embed New Chunks
    print("[INFO] Generating embeddings for new data...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    new_embeddings = model.encode([c["text"] for c in new_chunks], show_progress_bar=True)

    # 5. Update Indices
    
    # A. Update FAISS
    dimension = new_embeddings.shape[1]
    if index is None:
        # Create new if didn't exist
        print("[INFO] Creating new FAISS index...")
        index = faiss.IndexFlatL2(dimension)
    
    print(f"[INFO] Adding {len(new_embeddings)} vectors to FAISS...")
    index.add(np.array(new_embeddings).astype('float32'))

    # B. Merge Metadata
    all_chunks = existing_chunks + new_chunks

    # C. Rebuild BM25 (BM25 must be rebuilt fully to calculate frequencies correctly)
    print("[INFO] Rebuilding BM25 Index (Full Corpus)...")
    tokenized_corpus = [c["text"].split(" ") for c in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # 6. Save Everything
    print("[INFO] Saving updated artifacts...")
    faiss.write_index(index, os.path.join(ARTIFACTS_DIR, "vector_index.faiss"))
    
    with open(os.path.join(ARTIFACTS_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(all_chunks, f)
        
    with open(os.path.join(ARTIFACTS_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    print(f"--- Success! Total documents in index: {len(processed_files) + new_count} ---")

if __name__ == "__main__":
    main()
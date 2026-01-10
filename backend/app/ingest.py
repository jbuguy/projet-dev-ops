import os
import glob
import pickle
import numpy as np
import faiss
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

DATA_PATH = os.getenv("DATA_PATH", "/app/data")
RAW_DATA_DIR = os.path.join(DATA_PATH, "raw")
ARTIFACTS_DIR = os.path.join(DATA_PATH, "artifacts")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_documents(directory):
    """
    Scans a directory for PDF and TXT files and loads them using LangChain loaders.
    """
    documents = []
    
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    all_files = pdf_files + txt_files
    
    if not all_files:
        print(f"[WARN] No files found in {directory}. Did DVC pull work?")
        return []

    print(f"[INFO] Found {len(all_files)} files. Loading...")

    for filepath in all_files:
        try:
            if filepath.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
            elif filepath.endswith(".txt"):
                loader = TextLoader(filepath, encoding="utf-8")
            
            docs = loader.load()
            documents.extend(docs)
            print(f"   [OK] Loaded: {os.path.basename(filepath)}")
        except Exception as e:
            print(f"   [ERR] Error loading {os.path.basename(filepath)}: {e}")

    return documents

def main():
    print(f"--- Starting Ingestion Pipeline ---")
    print(f"Reading from: {RAW_DATA_DIR}")
    print(f"Saving to:    {ARTIFACTS_DIR}")

    raw_docs = load_documents(RAW_DATA_DIR)
    if not raw_docs:
        print("[ERR] Stopping pipeline: No documents loaded.")
        return

    print(f"[INFO] Total raw pages/documents loaded: {len(raw_docs)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(raw_docs)
    print(f"[INFO] Generated {len(split_docs)} chunks.")

    chunks = []
    for doc in split_docs:
        source_name = os.path.basename(doc.metadata.get("source", "unknown"))
        page_num = doc.metadata.get("page", 0) + 1
        
        chunks.append({
            "text": doc.page_content,
            "metadata": {
                "source": source_name,
                "page": page_num
            }
        })

    print("[INFO] Loading Embedding Model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    text_content = [c["text"] for c in chunks]
    embeddings = model.encode(text_content, show_progress_bar=True)
    
    print("[INFO] Building FAISS Index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    print("[INFO] Building BM25 Index...")
    tokenized_corpus = [doc.split(" ") for doc in text_content]
    bm25 = BM25Okapi(tokenized_corpus)

    print("[INFO] Saving artifacts to disk...")
    
    faiss.write_index(index, os.path.join(ARTIFACTS_DIR, "vector_index.faiss"))
    
    with open(os.path.join(ARTIFACTS_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(chunks, f)
        
    with open(os.path.join(ARTIFACTS_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    print("--- Pipeline Finished Successfully ---")
    print(f"Files created in {ARTIFACTS_DIR}:")
    print(os.listdir(ARTIFACTS_DIR))

if __name__ == "__main__":
    main()
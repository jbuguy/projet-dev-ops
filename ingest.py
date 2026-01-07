import os
import json
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuration
SOURCE_DIRECTORY = "source_documents"
OUTPUT_FILE = "processed_chunks.json"

def ingest_and_chunk():
    """
    Loads PDFs from the source directory, cleans them, chunks them,
    and saves them with metadata for the retrieval step.
    """
    print(f"Loading documents from {SOURCE_DIRECTORY}...")
    
    # 1. Load PDFs
    # We use DirectoryLoader with PyPDFLoader to handle all PDFs in the folder
    loader = DirectoryLoader(
        SOURCE_DIRECTORY, 
        glob="./*.pdf", 
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from raw documents.")

    # 2. Chunking Strategy
    # Using RecursiveCharacterTextSplitter to respect sentence boundaries.
    # Chunk size 1000 with overlap 100 is a standard starting point for RAG.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    # 3. Serialize and Save
    # We convert the Document objects to a simplified JSON format.
    # This preserves 'source' and 'page' metadata required for citations.
    serialized_chunks = []
    for chunk in chunks:
        serialized_chunks.append({
            "content": chunk.page_content,
            "metadata": {
                # Clean up the path to just get the filename for cleaner citations later
                "source": os.path.basename(chunk.metadata.get("source", "unknown")),
                "page": chunk.metadata.get("page", 0) + 1  # 1-based indexing for humans
            }
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(serialized_chunks, f, indent=4, ensure_ascii=False)
    
    print(f"Successfully saved chunks to {OUTPUT_FILE}")

if __name__ == "__main__":
    # Create the directory if it doesn't exist (to prevent errors)
    if not os.path.exists(SOURCE_DIRECTORY):
        os.makedirs(SOURCE_DIRECTORY)
        print(f"Created folder '{SOURCE_DIRECTORY}'. Please put your PDFs inside and run again.")
    else:
        ingest_and_chunk()
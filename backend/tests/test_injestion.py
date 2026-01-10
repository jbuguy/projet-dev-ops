import os
import pytest
import pickle
import numpy as np
from unittest.mock import MagicMock, patch
from app.ingest import load_new_documents, main

# --- Test 1: File Loading Logic (Does it find files?) ---
def test_load_new_documents(tmp_path):
    """
    Create fake files and ensure the loader finds them.
    """
    # 1. Setup fake directory
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    
    # 2. Create a dummy text file
    (raw_dir / "test.txt").write_text("Hello DevOps world!", encoding="utf-8")
    
    # 3. Run your function
    processed_files = set() # simulating empty history
    docs, count = load_new_documents(str(raw_dir), processed_files)
    
    # 4. Assertions
    assert count == 1
    assert len(docs) == 1
    assert docs[0].page_content == "Hello DevOps world!"

# --- Test 2: The Full Pipeline (Mocking the AI) ---
@patch("app.ingest.SentenceTransformer")  # Fake the Embedding Model
@patch("app.ingest.faiss")                # Fake the Vector Database
def test_full_ingestion_flow(mock_faiss, mock_model_class, tmp_path, monkeypatch):
    """
    Tests the entire main() function by mocking out the slow AI parts.
    This proves the 'plumbing' works without needing a GPU.
    """
    
    # A. Setup fake environment variables to use temp dir
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    artifacts_dir = data_dir / "artifacts"
    
    data_dir.mkdir()
    raw_dir.mkdir()
    artifacts_dir.mkdir()
    # Point the environment variable DATA_PATH to our temp dir
    monkeypatch.setenv("DATA_PATH", str(data_dir))

    # B. Create a dummy input file
    (raw_dir / "policy.txt").write_text("This is a policy document about fees.", encoding="utf-8")

    # C. Configure the Mocks (The Fake AI)
    # 1. Mock the Embedding Model instance
    mock_model_instance = MagicMock()
    mock_model_class.return_value = mock_model_instance
    # When encode() is called, return a fake vector (size 768)
    mock_model_instance.encode.return_value = np.random.rand(1, 768).astype('float32')

    # 2. Mock the FAISS index
    mock_index = MagicMock()
    mock_faiss.IndexFlatL2.return_value = mock_index
    # We also need to mock read_index so it returns None initially
    mock_faiss.read_index.side_effect = Exception("No index found") 

    # D. Run the Script
    # We need to import main locally or ensure paths are correct
    # (Assuming app.ingest imports are handled correctly)
    
    # Run the ingestion!
    # Note: We need to patch os.path.join or ensure the script uses the env var we set.
    # If your script uses global variables for paths, we might need to patch those specific constants.
    
    with patch("app.ingest.DATA_PATH", str(data_dir)), \
         patch("app.ingest.RAW_DATA_DIR", str(raw_dir)), \
         patch("app.ingest.ARTIFACTS_DIR", str(artifacts_dir)):
         
         from app import ingest
         ingest.main()

    # E. Assertions (Did it try to save things?)
    
    # 1. Did we create the artifacts directory?
    assert os.path.exists(artifacts_dir)
    
    # 2. Did we verify metadata was saved?
    assert os.path.exists(artifacts_dir / "metadata.pkl")
    
    # 3. Did we try to add vectors to FAISS?
    # Verify add() was called once
    assert mock_index.add.called
    
    # 4. Did we try to write the index to disk?
    # Verify write_index was called
    assert mock_faiss.write_index.called
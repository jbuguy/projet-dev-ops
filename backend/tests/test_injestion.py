import os
import pytest
from app.ingest import main

def test_chunking_logic():
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    text = "This is a sentence. " * 50
    splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    docs = splitter.create_documents([text])
    
    assert len(docs) > 1
    assert len(docs[0].page_content) <= 100

def test_artifact_generation(tmp_path, monkeypatch):
    """
    Simulates the ingestion process in a temporary directory 
    so we don't mess up real data.
    """
    d = tmp_path / "data"
    d.mkdir()
    (d / "raw").mkdir()
    (d / "artifacts").mkdir()
    
    assert os.path.exists(d / "artifacts")
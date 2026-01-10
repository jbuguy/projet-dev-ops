import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.rag import rag_service

# TestClient acts like a web browser for your API
client = TestClient(app)

def test_health_check():
    """
    Test 1: Does the /health endpoint return 200 OK?
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_chat_endpoint_success(mocker):
    """
    Test 2: Does the /chat endpoint work when the AI is ready?
    We MOCK the heavy AI parts so this test runs in milliseconds.
    """
    # 1. Trick the API into thinking the RAG service is ready
    mocker.patch.object(rag_service, 'is_ready', True)

    # 2. Fake the search results (so we don't need real PDFs)
    mocker.patch.object(rag_service, 'search', return_value=[
        {
            "text": "The fees are 5000 TND.", 
            "metadata": {"source": "fees.pdf", "page": 1}
        }
    ])

    # 3. Fake the answer generation
    mocker.patch.object(rag_service, 'generate_answer', return_value={
        "answer": "The fees are 5000 TND.",
        "sources": ["fees.pdf (Page 1)"]
    })

    # 4. Make the request
    payload = {"query": "How much does it cost?"}
    response = client.post("/chat", json=payload)

    # 5. Check the result
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The fees are 5000 TND."
    assert "fees.pdf (Page 1)" in data["sources"]

def test_chat_service_not_ready(mocker):
    """
    Test 3: Does the API gracefully handle errors if the AI isn't loaded?
    """
    # Force the service to report 'not ready'
    mocker.patch.object(rag_service, 'is_ready', False)

    payload = {"query": "Hello"}
    response = client.post("/chat", json=payload)

    # It should return a 503 Service Unavailable error
    assert response.status_code == 503
from fastapi.testclient import TestClient
from app.main import app
from app.core.rag import rag_service

# Create a test client
client = TestClient(app)

def test_health_check():
    """Test if the app starts up correctly."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_chat_endpoint_no_data(monkeypatch):
    """
    Test the chat endpoint handling when RAG is NOT ready.
    We force is_ready = False to simulate a fresh CI environment.
    """
    monkeypatch.setattr(rag_service, "is_ready", False)
    
    response = client.post("/chat", json={"query": "Hello"})
    
    # It should fail gracefully with a 503 error
    assert response.status_code == 503
    assert "Search index not ready" in response.json()['detail']
import pytest
from app.core.rag import rag_service


def test_health_check(client): 
    """
    Test 1: Does the /health endpoint return 200 OK?
    The 'client' fixture automatically sets rag_service.is_ready = True
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_chat_endpoint_success(client, mocker): 
    """
    Test 2: Does the /chat endpoint work when the AI is ready?
    """
    mocker.patch.object(rag_service, 'is_ready', True)

    mocker.patch.object(rag_service, 'search', return_value=[
        {
            "text": "The fees are 5000 TND.", 
            "metadata": {"source": "fees.pdf", "page": 1}
        }
    ])

    mocker.patch.object(rag_service, 'generate_answer', return_value={
        "answer": "The fees are 5000 TND.",
        "sources": ["fees.pdf (Page 1)"]
    })

    payload = {"query": "How much does it cost?"}
    response = client.post("/chat", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The fees are 5000 TND."
    assert "fees.pdf (Page 1)" in data["sources"]

def test_chat_service_not_ready(client, mocker): 
    """
    Test 3: Does the API gracefully handle errors if the AI isn't loaded?
    """
    mocker.patch.object(rag_service, 'is_ready', False)

    payload = {"query": "Hello"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 503
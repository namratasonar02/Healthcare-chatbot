from fastapi.testclient import TestClient
from app.main import app
from unittest.mock import patch

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@patch("app.rag.RAGService.ingest_documents")
def test_ingest_documents(mock_ingest):
    mock_ingest.return_value = {
        "message": "Documents ingested successfully",
        "documents": 8,
        "chunks": 42
    }
    
    response = client.post("/ingest")
    
    assert response.status_code == 200
    assert response.json() == {
        "message": "Documents ingested successfully",
        "documents": 8,
        "chunks": 42
    }
    mock_ingest.assert_called_once()

@patch("app.agent.Agent.process_query")
def test_ask_appointment(mock_process):
    mock_process.return_value = {
        "answer": "Cardiology:\n- Monday 10:00 AM",
        "sources": [],
        "confidence": "high"
    }
    
    response = client.post("/ask", json={"question": "appointment"})
    
    assert response.status_code == 200
    data = response.json()
    assert "Cardiology" in data["answer"]
    assert data["confidence"] == "high"
    mock_process.assert_called_once_with("appointment")

@patch("app.agent.Agent.process_query")
def test_ask_rag_query(mock_process):
    mock_process.return_value = {
        "answer": "Telehealth policy info...",
        "sources": [
            {
                "document": "telehealth_policy.txt",
                "chunk": "info..."
            }
        ],
        "confidence": "high"
    }
    
    response = client.post("/ask", json={"question": "telehealth"})
    
    assert response.status_code == 200
    data = response.json()
    assert "telehealth_policy.txt" in data["sources"][0]["document"]
    mock_process.assert_called_once_with("telehealth")

def test_api_info():
    response = client.get("/api/info")
    assert response.status_code == 200
    data = response.json()
    assert "llm_source" in data
    assert "embedding_source" in data
    assert "has_gemini_key" in data
    assert "ollama_model" in data


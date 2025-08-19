"""Test FastAPI endpoints."""
from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_predict_endpoint():
    """Verify risk score format."""
    response = client.post("/predict", json={"user_id": "test", "transaction_history": []})
    assert 0 <= response.json()["risk_score"] <= 1

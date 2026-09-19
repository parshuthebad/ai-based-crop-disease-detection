"""Basic API tests (model-independent)."""
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rejects_non_image():
    response = client.post(
        "/api/predict", files={"file": ("note.txt", b"hello", "text/plain")}
    )
    assert response.status_code == 415

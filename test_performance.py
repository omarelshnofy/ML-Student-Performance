from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Student Performance API is running"


def test_model_info():
    response = client.get("/model_info")
    assert response.status_code == 200
    assert response.json()["model"] == "RandomForestRegressor"
    assert response.json()["target"] == "G3"


def test_performance_level():
    response = client.get("/performance_level?grade=14")
    assert response.status_code == 200
    assert response.json()["performance_level"] == "Good"
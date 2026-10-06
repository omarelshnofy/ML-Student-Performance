from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_predict():
    payload = {
        "school": "GP",
        "sex": "F",
        "age": 17,
        "address": "U",
        "famsize": "GT3",
        "Pstatus": "A",
        "Medu": 4,
        "Fedu": 4,
        "Mjob": "at_home",
        "Fjob": "teacher",
        "reason": "course",
        "guardian": "mother",
        "traveltime": 2,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "yes",
        "famsup": "no",
        "paid": "no",
        "activities": "no",
        "nursery": "yes",
        "higher": "yes",
        "internet": "yes",
        "romantic": "no",
        "famrel": 4,
        "freetime": 3,
        "goout": 3,
        "Dalc": 1,
        "Walc": 1,
        "health": 5,
        "absences": 4
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "predicted_grade" in response.json()
    assert "target" in response.json()
    assert "model_version" in response.json()
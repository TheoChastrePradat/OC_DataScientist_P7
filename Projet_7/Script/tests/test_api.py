import os

os.environ["SKIP_MODEL_LOAD"] = "1"

from fastapi.testclient import TestClient
from app import app, EXPECTED_FEATURES

client = TestClient(app)

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_metadata_shape():
    r = client.get("/metadata")
    assert r.status_code == 200
    m = r.json()
    assert m["n_features"] == len(m["expected_features"]) > 0


def test_predict_mocked():
    payload = {"features": {f: 0.1 for f in EXPECTED_FEATURES}}
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "probability" in data and "decision" in data
"""The prediction service, tested with FastAPI's TestClient and a tiny model."""

import pytest
from fastapi.testclient import TestClient

import app.main as service
from mp290.model import build_pipeline
from conftest import NUMERIC, CATEGORICAL

GOOD = {"hour": 18, "nearby_offices": 60, "temp_c": 30.5, "rain_mm": 2.0,
        "feeder_buses": 20, "ticket_price_rs": 35, "day_type": "weekday", "line": "Red Line"}


@pytest.fixture
def client(small_train):
    model = build_pipeline({"model": "ridge", "alpha": 1.0}, NUMERIC, CATEGORICAL)
    model.fit(small_train[NUMERIC + CATEGORICAL], small_train["entries_per_hour"])
    service.model = model                      # no model file needed
    yield TestClient(service.app)
    service.model = None


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_stats(client):
    r = client.get("/stats")
    assert r.status_code == 200
    assert "v2_check_mae" in r.json()


def test_predict_returns_a_number(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    assert isinstance(r.json()["entries_per_hour"], float)


@pytest.mark.parametrize("field,value", [
    ("nearby_offices", 151),          # client rule 1
    ("ticket_price_rs", 4.99),        # client rule 2
    ("line", "Green Line"),
    ("hour", 24),
])
def test_bad_input_gets_422(client, field, value):
    assert client.post("/predict", json={**GOOD, field: value}).status_code == 422


def test_boundary_values_are_allowed(client):
    assert client.post("/predict", json={**GOOD, "nearby_offices": 150, "ticket_price_rs": 5}).status_code == 200


def test_missing_or_extra_field_gets_422(client):
    missing = {k: v for k, v in GOOD.items() if k != "line"}
    assert client.post("/predict", json=missing).status_code == 422
    assert client.post("/predict", json={**GOOD, "gate_count_end_of_hour": 2000}).status_code == 422


def test_one_json_log_line_per_prediction(client, capsys):
    import json
    client.post("/predict", json=GOOD)
    lines = [json.loads(l) for l in capsys.readouterr().out.splitlines() if l.startswith("{")]
    events = [l["event"] for l in lines]
    assert events.count("prediction") == 1
    pred = next(l for l in lines if l["event"] == "prediction")
    assert pred["line"] == "Red Line" and "prediction" in pred

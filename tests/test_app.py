"""Tests for the churn prediction API."""

import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_churn_likely(client):
    """Month-to-month, short tenure, high charges — likely to churn."""
    resp = client.post(
        "/predict",
        json={
            "gender": "Female",
            "tenure": 1,
            "MonthlyCharges": 95.0,
            "TotalCharges": 95.0,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == "Yes"
    assert body["churn_probability"] > 0.5


def test_predict_churn_unlikely(client):
    """Two-year contract, long tenure — unlikely to churn."""
    resp = client.post(
        "/predict",
        json={
            "gender": "Male",
            "tenure": 60,
            "MonthlyCharges": 20.0,
            "TotalCharges": 1200.0,
            "Contract": "Two year",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "TechSupport": "Yes",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] == "No"
    assert body["churn_probability"] < 0.5


def test_predict_missing_required_field(client):
    """Omitting a required field should return 422."""
    resp = client.post(
        "/predict",
        json={"gender": "Male"},  # missing tenure, MonthlyCharges, Contract
    )
    assert resp.status_code == 422


def test_predict_invalid_tenure(client):
    """Negative tenure should be rejected by validation."""
    resp = client.post(
        "/predict",
        json={
            "gender": "Male",
            "tenure": -5,
            "MonthlyCharges": 50.0,
            "Contract": "One year",
        },
    )
    assert resp.status_code == 422


def test_predict_inconsistent_internet_service(client):
    """OnlineSecurity='Yes' with InternetService='No' should be rejected."""
    resp = client.post(
        "/predict",
        json={
            "gender": "Male",
            "tenure": 12,
            "MonthlyCharges": 50.0,
            "Contract": "One year",
            "InternetService": "No",
            "OnlineSecurity": "Yes",
        },
    )
    assert resp.status_code == 422


def test_predict_inconsistent_phone_service(client):
    """MultipleLines='Yes' with PhoneService='No' should be rejected."""
    resp = client.post(
        "/predict",
        json={
            "gender": "Female",
            "tenure": 24,
            "MonthlyCharges": 30.0,
            "Contract": "Two year",
            "PhoneService": "No",
            "MultipleLines": "Yes",
        },
    )
    assert resp.status_code == 422

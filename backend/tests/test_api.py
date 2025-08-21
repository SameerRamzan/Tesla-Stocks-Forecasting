"""Test API endpoints."""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_root_endpoint():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Tesla Forecasting API" in data["message"]


def test_api_health():
    """Test API health endpoint."""
    response = client.get("/api/v1/health/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_openapi_docs():
    """Test that OpenAPI docs are accessible."""
    response = client.get("/api/v1/docs")
    assert response.status_code == 200


def test_invalid_endpoint():
    """Test invalid endpoint returns 404."""
    response = client.get("/api/v1/nonexistent")
    assert response.status_code == 404
"""
Tests for the FastAPI application.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "1.0.0"

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "endpoints" in data


class TestAnalysisEndpoints:
    """Tests for the analysis API endpoints."""

    def test_detect_anomalies(self, client):
        """Test the anomaly detection endpoint."""
        payload = {
            "telemetry_data": [
                {
                    "uav_id": "UAV-001",
                    "timestamp": "2025-06-15T10:00:00",
                    "latitude": 39.9042,
                    "longitude": 116.4074,
                    "altitude": 100.0,
                    "speed": 10.0,
                    "battery_level": 85.0,
                    "temperature": 35.0,
                    "vibration": 1.5,
                    "signal_strength": -50.0,
                    "motor_rpm": [5000, 5100, 4900, 5050],
                },
                {
                    "uav_id": "UAV-002",
                    "timestamp": "2025-06-15T10:00:00",
                    "latitude": 39.9052,
                    "longitude": 116.4084,
                    "altitude": 120.0,
                    "speed": 12.0,
                    "battery_level": 5.0,  # Low battery - anomaly
                    "temperature": 95.0,   # High temp - anomaly
                    "vibration": 1.8,
                    "signal_strength": -55.0,
                    "motor_rpm": [5000, 5000, 5000, 2000],  # Motor imbalance
                },
            ],
            "user_query": "test detection",
        }

        response = client.post("/api/v1/analysis/detect", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["anomaly_results"]["anomalies_found"] > 0

    def test_full_analysis(self, client):
        """Test the full analysis pipeline endpoint."""
        payload = {
            "telemetry_data": [
                {
                    "uav_id": "UAV-001",
                    "timestamp": "2025-06-15T10:00:00",
                    "latitude": 39.9042,
                    "longitude": 116.4074,
                    "altitude": 100.0,
                    "speed": 10.0,
                    "battery_level": 85.0,
                    "temperature": 35.0,
                    "vibration": 1.5,
                    "signal_strength": -50.0,
                },
            ],
        }

        response = client.post("/api/v1/analysis/full", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"

    def test_detect_empty_payload_fails(self, client):
        """Test that empty payload returns validation error."""
        response = client.post("/api/v1/analysis/detect", json={})
        assert response.status_code == 422  # Validation error


class TestRAGEndpoints:
    """Tests for the RAG API endpoints."""

    def test_rag_status(self, client):
        """Test RAG status endpoint."""
        response = client.get("/api/v1/rag/status")
        assert response.status_code == 200
        data = response.json()
        assert "initialized" in data

    def test_rag_query_without_api_key(self, client):
        """Test RAG query returns 503 when not initialized."""
        response = client.post(
            "/api/v1/rag/query",
            json={"question": "test question"},
        )
        # Should be 503 since API key is not configured in test
        assert response.status_code == 503

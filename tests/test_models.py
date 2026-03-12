"""
Tests for data processing and anomaly detection models.
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.data_processor import DataProcessor, UAVTelemetry
from src.models.anomaly_detector import (
    AnomalyDetector, AnomalyLevel, AnomalyType, AnomalyResult,
)


# ========== Data Processor Tests ==========

class TestDataProcessor:
    """Tests for the DataProcessor class."""

    def setup_method(self):
        self.processor = DataProcessor()
        self.sample_telemetry = {
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
            "heading": 180.0,
            "vertical_speed": 0.5,
        }

    def test_process_single_telemetry(self):
        """Test processing a single telemetry record."""
        result = self.processor.process_telemetry(self.sample_telemetry)

        assert "derived" in result
        assert "data_quality" in result
        assert result["uav_id"] == "UAV-001"

    def test_derived_features(self):
        """Test that derived features are computed correctly."""
        # Process twice to have history for derivative features
        self.processor.process_telemetry(self.sample_telemetry)
        telemetry2 = dict(self.sample_telemetry)
        telemetry2["battery_level"] = 83.0
        telemetry2["speed"] = 15.0

        result = self.processor.process_telemetry(telemetry2)
        derived = result["derived"]

        assert "battery_drain_rate" in derived
        assert "speed_delta" in derived
        assert "motor_rpm_variance" in derived
        assert "altitude_stability" in derived
        assert derived["battery_drain_rate"] == pytest.approx(2.0, abs=0.01)
        assert derived["speed_delta"] == pytest.approx(5.0, abs=0.01)

    def test_data_quality_assessment(self):
        """Test data quality scoring."""
        result = self.processor.process_telemetry(self.sample_telemetry)
        quality = result["data_quality"]

        assert quality["score"] == 1.0  # All fields present
        assert quality["missing_fields"] == []
        assert quality["issues"] == []

    def test_data_quality_with_missing_fields(self):
        """Test data quality with missing fields."""
        incomplete = {"uav_id": "UAV-001", "timestamp": "2025-06-15T10:00:00"}
        result = self.processor.process_telemetry(incomplete)
        quality = result["data_quality"]

        assert quality["score"] < 1.0
        assert len(quality["missing_fields"]) > 0

    def test_batch_processing(self):
        """Test batch processing of telemetry records."""
        batch = [self.sample_telemetry] * 5
        df = self.processor.process_batch(batch)

        assert len(df) == 5
        assert "data_quality_score" in df.columns

    def test_fleet_statistics(self):
        """Test fleet statistics computation."""
        telemetry_list = []
        for i in range(3):
            t = dict(self.sample_telemetry)
            t["uav_id"] = f"UAV-{i+1:03d}"
            t["altitude"] = 100.0 + i * 10
            telemetry_list.append(t)

        stats = self.processor.get_fleet_statistics(telemetry_list)

        assert stats["fleet_size"] == 3
        assert stats["total_records"] == 3
        assert "altitude" in stats
        assert stats["altitude"]["mean"] == pytest.approx(110.0, abs=0.1)

    def test_uav_telemetry_dataclass(self):
        """Test UAVTelemetry dataclass."""
        telemetry = UAVTelemetry(
            uav_id="UAV-001",
            timestamp="2025-06-15T10:00:00",
            latitude=39.9,
            longitude=116.4,
            altitude=100.0,
            speed=10.0,
            battery_level=85.0,
            temperature=35.0,
            vibration=1.5,
            signal_strength=-50.0,
        )
        d = telemetry.to_dict()
        assert d["uav_id"] == "UAV-001"
        assert d["altitude"] == 100.0


# ========== Anomaly Detector Tests ==========

class TestAnomalyDetector:
    """Tests for the AnomalyDetector class."""

    def setup_method(self):
        self.detector = AnomalyDetector()
        self.normal_telemetry = {
            "uav_id": "UAV-001",
            "temperature": 35.0,
            "vibration": 1.5,
            "battery_level": 85.0,
            "signal_strength": -50.0,
            "motor_rpm": [5000, 5100, 4900, 5050],
        }

    def test_normal_telemetry_no_anomaly(self):
        """Test that normal telemetry produces no anomaly."""
        result = self.detector.detect(self.normal_telemetry)

        assert result.is_anomaly is False
        assert result.level == AnomalyLevel.NORMAL
        assert len(result.anomaly_types) == 0
        assert result.confidence >= 0.9

    def test_high_temperature_anomaly(self):
        """Test temperature anomaly detection."""
        telemetry = dict(self.normal_telemetry)
        telemetry["temperature"] = 95.0  # Exceeds 80°C threshold

        result = self.detector.detect(telemetry)

        assert result.is_anomaly is True
        assert AnomalyType.TEMPERATURE in result.anomaly_types

    def test_high_vibration_anomaly(self):
        """Test vibration anomaly detection."""
        telemetry = dict(self.normal_telemetry)
        telemetry["vibration"] = 15.0  # Exceeds 10g threshold

        result = self.detector.detect(telemetry)

        assert result.is_anomaly is True
        assert AnomalyType.VIBRATION in result.anomaly_types

    def test_low_battery_anomaly(self):
        """Test low battery anomaly detection."""
        telemetry = dict(self.normal_telemetry)
        telemetry["battery_level"] = 5.0  # Below 10% threshold

        result = self.detector.detect(telemetry)

        assert result.is_anomaly is True
        assert AnomalyType.BATTERY in result.anomaly_types
        assert result.level == AnomalyLevel.CRITICAL  # Battery is critical

    def test_weak_signal_anomaly(self):
        """Test weak signal anomaly detection."""
        telemetry = dict(self.normal_telemetry)
        telemetry["signal_strength"] = -95.0  # Below -90dBm threshold

        result = self.detector.detect(telemetry)

        assert result.is_anomaly is True
        assert AnomalyType.SIGNAL in result.anomaly_types

    def test_motor_imbalance_detection(self):
        """Test motor RPM imbalance detection."""
        telemetry = dict(self.normal_telemetry)
        telemetry["motor_rpm"] = [5000, 5000, 5000, 2000]  # One motor very low

        result = self.detector.detect(telemetry)

        assert result.is_anomaly is True
        assert AnomalyType.MOTOR in result.anomaly_types

    def test_multiple_anomalies(self):
        """Test detection of multiple simultaneous anomalies."""
        telemetry = dict(self.normal_telemetry)
        telemetry["temperature"] = 95.0
        telemetry["battery_level"] = 5.0
        telemetry["vibration"] = 15.0

        result = self.detector.detect(telemetry)

        assert result.is_anomaly is True
        assert len(result.anomaly_types) >= 3
        assert result.level == AnomalyLevel.CRITICAL

    def test_trend_based_detection(self):
        """Test trend-based anomaly detection using derived features."""
        telemetry = dict(self.normal_telemetry)
        derived = {
            "battery_drain_rate": 8.0,  # Rapid drain
            "altitude_stability": 0.1,  # Very unstable
            "speed_delta": 25.0,  # Sudden speed change
        }

        result = self.detector.detect(telemetry, derived)

        assert result.is_anomaly is True
        assert AnomalyType.BATTERY in result.anomaly_types or AnomalyType.ALTITUDE in result.anomaly_types

    def test_batch_detection(self):
        """Test batch anomaly detection."""
        batch = [
            self.normal_telemetry,
            {**self.normal_telemetry, "temperature": 95.0},
            {**self.normal_telemetry, "battery_level": 5.0},
        ]

        results = self.detector.detect_batch(batch)

        assert len(results) == 3
        assert results[0].is_anomaly is False
        assert results[1].is_anomaly is True
        assert results[2].is_anomaly is True

    def test_recommendations_generated(self):
        """Test that recommendations are generated for anomalies."""
        telemetry = dict(self.normal_telemetry)
        telemetry["temperature"] = 95.0

        result = self.detector.detect(telemetry)

        assert len(result.recommendations) > 0
        assert any("thermal" in r.lower() or "temperature" in r.lower() or "cool" in r.lower()
                    for r in result.recommendations)

    def test_anomaly_result_to_dict(self):
        """Test AnomalyResult serialization."""
        result = self.detector.detect(self.normal_telemetry)
        d = result.to_dict()

        assert "uav_id" in d
        assert "is_anomaly" in d
        assert "level" in d
        assert "confidence" in d
        assert isinstance(d["level"], str)

    def test_isolation_forest_training(self):
        """Test Isolation Forest model training and prediction."""
        # Generate training data
        np.random.seed(42)
        normal_data = np.random.randn(100, 4) * 2 + 5

        self.detector.train_isolation_forest(normal_data)

        # Test with normal data
        normal_test = np.array([[5.0, 5.0, 5.0, 5.0]])
        predictions = self.detector.predict_isolation_forest(normal_test)
        assert len(predictions) == 1

        # Test with extreme outlier
        outlier = np.array([[100.0, 100.0, 100.0, 100.0]])
        predictions = self.detector.predict_isolation_forest(outlier)
        assert len(predictions) == 1

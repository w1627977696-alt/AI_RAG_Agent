"""
Anomaly Detector - Small Model Module
Implements rule-based and statistical anomaly detection for UAV telemetry.
Acts as the "small model" for fast, local anomaly detection.
"""
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from sklearn.ensemble import IsolationForest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config.settings import ANOMALY_THRESHOLDS


class AnomalyLevel(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class AnomalyType(str, Enum):
    TEMPERATURE = "temperature_anomaly"
    VIBRATION = "vibration_anomaly"
    BATTERY = "battery_anomaly"
    SIGNAL = "signal_anomaly"
    MOTOR = "motor_imbalance"
    ALTITUDE = "altitude_anomaly"
    SPEED = "speed_anomaly"
    STATISTICAL = "statistical_anomaly"


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single UAV."""
    uav_id: str
    is_anomaly: bool
    level: AnomalyLevel
    anomaly_types: list[AnomalyType]
    details: dict
    confidence: float
    recommendations: list[str]

    def to_dict(self) -> dict:
        return {
            "uav_id": self.uav_id,
            "is_anomaly": self.is_anomaly,
            "level": self.level.value,
            "anomaly_types": [a.value for a in self.anomaly_types],
            "details": self.details,
            "confidence": self.confidence,
            "recommendations": self.recommendations,
        }


class AnomalyDetector:
    """
    Multi-strategy anomaly detector for UAV telemetry.
    Combines rule-based thresholds, statistical analysis, and Isolation Forest.
    """

    def __init__(self, thresholds: Optional[dict] = None):
        self.thresholds = thresholds or ANOMALY_THRESHOLDS
        self._isolation_forest: Optional[IsolationForest] = None
        self._history: dict[str, list[dict]] = {}

    def detect(self, telemetry: dict, derived: Optional[dict] = None) -> AnomalyResult:
        """
        Run all anomaly detection strategies on a single telemetry record.
        Returns an AnomalyResult with combined findings.
        """
        uav_id = telemetry.get("uav_id", "unknown")
        anomalies = []
        details = {}
        recommendations = []

        # Strategy 1: Rule-based threshold detection
        rule_anomalies = self._rule_based_detection(telemetry)
        anomalies.extend(rule_anomalies)
        if rule_anomalies:
            details["rule_based"] = [a.value for a in rule_anomalies]

        # Strategy 2: Motor imbalance detection
        motor_anomaly = self._motor_imbalance_detection(telemetry)
        if motor_anomaly:
            anomalies.append(motor_anomaly)
            details["motor_imbalance"] = True

        # Strategy 3: Trend-based detection using derived features
        if derived:
            trend_anomalies = self._trend_based_detection(derived)
            anomalies.extend(trend_anomalies)
            if trend_anomalies:
                details["trend_based"] = [a.value for a in trend_anomalies]

        # Determine severity level
        level = self._determine_level(anomalies, telemetry)

        # Generate recommendations
        recommendations = self._generate_recommendations(anomalies, telemetry)

        # Compute confidence
        confidence = self._compute_confidence(anomalies, telemetry)

        return AnomalyResult(
            uav_id=uav_id,
            is_anomaly=len(anomalies) > 0,
            level=level,
            anomaly_types=anomalies,
            details=details,
            confidence=confidence,
            recommendations=recommendations,
        )

    def detect_batch(self, telemetry_list: list[dict]) -> list[AnomalyResult]:
        """Detect anomalies for a batch of telemetry records."""
        results = []
        for t in telemetry_list:
            derived = t.get("derived", None)
            results.append(self.detect(t, derived))
        return results

    def train_isolation_forest(self, feature_matrix: np.ndarray) -> None:
        """Train the Isolation Forest model on historical normal data."""
        self._isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
        )
        self._isolation_forest.fit(feature_matrix)

    def predict_isolation_forest(self, features: np.ndarray) -> list[bool]:
        """Use trained Isolation Forest to predict anomalies."""
        if self._isolation_forest is None:
            return [False] * len(features)
        predictions = self._isolation_forest.predict(features)
        return [p == -1 for p in predictions]

    def _rule_based_detection(self, telemetry: dict) -> list[AnomalyType]:
        """Apply threshold-based rules for anomaly detection."""
        anomalies = []

        temp = telemetry.get("temperature")
        if temp is not None:
            if temp < self.thresholds["temperature_min"] or temp > self.thresholds["temperature_max"]:
                anomalies.append(AnomalyType.TEMPERATURE)

        vibration = telemetry.get("vibration")
        if vibration is not None:
            if vibration > self.thresholds["vibration_max"]:
                anomalies.append(AnomalyType.VIBRATION)

        battery = telemetry.get("battery_level")
        if battery is not None:
            if battery < self.thresholds["battery_min"]:
                anomalies.append(AnomalyType.BATTERY)

        signal = telemetry.get("signal_strength")
        if signal is not None:
            if signal < self.thresholds["signal_min"]:
                anomalies.append(AnomalyType.SIGNAL)

        return anomalies

    def _motor_imbalance_detection(self, telemetry: dict) -> Optional[AnomalyType]:
        """Detect motor RPM imbalance across rotors."""
        motor_rpm = telemetry.get("motor_rpm", [])
        if not motor_rpm or len(motor_rpm) < 2:
            return None

        rpm_array = np.array(motor_rpm, dtype=float)
        mean_rpm = np.mean(rpm_array)
        if mean_rpm == 0:
            return None

        cv = np.std(rpm_array) / mean_rpm  # Coefficient of variation
        if cv > 0.15:  # More than 15% variation indicates imbalance
            return AnomalyType.MOTOR

        return None

    def _trend_based_detection(self, derived: dict) -> list[AnomalyType]:
        """Detect anomalies based on derived trend features."""
        anomalies = []

        battery_drain = derived.get("battery_drain_rate", 0)
        if battery_drain > 5.0:  # Rapid battery drain
            anomalies.append(AnomalyType.BATTERY)

        altitude_stability = derived.get("altitude_stability", 1.0)
        if altitude_stability < 0.3:  # Unstable altitude
            anomalies.append(AnomalyType.ALTITUDE)

        speed_delta = derived.get("speed_delta", 0)
        if abs(speed_delta) > 20.0:  # Sudden speed change
            anomalies.append(AnomalyType.SPEED)

        return anomalies

    def _determine_level(self, anomalies: list[AnomalyType], telemetry: dict) -> AnomalyLevel:
        """Determine the overall anomaly severity level."""
        if not anomalies:
            return AnomalyLevel.NORMAL

        critical_types = {AnomalyType.BATTERY, AnomalyType.MOTOR, AnomalyType.ALTITUDE}
        has_critical = any(a in critical_types for a in anomalies)

        battery = telemetry.get("battery_level", 100)
        if has_critical or len(anomalies) >= 3 or (battery is not None and battery < 5):
            return AnomalyLevel.CRITICAL

        return AnomalyLevel.WARNING

    def _compute_confidence(self, anomalies: list[AnomalyType], telemetry: dict) -> float:
        """Compute detection confidence score (0-1)."""
        if not anomalies:
            return 0.95  # High confidence in normal state

        # More anomaly signals = higher confidence in anomaly detection
        base = 0.7
        bonus = min(0.25, len(anomalies) * 0.08)
        return round(base + bonus, 2)

    def _generate_recommendations(self, anomalies: list[AnomalyType], telemetry: dict) -> list[str]:
        """Generate actionable recommendations based on detected anomalies."""
        recommendations = []

        recommendation_map = {
            AnomalyType.TEMPERATURE: "Monitor thermal management system; consider altitude adjustment for cooling.",
            AnomalyType.VIBRATION: "Inspect propellers and motor mounts; check for debris or damage.",
            AnomalyType.BATTERY: "Initiate return-to-home procedure; prioritize landing sequence.",
            AnomalyType.SIGNAL: "Move to area with better signal coverage; check antenna integrity.",
            AnomalyType.MOTOR: "Reduce payload and speed; schedule motor maintenance.",
            AnomalyType.ALTITUDE: "Stabilize altitude control; check barometer and GPS sensors.",
            AnomalyType.SPEED: "Verify wind conditions; check flight controller calibration.",
        }

        for anomaly in anomalies:
            if anomaly in recommendation_map:
                recommendations.append(recommendation_map[anomaly])

        if not recommendations:
            recommendations.append("Continue normal operations monitoring.")

        return recommendations

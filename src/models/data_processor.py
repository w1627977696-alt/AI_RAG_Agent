"""
Data Processor - Small Model Module
Handles UAV telemetry data preprocessing, feature engineering, and normalization.
Acts as the "small model" in the large-small model collaboration architecture.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class UAVTelemetry:
    """Single UAV telemetry data point."""
    uav_id: str
    timestamp: str
    latitude: float
    longitude: float
    altitude: float
    speed: float
    battery_level: float
    temperature: float
    vibration: float
    signal_strength: float
    motor_rpm: list = field(default_factory=lambda: [0.0] * 4)
    heading: float = 0.0
    vertical_speed: float = 0.0

    def to_dict(self) -> dict:
        return {
            "uav_id": self.uav_id,
            "timestamp": self.timestamp,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,
            "speed": self.speed,
            "battery_level": self.battery_level,
            "temperature": self.temperature,
            "vibration": self.vibration,
            "signal_strength": self.signal_strength,
            "motor_rpm": self.motor_rpm,
            "heading": self.heading,
            "vertical_speed": self.vertical_speed,
        }


class DataProcessor:
    """
    UAV telemetry data processor.
    Performs data cleaning, feature engineering, and statistical analysis.
    This serves as the 'small model' for efficient local data processing.
    """

    def __init__(self):
        self._history: dict[str, list[dict]] = {}

    def process_telemetry(self, telemetry: dict) -> dict:
        """
        Process raw telemetry data: clean, validate, and compute derived features.
        Returns enriched telemetry with derived metrics.
        """
        uav_id = telemetry.get("uav_id", "unknown")

        # Store history for trend analysis
        if uav_id not in self._history:
            self._history[uav_id] = []
        self._history[uav_id].append(telemetry)

        # Keep last 100 records for memory efficiency
        if len(self._history[uav_id]) > 100:
            self._history[uav_id] = self._history[uav_id][-100:]

        # Compute derived features
        processed = dict(telemetry)
        processed["derived"] = self._compute_derived_features(uav_id, telemetry)
        processed["data_quality"] = self._assess_data_quality(telemetry)

        return processed

    def process_batch(self, telemetry_list: list[dict]) -> pd.DataFrame:
        """Process a batch of telemetry records and return a DataFrame."""
        processed = [self.process_telemetry(t) for t in telemetry_list]

        # Flatten derived features into the dataframe
        rows = []
        for p in processed:
            row = {k: v for k, v in p.items() if k not in ("derived", "data_quality", "motor_rpm")}
            if "derived" in p:
                for dk, dv in p["derived"].items():
                    row[f"derived_{dk}"] = dv
            row["data_quality_score"] = p.get("data_quality", {}).get("score", 0)
            rows.append(row)

        return pd.DataFrame(rows)

    def get_fleet_statistics(self, telemetry_list: list[dict]) -> dict:
        """Compute fleet-wide statistics from telemetry data."""
        if not telemetry_list:
            return {"error": "No telemetry data provided"}

        df = pd.DataFrame(telemetry_list)
        numeric_cols = ["altitude", "speed", "battery_level", "temperature",
                        "vibration", "signal_strength"]
        available_cols = [c for c in numeric_cols if c in df.columns]

        stats = {}
        for col in available_cols:
            series = pd.to_numeric(df[col], errors="coerce")
            stats[col] = {
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2),
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
            }

        stats["fleet_size"] = int(df["uav_id"].nunique()) if "uav_id" in df.columns else 0
        stats["total_records"] = len(df)

        return stats

    def _compute_derived_features(self, uav_id: str, telemetry: dict) -> dict:
        """Compute derived features from raw telemetry and history."""
        derived = {}
        history = self._history.get(uav_id, [])

        # Battery drain rate (per minute)
        if len(history) >= 2:
            prev = history[-2]
            curr_battery = telemetry.get("battery_level", 0)
            prev_battery = prev.get("battery_level", 0)
            derived["battery_drain_rate"] = round(prev_battery - curr_battery, 4)
        else:
            derived["battery_drain_rate"] = 0.0

        # Speed change rate
        if len(history) >= 2:
            prev = history[-2]
            derived["speed_delta"] = round(
                telemetry.get("speed", 0) - prev.get("speed", 0), 4
            )
        else:
            derived["speed_delta"] = 0.0

        # Motor RPM variance (indicates mechanical imbalance)
        motor_rpm = telemetry.get("motor_rpm", [])
        if motor_rpm and len(motor_rpm) > 1:
            derived["motor_rpm_variance"] = round(float(np.var(motor_rpm)), 4)
            derived["motor_rpm_mean"] = round(float(np.mean(motor_rpm)), 2)
        else:
            derived["motor_rpm_variance"] = 0.0
            derived["motor_rpm_mean"] = 0.0

        # Distance from fleet center (if we have history of multiple UAVs)
        derived["altitude_stability"] = self._compute_altitude_stability(uav_id)

        return derived

    def _compute_altitude_stability(self, uav_id: str) -> float:
        """Compute altitude stability from recent history."""
        history = self._history.get(uav_id, [])
        if len(history) < 3:
            return 1.0

        altitudes = [h.get("altitude", 0) for h in history[-10:]]
        std = float(np.std(altitudes))
        # Normalize: lower std means more stable
        stability = max(0.0, 1.0 - std / 50.0)
        return round(stability, 4)

    def _assess_data_quality(self, telemetry: dict) -> dict:
        """Assess the quality of incoming telemetry data."""
        required_fields = ["uav_id", "timestamp", "latitude", "longitude",
                           "altitude", "speed", "battery_level", "temperature",
                           "vibration", "signal_strength"]
        present = sum(1 for f in required_fields if f in telemetry and telemetry[f] is not None)
        total = len(required_fields)

        issues = []
        if telemetry.get("battery_level", 100) < 0 or telemetry.get("battery_level", 0) > 100:
            issues.append("battery_level_out_of_range")
        if telemetry.get("altitude", 0) < 0:
            issues.append("negative_altitude")

        return {
            "score": round(present / total, 2),
            "missing_fields": [f for f in required_fields if f not in telemetry or telemetry[f] is None],
            "issues": issues,
        }

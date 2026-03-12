"""
Sample Data Generator
Generates realistic UAV swarm telemetry data for testing and demonstration.
"""
import json
import random
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def generate_uav_telemetry(
    num_uavs: int = 10,
    num_records_per_uav: int = 50,
    anomaly_ratio: float = 0.1,
    seed: int = 42,
) -> list[dict]:
    """
    Generate realistic UAV swarm telemetry data.

    Args:
        num_uavs: Number of UAVs in the fleet.
        num_records_per_uav: Number of telemetry records per UAV.
        anomaly_ratio: Fraction of records that should contain anomalies.
        seed: Random seed for reproducibility.

    Returns:
        List of telemetry dictionaries.
    """
    rng = np.random.RandomState(seed)
    random.seed(seed)
    records = []

    base_lat, base_lon = 39.9042, 116.4074  # Beijing coordinates
    start_time = datetime(2025, 6, 15, 10, 0, 0)

    for uav_idx in range(num_uavs):
        uav_id = f"UAV-{uav_idx + 1:03d}"
        battery = 100.0
        lat = base_lat + rng.uniform(-0.01, 0.01)
        lon = base_lon + rng.uniform(-0.01, 0.01)
        altitude = rng.uniform(50, 200)
        heading = rng.uniform(0, 360)

        for t_idx in range(num_records_per_uav):
            timestamp = (start_time + timedelta(seconds=t_idx * 2)).isoformat()
            is_anomaly = rng.random() < anomaly_ratio

            # Normal telemetry generation
            lat += rng.uniform(-0.0001, 0.0001)
            lon += rng.uniform(-0.0001, 0.0001)
            altitude += rng.uniform(-2, 2)
            altitude = max(10, min(500, altitude))
            speed = rng.uniform(5, 25)
            battery -= rng.uniform(0.05, 0.2)
            battery = max(0, battery)
            temperature = rng.uniform(20, 45)
            vibration = rng.uniform(0.5, 3.0)
            signal_strength = rng.uniform(-70, -30)
            heading = (heading + rng.uniform(-5, 5)) % 360
            vertical_speed = rng.uniform(-2, 2)
            motor_rpm = [rng.uniform(4500, 5500) for _ in range(4)]

            # Inject anomalies
            if is_anomaly:
                anomaly_type = random.choice([
                    "temperature", "vibration", "battery",
                    "signal", "motor", "altitude"
                ])
                if anomaly_type == "temperature":
                    temperature = rng.uniform(85, 120)
                elif anomaly_type == "vibration":
                    vibration = rng.uniform(12, 25)
                elif anomaly_type == "battery":
                    battery = rng.uniform(1, 8)
                elif anomaly_type == "signal":
                    signal_strength = rng.uniform(-100, -92)
                elif anomaly_type == "motor":
                    motor_rpm[random.randint(0, 3)] *= rng.uniform(0.3, 0.6)
                elif anomaly_type == "altitude":
                    altitude = rng.uniform(0, 5)

            record = {
                "uav_id": uav_id,
                "timestamp": timestamp,
                "latitude": round(lat, 6),
                "longitude": round(lon, 6),
                "altitude": round(altitude, 2),
                "speed": round(speed, 2),
                "battery_level": round(battery, 2),
                "temperature": round(temperature, 2),
                "vibration": round(vibration, 4),
                "signal_strength": round(signal_strength, 2),
                "motor_rpm": [round(r, 1) for r in motor_rpm],
                "heading": round(heading, 2),
                "vertical_speed": round(vertical_speed, 2),
                "_is_anomaly_injected": is_anomaly,
            }
            records.append(record)

    return records


def save_sample_data(output_dir: str = "data/sample"):
    """Generate and save sample data to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate normal fleet data
    records = generate_uav_telemetry(
        num_uavs=10,
        num_records_per_uav=50,
        anomaly_ratio=0.1,
    )

    with open(output_path / "fleet_telemetry.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Generate a small real-time simulation batch
    batch = generate_uav_telemetry(
        num_uavs=5,
        num_records_per_uav=10,
        anomaly_ratio=0.15,
        seed=123,
    )

    with open(output_path / "realtime_batch.json", "w", encoding="utf-8") as f:
        json.dump(batch, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(records)} fleet records -> {output_path / 'fleet_telemetry.json'}")
    print(f"Generated {len(batch)} batch records -> {output_path / 'realtime_batch.json'}")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    save_sample_data(str(project_root / "data" / "sample"))

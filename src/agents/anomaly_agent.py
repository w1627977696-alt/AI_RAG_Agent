"""
Anomaly Detection Agent
Wraps the AnomalyDetector as a LangChain-compatible agent tool.
Responsible for detecting anomalies in UAV telemetry data.
"""
from typing import Any

from langchain_core.tools import tool

from src.models.anomaly_detector import AnomalyDetector, AnomalyResult

# Shared detector instance
_detector = AnomalyDetector()


@tool
def detect_anomalies(telemetry_json: str) -> str:
    """
    Detect anomalies in a single UAV telemetry record.
    Input: JSON string of a processed telemetry record (with optional 'derived' field).
    Output: Anomaly detection result as JSON string.
    """
    import json
    try:
        telemetry = json.loads(telemetry_json)
        derived = telemetry.get("derived", None)
        result = _detector.detect(telemetry, derived)
        return json.dumps(result.to_dict(), ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def detect_anomalies_batch(telemetry_list_json: str) -> str:
    """
    Detect anomalies in a batch of UAV telemetry records.
    Input: JSON string of a list of processed telemetry records.
    Output: Summary of anomaly detection results as JSON string.
    """
    import json
    try:
        telemetry_list = json.loads(telemetry_list_json)
        results = _detector.detect_batch(telemetry_list)

        anomalous = [r for r in results if r.is_anomaly]
        summary = {
            "total_checked": len(results),
            "anomalies_found": len(anomalous),
            "anomaly_rate": round(len(anomalous) / max(len(results), 1), 4),
            "critical_count": sum(1 for r in anomalous if r.level.value == "critical"),
            "warning_count": sum(1 for r in anomalous if r.level.value == "warning"),
            "anomaly_details": [r.to_dict() for r in anomalous[:10]],
        }
        return json.dumps(summary, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_anomaly_agent_tools():
    """Return the list of tools for the anomaly detection agent."""
    return [detect_anomalies, detect_anomalies_batch]


def run_anomaly_detection(processed_records: list[dict]) -> dict:
    """
    Direct function call for anomaly detection (used by orchestrator).
    Returns anomaly detection results for all records.
    """
    results = []
    for record in processed_records:
        derived = record.get("derived", None)
        result = _detector.detect(record, derived)
        results.append(result)

    anomalous = [r for r in results if r.is_anomaly]

    return {
        "total_checked": len(results),
        "anomalies_found": len(anomalous),
        "anomaly_rate": round(len(anomalous) / max(len(results), 1), 4),
        "critical_count": sum(1 for r in anomalous if r.level.value == "critical"),
        "warning_count": sum(1 for r in anomalous if r.level.value == "warning"),
        "anomaly_details": [r.to_dict() for r in anomalous],
        "all_results": [r.to_dict() for r in results],
    }

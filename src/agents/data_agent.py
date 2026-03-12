"""
Data Processing Agent
Wraps the DataProcessor as a LangChain-compatible agent tool.
Handles data ingestion, cleaning, and feature engineering.
"""
from typing import Any

from langchain_core.tools import tool

from src.models.data_processor import DataProcessor

# Shared processor instance
_processor = DataProcessor()


@tool
def process_telemetry(telemetry_json: str) -> str:
    """
    Process raw UAV telemetry data.
    Input: JSON string of a single telemetry record.
    Output: Processed telemetry with derived features as JSON string.
    """
    import json
    try:
        telemetry = json.loads(telemetry_json)
        result = _processor.process_telemetry(telemetry)
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def process_telemetry_batch(telemetry_list_json: str) -> str:
    """
    Process a batch of UAV telemetry records.
    Input: JSON string of a list of telemetry records.
    Output: Fleet statistics and processed data summary as JSON string.
    """
    import json
    try:
        telemetry_list = json.loads(telemetry_list_json)
        df = _processor.process_batch(telemetry_list)
        stats = _processor.get_fleet_statistics(telemetry_list)

        return json.dumps({
            "records_processed": len(df),
            "fleet_statistics": stats,
            "sample_processed": df.head(3).to_dict(orient="records") if len(df) > 0 else [],
        }, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_fleet_statistics(telemetry_list_json: str) -> str:
    """
    Compute fleet-wide statistics from telemetry data.
    Input: JSON string of a list of telemetry records.
    Output: Statistical summary of the fleet as JSON string.
    """
    import json
    try:
        telemetry_list = json.loads(telemetry_list_json)
        stats = _processor.get_fleet_statistics(telemetry_list)
        return json.dumps(stats, ensure_ascii=False, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


def get_data_agent_tools():
    """Return the list of tools for the data processing agent."""
    return [process_telemetry, process_telemetry_batch, get_fleet_statistics]


def run_data_processing(telemetry_list: list[dict]) -> dict:
    """
    Direct function call for data processing (used by orchestrator).
    Returns processed data with derived features and fleet statistics.
    """
    processed_records = []
    for t in telemetry_list:
        processed = _processor.process_telemetry(t)
        processed_records.append(processed)

    stats = _processor.get_fleet_statistics(telemetry_list)

    return {
        "processed_records": processed_records,
        "fleet_statistics": stats,
        "total_processed": len(processed_records),
    }

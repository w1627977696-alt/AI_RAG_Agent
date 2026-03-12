"""
Analysis Routes
Provides endpoints for UAV telemetry analysis pipeline.
"""
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

from src.api.schemas import AnalysisRequest, AnalysisResponse, TelemetryRecord
from src.agents.orchestrator import run_pipeline
from src.agents.data_agent import run_data_processing
from src.agents.anomaly_agent import run_anomaly_detection

router = APIRouter(prefix="/api/v1/analysis", tags=["analysis"])


@router.post("/full", response_model=AnalysisResponse)
async def run_full_analysis(request: AnalysisRequest):
    """
    Run the full multi-agent analysis pipeline.
    Processes telemetry data through: Data Processing -> Anomaly Detection
    -> Impact Assessment -> Report Generation.
    """
    try:
        telemetry_dicts = [t.model_dump() for t in request.telemetry_data]
        result = run_pipeline(telemetry_dicts, request.user_query)
        return AnalysisResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect")
async def detect_anomalies(request: AnalysisRequest):
    """
    Run only data processing and anomaly detection (no LLM required).
    Faster and suitable for real-time monitoring.
    """
    try:
        telemetry_dicts = [t.model_dump() for t in request.telemetry_data]

        # Step 1: Data processing
        processed = run_data_processing(telemetry_dicts)

        # Step 2: Anomaly detection
        anomaly_results = run_anomaly_detection(processed["processed_records"])

        return {
            "status": "completed",
            "fleet_statistics": processed["fleet_statistics"],
            "anomaly_results": {
                k: v for k, v in anomaly_results.items() if k != "all_results"
            },
            "total_processed": processed["total_processed"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample")
async def get_sample_data():
    """Load and return sample telemetry data for testing."""
    sample_path = Path(__file__).parent.parent.parent.parent / "data" / "sample" / "realtime_batch.json"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample data not found. Run scripts/generate_sample_data.py first.")

    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"records": data, "count": len(data)}


@router.post("/sample/analyze")
async def analyze_sample_data():
    """Run analysis on the built-in sample data."""
    sample_path = Path(__file__).parent.parent.parent.parent / "data" / "sample" / "realtime_batch.json"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample data not found. Run scripts/generate_sample_data.py first.")

    with open(sample_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Remove internal fields
    for record in data:
        record.pop("_is_anomaly_injected", None)

    result = run_pipeline(data, "分析样本数据中的异常情况")
    return result

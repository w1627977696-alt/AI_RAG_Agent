"""
Pydantic Schemas for FastAPI
Defines request/response models for the API.
"""
from pydantic import BaseModel, Field
from typing import Optional


class TelemetryRecord(BaseModel):
    """Single UAV telemetry record."""
    uav_id: str = Field(..., description="Unique identifier of the UAV")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    latitude: float = Field(..., description="GPS latitude")
    longitude: float = Field(..., description="GPS longitude")
    altitude: float = Field(..., description="Altitude in meters")
    speed: float = Field(..., description="Speed in m/s")
    battery_level: float = Field(..., description="Battery level percentage (0-100)")
    temperature: float = Field(..., description="Temperature in Celsius")
    vibration: float = Field(..., description="Vibration level in g")
    signal_strength: float = Field(..., description="Signal strength in dBm")
    motor_rpm: list[float] = Field(default_factory=lambda: [0.0] * 4, description="Motor RPM values")
    heading: float = Field(default=0.0, description="Heading in degrees")
    vertical_speed: float = Field(default=0.0, description="Vertical speed in m/s")


class AnalysisRequest(BaseModel):
    """Request for full pipeline analysis."""
    telemetry_data: list[TelemetryRecord] = Field(..., description="List of telemetry records")
    user_query: str = Field(default="", description="Optional user query or context")


class AnalysisResponse(BaseModel):
    """Response from the analysis pipeline."""
    status: str
    report: dict
    anomaly_results: dict
    assessment_result: dict
    fleet_statistics: dict
    errors: list[str]
    timing: dict


class RAGQueryRequest(BaseModel):
    """Request for RAG-based Q&A."""
    question: str = Field(..., description="Question to answer using the knowledge base")
    k: int = Field(default=4, description="Number of documents to retrieve")


class RAGQueryResponse(BaseModel):
    """Response from RAG Q&A."""
    answer: str
    sources: list[dict]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    components: dict

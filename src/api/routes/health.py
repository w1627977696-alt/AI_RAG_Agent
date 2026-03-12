"""
Health Check Routes
"""
from fastapi import APIRouter
from src.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components={
            "data_processor": "available",
            "anomaly_detector": "available",
            "pipeline": "available",
            "rag": "requires_api_key",
        },
    )

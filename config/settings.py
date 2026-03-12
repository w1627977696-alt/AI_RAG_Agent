"""
UAV Swarm AI Operations Platform - Configuration
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")

# Vector Store
VECTOR_STORE_PATH = PROJECT_ROOT / os.getenv("VECTOR_STORE_PATH", "data/vector_store")
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Streamlit Configuration
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# Anomaly Detection Thresholds
ANOMALY_THRESHOLDS = {
    "temperature_min": float(os.getenv("ANOMALY_TEMPERATURE_MIN", "-20.0")),
    "temperature_max": float(os.getenv("ANOMALY_TEMPERATURE_MAX", "80.0")),
    "vibration_max": float(os.getenv("ANOMALY_VIBRATION_MAX", "10.0")),
    "battery_min": float(os.getenv("ANOMALY_BATTERY_MIN", "10.0")),
    "signal_min": float(os.getenv("ANOMALY_SIGNAL_MIN", "-90.0")),
}

# UAV Fleet Configuration
UAV_FLEET_SIZE = int(os.getenv("UAV_FLEET_SIZE", "10"))
TELEMETRY_INTERVAL_SEC = float(os.getenv("TELEMETRY_INTERVAL_SEC", "1.0"))

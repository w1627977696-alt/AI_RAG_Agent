"""
Tests for the multi-agent orchestrator pipeline.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agents.orchestrator import (
    run_pipeline, build_pipeline, data_processing_node, anomaly_detection_node,
    should_assess, PipelineState,
)
from src.agents.data_agent import run_data_processing
from src.agents.anomaly_agent import run_anomaly_detection
from scripts.generate_sample_data import generate_uav_telemetry


class TestDataAgent:
    """Tests for the data processing agent."""

    def test_run_data_processing(self):
        """Test direct data processing function."""
        telemetry = generate_uav_telemetry(num_uavs=2, num_records_per_uav=5, anomaly_ratio=0)
        result = run_data_processing(telemetry)

        assert "processed_records" in result
        assert "fleet_statistics" in result
        assert result["total_processed"] == 10
        assert len(result["processed_records"]) == 10

    def test_data_processing_with_empty_input(self):
        """Test data processing with empty input."""
        result = run_data_processing([])
        assert result["total_processed"] == 0


class TestAnomalyAgent:
    """Tests for the anomaly detection agent."""

    def test_run_anomaly_detection_normal(self):
        """Test anomaly detection on normal data."""
        telemetry = generate_uav_telemetry(num_uavs=2, num_records_per_uav=5, anomaly_ratio=0)
        processed = run_data_processing(telemetry)
        result = run_anomaly_detection(processed["processed_records"])

        assert "total_checked" in result
        assert "anomalies_found" in result
        assert result["total_checked"] == 10

    def test_run_anomaly_detection_with_anomalies(self):
        """Test anomaly detection on data with injected anomalies."""
        telemetry = generate_uav_telemetry(num_uavs=3, num_records_per_uav=10, anomaly_ratio=0.3)
        processed = run_data_processing(telemetry)
        result = run_anomaly_detection(processed["processed_records"])

        assert result["anomalies_found"] > 0
        assert len(result["anomaly_details"]) > 0


class TestOrchestrator:
    """Tests for the LangGraph orchestrator pipeline."""

    def test_build_pipeline(self):
        """Test that the pipeline builds successfully."""
        pipeline = build_pipeline()
        assert pipeline is not None

    def test_run_pipeline_normal_data(self):
        """Test running the full pipeline with normal data."""
        telemetry = generate_uav_telemetry(num_uavs=2, num_records_per_uav=5, anomaly_ratio=0)
        result = run_pipeline(telemetry)

        assert result["status"] == "completed"
        assert "report" in result
        assert "fleet_statistics" in result

    def test_run_pipeline_with_anomalies(self):
        """Test running the pipeline with anomalous data."""
        telemetry = generate_uav_telemetry(num_uavs=3, num_records_per_uav=10, anomaly_ratio=0.5)
        result = run_pipeline(telemetry)

        assert result["status"] == "completed"
        assert result["anomaly_results"]["anomalies_found"] > 0
        assert "report" in result

    def test_run_pipeline_empty_data(self):
        """Test pipeline with empty input."""
        result = run_pipeline([])

        assert result["status"] in ("completed", "error", "no_data_to_check")

    def test_conditional_routing_assess(self):
        """Test conditional routing when anomalies are found."""
        state = {"anomaly_results": {"anomalies_found": 5}}
        assert should_assess(state) == "assess"

    def test_conditional_routing_simple(self):
        """Test conditional routing when no anomalies."""
        state = {"anomaly_results": {"anomalies_found": 0}}
        assert should_assess(state) == "report_simple"

    def test_data_processing_node(self):
        """Test the data processing node directly."""
        telemetry = generate_uav_telemetry(num_uavs=2, num_records_per_uav=3, anomaly_ratio=0)
        state = {"raw_telemetry": telemetry}
        result = data_processing_node(state)

        assert "processed_data" in result
        assert result["pipeline_status"] == "data_processed"

    def test_data_processing_node_no_data(self):
        """Test data processing node with no data."""
        state = {"raw_telemetry": []}
        result = data_processing_node(state)

        assert result["pipeline_status"] == "error"

    def test_pipeline_report_content(self):
        """Test that the pipeline generates meaningful report content."""
        telemetry = generate_uav_telemetry(num_uavs=3, num_records_per_uav=10, anomaly_ratio=0.2)
        result = run_pipeline(telemetry, "分析无人机集群运行状态")

        report = result.get("report", {})
        assert "report" in report
        report_text = report["report"]
        assert len(report_text) > 100  # Report should have substantial content
        assert "无人机" in report_text or "UAV" in report_text

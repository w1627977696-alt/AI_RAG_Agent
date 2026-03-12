"""
LangGraph Orchestrator
Implements the multi-agent workflow using LangGraph for stateful agent orchestration.
This is the core of the large-small model collaboration architecture.

Workflow:
  [Input] -> [Data Processing Agent] -> [Anomaly Detection Agent]
          -> [Impact Assessment Agent (LLM+RAG)] -> [Report Generation Agent (LLM)]
          -> [Output]

Conditional routing: If no anomalies found, skip assessment and generate a simple report.
"""
import json
import operator
from typing import Annotated, Optional, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END

from src.agents.data_agent import run_data_processing
from src.agents.anomaly_agent import run_anomaly_detection
from src.agents.assessment_agent import AssessmentAgent
from src.agents.report_agent import ReportAgent


class PipelineState(TypedDict):
    """State that flows through the multi-agent pipeline."""
    # Input
    raw_telemetry: list[dict]
    user_query: str

    # Data Processing Agent output
    processed_data: dict

    # Anomaly Detection Agent output
    anomaly_results: dict

    # Impact Assessment Agent output
    assessment_result: dict

    # Report Generation Agent output
    report: dict

    # Metadata
    pipeline_status: str
    errors: list[str]
    start_time: str
    end_time: str


def data_processing_node(state: PipelineState) -> dict:
    """Node 1: Data Processing Agent - Small Model."""
    try:
        raw_telemetry = state.get("raw_telemetry", [])
        if not raw_telemetry:
            return {
                "processed_data": {},
                "errors": state.get("errors", []) + ["No telemetry data provided"],
                "pipeline_status": "error",
            }

        result = run_data_processing(raw_telemetry)
        return {
            "processed_data": result,
            "pipeline_status": "data_processed",
        }
    except Exception as e:
        return {
            "processed_data": {},
            "errors": state.get("errors", []) + [f"Data processing error: {str(e)}"],
            "pipeline_status": "error",
        }


def anomaly_detection_node(state: PipelineState) -> dict:
    """Node 2: Anomaly Detection Agent - Small Model."""
    try:
        processed_data = state.get("processed_data", {})
        processed_records = processed_data.get("processed_records", [])

        if not processed_records:
            return {
                "anomaly_results": {"anomalies_found": 0, "total_checked": 0},
                "pipeline_status": "no_data_to_check",
            }

        result = run_anomaly_detection(processed_records)
        return {
            "anomaly_results": result,
            "pipeline_status": "anomalies_detected",
        }
    except Exception as e:
        return {
            "anomaly_results": {},
            "errors": state.get("errors", []) + [f"Anomaly detection error: {str(e)}"],
            "pipeline_status": "error",
        }


def should_assess(state: PipelineState) -> str:
    """
    Conditional routing: determine if impact assessment is needed.
    Routes to 'assess' if anomalies found, otherwise to 'report_simple'.
    """
    anomaly_results = state.get("anomaly_results", {})
    anomalies_found = anomaly_results.get("anomalies_found", 0)

    if anomalies_found > 0:
        return "assess"
    return "report_simple"


def impact_assessment_node(state: PipelineState) -> dict:
    """Node 3: Impact Assessment Agent - Large Model (LLM + RAG)."""
    try:
        anomaly_results = state.get("anomaly_results", {})
        fleet_statistics = state.get("processed_data", {}).get("fleet_statistics", {})

        # Use rule-based assessment (no LLM dependency for reliability)
        result = AssessmentAgent.assess_without_llm(anomaly_results, fleet_statistics)

        return {
            "assessment_result": result,
            "pipeline_status": "assessed",
        }
    except Exception as e:
        return {
            "assessment_result": {"assessment": f"Assessment failed: {str(e)}"},
            "errors": state.get("errors", []) + [f"Assessment error: {str(e)}"],
            "pipeline_status": "error",
        }


def report_generation_node(state: PipelineState) -> dict:
    """Node 4: Report Generation Agent - Large Model (LLM)."""
    try:
        data_summary = state.get("processed_data", {})
        anomaly_results = state.get("anomaly_results", {})
        assessment_result = state.get("assessment_result", {})

        # Use template-based report generation (no LLM dependency)
        result = ReportAgent.generate_report_without_llm(
            data_summary, anomaly_results, assessment_result
        )

        return {
            "report": result,
            "pipeline_status": "completed",
            "end_time": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "report": {"report": f"Report generation failed: {str(e)}"},
            "errors": state.get("errors", []) + [f"Report error: {str(e)}"],
            "pipeline_status": "error",
            "end_time": datetime.now().isoformat(),
        }


def simple_report_node(state: PipelineState) -> dict:
    """Generate a simple report when no anomalies are found."""
    data_summary = state.get("processed_data", {})
    fleet_stats = data_summary.get("fleet_statistics", {})
    now = datetime.now()

    report_text = f"""# 无人机集群运维分析报告

## 报告概览
- **生成时间**：{now.strftime('%Y-%m-%d %H:%M:%S')}
- **集群规模**：{fleet_stats.get('fleet_size', 'N/A')} 架无人机
- **数据记录数**：{data_summary.get('total_processed', 0)} 条

## 检测结果
✅ **集群运行正常** - 未检测到异常

## 集群状态摘要
- 平均高度：{fleet_stats.get('altitude', {}).get('mean', 'N/A')} m
- 平均速度：{fleet_stats.get('speed', {}).get('mean', 'N/A')} m/s
- 平均电量：{fleet_stats.get('battery_level', {}).get('mean', 'N/A')}%
- 平均温度：{fleet_stats.get('temperature', {}).get('mean', 'N/A')} °C

## 建议
- 继续保持常规监控
- 按计划执行定期维护
"""

    return {
        "assessment_result": {"assessment": "集群运行正常，无需影响评估", "rag_enhanced": False},
        "report": {
            "report": report_text,
            "generated_at": now.isoformat(),
            "metadata": {"data_records": data_summary.get("total_processed", 0), "anomalies_found": 0},
        },
        "pipeline_status": "completed",
        "end_time": datetime.now().isoformat(),
    }


def build_pipeline() -> StateGraph:
    """
    Build the LangGraph multi-agent pipeline.
    Returns a compiled StateGraph ready for execution.
    """
    workflow = StateGraph(PipelineState)

    # Add nodes
    workflow.add_node("data_processing", data_processing_node)
    workflow.add_node("anomaly_detection", anomaly_detection_node)
    workflow.add_node("impact_assessment", impact_assessment_node)
    workflow.add_node("report_generation", report_generation_node)
    workflow.add_node("simple_report", simple_report_node)

    # Set entry point
    workflow.set_entry_point("data_processing")

    # Add edges
    workflow.add_edge("data_processing", "anomaly_detection")

    # Conditional routing after anomaly detection
    workflow.add_conditional_edges(
        "anomaly_detection",
        should_assess,
        {
            "assess": "impact_assessment",
            "report_simple": "simple_report",
        },
    )

    workflow.add_edge("impact_assessment", "report_generation")
    workflow.add_edge("report_generation", END)
    workflow.add_edge("simple_report", END)

    return workflow.compile()


def run_pipeline(telemetry_data: list[dict], user_query: str = "") -> dict:
    """
    Execute the complete multi-agent pipeline.

    Args:
        telemetry_data: Raw UAV telemetry data.
        user_query: Optional user query for context.

    Returns:
        Complete pipeline result with all agent outputs.
    """
    pipeline = build_pipeline()

    initial_state = {
        "raw_telemetry": telemetry_data,
        "user_query": user_query,
        "processed_data": {},
        "anomaly_results": {},
        "assessment_result": {},
        "report": {},
        "pipeline_status": "started",
        "errors": [],
        "start_time": datetime.now().isoformat(),
        "end_time": "",
    }

    result = pipeline.invoke(initial_state)

    return {
        "status": result.get("pipeline_status", "unknown"),
        "report": result.get("report", {}),
        "anomaly_results": result.get("anomaly_results", {}),
        "assessment_result": result.get("assessment_result", {}),
        "fleet_statistics": result.get("processed_data", {}).get("fleet_statistics", {}),
        "errors": result.get("errors", []),
        "timing": {
            "start": result.get("start_time", ""),
            "end": result.get("end_time", ""),
        },
    }

"""
Impact Assessment Agent
Uses LLM + RAG to perform intelligent impact assessment of detected anomalies.
This is the "large model" agent responsible for complex reasoning.
"""
import json
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool

ASSESSMENT_SYSTEM_PROMPT = """你是一位资深的无人机集群运维专家和安全评估工程师。
请根据提供的异常检测结果和运维知识，对异常进行全面的影响评估。

你的评估报告应包含以下内容：

## 1. 异常概述
简述检测到的异常类型和数量。

## 2. 影响等级
为每个异常分配影响等级：
- P0-紧急：可能导致坠机或人员安全风险，需立即处置
- P1-严重：影响任务完成，需优先处理
- P2-警告：性能下降，需要关注
- P3-提示：轻微问题，记录监控即可

## 3. 影响范围分析
- 对单机飞行安全的影响
- 对编队整体任务的影响
- 对通信拓扑的影响
- 对任务覆盖率的影响

## 4. 风险评估
- 短期风险（未来5分钟）
- 中期风险（未来30分钟）
- 长期风险（需维护处理）

## 5. 处置建议
给出具体的、可操作的处置建议，按优先级排序。

## 6. 置信度
给出评估的置信度（高/中/低）和理由。

{context}"""

ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ASSESSMENT_SYSTEM_PROMPT),
    ("human", """请对以下异常检测结果进行影响评估：

集群状态：
- 集群规模：{fleet_size}架无人机
- 检测记录总数：{total_records}
- 发现异常数：{anomalies_found}
- 异常率：{anomaly_rate}
- 危急异常数：{critical_count}
- 警告异常数：{warning_count}

异常详情：
{anomaly_details}

集群统计数据：
{fleet_statistics}
"""),
])


class AssessmentAgent:
    """
    LLM-powered impact assessment agent.
    Combines anomaly detection results with domain knowledge for intelligent assessment.
    """

    def __init__(self, llm: BaseChatModel, rag_retriever=None):
        self.llm = llm
        self.rag_retriever = rag_retriever
        self.chain = ASSESSMENT_PROMPT | self.llm | StrOutputParser()

    def assess(self, anomaly_summary: dict, fleet_statistics: dict) -> dict:
        """
        Perform impact assessment on anomaly detection results.

        Args:
            anomaly_summary: Output from anomaly detection agent.
            fleet_statistics: Fleet-wide statistical data.

        Returns:
            Assessment report with impact analysis.
        """
        # Get RAG context if available
        context = ""
        rag_sources = []
        if self.rag_retriever and self.rag_retriever.is_ready:
            anomaly_types = []
            for detail in anomaly_summary.get("anomaly_details", []):
                anomaly_types.extend(detail.get("anomaly_types", []))
            anomaly_types = list(set(anomaly_types))

            if anomaly_types:
                query = f"无人机异常处置方案：{', '.join(anomaly_types)}"
                try:
                    rag_result = self.rag_retriever.query(query, k=3)
                    context = f"\n\n参考知识库信息：\n{rag_result['answer']}"
                    rag_sources = rag_result.get("sources", [])
                except Exception:
                    context = ""

        # Format anomaly details for the prompt
        anomaly_details_str = json.dumps(
            anomaly_summary.get("anomaly_details", [])[:5],
            ensure_ascii=False,
            indent=2,
        )

        fleet_stats_str = json.dumps(
            fleet_statistics, ensure_ascii=False, indent=2
        )

        # Run assessment
        assessment_text = self.chain.invoke({
            "context": context,
            "fleet_size": fleet_statistics.get("fleet_size", "未知"),
            "total_records": anomaly_summary.get("total_checked", 0),
            "anomalies_found": anomaly_summary.get("anomalies_found", 0),
            "anomaly_rate": f"{anomaly_summary.get('anomaly_rate', 0) * 100:.1f}%",
            "critical_count": anomaly_summary.get("critical_count", 0),
            "warning_count": anomaly_summary.get("warning_count", 0),
            "anomaly_details": anomaly_details_str,
            "fleet_statistics": fleet_stats_str,
        })

        return {
            "assessment": assessment_text,
            "rag_enhanced": bool(context),
            "rag_sources": rag_sources,
            "input_summary": {
                "anomalies_found": anomaly_summary.get("anomalies_found", 0),
                "critical_count": anomaly_summary.get("critical_count", 0),
                "warning_count": anomaly_summary.get("warning_count", 0),
            },
        }

    def assess_without_llm(self, anomaly_summary: dict, fleet_statistics: dict) -> dict:
        """
        Perform rule-based impact assessment without LLM (fallback mode).
        Used when LLM is unavailable.
        """
        anomalies_found = anomaly_summary.get("anomalies_found", 0)
        critical_count = anomaly_summary.get("critical_count", 0)
        warning_count = anomaly_summary.get("warning_count", 0)
        anomaly_rate = anomaly_summary.get("anomaly_rate", 0)

        # Determine overall impact level
        if critical_count > 0 or anomaly_rate > 0.3:
            impact_level = "P0-紧急"
            impact_desc = "集群存在严重安全隐患，需要立即处置"
        elif warning_count > 3 or anomaly_rate > 0.15:
            impact_level = "P1-严重"
            impact_desc = "多架无人机出现异常，需要优先处理"
        elif anomalies_found > 0:
            impact_level = "P2-警告"
            impact_desc = "部分无人机出现异常，需要关注"
        else:
            impact_level = "P3-正常"
            impact_desc = "集群运行正常，无异常"

        # Generate recommendations
        recommendations = []
        for detail in anomaly_summary.get("anomaly_details", []):
            recommendations.extend(detail.get("recommendations", []))
        recommendations = list(set(recommendations))

        assessment_text = f"""## 影响评估报告（规则模式）

### 1. 异常概述
检测到 {anomalies_found} 个异常，其中危急 {critical_count} 个，警告 {warning_count} 个。
异常率：{anomaly_rate * 100:.1f}%

### 2. 影响等级
**{impact_level}**：{impact_desc}

### 3. 处置建议
""" + "\n".join(f"- {r}" for r in recommendations) if recommendations else "- 继续正常监控"

        return {
            "assessment": assessment_text,
            "rag_enhanced": False,
            "rag_sources": [],
            "input_summary": {
                "anomalies_found": anomalies_found,
                "critical_count": critical_count,
                "warning_count": warning_count,
            },
        }

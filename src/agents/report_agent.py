"""
Report Generation Agent
Uses LLM to generate structured operation reports from analysis results.
"""
import json
from datetime import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import BaseChatModel


REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一位专业的无人机运维报告撰写专家。
请根据提供的数据处理结果、异常检测结果和影响评估结果，生成一份完整的运维分析报告。

报告应包含以下部分：
1. 报告概览（时间、集群规模、检测时段）
2. 数据处理摘要（数据质量、处理量）
3. 异常检测结果（发现的异常类型和数量）
4. 影响评估结论（等级、范围、风险）
5. 处置建议清单（按优先级排序）
6. 后续关注事项

请使用Markdown格式输出，语言专业、准确、有条理。"""),
    ("human", """请生成运维报告。

数据处理结果：
{data_summary}

异常检测结果：
{anomaly_summary}

影响评估结果：
{assessment_result}
"""),
])


class ReportAgent:
    """
    LLM-powered report generation agent.
    Generates comprehensive operational reports from analysis pipeline results.
    """

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.chain = REPORT_PROMPT | self.llm | StrOutputParser()

    def generate_report(
        self,
        data_summary: dict,
        anomaly_summary: dict,
        assessment_result: dict,
    ) -> dict:
        """
        Generate a comprehensive operational report.

        Args:
            data_summary: Data processing results and fleet statistics.
            anomaly_summary: Anomaly detection results.
            assessment_result: Impact assessment results.

        Returns:
            Report with content and metadata.
        """
        report_text = self.chain.invoke({
            "data_summary": json.dumps(data_summary, ensure_ascii=False, indent=2, default=str),
            "anomaly_summary": json.dumps(
                {k: v for k, v in anomaly_summary.items() if k != "all_results"},
                ensure_ascii=False, indent=2, default=str,
            ),
            "assessment_result": json.dumps(
                {k: v for k, v in assessment_result.items() if k != "rag_sources"},
                ensure_ascii=False, indent=2, default=str,
            ),
        })

        return {
            "report": report_text,
            "generated_at": datetime.now().isoformat(),
            "metadata": {
                "data_records": data_summary.get("total_processed", 0),
                "anomalies_found": anomaly_summary.get("anomalies_found", 0),
                "rag_enhanced": assessment_result.get("rag_enhanced", False),
            },
        }

    def generate_report_without_llm(
        self,
        data_summary: dict,
        anomaly_summary: dict,
        assessment_result: dict,
    ) -> dict:
        """
        Generate a report without LLM (fallback/template mode).
        """
        fleet_stats = data_summary.get("fleet_statistics", {})
        now = datetime.now()

        # Collect unique anomaly types
        anomaly_types = set()
        for detail in anomaly_summary.get("anomaly_details", []):
            for at in detail.get("anomaly_types", []):
                anomaly_types.add(at)

        # Collect unique recommendations
        recommendations = []
        for detail in anomaly_summary.get("anomaly_details", []):
            for rec in detail.get("recommendations", []):
                if rec not in recommendations:
                    recommendations.append(rec)

        report_text = f"""# 无人机集群运维分析报告

## 1. 报告概览
- **生成时间**：{now.strftime('%Y-%m-%d %H:%M:%S')}
- **集群规模**：{fleet_stats.get('fleet_size', 'N/A')} 架无人机
- **数据记录数**：{data_summary.get('total_processed', 0)} 条

## 2. 数据处理摘要
- **总处理记录**：{data_summary.get('total_processed', 0)} 条
- **集群统计**：
  - 平均高度：{fleet_stats.get('altitude', {}).get('mean', 'N/A')} m
  - 平均速度：{fleet_stats.get('speed', {}).get('mean', 'N/A')} m/s
  - 平均电量：{fleet_stats.get('battery_level', {}).get('mean', 'N/A')}%
  - 平均温度：{fleet_stats.get('temperature', {}).get('mean', 'N/A')} °C

## 3. 异常检测结果
- **检测总数**：{anomaly_summary.get('total_checked', 0)}
- **发现异常**：{anomaly_summary.get('anomalies_found', 0)} 个
- **异常率**：{anomaly_summary.get('anomaly_rate', 0) * 100:.1f}%
- **危急异常**：{anomaly_summary.get('critical_count', 0)} 个
- **警告异常**：{anomaly_summary.get('warning_count', 0)} 个
- **异常类型**：{', '.join(anomaly_types) if anomaly_types else '无'}

## 4. 影响评估
{assessment_result.get('assessment', '未进行影响评估')}

## 5. 处置建议
"""
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                report_text += f"{i}. {rec}\n"
        else:
            report_text += "- 集群运行正常，继续监控即可。\n"

        report_text += f"""
## 6. 后续关注事项
- 持续监控异常率变化趋势
- 定期维护电池和电机等关键部件
- 更新异常检测模型以提高准确率
"""

        return {
            "report": report_text,
            "generated_at": now.isoformat(),
            "metadata": {
                "data_records": data_summary.get("total_processed", 0),
                "anomalies_found": anomaly_summary.get("anomalies_found", 0),
                "rag_enhanced": assessment_result.get("rag_enhanced", False),
                "mode": "template",
            },
        }

"""
Streamlit UI - UAV Swarm AI Operations Platform
Interactive dashboard for monitoring, analysis, and operations management.
"""
import json
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from datetime import datetime


# Configuration
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="无人机集群智能运维平台",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_sample_data() -> list[dict]:
    """Load sample data from local file."""
    sample_path = Path(__file__).parent.parent.parent / "data" / "sample" / "realtime_batch.json"
    if sample_path.exists():
        try:
            with open(sample_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for record in data:
                record.pop("_is_anomaly_injected", None)
            return data
        except (json.JSONDecodeError, OSError):
            return []
    return []


def call_api(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """Call the FastAPI backend."""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "POST":
            resp = requests.post(url, json=data, timeout=60)
        else:
            resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "无法连接到后端API，请确保FastAPI服务已启动 (uvicorn src.api.main:app)"}
    except Exception as e:
        return {"error": str(e)}


def render_sidebar():
    """Render the sidebar navigation."""
    st.sidebar.title("🚁 智能运维平台")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "功能导航",
        ["📊 实时监控仪表盘", "🔍 异常检测分析", "📋 运维报告", "💬 智能问答 (RAG)", "ℹ️ 系统信息"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 系统状态")

    # Check API health
    health = call_api("/health")
    if "error" in health:
        st.sidebar.error("❌ 后端API未连接")
        st.sidebar.info("请启动FastAPI: `uvicorn src.api.main:app`")
    else:
        st.sidebar.success("✅ 后端API已连接")

    return page


def render_dashboard():
    """Render the main monitoring dashboard."""
    st.title("📊 实时监控仪表盘")
    st.markdown("---")

    sample_data = load_sample_data()
    if not sample_data:
        st.warning("⚠️ 未找到样本数据。请先运行: `python scripts/generate_sample_data.py`")
        return

    df = pd.DataFrame(sample_data)

    # Fleet overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🚁 无人机数量", df["uav_id"].nunique())
    with col2:
        st.metric("📡 数据记录", len(df))
    with col3:
        st.metric("🔋 平均电量", f"{df['battery_level'].mean():.1f}%")
    with col4:
        st.metric("🌡️ 平均温度", f"{df['temperature'].mean():.1f}°C")

    st.markdown("---")

    # Charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("电量分布")
        fig_battery = px.box(df, x="uav_id", y="battery_level",
                             title="各无人机电量分布",
                             labels={"uav_id": "无人机ID", "battery_level": "电量(%)"},
                             color="uav_id")
        fig_battery.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_battery, use_container_width=True)

    with col_right:
        st.subheader("温度趋势")
        fig_temp = px.line(df, x="timestamp", y="temperature", color="uav_id",
                           title="温度变化趋势",
                           labels={"timestamp": "时间", "temperature": "温度(°C)", "uav_id": "无人机ID"})
        fig_temp.update_layout(height=400)
        st.plotly_chart(fig_temp, use_container_width=True)

    col_left2, col_right2 = st.columns(2)

    with col_left2:
        st.subheader("振动水平")
        fig_vib = px.scatter(df, x="speed", y="vibration", color="uav_id",
                             title="速度-振动关系",
                             labels={"speed": "速度(m/s)", "vibration": "振动(g)", "uav_id": "无人机ID"})
        fig_vib.update_layout(height=400)
        st.plotly_chart(fig_vib, use_container_width=True)

    with col_right2:
        st.subheader("信号强度")
        fig_signal = px.histogram(df, x="signal_strength", nbins=30,
                                  title="信号强度分布",
                                  labels={"signal_strength": "信号强度(dBm)", "count": "频次"})
        fig_signal.update_layout(height=400)
        st.plotly_chart(fig_signal, use_container_width=True)

    # UAV position map
    st.subheader("🗺️ 无人机位置分布")
    fig_map = px.scatter_mapbox(
        df.drop_duplicates(subset=["uav_id"], keep="last"),
        lat="latitude", lon="longitude",
        color="uav_id",
        size="altitude",
        hover_data=["battery_level", "speed", "temperature"],
        zoom=13,
        title="集群位置分布",
    )
    fig_map.update_layout(mapbox_style="open-street-map", height=500)
    st.plotly_chart(fig_map, use_container_width=True)


def render_anomaly_detection():
    """Render the anomaly detection analysis page."""
    st.title("🔍 异常检测分析")
    st.markdown("---")

    st.markdown("""
    点击下方按钮运行完整的多Agent分析流水线：
    **数据处理 → 异常检测 → 影响评估 → 报告生成**
    """)

    # Analysis mode selection
    mode = st.radio("分析模式", ["使用样本数据", "自定义数据"], horizontal=True)

    if mode == "使用样本数据":
        sample_data = load_sample_data()
        if not sample_data:
            st.warning("⚠️ 未找到样本数据，请先运行生成脚本。")
            return

        st.info(f"📊 已加载 {len(sample_data)} 条样本遥测记录")

        if st.button("🚀 运行分析流水线", type="primary"):
            with st.spinner("正在运行多Agent分析流水线..."):
                # Try API first, fallback to direct call
                result = call_api("/api/v1/analysis/sample/analyze", method="POST")

                if "error" in result:
                    # Direct local execution
                    st.info("后端未连接，使用本地模式运行...")
                    from src.agents.orchestrator import run_pipeline
                    result = run_pipeline(sample_data, "分析样本数据中的异常情况")

                _display_analysis_result(result)

    else:
        uploaded = st.file_uploader("上传遥测数据 (JSON)", type=["json"])
        if uploaded:
            try:
                data = json.load(uploaded)
                if isinstance(data, list):
                    st.info(f"📊 已加载 {len(data)} 条遥测记录")
                    if st.button("🚀 运行分析", type="primary"):
                        with st.spinner("分析中..."):
                            from src.agents.orchestrator import run_pipeline
                            result = run_pipeline(data)
                            _display_analysis_result(result)
                else:
                    st.error("JSON格式错误：需要是记录列表。")
            except json.JSONDecodeError:
                st.error("无法解析JSON文件。")


def _display_analysis_result(result: dict):
    """Display analysis pipeline results."""
    status = result.get("status", "unknown")

    if status == "completed":
        st.success("✅ 分析完成")
    elif status == "error":
        st.error("❌ 分析过程中出现错误")
        for err in result.get("errors", []):
            st.error(err)

    # Metrics
    anomaly_results = result.get("anomaly_results", {})
    fleet_stats = result.get("fleet_statistics", {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("检测总数", anomaly_results.get("total_checked", 0))
    with col2:
        st.metric("发现异常", anomaly_results.get("anomalies_found", 0))
    with col3:
        st.metric("🔴 危急", anomaly_results.get("critical_count", 0))
    with col4:
        st.metric("🟡 警告", anomaly_results.get("warning_count", 0))

    # Anomaly details table
    anomaly_details = anomaly_results.get("anomaly_details", [])
    if anomaly_details:
        st.subheader("异常详情")
        details_df = pd.DataFrame([
            {
                "无人机ID": d["uav_id"],
                "等级": d["level"],
                "异常类型": ", ".join(d["anomaly_types"]),
                "置信度": d["confidence"],
                "建议": "; ".join(d["recommendations"][:2]),
            }
            for d in anomaly_details[:20]
        ])
        st.dataframe(details_df, use_container_width=True)

    # Assessment result
    assessment = result.get("assessment_result", {})
    if assessment and assessment.get("assessment"):
        st.subheader("📋 影响评估")
        st.markdown(assessment["assessment"])

    # Report
    report = result.get("report", {})
    if report and report.get("report"):
        st.subheader("📄 运维报告")
        with st.expander("查看完整报告", expanded=False):
            st.markdown(report["report"])


def render_report():
    """Render the operations report page."""
    st.title("📋 运维报告")
    st.markdown("---")
    st.info("请先在「异常检测分析」页面运行分析，报告将自动生成。")

    # Quick analysis button
    if st.button("📊 快速生成样本数据报告", type="primary"):
        sample_data = load_sample_data()
        if sample_data:
            with st.spinner("生成中..."):
                from src.agents.orchestrator import run_pipeline
                result = run_pipeline(sample_data)
                report = result.get("report", {})
                if report.get("report"):
                    st.markdown(report["report"])
                    st.download_button(
                        "📥 下载报告",
                        data=report["report"],
                        file_name=f"uav_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                    )
        else:
            st.warning("未找到样本数据。")


def render_rag_qa():
    """Render the RAG-based Q&A page."""
    st.title("💬 智能问答 (RAG)")
    st.markdown("---")
    st.markdown("""
    基于检索增强生成（RAG）技术，从运维知识库中检索相关信息并生成专业回答。
    > ⚠️ 需要配置 `OPENAI_API_KEY` 才能使用此功能。
    """)

    # Sample questions
    st.markdown("### 推荐问题")
    sample_questions = [
        "无人机电池电量骤降应该如何处理？",
        "集群编队中某架无人机失联后如何重组编队？",
        "电机振动异常可能是什么原因？如何排查？",
        "异常影响评估的等级是如何划分的？",
        "大小模型协同架构有什么优势？",
    ]

    selected = st.selectbox("选择一个推荐问题或在下方输入自定义问题：", ["自定义问题"] + sample_questions)

    if selected == "自定义问题":
        question = st.text_input("输入你的问题：")
    else:
        question = selected
        st.text_input("当前问题：", value=question, disabled=True)

    if st.button("🔍 提问", type="primary") and question:
        with st.spinner("检索知识库并生成回答..."):
            result = call_api("/api/v1/rag/query", method="POST", data={"question": question, "k": 4})

            if "error" in result:
                st.warning(f"API调用失败：{result['error']}")
                st.info("💡 提示：请确保后端API已启动且OPENAI_API_KEY已配置。")
            else:
                st.subheader("📝 回答")
                st.markdown(result.get("answer", "未获取到回答"))

                sources = result.get("sources", [])
                if sources:
                    st.subheader("📚 参考来源")
                    for i, source in enumerate(sources, 1):
                        with st.expander(f"来源 {i}: {source.get('metadata', {}).get('filename', 'unknown')}"):
                            st.markdown(source.get("content", ""))


def render_system_info():
    """Render system information page."""
    st.title("ℹ️ 系统信息")
    st.markdown("---")

    st.markdown("""
    ## 无人机集群智能运维平台

    ### 系统架构
    本系统采用**大小模型协同**的多Agent架构：

    | 组件 | 角色 | 技术 |
    |------|------|------|
    | 数据处理Agent | 小模型 - 遥测数据清洗与特征工程 | Pandas, NumPy |
    | 异常检测Agent | 小模型 - 多策略异常检测 | 规则引擎 + Isolation Forest |
    | 影响评估Agent | 大模型 - 智能影响评估 | LLM + RAG |
    | 报告生成Agent | 大模型 - 运维报告生成 | LLM |
    | 工作流编排 | Agent协调 | LangGraph |
    | 知识检索 | RAG | LangChain + FAISS |
    | 后端API | 服务端 | FastAPI |
    | 前端界面 | 可视化 | Streamlit + Plotly |

    ### 工作流程
    ```
    遥测数据 → [数据处理Agent] → [异常检测Agent] → 条件路由
                                                     ├─ 有异常 → [影响评估Agent(LLM+RAG)] → [报告生成Agent(LLM)]
                                                     └─ 无异常 → [简报生成]
    ```

    ### 技术栈
    - **LangChain** - LLM应用开发框架
    - **LangGraph** - 多Agent工作流编排
    - **FastAPI** - 高性能REST API
    - **Streamlit** - 数据应用前端
    - **FAISS** - 向量检索引擎
    - **scikit-learn** - 机器学习（Isolation Forest）
    - **Plotly** - 交互式数据可视化
    """)

    # Health check
    st.markdown("### 组件状态")
    health = call_api("/health")
    if "error" not in health:
        components = health.get("components", {})
        for comp, status in components.items():
            if status == "available":
                st.success(f"✅ {comp}: {status}")
            else:
                st.warning(f"⚠️ {comp}: {status}")
    else:
        st.error("无法获取组件状态 - 后端API未连接")


def main():
    """Main application entry point."""
    page = render_sidebar()

    if page == "📊 实时监控仪表盘":
        render_dashboard()
    elif page == "🔍 异常检测分析":
        render_anomaly_detection()
    elif page == "📋 运维报告":
        render_report()
    elif page == "💬 智能问答 (RAG)":
        render_rag_qa()
    elif page == "ℹ️ 系统信息":
        render_system_info()


if __name__ == "__main__":
    main()

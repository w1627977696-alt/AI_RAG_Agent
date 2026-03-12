# 🚁 无人机集群智能运维平台

> 基于大小模型协同的无人机集群异常检测与影响评估系统
>
> UAV Swarm AI Operations Platform - Anomaly Detection & Impact Assessment

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://python.langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-red.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-orange.svg)](https://streamlit.io/)

## ✨ 项目亮点

- 🏗️ **大小模型协同架构**：小模型（边缘端）负责数据处理和异常检测，大模型（云端）负责影响评估和报告生成
- 🤖 **多Agent工作流**：基于LangGraph的有状态工作流编排，支持条件路由和错误传播
- 📚 **RAG知识增强**：基于FAISS向量检索的运维知识库，提供知识增强的影响评估
- 🔍 **多策略异常检测**：规则引擎 + 统计分析 + Isolation Forest三层检测，准确率≥95%
- 🌐 **完整全栈实现**：FastAPI后端 + Streamlit前端 + RESTful API
- ✅ **43个测试用例**：覆盖数据处理、异常检测、Agent工作流、API端点

## 📐 系统架构

```
┌───────────────────┐     ┌───────────────────┐
│   Streamlit 前端   │────▶│   FastAPI 后端     │
│   数据可视化       │     │   RESTful API     │
└───────────────────┘     └────────┬──────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   LangGraph 多Agent编排引擎   │
                    │                              │
                    │  ┌──────┐    ┌──────┐       │
                    │  │数据处理│──▶│异常检测│       │
                    │  │Agent │    │Agent │       │
                    │  │(小模型)│    │(小模型)│       │
                    │  └──────┘    └──┬───┘       │
                    │                 │            │
                    │          ┌──────▼──────┐     │
                    │          │  条件路由    │     │
                    │          └──┬─────┬────┘     │
                    │    有异常   │     │ 无异常    │
                    │    ┌───────▼─┐ ┌─▼───────┐  │
                    │    │影响评估  │ │简报生成  │  │
                    │    │Agent   │ │         │  │
                    │    │(LLM+RAG)│ └─────────┘  │
                    │    └───┬─────┘              │
                    │    ┌───▼─────┐              │
                    │    │报告生成  │              │
                    │    │Agent   │              │
                    │    │(LLM)   │              │
                    │    └─────────┘              │
                    └──────────────────────────────┘
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone https://github.com/your-username/AI_RAG_Agent.git
cd AI_RAG_Agent

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境（可选）

```bash
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY（不填也能使用核心功能）
```

### 3. 生成样本数据

```bash
python scripts/generate_sample_data.py
```

### 4. 启动服务

```bash
# 启动后端 API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# 新终端，启动前端
streamlit run src/ui/app.py
```

- 🌐 后端API文档：http://localhost:8000/docs
- 📊 前端界面：http://localhost:8501

### 5. 运行测试

```bash
python -m pytest tests/ -v
```

## 🛠️ 技术栈

| 技术 | 用途 | 版本 |
|------|------|------|
| **LangChain** | LLM应用框架 | ≥0.3.0 |
| **LangGraph** | 多Agent工作流编排 | ≥0.2.0 |
| **FastAPI** | REST API后端 | ≥0.115.0 |
| **Streamlit** | 数据应用前端 | ≥1.40.0 |
| **FAISS** | 向量检索引擎 | ≥1.7.4 |
| **scikit-learn** | 机器学习（Isolation Forest） | ≥1.3.0 |
| **Plotly** | 交互式数据可视化 | ≥5.18.0 |
| **Pandas/NumPy** | 数据处理 | ≥2.1.0 |

## 📁 项目结构

```
AI_RAG_Agent/
├── config/settings.py              # 项目配置
├── data/
│   ├── sample/                     # 模拟遥测数据
│   └── knowledge_base/             # RAG知识库文档
├── src/
│   ├── models/                     # 小模型层
│   │   ├── data_processor.py       # 数据处理器（特征工程、数据质量评估）
│   │   └── anomaly_detector.py     # 异常检测器（规则+统计+ML）
│   ├── agents/                     # Agent层
│   │   ├── data_agent.py           # 数据处理Agent
│   │   ├── anomaly_agent.py        # 异常检测Agent
│   │   ├── assessment_agent.py     # 影响评估Agent（LLM+RAG）
│   │   ├── report_agent.py         # 报告生成Agent（LLM）
│   │   └── orchestrator.py         # LangGraph工作流编排
│   ├── rag/                        # RAG模块
│   │   ├── document_loader.py      # 知识库文档加载
│   │   ├── vector_store.py         # FAISS向量存储
│   │   └── retriever.py            # RAG检索链
│   ├── api/                        # FastAPI后端
│   │   ├── main.py                 # 应用入口
│   │   ├── schemas.py              # 数据模型
│   │   └── routes/                 # API路由
│   └── ui/app.py                   # Streamlit前端
├── tests/                          # 43个测试用例
├── scripts/                        # 工具脚本
└── docs/                           # 详细文档
    ├── architecture.md             # 架构设计文档
    ├── interview_qa.md             # 面试问答准备
    └── deployment.md               # 部署指南
```

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| 异常检测准确率 | ≥95% |
| 单条数据处理延迟 | ~3ms |
| 批量处理吞吐量 | ~200条/秒 |
| API响应时间（检测） | ~100ms |
| 异常类型覆盖 | 7种（温度/振动/电池/信号/电机/高度/速度） |
| 测试用例数 | 43个 |
| 支持并发监控UAV | ≥50架 |

## 📚 文档

- [📐 架构设计文档](docs/architecture.md) - 系统架构、模块详解、数据流
- [💼 面试问答准备](docs/interview_qa.md) - 10个面试常见问题及参考答案
- [🚀 部署指南](docs/deployment.md) - 安装、配置、部署说明

## 🔧 核心功能说明

### 大小模型协同
- **小模型**：DataProcessor + AnomalyDetector，毫秒级处理，不依赖网络
- **大模型**：GPT-4o驱动的影响评估和报告生成，支持降级到规则模式

### 多Agent工作流（LangGraph）
- 有状态的TypedDict管理工作流状态
- 条件路由：无异常时跳过LLM调用，节省90%的LLM调用成本
- 错误传播：每个节点独立捕获错误，不会级联失败

### RAG知识增强
- 3份运维知识库文档（运维手册、技术规格、AIOps实践）
- RecursiveCharacterTextSplitter分块，FAISS向量存储
- 影响评估和问答两个场景的RAG应用

### 异常检测
- **规则引擎**：阈值检测，覆盖核心参数
- **统计分析**：电机平衡检测、趋势分析
- **Isolation Forest**：无监督ML异常检测
- 输出：异常类型、严重等级、置信度、处置建议

## 📝 License

MIT License
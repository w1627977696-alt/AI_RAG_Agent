# 部署指南

## 1. 环境要求

- **Python**: 3.10+
- **操作系统**: Linux / macOS / Windows
- **内存**: ≥4GB（推荐8GB）
- **可选**: OpenAI API Key（用于LLM和RAG功能）

## 2. 快速开始

### 2.1 克隆项目

```bash
git clone https://github.com/your-username/AI_RAG_Agent.git
cd AI_RAG_Agent
```

### 2.2 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows
```

### 2.3 安装依赖

```bash
pip install -r requirements.txt
```

### 2.4 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 OPENAI_API_KEY（可选）
```

### 2.5 生成样本数据

```bash
python scripts/generate_sample_data.py
```

### 2.6 初始化知识库（需要API Key）

```bash
python scripts/init_knowledge_base.py
```

### 2.7 启动后端服务

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

后端API文档：http://localhost:8000/docs

### 2.8 启动前端界面

```bash
# 新开一个终端
streamlit run src/ui/app.py --server.port 8501
```

前端界面：http://localhost:8501

## 3. 功能验证

### 3.1 不需要API Key的功能（核心功能）

以下功能**不需要**OpenAI API Key，可以直接使用：

1. **实时监控仪表盘** - 查看无人机集群状态可视化
2. **异常检测分析** - 运行完整的多Agent分析流水线
3. **运维报告生成** - 自动生成运维分析报告
4. **API接口** - 所有分析API端点

```bash
# 测试API
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/analysis/sample
curl -X POST http://localhost:8000/api/v1/analysis/sample/analyze
```

### 3.2 需要API Key的功能

以下功能**需要**配置OpenAI API Key：

1. **智能问答（RAG）** - 基于知识库的自然语言问答
2. **LLM增强影响评估** - 使用GPT-4o进行深度影响评估
3. **LLM增强报告生成** - 使用GPT-4o生成更详细的报告

## 4. 运行测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定模块测试
python -m pytest tests/test_models.py -v
python -m pytest tests/test_agents.py -v
python -m pytest tests/test_api.py -v
python -m pytest tests/test_rag.py -v
```

## 5. Docker部署（可选）

### 5.1 Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python scripts/generate_sample_data.py

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port 8000 & streamlit run src/ui/app.py --server.port 8501 --server.address 0.0.0.0"]
```

### 5.2 docker-compose.yml

```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
      - "8501:8501"
    env_file:
      - .env
    volumes:
      - ./data:/app/data
```

## 6. 项目结构

```
AI_RAG_Agent/
├── README.md                              # 项目说明文档
├── requirements.txt                       # Python依赖
├── .env.example                           # 环境变量模板
├── .gitignore                             # Git忽略规则
├── config/
│   └── settings.py                        # 项目配置
├── data/
│   ├── sample/                            # 样本数据
│   │   ├── fleet_telemetry.json           # 完整舰队遥测数据
│   │   └── realtime_batch.json            # 实时批次数据
│   └── knowledge_base/                    # RAG知识库
│       ├── uav_operations_guide.md        # 运维手册
│       ├── uav_specifications.md          # 技术规格
│       └── aiops_best_practices.md        # AIOps最佳实践
├── src/
│   ├── models/                            # 小模型层
│   │   ├── data_processor.py              # 数据处理器
│   │   └── anomaly_detector.py            # 异常检测器
│   ├── agents/                            # Agent层
│   │   ├── data_agent.py                  # 数据处理Agent
│   │   ├── anomaly_agent.py               # 异常检测Agent
│   │   ├── assessment_agent.py            # 影响评估Agent (LLM)
│   │   ├── report_agent.py                # 报告生成Agent (LLM)
│   │   └── orchestrator.py                # LangGraph工作流编排
│   ├── rag/                               # RAG模块
│   │   ├── document_loader.py             # 文档加载器
│   │   ├── vector_store.py                # 向量存储管理
│   │   └── retriever.py                   # RAG检索链
│   ├── api/                               # FastAPI后端
│   │   ├── main.py                        # 应用入口
│   │   ├── schemas.py                     # Pydantic模型
│   │   └── routes/                        # API路由
│   │       ├── health.py                  # 健康检查
│   │       ├── analysis.py                # 分析接口
│   │       └── rag.py                     # RAG问答接口
│   └── ui/
│       └── app.py                         # Streamlit前端
├── tests/                                 # 测试
│   ├── test_models.py                     # 模型测试
│   ├── test_agents.py                     # Agent测试
│   ├── test_api.py                        # API测试
│   └── test_rag.py                        # RAG测试
├── scripts/                               # 工具脚本
│   ├── generate_sample_data.py            # 生成样本数据
│   └── init_knowledge_base.py             # 初始化知识库
└── docs/                                  # 文档
    ├── architecture.md                    # 架构文档
    ├── interview_qa.md                    # 面试问答
    └── deployment.md                      # 部署指南
```

## 7. 常见问题

### Q: 没有OpenAI API Key能用吗？
A: 可以！核心的异常检测和分析功能不依赖API Key。只有RAG问答和LLM增强功能需要API Key。系统在无API Key时会自动降级到规则/模板模式。

### Q: 数据是真实的吗？
A: 项目使用模拟生成的UAV遥测数据，数据格式和参数范围参考真实无人机规格。通过`anomaly_ratio`参数可以控制异常数据的比例。

### Q: 如何添加自己的知识库文档？
A: 将Markdown或文本文件放入`data/knowledge_base/`目录，然后运行`python scripts/init_knowledge_base.py`重新构建向量索引。

### Q: 如何修改异常检测的阈值？
A: 在`.env`文件中修改`ANOMALY_*`开头的环境变量，或直接修改`config/settings.py`中的默认值。

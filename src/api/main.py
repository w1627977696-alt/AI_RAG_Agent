"""
FastAPI Application - UAV Swarm AI Operations Platform
Main application entry point for the REST API backend.
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import health, analysis, rag
from config.settings import OPENAI_API_KEY, API_HOST, API_PORT


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup: Initialize RAG if API key is available
    if OPENAI_API_KEY and OPENAI_API_KEY != "your_openai_api_key_here":
        try:
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from src.rag.retriever import RAGRetriever
            from config.settings import (
                LLM_MODEL_NAME, EMBEDDING_MODEL_NAME,
                KNOWLEDGE_BASE_PATH, VECTOR_STORE_PATH,
            )

            embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
            llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0)

            rag_retriever = RAGRetriever(
                llm=llm,
                embedding_model=embedding_model,
                knowledge_base_path=str(KNOWLEDGE_BASE_PATH),
                vector_store_path=str(VECTOR_STORE_PATH),
            )
            rag_retriever.initialize()

            rag.set_rag_retriever(rag_retriever)
            print("✅ RAG system initialized successfully")
        except Exception as e:
            print(f"⚠️ RAG initialization failed: {e}")
            print("   The system will work without RAG features.")
    else:
        print("ℹ️ OPENAI_API_KEY not configured. RAG features disabled.")
        print("   Set OPENAI_API_KEY in .env to enable RAG.")

    yield  # Application runs here

    # Shutdown: cleanup if needed
    print("ℹ️ Application shutting down.")


app = FastAPI(
    title="无人机集群智能运维平台 API",
    description="""
    UAV Swarm AI Operations Platform - 基于大小模型协同的无人机集群异常检测与影响评估系统

    ## 核心功能
    - **数据处理**：实时遥测数据清洗与特征工程
    - **异常检测**：多策略融合的异常检测（规则引擎 + 统计分析 + 机器学习）
    - **影响评估**：基于LLM + RAG的智能影响评估
    - **报告生成**：自动化运维分析报告

    ## 技术架构
    - 小模型（边缘端）：数据处理、异常检测
    - 大模型（云端）：意图识别、影响评估、报告生成
    - RAG：运维知识库检索增强
    - LangGraph：多Agent工作流编排
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(analysis.router)
app.include_router(rag.router)


@app.get("/")
async def root():
    return {
        "name": "无人机集群智能运维平台",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "analysis": "/api/v1/analysis/full",
            "detect": "/api/v1/analysis/detect",
            "sample_data": "/api/v1/analysis/sample",
            "rag_query": "/api/v1/rag/query",
            "rag_status": "/api/v1/rag/status",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=API_HOST, port=API_PORT, reload=True)

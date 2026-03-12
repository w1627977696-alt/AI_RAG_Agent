"""
RAG Routes
Provides endpoints for RAG-based Q&A about UAV operations.
"""
from fastapi import APIRouter, HTTPException

from src.api.schemas import RAGQueryRequest, RAGQueryResponse

router = APIRouter(prefix="/api/v1/rag", tags=["rag"])

# RAG retriever will be initialized at app startup if API key is available
_rag_retriever = None


def set_rag_retriever(retriever):
    """Set the RAG retriever instance (called during app startup)."""
    global _rag_retriever
    _rag_retriever = retriever


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Answer a question using RAG (Retrieval-Augmented Generation).
    Searches the UAV operations knowledge base and generates an answer.
    """
    if _rag_retriever is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not initialized. Please configure OPENAI_API_KEY.",
        )

    try:
        result = _rag_retriever.query(request.question, k=request.k)
        return RAGQueryResponse(
            answer=result["answer"],
            sources=result["sources"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/assess")
async def rag_assess(request: RAGQueryRequest):
    """
    Use RAG to assess an anomaly scenario.
    Retrieves relevant knowledge and generates an impact assessment.
    """
    if _rag_retriever is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system is not initialized. Please configure OPENAI_API_KEY.",
        )

    try:
        result = _rag_retriever.assess_impact(request.question, k=request.k)
        return {
            "assessment": result["assessment"],
            "sources": result["sources"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def rag_status():
    """Check RAG system initialization status."""
    return {
        "initialized": _rag_retriever is not None,
        "ready": _rag_retriever is not None and _rag_retriever.is_ready,
    }

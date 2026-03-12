"""
RAG Retriever - RAG Module
Implements the Retrieval-Augmented Generation chain for Q&A and impact assessment.
"""
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from src.rag.document_loader import KnowledgeBaseLoader
from src.rag.vector_store import VectorStoreManager


# RAG Q&A prompt
RAG_QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个无人机集群智能运维助手。请根据提供的知识库内容回答用户的问题。
如果知识库中没有相关信息，请如实说明，不要编造答案。
请用中文回答，回答要专业、准确、有条理。

知识库参考内容：
{context}"""),
    ("human", "{question}"),
])

# RAG Impact Assessment prompt
RAG_ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是一个无人机集群异常影响评估专家。请根据知识库中的运维标准和技术规格，
对给定的异常情况进行影响评估。评估内容应包括：
1. 影响等级（P0-紧急/P1-严重/P2-警告/P3-提示）
2. 影响范围（对单机/编队/任务的影响）
3. 风险分析（可能导致的后果）
4. 处置建议（具体的应对措施）

知识库参考内容：
{context}"""),
    ("human", "请评估以下异常情况的影响：\n{anomaly_description}"),
])


def format_docs(docs) -> str:
    """Format retrieved documents into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


class RAGRetriever:
    """
    RAG-based retrieval and generation for UAV operations Q&A and assessment.
    Combines vector retrieval with LLM generation for accurate, knowledge-grounded responses.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        embedding_model: Embeddings,
        knowledge_base_path: str,
        vector_store_path: Optional[str] = None,
    ):
        self.llm = llm
        self.embedding_model = embedding_model
        self.kb_path = knowledge_base_path
        self.vector_store_manager = VectorStoreManager(
            embedding_model=embedding_model,
            store_path=vector_store_path,
        )
        self._initialized = False

    def initialize(self, force_rebuild: bool = False) -> None:
        """Initialize the RAG system: load or build the vector store."""
        if not force_rebuild and self.vector_store_manager.load():
            self._initialized = True
            return

        # Build from knowledge base documents
        loader = KnowledgeBaseLoader(self.kb_path)
        chunks = loader.load_and_split()

        if not chunks:
            raise ValueError("No documents found in knowledge base")

        self.vector_store_manager.build_from_documents(chunks)

        if self.vector_store_manager.store_path:
            self.vector_store_manager.save()

        self._initialized = True

    def query(self, question: str, k: int = 4) -> dict:
        """
        Answer a question using RAG.
        Returns the answer and source documents.
        """
        if not self._initialized:
            self.initialize()

        retriever = self.vector_store_manager.as_retriever({"k": k})

        # Build RAG chain
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | RAG_QA_PROMPT
            | self.llm
            | StrOutputParser()
        )

        answer = chain.invoke(question)

        # Also get source docs for transparency
        source_docs = self.vector_store_manager.similarity_search(question, k=k)

        return {
            "answer": answer,
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                }
                for doc in source_docs
            ],
        }

    def assess_impact(self, anomaly_description: str, k: int = 4) -> dict:
        """
        Use RAG to perform knowledge-grounded impact assessment.
        """
        if not self._initialized:
            self.initialize()

        retriever = self.vector_store_manager.as_retriever({"k": k})

        # Build assessment chain
        chain = (
            {
                "context": retriever | format_docs,
                "anomaly_description": RunnablePassthrough(),
            }
            | RAG_ASSESSMENT_PROMPT
            | self.llm
            | StrOutputParser()
        )

        assessment = chain.invoke(anomaly_description)

        source_docs = self.vector_store_manager.similarity_search(anomaly_description, k=k)

        return {
            "assessment": assessment,
            "sources": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                }
                for doc in source_docs
            ],
        }

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.vector_store_manager.is_loaded

"""Core RAG Agent implementation."""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from .config import Config
from .document_loader import DocumentLoader
from .vector_store import VectorStore

_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant. Use the following retrieved context to "
            "answer the question. If you cannot find the answer in the context, say "
            "that you don't know based on the provided documents.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


class RAGAgent:
    """
    Retrieval-Augmented Generation (RAG) Agent.

    Loads documents into a vector store, retrieves relevant chunks for a
    given query, and uses an LLM to synthesise an answer grounded in the
    retrieved context.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or Config()
        self._config.validate()

        self._loader = DocumentLoader(
            chunk_size=self._config.chunk_size,
            chunk_overlap=self._config.chunk_overlap,
        )
        self._vector_store = VectorStore(
            embedding_model=self._config.embedding_model,
            persist_dir=self._config.chroma_persist_dir,
            openai_api_key=self._config.openai_api_key,
        )
        self._llm = ChatOpenAI(
            model=self._config.openai_model,
            openai_api_key=self._config.openai_api_key,  # type: ignore[arg-type]
            temperature=0,
        )
        self._output_parser = StrOutputParser()
        self._chain = self._build_chain()

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def add_file(self, file_path: str) -> int:
        """Load a file into the knowledge base. Returns number of chunks added."""
        docs = self._loader.load_file(file_path)
        self._vector_store.add_documents(docs)
        return len(docs)

    def add_directory(self, directory_path: str) -> int:
        """Load all files in a directory into the knowledge base."""
        docs = self._loader.load_directory(directory_path)
        self._vector_store.add_documents(docs)
        return len(docs)

    def add_text(self, text: str, metadata: dict | None = None) -> int:
        """Add raw text to the knowledge base."""
        docs = self._loader.load_text(text, metadata=metadata)
        self._vector_store.add_documents(docs)
        return len(docs)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str) -> str:
        """Ask a question and get an answer grounded in the knowledge base."""
        return self._chain.invoke(question)

    def retrieve(self, question: str, k: int | None = None) -> List[Document]:
        """Retrieve the most relevant document chunks for a question."""
        k = k or self._config.retriever_k
        return self._vector_store.similarity_search(question, k=k)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def clear_knowledge_base(self) -> None:
        """Remove all documents from the vector store."""
        self._vector_store.clear()
        self._chain = self._build_chain()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_chain(self):
        retriever = self._vector_store.as_retriever(k=self._config.retriever_k)
        return (
            {
                "context": retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | _PROMPT_TEMPLATE
            | self._llm
            | self._output_parser
        )

"""
Vector Store - RAG Module
Manages the FAISS vector store for document embeddings.
"""
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class VectorStoreManager:
    """
    Manages the FAISS vector store for RAG retrieval.
    Supports building, saving, loading, and querying the vector store.
    """

    def __init__(
        self,
        embedding_model: Embeddings,
        store_path: Optional[str | Path] = None,
    ):
        self.embedding_model = embedding_model
        self.store_path = Path(store_path) if store_path else None
        self._vector_store = None

    def build_from_documents(self, documents: list[Document]) -> None:
        """Build a new vector store from a list of document chunks."""
        from langchain_community.vectorstores import FAISS

        if not documents:
            raise ValueError("No documents provided to build vector store")

        self._vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model,
        )

    def save(self, path: Optional[str | Path] = None) -> None:
        """Save the vector store to disk."""
        save_path = Path(path) if path else self.store_path
        if save_path is None:
            raise ValueError("No save path specified")
        if self._vector_store is None:
            raise ValueError("No vector store to save. Build or load one first.")

        save_path.mkdir(parents=True, exist_ok=True)
        self._vector_store.save_local(str(save_path))

    def load(self, path: Optional[str | Path] = None) -> bool:
        """Load a vector store from disk. Returns True if successful."""
        from langchain_community.vectorstores import FAISS

        load_path = Path(path) if path else self.store_path
        if load_path is None or not load_path.exists():
            return False

        try:
            self._vector_store = FAISS.load_local(
                str(load_path),
                self.embedding_model,
                allow_dangerous_deserialization=True,
            )
            return True
        except Exception as e:
            print(f"Warning: Failed to load vector store: {e}")
            return False

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """Search the vector store for similar documents."""
        if self._vector_store is None:
            raise ValueError("No vector store available. Build or load one first.")
        return self._vector_store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> list[tuple[Document, float]]:
        """Search with relevance scores."""
        if self._vector_store is None:
            raise ValueError("No vector store available. Build or load one first.")
        return self._vector_store.similarity_search_with_score(query, k=k)

    def as_retriever(self, search_kwargs: Optional[dict] = None):
        """Get a LangChain retriever interface."""
        if self._vector_store is None:
            raise ValueError("No vector store available. Build or load one first.")
        kwargs = search_kwargs or {"k": 4}
        return self._vector_store.as_retriever(search_kwargs=kwargs)

    @property
    def is_loaded(self) -> bool:
        return self._vector_store is not None

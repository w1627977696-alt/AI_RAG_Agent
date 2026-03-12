"""Vector store management for the AI RAG Agent."""

from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


class VectorStore:
    """Manages a persistent Chroma vector store for document retrieval."""

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag_collection",
        openai_api_key: Optional[str] = None,
    ) -> None:
        self._embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key,  # type: ignore[arg-type]
        )
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._store: Optional[Chroma] = None

    def _get_store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embeddings,
                persist_directory=self._persist_dir,
            )
        return self._store

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store and return their IDs."""
        if not documents:
            return []
        store = self._get_store()
        return store.add_documents(documents)

    def as_retriever(self, k: int = 4):
        """Return a retriever that fetches the top-k most relevant documents."""
        return self._get_store().as_retriever(search_kwargs={"k": k})

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Return top-k documents most similar to the query."""
        return self._get_store().similarity_search(query, k=k)

    def clear(self) -> None:
        """Delete all documents from the collection."""
        store = self._get_store()
        store.delete_collection()
        self._store = None

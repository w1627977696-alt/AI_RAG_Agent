"""Tests for the VectorStore module (uses mocks to avoid real API calls)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.vector_store import VectorStore


@pytest.fixture()
def mock_embeddings():
    with patch("src.vector_store.OpenAIEmbeddings") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


@pytest.fixture()
def mock_chroma(mock_embeddings):
    with patch("src.vector_store.Chroma") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_instance


def make_store(mock_embeddings):
    """Helper: create a VectorStore without triggering real I/O."""
    return VectorStore(
        embedding_model="text-embedding-3-small",
        persist_dir="/tmp/test_chroma",
        openai_api_key="sk-test",
    )


def test_add_documents_returns_ids(mock_chroma, mock_embeddings):
    """add_documents delegates to Chroma and returns IDs."""
    mock_chroma.add_documents.return_value = ["id1", "id2"]
    store = make_store(mock_embeddings)

    docs = [Document(page_content="hello"), Document(page_content="world")]
    ids = store.add_documents(docs)

    mock_chroma.add_documents.assert_called_once_with(docs)
    assert ids == ["id1", "id2"]


def test_add_documents_empty_list(mock_chroma, mock_embeddings):
    """add_documents with an empty list returns [] without calling Chroma."""
    store = make_store(mock_embeddings)
    ids = store.add_documents([])
    mock_chroma.add_documents.assert_not_called()
    assert ids == []


def test_similarity_search_delegates(mock_chroma, mock_embeddings):
    """similarity_search delegates to the underlying Chroma store."""
    expected = [Document(page_content="relevant chunk")]
    mock_chroma.similarity_search.return_value = expected

    store = make_store(mock_embeddings)
    results = store.similarity_search("test query", k=2)

    mock_chroma.similarity_search.assert_called_once_with("test query", k=2)
    assert results == expected


def test_as_retriever_delegates(mock_chroma, mock_embeddings):
    """as_retriever calls Chroma.as_retriever with correct k."""
    store = make_store(mock_embeddings)
    store.as_retriever(k=5)
    mock_chroma.as_retriever.assert_called_once_with(search_kwargs={"k": 5})


def test_clear_deletes_collection(mock_chroma, mock_embeddings):
    """clear() calls delete_collection and resets internal store."""
    store = make_store(mock_embeddings)
    # Access the store once so it's initialised
    _ = store._get_store()
    store.clear()
    mock_chroma.delete_collection.assert_called_once()
    assert store._store is None

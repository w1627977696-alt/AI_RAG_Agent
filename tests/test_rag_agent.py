"""Tests for the RAGAgent (uses mocks to avoid real API calls)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.config import Config
from src.rag_agent import RAGAgent


@pytest.fixture()
def mock_config(monkeypatch):
    """Provide a Config with a fake API key so validate() passes."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return Config()


def test_add_text(mock_config):
    """add_text loads and indexes raw text, returning the chunk count."""
    with (
        patch("src.rag_agent.DocumentLoader") as MockLoader,
        patch("src.rag_agent.VectorStore"),
        patch("src.rag_agent.ChatOpenAI"),
        patch("src.rag_agent.StrOutputParser"),
    ):
        fake_docs = [Document(page_content="chunk1"), Document(page_content="chunk2")]
        MockLoader.return_value.load_text.return_value = fake_docs

        agent = RAGAgent(config=mock_config)
        count = agent.add_text("some text")

        MockLoader.return_value.load_text.assert_called_once_with(
            "some text", metadata=None
        )
        assert count == 2


def test_add_file(mock_config):
    """add_file delegates to the loader and indexes the resulting chunks."""
    with (
        patch("src.rag_agent.DocumentLoader") as MockLoader,
        patch("src.rag_agent.VectorStore"),
        patch("src.rag_agent.ChatOpenAI"),
        patch("src.rag_agent.StrOutputParser"),
    ):
        fake_docs = [Document(page_content="c1")]
        MockLoader.return_value.load_file.return_value = fake_docs

        agent = RAGAgent(config=mock_config)
        count = agent.add_file("/some/file.txt")

        MockLoader.return_value.load_file.assert_called_once_with("/some/file.txt")
        assert count == 1


def test_query_returns_answer(mock_config):
    """query() calls the retrieval chain and returns the answer string."""
    with (
        patch("src.rag_agent.DocumentLoader"),
        patch("src.rag_agent.VectorStore"),
        patch("src.rag_agent.ChatOpenAI"),
        patch("src.rag_agent.StrOutputParser"),
    ):
        agent = RAGAgent(config=mock_config)
        # Replace the built chain with a simple mock
        fake_chain = MagicMock()
        fake_chain.invoke.return_value = "Deep learning is a subset of ML."
        agent._chain = fake_chain

        answer = agent.query("What is deep learning?")

        fake_chain.invoke.assert_called_once_with("What is deep learning?")
        assert answer == "Deep learning is a subset of ML."


def test_retrieve_uses_vector_store(mock_config):
    """retrieve() calls similarity_search on the vector store."""
    with (
        patch("src.rag_agent.DocumentLoader"),
        patch("src.rag_agent.VectorStore") as MockVStore,
        patch("src.rag_agent.ChatOpenAI"),
        patch("src.rag_agent.StrOutputParser"),
    ):
        expected = [Document(page_content="relevant")]
        MockVStore.return_value.similarity_search.return_value = expected

        agent = RAGAgent(config=mock_config)
        docs = agent.retrieve("query", k=2)

        MockVStore.return_value.similarity_search.assert_called_once_with("query", k=2)
        assert docs == expected


def test_clear_knowledge_base(mock_config):
    """clear_knowledge_base() clears the vector store and rebuilds the chain."""
    with (
        patch("src.rag_agent.DocumentLoader"),
        patch("src.rag_agent.VectorStore") as MockVStore,
        patch("src.rag_agent.ChatOpenAI"),
        patch("src.rag_agent.StrOutputParser"),
    ):
        agent = RAGAgent(config=mock_config)
        agent.clear_knowledge_base()
        MockVStore.return_value.clear.assert_called_once()


def test_validate_called_on_init(monkeypatch):
    """RAGAgent raises if the config is invalid (missing API key)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = Config()
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        RAGAgent(config=cfg)

"""Tests for the configuration module."""

import os
import pytest

from src.config import Config


def test_config_defaults():
    """Config uses sensible defaults when env vars are not set."""
    # Temporarily unset the API key to test defaults
    original = os.environ.pop("OPENAI_API_KEY", None)
    try:
        cfg = Config()
        assert cfg.openai_model == "gpt-4o-mini"
        assert cfg.embedding_model == "text-embedding-3-small"
        assert cfg.chunk_size == 1000
        assert cfg.chunk_overlap == 200
        assert cfg.retriever_k == 4
    finally:
        if original is not None:
            os.environ["OPENAI_API_KEY"] = original


def test_config_reads_env_vars(monkeypatch):
    """Config correctly reads values from environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("RETRIEVER_K", "6")

    cfg = Config()
    assert cfg.openai_api_key == "test-key-123"
    assert cfg.openai_model == "gpt-4o"
    assert cfg.chunk_size == 500
    assert cfg.retriever_k == 6


def test_config_validate_raises_when_api_key_missing(monkeypatch):
    """validate() raises ValueError when OPENAI_API_KEY is missing."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = Config()
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        cfg.validate()


def test_config_validate_passes_with_api_key(monkeypatch):
    """validate() does not raise when OPENAI_API_KEY is set."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = Config()
    cfg.validate()  # should not raise

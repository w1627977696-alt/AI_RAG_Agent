"""
Tests for the RAG module (document loader, vector store).
Tests that don't require an API key.
"""
import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag.document_loader import KnowledgeBaseLoader


class TestKnowledgeBaseLoader:
    """Tests for the KnowledgeBaseLoader."""

    def test_load_documents(self):
        """Test loading documents from knowledge base."""
        kb_path = Path(__file__).parent.parent / "data" / "knowledge_base"
        loader = KnowledgeBaseLoader(str(kb_path))
        docs = loader.load_documents()

        assert len(docs) >= 3  # We created 3 knowledge base files
        for doc in docs:
            assert doc.page_content
            assert "source" in doc.metadata

    def test_load_and_split(self):
        """Test loading and splitting documents into chunks."""
        kb_path = Path(__file__).parent.parent / "data" / "knowledge_base"
        loader = KnowledgeBaseLoader(str(kb_path), chunk_size=500, chunk_overlap=100)
        chunks = loader.load_and_split()

        assert len(chunks) > 3  # Should be more chunks than documents
        for chunk in chunks:
            assert len(chunk.page_content) <= 600  # Some tolerance for splitting

    def test_load_nonexistent_path(self):
        """Test loading from non-existent path raises error."""
        loader = KnowledgeBaseLoader("/nonexistent/path")
        with pytest.raises(FileNotFoundError):
            loader.load_documents()

    def test_chunk_metadata_preserved(self):
        """Test that metadata is preserved after splitting."""
        kb_path = Path(__file__).parent.parent / "data" / "knowledge_base"
        loader = KnowledgeBaseLoader(str(kb_path))
        chunks = loader.load_and_split()

        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "filename" in chunk.metadata

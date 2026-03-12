"""Tests for the DocumentLoader module."""

import os
import tempfile
from pathlib import Path

import pytest

from src.document_loader import DocumentLoader


@pytest.fixture()
def loader():
    return DocumentLoader(chunk_size=100, chunk_overlap=10)


@pytest.fixture()
def sample_txt_file(tmp_path):
    """Create a temporary text file with known content."""
    content = "Hello World. " * 50  # ~650 characters – will be split into chunks
    file_path = tmp_path / "test.txt"
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_load_text_returns_documents(loader):
    """load_text splits raw text into Document chunks."""
    text = "sentence " * 100  # long enough to produce multiple chunks
    docs = loader.load_text(text)
    assert len(docs) >= 1
    for doc in docs:
        assert doc.page_content  # each chunk has content


def test_load_text_attaches_metadata(loader):
    """load_text stores provided metadata on each Document."""
    docs = loader.load_text("some text content", metadata={"source": "unit-test"})
    assert docs
    assert docs[0].metadata["source"] == "unit-test"


def test_load_text_empty_metadata_default(loader):
    """load_text uses an empty dict as default metadata."""
    docs = loader.load_text("hello world")
    assert docs
    # metadata should be a dict (may include extra keys from splitter)
    assert isinstance(docs[0].metadata, dict)


def test_load_file_text(loader, sample_txt_file):
    """load_file correctly loads a .txt file and returns chunks."""
    docs = loader.load_file(sample_txt_file)
    assert len(docs) >= 1
    combined = " ".join(d.page_content for d in docs)
    assert "Hello World" in combined


def test_load_file_not_found(loader):
    """load_file raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        loader.load_file("/nonexistent/path/file.txt")


def test_load_directory_loads_txt_files(loader, tmp_path):
    """load_directory reads all .txt files inside a directory."""
    (tmp_path / "a.txt").write_text("content A " * 20, encoding="utf-8")
    (tmp_path / "b.txt").write_text("content B " * 20, encoding="utf-8")

    docs = loader.load_directory(tmp_path)
    assert len(docs) >= 1
    combined = " ".join(d.page_content for d in docs)
    assert "content A" in combined
    assert "content B" in combined


def test_load_directory_not_a_directory(loader, tmp_path):
    """load_directory raises NotADirectoryError for a file path."""
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("hello", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        loader.load_directory(file_path)


def test_load_directory_empty(loader, tmp_path):
    """load_directory returns empty list for a directory with no supported files."""
    docs = loader.load_directory(tmp_path)
    assert docs == []


def test_chunk_size_respected(loader, tmp_path):
    """Documents are split so that no chunk exceeds chunk_size (approximately)."""
    long_text = "word " * 200  # ~1000 chars, chunk_size=100
    file_path = tmp_path / "long.txt"
    file_path.write_text(long_text, encoding="utf-8")

    docs = loader.load_file(file_path)
    assert len(docs) > 1  # must have been split
    for doc in docs:
        # Allow some leeway for the splitter algorithm
        assert len(doc.page_content) <= loader.chunk_size * 2

"""Document loading utilities for the AI RAG Agent."""

import os
from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentLoader:
    """Loads and splits documents from various sources."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """Load a single file (PDF or text) and return split documents."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path), encoding="utf-8")

        docs = loader.load()
        return self._splitter.split_documents(docs)

    def load_directory(
        self,
        directory_path: Union[str, Path],
        glob: str = "**/*.*",
    ) -> List[Document]:
        """Load all supported files from a directory and return split documents."""
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")

        all_docs: List[Document] = []

        for ext, loader_cls in [("*.txt", TextLoader), ("*.pdf", PyPDFLoader)]:
            for file_path in directory_path.rglob(ext):
                try:
                    if loader_cls is TextLoader:
                        loader = loader_cls(str(file_path), encoding="utf-8")
                    else:
                        loader = loader_cls(str(file_path))
                    docs = loader.load()
                    all_docs.extend(docs)
                except (FileNotFoundError, PermissionError, UnicodeDecodeError, OSError) as exc:
                    print(f"Warning: could not load {file_path}: {exc}")

        return self._splitter.split_documents(all_docs)

    def load_text(self, text: str, metadata: dict | None = None) -> List[Document]:
        """Load raw text directly and return split documents."""
        doc = Document(page_content=text, metadata=metadata or {})
        return self._splitter.split_documents([doc])

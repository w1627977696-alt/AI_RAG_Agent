"""
Document Loader - RAG Module
Loads and chunks documents from the knowledge base for vector storage.
"""
from pathlib import Path
from typing import Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class KnowledgeBaseLoader:
    """Loads documents from the knowledge base directory and splits them into chunks."""

    SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf", ".docx"}

    def __init__(
        self,
        knowledge_base_path: str | Path,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ):
        self.kb_path = Path(knowledge_base_path)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", "。", ".", " "],
        )

    def load_documents(self) -> list[Document]:
        """Load all supported documents from the knowledge base directory."""
        if not self.kb_path.exists():
            raise FileNotFoundError(f"Knowledge base path not found: {self.kb_path}")

        documents = []
        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in self.kb_path.glob(f"**/*{ext}"):
                docs = self._load_single_file(file_path)
                documents.extend(docs)

        return documents

    def load_and_split(self) -> list[Document]:
        """Load documents and split them into chunks for vector storage."""
        documents = self.load_documents()
        chunks = self.text_splitter.split_documents(documents)
        return chunks

    def _load_single_file(self, file_path: Path) -> list[Document]:
        """Load a single file and return as Document objects."""
        try:
            if file_path.suffix in (".md", ".txt"):
                return self._load_text_file(file_path)
            else:
                # PDF and DOCX are listed as supported but not yet implemented;
                # attempt plain-text read as a best-effort fallback.
                return self._load_text_file(file_path)
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            return []

    def _load_text_file(self, file_path: Path) -> list[Document]:
        """Load a text/markdown file."""
        content = file_path.read_text(encoding="utf-8")
        metadata = {
            "source": str(file_path),
            "filename": file_path.name,
            "file_type": file_path.suffix,
        }
        return [Document(page_content=content, metadata=metadata)]

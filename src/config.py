"""Configuration module for the AI RAG Agent."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Configuration settings for the RAG Agent."""

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    chroma_persist_dir: str = field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    )
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200"))
    )
    retriever_k: int = field(
        default_factory=lambda: int(os.getenv("RETRIEVER_K", "4"))
    )

    def validate(self) -> None:
        """Validate that required configuration values are set."""
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Please copy .env.example to .env and fill in your API key."
            )

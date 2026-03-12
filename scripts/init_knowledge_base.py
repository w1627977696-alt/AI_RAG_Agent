"""
Initialize the RAG knowledge base.
Loads documents from data/knowledge_base, creates embeddings, and saves the vector store.
Requires OPENAI_API_KEY to be configured.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import (
    OPENAI_API_KEY, EMBEDDING_MODEL_NAME,
    KNOWLEDGE_BASE_PATH, VECTOR_STORE_PATH,
)
from src.rag.document_loader import KnowledgeBaseLoader
from src.rag.vector_store import VectorStoreManager


def init_knowledge_base():
    """Initialize the knowledge base vector store."""
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
        print("❌ Error: OPENAI_API_KEY not configured.")
        print("   Please set OPENAI_API_KEY in .env file.")
        return False

    print(f"📂 Loading documents from: {KNOWLEDGE_BASE_PATH}")

    loader = KnowledgeBaseLoader(str(KNOWLEDGE_BASE_PATH))
    chunks = loader.load_and_split()
    print(f"📄 Loaded and split into {len(chunks)} chunks")

    if not chunks:
        print("❌ No documents found in knowledge base directory.")
        return False

    print(f"🔧 Creating embeddings with model: {EMBEDDING_MODEL_NAME}")

    from langchain_openai import OpenAIEmbeddings
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

    store_manager = VectorStoreManager(
        embedding_model=embedding_model,
        store_path=str(VECTOR_STORE_PATH),
    )

    print("🏗️ Building vector store...")
    store_manager.build_from_documents(chunks)

    print(f"💾 Saving vector store to: {VECTOR_STORE_PATH}")
    store_manager.save()

    print("✅ Knowledge base initialized successfully!")
    return True


if __name__ == "__main__":
    init_knowledge_base()

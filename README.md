# AI RAG Agent

A Python-based **Retrieval-Augmented Generation (RAG) Agent** that lets you load documents into a persistent vector knowledge base and ask questions answered by an LLM grounded in that context.

## Features

- Load **text files**, **PDF files**, or **raw text** into the knowledge base
- Persistent vector storage via **ChromaDB**
- OpenAI-powered embeddings and chat completions
- Clean **LCEL** (LangChain Expression Language) pipeline
- Interactive **CLI** for exploration
- Fully configurable via `.env`

## Architecture

```
User Query
    │
    ▼
DocumentLoader ──► VectorStore (ChromaDB)
                        │
                   Retriever (top-k)
                        │
                        ▼
              Context + Question ──► ChatOpenAI ──► Answer
```

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/w1627977696-alt/AI_RAG_Agent.git
cd AI_RAG_Agent
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 3. Run the interactive CLI

```bash
python main.py
```

**CLI commands:**

| Command | Description |
|---|---|
| `add-file <path>` | Load a file (`.txt` or `.pdf`) into the knowledge base |
| `add-dir <path>` | Load all files in a directory |
| `add-text` | Enter raw text directly |
| `clear` | Remove all documents from the knowledge base |
| `quit` | Exit |
| *anything else* | Ask a question |

**Example session:**

```
>>> add-file data/sample.txt
Added 5 chunk(s) from 'data/sample.txt'.

>>> What is RAG?
Answer: RAG (Retrieval-Augmented Generation) is an AI framework that retrieves
relevant documents from a knowledge base and uses them as context when generating
answers. This helps language models produce more accurate and up-to-date responses
by grounding them in real data.
```

### 4. Use as a library

```python
from src.config import Config
from src.rag_agent import RAGAgent

config = Config()          # reads from .env / environment
agent = RAGAgent(config)

# Ingest knowledge
agent.add_file("docs/report.pdf")
agent.add_text("The company was founded in 2010 in San Francisco.")

# Query
answer = agent.query("When was the company founded?")
print(answer)
```

## Project Structure

```
AI_RAG_Agent/
├── src/
│   ├── config.py           # Configuration (env vars)
│   ├── document_loader.py  # File/text ingestion & splitting
│   ├── vector_store.py     # ChromaDB wrapper
│   └── rag_agent.py        # Core RAG agent (LCEL pipeline)
├── data/
│   └── sample.txt          # Example document
├── tests/
│   ├── test_config.py
│   ├── test_document_loader.py
│   ├── test_vector_store.py
│   └── test_rag_agent.py
├── main.py                 # Interactive CLI
├── requirements.txt
└── .env.example
```

## Configuration

All settings can be overridden via environment variables (copy `.env.example` → `.env`):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model to use |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Vector DB persistence path |
| `CHUNK_SIZE` | `1000` | Document chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `RETRIEVER_K` | `4` | Number of chunks to retrieve per query |

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Requirements

- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys)

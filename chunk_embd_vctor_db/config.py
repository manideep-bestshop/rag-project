"""
Central configuration for the RAG pipeline.
Everything here is overridable via environment variables (see .env).
"""

import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Document loading -----------------------------------------------
CORPUS_DIR = os.getenv(
    "CORPUS_DIR",
    "C:/RAG/corpus",
)
# Keep this small on purpose -- this is a learning project, not a bulk ingest job.
MAX_PDF_FILES = int(os.getenv("MAX_PDF_FILES", "5"))

# --- 2. Chunking ---------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# --- 3. Embeddings (local Ollama) ----------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

# --- 4. Local LLM used only for the final "ask a question" demo ---------
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
TOP_K = int(os.getenv("TOP_K", "4"))

# --- 5. Postgres + pgvector ------------------------------------------------
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "manideep@123")
POSTGRES_DB = os.getenv("POSTGRES_DB", "aventro_rag_database")
TABLE_NAME = os.getenv("TABLE_NAME", "document_chunks")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

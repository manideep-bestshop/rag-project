"""
Central configuration for the Hybrid RAG pipeline (BM25 + pgvector).
Everything here is overridable via environment variables (see .env).

This project is a sibling of ../chunk_embd_vctor_db and intentionally reuses
its corpus, embedding model, LLM, and Postgres/pgvector setup -- the only
new moving part is the BM25 sparse index + the fusion step that combines it
with dense vector search.
"""

import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- 1. Document loading -------------------------------------------------
# Reuses the same Aventro Motors PDF corpus as chunk_embd_vctor_db.
CORPUS_DIR = os.getenv(
    "CORPUS_DIR",
    "/Users/srinivaskotipalli/mai-practitioner/RAG/chunk_embd_vctor_db/aventro/pdf",
)
# Keep this small on purpose -- this is a learning project, not a bulk ingest job.
MAX_PDF_FILES = int(os.getenv("MAX_PDF_FILES", "5"))

# --- 2. Chunking ----------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# --- 3. Dense embeddings (local Ollama) -----------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

# --- 4. Local LLM used only for the final "ask a question" demo ----------
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")

# --- 5. Postgres + pgvector (the "existing vector database") -------------
# Deliberately a *separate* database/table from chunk_embd_vctor_db so the
# two projects never collide on disk.
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "aventro_hybrid_rag_db")
TABLE_NAME = os.getenv("TABLE_NAME", "hybrid_document_chunks")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

# --- 6. BM25 sparse index (new) -------------------------------------------
# Persisted as a local pickle file -- no server, no external service.
# This *is* the "local vector database" counterpart for sparse/lexical search,
# the same way pgvector is the local vector database for dense search.
BM25_INDEX_PATH = os.getenv(
    "BM25_INDEX_PATH", os.path.join(BASE_DIR, "bm25_store", "bm25_index.pkl")
)
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))   # term-frequency saturation
BM25_B = float(os.getenv("BM25_B", "0.75"))    # document-length normalization

# --- 7. Hybrid retrieval + fusion -----------------------------------------
# How many candidates each retriever contributes BEFORE fusion.
TOP_K_DENSE = int(os.getenv("TOP_K_DENSE", "10"))
TOP_K_SPARSE = int(os.getenv("TOP_K_SPARSE", "10"))
# How many fused results are kept AFTER fusion, to send to the LLM.
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "4"))

# Fusion method: "rrf" (Reciprocal Rank Fusion, default) or "weighted"
# (normalized weighted-sum of dense similarity + BM25 score).
FUSION_METHOD = os.getenv("FUSION_METHOD", "rrf")
RRF_K = int(os.getenv("RRF_K", "60"))          # standard RRF damping constant
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.5"))   # only used by "weighted"
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.5"))  # only used by "weighted"

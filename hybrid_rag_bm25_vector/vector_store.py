"""
STEP 3b: Dense vector storage -- local PostgreSQL + pgvector.

This is the SAME local vector database approach as ../chunk_embd_vctor_db
(psycopg3 + pgvector, HNSW cosine index) -- reused as-is here, in its own
database/table so the two projects never collide. In the hybrid pipeline
this module answers only ONE of the two retrieval questions:

    "which chunks are closest in MEANING to the query?" (dense / semantic)

The other question -- "which chunks share the query's exact WORDS?"
(sparse / lexical) -- is answered by bm25_index.py. hybrid_retriever.py
asks both and fuses the two ranked lists.

Schema (table: hybrid_document_chunks):
    id          UUID PRIMARY KEY   -- same as chunk_metadata["chunk_id"]
    content     TEXT               -- the chunk text
    embedding   VECTOR(768)        -- nomic-embed-text output
    source_file TEXT               -- which PDF this came from
    chunk_index INT                -- position within that PDF's chunks
    metadata    JSONB              -- full chunk metadata (page, chunk_id, etc.)
    created_at  TIMESTAMPTZ        -- row insert time
"""

import json
from typing import List, Optional

import psycopg
from pgvector.psycopg import register_vector

import config


def _connect(dbname: Optional[str] = None) -> psycopg.Connection:
    return psycopg.connect(
        host=config.POSTGRES_HOST,
        port=config.POSTGRES_PORT,
        user=config.POSTGRES_USER,
        password=config.POSTGRES_PASSWORD,
        dbname=dbname or config.POSTGRES_DB,
        autocommit=True,
        connect_timeout=10,
    )


def ensure_database() -> None:
    conn = _connect(dbname="postgres")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (config.POSTGRES_DB,))
            if cur.fetchone() is None:
                cur.execute(f'CREATE DATABASE "{config.POSTGRES_DB}"')
                print(f"Created database '{config.POSTGRES_DB}'")
    finally:
        conn.close()


def get_connection() -> psycopg.Connection:
    conn = _connect()
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_schema() -> None:
    ensure_database()
    conn = get_connection()
    try:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.TABLE_NAME} (
                id UUID PRIMARY KEY,
                content TEXT NOT NULL,
                embedding VECTOR({config.EMBEDDING_DIMENSIONS}) NOT NULL,
                source_file TEXT,
                chunk_index INT,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT now()
            )
            """
        )
        conn.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {config.TABLE_NAME}_embedding_hnsw_idx
            ON {config.TABLE_NAME} USING hnsw (embedding vector_cosine_ops)
            """
        )
    finally:
        conn.close()
    print(f"Schema ready: table '{config.TABLE_NAME}' (db '{config.POSTGRES_DB}') with HNSW cosine index")


def insert_chunks(records: List[dict]) -> None:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            for r in records:
                meta = r["metadata"]
                cur.execute(
                    f"""
                    INSERT INTO {config.TABLE_NAME}
                        (id, content, embedding, source_file, chunk_index, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (
                        meta["chunk_id"],
                        r["content"],
                        r["embedding"],
                        meta.get("source_file"),
                        meta.get("chunk_index"),
                        json.dumps(meta),
                    ),
                )
    finally:
        conn.close()
    print(f"Inserted {len(records)} chunk(s) into '{config.TABLE_NAME}'")


def dense_search(query_embedding: List[float], k: Optional[int] = None) -> List[dict]:
    """Return the k chunks whose embeddings are closest to query_embedding,
    using cosine distance (the `<=>` operator from pgvector). This is the
    "dense" retriever half of the hybrid pipeline."""
    k = k or config.TOP_K_DENSE
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT metadata->>'chunk_id' AS chunk_id, content, source_file,
                       chunk_index, metadata, embedding <=> %s::vector AS distance
                FROM {config.TABLE_NAME}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_embedding, query_embedding, k),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    results = []
    for chunk_id, content, source_file, chunk_index, metadata, distance in rows:
        results.append(
            {
                "chunk_id": chunk_id,
                "content": content,
                "source_file": source_file,
                "chunk_index": chunk_index,
                "metadata": metadata,
                "distance": distance,
                "dense_score": 1 - distance,  # cosine distance -> similarity
            }
        )
    return results


# Backwards-compatible alias (matches the sibling project's function name).
similarity_search = dense_search


def count_chunks() -> int:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {config.TABLE_NAME}")
            return cur.fetchone()[0]
    finally:
        conn.close()


def clear_table() -> None:
    conn = get_connection()
    try:
        conn.execute(f"TRUNCATE TABLE {config.TABLE_NAME}")
    finally:
        conn.close()
    print(f"Cleared table '{config.TABLE_NAME}'")


if __name__ == "__main__":
    init_schema()
    print(f"Row count: {count_chunks()}")

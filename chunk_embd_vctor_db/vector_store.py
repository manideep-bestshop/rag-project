"""
STEP 4: Vector storage -- local PostgreSQL + pgvector.

Built directly on psycopg3 + the `pgvector` adapter (same combination used
in postgres_embeddings_rag_classroom.ipynb), rather than langchain_chroma
(used as an example in production-course-main-code/vector_stores.py), because
the goal here is a real local Postgres database, not an in-memory/on-disk
Chroma collection.

Connection details come from `config.py` / `.env`:
    POSTGRES_HOST=localhost
    POSTGRES_PORT=5432
    POSTGRES_USER=postgres
    POSTGRES_PASSWORD=postgres
    POSTGRES_DB=aventro_rag_db

Note on the password: your local Postgres (checked in pg_hba.conf) uses
`trust` authentication for localhost/127.0.0.1, meaning the server does not
actually check the password for local connections -- any value works. If you
later connect to a *different* Postgres instance that enforces a real
password, just set POSTGRES_PASSWORD (or the full DATABASE_URL) in `.env`.

Schema (table: document_chunks):
    id          UUID PRIMARY KEY   -- same as chunk_metadata["chunk_id"]
    content     TEXT               -- the chunk text
    embedding   VECTOR(768)        -- nomic-embed-text output
    source_file TEXT               -- which PDF this came from
    chunk_index INT                -- position within that PDF's chunks
    metadata    JSONB              -- full chunk metadata (page, chunk_id, etc.)
    created_at  TIMESTAMPTZ        -- row insert time

An HNSW index with cosine distance is used so `similarity_search` can run
fast approximate nearest-neighbor search even as the table grows.
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
    """Create the target database if it doesn't exist yet (connects to the
    default 'postgres' database to do so, since you can't CREATE DATABASE
    from inside the database you're creating)."""
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
    """Connect to the project database, enable pgvector, and register the
    numpy<->VECTOR adapter so we can pass Python lists/arrays directly."""
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


def similarity_search(query_embedding: List[float], k: Optional[int] = None) -> List[dict]:
    """Return the k chunks whose embeddings are closest to query_embedding,
    using cosine distance (the `<=>` operator from pgvector)."""
    k = k or config.TOP_K
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT content, source_file, chunk_index, metadata,
                       embedding <=> %s::vector AS distance
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
    for content, source_file, chunk_index, metadata, distance in rows:
        results.append(
            {
                "content": content,
                "source_file": source_file,
                "chunk_index": chunk_index,
                "metadata": metadata,
                "distance": distance,
                "similarity": 1 - distance,  # cosine distance -> similarity
            }
        )
    return results


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

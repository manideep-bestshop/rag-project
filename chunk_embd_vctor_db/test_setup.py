"""
Preflight checks -- run this before `ingest.py` to confirm:
  1. The PDF corpus directory is reachable and has files
  2. Ollama is running locally with the embedding + LLM models pulled
  3. Postgres is reachable with the configured credentials

Usage:
    python test_setup.py
"""

from pathlib import Path

import requests
import psycopg

import config


def check_corpus() -> bool:
    pdfs = list(Path(config.CORPUS_DIR).glob("*.pdf"))
    ok = len(pdfs) > 0
    status = "OK" if ok else "FAIL"
    print(f"[{status}] Corpus dir has {len(pdfs)} PDF file(s): {config.CORPUS_DIR}")
    if ok:
        print(f"       (will use the first {config.MAX_PDF_FILES} alphabetically)")
    return ok


def check_ollama() -> bool:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
        tags = [m["name"] for m in resp.json().get("models", [])]
    except Exception as e:
        print(f"[FAIL] Could not reach Ollama at localhost:11434 ({e})")
        print("       Start it with: ollama serve")
        return False

    has_embed = any(config.EMBEDDING_MODEL in t for t in tags)
    has_llm = any(config.LLM_MODEL in t for t in tags)

    print(f"[{'OK' if has_embed else 'FAIL'}] Embedding model '{config.EMBEDDING_MODEL}' available")
    if not has_embed:
        print(f"       Pull it with: ollama pull {config.EMBEDDING_MODEL}")

    print(f"[{'OK' if has_llm else 'FAIL'}] LLM model '{config.LLM_MODEL}' available")
    if not has_llm:
        print(f"       Pull it with: ollama pull {config.LLM_MODEL}")

    return has_embed and has_llm


def check_postgres() -> bool:
    try:
        conn = psycopg.connect(
            host=config.POSTGRES_HOST,
            port=config.POSTGRES_PORT,
            user=config.POSTGRES_USER,
            password=config.POSTGRES_PASSWORD,
            dbname="postgres",
            connect_timeout=5,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
            has_vector = cur.fetchone() is not None
        conn.close()
    except Exception as e:
        print(f"[FAIL] Could not connect to Postgres at {config.POSTGRES_HOST}:{config.POSTGRES_PORT} ({e})")
        return False

    print(f"[OK] Connected to Postgres at {config.POSTGRES_HOST}:{config.POSTGRES_PORT}")
    print(f"[{'OK' if has_vector else 'FAIL'}] pgvector extension available on this server")
    if not has_vector:
        print("       Install it (e.g. `brew install pgvector` then restart Postgres)")
    return has_vector


if __name__ == "__main__":
    print("Running preflight checks...\n")
    results = [check_corpus(), check_ollama(), check_postgres()]
    print()
    if all(results):
        print("All checks passed -- you're ready to run: python ingest.py")
    else:
        print("Some checks failed -- fix the items above before running ingest.py")

"""
Orchestrates the full ingestion pipeline:

    PDF files --> Document Loader --> Text Chunker (+metadata)
              --> Embeddings (Ollama) --> Postgres + pgvector

Usage:
    python ingest.py            # ingest (append to existing table)
    python ingest.py --reset    # wipe the table first, then ingest fresh
"""

import sys

from document_loader import load_pdfs
from text_chunker import chunk_documents
from embedder import embed_chunks
from vector_store import init_schema, insert_chunks, clear_table, count_chunks
import config


def run(reset: bool = False) -> None:
    print("=" * 70)
    print("STEP 1/4: Loading PDF documents")
    print("=" * 70)
    documents = load_pdfs()

    print("\n" + "=" * 70)
    print("STEP 2/4: Chunking + building metadata")
    print("=" * 70)
    chunks = chunk_documents(documents)

    print("\n" + "=" * 70)
    print("STEP 3/4: Generating embeddings (Ollama)")
    print("=" * 70)
    records = embed_chunks(chunks)

    print("\n" + "=" * 70)
    print("STEP 4/4: Storing in Postgres + pgvector")
    print("=" * 70)
    init_schema()
    if reset:
        clear_table()
    insert_chunks(records)

    print("\n" + "=" * 70)
    print("Ingestion complete")
    print("=" * 70)
    print(f"  PDF files processed : {min(config.MAX_PDF_FILES, len({d.metadata['source_file'] for d in documents}))}")
    print(f"  Chunks stored       : {len(records)}")
    print(f"  Table               : {config.TABLE_NAME} (db: {config.POSTGRES_DB})")
    print(f"  Total rows in table : {count_chunks()}")


if __name__ == "__main__":
    run(reset="--reset" in sys.argv)

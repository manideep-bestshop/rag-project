"""
Orchestrates the full hybrid ingestion pipeline:

    PDF files --> Document Loader --> Text Chunker (+metadata)
              --> Embeddings (Ollama) --> Postgres + pgvector  (dense index)
              --> BM25 tokenizer      --> local pickle file    (sparse index)

Both indexes are built from the SAME chunk list, so every chunk_id exists in
both places and hybrid_retriever.py can fuse results by that shared key.

Usage:
    python ingest.py            # ingest (append to existing table/index)
    python ingest.py --reset    # wipe Postgres table + BM25 index first
"""

import sys

from document_loader import load_pdfs
from text_chunker import chunk_documents
from embedder import embed_chunks
from vector_store import init_schema, insert_chunks, clear_table, count_chunks
from bm25_index import BM25Index
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
    print("STEP 3/4: Dense embeddings (Ollama) -> Postgres/pgvector")
    print("=" * 70)
    records = embed_chunks(chunks)
    init_schema()
    if reset:
        clear_table()
    insert_chunks(records)

    print("\n" + "=" * 70)
    print("STEP 4/4: Sparse index (BM25) -> local pickle file")
    print("=" * 70)
    bm25 = BM25Index.build(chunks)
    bm25.save()

    print("\n" + "=" * 70)
    print("Hybrid ingestion complete")
    print("=" * 70)
    print(f"  PDF files processed : {min(config.MAX_PDF_FILES, len({d.metadata['source_file'] for d in documents}))}")
    print(f"  Chunks stored       : {len(records)}")
    print(f"  Dense store         : {config.TABLE_NAME} (db: {config.POSTGRES_DB}), rows = {count_chunks()}")
    print(f"  Sparse store        : {config.BM25_INDEX_PATH}")


if __name__ == "__main__":
    run(reset="--reset" in sys.argv)

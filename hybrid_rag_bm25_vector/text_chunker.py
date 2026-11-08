"""
STEP 2: Chunking + metadata.

Splits each loaded PDF page into smaller, overlapping chunks using
RecursiveCharacterTextSplitter, then enriches every chunk with metadata
that explains exactly where it came from and how it was built.

This is identical to ../chunk_embd_vctor_db/text_chunker.py on purpose: the
chunk IS the unit of retrieval for BOTH retrievers in this project. The same
`chunk_id` ends up as:
  - a row in Postgres (dense/vector search), and
  - a document in the BM25 index (sparse/keyword search),
so fusion later can simply join on `chunk_id`.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def chunk_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    pages_by_file: dict[str, List[Document]] = {}
    for doc in documents:
        key = doc.metadata.get("source_file", "unknown")
        pages_by_file.setdefault(key, []).append(doc)

    all_chunks: List[Document] = []
    for source_file, pages in pages_by_file.items():
        file_chunks = splitter.split_documents(pages)
        total = len(file_chunks)
        now = datetime.now(timezone.utc).isoformat()

        for idx, chunk in enumerate(file_chunks):
            chunk.metadata.update(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_index": idx,
                    "total_chunks": total,
                    "chunk_char_count": len(chunk.page_content),
                    "chunk_size_setting": chunk_size,
                    "chunk_overlap_setting": chunk_overlap,
                    "ingested_at": now,
                }
            )
            all_chunks.append(chunk)

        print(f"  {source_file:<45} -> {total} chunk(s)")

    print(f"\nChunked {len(documents)} page-document(s) into {len(all_chunks)} chunk(s) total")
    return all_chunks


if __name__ == "__main__":
    from document_loader import load_pdfs

    docs = load_pdfs()
    chunks = chunk_documents(docs)
    print("\n--- Sample chunk ---")
    print("metadata:", chunks[0].metadata)
    print("content:", chunks[0].page_content[:300].replace("\n", " "), "...")

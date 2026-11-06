"""
STEP 2: Chunking + metadata.

Splits each loaded PDF page into smaller, overlapping chunks using
RecursiveCharacterTextSplitter (same splitter used in
production-course-main-code/text_splitters.py), then enriches every chunk
with metadata that explains exactly where it came from and how it was built.

Why chunk at all?
  Embedding models and LLM context windows work best on small, focused
  pieces of text. A whole PDF page mixes too many ideas into one vector,
  which hurts retrieval precision. Smaller chunks -> sharper embeddings ->
  more relevant search results.

Why overlap?
  Without overlap, a sentence that spans a chunk boundary gets cut in half
  and loses meaning in both halves. A small overlap (here: 120 of 800 chars,
  ~15%) repeats the tail of one chunk at the start of the next so context
  isn't lost at the seam.

Metadata added per chunk (beyond what PyPDFLoader already provides):
  - chunk_id             unique UUID for this chunk (primary key in Postgres)
  - chunk_index          position of this chunk within its *source file*
  - total_chunks         how many chunks that source file produced
  - chunk_char_count     length of the chunk text (useful for debugging/QA)
  - chunk_size_setting   the chunk_size the splitter was configured with
  - chunk_overlap_setting the chunk_overlap the splitter was configured with
  - ingested_at          UTC timestamp of when this chunk was processed

Combined with what PyPDFLoader already sets (`source_file`, `page`, `source`),
every chunk is fully traceable back to an exact PDF, page, and position.
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
        # Tries paragraph breaks first, then lines, then sentences, then
        # words -- only falling back to a hard character cut as a last resort.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Group pages by source file so chunk_index/total_chunks describe
    # position within a *document*, not across the whole unrelated batch.
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

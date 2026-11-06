"""
STEP 1: Document loading.

Reads a handful of PDF files from the Aventro Motors corpus and turns each
page into a LangChain `Document` (page_content + metadata). This mirrors the
`pdf_loader()` pattern in production-course-main-code/document_loaders.py,
but loops over a directory instead of a single file, and tags each page with
which source file it came from so later stages (chunking, storage) can keep
that lineage.
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

import config


def load_pdfs(corpus_dir: Optional[str] = None, max_files: Optional[int] = None) -> List[Document]:
    """Load the first `max_files` PDFs from `corpus_dir`.

    Returns one Document per PDF *page* (PyPDFLoader's default behaviour),
    each already carrying `source_file` and `page` metadata.
    """
    corpus_dir = corpus_dir or config.CORPUS_DIR
    max_files = config.MAX_PDF_FILES if max_files is None else max_files

    pdf_paths = sorted(Path(corpus_dir).glob("*.pdf"))[:max_files]
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {corpus_dir}")

    all_documents: List[Document] = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages = loader.load()

        for page in pages:
            # PyPDFLoader already sets metadata["source"] to the full file path
            # and metadata["page"] to the page index. We add a short, friendly
            # filename so downstream metadata stays readable.
            page.metadata["source_file"] = pdf_path.name

        all_documents.extend(pages)
        print(f"  loaded {pdf_path.name:<45} -> {len(pages)} page(s)")

    print(f"\nTotal: {len(pdf_paths)} PDF file(s), {len(all_documents)} page-document(s)")
    return all_documents


if __name__ == "__main__":
    docs = load_pdfs()
    print("\n--- Sample page ---")
    print("metadata:", docs[0].metadata)
    print("content preview:", docs[0].page_content[:300].replace("\n", " "), "...")

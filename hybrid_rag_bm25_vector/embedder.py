"""
STEP 3a: Dense embeddings (local, via Ollama).

Uses `nomic-embed-text` running on your local Ollama server to turn each
chunk's text into a 768-dimension vector. This is the "dense" half of
hybrid retrieval -- it captures *meaning* (semantic similarity) rather than
exact words, so a query like "car won't start" can match a chunk that says
"engine fails to crank" even though they share almost no vocabulary.

The complementary "sparse" half (BM25, exact keyword matching) lives in
bm25_index.py.
"""

from typing import List

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

import config

embeddings_model = OllamaEmbeddings(model=config.EMBEDDING_MODEL)


def embed_chunks(chunks: List[Document]) -> List[dict]:
    """Embed a batch of chunks. Returns a list of records ready for storage:
    {"content": str, "embedding": list[float], "metadata": dict}
    """
    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings_model.embed_documents(texts)

    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append(
            {
                "content": chunk.page_content,
                "embedding": vector,
                "metadata": chunk.metadata,
            }
        )

    dims = len(vectors[0]) if vectors else 0
    print(f"Embedded {len(records)} chunk(s) with '{config.EMBEDDING_MODEL}' ({dims} dimensions)")
    return records


def embed_query(text: str) -> List[float]:
    """Embed a single piece of text (e.g. a user's question) for similarity search."""
    return embeddings_model.embed_query(text)


if __name__ == "__main__":
    from document_loader import load_pdfs
    from text_chunker import chunk_documents

    docs = load_pdfs()
    chunks = chunk_documents(docs)
    records = embed_chunks(chunks[:3])
    print("\n--- Sample record ---")
    print("metadata:", records[0]["metadata"])
    print("embedding (first 5 values):", records[0]["embedding"][:5])

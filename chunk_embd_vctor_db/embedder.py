"""
STEP 3: Embeddings (local, via Ollama).

Uses `nomic-embed-text` running on your local Ollama server to turn each
chunk's text into a 768-dimension vector -- the same model and the same
`langchain_ollama.OllamaEmbeddings` wrapper used in
production-course-main-code/embeddings_deep.py.

Each embedding record keeps the chunk's full metadata alongside its vector,
so nothing gets lost between "chunk" and "row in the vector database."
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
    records = embed_chunks(chunks[:3])  # just a quick smoke test
    print("\n--- Sample record ---")
    print("metadata:", records[0]["metadata"])
    print("embedding (first 5 values):", records[0]["embedding"][:5])

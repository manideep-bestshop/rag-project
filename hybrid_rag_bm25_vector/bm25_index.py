"""
STEP 3c: Sparse lexical index -- BM25 (new in this project).

BM25 ("Best Match 25") is the classic keyword-ranking algorithm behind most
search engines before/alongside embeddings (it's what Elasticsearch/Lucene
use by default). It scores a chunk against a query using:

  - Term frequency: how many times a query word appears in the chunk
    (saturating -- the 10th occurrence barely matters more than the 5th;
    controlled by `config.BM25_K1`).
  - Inverse document frequency: rare words (e.g. "ACC", "pgvector") score
    higher than common words ("the", "car") because they're more
    discriminating.
  - Document-length normalization: a long chunk naturally contains more
    words, so raw counts are normalized against chunk length (controlled by
    `config.BM25_B`).

Unlike the dense embeddings in embedder.py, BM25 has NO notion of meaning --
it cannot tell that "won't start" relates to "engine fails to crank". What it
is extremely good at is exact/near-exact term matches: model names, part
numbers, acronyms (ACC, ABS, SUV), error codes -- the kind of query where a
semantic embedding can blur precision in favor of "generally similar" text.
That's precisely the gap hybrid retrieval closes.

This module is the "local vector database" counterpart for sparse search:
instead of a server, the whole index is a small Python object (tokenized
corpus + term statistics) pickled to disk at `config.BM25_INDEX_PATH`. No
service to run, no port to open -- just a file, built once in ingest.py and
loaded at query time in hybrid_retriever.py / ask.py.
"""

import os
import pickle
import re
from typing import List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

import config

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase + alphanumeric word tokenizer.

    Deliberately simple: BM25's job is exact-term matching, so the only
    thing that matters is that the SAME tokenizer runs at index time and at
    query time. No stemming/stopword removal -- keeping raw tokens makes it
    easy to explain scores by hand (see README's worked example).
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Thin wrapper around rank_bm25.BM25Okapi that also remembers, per
    document, the chunk_id/content/metadata needed to turn a BM25 hit back
    into a real chunk -- rank_bm25 itself only knows token lists and scores.
    """

    def __init__(self, chunk_ids: List[str], contents: List[str], metadatas: List[dict]):
        self.chunk_ids = chunk_ids
        self.contents = contents
        self.metadatas = metadatas
        tokenized_corpus = [tokenize(text) for text in contents]
        self._bm25 = BM25Okapi(tokenized_corpus, k1=config.BM25_K1, b=config.BM25_B)

    @classmethod
    def build(cls, chunks: List[Document]) -> "BM25Index":
        chunk_ids = [c.metadata["chunk_id"] for c in chunks]
        contents = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        index = cls(chunk_ids, contents, metadatas)
        print(f"Built BM25 index over {len(chunks)} chunk(s) (k1={config.BM25_K1}, b={config.BM25_B})")
        return index

    def search(self, query: str, k: Optional[int] = None) -> List[dict]:
        """Return the top-k chunks ranked by BM25 score for `query`."""
        k = k or config.TOP_K_SPARSE
        scores = self._bm25.get_scores(tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = []
        for i in ranked_idx:
            if scores[i] <= 0:
                continue  # no shared terms at all -- not a real match
            results.append(
                {
                    "chunk_id": self.chunk_ids[i],
                    "content": self.contents[i],
                    "source_file": self.metadatas[i].get("source_file"),
                    "chunk_index": self.metadatas[i].get("chunk_index"),
                    "metadata": self.metadatas[i],
                    "sparse_score": float(scores[i]),
                }
            )
        return results

    def save(self, path: Optional[str] = None) -> None:
        path = path or config.BM25_INDEX_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"chunk_ids": self.chunk_ids, "contents": self.contents, "metadatas": self.metadatas},
                f,
            )
        print(f"Saved BM25 index -> {path}")

    @classmethod
    def load(cls, path: Optional[str] = None) -> "BM25Index":
        path = path or config.BM25_INDEX_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No BM25 index found at {path}. Run `python ingest.py` first."
            )
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["chunk_ids"], data["contents"], data["metadatas"])


if __name__ == "__main__":
    from document_loader import load_pdfs
    from text_chunker import chunk_documents

    docs = load_pdfs()
    chunks = chunk_documents(docs)
    index = BM25Index.build(chunks)
    print("\n--- Sample BM25 search: 'ACC adaptive cruise control' ---")
    for r in index.search("ACC adaptive cruise control", k=3):
        preview = r["content"][:120].replace("\n", " ")
        print(f"  score={r['sparse_score']:.3f} | {r['source_file']} | {preview}...")

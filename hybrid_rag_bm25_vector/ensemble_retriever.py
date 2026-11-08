"""
STEP 4, alternative engine: LangChain's built-in EnsembleRetriever.

hybrid_retriever.py (the default engine) is a hand-rolled RRF/weighted
fusion -- written out explicitly so the math is visible. This module wires
the SAME two local indexes (the BM25 pickle + the Postgres/pgvector table)
into LangChain's own EnsembleRetriever, which performs the identical
weighted Reciprocal Rank Fusion internally:

    score(doc) = Σ  weight_i / (c + rank_in_retriever_i)

...with the SAME damping constant (`c`, default 60) this project already
uses as `config.RRF_K`. The difference is purely architectural: this path
goes through LangChain's standard `BaseRetriever` interface, so it composes
with any other LangChain retriever/chain/tool (a reranker, a
`RunnableSequence`, LangGraph, etc.) for free -- at the cost of being a bit
more "magic" than reading the fusion loop yourself in hybrid_retriever.py.

Use `python ask.py --engine ensemble "..."` to run this path.

Note on corpus size: like bm25_index.py, LangChain's BM25Retriever keeps its
whole tokenized corpus in memory (no server) -- `build_ensemble_retriever()`
rebuilds it from the same persisted `bm25_store/bm25_index.pkl` each call,
exactly as `BM25Index.load()` does for the custom engine. The dense side
goes through `PgVectorRetriever` below, a ~15-line adapter that lets
LangChain's EnsembleRetriever talk to this project's existing
psycopg3/pgvector store without that store having to be a LangChain
`VectorStore` subclass.

Why the resolver below instead of a plain top-level import: LangChain has
relocated `EnsembleRetriever`/`BM25Retriever` more than once across major
versions (`langchain.retrievers` -> `langchain_community.retrievers` ->
`langchain_classic.retrievers`, with some versions re-exporting from more
than one of those at once). Rather than hard-coding today's location and
breaking on someone else's installed version, `_resolve()` tries every
known location in order and uses whichever one is actually importable --
and the whole lookup is deferred into `build_ensemble_retriever()` so that
`--engine custom` keeps working even on a LangChain install where NONE of
them are present (`ask.py` imports this module unconditionally).
"""

import importlib
from typing import Any, List, Optional, Type

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

import config
from bm25_index import BM25Index, tokenize
from embedder import embed_query
from vector_store import dense_search

# (module, attribute) pairs to try, newest/most-likely-correct first. Covers
# every location these two classes have lived at across LangChain's 0.x ->
# 1.0 restructuring.
_ENSEMBLE_CANDIDATES = [
    ("langchain_classic.retrievers.ensemble", "EnsembleRetriever"),
    ("langchain_classic.retrievers", "EnsembleRetriever"),
    ("langchain_community.retrievers.ensemble", "EnsembleRetriever"),
    ("langchain_community.retrievers", "EnsembleRetriever"),
    ("langchain.retrievers.ensemble", "EnsembleRetriever"),
    ("langchain.retrievers", "EnsembleRetriever"),
]
_BM25_CANDIDATES = [
    ("langchain_community.retrievers.bm25", "BM25Retriever"),
    ("langchain_community.retrievers", "BM25Retriever"),
    ("langchain_classic.retrievers.bm25", "BM25Retriever"),
    ("langchain_classic.retrievers", "BM25Retriever"),
    ("langchain.retrievers.bm25", "BM25Retriever"),
    ("langchain.retrievers", "BM25Retriever"),
]


def _resolve(candidates: list, what: str) -> Type:
    """Return the first importable class among `candidates`. Raises a
    single clear ImportError (listing every location tried) only if NONE
    of them work -- this runs lazily, inside build_ensemble_retriever(),
    never at module import time.
    """
    errors = []
    for module_path, attr in candidates:
        try:
            module = importlib.import_module(module_path)
            return getattr(module, attr)
        except (ImportError, AttributeError) as e:
            errors.append(f"  {module_path}.{attr} -- {type(e).__name__}: {e}")

    tried = "\n".join(errors)
    raise ImportError(
        f"Could not find {what} in any known LangChain location. Tried:\n{tried}\n\n"
        f"Install one of: pip install langchain-classic  (or langchain-community, "
        f"or langchain, whichever your LangChain version ships it under), "
        f"then retry. --engine custom (hybrid_retriever.py) works regardless."
    )


class PgVectorRetriever(BaseRetriever):
    """Minimal LangChain BaseRetriever wrapper around this project's
    existing Postgres + pgvector dense store (vector_store.dense_search).

    EnsembleRetriever only requires `.invoke(query) -> list[Document]` from
    each of its sub-retrievers -- it doesn't care that the store underneath
    isn't a `langchain_core.vectorstores.VectorStore`.
    """

    k: int = Field(default=4)

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> List[Document]:
        query_vector = embed_query(query)
        results = dense_search(query_vector, k=self.k)
        return [
            Document(
                page_content=r["content"],
                metadata={**r["metadata"], "chunk_id": r["chunk_id"], "dense_score": r["dense_score"]},
            )
            for r in results
        ]


def build_ensemble_retriever() -> Any:
    """Load the persisted BM25 pickle + the live pgvector table and wire
    both into one LangChain EnsembleRetriever (weighted RRF, c=RRF_K),
    deduplicating documents across the two retrievers by `chunk_id`.

    Resolves BM25Retriever/EnsembleRetriever from whichever LangChain
    package actually provides them on this machine (see module docstring).
    """
    BM25Retriever = _resolve(_BM25_CANDIDATES, "BM25Retriever")
    EnsembleRetriever = _resolve(_ENSEMBLE_CANDIDATES, "EnsembleRetriever")

    bm25_data = BM25Index.load()
    bm25_retriever = BM25Retriever.from_texts(
        texts=bm25_data.contents,
        metadatas=[
            {**meta, "chunk_id": cid}
            for cid, meta in zip(bm25_data.chunk_ids, bm25_data.metadatas)
        ],
        preprocess_func=tokenize,  # same tokenizer as bm25_index.py, for apples-to-apples comparison
    )
    bm25_retriever.k = config.TOP_K_SPARSE

    pgvector_retriever = PgVectorRetriever(k=config.TOP_K_DENSE)

    return EnsembleRetriever(
        retrievers=[bm25_retriever, pgvector_retriever],
        weights=[config.SPARSE_WEIGHT, config.DENSE_WEIGHT],
        c=config.RRF_K,
        id_key="chunk_id",
    )


def ensemble_search(query: str, k_final: Optional[int] = None) -> List[dict]:
    """Run LangChain's EnsembleRetriever and normalize its output into the
    same result-dict shape `hybrid_retriever.hybrid_search()` returns, so
    `ask.py` can print/consume either engine identically.
    """
    k_final = k_final or config.TOP_K_FINAL
    retriever = build_ensemble_retriever()
    docs = retriever.invoke(query)

    results = []
    for rank, doc in enumerate(docs[:k_final], start=1):
        meta = doc.metadata
        results.append(
            {
                "chunk_id": meta.get("chunk_id"),
                "content": doc.page_content,
                "source_file": meta.get("source_file"),
                "chunk_index": meta.get("chunk_index"),
                "metadata": meta,
                "ensemble_rank": rank,
            }
        )
    return results


if __name__ == "__main__":
    print("--- LangChain EnsembleRetriever: 'ACC adaptive cruise control' ---")
    for r in ensemble_search("ACC adaptive cruise control"):
        preview = r["content"][:120].replace("\n", " ")
        print(f"  rank={r['ensemble_rank']} | {r['source_file']} (chunk {r['chunk_index']}) | {preview}...")

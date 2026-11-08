"""
STEP 4: Hybrid retrieval -- fuse dense (pgvector) + sparse (BM25) rankings.

This is the heart of "Hybrid RAG": run BOTH retrievers against the same
query, then merge their two ranked lists into one. Why not just pick the
better one?

  - Dense/vector search (vector_store.dense_search) is great at MEANING:
    "my car won't start" matches a chunk about "engine fails to crank"
    even with zero shared words. It's comparatively weak on rare, exact
    tokens (model codes, acronyms like "ACC"/"ABS") because the embedding
    model may not have learned a sharp, distinct vector for them.

  - Sparse/BM25 search (bm25_index.BM25Index.search) is great at EXACT
    TERMS: an acronym, a part number, a filename-sounding phrase. It's
    comparatively weak on paraphrase/synonymy: "won't start" vs "fails to
    crank" share no tokens, so BM25 scores that pair near zero.

Combining both covers more queries than either alone.

Fusion method -- Reciprocal Rank Fusion (RRF), the default:
    score(chunk) = sum over each retriever R that returned this chunk of
                   1 / (RRF_K + rank_in_R)
  RRF only looks at RANK POSITION, not raw score magnitude, which sidesteps
  the problem that cosine similarity (0..1) and BM25 scores (unbounded,
  corpus-dependent) are not on comparable scales. This is the same fusion
  method used in Elasticsearch's and Azure AI Search's hybrid query APIs.

A simpler "weighted" method is also included for comparison: normalize each
retriever's raw scores to [0, 1] and combine with DENSE_WEIGHT/SPARSE_WEIGHT.
It's more interpretable but more fragile (sensitive to how you normalize).
"""

from typing import List, Optional

import config
from bm25_index import BM25Index
from embedder import embed_query
from vector_store import dense_search


def _rrf_fuse(dense_results: List[dict], sparse_results: List[dict], rrf_k: int) -> List[dict]:
    scores: dict[str, float] = {}
    chunks: dict[str, dict] = {}

    for rank, r in enumerate(dense_results, start=1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        chunks.setdefault(cid, {**r, "dense_rank": rank, "sparse_rank": None})
        chunks[cid]["dense_rank"] = rank
        chunks[cid]["dense_score"] = r.get("dense_score")

    for rank, r in enumerate(sparse_results, start=1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        if cid not in chunks:
            chunks[cid] = {**r, "dense_rank": None, "sparse_rank": rank}
        else:
            chunks[cid]["sparse_rank"] = rank
            chunks[cid]["sparse_score"] = r.get("sparse_score")

    fused = []
    for cid, rrf_score in scores.items():
        row = dict(chunks[cid])
        row["rrf_score"] = rrf_score
        fused.append(row)

    fused.sort(key=lambda r: r["rrf_score"], reverse=True)
    return fused


def _normalize(results: List[dict], key: str) -> dict:
    """Return {chunk_id: normalized_score in [0,1]} for a list of result dicts."""
    if not results:
        return {}
    values = [r[key] for r in results]
    lo, hi = min(values), max(values)
    span = (hi - lo) or 1.0
    return {r["chunk_id"]: (r[key] - lo) / span for r in results}


def _weighted_fuse(dense_results: List[dict], sparse_results: List[dict]) -> List[dict]:
    dense_norm = _normalize(dense_results, "dense_score")
    sparse_norm = _normalize(sparse_results, "sparse_score")

    chunks: dict[str, dict] = {}
    for r in dense_results:
        chunks.setdefault(r["chunk_id"], {**r})
    for r in sparse_results:
        chunks.setdefault(r["chunk_id"], {**r}).update(
            {k: v for k, v in r.items() if k not in chunks[r["chunk_id"]]}
        )
        chunks[r["chunk_id"]]["sparse_score"] = r["sparse_score"]

    fused = []
    for cid, row in chunks.items():
        d = dense_norm.get(cid, 0.0)
        s = sparse_norm.get(cid, 0.0)
        row = dict(row)
        row["weighted_score"] = config.DENSE_WEIGHT * d + config.SPARSE_WEIGHT * s
        fused.append(row)

    fused.sort(key=lambda r: r["weighted_score"], reverse=True)
    return fused


def hybrid_search(
    query: str,
    bm25_index: BM25Index,
    k_final: Optional[int] = None,
    k_dense: Optional[int] = None,
    k_sparse: Optional[int] = None,
    method: Optional[str] = None,
) -> List[dict]:
    """Run dense + sparse retrieval for `query` and return the fused top-k."""
    k_final = k_final or config.TOP_K_FINAL
    k_dense = k_dense or config.TOP_K_DENSE
    k_sparse = k_sparse or config.TOP_K_SPARSE
    method = method or config.FUSION_METHOD

    query_vector = embed_query(query)
    dense_results = dense_search(query_vector, k=k_dense)
    sparse_results = bm25_index.search(query, k=k_sparse)

    if method == "weighted":
        fused = _weighted_fuse(dense_results, sparse_results)
    else:
        fused = _rrf_fuse(dense_results, sparse_results, rrf_k=config.RRF_K)

    return fused[:k_final]


if __name__ == "__main__":
    index = BM25Index.load()
    for r in hybrid_search("ACC adaptive cruise control", index):
        print(
            f"rrf={r.get('rrf_score', 0):.4f} dense_rank={r.get('dense_rank')} "
            f"sparse_rank={r.get('sparse_rank')} | {r['source_file']}"
        )

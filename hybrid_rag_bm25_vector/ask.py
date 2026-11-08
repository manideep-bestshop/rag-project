"""
Query the hybrid index: run BM25 (sparse) and pgvector (dense) retrieval in
parallel, fuse their rankings (RRF by default), and hand the fused top-k
chunks to a local Ollama LLM to generate a grounded answer.

Two fusion ENGINES are available, both doing the same RRF math:
  - "custom"   (default) -- hybrid_retriever.py's hand-rolled fusion loop.
  - "ensemble"            -- LangChain's own retrievers.EnsembleRetriever,
                              wired up in ensemble_retriever.py.

Usage:
    python ask.py "How do I change a tyre on an Aventro car?"
    python ask.py --no-generate "ACC"
    python ask.py --method weighted "adaptive cruise control"
    python ask.py --compare "ACC"             # also show dense-only vs sparse-only
    python ask.py --engine ensemble "ACC"     # use LangChain's EnsembleRetriever
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from bm25_index import BM25Index
from embedder import embed_query
from vector_store import dense_search
from hybrid_retriever import hybrid_search
from ensemble_retriever import ensemble_search
import config


def generate_answer(query: str, results: list[dict]) -> str:
    from langchain_ollama import ChatOllama

    context = "\n\n".join(
        f"[Source: {r['source_file']}]\n{r['content']}" for r in results
    )
    prompt = (
        "Answer the question using ONLY the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    llm = ChatOllama(model=config.LLM_MODEL, temperature=0)
    response = llm.invoke(prompt)
    return response.content


def _print_results(results: list[dict], score_key: str) -> None:
    for i, r in enumerate(results, 1):
        preview = r["content"][:160].replace("\n", " ").strip()
        print(f"  [{i}] {score_key}={r.get(score_key, 0):.4f} | {r['source_file']} (chunk {r['chunk_index']})")
        print(f"      {preview}...")


def ask(
    query: str,
    generate: bool = True,
    method: str = None,
    compare: bool = False,
    engine: str = "custom",
) -> None:
    print(f"\nQuery: {query}")

    if compare:
        bm25 = BM25Index.load()
        print("\n--- Dense-only (pgvector / semantic) ---")
        _print_results(dense_search(embed_query(query)), "dense_score")

        print("\n--- Sparse-only (BM25 / keyword) ---")
        _print_results(bm25.search(query), "sparse_score")

    fused = None
    score_key = None
    if engine == "ensemble":
        try:
            fused = ensemble_search(query)
            score_key = "ensemble_rank"
            print("\n--- Hybrid fused (engine=ensemble, LangChain EnsembleRetriever) ---")
        except ImportError as e:
            print(f"\n[engine=ensemble unavailable] {e}\n")
            print("Falling back to --engine custom (hybrid_retriever.py) instead...\n")
            engine = "custom"

    if engine != "ensemble" or fused is None:
        bm25 = BM25Index.load()
        fused = hybrid_search(query, bm25, method=method)
        score_key = "weighted_score" if (method or config.FUSION_METHOD) == "weighted" else "rrf_score"
        print(f"\n--- Hybrid fused (engine=custom, method={method or config.FUSION_METHOD}) ---")

    _print_results(fused, score_key)

    if generate:
        print("\nGenerating answer with local LLM...")
        answer = generate_answer(query, fused)
        print("\n" + "-" * 70)
        print("Answer:")
        print(answer)
        print("-" * 70)


if __name__ == "__main__":
    args = sys.argv[1:]
    do_generate = "--no-generate" not in args
    do_compare = "--compare" in args
    args = [a for a in args if a not in ("--no-generate", "--compare")]

    fusion_method = None
    if "--method" in args:
        idx = args.index("--method")
        fusion_method = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    engine = "custom"
    if "--engine" in args:
        idx = args.index("--engine")
        engine = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    query = " ".join(args) or "How do I change a tyre on an Aventro car?"
    ask(query, generate=do_generate, method=fusion_method, compare=do_compare, engine=engine)

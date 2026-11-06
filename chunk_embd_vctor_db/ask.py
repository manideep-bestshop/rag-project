"""
Query the vector database: embed a question, retrieve the top-k most
similar chunks from Postgres, and (optionally) ask a local Ollama LLM to
answer using only that retrieved context. This is the "R" + "AG" of RAG:
Retrieval (vector search) feeding Generation (the LLM call).

Usage:
    python ask.py "How do I change a tyre on an Aventro car?"
    python ask.py --no-generate "adaptive cruise control"
"""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from embedder import embed_query
from vector_store import similarity_search
import config


def retrieve(query: str, k: int = None):
    k = k or config.TOP_K
    query_vector = embed_query(query)
    return similarity_search(query_vector, k=k)


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


def ask(query: str, k: int = None, generate: bool = True) -> None:
    results = retrieve(query, k=k)

    print(f"\nQuery: {query}")
    print(f"Top {len(results)} chunk(s) by cosine similarity:\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] similarity={r['similarity']:.4f} | {r['source_file']} (chunk {r['chunk_index']})")
        preview = r["content"][:220].replace("\n", " ").strip()
        print(f"    {preview}...\n")

    if generate:
        print("Generating answer with local LLM...")
        answer = generate_answer(query, results)
        print("\n" + "-" * 70)
        print("Answer:")
        print(answer)
        print("-" * 70)


if __name__ == "__main__":
    args = sys.argv[1:]
    do_generate = "--no-generate" not in args
    args = [a for a in args if a != "--no-generate"]
    query = " ".join(args) or "How do I change a tyre on an Aventro car?"
    ask(query, generate=do_generate)

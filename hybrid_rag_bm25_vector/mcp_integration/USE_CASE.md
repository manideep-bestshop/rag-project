# Use Case: Aventro Knowledge MCP Server

**Status:** proposed design (not yet implemented) — see `ARCHITECTURE.md` for the
technical design and `../README.md` for the hybrid RAG pipeline this builds on.

---

## 1. The problem today

The hybrid RAG pipeline in this project (`hybrid_rag_bm25_vector`) is a
**script you run**: `python ingest.py` once, then `python ask.py "question"`
per query, from a terminal, in this project's own `.venv`. That's fine for
development and for the worked examples in the README, but it doesn't
compose with anything else:

- A support agent using **Claude Desktop** can't ask it a question without
  switching to a terminal, activating the venv, and typing a CLI command.
- A **second project** that also wants to answer Aventro questions (a Slack
  bot, an internal dashboard, a different Claude Code session) would have to
  either shell out to `ask.py` or re-implement the whole
  BM25 + pgvector + RRF + Ollama pipeline itself.
- There's no way for an AI assistant to *discover* what this tool can do —
  it has to be told, by a human, to run a specific Python script with
  specific flags.

**MCP (Model Context Protocol)** exists exactly for this gap: it's a
standard way to expose a capability — here, "search/ask the Aventro
knowledge base" — as a set of typed, discoverable tools that *any*
MCP-aware client can call, the same way this very session calls `Read`,
`Bash`, or `Artifact`.

---

## 2. The use case

> **Turn the hybrid retriever into a reusable MCP tool, so any MCP client —
> Claude Desktop, Claude Code, a Slack bot, an internal support console —
> can query the Aventro Motors knowledge base directly inside a
> conversation, grounded in the real PDFs, with zero per-client
> reimplementation and zero network exposure (everything stays local).**

### 2.1 Primary persona — dealership support staff, via Claude Desktop

A support rep is on a call with a customer who has a flat tyre on an
Aventro Swift Sedan. Today they'd have to find the right PDF manually. With
the MCP server configured in Claude Desktop, they just type:

> "How do I walk a customer through changing a tyre on an Aventro Swift
> Sedan?"

Claude Desktop recognizes an `aventro-rag` MCP server is available, calls
its `ask_aventro_question` tool, which runs the *exact same* hybrid
retrieval (BM25 keyword match + pgvector semantic search, fused via RRF)
and local `llama3.2` generation already proven in this project's README —
and returns a grounded, cited answer inline in the chat. No terminal, no
copy-pasting PDF text, no separate app.

### 2.2 Secondary persona — a developer building a different tool, via Claude Code

A developer is building an unrelated internal tool (say, a service-ticket
triage assistant) in a completely different repo, using Claude Code. They
need to look up Aventro service-center policy for one feature. Instead of
vendoring this project's retrieval code, they add this project's MCP server
to their Claude Code session's `.mcp.json` and call
`search_aventro_docs("warranty claim process")` directly from *their*
conversation — the knowledge base is reused, not duplicated.

### 2.3 Tertiary persona — an automated QA/regression script

A nightly script (itself an MCP client, e.g. built with the Claude Agent
SDK) calls `search_aventro_docs` for a fixed list of known questions and
checks that the top result's `source_file` still matches the expected PDF
after each re-ingestion — a cheap regression test for "did changing
`CHUNK_SIZE` or re-ordering the corpus silently break retrieval for X?"

---

## 3. Why MCP specifically (not just a REST API)

| Option | Problem it has here |
|---|---|
| Keep it a CLI script (`ask.py`) | Not callable by an AI assistant without a human running a terminal command for it each time. |
| Wrap it in a REST API (FastAPI, Flask) | Works, but now you're running/securing a network service, picking a port, handling auth — for something that's meant to stay 100% local and single-user. Every client also has to hand-write its own HTTP client + JSON schema instead of getting one for free. |
| **MCP server over stdio** | The MCP host (Claude Desktop, Claude Code, etc.) spawns the server as a local subprocess and talks to it over stdin/stdout — no port, no network exposure, no auth needed, and the tool's input/output schema is self-describing, so any MCP-aware client gets autodiscovery and type-checked calls for free. |

This also matches the project's existing "100% local" constraint (see the
root `README.md`): stdio transport means the knowledge base never leaves
the machine, same as Ollama + pgvector + the BM25 pickle today.

---

## 4. What gets exposed (tool/resource sketch)

See `ARCHITECTURE.md` §4 for the full design; summary:

| MCP primitive | Name | Wraps | Purpose |
|---|---|---|---|
| Tool | `search_aventro_docs` | `hybrid_retriever.hybrid_search()` / `ensemble_retriever.ensemble_search()` | Retrieval only — returns fused chunks + scores + citations, no LLM call. Cheap, used for quick lookups or by other agents that want to do their own synthesis. |
| Tool | `ask_aventro_question` | `ask.py`'s `retrieve + generate_answer()` | Full RAG answer: retrieves, then asks local `llama3.2` to answer from the fused context. What a support rep would actually use. |
| Tool | `list_ingested_sources` | `vector_store.count_chunks()` + a `DISTINCT source_file` query | Lets a client (or a human) see which PDFs are currently indexed, for trust/debugging — "is the answer based on the *current* Aventro Swift Sedan manual, or last month's?" |
| Resource | `aventro://corpus/{source_file}` | raw chunk text for one PDF | Optional: lets a client read a whole document's indexed text directly, e.g. to quote a full procedure rather than a 800-char chunk. |

---

## 5. Non-goals / what this does NOT change

- **Ingestion stays a CLI step.** `python ingest.py` still builds the two
  indexes (pgvector table + BM25 pickle) exactly as today. The MCP server
  is a *query-time* interface only — it reads the indexes, it doesn't build
  them. (A `reindex` tool is listed as a future extension in
  `ARCHITECTURE.md §7`, not part of this use case.)
- **No new retrieval logic.** The MCP server is a thin adapter around
  `hybrid_retriever.py` / `ensemble_retriever.py` / `ask.py`, which are
  unchanged. Two engines (`custom` RRF and LangChain `EnsembleRetriever`,
  with its fallback resolver) both stay usable, selected via a tool
  parameter.
- **`ask.py` keeps working standalone.** The MCP server is additive, not a
  replacement for the terminal workflow documented in the main README.

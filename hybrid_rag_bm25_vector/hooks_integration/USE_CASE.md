# Use Case: Claude Code Hooks for the Aventro Hybrid RAG Project

**Status:** proposed design (not yet implemented) — see `ARCHITECTURE.md` for
the technical design (including a primer on how Claude Code hooks actually
work) and `../README.md` for the hybrid RAG pipeline this builds on. Compare
with `../mcp_integration/` — that proposal exposes this project as tools an
agent *chooses* to call; this one makes the project act *automatically* on
events in a Claude Code session, with no explicit tool call at all.

---

## 1. The problem today

Everything in this project is **invoked on purpose**: a human (or an agent
via the MCP proposal) has to decide "I should search Aventro docs now" and
explicitly run `ask.py` or call a tool. Two things never happen
automatically:

1. **Grounding.** If someone is chatting with Claude Code in this repo and
   asks "what's the procedure for a dead key fob battery?", Claude has no
   reason to know it should consult this project's corpus unless told to —
   it'll either guess from general knowledge or the user has to remember to
   run `ask.py` themselves first and paste the answer back in.
2. **Freshness.** If a new PDF gets added to `aventro/pdf/` (or an existing
   one gets edited) during a session, nothing reminds anyone that
   `bm25_store/bm25_index.pkl` and the `hybrid_document_chunks` table are
   now stale until someone happens to remember to re-run `python ingest.py`.

**Claude Code hooks** exist for exactly this: shell commands that the
harness runs automatically on specific lifecycle events (a prompt being
submitted, a tool about to run, a tool having just run, etc.), without
anyone having to ask for it each time.

---

## 2. The use cases

### 2.1 Primary — Auto-Grounding Hook (`UserPromptSubmit`)

> **Every time a prompt is submitted in a Claude Code session working in
> this repo, a hook silently checks whether it's Aventro-related, and if
> so, runs the SAME hybrid retriever this project already has and injects
> the retrieved chunks as extra context — before Claude ever sees the raw
> prompt.**

**Scenario:** a developer is in a Claude Code session in this repo (or any
repo with this hook installed) and types:

> "What's the fastest way to walk someone through a dead key-fob battery?"

Without the hook, Claude answers from general knowledge (or has to be told
"use ask.py"). With the hook:

1. `UserPromptSubmit` fires before Claude sees the prompt.
2. The hook script runs a **cheap BM25-only check first** (no Ollama round
   trip) against the persisted `bm25_store/bm25_index.pkl` — if nothing
   scores above a relevance threshold, the hook exits immediately and
   Claude proceeds exactly as if no hook existed (near-zero cost on
   unrelated prompts, like "fix this failing test").
3. If the BM25 pre-check *does* find a plausible match, the hook runs the
   full hybrid search (`hybrid_retriever.hybrid_search()`, BM25 + pgvector,
   RRF-fused — identical code path to `ask.py`) and injects the top chunks
   as additional context.
4. Claude answers the actual question, now grounded in
   `How to Open the Door When the Car Key Remote Battery is Dead.pdf`,
   with no tool call visible in the transcript — it just already knew.

This is the single most useful case: it turns the project from "a tool you
have to remember to use" into **ambient grounding** for anyone working in
or near this corpus.

### 2.2 Secondary — Auto-Reindex Reminder (`PostToolUse`)

> **Whenever a `Write` or `Edit` tool call touches a file under
> `aventro/pdf/` or `aventro/md/`, a hook fires right after and flags that
> the two indexes are now stale — so staleness is caught the moment it
> happens, not discovered later from a wrong answer.**

**Scenario:** someone asks Claude Code to add a new PDF-derived markdown
note under `aventro/md/` (or edits an existing one). The `PostToolUse`
hook sees a write under a watched path and prints:

```
[hooks] aventro/md/New Model Launch FAQ.md changed — bm25_store/bm25_index.pkl
and the hybrid_document_chunks table do not include this yet. Run:
  python ingest.py --reset
```

By default this hook only **warns** (non-blocking, visible to the user and
the model) rather than silently re-running a multi-minute ingestion job on
every keystroke — see `ARCHITECTURE.md §4.2` for why auto-run is explicitly
not the default.

### 2.3 Tertiary — Destructive-Action Guardrail (`PreToolUse`)

> **Before any `Bash` tool call runs, a hook checks whether the command
> would wipe this project's indexes (`ingest.py --reset`, `TRUNCATE
> hybrid_document_chunks`, `rm -rf bm25_store`, `DROP DATABASE
> aventro_hybrid_rag_db`) and, if so, blocks it with an explanation instead
> of letting it run silently.**

This mirrors a rule this very session already follows by instruction (see
the "Executing actions with care" guidance governing this conversation): a
`PreToolUse` hook is how you turn that kind of judgment call into something
enforced mechanically, every time, regardless of which prompt is driving
the session.

---

## 3. Why hooks specifically (not just "ask the model to remember")

| Option | Problem it has here |
|---|---|
| Tell Claude in a system prompt / CLAUDE.md "always check the Aventro docs first" | Advisory only — depends on the model choosing to comply every single time, costs a tool call + round trip even for irrelevant prompts, and a differently-configured session (or a different model) won't have it at all. |
| The MCP server (`../mcp_integration/`) | Requires the model to *decide* to call a tool. Great for "the agent should be able to look this up," wrong fit for "this should just always happen, deterministically, for every matching event." |
| **Hooks** | Run deterministically on the harness side for a specific event, regardless of what the model decides — the model doesn't even have to know the hook exists. Zero reliance on the model "remembering" to ground itself or an engineer remembering to re-ingest. |

Hooks and the MCP server proposal are complementary, not competing: hooks
handle "this should happen automatically, every time this event occurs,"
while MCP handles "an agent should be able to ask for this on demand."

---

## 4. Non-goals / what this does NOT change

- **No new retrieval logic.** Exactly like the MCP proposal, every hook
  script here is a thin wrapper that imports and calls
  `hybrid_retriever.py` / `bm25_index.py` / `vector_store.py` as a library
  — nothing about retrieval itself changes.
- **Ingestion stays explicit by default.** The reindex hook (§2.2) warns;
  it does not auto-run `ingest.py` unless explicitly opted into (see
  `ARCHITECTURE.md §4.2`).
- **Hooks are opt-in, scoped to a settings.json.** Nothing here runs unless
  a developer deliberately adds these entries to `.claude/settings.json`
  (project-level) or their own user settings — see `ARCHITECTURE.md §2`
  for precedence and scope.

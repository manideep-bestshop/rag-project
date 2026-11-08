# Use Case: Combining Hooks + MCP Tool Calling — the Aventro Support Copilot

**Status:** proposed design (not yet implemented). This builds directly on
the two earlier proposals — `../hooks_integration/` (automatic, deterministic
triggers) and `../mcp_integration/` (on-demand, agent-chosen tools) — and
shows them working *together* as one system instead of two unrelated
options. Read those two first if you haven't; this document assumes their
designs and gives each one a job the other can't do alone.

---

## 1. The gap neither one closes by itself

- **Hooks alone** (`../hooks_integration/`) are great at "this should always
  happen" — but a hook is a short, cheap script that fires on every event.
  It can't afford to run a deep, multi-engine retrieval on *every* prompt,
  and it has no way to ask a human/agent "are you sure?" before doing
  something destructive — it can only allow or block, mechanically.
- **MCP tools alone** (`../mcp_integration/`) are great at "an agent can ask
  for this deliberately" — but nothing *makes* an agent remember to call
  them. A support rep's prompt might be Aventro-relevant and the model
  simply doesn't reach for the tool. And a destructive action (re-ingesting
  the corpus) is just as easy to request via the MCP tool as a harmless
  one — nothing inherently gates it.

**The fix is to give each piece the job it's actually good at**, and have
hooks actively *route into* MCP tools rather than the two living side by
side:

| Job | Owner | Why |
|---|---|---|
| Decide whether *anything* needs to happen at all, on every single prompt, at near-zero cost | **Hook** (`UserPromptSubmit`) | Hooks run deterministically, every time, before the model sees anything — exactly the "always check" job. |
| Do a deep, possibly-expensive retrieval when it's actually warranted | **MCP tool** (`ask_aventro_question` / `search_aventro_docs`) | Tool calls are the agent's deliberate choice — the right place for something that costs real latency/tokens and should only run when it'll pay off. |
| Mutate the indexes (re-ingest, reset) | **MCP tool, gated by a hook** | A hook can *block* the raw dangerous command outright; the MCP tool is the one sanctioned, confirmable path left to actually do it. |

---

## 2. The combined flow — three tiers of relevance, not two

`../hooks_integration/`'s auto-grounding hook had one threshold: "relevant
enough to inject, or not." This proposal splits that into **three** bands,
because "inject raw chunks automatically" and "do nothing" aren't the only
two sensible responses to an ambiguous prompt:

| BM25 top score | Band | What the hook does |
|---|---|---|
| `< RELEVANCE_FLOOR_LOW` | **Not relevant** | Nothing. Exit 0, silent — identical to `../hooks_integration/`'s skip path. |
| `RELEVANCE_FLOOR_LOW` … `RELEVANCE_FLOOR_HIGH` | **Ambiguous** | Inject a *short hint*, not full context: "This may relate to Aventro Motors documentation — consider calling `search_aventro_docs` or `ask_aventro_question` for a grounded answer." The hook deliberately does NOT decide for the model here — it hands the decision to the agent, via the MCP tool, because a single BM25 score isn't confident enough to justify a 4-chunk context injection the user never asked for. |
| `>= RELEVANCE_FLOOR_HIGH` | **Confidently relevant** | Auto-inject full hybrid-search context directly, exactly like `../hooks_integration/`'s fast path — no tool call needed, no round trip. |

This is the first place hooks and MCP genuinely combine: **the hook's job
in the middle band is explicitly to recommend the tool, not replace it.**

---

## 3. The combined flow — guardrail hands off to a confirmable tool

`../hooks_integration/`'s guardrail hook (§5 there) only *blocks* dangerous
Bash commands (`ingest.py --reset`, `rm -rf bm25_store`, ...). Blocking
alone leaves a dead end: now what? This proposal adds the other half —
a new MCP tool, `reindex_aventro_corpus(confirm: bool)`, which is the
**one sanctioned way** to actually rebuild the indexes once the raw
command is blocked:

1. Someone (or Claude, following an instruction) tries to run
   `python ingest.py --reset` directly in `Bash`.
2. `PreToolUse` guardrail hook blocks it — same as before — but the block
   reason now explicitly says: *"Use the `reindex_aventro_corpus` MCP tool
   instead (requires `confirm=true`)."*
3. Claude calls `reindex_aventro_corpus(confirm=false)` first (or the
   human asks it to) — gets back a **dry-run summary** ("3 new/changed
   files since last ingest: ..., this would replace 28 chunks with ~34").
4. Only on `reindex_aventro_corpus(confirm=true)` does it actually run
   `ingest.py --reset` under the hood.

The raw destructive path is closed off entirely by the hook; the only way
through is the tool, and the tool itself requires an explicit second step.
That's two independent layers (hook gate + tool confirmation), neither of
which alone would be as safe.

---

## 4. The combined flow — reindex *reminder* becomes a reindex *nudge*

`../hooks_integration/`'s `PostToolUse` reindex hook (§4 there) only
printed a warning when a corpus file changed. Combined with the new
`reindex_aventro_corpus` tool from §3, that warning can now point at a
concrete next action instead of just a command to copy-paste:

```
[hooks] "New Model Launch FAQ.md" changed under the Aventro corpus.
Consider calling reindex_aventro_corpus(confirm=true) (via the aventro-rag
MCP server) to bring the indexes back in sync.
```

Claude — not the hook — decides whether and when to actually call it
(e.g., it might reasonably wait until a batch of edits is done rather than
re-indexing after every single file change). The hook's job stays narrow:
notice the staleness and point at the sanctioned fix; the tool call itself
stays a deliberate, agent-initiated decision.

---

## 5. Primary persona — dealership support rep, end to end

> "Can you check if the Aventro Swift Sedan's tyre comes with roadside
> assistance?"

1. `UserPromptSubmit` hook: BM25 top score lands in the **ambiguous** band
   (query mixes a model name with a policy question — no single chunk is a
   slam dunk). Hook injects the short hint, not full context.
2. Claude, seeing the hint, calls `ask_aventro_question` (MCP tool) itself
   — a deliberate choice informed by the hint, not an automatic injection.
3. The MCP server runs the full hybrid retriever (`hybrid_retriever.py`,
   same RRF fusion as everywhere else in this project) and
   `generate_answer()`, returns a cited answer.
4. Claude relays it to the rep, with the PDF citation.

Later, the rep (or a teammate) updates
`aventro/md/Aventro Swift Sedan.md` with a new roadside-assistance clause.

5. `PostToolUse` hook fires, sees the change, nudges toward
   `reindex_aventro_corpus`.
6. Claude calls it with `confirm=false` first, sees the dry-run summary,
   confirms with the rep, then calls it again with `confirm=true`.
7. The next question about that clause is now answered from the updated
   text — the whole loop, from "doc changed" to "index caught up," never
   required a human to remember a CLI command.

---

## 6. Non-goals

- This does not change anything about `../mcp_integration/`'s existing
  three tools (`search_aventro_docs`, `ask_aventro_question`,
  `list_ingested_sources`) — they're reused as-is. Only one new tool is
  introduced here (`reindex_aventro_corpus`).
- This does not change anything about `../hooks_integration/`'s auto-reindex
  hook's *default* of warning rather than auto-running ingestion — see
  `ARCHITECTURE.md §4` here for why that stays true even with a confirmable
  tool now available.
- Still 100% local, still no new secrets, still nothing implemented yet —
  this is a design document, same as the two it builds on.

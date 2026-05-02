# Token Budget Reference

Cheat sheet for every hardcoded `max_tokens` value in `dova serve` / `dova mcp serve`.

## Provider-level defaults (config/providers.py)

Applied when an LLM call does NOT override `max_tokens`. Per `TaskType`:

| TaskType | tokens | use |
|---|---:|---|
| `QUICK` / `FAST` | 10,240 | classifier / gate calls |
| `STANDARD` | 20,480 | default conversational |
| `REASONING` | 20,480 | deliberation, analyst calls |
| `DEEP_ANALYSIS` | 81,920 | long-form research synthesis |
| `CREATIVE` / `SUMMARIZATION` | 40,960 | narrative generation |
| `EMBEDDING` | 81,920 | embedding requests |
| OpenAI fallback | 16,384 | `OPENAI_DEFAULT_MAX_TOKENS` |

`LLMRouter._clamp_max_tokens` silently caps these at the provider's hard
model ceiling (e.g. Opus 4.7 output ceiling), so requesting more than the
model supports is a no-op — don't chase bigger numbers here.

## Per-call explicit overrides

Sorted by file, showing only calls that matter to `dova serve` and
`dova mcp serve` (CLI-only paths in `cli/interact.py` are excluded).

| location | tokens | purpose | notes |
|---|---:|---|---|
| `agents/synthesis.py:184` | 40,000 | Main synthesis answer | long-form research output |
| `agents/synthesis.py:427` | 24,000 | Emergent-insight sub-synthesis | raised from 15,000 to give richer insight JSON headroom |
| `agents/thinking_orchestrator.py:1472` | 3,000 | Cross-domain AI↔Bio bridges | raised from 1,200 after observing JSON truncation |
| `agents/profiling.py:327` | 100 | Tiny classifier call | intentionally small |
| `config/providers.py:408` | 10 | Provider health probe | intentionally tiny |

## Guidelines for future changes

1. **Ceilings beat bumps.** Bedrock/Anthropic/OpenAI models cap output at
   32K–64K. Raising above that is silently clamped and costs nothing but
   clarity.
2. **Output tokens are billed.** Don't budget for slack "just in case" —
   the model stops when it's done. Over-budgeting costs nothing on quiet
   calls but attracts bloat over time.
3. **Prompt changes beat max_tokens changes.** If you want richer output,
   tell the prompt to produce longer output; `max_tokens` is the ceiling,
   not the target.
4. **If a call truncates, grep for it first.** Check `llm_call_complete`
   logs for `output_tokens == max_tokens`; that's the signature of
   truncation. Don't raise speculatively.

# DOVA Debate-First Architecture

## Goal
Make Multi-Round Adversarial Debates (Bull/Bear) run **before** synthesis so the final
user-visible answer is informed by — not isolated from — the debate. Then add a
refinement iteration so low-confidence syntheses get a second pass that addresses
bear concerns explicitly.

## Current State (verified)

File: `src/dova/agents/thinking_orchestrator.py`

Current ordering in `execute()`:
1. Deliberate (L511–683)
2. If `USE_TOOLS`: Research (L694) → **Synthesize** (L722–728)
3. Optional Debate (L733–768) — **AFTER** synthesis
4. Finalize (L770–802)

The `response` string returned to the user is produced at step 2. The debate at
step 3 only adds side-car keys (`bull_strengths`, `bear_concerns`, etc.) to
`action_result`. The primary response never reflects the debate.

## Target State

1. Deliberate
2. If `USE_TOOLS`: Research
3. **Multi-Round Bull/Bear Debate** (uses research evidence + query as context)
4. **Informed Synthesis** (synthesis prompt includes debate conclusions)
5. **Refinement** (one extra synthesis pass if `confidence_score < 0.7`)
6. Finalize

Trigger conditions unchanged: `force_debate` or (`auto_debate` AND evaluative query).
Non-debate queries keep the existing fast single-synthesis path.

## Implementation Checklist

- [ ] 1. **Move debate call before synthesis** in `ThinkingOrchestrator.execute()`
      (`thinking_orchestrator.py`, ~L693–768). Compute `should_debate` right
      after research, invoke `_run_debate` with tool results only, then pass
      `debate_output` into synthesis.

- [ ] 2. **Adapt `_run_debate`** (L1196–1248): make `synthesized_answer`
      parameter optional. When absent, context is research evidence + query only.

- [ ] 3. **Extend `_build_synthesis_prompt`** (L2161+) to accept optional
      `debate_output: dict | None`. When provided, inject an "Adversarial
      Analysis" section with bull strengths, bear concerns, balanced assessment
      and instruct the LLM to weigh both sides in its answer.

- [ ] 4. **Thread `debate_output` through** `_synthesize_with_results` and
      `_synthesize_with_results_stream` (L2114–2159).

- [ ] 5. **Add `_refine_synthesis` method**: takes the initial response +
      debate concerns + evidence, produces a refined response that explicitly
      addresses concerns. Only invoked when `debate.confidence_score <
      REFINE_THRESHOLD` (default 0.7). Capped at one round. Disabled when
      `task.params.get("refine") is False`.

- [ ] 6. **Update progress events**: fire `stage=debating` before
      `stage=synthesizing`; add `stage=refining` when refinement runs.

- [ ] 7. **Unit tests** (`tests/unit/agents/test_thinking_orchestrator.py`):
      - `test_debate_runs_before_synthesis` — assert call order
      - `test_synthesis_prompt_contains_debate_section` — assert prompt content
      - `test_refinement_triggers_on_low_confidence` — boundary at 0.7
      - `test_no_debate_preserves_single_synthesis_path` — regression guard
      - `test_non_evaluative_query_skips_debate` — trigger logic intact

## Success Criteria

- Final `response` text provably reflects debate content (e.g., visible hedging
  on bear-concern topics, explicit acknowledgement of strengths).
- When `confidence_score < 0.7`, a second synthesis runs and the final response
  differs from the first draft.
- All existing tests pass (`pytest tests/unit` and `pytest tests/integration`).
- Non-debate queries run in unchanged latency.
- Streaming progress events fire in the new order.

## Risks / Trade-offs

- **Latency**: debate adds ~4–5 extra LLM calls in front of synthesis.
  Mitigation: debate stays opt-in; default remains off.
- **Token cost**: synthesis prompt larger when debate output injected.
  Mitigation: compact summary format (strengths/concerns as bullets only).
- **Streaming UX**: first synthesis token appears later.
  Mitigation: emit richer progress events during debate rounds.

## Review Section — Implemented 2026-05-01

### What changed

**`src/dova/agents/thinking_orchestrator.py`** — rewired the USE_TOOLS branch of
`execute()` so the bull/bear debate runs **between** research and synthesis, not
after. Also injected debate output into the synthesis prompt and added an
optional refinement pass.

New flow:
```
Deliberate → Research → Debate (bull/bear, N rounds) → Synthesis (debate-informed) →
  Refinement (conditional, once, when debate confidence < threshold) → Finalize
```

Key code changes:

- `execute()` L684–802: reordered USE_TOOLS branch. Debate trigger computed
  once up front. `_run_debate` now runs on raw evidence (research output),
  and the produced `debate_out` is threaded into synthesis. A refinement
  pass runs once when `confidence_score < refine_threshold` (default 0.7).

- `_run_debate()` L1196+: made `synthesized_answer` optional (was required).
  When absent, the debate context contains only research evidence — matching
  the new pre-synthesis placement.

- `_synthesize_with_results()` and `_synthesize_with_results_stream()`:
  added optional `debate_output` kwarg. Threaded through to
  `_build_synthesis_prompt()`.

- `_build_synthesis_prompt()`: added optional `debate_output` kwarg that
  injects an "Adversarial Analysis" block between the search results and
  style instructions. The block contains bull strengths (as supporting
  bullets), bear concerns (with a rule that the synthesis MUST address
  every one — either rebut with evidence or acknowledge as a caveat),
  moderator's balanced assessment, recommendation, and confidence.
  Also includes explicit "DEBATE INTEGRATION RULES" so the LLM doesn't
  write meta-commentary about the debate itself.

- `_refine_synthesis()` — NEW method. Takes the initial synthesis response
  and the debate output, produces a revised answer that explicitly
  addresses each bear concern. Uses lower temperature (0.4) and compact
  prompt. Invoked at most once per query.

Progress events (streaming UX):
- `stage=debating` now fires BEFORE `stage=synthesizing` (was after).
- New `stage=refining` event fires only when refinement runs.
- `reflection` event content varies based on whether debate output is available.

**`tests/unit/agents/test_thinking_orchestrator.py`** — added `TestDebateFirstFlow`
class with 10 tests:
1. `test_build_prompt_without_debate_omits_block` — no debate → no Adversarial Analysis
2. `test_build_prompt_with_debate_injects_block` — debate → block present w/ all fields
3. `test_debate_runs_before_synthesis` — asserts call order: research → debate → synthesis
4. `test_no_debate_preserves_single_synthesis_path` — regression guard for default path
5. `test_non_evaluative_query_skips_debate` — trigger logic intact
6. `test_force_debate_overrides_evaluative_check` — `force_debate=True` bypasses pattern match
7. `test_refinement_triggers_on_low_confidence` — confidence 0.45 < 0.7 triggers refine
8. `test_high_confidence_skips_refinement` — confidence 0.85 ≥ 0.7 skips refine
9. `test_refinement_can_be_disabled` — `refine=False` disables refinement
10. `test_run_debate_accepts_no_synthesized_answer` — new optional arg works

### QA evidence

- `pytest tests/unit/agents/test_thinking_orchestrator.py::TestDebateFirstFlow`:
  **10/10 passed** in 0.07s
- `pytest tests/unit/`: **277/277 passed** (was 267 before; +10 new tests, 0 regressions)
- `pytest tests/`: **301/301 passed** (unit + integration)
- `python -c "ast.parse(...)"`: syntax valid

### Success criteria met

- [x] Debate runs before synthesis (proven by `test_debate_runs_before_synthesis`
      asserting call order)
- [x] Synthesis prompt contains debate output (proven by
      `test_build_prompt_with_debate_injects_block`)
- [x] Refinement triggers on low confidence (proven by
      `test_refinement_triggers_on_low_confidence`)
- [x] Non-debate queries unchanged (proven by
      `test_no_debate_preserves_single_synthesis_path`)
- [x] All existing tests pass

### Surprises / deviations from plan

- **Default `refine=True`** — the plan said "Controllable via
  `task.params.get('refine')`" without specifying a default. Chose `True` so
  refinement happens automatically when the debate reveals low confidence,
  which best fulfils the goal of enhancing accuracy/validity. Can be
  disabled with `refine=False`.

- **Debate trigger moved up** — originally scoped inside the debate step,
  but moving it to the top of section 3 made the code cleaner and lets
  the RESPOND_DIRECTLY branch remain simple (debate doesn't apply to
  context-only responses anyway).

- **No `SynthesisAgent` integration** — the legacy `synthesis.py` module
  exists but is unused by `ThinkingOrchestrator`. The Explore agent flagged
  this. Left alone — integrating it is a separate concern from the requested
  debate-first rewiring, and touching it risks scope creep.

### Files changed

```
src/dova/agents/thinking_orchestrator.py    (~130 lines changed/added)
tests/unit/agents/test_thinking_orchestrator.py  (+~340 lines, new class)
tasks/todo.md                                (rewritten for this task)
```

---

## Lessons

Captured in `tasks/lessons.md` after any user correction.

---

## Previous task archive

The previous content of this file (bio/pharma MCP servers integration, completed 2026-04-29)
has been preserved in git history (commit 0ed9024 / v1.7).

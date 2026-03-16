# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 0. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 1. Workflow

Before start writing code immediately:
- Start complex tasks in Plan mode
- Get plan approval before implementation
- Break large changes into reviewable chunks


## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

## 5. The Context Stuffing Trap

**Avoid context rot. Stay consise and focused

When implementing authentication:
- validate inputs, handle errors securely, follow auth/ patterns

# 6. VIBE CODING VALIDATION REQUIREMENTS
All code generated through vibe coding MUST be thoroughly tested and validated before deployment:
### 6.1 Feature Validation
- [ ] Verify ALL requested features are implemented correctly
- [ ] Confirm edge cases are handled appropriately
- [ ] Validate input/output behavior matches specifications
### 6.2 Functionality Testing
- [ ] Execute unit tests for all functions/methods
- [ ] Run integration tests for component interactions
- [ ] Perform end-to-end testing for complete workflows
- [ ] Test error handling and exception scenarios
### 6.3 Operability Verification
- [ ] Confirm code runs in target environment(s)
- [ ] Validate dependencies are properly declared
- [ ] Test performance under expected load
- [ ] Verify logging, monitoring, and observability
- [ ] Ensure graceful degradation and recovery
### 6.4 Code Quality
- [ ] Review for security vulnerabilities
- [ ] Check for code smells and anti-patterns
- [ ] Validate documentation completeness
- [ ] Confirm coding standards compliance

**🚫 DO NOT deploy or merge vibe-coded solutions without completing this checklist.**

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

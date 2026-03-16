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

## 0. Task Management
**Plan First**: Write plan to tasks/todo.md with checkable items
**Verify Plan**: Check in before starting implementation
**Track Progress**: Mark items complete as you go
**Explain Changes**: High-level summary at each step
**Document Results**: Add review section to tasks/todo.md
**Capture Lessons**: Update tasks/lessons.md after corrections

## 1. Core Principles

**Simplicity First**: Make every change as simple as possible. Impact minimal code.
**No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
**Minimat Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

## 2. Workflow

Before start writing code immediately:
- Start complex tasks in Plan mode
- Get plan approval before implementation
- Break large changes into reviewable chunks


## 3. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 4. Surgical Changes

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

## 5. Goal-Driven Execution

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

## 6. The Context Stuffing Trap

**Avoid context rot. Stay consise and focused

When implementing authentication:
- validate inputs, handle errors securely, follow auth/ patterns

# 7. VIBE CODING VALIDATION REQUIREMENTS
All code generated through vibe coding MUST be thoroughly tested and validated before deployment:
### 7.1 Feature Validation
- [ ] Verify ALL requested features are implemented correctly
- [ ] Confirm edge cases are handled appropriately
- [ ] Validate input/output behavior matches specifications
### 7.2 Functionality Testing
- [ ] Execute unit tests for all functions/methods
- [ ] Run integration tests for component interactions
- [ ] Perform end-to-end testing for complete workflows
- [ ] Test error handling and exception scenarios
### 7.3 Operability Verification
- [ ] Confirm code runs in target environment(s)
- [ ] Validate dependencies are properly declared
- [ ] Test performance under expected load
- [ ] Verify logging, monitoring, and observability
- [ ] Ensure graceful degradation and recovery
### 7.4 Code Quality
- [ ] Review for security vulnerabilities
- [ ] Check for code smells and anti-patterns
- [ ] Validate documentation completeness
- [ ] Confirm coding standards compliance

**🚫 DO NOT deploy or merge vibe-coded solutions without completing this checklist.**

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## 8. Self-Improvement Loop
- After ANY correction from the user: update tasks/lessons.md with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

## 9. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

## 10. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

## 11. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

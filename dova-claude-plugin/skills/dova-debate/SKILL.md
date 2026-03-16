---
name: dova-debate
description: Run a structured Bull vs Bear debate on an AI/ML topic using Dova's debate agents
allowed-tools:
  - mcp__dova__dova_debate
---

# Dova Debate

Run a structured debate on the following topic:

**Topic:** $ARGUMENTS

## Instructions

1. Call `dova_debate` with the topic and 2 rounds.
2. Parse the JSON response and present the debate clearly:

### Output Format

**Bull Case (Strengths):**
- List the key strengths and positive arguments

**Bear Case (Concerns):**
- List the key concerns and critical arguments

**Balanced Assessment:**
- Present the synthesized conclusion

**Recommendation:**
- Present the final recommendation with confidence score

3. Keep the presentation balanced — give equal weight to both sides.

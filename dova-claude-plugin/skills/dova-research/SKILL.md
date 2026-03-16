---
name: dova-research
description: Research AI/ML topics across ArXiv papers, GitHub repos, HuggingFace models, and the web using Dova's multi-agent research platform
allowed-tools:
  - mcp__dova__dova_research
  - mcp__dova__dova_search
  - mcp__dova__dova_web_search
---

# Dova Research

Research the following topic using Dova's multi-agent research tools:

**Query:** $ARGUMENTS

## Instructions

1. Call `dova_research` with the query to search across all sources (ArXiv, GitHub, HuggingFace, web).
2. Parse the JSON response and present findings organized by source:

### Output Format

**Papers** (from ArXiv):
- Title, authors, year, key findings, link

**Repositories** (from GitHub):
- Name, description, stars, language, link

**Models & Datasets** (from HuggingFace):
- Name, description, downloads, link

**Web Results**:
- Title, snippet, link

3. End with a brief synthesis of the key themes across all sources.
4. If a specific source returns no results, skip that section silently.

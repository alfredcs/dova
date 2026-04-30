# Bio-Related MCP Servers — Comprehensive Directory

_Compiled 2026-04-29 via live GitHub / web fetches. URLs verified at time of compilation. Items that could not be verified are marked `unverified`._

---

## ⭐ Integrated into DOVA

As of 2026-04-29, DOVA ships with a curated `bio` source that routes queries to the
three **cyanheads hosted Streamable-HTTP endpoints** below. These require no local
install and are reachable from the default runtime:

| DOVA server name | URL | Primary tool | Used when query mentions |
|---|---|---|---|
| `pubmed-bio` | https://pubmed.caseyjhand.com/mcp | `pubmed_search_articles` | PubMed, PMID, PMC, MeSH, literature, MEDLINE, systematic review, meta-analysis (DEFAULT for bio umbrella) |
| `clinicaltrials-bio` | https://clinicaltrials.caseyjhand.com/mcp | `clinicaltrials_search_studies` | clinical trial, NCT, phase I–IV, enrollment, recruiting, cohort, eligibility |
| `pubchem-bio` | https://pubchem.caseyjhand.com/mcp | `pubchem_search_compounds` | compound, SMILES, InChI, chemical formula, CID, bioassay, cheminformatics |

All three are **Apache-2.0 licensed**, public, and require no authentication. They
appear to the orchestrator as a single aggregated `bio` tool (mirroring how the
75+ `awslabs.*` servers are aggregated) so the deliberation LLM sees only
`bio: Biomedical / pharma data (use for: PubMed literature, clinical trials,
drug/chemical compounds, gene/protein/disease queries)` — keyword routing picks
the right backend at execution time. This keeps the deliberation prompt small
and token-efficient while still giving access to the full biomedical surface.

### Why this specific set (and not the other ~80 servers below)

1. **Hosted HTTP endpoints** — zero local install, unlike the overwhelming majority of bio MCP servers which are STDIO-only with `uvx`/`npx`/Conda dependencies.
2. **Live-validated at registration time** — all three endpoints returned HTTP 200 and successful `initialize + tools/list + tools/call` handshakes with real biomed queries (aspirin, BRAF V600E, caffeine).
3. **Non-overlapping coverage** — literature / trials / compounds is the complete biomed-research triangle; adding Ensembl/UniProt/STRING on top would overlap with PubMed citations or require API keys.
4. **Token efficiency** — surfacing one `bio` tool instead of 80+ keeps the deliberation prompt under 2k tokens and avoids confusing the LLM on which server to pick.

### How routing works

Two layers of semantic selection:

1. **Cross-group (deliberation-first)**: when the user does not pre-select sources,
   the ThinkingOrchestrator's deliberation LLM chooses among `AI / Web / Bio` based
   on the query's understanding — not keyword matching. Bio surfaces to the LLM as
   a single tool, keeping the deliberation prompt small.
2. **Within-group (semantic fan-out)**: when `bio` is chosen, a keyword-scored
   multi-select (`_select_bio_servers`) runs every sub-server whose signal is
   positive in parallel. Cross-domain queries ("PubMed review of BRAF trials")
   hit both PubMed and ClinicalTrials simultaneously; single-domain queries
   ("SMILES of aspirin") go to just PubChem.

### How to invoke directly

```python
from dova.tools.research_tools import search_bio_tool

# Auto-route by domain hint
search_bio_tool("GLP-1 agonists in type 2 diabetes", domain="literature")
search_bio_tool("BRAF V600E melanoma", domain="trials")
search_bio_tool("caffeine", domain="compounds")
```

From the frontend, the Dashboard exposes three group checkboxes — **AI**, **Web**,
**Bio**. Ticking **Bio** adds `"bio"` to the request's `sources` list; the
orchestrator then fans out across the three bio endpoints by keyword score.

---

## Table of Contents
- [Scientific Literature & Preprints](#-scientific-literature--preprints)
- [Genome & Gene Databases](#-genome--gene-databases)
- [Gene Ontology & Functional Data](#-gene-ontology--functional-data)
- [Protein Databases & Structures](#-protein-databases--structures)
- [Pathways & Networks](#-pathways--networks)
- [Chemicals & Drug Data](#-chemicals--drug-data)
- [Clinical Trials & Healthcare](#-clinical-trials--healthcare)
- [Expression Data](#-expression-data)
- [Single-Cell & Spatial Analysis](#-single-cell--spatial-analysis)
- [Bioinformatics Tools & Workflows](#-bioinformatics-tools--workflows)
- [Integrated & Utility Servers](#-integrated--utility-servers)
- [Emerging / Specialized](#-emerging--specialized)
- [Enterprise / Commercial](#-enterprise--commercial)
- [Key Hubs & Directories](#key-hubs--directories)
- [Auth patterns at a glance](#auth-patterns-at-a-glance)
- [Items not confirmed](#items-not-confirmed)

---

## 📚 Scientific Literature & Preprints

| Server | Repo | Auth | Notes |
|---|---|---|---|
| **PubMed MCP** (cyanheads) | https://github.com/cyanheads/pubmed-mcp-server | None required; NCBI API key optional for higher rate limits | STDIO + HTTP, Apache 2.0 |
| **PubMed MCP** (JackKuo666) | https://github.com/JackKuo666/PubMed-MCP-Server | None | 108⭐, alternative implementation |
| **medRxiv MCP** (JackKuo666) | https://github.com/JackKuo666/medRxiv-MCP-Server | None (public preprint API) | 7⭐ |
| **bioRxiv MCP** (JackKuo666) | https://github.com/JackKuo666/bioRxiv-MCP-Server | None | 22⭐ |
| **Semantic Scholar MCP** (JackKuo666) | https://github.com/JackKuo666/semanticscholar-MCP-Server | Optional `S2_API_KEY` | 64⭐ |
| **PubTator MCP** (JackKuo666) | https://github.com/JackKuo666/PubTator-MCP-Server | None | 9⭐ |
| **paperscraper MCP** (JackKuo666) | https://github.com/JackKuo666/paperscraper-MCP-Server | None | Scrapes PubMed/arXiv/medRxiv/bioRxiv/ChemRxiv |
| **paperscraper MCP** (MCPmed) | https://github.com/MCPmed/paperscraperMCP | None | Fork of jannisborn/paperscraper |
| **Sci-Hub MCP** (JackKuo666) | https://github.com/JackKuo666/Sci-Hub-MCP-Server | None | 84⭐ — legal gray area |
| **Crossref MCP** (JackKuo666) | https://github.com/JackKuo666/Crossref-MCP-Server | None | 4⭐ |
| **Google Scholar MCP** (JackKuo666) | https://github.com/JackKuo666/Google-Scholar-MCP-Server | None | 299⭐ |
| **OpenAlex MCP** (cyanheads) | https://github.com/cyanheads/openalex-mcp-server | None | 270M+ publications |
| **arXiv MCP** (cyanheads) | https://github.com/cyanheads/arxiv-mcp-server | None | — |

---

## 🧬 Genome & Gene Databases

| Server | Repo | Auth |
|---|---|---|
| **Ensembl MCP** (Augmented Nature) | https://github.com/Augmented-Nature/Ensembl-MCP-Server | None (public REST) |
| **NCBI Datasets MCP** (Augmented Nature) | https://github.com/Augmented-Nature/NCBI-Datasets-MCP-Server | Optional NCBI API key |
| **GTEx MCP** (Augmented Nature) | https://github.com/Augmented-Nature/GTEx-MCP-Server | None |
| **GWAS Catalog MCP** (EBISPOT — official) | https://github.com/EBISPOT/gwas-mcp | None |
| **GWAS MCP** (openpharma) | https://github.com/openpharma-org/gwas-mcp-server | None — JS alternative |
| **GWAS MCP** (muslus) | https://github.com/muslus/gwas-mcp | None |
| **Liftover MCP** | https://github.com/koido/liftover-mcp | None |
| **OpenGenes MCP** (longevity-genie) | https://github.com/longevity-genie/opengenes-mcp | None — 17⭐ |
| **AlphaGenome MCP** (longevity-genie) | https://github.com/longevity-genie/alphagenome-mcp | Google DeepMind AlphaGenome API key |

---

## 🔬 Gene Ontology & Functional Data

| Server | Repo | Auth |
|---|---|---|
| **Gene Ontology MCP** (Augmented Nature) | https://github.com/Augmented-Nature/GeneOntology-MCP-Server | None |
| **BioOntology MCP** (Augmented Nature) | https://github.com/Augmented-Nature/BioOntology-MCP-Server | May require BioPortal API key |

---

## 🧪 Protein Databases & Structures

| Server | Repo | Auth |
|---|---|---|
| **UniProt MCP** (Augmented Nature) | https://github.com/Augmented-Nature/Augmented-Nature-UniProt-MCP-Server | None |
| **EMBL-EBI Protein MCP** (MCPmed) | https://github.com/MCPmed/EMBL-EBI-Protein-mcp | None — BSD-3 |
| **RCSB PDB MCP** (cnyambura) | https://github.com/cnyambura/rcsb-mcp | None — Python |
| **RCSB PDB MCP** (piehld) | https://github.com/piehld/rcsb-pdb-mcp-server | None — JS alternative |
| **PDB MCP** (Augmented Nature) | https://github.com/Augmented-Nature/PDB-MCP-Server | None |
| **PDBe MCP Servers** (PDBe Europe — official) | https://github.com/PDBeurope/PDBe-MCP-Servers | None |
| **Protein MCP** (cyanheads) | https://github.com/cyanheads/protein-mcp-server | None — bundles PDB/PDBe/UniProt |
| **AlphaFold MCP** (Augmented Nature) | https://github.com/Augmented-Nature/AlphaFold-MCP-Server | None |
| **AlphaFold MCP** (openpharma) | https://github.com/openpharma-org/alphafold-mcp-server | None |
| **chimerax-alphafold-mcp** (GDAmitha) | https://github.com/GDAmitha/chimerax-alphafold-mcp | None |
| **Protein Atlas MCP** (Augmented Nature) | https://github.com/Augmented-Nature/ProteinAtlas-MCP-Server | None |
| **PyMOL MCP** (vrtejus) | https://github.com/vrtejus/pymol-mcp | None — 53⭐, requires local PyMOL |
| **PyMOL AI MCP** (Masterchiefm) | https://github.com/Masterchiefm/pymol-ai-MCP | None |
| **ChimeraX MCP** (BenWertoski) | https://github.com/BenWertoski/chimeraX-mcp | None — requires ChimeraX |
| **biotite-mcp** (longevity-genie) | https://github.com/longevity-genie/biotite-mcp | None — wraps biotite lib |
| **atomica-mcp** (longevity-genie) | https://github.com/longevity-genie/atomica-mcp | None — PDB resolution |
| **protein_hunter_mcp** (longevity-genie) | https://github.com/longevity-genie/protein_hunter_mcp | None |

---

## 🕸️ Pathways & Networks

| Server | Repo | Auth |
|---|---|---|
| **Reactome MCP** (Augmented Nature) | https://github.com/Augmented-Nature/Reactome-MCP-Server | None |
| **STRING MCP** (MCPmed) | https://github.com/MCPmed/STRINGmcp | None — BSD-3 |
| **STRING-db MCP** (Augmented Nature) | https://github.com/Augmented-Nature/STRING-db-MCP-Server | None |
| **Oh My KEGG MCP** (martinuslee) | https://github.com/martinuslee/oh-my-kegg-mcp | None — 30+ tools, LangChain |
| **KEGG MCP** (Augmented Nature) | https://github.com/Augmented-Nature/KEGG-MCP-Server | KEGG license applies commercially |
| **KEGG MCP** (Lucas-Servi, Python) | https://github.com/Lucas-Servi/kegg-mcp-server-python | None — 33 tools, TTL caching |
| **KEGG MCP** (openpharma) | https://github.com/openpharma-org/kegg-mcp-server | — |
| **OpenTargets MCP** (Augmented Nature) | https://github.com/Augmented-Nature/OpenTargets-MCP-Server | None — public GraphQL |

---

## 💊 Chemicals & Drug Data

| Server | Repo | Auth |
|---|---|---|
| **PubChem MCP** (cyanheads) | https://github.com/cyanheads/pubchem-mcp-server | None |
| **PubChem MCP** (Augmented Nature) | https://github.com/Augmented-Nature/PubChem-MCP-Server | None — 36⭐ |
| **PubChem MCP** (JackKuo666) | https://github.com/JackKuo666/PubChem-MCP-Server | None |
| **ChEMBL MCP** (Augmented Nature) | https://github.com/Augmented-Nature/ChEMBL-MCP-Server | None — 83⭐ |
| **ChEMBL MCP** (JackKuo666) | https://github.com/JackKuo666/ChEMBL-MCP-Server | None |
| **chembl-mcp** (longevity-genie) | https://github.com/longevity-genie/chembl-mcp | None |
| **SureChEMBL MCP** (Augmented Nature) | https://github.com/Augmented-Nature/SureChEMBL-MCP-Server | None |
| **OpenFDA MCP** (Augmented Nature) | https://github.com/Augmented-Nature/OpenFDA-MCP-Server | Optional `OPENFDA_API_KEY` for rate limits |
| **Pharmacology MCP** (longevity-genie) | https://github.com/longevity-genie/pharmacology-mcp | None — Guide to PHARMACOLOGY |
| **PLSDB MCP** (MCPmed) | https://github.com/MCPmed/PLSDBmcp | None — plasmid DB |

---

## 🏥 Clinical Trials & Healthcare

| Server | Repo | Auth |
|---|---|---|
| **ClinicalTrials.gov MCP** (cyanheads) | https://github.com/cyanheads/clinicaltrialsgov-mcp-server | None — public v2 API |
| **ClinicalTrials MCP** (JackKuo666) | https://github.com/JackKuo666/ClinicalTrials-MCP-Server | None |
| **Healthcare MCP** (Cicatriiz) | https://github.com/Cicatriiz/healthcare-mcp-public | None for public APIs — 113⭐; bundles FDA / PubMed / medRxiv / NCBI Bookshelf / trials |
| **PopHive MCP** (Cicatriiz) | https://github.com/Cicatriiz/pophive-mcp-server | None — Yale PopHIVE + HHS / CDC |
| **CDISC Library MCP** (Cicatriiz) | https://github.com/Cicatriiz/cdisc-library-mcp-server | CDISC Library account token |
| **BioMed MCP** (JackKuo666) | https://github.com/JackKuo666/BioMed-MCP-Server | None |
| **GBD MCP** (JackKuo666) | https://github.com/JackKuo666/GBD-MCP-Server | Global Burden of Disease data |

---

## 📊 Expression Data

| Server | Repo | Auth |
|---|---|---|
| **GEO MCP** (MCPmed) | https://github.com/MCPmed/GEOmcp | Optional NCBI API key — BSD-3 |
| **UCSC Cell Browser MCP** (MCPmed) | https://github.com/MCPmed/UCSCCBmcp | None |

---

## 🧫 Single-Cell & Spatial Analysis

| Server | Repo | Auth |
|---|---|---|
| **Scanpy MCP** (huang-sh) | https://github.com/huang-sh/scanpy-mcp | None — local Python deps |
| **Scanpy MCP** (scmcphub) | https://github.com/scmcphub/scanpy-mcp | None |
| **SCMCP** (huang-sh) | https://github.com/huang-sh/scmcp | None — 12⭐ |
| **SCMCP** (scmcphub) | https://github.com/scmcphub/scmcp | None — 15⭐ |
| **scmcp-shared** | https://github.com/scmcphub/scmcp-shared | Shared library |
| **ChatSpatial** (cafferychen777) | https://github.com/cafferychen777/ChatSpatial | None — 32⭐, 60+ spatial methods |
| **cellxgene MCP** (longevity-genie) | https://github.com/longevity-genie/cellxgene-mcp | None |
| **PBMCpedia MCP** (MCPmed) | https://github.com/MCPmed/PBMCpediaMCP | None |

---

## 🛠️ Bioinformatics Tools & Workflows

### `bio-mcp` org (https://github.com/bio-mcp)
All Python, `pip install -e .`, no auth, Conda-installed tools.

| Server | Repo |
|---|---|
| bio-mcp-blast | https://github.com/bio-mcp/bio-mcp-blast |
| bio-mcp-bwa | https://github.com/bio-mcp/bio-mcp-bwa |
| bio-mcp-samtools | https://github.com/bio-mcp/bio-mcp-samtools |
| bio-mcp-seqkit | https://github.com/bio-mcp/bio-mcp-seqkit |
| bio-mcp-queue | https://github.com/bio-mcp/bio-mcp-queue (Redis + Celery + MinIO) |
| bio-mcp-bcftools | https://github.com/bio-mcp/bio-mcp-bcftools |
| bio-mcp-bedtools | https://github.com/bio-mcp/bio-mcp-bedtools |
| bio-mcp-fastqc | https://github.com/bio-mcp/bio-mcp-fastqc |
| bio-mcp-interpro | https://github.com/bio-mcp/bio-mcp-interpro |
| bio-mcp-amber | https://github.com/bio-mcp/bio-mcp-amber |
| bio-mcp-evo2 | https://github.com/bio-mcp/bio-mcp-evo2 |
| bio-mcp-template | https://github.com/bio-mcp/bio-mcp-template |

### Other workflow / tool servers

| Server | Repo | Auth |
|---|---|---|
| **Galaxy MCP** (official) | https://github.com/galaxyproject/galaxy-mcp | Galaxy server URL + API key |
| **Bioconda MCP** (JackKuo666) | https://github.com/JackKuo666/Bioconda-MCP-Server | None |
| **nf-core MCP** (wjlim) | https://github.com/wjlim/nf-core_mcp | None — dev assistant |
| **BioinfoMCP** (florensiawidjaja) | https://github.com/florensiawidjaja/BioinfoMCP | None — auto-converts CLI bioinfo tools to MCP |
| **Bio-OS MCP** | `unverified` — no public repo found under that exact name |
| **grobid MCP** (JackKuo666) | https://github.com/JackKuo666/grobid-MCP-Server | Requires local GROBID |
| **pdffigures2 MCP** (JackKuo666) | https://github.com/JackKuo666/pdffigures2-MCP-Server | None |
| **pydna-mcp** (longevity-genie) | https://github.com/longevity-genie/pydna-mcp | None |

---

## 🧩 Integrated & Utility Servers

| Server | Repo | Auth |
|---|---|---|
| **Biotools MCP** (BACH-AI-Tools) | https://github.com/BACH-AI-Tools/biotools-mcp-server | None — TypeScript, integrates PubMed / UniProt / GenBank / KEGG / PDB |
| **BioThings MCP** (Augmented Nature) | https://github.com/Augmented-Nature/BioThings-MCP-Server | None — MyGene / MyVariant |
| **biothings-mcp** (longevity-genie) | https://github.com/longevity-genie/biothings-mcp | None — 31⭐ |
| **Holy Bio MCP** (longevity-genie) | https://github.com/longevity-genie/holy-bio-mcp | Aggregates longevity-genie agentic biotools |
| **gget-mcp** (longevity-genie) | https://github.com/longevity-genie/gget-mcp | None — wraps `gget` — 27⭐ |
| **synergy-age-mcp** (longevity-genie) | https://github.com/longevity-genie/synergy-age-mcp | None — 10⭐ |
| **BioMCP-Hub** (JackKuo666) | https://github.com/JackKuo666/BioMCP-Hub | Modular bio platform |
| **biomcp** (abhishektiwari — unrelated to GenomOncology) | https://github.com/abhishektiwari/biomcp | None — gateway / server |
| **biomodels-mcp** (longevity-genie) | https://github.com/longevity-genie/biomodels-mcp | None |

---

## 🆕 Emerging / Specialized

| Server | Repo | Auth |
|---|---|---|
| **FutureHouse MCP** / `edison_mcp` (longevity-genie) | https://github.com/longevity-genie/edison_mcp | FutureHouse API key |
| **Addgene MCP** (longevity-genie) | https://github.com/longevity-genie/addgene-mcp | None — alpha |
| **Boltz MCP** (longevity-genie) | https://github.com/longevity-genie/boltz-mcp | None — runs Boltz-2 locally, alpha |
| **Benchling MCP** (longevity-genie) | https://github.com/longevity-genie/benchling-mcp | Benchling API key + tenant URL — alpha |
| **cell2sentence4longevity-mcp** (longevity-genie) | https://github.com/longevity-genie/cell2sentence4longevity-mcp | None |

---

## 🏢 Enterprise / Commercial

| Server | Repo | Auth / Licensing |
|---|---|---|
| **BioMCP** (GenomOncology — official) | https://github.com/genomoncology/biomcp | **MIT open-source**; 497⭐, Rust binary. Optional keys: `NCBI_API_KEY`, `S2_API_KEY`, `ONCOKB_TOKEN`, `DISGENET_API_KEY`, `OPENFDA_API_KEY`, `NCI_API_KEY`, `ALPHAGENOME_API_KEY`. Install: `curl -fsSL https://biomcp.org/install.sh \| bash` or `uv tool install biomcp-cli`. Covers 13 entity types (gene / variant / article / trial / drug / disease / pathway / protein / AE / PGx / GWAS / diagnostic / phenotype). |
| **biomcp-examples** | https://github.com/genomoncology/biomcp-examples | MIT — example workflows |
| **biomcp-trialgpt** | https://github.com/genomoncology/biomcp-trialgpt | MIT |
| **OncoMCP** (GenomOncology) | No public repo located. GenomOncology positions it as a commercial precision-oncology product layered on BioMCP; `ONCOKB_TOKEN` is the evidence-provider key. Treat as commercial product, not OSS. |

---

## Key Hubs & Directories

| Resource | URL |
|---|---|
| **MCPmed** (Saarland University Clinical Bioinformatics initiative) | https://mcpmed.org · https://github.com/MCPmed |
| **MCPmed cookiecutter template** | https://github.com/MCPmed/Cookiecutter-MCPmed |
| **Augmented Nature** (author org, 24+ bio MCP repos) | https://github.com/Augmented-Nature |
| **longevity-genie** (21+ bio MCP repos) | https://github.com/longevity-genie |
| **bio-mcp** (workflow tool collection) | https://github.com/bio-mcp |
| **scmcphub** (single-cell) | https://github.com/scmcphub |
| **JackKuo666** (literature / preprint collection) | https://github.com/JackKuo666 |
| **cyanheads** (bio-adjacent) | https://github.com/cyanheads |

---

## Auth patterns at a glance

- **No auth required** (vast majority): public REST / GraphQL APIs — PubMed, PDB, UniProt, Reactome, STRING, Ensembl, ClinicalTrials.gov, GWAS Catalog, Gene Ontology, medRxiv / bioRxiv, GTEx, AlphaFold, Scanpy / SCMCP (local), etc.
- **Optional API key for rate limits**: NCBI (`NCBI_API_KEY`), OpenFDA (`OPENFDA_API_KEY`), Semantic Scholar (`S2_API_KEY`).
- **Required API key / account**: Benchling (tenant + token), Galaxy (server URL + key), OncoKB via BioMCP (`ONCOKB_TOKEN`), DisGeNET (`DISGENET_API_KEY`), NCI CTS (`NCI_API_KEY`), AlphaGenome (`ALPHAGENOME_API_KEY`), FutureHouse (via `edison_mcp`).
- **Commercial / private**: OncoMCP (GenomOncology).
- **License-sensitive data**: KEGG (free for academic, paid commercial — per KEGG terms, even though the MCP servers are OSS).

---

## Items not confirmed

- **Bio-OS MCP** — no matching public repo surfaced in search
- **OncoMCP** public repo — appears to be a closed commercial product, not on GitHub
- ~~Exact AlphaFold / SureChEMBL repo paths under `Augmented-Nature`~~ — **confirmed 2026-04-29**: both `AlphaFold-MCP-Server` (34⭐) and `SureChEMBL-MCP-Server` (7⭐) exist at the expected canonical paths
- **Augmented-Nature UniProt MCP** — the short path `Augmented-Nature/UniProt-MCP-Server` redirects (301) to `Augmented-Nature/Augmented-Nature-UniProt-MCP-Server` (the actual repo name, 19⭐). Use the full name.
- MCPHub (certification framework) — site at mcphub.com timed out during fetch

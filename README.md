# AI Notebook Builder

A tool that analyzes existing **DSEP / El Camino** Jupyter notebooks and helps generate new starter notebooks using corpus patterns.

Curriculum developers can learn from successful notebooks in the corpus, search for examples, plan new lessons, and export runnable starter `.ipynb` files — without copying notebooks by hand.

---

## The problem

Building educational notebooks takes repeated work:

- Choosing a teaching structure (introduction, objectives, exercises, reflection)
- Writing consistent markdown and section headings
- Scaffolding code cells for students
- Adding widgets and visualizations
- Checking that notebooks are complete and usable before review

Teams often recreate the same patterns notebook after notebook. **AI Notebook Builder** reduces that repetition by learning from an existing notebook corpus and turning those patterns into outlines, drafts, exports, and validation checks.

---

## Features

- **Parse existing notebooks** — extract headings, code, exercises, widgets, and metadata
- **Analyze curriculum patterns** — summarize teaching structures across the corpus
- **Search prior notebook examples** — retrieve and synthesize examples with Notebook Copilot
- **Generate notebook outlines** — plan a new notebook from user inputs
- **Draft reusable notebook sections** — create section-level markdown drafts
- **Export starter `.ipynb` notebooks** — convert outlines or drafts into Jupyter notebooks
- **Validate generated notebooks** — check structure, syntax, and quality before review

---

## Project workflow

```
notebooks → parser → corpus analysis → retrieval/copilot → outline generator → notebook export → validation
```

```mermaid
flowchart LR
    A[data/notebooks] --> B[parse_notebooks.py]
    B --> C[analyze_corpus.py]
    C --> D[notebook_copilot.py]
    D --> E[generate_notebook_outline.py]
    E --> F[export_to_ipynb.py]
    F --> G[validate_generated_notebook.py]
    G --> H[starter_notebook.ipynb]
```

**Typical path:** run the full pipeline with one command, or use individual scripts for search, drafting, and export.

---

## Project structure

```
ai-notebook-builder/
├── data/
│   ├── notebooks/       # Source .ipynb files (read-only for the parser)
│   ├── parsed/          # Parsed text, metadata, curriculum report
│   ├── generated/       # Outlines, drafts, exported notebooks, validation
│   └── embeddings/      # Local search index for Notebook Copilot
├── scripts/             # All pipeline and utility scripts
└── README.md
```

---

## Quick start

1. Place reference notebooks in `data/notebooks/`.
2. Run the full pipeline:

```bash
python3 scripts/run_pipeline.py
```

Step 3 asks interactive questions (discipline, topic, dataset, coding level, widgets, reflection, length). The pipeline stops if any required step fails.

**Outputs:**

- `data/generated/notebook_outline.md`
- `data/generated/starter_notebook.ipynb`
- `data/generated/validation_report.md`

---

## Individual commands

```bash
# Parse and analyze the corpus
python3 scripts/parse_notebooks.py
python3 scripts/analyze_corpus.py

# Search and draft from prior notebooks
python3 scripts/notebook_copilot.py build   # first time only
python3 scripts/notebook_copilot.py "Show me examples of widget usage"
python3 scripts/notebook_copilot.py --draft "Create a beginner widget section for a calculus notebook on Riemann sums"

# Plan, export, and validate a new notebook
python3 scripts/generate_notebook_outline.py
python3 scripts/export_to_ipynb.py --outline
python3 scripts/validate_generated_notebook.py
```

On macOS, use `python3` (not `python`).

---

## Example output

A generated **calculus derivatives and integrals** notebook (no dataset) includes:

- **Title and learning objectives** for exploring derivatives and integrals with interactive visualizations
- **Runnable setup code** — `numpy`, `matplotlib`, `ipywidgets`, `interact`
- **Function plotting** — defines `f(x) = x²` and plots it over an interval
- **Derivative visualization** — tangent line at a chosen x-value with a slider
- **Integral visualization** — Riemann sum rectangles with sliders for count and interval bounds
- **Student prompts** — “Try changing…” and “What do you notice?” after interactive cells

The validation report checks structure, Python syntax (without executing code), and flags issues like dataset language when no dataset is used.

---

## Current limitations

- **Rule-based generation** — outlines, drafts, and exports use templates and corpus patterns, not an LLM
- **Small notebook corpus** — patterns are learned from a limited set of DSEP / El Camino notebooks
- **No LLM API yet** — retrieval uses local TF-IDF embeddings by default (optional semantic embeddings via `requirements-copilot.txt`)
- **Human review still required** — generated notebooks are starters, not finished curriculum

---

## Future improvements

- Add LLM / RAG generation for richer outlines and section drafts
- Add a Streamlit or web interface for non-technical users
- Support more notebook templates (biology EDA, case studies, coding labs, ethics)
- Improve dataset-aware generation and discipline-specific exports
- Execute notebooks during validation to catch runtime errors
- Export separate instructor and student versions

---

## Requirements

- Python 3.10+
- **numpy** (Notebook Copilot embeddings)

Optional:

```bash
pip install -r requirements-copilot.txt
```

Installs `sentence-transformers` for semantic search in Notebook Copilot.

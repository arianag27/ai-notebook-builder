#!/usr/bin/env python3
"""
AI notebook generator using Retrieval-Augmented Generation (RAG).

Retrieves relevant chunks from the vector index, builds a structured prompt,
and generates a notebook draft with a local Ollama model (default).

Usage:
    python3 scripts/build_vector_index.py   # first time (or after syncing notebooks)
    python3 scripts/llm_generator.py
    python3 scripts/llm_generator.py --model llama3.1:8b

Output:
    data/generated/llm_notebook_draft.md

Requires Ollama running locally (default: qwen3:8b at http://localhost:11434).
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from providers.factory import get_llm_provider, provider_label
from providers.ollama_provider import OllamaProvider
import rag_utils

GENERATED_DIR = PROJECT_ROOT / "data" / "generated"
OUTPUT_PATH = GENERATED_DIR / "llm_notebook_draft.md"

NO_DATA_PHRASES = {
    "",
    "no data",
    "none",
    "n/a",
    "na",
    "not using data",
    "no dataset",
    "not applicable",
}

# Retrieval queries keyed by section type
RETRIEVAL_QUERIES = [
    ("learning objectives", "learning objectives goals outcomes"),
    ("introduction", "introduction background context real world"),
    ("widget examples", "widget interactive ipywidgets slider interact"),
    ("coding exercises", "coding exercise scaffold hint your turn"),
    ("reflection questions", "reflection discussion question free response"),
    ("dataset overview", "dataset columns preview load data"),
    ("visualization", "plot chart visualization matplotlib"),
]

MAX_EXAMPLES_IN_PROMPT = 12
MAX_CHARS_PER_EXAMPLE = 1000


def prompt_user() -> dict:
    """Collect notebook requirements from the user."""
    print("\n=== AI Notebook Builder — LLM Generator ===\n")
    print("Answer the prompts below. Examples from the corpus will guide generation.\n")
    print("(Enter 'no data' if the notebook does not use an external dataset.)\n")

    discipline = input("Subject / discipline (e.g. Biology, Calculus): ").strip()
    topic = input("Topic (e.g. wastewater surveillance): ").strip()
    dataset = input("Dataset description (or 'no data'): ").strip()
    learning_goals = input(
        "Learning goals (optional, brief description): "
    ).strip()

    coding_level = ""
    while coding_level not in ("beginner", "intermediate", "advanced"):
        coding_level = input(
            "Coding level (beginner / intermediate / advanced): "
        ).strip().lower()

    wants_widgets = input("Include widgets? (yes / no): ").strip().lower() in ("yes", "y")
    wants_reflection = input(
        "Include reflection questions? (yes / no): "
    ).strip().lower() in ("yes", "y")

    length = ""
    while length not in ("short", "medium", "long"):
        length = input(
            "Approximate length (short ~30min / medium ~60min / long ~90min): "
        ).strip().lower()

    return {
        "discipline": discipline or "General",
        "topic": topic or "Untitled Topic",
        "dataset": dataset,
        "learning_goals": learning_goals,
        "coding_level": coding_level,
        "wants_widgets": wants_widgets,
        "wants_reflection": wants_reflection,
        "length": length,
        "uses_dataset": dataset.strip().lower() not in NO_DATA_PHRASES,
    }


def ensure_index_exists() -> None:
    """Make sure the vector index is available."""
    chunks, embeddings, _ = rag_utils.load_index()
    if chunks and embeddings.size:
        return

    print("Vector index not found. Building it now...\n")
    from build_vector_index import build_index

    build_index(use_openai=False)


def parse_cli_args() -> dict:
    """Parse optional --model and --provider flags."""
    args = {"model": None, "provider": None}
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        if argv[i] == "--model" and i + 1 < len(argv):
            args["model"] = argv[i + 1]
            i += 2
        elif argv[i] == "--provider" and i + 1 < len(argv):
            args["provider"] = argv[i + 1]
            i += 2
        else:
            i += 1
    return args


def retrieve_examples(requirements: dict) -> list[dict]:
    """Retrieve relevant notebook chunks for the user's request."""
    discipline = requirements["discipline"]
    topic = requirements["topic"]
    coding_level = requirements["coding_level"]

    base_query = f"{discipline} {topic} {coding_level}"
    seen_keys: set[str] = set()
    results: list[dict] = []

    query_plan = list(RETRIEVAL_QUERIES)
    if not requirements["uses_dataset"]:
        query_plan = [q for q in query_plan if q[0] != "dataset overview"]
    if not requirements["wants_widgets"]:
        query_plan = [q for q in query_plan if q[0] != "widget examples"]
    if not requirements["wants_reflection"]:
        query_plan = [q for q in query_plan if q[0] != "reflection questions"]

    for section_type, keywords in query_plan:
        query = f"{base_query} {keywords}"
        hits = rag_utils.search_chunks(query, top_k=3, discipline=discipline)
        for hit in hits:
            key = f"{hit['notebook']}::{hit['section']}"
            if key in seen_keys:
                continue
            seen_keys.add(key)
            hit["retrieval_type"] = section_type
            results.append(hit)
            if len(results) >= MAX_EXAMPLES_IN_PROMPT:
                return results

    return results


def format_examples_for_prompt(examples: list[dict]) -> str:
    """Format retrieved chunks for inclusion in the LLM prompt."""
    if not examples:
        return "(No examples retrieved — generate from best practices.)"

    blocks = []
    for i, example in enumerate(examples, start=1):
        text = rag_utils.truncate_text(example["text"], MAX_CHARS_PER_EXAMPLE)
        blocks.append(
            f"### Example {i}\n"
            f"- Notebook: {example['notebook']}\n"
            f"- Discipline: {example.get('discipline', 'Unknown')}\n"
            f"- Section type: {example.get('section', example.get('retrieval_type', ''))}\n"
            f"- Content:\n{text}\n"
        )
    return "\n".join(blocks)


def build_length_requirements(requirements: dict) -> str:
    """Return strict length-specific structure rules for the LLM prompt."""
    length = requirements["length"]
    wants_reflection = requirements["wants_reflection"]
    wants_widgets = requirements["wants_widgets"]

    reflection_short = (
        "- Include a brief `## Reflection Questions` section (2–3 prompts)."
        if wants_reflection
        else "- Reflection is optional for short notebooks; include only if it fits naturally."
    )
    reflection_medium = (
        "- Include a `## Reflection Questions` section with open-ended prompts."
        if wants_reflection
        else "- Do not include a reflection section."
    )
    reflection_long = (
        "- Include a detailed `## Reflection Questions` section with multiple open-ended "
        "and discussion-style prompts."
        if wants_reflection
        else "- Do not include a reflection section."
    )
    widget_note = (
        "- Include at least one interactive widget section with ipywidgets."
        if wants_widgets
        else "- Do not include widget sections."
    )

    guides = {
        "short": f"""### SHORT notebook requirements (strict)

**Major sections:** exactly **4** major lesson sections (use `##` headings such as `## 1. ...` through `## 4. ...`).

**Per-section content:**
- Each major section must include **at least one learning activity** (explanation, guided task, or prompt).
- Include **at least 1** ````python` code activity somewhere in the notebook.
- Keep sections concise but complete — not a bare outline.

**Structure:**
- Use `###` subsections only when needed; subsections are optional for short notebooks.
- {widget_note}
{reflection_short}
- `## Extension Activities` is optional for short notebooks.

**Do not** produce fewer than 4 major lesson sections.""",

        "medium": f"""### MEDIUM notebook requirements (strict)

**Major sections:** exactly **6** major lesson sections (use `##` headings such as `## 1. ...` through `## 6. ...`).

**Per-section content:**
- **Some** major sections must include `###` subsections (at least 3 sections should have subsections).
- Include **multiple** code and/or visualization activities (at least 3 ````python` blocks total).
- Include **at least one** dedicated student practice section (e.g. `## Guided Practice` or a section titled with "Your Turn" / "Practice") with scaffolded exercises.
- Each major section should include at least one learning activity.

**Structure:**
- {widget_note}
{reflection_medium}
- Include `## Extension Activities` with 2–3 optional stretch tasks.

**Do not** generate a short outline. Medium means 6 full major sections with substantive content.""",

        "long": f"""### LONG notebook requirements (strict)

**Major sections:** **6 to 8** major lesson sections (use `##` headings such as `## 1. ...` through `## 6–8. ...`).

**Per-section content:**
- **Most** major sections (at least 5 of them) must include `###` subsections.
- Include **multiple** coding activities (at least 4 ````python` blocks total).
- Include **scaffolded student exercises** with hints, starter code, and `# TODO` placeholders in at least 2 sections.
- Each major section must include at least one learning activity.

**Structure:**
- {widget_note}
{reflection_long}
- Include `## Extension Activities` with 3+ optional stretch tasks.

**Do not** generate a short or medium-length outline. Long means 6–8 full major sections with deep scaffolding.""",
    }

    return guides[length]


def build_generation_prompt(requirements: dict, examples: list[dict]) -> str:
    """Build the full prompt sent to the LLM."""
    widget_line = "Yes — include interactive widget activities" if requirements["wants_widgets"] else "No widgets"
    reflection_line = (
        "Yes — include reflection questions"
        if requirements["wants_reflection"]
        else "No reflection questions"
    )
    dataset_line = (
        requirements["dataset"]
        if requirements["uses_dataset"]
        else "No external dataset — use conceptual or simulated activities only"
    )
    goals_line = requirements["learning_goals"] or "Infer appropriate goals from the topic"
    length_requirements = build_length_requirements(requirements)

    return f"""You are helping generate DSEP / El Camino educational Jupyter notebooks.
Write in a clear, student-friendly style similar to the retrieved examples.
Use markdown headings and include runnable Python code in fenced ```python blocks.

## Retrieved Examples from the Corpus

Study these real notebook sections for structure, tone, and pacing:

{format_examples_for_prompt(examples)}

## User Request

- Discipline: {requirements['discipline']}
- Topic: {requirements['topic']}
- Dataset: {dataset_line}
- Coding level: {requirements['coding_level']}
- Learning goals: {goals_line}
- Widgets: {widget_line}
- Reflection: {reflection_line}
- **Selected length: {requirements['length'].upper()}** — you MUST follow the length rules below

## Length Requirements (MANDATORY)

{length_requirements}

**Critical:** Do not generate a short outline when the user selected **medium** or **long**.
Match the selected length exactly. A medium or long notebook must have the required number
of major `##` lesson sections with real content, not brief bullet-point stubs.

## Your Task

Generate a complete notebook draft in Markdown with this structure:

1. `#` Title (one compelling title line)
2. `## Introduction` — context and why the topic matters
3. `## Learning Objectives` — bullet list using "By the end of this notebook, you will..."
4. `## Suggested Libraries` — bullet list of imports students will need
5. **Major lesson sections** — follow the length requirements above (`## 1. ...`, `## 2. ...`, etc.)
6. Include ````python` code blocks for coding activities matched to {requirements['coding_level']} level
7. End with reflection and/or extension sections as required by the length rules

Important rules:
- Adapt patterns from the retrieved examples; do not copy them verbatim
- Use placeholders like `# TODO` or `*Your answer here*` where students should fill in work
- Keep code cells runnable where possible (imports first, then activities)
- Count major lesson sections carefully — Introduction, Learning Objectives, Suggested Libraries,
  Reflection, and Extension do NOT count toward the major-section total
- Do NOT include an "Influenced By" section — that will be added separately
- Output only the notebook draft markdown, no preamble or explanation
"""


def unique_notebook_names(examples: list[dict]) -> list[str]:
    """Collect unique source notebook filenames for citations."""
    seen: set[str] = set()
    names: list[str] = []
    for example in examples:
        notebook = example.get("notebook", "")
        if notebook and notebook not in seen:
            seen.add(notebook)
            names.append(notebook)
    return sorted(names)


def append_influenced_by(draft: str, notebook_names: list[str]) -> str:
    """Add a source citation section to the generated draft."""
    if "## Influenced By" in draft:
        return draft

    lines = [draft.rstrip(), "", "## Influenced By", ""]
    if notebook_names:
        for name in notebook_names:
            lines.append(f"- {name}")
    else:
        lines.append("- (no specific corpus notebooks cited)")
    return "\n".join(lines) + "\n"


def write_draft(content: str, requirements: dict, examples: list[dict]) -> Path:
    """Save the generated draft with a short header."""
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    header = (
        f"# LLM Notebook Draft\n\n"
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}\n"
        f"**Discipline:** {requirements['discipline']}\n"
        f"**Topic:** {requirements['topic']}\n"
        f"**Coding level:** {requirements['coding_level']}\n"
        f"**Retrieved examples:** {len(examples)}\n\n"
        f"---\n\n"
    )

    OUTPUT_PATH.write_text(header + content, encoding="utf-8")
    return OUTPUT_PATH


def main() -> int:
    try:
        cli = parse_cli_args()
        requirements = prompt_user()
        ensure_index_exists()

        print("\nRetrieving relevant notebook examples...")
        examples = retrieve_examples(requirements)
        print(f"  Found {len(examples)} example chunks from the corpus")

        if not examples:
            print("WARNING: No examples retrieved. Generation will rely on general instructions.")

        prompt = build_generation_prompt(requirements, examples)

        provider = get_llm_provider(provider_name=cli["provider"], model=cli["model"])
        label = provider_label(provider)
        print(f"\nGenerating notebook draft with {label}...")

        if isinstance(provider, OllamaProvider):
            provider.check_ready()

        draft = provider.generate(prompt)
        if not draft:
            print("ERROR: The model returned an empty response.")
            return 1

        influenced = unique_notebook_names(examples)
        draft = append_influenced_by(draft, influenced)
        output_path = write_draft(draft, requirements, examples)

        print("\n=== Generation Complete ===")
        print(f"Draft saved to: {output_path}")
        print(f"Influenced by {len(influenced)} notebook(s):")
        for name in influenced[:8]:
            print(f"  - {name}")
        if len(influenced) > 8:
            print(f"  ... and {len(influenced) - 8} more")
        print("\nNext step:")
        print("  python3 scripts/export_to_ipynb.py --llm")
        return 0

    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    except ImportError as exc:
        print(f"ERROR: {exc}")
        return 1
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

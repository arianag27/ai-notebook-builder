#!/usr/bin/env python3
"""
Parse Jupyter notebooks into clean text files and metadata for RAG / embeddings.

Reads .ipynb files from data/notebooks/ and writes parsed output to data/parsed/.
Does not modify original notebooks.
"""

import json
import re
import sys
from pathlib import Path

# --- Paths (relative to project root) ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = PROJECT_ROOT / "data" / "notebooks"
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"

# --- Section labels used for structure / pattern detection ---

SECTION_LABELS = [
    "introduction",
    "learning objectives",
    "table of contents",
    "dataset",
    "dataset overview",
    "data overview",
    "background",
    "background reading",
    "visualization",
    "analysis",
    "exercise",
    "coding exercise",
    "question",
    "reflection",
    "discussion",
    "conclusion",
    "hint",
    "answer",
    "solution",
    "extension",
    "widget",
    "interactive",
    "explore",
    "sources",
]

# Keywords that suggest real-world context
REAL_WORLD_KEYWORDS = [
    "case study",
    "real-world",
    "real world",
    "industry",
    "policy",
    "public health",
    "environmental",
    "incarceration",
    "housing",
    "marketing",
    "biomanufacturing",
    "cancer",
    "ethics",
    "california",
    "county",
    "population",
]

# Topic hints for discipline guessing (keyword -> topic)
TOPIC_KEYWORDS = {
    "ethics": "Ethics / Data Science",
    "trolley": "Ethics / Philosophy",
    "housing": "Economics / Housing",
    "incarceration": "Social Science / Criminal Justice",
    "riemann": "Mathematics / Calculus",
    "orbital": "Chemistry / Physics",
    "enzyme": "Biology / Biomanufacturing",
    "cancer": "Biology / Genomics",
    "marketing": "Business / Marketing",
    "calenviro": "Environmental Science",
    "parental": "Public Health / Epidemiology",
    "mutation": "Biology / Genomics",
    "sum": "Mathematics",
}


def get_cell_text(cell: dict) -> str:
    """Turn a cell's source field into a single string."""
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def load_notebook(path: Path) -> dict:
    """Load a notebook JSON file. Raises on invalid JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_title(markdown_texts: list[str], filename: str) -> str:
    """Get notebook title from the first top-level heading, or fall back to filename."""
    for text in markdown_texts:
        match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if match:
            title = match.group(1).strip()
            # Skip generic section headers
            if title.lower() not in ("table of contents", "learning objectives"):
                return title
    return Path(filename).stem.replace("_", " ")


def extract_headings(all_text: str) -> list[str]:
    """Find markdown headings (# through ####)."""
    headings = []
    for line in all_text.splitlines():
        match = re.match(r"^(#{1,4})\s+(.+)$", line.strip())
        if match:
            # Strip HTML tags from heading text
            text = re.sub(r"<[^>]+>", "", match.group(2)).strip()
            if text:
                headings.append(text)
    return headings


def extract_section(label: str, all_text: str, max_chars: int = 3000) -> list[str]:
    """
    Pull blocks of text that mention a section label (e.g. 'learning objectives').
    Returns matching paragraphs / nearby content.
    """
    results = []
    pattern = re.compile(
        rf"(?:^|\n)(?:#{{1,4}}\s+.*{re.escape(label)}.*|.*\b{re.escape(label)}\b.*)",
        re.IGNORECASE | re.MULTILINE,
    )
    for match in pattern.finditer(all_text):
        start = match.start()
        # Grab a chunk after the match (until next major heading or limit)
        chunk = all_text[start : start + 800]
        next_heading = re.search(r"\n#{1,2}\s+", chunk[50:])
        if next_heading:
            chunk = chunk[: 50 + next_heading.start()]
        chunk = chunk.strip()
        if chunk and chunk not in results:
            results.append(chunk[:max_chars])
    return results


def extract_learning_objectives(all_text: str) -> list[str]:
    """Find learning objectives sections or bullet lists under that heading."""
    objectives = extract_section("learning objectives", all_text)
    if not objectives:
        # Look for "By the end of this notebook" style blocks
        match = re.search(
            r"(?:By the end|In this (?:lesson|notebook|lab)).*?(?=\n#{1,2}\s|\Z)",
            all_text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            objectives.append(match.group(0).strip()[:2000])
    return objectives


def extract_dataset_descriptions(all_text: str) -> list[str]:
    """Find dataset / data overview content."""
    descriptions = []
    for label in ("dataset overview", "dataset", "data overview", "overview of data", "load the data"):
        descriptions.extend(extract_section(label, all_text))
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for d in descriptions:
        key = d[:100]
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def extract_exercises(all_text: str) -> list[str]:
    """Find exercise and question prompts."""
    exercises = []
    patterns = [
        r"(?:^|\n)(?:#{1,4}\s+)?(?:\*\*)?(?:Question|Exercise|Free Response)[^*\n]*(?:\*\*)?[:\s].*",
        r"<h2>\s*Question\s+\d+.*?</h2>",
        r"###\s*<span[^>]*>Question\s+\w+</span>",
    ]
    for pat in patterns:
        for match in re.finditer(pat, all_text, re.IGNORECASE | re.MULTILINE):
            # Grab the prompt and a bit of context
            start = match.start()
            chunk = all_text[start : start + 500].strip()
            if chunk not in exercises:
                exercises.append(chunk)
    return exercises


def extract_hints(all_text: str) -> list[str]:
    """Find hint text."""
    hints = []
    for match in re.finditer(
        r"(?:\*Hint:\*|\*\*Hint:\*\*|#\s*Hint|Hint:)\s*.+",
        all_text,
        re.IGNORECASE,
    ):
        hints.append(match.group(0).strip()[:500])
    return hints


def extract_answers(all_text: str, code_texts: list[str]) -> list[str]:
    """Find answer / solution content in markdown and code."""
    answers = []
    for match in re.finditer(
        r"(?:^|\n)(?:#{1,4}\s+)?(?:\*\*)?(?:Answer|Solution)[^*\n]*(?:\*\*)?[:\s].*",
        all_text,
        re.IGNORECASE | re.MULTILINE,
    ):
        answers.append(all_text[match.start() : match.start() + 400].strip())

    if re.search(r"your answer here", all_text, re.IGNORECASE):
        answers.append("[Placeholder: 'Your Answer Here' cells found]")

    # Otter grader cells often hold hidden answers
    for code in code_texts:
        if "grader.check" in code or "otter" in code.lower():
            answers.append("[Otter autograder checks present in code cells]")

    return answers


def extract_reflections(all_text: str) -> list[str]:
    """Find reflection and discussion prompts."""
    reflections = []
    patterns = [
        r"(?:\*\*Reflect\*\*|Reflection Prompt|reflection|discussion)[:\s].*",
        r"##\s+.*[Rr]eflection.*",
        r"Free Response Questions? and Reflections?",
    ]
    for pat in patterns:
        for match in re.finditer(pat, all_text, re.IGNORECASE | re.MULTILINE):
            chunk = all_text[match.start() : match.start() + 600].strip()
            if chunk not in reflections:
                reflections.append(chunk)
    return reflections


def extract_imports(code_texts: list[str]) -> list[str]:
    """Collect import statements from code cells."""
    imports = []
    import_pattern = re.compile(
        r"^(?:import |from \S+ import ).+$", re.MULTILINE
    )
    for code in code_texts:
        for match in import_pattern.finditer(code):
            line = match.group(0).strip()
            if line not in imports:
                imports.append(line)
    return imports


def detect_library_usage(code_texts: list[str]) -> dict[str, bool]:
    """Check which common libraries appear in code."""
    combined = "\n".join(code_texts).lower()
    return {
        "uses_pandas": "pandas" in combined or "import pd" in combined,
        "uses_numpy": "numpy" in combined or "import np" in combined,
        "uses_matplotlib": "matplotlib" in combined,
        "uses_seaborn": "seaborn" in combined,
        "uses_plotly": "plotly" in combined,
        "uses_ipywidgets": "ipywidgets" in combined or "interact(" in combined,
    }


def extract_visualization_patterns(code_texts: list[str]) -> list[str]:
    """Note plotting / chart patterns in code."""
    patterns = []
    viz_keywords = [
        ("plt.", "matplotlib pyplot calls"),
        ("sns.", "seaborn calls"),
        ("px.", "plotly express"),
        ("go.Figure", "plotly graph objects"),
        (".plot(", "pandas/matplotlib .plot()"),
        ("histogram", "histogram"),
        ("bar(", "bar chart"),
        ("scatter", "scatter plot"),
        ("heatmap", "heatmap"),
        ("subplot", "subplots"),
    ]
    combined = "\n".join(code_texts)
    for keyword, label in viz_keywords:
        if keyword in combined and label not in patterns:
            patterns.append(label)
    return patterns


def extract_widget_patterns(code_texts: list[str], all_text: str) -> list[str]:
    """Note interactive / widget patterns."""
    patterns = []
    combined = "\n".join(code_texts)
    checks = [
        ("ipywidgets", "ipywidgets library"),
        ("interact(", "@interact decorator / interact()"),
        ("widgets.", "widgets API"),
        ("Dropdown", "dropdown widget"),
        ("Slider", "slider widget"),
        ("Button", "button widget"),
        ("YouTubeVideo", "embedded YouTube video"),
        ("%run widget", "external widget file (%run)"),
        ("build_ethics_widget", "custom ethics widget"),
    ]
    for keyword, label in checks:
        if keyword in combined or keyword in all_text:
            if label not in patterns:
                patterns.append(label)
    return patterns


def guess_discipline(title: str, filename: str, all_text: str) -> str:
    """Best-guess topic from filename, title, and content keywords."""
    search_text = f"{filename} {title} {all_text[:3000]}".lower()
    for keyword, topic in TOPIC_KEYWORDS.items():
        if keyword in search_text:
            return topic
    return "General / Interdisciplinary"


def estimate_coding_level(
    num_code_cells: int, code_texts: list[str], libraries: dict[str, bool]
) -> str:
    """Rough beginner / intermediate / advanced estimate."""
    combined = "\n".join(code_texts).lower()
    advanced_signals = [
        "sklearn",
        "linear_model",
        "train_test_split",
        "cross_val",
        "gradient",
        "otter",
        "grader.check",
    ]
    advanced_count = sum(1 for s in advanced_signals if s in combined)

    if num_code_cells <= 5 and advanced_count == 0:
        return "beginner"
    if advanced_count >= 2 or num_code_cells > 25:
        return "advanced"
    if libraries["uses_pandas"] or num_code_cells > 10:
        return "intermediate"
    return "beginner"


def has_real_world_context(all_text: str, title: str) -> bool:
    """Check if notebook mentions applied / real-world framing."""
    search = f"{title} {all_text[:5000]}".lower()
    return any(kw in search for kw in REAL_WORLD_KEYWORDS)


def normalize_heading_for_pattern(heading: str) -> str | None:
    """Map a heading to a known section label, if possible."""
    clean = re.sub(r"<[^>]+>", "", heading).lower()
    clean = re.sub(r"^\d+[\.\)]\s*", "", clean)  # strip leading numbers
    clean = re.sub(r"^#+\s*", "", clean).strip()

    mapping = [
        (["learning objective"], "Learning Objectives"),
        (["table of contents"], "Table of Contents"),
        (["introduction", "intro"], "Introduction"),
        (["dataset", "data overview", "overview of data", "load the data"], "Dataset"),
        (["background"], "Background"),
        (["visualization", "visualizing", "interactive visualization"], "Visualization"),
        (["analysis", "explore", "modeling"], "Analysis"),
        (["exercise", "coding exercise"], "Exercise"),
        (["question", "free response"], "Exercise"),
        (["hint"], "Hint"),
        (["answer", "solution"], "Answer"),
        (["reflection", "reflect"], "Reflection"),
        (["discussion"], "Discussion"),
        (["conclusion", "wrap up", "summary"], "Conclusion"),
        (["widget", "interactive"], "Interactive Widget"),
        (["extension"], "Extension Activity"),
        (["source"], "Sources"),
    ]
    for keywords, label in mapping:
        if any(kw in clean for kw in keywords):
            return label
    return None


def detect_notebook_pattern(headings: list[str]) -> str:
    """
    Build a teaching-structure pattern from major headings.
    Example: Introduction → Learning Objectives → Dataset → Visualization → Reflection
    """
    seen = set()
    pattern_parts = []
    for heading in headings:
        label = normalize_heading_for_pattern(heading)
        if label and label not in seen and label != "Table of Contents":
            seen.add(label)
            pattern_parts.append(label)
    if not pattern_parts:
        return "Unknown structure"
    return " → ".join(pattern_parts)


def summarize_structure(headings: list[str], num_md: int, num_code: int) -> str:
    """Short prose summary of how the notebook is organized."""
    major = [h for h in headings if h and not h.startswith("Table of")]
    major_preview = ", ".join(major[:6])
    if len(major) > 6:
        major_preview += ", ..."
    return (
        f"{num_md} markdown cells and {num_code} code cells. "
        f"Major sections include: {major_preview or 'none detected'}."
    )


def parse_notebook(nb: dict, filename: str) -> dict:
    """Extract all structured information from one notebook."""
    markdown_texts = []
    code_texts = []

    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type", "")
        text = get_cell_text(cell)
        if cell_type == "markdown":
            markdown_texts.append(text)
        elif cell_type == "code":
            code_texts.append(text)

    all_markdown = "\n\n".join(markdown_texts)
    all_text = all_markdown + "\n\n" + "\n\n".join(code_texts)

    title = extract_title(markdown_texts, filename)
    headings = extract_headings(all_markdown)
    learning_objectives = extract_learning_objectives(all_markdown)
    dataset_descriptions = extract_dataset_descriptions(all_markdown)
    exercises = extract_exercises(all_markdown)
    hints = extract_hints(all_text)
    answers = extract_answers(all_markdown, code_texts)
    reflections = extract_reflections(all_markdown)
    imports = extract_imports(code_texts)
    libraries = detect_library_usage(code_texts)
    viz_patterns = extract_visualization_patterns(code_texts)
    widget_patterns = extract_widget_patterns(code_texts, all_markdown)
    pattern = detect_notebook_pattern(headings)

    num_md = len(markdown_texts)
    num_code = len(code_texts)
    coding_level = estimate_coding_level(num_code, code_texts, libraries)

    return {
        "filename": filename,
        "title": title,
        "discipline": guess_discipline(title, filename, all_markdown),
        "num_markdown_cells": num_md,
        "num_code_cells": num_code,
        "coding_level": coding_level,
        "headings": headings,
        "markdown_cells": markdown_texts,
        "code_cells": code_texts,
        "learning_objectives": learning_objectives,
        "dataset_descriptions": dataset_descriptions,
        "exercises": exercises,
        "hints": hints,
        "answers": answers,
        "reflections": reflections,
        "imports": imports,
        "visualization_patterns": viz_patterns,
        "widget_patterns": widget_patterns,
        "notebook_pattern": pattern,
        "structure_summary": summarize_structure(headings, num_md, num_code),
        "contains_exercises": len(exercises) > 0,
        "contains_hints": len(hints) > 0,
        "contains_answers": len(answers) > 0,
        "contains_reflection_questions": len(reflections) > 0,
        "contains_real_world_context": has_real_world_context(all_markdown, title),
        **libraries,
    }


def format_section(title: str, items: list[str], empty_message: str = "(none found)") -> str:
    """Format a list of strings as a named section for the output file."""
    lines = [f"=== {title} ===", ""]
    if not items:
        lines.append(empty_message)
    else:
        for i, item in enumerate(items, 1):
            lines.append(f"--- Item {i} ---")
            lines.append(item)
            lines.append("")
    lines.append("")
    return "\n".join(lines)


def write_parsed_file(parsed: dict, output_path: Path) -> None:
    """Write one human-readable parsed text file."""
    sections = [
        f"Notebook: {parsed['filename']}",
        f"Title: {parsed['title']}",
        "",
        format_section("HEADINGS", parsed["headings"]),
        format_section("LEARNING OBJECTIVES", parsed["learning_objectives"]),
        format_section("DATASET DESCRIPTIONS", parsed["dataset_descriptions"]),
        format_section(
            "MARKDOWN CELLS",
            [t[:2000] + ("..." if len(t) > 2000 else "") for t in parsed["markdown_cells"]],
        ),
        format_section(
            "CODE CELLS",
            [t[:1500] + ("..." if len(t) > 1500 else "") for t in parsed["code_cells"]],
        ),
        format_section("IMPORTS / LIBRARIES", parsed["imports"]),
        format_section("EXERCISES", parsed["exercises"]),
        format_section("HINTS", parsed["hints"]),
        format_section("ANSWERS / SOLUTIONS", parsed["answers"]),
        format_section("REFLECTION QUESTIONS", parsed["reflections"]),
        format_section("VISUALIZATION PATTERNS", parsed["visualization_patterns"]),
        format_section("WIDGET PATTERNS", parsed["widget_patterns"]),
        "=== NOTEBOOK STRUCTURE ===",
        "",
        f"Detected pattern: {parsed['notebook_pattern']}",
        "",
        parsed["structure_summary"],
        "",
    ]
    output_path.write_text("\n".join(sections), encoding="utf-8")


def build_metadata_entry(parsed: dict) -> dict:
    """Build one entry for notebook_metadata.json."""
    return {
        "filename": parsed["filename"],
        "title": parsed["title"],
        "discipline": parsed["discipline"],
        "num_markdown_cells": parsed["num_markdown_cells"],
        "num_code_cells": parsed["num_code_cells"],
        "coding_level": parsed["coding_level"],
        "uses_pandas": parsed["uses_pandas"],
        "uses_numpy": parsed["uses_numpy"],
        "uses_matplotlib": parsed["uses_matplotlib"],
        "uses_seaborn": parsed["uses_seaborn"],
        "uses_plotly": parsed["uses_plotly"],
        "uses_ipywidgets": parsed["uses_ipywidgets"],
        "contains_exercises": parsed["contains_exercises"],
        "contains_hints": parsed["contains_hints"],
        "contains_answers": parsed["contains_answers"],
        "contains_reflection_questions": parsed["contains_reflection_questions"],
        "contains_real_world_context": parsed["contains_real_world_context"],
        "major_headings": parsed["headings"][:20],
        "notebook_pattern": parsed["notebook_pattern"],
        "structure_summary": parsed["structure_summary"],
    }


def process_all_notebooks() -> tuple[int, int]:
    """
    Process every .ipynb in data/notebooks/.
    Returns (success_count, error_count).
    """
    PARSED_DIR.mkdir(parents=True, exist_ok=True)

    notebook_files = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    total = len(notebook_files)

    if total == 0:
        print(f"No notebooks found in {NOTEBOOKS_DIR}")
        return 0, 0

    metadata_list = []
    success = 0
    errors = 0

    for i, nb_path in enumerate(notebook_files, 1):
        print(f"Processing notebook {i} of {total}... ({nb_path.name})")

        try:
            nb = load_notebook(nb_path)
            parsed = parse_notebook(nb, nb_path.name)

            # Write parsed text file (use stem to avoid spaces in extension)
            output_name = nb_path.stem + "_parsed.txt"
            write_parsed_file(parsed, PARSED_DIR / output_name)

            metadata_list.append(build_metadata_entry(parsed))
            success += 1

        except json.JSONDecodeError as e:
            print(f"  ERROR: Invalid JSON in {nb_path.name}: {e}")
            errors += 1
        except Exception as e:
            print(f"  ERROR: Failed to process {nb_path.name}: {e}")
            errors += 1

    # Write combined metadata
    metadata_path = PARSED_DIR / "notebook_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {success} notebook(s) parsed, {errors} error(s).")
    print(f"Output written to {PARSED_DIR}")

    return success, errors


def main() -> None:
    if not NOTEBOOKS_DIR.exists():
        print(f"Notebooks directory not found: {NOTEBOOKS_DIR}")
        print("Create data/notebooks/ and add .ipynb files.")
        sys.exit(1)

    success, errors = process_all_notebooks()
    sys.exit(1 if errors > 0 and success == 0 else 0)


if __name__ == "__main__":
    main()

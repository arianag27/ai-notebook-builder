#!/usr/bin/env python3
"""
Export a generated markdown draft or outline to a starter Jupyter notebook.

Reads from data/generated/draft_section.md or notebook_outline.md and writes
data/generated/starter_notebook.ipynb. Does not modify original notebooks.

Usage:
    python3 scripts/export_to_ipynb.py
    python3 scripts/export_to_ipynb.py --draft
    python3 scripts/export_to_ipynb.py --outline
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = PROJECT_ROOT / "data" / "generated"
DRAFT_PATH = GENERATED_DIR / "draft_section.md"
OUTLINE_PATH = GENERATED_DIR / "notebook_outline.md"
OUTPUT_PATH = GENERATED_DIR / "starter_notebook.ipynb"

CODE_HEADER = "# --- Starter code: customize for your lesson ---"

CALCULUS_KEYWORDS = (
    "derivative", "derivatives", "integral", "integrals",
    "riemann", "calculus", "tangent", "area under",
)

# Runnable starter code for math/calculus notebooks (no dataset)
CALCULUS_SETUP_CODE = """import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, FloatSlider, IntSlider

%matplotlib inline"""

CALCULUS_DEFINE_FUNCTION_CODE = """def f(x):
    \"\"\"Example function: f(x) = x². Change the formula to explore other functions.\"\"\"
    return x ** 2"""

CALCULUS_PLOT_FUNCTION_CODE = """a, b = 0, 2
x = np.linspace(a, b, 200)

plt.figure(figsize=(8, 4))
plt.plot(x, f(x), label="f(x) = x²")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Plotting a function over an interval")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""

CALCULUS_TANGENT_CODE = """def plot_tangent(x0=1.0):
    \"\"\"Show the function and a tangent line at x = x0.\"\"\"
    h = 0.0001
    slope = (f(x0 + h) - f(x0 - h)) / (2 * h)

    x = np.linspace(-1, 3, 200)
    y = f(x)
    tangent = slope * (x - x0) + f(x0)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label="f(x)")
    plt.plot(x, tangent, "--", label=f"tangent at x = {x0:.2f}")
    plt.scatter([x0], [f(x0)], color="red", zorder=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Derivative intuition: tangent line")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

interact(
    plot_tangent,
    x0=FloatSlider(min=-1.0, max=3.0, step=0.1, value=1.0, description="x value"),
)"""

CALCULUS_RIEMANN_CODE = """def plot_riemann(n_rectangles=10, a=0.0, b=2.0):
    \"\"\"Approximate area under f(x) using left-endpoint rectangles.\"\"\"
    x = np.linspace(a, b, 200)

    plt.figure(figsize=(8, 4))
    plt.plot(x, f(x), label="f(x)")

    dx = (b - a) / n_rectangles
    for i in range(n_rectangles):
        left = a + i * dx
        height = f(left)
        rect = plt.Rectangle(
            (left, 0), dx, height,
            alpha=0.4, edgecolor="gray", facecolor="skyblue",
        )
        plt.gca().add_patch(rect)

    plt.xlim(a, b)
    plt.ylim(0, max(f(a), f(b), f((a + b) / 2)) * 1.2)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"Riemann sum with {n_rectangles} rectangles")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

interact(
    plot_riemann,
    n_rectangles=IntSlider(min=2, max=50, step=1, value=10, description="rectangles"),
    a=FloatSlider(min=-1.0, max=1.0, step=0.1, value=0.0, description="left bound"),
    b=FloatSlider(min=1.0, max=3.0, step=0.1, value=2.0, description="right bound"),
)"""


def choose_input_file(prefer: str | None = None) -> Path:
    """Pick draft or outline file. Prefer draft by default if both exist."""
    if prefer == "outline":
        if OUTLINE_PATH.exists():
            return OUTLINE_PATH
        print(f"Outline not found: {OUTLINE_PATH}")
        sys.exit(1)

    if prefer == "draft":
        if DRAFT_PATH.exists():
            return DRAFT_PATH
        print(f"Draft not found: {DRAFT_PATH}")
        sys.exit(1)

    if DRAFT_PATH.exists():
        return DRAFT_PATH
    if OUTLINE_PATH.exists():
        return OUTLINE_PATH

    print("No input file found. Generate one first:")
    print("  python3 scripts/generate_notebook_outline.py")
    print("  python3 scripts/notebook_copilot.py --draft \"Create a ...\"")
    sys.exit(1)


def extract_draft_body(text: str) -> str:
    """
    Pull only the draft section from draft_section.md.
    Skips metadata, influence notes, and source chunks.
    """
    if "## Notes on Influencing Examples" in text:
        start = text.find("---\n")
        if start == -1:
            return text
        start += len("---\n")
        end = text.find("## Notes on Influencing Examples")
        return text[start:end].strip()

    # Fallback: content after first --- block
    parts = text.split("\n---\n")
    if len(parts) >= 2:
        return parts[1].strip()
    return text


def extract_outline_title(text: str) -> str:
    """Get notebook title from outline if present."""
    match = re.search(r"## Title\s*\n+(.+)", text)
    if match:
        return match.group(1).strip()
    return "Starter Notebook"


def prepare_outline_markdown(text: str) -> str:
    """Clean up outline markdown for notebook export."""
    title = extract_outline_title(text)

    # Remove planning metadata not needed in the notebook
    text = re.sub(r"^# Notebook Outline\s*\n", "", text)
    text = re.sub(r"\*\*Generated for:\*\*[^\n]*\n[^\n]*\n\n---\n\n", "", text)
    text = re.sub(r"## Title\s*\n+.+\s*\n+", "", text, count=1)
    if "## Similar Notebooks in Corpus" in text:
        text = text[: text.index("## Similar Notebooks in Corpus")].rstrip()

    return f"# {title}\n\n{text.strip()}"


def prepare_markdown(text: str, source: Path) -> str:
    """Prepare markdown content based on source file type."""
    if source.name == "draft_section.md":
        return extract_draft_body(text)
    return prepare_outline_markdown(text)


def is_section_heading(line: str) -> bool:
    """True for # or ## headings that start a new markdown cell."""
    return bool(re.match(r"^#{1,2}\s+", line))


def enhance_code_cell(code: str) -> str:
    """Add helpful comments to starter code cells."""
    lines = code.splitlines()
    has_header = any("customize" in ln.lower() or "starter code" in ln.lower() for ln in lines[:2])

    new_lines = []
    if not has_header:
        new_lines.extend([CODE_HEADER, ""])

    for line in lines:
        stripped = line.strip()
        if stripped == "pass":
            indent = line[: len(line) - len(line.lstrip())]
            new_lines.append(f"{indent}pass  # TODO: implement this step")
        elif "# TODO" in line or "# todo" in line:
            new_lines.append(line)
        elif stripped.startswith("#") and "TODO" not in line and "customize" not in line.lower():
            new_lines.append(line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def make_markdown_cell(text: str) -> dict:
    """Create a markdown notebook cell."""
    # Jupyter expects source as a list of lines ending with \n
    source_lines = [line + "\n" for line in text.splitlines()]
    if source_lines:
        source_lines[-1] = source_lines[-1].rstrip("\n") or "\n"
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines if source_lines else [""],
    }


def make_code_cell(code: str, enhance: bool = True) -> dict:
    """Create a code notebook cell. Set enhance=False for ready-to-run template code."""
    final_code = enhance_code_cell(code) if enhance else code
    source_lines = [line + "\n" for line in final_code.splitlines()]
    if source_lines:
        source_lines[-1] = source_lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source_lines,
        "outputs": [],
        "execution_count": None,
    }


def parse_markdown_to_cells(markdown: str) -> list[dict]:
    """
    Convert markdown to notebook cells.

    - # and ## headings start new markdown cells
    - ### and lower stay in the current cell
    - ```python blocks become code cells
    """
    cells = []
    lines = markdown.splitlines()
    md_buffer: list[str] = []
    i = 0

    def flush_markdown() -> None:
        if not md_buffer:
            return
        text = "\n".join(md_buffer).strip()
        if text:
            cells.append(make_markdown_cell(text))
        md_buffer.clear()

    while i < len(lines):
        line = lines[i]

        # Fenced Python code block
        if line.strip().startswith("```python"):
            flush_markdown()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            cells.append(make_code_cell("\n".join(code_lines)))
            i += 1  # skip closing fence
            continue

        # Top-level headings start a new markdown cell
        if is_section_heading(line):
            flush_markdown()
            md_buffer = [line]
        else:
            md_buffer.append(line)

        i += 1

    flush_markdown()
    return cells


def parse_libraries_from_text(text: str) -> list[str]:
    """Extract library names from a Suggested Libraries section."""
    libs = []
    mapping = {
        "numpy": "import numpy as np",
        "pandas": "import pandas as pd",
        "matplotlib": "import matplotlib.pyplot as plt",
        "seaborn": "import seaborn as sns",
        "plotly": "import plotly.express as px",
        "ipywidgets": "import ipywidgets as widgets",
        "sklearn": "from sklearn import linear_model",
    }
    lower = text.lower()
    for key, stmt in mapping.items():
        if key in lower:
            libs.append(stmt)
    return libs


def add_outline_code_stubs(cells: list[dict]) -> list[dict]:
    """
    For notebook outlines, insert placeholder code cells after key sections.
    """
    result = []
    for cell in cells:
        result.append(cell)
        if cell["cell_type"] != "markdown":
            continue

        source = "".join(cell["source"])

        if "## Suggested Libraries" in source:
            imports = parse_libraries_from_text(source)
            if not imports:
                imports = ["import numpy as np", "import matplotlib.pyplot as plt"]
            code = CODE_HEADER + "\n\n" + "\n".join(imports)
            code += "\n\n# %matplotlib inline  # uncomment for inline plots"
            result.append(make_code_cell(code))

        if "## Suggested Code Activities" in source:
            stub = "\n".join([
                CODE_HEADER,
                "",
                "# TODO: Add your first coding activity here",
                "# Example: load data, define a function, or create a plot",
                "",
                "pass  # replace with your code",
            ])
            result.append(make_code_cell(stub))

        if "## Optional Widget Ideas" in source and "none" not in source.lower()[:200]:
            stub = "\n".join([
                CODE_HEADER,
                "",
                "# TODO: Add an interactive widget",
                "# from ipywidgets import interact",
                "",
                "# @interact",
                "# def explore(parameter=1):",
                "#     pass  # build your widget visualization",
            ])
            result.append(make_code_cell(stub))

    return result


def add_placeholder_cells(cells: list[dict]) -> list[dict]:
    """Add a student response cell after markdown with answer placeholders."""
    result = []
    placeholders = ("*your answer here*", "*your reflections here*", "*type your answer*")

    for cell in cells:
        result.append(cell)
        if cell["cell_type"] != "markdown":
            continue
        source = "".join(cell["source"]).lower()
        if any(p in source for p in placeholders):
            result.append(make_markdown_cell("*(Student response cell — edit and run when finished)*"))
    return result


def parse_outline_metadata(text: str) -> dict:
    """Extract planning metadata from notebook_outline.md."""
    meta = {
        "uses_dataset": True,
        "subject_type": "general",
        "topic": "",
        "reflection": True,
        "export_template": None,
        "title": extract_outline_title(text),
        "intro": "",
        "objectives": [],
    }

    if "Uses dataset: No" in text:
        meta["uses_dataset"] = False
    if "Subject type: math" in text:
        meta["subject_type"] = "math"

    match = re.search(r"\*\*Generated for:\*\*\s*([^—\n]+)—\s*(.+)", text)
    if match:
        meta["topic"] = match.group(2).strip()

    if re.search(r"\*\*Reflection questions:\*\*\s*No", text, re.IGNORECASE):
        meta["reflection"] = False
    if "reflection not requested" in text.lower():
        meta["reflection"] = False

    template_match = re.search(r"\*\*Export template:\*\*\s*(\S+)", text)
    if template_match:
        meta["export_template"] = template_match.group(1)

    intro_match = re.search(r"## Short Introduction\s*\n+(.+?)(?=\n## )", text, re.DOTALL)
    if intro_match:
        meta["intro"] = intro_match.group(1).strip()

    obj_match = re.search(r"## Learning Objectives\s*\n+((?:- .+\n?)+)", text)
    if obj_match:
        meta["objectives"] = [
            line[2:].strip()
            for line in obj_match.group(1).splitlines()
            if line.startswith("- ")
        ]

    return meta


def is_calculus_topic(text: str) -> bool:
    """Check if the topic involves derivatives, integrals, or Riemann sums."""
    lower = text.lower()
    return any(kw in lower for kw in CALCULUS_KEYWORDS)


def should_use_calculus_template(text: str, source: Path) -> bool:
    """Decide whether to export a rich calculus starter notebook."""
    if source.name == "notebook_outline.md":
        meta = parse_outline_metadata(text)
        if meta["export_template"] == "math_calculus_interactive":
            return True
        return (
            not meta["uses_dataset"]
            and meta["subject_type"] == "math"
            and is_calculus_topic(meta["topic"])
        )

    if source.name == "draft_section.md":
        body = extract_draft_body(text).lower()
        return is_calculus_topic(body) and "widget" in body

    return False


def build_calculus_interactive_cells(meta: dict) -> list[dict]:
    """Build a runnable calculus starter notebook (no dataset)."""
    title = meta.get("title", "Exploring Derivatives and Integrals")
    intro = meta.get("intro") or (
        "In this notebook, you will explore **derivatives and integrals** through "
        "interactive plots and sliders. No dataset is required — you will define and "
        "visualize functions directly in Python."
    )
    objectives = meta.get("objectives") or [
        "Explain the relationship between derivatives and slopes of functions.",
        "Plot a function over an interval.",
        "Use a tangent line to explore derivative intuition.",
        "Approximate area under a curve with Riemann sum rectangles.",
    ]

    obj_lines = "\n".join(f"- {obj}" for obj in objectives)
    cells = [
        make_markdown_cell(f"# {title}"),
        make_markdown_cell(f"## Introduction\n\n{intro}"),
        make_markdown_cell(f"## Learning Objectives\n\n{obj_lines}"),
        make_markdown_cell(
            "## Setup\n\n"
            "Run the cell below to import the libraries used in this notebook."
        ),
        make_code_cell(CALCULUS_SETUP_CODE, enhance=False),
        make_markdown_cell(
            "## Define a Function\n\n"
            "We start with a simple example function. "
            "You can change the formula later to explore other functions."
        ),
        make_code_cell(CALCULUS_DEFINE_FUNCTION_CODE, enhance=False),
        make_markdown_cell(
            "## Plot the Function\n\n"
            "Before studying derivatives and integrals, plot the function over an interval. "
            "This helps you see the shape of the graph."
        ),
        make_code_cell(CALCULUS_PLOT_FUNCTION_CODE, enhance=False),
        make_markdown_cell(
            "## Explore the Derivative\n\n"
            "A derivative measures how fast a function changes at a point. "
            "One way to build intuition is to look at the **tangent line** at a chosen "
            "x-value.\n\n"
            "Run the interactive cell below and move the slider."
        ),
        make_code_cell(CALCULUS_TANGENT_CODE, enhance=False),
        make_markdown_cell(
            "**Try changing the slider.** What happens to the tangent line when you move "
            "to a different x-value?\n\n"
            "**What do you notice?** How does the slope of the tangent relate to how steep "
            "the curve looks at that point?"
        ),
        make_markdown_cell(
            "## Explore the Integral\n\n"
            "An integral measures accumulated area under a curve. "
            "A **Riemann sum** approximates this area using rectangles.\n\n"
            "Use the sliders below to change the number of rectangles and the interval."
        ),
        make_code_cell(CALCULUS_RIEMANN_CODE, enhance=False),
        make_markdown_cell(
            "**Try changing the number of rectangles.** Does the approximation look closer "
            "to the area under the curve when you use more rectangles?\n\n"
            "**What do you notice?** What happens when you change the interval bounds?"
        ),
    ]

    if meta.get("reflection"):
        cells.append(make_markdown_cell(
            "## Reflection\n\n"
            "1. How did the tangent line visualization connect to the idea of a derivative?\n"
            "2. How did increasing the number of rectangles affect the Riemann sum?\n\n"
            "*Your reflections here*"
        ))

    return cells


def build_notebook(cells: list[dict], source_file: Path) -> dict:
    """Wrap cells in a valid nbformat v4 notebook structure."""
    header = make_markdown_cell(
        f"> **Starter notebook** exported from `{source_file.name}`.\n"
        "> Replace placeholders and customize code cells before publishing."
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "cells": [header] + cells,
    }


def export_to_ipynb(input_path: Path, output_path: Path) -> int:
    """Read markdown, convert to .ipynb, and save."""
    raw_text = input_path.read_text(encoding="utf-8")

    if should_use_calculus_template(raw_text, input_path):
        meta = parse_outline_metadata(raw_text) if input_path.name == "notebook_outline.md" else {
            "title": "Exploring Derivatives and Integrals with Interactive Visualizations",
            "intro": "",
            "objectives": [],
            "reflection": "reflection not requested" not in raw_text.lower(),
        }
        cells = build_calculus_interactive_cells(meta)
        notebook = build_notebook(cells, input_path)
    else:
        markdown = prepare_markdown(raw_text, input_path)
        cells = parse_markdown_to_cells(markdown)

        if input_path.name == "notebook_outline.md":
            cells = add_outline_code_stubs(cells)

        cells = add_placeholder_cells(cells)
        notebook = build_notebook(cells, input_path)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    return len(notebook["cells"])


def parse_args() -> str | None:
    """Return 'draft', 'outline', or None based on CLI flags."""
    if "--outline" in sys.argv:
        return "outline"
    if "--draft" in sys.argv:
        return "draft"
    return None


def main() -> None:
    prefer = parse_args()
    input_path = choose_input_file(prefer)
    cell_count = export_to_ipynb(input_path, OUTPUT_PATH)

    print(f"Exported {cell_count} cells from {input_path.name}")
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

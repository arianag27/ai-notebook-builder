#!/usr/bin/env python3
"""
Validate a generated starter notebook before human review.

Checks structure, Python syntax (without running code), and common quality issues.
Reads expectations from notebook_outline.md when available.

Usage:
    python3 scripts/validate_generated_notebook.py
"""

import ast
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = PROJECT_ROOT / "data" / "generated"
NOTEBOOK_PATH = GENERATED_DIR / "starter_notebook.ipynb"
OUTLINE_PATH = GENERATED_DIR / "notebook_outline.md"
REPORT_PATH = GENERATED_DIR / "validation_report.md"

# Words that suggest a dataset is required (warn when outline says no dataset)
DATASET_LANGUAGE = [
    "load the dataset",
    "load dataset",
    "read_csv",
    "dataframe",
    "preview rows",
    "column names",
    "pd.read",
    "dataset overview",
    "explore the data",
]

# Patterns that indicate unfinished generic starter code
GENERIC_PLACEHOLDER_PATTERNS = [
    r"\bpass\s*#\s*TODO",
    r"\bpass\s*#\s*replace",
    r"#\s*TODO:",
    r"replace with your code",
    r"add your first coding activity",
]

# Phrases that count as student observation prompts (OK when reflection = no)
OBSERVATION_PROMPTS = [
    "try changing",
    "what do you notice",
    "what happens when",
    "what would you",
]

def is_interactive_code(code: str) -> bool:
    """True when a code cell uses widgets for interaction (not just imports)."""
    if "interact(" in code or "@interact" in code:
        return True
    if re.search(r"widgets\.(interact|IntSlider|FloatSlider|Dropdown)", code):
        return True
    return False


def load_notebook(path: Path) -> dict:
    """Load a .ipynb file safely. Exit with a clear message if missing or invalid."""
    if not path.exists():
        print(f"Notebook not found: {path}")
        print("Run: python3 scripts/export_to_ipynb.py")
        sys.exit(1)

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        print(f"Invalid notebook JSON in {path}: {exc}")
        sys.exit(1)


def get_cell_text(cell: dict) -> str:
    """Turn a cell source field into one string."""
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def load_outline_expectations() -> dict:
    """Read planning flags from notebook_outline.md if it exists."""
    expectations = {
        "uses_dataset": None,
        "reflection": None,
        "topic": "",
    }

    if not OUTLINE_PATH.exists():
        return expectations

    text = OUTLINE_PATH.read_text(encoding="utf-8")

    if "Uses dataset: No" in text:
        expectations["uses_dataset"] = False
    elif "Uses dataset: Yes" in text:
        expectations["uses_dataset"] = True

    if re.search(r"\*\*Reflection questions:\*\*\s*No", text, re.IGNORECASE):
        expectations["reflection"] = False
    elif re.search(r"\*\*Reflection questions:\*\*\s*Yes", text, re.IGNORECASE):
        expectations["reflection"] = True
    elif "reflection not requested" in text.lower():
        expectations["reflection"] = False

    match = re.search(r"\*\*Generated for:\*\*\s*[^—\n]+—\s*(.+)", text)
    if match:
        expectations["topic"] = match.group(1).strip()

    return expectations


def strip_ipython_magics(code: str) -> str:
    """Remove lines like %matplotlib inline before syntax checking."""
    lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("%") or stripped.startswith("!"):
            continue
        lines.append(line)
    return "\n".join(lines)


class Validator:
    """Collect passed checks, warnings, and errors."""

    def __init__(self) -> None:
        self.passed: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []

    def ok(self, message: str) -> None:
        self.passed.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def fail(self, message: str) -> None:
        self.errors.append(message)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


def validate_structure(cells: list[dict], v: Validator) -> None:
    """Check required notebook sections and cell content."""
    markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]
    code_cells = [c for c in cells if c.get("cell_type") == "code"]

    if markdown_cells:
        v.ok(f"Has markdown cells ({len(markdown_cells)})")
    else:
        v.fail("No markdown cells found")

    if code_cells:
        v.ok(f"Has code cells ({len(code_cells)})")
    else:
        v.fail("No code cells found")

    # Empty cells
    for i, cell in enumerate(cells):
        text = get_cell_text(cell).strip()
        if not text:
            v.fail(f"Cell {i} is empty")

    if not any(get_cell_text(c).strip() == "" for c in cells):
        v.ok("No empty cells")

    # Title: single # heading (skip export banner in cell 0)
    has_title = False
    for cell in markdown_cells:
        text = get_cell_text(cell).strip()
        if re.match(r"^#\s+\S", text) and not text.startswith("##"):
            has_title = True
            break
    if has_title:
        v.ok("Has a notebook title (# heading)")
    else:
        v.fail("Missing notebook title (# heading)")

    # Learning objectives
    combined_md = "\n".join(get_cell_text(c) for c in markdown_cells).lower()
    if "learning objectives" in combined_md:
        v.ok("Has learning objectives section")
    else:
        v.fail("Missing learning objectives section")

    # Setup / imports
    has_setup_heading = "## setup" in combined_md
    has_imports = any(
        re.search(r"^\s*import\s+", get_cell_text(c), re.MULTILINE)
        for c in code_cells
    )
    if has_setup_heading and has_imports:
        v.ok("Has setup section with import code")
    elif has_imports:
        v.ok("Has import code (setup heading not found)")
    else:
        v.fail("Missing setup/imports")


def validate_code_syntax(cells: list[dict], v: Validator) -> None:
    """Parse each code cell with ast.parse and report syntax errors."""
    checked = 0
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        code = get_cell_text(cell)
        cleaned = strip_ipython_magics(code).strip()
        if not cleaned:
            v.fail(f"Code cell {i} has no Python code after removing magics")
            continue

        try:
            ast.parse(cleaned)
            checked += 1
        except SyntaxError as exc:
            v.fail(f"Syntax error in code cell {i}: {exc.msg} (line {exc.lineno})")

    if checked:
        v.ok(f"All {checked} code cell(s) passed syntax check")


def validate_no_generic_placeholders(cells: list[dict], v: Validator) -> None:
    """Flag leftover TODO/pass starter stubs."""
    found = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        code = get_cell_text(cell)
        for pattern in GENERIC_PLACEHOLDER_PATTERNS:
            if re.search(pattern, code, re.IGNORECASE):
                found.append(i)
                break

    if found:
        v.fail(
            f"Generic TODO/pass placeholders in code cell(s): {', '.join(map(str, found))}"
        )
    else:
        v.ok("No generic TODO/pass placeholders in code")


def validate_quality(
    cells: list[dict],
    expectations: dict,
    v: Validator,
) -> None:
    """Detect common quality issues using outline expectations when available."""
    all_text = "\n".join(get_cell_text(c) for c in cells)
    all_lower = all_text.lower()

    # Dataset language when dataset = no
    if expectations.get("uses_dataset") is False:
        hits = [phrase for phrase in DATASET_LANGUAGE if phrase in all_lower]
        # Allow mentioning "no dataset" or "no external dataset"
        hits = [
            h for h in hits
            if not re.search(r"no (external )?dataset", all_lower)
        ]
        if hits:
            v.warn(
                f"Dataset language found but outline says no dataset: {', '.join(hits[:3])}"
            )
        else:
            v.ok("No inappropriate dataset language (outline: no dataset)")

    # Reflection section when reflection = no
    if expectations.get("reflection") is False:
        has_reflection_section = bool(
            re.search(r"^##\s+reflection", all_lower, re.MULTILINE)
        )
        if has_reflection_section:
            v.warn("Reflection section found but outline says reflection = no")
        else:
            v.ok("No full reflection section (outline: reflection = no)")

    # Count generic customization placeholders in markdown
    placeholder_count = len(re.findall(
        r"\*your (?:answer|reflections?) here\*|\[add |\[describe |\[what ",
        all_lower,
    ))
    if placeholder_count > 3:
        v.warn(f"Many generic placeholders found ({placeholder_count})")
    elif placeholder_count > 0:
        v.ok(f"Limited placeholders found ({placeholder_count}) — acceptable")
    else:
        v.ok("No excessive generic placeholders")

    # Explanations before code cells
    missing_explanations = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        if i == 0:
            continue
        prev = cells[i - 1]
        if prev.get("cell_type") != "markdown":
            missing_explanations.append(i)
            continue
        prev_text = get_cell_text(prev).strip()
        if len(prev_text) < 40:
            missing_explanations.append(i)

    if missing_explanations:
        v.warn(
            "Code cell(s) may lack a beginner-friendly explanation in the previous markdown cell: "
            + ", ".join(map(str, missing_explanations))
        )
    else:
        v.ok("Code cells have markdown explanations before them")

    # Student prompts after interactive code cells
    interactive_cells = []
    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        code = get_cell_text(cell)
        if is_interactive_code(code):
            interactive_cells.append(i)

    missing_prompts = []
    for idx in interactive_cells:
        if idx + 1 >= len(cells):
            missing_prompts.append(idx)
            continue
        next_text = get_cell_text(cells[idx + 1]).lower()
        if not any(prompt in next_text for prompt in OBSERVATION_PROMPTS):
            missing_prompts.append(idx)

    if interactive_cells and missing_prompts:
        v.warn(
            "Interactive code cell(s) missing student prompts afterward: "
            + ", ".join(map(str, missing_prompts))
        )
    elif interactive_cells:
        v.ok("Interactive sections include student observation prompts")
    else:
        v.ok("No interactive cells to check for student prompts")


def build_report(
    notebook_path: Path,
    expectations: dict,
    v: Validator,
) -> str:
    """Format the full markdown validation report."""
    status = "PASSED" if v.success and not v.warnings else (
        "PASSED WITH WARNINGS" if v.success else "FAILED"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        "# Notebook Validation Report",
        "",
        f"**Notebook:** `{notebook_path.name}`",
        f"**Validated:** {timestamp}",
        f"**Status:** {status}",
        "",
    ]

    if expectations.get("uses_dataset") is not None:
        lines.append(f"**Expected dataset:** {'yes' if expectations['uses_dataset'] else 'no'}")
    if expectations.get("reflection") is not None:
        lines.append(f"**Expected reflection:** {'yes' if expectations['reflection'] else 'no'}")
    if expectations.get("topic"):
        lines.append(f"**Topic:** {expectations['topic']}")
    lines.append("")

    lines.extend(["## Passed Checks", ""])
    if v.passed:
        for item in v.passed:
            lines.append(f"- [x] {item}")
    else:
        lines.append("- None")
    lines.append("")

    lines.extend(["## Warnings", ""])
    if v.warnings:
        for item in v.warnings:
            lines.append(f"- [!] {item}")
    else:
        lines.append("- None")
    lines.append("")

    lines.extend(["## Errors", ""])
    if v.errors:
        for item in v.errors:
            lines.append(f"- [x] {item}")
    else:
        lines.append("- None")
    lines.append("")

    lines.extend([
        "## Next Steps",
        "",
    ])
    if v.errors:
        lines.append("- Fix the errors above, then re-run export and validation.")
    elif v.warnings:
        lines.append("- Review warnings before sharing the notebook with students.")
    else:
        lines.append("- Notebook looks ready for human review.")

    lines.append("")
    return "\n".join(lines)


def print_summary(v: Validator) -> None:
    """Print a short terminal summary."""
    print()
    print("=== Validation Summary ===")
    print(f"Passed: {len(v.passed)}")
    print(f"Warnings: {len(v.warnings)}")
    print(f"Errors: {len(v.errors)}")
    print()

    if v.errors:
        print("Errors:")
        for item in v.errors:
            print(f"  - {item}")
        print()

    if v.warnings:
        print("Warnings:")
        for item in v.warnings:
            print(f"  - {item}")
        print()

    if v.success:
        if v.warnings:
            print("Result: PASSED WITH WARNINGS")
        else:
            print("Result: PASSED")
    else:
        print("Result: FAILED")


def validate_notebook(notebook_path: Path = NOTEBOOK_PATH) -> Validator:
    """Run all validation checks and return results."""
    notebook = load_notebook(notebook_path)
    cells = notebook.get("cells", [])
    expectations = load_outline_expectations()

    v = Validator()
    validate_structure(cells, v)
    validate_code_syntax(cells, v)
    validate_no_generic_placeholders(cells, v)
    validate_quality(cells, expectations, v)

    return v


def main() -> None:
    v = validate_notebook()

    report = build_report(NOTEBOOK_PATH, load_outline_expectations(), v)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print_summary(v)
    print(f"Report saved to {REPORT_PATH}")

    sys.exit(0 if v.success else 1)


if __name__ == "__main__":
    main()

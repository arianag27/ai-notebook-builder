#!/usr/bin/env python3
"""
Analyze parsed notebook corpus and generate a curriculum patterns report.

Reads notebook_metadata.json and *_parsed.txt files from data/parsed/.
Writes curriculum_patterns_report.md to data/parsed/.
"""

import json
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
METADATA_PATH = PARSED_DIR / "notebook_metadata.json"
REPORT_PATH = PARSED_DIR / "curriculum_patterns_report.md"

LIBRARY_FIELDS = [
    ("uses_pandas", "pandas"),
    ("uses_numpy", "numpy"),
    ("uses_matplotlib", "matplotlib"),
    ("uses_seaborn", "seaborn"),
    ("uses_plotly", "plotly"),
    ("uses_ipywidgets", "ipywidgets"),
]

ACTIVITY_FIELDS = [
    ("contains_exercises", "Exercises / Questions"),
    ("contains_hints", "Hints"),
    ("contains_answers", "Answers / Solutions"),
    ("contains_reflection_questions", "Reflection Questions"),
    ("contains_real_world_context", "Real-World Context"),
]

ARCHETYPES = {
    "Introductory EDA Notebook": {
        "description": "Guides students through exploring a dataset with tables, summaries, and charts.",
        "signals": lambda m, p: (
            m["uses_pandas"]
            and m["num_code_cells"] >= 3
            and (m["uses_matplotlib"] or m["uses_seaborn"] or m["uses_plotly"])
            and m["coding_level"] in ("beginner", "intermediate")
        ),
        "common_sections": "Introduction → Learning Objectives → Dataset → Exploration → Visualization",
        "teaching_methods": [
            "Load and preview data early",
            "Column guides and data dictionaries",
            "Guided plots with interpretation prompts",
        ],
    },
    "Coding-Focused Notebook": {
        "description": "Students write and test code, often with autograding, hints, and scaffolded functions.",
        "signals": lambda m, p: (
            m["num_code_cells"] >= 12
            and m["contains_exercises"]
            and (m["contains_hints"] or m["contains_answers"] or m["coding_level"] == "advanced")
        ),
        "common_sections": "Introduction → Dataset → Coding Exercises → Model Building → Evaluation",
        "teaching_methods": [
            "Step-by-step coding tasks with blanks to fill in",
            "Hints before harder implementation steps",
            "Otter or inline answer-checking",
        ],
    },
    "Widget-Based Interactive Notebook": {
        "description": "Uses sliders, dropdowns, or custom widgets so students explore concepts interactively.",
        "signals": lambda m, p: m["uses_ipywidgets"] or len(p.get("widget_patterns", [])) > 0,
        "common_sections": "Background → Interactive Widget → Experimentation → Discussion",
        "teaching_methods": [
            "Run-once setup cell for widget imports",
            "Parameter exploration without rewriting code",
            "External widget files for complex UIs",
        ],
    },
    "Reflection-Based Notebook": {
        "description": "Prioritizes discussion, ethical reasoning, or written reflection over heavy coding.",
        "signals": lambda m, p: (
            m["contains_reflection_questions"]
            or len(p.get("reflections", [])) >= 2
            or (m["num_code_cells"] <= 8 and m["contains_reflection_questions"])
        ),
        "common_sections": "Introduction → Guided Exploration → Reflection → Conclusion",
        "teaching_methods": [
            "Free-response or short-answer prompts",
            "Scenario-based discussion cues",
            "Post-notebook reflection forms",
        ],
    },
    "Case Study Notebook": {
        "description": "Frames learning around a real-world scenario, policy issue, or industry problem.",
        "signals": lambda m, p: m["contains_real_world_context"] and (
            "case study" in m["title"].lower()
            or "case study" in " ".join(p.get("markdown_sample", [])).lower()
            or m["discipline"] not in ("Mathematics / Calculus", "General / Interdisciplinary")
        ),
        "common_sections": "Case Introduction → Data / Context → Analysis → Application → Wrap-Up",
        "teaching_methods": [
            "Narrative framing with domain vocabulary",
            "Multi-section storyline (e.g. 4 P's, policy eras)",
            "Connect data patterns to real decisions",
        ],
    },
}


def load_metadata() -> list[dict]:
    """Load notebook_metadata.json."""
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_parsed_file(stem: str) -> dict:
    """
    Load a parsed text file and extract named sections.
    Returns a dict of section_name -> list of item strings.
    """
    path = PARSED_DIR / f"{stem}_parsed.txt"
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    sections = {}
    parts = re.split(r"\n=== (.+?) ===\n", text)

    # parts[0] is header; then alternating section name, content
    for i in range(1, len(parts), 2):
        name = parts[i].strip()
        content = parts[i + 1] if i + 1 < len(parts) else ""
        items = []
        for block in re.split(r"--- Item \d+ ---\n", content):
            block = block.strip()
            if block and block != "(none found)":
                items.append(block)
        sections[name] = items

    return sections


def stem_from_filename(filename: str) -> str:
    """Turn 'my notebook.ipynb' into 'my notebook' for parsed file lookup."""
    return Path(filename).stem


def pct(count: int, total: int) -> str:
    """Format count with percentage."""
    if total == 0:
        return "0 (0%)"
    return f"{count} ({100 * count // total}%)"


def count_by_key(items: list[dict], key: str) -> Counter:
    """Count occurrences of a metadata field value."""
    return Counter(item[key] for item in items)


def analyze_libraries(metadata: list[dict]) -> list[tuple[str, int]]:
    """Count how many notebooks use each library."""
    counts = []
    total = len(metadata)
    for field, label in LIBRARY_FIELDS:
        count = sum(1 for m in metadata if m.get(field))
        counts.append((label, count))
    return sorted(counts, key=lambda x: -x[1])


def analyze_activities(metadata: list[dict]) -> list[tuple[str, int]]:
    """Count learning activity types across the corpus."""
    total = len(metadata)
    results = []
    for field, label in ACTIVITY_FIELDS:
        count = sum(1 for m in metadata if m.get(field))
        results.append((label, count))
    return sorted(results, key=lambda x: -x[1])


def analyze_patterns(metadata: list[dict]) -> Counter:
    """Count detected notebook structure patterns."""
    return Counter(m["notebook_pattern"] for m in metadata if m.get("notebook_pattern"))


def extract_section_sequences(metadata: list[dict]) -> Counter:
    """
    Build simplified section-order strings from notebook_pattern.
    e.g. 'Introduction → Dataset → Visualization'
    """
    sequences = []
    for m in metadata:
        pattern = m.get("notebook_pattern", "")
        if pattern and pattern != "Unknown structure":
            sequences.append(pattern)
    return Counter(sequences)


def analyze_visualization_activities(parsed_data: dict[str, dict]) -> Counter:
    """Count visualization patterns across all parsed files."""
    counts = Counter()
    for sections in parsed_data.values():
        for item in sections.get("VISUALIZATION PATTERNS", []):
            counts[item] += 1
    return counts


def analyze_reflection_styles(parsed_data: dict[str, dict]) -> Counter:
    """Categorize reflection / question styles from parsed reflection text."""
    styles = Counter()
    patterns = [
        (r"free response", "Free Response Questions"),
        (r"question \d|question one|question two", "Numbered Questions"),
        (r"reflect|reflection", "Reflection Prompts"),
        (r"discussion|discuss", "Discussion Prompts"),
        (r"your answer here|type your answer", "Answer Placeholder Cells"),
        (r"google\.com/forms|reflection form", "External Reflection Form"),
        (r"scenario", "Scenario-Based Prompts"),
    ]
    for sections in parsed_data.values():
        combined = " ".join(sections.get("REFLECTION QUESTIONS", [])).lower()
        combined += " ".join(sections.get("EXERCISES", [])).lower()
        matched = set()
        for regex, label in patterns:
            if re.search(regex, combined, re.IGNORECASE):
                matched.add(label)
        for label in matched:
            styles[label] += 1
    return styles


def analyze_widgets(metadata: list[dict], parsed_data: dict[str, dict]) -> dict:
    """Summarize widget usage across notebooks."""
    widget_notebooks = []
    widget_types = Counter()
    incorporation_notes = []

    for m in metadata:
        stem = stem_from_filename(m["filename"])
        sections = parsed_data.get(stem, {})
        widgets = sections.get("WIDGET PATTERNS", [])
        uses_widgets = m["uses_ipywidgets"] or len(widgets) > 0

        if not uses_widgets:
            continue

        widget_notebooks.append(m["filename"])
        for w in widgets:
            widget_types[w] += 1

        # Summarize how widgets are used from parsed content
        widget_text = " ".join(widgets).lower()
        if "interact" in widget_text:
            incorporation_notes.append(
                f"**{m['filename']}**: `@interact` / `interact()` for parameter exploration"
            )
        elif "external widget" in widget_text or "%run widget" in widget_text:
            incorporation_notes.append(
                f"**{m['filename']}**: External widget file keeps notebook readable"
            )
        elif "youtube" in widget_text:
            incorporation_notes.append(
                f"**{m['filename']}**: Embedded video for follow-along guidance"
            )
        elif "dropdown" in widget_text or "slider" in widget_text:
            incorporation_notes.append(
                f"**{m['filename']}**: Dropdown/slider for data or scenario selection"
            )
        else:
            incorporation_notes.append(
                f"**{m['filename']}**: Interactive elements support exploration sections"
            )

    return {
        "notebooks": widget_notebooks,
        "types": widget_types,
        "incorporation": incorporation_notes,
    }


def classify_archetypes(
    metadata: list[dict], parsed_data: dict[str, dict]
) -> dict[str, dict]:
    """Assign notebooks to archetypes based on metadata and parsed content."""
    results = {}
    for name, config in ARCHETYPES.items():
        matches = []
        for m in metadata:
            stem = stem_from_filename(m["filename"])
            sections = parsed_data.get(stem, {})
            enriched = {
                **sections,
                "markdown_sample": sections.get("MARKDOWN CELLS", [])[:3],
            }
            if config["signals"](m, enriched):
                matches.append(m["filename"])
        results[name] = {
            "matches": matches,
            "description": config["description"],
            "common_sections": config["common_sections"],
            "teaching_methods": config["teaching_methods"],
        }
    return results


def derive_reusable_templates(metadata: list[dict]) -> list[dict]:
    """
    Cluster common section sequences into named reusable templates.
    """
    pattern_counts = analyze_patterns(metadata)

    # Manually name the most common / meaningful clusters
    templates = []
    seen_patterns = set()

    for pattern, count in pattern_counts.most_common():
        if pattern in seen_patterns or count < 1:
            continue
        seen_patterns.add(pattern)

        # Assign a template letter based on content
        sections = [s.strip() for s in pattern.split("→")]
        if "Interactive Widget" in pattern:
            name = "Template B: Interactive Exploration"
        elif "Learning Objectives" in pattern and "Visualization" in pattern:
            name = "Template A: Structured Data Lesson"
        elif "Exercise" in pattern or "Reflection" in pattern:
            name = "Template C: Exercise + Reflection"
        elif len(sections) <= 2:
            name = f"Template: {sections[0]}-Focused"
        else:
            name = f"Template: {' + '.join(sections[:3])}"

        notebooks = [
            m["filename"] for m in metadata if m.get("notebook_pattern") == pattern
        ]
        templates.append({
            "name": name,
            "sections": pattern,
            "count": count,
            "notebooks": notebooks,
        })

    # Add composite templates observed across multiple notebooks (ideal structures)
    ideal_templates = [
        {
            "name": "Template A: Introductory Data Lesson",
            "sections": "Introduction → Learning Objectives → Dataset Overview → Visualization → Interpretation Questions → Reflection",
            "count": None,
            "notebooks": [
                m["filename"]
                for m in metadata
                if m.get("contains_exercises")
                and (m.get("uses_matplotlib") or m.get("uses_seaborn"))
                and m.get("contains_reflection_questions")
            ],
        },
        {
            "name": "Template B: Interactive Exploration",
            "sections": "Background → Interactive Widget → Experimentation → Discussion",
            "count": None,
            "notebooks": [
                m["filename"]
                for m in metadata
                if m.get("uses_ipywidgets") or "Interactive Widget" in m.get("notebook_pattern", "")
            ],
        },
        {
            "name": "Template C: Coding Lab with Scaffolding",
            "sections": "Introduction → Dataset → Guided Coding → Hints → Model / Analysis → Evaluation",
            "count": None,
            "notebooks": [
                m["filename"]
                for m in metadata
                if m.get("contains_hints") and m.get("num_code_cells", 0) >= 10
            ],
        },
    ]

    # Only include ideal templates that match at least one notebook
    for t in ideal_templates:
        if t["notebooks"]:
            templates.append(t)

    return templates


def build_recommendations(
    metadata: list[dict],
    libraries: list[tuple[str, int]],
    activities: list[tuple[str, int]],
    archetypes: dict,
) -> list[str]:
    """Generate actionable recommendations from corpus analysis."""
    total = len(metadata)
    recs = []

    # Most used components
    top_libs = [lib for lib, c in libraries if c >= total // 2]
    if top_libs:
        recs.append(
            f"**Core libraries**: {', '.join(top_libs)} appear in most notebooks — "
            "include import/setup cells and brief library introductions in generated notebooks."
        )

    top_activities = [a for a, c in activities if c >= total // 3]
    if top_activities:
        recs.append(
            f"**High-value activities**: {', '.join(top_activities)} are common — "
            "generated notebooks should plan for these components explicitly."
        )

    # Structure
    patterns = analyze_patterns(metadata)
    if patterns:
        top_pattern = patterns.most_common(1)[0][0]
        recs.append(
            f"**Reusable structure**: The most frequent detected pattern is "
            f"\"{top_pattern}\". Use this as a default outline when topics fit."
        )

    # Archetypes
    strong_archetypes = [
        name for name, info in archetypes.items() if len(info["matches"]) >= 2
    ]
    if strong_archetypes:
        recs.append(
            f"**Archetypes to model**: {', '.join(strong_archetypes)} each appear "
            "multiple times — store archetype tags in metadata for retrieval."
        )

    # Future generation data
    recs.append(
        "**Store for generation**: title, learning objectives, section order, "
        "dataset description, exercise prompts, hint placement, reflection style, "
        "visualization types, widget patterns, coding level, and discipline/topic."
    )

    recs.append(
        "**Style consistency**: Many notebooks open with estimated time, table of "
        "contents, and a library-import cell — preserve these as optional template blocks."
    )

    return recs


def build_report(
    metadata: list[dict],
    parsed_data: dict[str, dict],
) -> str:
    """Assemble the full markdown report."""
    total = len(metadata)
    disciplines = count_by_key(metadata, "discipline")
    coding_levels = count_by_key(metadata, "coding_level")
    libraries = analyze_libraries(metadata)
    activities = analyze_activities(metadata)
    patterns = analyze_patterns(metadata)
    section_orders = extract_section_sequences(metadata)
    viz_activities = analyze_visualization_activities(parsed_data)
    reflection_styles = analyze_reflection_styles(parsed_data)
    widget_analysis = analyze_widgets(metadata, parsed_data)
    archetypes = classify_archetypes(metadata, parsed_data)
    templates = derive_reusable_templates(metadata)
    recommendations = build_recommendations(
        metadata, libraries, activities, archetypes
    )

    lines = [
        "# Curriculum Patterns Report",
        "",
        "Analysis of DSEP / El Camino notebook corpus for curriculum design patterns.",
        "",
        f"*Generated from {total} notebooks in `data/parsed/`*",
        "",
        "---",
        "",
        "## 1. Notebook Overview",
        "",
        f"**Total notebooks analyzed:** {total}",
        "",
        "### Disciplines represented",
        "",
    ]

    for discipline, count in disciplines.most_common():
        lines.append(f"- {discipline}: {pct(count, total)}")

    lines.extend([
        "",
        "### Coding level distribution",
        "",
    ])
    for level in ("beginner", "intermediate", "advanced"):
        count = coding_levels.get(level, 0)
        lines.append(f"- {level.capitalize()}: {pct(count, total)}")

    lines.extend([
        "",
        "### Most common libraries used",
        "",
    ])
    for lib, count in libraries:
        if count > 0:
            lines.append(f"- {lib}: {pct(count, total)}")

    lines.extend([
        "",
        "---",
        "",
        "## 2. Teaching Pattern Analysis",
        "",
        "### Most common notebook structures",
        "",
    ])
    for pattern, count in patterns.most_common():
        lines.append(f"- **{pattern}** — {count} notebook(s)")

    lines.extend([
        "",
        "### Most common section orders",
        "",
    ])
    for order, count in section_orders.most_common():
        lines.append(f"- {order} ({count})")

    lines.extend([
        "",
        "### Most common learning activity types",
        "",
    ])
    for activity, count in activities:
        lines.append(f"- {activity}: {pct(count, total)}")

    lines.extend([
        "",
        "### Most common visualization activities",
        "",
    ])
    if viz_activities:
        for viz, count in viz_activities.most_common():
            lines.append(f"- {viz}: {count} notebook(s)")
    else:
        lines.append("- No visualization patterns detected in parsed files.")

    lines.extend([
        "",
        "### Most common reflection / question styles",
        "",
    ])
    if reflection_styles:
        for style, count in reflection_styles.most_common():
            lines.append(f"- {style}: {count} notebook(s)")
    else:
        lines.append("- No reflection styles detected.")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Interactivity Analysis",
        "",
        f"**Notebooks using widgets:** {len(widget_analysis['notebooks'])} of {total}",
        "",
    ])
    if widget_analysis["notebooks"]:
        for nb in widget_analysis["notebooks"]:
            lines.append(f"- {nb}")
    else:
        lines.append("- None detected.")

    lines.extend([
        "",
        "### Types of widgets used",
        "",
    ])
    if widget_analysis["types"]:
        for wtype, count in widget_analysis["types"].most_common():
            lines.append(f"- {wtype}: {count} notebook(s)")
    else:
        lines.append("- No widget patterns parsed.")

    lines.extend([
        "",
        "### How widgets are incorporated into learning",
        "",
    ])
    for note in widget_analysis["incorporation"]:
        lines.append(f"- {note}")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Notebook Archetypes",
        "",
    ])

    for name, info in archetypes.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"**Defining characteristics:** {info['description']}")
        lines.append("")
        lines.append(f"**Common section order:** {info['common_sections']}")
        lines.append("")
        lines.append("**Common teaching methods:**")
        for method in info["teaching_methods"]:
            lines.append(f"- {method}")
        lines.append("")
        if info["matches"]:
            lines.append("**Representative notebooks:**")
            for nb in info["matches"]:
                lines.append(f"- {nb}")
        else:
            lines.append("**Representative notebooks:** None matched in this corpus.")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 5. Reusable Templates",
        "",
        "Template structures that appear reusable across the corpus:",
        "",
    ])

    for i, t in enumerate(templates, 1):
        lines.append(f"### {t['name']}")
        lines.append("")
        sections = t["sections"].replace(" → ", "\n")
        lines.append(sections)
        lines.append("")
        if t.get("count"):
            lines.append(f"*Observed in {t['count']} notebook(s)*")
        lines.append("")
        if t["notebooks"]:
            lines.append("**Example notebooks:** " + ", ".join(t["notebooks"]))
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 6. Recommendations",
        "",
        "Based on this notebook corpus:",
        "",
    ])
    for rec in recommendations:
        lines.append(f"- {rec}")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if not METADATA_PATH.exists():
        print(f"Metadata not found: {METADATA_PATH}")
        print("Run parse_notebooks.py first.")
        return

    print("Loading metadata...")
    metadata = load_metadata()
    total = len(metadata)

    print(f"Loading {total} parsed text files...")
    parsed_data = {}
    for m in metadata:
        stem = stem_from_filename(m["filename"])
        parsed_data[stem] = load_parsed_file(stem)

    print("Building curriculum patterns report...")
    report = build_report(metadata, parsed_data)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"Done. Report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()

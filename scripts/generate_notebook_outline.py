#!/usr/bin/env python3
"""
Generate a notebook outline using rule-based logic from the parsed corpus.

Reads notebook_metadata.json and curriculum_patterns_report.md, prompts the user
for notebook requirements, and writes a structured outline to data/generated/.
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "parsed" / "notebook_metadata.json"
REPORT_PATH = PROJECT_ROOT / "data" / "parsed" / "curriculum_patterns_report.md"
OUTPUT_DIR = PROJECT_ROOT / "data" / "generated"
OUTPUT_PATH = OUTPUT_DIR / "notebook_outline.md"

# Dataset description values that mean "no external data"
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

# Keywords to detect subject area from discipline + topic
MATH_KEYWORDS = {
    "math", "mathematics", "calculus", "derivative", "derivatives",
    "integral", "integrals", "algebra", "geometry", "riemann", "tangent", "slope",
}
DATA_DISCIPLINE_KEYWORDS = {
    "biology", "health", "social", "environmental", "economics", "genomics",
    "epidemiology", "marketing", "ethics", "criminal", "housing", "science",
}
CALCULUS_KEYWORDS = {
    "derivative", "derivatives", "integral", "integrals", "riemann",
    "tangent", "slope", "calculus", "area under",
}

TEMPLATES = {
    "data_lesson": {
        "name": "Template A: Introductory Data Lesson",
        "sections": [
            "Introduction",
            "Learning Objectives",
            "Import Libraries",
            "Dataset Overview",
            "Data Exploration",
            "Visualization",
            "Interpretation Questions",
            "Reflection",
            "Conclusion",
        ],
        "short_sections": [
            "Introduction",
            "Learning Objectives",
            "Dataset Overview",
            "Visualization",
            "Reflection",
        ],
    },
    "math_concept": {
        "name": "Math Concept Exploration",
        "sections": [
            "Introduction",
            "Learning Objectives",
            "Import Libraries",
            "Mathematical Background",
            "Plotting Functions",
            "Interactive Exploration",
            "Guided Exercises",
            "Reflection",
            "Conclusion",
        ],
        "short_sections": [
            "Introduction",
            "Learning Objectives",
            "Plotting Functions",
            "Interactive Exploration",
            "Reflection",
        ],
    },
    "interactive": {
        "name": "Template B: Interactive Exploration",
        "sections": [
            "Introduction",
            "Background / Context",
            "Setup (imports + widget load)",
            "Interactive Widget Exploration",
            "Guided Analysis",
            "Discussion",
            "Reflection",
            "Conclusion",
        ],
        "short_sections": [
            "Introduction",
            "Interactive Widget Exploration",
            "Discussion",
            "Reflection",
        ],
    },
    "coding_lab": {
        "name": "Template C: Coding Lab with Scaffolding",
        "sections": [
            "Introduction",
            "Learning Objectives",
            "Dataset Overview",
            "Guided Coding Exercise 1",
            "Guided Coding Exercise 2",
            "Hint + Challenge Activity",
            "Model / Analysis",
            "Evaluation",
            "Conclusion",
        ],
        "short_sections": [
            "Introduction",
            "Dataset Overview",
            "Guided Coding Exercise",
            "Evaluation",
        ],
    },
    "case_study": {
        "name": "Case Study Notebook",
        "sections": [
            "Introduction",
            "Case Background",
            "Learning Objectives",
            "Explore the Data",
            "Analysis by Theme",
            "Application / Real-World Connection",
            "Reflection",
            "Sources",
        ],
        "short_sections": [
            "Introduction",
            "Case Background",
            "Explore the Data",
            "Reflection",
        ],
    },
}

# --- Data-focused defaults (used when uses_dataset=True) ---

DATA_CODE_ACTIVITIES = {
    "beginner": [
        "Run a setup cell to import libraries",
        "Load the dataset and preview the first few rows",
        "Display basic column names and data types",
        "Create one simple plot (bar chart or line chart)",
        "Answer a short interpretation question in a markdown cell",
    ],
    "intermediate": [
        "Import libraries and load the dataset",
        "Summarize the dataset with `.describe()` or value counts",
        "Filter or group data to answer a guided question",
        "Build 2–3 visualizations (distribution, comparison, trend)",
        "Compare patterns across groups and write a short interpretation",
        "Complete a 'Your Turn' coding prompt with scaffolded starter code",
    ],
    "advanced": [
        "Load and split data for analysis or modeling",
        "Write a helper function (e.g. error metric or summary statistic)",
        "Build and evaluate a simple model or analytical pipeline",
        "Compare results across approaches with a table or plot",
        "Interpret coefficients, metrics, or outputs in context",
        "Optional challenge: add a feature or try an extension method",
    ],
}

DATA_VISUALIZATIONS = {
    "beginner": [
        "Bar chart or line chart of a key variable",
        "Simple histogram of one column",
    ],
    "intermediate": [
        "Distribution plot (histogram or KDE)",
        "Grouped bar chart or box plot for comparisons",
        "Scatter plot to explore relationships",
        "Optional map or time-series chart if the dataset supports it",
    ],
    "advanced": [
        "Feature distribution and correlation heatmap",
        "Model performance comparison chart",
        "Residual or error plot",
        "Interactive plotly chart for exploration",
    ],
}

# --- Math / no-data defaults ---

MATH_CODE_ACTIVITIES = {
    "beginner": [
        "Run a setup cell to import numpy and matplotlib",
        "Define a simple function using numpy (e.g. f(x) = x²)",
        "Plot the function over an interval",
        "Answer a short conceptual question in a markdown cell",
    ],
    "intermediate": [
        "Define one or more functions with numpy",
        "Plot functions and compare their behavior on the same axes",
        "Visualize a tangent line or secant lines at a chosen point",
        "Approximate area under a curve using rectangles (Riemann sum)",
        "Complete a 'Your Turn' prompt with scaffolded starter code",
    ],
    "advanced": [
        "Implement a numerical method (e.g. Riemann sum or finite differences)",
        "Compare approximation methods and discuss accuracy",
        "Write a reusable function for plotting with optional parameters",
        "Explore how changing parameters affects the visualization",
        "Optional challenge: compare two functions or integration methods",
    ],
}

MATH_VISUALIZATIONS = [
    "Plot f(x) over a chosen interval",
    "Tangent line at a point on the curve",
    "Secant lines approaching a tangent (derivative intuition)",
    "Riemann sum rectangles approximating area under a curve",
    "Side-by-side plots comparing functions or parameter values",
]

CALCULUS_OBJECTIVES = [
    "Explain the relationship between derivatives and slopes of functions.",
    "Use visualizations to explore how functions change over an interval.",
    "Approximate area under a curve using rectangles.",
    "Connect integrals to accumulated area under a graph.",
]

CALCULUS_WIDGET_IDEAS = [
    "Slider for the x-value where the tangent line is drawn",
    "Slider for the number of rectangles in a Riemann sum",
    "Dropdown to choose which function to plot (e.g. x², sin(x), x³)",
    "Slider for interval bounds [a, b] on the x-axis",
]

GENERAL_MATH_WIDGET_IDEAS = [
    "Slider to adjust a function parameter (e.g. coefficient or frequency)",
    "Dropdown to select which function to visualize",
    "`@interact` so students change inputs without editing code",
]

DATA_WIDGET_IDEAS = [
    "Dropdown to select a variable, region, or category to explore",
    "Slider to adjust a parameter (year, threshold, bin count)",
    "`@interact` function so students change inputs without editing code",
]

MATH_REFLECTION_PROMPTS = [
    "How does the visualization help you understand the concept?",
    "What changes when you adjust the parameter or interval?",
    "How does increasing the number of rectangles affect the approximation?",
    "Where do you see the connection between the graph and the formula?",
]

DATA_REFLECTION_PROMPTS = [
    "What patterns did you notice in the data? Were any surprising?",
    "How does this analysis connect to the real-world topic?",
    "What limitations does this dataset have? What would you investigate next?",
    "Who might use these findings, and what decisions could they inform?",
]


def load_metadata() -> list[dict]:
    """Load notebook metadata. Exit with a helpful message if missing."""
    if not METADATA_PATH.exists():
        print(f"Metadata not found: {METADATA_PATH}")
        print("Run: python3 scripts/parse_notebooks.py")
        sys.exit(1)
    with open(METADATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_report_text() -> str:
    """Load curriculum report if available (optional but recommended)."""
    if REPORT_PATH.exists():
        return REPORT_PATH.read_text(encoding="utf-8")
    return ""


def prompt_user() -> dict:
    """Collect notebook requirements from the user."""
    print("\n=== AI Notebook Builder — Outline Generator ===\n")
    print("Answer the prompts below to generate a notebook plan.\n")
    print("(Enter 'no data' if the notebook does not use an external dataset.)\n")

    discipline = input("Subject / discipline (e.g. Math, Environmental Science): ").strip()
    topic = input("Topic (e.g. derivatives and integrals): ").strip()
    dataset = input("Dataset description (or 'no data'): ").strip()

    coding_level = ""
    while coding_level not in ("beginner", "intermediate", "advanced"):
        coding_level = input("Coding level (beginner / intermediate / advanced): ").strip().lower()

    wants_widgets = input("Include widgets? (yes / no): ").strip().lower() in ("yes", "y")
    wants_reflection = input("Include reflection questions? (yes / no): ").strip().lower() in ("yes", "y")

    length = ""
    while length not in ("short", "medium", "long"):
        length = input("Approximate length (short ~30min / medium ~60min / long ~90min): ").strip().lower()

    return {
        "discipline": discipline or "General",
        "topic": topic or "Untitled Topic",
        "dataset": dataset,
        "coding_level": coding_level,
        "wants_widgets": wants_widgets,
        "wants_reflection": wants_reflection,
        "length": length,
    }


def uses_dataset(dataset_description: str) -> bool:
    """Return False when the user indicates no external dataset."""
    return dataset_description.strip().lower() not in NO_DATA_PHRASES


def detect_subject_type(discipline: str, topic: str) -> str:
    """Classify as math, data_science, or general based on keywords."""
    text = f"{discipline} {topic}".lower()
    if any(kw in text for kw in MATH_KEYWORDS):
        return "math"
    if any(kw in text for kw in DATA_DISCIPLINE_KEYWORDS):
        return "data_science"
    return "general"


def is_calculus_topic(topic: str) -> bool:
    """Check if the topic involves derivatives, integrals, or related ideas."""
    text = topic.lower()
    return any(kw in text for kw in CALCULUS_KEYWORDS)


def enrich_requirements(requirements: dict) -> dict:
    """Add derived flags used by the rest of the generator."""
    requirements["uses_dataset"] = uses_dataset(requirements["dataset"])
    requirements["subject_type"] = detect_subject_type(
        requirements["discipline"], requirements["topic"]
    )
    requirements["is_calculus"] = is_calculus_topic(requirements["topic"])
    return requirements


def normalize_discipline(discipline: str) -> str:
    """Lowercase discipline for fuzzy matching."""
    return discipline.lower().strip()


SMALL_TITLE_WORDS = {"a", "an", "the", "and", "or", "of", "in", "on", "to", "for", "with"}


def title_case_topic(topic: str) -> str:
    """Capitalize topic words, keeping small words lowercase (except first word)."""
    words = topic.split()
    result = []
    for i, word in enumerate(words):
        if i > 0 and word.lower() in SMALL_TITLE_WORDS:
            result.append(word.lower())
        else:
            result.append(word.capitalize())
    return " ".join(result)


def find_similar_notebooks(requirements: dict, metadata: list[dict]) -> list[dict]:
    """Find corpus notebooks similar to the user's requirements."""
    target_discipline = normalize_discipline(requirements["discipline"])
    subject = requirements["subject_type"]
    scored = []

    for nb in metadata:
        score = 0
        nb_discipline = normalize_discipline(nb.get("discipline", ""))
        nb_title = nb.get("title", "").lower()

        if subject == "math":
            if "math" in nb_discipline or "calculus" in nb_discipline:
                score += 3
            if any(kw in nb_title for kw in ("riemann", "orbital", "sum")):
                score += 2
        elif target_discipline in nb_discipline or nb_discipline in target_discipline:
            score += 3
        elif any(word in nb_discipline for word in target_discipline.split() if len(word) > 3):
            score += 1

        if nb.get("coding_level") == requirements["coding_level"]:
            score += 2

        if requirements["wants_widgets"] and nb.get("uses_ipywidgets"):
            score += 2

        if requirements["wants_reflection"] and nb.get("contains_reflection_questions"):
            score += 2

        if requirements["uses_dataset"] and nb.get("uses_pandas"):
            score += 1
        if not requirements["uses_dataset"] and not nb.get("uses_pandas"):
            score += 1

        if nb.get("contains_exercises"):
            score += 1

        scored.append((score, nb))

    scored.sort(key=lambda x: -x[0])

    # For math notebooks, prefer math-adjacent corpus examples
    if subject == "math":
        math_adjacent = []
        for score, nb in scored:
            nb_text = f"{nb.get('discipline', '')} {nb.get('title', '')}".lower()
            if any(kw in nb_text for kw in ("math", "calculus", "riemann", "orbital", "sum")):
                math_adjacent.append(nb)
        if math_adjacent:
            return math_adjacent[:3]

    return [nb for score, nb in scored[:3] if score > 0]


def choose_template(requirements: dict) -> str:
    """Pick the best template key based on user requirements."""
    subject = requirements["subject_type"]
    level = requirements["coding_level"]
    widgets = requirements["wants_widgets"]
    has_data = requirements["uses_dataset"]
    discipline = normalize_discipline(requirements["discipline"])

    # Math without data: concept exploration, not data lab
    if subject == "math" and not has_data:
        return "interactive" if widgets else "math_concept"

    if not has_data and widgets:
        return "interactive"

    if level == "advanced" and has_data:
        return "coding_lab"

    if widgets and has_data:
        return "interactive"

    if "case" in requirements["topic"].lower() or "marketing" in discipline:
        return "case_study"

    if subject == "data_science" or has_data:
        return "data_lesson"

    return "math_concept" if subject == "math" else "interactive"


def trim_sections(sections: list[str], requirements: dict) -> list[str]:
    """Adjust section list based on length and optional components."""
    length = requirements["length"]
    result = list(sections)

    if length == "short":
        return result[:5] if len(result) > 5 else result

    if not requirements["wants_reflection"]:
        result = [
            s for s in result
            if "reflection" not in s.lower() and "discussion" not in s.lower()
        ]

    if not requirements["wants_widgets"]:
        result = [
            s for s in result
            if "widget" not in s.lower() and "interactive widget" not in s.lower()
        ]

    return result


def get_sections(template_key: str, requirements: dict) -> tuple[str, list[str]]:
    """Return template name and section list for the outline."""
    template = TEMPLATES[template_key]
    if requirements["length"] == "short":
        sections = list(template["short_sections"])
    else:
        sections = list(template["sections"])

    # Remove data-specific sections when no dataset is used
    if not requirements["uses_dataset"]:
        sections = [
            s for s in sections
            if "dataset" not in s.lower()
            and "data exploration" not in s.lower()
            and "explore the data" not in s.lower()
        ]

    return template["name"], trim_sections(sections, requirements)


def article(word: str) -> str:
    """Return 'a' or 'an' based on the next word."""
    return "an" if word[:1].lower() in "aeiou" else "a"


def generate_title(requirements: dict) -> str:
    """Build a notebook title matched to subject and data usage."""
    topic = title_case_topic(requirements["topic"])
    discipline = requirements["discipline"]
    subject = requirements["subject_type"]
    has_data = requirements["uses_dataset"]

    if subject == "math" and not has_data:
        if requirements["wants_widgets"]:
            return f"Exploring {topic} with Interactive Visualizations"
        return f"Exploring {topic} through Visualization and Computation"

    if not has_data:
        return f"Exploring {topic}: {article(discipline)} {discipline} Notebook"

    return f"Exploring {topic}: {article(discipline)} {discipline} Data Notebook"


def generate_intro(requirements: dict) -> str:
    """Write a short introductory paragraph."""
    time_map = {"short": "30", "medium": "60", "long": "90"}
    minutes = time_map[requirements["length"]]
    topic = requirements["topic"]
    discipline = requirements["discipline"]
    level = requirements["coding_level"]

    if not requirements["uses_dataset"]:
        approach = "interactive visualizations and Python code"
        if requirements["wants_widgets"]:
            approach = "interactive widgets, visualizations, and Python code"
        return (
            f"In this notebook, you will explore **{topic}** through **{approach}**. "
            f"No external dataset is required. This activity is designed for "
            f"{level}-level learners in {discipline} and should take approximately "
            f"{minutes} minutes to complete."
        )

    return (
        f"In this notebook, you will explore **{topic}** using "
        f"**{requirements['dataset']}**. This activity is designed for "
        f"{level}-level learners in {discipline} and should take approximately "
        f"{minutes} minutes to complete."
    )


def generate_learning_objectives(requirements: dict, sections: list[str]) -> list[str]:
    """Create learning objectives based on subject, data usage, and sections."""
    topic = requirements["topic"]
    level = requirements["coding_level"]
    subject = requirements["subject_type"]
    has_data = requirements["uses_dataset"]

    # Calculus / math without data
    if subject == "math" and not has_data:
        objectives = list(CALCULUS_OBJECTIVES) if requirements["is_calculus"] else [
            f"Explain the key ideas behind {topic}.",
            "Use numpy and matplotlib to define and plot functions.",
            "Explore how changing parameters affects graphs and behavior.",
            "Connect visual patterns to mathematical definitions.",
        ]
        if requirements["wants_widgets"]:
            objectives.append("Use interactive sliders to explore functions and parameters.")
        if requirements["wants_reflection"]:
            objectives.append("Reflect on how visualization deepens conceptual understanding.")
        return objectives

    # Data-focused notebooks
    objectives = [f"Describe the context and purpose of studying {topic}."]

    if has_data:
        objectives.append(f"Load and explore {requirements['dataset']}.")
    else:
        objectives.append("Use Python to simulate or compute examples related to the topic.")

    if any("visual" in s.lower() for s in sections):
        if has_data:
            objectives.append("Create and interpret visualizations to identify patterns in the data.")
        else:
            objectives.append("Create and interpret visualizations to build intuition.")

    if level in ("intermediate", "advanced") and has_data:
        objectives.append("Apply data analysis techniques to answer guided questions.")

    if level == "advanced" and has_data:
        objectives.append("Build and evaluate an analytical or modeling approach using Python.")

    if requirements["wants_widgets"]:
        objectives.append("Use interactive tools to explore how changes affect results.")

    if requirements["wants_reflection"]:
        if subject == "data_science" or has_data:
            objectives.append("Reflect on findings and connect results to real-world implications.")
        else:
            objectives.append("Reflect on what the visualizations reveal about the concept.")

    return objectives


def generate_libraries(requirements: dict) -> list[str]:
    """Suggest libraries based on subject, data usage, and widgets."""
    level = requirements["coding_level"]
    subject = requirements["subject_type"]
    has_data = requirements["uses_dataset"]
    widgets = requirements["wants_widgets"]

    # Math without data: numpy + matplotlib only (+ ipywidgets if requested)
    if subject == "math" and not has_data:
        libs = ["numpy", "matplotlib"]
        if widgets:
            libs.append("ipywidgets")
        return libs

    if not has_data:
        libs = ["numpy", "matplotlib"]
        if widgets:
            libs.append("ipywidgets")
        return libs

    # Data-focused libraries by level
    if level == "beginner":
        libs = ["pandas", "numpy", "matplotlib"]
    elif level == "intermediate":
        libs = ["pandas", "numpy", "matplotlib", "seaborn"]
    else:
        libs = ["pandas", "numpy", "matplotlib", "seaborn", "plotly", "sklearn"]

    if widgets:
        libs.append("ipywidgets")

    return libs


def get_export_template(requirements: dict) -> str | None:
    """Tag outlines that map to a rich export template in export_to_ipynb.py."""
    if (
        requirements["subject_type"] == "math"
        and not requirements["uses_dataset"]
        and requirements["is_calculus"]
    ):
        return "math_calculus_interactive"
    return None


def generate_code_activities(requirements: dict) -> list[str]:
    """Suggest code activities matched to subject and data usage."""
    level = requirements["coding_level"]
    subject = requirements["subject_type"]
    has_data = requirements["uses_dataset"]

    if subject == "math" and not has_data and requirements["is_calculus"]:
        return [
            "Import numpy, matplotlib, and ipywidgets",
            "Define a function such as f(x) = x²",
            "Plot the function over an interval",
            "Visualize a tangent line with a slider for the x-value",
            "Visualize Riemann sum rectangles with a slider for the number of rectangles",
        ]

    if subject == "math" and not has_data:
        base = list(MATH_CODE_ACTIVITIES[level])
    elif has_data:
        base = list(DATA_CODE_ACTIVITIES[level])
    else:
        base = [
            "Run a setup cell to import libraries",
            "Define or simulate values using numpy",
            "Create a plot to visualize the concept",
            "Answer a guided question in a markdown cell",
        ]

    if requirements["wants_widgets"]:
        if requirements["is_calculus"]:
            base.append("Use sliders to adjust tangent point, rectangle count, or interval bounds")
        else:
            base.append("Run an interactive widget cell to explore a parameter")

    if requirements["length"] == "short":
        return base[:4]
    if requirements["length"] == "medium":
        return base
    return base + ["Extension: complete an optional challenge activity"]


def generate_visualizations(requirements: dict, similar: list[dict]) -> list[str]:
    """Suggest visualizations based on subject and data usage."""
    subject = requirements["subject_type"]
    has_data = requirements["uses_dataset"]
    level = requirements["coding_level"]

    if subject == "math" and not has_data:
        viz = list(MATH_VISUALIZATIONS)
        if requirements["is_calculus"]:
            return viz[:5]
        return viz[:3]

    if not has_data:
        return [
            "Plot of a function or simulated values",
            "Comparison plot showing how a parameter changes the output",
        ]

    viz = list(DATA_VISUALIZATIONS[level])

    # Only borrow corpus viz patterns when using data
    for nb in similar:
        if nb.get("uses_plotly") and "Interactive plotly chart" not in viz:
            viz.append("Interactive plotly chart (used in similar corpus notebooks)")
        if nb.get("uses_seaborn") and "seaborn" not in " ".join(viz).lower():
            viz.append("Seaborn statistical plot (used in similar corpus notebooks)")

    return viz[:5]


def generate_widget_ideas(requirements: dict, similar: list[dict]) -> list[str]:
    """Suggest topic-specific widget ideas."""
    if not requirements["wants_widgets"]:
        return []

    if requirements["is_calculus"]:
        ideas = list(CALCULUS_WIDGET_IDEAS)
    elif requirements["subject_type"] == "math":
        ideas = list(GENERAL_MATH_WIDGET_IDEAS)
    else:
        ideas = list(DATA_WIDGET_IDEAS)

    for nb in similar:
        if nb.get("uses_ipywidgets"):
            ideas.append(f"See inspiration: {nb['filename']} (uses widgets in corpus)")
            break

    return ideas


def generate_reflection_questions(requirements: dict) -> list[str]:
    """Generate reflection/discussion questions matched to subject."""
    if not requirements["wants_reflection"]:
        return []

    topic = requirements["topic"]
    subject = requirements["subject_type"]
    has_data = requirements["uses_dataset"]

    if subject == "math" or not has_data:
        questions = list(MATH_REFLECTION_PROMPTS[:3])
        if requirements["is_calculus"]:
            questions[0] = "How does the tangent line visualization connect to the idea of a derivative?"
        return questions

    questions = [
        f"What did you learn about {topic} from this data?",
        DATA_REFLECTION_PROMPTS[1],
        DATA_REFLECTION_PROMPTS[2],
    ]

    if "ethics" in normalize_discipline(requirements["discipline"]):
        questions.append("What ethical or equity considerations should we keep in mind?")

    if requirements["length"] == "long":
        questions.append(DATA_REFLECTION_PROMPTS[3])

    return questions


def generate_extension(requirements: dict) -> str | None:
    """Suggest an extension activity for longer notebooks."""
    if requirements["length"] != "long":
        return None

    if requirements["subject_type"] == "math" and not requirements["uses_dataset"]:
        if requirements["is_calculus"]:
            return (
                "Compare left-endpoint, right-endpoint, and midpoint Riemann sums for "
                "the same function and interval."
            )
        return "Explore a different function and describe how its graph changes with parameters."

    if requirements["uses_dataset"]:
        return (
            "Add a new visualization using a column not yet explored, or compare "
            "two groups and explain the difference."
        )

    return "Try an optional challenge exercise with a hint provided."


def extract_report_recommendation(report_text: str, requirements: dict) -> str:
    """Pull corpus guidance relevant to this notebook type."""
    if not report_text:
        return "Run analyze_corpus.py to generate curriculum_patterns_report.md for richer suggestions."

    if requirements["subject_type"] == "math" and not requirements["uses_dataset"]:
        return (
            "Corpus reference: `sums.ipynb` and `Orbitals.ipynb` use numpy, matplotlib, "
            "and widgets for math/science visualization without external datasets."
        )

    match = re.search(r"\*\*Core libraries\*\*:.*", report_text)
    if match:
        return match.group(0).strip("- ")
    return "Follow DSEP / El Camino notebook conventions: table of contents, estimated time, import cell."


def build_outline(
    requirements: dict,
    metadata: list[dict],
    report_text: str,
) -> str:
    """Assemble the full markdown outline."""
    requirements = enrich_requirements(requirements)

    template_key = choose_template(requirements)
    template_name, sections = get_sections(template_key, requirements)
    similar = find_similar_notebooks(requirements, metadata)

    title = generate_title(requirements)
    intro = generate_intro(requirements)
    objectives = generate_learning_objectives(requirements, sections)
    code_activities = generate_code_activities(requirements)
    visualizations = generate_visualizations(requirements, similar)
    widget_ideas = generate_widget_ideas(requirements, similar)
    reflections = generate_reflection_questions(requirements)
    extension = generate_extension(requirements)
    libraries = generate_libraries(requirements)
    corpus_note = extract_report_recommendation(report_text, requirements)

    data_note = "Uses dataset: Yes" if requirements["uses_dataset"] else "Uses dataset: No"
    subject_note = f"Subject type: {requirements['subject_type']}"
    export_template = get_export_template(requirements)
    reflection_note = (
        "Yes" if requirements["wants_reflection"] else "No"
    )

    lines = [
        "# Notebook Outline",
        "",
        f"**Generated for:** {requirements['discipline']} — {requirements['topic']}",
        f"**{data_note}** | **{subject_note}**",
        f"**Reflection questions:** {reflection_note}",
    ]
    if export_template:
        lines.append(f"**Export template:** {export_template}")
    lines.extend([
        "",
        "---",
        "",
        "## Title",
        "",
        title,
        "",
        "## Short Introduction",
        "",
        intro,
        "",
        "## Learning Objectives",
        "",
    ])

    for obj in objectives:
        lines.append(f"- {obj}")

    lines.extend([
        "",
        "## Suggested Section Order",
        "",
        f"*Based on: {template_name}*",
        "",
    ])

    for i, section in enumerate(sections, 1):
        lines.append(f"{i}. {section}")

    lines.extend([
        "",
        "## Suggested Libraries",
        "",
        ", ".join(libraries),
        "",
        f"*Corpus guidance: {corpus_note}*",
        "",
        "## Suggested Code Activities",
        "",
    ])

    for activity in code_activities:
        lines.append(f"- {activity}")

    lines.extend([
        "",
        "## Suggested Visualizations",
        "",
    ])

    for viz in visualizations:
        lines.append(f"- {viz}")

    lines.extend([
        "",
        "## Optional Widget Ideas",
        "",
    ])

    if widget_ideas:
        for idea in widget_ideas:
            lines.append(f"- {idea}")
    else:
        lines.append("- None (widgets not requested)")

    lines.extend([
        "",
        "## Reflection / Discussion Questions",
        "",
    ])

    if reflections:
        for q in reflections:
            lines.append(f"- {q}")
    else:
        lines.append("- None (reflection not requested)")

    lines.extend([
        "",
        "## Possible Extension Activity",
        "",
    ])

    if extension:
        lines.append(extension)
    else:
        lines.append("None for this length — consider adding one for a longer notebook.")

    if similar:
        lines.extend([
            "",
            "## Similar Notebooks in Corpus",
            "",
            "Use these as style references when building the notebook:",
            "",
        ])
        for nb in similar:
            lines.append(
                f"- **{nb['title']}** (`{nb['filename']}`) — "
                f"{nb['discipline']}, {nb['coding_level']} level, "
                f"pattern: {nb.get('notebook_pattern', 'N/A')}"
            )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    metadata = load_metadata()
    report_text = load_report_text()

    if not report_text:
        print("Note: curriculum_patterns_report.md not found. Using metadata only.")
        print("Run: python3 scripts/analyze_corpus.py for richer outlines.\n")

    requirements = prompt_user()
    outline = build_outline(requirements, metadata, report_text)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(outline, encoding="utf-8")

    print(f"\nOutline saved to {OUTPUT_PATH}")
    print("Open the file to review your notebook plan.")


if __name__ == "__main__":
    main()

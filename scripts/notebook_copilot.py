#!/usr/bin/env python3
"""
Notebook Copilot — search, retrieval, and rule-based synthesis over the corpus.

Builds local embeddings from data/parsed/*_parsed.txt files, retrieves relevant
chunks, and synthesizes a concise answer for curriculum developers.

Usage:
    python3 scripts/notebook_copilot.py                  # interactive (synthesized)
    python3 scripts/notebook_copilot.py --raw            # interactive (raw results only)
    python3 scripts/notebook_copilot.py --draft "..."    # draft a notebook section
    python3 scripts/notebook_copilot.py build            # rebuild the index
    python3 scripts/notebook_copilot.py "your query"       # one-shot synthesized answer
    python3 scripts/notebook_copilot.py --raw "query"    # one-shot raw results only

Optional: pip install sentence-transformers for semantic embeddings
(see requirements-copilot.txt). Falls back to TF-IDF if not installed.
"""

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
GENERATED_DIR = PROJECT_ROOT / "data" / "generated"
METADATA_PATH = PARSED_DIR / "notebook_metadata.json"
DRAFT_PATH = GENERATED_DIR / "draft_section.md"

CHUNKS_PATH = EMBEDDINGS_DIR / "chunks.json"
EMBEDDINGS_PATH = EMBEDDINGS_DIR / "embeddings.npy"
VOCAB_PATH = EMBEDDINGS_DIR / "vocabulary.json"
INDEX_META_PATH = EMBEDDINGS_DIR / "index_meta.json"

TOP_K = 5
MAX_DISPLAY_CHARS = 600
MIN_CHUNK_CHARS = 30

# Sections where short items are combined into one chunk
COMBINE_ITEM_SECTIONS = {
    "WIDGET PATTERNS",
    "VISUALIZATION PATTERNS",
    "IMPORTS / LIBRARIES",
}

# Boost retrieval when query mentions these topics
DISCIPLINE_QUERY_MAP = {
    "biology": ["biology", "genomics", "biomanufacturing", "cancer"],
    "public health": ["public health", "epidemiology", "prams", "prenatal"],
    "math": ["math", "mathematics", "calculus", "riemann"],
    "social science": ["social", "criminal", "incarceration", "justice"],
    "environmental": ["environmental", "calenviro", "pollution"],
    "ethics": ["ethics", "trolley"],
    "marketing": ["marketing", "business"],
    "chemistry": ["chemistry", "physics", "orbital"],
    "economics": ["economics", "housing", "regression"],
}

SECTION_QUERY_MAP = {
    "widget": ["widget patterns"],
    "reflection": ["reflection questions"],
    "learning objective": ["learning objectives"],
    "dataset": ["dataset descriptions"],
    "exercise": ["exercises"],
    "hint": ["hints"],
    "visualization": ["visualization patterns"],
    "code": ["code cells"],
}

# Query topic keywords for synthesis routing
QUERY_TOPIC_KEYWORDS = {
    "widget": ["widget", "interactive", "ipywidgets", "slider", "dropdown", "interact"],
    "reflection": ["reflection", "discussion", "reflect", "free response"],
    "learning objective": ["learning objective", "learning outcomes", "learning goals"],
    "dataset": ["dataset", "data overview", "introduce data", "load data", "column"],
    "exercise": ["exercise", "question", "coding exercise", "your turn"],
    "hint": ["hint", "scaffold", "scaffolding"],
    "visualization": ["visualization", "plot", "chart", "graph", "matplotlib"],
}

# Rule-based reuse tips per topic (generic guidance for interns)
REUSE_TIPS = {
    "widget": [
        "Add one setup cell that imports widget libraries before the interactive section.",
        "Keep complex UI in a separate file if the notebook becomes hard to read.",
        "Pair each widget with a short markdown prompt explaining what to change and observe.",
    ],
    "reflection": [
        "Place 1–3 reflection prompts after major sections, not only at the end.",
        "Use open-ended questions that connect results to real-world context.",
        "Leave blank markdown cells or 'Your answer here' placeholders for student responses.",
    ],
    "learning objective": [
        "List 4–7 objectives at the start using 'By the end of this notebook, you will...'",
        "Mix conceptual goals (explain, interpret) with skill goals (use Python to...).",
        "Align each objective with at least one later section or exercise.",
    ],
    "dataset": [
        "Introduce the dataset with source, row meaning, and a column guide before coding.",
        "Show a small preview table so students know what each row represents.",
        "Explain why the dataset matters for the topic before asking analysis questions.",
    ],
    "exercise": [
        "Scaffold exercises: context → prompt → starter code → 'Your answer here' cell.",
        "Number questions and place them immediately after the concept they reinforce.",
        "For beginners, keep exercises short and run code cells frequently.",
    ],
    "hint": [
        "Place hints directly below the question, marked with *Hint:* or **Hint:**.",
        "Give hints that point to a function or approach without revealing the full answer.",
    ],
    "visualization": [
        "Introduce each chart with what to look for before showing the code cell.",
        "Follow plots with an interpretation question in markdown.",
    ],
    "general": [
        "Review the cited notebooks for section order and pacing.",
        "Adapt the closest example to your discipline before inventing a new structure.",
    ],
}

# Section types for --draft mode (order matters: more specific phrases first)
DRAFT_SECTION_KEYWORDS = [
    ("learning objectives", ["learning objective", "learning outcomes", "learning goals"]),
    ("reflection questions", ["reflection question", "reflection section", "discussion question"]),
    ("dataset overview", ["dataset overview", "dataset section", "data overview", "introduce dataset"]),
    ("widget section", ["widget section", "interactive section", "widget", "interact", "slider"]),
    ("coding exercise", ["coding exercise", "exercise section", "your turn", "coding activity"]),
    ("introduction", ["introduction", "intro section", "opening section"]),
]

# Map draft section -> corpus section names for retrieval filtering
DRAFT_SECTION_TO_CORPUS = {
    "learning objectives": "learning objectives",
    "introduction": "introduction",
    "dataset overview": "dataset descriptions",
    "widget section": "widget patterns",
    "coding exercise": "exercises",
    "reflection questions": "reflection questions",
}

# Discipline keywords in draft prompts
DRAFT_DISCIPLINE_KEYWORDS = {
    "Mathematics": ["calculus", "math", "mathematics", "riemann", "integral", "derivative"],
    "Biology": ["biology", "enzyme", "biomanufacturing", "genomics", "cancer"],
    "Ethics / Data Science": ["ethics", "ai ethics", "trolley"],
    "Public Health": ["public health", "epidemiology", "health", "prams"],
    "Environmental Science": ["environmental", "calenviro", "pollution"],
    "Social Science": ["social science", "incarceration", "criminal justice"],
    "Chemistry / Physics": ["chemistry", "physics", "orbital"],
    "Business / Marketing": ["marketing", "business"],
    "Economics": ["economics", "housing", "regression"],
}

# Try semantic embeddings if sentence-transformers is installed
try:
    from sentence_transformers import SentenceTransformer

    SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    EMBEDDING_METHOD = "semantic"
except ImportError:
    SEMANTIC_MODEL = None
    EMBEDDING_METHOD = "tfidf"


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer for TF-IDF."""
    return re.findall(r"[a-z0-9]+", text.lower())


def load_metadata_by_filename() -> dict[str, dict]:
    """Map notebook filename -> metadata entry."""
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH, encoding="utf-8") as f:
        entries = json.load(f)
    return {e["filename"]: e for e in entries}


def parse_parsed_file(path: Path, metadata_map: dict) -> list[dict]:
    """
    Split one *_parsed.txt file into searchable chunks.
    Each chunk = one section item with notebook + section citations.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    notebook = path.stem.replace("_parsed", "") + ".ipynb"
    if lines and lines[0].startswith("Notebook:"):
        notebook = lines[0].replace("Notebook:", "").strip()

    meta = metadata_map.get(notebook, {})
    discipline = meta.get("discipline", "Unknown")
    title = meta.get("title", notebook)
    coding_level = meta.get("coding_level", "unknown")

    chunks = []

    def add_chunk(section_name: str, item_text: str) -> None:
        if not item_text or item_text == "(none found)":
            return
        if len(item_text) < MIN_CHUNK_CHARS and section_name not in COMBINE_ITEM_SECTIONS:
            return
        search_text = (
            f"Notebook: {notebook}\n"
            f"Discipline: {discipline}\n"
            f"Coding level: {coding_level}\n"
            f"Section: {section_name}\n"
            f"{item_text}"
        )
        chunks.append({
            "notebook": notebook,
            "title": title,
            "discipline": discipline,
            "coding_level": coding_level,
            "section": section_name,
            "text": item_text,
            "search_text": search_text,
        })

    # Split on section headers
    parts = re.split(r"\n=== (.+?) ===\n", text)

    for i in range(1, len(parts), 2):
        section_name = parts[i].strip()
        content = parts[i + 1] if i + 1 < len(parts) else ""

        raw_items = re.split(r"--- Item \d+ ---\n", content)
        item_texts = [t.strip() for t in raw_items if t.strip() and t.strip() != "(none found)"]

        # Combine short pattern sections into one chunk
        if section_name in COMBINE_ITEM_SECTIONS and item_texts:
            combined = "\n".join(f"- {t}" for t in item_texts)
            add_chunk(section_name, combined)
            continue

        for item_text in item_texts:
            add_chunk(section_name, item_text)

    return chunks


def load_all_chunks() -> list[dict]:
    """Load chunks from every parsed notebook text file."""
    metadata_map = load_metadata_by_filename()
    parsed_files = sorted(PARSED_DIR.glob("*_parsed.txt"))

    if not parsed_files:
        print(f"No parsed files found in {PARSED_DIR}")
        print("Run: python3 scripts/parse_notebooks.py")
        sys.exit(1)

    all_chunks = []
    for path in parsed_files:
        file_chunks = parse_parsed_file(path, metadata_map)
        all_chunks.extend(file_chunks)
        print(f"  Chunked {path.name}: {len(file_chunks)} chunks")

    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def build_tfidf_embeddings(chunks: list[dict]) -> tuple[np.ndarray, dict]:
    """Build TF-IDF vectors for all chunks. Returns (matrix, vocab_info)."""
    docs_tokens = [tokenize(c["search_text"]) for c in chunks]
    n_docs = len(docs_tokens)

    # Document frequency per term
    df: dict[str, int] = {}
    for tokens in docs_tokens:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    vocabulary = sorted(df.keys())
    idf = {term: math.log((1 + n_docs) / (1 + df[term])) + 1 for term in vocabulary}
    term_to_idx = {term: i for i, term in enumerate(vocabulary)}

    matrix = np.zeros((n_docs, len(vocabulary)), dtype=np.float32)
    for doc_i, tokens in enumerate(docs_tokens):
        tf: dict[str, int] = {}
        for term in tokens:
            tf[term] = tf.get(term, 0) + 1
        max_tf = max(tf.values()) if tf else 1
        for term, count in tf.items():
            if term in term_to_idx:
                weight = (count / max_tf) * idf[term]
                matrix[doc_i, term_to_idx[term]] = weight

    # L2-normalize rows for cosine similarity via dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    matrix = matrix / norms

    vocab_info = {
        "method": "tfidf",
        "vocabulary": vocabulary,
        "idf": idf,
    }
    return matrix, vocab_info


def embed_query_tfidf(query: str, vocab_info: dict) -> np.ndarray:
    """Embed a query string using the saved TF-IDF vocabulary."""
    vocabulary = vocab_info["vocabulary"]
    idf = vocab_info["idf"]
    term_to_idx = {term: i for i, term in enumerate(vocabulary)}

    tokens = tokenize(query)
    vec = np.zeros(len(vocabulary), dtype=np.float32)
    tf: dict[str, int] = {}
    for term in tokens:
        tf[term] = tf.get(term, 0) + 1
    max_tf = max(tf.values()) if tf else 1

    for term, count in tf.items():
        if term in term_to_idx:
            vec[term_to_idx[term]] = (count / max_tf) * idf[term]

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def build_semantic_embeddings(chunks: list[dict]) -> np.ndarray:
    """Build embeddings with sentence-transformers."""
    texts = [c["search_text"] for c in chunks]
    embeddings = SEMANTIC_MODEL.encode(texts, show_progress_bar=True)
    return np.array(embeddings, dtype=np.float32)


def embed_query_semantic(query: str) -> np.ndarray:
    """Embed a query with sentence-transformers."""
    return np.array(SEMANTIC_MODEL.encode([query])[0], dtype=np.float32)


def build_index() -> None:
    """Chunk parsed notebooks, embed, and save locally."""
    print("Building notebook search index...\n")
    chunks = load_all_chunks()

    if not chunks:
        print("No chunks to index.")
        sys.exit(1)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    if EMBEDDING_METHOD == "semantic":
        print("Using semantic embeddings (sentence-transformers)...")
        embeddings = build_semantic_embeddings(chunks)
        vocab_info = {"method": "semantic", "model": "all-MiniLM-L6-v2"}
    else:
        print("Using TF-IDF embeddings (install sentence-transformers for semantic search).")
        embeddings, vocab_info = build_tfidf_embeddings(chunks)

    # Save index
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    np.save(EMBEDDINGS_PATH, embeddings)

    if vocab_info.get("method") == "tfidf":
        with open(VOCAB_PATH, "w", encoding="utf-8") as f:
            json.dump(vocab_info, f)

    index_meta = {
        "method": vocab_info.get("method", "tfidf"),
        "num_chunks": len(chunks),
        "embedding_dim": embeddings.shape[1],
    }
    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(index_meta, f, indent=2)

    print(f"\nIndex saved to {EMBEDDINGS_DIR}")
    print(f"  {len(chunks)} chunks, {embeddings.shape[1]}-dim embeddings ({index_meta['method']})")


def load_index() -> tuple[list[dict], np.ndarray, dict]:
    """Load chunks, embeddings, and vocab/metadata from disk."""
    if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        return [], np.array([]), {}

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(EMBEDDINGS_PATH)

    vocab_info = {}
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, encoding="utf-8") as f:
            vocab_info = json.load(f)
    elif INDEX_META_PATH.exists():
        with open(INDEX_META_PATH, encoding="utf-8") as f:
            vocab_info = json.load(f)

    return chunks, embeddings, vocab_info


def apply_retrieval_boosts(query: str, chunks: list[dict], scores: np.ndarray) -> np.ndarray:
    """Boost scores when query mentions a discipline, section, or coding level."""
    boosted = scores.copy()
    q = query.lower()

    for keywords in DISCIPLINE_QUERY_MAP.values():
        if any(kw in q for kw in keywords):
            for i, chunk in enumerate(chunks):
                disc = chunk.get("discipline", "").lower()
                if any(kw in disc for kw in keywords):
                    boosted[i] += 0.15

    for trigger, section_names in SECTION_QUERY_MAP.items():
        if trigger in q:
            for i, chunk in enumerate(chunks):
                if chunk.get("section", "").lower() in section_names:
                    boosted[i] += 0.20

    for level in ("beginner", "intermediate", "advanced"):
        if level in q:
            for i, chunk in enumerate(chunks):
                if chunk.get("coding_level") == level:
                    boosted[i] += 0.10

    return boosted


def search(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Find the most similar chunks to the query.
    Returns list of dicts with chunk info and similarity score.
    """
    chunks, embeddings, vocab_info = load_index()

    if not chunks:
        print("Index not found. Building now...\n")
        build_index()
        chunks, embeddings, vocab_info = load_index()

    method = vocab_info.get("method", "tfidf")

    if method == "semantic" and SEMANTIC_MODEL is not None:
        query_vec = embed_query_semantic(query)
    else:
        if not vocab_info.get("vocabulary"):
            # Rebuild vocab from saved chunks if needed
            _, vocab_info = build_tfidf_embeddings(chunks)
        query_vec = embed_query_tfidf(query, vocab_info)

    # Cosine similarity (vectors are normalized for tfidf; semantic model returns normalized)
    scores = embeddings @ query_vec
    scores = apply_retrieval_boosts(query, chunks, scores)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        results.append({
            **chunks[int(idx)],
            "score": float(scores[idx]),
        })

    return results


def truncate_text(text: str, max_chars: int = MAX_DISPLAY_CHARS) -> str:
    """Trim long text for display."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def detect_query_topic(query: str) -> str:
    """Pick a synthesis template based on query keywords."""
    q = query.lower()
    for topic, keywords in QUERY_TOPIC_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return topic
    return "general"


def dedupe_by_notebook(results: list[dict]) -> list[dict]:
    """Keep the top-scoring chunk per notebook for cleaner examples."""
    seen: set[str] = set()
    unique = []
    for r in results:
        nb = r["notebook"]
        if nb not in seen:
            seen.add(nb)
            unique.append(r)
    return unique


def extract_bullet_items(text: str) -> list[str]:
    """Pull bullet lines from chunk text."""
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("- "):
            items.append(line[2:].strip())
    return items


def first_sentence(text: str, max_len: int = 200) -> str:
    """Return the first sentence or line of text for brief quotes."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return ""
    match = re.match(r"^(.+?[.!?])(\s|$)", text)
    snippet = match.group(1) if match else text
    if len(snippet) > max_len:
        return snippet[:max_len].rstrip() + "..."
    return snippet


def is_vague_chunk(text: str, section: str) -> bool:
    """True when retrieved text lacks enough detail to describe usage."""
    text = text.strip()
    if section in COMBINE_ITEM_SECTIONS:
        lines = extract_bullet_items(text)
        # Only short labels like "ipywidgets library" with no prose
        if lines and all(len(line) < 50 for line in lines) and len(text) < 300:
            return True
    if len(text) < 80:
        return True
    return False


def infer_widget_usage(text: str) -> str:
    """Describe widget usage only from phrases found in the chunk."""
    t = text.lower()
    usages = []
    checks = [
        (["%run widget", "external widget file", "widget.ipy"], "loads interactivity from an external widget file"),
        (["@interact", "interact("], "uses `@interact` for parameter exploration"),
        (["dropdown"], "includes a dropdown for selecting options"),
        (["slider"], "uses sliders to adjust parameters"),
        (["button"], "includes button-based interaction"),
        (["youtube"], "embeds a YouTube video for follow-along guidance"),
        (["animation"], "uses animation as part of the interactive experience"),
        (["checkbox"], "includes checkbox widgets"),
    ]
    for keywords, description in checks:
        if any(kw in t for kw in keywords):
            usages.append(description)
    return "; ".join(usages)


def infer_general_usage(text: str, topic: str) -> str:
    """Infer how content is used, using only text evidence."""
    t = text.lower()
    if topic == "dataset":
        if "column" in t:
            return "describes columns or variables in the dataset"
        if "load" in t or "read" in t:
            return "introduces how to load the dataset"
        if "row" in t or "granularity" in t:
            return "explains what each row represents"
    if topic == "reflection":
        if "reflect" in t or "discussion" in t:
            return "prompts reflection or discussion"
        if "your answer" in t:
            return "asks students to write a response in a markdown cell"
    if topic == "exercise":
        if "your turn" in t:
            return "structured as a 'Your Turn' coding prompt"
        if "question" in t:
            return "poses a guided question for students"
        if "your answer" in t:
            return "includes a placeholder for student answers"
    if topic == "hint":
        if "hint" in t:
            return "provides a hint below a question"
    if topic == "visualization":
        items = extract_bullet_items(text)
        if items:
            return "uses: " + ", ".join(items[:4])
    if topic == "learning objective":
        if "by the end" in t or "you will" in t:
            return "lists objectives students should meet by the end"
    return first_sentence(text, 150)


def build_example_entry(result: dict, topic: str) -> dict:
    """Build one structured example from a retrieved chunk."""
    text = result["text"]
    section = result.get("section", "")

    entry = {
        "notebook": result["notebook"],
        "discipline": result.get("discipline", "Unknown"),
        "section": section,
        "vague": False,
    }

    if topic == "widget":
        types = extract_bullet_items(text)
        if not types:
            # Scan for widget-related labels in prose
            for label in ("ipywidgets", "dropdown", "slider", "interact", "button"):
                if label in text.lower():
                    types.append(label)
        entry["types"] = types or ["(type not specified in retrieved text)"]
        entry["usage"] = infer_widget_usage(text)
    elif topic == "visualization":
        entry["types"] = extract_bullet_items(text) or ["(see retrieved text)"]
        entry["usage"] = infer_general_usage(text, topic)
    else:
        entry["types"] = [section]
        entry["usage"] = infer_general_usage(text, topic)

    entry["snippet"] = first_sentence(text, 180)
    # Vague only when we lack both usage inference and substantive prose
    entry["vague"] = not entry["usage"] and is_vague_chunk(text, section)
    return entry


def summarize_patterns(query: str, topic: str, examples: list[dict], results: list[dict]) -> str:
    """Write a short cross-notebook summary from retrieved evidence only."""
    if not results:
        return "No matching notebook content was found. Try rephrasing your question."

    notebooks = {r["notebook"] for r in results}
    disciplines = sorted({r.get("discipline", "Unknown") for r in results})
    disc_text = ", ".join(disciplines[:4])
    if len(disciplines) > 4:
        disc_text += ", ..."

    if topic == "widget":
        all_types: list[str] = []
        for ex in examples:
            all_types.extend(ex.get("types", []))
        unique_types = list(dict.fromkeys(all_types))  # preserve order, dedupe
        type_text = ", ".join(unique_types[:6]) if unique_types else "unspecified widget types"
        vague_count = sum(1 for ex in examples if ex["vague"])
        summary = (
            f"Retrieved {len(results)} chunk(s) from {len(notebooks)} notebook(s) "
            f"({disc_text}). Common widget-related patterns include: {type_text}."
        )
        if vague_count:
            summary += (
                f" {vague_count} example(s) only list widget types without usage details "
                "— check those notebooks for full implementation."
            )
        return summary

    sections = sorted({r.get("section", "") for r in results})
    section_text = ", ".join(sections[:4])
    summary = (
        f"Retrieved {len(results)} chunk(s) from {len(notebooks)} notebook(s) "
        f"({disc_text}), mainly from sections: {section_text}."
    )

    vague_count = sum(1 for ex in examples if ex["vague"])
    if vague_count:
        summary += (
            f" Some retrieved text is brief — open the cited notebooks for full context."
        )
    return summary


def format_specific_examples(examples: list[dict], topic: str) -> list[str]:
    """Format the specific examples section."""
    lines = []
    for i, ex in enumerate(examples, 1):
        lines.append(f"**Example {i}: {ex['notebook']}**")
        lines.append(f"- Discipline: {ex['discipline']}")
        if topic == "widget":
            types = ", ".join(ex.get("types", []))
            lines.append(f"- Widget type(s): {types}")
        elif topic == "visualization":
            types = ", ".join(ex.get("types", []))
            lines.append(f"- Visualization pattern(s): {types}")
        else:
            lines.append(f"- Section: {ex['section']}")

        usage = ex.get("usage", "")
        if usage:
            lines.append(f"- How it seems to be used: {usage}")
        elif ex["vague"]:
            lines.append(
                "- How it seems to be used: Not enough detail in retrieved text; "
                "check the original notebook."
            )
        else:
            lines.append(f"- Excerpt: {ex.get('snippet', '(no excerpt)')}")
        lines.append("")
    return lines


def synthesize_answer(query: str, results: list[dict]) -> str:
    """
    Turn retrieved chunks into a concise, rule-based answer.
    Only uses information found in the retrieved text.
    """
    topic = detect_query_topic(query)
    examples = [build_example_entry(r, topic) for r in dedupe_by_notebook(results)]

    lines = [
        "",
        "## Answer",
        "",
        f'**Query:** "{query}"',
        "",
        "### Summary",
        "",
        summarize_patterns(query, topic, examples, results),
        "",
        "### Specific Examples",
        "",
    ]

    if examples:
        lines.extend(format_specific_examples(examples, topic))
    else:
        lines.append("No specific examples could be extracted from the retrieved chunks.")
        lines.append("")

    lines.extend([
        "### Suggestions for Reuse",
        "",
    ])
    for tip in REUSE_TIPS.get(topic, REUSE_TIPS["general"]):
        lines.append(f"- {tip}")

    lines.append("")
    return "\n".join(lines)


def format_response(query: str, results: list[dict], raw_only: bool = False) -> str:
    """Format full output: synthesized answer + sources, or raw results only."""
    if raw_only:
        return format_results(query, results)

    parts = [
        synthesize_answer(query, results),
        "",
        "---",
        "",
        "## Sources",
        format_results(query, results),
    ]
    return "\n".join(parts)


def format_results(query: str, results: list[dict]) -> str:
    """Format raw search results with source citations."""
    lines = [
        "",
        f"Found {len(results)} result(s)",
        "",
    ]

    if not results:
        lines.append("No matching notebook content found. Try rephrasing your question.")
        return "\n".join(lines)

    for i, r in enumerate(results, 1):
        lines.extend([
            "-" * 60,
            f"Result {i}  (similarity: {r['score']:.3f})",
            "",
            "Source:",
            r["notebook"],
            "",
            "Section:",
            r["section"],
            "",
            "Discipline:",
            r.get("discipline", "Unknown"),
            "",
            "Text:",
            truncate_text(r["text"]),
            "",
        ])

    return "\n".join(lines)


# --- Draft section generation (--draft mode) ---


def detect_draft_section_type(prompt: str) -> str:
    """Identify which notebook section the user wants to draft."""
    p = prompt.lower()
    for section_type, keywords in DRAFT_SECTION_KEYWORDS:
        if any(kw in p for kw in keywords):
            return section_type
    return "introduction"


def detect_draft_discipline(prompt: str) -> str:
    """Guess discipline from the draft prompt."""
    p = prompt.lower()
    for discipline, keywords in DRAFT_DISCIPLINE_KEYWORDS.items():
        if any(kw in p for kw in keywords):
            return discipline
    return "General"


def extract_draft_topic(prompt: str) -> str:
    """Pull the lesson topic from phrases like 'on X' or 'about X'."""
    patterns = [
        r"notebook on\s+(.+)$",
        r"notebook about\s+(.+)$",
        r"for (?:an?|the)\s+(.+?)\s+notebook",
        r"(?:section|questions|objectives)\s+(?:for|on|about)\s+(.+)$",
        r"(?:on|about)\s+(.+)$",
    ]
    for pat in patterns:
        match = re.search(pat, prompt, re.IGNORECASE)
        if match:
            topic = match.group(1).strip().rstrip(".")
            topic = re.sub(r"\s+notebook$", "", topic, flags=re.IGNORECASE)
            topic = re.sub(r"^(?:an?|the)\s+", "", topic, flags=re.IGNORECASE)
            if len(topic) > 2:
                return topic
    return ""


def extract_draft_coding_level(prompt: str) -> str:
    """Detect beginner / intermediate / advanced from the prompt."""
    p = prompt.lower()
    for level in ("beginner", "intermediate", "advanced"):
        if level in p:
            return level
    return "intermediate"


def parse_draft_request(prompt: str) -> dict:
    """Parse a --draft prompt into structured fields."""
    discipline = detect_draft_discipline(prompt)
    topic = extract_draft_topic(prompt) or discipline
    return {
        "prompt": prompt,
        "section_type": detect_draft_section_type(prompt),
        "discipline": discipline,
        "topic": topic,
        "coding_level": extract_draft_coding_level(prompt),
    }


def build_draft_search_query(request: dict) -> str:
    """Build a retrieval query tailored to the draft request."""
    corpus_section = DRAFT_SECTION_TO_CORPUS.get(request["section_type"], "")
    return (
        f"{request['section_type']} {corpus_section} "
        f"{request['discipline']} {request['topic']} {request['coding_level']}"
    )


def prioritize_draft_results(results: list[dict], section_type: str) -> list[dict]:
    """Put chunks from the target corpus section first."""
    target = DRAFT_SECTION_TO_CORPUS.get(section_type, "").lower()
    if not target:
        return results
    preferred = [r for r in results if target in r.get("section", "").lower()]
    other = [r for r in results if r not in preferred]
    return preferred + other


def title_case_phrase(phrase: str) -> str:
    """Title-case a short phrase for headings."""
    return " ".join(w.capitalize() for w in phrase.split())


def observe_style_patterns(results: list[dict], section_type: str) -> list[str]:
    """Note structural patterns seen in retrieved examples (no copying)."""
    notes = []
    combined = " ".join(r["text"].lower() for r in results)

    if section_type == "learning objectives":
        if "by the end of this notebook" in combined:
            notes.append("Uses an opening line like 'By the end of this notebook, you will...'")
        if re.search(r"\n\d+\.", combined):
            notes.append("Uses numbered objectives")
        if "- " in combined:
            notes.append("Uses bullet-style objectives")

    if section_type == "widget section":
        widgets = []
        for label, name in [
            ("interact", "interact()"),
            ("slider", "sliders"),
            ("dropdown", "dropdowns"),
            ("ipywidgets", "ipywidgets"),
        ]:
            if label in combined:
                widgets.append(name)
        if widgets:
            notes.append(f"References interactive tools: {', '.join(widgets)}")

    if section_type == "dataset overview":
        if "column" in combined:
            notes.append("Includes column or variable descriptions")
        if "row" in combined or "granularity" in combined:
            notes.append("Explains what each row represents")

    if section_type == "coding exercise":
        if "your turn" in combined:
            notes.append("Uses 'Your Turn' framing for student tasks")
        if "your answer" in combined:
            notes.append("Includes answer placeholder cells")
        if "hint" in combined:
            notes.append("Pairs exercises with hints")

    if section_type == "reflection questions":
        if "reflect" in combined:
            notes.append("Uses direct reflection prompts")
        if "discuss" in combined:
            notes.append("Includes discussion-style questions")

    if section_type == "introduction":
        if "estimated time" in combined:
            notes.append("Mentions estimated completion time")
        if "table of contents" in combined:
            notes.append("Includes a table of contents")

    return notes


def draft_learning_objectives(request: dict, style_notes: list[str]) -> str:
    """Draft a learning objectives section."""
    topic = request["topic"]
    discipline = request["discipline"]
    level = request["coding_level"]

    objectives = [
        f"Explain the main ideas behind **{topic}** and why they matter in {discipline}.",
        f"Describe key vocabulary and concepts students need for **{topic}**.",
    ]

    if request["section_type"] != "widget section" and "widget" not in request["prompt"].lower():
        if discipline in ("Biology", "Public Health", "Environmental Science", "Social Science", "Economics"):
            objectives.append("Load and explore a dataset relevant to the lesson topic.")
            objectives.append("Create visualizations to identify patterns in the data.")
        elif discipline == "Mathematics":
            objectives.append("Use Python to plot functions and explore mathematical relationships.")
            objectives.append("Connect visual patterns to definitions and formulas.")
        else:
            objectives.append("Use Python to investigate examples related to the lesson topic.")

    if "widget" in request["prompt"].lower() or any("interact" in n for n in style_notes):
        objectives.append("Use interactive widgets to explore how changes affect results.")

    if level == "beginner":
        objectives.append("Follow guided notebook steps and explain findings in your own words.")
    else:
        objectives.append("Answer guided questions that connect practice to conceptual understanding.")

    objectives.append("Reflect on what you learned and what questions remain.")

    lines = [
        "## Learning Objectives",
        "",
        "By the end of this notebook, you should be able to:",
        "",
    ]
    for i, obj in enumerate(objectives[:7], 1):
        lines.append(f"{i}. {obj}")

    return "\n".join(lines)


def draft_introduction(request: dict) -> str:
    """Draft an introduction section."""
    topic = title_case_phrase(request["topic"])
    discipline = request["discipline"]
    level = request["coding_level"]
    time = {"beginner": "30–45", "intermediate": "45–60", "advanced": "60–90"}[level]

    return "\n".join([
        f"## 1. Introduction",
        "",
        f"Welcome to this notebook on **{topic}**.",
        "",
        f"In this lesson, you will explore **{request['topic']}** through hands-on activities "
        f"designed for {level}-level learners in **{discipline}**.",
        "",
        f"**Estimated time:** {time} minutes",
        "",
        "### What you will do",
        "",
        "- Read short background sections",
        "- Run provided code cells",
        "- Complete guided questions and reflections",
        "",
        "### Table of Contents",
        "",
        "1. Introduction",
        "2. Learning Objectives",
        "3. Main Activity",
        "4. Reflection",
        "5. Conclusion",
        "",
    ])


def draft_dataset_overview(request: dict, style_notes: list[str]) -> str:
    """Draft a dataset overview section."""
    topic = request["topic"]
    lines = [
        "## Dataset Overview",
        "",
        f"This notebook uses data related to **{topic}**.",
        "",
        "### About the data",
        "",
        "- **Source:** [Add dataset source / citation]",
        "- **Rows:** Each row represents [describe unit of observation]",
        "- **Purpose:** This dataset helps us investigate [connection to topic]",
        "",
        "### Key columns",
        "",
        "| Column | Description |",
        "|--------|-------------|",
        "| `[column_1]` | [What this column measures] |",
        "| `[column_2]` | [What this column measures] |",
        "| `[column_3]` | [What this column measures] |",
        "",
        "### Preview the data",
        "",
        "Run the cell below to load and preview the dataset.",
        "",
        "```python",
        "import pandas as pd",
        "",
        "# Load your dataset",
        "# df = pd.read_csv('your_data.csv')",
        "# df.head()",
        "```",
        "",
        "**Question:** In 2–3 sentences, describe what this dataset represents and one question you hope to answer.",
        "",
        "*Your answer here*",
        "",
    ]
    return "\n".join(lines)


def draft_widget_section(request: dict, style_notes: list[str]) -> str:
    """Draft an interactive widget section."""
    topic = title_case_phrase(request["topic"])
    level = request["coding_level"]
    is_calculus = any(kw in request["topic"].lower() for kw in ("riemann", "integral", "calculus", "sum"))

    lines = [
        f"## Interactive Exploration: {topic}",
        "",
        "Use the widgets below to explore the lesson concept without rewriting code each time.",
        "",
        "### Setup",
        "",
        "Run this cell once to load libraries:",
        "",
        "```python",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import ipywidgets as widgets",
        "from ipywidgets import interact, IntSlider, FloatSlider, Dropdown",
        "%matplotlib inline",
        "```",
        "",
        "### Instructions",
        "",
    ]

    if is_calculus:
        lines.extend([
            "Adjust the controls and observe how the plot changes:",
            "",
            "- **Function:** Choose which function to visualize",
            "- **Number of rectangles:** Change the partition count for the Riemann sum",
            "- **Interval bounds:** Set the left and right endpoints",
            "- **Tangent point:** Move the x-value where the tangent line is drawn",
            "",
        ])
    else:
        lines.extend([
            "Adjust the controls and observe how the output changes:",
            "",
            "- **Parameter 1:** [Describe what this slider changes]",
            "- **Parameter 2:** [Describe what this dropdown selects]",
            "",
        ])

    lines.extend([
        "### Interactive cell",
        "",
        "```python",
        "# Starter scaffold — customize for your lesson",
    ])

    if is_calculus and level == "beginner":
        lines.extend([
            "@interact",
            "def explore_riemann(n_rectangles=10, a=0.0, b=1.0):",
            "    # TODO: plot function and Riemann sum rectangles",
            "    pass",
        ])
    else:
        lines.extend([
            "@interact",
            "def explore(parameter=1):",
            "    # TODO: build widget visualization",
            "    pass",
        ])

    lines.extend([
        "```",
        "",
        "### What to notice",
        "",
        "- What pattern do you see when you change the first parameter?",
        "- Does the result match what you expected from the definition?",
        "- What would you try next to test your understanding?",
        "",
    ])
    return "\n".join(lines)


def draft_coding_exercise(request: dict, style_notes: list[str]) -> str:
    """Draft a coding exercise section."""
    topic = request["topic"]
    level = request["coding_level"]
    has_hints = any("hint" in n.lower() for n in style_notes)

    lines = [
        f"## Coding Exercise: {title_case_phrase(topic)}",
        "",
        "### Your Turn",
        "",
        f"Complete the activity below to practice **{topic}**.",
        "",
        "**Question:** [Write a clear, specific task for students]",
        "",
    ]

    if level == "beginner":
        lines.extend([
            "*Fill in the code below. Run the cell when you are finished.*",
            "",
            "```python",
            "# Starter code",
            "import numpy as np",
            "",
            "# Step 1: [first small task]",
            "# Step 2: [second small task]",
            "```",
            "",
        ])
    else:
        lines.extend([
            "*Use the starter code below, then extend it to answer the question.*",
            "",
            "```python",
            "# Starter code",
            "import pandas as pd",
            "import numpy as np",
            "",
            "def your_function(data):",
            "    # TODO: implement",
            "    pass",
            "```",
            "",
        ])

    if has_hints:
        lines.append("*Hint: [Point students toward a function or approach without giving the full answer]*")
        lines.append("")

    lines.extend([
        "**Reflection:** In 1–2 sentences, explain what your code shows.",
        "",
        "*Your answer here*",
        "",
    ])
    return "\n".join(lines)


def draft_reflection_questions(request: dict, style_notes: list[str]) -> str:
    """Draft reflection / discussion questions."""
    topic = request["topic"]
    discipline = request["discipline"]

    questions = [
        f"What did you learn about **{topic}** from this notebook?",
        "What result or observation surprised you? Why?",
        "What limitations should we keep in mind when interpreting these findings?",
    ]

    if discipline == "Ethics / Data Science":
        questions.extend([
            "What patterns did you notice across different scenarios?",
            "What kinds of reasoning appeared in the responses (rules, outcomes, uncertainty)?",
            "What would you want to know before trusting an AI system in a real decision?",
        ])
    elif discipline == "Mathematics":
        questions.extend([
            "How did the visualization help you understand the concept?",
            "What changes when you adjust the parameters in the interactive section?",
        ])
    else:
        questions.append("How could these findings matter to a real-world decision or policy?")

    lines = [
        "## Reflection Questions",
        "",
        "Take a few minutes to reflect. Discuss with a partner or write brief notes below.",
        "",
    ]
    for i, q in enumerate(questions[:5], 1):
        lines.append(f"{i}. {q}")

    lines.extend([
        "",
        "*Your reflections here*",
        "",
    ])
    return "\n".join(lines)


def generate_section_draft(request: dict, results: list[dict]) -> str:
    """Pick the right template and generate the draft section body."""
    section_type = request["section_type"]
    style_notes = observe_style_patterns(results, section_type)

    if section_type == "learning objectives":
        return draft_learning_objectives(request, style_notes)
    if section_type == "introduction":
        return draft_introduction(request)
    if section_type == "dataset overview":
        return draft_dataset_overview(request, style_notes)
    if section_type == "widget section":
        return draft_widget_section(request, style_notes)
    if section_type == "coding exercise":
        return draft_coding_exercise(request, style_notes)
    if section_type == "reflection questions":
        return draft_reflection_questions(request, style_notes)
    return draft_introduction(request)


def build_influence_notes(request: dict, results: list[dict]) -> list[str]:
    """Describe which corpus notebooks influenced the draft."""
    notes = []
    seen = set()
    style_patterns = observe_style_patterns(results, request["section_type"])

    for r in results:
        nb = r["notebook"]
        if nb in seen:
            continue
        seen.add(nb)
        notes.append(
            f"- **{nb}** ({r.get('discipline', 'Unknown')}) — "
            f"referenced the *{r.get('section', 'unknown')}* section"
        )
        if len(seen) >= 4:
            break

    for pattern in style_patterns:
        notes.append(f"- Style pattern adopted: {pattern}")

    if not notes:
        notes.append("- No close corpus matches found; draft uses default DSEP / El Camino template.")

    return notes


def format_draft_output(request: dict, draft_body: str, results: list[dict]) -> str:
    """Assemble the full draft output with notes and sources."""
    influence = build_influence_notes(request, results)

    lines = [
        "# Draft Section",
        "",
        f"**Request:** {request['prompt']}",
        f"**Section type:** {request['section_type']}",
        f"**Discipline:** {request['discipline']}",
        f"**Topic:** {request['topic']}",
        f"**Coding level:** {request['coding_level']}",
        "",
        "---",
        "",
        draft_body,
        "",
        "---",
        "",
        "## Notes on Influencing Examples",
        "",
        "This draft was shaped by the following corpus patterns (not copied verbatim):",
        "",
    ]
    lines.extend(influence)

    lines.extend([
        "",
        "---",
        "",
        "## Source Chunks Used",
        format_results(request["prompt"], results),
    ])
    return "\n".join(lines)


def run_draft_mode(prompt: str) -> str:
    """Retrieve examples, generate a draft, save to disk, and return output."""
    request = parse_draft_request(prompt)
    search_query = build_draft_search_query(request)
    results = search(search_query)
    results = prioritize_draft_results(results, request["section_type"])

    if not results:
        return "No relevant chunks found. Try rephrasing your draft request."

    draft_body = generate_section_draft(request, results)
    output = format_draft_output(request, draft_body, results)

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    DRAFT_PATH.write_text(output, encoding="utf-8")

    return output


def parse_cli_args() -> tuple[bool, bool, list[str]]:
    """Parse CLI flags and return (raw_only, draft_mode, remaining_args)."""
    raw_only = "--raw" in sys.argv
    draft_mode = "--draft" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("--raw", "--draft")]
    return raw_only, draft_mode, args


def interactive_loop(raw_only: bool = False) -> None:
    """Run an interactive question loop."""
    mode = "raw retrieval" if raw_only else "synthesized answers"
    print(f"\n=== Notebook Copilot — {mode} ===\n")
    print("Ask questions about the notebook corpus.")
    print("Examples:")
    print('  - "Show me examples of widget usage"')
    print('  - "How do biology notebooks introduce datasets?"')
    print('  - "Give me examples of reflection questions"')
    print('  - "Show me how beginner coding exercises are structured"')
    if not raw_only:
        print("\nAdd --raw to see retrieval results only (no synthesis).")
    print("\nType 'quit' or 'exit' to stop.\n")

    while True:
        try:
            query = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        results = search(query)
        print(format_response(query, results, raw_only=raw_only))


def main() -> None:
    raw_only, draft_mode, args = parse_cli_args()

    if args:
        command = args[0].lower()

        if command == "build":
            build_index()
            return

        if command in ("help", "-h", "--help"):
            print(__doc__)
            return

        query = " ".join(args)

        if draft_mode:
            chunks, _, _ = load_index()
            if not chunks:
                build_index()
            output = run_draft_mode(query)
            print(output)
            print(f"\nDraft saved to {DRAFT_PATH}")
            return

        results = search(query)
        print(format_response(query, results, raw_only=raw_only))
        return

    if draft_mode:
        print("Usage: python3 scripts/notebook_copilot.py --draft \"Create a widget section for...\"")
        sys.exit(1)

    # Default: ensure index exists, then interactive mode
    chunks, _, _ = load_index()
    if not chunks:
        build_index()

    interactive_loop(raw_only=raw_only)


if __name__ == "__main__":
    main()

"""
Shared helpers for RAG: chunk parsed notebooks, build embeddings, and search.

Used by build_vector_index.py and llm_generator.py.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
METADATA_PATH = PARSED_DIR / "notebook_metadata.json"

CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.json"
EMBEDDINGS_PATH = VECTOR_STORE_DIR / "embeddings.npy"
VOCAB_PATH = VECTOR_STORE_DIR / "vocabulary.json"
INDEX_META_PATH = VECTOR_STORE_DIR / "index_meta.json"

MIN_CHUNK_CHARS = 30
COMBINE_ITEM_SECTIONS = {
    "WIDGET PATTERNS",
    "VISUALIZATION PATTERNS",
    "IMPORTS / LIBRARIES",
}


def tokenize(text: str) -> list[str]:
    """Simple word tokenizer for TF-IDF fallback."""
    return re.findall(r"[a-z0-9]+", text.lower())


def load_metadata_by_filename() -> dict[str, dict]:
    """Map notebook filename -> metadata entry."""
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH, encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["filename"]: entry for entry in entries}


def parse_parsed_file(path: Path, metadata_map: dict) -> list[dict]:
    """Split one *_parsed.txt file into searchable chunks."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    notebook = path.stem.replace("_parsed", "") + ".ipynb"
    if lines and lines[0].startswith("Notebook:"):
        notebook = lines[0].replace("Notebook:", "").strip()

    meta = metadata_map.get(notebook, {})
    discipline = meta.get("discipline", "Unknown")
    title = meta.get("title", notebook)
    coding_level = meta.get("coding_level", "unknown")

    chunks: list[dict] = []

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

    parts = re.split(r"\n=== (.+?) ===\n", text)
    for i in range(1, len(parts), 2):
        section_name = parts[i].strip()
        content = parts[i + 1] if i + 1 < len(parts) else ""

        raw_items = re.split(r"--- Item \d+ ---\n", content)
        item_texts = [
            item.strip()
            for item in raw_items
            if item.strip() and item.strip() != "(none found)"
        ]

        if section_name in COMBINE_ITEM_SECTIONS and item_texts:
            combined = "\n".join(f"- {item}" for item in item_texts)
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

    all_chunks: list[dict] = []
    for path in parsed_files:
        file_chunks = parse_parsed_file(path, metadata_map)
        all_chunks.extend(file_chunks)
        print(f"  Chunked {path.name}: {len(file_chunks)} chunks")

    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks


def build_tfidf_embeddings(chunks: list[dict]) -> tuple[np.ndarray, dict]:
    """Build TF-IDF vectors for all chunks."""
    docs_tokens = [tokenize(chunk["search_text"]) for chunk in chunks]
    n_docs = len(docs_tokens)

    df: dict[str, int] = {}
    for tokens in docs_tokens:
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1

    vocabulary = sorted(df.keys())
    idf = {term: math.log((1 + n_docs) / (1 + df[term])) + 1 for term in vocabulary}
    term_to_idx = {term: idx for idx, term in enumerate(vocabulary)}

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
    term_to_idx = {term: idx for idx, term in enumerate(vocabulary)}

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


def save_index(chunks: list[dict], embeddings: np.ndarray, vocab_info: dict) -> None:
    """Save chunks, embeddings, and metadata to data/vector_store/."""
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    np.save(EMBEDDINGS_PATH, embeddings)

    if vocab_info.get("method") == "tfidf":
        with open(VOCAB_PATH, "w", encoding="utf-8") as f:
            json.dump(vocab_info, f)
    elif VOCAB_PATH.exists():
        VOCAB_PATH.unlink()

    index_meta = {
        "method": vocab_info.get("method", "tfidf"),
        "model": vocab_info.get("model"),
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
    }
    with open(INDEX_META_PATH, "w", encoding="utf-8") as f:
        json.dump(index_meta, f, indent=2)


def load_index() -> tuple[list[dict], np.ndarray, dict]:
    """Load chunks, embeddings, and vocab/metadata from disk."""
    if not CHUNKS_PATH.exists() or not EMBEDDINGS_PATH.exists():
        return [], np.array([]), {}

    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)

    embeddings = np.load(EMBEDDINGS_PATH)

    vocab_info: dict = {}
    if VOCAB_PATH.exists():
        with open(VOCAB_PATH, encoding="utf-8") as f:
            vocab_info = json.load(f)
    elif INDEX_META_PATH.exists():
        with open(INDEX_META_PATH, encoding="utf-8") as f:
            vocab_info = json.load(f)

    return chunks, embeddings, vocab_info


def embed_query_openai(query: str, model: str) -> np.ndarray:
    """Embed a single query with OpenAI."""
    from providers.openai_provider import embed_texts

    return embed_texts([query], model=model)[0]


def search_chunks(
    query: str,
    top_k: int = 5,
    discipline: str | None = None,
) -> list[dict]:
    """
    Find the most similar chunks to a query.

    Returns chunk dicts with an added "score" field.
    """
    chunks, embeddings, vocab_info = load_index()
    if not chunks:
        return []

    method = vocab_info.get("method", "tfidf")

    if method == "openai":
        model = vocab_info.get("model", "text-embedding-3-small")
        query_vec = embed_query_openai(query, model)
    else:
        if not vocab_info.get("vocabulary"):
            _, vocab_info = build_tfidf_embeddings(chunks)
        query_vec = embed_query_tfidf(query, vocab_info)

    scores = embeddings @ query_vec

    if discipline:
        discipline_lower = discipline.lower()
        for i, chunk in enumerate(chunks):
            if discipline_lower in chunk.get("discipline", "").lower():
                scores[i] += 0.15

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


def truncate_text(text: str, max_chars: int = 1200) -> str:
    """Trim long chunk text for prompt context."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."

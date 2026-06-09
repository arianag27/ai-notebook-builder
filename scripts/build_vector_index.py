#!/usr/bin/env python3
"""
Build a searchable embedding index from parsed notebooks for RAG.

Reads all data/parsed/*_parsed.txt files, creates chunk embeddings, and saves
them to data/vector_store/.

Usage:
    python3 scripts/build_vector_index.py
    python3 scripts/build_vector_index.py --openai   # use OpenAI embeddings (optional)

Default: TF-IDF embeddings (free, offline, no API key).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

import rag_utils


def build_index(use_openai: bool = False) -> None:
    """Chunk parsed notebooks, embed, and save to data/vector_store/."""
    print("Building vector index for RAG...\n")
    chunks = rag_utils.load_all_chunks()

    if not chunks:
        print("No chunks to index.")
        sys.exit(1)

    if use_openai:
        from providers.openai_provider import DEFAULT_EMBEDDING_MODEL, embed_texts, get_api_key

        if not get_api_key():
            print("WARNING: OPENAI_API_KEY not set. Using TF-IDF instead.\n")
            use_openai = False

    if use_openai:
        from providers.openai_provider import DEFAULT_EMBEDDING_MODEL, embed_texts

        print(f"Using OpenAI embeddings ({DEFAULT_EMBEDDING_MODEL})...")
        try:
            texts = [chunk["search_text"] for chunk in chunks]
            embeddings = embed_texts(texts, model=DEFAULT_EMBEDDING_MODEL)
            vocab_info = {
                "method": "openai",
                "model": DEFAULT_EMBEDDING_MODEL,
            }
        except (ValueError, RuntimeError, ImportError) as exc:
            print(f"OpenAI embedding failed: {exc}")
            print("Falling back to TF-IDF embeddings.\n")
            use_openai = False

    if not use_openai:
        print("Using TF-IDF embeddings (numpy only, no API key).")
        embeddings, vocab_info = rag_utils.build_tfidf_embeddings(chunks)

    rag_utils.save_index(chunks, embeddings, vocab_info)

    print(f"\nIndex saved to {rag_utils.VECTOR_STORE_DIR}")
    print(
        f"  {len(chunks)} chunks, {embeddings.shape[1]}-dim embeddings "
        f"({vocab_info.get('method', 'tfidf')})"
    )


def main() -> None:
    use_openai = "--openai" in sys.argv
    build_index(use_openai=use_openai)


if __name__ == "__main__":
    main()

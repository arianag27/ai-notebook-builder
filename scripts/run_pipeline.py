#!/usr/bin/env python3
"""
Run the full AI Notebook Builder workflow from one command.

Steps:
  0. sync_notebooks_from_repo.py (optional) — sync ecc-* notebooks from GitHub
  1. parse_notebooks.py      — parse corpus notebooks
  2. analyze_corpus.py       — build curriculum patterns report
  3. generate_notebook_outline.py — interactive outline (you answer prompts)
  4. export_to_ipynb.py      — create starter Jupyter notebook
  5. validate_generated_notebook.py — check notebook quality

Usage:
    python3 scripts/run_pipeline.py
    python3 scripts/run_pipeline.py --skip-sync
    python3 scripts/run_pipeline.py --repo-url https://github.com/ds-modules/ecc-textbook.git
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
GENERATED_DIR = PROJECT_ROOT / "data" / "generated"

DEFAULT_REPO_URL = "https://github.com/ds-modules/ecc-textbook.git"

# Each step: (description, script name, extra CLI args)
PIPELINE_STEPS = [
    ("Parse notebooks", "parse_notebooks.py", []),
    ("Analyze corpus", "analyze_corpus.py", []),
    ("Generate notebook outline", "generate_notebook_outline.py", []),
    ("Export to Jupyter notebook", "export_to_ipynb.py", ["--outline"]),
    ("Validate generated notebook", "validate_generated_notebook.py", []),
]

FINAL_OUTPUTS = [
    GENERATED_DIR / "notebook_outline.md",
    GENERATED_DIR / "starter_notebook.ipynb",
    GENERATED_DIR / "validation_report.md",
]


def parse_pipeline_args() -> tuple[bool, str]:
    """
    Parse --skip-sync and --repo-url from command line.
    Returns (skip_sync, repo_url).
    """
    skip_sync = "--skip-sync" in sys.argv
    repo_url = DEFAULT_REPO_URL

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--repo-url" and i + 1 < len(args):
            repo_url = args[i + 1]
            i += 2
        else:
            i += 1

    return skip_sync, repo_url


def run_step(step_number: int, total: int, description: str, script_name: str, args: list[str]) -> int:
    """Run one pipeline script and return its exit code."""
    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return 1

    print()
    print("=" * 60)
    print(f"Step {step_number} of {total}: {description}")
    print(f"Running: python3 scripts/{script_name}", " ".join(args))
    print("=" * 60)
    print()

    command = [sys.executable, str(script_path), *args]
    result = subprocess.run(command, cwd=PROJECT_ROOT)

    if result.returncode == 0:
        print()
        print(f"Step {step_number} completed successfully.")
    else:
        print()
        print(f"Step {step_number} FAILED (exit code {result.returncode}).")

    return result.returncode


def show_final_outputs() -> None:
    """Print paths to the main generated files."""
    print()
    print("=" * 60)
    print("Generated files")
    print("=" * 60)

    for path in FINAL_OUTPUTS:
        if path.exists():
            print(f"  [ok] {path}")
        else:
            print(f"  [--] {path} (not found)")


def main() -> None:
    skip_sync, repo_url = parse_pipeline_args()

    sync_step_count = 0 if skip_sync else 1
    total = sync_step_count + len(PIPELINE_STEPS)
    step_number = 1

    print()
    print("AI Notebook Builder — Full Pipeline")
    print()

    if skip_sync:
        print("Sync skipped — using existing notebooks in data/notebooks/.")
    else:
        print(f"Step 0: Sync notebooks from {repo_url}")

    print("Later steps will parse, analyze, ask outline questions, export, and validate.")
    print()

    if not skip_sync:
        exit_code = run_step(
            step_number,
            total,
            "Sync notebooks from GitHub",
            "sync_notebooks_from_repo.py",
            ["--repo-url", repo_url],
        )
        if exit_code != 0:
            print()
            print("Pipeline stopped because sync failed.")
            print("Fix the issue above, or rerun with --skip-sync to use local notebooks.")
            sys.exit(exit_code)
        step_number += 1

    for description, script_name, args in PIPELINE_STEPS:
        exit_code = run_step(step_number, total, description, script_name, args)
        if exit_code != 0:
            print()
            print("Pipeline stopped because a step failed.")
            print("Fix the issue above, then run the pipeline again.")
            show_final_outputs()
            sys.exit(exit_code)
        step_number += 1

    print()
    print("=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

    show_final_outputs()
    print()
    print("Next: open starter_notebook.ipynb and validation_report.md to review.")


if __name__ == "__main__":
    main()

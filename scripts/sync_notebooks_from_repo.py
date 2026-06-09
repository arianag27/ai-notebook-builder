#!/usr/bin/env python3
"""
Sync Jupyter notebooks from the ECC textbook GitHub repo into data/notebooks/.

Only copies notebooks from top-level folders whose names start with "ecc-".
Tracks synced files in sync_manifest.json so re-syncing does not delete
manually added notebooks.

Usage:
    python3 scripts/sync_notebooks_from_repo.py
    python3 scripts/sync_notebooks_from_repo.py --repo-url https://github.com/ds-modules/ecc-textbook.git
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO_URL = "https://github.com/ds-modules/ecc-textbook.git"
SOURCE_REPOS_DIR = PROJECT_ROOT / "data" / "source_repos"
NOTEBOOKS_DIR = PROJECT_ROOT / "data" / "notebooks"
MANIFEST_PATH = NOTEBOOKS_DIR / "sync_manifest.json"

ECC_PREFIX = "ecc-"


def repo_folder_name(repo_url: str) -> str:
    """Turn a git URL into a local folder name (e.g. ecc-textbook)."""
    name = repo_url.rstrip("/").split("/")[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name


def check_git_installed() -> None:
    """Raise a clear error if git is not available."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "git is not installed. Install git and try again."
        ) from exc

    if result.returncode != 0:
        raise RuntimeError("git is not working correctly. Check your installation.")


def run_git(command: list[str], cwd: Path | None = None, action: str = "git command") -> None:
    """Run a git command and raise a readable error on failure."""
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"{action} failed.\n{detail}")


def clone_or_pull(repo_url: str, local_repo_path: Path) -> str:
    """Clone the repo if missing, otherwise pull latest changes."""
    if local_repo_path.exists() and (local_repo_path / ".git").exists():
        run_git(["git", "pull"], cwd=local_repo_path, action="git pull")
        return "pulled latest changes"

    if local_repo_path.exists():
        shutil.rmtree(local_repo_path)

    local_repo_path.parent.mkdir(parents=True, exist_ok=True)
    run_git(
        ["git", "clone", repo_url, str(local_repo_path)],
        action="git clone",
    )
    return "cloned repository"


def find_ecc_folders(repo_path: Path) -> list[Path]:
    """Return top-level folders whose names start with ecc-."""
    if not repo_path.exists():
        return []

    folders = []
    for item in sorted(repo_path.iterdir()):
        if item.is_dir() and item.name.startswith(ECC_PREFIX):
            folders.append(item)
    return folders


def is_checkpoint_notebook(path: Path) -> bool:
    """True for checkpoint paths or checkpoint notebook files."""
    if ".ipynb_checkpoints" in path.parts:
        return True
    if path.name.endswith("-checkpoint.ipynb"):
        return True
    return False


def find_notebooks(ecc_folders: list[Path]) -> tuple[list[Path], int]:
    """Recursively find .ipynb files inside ecc-* folders. Returns (paths, skipped)."""
    notebooks = []
    skipped = 0
    for folder in ecc_folders:
        for path in folder.rglob("*.ipynb"):
            if is_checkpoint_notebook(path):
                skipped += 1
                continue
            notebooks.append(path)
    return sorted(notebooks), skipped


def dest_filename(repo_path: Path, notebook_path: Path) -> str:
    """
    Build a unique destination filename that preserves folder context.

    Example:
        ecc-biology/SARS-CoV2/wastewater_covid_notebook.ipynb
        -> ecc-biology__SARS-CoV2__wastewater_covid_notebook.ipynb
    """
    rel = notebook_path.relative_to(repo_path)
    return "__".join(rel.parts)


def load_manifest() -> dict:
    """Load the previous sync manifest, or return an empty one."""
    if not MANIFEST_PATH.exists():
        return {"files": []}

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


def remove_previous_synced_files(manifest: dict) -> int:
    """Delete notebooks listed in the previous manifest only."""
    removed = 0
    for entry in manifest.get("files", []):
        filename = entry.get("dest_filename")
        if not filename:
            continue
        dest_path = NOTEBOOKS_DIR / filename
        if dest_path.exists():
            dest_path.unlink()
            removed += 1
    return removed


def copy_notebooks(repo_path: Path, notebooks: list[Path]) -> list[dict]:
    """Copy notebooks into data/notebooks/. Returns manifest entries."""
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)

    entries = []
    for notebook_path in notebooks:
        dest_name = dest_filename(repo_path, notebook_path)
        dest_path = NOTEBOOKS_DIR / dest_name
        shutil.copy2(notebook_path, dest_path)

        entries.append({
            "dest_filename": dest_name,
            "source_path": notebook_path.relative_to(repo_path).as_posix(),
        })

    return entries


def write_manifest(repo_url: str, entries: list[dict]) -> None:
    """Save the new sync manifest."""
    manifest = {
        "repo_url": repo_url,
        "synced_at": datetime.now(timezone.utc).isoformat(),
        "files": entries,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def print_summary(
    repo_url: str,
    local_repo_path: Path,
    ecc_folders: list[Path],
    notebooks: list[Path],
    copied: int,
    skipped: int,
    removed: int,
    git_action: str,
) -> None:
    """Print a clear summary for the user."""
    print()
    print("=" * 60)
    print("Notebook Sync Summary")
    print("=" * 60)
    print(f"Repo URL:          {repo_url}")
    print(f"Local repo path:   {local_repo_path}")
    print(f"Git action:        {git_action}")
    print(f"ECC folders found: {len(ecc_folders)}")
    for folder in ecc_folders:
        print(f"  - {folder.name}")
    print(f"Notebooks found:   {len(notebooks)}")
    print(f"Notebooks copied:  {copied}")
    print(f"Notebooks skipped: {skipped} (checkpoints ignored)")
    print(f"Old synced removed:{removed}")
    print(f"Destination:       {NOTEBOOKS_DIR}")
    print(f"Manifest:          {MANIFEST_PATH}")
    print("=" * 60)


def sync_from_repo(repo_url: str = DEFAULT_REPO_URL) -> int:
    """
    Run the full sync workflow.
    Returns exit code 0 on success, 1 on error.
    """
    local_repo_path = SOURCE_REPOS_DIR / repo_folder_name(repo_url)

    try:
        check_git_installed()
        git_action = clone_or_pull(repo_url, local_repo_path)

        ecc_folders = find_ecc_folders(local_repo_path)
        if not ecc_folders:
            print(f"ERROR: No folders starting with '{ECC_PREFIX}' found in {local_repo_path}")
            return 1

        notebooks, skipped = find_notebooks(ecc_folders)
        if not notebooks:
            print(f"ERROR: No .ipynb files found inside {ECC_PREFIX}* folders.")
            return 1

        old_manifest = load_manifest()
        removed = remove_previous_synced_files(old_manifest)

        entries = copy_notebooks(local_repo_path, notebooks)
        write_manifest(repo_url, entries)

        print_summary(
            repo_url=repo_url,
            local_repo_path=local_repo_path,
            ecc_folders=ecc_folders,
            notebooks=notebooks,
            copied=len(entries),
            skipped=skipped,
            removed=removed,
            git_action=git_action,
        )
        return 0

    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:
        print(f"ERROR: Unexpected failure during sync: {exc}")
        return 1


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Sync ecc-* notebooks from the ECC textbook GitHub repo.",
    )
    parser.add_argument(
        "--repo-url",
        default=DEFAULT_REPO_URL,
        help=f"Git repository URL (default: {DEFAULT_REPO_URL})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return sync_from_repo(args.repo_url)


if __name__ == "__main__":
    sys.exit(main())

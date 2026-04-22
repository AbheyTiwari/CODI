# codi_agent/cli.py
#
# This is the entry point registered by `pip install -e .`
# When the user types `codi` in any terminal directory, this runs.
#
# What it does:
#   1. Resolves where the Codi source code lives (the git-cloned repo)
#   2. Adds the repo to sys.path so all Codi modules can be imported
#   3. Sets CODI_WORKING_DIR to wherever the user ran `codi` from
#   4. Auto-indexes that directory into ChromaDB (incremental — skips unchanged files)
#   5. Hands off to main.py's main() as normal

import os
import sys

# ── Find the repo root ────────────────────────────────────────────────────────
# __file__ is  <repo>/cli.py  (cli.py is at the repo root, not in a subdirectory)
# so _REPO_ROOT is just the directory containing this file.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    # 1. Put the repo on sys.path so all Codi modules import correctly
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    # 2. Record the directory where the user typed `codi`
    cwd = os.getcwd()
    os.environ["CODI_WORKING_DIR"] = cwd

    # 3. Point the ChromaDB at a per-project location so each repo has its own index
    #    We store it inside the Codi repo itself, namespaced by the project path hash
    import hashlib
    path_hash = hashlib.md5(cwd.encode()).hexdigest()[:10]
    chroma_dir = os.path.join(_REPO_ROOT, "chroma_db", path_hash)
    os.environ["CODI_CHROMA_DIR"] = chroma_dir

    # 4. Load .env from the repo root (where the user keeps their API keys)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))

    # 5. Auto-index the working directory (incremental — only changed files re-index)
    _auto_index(cwd, chroma_dir)

    # 6. Launch the CLI
    from main import main as _main
    _main()


def _auto_index(project_path: str, chroma_dir: str):
    """
    Incrementally index the project directory.
    Skips files that haven't changed since last run (hash-based cache).
    Shows a one-line status — doesn't block the UI for more than a second or two
    on small projects.
    """
    from rich.console import Console
    console = Console()

    # Skip indexing if the directory is huge (e.g. user is in C:\ or ~)
    # Heuristic: count files first, bail if >5000
    skip_dirs = {'.git', 'node_modules', '__pycache__', 'venv', 'dist', 'build',
                 '.idea', 'chroma_db', '.mypy_cache', '.pytest_cache'}
    file_count = 0
    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
        file_count += len(files)
        if file_count > 5000:
            console.print(
                f"[dim]⚠ Directory has >5000 files — skipping auto-index. "
                f"Run [bold]/index {project_path}[/bold] manually if needed.[/dim]"
            )
            return

    with console.status(f"[dim]Indexing {project_path} ...[/dim]", spinner="dots"):
        try:
            from indexer import index_codebase
            index_codebase(project_path, db_path=chroma_dir)
        except Exception as e:
            console.print(f"[dim yellow]Auto-index warning: {e}[/dim yellow]")

    console.print(f"[dim]✓ Indexed [bold]{project_path}[/bold][/dim]")
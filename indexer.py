import os
import json
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from quantized_embeddings import QuantizedEmbeddings

# ── ChromaDB location ─────────────────────────────────────────────────────────
# When launched via `codi` CLI, CODI_CHROMA_DIR is set per-project by cli.py.
# When run directly (python indexer.py), fall back to the repo-local chroma_db/.
_REPO_CHROMA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
CHROMA_PERSIST_DIR = os.environ.get("CODI_CHROMA_DIR", _REPO_CHROMA)

# ── Global embeddings singleton (cache) ────────────────────────────────────────
_embeddings_instance = None
_use_quantization = os.environ.get("CODI_USE_QUANTIZATION", "true").lower() == "true"

def get_embeddings():
    """
    Get or create embeddings instance (singleton pattern).
    Uses quantized embeddings by default for 4x compression and faster search.
    Disable with: CODI_USE_QUANTIZATION=false
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        try:
            _embeddings_instance = QuantizedEmbeddings(
                model_name="all-MiniLM-L6-v2",
                use_quantization=_use_quantization
            )
        except Exception:
            # Fall back to regular embeddings if quantized embeddings are unavailable
            _embeddings_instance = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings_instance

def get_vectorstore():
    chroma_dir = os.environ.get("CODI_CHROMA_DIR", CHROMA_PERSIST_DIR)
    if not os.path.exists(chroma_dir) or not os.listdir(chroma_dir):
        return None
    try:
        embeddings = get_embeddings()
        return Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    except Exception:
        return None

def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

_SKIP_DIRS = {
    '.git', 'node_modules', '__pycache__', 'venv', '.venv', 'dist', 'build',
    '.idea', 'chroma_db', '.mypy_cache', '.pytest_cache', '.tox',
}
_CODE_EXTS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.scss',
    '.json', '.yaml', '.yml', '.toml', '.md', '.txt', '.sh', '.bash',
    '.c', '.cpp', '.h', '.hpp', '.java', '.go', '.rs', '.rb', '.php',
    '.sql', '.graphql',
}
_NAMED_FILES = {'Dockerfile', 'Makefile', '.env.example'}

# Files larger than this are skipped entirely rather than chunked. A single
# huge file (minified vendor bundle, generated JSON fixture, lockfile) can
# otherwise produce tens of thousands of chunks that both blow past
# Chroma's max add_documents() batch size and add little semantic search
# value anyway. ~2MB of text is already thousands of chunks at 800 chars each.
_MAX_FILE_CHARS = int(os.environ.get("CODI_MAX_INDEX_FILE_CHARS", "2000000"))


def _is_eligible(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in _CODE_EXTS or filename in _NAMED_FILES


def count_eligible_files(root_path: str) -> int:
    """
    Cheap pre-scan (stat only, no file reads) so a progress callback can show
    "N / total" instead of an unbounded, seemingly-hung spinner. This mirrors
    the same skip_dirs/code_exts rules walk_codebase() uses so the count is
    accurate, not just a rough guess.
    """
    total = 0
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith('.')]
        total += sum(1 for f in files if _is_eligible(f))
    return total


def walk_codebase(root_path: str):
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS and not d.startswith('.')]
        for f in files:
            if not _is_eligible(f):
                continue
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    content = file.read()
                if content.strip():
                    yield path, content
            except Exception:
                continue

def index_codebase(root_path: str, db_path: str = None, progress_callback=None, quiet: bool = False):
    """
    Incrementally index a codebase into ChromaDB.

    progress_callback, if given, is called as progress_callback(done, total,
    current_path, changed) after every file is considered (whether it was
    actually re-embedded or skipped via the hash cache). This is what lets a
    caller (e.g. main.py's Rich status spinner) show real "N/total" progress
    instead of a spinner that looks identical whether it's on file 1 or file
    4000 of a large first-time index.
    """
    if db_path is None:
        db_path = os.environ.get("CODI_CHROMA_DIR", CHROMA_PERSIST_DIR)

    os.makedirs(db_path, exist_ok=True)

    # Pre-scan for a total count. This is a directory walk + splitext check
    # only (no file reads), so it's cheap even on large trees, and it's what
    # turns "indexing..." into "indexing... (128/4302) app/models/user.py".
    total_files = count_eligible_files(root_path)
    if progress_callback is None:
        def progress_callback(done, total, path, changed):  # noqa: ARG001
            if quiet:
                return
            # Default: print every ~5% of progress (or every file for small
            # projects) instead of flooding stdout on large repos.
            step = max(1, total // 20)
            if done == total or done % step == 0:
                label = os.path.basename(path) if path else ""
                print(f"  Indexing {done}/{total} — {label}")

    if not quiet:
        print(f"  Indexing: {root_path} ({total_files} eligible files)")

    # An empty directory has nothing to embed. Avoid loading/downloading the
    # Hugging Face model solely to announce zero files.
    if total_files == 0:
        if not quiet:
            print("  Indexed 0 changed / 0 total files.")
        return

    ef = get_embeddings()

    cache_path = os.path.join(db_path, "file_hashes.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    vectorstore = Chroma(persist_directory=db_path, embedding_function=ef)

    new_cache = {}
    updated = 0
    processed = 0
    skipped_large = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    
    # Batch processing for faster indexing
    batch_docs = []
    batch_size = 32

    def _flush():
        nonlocal batch_docs
        if batch_docs:
            vectorstore.add_documents(batch_docs)
            batch_docs = []

    for fpath, content in walk_codebase(root_path):
        processed += 1

        # A single huge file (a bundled/minified vendor script, a large
        # generated JSON fixture, a lockfile) can produce tens of thousands
        # of chunks in one create_documents() call. The old code only
        # flushed batch_docs AFTER the whole file's docs were appended, so
        # one such file could hand Chroma a single add_documents() batch far
        # larger than its own internal max_batch_size (observed: a batch of
        # 30586 against a 5461 ceiling) — the whole index run then raised
        # and aborted with nothing flushed. Skip absolute monsters outright;
        # they are almost never useful for semantic code search anyway.
        if len(content) > _MAX_FILE_CHARS:
            skipped_large.append(fpath)
            progress_callback(processed, total_files, fpath, False)
            continue

        try:
            h = file_hash(fpath)
        except Exception:
            progress_callback(processed, total_files, fpath, False)
            continue

        new_cache[fpath] = h
        if cache.get(fpath) == h:
            progress_callback(processed, total_files, fpath, False)
            continue

        try:
            vectorstore.delete(where={"source": fpath})
        except Exception:
            pass

        splits = text_splitter.split_text(content)
        if splits:
            docs = text_splitter.create_documents(
                splits,
                metadatas=[{"source": fpath}] * len(splits)
            )
            # Flush in fixed-size slices as we go, instead of extending
            # batch_docs with the entire file's docs and checking the size
            # only afterward. This guarantees add_documents() is NEVER
            # called with more than batch_size documents, no matter how
            # many chunks a single file produces.
            for i in range(0, len(docs), batch_size):
                batch_docs.extend(docs[i:i + batch_size])
                if len(batch_docs) >= batch_size:
                    _flush()

            updated += 1

        progress_callback(processed, total_files, fpath, True)

    # Add remaining documents
    _flush()

    json.dump(new_cache, open(cache_path, "w"))
    if not quiet:
        print(f"  Indexed {updated} changed / {len(new_cache)} total files.")
    if skipped_large:
        if not quiet:
            print(
                f"  Skipped {len(skipped_large)} file(s) over "
                f"{_MAX_FILE_CHARS // 1000}k chars (too large to usefully "
                f"chunk for semantic search): "
                + ", ".join(os.path.basename(p) for p in skipped_large[:5])
                + (f" and {len(skipped_large) - 5} more" if len(skipped_large) > 5 else "")
            )

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        index_codebase(sys.argv[1])
    else:
        print("Usage: python indexer.py /path/to/project")

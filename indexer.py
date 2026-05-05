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
        except ImportError:
            # Fall back to regular embeddings if turboquant not available
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

def walk_codebase(root_path: str):
    skip_dirs = {
        '.git', 'node_modules', '__pycache__', 'venv', 'dist', 'build',
        '.idea', 'chroma_db', '.mypy_cache', '.pytest_cache', '.tox',
    }
    code_exts = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.scss',
        '.json', '.yaml', '.yml', '.toml', '.md', '.txt', '.sh', '.bash',
        '.c', '.cpp', '.h', '.hpp', '.java', '.go', '.rs', '.rb', '.php',
        '.sql', '.graphql',
    }
    named_files = {'Dockerfile', 'Makefile', '.env.example'}

    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
        for f in files:
            _, ext = os.path.splitext(f)
            if ext.lower() not in code_exts and f not in named_files:
                continue
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    content = file.read()
                if content.strip():
                    yield path, content
            except Exception:
                continue

def index_codebase(root_path: str, db_path: str = None):
    if db_path is None:
        db_path = os.environ.get("CODI_CHROMA_DIR", CHROMA_PERSIST_DIR)

    print(f"  Indexing: {root_path}")
    os.makedirs(db_path, exist_ok=True)
    ef = get_embeddings()

    cache_path = os.path.join(db_path, "file_hashes.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    vectorstore = Chroma(persist_directory=db_path, embedding_function=ef)

    new_cache = {}
    updated = 0
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    
    # Batch processing for faster indexing
    batch_docs = []
    batch_size = 32

    for fpath, content in walk_codebase(root_path):
        try:
            h = file_hash(fpath)
        except Exception:
            continue

        new_cache[fpath] = h
        if cache.get(fpath) == h:
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
            batch_docs.extend(docs)
            
            # Add documents in batches for better performance
            if len(batch_docs) >= batch_size:
                vectorstore.add_documents(batch_docs)
                batch_docs = []
            
            updated += 1

    # Add remaining documents
    if batch_docs:
        vectorstore.add_documents(batch_docs)

    json.dump(new_cache, open(cache_path, "w"))
    print(f"  Indexed {updated} changed / {len(new_cache)} total files.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        index_codebase(sys.argv[1])
    else:
        print("Usage: python indexer.py /path/to/project")
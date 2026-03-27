import os
import json
import hashlib
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma          # updated — replaces deprecated langchain_community.vectorstores.Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore():
    if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
        return None
    try:
        embeddings = get_embeddings()
        return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    except Exception:
        return None

def file_hash(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def walk_codebase(root_path: str):
    skip_dirs = {'.git', 'node_modules', '__pycache__', 'venv', 'dist', 'build', '.idea', 'chroma_db'}
    for root, dirs, files in os.walk(root_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
        for f in files:
            path = os.path.join(root, f)
            try:
                with open(path, "r", encoding="utf-8") as file:
                    yield path, file.read()
            except Exception:
                continue

def index_codebase(root_path: str, db_path: str = CHROMA_PERSIST_DIR):
    print(f"Indexing codebase at: {root_path} ...")
    os.makedirs(db_path, exist_ok=True)
    ef = get_embeddings()

    cache_path = os.path.join(db_path, "file_hashes.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    vectorstore = Chroma(persist_directory=db_path, embedding_function=ef)

    new_cache = {}
    updated = 0
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    for fpath, content in walk_codebase(root_path):
        try:
            h = file_hash(fpath)
        except Exception:
            continue

        new_cache[fpath] = h
        if cache.get(fpath) == h:
            continue  # unchanged, skip

        try:
            vectorstore.delete(where={"source": fpath})
        except Exception:
            pass

        splits = text_splitter.split_text(content)
        if splits:
            docs = text_splitter.create_documents(splits, metadatas=[{"source": fpath}] * len(splits))
            vectorstore.add_documents(docs)
        updated += 1

    json.dump(new_cache, open(cache_path, "w"))
    print(f"Indexed {updated} changed files. {len(new_cache)} total tracked.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        index_codebase(sys.argv[1])
    else:
        print("Usage: python indexer.py /path/to/project")
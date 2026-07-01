# tools/local/search_tools.py

import functools

from context_trimmer import trim_tool_output
from logger import log


@functools.lru_cache(maxsize=128)
def _cached_search(query: str, k: int = 5):
    from indexer import get_vectorstore

    vs = get_vectorstore()
    if vs is None:
        return ()

    docs = vs.similarity_search(query, k=k)
    return tuple((doc.page_content, dict(doc.metadata)) for doc in docs)


def search_codebase(args: dict) -> str:
    """Semantic search across the indexed project codebase. Returns top matching chunks."""
    query = args.get("query", "")
    if not query:
        return "ERROR: no query provided"

    log("tool_call", {"tool": "search_codebase", "query": query})
    try:
        docs = _cached_search(query, k=5)
        if not docs:
            return "Codebase not indexed yet. Run /index first." if not docs else "No matching code chunks found."
        results = []
        for i, (content, metadata) in enumerate(docs):
            source = metadata.get("source", "unknown")
            chunk = trim_tool_output(content, max_tokens=200)
            results.append(f"--- Chunk {i+1} [{source}] ---\n{chunk}")
        log("tool_result", {"tool": "search_codebase", "chunks": len(docs), "status": "ok"})
        return "\n\n".join(results)
    except Exception as e:
        return f"ERROR searching codebase: {e}"


def register_search_tools(registry):
    registry.register_local("search_codebase", search_codebase)

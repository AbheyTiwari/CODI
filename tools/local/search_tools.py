# tools/local/search_tools.py

from context_trimmer import trim_tool_output
from logger import log


def search_codebase(args: dict) -> str:
    """Semantic search across the indexed project codebase. Returns top matching chunks."""
    query = args.get("query", "")
    if not query:
        return "ERROR: no query provided"

    log("tool_call", {"tool": "search_codebase", "query": query})
    try:
        from indexer import get_vectorstore
        vs = get_vectorstore()
        if vs is None:
            return "Codebase not indexed yet. Run /index first."
        docs = vs.similarity_search(query, k=5)
        if not docs:
            return "No matching code chunks found."
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "unknown")
            chunk  = trim_tool_output(doc.page_content, max_tokens=200)
            results.append(f"--- Chunk {i+1} [{source}] ---\n{chunk}")
        log("tool_result", {"tool": "search_codebase", "chunks": len(docs), "status": "ok"})
        return "\n\n".join(results)
    except Exception as e:
        return f"ERROR searching codebase: {e}"


def register_search_tools(registry):
    registry.register_local("search_codebase", search_codebase)

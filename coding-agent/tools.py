import os
import subprocess
from langchain_core.tools import tool
from indexer import get_vectorstore

@tool
def run_command(command: str) -> str:
    """Runs a shell command and returns the stdout and stderr. Use this to execute CLI commands."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=60
        )
        output = result.stdout
        if result.stderr:
            output += "\nError Output:\n" + result.stderr
        return output
    except subprocess.TimeoutExpired:
        return "Command timed out after 60 seconds."
    except Exception as e:
        return f"Error executing command: {e}"

@tool
def read_file(path: str) -> str:
    """Reads and returns the full contents of any file by path."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

@tool
def write_file(path_and_content: str) -> str:
    """Write content to a file. Input must be formatted as: path::content
    Example: index.html::<html>...</html>"""
    if "::" not in path_and_content:
        return "ERROR: Input must be formatted as path::content"
    path, content = path_and_content.split("::", 1)
    path = path.strip()
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(os.path.expanduser(path), "w", encoding="utf-8") as f:
            f.write(content)
        return f"Written {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR writing file: {e}"

@tool
def search_codebase(query: str) -> str:
    """Semantic search across ChromaDB — returns top 5 relevant chunks with file + chunk metadata."""
    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            return "Codebase not indexed yet. Run /index <path> first."
        
        docs = vectorstore.similarity_search(query, k=5)
        if not docs:
            return "No matching code chunks found."
        
        results = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown file")
            results.append(f"--- Chunk {i+1} from {source} ---\n{doc.page_content}\n")
        return "\n".join(results)
    except Exception as e:
        if "not indexed" in str(e).lower() or "does not exist" in str(e).lower():
           return "Codebase not indexed yet. Run /index <path> first."
        return f"Error searching codebase: {e}"

@tool
def list_files(dir_path: str) -> str:
    """Lists all files in a directory recursively (skips junk dirs)."""
    skip_dirs = {'.git', 'node_modules', '__pycache__', 'venv', 'dist', 'build'}
    try:
        file_list = []
        for root, dirs, files in os.walk(dir_path):
            dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
            for file in files:
                file_list.append(os.path.join(root, file))
        return "\n".join(file_list)
    except Exception as e:
        return f"Error listing files: {e}"

from mcp_client import load_all_mcp_tools

def get_all_tools():
    base_tools = [run_command, read_file, write_file, search_codebase, list_files]
    mcp_tools = load_all_mcp_tools()
    
    all_tools = base_tools + mcp_tools
    print(f"  [Tools] {len(base_tools)} native + {len(mcp_tools)} MCP = {len(all_tools)} total tools")
    return all_tools

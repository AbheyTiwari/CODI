import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore():
    # Chroma requires the directory to exist, but if it doesn't have a valid index, it might throw if queried.
    # To be safe, we check if there's an index created (e.g., chroma.sqlite3 exists inside).
    if not os.path.exists(CHROMA_PERSIST_DIR) or not os.listdir(CHROMA_PERSIST_DIR):
        return None
    try:
        embeddings = get_embeddings()
        return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    except Exception:
        return None

def index_codebase(directory_path: str):
    print(f"Indexing codebase at: {directory_path} ...")
    
    try:
        # User requested TextLoader paired with DirectoryLoader
        # Note: glob="**/*" tries to read all files which might include binary ones. 
        # Using a restricted glob if necessary, but we'll autodetect encoding and swallow errors on binaries if needed.
        # It's cleaner to list allowed file types if possible, or just ignore errors.
        
        loader = DirectoryLoader(
            directory_path, 
            glob="**/*",
            exclude=[
                "**/.git/**/*", "**/node_modules/**/*", "**/__pycache__/**/*", 
                "**/venv/**/*", "**/dist/**/*", "**/build/**/*", "**/.*/"
            ],
            loader_cls=TextLoader, 
            loader_kwargs={"autodetect_encoding": True},
            show_progress=True,
            use_multithreading=True
        )
        docs = loader.load()
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    if not docs:
        print("No readable documents found to index.")
        return

    print(f"Loaded {len(docs)} documents. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    print(f"Created {len(splits)} chunks. Generating embeddings and storing in ChromaDB...")
    embeddings = get_embeddings()
    
    if os.path.exists(CHROMA_PERSIST_DIR):
        import shutil
        shutil.rmtree(CHROMA_PERSIST_DIR)
        
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings, 
        persist_directory=CHROMA_PERSIST_DIR
    )
    print("Indexing complete!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        index_codebase(sys.argv[1])
    else:
        print("Usage: python indexer.py /path/to/project")

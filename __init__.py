# core/__init__.py


#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║     CAVE MERGER  —  Unified Knowledge Base Builder           ║
║  Takes one or more scraped_data/ output folders and merges  ║
║  everything into a single, deduplicated knowledge base.      ║
╚══════════════════════════════════════════════════════════════╝

Use this when you've run scraper.py on multiple sources separately
and want to combine them, OR to re-merge after adding a new source
without re-scraping everything.

What it does:
  • Merges rag_chunks — content-hash deduplication (no duplicates)
  • Merges glossary_terms — longest definition wins, sources unioned
  • Merges tables, images, PDFs metadata
  • Produces a master knowledge_base.json keyed by term
  • Produces unified rag_chunks.jsonl ready for any vector store

Usage:
    python merger.py scraped_data/
    python merger.py docs_scraped/ tomtom_scraped/
    python merger.py docs_scraped/ tomtom_scraped/ --output kb/
"""

import os
import sys
import json
import hashlib
import argparse
from datetime import datetime, timezone

R="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
CYAN="\033[96m"; GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"; PURPLE="\033[95m"

def cprint(msg, c=R): print(f"{c}{msg}{R}", flush=True)

def uid(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:10]

def content_hash(text: str) -> str:
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_json_file(path: str) -> list | dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def load_jsonl_file(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except Exception:
                    pass
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  MERGE: RAG chunks (content-hash dedup)
# ─────────────────────────────────────────────────────────────────────────────

def merge_rag_chunks(folders: list[str]) -> list[dict]:
    """Load and deduplicate all rag_chunks across all input folders."""
    all_chunks: list[dict] = []
    seen_hashes: set[str] = set()

    for folder in folders:
        # Prefer .jsonl (faster), fall back to .json
        chunks = load_jsonl_file(os.path.join(folder, "rag_chunks.jsonl"))
        if not chunks:
            chunks = load_json_file(os.path.join(folder, "rag_chunks.json")) or []

        added = 0
        for chunk in chunks:
            h = content_hash(chunk.get("text", ""))
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_chunks.append(chunk)
                added += 1

        cprint(f"  {folder:<35}  {len(chunks):>5} chunks loaded  {added:>5} new (unique)", DIM)

    return all_chunks


# ─────────────────────────────────────────────────────────────────────────────
#  MERGE: Glossary (longest definition wins, sources unioned)
# ─────────────────────────────────────────────────────────────────────────────

def merge_glossaries(folders: list[str]) -> dict[str, dict]:
    """
    Merge glossary_terms.json files from all input folders.
    Key = normalised term (lowercase stripped).
    Merge rules:
      - Longer definition wins
      - sources list is unioned
      - images / links lists are unioned
    """
    knowledge: dict[str, dict] = {}

    for folder in folders:
        entries = load_jsonl_file(os.path.join(folder, "glossary_terms.jsonl"))
        if not entries:
            entries = load_json_file(os.path.join(folder, "glossary_terms.json")) or []

        for entry in entries:
            key = entry.get("term", "").lower().strip()
            if not key:
                continue

            if key not in knowledge:
                knowledge[key] = {
                    "term":       entry.get("term", ""),
                    "definition": entry.get("definition", ""),
                    "sources":    list(entry.get("sources", [])),
                    "images":     list(entry.get("images", [])),
                    "links":      list(entry.get("links", [])),
                    "tables":     list(entry.get("tables", [])),
                }
            else:
                existing = knowledge[key]
                # Longer definition wins
                if len(entry.get("definition", "")) > len(existing["definition"]):
                    existing["definition"] = entry["definition"]
                # Union sources
                for src in entry.get("sources", []):
                    if src not in existing["sources"]:
                        existing["sources"].append(src)
                # Union images
                for img in entry.get("images", []):
                    if img not in existing["images"]:
                        existing["images"].append(img)
                # Union links
                for lnk in entry.get("links", []):
                    if lnk not in existing["links"]:
                        existing["links"].append(lnk)

        cprint(f"  {folder:<35}  {len(entries):>5} glossary entries", DIM)

    return knowledge


# ─────────────────────────────────────────────────────────────────────────────
#  MERGE: Images and Tables (deduplicate by URL/content)
# ─────────────────────────────────────────────────────────────────────────────

def merge_images(folders: list[str]) -> list[dict]:
    seen_urls: set[str] = set()
    all_images: list[dict] = []
    for folder in folders:
        images = load_json_file(os.path.join(folder, "images.json")) or []
        for img in images:
            url = img.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_images.append(img)
    return all_images


def merge_tables(folders: list[str]) -> list[dict]:
    seen_hashes: set[str] = set()
    all_tables: list[dict] = []
    for folder in folders:
        tables = load_json_file(os.path.join(folder, "tables.json")) or []
        for tbl in tables:
            h = content_hash(tbl.get("as_text", "") or tbl.get("caption", ""))
            if h not in seen_hashes:
                seen_hashes.add(h)
                all_tables.append(tbl)
    return all_tables


def merge_pdfs(folders: list[str]) -> list[dict]:
    seen_urls: set[str] = set()
    all_pdfs: list[dict] = []
    for folder in folders:
        pdfs = load_json_file(os.path.join(folder, "pdfs.json")) or []
        for pdf in pdfs:
            url = pdf.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_pdfs.append(pdf)
    return all_pdfs


# ─────────────────────────────────────────────────────────────────────────────
#  BUILD MASTER KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────────────────────

def build_knowledge_base(glossary: dict[str, dict], chunks: list[dict]) -> dict:
    """
    Build a master knowledge_base.json:
    - Each glossary term is the primary key
    - Related RAG chunks are attached by scanning for term mentions
    - Produces a structure suitable for both lookup and semantic search
    """
    # Index chunks by source_type for fast lookup
    glossary_chunks = [c for c in chunks if c.get("source_type") == "glossary"]

    kb: dict[str, dict] = {}
    for key, entry in glossary.items():
        kb[key] = {
            **entry,
            "rag_chunk_ids": [],
        }

    # Attach glossary-type chunks to their terms
    for chunk in glossary_chunks:
        term_key = chunk.get("term", "").lower().strip()
        if term_key and term_key in kb:
            kb[term_key]["rag_chunk_ids"].append(chunk.get("id", ""))

    return kb


# ─────────────────────────────────────────────────────────────────────────────
#  SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_merged(
    chunks: list[dict],
    glossary: dict[str, dict],
    knowledge_base: dict,
    images: list[dict],
    tables: list[dict],
    pdfs: list[dict],
    out_dir: str,
    source_folders: list[str],
):
    os.makedirs(out_dir, exist_ok=True)

    # ── RAG chunks ─────────────────────────────────────────────────────────
    chunks_json = os.path.join(out_dir, "rag_chunks.json")
    with open(chunks_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    chunks_jsonl = os.path.join(out_dir, "rag_chunks.jsonl")
    with open(chunks_jsonl, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # ── Glossary ──────────────────────────────────────────────────────────
    glossary_list = sorted(glossary.values(), key=lambda x: x["term"].lower())

    glossary_json = os.path.join(out_dir, "glossary_terms.json")
    with open(glossary_json, "w", encoding="utf-8") as f:
        json.dump(glossary_list, f, indent=2, ensure_ascii=False)

    glossary_jsonl = os.path.join(out_dir, "glossary_terms.jsonl")
    with open(glossary_jsonl, "w", encoding="utf-8") as f:
        for entry in glossary_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Master knowledge base ──────────────────────────────────────────────
    kb_path = os.path.join(out_dir, "knowledge_base.json")
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "built_at":      datetime.now(timezone.utc).isoformat(),
                "source_folders": source_folders,
                "total_terms":   len(knowledge_base),
                "total_chunks":  len(chunks),
            },
            "terms": knowledge_base,
        }, f, indent=2, ensure_ascii=False)

    # ── Images / tables / PDFs ────────────────────────────────────────────
    with open(os.path.join(out_dir, "images.json"), "w", encoding="utf-8") as f:
        json.dump(images, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "tables.json"), "w", encoding="utf-8") as f:
        json.dump(tables, f, indent=2, ensure_ascii=False)

    with open(os.path.join(out_dir, "pdfs.json"), "w", encoding="utf-8") as f:
        json.dump(pdfs, f, indent=2, ensure_ascii=False)

    # ── Summary stats ─────────────────────────────────────────────────────
    # Chunk breakdown by source_type
    type_counts: dict[str, int] = {}
    for c in chunks:
        t = c.get("source_type", "text")
        type_counts[t] = type_counts.get(t, 0) + 1

    # Source breakdown by source_url domain
    domain_counts: dict[str, int] = {}
    for c in chunks:
        from urllib.parse import urlparse
        domain = urlparse(c.get("source_url", "")).netloc or "unknown"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    summary = {
        "built_at":       datetime.now(timezone.utc).isoformat(),
        "source_folders": source_folders,
        "total_chunks":   len(chunks),
        "total_terms":    len(glossary_list),
        "total_images":   len(images),
        "total_tables":   len(tables),
        "total_pdfs":     len(pdfs),
        "chunks_by_type": type_counts,
        "chunks_by_domain": domain_counts,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Cave Merger — unified knowledge base builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("folders", nargs="+", help="One or more scraped_data/ output folders")
    p.add_argument("--output", default="knowledge_base", help="Output folder (default: knowledge_base/)")
    args = p.parse_args()

    # Validate folders
    for folder in args.folders:
        if not os.path.isdir(folder):
            cprint(f"  ✕  Folder not found: {folder}", RED)
            sys.exit(1)

    cprint(f"\n{'═'*60}", CYAN)
    cprint(f"  CAVE MERGER  ⑂  Unified Knowledge Base Builder", BOLD)
    for folder in args.folders:
        cprint(f"  Source: {folder}", DIM)
    cprint(f"  Output: {args.output}/", DIM)
    cprint(f"{'═'*60}\n", CYAN)

    cprint(f"  Merging RAG chunks...", CYAN)
    chunks = merge_rag_chunks(args.folders)
    cprint(f"  → {len(chunks)} unique chunks total\n", GREEN)

    cprint(f"  Merging glossary terms...", CYAN)
    glossary = merge_glossaries(args.folders)
    cprint(f"  → {len(glossary)} unique terms total\n", GREEN)

    cprint(f"  Merging images, tables, PDFs...", CYAN)
    images = merge_images(args.folders)
    tables = merge_tables(args.folders)
    pdfs   = merge_pdfs(args.folders)
    cprint(f"  → {len(images)} images  |  {len(tables)} tables  |  {len(pdfs)} PDFs\n", GREEN)

    cprint(f"  Building master knowledge base...", CYAN)
    kb = build_knowledge_base(glossary, chunks)
    cprint(f"  → {len(kb)} terms in knowledge base\n", GREEN)

    cprint(f"  Saving to {args.output}/...", CYAN)
    summary = save_merged(chunks, glossary, kb, images, tables, pdfs, args.output, args.folders)

    cprint(f"\n{'═'*60}", GREEN)
    cprint(f"  ✓  MERGE COMPLETE", GREEN + BOLD)
    cprint(f"{'═'*60}", GREEN)
    cprint(f"\n  OUTPUT FILES in {args.output}/:", CYAN + BOLD)
    cprint(f"  {'rag_chunks.jsonl':<28}  {summary['total_chunks']} chunks  ← feed to your embedder", GREEN)
    cprint(f"  {'rag_chunks.json':<28}  same, formatted", GREEN)
    cprint(f"  {'glossary_terms.json':<28}  {summary['total_terms']} unique terms", PURPLE)
    cprint(f"  {'glossary_terms.jsonl':<28}  same, one per line", PURPLE)
    cprint(f"  {'knowledge_base.json':<28}  master KB (terms + chunk refs + metadata)", YELLOW)
    cprint(f"  {'summary.json':<28}  stats breakdown", DIM)
    cprint(f"  {'images.json':<28}  {summary['total_images']} images", DIM)
    cprint(f"  {'tables.json':<28}  {summary['total_tables']} tables", DIM)
    cprint(f"  {'pdfs.json':<28}  {summary['total_pdfs']} PDFs", DIM)

    if summary["chunks_by_type"]:
        cprint(f"\n  Chunks by type:", CYAN)
        for t, count in sorted(summary["chunks_by_type"].items(), key=lambda x: -x[1]):
            cprint(f"    {t:<25}  {count}", DIM)

    if summary["chunks_by_domain"]:
        cprint(f"\n  Chunks by source domain:", CYAN)
        for domain, count in sorted(summary["chunks_by_domain"].items(), key=lambda x: -x[1]):
            cprint(f"    {domain:<40}  {count}", DIM)

    cprint(f"\n  → Load {args.output}/rag_chunks.jsonl into your vector store and go!\n", YELLOW + BOLD)


if __name__ == "__main__":
    main()

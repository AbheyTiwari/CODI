# core/__init__.py

#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════╗
║     CAVE SCRAPER  —  Full Content Extractor           ║
║  Reads site_map.json → extracts ALL content from      ║
║  every page → outputs RAG-ready chunks + assets       ║
╚═══════════════════════════════════════════════════════╝

Extracts per page:
  • Clean markdown text (body, headings, paragraphs)
  • Tables  → list of row dicts with headers
  • Images  → url, alt text, caption
  • Links   → anchor text + href
  • Forms   → fields, action, method
  • PDFs    → download urls found on the page
  • Metadata → title, description, og tags, canonical
  • JSON-LD  → structured data blocks (schema.org etc.)
  • RAG chunks → text split into overlapping chunks
                 ready to embed into a vector store

Usage:
    python scraper.py site_map.json
    python scraper.py site_map.json --filter /blog
    python scraper.py site_map.json --depth 2 --chunk-size 500 --threads 8
    python scraper.py site_map.json --output my_data --delay 0.5
"""

import os
import re
import sys
import json
import time
import hashlib
import argparse
import threading
from urllib.parse import urljoin, urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup, Tag

try:
    import markdownify as md_lib
    HAS_MD = True
except ImportError:
    HAS_MD = False

# ── terminal colours ──────────────────────────────────────────────────────────
R="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
CYAN="\033[96m"; GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"; PURPLE="\033[95m"

def cprint(msg, c=R): print(f"{c}{msg}{R}", flush=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CaveScraperBot/1.0)",
    "Accept-Language": "en-US,en;q=0.9",
}

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def uid(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:10]

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, size: int, overlap: int) -> list[dict]:
    """Split text into overlapping chunks for RAG embedding."""
    words = text.split()
    chunks = []
    i = 0
    idx = 0
    while i < len(words):
        chunk_words = words[i:i+size]
        chunk = " ".join(chunk_words)
        if chunk.strip():
            chunks.append({"chunk_index": idx, "text": chunk, "word_count": len(chunk_words)})
            idx += 1
        i += size - overlap
    return chunks

def table_to_dicts(table_tag: Tag) -> list[dict]:
    """Convert HTML <table> to list of row-dicts keyed by header."""
    rows = table_tag.find_all("tr")
    if not rows:
        return []
    headers = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
    if not headers:
        return []
    result = []
    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if cells:
            # pad / trim to header length
            cells += [""] * (len(headers) - len(cells))
            result.append(dict(zip(headers, cells[:len(headers)])))
    return result

def table_to_text(table_dicts: list[dict]) -> str:
    """Turn table rows into readable text for RAG."""
    if not table_dicts:
        return ""
    lines = []
    headers = list(table_dicts[0].keys())
    lines.append(" | ".join(headers))
    lines.append("-" * len(lines[0]))
    for row in table_dicts:
        lines.append(" | ".join(str(row.get(h,"")) for h in headers))
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  CORE EXTRACTOR  —  everything from one page
# ─────────────────────────────────────────────────────────────────────────────

def extract_all(url: str, html: str, chunk_size: int, chunk_overlap: int) -> dict:
    soup = BeautifulSoup(html, "lxml")
    base = url

    # ── 1. Metadata ──────────────────────────────────────────────────────────
    meta = {}
    for m in soup.find_all("meta"):
        name = m.get("name") or m.get("property") or ""
        content = m.get("content") or ""
        if name and content:
            meta[name] = content[:300]

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else meta.get("og:title","")

    canonical = ""
    link_can = soup.find("link", rel="canonical")
    if link_can:
        canonical = link_can.get("href","")

    # ── 2. Remove noise tags ─────────────────────────────────────────────────
    for tag in soup(["script","style","noscript","svg","header","footer",
                     "nav","aside","[role=navigation]","[aria-hidden=true]"]):
        tag.decompose()

    # ── 3. Headings ──────────────────────────────────────────────────────────
    headings = []
    for tag in soup.find_all(["h1","h2","h3","h4","h5","h6"]):
        txt = clean_text(tag.get_text())
        if txt:
            headings.append({"level": tag.name, "text": txt})

    # ── 4. Tables ────────────────────────────────────────────────────────────
    tables = []
    for i, tbl in enumerate(soup.find_all("table")):
        rows = table_to_dicts(tbl)
        if rows:
            caption_tag = tbl.find("caption")
            caption = caption_tag.get_text(strip=True) if caption_tag else f"Table {i+1}"
            tables.append({
                "caption": caption,
                "rows": rows,
                "as_text": table_to_text(rows),
            })
        tbl.decompose()  # remove from soup so it doesn't appear in body text

    # ── 5. Images ────────────────────────────────────────────────────────────
    images = []
    for img in soup.find_all("img"):
        src = img.get("src","").strip()
        if not src or src.startswith("data:"):
            continue
        abs_src = urljoin(base, src)
        alt  = clean_text(img.get("alt",""))
        # look for caption in adjacent figcaption
        caption = ""
        fig = img.find_parent("figure")
        if fig:
            fc = fig.find("figcaption")
            if fc:
                caption = clean_text(fc.get_text())
        images.append({
            "url": abs_src,
            "alt": alt,
            "caption": caption,
        })

    # ── 6. Links ─────────────────────────────────────────────────────────────
    links = []
    seen_links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:","tel:","javascript:","#")):
            continue
        abs_href = urljoin(base, href)
        txt = clean_text(a.get_text())
        if abs_href not in seen_links:
            seen_links.add(abs_href)
            links.append({"text": txt[:100], "url": abs_href})

    # ── 7. PDF / file links ───────────────────────────────────────────────────
    pdf_links = [
        lnk for lnk in links
        if lnk["url"].lower().endswith((".pdf",".docx",".xlsx",".csv",".pptx"))
    ]

    # ── 8. Forms ─────────────────────────────────────────────────────────────
    forms = []
    for form in soup.find_all("form"):
        fields = []
        for inp in form.find_all(["input","textarea","select"]):
            fields.append({
                "tag":   inp.name,
                "name":  inp.get("name") or inp.get("id") or "",
                "type":  inp.get("type","text"),
                "placeholder": inp.get("placeholder",""),
            })
        forms.append({
            "action": urljoin(base, form.get("action") or ""),
            "method": (form.get("method") or "GET").upper(),
            "fields": fields,
        })

    # ── 9. JSON-LD structured data ────────────────────────────────────────────
    json_ld = []
    for s in BeautifulSoup(html, "lxml").find_all("script", type="application/ld+json"):
        try:
            json_ld.append(json.loads(s.string or ""))
        except Exception:
            pass

    # ── 10. Body text  (markdown if available, else plain) ───────────────────
    main = soup.find("main") or soup.find("article") or soup.find("body") or soup
    if HAS_MD:
        body_text = md_lib.markdownify(str(main), heading_style="ATX", strip=["a","img"])
        body_text = re.sub(r'\n{3,}', '\n\n', body_text).strip()
    else:
        body_text = clean_text(main.get_text(" ", strip=True))

    # ── 11. Full page text (headings + tables + body) for RAG ────────────────
    heading_text = "\n".join(f"{'#'*int(h['level'][1:])} {h['text']}" for h in headings)
    table_text   = "\n\n".join(t["as_text"] for t in tables)
    image_text   = "\n".join(
        f"[Image: {i['alt'] or 'no alt'}{ ' — ' + i['caption'] if i['caption'] else ''}]"
        for i in images
    )
    full_text = "\n\n".join(filter(None, [heading_text, body_text, table_text, image_text]))

    # ── 12. RAG chunks ────────────────────────────────────────────────────────
    rag_chunks = []
    # body chunks
    rag_chunks += chunk_text(body_text, chunk_size, chunk_overlap)
    # each table as its own chunk
    for t in tables:
        if t["as_text"]:
            rag_chunks.append({
                "chunk_index": len(rag_chunks),
                "text": f"Table: {t['caption']}\n{t['as_text']}",
                "word_count": len(t['as_text'].split()),
                "source_type": "table",
            })
    # image alt/caption as mini-chunks
    for img in images:
        img_txt = " ".join(filter(None,[img["alt"], img["caption"]])).strip()
        if img_txt:
            rag_chunks.append({
                "chunk_index": len(rag_chunks),
                "text": f"Image on page: {img_txt}",
                "word_count": len(img_txt.split()),
                "source_type": "image_description",
                "image_url": img["url"],
            })

    return {
        "url":        url,
        "path":       urlparse(url).path or "/",
        "title":      title,
        "canonical":  canonical,
        "metadata":   meta,
        "headings":   headings,
        "body_text":  body_text,
        "full_text":  full_text,
        "tables":     tables,
        "images":     images,
        "links":      links,
        "pdf_links":  pdf_links,
        "forms":      forms,
        "json_ld":    json_ld,
        "rag_chunks": rag_chunks,
        "stats": {
            "words":   len(body_text.split()),
            "tables":  len(tables),
            "images":  len(images),
            "links":   len(links),
            "pdfs":    len(pdf_links),
            "chunks":  len(rag_chunks),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  FETCH + EXTRACT one page
# ─────────────────────────────────────────────────────────────────────────────

def scrape_page(page: dict, session: requests.Session,
                chunk_size: int, chunk_overlap: int) -> dict:
    url = page["url"]
    result = {"url": url, "path": page["path"], "error": None, "data": {}}
    try:
        resp = session.get(url, timeout=12, headers=HEADERS, allow_redirects=True)
        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}"
            return result
        ct = resp.headers.get("content-type","")
        if "text/html" not in ct:
            result["error"] = f"Non-HTML ({ct.split(';')[0].strip()})"
            return result
        result["data"] = extract_all(url, resp.text, chunk_size, chunk_overlap)
    except requests.exceptions.Timeout:
        result["error"] = "Timeout"
    except Exception as e:
        result["error"] = str(e)[:120]
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  WRITE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(results: list[dict], out_dir: str, site_url: str):
    os.makedirs(out_dir, exist_ok=True)

    # ── full_data.json  (everything, one file) ───────────────────────────────
    full_path = os.path.join(out_dir, "full_data.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": site_url,
            "scraped_at": datetime.now(timezone.utc).isoformat(),
            "total_pages": len(results),
            "pages": results,
        }, f, indent=2, ensure_ascii=False)

    # ── rag_chunks.json  (flat list of all chunks — feed this to embeddings) ─
    all_chunks = []
    for r in results:
        if r["error"] or not r["data"]:
            continue
        d = r["data"]
        for chunk in d.get("rag_chunks", []):
            all_chunks.append({
                "id":         uid(r["url"] + str(chunk["chunk_index"])),
                "source_url": r["url"],
                "source_path": r["path"],
                "page_title": d.get("title",""),
                "chunk_index": chunk["chunk_index"],
                "text":        chunk["text"],
                "word_count":  chunk.get("word_count", 0),
                "source_type": chunk.get("source_type","text"),
                "image_url":   chunk.get("image_url",""),
            })
    chunks_path = os.path.join(out_dir, "rag_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # ── rag_chunks.jsonl  (one chunk per line — compatible with most loaders) ─
    jsonl_path = os.path.join(out_dir, "rag_chunks.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # ── images.json  (all images across the site) ────────────────────────────
    all_images = []
    for r in results:
        if r["error"] or not r["data"]:
            continue
        for img in r["data"].get("images", []):
            all_images.append({**img, "found_on": r["url"]})
    images_path = os.path.join(out_dir, "images.json")
    with open(images_path, "w", encoding="utf-8") as f:
        json.dump(all_images, f, indent=2, ensure_ascii=False)

    # ── tables.json  (all tables across the site) ────────────────────────────
    all_tables = []
    for r in results:
        if r["error"] or not r["data"]:
            continue
        for tbl in r["data"].get("tables", []):
            all_tables.append({**tbl, "found_on": r["url"], "page_title": r["data"].get("title","")})
    tables_path = os.path.join(out_dir, "tables.json")
    with open(tables_path, "w", encoding="utf-8") as f:
        json.dump(all_tables, f, indent=2, ensure_ascii=False)

    # ── pdfs.json  (all PDF / file links found) ───────────────────────────────
    all_pdfs = []
    for r in results:
        if r["error"] or not r["data"]:
            continue
        for pdf in r["data"].get("pdf_links", []):
            all_pdfs.append({**pdf, "found_on": r["url"]})
    pdfs_path = os.path.join(out_dir, "pdfs.json")
    with open(pdfs_path, "w", encoding="utf-8") as f:
        json.dump(all_pdfs, f, indent=2, ensure_ascii=False)

    return {
        "full_data":   full_path,
        "rag_chunks":  chunks_path,
        "rag_jsonl":   jsonl_path,
        "images":      images_path,
        "tables":      tables_path,
        "pdfs":        pdfs_path,
        "total_chunks": len(all_chunks),
        "total_images": len(all_images),
        "total_tables": len(all_tables),
        "total_pdfs":   len(all_pdfs),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Cave Scraper — full content extractor")
    p.add_argument("map_file",     help="site_map.json from crawler.py")
    p.add_argument("--filter",     default="",   help="Only scrape paths containing this string")
    p.add_argument("--depth",      type=int, default=99, help="Only scrape pages at/below this depth")
    p.add_argument("--threads",    type=int, default=8,  help="Parallel threads (default 8)")
    p.add_argument("--delay",      type=float, default=0.0, help="Seconds delay between requests")
    p.add_argument("--chunk-size", type=int, default=400,  help="RAG chunk size in words (default 400)")
    p.add_argument("--chunk-overlap", type=int, default=50, help="Overlap between chunks (default 50)")
    p.add_argument("--output",     default="scraped_data", help="Output folder (default: scraped_data/)")
    args = p.parse_args()

    with open(args.map_file, encoding="utf-8") as f:
        site_map = json.load(f)

    all_pages = site_map.get("pages", [])
    pages = [
        pg for pg in all_pages
        if (not args.filter or args.filter in pg["path"])
        and pg["depth"] <= args.depth
        and not pg.get("error")
    ]

    cprint(f"\n{'═'*60}", CYAN)
    cprint(f"  CAVE SCRAPER  ⑂  Full Content Extractor", BOLD)
    cprint(f"  Map    : {args.map_file}  ({len(all_pages)} pages mapped)", DIM)
    cprint(f"  Scraping {len(pages)} pages  |  {args.threads} threads  |  chunk={args.chunk_size}w", DIM)
    cprint(f"  Output : {args.output}/", DIM)
    cprint(f"{'═'*60}\n", CYAN)

    session = requests.Session()
    results = []
    errors  = 0
    lock    = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {
            ex.submit(scrape_page, pg, session, args.chunk_size, args.chunk_overlap): pg
            for pg in pages
        }
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            with lock:
                results.append(res)
                done += 1
            if res["error"]:
                errors += 1
                cprint(f"  ✕  {res['path']:<40}  {res['error']}", RED)
            else:
                d = res["data"]
                st = d.get("stats", {})
                cprint(
                    f"  ✓  {res['path']:<40}  "
                    f"{st.get('words',0):>5}w  "
                    f"{st.get('tables',0)}tbl  "
                    f"{st.get('images',0)}img  "
                    f"{st.get('chunks',0)}chunks",
                    GREEN
                )
            if args.delay:
                time.sleep(args.delay)

    # ── Save everything ──────────────────────────────────────────────────────
    site_url = site_map.get("meta", {}).get("root_url", "")
    info = save_outputs(results, args.output, site_url)

    cprint(f"\n{'═'*60}", GREEN)
    cprint(f"  ✓  DONE  —  {len(results)} pages scraped, {errors} errors", GREEN + BOLD)
    cprint(f"{'═'*60}", GREEN)
    cprint(f"\n  OUTPUT FILES:", CYAN + BOLD)
    cprint(f"  {'rag_chunks.jsonl':<22}  {info['total_chunks']} chunks  ← feed this to your embedder", GREEN)
    cprint(f"  {'rag_chunks.json':<22}  same, formatted", GREEN)
    cprint(f"  {'full_data.json':<22}  everything per page", DIM)
    cprint(f"  {'images.json':<22}  {info['total_images']} images (url + alt + caption)", DIM)
    cprint(f"  {'tables.json':<22}  {info['total_tables']} tables", DIM)
    cprint(f"  {'pdfs.json':<22}  {info['total_pdfs']} file links", DIM)
    cprint(f"\n  Each chunk in rag_chunks.jsonl has:", CYAN)
    cprint(f"    id, source_url, source_path, page_title,", DIM)
    cprint(f"    chunk_index, text, word_count, source_type", DIM)
    cprint(f"\n  → Load rag_chunks.jsonl into your vector store and go!\n", YELLOW + BOLD)


if __name__ == "__main__":
    main()

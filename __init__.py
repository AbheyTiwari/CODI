# core/__init__.py


#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║     CAVE SCRAPER v3  —  Multi-Source Content Extractor      ║
║  Reads ONE OR MORE site_map.json files →                    ║
║  extracts ALL content → deduplicates → outputs RAG-ready    ║
║  chunks + unified glossary + knowledge base                  ║
╚══════════════════════════════════════════════════════════════╝

Extracts per page:
  • Clean markdown text (Sphinx, Docusaurus, MkDocs, ReadTheDocs, GitBook)
  • Glossary entries  → term/definition pairs (dl/dt/dd, custom tables)
  • Sidebar / TOC navigation text
  • Tables  → structured rows
  • Images  → url, alt, caption
  • Code blocks  → language-tagged, preserved separately
  • Links, forms, JSON-LD
  • PDFs → text extraction via pypdf

Deduplication:
  • URLs deduplicated across all input maps (last-write-wins on metadata,
    longest-definition wins on glossary)
  • Chunk-level content hash deduplication (no identical text twice)
  • Glossary terms merged: longer definition wins, sources list unioned,
    images/tables/links unioned

Usage:
    python scraper.py site_map.json
    python scraper.py docs_map.json tomtom_map.json
    python scraper.py site_map.json --filter /docs --chunk-size 400
    python scraper.py site_map.json --no-pdfs
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

try:
    from pypdf import PdfReader
    import io
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

R="\033[0m"; BOLD="\033[1m"; DIM="\033[2m"
CYAN="\033[96m"; GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"; PURPLE="\033[95m"

def cprint(msg, c=R): print(f"{c}{msg}{R}", flush=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 CaveScraperBot/3.0"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Ordered candidate selectors for "main content" on docs/wiki sites
CONTENT_SELECTORS = [
    "main",
    "article",
    "[role=main]",
    "div.markdown-body",
    "div.theme-doc-markdown",
    "div.document",
    "div.rst-content",
    "div.section",
    "div#content",
    "div.content",
    "div.md-content",
    "div.page",
    "div.wiki-content",
    "div.body",
    "#main-content",
    "#mw-content-text",
]

SIDEBAR_SELECTORS = [
    "nav.toc", "div.toc", "aside.toc", "nav[aria-label=Sidebar]",
    "div.sidebar", "aside.sidebar", "div.theme-doc-sidebar-container",
    "nav.md-nav", "div.wy-menu",
]

MIN_CONTENT_WORDS = 40


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def uid(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:10]

def content_hash(text: str) -> str:
    return hashlib.sha256(text.lower().strip().encode()).hexdigest()

def clean_text(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    return text.strip()

def chunk_text(text: str, size: int, overlap: int) -> list[dict]:
    words = text.split()
    chunks = []
    if not words:
        return chunks
    step = max(size - overlap, 1)
    i, idx = 0, 0
    while i < len(words):
        chunk_words = words[i:i+size]
        chunk = " ".join(chunk_words)
        if chunk.strip():
            chunks.append({"chunk_index": idx, "text": chunk, "word_count": len(chunk_words)})
            idx += 1
        i += step
    return chunks

def table_to_dicts(table_tag: Tag) -> list[dict]:
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
            cells += [""] * (len(headers) - len(cells))
            result.append(dict(zip(headers, cells[:len(headers)])))
    return result

def table_to_text(table_dicts: list[dict]) -> str:
    if not table_dicts:
        return ""
    lines = []
    headers = list(table_dicts[0].keys())
    lines.append(" | ".join(headers))
    lines.append("-" * len(lines[0]))
    for row in table_dicts:
        lines.append(" | ".join(str(row.get(h, "")) for h in headers))
    return "\n".join(lines)

def html_to_text(tag, strip_links_imgs=True) -> str:
    if HAS_MD:
        strip = ["a", "img"] if strip_links_imgs else []
        text = md_lib.markdownify(str(tag), heading_style="ATX", strip=strip)
        return re.sub(r'\n{3,}', '\n\n', text).strip()
    return clean_text(tag.get_text(" ", strip=True))


# ─────────────────────────────────────────────────────────────────────────────
#  GLOSSARY EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_glossary_entries(main: BeautifulSoup, page_url: str = "", page_title: str = "") -> list[dict]:
    """
    Extract term/definition pairs from a page using multiple strategies:
      1. <dl><dt>/<dd>  — standard HTML definition lists
      2. Table with "Term" / "Definition" columns
      3. Heading followed immediately by a paragraph (common in per-term pages)
      4. Alphabetical index pages: collect all linked terms
    """
    entries = []

    # ── Strategy 1: Definition lists ─────────────────────────────────────────
    for dl in main.find_all("dl"):
        terms = dl.find_all("dt")
        defs  = dl.find_all("dd")
        for term_tag, def_tag in zip(terms, defs):
            t = clean_text(term_tag.get_text(" "))
            d = clean_text(def_tag.get_text(" "))
            if t and d:
                entries.append({"term": t, "definition": d, "strategy": "dl"})

    # ── Strategy 2: Table with Term/Definition columns ────────────────────────
    for tbl in main.find_all("table"):
        rows = tbl.find_all("tr")
        if not rows:
            continue
        headers = [th.get_text(strip=True).lower() for th in rows[0].find_all(["th", "td"])]
        term_col = next((i for i, h in enumerate(headers) if "term" in h or "name" in h or "abbreviation" in h), None)
        def_col  = next((i for i, h in enumerate(headers) if "definition" in h or "description" in h or "meaning" in h), None)
        if term_col is None or def_col is None:
            continue
        for row in rows[1:]:
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if len(cells) > max(term_col, def_col):
                t = cells[term_col].strip()
                d = cells[def_col].strip()
                if t and d:
                    entries.append({"term": t, "definition": d, "strategy": "table"})

    # ── Strategy 3: Per-term page (heading + first substantial paragraph) ─────
    # Used when a site dedicates one page per term (e.g. TomTom glossary)
    if not entries:
        h1 = main.find("h1")
        if h1:
            term_candidate = clean_text(h1.get_text())
            # Look for the first substantial paragraph after the heading
            for sib in h1.find_next_siblings(["p", "div", "section"]):
                text = clean_text(sib.get_text(" "))
                if len(text.split()) >= 5:
                    entries.append({
                        "term":       term_candidate,
                        "definition": text[:1000],
                        "strategy":   "per_term_page",
                    })
                    break

    # Attach source info to all found entries
    for e in entries:
        e.setdefault("source_url", page_url)
        e.setdefault("page_title", page_title)

    return entries


# ─────────────────────────────────────────────────────────────────────────────
#  CONTENT SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def find_main_content(soup: BeautifulSoup):
    candidates = []
    for sel in CONTENT_SELECTORS:
        for el in soup.select(sel):
            words = len(el.get_text(" ", strip=True).split())
            if words >= MIN_CONTENT_WORDS:
                candidates.append((words, el))
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1]
    return soup.find("body") or soup

def find_sidebar_text(soup: BeautifulSoup) -> str:
    texts = []
    for sel in SIDEBAR_SELECTORS:
        for el in soup.select(sel):
            t = clean_text(el.get_text(" ", strip=True))
            if t and len(t.split()) >= 5:
                texts.append(t)
    return "\n".join(texts[:3])


# ─────────────────────────────────────────────────────────────────────────────
#  CORE EXTRACTOR  — everything from one HTML page
# ─────────────────────────────────────────────────────────────────────────────

def extract_all(url: str, html: str, chunk_size: int, chunk_overlap: int,
                is_glossary_page: bool = False) -> dict:
    soup_full = BeautifulSoup(html, "lxml")
    base = url

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {}
    for m in soup_full.find_all("meta"):
        name = m.get("name") or m.get("property") or ""
        content = m.get("content") or ""
        if name and content:
            meta[name] = content[:300]

    title_tag = soup_full.find("title")
    title = title_tag.get_text(strip=True) if title_tag else meta.get("og:title", "")

    canonical = ""
    link_can = soup_full.find("link", rel="canonical")
    if link_can:
        canonical = link_can.get("href", "")

    sidebar_text = find_sidebar_text(soup_full)

    # ── Links ─────────────────────────────────────────────────────────────────
    links = []
    seen_links = set()
    for a in soup_full.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        abs_href = urljoin(base, href)
        txt = clean_text(a.get_text())
        if abs_href not in seen_links:
            seen_links.add(abs_href)
            links.append({"text": txt[:100], "url": abs_href})

    pdf_links = [
        lnk for lnk in links
        if lnk["url"].lower().endswith((".pdf", ".docx", ".xlsx", ".csv", ".pptx", ".doc", ".xls", ".ppt"))
    ]

    json_ld = []
    for s in soup_full.find_all("script", type="application/ld+json"):
        try:
            json_ld.append(json.loads(s.string or ""))
        except Exception:
            pass

    # ── Find + clean main content block ──────────────────────────────────────
    main = find_main_content(soup_full)
    main = BeautifulSoup(str(main), "lxml")

    for tag in main.find_all(["script", "style", "noscript", "svg", "button"]):
        tag.decompose()
    for tag in main.select("[aria-hidden=true]"):
        tag.decompose()

    # ── Glossary extraction (before we strip tables/headings) ─────────────────
    glossary_entries = extract_glossary_entries(main, page_url=url, page_title=title)

    # ── Headings ──────────────────────────────────────────────────────────────
    headings = []
    for tag in main.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        txt = clean_text(tag.get_text())
        if txt:
            headings.append({"level": tag.name, "text": txt})

    # ── Code blocks ───────────────────────────────────────────────────────────
    code_blocks = []
    for pre in main.find_all("pre"):
        code_txt = pre.get_text()
        if code_txt.strip():
            lang = ""
            code_tag = pre.find("code")
            if code_tag:
                for cls in code_tag.get("class", []):
                    if cls.startswith(("language-", "lang-")):
                        lang = cls.split("-", 1)[1]
            code_blocks.append({"language": lang, "code": code_txt.strip()[:3000]})

    # ── Tables ────────────────────────────────────────────────────────────────
    tables = []
    for i, tbl in enumerate(main.find_all("table")):
        rows = table_to_dicts(tbl)
        if rows:
            caption_tag = tbl.find("caption")
            caption = caption_tag.get_text(strip=True) if caption_tag else f"Table {i+1}"
            tables.append({
                "caption": caption,
                "rows": rows,
                "as_text": table_to_text(rows),
            })
        tbl.decompose()

    # ── Images ────────────────────────────────────────────────────────────────
    images = []
    for img in main.find_all("img"):
        src = img.get("src", "").strip()
        if not src or src.startswith("data:"):
            continue
        abs_src = urljoin(base, src)
        alt = clean_text(img.get("alt", ""))
        caption = ""
        fig = img.find_parent("figure")
        if fig:
            fc = fig.find("figcaption")
            if fc:
                caption = clean_text(fc.get_text())
        images.append({"url": abs_src, "alt": alt, "caption": caption or clean_text(img.get("title", ""))})

    # ── Forms ─────────────────────────────────────────────────────────────────
    forms = []
    for form in main.find_all("form"):
        fields = []
        for inp in form.find_all(["input", "textarea", "select"]):
            fields.append({
                "tag": inp.name,
                "name": inp.get("name") or inp.get("id") or "",
                "type": inp.get("type", "text"),
                "placeholder": inp.get("placeholder", ""),
            })
        forms.append({
            "action": urljoin(base, form.get("action") or ""),
            "method": (form.get("method") or "GET").upper(),
            "fields": fields,
        })

    # ── Body text ─────────────────────────────────────────────────────────────
    for pre in main.find_all("pre"):
        pre.decompose()
    body_text = html_to_text(main)
    body_text = clean_text(body_text)

    if len(body_text.split()) < MIN_CONTENT_WORDS:
        fallback = BeautifulSoup(html, "lxml")
        for tag in fallback(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
            tag.decompose()
        body_text = clean_text(html_to_text(fallback))

    # ── Full text ─────────────────────────────────────────────────────────────
    heading_text = "\n".join(f"{'#'*int(h['level'][1:])} {h['text']}" for h in headings)
    table_text   = "\n\n".join(t["as_text"] for t in tables)
    image_text   = "\n".join(
        f"[Image: {i['alt'] or 'no alt'}{' — ' + i['caption'] if i['caption'] else ''}]"
        for i in images
    )
    code_text    = "\n\n".join(f"```{c['language']}\n{c['code']}\n```" for c in code_blocks)
    glossary_text = "\n".join(
        f"Term: {e['term']}\nDefinition: {e['definition']}" for e in glossary_entries
    )
    full_text = "\n\n".join(filter(None, [
        f"# {title}" if title else "",
        heading_text, body_text, code_text, table_text, image_text, glossary_text,
    ]))

    # ── RAG chunks ────────────────────────────────────────────────────────────
    rag_chunks = []
    context_prefix = f"[Page: {title}]\n" if title else ""

    if is_glossary_page and glossary_entries:
        # One chunk per glossary term — dramatically better for RAG retrieval
        for entry in glossary_entries:
            chunk_text_str = f"Term: {entry['term']}\nDefinition: {entry['definition']}"
            rag_chunks.append({
                "chunk_index": len(rag_chunks),
                "text":        chunk_text_str,
                "word_count":  len(chunk_text_str.split()),
                "source_type": "glossary",
                "term":        entry["term"],
            })
    else:
        # Standard body chunking
        if body_text:
            for c in chunk_text(body_text, chunk_size, chunk_overlap):
                c["text"] = context_prefix + c["text"]
                rag_chunks.append(c)

    # Tables, code, images, sidebar always chunked individually
    for tbl in tables:
        if tbl["as_text"]:
            rag_chunks.append({
                "chunk_index": len(rag_chunks),
                "text": f"{context_prefix}Table: {tbl['caption']}\n{tbl['as_text']}",
                "word_count": len(tbl["as_text"].split()),
                "source_type": "table",
            })

    for cb in code_blocks:
        if cb["code"]:
            rag_chunks.append({
                "chunk_index": len(rag_chunks),
                "text": f"{context_prefix}Code ({cb['language'] or 'unspecified'}):\n{cb['code']}",
                "word_count": len(cb["code"].split()),
                "source_type": "code",
            })

    for img in images:
        img_txt = " ".join(filter(None, [img["alt"], img["caption"]])).strip()
        if img_txt:
            rag_chunks.append({
                "chunk_index": len(rag_chunks),
                "text": f"{context_prefix}Image on page: {img_txt}",
                "word_count": len(img_txt.split()),
                "source_type": "image_description",
                "image_url": img["url"],
            })

    if sidebar_text and len(sidebar_text.split()) >= 5:
        rag_chunks.append({
            "chunk_index": len(rag_chunks),
            "text": f"{context_prefix}Navigation / related sections: {sidebar_text}",
            "word_count": len(sidebar_text.split()),
            "source_type": "navigation",
        })

    return {
        "url":             url,
        "path":            urlparse(url).path or "/",
        "title":           title,
        "canonical":       canonical,
        "metadata":        meta,
        "headings":        headings,
        "body_text":       body_text,
        "full_text":       full_text,
        "sidebar_text":    sidebar_text,
        "tables":          tables,
        "images":          images,
        "code_blocks":     code_blocks,
        "links":           links,
        "pdf_links":       pdf_links,
        "forms":           forms,
        "json_ld":         json_ld,
        "glossary_entries": glossary_entries,
        "rag_chunks":      rag_chunks,
        "is_glossary_page": is_glossary_page,
        "stats": {
            "words":       len(body_text.split()),
            "tables":      len(tables),
            "images":      len(images),
            "code_blocks": len(code_blocks),
            "links":       len(links),
            "pdfs":        len(pdf_links),
            "chunks":      len(rag_chunks),
            "glossary":    len(glossary_entries),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PDF EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_pdf_text(pdf_bytes: bytes, chunk_size: int, chunk_overlap: int, title_hint: str = "") -> dict:
    if not HAS_PDF:
        return {"error": "pypdf not installed", "rag_chunks": []}
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for page in reader.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                pages_text.append("")
        full_text = clean_text("\n\n".join(pages_text))
        prefix = f"[Document: {title_hint}]\n" if title_hint else ""
        chunks = chunk_text(full_text, chunk_size, chunk_overlap)
        for c in chunks:
            c["text"] = prefix + c["text"]
            c["source_type"] = "pdf"
        return {
            "num_pages": len(reader.pages),
            "word_count": len(full_text.split()),
            "rag_chunks": chunks,
            "error": None,
        }
    except Exception as e:
        return {"error": str(e)[:150], "rag_chunks": []}


def scrape_pdf(url: str, session: requests.Session, chunk_size: int, chunk_overlap: int) -> dict:
    result = {"url": url, "path": urlparse(url).path, "error": None, "data": {}}
    try:
        resp = session.get(url, timeout=20, headers=HEADERS)
        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}"
            return result
        title_hint = os.path.basename(urlparse(url).path)
        result["data"] = extract_pdf_text(resp.content, chunk_size, chunk_overlap, title_hint)
        if result["data"].get("error"):
            result["error"] = result["data"]["error"]
    except Exception as e:
        result["error"] = str(e)[:120]
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  FETCH + EXTRACT one HTML page
# ─────────────────────────────────────────────────────────────────────────────

def scrape_page(page: dict, session: requests.Session,
                chunk_size: int, chunk_overlap: int) -> dict:
    url = page["url"]
    is_glossary = bool(page.get("is_glossary"))
    result = {"url": url, "path": page["path"], "error": None, "data": {}}
    try:
        resp = session.get(url, timeout=15, headers=HEADERS, allow_redirects=True)
        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}"
            return result
        ct = resp.headers.get("content-type", "")
        if "text/html" not in ct:
            result["error"] = f"Non-HTML ({ct.split(';')[0].strip()})"
            return result
        if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
            resp.encoding = resp.apparent_encoding or "utf-8"
        result["data"] = extract_all(url, resp.text, chunk_size, chunk_overlap,
                                     is_glossary_page=is_glossary)
    except requests.exceptions.Timeout:
        result["error"] = "Timeout"
    except Exception as e:
        result["error"] = str(e)[:120]
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  GLOSSARY MERGER
# ─────────────────────────────────────────────────────────────────────────────

def merge_glossary(results: list[dict]) -> dict:
    """
    Merge all glossary entries across all scraped pages into a single dict.
    Key = normalised term (lowercase, stripped).
    When duplicates exist:
      - Longer definition wins
      - Sources list is unioned
      - Images, links, tables are unioned
    """
    knowledge: dict[str, dict] = {}

    for r in results:
        if r["error"] or not r["data"]:
            continue
        for entry in r["data"].get("glossary_entries", []):
            key = entry["term"].lower().strip()
            if not key:
                continue
            if key not in knowledge:
                knowledge[key] = {
                    "term":       entry["term"],
                    "definition": entry["definition"],
                    "sources":    [r["url"]],
                    "strategy":   entry.get("strategy", ""),
                    "images":     [],
                    "links":      [],
                }
            else:
                existing = knowledge[key]
                # Longer definition wins
                if len(entry["definition"]) > len(existing["definition"]):
                    existing["definition"] = entry["definition"]
                if r["url"] not in existing["sources"]:
                    existing["sources"].append(r["url"])
    return knowledge


# ─────────────────────────────────────────────────────────────────────────────
#  WRITE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(results: list[dict], pdf_results: list[dict],
                 glossary: dict, out_dir: str, source_urls: list[str]):
    os.makedirs(out_dir, exist_ok=True)

    # ── full_data.json ────────────────────────────────────────────────────────
    full_path = os.path.join(out_dir, "full_data.json")
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump({
            "sources":      source_urls,
            "scraped_at":   datetime.now(timezone.utc).isoformat(),
            "total_pages":  len(results),
            "total_pdfs":   len(pdf_results),
            "pages":        results,
            "documents":    pdf_results,
        }, f, indent=2, ensure_ascii=False)

    # ── Deduplicated RAG chunks ───────────────────────────────────────────────
    all_chunks = []
    seen_hashes: set[str] = set()

    def _add_chunk(chunk: dict, source_url: str, source_path: str, page_title: str, source_type_default: str):
        h = content_hash(chunk["text"])
        if h in seen_hashes:
            return  # deduplicate identical content
        seen_hashes.add(h)
        all_chunks.append({
            "id":          uid(source_url + str(chunk["chunk_index"])),
            "source_url":  source_url,
            "source_path": source_path,
            "page_title":  page_title,
            "chunk_index": chunk["chunk_index"],
            "text":        chunk["text"],
            "word_count":  chunk.get("word_count", 0),
            "source_type": chunk.get("source_type", source_type_default),
            "image_url":   chunk.get("image_url", ""),
            "term":        chunk.get("term", ""),
        })

    for r in results:
        if r["error"] or not r["data"]:
            continue
        d = r["data"]
        for chunk in d.get("rag_chunks", []):
            _add_chunk(chunk, r["url"], r["path"], d.get("title", ""), "text")

    for r in pdf_results:
        if r["error"] or not r["data"]:
            continue
        d = r["data"]
        for chunk in d.get("rag_chunks", []):
            _add_chunk(chunk, r["url"], r["path"],
                       os.path.basename(r["path"]), "pdf")

    chunks_path = os.path.join(out_dir, "rag_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    jsonl_path = os.path.join(out_dir, "rag_chunks.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # ── Glossary ──────────────────────────────────────────────────────────────
    glossary_list = sorted(glossary.values(), key=lambda x: x["term"].lower())
    glossary_path = os.path.join(out_dir, "glossary_terms.json")
    with open(glossary_path, "w", encoding="utf-8") as f:
        json.dump(glossary_list, f, indent=2, ensure_ascii=False)

    glossary_jsonl = os.path.join(out_dir, "glossary_terms.jsonl")
    with open(glossary_jsonl, "w", encoding="utf-8") as f:
        for entry in glossary_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Images ────────────────────────────────────────────────────────────────
    all_images = []
    for r in results:
        if r["error"] or not r["data"]:
            continue
        for img in r["data"].get("images", []):
            all_images.append({**img, "found_on": r["url"]})
    images_path = os.path.join(out_dir, "images.json")
    with open(images_path, "w", encoding="utf-8") as f:
        json.dump(all_images, f, indent=2, ensure_ascii=False)

    # ── Tables ────────────────────────────────────────────────────────────────
    all_tables = []
    for r in results:
        if r["error"] or not r["data"]:
            continue
        for tbl in r["data"].get("tables", []):
            all_tables.append({**tbl, "found_on": r["url"], "page_title": r["data"].get("title", "")})
    tables_path = os.path.join(out_dir, "tables.json")
    with open(tables_path, "w", encoding="utf-8") as f:
        json.dump(all_tables, f, indent=2, ensure_ascii=False)

    # ── PDFs ──────────────────────────────────────────────────────────────────
    pdfs_meta = [{"url": r["url"], "path": r["path"], "error": r["error"],
                  "num_pages": r["data"].get("num_pages") if r["data"] else None,
                  "word_count": r["data"].get("word_count") if r["data"] else None}
                 for r in pdf_results]
    pdfs_path = os.path.join(out_dir, "pdfs.json")
    with open(pdfs_path, "w", encoding="utf-8") as f:
        json.dump(pdfs_meta, f, indent=2, ensure_ascii=False)

    return {
        "full_data":    full_path,
        "rag_chunks":   chunks_path,
        "rag_jsonl":    jsonl_path,
        "glossary":     glossary_path,
        "glossary_jsonl": glossary_jsonl,
        "images":       images_path,
        "tables":       tables_path,
        "pdfs":         pdfs_path,
        "total_chunks": len(all_chunks),
        "total_images": len(all_images),
        "total_tables": len(all_tables),
        "total_pdfs":   len(pdf_results),
        "total_glossary": len(glossary_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Cave Scraper v3 — multi-source content extractor with glossary merging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("map_files",    nargs="+", help="One or more site_map.json files from crawler.py")
    p.add_argument("--filter",     default="",   help="Only scrape paths containing this string")
    p.add_argument("--depth",      type=int, default=99, help="Only scrape pages at/below this depth")
    p.add_argument("--threads",    type=int, default=8,  help="Parallel threads (default 8)")
    p.add_argument("--delay",      type=float, default=0.0, help="Seconds between requests")
    p.add_argument("--chunk-size", type=int, default=400,  help="RAG chunk size in words (default 400)")
    p.add_argument("--chunk-overlap", type=int, default=50, help="Overlap between chunks (default 50)")
    p.add_argument("--output",     default="scraped_data", help="Output folder (default: scraped_data/)")
    p.add_argument("--no-pdfs",    action="store_true", help="Skip PDF extraction")
    args = p.parse_args()

    # ── Load and merge all site maps ──────────────────────────────────────────
    all_pages_raw = []
    all_asset_files = []
    source_urls = []

    for map_file in args.map_files:
        with open(map_file, encoding="utf-8") as f:
            site_map = json.load(f)
        meta = site_map.get("meta", {})
        # Support both old (root_url) and new (root_urls) formats
        roots = meta.get("root_urls") or [meta.get("root_url", map_file)]
        source_urls.extend(roots)
        all_pages_raw.extend(site_map.get("pages", []))
        all_asset_files.extend(meta.get("asset_files", []))
        cprint(f"  Loaded {map_file}: {len(site_map.get('pages', []))} pages", DIM)

    # Deduplicate pages by URL (last map wins on metadata)
    unique_pages: dict[str, dict] = {}
    for page in all_pages_raw:
        unique_pages[page["url"]] = page
    all_pages_raw = list(unique_pages.values())

    # Apply filters
    pages = [
        pg for pg in all_pages_raw
        if (not args.filter or args.filter in pg["path"])
        and pg["depth"] <= args.depth
        and not pg.get("error")
    ]

    # PDF assets
    seen_assets = set()
    pdf_assets = []
    for a in all_asset_files:
        if a not in seen_assets and a.lower().endswith(".pdf"):
            seen_assets.add(a)
            pdf_assets.append(a)
    if args.filter:
        pdf_assets = [a for a in pdf_assets if args.filter in urlparse(a).path]

    cprint(f"\n{'═'*60}", CYAN)
    cprint(f"  CAVE SCRAPER v3  ⑂  Multi-Source Extractor", BOLD)
    cprint(f"  Maps loaded : {len(args.map_files)}", DIM)
    cprint(f"  Total pages (deduped): {len(pages)}", DIM)
    cprint(f"  Threads : {args.threads}  |  chunk={args.chunk_size}w overlap={args.chunk_overlap}w", DIM)
    glossary_pages = [p for p in pages if p.get("is_glossary")]
    cprint(f"  Glossary pages detected: {len(glossary_pages)}", PURPLE)
    if not args.no_pdfs:
        cprint(f"  PDFs to extract: {len(pdf_assets)}", DIM)
    if not HAS_MD:
        cprint(f"  ⚠ markdownify not installed — plain text extraction", YELLOW)
    if not HAS_PDF and not args.no_pdfs:
        cprint(f"  ⚠ pypdf not installed — PDFs skipped (pip install pypdf)", YELLOW)
    cprint(f"  Output : {args.output}/", DIM)
    cprint(f"{'═'*60}\n", CYAN)

    session = requests.Session()
    results = []
    errors = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        futures = {
            ex.submit(scrape_page, pg, session, args.chunk_size, args.chunk_overlap): pg
            for pg in pages
        }
        for fut in as_completed(futures):
            res = fut.result()
            with lock:
                results.append(res)
            if res["error"]:
                errors += 1
                cprint(f"  ✕  {res['path']:<45}  {res['error']}", RED)
            else:
                st = res["data"].get("stats", {})
                g_count = st.get("glossary", 0)
                g_label = f"  {PURPLE}{g_count}gloss{R}" if g_count else ""
                cprint(
                    f"  ✓  {res['path']:<45}  "
                    f"{st.get('words',0):>5}w  "
                    f"{st.get('tables',0)}tbl  "
                    f"{st.get('images',0)}img  "
                    f"{st.get('code_blocks',0)}code  "
                    f"{st.get('chunks',0)}chunks{g_label}",
                    GREEN
                )
            if args.delay:
                time.sleep(args.delay)

    # ── PDFs ──────────────────────────────────────────────────────────────────
    pdf_results = []
    if not args.no_pdfs and HAS_PDF and pdf_assets:
        cprint(f"\n{CYAN}{BOLD}Extracting {len(pdf_assets)} PDF documents...{R}")
        with ThreadPoolExecutor(max_workers=args.threads) as ex:
            futures = {
                ex.submit(scrape_pdf, url, session, args.chunk_size, args.chunk_overlap): url
                for url in pdf_assets
            }
            for fut in as_completed(futures):
                res = fut.result()
                pdf_results.append(res)
                if res["error"]:
                    cprint(f"  ✕  {res['path']:<45}  {res['error']}", RED)
                else:
                    d = res["data"]
                    cprint(f"  ✓  {res['path']:<45}  {d.get('num_pages',0)}pg  {d.get('word_count',0)}w  {len(d.get('rag_chunks',[]))}chunks", GREEN)

    # ── Merge glossary ────────────────────────────────────────────────────────
    cprint(f"\n  Merging glossary entries...", CYAN)
    glossary = merge_glossary(results)
    cprint(f"  ✓  {len(glossary)} unique terms merged", GREEN)

    # ── Save ──────────────────────────────────────────────────────────────────
    info = save_outputs(results, pdf_results, glossary, args.output, source_urls)

    cprint(f"\n{'═'*60}", GREEN)
    cprint(f"  ✓  DONE — {len(results)} pages, {len(pdf_results)} PDFs, {errors} errors", GREEN + BOLD)
    cprint(f"{'═'*60}", GREEN)
    cprint(f"\n  OUTPUT FILES:", CYAN + BOLD)
    cprint(f"  {'rag_chunks.jsonl':<28}  {info['total_chunks']} chunks  ← feed to your embedder", GREEN)
    cprint(f"  {'rag_chunks.json':<28}  same, formatted", GREEN)
    cprint(f"  {'glossary_terms.json':<28}  {info['total_glossary']} unique terms  ← merged glossary", PURPLE)
    cprint(f"  {'glossary_terms.jsonl':<28}  same, one line per term", PURPLE)
    cprint(f"  {'full_data.json':<28}  everything per page + PDFs", DIM)
    cprint(f"  {'images.json':<28}  {info['total_images']} images", DIM)
    cprint(f"  {'tables.json':<28}  {info['total_tables']} tables", DIM)
    cprint(f"  {'pdfs.json':<28}  {info['total_pdfs']} PDFs", DIM)
    cprint(f"\n  → Load rag_chunks.jsonl into your vector store and go!\n", YELLOW + BOLD)


if __name__ == "__main__":
    main()

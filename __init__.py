# core/__init__.py


#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║         CAVE EXPLORER v3  — Multi-Site Mapper            ║
║  Splits into bots at every junction, explores all paths  ║
║  across ONE OR MORE root URLs, merges everything into a  ║
║  single unified site_map.json ready for scraping.        ║
╚══════════════════════════════════════════════════════════╝

Usage:
    python crawler.py https://example.com
    python crawler.py https://docs.site.com https://glossary.site.com
    python crawler.py https://a.com https://b.com --depth 3 --max-pages 2000 --threads 10
    python crawler.py https://a.com --output my_site_map.json
"""

import sys
import json
import time
import argparse
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse, urldefrag
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

RESET  = "\033[0m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
PURPLE = "\033[95m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

BOT_COLORS = [CYAN, PURPLE, GREEN, YELLOW]

def cprint(msg, color=RESET, end="\n"):
    print(f"{color}{msg}{RESET}", end=end, flush=True)


# ─── Files we should record but NOT try to parse as HTML ──────────────────────
ASSET_EXTENSIONS = (
    ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".csv", ".pptx", ".ppt",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".bmp",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".mp4", ".mp3", ".wav", ".avi", ".mov",
    ".woff", ".woff2", ".ttf", ".eot",
    ".css", ".js", ".json", ".xml",
)

SKIP_PATH_FRAGMENTS = (
    "/_static/", "/_images/", "/_sources/", "/_downloads/",
    "/assets/", "/static/", "/cdn-cgi/",
)

# Patterns that suggest a page is a glossary
GLOSSARY_PATTERNS = (
    "/glossary", "glossary/", "/terms", "/abbreviations", "/definitions",
)


# ─── Thread-safe visited set (shared across all root explorers) ────────────────
class VisitedSet:
    def __init__(self):
        self._set = set()
        self._lock = threading.Lock()

    def add_if_new(self, url):
        with self._lock:
            if url in self._set:
                return False
            self._set.add(url)
            return True

    def peek(self, url):
        with self._lock:
            return url in self._set

    def __len__(self):
        with self._lock:
            return len(self._set)


# ─── Page node ────────────────────────────────────────────────────────────────
class PageNode:
    def __init__(self, url, depth, source_label=""):
        self.url          = url
        self.depth        = depth
        self.path         = urlparse(url).path or "/"
        self.source_label = source_label   # which root URL this came from
        self.title        = ""
        self.status       = None
        self.content_type = ""
        self.links_found  = []
        self.asset_links  = []
        self.forms        = []
        self.meta         = {}
        self.headings     = []
        self.images       = []
        self.scripts      = []
        self.error        = None
        self.children     = []
        self.is_glossary  = False


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 CaveExplorerBot/3.0"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def is_asset_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(ASSET_EXTENSIONS)


def should_skip(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(frag in path for frag in SKIP_PATH_FRAGMENTS)


def is_glossary_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(p in path for p in GLOSSARY_PATTERNS)


def fetch_and_parse(node: PageNode, base_domains: set, session: requests.Session):
    """Fetch a page and extract metadata, links, assets."""
    try:
        resp = session.get(node.url, timeout=15, allow_redirects=True,
                           headers=HEADERS, stream=False)
        node.status = resp.status_code
        node.content_type = resp.headers.get("content-type", "")

        if resp.status_code != 200:
            node.error = f"HTTP {resp.status_code}"
            return

        if "text/html" not in node.content_type:
            node.error = f"Non-HTML: {node.content_type.split(';')[0].strip()}"
            return

        if not resp.encoding or resp.encoding.lower() == "iso-8859-1":
            resp.encoding = resp.apparent_encoding or "utf-8"

        soup = BeautifulSoup(resp.text, "lxml")

        t = soup.find("title")
        node.title = t.get_text(strip=True)[:120] if t else ""

        for m in soup.find_all("meta"):
            name = m.get("name") or m.get("property") or ""
            content = m.get("content") or ""
            if name and content:
                node.meta[name] = content[:200]

        # Mark as glossary if URL or meta suggest it
        if is_glossary_url(node.url):
            node.is_glossary = True
            node.meta["is_glossary"] = "true"

        for tag in soup.find_all(["h1", "h2", "h3"]):
            txt = tag.get_text(strip=True)
            if txt:
                node.headings.append({"level": tag.name, "text": txt})

        # ── Links: separate page-links vs asset-links ───────────────────────
        seen_links, seen_assets = set(), set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            abs_url, _ = urldefrag(urljoin(node.url, href))
            parsed = urlparse(abs_url)
            if parsed.scheme not in ("http", "https"):
                continue
            # Accept links from ANY of our tracked domains (multi-root support)
            if parsed.netloc not in base_domains:
                continue
            clean = parsed._replace(fragment="").geturl()

            if is_asset_url(clean):
                if clean not in seen_assets:
                    seen_assets.add(clean)
                    node.asset_links.append(clean)
                continue

            if should_skip(clean):
                continue

            if clean not in seen_links:
                seen_links.add(clean)
                node.links_found.append(clean)

        # ── <link rel="next/prev"> pagination ──────────────────────────────
        for ln in soup.find_all("link", href=True):
            rel = " ".join(ln.get("rel", []))
            if rel in ("next", "prev"):
                abs_url, _ = urldefrag(urljoin(node.url, ln["href"]))
                parsed = urlparse(abs_url)
                if parsed.netloc in base_domains:
                    clean = parsed._replace(fragment="").geturl()
                    if clean not in seen_links and not is_asset_url(clean):
                        seen_links.add(clean)
                        node.links_found.append(clean)

        for form in soup.find_all("form"):
            fields = []
            for inp in form.find_all(["input", "textarea", "select"]):
                fields.append({
                    "tag":  inp.name,
                    "name": inp.get("name") or inp.get("id") or "",
                    "type": inp.get("type") or "text",
                })
            node.forms.append({
                "action": urljoin(node.url, form.get("action") or ""),
                "method": (form.get("method") or "GET").upper(),
                "fields": fields,
            })

        for img in soup.find_all("img", src=True):
            node.images.append(urljoin(node.url, img["src"]))

        for s in soup.find_all("script", src=True):
            node.scripts.append(urljoin(node.url, s["src"]))

    except requests.exceptions.Timeout:
        node.error = "Timeout"
    except requests.exceptions.ConnectionError as e:
        node.error = f"Connection error: {str(e)[:80]}"
    except Exception as e:
        node.error = f"Error: {str(e)[:80]}"


class CaveExplorer:
    """
    Multi-root crawler. Pass multiple root URLs and they will be crawled
    concurrently, sharing a single visited set so there's no duplication.
    """

    def __init__(self, root_urls: list[str], max_depth=2, max_pages=1000, max_threads=10):
        self.root_urls   = []
        self.base_domains = set()
        for url in root_urls:
            parsed = urlparse(url)
            clean  = parsed._replace(fragment="").geturl()
            self.root_urls.append(clean)
            self.base_domains.add(parsed.netloc)

        self.max_depth   = max_depth
        self.max_pages   = max_pages
        self.max_threads = max_threads
        self.visited     = VisitedSet()
        self.session     = requests.Session()
        self._lock       = threading.Lock()
        self._bot_id     = 0
        self.stats       = defaultdict(int)
        self.all_assets  = set()
        self.all_assets_lock = threading.Lock()

    def _new_bot_id(self):
        with self._lock:
            self._bot_id += 1
            return self._bot_id

    def _log_split(self, bot_id, url, n_children, depth):
        color = BOT_COLORS[depth % len(BOT_COLORS)]
        indent = "  " * depth
        cprint(f"{indent}⑂  BOT-{bot_id:03d}  SPLIT → {n_children} child bots  [{urlparse(url).path or '/'}]", color)

    def _log_scan(self, bot_id, url, depth):
        indent = "  " * depth
        cprint(f"{indent}◉  BOT-{bot_id:03d}  scanning  {urlparse(url).path or '/'}", DIM)

    def _log_merge(self, bot_id, url, n, depth):
        indent = "  " * depth
        cprint(f"{indent}⇒  BOT-{bot_id:03d}  MERGE ← {n} children  [{urlparse(url).path or '/'}]", GREEN)

    def _log_error(self, bot_id, url, err, depth):
        indent = "  " * depth
        cprint(f"{indent}✕  BOT-{bot_id:03d}  {urlparse(url).path or '/'}  — {err}", RED)

    def _explore(self, url: str, depth: int, source_label: str = "") -> PageNode | None:
        if len(self.visited) >= self.max_pages:
            return None
        if not self.visited.add_if_new(url):
            return None

        bot_id = self._new_bot_id()
        node   = PageNode(url, depth, source_label=source_label)

        self._log_scan(bot_id, url, depth)
        fetch_and_parse(node, self.base_domains, self.session)
        self.stats["pages_fetched"] += 1

        if node.asset_links:
            with self.all_assets_lock:
                for a in node.asset_links:
                    self.all_assets.add(a)

        if node.error:
            self._log_error(bot_id, url, node.error, depth)
            self.stats["errors"] += 1
            return node

        self.stats["links_total"] += len(node.links_found)

        if depth >= self.max_depth:
            return node

        new_links = [lnk for lnk in node.links_found if not self.visited.peek(lnk)]
        if not new_links:
            return node

        self._log_split(bot_id, url, len(new_links), depth)

        with ThreadPoolExecutor(max_workers=min(len(new_links), self.max_threads)) as ex:
            futures = {
                ex.submit(self._explore, lnk, depth + 1, source_label): lnk
                for lnk in new_links
            }
            for fut in as_completed(futures):
                child = fut.result()
                if child is not None:
                    node.children.append(child)

        self._log_merge(bot_id, url, len(node.children), depth)
        return node

    def run(self) -> list[PageNode]:
        cprint(f"\n{'═'*60}", CYAN)
        cprint(f"  CAVE EXPLORER v3  ⑂  Multi-Site Mapper", BOLD)
        for url in self.root_urls:
            cprint(f"  Root: {url}", CYAN)
        cprint(f"  depth={self.max_depth}  max_pages={self.max_pages}  threads={self.max_threads}", DIM)
        cprint(f"{'═'*60}\n", CYAN)

        start = time.time()
        roots = []

        # Crawl all root URLs concurrently (sharing the visited set)
        with ThreadPoolExecutor(max_workers=len(self.root_urls)) as ex:
            futures = {
                ex.submit(self._explore, url, 0, urlparse(url).netloc): url
                for url in self.root_urls
            }
            for fut in as_completed(futures):
                root = fut.result()
                if root:
                    roots.append(root)

        elapsed = time.time() - start

        cprint(f"\n{'═'*60}", GREEN)
        cprint(f"  ✓  EXPLORATION COMPLETE in {elapsed:.1f}s", GREEN + BOLD)
        cprint(f"  Pages visited : {len(self.visited)}", GREEN)
        cprint(f"  Errors        : {self.stats['errors']}", YELLOW if self.stats['errors'] else GREEN)
        cprint(f"  Links found   : {self.stats['links_total']}", GREEN)
        cprint(f"  Asset files   : {len(self.all_assets)}", GREEN)
        cprint(f"  Root trees    : {len(roots)}", GREEN)
        cprint(f"{'═'*60}\n", GREEN)

        return roots


def node_to_dict(node: PageNode) -> dict:
    return {
        "url":          node.url,
        "path":         node.path,
        "depth":        node.depth,
        "source_label": node.source_label,
        "status":       node.status,
        "title":        node.title,
        "content_type": node.content_type,
        "error":        node.error,
        "meta":         node.meta,
        "headings":     node.headings,
        "forms":        node.forms,
        "images":       node.images[:30],
        "scripts":      node.scripts[:20],
        "asset_links":  node.asset_links,
        "links_found":  node.links_found,
        "is_glossary":  node.is_glossary,
        "children":     [node_to_dict(c) for c in node.children],
    }


def flatten(node: PageNode, out=None) -> list[dict]:
    if out is None:
        out = []
    out.append({
        "url":          node.url,
        "path":         node.path,
        "depth":        node.depth,
        "source_label": node.source_label,
        "title":        node.title,
        "status":       node.status,
        "error":        node.error,
        "forms":        node.forms,
        "meta":         node.meta,
        "is_glossary":  node.is_glossary,
    })
    for child in node.children:
        flatten(child, out)
    return out


def flatten_all(roots: list[PageNode]) -> list[dict]:
    """Flatten multiple root trees, deduplicating by URL."""
    all_pages = []
    seen_urls = set()
    for root in roots:
        for page in flatten(root):
            if page["url"] not in seen_urls:
                seen_urls.add(page["url"])
                all_pages.append(page)
    return all_pages


def print_tree(node: PageNode, prefix="", is_last=True, depth_limit=99):
    connector = "└── " if is_last else "├── "
    color     = BOT_COLORS[node.depth % len(BOT_COLORS)]
    status    = f" [{node.status}]" if node.status else ""
    err       = f"  ← {node.error}" if node.error else ""
    title     = f"  \"{node.title}\"" if node.title else ""
    glyph     = " 📖" if node.is_glossary else ""
    cprint(f"{prefix}{connector}{node.path}{status}{title}{err}{glyph}", color)
    if node.depth >= depth_limit:
        return
    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(node.children):
        print_tree(child, child_prefix, i == len(node.children) - 1, depth_limit)


def main():
    parser = argparse.ArgumentParser(
        description="Cave Explorer v3 — multi-site recursive mapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("urls", nargs="+", help="One or more root URLs to explore")
    parser.add_argument("--depth",     type=int, default=3,    help="Max crawl depth (default 3)")
    parser.add_argument("--max-pages", type=int, default=1000, help="Max pages to visit total (default 1000)")
    parser.add_argument("--threads",   type=int, default=10,   help="Parallel bot threads (default 10)")
    parser.add_argument("--output",    default="site_map.json", help="Output JSON file (default site_map.json)")
    args = parser.parse_args()

    explorer = CaveExplorer(
        root_urls   = args.urls,
        max_depth   = args.depth,
        max_pages   = args.max_pages,
        max_threads = args.threads,
    )

    roots = explorer.run()

    cprint("SITE TREE\n", CYAN + BOLD)
    for root in roots:
        cprint(f"  ▶ {root.source_label}", CYAN)
        print_tree(root)
        print()

    all_pages = flatten_all(roots)

    output = {
        "meta": {
            "root_urls":   args.urls,
            "crawled_at":  datetime.now(timezone.utc).isoformat(),
            "depth":       args.depth,
            "pages_total": len(all_pages),
            "asset_files": sorted(explorer.all_assets),
            "glossary_pages": [p["url"] for p in all_pages if p.get("is_glossary")],
        },
        "trees":  [node_to_dict(r) for r in roots],
        "pages":  all_pages,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    glossary_count = sum(1 for p in all_pages if p.get("is_glossary"))
    cprint(f"\n✓  Map saved → {args.output}", GREEN + BOLD)
    cprint(f"  {len(all_pages)} pages  |  {len(explorer.all_assets)} assets  |  {glossary_count} glossary pages detected\n", DIM)

    cprint("QUICK SUMMARY", CYAN + BOLD)
    for url in args.urls:
        domain = urlparse(url).netloc
        domain_pages = [p for p in all_pages if urlparse(p["url"]).netloc == domain]
        cprint(f"  {domain:<40} {len(domain_pages)} pages")
    forms_total = sum(len(p['forms']) for p in all_pages)
    cprint(f"  Forms    : {forms_total}")
    cprint(f"  Assets   : {len(explorer.all_assets)}")
    errors = [p for p in all_pages if p['error']]
    cprint(f"  Errors   : {len(errors)}")
    if errors:
        for e in errors[:5]:
            cprint(f"    {e['path']}  — {e['error']}", YELLOW)


if __name__ == "__main__":
    main()

# core/__init__.py

#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════╗
║         CAVE EXPLORER  — Site Mapper         ║
║  Splits into bots at every junction,         ║
║  explores all paths, merges back, outputs    ║
║  a complete site map ready for scraping.     ║
╚══════════════════════════════════════════════╝

Usage:
    python crawler.py https://example.com
    python crawler.py https://example.com --depth 3 --max-pages 100 --threads 10
    python crawler.py https://example.com --output my_site_map.json
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

# ─── Colours for terminal output ───────────────────────────────────────────
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


# ─── Thread-safe visited set ────────────────────────────────────────────────
class VisitedSet:
    def __init__(self):
        self._set = set()
        self._lock = threading.Lock()

    def add_if_new(self, url):
        """Returns True if url was not yet visited (and adds it)."""
        with self._lock:
            if url in self._set:
                return False
            self._set.add(url)
            return True

    def __len__(self):
        with self._lock:
            return len(self._set)


# ─── Page node ─────────────────────────────────────────────────────────────
class PageNode:
    def __init__(self, url, depth):
        self.url        = url
        self.depth      = depth
        self.path       = urlparse(url).path or "/"
        self.title      = ""
        self.status     = None
        self.content_type = ""
        self.links_found  = []   # absolute URLs on same domain
        self.forms        = []   # list of {action, method, fields}
        self.meta         = {}   # og:title, description, etc.
        self.headings     = []   # h1..h3 text
        self.images       = []   # src of <img>
        self.scripts      = []   # src of <script>
        self.error        = None
        self.children     = []   # child PageNode objects


# ─── Core fetcher / parser ──────────────────────────────────────────────────
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CaveExplorerBot/1.0; "
        "+https://github.com/cave-explorer)"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

def fetch_and_parse(node: PageNode, base_domain: str, session: requests.Session):
    """Fetch a URL and populate node with extracted data."""
    try:
        resp = session.get(node.url, timeout=10, allow_redirects=True,
                           headers=HEADERS, stream=False)
        node.status = resp.status_code
        node.content_type = resp.headers.get("content-type", "")

        if resp.status_code != 200:
            node.error = f"HTTP {resp.status_code}"
            return

        if "text/html" not in node.content_type:
            node.error = f"Non-HTML: {node.content_type.split(';')[0].strip()}"
            return

        soup = BeautifulSoup(resp.text, "lxml")

        # Title
        t = soup.find("title")
        node.title = t.get_text(strip=True)[:120] if t else ""

        # Meta tags
        for m in soup.find_all("meta"):
            name = m.get("name") or m.get("property") or ""
            content = m.get("content") or ""
            if name and content:
                node.meta[name] = content[:200]

        # Headings
        for tag in soup.find_all(["h1", "h2", "h3"]):
            txt = tag.get_text(strip=True)
            if txt:
                node.headings.append({"level": tag.name, "text": txt[:100]})

        # Links
        origin = urlparse(node.url)
        seen_links = set()
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith(("mailto:", "tel:", "javascript:", "#")):
                continue
            abs_url, _ = urldefrag(urljoin(node.url, href))
            parsed = urlparse(abs_url)
            if parsed.scheme not in ("http", "https"):
                continue
            if parsed.netloc != base_domain:
                continue
            clean = parsed._replace(fragment="").geturl()
            if clean not in seen_links:
                seen_links.add(clean)
                node.links_found.append(clean)

        # Forms
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

        # Images
        for img in soup.find_all("img", src=True):
            node.images.append(urljoin(node.url, img["src"]))

        # Scripts
        for s in soup.find_all("script", src=True):
            node.scripts.append(urljoin(node.url, s["src"]))

    except requests.exceptions.Timeout:
        node.error = "Timeout"
    except requests.exceptions.ConnectionError as e:
        node.error = f"Connection error: {str(e)[:80]}"
    except Exception as e:
        node.error = f"Error: {str(e)[:80]}"


# ─── Recursive explorer (the "cave bot") ───────────────────────────────────
class CaveExplorer:
    def __init__(self, root_url, max_depth=2, max_pages=100, max_threads=8):
        parsed = urlparse(root_url)
        self.root_url    = parsed._replace(fragment="").geturl()
        self.base_domain = parsed.netloc
        self.max_depth   = max_depth
        self.max_pages   = max_pages
        self.max_threads = max_threads
        self.visited     = VisitedSet()
        self.session     = requests.Session()
        self._lock       = threading.Lock()
        self._bot_id     = 0
        self.stats       = defaultdict(int)

    def _new_bot_id(self):
        with self._lock:
            self._bot_id += 1
            return self._bot_id

    def _log_split(self, bot_id, url, n_children, depth):
        color = BOT_COLORS[depth % len(BOT_COLORS)]
        indent = "  " * depth
        cprint(f"{indent}⑂  BOT-{bot_id:03d}  SPLIT → {n_children} child bots  [{urlparse(url).path or '/'}]", color)

    def _log_scan(self, bot_id, url, depth):
        color = BOT_COLORS[depth % len(BOT_COLORS)]
        indent = "  " * depth
        cprint(f"{indent}◉  BOT-{bot_id:03d}  scanning  {urlparse(url).path or '/'}", DIM)

    def _log_merge(self, bot_id, url, n, depth):
        color = BOT_COLORS[depth % len(BOT_COLORS)]
        indent = "  " * depth
        cprint(f"{indent}⇒  BOT-{bot_id:03d}  MERGE ← {n} children  [{urlparse(url).path or '/'}]", GREEN)

    def _log_error(self, bot_id, url, err, depth):
        indent = "  " * depth
        cprint(f"{indent}✕  BOT-{bot_id:03d}  {urlparse(url).path or '/'}  — {err}", RED)

    def _explore(self, url: str, depth: int) -> PageNode | None:
        """Recursively explore a URL. Returns a populated PageNode or None."""
        if len(self.visited) >= self.max_pages:
            return None
        if not self.visited.add_if_new(url):
            return None  # already claimed by another bot

        bot_id = self._new_bot_id()
        node   = PageNode(url, depth)

        self._log_scan(bot_id, url, depth)
        fetch_and_parse(node, self.base_domain, self.session)
        self.stats["pages_fetched"] += 1

        if node.error:
            self._log_error(bot_id, url, node.error, depth)
            self.stats["errors"] += 1
            return node

        self.stats["links_total"] += len(node.links_found)

        # Stop recursing at max depth
        if depth >= self.max_depth:
            return node

        # New links to explore
        new_links = [
            lnk for lnk in node.links_found
            if lnk not in self.visited._set   # fast non-locking peek
        ]
        if not new_links:
            return node

        self._log_split(bot_id, url, len(new_links), depth)

        # Spawn child bots in parallel
        with ThreadPoolExecutor(max_workers=min(len(new_links), self.max_threads)) as ex:
            futures = {ex.submit(self._explore, lnk, depth + 1): lnk
                       for lnk in new_links}
            for fut in as_completed(futures):
                child = fut.result()
                if child is not None:
                    node.children.append(child)

        self._log_merge(bot_id, url, len(node.children), depth)
        return node

    def run(self) -> PageNode:
        cprint(f"\n{'═'*60}", CYAN)
        cprint(f"  CAVE EXPLORER  ⑂  {self.root_url}", BOLD)
        cprint(f"  depth={self.max_depth}  max_pages={self.max_pages}  threads={self.max_threads}", DIM)
        cprint(f"{'═'*60}\n", CYAN)

        start = time.time()
        root  = self._explore(self.root_url, 0)
        elapsed = time.time() - start

        cprint(f"\n{'═'*60}", GREEN)
        cprint(f"  ✓  EXPLORATION COMPLETE in {elapsed:.1f}s", GREEN + BOLD)
        cprint(f"  Pages visited : {len(self.visited)}", GREEN)
        cprint(f"  Errors        : {self.stats['errors']}", YELLOW if self.stats['errors'] else GREEN)
        cprint(f"  Links found   : {self.stats['links_total']}", GREEN)
        cprint(f"{'═'*60}\n", GREEN)

        return root


# ─── Serialise tree → dict ──────────────────────────────────────────────────
def node_to_dict(node: PageNode) -> dict:
    return {
        "url":          node.url,
        "path":         node.path,
        "depth":        node.depth,
        "status":       node.status,
        "title":        node.title,
        "content_type": node.content_type,
        "error":        node.error,
        "meta":         node.meta,
        "headings":     node.headings,
        "forms":        node.forms,
        "images":       node.images[:20],     # cap to keep JSON sane
        "scripts":      node.scripts[:20],
        "links_found":  node.links_found,
        "children":     [node_to_dict(c) for c in node.children],
    }


# ─── Flat list of all pages (for scraping) ─────────────────────────────────
def flatten(node: PageNode, out=None) -> list[dict]:
    if out is None:
        out = []
    out.append({
        "url":    node.url,
        "path":   node.path,
        "depth":  node.depth,
        "title":  node.title,
        "status": node.status,
        "error":  node.error,
        "forms":  node.forms,
        "meta":   node.meta,
    })
    for child in node.children:
        flatten(child, out)
    return out


# ─── Pretty tree printer ────────────────────────────────────────────────────
def print_tree(node: PageNode, prefix="", is_last=True, depth_limit=99):
    connector = "└── " if is_last else "├── "
    color     = BOT_COLORS[node.depth % len(BOT_COLORS)]
    status    = f" [{node.status}]" if node.status else ""
    err       = f"  ← {node.error}" if node.error else ""
    title     = f"  \"{node.title}\"" if node.title else ""
    cprint(f"{prefix}{connector}{node.path}{status}{title}{err}", color)

    if node.depth >= depth_limit:
        return

    child_prefix = prefix + ("    " if is_last else "│   ")
    for i, child in enumerate(node.children):
        print_tree(child, child_prefix, i == len(node.children) - 1, depth_limit)


# ─── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Cave Explorer — recursive website mapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("url",         help="Root URL to explore")
    parser.add_argument("--depth",     type=int, default=2,   help="Max crawl depth (default 2)")
    parser.add_argument("--max-pages", type=int, default=100, help="Max pages to visit (default 100)")
    parser.add_argument("--threads",   type=int, default=8,   help="Parallel bot threads (default 8)")
    parser.add_argument("--output",    default="site_map.json", help="Output JSON file (default site_map.json)")
    args = parser.parse_args()

    explorer = CaveExplorer(
        root_url   = args.url,
        max_depth  = args.depth,
        max_pages  = args.max_pages,
        max_threads= args.threads,
    )

    root = explorer.run()

    # ── Print the tree ──
    cprint("SITE TREE\n", CYAN + BOLD)
    print_tree(root)

    # ── Build output ──
    all_pages = flatten(root)
    output = {
        "meta": {
            "root_url":    args.url,
            "crawled_at":  datetime.now(timezone.utc).isoformat(),
            "depth":       args.depth,
            "pages_total": len(all_pages),
        },
        "tree":  node_to_dict(root),
        "pages": all_pages,        # ← flat list, perfect for scraping loops
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    cprint(f"\n✓  Map saved → {args.output}", GREEN + BOLD)
    cprint(f"  {len(all_pages)} pages  |  use output['pages'] to iterate for scraping\n", DIM)

    # ── Quick summary ──
    cprint("QUICK SUMMARY", CYAN + BOLD)
    cprint(f"  Domain   : {urlparse(args.url).netloc}")
    cprint(f"  Pages    : {len(all_pages)}")
    forms_total = sum(len(p['forms']) for p in all_pages)
    cprint(f"  Forms    : {forms_total}")
    errors = [p for p in all_pages if p['error']]
    cprint(f"  Errors   : {len(errors)}")
    if errors:
        for e in errors[:5]:
            cprint(f"    {e['path']}  — {e['error']}", YELLOW)


if __name__ == "__main__":
    main()

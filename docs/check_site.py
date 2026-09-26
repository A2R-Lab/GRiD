#!/usr/bin/env python3
"""Check local page/asset targets, homepage anchors, and legacy redirects."""
import hashlib
import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from build_site import LEGACY_ROUTES


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()
        self.home_links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "data-grid-home" in attrs:
            self.home_links.append(attrs.get("href", ""))
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for key in ("href", "src"):
            if key in attrs:
                self.links.append(attrs[key])


def check(root):
    errors = []
    parsed = {}

    def parse(page):
        if page not in parsed:
            parser = Links()
            parser.feed(page.read_text(encoding="utf-8"))
            parsed[page] = parser
        return parsed[page]

    pages = [p for p in root.rglob("*.html")
             if not any(part.startswith("_") for part in p.relative_to(root).parts)]
    if not (root / "index.html").is_file() or not (root / "docs/index.html").is_file():
        raise SystemExit("Missing cover page or /docs/index.html")
    for page in pages:
        for link in parse(page).home_links:
            target = (page.parent / link).resolve()
            if target != root:
                errors.append(f"Project home must return to cover: {page.relative_to(root)}")
        for link in parse(page).links:
            url = urlsplit(link)
            if url.scheme or url.netloc:
                continue
            target = ((root / unquote(url.path.lstrip("/"))) if url.path.startswith("/")
                      else page.parent / unquote(url.path)) if url.path else page
            target = target.resolve()
            if target.is_dir():
                target /= "index.html"
            if not target.is_relative_to(root) or not target.exists():
                errors.append(f"{page.relative_to(root)}: missing local target {link}")
            elif page == root / "index.html" and url.fragment and target.suffix == ".html":
                if unquote(url.fragment) not in parse(target).ids:
                    errors.append(f"Homepage link has missing anchor: {link}")
    routes = {p.relative_to(root / "docs").as_posix(): p.relative_to(root / "docs").as_posix()
              for p in pages if p.is_relative_to(root / "docs")
              and p != root / "docs/index.html"}
    routes.update(LEGACY_ROUTES)
    for old, new in routes.items():
        legacy = root / old
        if not legacy.is_file():
            errors.append(f"Missing legacy redirect: {old}")
            continue
        expected = (root / "docs" / new).resolve()
        if not any((legacy.parent / unquote(urlsplit(link).path)).resolve() == expected
                   for link in parse(legacy).links):
            errors.append(f"Wrong legacy redirect destination: {old}")
        if "location.search + location.hash" not in legacy.read_text(encoding="utf-8"):
            errors.append(f"Legacy query/fragment preservation missing: {old}")
    if errors:
        raise SystemExit("\n".join(errors))
    assets = root / "docs/_static/release"
    if (assets / "manifest.json").exists():
        manifest = json.loads((assets / "manifest.json").read_text())
        if not manifest.get("approved"):
            raise SystemExit("Release figures are tracked but not approved (docs/plot_release_figures.py --approve)")
        for name, digest in manifest["outputs"].items():
            path = assets / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise SystemExit(f"Release figure asset hash mismatch: {name}")
    print(f"PASS: {len(pages)} pages; local links/assets, homepage anchors, {len(routes)} redirects")


if __name__ == "__main__":
    check(Path(sys.argv[1]).resolve())

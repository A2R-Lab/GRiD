#!/usr/bin/env python3
"""Assemble the GRiD cover, /docs reference, and old documentation redirects.

No GPU is used. Requires a Sphinx HTML build and a fresh output directory.
"""
import argparse
import html
import json
import shutil
from pathlib import Path
from urllib.parse import quote

# Pages on the old public site whose source paths changed before this redesign.
LEGACY_ROUTES = {
    "user_guide/landing_page.html": "index.html",
    "user_guide/concepts/grid_methedology.html": "user_guide/concepts/codegen_architecture.html",
    "user_guide/concepts/algorithms/rnea.html": "user_guide/concepts/algorithms/inverse_dynamics.html",
    "faq.html": "how_do_i.html",
    "todo_list.html": "contribution_guidelines.html",
}


def redirect_page(target):
    escaped = html.escape(target, quote=True)
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GRiD documentation has moved</title>
<script>location.replace({json.dumps(target)} + location.search + location.hash);</script>
</head><body><p>GRiD documentation has moved. <a href="{escaped}">Continue to the documentation</a>.</p></body></html>
'''


def assemble(sphinx, landing, output):
    if not (sphinx / "index.html").is_file():
        raise ValueError(f"Missing Sphinx HTML build: {sphinx}")
    if output.exists():
        raise ValueError(f"Output exists; choose a fresh directory: {output}")
    if output.is_relative_to(sphinx) or sphinx.is_relative_to(output):
        raise ValueError("Sphinx source and site output must not overlap")
    for target in LEGACY_ROUTES.values():
        if not (sphinx / target).is_file():
            raise ValueError(f"Missing legacy route destination: {target}")
    # Keep legacy assets/downloads available as well as the new /docs paths.
    shutil.copytree(sphinx, output, ignore=shutil.ignore_patterns(".doctrees"))
    shutil.copytree(sphinx, output / "docs", ignore=shutil.ignore_patterns(".doctrees"))
    redirects = {p.relative_to(sphinx).as_posix(): p.relative_to(sphinx).as_posix()
                 for p in sphinx.rglob("*.html")
                 if not any(part.startswith("_") for part in p.relative_to(sphinx).parts)
                 and p.relative_to(sphinx).as_posix() != "index.html"}
    redirects.update(LEGACY_ROUTES)
    for old, new in redirects.items():
        page = output / old
        page.parent.mkdir(parents=True, exist_ok=True)
        target = "../" * (len(Path(old).parts) - 1) + "docs/" + quote(new, safe="/")
        page.write_text(redirect_page(target), encoding="utf-8")
    shutil.copytree(landing, output / "landing",
                    ignore=shutil.ignore_patterns("README.md", "index.html"))
    shutil.copyfile(landing / "index.html", output / "index.html")
    (output / ".nojekyll").touch()
    print(f"Review site: {output}")


def main():
    docs = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sphinx", type=Path, default=docs / "build" / "html")
    parser.add_argument("--output", type=Path, default=docs / "build" / "site")
    args = parser.parse_args()
    assemble(args.sphinx.resolve(), docs / "landing", args.output.resolve())


if __name__ == "__main__":
    main()

# GRiD website

The cover page lives at `/GRiD/`, with Sphinx at `/GRiD/docs/`.
Building the website does not trigger GPU runs or collect benchmark data.

## Build and review locally

The build recipe is kept in one place — the *Documentation* section of
`docs/source/contribution_guidelines.rst` (rendered at
`/GRiD/docs/contribution_guidelines.html`). It is exactly what the GitHub Pages
workflow (`.github/workflows/gh-pages.yml`) runs, from the repository root; a
developer install already has the docs dependencies (`docs/requirements.txt`):

```sh
.venv/bin/python -m unittest discover -s docs -p 'test_site.py'
.venv/bin/python -m sphinx -b html -W --keep-going docs/source docs/build/html
.venv/bin/python docs/build_site.py          # cover + /docs + legacy redirects -> docs/build/site
.venv/bin/python docs/check_site.py docs/build/site
.venv/bin/python -m http.server -d docs/build/site 8000   # preview at http://localhost:8000
```

The `external/URDFParser` and `external/RBDReference` submodules must be populated
for autodoc (`git submodule update --init --recursive` on a fresh checkout).
No CUDA build, GPU, JAX, or PyTorch installation is required to build the site.
`GRID_DOCS_REF` (default `main`) is the branch the theme's "view source" links
point at; set it only when previewing another branch. Open
`http://localhost:8000/` and `http://localhost:8000/docs/`. The assembly script
refuses to overwrite an existing site, so remove `docs/build/site` between
rebuilds. Relative URLs also support hosting below `/GRiD/`.

## Data and release review

The single source of truth for the release measurements is
[`../source/release_measurements.rst`](../source/release_measurements.rst), rendered
at `/docs/release_measurements.html`: the protocol, the three figures (stacked
absolute timings, speedup against Pinocchio, speedup against the GPU libraries)
and the full table with every cell's status. The homepage embeds the figures from
`docs/source/_static/release/`, which `docs/plot_release_figures.py --approve`
writes from an audited report directory (`python -m test.benchmarks.release.report`);
`check_site.py` verifies the manifest hashes and local page anchors.
Run `docs/export_release_tables.py` after plotting to export the pinned full
tables, their audit, and provenance. This dataset excludes collision timings.

The original paper remains tied to the archival
`robot-acceleration/GRiD`; current development points to `A2R-Lab/GRiD`.

The workflow builds on pull requests that touch the docs, bindings, generator,
peers or packaging, and deploys only on pushes to `main`. Old HTML URLs redirect
to the corresponding `/docs/` page, preserving query strings and fragments.
Explicit aliases cover the old user-guide landing page and renamed/retired pages;
anchors on retired pages may no longer have a matching section.

Styling follows GLASS / GATO / Nerfies under CC BY-SA 4.0, with attribution in the
footer. Fonts use local system stacks so the cover needs no third-party assets.

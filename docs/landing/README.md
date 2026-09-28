# GRiD website

The cover page lives at `/GRiD/`, with Sphinx at `/GRiD/docs/`.
Building the website does not trigger GPU runs or collect benchmark data.

## Build and review locally

From the GRiD checkout, use an existing docs environment or create one:

```sh
python3 -m venv /tmp/grid-docs-venv
source /tmp/grid-docs-venv/bin/activate
python -m pip install -r docs/requirements.txt
```

The `external/URDFParser` and `external/RBDReference` submodules must be populated
for autodoc. From a new checkout, use `git submodule update --init --recursive`.
No CUDA build, GPU, JAX, or PyTorch installation is required to build the site.

```sh
preview_root=$(mktemp -d)
GRID_DOCS_REF=modernizing-tests python -m sphinx -b html -W --keep-going \
  docs/source "$preview_root/html"
python docs/build_site.py --sphinx "$preview_root/html" --output "$preview_root/site"
python docs/check_site.py "$preview_root/site"
python -m http.server 8000 --bind 127.0.0.1 --directory "$preview_root/site"
```

Open `http://localhost:8000/` and `http://localhost:8000/docs/`.
Each rebuild should use a fresh `preview_root`. The assembly script refuses to
overwrite an existing site and does not touch the user's `docs/_build/` output.
Relative URLs also support hosting below `/GRiD/`.

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

The workflow builds pull requests and pushes to `modernizing-tests` without
deploying. Production deployment is restricted to `main`. Old HTML URLs redirect
to the corresponding `/docs/` page, preserving query strings and fragments.
Explicit aliases cover the old user-guide landing page and renamed/retired pages;
anchors on retired pages may no longer have a matching section.

Styling follows GLASS / GATO / Nerfies under CC BY-SA 4.0, with attribution in the
footer. Fonts use local system stacks so the cover needs no third-party assets.

Contributing to GRiD
====================

Open an issue to discuss substantial changes, then submit a focused pull
request with the motivation, implementation and validation results.
Be considerate and respectful in discussions and reviews.

Code and tests
--------------

* Follow the surrounding Python and CUDA style. Use descriptive names and
  document public inputs, output shapes and conventions.
* Add regression tests for fixes and numerical checks for new algorithms.
  See :doc:`user_guide/tutorials/cuda_validation` for the CPU and GPU suites.
* Do not weaken tolerances or remove failing cases to make tests pass.
* For a new algorithm, follow :doc:`user_guide/tutorials/adding_an_algorithm`.
  Generated wrapper regions must be regenerated from their specifications,
  not edited by hand.
* Change peer libraries in their own repositories, then update GRiD's
  submodule pins to commits available from those remotes.

Documentation
-------------

Update the relevant guide and API documentation when behavior changes.
Use runnable examples and state limitations alongside the affected feature.
Documentation and website edits are outside the receipt fingerprint, so they
never require a GPU run.

The site is Sphinx (``pydata_sphinx_theme``) under ``docs/source/`` plus the
cover page under ``docs/landing/``:

* Homepage text and first-screen navigation: ``docs/source/index.rst``.
* Install, examples, CUDA status, validation and benchmark pages:
  ``docs/source/user_guide/``; API landing text:
  ``docs/source/api_reference/index.rst``.
* Add a new page to the nearest ``.. toctree::`` so Sphinx includes it. When a
  page moves or is merged, add its old path to ``LEGACY_ROUTES`` in
  ``docs/build_site.py`` so published links keep working.
* The navigation logo is ``docs/source/_static/a2r_lab.png`` (configured in
  ``docs/source/conf.py``); theme assets such as the favicon and
  ``custom.css`` live under ``docs/source/_static/``. Put figures next to the
  page that uses them or under ``_static``.

Build exactly what the GitHub Pages workflow (``.github/workflows/gh-pages.yml``)
builds, from the repository root; developer installs already include the
documentation dependencies (``docs/requirements.txt``):

.. code-block:: bash

   .venv/bin/python -m unittest discover -s docs -p 'test_site.py'
   .venv/bin/python -m sphinx -b html -W --keep-going docs/source docs/build/html
   .venv/bin/python docs/build_site.py          # cover + /docs + legacy redirects -> docs/build/site
   .venv/bin/python docs/check_site.py docs/build/site
   .venv/bin/python -m http.server -d docs/build/site 8000   # preview at http://localhost:8000

Keep the strict build warning-free; the workflow treats warnings as errors.

Writing style
~~~~~~~~~~~~~

Write documentation the way the GRiD, MPCGPU and GATO papers are written.
Use plain declarative sentences of one idea each, "we" for what the project
does and the imperative for instructions. State a claim once, with its number,
and put caveats in their own sentences. Use one name per thing throughout.
Avoid em-dashes, arrows in prose, capitalised emphasis, bold lead-ins, emoji,
exclamation marks, rhetorical questions and filler such as "simply", "note
that" or "importantly". Describe current behaviour as fact; dates, decision
records and the history of a change belong in the changelog or in
``docs/design_history/``.

License
-------

GRiD software contributions are licensed under the repository's MIT license.
The landing-page design attribution and its CC BY-SA 4.0 terms are recorded
in the website footer.

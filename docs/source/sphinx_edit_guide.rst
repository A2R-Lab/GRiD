Editing The Documentation
=========================

GRiD uses Sphinx with the ``pydata_sphinx_theme`` theme. Most content changes
are plain reStructuredText edits under ``docs/source/``.

Common Edits
------------

* Homepage text and first-screen navigation live in ``docs/source/index.rst``.
* Install, examples, CUDA status, validation, and benchmark pages live under
  ``docs/source/user_guide/``.
* API landing-page text lives in ``docs/source/api_reference/index.rst``.
* Add new pages to the nearest ``.. toctree::`` so Sphinx includes them in the
  site.

Logo, Images, And Styling
-------------------------

The top navigation logo is configured in ``docs/source/conf.py``:

.. code-block:: python

   html_theme_options = {
       "logo": {
           "image_light": "_static/a2r_lab.jpg",
           "image_dark": "_static/a2r_lab.jpg",
       },
   }

To update the current A2R Lab logo, replace
``docs/source/_static/a2r_lab.jpg`` with a new image using the same filename.
To use separate light and dark logos, add both files under
``docs/source/_static/`` and update the two paths in ``conf.py``.

Use ``docs/source/user_guide/imgs/`` for figures that belong to documentation
pages. Use ``docs/source/_static/`` for theme assets such as logos, favicons,
and CSS. Local CSS overrides live in ``docs/source/_static/custom.css``.

Build Locally
-------------

Install the docs dependencies:

.. code-block:: bash

   .venv/bin/python -m pip install -r docs/requirements.txt

Build the docs from the repository root:

.. code-block:: bash

   .venv/bin/python -m sphinx -W --keep-going -b html docs/source docs/build/html

Preview in a browser:

.. code-block:: bash

   .venv/bin/python -m http.server -d docs/build/html 8000

Then open ``http://localhost:8000``.

GitHub Pages
------------

The GitHub Pages workflow installs ``docs/requirements.txt`` and builds the
same Sphinx source tree. Keep local builds warning-free before pushing docs
changes.

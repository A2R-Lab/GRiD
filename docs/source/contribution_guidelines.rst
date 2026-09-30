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
See :doc:`sphinx_edit_guide` for the strict build and local preview commands.

.. toctree::
   :hidden:

   sphinx_edit_guide

License
-------

GRiD software contributions are licensed under the repository's MIT license.
The landing-page design attribution and its CC BY-SA 4.0 terms are recorded
in the website footer.

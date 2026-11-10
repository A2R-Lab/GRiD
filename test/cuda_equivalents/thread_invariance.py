"""Shared thread-count sweep for the CUDA equivalence runners.

A kernel's output must be identical at every thread count (CLAUDE.md,
"Thread-count invariant"). Every runner-driven module used to carry the same
loop by hand: run the executable at each count, compare the values with the
oracle, and assert the values are bit-identical to the previous count's.
``thread_sweep`` is that loop. A module keeps its own oracle comparison in
``check`` and its own parsing in ``run``; the helper owns only the sweep and
the bit-identity assertion, so a thread-count regression reads the same in
every module.

Imported as ``from test.cuda_equivalents.thread_invariance import thread_sweep``
at column 0 so the split driver's helper-import rule fingerprints this file
into every shard that uses it (test/run_split_suite.py, _LOCAL_HELPER_IMPORT).
"""
from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np


def thread_sweep(threads: Iterable[int], run: Callable[[int], Optional[np.ndarray]],
                 check: Optional[Callable[[np.ndarray, int], None]] = None,
                 select: Optional[Callable[[np.ndarray, int], np.ndarray]] = None) -> np.ndarray:
    """Run ``run(count)`` for every thread count, check each result against the
    oracle with ``check(values, count)``, and assert that ``select(values,
    count)`` (default: the whole array) is bit-identical across the counts
    that ran. ``run`` returns None for a count the runner refused before
    launch (an admission limit it has already asserted on); that count is
    left out of the sweep. At least one count must run. Returns the last
    selected values, the baseline every other count matched."""
    baseline = None
    ran = 0
    for count in threads:
        values = run(count)
        if values is None:
            continue
        ran += 1
        if check is not None:
            check(values, count)
        probe = np.asarray(values if select is None else select(values, count))
        if baseline is not None:
            np.testing.assert_array_equal(
                probe, baseline, err_msg=f'thread count {count} is not bit-identical to the baseline')
        baseline = probe
    assert ran > 0, 'no thread count ran'
    return baseline

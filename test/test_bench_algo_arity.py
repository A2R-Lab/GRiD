"""Golden pin for the bench ALGOS arity derivation (H6).

timeGRiD_bindings.ALGOS arities now derive from AbiSpec.inputs. This golden is
the pre-derivation hand table verbatim: a spec-row edit that would change what
the bench feeds an algo fails HERE (loudly, diffably) instead of silently
shifting timing inputs. Update the golden only with a reviewed arity change.
CPU-only.
"""
from __future__ import annotations

GOLDEN = [
    ("inverse_dynamics",          ("q", "v", "a")),
    ("forward_dynamics",          ("q", "v", "u")),
    ("inverse_dynamics_gradient", ("q", "v", "a")),
    ("forward_dynamics_gradient", ("q", "v", "u")),
    ("crba",                      ("q",)),
    ("minv",                      ("q",)),
    ("aba",                       ("q", "v", "u")),
    ("idsva_so",                  ("q", "v", "a")),
    ("fdsva_so",                  ("q", "v", "u")),
    ("end_effector_pose",         ("q",)),
    ("end_effector_pose_gradient", ("q",)),
]


def test_bench_algos_match_golden():
    from test.benchmarks.baselines.grid.timeGRiD_bindings import ALGOS
    assert ALGOS == GOLDEN

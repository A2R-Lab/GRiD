"""M3 F0/F1 referees — per-algorithm header fragments of grid.cuh.

F0 (slicing): gen_all_code drops sentinel comment lines at top-level fragment
boundaries and strips them after the post-passes, so the WRITTEN grid.cuh is
byte-identical to the pre-F0 emission (proven ×7 cells by tools/byte_gate.py
at landing; kept honest here on the iiwa14 cell). self.header_fragments holds
the ordered (name, text) slices; fragments_dir= writes grid_frag_<name>.cuh.

F1 (closure): ALGO_TO_FRAGMENT maps algorithm_list keys to the fragment that
carries them, and the def-before-use referee proves every *_inner/*_device
symbol used by a fragment is defined in the SAME or an EARLIER fragment —
the property a future #include roll-up (design doc F3) needs.

Design: docs/open-tasks/header_fragments_design_2026-09-14.md.
"""
import re

import pytest

from grid_codegen import GRiDCodeGenerator
from grid_codegen.helpers._code_generation_helpers import (
    ALGO_TO_FRAGMENT,
    FRAGMENT_SENTINEL,
    split_fragment_sentinels,
)
from config import robot_urdf
from URDFParser import URDFParser


EXPECTED_ORDER = [
    "_prologue", "core", "ee_kinematics", "inverse_dynamics", "regressors",
    "minv", "forward_dynamics", "forward_dynamics_parameter_gradient",
    "inverse_dynamics_gradient", "forward_dynamics_gradient", "f_ext_gradient",
    "aba", "crba", "integrator", "second_order", "centroidal",
    "frame_jacobian_family", "ee_runtime", "combinations", "init_close",
    "grid_plant", "collision",
]


@pytest.fixture(scope="module", params=["iiwa14", "go2"])
def iiwa14_gen(request, tmp_path_factory):
    """iiwa14 = fixed-base; go2 = floating (exercises the floating dependency
    edges: fd_grad/id_grad -> crba_inner via forward declaration)."""
    name = request.param
    parser = URDFParser()
    robot = parser.parse(str(robot_urdf(name)), floating_base=(name == "go2"))
    gen = GRiDCodeGenerator(robot)
    out_dir = tmp_path_factory.mktemp("frag_" + name)
    out = out_dir / "grid.cuh"
    gen.gen_all_code(output_path=str(out), fragments_dir=str(out_dir / "fragments"))
    return gen, out, out_dir / "fragments"


def test_fragments_cover_file_and_strip_sentinels(iiwa14_gen):
    gen, out, _ = iiwa14_gen
    text = out.read_text()
    assert FRAGMENT_SENTINEL not in text, "sentinel leaked into the written header"
    assert text == gen.code_str
    # stripping is idempotent: re-splitting a stripped stream is one fragment
    refrags, restripped = split_fragment_sentinels(text)
    assert restripped == text and len(refrags) == 1


def test_fragment_names_and_order(iiwa14_gen):
    gen, _, _ = iiwa14_gen
    names = [n for n, _ in gen.header_fragments]
    assert names == EXPECTED_ORDER
    # default iiwa14 build: every default-profile algo fragment is non-empty;
    # opt-in-only groups are empty
    by_name = dict(gen.header_fragments)
    for name in ("core", "ee_kinematics", "inverse_dynamics", "minv",
                 "forward_dynamics", "inverse_dynamics_gradient",
                 "forward_dynamics_gradient", "aba", "crba", "integrator",
                 "second_order", "init_close", "grid_plant"):
        assert by_name[name].strip(), f"fragment {name} unexpectedly empty"
    assert not by_name["collision"].strip(), "collision fragment should be empty without collision_spec"


def test_fragment_files_written(iiwa14_gen):
    gen, _, frag_dir = iiwa14_gen
    files = {p.name for p in frag_dir.iterdir()}
    assert files == {f"grid_frag_{n}.cuh" for n, _ in gen.header_fragments}


def test_algo_to_fragment_map_is_honest(iiwa14_gen):
    gen, _, _ = iiwa14_gen
    by_name = dict(gen.header_fragments)
    assert set(ALGO_TO_FRAGMENT.values()) <= set(by_name), \
        "ALGO_TO_FRAGMENT names a fragment gen_all_code does not emit"
    # every default-emitted algo's mapped fragment mentions its kernel symbol
    # family (weak but robust presence check)
    for algo, frag in (("inverse_dynamics", "inverse_dynamics"),
                       ("minv", "minv"), ("crba", "crba"),
                       ("fdsva_so", "second_order")):
        assert algo in by_name[frag], f"{algo} not found in fragment {frag}"


_DEF_RE = re.compile(r"\b(?:void|int|size_t|T|bool|dim3)\s+([A-Za-z_]\w*_(?:inner|device))\s*\(")
_USE_RE = re.compile(r"\b([A-Za-z_]\w*_(?:inner|device))\s*[<(]")


def test_defs_precede_uses_across_fragments(iiwa14_gen):
    """F1: the roll-up include order (== emission order) must satisfy
    DECLARATION-or-definition-before-use for the cross-fragment
    *_inner/*_device composition surface (fd composes minv/id inners;
    floating gradients call crba_inner DEFINED later but forward-DECLARED
    earlier — _DEF_RE matches the declaration, which is what include order
    needs). A violation means a fragment include-selection cannot be a
    simple ordered subset — exactly what F2 relies on."""
    gen, _, _ = iiwa14_gen
    defined_at = {}
    for idx, (name, text) in enumerate(gen.header_fragments):
        for sym in _DEF_RE.findall(text):
            defined_at.setdefault(sym, idx)
    def _code_lines(text):
        # comment lines (the file-doc interface listing names every kernel)
        # are not uses
        return "\n".join(ln for ln in text.split("\n")
                         if not ln.lstrip().startswith(("//", "*", "/*")))
    violations = []
    for idx, (name, text) in enumerate(gen.header_fragments):
        for sym in set(_USE_RE.findall(_code_lines(text))):
            d = defined_at.get(sym)
            if d is not None and d > idx:
                violations.append(f"{sym} used in {name} (#{idx}) but defined in "
                                  f"{gen.header_fragments[d][0]} (#{d})")
    assert not violations, "def-after-use across fragments:\n" + "\n".join(violations)

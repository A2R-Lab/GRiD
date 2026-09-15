"""Referee: generated device-fn signature uniformity across robot topologies.

Pins the 2026-06-21 consumer-bug fixes (@610ee62 + @dba67d4, see
docs/open-tasks/archive/grid_interface_uniformity_2026-06-21.md): the
load_update_XmatsHom_helpers overloads must carry `int *s_topology_helpers`
on EVERY robot — serial chains included (they pass nullptr and skip the
topology-copy body) — so generic callers see one signature per arity, never
per-robot drift. The standalone `load_topology_helpers` device helper must
also exist for external/inline callers of the *_inner functions.

The 2026-09-15 parked-list audit found this uniformity unpinned (a regression
would only surface as a downstream consumer compile break); this referee
closes that gap. CPU-only: two codegen runs (serial fixed-base iiwa14 +
branched floating-base go2), pin-only for speed.
"""
import re

import pytest

from grid_codegen import GRiDCodeGenerator
from config import robot_urdf
from URDFParser import URDFParser


@pytest.fixture(scope="module", params=["iiwa14", "go2"])
def generated_header(request, tmp_path_factory):
    """iiwa14 = serial chain (identical-S fast path: the drift case — helpers
    must STILL carry the param); go2-floating = branched (helpers used)."""
    name = request.param
    parser = URDFParser()
    robot = parser.parse(str(robot_urdf(name)), floating_base=(name == "go2"))
    gen = GRiDCodeGenerator(robot)
    out = tmp_path_factory.mktemp("sig_" + name) / "grid.cuh"
    gen.gen_all_code(output_path=str(out), enable_mujoco_kernels=False)
    return name, out.read_text()


def test_xmatshom_helper_definitions_uniform(generated_header):
    name, text = generated_header
    defs = [ln for ln in text.split("\n")
            if re.search(r"\bvoid load_update_XmatsHom_helpers\(", ln)]
    assert defs, f"{name}: no load_update_XmatsHom_helpers definitions found"
    offenders = [ln.strip() for ln in defs if "int *s_topology_helpers" not in ln]
    assert not offenders, (
        f"{name}: load_update_XmatsHom_helpers definition(s) dropped the "
        f"uniform `int *s_topology_helpers` parameter (per-robot signature "
        f"drift, the @610ee62 consumer bug): {offenders}"
    )


def test_xmatshom_helper_calls_uniform(generated_header):
    name, text = generated_header
    calls = [ln for ln in text.split("\n")
             if "load_update_XmatsHom_helpers<" in ln]
    assert calls, f"{name}: no load_update_XmatsHom_helpers call sites found"
    offenders = [ln.strip() for ln in calls if "s_topology_helpers" not in ln]
    assert not offenders, (
        f"{name}: call site(s) stopped passing s_topology_helpers — the "
        f"uniform calling convention regressed: {offenders}"
    )


def test_standalone_topology_loader_exists(generated_header):
    name, text = generated_header
    assert ("void load_topology_helpers(int *s_topology_helpers, "
            "const robotModel<T> *d_robotModel)") in text, (
        f"{name}: standalone load_topology_helpers device helper missing "
        f"(@dba67d4 — external/inline *_inner callers self-provision topology)"
    )

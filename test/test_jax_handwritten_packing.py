"""CPU-only guards for hand-written JAX handlers outside the ABI generator.

The source checks deliberately bind source pitch AND copy width: checking only
destination offsets misses floating-base row misalignment and overreads.
Runtime numerical coverage lives in test_jax_floating_input_widths.py.
"""
from pathlib import Path
import re

import pytest


SOURCE = (Path(__file__).resolve().parents[1] /
          "bindings/grid_rbd/wrapper_template.cu").read_text()


@pytest.mark.parametrize("function,third,label", [
    ("grid_rbd_jax_idsva_so_impl", "qdd", "idsva_so: "),
    ("grid_rbd_jax_pack_qqdu", "u", ""),
])
def test_handwritten_jax_packing_uses_nv_source_rows(function, third, label):
    start = SOURCE.index("static ffi::Error " + function + "(")
    end = SOURCE.index("\n}", start)
    body = SOURCE[start:end]
    compact = re.sub(r"\s+", "", body)
    for operand in ("qd", third):
        assert (f'GRID_RBD_FFI_VALIDATE_ROWS({operand}, "{label}{operand}", '
                'grid::NUM_VEL, batch)') in body
    assert "constsize_tq_bytes=nj*sizeof(T);" in compact
    assert "constsize_tv_bytes=grid::NUM_VEL*sizeof(T);" in compact
    assert "constsize_tdst_pitch=3*nj*sizeof(T);" in compact
    for operand, offset, width in (("q", "0", "q_bytes"),
                                   ("qd", "nj", "v_bytes"),
                                   (third, "2*nj", "v_bytes")):
        assert (f"cudaMemcpy2DAsync(&g_data->d_q_qd_u[{offset}],dst_pitch,"
                f"{operand}.typed_data(),{width},{width},batch,") in compact


@pytest.mark.parametrize("function", ["grid_rbd_jax_integrator_body",
                                     "grid_rbd_jax_integrator_gradient_body"])
def test_both_integrator_handlers_use_checked_pack(function):
    start = SOURCE.index("static ffi::Error " + function + "(")
    body = SOURCE[start:SOURCE.index("\n}", start)]
    assert "grid_rbd_jax_pack_qqdu(g_ctx, stream, batch, nj, q, qd, u)" in body
    assert "if (_pe.failure()) return _pe;" in body

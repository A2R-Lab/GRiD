"""Emit the gated per-op grid_plant tails of wrapper_template.cu (jax + torch).

M2 (canonical plan 2026-09-13): the five GATED plant ops — plant_step,
plant_step_gradient, ee_pos_cost, com_cost, momentum_cost — have congruent
jax-FFI and torch bodies (stage inputs D→D into g_plant scratch, launch the
same grid_plant::* kernel, copy outputs D→D out). This module emits BOTH
surfaces' ``#ifdef GRID_PLANT_HAS_*`` tails from one table row per op, so a
new plant op is a PLANT_OPS row, not ~2×40 hand-written wrapper lines.

The un-gated quadratic-cost / barrier families deliberately STAY hand-written:
they are already table-shaped in C++ (one templated shared impl + thin
wrappers), so emitting them would move code into Python strings for zero
de-duplication.

REGION LOCATION — explicit BEGIN/END marker comments (house style, matching
wrapper_body_gen's regions): each surface's tail is the text BETWEEN its
``// ── BEGIN GENERATED <SURFACE> PLANT TAIL …`` and ``// ── END …`` lines.
(M2 initially shipped marker-less sentinel location because wrapper.cu bytes
feed the stage-2 content key and a cosmetic marker would have forced a full
robot-.so rebuild; the markers were added in the 2026-09-15 wrapper-staling
window as planned.) The emitted text stays byte-identical to the checked-in
file, asserted by test/test_wrapper_plant_block.py.

Regenerate (rewrites the regions in place; no-op while byte-identical):
  .venv/bin/python -m grid_codegen.wrapper_plant_gen
"""
from __future__ import annotations

from pathlib import Path

# ── the per-op table ─────────────────────────────────────────────────────────
# Fields deliberately mirror the emitted text's degrees of freedom:
#   gate            GRID_PLANT_HAS_* #ifdef macro
#   kernel          grid_plant:: kernel symbol
#   smem            grid:: *_DYNAMIC_SHARED_MEM_BYTES macro (no <T>() suffix)
#   check           the post-launch check string (matches the historical text —
#                   some ops used the kernel symbol, some the op name)
#   jax_comment / torch launcher comments etc. are kept per-op verbatim: they
#   carry op-specific facts (buffer reuse, packing, register pressure) that a
#   template cannot invent.
PLANT_STEP_OPS = {
    "plant_step": dict(
        gate="GRID_PLANT_HAS_STEP",
        kernel="plant_step_kernel",
        smem="INTEGRATOR_DYNAMIC_SHARED_MEM_BYTES",
        check="plant_step_kernel",
        name="plant_step",
        jax_comment=[
            "// plant_step(x, u; dt, it) → x_kp1  (B, NX). Reuses g_plant.d_grad as x_kp1",
            "// (size NX), matching the C-ABI launch_plant_step.",
        ],
        # launcher decls + kernel dim args (step passes grid::NUM_VEL inline;
        # gradient declares nv — historical text, kept verbatim)
        launcher_decls=["    const int nx = grid::NUM_POS + grid::NUM_VEL;"],
        dim_args="nx, grid::NUM_VEL",
        out_size="nx",
        out_name="x_kp1",
        ss_split=False,
        mjx_comment="// MUJOCO=true launches the plant_step kernel with MUJOCO_OUTPUT=true (floating only).",
    ),
    "plant_step_gradient": dict(
        gate="GRID_PLANT_HAS_STEP_GRADIENT",
        kernel="plant_step_gradient_kernel",
        smem="INTEGRATOR_DU_DYNAMIC_SHARED_MEM_BYTES",
        check="plant_step_gradient_kernel",
        name="plant_step_gradient",
        jax_comment=[
            "// plant_step_gradient(x, u; dt, it) → dAB  (B, 2*NV*3*NV col-major). Reuses",
            "// g_plant.d_grad as the dAB output (size 2*NV*3*NV), matching the C-ABI.",
        ],
        launcher_decls=[
            "    const int nx = grid::NUM_POS + grid::NUM_VEL;",
            "    const int nv = grid::NUM_VEL;",
        ],
        dim_args="nx, nv",
        out_size="dab",
        out_name="dAB",
        ss_split=True,
        mjx_comment="// MUJOCO=true launches the plant_step_gradient kernel with MUJOCO_OUTPUT=true (floating only).",
    ),
}

COST_OPS = {
    "ee_pos_cost": dict(
        gate="GRID_PLANT_HAS_EE_COST",
        kernel="ee_pos_cost_kernel",
        smem="END_EFFECTOR_POSE_GRADIENT_DYNAMIC_SHARED_MEM_BYTES",
        check="ee_pos_cost",
        comment=[
            "// ee_pos_cost(q, p_des, W) → (value (B,1), grad (B,NX), hess (B,NX*NX)).",
            "// MUJOCO=true launches with MUJOCO_OUTPUT=true (floating only).",
        ],
        ins=(("q", "nq"), ("p_des", "3 "), ("W", "3 ")),
        clamp_sym="ee_pos_cost_kernel<T, 0, MUJOCO>",
        launch_targs="T, 0, /*MUJOCO_OUTPUT=*/MUJOCO",
        extra_kargs=["g_plant.d_end_effector_pose", "g_plant.d_end_effector_pose_gradient"],
        jax_launch_targs="T, /*EE=*/0, /*MUJOCO_OUTPUT=*/MUJOCO",
        jax_kargs_layout="ee",     # 3-line karg layout (extra buffers on own line)
        bind="cost_macro",
    ),
    "com_cost": dict(
        gate="GRID_PLANT_HAS_COM_COST",
        kernel="com_cost_kernel",
        smem="COM_DYNAMIC_SHARED_MEM_BYTES",
        check="com_cost",
        comment=[
            "// com_cost(q, p_des, W) → (value (B,1), grad (B,NX), hess (B,NX*NX)).",
            "// MUJOCO=true launches with MUJOCO_OUTPUT=true (floating only).",
        ],
        ins=(("q", "nq"), ("p_des", "3 "), ("W", "3 ")),
        clamp_sym="com_cost_kernel<T, MUJOCO>",
        launch_targs="T, /*MUJOCO_OUTPUT=*/MUJOCO",
        extra_kargs=[],
        jax_launch_targs="T, /*MUJOCO_OUTPUT=*/MUJOCO",
        jax_kargs_layout="com",
        bind="cost_macro",
    ),
    "momentum_cost": dict(
        gate="GRID_PLANT_HAS_MOMENTUM_COST",
        kernel="momentum_cost_kernel",
        smem="CCRBA_DYNAMIC_SHARED_MEM_BYTES",
        check="momentum_cost_kernel",
        comment=[
            "// momentum_cost(q, qd, h_des, W) → (value (B,1), grad (B,NX), hess (B,NX*NX)).",
            "// h_des(6) and W(6) are packed into the two halves of d_in_c, matching the C-ABI.",
            "// MUJOCO=true launches with MUJOCO_OUTPUT=true (floating only).",
        ],
        ins=(("q", "nq"), ("qd", "nv"), ("h_des", "6"), ("W", "6")),
        clamp_sym="momentum_cost_kernel<T, MUJOCO>",
        clamp_comment="    // momentum_cost is register-heavy (~140 regs/thread): clamp to its launch cap.",
        launch_targs="T, /*MUJOCO_OUTPUT=*/MUJOCO",
        extra_kargs=[],
        jax_launch_targs="T, /*MUJOCO_OUTPUT=*/MUJOCO",
        jax_kargs_layout="momentum",
        bind="explicit4",
    ),
}


# ── jax surface ──────────────────────────────────────────────────────────────

def _jax_step_block(key: str) -> list[str]:
    op = PLANT_STEP_OPS[key]
    name, kern, smem = op["name"], op["kernel"], op["smem"]
    L = [f"#ifdef {op['gate']}"]
    L += op["jax_comment"]
    L += [
        "template <grid::IntegratorType IT, bool MUJOCO>",
        f"static void launch_{name}_jax(GridCtx *ctx, cudaStream_t stream, int batch, T gravity, T dt) {{",
        "    GRID_RBD_CTX_LOCALS(ctx);",
        *op["launcher_decls"],
        "    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);",
        f"    const size_t smem = grid::{smem}<T>();",
        f"    cudaFuncSetAttribute(grid_plant::{kern}<T, IT, MUJOCO>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);",
        f"    dim3 thr = grid_clamp_threads_for(grid_plant::{kern}<T, IT, MUJOCO>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));",
        f"    grid_plant::{kern}<T, IT, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr,",
        "        smem, stream>>>(",
        "            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,",
        f"            {op['dim_args']}, g_robot, (T)gravity, (T)dt, batch);",
        "}",
        "",
        op["mjx_comment"],
        "template <bool MUJOCO>",
        f"static ffi::Error grid_rbd_jax_{name}_impl(",
        "    cudaStream_t stream,",
        "    ffi::Buffer<GRID_FFI_T> x, ffi::Buffer<GRID_FFI_T> u,",
        f"    ffi::ResultBuffer<GRID_FFI_T> {op['out_name']},",
        "    T dt, int64_t it, T gravity, int64_t ctx_id)",
        "{",
        "    GRID_RBD_CTX_OR_FFI(ctx_id);",
        '    if (plant_alloc(g_ctx)) return ffi::Error::Internal("plant_alloc failed");',
        "    const int nx = grid::NUM_POS + grid::NUM_VEL;",
        "    const int nv = grid::NUM_VEL;",
    ]
    if op["out_size"] == "dab":
        L.append("    const int dab = 2 * nv * 3 * nv;")
    L += [
        "    auto dims = x.dimensions();",
        "    if (dims.size() != 2 || (int)dims[1] != nx)",
        f'        return ffi::Error::InvalidArgument("{name}: x must be 2D (B, NX)");',
        "    int batch = (int)dims[0];",
        f'    if (batch < 1) return ffi::Error::InvalidArgument("{name}: batch must be >= 1");',
        f'    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("{name}: batch > max_batch");',
        f'    GRID_RBD_FFI_VALIDATE_ROWS(u, "{name}: u", nv, batch);',
        "    cudaMemcpyAsync(g_plant.d_in_a, x.typed_data(), (size_t)batch * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);",
        "    cudaMemcpyAsync(g_plant.d_in_b, u.typed_data(), (size_t)batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);",
    ]
    if op["ss_split"]:
        L += [
            "    // mjx gradient is single-stage (euler/si) only; pin supports all integrator types.",
            "    if constexpr (MUJOCO) {",
            f"        GRID_RBD_IT_DISPATCH_FFI_MJX_SS((int)it, launch_{name}_jax, true, stream, batch, gravity, dt);",
            "    } else {",
            f"        GRID_RBD_IT_DISPATCH_FFI_MJX((int)it, launch_{name}_jax, false, stream, batch, gravity, dt);",
            "    }",
        ]
    else:
        L.append(f"    GRID_RBD_IT_DISPATCH_FFI_MJX((int)it, launch_{name}_jax, MUJOCO, stream, batch, gravity, dt);")
    L += [
        f'    GRID_RBD_FFI_CHECK_LAUNCH("{op["check"]}");',
        f"    cudaMemcpyAsync({op['out_name']}->typed_data(), g_plant.d_grad, (size_t)batch * {op['out_size']} * sizeof(T), cudaMemcpyDeviceToDevice, stream);",
        "    return ffi::Error::Success();",
        "}",
        "",
        f"GRID_RBD_JAX_BIND_2IN_DT_IT_GRAV(grid_rbd_jax_{name}, grid_rbd_jax_{name}_impl<false>);",
        "#ifdef GRID_RBD_WITH_MUJOCO",
        f"GRID_RBD_JAX_BIND_2IN_DT_IT_GRAV(grid_rbd_jax_{name}_mujoco, grid_rbd_jax_{name}_impl<true>);",
        "#endif  // GRID_RBD_WITH_MUJOCO",
        f"#endif  // {op['gate']}",
    ]
    return L


def _size_expr(dim: str) -> str:
    return f"(size_t)batch * {dim} * sizeof(T)"


def _jax_cost_block(key: str) -> list[str]:
    op = COST_OPS[key]
    kern = op["kernel"]
    four_in = len(op["ins"]) == 4
    L = [f"#ifdef {op['gate']}"]
    L += op["comment"]
    L += ["template <bool MUJOCO>", f"static ffi::Error grid_rbd_jax_plant_{key}_impl("]
    if four_in:
        L += [
            "    cudaStream_t stream,",
            "    ffi::Buffer<GRID_FFI_T> q, ffi::Buffer<GRID_FFI_T> qd,",
            "    ffi::Buffer<GRID_FFI_T> h_des, ffi::Buffer<GRID_FFI_T> W,",
        ]
    else:
        a, b, c = (n for n, _ in op["ins"])
        L += [
            "    cudaStream_t stream,",
            f"    ffi::Buffer<GRID_FFI_T> {a}, ffi::Buffer<GRID_FFI_T> {b}, ffi::Buffer<GRID_FFI_T> {c},",
        ]
    L += [
        "    ffi::ResultBuffer<GRID_FFI_T> out, ffi::ResultBuffer<GRID_FFI_T> grad,",
        "    ffi::ResultBuffer<GRID_FFI_T> hess, int64_t ctx_id)",
        "{",
        "    GRID_RBD_CTX_OR_FFI(ctx_id);",
        '    if (plant_alloc(g_ctx)) return ffi::Error::Internal("plant_alloc failed");',
    ]
    if four_in:
        L += [
            "    const int nq = grid::NUM_POS;",
            "    const int nv = grid::NUM_VEL;",
            "    const int nx = nq + nv;",
        ]
    else:
        L += [
            "    const int nq = grid::NUM_POS;",
            "    const int nx = grid::NUM_POS + grid::NUM_VEL;",
        ]
    L += [
        "    auto dims = q.dimensions();",
        "    if (dims.size() != 2 || (int)dims[1] != nq)",
        f'        return ffi::Error::InvalidArgument("{key}: q must be 2D (B, NQ)");',
        "    int batch = (int)dims[0];",
        f'    if (batch < 1) return ffi::Error::InvalidArgument("{key}: batch must be >= 1");',
        f'    if (batch > kMaxBatch) return ffi::Error::InvalidArgument("{key}: batch > max_batch");',
    ]
    # W03: the other operands must carry the leading batch (copies sized by it).
    for n, d in op["ins"][1:]:
        L.append(f'    GRID_RBD_FFI_VALIDATE_ROWS({n}, "{key}: {n}", {d.strip()}, batch);')
    if four_in:
        L += [
            "    cudaMemcpyAsync(g_plant.d_in_a, q.typed_data(),  (size_t)batch * nq * sizeof(T), cudaMemcpyDeviceToDevice, stream);",
            "    cudaMemcpyAsync(g_plant.d_in_b, qd.typed_data(), (size_t)batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);",
            "    cudaMemcpyAsync(g_plant.d_in_c,                  h_des.typed_data(), (size_t)batch * 6 * sizeof(T), cudaMemcpyDeviceToDevice, stream);",
            "    cudaMemcpyAsync(g_plant.d_in_c + (size_t)batch * 6, W.typed_data(),  (size_t)batch * 6 * sizeof(T), cudaMemcpyDeviceToDevice, stream);",
        ]
    else:
        w = max(len(n) for n, _ in op["ins"])
        for slot, (n, d) in zip(("a", "b", "c"), op["ins"]):
            pad = " " * (w - len(n) + 1)
            L.append(f"    cudaMemcpyAsync(g_plant.d_in_{slot}, {n}.typed_data(),{pad}{_size_expr(d)}, cudaMemcpyDeviceToDevice, stream);")
    L.append(f"    size_t smem = grid::{op['smem']}<T>();")
    L.append("    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);")
    if "clamp_comment" in op:
        L.append(op["clamp_comment"])
    L.append(f"    dim3 thr = grid_clamp_threads_for(grid_plant::{op['clamp_sym']}, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));")
    L.append(f"    grid_plant::{kern}<{op['jax_launch_targs']}><<<grid_dim, thr, smem, stream>>>(")
    layout = op["jax_kargs_layout"]
    if layout == "ee":
        L += [
            "        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,",
            "        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,",
            "        g_plant.d_end_effector_pose, g_plant.d_end_effector_pose_gradient, g_robot, batch);",
        ]
    elif layout == "com":
        L += [
            "        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,",
            "        g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,",
            "        g_robot, batch);",
        ]
    else:  # momentum
        L += [
            "        g_plant.d_out, g_plant.d_grad, g_plant.d_hess,",
            "        g_plant.d_in_a, g_plant.d_in_b,",
            "        g_plant.d_in_c, g_plant.d_in_c + (size_t)batch * 6,",
            "        g_robot, batch);",
        ]
    L += [
        f'    GRID_RBD_FFI_CHECK_LAUNCH("{op["check"]}");',
        "    cudaMemcpyAsync(out->typed_data(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);",
        "    cudaMemcpyAsync(grad->typed_data(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);",
        "    cudaMemcpyAsync(hess->typed_data(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);",
        "    return ffi::Error::Success();",
        "}",
        "",
    ]
    if op["bind"] == "cost_macro":
        L += [
            f"GRID_RBD_JAX_PLANT_COST_BIND(grid_rbd_jax_plant_{key},",
            f"                             grid_rbd_jax_plant_{key}_impl<false>);",
            "#ifdef GRID_RBD_WITH_MUJOCO",
            f"GRID_RBD_JAX_PLANT_COST_BIND(grid_rbd_jax_plant_{key}_mujoco,",
            f"                             grid_rbd_jax_plant_{key}_impl<true>);",
            "#endif  // GRID_RBD_WITH_MUJOCO",
        ]
    else:  # explicit 4-input bind (momentum)
        for suffix, flag in (("", "false"), ("_mujoco", "true")):
            if suffix:
                L.append("#ifdef GRID_RBD_WITH_MUJOCO")
            L += [
                "XLA_FFI_DEFINE_HANDLER_SYMBOL(",
                f"    grid_rbd_jax_plant_{key}{suffix},",
                f"    grid_rbd_jax_plant_{key}_impl<{flag}>,",
                "    ffi::Ffi::Bind()",
                "        .Ctx<ffi::PlatformStream<cudaStream_t>>()",
                "        .Arg<ffi::Buffer<GRID_FFI_T>>().Arg<ffi::Buffer<GRID_FFI_T>>()",
                "        .Arg<ffi::Buffer<GRID_FFI_T>>().Arg<ffi::Buffer<GRID_FFI_T>>()",
                "        .Ret<ffi::Buffer<GRID_FFI_T>>().Ret<ffi::Buffer<GRID_FFI_T>>().Ret<ffi::Buffer<GRID_FFI_T>>()",
                '        .Attr<int64_t>("ctx_id")',
                ");",
            ]
            if suffix:
                L.append("#endif  // GRID_RBD_WITH_MUJOCO")
    L.append(f"#endif  // {op['gate']}")
    return L


def gen_jax_plant_tail() -> str:
    parts = (_jax_step_block("plant_step") + [""]
             + _jax_step_block("plant_step_gradient") + [""]
             + _jax_cost_block("ee_pos_cost") + [""]
             + _jax_cost_block("com_cost") + [""]
             + _jax_cost_block("momentum_cost"))
    return "\n".join(parts) + "\n"


# ── torch surface ────────────────────────────────────────────────────────────

def _torch_step_block(key: str) -> list[str]:
    op = PLANT_STEP_OPS[key]
    name, kern, smem = op["name"], op["kernel"], op["smem"]
    L = [f"#ifdef {op['gate']}"]
    L += [
        "template <grid::IntegratorType IT, bool MUJOCO>",
        f"static void torch_launch_{name}(GridCtx *ctx, cudaStream_t stream, int batch, double gravity, double dt) {{",
        "    GRID_RBD_CTX_LOCALS(ctx);",
    ]
    # torch launchers fold the decls onto one line for the gradient (historical)
    if key == "plant_step":
        L.append("    const int nx = grid::NUM_POS + grid::NUM_VEL;")
    else:
        L.append("    const int nx = grid::NUM_POS + grid::NUM_VEL, nv = grid::NUM_VEL;")
    L += [
        "    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);",
        f"    const size_t smem = grid::{smem}<T>();",
        f"    cudaFuncSetAttribute(grid_plant::{kern}<T, IT, MUJOCO>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);",
        f"    dim3 thr = grid_clamp_threads_for(grid_plant::{kern}<T, IT, MUJOCO>, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));",
        f"    grid_plant::{kern}<T, IT, /*MUJOCO_OUTPUT=*/MUJOCO><<<grid_dim, thr,",
        "        smem, stream>>>(",
        "            g_plant.d_grad, g_plant.d_in_a, g_plant.d_in_b,",
        f"            {op['dim_args'] if key != 'plant_step' else 'nx, grid::NUM_VEL'}, g_robot, (T)gravity, (T)dt, batch);",
        "}",
        "",
        "template <bool MUJOCO>",
        f"torch::Tensor torch_{name}(torch::Tensor x, torch::Tensor u, double dt, int64_t it, double gravity, int64_t ctx_id) {{",
        "    GRID_RBD_CTX_OR_THROW(ctx_id);",
        "    grid_torch_plant_init(g_ctx);",
        "    const int nx = grid::NUM_POS + grid::NUM_VEL, nv = grid::NUM_VEL;",
    ]
    if op["out_size"] == "dab":
        L.append("    const int dab = 2 * nv * 3 * nv;")
    L += [
        f'    grid_torch_check_n(x, "{name}: x", nx);',
        f'    grid_torch_check_n(u, "{name}: u", nv);',
        "    int batch = grid_torch_batch(x);",
        f'    grid_torch_check_rows(u, batch, "{name}: u");',
        "    cudaStream_t stream = at::cuda::getCurrentCUDAStream();",
        "    cudaMemcpyAsync(g_plant.d_in_a, x.data_ptr<T>(), (size_t)batch * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);",
        "    cudaMemcpyAsync(g_plant.d_in_b, u.data_ptr<T>(), (size_t)batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);",
        f"    auto out = grid_torch_empty(batch, {op['out_size']}, x);",
    ]
    if op["ss_split"]:
        L += [
            "    // mjx gradient is single-stage (euler/si) only; pin supports all integrator types.",
            "    if constexpr (MUJOCO) {",
            f"        GRID_RBD_IT_DISPATCH_TORCH_MJX_SS((int)it, torch_launch_{name}, true, stream, batch, gravity, dt);",
            "    } else {",
            f"        GRID_RBD_IT_DISPATCH_TORCH_MJX((int)it, torch_launch_{name}, false, stream, batch, gravity, dt);",
            "    }",
        ]
    else:
        L.append(f"    GRID_RBD_IT_DISPATCH_TORCH_MJX((int)it, torch_launch_{name}, MUJOCO, stream, batch, gravity, dt);")
    L += [
        f'    grid_torch_check_launch("{op["check"]}");',
        f"    cudaMemcpyAsync(out.data_ptr<T>(), g_plant.d_grad, (size_t)batch * {op['out_size']} * sizeof(T), cudaMemcpyDeviceToDevice, stream);",
        "    return out;",
        "}",
        f"#endif  // {op['gate']}",
    ]
    return L


def _torch_cost_block(key: str) -> list[str]:
    op = COST_OPS[key]
    kern = op["kernel"]
    four_in = len(op["ins"]) == 4
    args = ", ".join(f"torch::Tensor {n}" for n, _ in op["ins"]) + ", int64_t ctx_id"
    L = [
        f"#ifdef {op['gate']}",
        "template <bool MUJOCO>",
        f"std::vector<torch::Tensor> torch_{key}({args}) {{",
        "    GRID_RBD_CTX_OR_THROW(ctx_id);",
        "    grid_torch_plant_init(g_ctx);",
    ]
    if four_in:
        L.append("    const int nq = grid::NUM_POS, nv = grid::NUM_VEL, nx = nq + nv;")
    else:
        L.append("    const int nq = grid::NUM_POS, nx = grid::NUM_POS + grid::NUM_VEL;")
    for n, d in op["ins"]:
        L.append(f'    grid_torch_check_n({n}, "{key}: {n}", {d.strip()});')
    first = op["ins"][0][0]
    L += [
        f"    int batch = grid_torch_batch({first});",
    ]
    L += [f'    grid_torch_check_rows({n}, batch, "{key}: {n}");' for n, _ in op["ins"][1:]]
    L += [
        "    cudaStream_t stream = at::cuda::getCurrentCUDAStream();",
    ]
    if four_in:
        L += [
            "    cudaMemcpyAsync(g_plant.d_in_a, q.data_ptr<T>(),  (size_t)batch * nq * sizeof(T), cudaMemcpyDeviceToDevice, stream);",
            "    cudaMemcpyAsync(g_plant.d_in_b, qd.data_ptr<T>(), (size_t)batch * nv * sizeof(T), cudaMemcpyDeviceToDevice, stream);",
            "    cudaMemcpyAsync(g_plant.d_in_c,                  h_des.data_ptr<T>(), (size_t)batch * 6 * sizeof(T), cudaMemcpyDeviceToDevice, stream);",
            "    cudaMemcpyAsync(g_plant.d_in_c + (size_t)batch * 6, W.data_ptr<T>(),  (size_t)batch * 6 * sizeof(T), cudaMemcpyDeviceToDevice, stream);",
        ]
    else:
        w = max(len(n) for n, _ in op["ins"])
        for slot, (n, d) in zip(("a", "b", "c"), op["ins"]):
            pad = " " * (w - len(n) + 1)
            L.append(f"    cudaMemcpyAsync(g_plant.d_in_{slot}, {n}.data_ptr<T>(),{pad}{_size_expr(d)}, cudaMemcpyDeviceToDevice, stream);")
    L += [
        f"    auto out  = grid_torch_empty(batch, 1, {first});",
        f"    auto grad = grid_torch_empty(batch, nx, {first});",
        f"    auto hess = grid_torch_empty(batch, nx * nx, {first});",
        f"    size_t smem = grid::{op['smem']}<T>();",
        "    dim3 grid_dim = grid_rbd_grid_for(g_ctx, batch);",
    ]
    if "clamp_comment" in op:
        L.append(op["clamp_comment"])
    L.append(f"    dim3 thr = grid_clamp_threads_for(grid_plant::{op['clamp_sym']}, grid_rbd_launch_threads_n<grid::GRID_ALGO_COUNT>(g_ctx, batch));")
    targs = op["launch_targs"].replace("T, 0, ", "T, 0, ")
    L.append(f"    grid_plant::{kern}<{targs}><<<grid_dim, thr, smem, stream>>>(")
    if four_in:
        L += [
            "        g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b,",
            "        g_plant.d_in_c, g_plant.d_in_c + (size_t)batch * 6, g_robot, batch);",
        ]
    elif op["extra_kargs"]:
        L += [
            "        g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,",
            "        " + ", ".join(op["extra_kargs"]) + ", g_robot, batch);",
        ]
    else:
        L += [
            "        g_plant.d_out, g_plant.d_grad, g_plant.d_hess, g_plant.d_in_a, g_plant.d_in_b, g_plant.d_in_c,",
            "        g_robot, batch);",
        ]
    L += [
        f'    grid_torch_check_launch("{op["check"]}");',
        "    cudaMemcpyAsync(out.data_ptr<T>(),  g_plant.d_out,  (size_t)batch * sizeof(T),           cudaMemcpyDeviceToDevice, stream);",
        "    cudaMemcpyAsync(grad.data_ptr<T>(), g_plant.d_grad, (size_t)batch * nx * sizeof(T),      cudaMemcpyDeviceToDevice, stream);",
        "    cudaMemcpyAsync(hess.data_ptr<T>(), g_plant.d_hess, (size_t)batch * nx * nx * sizeof(T), cudaMemcpyDeviceToDevice, stream);",
        "    return {out, grad, hess};",
        "}",
        f"#endif  // {op['gate']}",
    ]
    return L


def gen_torch_plant_tail() -> str:
    parts = (_torch_step_block("plant_step") + [""]
             + _torch_step_block("plant_step_gradient") + [""]
             + _torch_cost_block("ee_pos_cost") + [""]
             + _torch_cost_block("com_cost") + [""]
             + _torch_cost_block("momentum_cost"))
    return "\n".join(parts) + "\n"


# ── region location + rewrite ────────────────────────────────────────────────

_TEMPLATE = Path(__file__).resolve().parents[1] / "bindings" / "grid_rbd" / "wrapper_template.cu"
_MARKERS = {
    "jax": ("// ── BEGIN GENERATED JAX PLANT TAIL (grid_codegen/wrapper_plant_gen.py — do not hand-edit) ──",
            "// ── END GENERATED JAX PLANT TAIL ──"),
    "torch": ("// ── BEGIN GENERATED TORCH PLANT TAIL (grid_codegen/wrapper_plant_gen.py — do not hand-edit) ──",
              "// ── END GENERATED TORCH PLANT TAIL ──"),
}


def plant_tail_spans(text: str) -> dict[str, tuple[int, int]]:
    """{surface: (begin, end)} inclusive line indices of each surface's emitted
    tail content — the lines BETWEEN that surface's BEGIN/END marker lines
    (the markers themselves are hand-written and stay outside the region)."""
    lines = text.split("\n")
    spans = {}
    for surface, (mark_begin, mark_end) in _MARKERS.items():
        b = lines.index(mark_begin) + 1
        e = lines.index(mark_end) - 1
        spans[surface] = (b, e)
    return spans


def rewrite_plant_tails() -> bool:
    """Replace both tails with the emitted text; returns True if bytes changed."""
    text = _TEMPLATE.read_text()
    lines = text.split("\n")
    spans = plant_tail_spans(text)
    # splice torch first (later in the file) so jax indices stay valid
    for surface, gen in (("torch", gen_torch_plant_tail), ("jax", gen_jax_plant_tail)):
        b, e = spans[surface]
        lines[b:e + 1] = gen().rstrip("\n").split("\n")
    out = "\n".join(lines)
    changed = out != text
    if changed:
        _TEMPLATE.write_text(out)
    return changed


def main() -> None:
    changed = rewrite_plant_tails()
    print("rewrote plant tails (bytes changed)" if changed
          else "plant tails already byte-identical (no write)")


if __name__ == "__main__":
    main()

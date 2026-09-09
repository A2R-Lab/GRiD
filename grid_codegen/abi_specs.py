"""Per-algorithm C-ABI body metadata (P0 of the table-driven wrapper collapse).

Each AbiSpec row transcribes ONE algorithm's hand-written body in
bindings/grid_rbd/wrapper_template.cu into declarative fields, so the P1
collapse can generate the body from the row (and H6 can derive the python
mirrors from the same source of truth). Until P1 lands, NOTHING consumes
these rows at emission time — the contract is enforced by the CPU cross-check
test (test/test_abi_spec_crosscheck.py), which parses wrapper_template.cu and
validates every field against the actual code. Field shapes may still be
refined when P1 consumes them; the cross-check is what keeps transcription
honest in the meantime.

Vocabulary (observed variance, 2026-08-28 wrapper audit):
- pack_mode:  how q/qd/u map onto pack_q_qd_u
    "q_qd_u"     pack_q_qd_u(q, qd, u, ...)
    "q_qd_null"  pack_q_qd_u(q, qd, nullptr, ...)
    "q_q_null"   pack_q_qd_u(q, q, nullptr, ...)   (dummy-qd kinematics path)
    "qdd_u_slot" pack_q_qd_u(q, qd, qdd, ...)      (qdd rides the u slot)
    "pack_q"     pack_q(q, ...)                     (compressed single-input)
    "custom"     bespoke staging (body_override rows)
- qdd_route:  none | u_slot | flag_fork (USE_QDD_FLAG if/else launch fork)
- f_ext_mode: none | optional (apply_f_ext + reset epilogue) | produces
- it_dispatch: None | "FULL" (cases 0-5) | "HESSIAN" (EULER/SI-E only)
- out_copy:   "memcpy_h" (sync + std::memcpy from h_*) |
              "cudaMemcpy_d" (device-direct D2H from d_* — the nj-stride
              workaround sites; see wrapper comments)
- body_override=True: the body is genuinely bespoke (S-cases from the audit);
  P1 keeps it literal and the cross-check only validates identity fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AbiSpec:
    key: str                                  # join to AlgoDescriptor.key
    # ── identity & gating ───────────────────────────────────────────────
    abi_stem: str | None = None               # extern "C" grid_rbd_<stem>; None -> key
    grid_symbol: str | None = None            # host callee; None -> "grid::"+key; may be a macro
    gate_macro: str | None = None             # None -> GRID_HAS_<KEY>
    gate_form: str = "if"                     # "if" | "ifdef"
    sig_mjx_macro: str | None = None          # GRID_RBD_SIG_MJX_* when host template has MUJOCO_OUTPUT
    not_built_msg: str = "subset"             # "subset" | "reduced" | "bare" | literal stub text
                                              # (3 rows carry bespoke phrasings — see D1 notes)
    # Which kind of body/surface the row describes (drives which referee checks
    # apply and which emitter, if any, consumes it):
    #   "cabi"        generated extern "C" grid_rbd_<stem> body (the default;
    #                 full referee validation + wrapper_body_gen emission)
    #   "plant"       hand-written extern "C" grid_plant_<stem> body (PlantBuffers
    #                 section; python-surface fields only, body stays literal)
    #   "ffi_only"    no C-ABI body — surfaces ONLY through the jax FFI handler
    #                 + torch op (forward_dynamics_parameter_gradient)
    #   "kernel_only" grid.cuh kernel with NO binding surface at all yet
    #                 (f_ext_gradient[_dq]; registry + kernel ceiling only)
    surface_class: str = "cabi"
    # ── inputs ──────────────────────────────────────────────────────────
    inputs: tuple[tuple[str, str], ...] = ()  # ordered (c_name, c_type) of the extern "C" params
    pack_mode: str = "q_qd_null"
    qdd_route: str = "none"
    f_ext_mode: str = "none"
    takes_gravity: bool = False
    takes_dt_it: bool = False
    it_dispatch: str | None = None
    trailing_runtime_args: tuple[str, ...] = ()
    # ── launch ──────────────────────────────────────────────────────────
    launch_algo: str | None = None            # GRID_ALGO_* name; None -> from key; "GRID_ALGO_COUNT" = untuned
    clamp_kernel: str | None = None           # grid_clamp_threads_for target, when used
    template_shape: str = "plain"             # "plain" (tight <T>) | "std5" (COMPRESSED+KIND[+MJX]+TIER) | "so4" (KIND[+MJX]+TIER)
    pre_launch_check: bool = False            # the pre-launch sticky-error 200+ block
    # ── output ──────────────────────────────────────────────────────────
    out_buffer: str | None = None             # gridData member (h_c / d_M / ...)
    out_copy: str = "memcpy_h"
    out_size_expr: str | None = None          # per-batch-item element count, C expression
    # ── mjx twin ────────────────────────────────────────────────────────
    has_mjx_twin: bool = False
    mjx_omits_tier: bool = False
    mjx_requires_qdd: bool = False            # `if (!qdd_opt) return 4;`
    mjx_it_dispatch: str | None = None        # twin's IT dispatch when it differs
    mjx_post_launch_check: bool = False       # twin's 200+e cudaGetLastError block
    # The mjx kernel twins do NOT reframe external wrenches: dispatching a twin
    # with a caller f_ext returns silently wrong torques/Jacobians (2026-09-09
    # layout audit). The jax/torch surfaces refuse f_ext under the ACTIVE mjx
    # convention for every row flagged here (numpy falls back to the validated
    # pin-kernel + host-rotation path instead). Referee invariant: True exactly
    # when f_ext_mode == "optional" and has_mjx_twin — flip per-row if a future
    # twin learns to reframe.
    mjx_rejects_f_ext: bool = False
    # ── python (pybind _core.cpp) surface — C4 arc, one field/many consumers ──
    # py_out_dims: trailing per-batch-item out dims as the VERBATIM C++ exprs the
    # pybind method allocates ({batch, *py_out_dims}); the jax/torch reshape
    # collapse derives python dims from the same tuple by token substitution
    # (num_joints_->nj, num_vel_->nv, num_ees_->nee, num_bodies_->nb,
    # second_order_tensor_size->so_size).
    py_out_dims: tuple[str, ...] | None = None
    # out_layout: HOW the flat per-item buffer is ordered / what transform the
    # python surfaces must apply (A3 slice 4, from the 2026-09-09 layout audit).
    # ⚠py_out_dims describes the ALLOCATION only — for the col-major rows a
    # naive reshape(py_out_dims) silently transposes Jacobians. Classes (see
    # bindings/grid_rbd/_out_transform.py, the ONE implementation all surfaces
    # share): "flat" (pass-through), ("reshape", dims) row-major,
    # ("colmajor", rows, cols) reshape(cols,rows)+swap, ("split_colmajor",
    # head, rows, cols) com/ccrba 2-tuples, ("grad_concat",) id_du/fd_du,
    # ("colmajor_whole", rows, cols) integrator_gradient, ("ee_grad",),
    # ("dccrba",), ("so_slabs",), ("minv",) pin-symmetrize/mjx-dense.
    # dim entries are py_dim_tokens tokens or int literals.
    out_layout: tuple | str | None = None
    # which python surfaces expose the method ("numpy","jax","torch"); None =
    # all three. fk_batched is numpy-only BY DESIGN (_surface_common table).
    py_surfaces: tuple[str, ...] | None = None
    # jax PLANT ops are deliberately 2-D-only (hard-coded ShapeDtypeStruct —
    # not vmap-able); a table-driven emitter must not silently make them
    # vmap-able (changes FFI dispatch). The integrator pair became vmap-able
    # 2026-09-09 (routed through _out + broadcast_all like everything else).
    py_vmap_ok: bool = True
    py_rc3_msg: str | None = None             # _core's rc==3 message (differs from the wrapper stub)
    py_twin_guard: str | None = None          # the *_mujoco method's null-fn guard message
    # ── escape hatch ────────────────────────────────────────────────────
    body_override: bool = False


# Seed rows: the four representatives from the 2026-08-28 wrapper audit.
# (Agents transcribe the remainder; the cross-check test is the referee.)
ABI_SPECS: dict[str, AbiSpec] = {
    "crba": AbiSpec(
        "crba",
        inputs=(("q", "const T*"), ("m_out", "T*"), ("batch", "int"),
                ("gravity", "T")),
        pack_mode="q_q_null",
        takes_gravity=True,       # accepted-but-unused by the algorithm
        sig_mjx_macro="GRID_RBD_SIG_MJX_CRBA",
        template_shape="std5",
        out_buffer="d_M", out_copy="cudaMemcpy_d", out_size_expr="grid::NUM_VEL*grid::NUM_VEL",
        has_mjx_twin=True,
        py_out_dims=('num_vel_', 'num_vel_'),
        out_layout=("reshape", ("num_vel_", "num_vel_")),
        py_rc3_msg="crba not built into this robot .so — add 'crba' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='crba_mujoco unavailable: this .so has no mjx CRBA kernel (only floating-base robots export grid_rbd_crba_mujoco)',
    ),
    "inverse_dynamics": AbiSpec(
        "inverse_dynamics",
        inputs=(("q", "const T*"), ("qd", "const T*"), ("qdd_opt", "const T*"),
                ("c_out", "T*"), ("batch", "int"), ("gravity", "T"),
                ("f_ext", "const T*")),
        pack_mode="q_qd_null",
        qdd_route="flag_fork",
        f_ext_mode="optional",
        takes_gravity=True,
        sig_mjx_macro="GRID_RBD_SIG_MJX_INVERSE_DYNAMICS",
        template_shape="qdd6",
        out_buffer="h_c", out_copy="memcpy_h", out_size_expr="grid::NUM_JOINTS",
        has_mjx_twin=True,
        mjx_rejects_f_ext=True, mjx_requires_qdd=True,
        py_out_dims=('num_joints_',),
        out_layout="flat",
        py_rc3_msg="inverse_dynamics not built into this robot .so — add 'inverse_dynamics' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='inverse_dynamics_mujoco unavailable: this .so has no mjx ID kernel (only floating-base robots export grid_rbd_inverse_dynamics_mujoco)',
    ),
    "integrator": AbiSpec(
        "integrator",
        inputs=(("q", "const T*"), ("qd", "const T*"), ("u", "const T*"),
                ("x_kp1_out", "T*"), ("batch", "int"), ("gravity", "T"),
                ("dt", "T"), ("it", "int")),
        pack_mode="q_qd_u",
        takes_gravity=True,
        takes_dt_it=True, it_dispatch="FULL",
        pre_launch_check=True,
        # Consumed by the hand-written launch_integrator_host sig fork (NOT by
        # the generated body — plain/IT rows emit no fork) and by _compile.py's
        # SIG_MJX flag derivation. Was missing until 2026-09-06 (H6): the
        # _compile dict and this table disagreed about the same header.
        sig_mjx_macro="GRID_RBD_SIG_MJX_INTEGRATOR",
        out_buffer="h_x_kp1", out_copy="memcpy_h",
        out_size_expr="(grid::NUM_POS + grid::NUM_VEL)",
        has_mjx_twin=True,
        py_out_dims=('num_joints_ + num_vel_',),
        out_layout="flat",
        py_rc3_msg="integrator not built into this robot .so — add 'integrator' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='integrator_mujoco unavailable: floating-base .so only',
    ),
    "f_ext_contact": AbiSpec(
        "f_ext_contact",
        abi_stem="tool_fext",
        gate_macro="GRID_HAS_CONTACT_RUNTIME", gate_form="ifdef",
        inputs=(("q", "const T*"), ("wrench", "const T*"), ("jid", "int"),
                ("rc", "const T*"), ("out", "T*"), ("batch", "int")),
        pack_mode="custom",
        f_ext_mode="produces",
        launch_algo="GRID_ALGO_COUNT",
        pre_launch_check=True,
        out_buffer="d_f_ext", out_copy="cudaMemcpy_d",
        out_size_expr="6*NUM_BODIES",
        body_override=True,       # bespoke kernel + malloc/smem/funcattr + memset-after
    ),

    # ── dynamics/derivatives half (agent-transcribed 2026-08-28) ──────

    "minv": AbiSpec(
        "minv",
        inputs=(("q", "const T*"), ("minv_out", "T*"), ("batch", "int")),
        template_shape="std5",
        pack_mode="q_q_null",     # pack_q_qd_u(q, /*qd=*/q, /*u=*/nullptr) — qd/u unused
        sig_mjx_macro="GRID_RBD_SIG_MJX_MINV",
        out_buffer="d_Minv", out_copy="cudaMemcpy_d", out_size_expr="grid::NUM_VEL*grid::NUM_VEL",
        has_mjx_twin=True,
        # NOTE: no gravity param at all (unlike crba, which accepts-but-ignores one).
        py_out_dims=('num_vel_', 'num_vel_'),
        out_layout=("minv",),
        py_rc3_msg="minv not built into this robot .so — add 'minv' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='minv_mujoco unavailable: this .so has no mjx Minv kernel (only floating-base robots export grid_rbd_minv_mujoco)',
    ),

    "forward_dynamics": AbiSpec(
        "forward_dynamics",
        inputs=(("q", "const T*"), ("qd", "const T*"), ("u", "const T*"),
                ("qdd_out", "T*"), ("batch", "int"), ("gravity", "T"),
                ("f_ext", "const T*")),
        pack_mode="q_qd_u",
        f_ext_mode="optional",
        takes_gravity=True,
        sig_mjx_macro="GRID_RBD_SIG_MJX_FORWARD_DYNAMICS",
        template_shape="so4",
        out_buffer="h_qdd", out_copy="memcpy_h", out_size_expr="grid::NUM_JOINTS",
        has_mjx_twin=True,
        mjx_rejects_f_ext=True,
        # Twin note (no field): _mujoco path only valid for null f_ext (kernel does
        # not reframe f_ext); enforced by the python dispatch, NOT by a return-4 here.
        py_out_dims=('num_joints_',),
        out_layout="flat",
        py_rc3_msg="forward_dynamics not built into this robot .so — add 'forward_dynamics' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='forward_dynamics_mujoco unavailable: floating-base .so only',
    ),

    "aba": AbiSpec(
        "aba",
        inputs=(("q", "const T*"), ("qd", "const T*"), ("u", "const T*"),
                ("qdd_out", "T*"), ("batch", "int"), ("gravity", "T"),
                ("f_ext", "const T*")),
        pack_mode="q_qd_u",
        f_ext_mode="optional",
        takes_gravity=True,
        sig_mjx_macro="GRID_RBD_SIG_MJX_ABA",
        template_shape="so4",
        out_buffer="h_qdd", out_copy="memcpy_h", out_size_expr="grid::NUM_JOINTS",
        has_mjx_twin=True,
        mjx_rejects_f_ext=True,
        py_out_dims=('num_joints_',),
        out_layout="flat",
        py_rc3_msg="aba not built into this robot .so — add 'aba' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='aba_mujoco unavailable: floating-base .so only',
    ),

    "inverse_dynamics_gradient": AbiSpec(
        "inverse_dynamics_gradient",
        inputs=(("q", "const T*"), ("qd", "const T*"), ("qdd_opt", "const T*"),
                ("dc_du_out", "T*"), ("batch", "int"), ("gravity", "T"),
                ("f_ext", "const T*")),
        pack_mode="q_qd_null",
        qdd_route="flag_fork",    # USE_QDD_FLAG=true/false launch fork on qdd_opt
        f_ext_mode="optional",
        takes_gravity=True,
        sig_mjx_macro="GRID_RBD_SIG_MJX_INVERSE_DYNAMICS_GRADIENT",
        template_shape="qdd6",
        out_buffer="d_dc_du", out_copy="cudaMemcpy_d",
        out_size_expr="2*grid::NUM_VEL*grid::NUM_VEL",  # code: (size_t)batch * 2 * nv * nv * sizeof(T)
        has_mjx_twin=True,
        mjx_rejects_f_ext=True, mjx_requires_qdd=True,
        py_out_dims=('num_vel_', '2 * num_vel_'),
        out_layout=("grad_concat",),
        py_rc3_msg="inverse_dynamics_gradient not built into this robot .so — add 'inverse_dynamics_gradient' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='inverse_dynamics_gradient_mujoco unavailable: floating-base .so only',
    ),

    "forward_dynamics_gradient": AbiSpec(
        "forward_dynamics_gradient",
        inputs=(("q", "const T*"), ("qd", "const T*"), ("u", "const T*"),
                ("df_du_out", "T*"), ("batch", "int"), ("gravity", "T"),
                ("f_ext", "const T*")),
        pack_mode="q_qd_u",
        # host template hard-codes USE_QDD_MINV_FLAG=false in BOTH forks; no qdd
        # input at all -> qdd_route="none".
        f_ext_mode="optional",
        takes_gravity=True,
        sig_mjx_macro="GRID_RBD_SIG_MJX_FORWARD_DYNAMICS_GRADIENT",
        template_shape="fdgrad5",
        out_buffer="d_df_du", out_copy="cudaMemcpy_d", out_size_expr="2*grid::NUM_VEL*grid::NUM_VEL",
        has_mjx_twin=True,
        mjx_rejects_f_ext=True,
        py_out_dims=('num_vel_', '2 * num_vel_'),
        out_layout=("grad_concat",),
        py_rc3_msg="forward_dynamics_gradient not built into this robot .so — add 'forward_dynamics_gradient' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='forward_dynamics_gradient_mujoco unavailable: floating-base .so only',
    ),

    "idsva_so": AbiSpec(
        "idsva_so",
        inputs=(("q", "const T*"), ("qd", "const T*"), ("qdd", "const T*"),
                ("out", "T*"), ("batch", "int"), ("gravity", "T")),
        pack_mode="qdd_u_slot",   # pack_q_qd_u(q, qd, qdd): kernel reads s_qdd from u-slot
        qdd_route="u_slot",       # qdd is REQUIRED positional (no null fork, no return-4)
        takes_gravity=True,
        sig_mjx_macro="GRID_RBD_SIG_MJX_IDSVA_SO",
        template_shape="so4",
        out_buffer="h_idsva_so", out_copy="memcpy_h",
        out_size_expr="grid::SECOND_ORDER_TENSOR_SIZE",
        has_mjx_twin=True, mjx_post_launch_check=True,
        # VOCAB GAP (no field): the MJX TWIN ONLY has the post-launch
        # `cudaGetLastError() -> return 200+e` check (register-heavy kernel,
        # pre_launch_check stays False.
        py_out_dims=('second_order_tensor_size',),
        out_layout=("so_slabs",),
        py_rc3_msg="idsva_so not built into this robot .so — add 'idsva_so_body_frame' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='idsva_so_mujoco unavailable: floating-base .so only',
    ),

    "fdsva_so": AbiSpec(
        "fdsva_so",
        inputs=(("q", "const T*"), ("qd", "const T*"), ("u", "const T*"),
                ("out", "T*"), ("batch", "int"), ("gravity", "T")),
        pack_mode="q_qd_u",
        takes_gravity=True,
        sig_mjx_macro="GRID_RBD_SIG_MJX_FDSVA_SO",
        template_shape="so4",
        out_buffer="h_df2", out_copy="memcpy_h",
        out_size_expr="grid::SECOND_ORDER_TENSOR_SIZE",
        has_mjx_twin=True, mjx_post_launch_check=True,
        # VOCAB GAP (no field): mjx-twin-only 200+ post-launch check
        # (wrapper_template.cu:1601-1602); pin body has none.
        py_out_dims=('second_order_tensor_size',),
        out_layout=("so_slabs",),
        py_rc3_msg="fdsva_so not built into this robot .so — add 'fdsva_so' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='fdsva_so_mujoco unavailable: floating-base .so only',
    ),

    "inverse_dynamics_regressor": AbiSpec(
        "inverse_dynamics_regressor",
        inputs=(("q", "const T*"), ("qd", "const T*"), ("qdd", "const T*"),
                ("out", "T*"), ("batch", "int"), ("gravity", "T")),
        pack_mode="qdd_u_slot",   # qdd rides the u-slot (like idsva_so)
        qdd_route="u_slot",
        takes_gravity=True,
        out_buffer="h_Y", out_copy="memcpy_h",
        out_size_expr="grid::NUM_VEL * 10 * grid::NUM_BODIES",
        has_mjx_twin=True, mjx_omits_tier=True, mjx_post_launch_check=True,  # twin: <T,false,GRID_DATA_ALL,true>, no TIER
        # No GRID_RBD_SIG_MJX_* fork in either body (sig_mjx_macro=None).
        # VOCAB GAP (no field): mjx-twin-only 200+ post-launch check
        # (wrapper_template.cu:1544-1545).
        py_out_dims=('num_vel_ * 10 * num_bodies_',),
        out_layout=("reshape", ("num_vel_", "10*num_bodies_")),
        py_rc3_msg="inverse_dynamics_regressor not built into this robot .so — add 'inverse_dynamics_regressor' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='inverse_dynamics_regressor_mujoco unavailable: floating-base .so only',
    ),

    "integrator_gradient": AbiSpec(
        "integrator_gradient",
        inputs=(("q", "const T*"), ("qd", "const T*"), ("u", "const T*"),
                ("dAB_out", "T*"), ("batch", "int"), ("gravity", "T"),
                ("dt", "T"), ("it", "int")),
        pack_mode="q_qd_u",
        takes_gravity=True,
        takes_dt_it=True, it_dispatch="FULL",   # GRID_RBD_IT_DISPATCH cases 0-5
                                  # (no TIER, no SIG_MJX fork — UNLIKE the integrator's
                                  # launcher, which passes TIER and forks on
                                  # GRID_RBD_SIG_MJX_INTEGRATOR)
        pre_launch_check=True,    # { cudaGetLastError() -> return 200+e } after dispatch
        out_buffer="h_dAB", out_copy="memcpy_h",
        out_size_expr="(2 * grid::NUM_VEL) * (3 * grid::NUM_VEL)",
        has_mjx_twin=True,
        mjx_it_dispatch="HESSIAN",  # twin dispatches GRID_RBD_IT_DISPATCH_HESSIAN
                                    # (EULER / SEMI_IMPLICIT_EULER only)
        # Twin host helper launch_integrator_grad_host_mujoco DOES pass TIER +
        # MUJOCO_OUTPUT=true (mjx_omits_tier=False).
        py_out_dims=('2 * num_vel_ * 3 * num_vel_',),
        out_layout=("colmajor_whole", ("2*num_vel_", "3*num_vel_")),
        py_rc3_msg="integrator_gradient not built into this robot .so — add 'integrator_gradient' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='integrator_gradient_mujoco unavailable: floating-base .so only',
    ),

    "kinetic_energy_regressor": AbiSpec(
        "kinetic_energy_regressor",
        inputs=(("q", "const T*"), ("qd", "const T*"), ("out", "T*"),
                ("batch", "int"), ("gravity", "T")),
        pack_mode="q_qd_null",
        takes_gravity=True,       # accepted + forwarded (KE regressor is gravity-independent)
        out_buffer="h_ke_regressor", out_copy="memcpy_h",
        out_size_expr="10*grid::NUM_BODIES",
        has_mjx_twin=True, mjx_omits_tier=True,
        py_out_dims=('10 * num_bodies_',),
        out_layout="flat",
        py_twin_guard='kinetic_energy_regressor_mujoco unavailable: floating-base .so only (re-register with force_rebuild=True)',
    ),

    "potential_energy_regressor": AbiSpec(
        "potential_energy_regressor",
        inputs=(("q", "const T*"), ("out", "T*"), ("batch", "int"),
                ("gravity", "T")),
        pack_mode="pack_q",       # COMPRESSED input layout (h_q / d_q), like com
        takes_gravity=True,
        out_buffer="h_pe_regressor", out_copy="memcpy_h",
        out_size_expr="10*grid::NUM_BODIES",
        has_mjx_twin=True, mjx_omits_tier=True,
        py_out_dims=('10 * num_bodies_',),
        out_layout="flat",
        py_twin_guard='potential_energy_regressor_mujoco unavailable: floating-base .so only (re-register with force_rebuild=True)',
    ),

    "energy": AbiSpec(
        "energy",
        gate_form="ifdef",        # `#ifdef GRID_HAS_ENERGY` (macro is the default name)
        not_built_msg="reduced",  # "not generated for this robot (reduced codegen profile)"
        inputs=(("q", "const T*"), ("qd", "const T*"), ("out", "T*"),
                ("batch", "int"), ("gravity", "T")),
        pack_mode="q_qd_null",
        takes_gravity=True,
        out_buffer="h_energy", out_copy="memcpy_h", out_size_expr="3",
        has_mjx_twin=True,
        # VOCAB GAP (no field): INVERTED tier asymmetry — the pin host call omits
        # TIER but the MJX TWIN PASSES it (wrapper_template.cu:1743-1745).
        # mjx_omits_tier=False is literally true of the twin, but no field records
        # that the twin ADDS a tier the main body lacks.
        # Also: the rc=3 stub voids only (q, qd, out, batch) — gravity un-voided.
        py_out_dims=('3',),
        out_layout="flat",
        py_rc3_msg='energy not available for this robot: it is not generated for mimic robots (the per-body Jacobian fold is not yet mimic-reduced)',
        py_twin_guard='energy_mujoco unavailable: floating-base .so with energy only (re-register with force_rebuild=True)',
    ),

    # ── kinematics/centroidal/runtime half (agent-transcribed 2026-08-28) ─

    # ── EE pose family (baked targets; macro callee + SIG_MJX signature fork) ──
    "end_effector_pose": AbiSpec(
        "end_effector_pose",
        grid_symbol="grid::GRID_RBD_EE_POSE_FN",              # [D7] macro callee
        sig_mjx_macro="GRID_RBD_SIG_MJX_EE_POSE",
        template_shape="std5",
        inputs=(("q", "const T*"), ("ee_out", "T*"), ("batch", "int")),
        pack_mode="q_q_null",
        out_buffer="h_end_effector_pose", out_copy="memcpy_h",
        out_size_expr="6*GRID_RBD_NUM_EES",
        has_mjx_twin=True,                                     # twin keeps explicit TIER
        py_out_dims=('6 * num_ees_',),
        out_layout="flat",
        py_rc3_msg="end_effector_pose not built into this robot .so — add 'end_effector_pose' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='end_effector_pose_mujoco unavailable: floating-base .so only',
    ),
    "end_effector_pose_gradient": AbiSpec(
        "end_effector_pose_gradient",
        grid_symbol="grid::GRID_RBD_EE_POSE_GRADIENT_FN",     # [D7]
        sig_mjx_macro="GRID_RBD_SIG_MJX_EE_POSE_GRADIENT",
        template_shape="std5",
        inputs=(("q", "const T*"), ("dee_out", "T*"), ("batch", "int")),
        pack_mode="q_q_null",
        out_buffer="h_end_effector_pose_gradient", out_copy="memcpy_h",
        out_size_expr="6*GRID_RBD_NUM_EES*grid::NUM_VEL",
        has_mjx_twin=True,
        py_out_dims=('6 * num_ees_', 'num_vel_'),
        out_layout=("ee_grad",),
        py_rc3_msg="end_effector_pose_gradient not built into this robot .so — add 'end_effector_pose_gradient' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='end_effector_pose_gradient_mujoco unavailable: floating-base .so only',
    ),
    "end_effector_pose_hessian": AbiSpec(
        "end_effector_pose_hessian",
        grid_symbol="grid::GRID_RBD_EE_POSE_HESSIAN_FN",      # [D7]
        sig_mjx_macro="GRID_RBD_SIG_MJX_EE_POSE_HESSIAN",
        template_shape="std5",
        inputs=(("q", "const T*"), ("d2ee_out", "T*"), ("batch", "int")),
        pack_mode="q_q_null",
        out_buffer="h_end_effector_pose_hessian", out_copy="memcpy_h",
        out_size_expr="6*GRID_RBD_NUM_EES*grid::NUM_VEL*grid::NUM_VEL",
        has_mjx_twin=True,
        py_out_dims=('6 * num_ees_', 'num_vel_', 'num_vel_'),
        out_layout=("reshape", ("6*num_ees_", "num_vel_", "num_vel_")),
        py_rc3_msg="end_effector_pose_hessian not built into this robot .so — add 'end_effector_pose_hessian' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='end_effector_pose_hessian_mujoco unavailable: floating-base .so only',
    ),

    # ── batched FK (no registry row yet — flagged in the report) ──────────────
    "fk_batched": AbiSpec(
        "fk_batched",
        grid_symbol="grid::ee_pose_fk_batched",
        gate_form="ifdef",                                     # #ifdef GRID_HAS_FK_BATCHED
        not_built_msg="not supported for this robot (floating-base / spherical; mimic supported since 2026-08-01)",  # [D1]
        inputs=(("q", "const T*"), ("pose7_out", "T*"),
                ("batch", "int"), ("use_warp", "int")),
        pack_mode="custom",                                    # cudaMemcpy q -> static d_q_fk (stride NUM_POS)
        launch_algo="GRID_ALGO_COUNT",
        out_buffer="d_pose7",                                  # [D4] static scratch, not gridData
        out_copy="cudaMemcpy_d",
        out_size_expr="7",
        body_override=True,   # static cudaMalloc scratch + use_warp template fork
                              # (USE_WARP=true clamps threads.x to >=32) + no g_data launch shape
        py_out_dims=('7',),
        out_layout="flat",
        py_surfaces=("numpy",),
        py_rc3_msg='fk_batched: not supported for this robot (floating-base / mimic)',
    ),

    # ── frame_jacobian family (opt-in codegen; runtime frame trailing args) ───
    "frame_jacobian": AbiSpec(
        "frame_jacobian",
        gate_form="ifdef",                                     # #ifdef GRID_HAS_FRAME_JACOBIAN
        not_built_msg="not generated for this .so",            # [D1]
        inputs=(("q", "const T*"), ("out", "T*"), ("batch", "int"),
                ("target_jid", "int"), ("reference_frame", "int")),
        pack_mode="q_q_null",
        trailing_runtime_args=("target_jid", "reference_frame"),
        out_buffer="h_frame_jacobian", out_copy="memcpy_h",
        out_size_expr="6*grid::NUM_VEL",
        has_mjx_twin=True, mjx_omits_tier=True,  # [D6] nested twin block
        py_out_dims=('6 * num_vel_',),
        out_layout=("colmajor", ("6", "num_vel_")),
        py_rc3_msg="frame_jacobian not built into this robot .so — add 'frame_jacobian' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='frame_jacobian_mujoco unavailable: floating-base .so with frame_jacobian only',
    ),
    "frame_jacobian_dot": AbiSpec(
        "frame_jacobian_dot",
        gate_form="ifdef",                                     # #ifdef GRID_HAS_FRAME_JACOBIAN_DOT
        not_built_msg="bare",
        inputs=(("q", "const T*"), ("qd", "const T*"), ("out", "T*"),
                ("batch", "int"), ("target_jid", "int"), ("reference_frame", "int")),
        pack_mode="q_qd_null",
        trailing_runtime_args=("target_jid", "reference_frame"),
        out_buffer="h_frame_jacobian_dot", out_copy="memcpy_h",
        out_size_expr="6*grid::NUM_VEL",
        has_mjx_twin=True, mjx_omits_tier=True,  # [D6] inner #ifdef in FJ twin block
        py_out_dims=('6 * num_vel_',),
        out_layout=("colmajor", ("6", "num_vel_")),
        py_rc3_msg="frame_jacobian_dot not built into this robot .so — add 'frame_jacobian_dot' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='frame_jacobian_dot_mujoco unavailable: floating-base .so with frame_jacobian only',
    ),
    "osc_inertia": AbiSpec(
        "osc_inertia",
        gate_form="ifdef",                                     # #ifdef GRID_HAS_OSC_INERTIA
        not_built_msg="bare",
        inputs=(("q", "const T*"), ("out", "T*"), ("batch", "int")),
        pack_mode="q_q_null",
        out_buffer="h_osc_inertia", out_copy="memcpy_h",
        out_size_expr="36",
        has_mjx_twin=True, mjx_omits_tier=True,  # frame bakes at codegen; no trailing args
        py_out_dims=('36',),
        out_layout=("reshape", ("6", "6")),
        py_rc3_msg="osc_inertia not built into this robot .so — add 'osc_inertia' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='osc_inertia_mujoco unavailable: floating-base .so with frame_jacobian only',
    ),

    # ── RNEA-derived centroidal/bias quantities ───────────────────────────────
    "generalized_gravity": AbiSpec(
        "generalized_gravity",
        # #if GRID_HAS_GENERALIZED_GRAVITY (value-test form, defaults)
        inputs=(("q", "const T*"), ("out", "T*"), ("batch", "int"), ("gravity", "T")),
        pack_mode="q_q_null",                                  # qd unused (zeroed internally)
        takes_gravity=True,
        out_buffer="h_c", out_copy="memcpy_h",
        out_size_expr="grid::NUM_VEL",
        has_mjx_twin=True, mjx_omits_tier=True,  # twin <T,false,GRID_DATA_ALL,true>, no tier
        py_out_dims=('num_vel_',),
        out_layout="flat",
        py_twin_guard='generalized_gravity_mujoco unavailable: floating-base .so only',
    ),
    "nonlinear_effects": AbiSpec(
        "nonlinear_effects",
        inputs=(("q", "const T*"), ("qd", "const T*"), ("out", "T*"),
                ("batch", "int"), ("gravity", "T")),
        pack_mode="q_qd_null",
        takes_gravity=True,
        out_buffer="h_c", out_copy="memcpy_h",
        out_size_expr="grid::NUM_VEL",
        has_mjx_twin=True, mjx_omits_tier=True,
        py_out_dims=('num_vel_',),
        out_layout="flat",
        py_twin_guard='nonlinear_effects_mujoco unavailable: floating-base .so only',
    ),
    "coriolis_matrix": AbiSpec(
        "coriolis_matrix",
        inputs=(("q", "const T*"), ("qd", "const T*"), ("out", "T*"),
                ("batch", "int"), ("gravity", "T")),
        pack_mode="q_qd_null",
        takes_gravity=True,
        out_buffer="h_coriolis", out_copy="memcpy_h",
        out_size_expr="grid::NUM_VEL*grid::NUM_VEL",
        has_mjx_twin=True, mjx_omits_tier=True,
        py_out_dims=('num_vel_ * num_vel_',),
        out_layout=("reshape", ("num_vel_", "num_vel_")),
        py_twin_guard='coriolis_matrix_mujoco unavailable: floating-base .so only',
    ),

    # ── centroidal (compressed pack_q on com/dccrba; clamped launches) ────────
    "com": AbiSpec(
        "com",
        gate_form="ifdef",                                     # #ifdef GRID_HAS_COM
        not_built_msg="reduced",
        inputs=(("q", "const T*"), ("out", "T*"), ("batch", "int")),
        pack_mode="pack_q",                                    # compressed h_q layout
        out_buffer="h_com", out_copy="memcpy_h",
        out_size_expr="(3 + 3 * grid::NUM_VEL)",
        has_mjx_twin=True,
        py_out_dims=('3 + 3 * num_vel_',),
        out_layout=("vec_then_colmajor", "3", ("3", "num_vel_")),
        py_rc3_msg='com not available for this robot: it is not generated for mimic robots (the per-body Jacobian fold is not yet mimic-reduced)',
        py_twin_guard='com_mujoco unavailable: floating-base .so with com only (re-register with force_rebuild=True)',
    ),
    "ccrba": AbiSpec(
        "ccrba",
        gate_form="ifdef",                                     # #ifdef GRID_HAS_CCRBA
        not_built_msg="reduced",
        inputs=(("q", "const T*"), ("qd", "const T*"), ("out", "T*"), ("batch", "int")),
        pack_mode="q_qd_null",
        clamp_kernel="grid::ccrba_kernel<T>",                  # pin only [D3]
        out_buffer="h_ccrba", out_copy="memcpy_h",
        out_size_expr="(6 * grid::NUM_VEL + 6)",
        has_mjx_twin=True,
        py_out_dims=('6 * num_vel_ + 6',),
        out_layout=("colmajor_then_vec", ("6", "num_vel_"), "6"),
        py_rc3_msg='ccrba not available for this robot: it is not generated for mimic robots (the per-body Jacobian fold is not yet mimic-reduced)',
        py_twin_guard='ccrba_mujoco unavailable: floating-base .so with ccrba only (re-register with force_rebuild=True)',
    ),
    "dccrba": AbiSpec(
        "dccrba",
        gate_form="ifdef",                                     # #ifdef GRID_HAS_DCCRBA
        not_built_msg="reduced",
        inputs=(("q", "const T*"), ("out", "T*"), ("batch", "int")),
        pack_mode="pack_q",                                    # compressed h_q layout
        clamp_kernel="grid::dccrba_kernel<T>",                 # pin only [D3]
        out_buffer="h_dccrba", out_copy="memcpy_h",
        out_size_expr="6*grid::NUM_VEL*grid::NUM_VEL",
        has_mjx_twin=True, mjx_omits_tier=True,
        py_out_dims=('6 * num_vel_ * num_vel_',),
        out_layout=("dccrba",),
        py_rc3_msg='dccrba not available for this robot: it is not generated for mimic robots (the per-body Jacobian fold is not yet mimic-reduced)',
        py_twin_guard='dccrba_mujoco unavailable: floating-base non-mimic .so only',
    ),
    "cmm_time_variation": AbiSpec(
        "cmm_time_variation",
        gate_form="ifdef",                                     # #ifdef GRID_HAS_CMM_TIME_VARIATION
        not_built_msg="not generated for this robot (mimic)",  # [D1]
        inputs=(("q", "const T*"), ("qd", "const T*"), ("out", "T*"), ("batch", "int")),
        pack_mode="q_qd_null",
        clamp_kernel="grid::cmm_time_variation_kernel<T>",     # pin only [D3]
        out_buffer="h_cmm_time_variation", out_copy="memcpy_h",
        out_size_expr="6*grid::NUM_VEL",
        has_mjx_twin=True, mjx_omits_tier=True,
        py_out_dims=('6 * num_vel_',),
        out_layout=("colmajor", ("6", "num_vel_")),
        py_rc3_msg='cmm_time_variation not available for this robot: it is not generated for mimic robots (the per-body Jacobian fold is not yet mimic-reduced)',
        py_twin_guard='cmm_time_variation_mujoco unavailable: floating-base .so only',
    ),

    # ── runtime-target EE pose (Xtool staging → body_override [D5]) ───────────
    "end_effector_pose_runtime": AbiSpec(
        "end_effector_pose_runtime",
        gate_form="ifdef",                          # #ifdef GRID_HAS_END_EFFECTOR_POSE_RUNTIME
        not_built_msg="bare",
        inputs=(("q", "const T*"), ("out", "T*"), ("batch", "int"),
                ("target_jid", "int"), ("offset", "const T*")),
        pack_mode="q_q_null",                                  # qd/u unused
        trailing_runtime_args=("target_jid",),                 # offset goes via device staging, not the call
        launch_algo="GRID_ALGO_COUNT",                         # untuned
        out_buffer="h_eePose", out_copy="memcpy_h",
        out_size_expr="6",
        has_mjx_twin=True, mjx_omits_tier=True,  # [D6] twin nested in #ifdef GRID_RBD_WITH_MUJOCO, own #ifdef+stub inside
        # Xtool staging (16-float identity/copy + cudaMemcpy->d_eepose_runtime_offset,
        # rc=101) is emitted by the XTOOL_STAGING feature in wrapper_body_gen.py.
        py_out_dims=('6',),
        out_layout="flat",
        py_rc3_msg="end_effector_pose_runtime not built into this robot .so — add 'end_effector_pose_runtime' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='end_effector_pose_runtime_mujoco unavailable: floating-base .so only',
    ),
    "end_effector_pose_gradient_runtime": AbiSpec(
        "end_effector_pose_gradient_runtime",
        gate_form="ifdef",                 # #ifdef GRID_HAS_END_EFFECTOR_POSE_GRADIENT_RUNTIME
        not_built_msg="bare",
        inputs=(("q", "const T*"), ("out", "T*"), ("batch", "int"),
                ("target_jid", "int"), ("offset", "const T*")),
        pack_mode="q_q_null",
        trailing_runtime_args=("target_jid",),
        launch_algo="GRID_ALGO_COUNT",
        out_buffer="h_eePoseGrad", out_copy="memcpy_h",
        out_size_expr="6*grid::NUM_VEL",
        has_mjx_twin=True, mjx_omits_tier=True,
        # Same XTOOL_STAGING emission as the pose variant.
        py_out_dims=('6 * num_vel_',),
        out_layout=("colmajor", ("6", "num_vel_")),
        py_rc3_msg="end_effector_pose_gradient_runtime not built into this robot .so — add 'end_effector_pose_gradient_runtime' to algorithm_list in register_robot() and rebuild",
        py_twin_guard='end_effector_pose_gradient_runtime_mujoco unavailable: floating-base .so only',
    ),

    # ── non-"cabi" rows (wave-2 item 5, 2026-09-09): python-surface metadata
    # for the hand-written plant section, the FFI-only param gradient, and the
    # two surface-less kernels — referee checks branch on surface_class. ──────

    "forward_dynamics_parameter_gradient": AbiSpec(
        "forward_dynamics_parameter_gradient",
        surface_class="ffi_only",             # no C-ABI body: jax FFI handler
                                              # + torch op only (sysID ∂qdd/∂π)
        takes_gravity=True,
        py_out_dims=('num_vel_', '10 * num_bodies_'),
        out_layout=("reshape", ("num_vel_", "10*num_bodies_")),
        py_surfaces=("jax", "torch"),
        py_rc3_msg="forward_dynamics_parameter_gradient not built into this robot .so — add 'forward_dynamics_parameter_gradient' to algorithm_list in register_robot() and rebuild",
    ),
    "f_ext_gradient": AbiSpec(
        "f_ext_gradient",
        surface_class="kernel_only",          # grid.cuh kernel + registry +
        py_surfaces=(),                       # ceiling entry; NO binding yet
    ),
    "f_ext_gradient_dq": AbiSpec(
        "f_ext_gradient_dq",
        surface_class="kernel_only",
        py_surfaces=(),
    ),
    "plant_step": AbiSpec(
        "plant_step",
        surface_class="plant",
        py_vmap_ok=False,                # extern "C" grid_plant_step —
                                              # hand-written PlantBuffers body
        inputs=(("x", "const T*"), ("u", "const T*"), ("x_kp1", "T*"),
                ("batch", "int"), ("gravity", "float"), ("dt", "float"),
                ("it", "int")),
        takes_gravity=True, takes_dt_it=True,
        has_mjx_twin=True,
        py_out_dims=('num_joints_ + num_vel_',),
        out_layout="flat",
    ),
    "plant_step_gradient": AbiSpec(
        "plant_step_gradient",
        surface_class="plant",
        py_vmap_ok=False,
        inputs=(("x", "const T*"), ("u", "const T*"), ("dAB", "T*"),
                ("batch", "int"), ("gravity", "float"), ("dt", "float"),
                ("it", "int")),
        takes_gravity=True, takes_dt_it=True,
        has_mjx_twin=True,
        py_out_dims=('2 * num_vel_ * 3 * num_vel_',),
        out_layout=("colmajor_whole", ("2*num_vel_", "3*num_vel_")),
    ),
    "plant_step_hessian": AbiSpec(
        "plant_step_hessian",
        surface_class="plant",
        py_vmap_ok=False,
        inputs=(("x", "const T*"), ("u", "const T*"), ("d2AB", "T*"),
                ("batch", "int"), ("gravity", "float"), ("dt", "float"),
                ("it", "int")),
        takes_gravity=True, takes_dt_it=True,
        has_mjx_twin=True,
        py_out_dims=('2 * num_vel_ * 3 * num_vel_ * 3 * num_vel_',),
        # row-major already (H is C-order) — reshape only, no transpose.
        out_layout=("reshape", ("2*num_vel_", "3*num_vel_", "3*num_vel_")),
        py_surfaces=("numpy",),               # jax/torch don't expose it yet
    ),
}

# ── transcription deviation notes (2026-08-28 agents) ──────────────────
#
# out_size_expr spelling follows the seed-row convention: the code's per-item
# element-count expression with whitespace around '*' collapsed (seed "crba"
# spells `nv * nv` as "nv*nv").
# AbiSpec transcription — kinematics/centroidal/runtime half of
# bindings/grid_rbd/wrapper_template.cu (P0 table-driven wrapper refactor).
#
# VOCABULARY DEVIATIONS (flagged, not force-fit — see final report):
#   [D1] not_built_msg: three stubs use phrasings outside the subset|reduced|bare
#        vocabulary; the literal comment text is stored instead:
#          - fk_batched  : "not supported for this robot (floating-base / spherical; mimic supported since 2026-08-01)"
#          - cmm_time_variation : "not generated for this robot (mimic)"
#          - frame_jacobian     : "not generated for this .so"   (no profile parenthetical)
#        Pure `return 3;` stubs with NO comment (frame_jacobian_dot, osc_inertia,
#        both runtime-EE fns) are recorded as "bare".
#   [D2] mjx ADDS tier (inverse of mjx_omits_tier, no field for it): the pin
#        launches of com/ccrba use plain grid::com<T>/grid::ccrba<T> (NO explicit
#        RESOURCE_TIER — template_shape="plain"), but their _mujoco twins DO
#        pass /*RESOURCE_TIER=*/grid::launch_cfg<GRID_ALGO_{COM,CCRBA}>::TIER
#        explicitly. mjx_omits_tier stays False everywhere in this half (no row
#        has pin-with-explicit-tier + twin-without).
#   [D3] clamp asymmetry (no field): ccrba/dccrba/cmm_time_variation clamp via
#        grid_clamp_threads_for(<kernel>, ...) in the PIN launch only; their
#        _mujoco twins launch with the UNclamped grid_rbd_launch_threads_n value.
#   [D4] fk_batched out_buffer "d_pose7" is a function-local static cudaMalloc
#        scratch, NOT a gridData member (field doc says gridData member);
#        likewise its input staging cudaMemcpy's into static d_q_fk. body_override.
#   [D5] runtime-EE offset staging: `offset` (16-float col-major SE(3) Xtool,
#        nullptr => identity) is staged host->device into
#        g_data->d_eepose_runtime_offset with rc=101 on cudaMemcpy failure —
#        no pack/trailing-arg vocabulary covers it, so both rows (and their
#        twins, which repeat the staging verbatim) are body_override=True with
#        the standard-shaped fields still filled faithfully.
#   [D6] twin gate forms vary and have no field: `#if defined(GRID_RBD_WITH_MUJOCO)
#        && GRID_HAS_X` (ee_pose family, generalized_gravity, nonlinear_effects,
#        coriolis_matrix), `#if defined(GRID_HAS_X) && defined(GRID_RBD_WITH_MUJOCO)`
#        (com, ccrba, dccrba, cmm), one nested block for the frame_jacobian family
#        (outer `#if defined(GRID_HAS_FRAME_JACOBIAN) && defined(GRID_RBD_WITH_MUJOCO)`,
#        inner #ifdef for _dot/osc), and the runtime-EE twins sit in a bare
#        `#ifdef GRID_RBD_WITH_MUJOCO` with the GRID_HAS_* #ifdef + rc=3 stub
#        INSIDE the twin body.
#   [D7] ee_pose family grid_symbol is a compile-time #define macro
#        (grid::GRID_RBD_EE_POSE*_FN, resolved by _compile.py from the generated
#        header) — allowed by the docstring ("may be a macro"), noted for P1.
# from grid_codegen.abi_specs import AbiSpec



# ── python-side out-dim expansion (H6-w2 item 6 / A2, 2026-09-09) ────────────
# py_out_dims holds VERBATIM C++ trailing-dim expressions; python consumers
# (handle.capabilities(), the coming jax/torch reshape collapse) expand them by
# token substitution. ONE table so the token set can't drift per consumer.
# `second_order_tensor_size` = the C++ SECOND_ORDER_TENSOR_SIZE constant
# (four rank-3 nv tensors: 4 * nv^3).

# The ONE spelling of the f_ext-under-mjx refusal (see AbiSpec.mjx_rejects_f_ext;
# raised by BaseDelegateMixin._refuse_mjx_f_ext on the jax/torch surfaces).
MJX_F_EXT_REFUSAL = (
    "mjx-convention {name} does not accept f_ext on the jax/torch surfaces: "
    "the mjx kernel twins do not reframe external wrenches yet, and dispatching "
    "them with f_ext would return silently wrong torques (found in the "
    "2026-09-09 layout audit). Use the numpy handle (which falls back to the "
    "validated pin-kernel + host-rotation path), or pass f_ext in pinocchio "
    "convention.")


def py_dim_tokens(nq: int, nv: int, nee: int, nb: int) -> dict[str, int]:
    """The substitution table for expanding an AbiSpec.py_out_dims entry."""
    return {
        "num_joints_": nq,
        "num_vel_": nv,
        "num_ees_": nee,
        "num_bodies_": nb,
        "second_order_tensor_size": 4 * nv ** 3,
    }


def expand_py_out_dims(spec: "AbiSpec", nq: int, nv: int, nee: int, nb: int):
    """Expand spec.py_out_dims to concrete ints, or None when dims are absent
    or reference an unavailable token value (e.g. nb unknown on an old .so)."""
    if not spec.py_out_dims:
        return None
    toks = py_dim_tokens(nq, nv, nee, nb)
    out = []
    for d in spec.py_out_dims:
        expr = d
        for tok, val in toks.items():
            if val is not None:
                expr = expr.replace(tok, str(val))
        try:
            out.append(int(eval(expr, {"__builtins__": {}})))  # arithmetic-only exprs
        except Exception:
            return None
    return tuple(out)

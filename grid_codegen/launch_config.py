"""Launch-config bake: config/launch_configs JSON loading + the generated
launch_cfg<ALGO> emission. H4 move from GRiDCodeGenerator.py (2026-08-27,
verbatim). Module-level names are re-exported from GRiDCodeGenerator for
the external importers (bindings, autotune_ffi, parity goldens)."""
import os
import json
import warnings

from .algo_registry import ALGO_DESCRIPTORS, build_launch_config_algo_to_symbol, launch_config_descriptors

# ─── A1b launch-config bake (single source of truth) ─────────────────────────
# The autotuned per-(robot,base,algo) {tier,threads} live in
# config/launch_configs/<robot>/<DEFAULT_GPU>.json. At codegen time we read the
# matching entry and bake it into the generated header as host-side constants
# (grid::launch_cfg<ALGO_*>). This is purely ADDITIVE host content — kernel
# bodies are untouched (Gate-A). A missing robot / GPU / algo falls back to the
# conservative MAX_PERF_LEVEL_THREADS default so un-tuned robots are unaffected.

# DEFAULT GPU whose autotuned config ships baked-in.
LAUNCH_CONFIG_DEFAULT_GPU = "rtx5090_sm120"

# JSON tier name -> emitted TIER_* enum symbol. Global (not per-algo), kept inline.
LAUNCH_CONFIG_TIER_SYMBOL = {
    "shared":  "TIER_SHARED",
    "lite":    "TIER_LITE",
    "minimal": "TIER_MINIMAL",
}

# JSON (bench-abbreviated) algo key -> canonical grid:: host-launcher symbol.
# DERIVED from the per-algo descriptor table (algo_registry.ALGO_DESCRIPTORS):
# {autotune_key: descriptor.key} for every descriptor carrying a launch_cfg. The
# launch_configs JSON inherits the autotune-sweep's short keys; this maps them to
# the host-launcher / kernel base names so the baked enum is unambiguous. Algos
# present in the registry but absent from the autotune sweep simply get no baked
# override (they fall back to the conservative default). Built once at import.
_ALGO_TO_SYMBOL = build_launch_config_algo_to_symbol()


def _launch_configs_dir():
    """Absolute path to the repo's config/launch_configs/ dir (sibling of GRiDCodeGenerator)."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "launch_configs")


def load_launch_config(robot_id, floating_base, gpu = LAUNCH_CONFIG_DEFAULT_GPU, profile = "host"):
    """Return {grid_symbol: {"tier": TIER_*, "threads": int}} for (robot_id, base).

    Reads config/launch_configs/<robot_id>/<gpu>.json and picks the matching base
    ("floating" or "fixed"). Returns {} (-> full fallback) when the file is
    absent / unreadable / lacks the base. Unknown algo keys / tiers are skipped
    individually (so a partially-populated config still bakes what it can).

    `profile` selects WHICH autotune the binding/build wants — the optimal
    thread count is launch-path AND use-case dependent (see autotune_ffi.py):
      * "host" (default): the C++/host launch path, throughput-optimal — the
        `bases` block, baked by the C++ run.py autotune.
      * "ffi":  the jax/torch FFI launch path, batch-to-land (single batched
        launch, wait for all N to land) — the `ffi_bases` block, baked by
        autotune_ffi.py. The SAME kernel has a DIFFERENT thread optimum under
        the FFI launch path (e.g. iiwa14 fd: host best=128, FFI best=768), so a
        binding that inherits the host pick runs ~1.6x slow. Per-algo fallback
        to `bases` when an algo has no FFI entry (partial FFI tuning is fine)."""
    if not robot_id:
        return {}
    path = os.path.join(_launch_configs_dir(), str(robot_id), str(gpu) + ".json")
    if not os.path.exists(path):
        # The URDF <robot name=...> often differs from the canonical config dir
        # ("KUKAiiwa14" vs iiwa14/, "indy" vs indy7/) — such a miss silently fell
        # back to the conservative tier/threads for EVERY algo even though a
        # tuned table exists (GATO FYI, 2026-08-09). Match case-insensitively:
        # a config dir contained in the robot name, or one the name starts, and
        # take the longest (most specific) hit. No hit -> untuned fallback.
        try:
            rid = str(robot_id).lower()
            cands = [d for d in os.listdir(_launch_configs_dir())
                     if os.path.isdir(os.path.join(_launch_configs_dir(), d))
                     and (d.lower() in rid or d.lower().startswith(rid))]
            if cands:
                best = max(cands, key=len)
                path = os.path.join(_launch_configs_dir(), best, str(gpu) + ".json")
        except OSError:
            pass
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError):
        return {}
    base = "floating" if floating_base else "fixed"
    host_block = (doc.get("bases") or {}).get(base) or {}
    # Any non-host profile overlays its <profile>_bases on top of the host bases
    # (per-algo fallback): "ffi" -> ffi_bases (jax), "torch" -> torch_bases,
    # "pybind" -> pybind_bases. An unknown profile or a missing block leaves the
    # host bases untouched -> byte-identical to an un-tuned robot. (ffi behavior is
    # unchanged since "ffi" -> "ffi_bases".)
    if profile != "host":
        overlay = (doc.get(str(profile) + "_bases") or {}).get(base) or {}
        base_block = dict(host_block)
        base_block.update(overlay)
    else:
        base_block = host_block
    out = {}
    for algo_key, cfg in base_block.items():
        symbol = _ALGO_TO_SYMBOL.get(algo_key)
        if symbol is None:
            continue
        tier_sym = LAUNCH_CONFIG_TIER_SYMBOL.get(str(cfg.get("tier", "")).lower())
        threads = cfg.get("threads")
        if tier_sym is None or not isinstance(threads, int) or threads < 1:
            continue
        out[symbol] = {"tier": tier_sym, "threads": int(threads)}
    return out


def baked_launch_cfg(codegen):
    """The launch-config table THIS codegen run bakes, with the same robot/
    profile resolution gen_add_launch_config_helpers uses — the single load
    path, so the launch_cfg<> specializations and the kernel-attribute
    registrations (B2: divergent-tier opt-ins) can never disagree.
    Returns {symbol: {"tier": "TIER_*", "threads": n}}."""
    robot_id = (codegen.launch_config_robot if codegen.launch_config_robot is not None
                else codegen.robot.get_name())
    profile = getattr(codegen, "launch_config_profile", "host")
    return load_launch_config(robot_id, codegen.robot.floating_base, profile=profile)


def gen_add_launch_config_helpers(self):
    """Emit the A1b baked launch-config table (single source of truth).

    Reads config/launch_configs/<robot>/<DEFAULT_GPU>.json for this robot+base and
    emits per-algo `grid::launch_cfg<GRID_ALGO_*>` specializations carrying
    the autotuned {TIER, THREADS}. The primary template falls back to the
    conservative (GRID_DEFAULT_RESOURCE_TIER, MAX_PERF_LEVEL_THREADS) default,
    so any algo without a baked entry — and any robot/GPU without a config —
    compiles exactly as before. Purely additive host-side content: no kernel
    body is emitted here (Gate-A: kernel `__global__` bodies byte-identical).

    HOST launchers / bindings read grid::launch_cfg<ALGO>::{TIER,THREADS} to
    default their launch config, fixing the FFI thread-default pathology at
    the C++ root. Explicit caller-supplied threads still override."""
    cfg = baked_launch_cfg(self)
    robot_id = self.launch_config_robot if self.launch_config_robot is not None else self.robot.get_name()
    profile = getattr(self, "launch_config_profile", "host")
    # Stable, declaration-ordered list of every algo that COULD carry a config
    # (the canonical grid:: symbols). Emit one enumerator per algo so the
    # table is complete regardless of which algos this robot tuned.
    # Driven from the descriptor table: every algo carrying a baked launch_cfg,
    # in launch_config_descriptors() order (batch 1 = the original 17 as a
    # stable ABI prefix, batch 2+ appended — see algo_registry.py).
    launch_descriptors = launch_config_descriptors()
    algo_symbols = [d.key for d in launch_descriptors]
    enum_names = {d.key: d.enum_name for d in launch_descriptors}
    base_name = "floating" if self.robot.floating_base else "fixed"
    if cfg:
        src = "config/launch_configs/" + str(robot_id) + "/" + LAUNCH_CONFIG_DEFAULT_GPU + ".json (" + base_name + ", profile=" + profile + ")"
    else:
        src = "NONE found for robot=" + str(robot_id) + " base=" + base_name + " gpu=" + LAUNCH_CONFIG_DEFAULT_GPU + " profile=" + profile + " -> conservative fallback"
    self.gen_add_code_lines([
        "",
        "// ─── A1b baked launch config (single source of truth) ───────────────",
        "// Autotuned per-algo {resource tier, threads-per-block} for THIS robot",
        "// + base, baked from " + src + ".",
        "// GRiD kernels are single-block + thread-count-invariant, so (tier,threads)",
        "// is a pure PERFORMANCE choice; HOST launchers / python-jax-torch bindings",
        "// default their launch config from grid::launch_cfg<GRID_ALGO_*>. An algo",
        "// with no autotuned entry (or a robot/GPU with no launch_configs file)",
        "// falls back to the conservative (GRID_DEFAULT_RESOURCE_TIER,",
        "// MAX_PERF_LEVEL_THREADS) default -> un-tuned robots are unaffected.",
        "enum GridAlgo {",
    ])
    for sym in algo_symbols:
        self.gen_add_code_line("    " + enum_names[sym] + ",", )
    self.gen_add_code_lines([
        "    GRID_ALGO_COUNT",
        "};",
        "// Primary template = conservative fallback (matches the historical default).",
        "template <int ALGO> struct launch_cfg {",
        "    static constexpr int TIER    = GRID_DEFAULT_RESOURCE_TIER;",
        "    static constexpr int THREADS = MAX_PERF_LEVEL_THREADS;",
        "};",
    ])
    # Per-algo specializations (only for algos with a baked entry).
    # THREADS is CLAMPED to the tier's __launch_bounds__ (tier_max_threads<TIER>):
    # an autotune may report a count above the tier's launch_bounds because the
    # jax/torch FFI path clamps at launch (grid_clamp_threads_for), but the
    # numpy/pybind host-wrapper path passes the count RAW — an unclamped value >
    # launch_bounds is cudaErrorInvalidValue ("invalid argument"). launch_bounds
    # guarantees tier_max_threads threads are always launchable, so the clamp is
    # exact for the FFI path (a no-op there) and correct for the host path.
    for sym in algo_symbols:
        entry = cfg.get(sym)
        if entry is None:
            continue
        n = str(entry["threads"])
        tmax = "tier_max_threads<" + entry["tier"] + ">()"
        self.gen_add_code_line(
            "template <> struct launch_cfg<" + enum_names[sym] + "> { "
            "static constexpr int TIER = " + entry["tier"] + "; "
            "static constexpr int THREADS = ((" + n + ") < " + tmax + ") ? (" + n + ") : " + tmax + "; };"
        )
    self.gen_add_code_line("")

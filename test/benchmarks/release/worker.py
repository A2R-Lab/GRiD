"""One robot/backend/operation per process. Failures never become timing wins."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import traceback
import numpy as np
from .protocol import agreement, digest, timed, write_json, source_fingerprints, leaves, WARM_SECONDS
from .protocol import ACCURACY_POLICIES, ACCURACY_POLICY_VERSION, accuracy_status, cell_accuracy_status


def gradient_blocks(value, nv):
    """A full Jacobian is checked as two blocks (d/dq, d/dv) so the gross-error
    backstop cannot hide a wrong velocity block under a large position block."""
    return tuple(np.asarray(value)[..., k*nv:(k+1)*nv] for k in range(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--operation", required=True)
    ap.add_argument("--batches", nargs="+", type=int, required=True)
    ap.add_argument("--iterations", type=int, required=True)
    ap.add_argument("--warmups", type=int, required=True)
    ap.add_argument("--warm-seconds", type=float, default=WARM_SECONDS,
                    help="Sustain calls at least this long before sampling so the device reaches its steady clock")
    ap.add_argument("--cpu-threads", type=int, default=1)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--accuracy-policy", choices=ACCURACY_POLICIES, default="strict")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    if args.warm_seconds < 0:
        ap.error("--warm-seconds must be non-negative")
    capture = {"schema": 1, "robot": args.robot, "backend": args.backend,
               "operation": args.operation, "cells": [], "comparison_eligible": False,
               "collector_sources": source_fingerprints(), "accuracy_policy": args.accuracy_policy,
               "accuracy_policy_version": ACCURACY_POLICY_VERSION,
               "warm_seconds": args.warm_seconds,
               "purpose": "preparation" if args.prepare_only else "measurement"}
    failed = False
    adapter = None
    try:
        from .fixtures import Fixture
        fixture = Fixture(args.robot, max(args.batches))
        fixture_path = args.output.with_suffix(".inputs.npz")
        np.savez(fixture_path, q=fixture.q, v=fixture.v, a=fixture.a, u=fixture.u)
        capture.update(fixture=fixture.metadata, inputs_sha256=digest(fixture_path),
            input_values_sha256=hashlib.sha256(b"".join(x.tobytes() for x in
                (fixture.q, fixture.v, fixture.a, fixture.u))).hexdigest())
        if args.backend.startswith("grid_"):
            from .grid_adapter import GridAdapter
            adapter = GridAdapter(args.backend, args.operation, fixture, max(args.batches), args.output.parent)
        else:
            from .competitors import make_adapter
            adapter = make_adapter(args.backend, args.operation, fixture, args.output.parent,
                                   cpu_threads=min(args.cpu_threads, max(1,max(args.batches)//16)))
        capture["adapter"] = adapter.metadata
        blocks = (lambda v: gradient_blocks(v, fixture.nv)) if args.operation.endswith("_gradient") else (lambda v: v)
        def check(actual, expected, **kw):
            return agreement(blocks(actual), blocks(expected), **kw)
        bridged = args.backend in {"grid_native", "grid_cuda"}
        for batch in args.batches:
            cell = {"batch": batch, "status": "error", "comparison_eligible": False}
            try:
                adapter.prepare(batch)
                cell["adapter"] = dict(adapter.metadata)
                if args.prepare_only:
                    if args.backend == "grid_native":
                        adapter.native_library()
                    for call in (adapter.host, adapter.resident):
                        if call is not None:
                            output = adapter.normalize(call())
                            if not all(np.isfinite(x).all() for x in leaves(output)):
                                raise ValueError("Non-finite preparation output")
                    cell.update(status="prepared", note="Build/JIT and finite-output check only; no oracle validation or timings")
                    cell["adapter"] = dict(adapter.metadata)
                    capture["cells"].append(cell)
                    write_json(args.output, capture)
                    print(json.dumps({"batch": batch, "status": "prepared"}), flush=True)
                    continue
                expected = adapter.expected(batch) if hasattr(adapter, "expected") else fixture.expected(args.operation, batch)
                host_output = adapter.normalize(adapter.host())
                agreement_check = check(host_output, expected)
                cell["oracle_agreement"] = agreement_check
                status = accuracy_status(agreement_check, args.accuracy_policy, args.operation, adapter.metadata["dtype"])
                if not agreement_check["passed"]:
                    cell["status"] = "validation_failed"
                    failed_path = args.output.with_suffix(f".b{batch}.failure.npz")
                    np.savez(failed_path, **{f"actual_{i}": x for i,x in enumerate(leaves(host_output))},
                             **{f"oracle_{i}": x for i,x in enumerate(leaves(expected))})
                    cell["failure_outputs"] = {"path": failed_path.name, "sha256": digest(failed_path)}
                    if status == "validation_failed":
                        raise ValueError("Independent-oracle validation failed; no timing collected")
                    cell["accuracy_warning"] = "Entrywise gate exceeded; retained under explicit fp32 FD warning policy"
                resident_time = None
                if bridged:
                    # Native loops time the SAME artifact outside Python; their
                    # outputs must be bitwise the NumPy wrapper's, never "close".
                    if args.backend == "grid_native":
                        host_time, native_out = adapter.native_time(batch, args.warmups, args.iterations, args.warm_seconds)
                    else:
                        host_time, resident_time, native_out, kernel_out = adapter.kernel_time(
                            batch, args.warmups, args.iterations, args.warm_seconds)
                        cell["kernel_compute_agreement"] = agreement(kernel_out, native_out, rtol=0, atol=0)
                        if not cell["kernel_compute_agreement"]["passed"]:
                            raise ValueError("Compute-only and with-memory kernel outputs differ")
                    cell["native_wrapper_agreement"] = agreement(native_out, host_output, rtol=0, atol=0)
                    if not cell["native_wrapper_agreement"]["passed"]:
                        raise ValueError("Native and NumPy results differ")
                elif hasattr(adapter, "time_host"):
                    host_time, extra = adapter.time_host(args.warmups, args.iterations, args.warm_seconds)
                    cell.update(extra)
                else:
                    host_time = timed(adapter.host, adapter.sync, args.warmups, args.iterations, args.warm_seconds)
                cell["host_to_host"] = host_time
                if adapter.resident is not None:
                    resident_output = adapter.normalize(adapter.resident())
                    cell["boundary_agreement"] = check(resident_output, host_output)
                    cell["resident_oracle_agreement"] = check(resident_output, expected)
                    if any(accuracy_status(cell[k], args.accuracy_policy, args.operation, adapter.metadata["dtype"])
                           == "validation_failed" for k in ("boundary_agreement", "resident_oracle_agreement")):
                        cell["status"] = "validation_failed"
                        raise ValueError("Resident output exceeds oracle or boundary error budget")
                    cell["resident"] = resident_time if resident_time is not None else timed(
                        adapter.resident, adapter.sync, args.warmups, args.iterations, args.warm_seconds)
                    if hasattr(adapter, "extra_timings"):
                        cell.update(adapter.extra_timings(args.warmups, args.iterations, args.warm_seconds))
                # Catch stateful/mutating implementations that only pass once.
                post_output = adapter.normalize(adapter.host())
                cell["post_timing_agreement"] = check(post_output, expected)
                # Record variation separately from oracle error. Both must fit
                # the explicit policy; a small inter-call change alone is insufficient.
                cell["repeatability_agreement"] = check(post_output, host_output)
                if adapter.resident is not None:
                    resident_post = adapter.normalize(adapter.resident())
                    cell["resident_post_timing_agreement"] = check(resident_post, expected)
                    cell["resident_repeatability_agreement"] = check(resident_post, resident_output)
                    cell["post_boundary_agreement"] = check(resident_post, post_output)
                status = cell_accuracy_status(cell, args.accuracy_policy, args.operation, adapter.metadata["dtype"])
                if status == "validation_failed":
                    cell["status"] = "validation_failed"
                    raise ValueError("Post-timing oracle or variation error budget exceeded")
                cell.update(status=status, comparison_eligible=True)
                cell["adapter"] = dict(adapter.metadata)
                if status == "accuracy_warning":
                    cell["warning_checks"] = [k for k,v in cell.items() if k.endswith("agreement") and not v["passed"]]
                    cell["accuracy_warning"] = "Bounded fp32 oracle/variation discrepancies retained; not a strict validation pass"
            except Exception as error:
                failed = True
                cell.update(reason=f"{type(error).__name__}: {error}")
                traceback.print_exc()
            capture["cells"].append(cell)
            write_json(args.output, capture)
            print(json.dumps({"batch": batch, "status": cell["status"]}), flush=True)
    except Exception as error:
        failed = True
        capture["setup_error"] = f"{type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        if adapter is not None and hasattr(adapter, "close"):
            try:
                adapter.close()
            except Exception:
                traceback.print_exc()
    write_json(args.output, capture)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

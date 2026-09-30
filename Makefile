# GRiD developer entry points (N2.3 ergonomics, 2026-09-08).
#
# The one rule this file exists to enforce: the checked-in generated regions
# (wrapper_template.cu C-ABI bodies + _core.cpp pybind bodies, both driven by
# grid_codegen/abi_specs.py) MUST be regenerated before _core is rebuilt —
# a stale _core after a table edit fails silently at runtime, not at build.
# `make build` therefore depends on `make gen`.
#
# Everything here is a thin veneer over the real commands (printed as it runs)
# so the underlying invocations stay copy-pasteable.

PY := .venv/bin/python

.PHONY: help gen gen-check build test-cpu test-gpu receipt-refresh verify clean-pycache

help:
	@echo "GRiD targets:"
	@echo "  gen             regenerate the checked-in generated regions (wrapper_body_gen + core_body_gen)"
	@echo "  gen-check       fail if a generated region is out of date vs abi_specs.py"
	@echo "  build           gen + rebuild the grid-rbd package (_core pybind module) via pip install -e ."
	@echo "  test-cpu        the CPU-only suite (drift gates, partition logic, pinocchio oracle smoke)"
	@echo "  test-gpu        full GPU pass via the crash-isolated split driver (hours; quiet box!)"
	@echo "  receipt-refresh re-run ONLY the stale receipt shards and re-sign gpu-proof.json"
	@echo "  verify          verify the committed gpu-proof.json against the everyday policy"
	@echo "  clean-pycache   drop grid_codegen bytecode (required after codegen edits)"

gen:
	$(PY) -m grid_codegen.wrapper_body_gen
	$(PY) -m grid_codegen.core_body_gen

gen-check:
	$(PY) -m grid_codegen.wrapper_body_gen --check
	$(PY) -m grid_codegen.core_body_gen --check

# build DEPENDS on gen: rebuilding _core against a stale generated region is
# the silent-trap this Makefile exists to kill.
build: gen
	.venv/bin/pip install -e . --no-deps

test-cpu:
	$(PY) -m pytest -q -m "not cuda_equivalence and not python_wrappers and not notebooks" test/

# Full GPU receipt pass (SPLIT = crash-isolated shards; see test/run_gpu_proof.sh --help).
test-gpu:
	SPLIT=1 test/run_gpu_proof.sh

# Everyday receipt refresh: re-runs only shards whose fingerprints changed,
# carries the rest, re-signs gpu-proof.json (commit it after).
receipt-refresh:
	SPLIT=1 SPLIT_REFRESH=1 test/run_gpu_proof.sh

verify:
	.venv/bin/gpu-proof verify --receipt gpu-proof.json --policy test/gpu-proof-policy.yaml

clean-pycache:
	rm -rf grid_codegen/__pycache__ grid_codegen/algorithms/__pycache__

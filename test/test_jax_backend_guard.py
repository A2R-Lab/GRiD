"""A CPU-only jax is refused at JAX-handle construction with an install hint
(release check 2026-09-24: the plain `jax` wheel fails deep inside the first FFI
call with "No FFI handler registered ... on a platform Host")."""
import pytest

jax = pytest.importorskip("jax")


def test_cpu_only_jax_is_refused_with_a_hint(monkeypatch):
    from grid_rbd.jax import _require_gpu_backend
    monkeypatch.setattr(jax, "default_backend", lambda: "cpu")
    with pytest.raises(RuntimeError, match=r"jax\[cuda12\]"):
        _require_gpu_backend()


def test_gpu_jax_passes_the_guard(monkeypatch):
    from grid_rbd.jax import _require_gpu_backend
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    _require_gpu_backend()

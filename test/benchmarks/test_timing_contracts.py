"""CPU-only regressions for truthful summary statistics and rollout outputs."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from test.benchmarks.timing_parser import _single_stats, _stats, parse_grid_output

ROOT = Path(__file__).resolve().parents[2]


class TimingContracts(unittest.TestCase):
    def test_average_is_not_a_median(self):
        stats = _stats(7.0, 2.0, 1.0, 11.0)
        self.assertEqual(stats["mean"], 7.0)
        self.assertNotIn("median", stats)
        self.assertEqual(stats.get("median", stats.get("mean")), 7.0)

    def test_single_summary_does_not_invent_distribution(self):
        self.assertEqual(_single_stats(2.5), {"mean": 2.5})
        self.assertEqual(_single_stats(2.5, "median"), {"median": 2.5})

    def test_python_single_median_is_not_a_mean(self):
        out = parse_grid_output("Single Call INVERSE_DYNAMICS 2.5us\n",
                                single_statistic="median")
        self.assertEqual(out["inverse_dynamics"]["single_us"], {"median": 2.5})

    def test_missing_or_negative_overhead_is_not_zero(self):
        from test.benchmarks.plot_benchmarks import _overhead
        self.assertTrue(np.isnan(_overhead(None, 2.0)))
        self.assertTrue(np.isnan(_overhead(2.0, None)))
        self.assertEqual(_overhead(5.0, 2.0), 3.0)
        self.assertEqual(_overhead(2.0, 2.0), 0.0)
        with self.assertWarns(RuntimeWarning):
            self.assertTrue(np.isnan(_overhead(1.0, 2.0)))

    def test_text_parser_preserves_average_without_median(self):
        out = parse_grid_output(
            "[N:16]: INVERSE_DYNAMICS COMPUTE ONLY: Average[7.0us] Std Dev [2.0us] Min [1.0us] Max [11.0us]\n")
        cell = out["inverse_dynamics"]["batch_16_compute_only_us"]
        self.assertEqual(cell["mean"], 7.0)
        self.assertNotIn("median", cell)

    def test_rollout_examples_match_final_state_contract_on_cpu(self):
        # Extract only the pure nested workload functions, not main(), imports,
        # registration, CUDA compilation, or timing. NumPy stands in for JAX and
        # deterministic fake dynamics test loop/scan and output equivalence.
        for name, floating in [("jax_gpu_resident.py", False), ("jax_gpu_resident_go2.py", True)]:
            with self.subTest(example=name):
                tree = ast.parse((ROOT / "bindings/examples" / name).read_text())
                funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                         and n.name in {"rollout_resident", "rollout_host_roundtrip"}]
                self.assertEqual(len(funcs), 2)
                nq, nv, batch, dt = (4, 3, 2, .01) if floating else (3, 3, 2, .01)

                def scan(body, state, controls):
                    for control in controls:
                        state, output = body(state, control)
                        self.assertIsNone(output, "Resident path must not compute an extra cost history")
                    return state, None

                def fd(q, v, u):
                    return .1 * q + .2 * v + u

                def integrator(q, v, u, step):
                    velocity = v[:, :nv] + step * u[:, :nv]
                    position = q.copy()
                    position[:, :nv] += step * velocity
                    return np.concatenate([position, velocity], axis=1)

                scope = dict(np=np, jnp=np, nq=nq, nv=nv, B=batch, dt=dt,
                             pad=np.zeros((batch, nq - nv), np.float32),
                             h=SimpleNamespace(forward_dynamics=fd, integrator=integrator),
                             jax=SimpleNamespace(jit=lambda f: f, lax=SimpleNamespace(scan=scan)))
                exec(compile(ast.fix_missing_locations(ast.Module(body=funcs, type_ignores=[])), name, "exec"), scope)
                q = np.zeros((batch, nq), np.float32)
                v = np.zeros_like(q)
                controls = np.ones((5, batch, nq), np.float32) * .1
                resident = scope["rollout_resident"](q, v, controls)
                host = scope["rollout_host_roundtrip"](q, v, controls)
                self.assertEqual(len(resident), 2)
                for a, b in zip(resident, host):
                    np.testing.assert_allclose(a, b)
                # Every timed rollout completion must fence the whole return tree.
                calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == "block_until_ready"
                         and n.args and isinstance(n.args[0], ast.Call)
                         and isinstance(n.args[0].func, ast.Name) and n.args[0].func.id == "rollout_jit"]
                self.assertEqual(len(calls), 2)  # warmup and timed execution


if __name__ == "__main__":
    unittest.main()

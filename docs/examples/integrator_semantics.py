"""Check the full-state CPU update on constant acceleration, not GPU correctness."""
import numpy as np
from RBDReference import RBDReference


class ConstantAcceleration:
    """Exercise the real reference integrator with an exactly solvable RHS."""
    integrator = RBDReference.integrator
    _integrator_butcher = staticmethod(RBDReference._integrator_butcher)

    def integrate(self, q, dq):
        return q + dq

    def forward_dynamics(self, q, qd, u, f_ext=None):
        return np.ones_like(qd)


def position_error(steps, scheme="rk4"):
    system = ConstantAcceleration()
    state = np.zeros(2)
    for _ in range(steps):
        state = system.integrator(state[:1], state[1:], np.zeros(1),
                                  1.0 / steps, integrator_type=scheme)
    # q(0)=v(0)=0, acceleration=1, T=1 => q(T)=0.5, v(T)=1.
    return float(abs(state[0] - 0.5))


if __name__ == "__main__":
    print("steps  Euler position error  RK4 position error  constant-acceleration error")
    for steps in (10, 20, 40, 80):
        print(f"{steps:5d}  {position_error(steps, 'euler'):.8f}            "
              f"{position_error(steps):.3e}           "
              f"{position_error(steps, 'constant_acceleration'):.3e}")

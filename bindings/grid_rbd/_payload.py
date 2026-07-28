"""Payload / tool spatial-inertia composition for the runtime ``attach_tool`` API.

A rigidly-welded tool is dynamically just a payload added to the attach link's
spatial inertia. GRiD's runtime inertia table uses the frozen 10-parameter basis

    pi = [ m, h = m*c (3), I_O = [Ixx, Ixy, Ixz, Iyy, Iyz, Izz] (6) ]

where ``I_O`` is the rotational inertia about the LINK-FRAME ORIGIN and the implied
6x6 spatial inertia is

    I(pi) = [[ I_O_mat,     skew(h) ],
             [ skew(h)^T,   m * I3  ]].

Because that 6x6 is LINEAR in ``pi`` and both link and payload are expressed about
the SAME link origin, composing a payload is a pure ADD in this basis:

    combined_pi = baked_link_pi + payload_pi_about_link_origin.

This module builds ``payload_pi_about_link_origin`` from the physical payload
description (mass, CoM offset in the link frame, rotational inertia about the
payload CoM) via the parallel-axis theorem, mirroring URDFParser ``Link.build_spatial_inertia``.
"""

import numpy as np

__all__ = ["skew", "payload_inertia_params", "compose_payload_inertia", "params_to_spatial_6x6"]


def skew(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]], dtype=np.float64)


def _inertia_matrix_from_6(I6):
    ixx, ixy, ixz, iyy, iyz, izz = (float(x) for x in I6)
    return np.array([[ixx, ixy, ixz],
                     [ixy, iyy, iyz],
                     [ixz, iyz, izz]], dtype=np.float64)


def _inertia_6_from_matrix(M):
    return np.array([M[0, 0], M[0, 1], M[0, 2], M[1, 1], M[1, 2], M[2, 2]], dtype=np.float64)


def payload_inertia_params(mass, com=(0.0, 0.0, 0.0), inertia=None):
    """10-param spatial inertia of a payload, expressed about the LINK ORIGIN.

    Parameters
    ----------
    mass : float
        Payload mass (kg).
    com : array-like, shape (3,)
        Payload center of mass in the attach-link frame (m).
    inertia : array-like, optional
        Payload rotational inertia about ITS OWN CoM, either a 3x3 matrix or the
        6-vector ``[Ixx, Ixy, Ixz, Iyy, Iyz, Izz]``. Defaults to a point mass
        (zero rotational inertia about the CoM).

    Returns
    -------
    numpy.ndarray, shape (10,)
        ``[m, h=m*c, I_O]`` about the link origin (parallel-axis applied).
    """
    m = float(mass)
    c = np.asarray(com, dtype=np.float64).reshape(-1)[:3]
    if inertia is None:
        I_cm = np.zeros((3, 3), dtype=np.float64)
    else:
        A = np.asarray(inertia, dtype=np.float64)
        I_cm = A if A.shape == (3, 3) else _inertia_matrix_from_6(A.reshape(-1)[:6])
    # parallel-axis: I about origin = I about com + m * skew(c) skew(c)^T
    Sc = skew(c)
    I_O = I_cm + m * (Sc @ Sc.T)
    h = m * c
    return np.concatenate(([m], h, _inertia_6_from_matrix(I_O)))


def compose_payload_inertia(baked_params, mass, com=(0.0, 0.0, 0.0), inertia=None):
    """Add a payload to a link's baked 10-param spatial inertia (about the link origin).

    ``baked_params`` is the link's current 10-param vector (as returned by
    ``RobotHandle.inertia_params`` for that body). Returns the combined 10-param
    vector to poke back via ``set_inertia_params``. Additive by construction.
    """
    base = np.asarray(baked_params, dtype=np.float64).reshape(-1)
    if base.shape != (10,):
        raise ValueError(f"baked_params must be a length-10 vector, got shape {base.shape}.")
    return base + payload_inertia_params(mass, com, inertia)


def params_to_spatial_6x6(params):
    """Reconstruct the 6x6 spatial inertia from a 10-param vector (GRiD convention)."""
    p = np.asarray(params, dtype=np.float64).reshape(-1)
    m = p[0]
    h = p[1:4]
    I_O = _inertia_matrix_from_6(p[4:10])
    Sh = skew(h)
    top = np.hstack((I_O, Sh))
    bottom = np.hstack((Sh.T, m * np.eye(3)))
    return np.vstack((top, bottom))

"""Static configuration/tangent block layouts; no framework or CUDA imports."""


def validate_configuration_layout(blocks, nq, nv):
    """Validate serialized metadata once, before tracing a backward function."""
    result = tuple(tuple(b) for b in blocks)
    q_end = v_end = 0
    for kind, qi, vi, npos, nvel in result:
        expected = {"floating": (7, 6), "spherical": (4, 3)}.get(kind)
        if (qi != q_end or vi != v_end or npos <= 0 or nvel <= 0
                or (expected is not None and (npos, nvel) != expected)
                or (expected is None and (kind != "euclidean" or npos != nvel))):
            raise ValueError("Invalid configuration_layout metadata; re-register the robot")
        q_end += npos
        v_end += nvel
    if (q_end, v_end) != (nq, nv):
        raise ValueError("configuration_layout does not cover nq/nv; re-register the robot")
    return result


def configuration_layout_from_meta(meta, nq, nv):
    blocks = meta.get("configuration_layout")
    if blocks is None:
        # Older scalar-joint artifacts are unambiguous and need no rebuild.
        # Never guess spherical locations from nq-nv alone.
        if nq == nv:
            blocks = [("euclidean", 0, 0, nq, nv)] if nq else []
        elif meta.get("floating_base") and nq == nv + 1 and nv >= 6:
            blocks = [("floating", 0, 0, 7, 6)]
            if nv > 6:
                blocks.append(("euclidean", 7, 6, nq - 7, nv - 6))
        else:
            raise ValueError("This cached spherical-joint artifact lacks configuration_layout; "
                             "re-register the robot with the current generator")
    return validate_configuration_layout(blocks, nq, nv)

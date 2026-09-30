"""Materialize only the requested endpoint, not every simulator field."""
import warp as wp


@wp.kernel
def endpoint_pose(pos: wp.array2d(dtype=wp.vec3), rot: wp.array2d(dtype=wp.mat33),
                  body: int, out: wp.array2d(dtype=wp.float32)):
    b = wp.tid()
    p, r = pos[b, body], rot[b, body]
    out[b,0] = p[0]
    out[b,1] = p[1]
    out[b,2] = p[2]
    out[b,3] = wp.atan2(r[2,1], r[2,2])
    out[b,4] = wp.atan2(-r[2,0], wp.sqrt(r[2,2]*r[2,2]+r[2,1]*r[2,1]))
    out[b,5] = wp.atan2(r[1,0], r[0,0])

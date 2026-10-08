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


@wp.kernel
def endpoint_point(pos: wp.array2d(dtype=wp.vec3), body: int, out: wp.array(dtype=wp.vec3)):
    b = wp.tid()
    out[b] = pos[b, body]


@wp.kernel
def endpoint_pose_jacobian(rot: wp.array2d(dtype=wp.mat33), body: int,
                           jacp: wp.array3d(dtype=wp.float32), jacr: wp.array3d(dtype=wp.float32),
                           out: wp.array3d(dtype=wp.float32)):
    """d[xyz; rpy]/dv from the world-frame body Jacobian: translational rows pass
    through, rotational rows are E(rpy)^{-1} jacr with omega_world = E d(rpy)/dt
    for R = Rz(yaw) Ry(pitch) Rx(roll)."""
    b, j = wp.tid()
    r = rot[b, body]
    pitch = wp.atan2(-r[2,0], wp.sqrt(r[2,2]*r[2,2]+r[2,1]*r[2,1]))
    yaw = wp.atan2(r[1,0], r[0,0])
    cp, sp, cy, sy = wp.cos(pitch), wp.sin(pitch), wp.cos(yaw), wp.sin(yaw)
    wx, wy, wz = jacr[b,0,j], jacr[b,1,j], jacr[b,2,j]
    # E^{-1} = [[cy/cp, sy/cp, 0], [-sy, cy, 0], [cy*sp/cp, sy*sp/cp, 1]]
    out[b,0,j] = jacp[b,0,j]
    out[b,1,j] = jacp[b,1,j]
    out[b,2,j] = jacp[b,2,j]
    out[b,3,j] = (cy*wx + sy*wy) / cp
    out[b,4,j] = -sy*wx + cy*wy
    out[b,5,j] = (cy*sp*wx + sy*sp*wy) / cp + wz

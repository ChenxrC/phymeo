"""Independent FP32 rigid dynamics through the public tyx kernel API."""

import tyx as tx

from ._tyx_numeric import _add, _div, _mul


@tx.kernel
def integrate_rigid_bodies(
    count: tx.i32,
    dt: tx.f32,
    position: tx.ndarray(tx.f32, 2),
    orientation: tx.ndarray(tx.f32, 2),
    velocity: tx.ndarray(tx.f32, 2),
    angular: tx.ndarray(tx.f32, 2),
    force: tx.ndarray(tx.f32, 2),
    torque: tx.ndarray(tx.f32, 2),
    weights: tx.ndarray(tx.f32, 1),
    inertia: tx.ndarray(tx.f32, 2),
    gravity: tx.ndarray(tx.f32, 1),
    next_position: tx.ndarray(tx.f32, 2),
    next_orientation: tx.ndarray(tx.f32, 2),
    next_velocity: tx.ndarray(tx.f32, 2),
    next_angular: tx.ndarray(tx.f32, 2),
    failed: tx.ndarray(tx.i32, 1),
):
    for i in tx.range(count):
        if weights[i] == 0.0:
            for axis in tx.static_range(3):
                next_position[i, axis] = position[i, axis]
                next_velocity[i, axis] = 0.0
                next_angular[i, axis] = 0.0
            for component in tx.static_range(4):
                next_orientation[i, component] = orientation[i, component]
        else:
            for axis in tx.static_range(3):
                acceleration = _add(gravity[axis], _mul(weights[i], force[i, axis]))
                speed = _add(velocity[i, axis], _mul(dt, acceleration))
                coordinate = _add(position[i, axis], _mul(dt, speed))
                next_velocity[i, axis] = speed
                next_position[i, axis] = coordinate
                if (
                    not tx.isfinite(acceleration)
                    or not tx.isfinite(speed)
                    or (not tx.isfinite(coordinate))
                ):
                    failed[i] = 1
            qx = orientation[i, 0]
            qy = orientation[i, 1]
            qz = orientation[i, 2]
            qw = orientation[i, 3]
            qnorm = tx.sqrt(
                _add(_add(_add(_mul(qx, qx), _mul(qy, qy)), _mul(qz, qz)), _mul(qw, qw))
            )
            qx = _div(qx, qnorm)
            qy = _div(qy, qnorm)
            qz = _div(qz, qnorm)
            qw = _div(qw, qnorm)
            r00 = _add(1.0, -_mul(2.0, _add(_mul(qy, qy), _mul(qz, qz))))
            r01 = _mul(2.0, _add(_mul(qx, qy), -_mul(qz, qw)))
            r02 = _mul(2.0, _add(_mul(qx, qz), _mul(qy, qw)))
            r10 = _mul(2.0, _add(_mul(qx, qy), _mul(qz, qw)))
            r11 = _add(1.0, -_mul(2.0, _add(_mul(qx, qx), _mul(qz, qz))))
            r12 = _mul(2.0, _add(_mul(qy, qz), -_mul(qx, qw)))
            r20 = _mul(2.0, _add(_mul(qx, qz), -_mul(qy, qw)))
            r21 = _mul(2.0, _add(_mul(qy, qz), _mul(qx, qw)))
            r22 = _add(1.0, -_mul(2.0, _add(_mul(qx, qx), _mul(qy, qy))))
            ux = _add(
                _add(_mul(r00, angular[i, 0]), _mul(r10, angular[i, 1])), _mul(r20, angular[i, 2])
            )
            uy = _add(
                _add(_mul(r01, angular[i, 0]), _mul(r11, angular[i, 1])), _mul(r21, angular[i, 2])
            )
            uz = _add(
                _add(_mul(r02, angular[i, 0]), _mul(r12, angular[i, 1])), _mul(r22, angular[i, 2])
            )
            fx = _add(
                _add(_mul(r00, torque[i, 0]), _mul(r10, torque[i, 1])), _mul(r20, torque[i, 2])
            )
            fy = _add(
                _add(_mul(r01, torque[i, 0]), _mul(r11, torque[i, 1])), _mul(r21, torque[i, 2])
            )
            fz = _add(
                _add(_mul(r02, torque[i, 0]), _mul(r12, torque[i, 1])), _mul(r22, torque[i, 2])
            )
            ix = inertia[i, 0]
            iy = inertia[i, 1]
            iz = inertia[i, 2]
            ax = _div(fx, ix)
            ay = _div(fy, iy)
            az = _div(fz, iz)
            if iy != iz and uy != 0.0 and (uz != 0.0):
                ax = _add(ax, _mul(_mul(_div(_add(iy, -iz), ix), uy), uz))
            if iz != ix and uz != 0.0 and (ux != 0.0):
                ay = _add(ay, _mul(_mul(_div(_add(iz, -ix), iy), uz), ux))
            if ix != iy and ux != 0.0 and (uy != 0.0):
                az = _add(az, _mul(_mul(_div(_add(ix, -iy), iz), ux), uy))
            if not tx.isfinite(ux) or not tx.isfinite(uy) or (not tx.isfinite(uz)):
                failed[i] = 1
            if not tx.isfinite(ax) or not tx.isfinite(ay) or (not tx.isfinite(az)):
                failed[i] = 1
            wx = _add(
                angular[i, 0], _mul(dt, _add(_add(_mul(r00, ax), _mul(r01, ay)), _mul(r02, az)))
            )
            wy = _add(
                angular[i, 1], _mul(dt, _add(_add(_mul(r10, ax), _mul(r11, ay)), _mul(r12, az)))
            )
            wz = _add(
                angular[i, 2], _mul(dt, _add(_add(_mul(r20, ax), _mul(r21, ay)), _mul(r22, az)))
            )
            next_angular[i, 0] = wx
            next_angular[i, 1] = wy
            next_angular[i, 2] = wz
            rx = _mul(dt, wx)
            ry = _mul(dt, wy)
            rz = _mul(dt, wz)
            nx = qx
            ny = qy
            nz = qz
            nw = qw
            if not tx.isfinite(rx) or not tx.isfinite(ry) or (not tx.isfinite(rz)):
                failed[i] = 1
            else:
                scale = tx.max(tx.abs(rx), tx.max(tx.abs(ry), tx.abs(rz)))
                if scale > 0.0:
                    sx = _div(rx, scale)
                    sy = _div(ry, scale)
                    sz = _div(rz, scale)
                    scaled_norm = tx.sqrt(_add(_add(_mul(sx, sx), _mul(sy, sy)), _mul(sz, sz)))
                    angle = _mul(scale, scaled_norm)
                    if not tx.isfinite(angle):
                        failed[i] = 1
                    else:
                        sine = tx.sin(_mul(0.5, angle))
                        cosine = tx.cos(_mul(0.5, angle))
                        dx = _mul(_div(sx, scaled_norm), sine)
                        dy = _mul(_div(sy, scaled_norm), sine)
                        dz = _mul(_div(sz, scaled_norm), sine)
                        nx = _add(
                            _add(_add(_mul(cosine, qx), _mul(qw, dx)), _mul(dy, qz)), -_mul(dz, qy)
                        )
                        ny = _add(
                            _add(_add(_mul(cosine, qy), _mul(qw, dy)), _mul(dz, qx)), -_mul(dx, qz)
                        )
                        nz = _add(
                            _add(_add(_mul(cosine, qz), _mul(qw, dz)), _mul(dx, qy)), -_mul(dy, qx)
                        )
                        nw = _add(
                            _add(_add(_mul(cosine, qw), -_mul(dx, qx)), -_mul(dy, qy)),
                            -_mul(dz, qz),
                        )
            norm = tx.sqrt(_add(_add(_add(_mul(nx, nx), _mul(ny, ny)), _mul(nz, nz)), _mul(nw, nw)))
            next_orientation[i, 0] = _div(nx, norm)
            next_orientation[i, 1] = _div(ny, norm)
            next_orientation[i, 2] = _div(nz, norm)
            next_orientation[i, 3] = _div(nw, norm)

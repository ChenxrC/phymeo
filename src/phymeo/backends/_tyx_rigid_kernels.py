"""Independent FP32 rigid dynamics through the public tyx kernel API."""

import tyx as tx


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
                acceleration = gravity[axis] + weights[i] * force[i, axis]
                speed = velocity[i, axis] + dt * acceleration
                coordinate = position[i, axis] + dt * speed
                next_velocity[i, axis] = speed
                next_position[i, axis] = coordinate
                if (
                    not tx.isfinite(acceleration)
                    or not tx.isfinite(speed)
                    or not tx.isfinite(coordinate)
                ):
                    failed[i] = 1
            qx = orientation[i, 0]
            qy = orientation[i, 1]
            qz = orientation[i, 2]
            qw = orientation[i, 3]
            qnorm = tx.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            qx = qx / qnorm
            qy = qy / qnorm
            qz = qz / qnorm
            qw = qw / qnorm
            r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
            r01 = 2.0 * (qx * qy - qz * qw)
            r02 = 2.0 * (qx * qz + qy * qw)
            r10 = 2.0 * (qx * qy + qz * qw)
            r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
            r12 = 2.0 * (qy * qz - qx * qw)
            r20 = 2.0 * (qx * qz - qy * qw)
            r21 = 2.0 * (qy * qz + qx * qw)
            r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
            ux = r00 * angular[i, 0] + r10 * angular[i, 1] + r20 * angular[i, 2]
            uy = r01 * angular[i, 0] + r11 * angular[i, 1] + r21 * angular[i, 2]
            uz = r02 * angular[i, 0] + r12 * angular[i, 1] + r22 * angular[i, 2]
            fx = r00 * torque[i, 0] + r10 * torque[i, 1] + r20 * torque[i, 2]
            fy = r01 * torque[i, 0] + r11 * torque[i, 1] + r21 * torque[i, 2]
            fz = r02 * torque[i, 0] + r12 * torque[i, 1] + r22 * torque[i, 2]
            ix = inertia[i, 0]
            iy = inertia[i, 1]
            iz = inertia[i, 2]
            ax = fx / ix
            ay = fy / iy
            az = fz / iz
            # Euler equations in principal axes. Zero coefficients avoid 0*Inf
            # from an irrelevant squared spin for a spherical inertia tensor.
            if iy != iz and uy != 0.0 and uz != 0.0:
                ax = ax + ((iy - iz) / ix) * uy * uz
            if iz != ix and uz != 0.0 and ux != 0.0:
                ay = ay + ((iz - ix) / iy) * uz * ux
            if ix != iy and ux != 0.0 and uy != 0.0:
                az = az + ((ix - iy) / iz) * ux * uy
            if not tx.isfinite(ux) or not tx.isfinite(uy) or not tx.isfinite(uz):
                failed[i] = 1
            if not tx.isfinite(ax) or not tx.isfinite(ay) or not tx.isfinite(az):
                failed[i] = 1
            wx = angular[i, 0] + dt * (r00 * ax + r01 * ay + r02 * az)
            wy = angular[i, 1] + dt * (r10 * ax + r11 * ay + r12 * az)
            wz = angular[i, 2] + dt * (r20 * ax + r21 * ay + r22 * az)
            next_angular[i, 0] = wx
            next_angular[i, 1] = wy
            next_angular[i, 2] = wz
            rx = dt * wx
            ry = dt * wy
            rz = dt * wz
            nx = qx
            ny = qy
            nz = qz
            nw = qw
            if not tx.isfinite(rx) or not tx.isfinite(ry) or not tx.isfinite(rz):
                failed[i] = 1
            else:
                scale = tx.max(tx.abs(rx), tx.max(tx.abs(ry), tx.abs(rz)))
                if scale > 0.0:
                    sx = rx / scale
                    sy = ry / scale
                    sz = rz / scale
                    scaled_norm = tx.sqrt(sx * sx + sy * sy + sz * sz)
                    angle = scale * scaled_norm
                    if not tx.isfinite(angle):
                        failed[i] = 1
                    else:
                        sine = tx.sin(0.5 * angle)
                        cosine = tx.cos(0.5 * angle)
                        dx = (sx / scaled_norm) * sine
                        dy = (sy / scaled_norm) * sine
                        dz = (sz / scaled_norm) * sine
                        nx = cosine * qx + qw * dx + dy * qz - dz * qy
                        ny = cosine * qy + qw * dy + dz * qx - dx * qz
                        nz = cosine * qz + qw * dz + dx * qy - dy * qx
                        nw = cosine * qw - dx * qx - dy * qy - dz * qz
            norm = tx.sqrt(nx * nx + ny * ny + nz * nz + nw * nw)
            next_orientation[i, 0] = nx / norm
            next_orientation[i, 1] = ny / norm
            next_orientation[i, 2] = nz / norm
            next_orientation[i, 3] = nw / norm

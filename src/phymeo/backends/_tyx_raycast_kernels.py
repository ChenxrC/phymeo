"""One work item per beam. All surface roots, clip tests and first-hit selection run here."""

import tyx as tx


@tx.func
def _dot(x: tx.f32, y: tx.f32, z: tx.f32, u: tx.f32, v: tx.f32, w: tx.f32) -> tx.f32:
    return x * u + y * v + z * w


@tx.kernel
def first_hits(
    ray_count: tx.i32,
    patch_count: tx.i32,
    near: tx.f32,
    far: tx.f32,
    beams: tx.ndarray(tx.f32, 2),
    tags: tx.ndarray(tx.i32, 2),
    geom: tx.ndarray(tx.f32, 2),
    labels: tx.ndarray(tx.i32, 1),
    hits: tx.ndarray(tx.f32, 2),
):
    for ray in tx.range(ray_count):
        ox = beams[ray, 0]
        oy = beams[ray, 1]
        oz = beams[ray, 2]
        dx = beams[ray, 3]
        dy = beams[ray, 4]
        dz = beams[ray, 5]
        # Shear triangle vertices into this ray's transverse plane using its largest component.
        kz = 0
        if tx.abs(dy) > tx.abs(dx):
            kz = 1
        if tx.abs(dz) > tx.abs(beams[ray, 3 + kz]):
            kz = 2
        kx = 0 if kz == 2 else kz + 1
        ky = 0 if kx == 2 else kx + 1
        sx = beams[ray, 3 + kx] / beams[ray, 3 + kz]
        sy = beams[ray, 3 + ky] / beams[ray, 3 + kz]
        best = 0.0
        label = -1
        nx = 0.0
        ny = 0.0
        nz = 0.0
        issue = 0
        for patch in range(patch_count):
            kind = tags[patch, 0]
            # p0 is the sphere centre, start of the shaft, or first point of a flat surface.
            vx = ox - geom[patch, 0]
            vy = oy - geom[patch, 1]
            vz = oz - geom[patch, 2]
            ax = geom[patch, 3]
            ay = geom[patch, 4]
            az = geom[patch, 5]
            axis_ray = _dot(ax, ay, az, dx, dy, dz)
            axis_offset = _dot(ax, ay, az, vx, vy, vz)
            if kind <= 2:
                # 0 ball; 1 exterior hemisphere; 2 shaft bounded by both end planes.
                ux = dx
                uy = dy
                uz = dz
                if kind == 2:
                    ux = dy * az - dz * ay
                    uy = dz * ax - dx * az
                    uz = dx * ay - dy * ax
                    qx = vy * az - vz * ay
                    qy = vz * ax - vx * az
                    qz = vx * ay - vy * ax
                    vx = qx
                    vy = qy
                    vz = qz
                speed2 = _dot(ux, uy, uz, ux, uy, uz)
                if speed2 > 0.0:
                    mid = -_dot(ux, uy, uz, vx, vy, vz) / speed2
                    cx = vx + mid * ux
                    cy = vy + mid * uy
                    cz = vz + mid * uz
                    radius = geom[patch, 12]
                    gap = radius * radius - _dot(cx, cy, cz, cx, cy, cz)
                    if not tx.isfinite(mid) or not tx.isfinite(gap):
                        issue = 1
                    elif gap >= 0.0:
                        span = tx.sqrt(gap / speed2)
                        for root in tx.static_range(2):
                            offset = -span if root == 0 else span
                            time = mid + offset
                            # Decompose at the closest approach to preserve the radial normal,
                            # including opposite cap roots at a large distance from the source.
                            rx = cx + offset * ux
                            ry = cy + offset * uy
                            rz = cz + offset * uz
                            valid = 1
                            if kind == 1 and _dot(rx, ry, rz, ax, ay, az) < 0.0:
                                valid = 0
                            if kind == 2:
                                qx = ay * rz - az * ry
                                qy = az * rx - ax * rz
                                qz = ax * ry - ay * rx
                                rx = qx
                                ry = qy
                                rz = qz
                                height = axis_offset + time * axis_ray
                                if height < 0.0 or height > geom[patch, 13]:
                                    valid = 0
                            if not tx.isfinite(time):
                                issue = 1
                            elif valid == 1 and time >= near and time <= far:
                                if label < 0 or time < best:
                                    length = tx.sqrt(_dot(rx, ry, rz, rx, ry, rz))
                                    if length > 0.0 and tx.isfinite(length):
                                        nx = rx / length
                                        ny = ry / length
                                        nz = rz / length
                                        best = time
                                        label = tags[patch, 1]
                                    else:
                                        issue = 1
            elif axis_ray != 0.0:
                # 3 plane; 4 closed two-sided triangle. Shared edge endpoint products are
                # evaluated in the same ray frame for both triangles.
                time = -axis_offset / axis_ray
                if not tx.isfinite(time):
                    issue = 1
                elif time >= near and time <= far:
                    valid = 1
                    if kind == 4:
                        za = geom[patch, kz] - beams[ray, kz]
                        xa = geom[patch, kx] - beams[ray, kx] - sx * za
                        ya = geom[patch, ky] - beams[ray, ky] - sy * za
                        zb = geom[patch, 6 + kz] - beams[ray, kz]
                        xb = geom[patch, 6 + kx] - beams[ray, kx] - sx * zb
                        yb = geom[patch, 6 + ky] - beams[ray, ky] - sy * zb
                        zc = geom[patch, 9 + kz] - beams[ray, kz]
                        xc = geom[patch, 9 + kx] - beams[ray, kx] - sx * zc
                        yc = geom[patch, 9 + ky] - beams[ray, ky] - sy * zc
                        u = xb * yc - yb * xc
                        v = xc * ya - yc * xa
                        w = xa * yb - ya * xb
                        if (u < 0.0 or v < 0.0 or w < 0.0) and (u > 0.0 or v > 0.0 or w > 0.0):
                            valid = 0
                        elif u == 0.0 and v == 0.0 and w == 0.0:
                            issue = 1
                    if valid == 1 and (label < 0 or time < best):
                        sign = -1.0 if axis_ray > 0.0 else 1.0
                        nx = sign * ax
                        ny = sign * ay
                        nz = sign * az
                        best = time
                        label = tags[patch, 1]
        labels[ray] = -2 if issue != 0 else label
        hits[ray, 0] = best
        hits[ray, 1] = nx
        hits[ray, 2] = ny
        hits[ray, 3] = nz

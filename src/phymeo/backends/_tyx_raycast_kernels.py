"""One work item per beam. All surface roots, clip tests and first-hit selection run here."""

import tyx as tx


@tx.func
def _dot(x: tx.f32, y: tx.f32, z: tx.f32, u: tx.f32, v: tx.f32, w: tx.f32) -> tx.f32:
    return x * u + y * v + z * w


@tx.kernel
def first_hits(
    ray_count: tx.i32,
    patch_count: tx.i32,
    routed: tx.i32,
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
        begin = 0
        end = patch_count
        lower = near
        upper = far
        if routed != 0:
            begin = tags[patch_count + ray, 0]
            end = tags[patch_count + ray, 1]
            lower = beams[ray, 6]
            upper = beams[ray, 7]
        for patch in range(begin, end):
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
                            elif valid == 1 and time >= lower and time <= upper:
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
            elif kind <= 4 and axis_ray != 0.0:
                # 3 plane; 4 closed two-sided triangle. Shared edge endpoint products are
                # evaluated in the same ray frame for both triangles.
                time = -axis_offset / axis_ray
                if not tx.isfinite(time):
                    issue = 1
                elif time >= lower and time <= upper:
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
            elif kind >= 5:
                # 5 cylinder; 6 cone (apex at +half); 7 triaxial ellipsoid. Transform vectors,
                # solve in a metric ray frame and rotate the implicit gradient back to world.
                lx = axis_offset
                ux = axis_ray
                ly = _dot(vx, vy, vz, geom[patch, 6], geom[patch, 7], geom[patch, 8])
                uy = _dot(dx, dy, dz, geom[patch, 6], geom[patch, 7], geom[patch, 8])
                lz = _dot(vx, vy, vz, geom[patch, 9], geom[patch, 10], geom[patch, 11])
                uz = _dot(dx, dy, dz, geom[patch, 9], geom[patch, 10], geom[patch, 11])
                # Expand about the point nearest the shape centre, so a distant sensor does not
                # subtract two large, nearly equal terms in the discriminant.
                shift = -_dot(lx, ly, lz, ux, uy, uz) / _dot(ux, uy, uz, ux, uy, uz)
                cx = lx + shift * ux
                cy = ly + shift * uy
                cz = lz + shift * uz
                r = geom[patch, 12]
                half = geom[patch, 13]
                oxu = cx / r
                oyu = cy / r
                dxu = ux / r
                dyu = uy / r
                a = dxu * dxu + dyu * dyu
                b = dxu * oxu + dyu * oyu
                c = oxu * oxu + oyu * oyu - 1.0
                generator = 0
                if kind == 5:
                    # A ray parallel to the axis may lie on the whole finite barrel.
                    if a == 0.0 and lx * lx + ly * ly == r * r:
                        generator = 1
                elif kind == 6:
                    zscale = 2.0 * half
                    ozc = (half - cz) / zscale
                    dzc = -uz / zscale
                    radial = tx.sqrt(a)
                    # A factored difference retains opposite generators and the linear case.
                    a = (radial - tx.abs(dzc)) * (radial + tx.abs(dzc))
                    b = b - ozc * dzc
                    c = oxu * oxu + oyu * oyu - ozc * ozc
                    if a == 0.0:
                        gx = lx / r
                        gy = ly / r
                        gz = (half - lz) / zscale
                        if gx * gx + gy * gy == gz * gz and gx * dxu + gy * dyu == gz * dzc:
                            generator = 1
                else:
                    oyu = cy / half
                    dyu = uy / half
                    ozu = cz / geom[patch, 14]
                    dzu = uz / geom[patch, 14]
                    a = _dot(dxu, dyu, dzu, dxu, dyu, dzu)
                    b = _dot(oxu, oyu, ozu, dxu, dyu, dzu)
                    c = _dot(oxu, oyu, ozu, oxu, oyu, ozu) - 1.0
                disc = b * b - a * c
                if not tx.isfinite(disc):
                    issue = 1
                for feature in tx.static_range(5):
                    # Side roots precede a possible coincident side interval and the end discs.
                    valid = 0
                    delta = 0.0
                    time = 0.0
                    cap_sign = 0.0
                    if feature < 2 and generator == 0:
                        if a != 0.0 and disc >= 0.0:
                            root_span = tx.sqrt(disc)
                            q = -b - root_span if b >= 0.0 else -b + root_span
                            delta = q / a
                            if feature == 1 and q != 0.0:
                                delta = c / q
                            valid = 1
                        elif a == 0.0 and b != 0.0 and feature == 0:
                            delta = -c / (2.0 * b)
                            valid = 1
                        time = shift + delta
                    elif feature == 2 and generator == 1 and uz != 0.0:
                        low = shift + (-half - cz) / uz
                        high = shift + (half - cz) / uz
                        time = tx.max(lower, tx.min(low, high))
                        delta = time - shift
                        if time <= tx.max(low, high):
                            valid = 1
                    elif feature >= 3 and kind != 7 and uz != 0.0:
                        if kind == 5 or feature == 3:
                            cap_sign = -1.0 if feature == 3 else 1.0
                            delta = (cap_sign * half - cz) / uz
                            time = shift + delta
                            valid = 1
                    if valid == 1 and not tx.isfinite(time):
                        issue = 1
                    if valid == 1 and time >= lower and time <= upper and (label < 0 or time < best):
                        px = cx + delta * ux
                        py = cy + delta * uy
                        pz = cz + delta * uz
                        gx = px
                        gy = py
                        gz = 0.0
                        if cap_sign != 0.0:
                            if px * px + py * py > r * r:
                                valid = 0
                            gx = 0.0
                            gy = 0.0
                            gz = cap_sign
                        elif kind == 7:
                            gx = px / (r * r)
                            gy = py / (half * half)
                            zsize = geom[patch, 14]
                            gz = pz / (zsize * zsize)
                        else:
                            if pz < -half or pz > half:
                                valid = 0
                            if kind == 6:
                                slope = r / (2.0 * half)
                                gz = slope * slope * (half - pz)
                        if valid == 1:
                            wx = gx * ax + gy * geom[patch, 6] + gz * geom[patch, 9]
                            wy = gx * ay + gy * geom[patch, 7] + gz * geom[patch, 10]
                            wz = gx * az + gy * geom[patch, 8] + gz * geom[patch, 11]
                            length = tx.sqrt(_dot(wx, wy, wz, wx, wy, wz))
                            if not tx.isfinite(length):
                                issue = 1
                            elif length == 0.0 and (
                                kind != 6 or px != 0.0 or py != 0.0 or pz != half
                            ):
                                issue = 1
                            else:
                                nx = wx / length if length > 0.0 else 0.0
                                ny = wy / length if length > 0.0 else 0.0
                                nz = wz / length if length > 0.0 else 0.0
                                best = time
                                label = tags[patch, 1]
        labels[ray] = -2 if issue != 0 else label
        hits[ray, 0] = best
        hits[ray, 1] = nx
        hits[ray, 2] = ny
        hits[ray, 3] = nz

"""Selected device rays against shared local geometry and borrowed endpoint poses."""

import tyx as tx

from ._tyx_raycast_kernels import cast_patches

_MISS = float("inf")


@tx.func
def rotate(q: tx.vec4f, v: tx.vec3f) -> tx.vec3f:
    axis = tx.vec3f(q[0], q[1], q[2])
    return v + 2.0 * tx.cross(axis, tx.cross(axis, v) + q[3] * v)


@tx.kernel(gradient="unsupported")
def query(
    selected: tx.Array[tx.i32],
    count: tx.Array[tx.i32],
    gate: tx.Array[tx.i32],
    native_error: tx.Array[tx.i32],
    bodies: tx.Array[tx.i32],
    local_pose: tx.ndarray(tx.f32, 2),
    ranges: tx.ndarray(tx.i32, 2),
    handles: tx.Array[tx.u64],
    tags: tx.ndarray(tx.i32, 2),
    geom: tx.ndarray(tx.f32, 2),
    positions: tx.ndarray(tx.f32, 3),
    rotations: tx.ndarray(tx.f32, 3),
    origins: tx.ndarray(tx.f32, 3),
    directions: tx.ndarray(tx.f32, 3),
    labels: tx.Array[tx.i32],
    hits: tx.ndarray(tx.f32, 2),
    errors: tx.Array[tx.i32],
    rows: tx.i32,
    ray_count: tx.i32,
    near: tx.f32,
    far: tx.f32,
):
    for i in tx.range(rows * ray_count):
        if gate[0] == 0 and native_error[0] == 0:
            row, ray = i // ray_count, i % ray_count
            if row < count[0]:
                world = selected[row]
                if world < 0 or world >= positions.shape[0]:
                    tx.atomic_or(errors[0], 1)
                else:
                    origin = tx.vec3f(
                        origins[row, ray, 0], origins[row, ray, 1], origins[row, ray, 2]
                    )
                    direction = tx.vec3f(
                        directions[row, ray, 0], directions[row, ray, 1], directions[row, ray, 2]
                    )
                    valid = True
                    for j in range(3):
                        if not tx.isfinite(origin[j]) or not tx.isfinite(direction[j]):
                            valid = False
                    if tx.abs(tx.dot(direction, direction) - 1.0) > 1.0e-5:
                        valid = False
                    if not valid:
                        tx.atomic_or(errors[0], 1)
                    else:
                        best, label = far, -1
                        normal = tx.vec3f(0.0, 0.0, 0.0)
                        for shape in range(bodies.shape[0]):
                            body = bodies[shape]
                            q = tx.vec4f(
                                -rotations[world, body, 1],
                                -rotations[world, body, 2],
                                -rotations[world, body, 3],
                                rotations[world, body, 0],
                            )
                            lq = tx.vec4f(
                                -local_pose[shape, 3],
                                -local_pose[shape, 4],
                                -local_pose[shape, 5],
                                local_pose[shape, 6],
                            )
                            offset = tx.vec3f(
                                positions[world, body, 0],
                                positions[world, body, 1],
                                positions[world, body, 2],
                            )
                            lp = tx.vec3f(
                                local_pose[shape, 0], local_pose[shape, 1], local_pose[shape, 2]
                            )
                            o = rotate(lq, rotate(q, origin - offset) - lp)
                            d = rotate(lq, rotate(q, direction))
                            found = -1
                            hit = tx.vec4f(0.0, 0.0, 0.0, 0.0)
                            clipped = o + near * d
                            safe = True
                            for j in range(3):
                                if (
                                    not tx.isfinite(o[j])
                                    or not tx.isfinite(d[j])
                                    or not tx.isfinite(clipped[j])
                                ):
                                    safe = False
                            if not safe:
                                tx.atomic_or(errors[0], 1)
                            elif handles[shape] != tx.u64(0):
                                mesh_hit = tx.mesh_query_ray(
                                    handles[shape], o + near * d, d, 3.402823466e38
                                )
                                if mesh_hit.result and mesh_hit.t + near <= far:
                                    found = shape
                                    n = mesh_hit.normal
                                    if tx.dot(n, d) > 0.0:
                                        n = -n
                                    hit = tx.vec4f(mesh_hit.t + near, n[0], n[1], n[2])
                            else:
                                found, hit = cast_patches(
                                    o, d, near, far, tags, geom, ranges[shape, 0], ranges[shape, 1]
                                )
                            if found == -2:
                                tx.atomic_or(errors[0], 1)
                            elif found >= 0 and (label < 0 or hit[0] < best):
                                best, label = hit[0], shape
                                n = tx.vec3f(hit[1], hit[2], hit[3])
                                q = tx.vec4f(-q[0], -q[1], -q[2], q[3])
                                lq = tx.vec4f(-lq[0], -lq[1], -lq[2], lq[3])
                                normal = rotate(q, rotate(lq, n))
                                # Prepared observations use one ray-facing policy for every surface.
                                if tx.dot(normal, direction) > 0.0:
                                    normal = -normal
                                if not tx.isfinite(best):
                                    tx.atomic_or(errors[0], 1)
                                for j in range(3):
                                    if not tx.isfinite(normal[j]):
                                        tx.atomic_or(errors[0], 1)
                        labels[i] = label
                        hits[i, 0] = best if label >= 0 else _MISS
                        for j in range(3):
                            hits[i, j + 1] = normal[j]

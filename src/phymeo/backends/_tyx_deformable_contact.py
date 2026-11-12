"""D3 force transpose only: the original normal law and physical integrators stay unchanged."""

import tyx as tx


@tx.kernel
def clear_response_poses(count: tx.i32, poses: tx.ndarray(tx.f32, 2)):
    for i in tx.range(count):
        for j in tx.static_range(12):
            poses[i, j] = 0.0


@tx.kernel
def transpose_response(
    worlds: tx.i32,
    sites: tx.i32,
    offsets: tx.ndarray(tx.i32, 1),
    identities: tx.ndarray(tx.i32, 2),
    barycentric: tx.ndarray(tx.f32, 2),
    levers: tx.ndarray(tx.f32, 2),
    forces: tx.ndarray(tx.f32, 2),
    node_force: tx.ndarray(tx.f32, 2),
    body_wrench: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    # One world owns all its reductions: no atomics or cross-world writes.
    for world in tx.range(worlds):
        valid = 1
        for node in range(offsets[world], offsets[world + 1]):
            for axis in tx.static_range(3):
                node_force[node, axis] = 0.0
        for axis in tx.static_range(6):
            body_wrench[world, axis] = 0.0
        for site in range(sites):
            if identities[site, 0] == world:
                if identities[site, 4] != world:
                    valid = 0
                for corner in tx.static_range(3):
                    node = identities[site, corner + 1]
                    if node < offsets[world] or node >= offsets[world + 1]:
                        valid = 0
                    else:
                        for axis in tx.static_range(3):
                            value = (
                                node_force[node, axis]
                                + barycentric[site, corner] * forces[site, axis + 2]
                            )
                            node_force[node, axis] = value
                            if not tx.isfinite(value):
                                valid = 0
                fx = -forces[site, 2]
                fy = -forces[site, 3]
                fz = -forces[site, 4]
                rx = levers[site, 0]
                ry = levers[site, 1]
                rz = levers[site, 2]
                body_wrench[world, 0] = body_wrench[world, 0] + fx
                body_wrench[world, 1] = body_wrench[world, 1] + fy
                body_wrench[world, 2] = body_wrench[world, 2] + fz
                body_wrench[world, 3] = body_wrench[world, 3] + ry * fz - rz * fy
                body_wrench[world, 4] = body_wrench[world, 4] + rz * fx - rx * fz
                body_wrench[world, 5] = body_wrench[world, 5] + rx * fy - ry * fx
        for axis in tx.static_range(6):
            if not tx.isfinite(body_wrench[world, axis]):
                valid = 0
        status[world] = valid

"""M-B1 contact-force transpose over complete per-world body slots.

The calibrated law and physical integrators are unchanged. A world owns every
node/body reduction in its two offset ranges; no atomics or cross-world writes.
"""

import tyx as tx


@tx.kernel
def transpose_multibody_response(
    worlds: tx.i32,
    sites: tx.i32,
    offsets: tx.ndarray(tx.i32, 1),
    body_offsets: tx.ndarray(tx.i32, 1),
    identities: tx.ndarray(tx.i32, 2),
    barycentric: tx.ndarray(tx.f32, 2),
    levers: tx.ndarray(tx.f32, 2),
    forces: tx.ndarray(tx.f32, 2),
    node_force: tx.ndarray(tx.f32, 2),
    body_wrench: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for world in tx.range(worlds):
        valid = 1
        for node in range(offsets[world], offsets[world + 1]):
            for axis in tx.static_range(3):
                node_force[node, axis] = 0.0
        for body in range(body_offsets[world], body_offsets[world + 1]):
            for axis in tx.static_range(6):
                body_wrench[body, axis] = 0.0
        for site in range(sites):
            if identities[site, 0] == world:
                body = identities[site, 4]
                if body < body_offsets[world] or body >= body_offsets[world + 1]:
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
                # An invalid body address never participates in any read or
                # write. In particular, a valid slot in another world is not
                # accepted merely because it lies within the allocated array.
                if body >= body_offsets[world] and body < body_offsets[world + 1]:
                    fx = -forces[site, 2]
                    fy = -forces[site, 3]
                    fz = -forces[site, 4]
                    rx = levers[site, 0]
                    ry = levers[site, 1]
                    rz = levers[site, 2]
                    body_wrench[body, 0] = body_wrench[body, 0] + fx
                    body_wrench[body, 1] = body_wrench[body, 1] + fy
                    body_wrench[body, 2] = body_wrench[body, 2] + fz
                    body_wrench[body, 3] = body_wrench[body, 3] + ry * fz - rz * fy
                    body_wrench[body, 4] = body_wrench[body, 4] + rz * fx - rx * fz
                    body_wrench[body, 5] = body_wrench[body, 5] + rx * fy - ry * fx
        for body in range(body_offsets[world], body_offsets[world + 1]):
            for axis in tx.static_range(6):
                if not tx.isfinite(body_wrench[body, axis]):
                    valid = 0
        status[world] = valid

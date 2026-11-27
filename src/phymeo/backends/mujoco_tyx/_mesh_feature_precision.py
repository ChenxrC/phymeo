"""Refine retained contact feature normals without rounded world vertices.

Static edges/faces are compiled from all original small-hull vertices in FP64
and stored as two FP32 lanes. Per-pose work is entirely on device. This is not
a CPU narrow phase: no pose, support selection or contact decision runs here
on the host. Candidate axes come from the original local topology, not the
possibly non-supporting triangles retained by GJK/EPA.
"""

import numpy as np
import tyx as tx

from ._mesh_query_kernels import sum_error


def compile_features(catalog):
    edge_offsets, face_offsets = [0], [0]
    edges, faces = [], []
    for g in range(catalog.ngeom):
        p = catalog.points[catalog.offsets[g] : catalog.offsets[g + 1]].astype(np.float64)
        n = len(p)
        if n <= 16:
            e = p[None, :, :] - p[:, None, :]
            f = np.cross(e[:, :, None, :], e[:, None, :, :]).reshape(-1, 3)
            length = np.linalg.norm(f, axis=1)
            f = f[length > 1e-30] / length[length > 1e-30, None]
            # Only original supporting faces, not arbitrary EPA triangles.
            # All original vertices participate. The tolerance covers FP64
            # static dot-product roundoff, never a geometric contact margin.
            origins = np.repeat(p, n * n, axis=0)[length > 1e-30]
            distances = np.einsum("ik,ijk->ij", f, p[None] - origins[:, None])
            tol = 64 * np.finfo(float).eps * np.max(np.ptp(p, axis=0))
            positive = distances.min(axis=1) >= -tol
            negative = distances.max(axis=1) <= tol
            f = f[positive | negative]
            f[positive[positive | negative]] *= -1
            f = np.unique(f, axis=0)
            edge = e[np.triu_indices(n, k=1)]
            size = np.linalg.norm(edge, axis=1)
            good = size > 1e-30
            edge[good] /= size[good, None]
            edge[~good] = 0
            edges.append(edge)
            faces.append(f)
            edge_offsets.append(edge_offsets[-1] + len(edge))
            face_offsets.append(face_offsets[-1] + len(f))
        else:
            edge_offsets.append(edge_offsets[-1])
            face_offsets.append(face_offsets[-1])
    result = dict(
        edge_offsets=np.asarray(edge_offsets, np.int32),
        face_offsets=np.asarray(face_offsets, np.int32),
    )
    for name, parts in (("edge_axes", edges), ("face_axes", faces)):
        data = np.concatenate(parts) if parts else np.zeros((1, 3))
        high = data.astype(np.float32)
        result[name] = high
        result[name + "_low"] = (data - high.astype(float)).astype(np.float32)
    return result


@tx.kernel
def refine_contact_features(
    count: tx.i32,
    first: tx.i32,
    ngeom: tx.i32,
    offsets: tx.ndarray(tx.i32, 1),
    local: tx.ndarray(tx.f32, 2),
    positions: tx.ndarray(tx.f32, 3),
    rotations: tx.ndarray(tx.f32, 4),
    positions_low: tx.ndarray(tx.f32, 3),
    rotations_low: tx.ndarray(tx.f32, 4),
    edge_offsets: tx.ndarray(tx.i32, 1),
    face_offsets: tx.ndarray(tx.i32, 1),
    edge_axes: tx.ndarray(tx.f32, 2),
    edge_axes_low: tx.ndarray(tx.f32, 2),
    face_axes: tx.ndarray(tx.f32, 2),
    face_axes_low: tx.ndarray(tx.f32, 2),
    pairs: tx.ndarray(tx.i32, 2),
    frames: tx.ndarray(tx.f32, 2),
    meta: tx.ndarray(tx.i32, 2),
    gap: tx.ndarray(tx.f32, 2),
    normal: tx.ndarray(tx.f32, 2),
    scratch: tx.ndarray(tx.f32, 4),
    corrected: tx.ndarray(tx.i32, 1),
    feature_axis: tx.ndarray(tx.f32, 3),
):
    for lane in tx.range(count):
        out = first + lane
        corrected[out] = 0
        for axis in range(3):
            feature_axis[out, axis, 0] = 0.0
            feature_axis[out, axis, 1] = 0.0
        if meta[out, 0] == 1 and gap[out, 0] < 0.0:
            ga = pairs[out, 0] % ngeom
            gb = pairs[out, 1] % ngeom
            world = pairs[out, 0] // ngeom
            na = offsets[ga + 1] - offsets[ga]
            nb = offsets[gb + 1] - offsets[gb]
            if na <= 16 and nb <= 16:
                best_gap = -3.402823e38
                best_x = 0.0
                best_y = 0.0
                best_z = 0.0
                found = 0
                # EPA witnesses do not prove which face is active. Consider
                # every original supporting face and every vertex-pair edge axis.
                # Extra diagonal axes are redundant SAT bounds, not new geometry.
                nfa = face_offsets[ga + 1] - face_offsets[ga]
                nfb = face_offsets[gb + 1] - face_offsets[gb]
                nf = nfa + nfb
                nea = edge_offsets[ga + 1] - edge_offsets[ga]
                neb = edge_offsets[gb + 1] - edge_offsets[gb]
                for feature in range(nf + nea * neb):
                    face_side = -1
                    valid_feature = 0
                    if feature < nfa:
                        face_side = 0
                        valid_feature = 1
                    elif feature < nf:
                        face_side = 1
                        valid_feature = 1
                    elif feature >= nf:
                        valid_feature = 1
                    corrected[out] = 0
                    for side in range(2):
                        g = ga
                        if side == 1:
                            g = gb
                        if valid_feature == 1 and (face_side == -1 or side == face_side):
                            slot = edge_offsets[g] + (feature - nf) // neb
                            if side == 1:
                                slot = edge_offsets[g] + (feature - nf) % neb
                            if face_side >= 0:
                                slot = face_offsets[g] + feature
                                if side == 1:
                                    slot = face_offsets[g] + feature - nfa
                            if valid_feature == 1:
                                for axis in range(3):
                                    hi = 0.0
                                    lo = 0.0
                                    for col in range(3):
                                        eh = edge_axes[0, col]
                                        el = edge_axes_low[0, col]
                                        if face_side == -1:
                                            eh = edge_axes[slot, col]
                                            el = edge_axes_low[slot, col]
                                        else:
                                            eh = face_axes[slot, col]
                                            el = face_axes_low[slot, col]
                                        rh = rotations[world, g, axis, col]
                                        rl = rotations_low[world, g, axis, col]
                                        prod = tx.mul_rn(eh, rh)
                                        pe = tx.fma(eh, rh, -prod) + eh * rl + el * rh + el * rl
                                        total = tx.add_rn(hi, prod)
                                        lo = lo + sum_error(hi, prod, total) + pe
                                        hi = total
                                    total = tx.add_rn(hi, lo)
                                    scratch[lane, side, axis, 0] = total
                                    scratch[lane, side, axis, 1] = sum_error(hi, lo, total)
                                if corrected[out] != -1:
                                    corrected[out] = 1
                            else:
                                corrected[out] = -1
                    square = 0.0
                    alignment = 0.0
                    if corrected[out] == 1:
                        for axis in range(3):
                            value = 0.0
                            value_low = 0.0
                            if face_side >= 0:
                                value = scratch[lane, face_side, axis, 0]
                                value_low = scratch[lane, face_side, axis, 1]
                            else:
                                j = (axis + 1) % 3
                                k = (axis + 2) % 3
                                a = scratch[lane, 0, j, 0]
                                b = scratch[lane, 1, k, 0]
                                c = scratch[lane, 0, k, 0]
                                d = scratch[lane, 1, j, 0]
                                prod_ab = tx.mul_rn(a, b)
                                prod_cd = tx.mul_rn(c, d)
                                value = tx.sub_rn(prod_ab, prod_cd)
                                value_low = (
                                    sum_error(prod_ab, -prod_cd, value)
                                    + tx.fma(a, b, -prod_ab)
                                    - tx.fma(c, d, -prod_cd)
                                    + a * scratch[lane, 1, k, 1]
                                    + b * scratch[lane, 0, j, 1]
                                    - c * scratch[lane, 1, j, 1]
                                    - d * scratch[lane, 0, k, 1]
                                    + scratch[lane, 0, j, 1] * scratch[lane, 1, k, 1]
                                    - scratch[lane, 0, k, 1] * scratch[lane, 1, j, 1]
                                )
                                total = tx.add_rn(value, value_low)
                                value_low = sum_error(value, value_low, total)
                                value = total
                            scratch[lane, 2, axis, 0] = value
                            scratch[lane, 2, axis, 1] = value_low
                            square = square + value * value
                            alignment = alignment + value * normal[out, axis]
                        length = tx.sqrt(square)
                        if length > 1e-12:
                            direction = 1.0
                            if alignment < 0.0:
                                direction = -1.0
                            difference = 0.0
                            for axis in range(3):
                                value = direction * scratch[lane, 2, axis, 0] / length
                                delta = value - normal[out, axis]
                                difference = difference + delta * delta
                            # Retained-feature correction must agree with the certified
                            # GJK normal's existing 2e-4 numerical comparison envelope.
                            if difference <= 4e-8:
                                amax_hi = -3.402823e38
                                amax_lo = 0.0
                                bmin_hi = 3.402823e38
                                bmin_lo = 0.0
                                for side in range(2):
                                    g = ga
                                    if side == 1:
                                        g = gb
                                    for v in range(offsets[g], offsets[g + 1]):
                                        projection = 0.0
                                        perr = 0.0
                                        for axis in range(3):
                                            ph = positions[world, g, axis]
                                            origin = -frames[lane, axis]
                                            hi = tx.add_rn(ph, origin)
                                            lo = positions_low[world, g, axis] + sum_error(
                                                ph, origin, hi
                                            )
                                            for col in range(3):
                                                rh = rotations[world, g, axis, col]
                                                vv = local[v, col]
                                                prod = tx.mul_rn(rh, vv)
                                                pe = (
                                                    tx.fma(rh, vv, -prod)
                                                    + rotations_low[world, g, axis, col] * vv
                                                )
                                                total = tx.add_rn(hi, prod)
                                                lo = lo + sum_error(hi, prod, total) + pe
                                                hi = total
                                            # Project with the original two-lane axis.
                                            # Rounding the normalized axis before support
                                            # selection loses near-coplanar depth bits.
                                            axis_n = direction * scratch[lane, 2, axis, 0]
                                            axis_low = direction * scratch[lane, 2, axis, 1]
                                            prod = tx.mul_rn(hi, axis_n)
                                            pe = (
                                                tx.fma(hi, axis_n, -prod)
                                                + lo * axis_n
                                                + hi * axis_low
                                                + lo * axis_low
                                            )
                                            total = tx.add_rn(projection, prod)
                                            perr = perr + sum_error(projection, prod, total) + pe
                                            projection = total
                                        if side == 0:
                                            if (projection - amax_hi) + (perr - amax_lo) > 0.0:
                                                amax_hi = projection
                                                amax_lo = perr
                                        else:
                                            if (projection - bmin_hi) + (perr - bmin_lo) < 0.0:
                                                bmin_hi = projection
                                                bmin_lo = perr
                                hi = tx.sub_rn(bmin_hi, amax_hi)
                                value = (
                                    hi + (sum_error(bmin_hi, -amax_hi, hi) + bmin_lo - amax_lo)
                                ) / (frames[lane, 3] * length)
                                # No epsilon activation: only an intersecting support
                                # interval may replace an already intersecting query.
                                if (
                                    value < 0.0
                                    and value > best_gap
                                    and tx.abs(value - gap[out, 0]) * frames[lane, 3] <= 5e-6
                                ):
                                    best_x = direction * scratch[lane, 2, 0, 0] / length
                                    best_y = direction * scratch[lane, 2, 1, 0] / length
                                    best_z = direction * scratch[lane, 2, 2, 0] / length
                                    best_gap = value
                                    for axis in range(3):
                                        feature_axis[out, axis, 0] = (
                                            direction * scratch[lane, 2, axis, 0]
                                        )
                                        feature_axis[out, axis, 1] = (
                                            direction * scratch[lane, 2, axis, 1]
                                        )
                                    found = 1
                corrected[out] = 0
                if found == 1:
                    normal[out, 0] = best_x
                    normal[out, 1] = best_y
                    normal[out, 2] = best_z
                    gap[out, 0] = best_gap
                    gap[out, 1] = tx.min(gap[out, 1], best_gap)
                    gap[out, 2] = tx.max(gap[out, 2], best_gap)
                    corrected[out] = 2

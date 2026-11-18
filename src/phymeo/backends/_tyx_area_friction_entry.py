"""Resident same-entry pack for the independent final-normal friction kernel.

This is a low-level diagnostic adapter, not a live MuJoCo integration API. It
never solves, advances state, or publishes an observation. The caller provides
same-entry buffers and invokes pack and solve in the same active TYX runtime.
Array metadata/device checks do not authenticate a source owner or its phase.

A failed lane has zero outputs, sizes[nv, ns] == (0, 0), and a persistent
pack_status. Thus the existing solve_batch rejects that lane before numerical
work, without a HOST readback. The caller must reject the complete batch if any
pack_status or solver status is nonzero. Padding lanes are zero and not failures.
"""

import numpy as np
import tyx as tx

from .mujoco_tyx._area_friction_layout import MAX_LIVE_DOF, friction_width

PACK_OK = 0
PACK_UPSTREAM_FAILED = 1
PACK_INDEX_ERROR = 2
PACK_DOMAIN_ERROR = 3
PACK_ARITHMETIC_ERROR = 4


@tx.kernel
def _pack_entry_kernel(
    capacity: tx.i32,
    count: tx.i32,
    nv: tx.i32,
    width: tx.i32,
    packed_width: tx.i32,
    nbody: tx.i32,
    npair: tx.i32,
    area_capacity: tx.i32,
    h: tx.f32,
    source_status: tx.ndarray(tx.i32, 1),
    source_mass: tx.ndarray(tx.f32, 3),
    source_vectors: tx.ndarray(tx.f32, 2),
    source_v: tx.ndarray(tx.f32, 2),
    source_bias: tx.ndarray(tx.f32, 2),
    source_passive: tx.ndarray(tx.f32, 2),
    source_applied: tx.ndarray(tx.f32, 2),
    motion: tx.ndarray(tx.f32, 4),
    body_jacobian: tx.ndarray(tx.f32, 4),
    area_input: tx.ndarray(tx.f32, 3),
    area_pair_index: tx.ndarray(tx.i32, 2),
    area_sizes: tx.ndarray(tx.i32, 1),
    area_body_ids: tx.ndarray(tx.i32, 2),
    pair_laws: tx.ndarray(tx.f32, 2),
    sizes: tx.ndarray(tx.i32, 2),
    mass: tx.ndarray(tx.f32, 3),
    inputs: tx.ndarray(tx.f32, 3),
    steps: tx.ndarray(tx.f32, 1),
    samples: tx.ndarray(tx.f32, 3),
    ja: tx.ndarray(tx.f32, 4),
    jb: tx.ndarray(tx.f32, 4),
    pack_status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(capacity):
        pack_status[w] = 0
        steps[w] = 0.0
        for i in range(4):
            sizes[w, i] = 0
        for i in range(packed_width):
            for j in range(packed_width):
                mass[w, i, j] = 0.0
            for j in range(3):
                inputs[w, j, i] = 0.0
        for s in range(128):
            for i in range(10):
                samples[w, s, i] = 0.0
            for i in range(3):
                for d in range(packed_width):
                    ja[w, s, i, d] = 0.0
                    jb[w, s, i, d] = 0.0
        code = 0
        ns = 0
        if w < count:
            # Preserve upstream failure, and never inspect its stale payload.
            if source_status[w] != 0:
                code = 1
            if code == 0:
                ns = area_sizes[w]
                if ns < 0 or ns > 128 or ns > area_capacity:
                    code = 2
            if code == 0:
                if not tx.isfinite(h) or h <= 0.0:
                    code = 3
                for d in range(nv):
                    if (
                        not tx.isfinite(source_vectors[1, d])
                        or source_vectors[1, d] < 0.0
                        or not tx.isfinite(source_v[w, d])
                        or not tx.isfinite(source_bias[w, d])
                        or not tx.isfinite(source_passive[w, d])
                        or not tx.isfinite(source_applied[w, d])
                    ):
                        code = 3
                    for j in range(nv):
                        if (
                            not tx.isfinite(source_mass[w, d, j])
                            or source_mass[w, d, j] != source_mass[w, j, d]
                        ):
                            code = 3
                    if source_mass[w, d, d] <= 0.0:
                        code = 3
                # Only fields used for point Jacobians are physical inputs here.
                for b in range(nbody):
                    for i in range(3):
                        if not tx.isfinite(motion[w, b, 5, i]):
                            code = 3
                    for i in range(6):
                        for d in range(nv):
                            if not tx.isfinite(body_jacobian[w, b, i, d]):
                                code = 3
            if code == 0:
                # Validate every declared pair, including pairs with empty area.
                for p in range(npair):
                    ba = area_body_ids[p, 0]
                    bb = area_body_ids[p, 1]
                    if ba < 0 or ba >= nbody or bb < 0 or bb >= nbody or ba == bb:
                        code = 2
                    if (
                        not tx.isfinite(pair_laws[p, 0])
                        or pair_laws[p, 0] < 0.0
                        or not tx.isfinite(pair_laws[p, 1])
                        or pair_laws[p, 1] <= 0.0
                    ):
                        if code == 0:
                            code = 3
            if code == 0:
                # No indirect body access occurs before ALL sample indices pass.
                for s in range(ns):
                    p = area_pair_index[w, s]
                    if p < 0 or p >= npair:
                        code = 2
                    for i in range(8):
                        if not tx.isfinite(area_input[w, s, i]):
                            if code == 0:
                                code = 3
                    norm2 = 0.0
                    for i in range(3):
                        norm2 = norm2 + area_input[w, s, 3 + i] * area_input[w, s, 3 + i]
                    if (
                        not tx.isfinite(norm2)
                        or tx.abs(norm2 - 1.0) > 0.000002
                        or area_input[w, s, 6] <= 0.0
                        or area_input[w, s, 7] > 0.0
                    ):
                        if code == 0:
                            code = 3
            if code == 0:
                sizes[w, 0] = nv
                sizes[w, 1] = ns
                sizes[w, 2] = 24
                sizes[w, 3] = 12
                steps[w] = h
                for d in range(nv):
                    for j in range(nv):
                        mass[w, d, j] = source_mass[w, d, j]
                    inputs[w, 0, d] = source_vectors[1, d]
                    inputs[w, 1, d] = source_v[w, d]
                    force = (
                        source_applied[w, d]
                        + source_passive[w, d]
                        - source_bias[w, d]
                        + source_vectors[1, d] * source_v[w, d]
                    )
                    inputs[w, 2, d] = force
                    if not tx.isfinite(force):
                        code = 4
                for s in range(ns):
                    p = area_pair_index[w, s]
                    ba = area_body_ids[p, 0]
                    bb = area_body_ids[p, 1]
                    for i in range(8):
                        samples[w, s, i] = area_input[w, s, i]
                    samples[w, s, 8] = pair_laws[p, 0]
                    samples[w, s, 9] = pair_laws[p, 1]
                    for i in range(3):
                        j = (i + 1) % 3
                        k = (i + 2) % 3
                        ra_j = area_input[w, s, j] - motion[w, ba, 5, j]
                        ra_k = area_input[w, s, k] - motion[w, ba, 5, k]
                        rb_j = area_input[w, s, j] - motion[w, bb, 5, j]
                        rb_k = area_input[w, s, k] - motion[w, bb, 5, k]
                        if (
                            not tx.isfinite(ra_j)
                            or not tx.isfinite(ra_k)
                            or not tx.isfinite(rb_j)
                            or not tx.isfinite(rb_k)
                        ):
                            code = 4
                        for d in range(nv):
                            value_a = (
                                body_jacobian[w, ba, i, d]
                                + body_jacobian[w, ba, 3 + j, d] * ra_k
                                - body_jacobian[w, ba, 3 + k, d] * ra_j
                            )
                            value_b = (
                                body_jacobian[w, bb, i, d]
                                + body_jacobian[w, bb, 3 + j, d] * rb_k
                                - body_jacobian[w, bb, 3 + k, d] * rb_j
                            )
                            ja[w, s, i, d] = value_a
                            jb[w, s, i, d] = value_b
                            if not tx.isfinite(value_a) or not tx.isfinite(value_b):
                                code = 4
            if code != 0:
                # Device-side solver gate: nv=0 makes solve_batch fail before
                # accessing sample data. Clear partial assembly as well.
                steps[w] = 0.0
                for i in range(4):
                    sizes[w, i] = 0
                for i in range(packed_width):
                    for j in range(packed_width):
                        mass[w, i, j] = 0.0
                    for j in range(3):
                        inputs[w, j, i] = 0.0
                for s in range(128):
                    for i in range(10):
                        samples[w, s, i] = 0.0
                    for i in range(3):
                        for d in range(packed_width):
                            ja[w, s, i, d] = 0.0
                            jb[w, s, i, d] = 0.0
                pack_status[w] = code


def _overlap(left, right, mode):
    if mode == "interpret":
        return bool(np.shares_memory(left, right))
    a, b = int(left.device_ptr), int(right.device_ptr)
    return a < b + int(right.nbytes) and b < a + int(left.nbytes)


def pack_entry(
    capacity,
    count,
    nv,
    width,
    nbody,
    npair,
    area_capacity,
    h,
    source_status,
    source_mass,
    source_vectors,
    source_v,
    source_bias,
    source_passive,
    source_applied,
    motion,
    body_jacobian,
    area_input,
    area_pair_index,
    area_sizes,
    area_body_ids,
    pair_laws,
    sizes,
    mass,
    inputs,
    steps,
    samples,
    ja,
    jb,
    pack_status,
    *,
    mode="cuda",
):
    """Metadata-only checked launch; does not read values, allocate, or sync.

    Source leading extent is count, output leading extent is capacity. Buffers
    must be in the caller's currently active same-device TYX runtime. Runtime
    owner/entry stamp authenticity remains the caller's responsibility.

    Codes: 0 success; 1 upstream source_status != 0; 2 count/index; 3 finite or
    physical domain; 4 finite-source assembly overflow. The original upstream
    code remains unchanged in source_status. Invalid metadata raises ValueError
    before any kernel launch/output write. SPD is checked by the downstream
    solver; this pack checks exact symmetry and positive diagonal only.
    """
    bounds = (
        (capacity, 1, 32),
        (count, 1, capacity),
        (nv, 1, MAX_LIVE_DOF),
        (nbody, 1, 128),
        (npair, 1, 16),
        (area_capacity, 1, 512),
    )
    if any(type(value) is not int or not low <= value <= high for value, low, high in bounds):
        raise ValueError("entry pack bounded integer dimensions")
    if type(width) is not int or width != max(8, nv):
        raise ValueError("entry pack native padded width")
    if type(mode) is not str or mode not in ("interpret", "cuda"):
        raise ValueError("entry pack mode")
    if isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, float, np.integer, np.floating)):
        raise ValueError("entry pack timestep scalar")
    try:
        packed_h = np.float32(h)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("entry pack timestep representation") from error
    source = (
        source_status,
        source_mass,
        source_vectors,
        source_v,
        source_bias,
        source_passive,
        source_applied,
        motion,
        body_jacobian,
        area_input,
        area_pair_index,
        area_sizes,
        area_body_ids,
        pair_laws,
    )
    outputs = (sizes, mass, inputs, steps, samples, ja, jb, pack_status)
    array_type = np.ndarray if mode == "interpret" else tx.CudaArray
    if not all(isinstance(value, array_type) for value in source + outputs):
        raise ValueError("entry pack array backend")
    vector_shape = source_vectors.shape
    if len(vector_shape) != 2 or vector_shape[0] != 9 or not 8 <= vector_shape[1] <= 45:
        raise ValueError("entry pack static vector extent")
    f32, i32 = np.dtype("float32"), np.dtype("int32")
    packed_width = friction_width(nv)
    specifications = (
        ((count,), i32),
        ((count, width, width), f32),
        (vector_shape, f32),
        ((count, width), f32),
        ((count, width), f32),
        ((count, width), f32),
        ((count, width), f32),
        ((count, nbody, 7, 3), f32),
        ((count, nbody, 6, width), f32),
        ((count, area_capacity, 8), f32),
        ((count, area_capacity), i32),
        ((count,), i32),
        ((npair, 2), i32),
        ((npair, 2), f32),
        ((capacity, 4), i32),
        ((capacity, packed_width, packed_width), f32),
        ((capacity, 3, packed_width), f32),
        ((capacity,), f32),
        ((capacity, 128, 10), f32),
        ((capacity, 128, 3, packed_width), f32),
        ((capacity, 128, 3, packed_width), f32),
        ((capacity,), i32),
    )
    ordinal = source_status.device_ordinal if mode == "cuda" else None
    for index, (value, (shape, dtype)) in enumerate(
        zip(source + outputs, specifications, strict=True)
    ):
        if mode == "interpret":
            if (
                value.shape != shape
                or value.dtype != dtype
                or not value.flags.c_contiguous
                or (index >= len(source) and not value.flags.writeable)
            ):
                raise ValueError("entry pack array layout")
        else:
            if (
                value.shape != shape
                or np.dtype(value.host_dtype) != dtype
                or value.device_ordinal != ordinal
            ):
                raise ValueError("entry pack device layout")
            value._ensure_alive()
    prior = list(source)
    for value in outputs:
        if any(_overlap(value, previous, mode) for previous in prior):
            raise ValueError("entry pack outputs overlap source or destination storage")
        prior.append(value)
    return _pack_entry_kernel(
        capacity,
        count,
        nv,
        width,
        packed_width,
        nbody,
        npair,
        area_capacity,
        packed_h,
        *source,
        *outputs,
        mode=mode,
    )

# Copyright 2025 The Newton Developers
# Copyright 2026 Phymeo contributors
# Licensed under Apache License 2.0; see LICENSE and NOTICE in this package.
# World-coordinate RNE/CRB, following _floating_kernels.py (mujoco-warp v0.0.2).
"""Instantaneous physical rigid terms, without integration or actuation."""

import tyx as tx


@tx.kernel
def mechanics(
    worlds: tx.i32,
    nb: tx.i32,
    nv: tx.i32,
    bodies: tx.ndarray(tx.i32, 2),
    dofs: tx.ndarray(tx.i32, 2),
    ancestor: tx.ndarray(tx.i32, 2),
    inertia: tx.ndarray(tx.f32, 3),
    weight: tx.ndarray(tx.f32, 1),
    armature: tx.ndarray(tx.f32, 1),
    gravity: tx.ndarray(tx.f32, 1),
    v: tx.ndarray(tx.f32, 2),
    state: tx.ndarray(tx.f32, 4),
    rotation: tx.ndarray(tx.f32, 4),
    axes: tx.ndarray(tx.f32, 3),
    anchors: tx.ndarray(tx.f32, 4),
    com: tx.ndarray(tx.f32, 3),
    com_low: tx.ndarray(tx.f32, 3),
    jac: tx.ndarray(tx.f32, 4),
    motion: tx.ndarray(tx.f32, 4),
    iw: tx.ndarray(tx.f32, 4),
    force: tx.ndarray(tx.f32, 3),
    scratch: tx.ndarray(tx.f32, 3),
    mass: tx.ndarray(tx.f32, 3),
    bias: tx.ndarray(tx.f32, 2),
    sampled_qvel: tx.ndarray(tx.f32, 2),
    status: tx.ndarray(tx.i32, 1),
):
    for w in tx.range(worlds):
        for d in range(nv):
            sampled_qvel[w, d] = v[w, d]
            if not tx.isfinite(v[w, d]):
                status[w] = 1
        for b in range(nb):
            node = bodies[b, 0]
            parent = bodies[b, 2]
            parent_node = bodies[parent, 0]
            kind = bodies[b, 3]
            dof = bodies[b, 4]
            # Jacobians use the retained low COM/anchor difference.
            for d in range(nv):
                for i in range(3):
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    lin = 0.0
                    ang = 0.0
                    if ancestor[b, d] != 0:
                        if dofs[d, 0] != 0:
                            rj = (com[w, b, j] - anchors[w, d, j, 0]) + (
                                com_low[w, b, j] - anchors[w, d, j, 1]
                            )
                            rk = (com[w, b, k] - anchors[w, d, k, 0]) + (
                                com_low[w, b, k] - anchors[w, d, k, 1]
                            )
                            lin = axes[w, d, j] * rk - axes[w, d, k] * rj
                            ang = axes[w, d, i]
                        else:
                            lin = axes[w, d, i]
                    jac[w, b, i, d] = lin
                    jac[w, b, i + 3, d] = ang
            for i in range(3):
                omega = 0.0
                for d in range(nv):
                    omega = omega + jac[w, b, i + 3, d] * v[w, d]
                motion[w, b, 0, i] = omega
                motion[w, b, 1, i] = 0.0
                motion[w, b, 2, i] = 0.0
                if b > 0:
                    motion[w, b, 1, i] = motion[w, parent, 1, i]
                scratch[w, 0, i] = (state[w, node, i, 0] - state[w, parent_node, i, 0]) + (
                    state[w, node, i, 1] - state[w, parent_node, i, 1]
                )
                if kind == 1:
                    scratch[w, 0, i] = (anchors[w, dof, i, 0] - state[w, parent_node, i, 0]) + (
                        anchors[w, dof, i, 1] - state[w, parent_node, i, 1]
                    )
            if b > 0 and kind != 3 and kind != 4:
                for i in range(3):
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    scratch[w, 1, i] = (
                        motion[w, parent, 0, j] * scratch[w, 0, k]
                        - motion[w, parent, 0, k] * scratch[w, 0, j]
                    )
                    if kind == 1:
                        motion[w, b, 1, i] = (
                            motion[w, b, 1, i]
                            + (
                                motion[w, parent, 0, j] * axes[w, dof, k]
                                - motion[w, parent, 0, k] * axes[w, dof, j]
                            )
                            * v[w, dof]
                        )
                for i in range(3):
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    motion[w, b, 2, i] = (
                        motion[w, parent, 2, i]
                        + motion[w, parent, 1, j] * scratch[w, 0, k]
                        - motion[w, parent, 1, k] * scratch[w, 0, j]
                        + motion[w, parent, 0, j] * scratch[w, 1, k]
                        - motion[w, parent, 0, k] * scratch[w, 1, j]
                    )
                    if kind == 2:
                        motion[w, b, 2, i] = (
                            motion[w, b, 2, i]
                            + 2.0
                            * (
                                motion[w, parent, 0, j] * axes[w, dof, k]
                                - motion[w, parent, 0, k] * axes[w, dof, j]
                            )
                            * v[w, dof]
                        )
                if kind == 1:
                    for i in range(3):
                        scratch[w, 0, i] = (state[w, node, i, 0] - anchors[w, dof, i, 0]) + (
                            state[w, node, i, 1] - anchors[w, dof, i, 1]
                        )
                    for i in range(3):
                        j = (i + 1) % 3
                        k = (i + 2) % 3
                        scratch[w, 1, i] = (
                            motion[w, b, 0, j] * scratch[w, 0, k]
                            - motion[w, b, 0, k] * scratch[w, 0, j]
                        )
                    for i in range(3):
                        j = (i + 1) % 3
                        k = (i + 2) % 3
                        motion[w, b, 2, i] = (
                            motion[w, b, 2, i]
                            + motion[w, b, 1, j] * scratch[w, 0, k]
                            - motion[w, b, 1, k] * scratch[w, 0, j]
                            + motion[w, b, 0, j] * scratch[w, 1, k]
                            - motion[w, b, 0, k] * scratch[w, 1, j]
                        )
            for i in range(3):
                scratch[w, 0, i] = (com[w, b, i] - state[w, node, i, 0]) + (
                    com_low[w, b, i] - state[w, node, i, 1]
                )
                for j in range(3):
                    value = 0.0
                    for a in range(3):
                        for c in range(3):
                            value = (
                                value
                                + rotation[w, node, 3 * i + a, 0]
                                * inertia[b, a, c]
                                * rotation[w, node, 3 * j + c, 0]
                            )
                    iw[w, b, i, j] = value
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                scratch[w, 1, i] = (
                    motion[w, b, 0, j] * scratch[w, 0, k] - motion[w, b, 0, k] * scratch[w, 0, j]
                )
                scratch[w, 2, i] = 0.0
                scratch[w, 3, i] = 0.0
                for j in range(3):
                    scratch[w, 2, i] = scratch[w, 2, i] + iw[w, b, i, j] * motion[w, b, 0, j]
                    scratch[w, 3, i] = scratch[w, 3, i] + iw[w, b, i, j] * motion[w, b, 1, j]
            for i in range(3):
                j = (i + 1) % 3
                k = (i + 2) % 3
                value = (
                    motion[w, b, 2, i]
                    + motion[w, b, 1, j] * scratch[w, 0, k]
                    - motion[w, b, 1, k] * scratch[w, 0, j]
                    + motion[w, b, 0, j] * scratch[w, 1, k]
                    - motion[w, b, 0, k] * scratch[w, 1, j]
                )
                motion[w, b, 3, i] = value
                force[w, b, i] = weight[b] * (value - gravity[i])
                force[w, b, i + 3] = (
                    scratch[w, 3, i]
                    + motion[w, b, 0, j] * scratch[w, 2, k]
                    - motion[w, b, 0, k] * scratch[w, 2, j]
                )
        for i in range(nv):
            value = 0.0
            for b in range(1, nb):
                for k in range(6):
                    value = value + jac[w, b, k, i] * force[w, b, k]
            bias[w, i] = value
            if not tx.isfinite(value):
                status[w] = 1
            for j in range(i, nv):
                value = 0.0
                if i == j:
                    value = armature[i]
                for b in range(1, nb):
                    for k in range(3):
                        value = value + weight[b] * jac[w, b, k, i] * jac[w, b, k, j]
                        for axis in range(3):
                            value = (
                                value
                                + jac[w, b, k + 3, i] * iw[w, b, k, axis] * jac[w, b, axis + 3, j]
                            )
                mass[w, i, j] = value
                mass[w, j, i] = value
                if not tx.isfinite(value):
                    status[w] = 1

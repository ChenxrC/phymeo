# Copyright 2025 The Newton Developers
# Copyright 2026 Phymeo contributors
# Licensed under Apache License, Version 2.0. See LICENSE and NOTICE in this package.
# Model fields and scalar passive/transmission semantics follow mujoco-warp 0.0.2.

"""Static native tree fields only: no configuration-specific M/J/bias freezing."""

from dataclasses import dataclass

import numpy as np

from . import MuJoCoTyxIssue as Issue
from . import MuJoCoTyxUnsupported
from ._model import fp32
from ._tree_model import TyxTreeModel, _rotation

MAX_BODIES = 128


@dataclass(frozen=True)
class TyxFloatingModel(TyxTreeModel):
    root_body: int


def compile_floating_model(mj, model, *, allow_normal_contacts=False):
    issues = []

    def issue(code, detail):
        issues.append(Issue(code, detail))

    free = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_FREE)
    scalar = np.isin(model.jnt_type, [mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE])
    if (
        len(free) != 1
        or int(free[0]) != 0
        or not np.all(scalar[1:])
        or not 6 <= model.nv <= 35
        or model.nq != model.nv + 1
        or model.njnt != model.nv - 5
    ):
        issue("joint-domain", "T2 requires one leading native free root and 0..29 hinge/slide DOFs")
        raise MuJoCoTyxUnsupported(issues)
    root = int(model.jnt_bodyid[0])
    if (
        model.body_parentid[root] != 0
        or model.body_jntnum[root] != 1
        or model.jnt_qposadr[0] != 0
        or model.jnt_dofadr[0] != 0
    ):
        issue("root-domain", "native free root must be a world child with only its free joint")
    for joint in range(1, model.njnt):
        cursor = int(model.jnt_bodyid[joint])
        while cursor and cursor != root:
            cursor = int(model.body_parentid[cursor])
        if cursor != root:
            issue("root-domain", "all scalar joints must descend from the single free root")
    if model.jnt_stiffness[0] != 0:
        issue("free-spring", "native quaternion free-joint springs are not migrated")
    if model.nbody > MAX_BODIES:
        issue("body-capacity", "T2 preserves every body and supports at most 128 native bodies")
    for field, code in (
        ("neq", "equality"),
        ("ntendon", "tendon"),
        ("nflex", "flex"),
        ("nplugin", "plugin"),
        ("nmocap", "mocap"),
        ("nuserdata", "userdata"),
        ("npair", "explicit-pairs"),
        ("nexclude", "explicit-excludes"),
    ):
        if int(getattr(model, field, 0)):
            issue(code, f"native {field} is not migrated in T2")
    if model.na:
        issue("actuator-state", "activation dynamics are not migrated")
    if int(model.opt.integrator) != int(mj.mjtIntegrator.mjINT_EULER):
        issue("integrator", "T2 ports native Euler only")
    if (
        int(model.opt.iterations) < 1
        or int(model.opt.noslip_iterations)
        or not np.isfinite(model.opt.tolerance)
        or model.opt.tolerance < 0
    ):
        issue(
            "solver-options",
            "positive source iterations/nonnegative tolerance and no noslip required",
        )
    if int(model.opt.solver) != int(mj.mjtSolver.mjSOL_PGS) and int(model.opt.ls_iterations) < 1:
        issue("solver-options", "Newton/CG source must permit a line-search iteration")
    if int(model.opt.disableflags) or int(model.opt.enableflags):
        issue("option-flags", "nondefault native simulation flags are not migrated")
    if np.any(model.opt.wind) or model.opt.viscosity or model.opt.density:
        issue("fluid", "fluid dynamics are not migrated")
    if np.any(model.body_gravcomp) or np.any(model.dof_frictionloss):
        issue("passive", "gravity compensation and dry joint friction are not migrated")
    if np.any(model.dof_armature < 0) or np.any(model.dof_damping < 0):
        issue("passive", "negative reflected inertia/damping is outside T2")
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        if (
            model.actuator_trntype[actuator] != mj.mjtTrn.mjTRN_JOINT
            or not 1 <= joint < model.njnt
            or model.actuator_dyntype[actuator] != mj.mjtDyn.mjDYN_NONE
            or model.actuator_gaintype[actuator] != mj.mjtGain.mjGAIN_FIXED
            or model.actuator_biastype[actuator]
            not in (mj.mjtBias.mjBIAS_NONE, mj.mjtBias.mjBIAS_AFFINE)
            or np.any(model.actuator_gear[actuator, 1:])
            or np.any(model.actuator_gainprm[actuator, 1:])
            or np.any(model.actuator_biasprm[actuator, 3:])
        ):
            issue("actuator-domain", f"actuator {actuator} is not a scalar stateless affine motor")
    if np.any(model.jnt_actfrclimited):
        issue("joint-actuator-limits", "joint-wide summed force clamps are not migrated")
    # Test potential pairs, not just collisions at the source initial pose.
    for left in range(model.ngeom):
        for right in range(left + 1, model.ngeom):
            b1, b2 = map(int, model.geom_bodyid[[left, right]])
            w1, w2 = map(int, model.body_weldid[[b1, b2]])
            p1 = int(model.body_weldid[int(model.body_parentid[w1])])
            p2 = int(model.body_weldid[int(model.body_parentid[w2])])
            if w1 == w2 or (p1 == w2 and w2 != 0) or (p2 == w1 and w1 != 0):
                continue
            if (int(model.geom_contype[left]) & int(model.geom_conaffinity[right])) or (
                int(model.geom_contype[right]) & int(model.geom_conaffinity[left])
            ):
                if not allow_normal_contacts:
                    issue("contact-domain", f"T2 has a possible native contact pair {left}/{right}")
                break
    if issues:
        raise MuJoCoTyxUnsupported(issues)
    body_data = np.zeros((model.nbody, 25))
    for body in range(model.nbody):
        body_data[body, :3] = model.body_pos[body]
        body_data[body, 3:12] = _rotation(model.body_quat[body]).reshape(-1)
        body_data[body, 12:15] = model.body_ipos[body]
        body_data[body, 15] = model.body_mass[body]
        principal = _rotation(model.body_iquat[body])
        body_data[body, 16:25] = (
            principal @ np.diag(model.body_inertia[body]) @ principal.T
        ).reshape(-1)
    joint_ids = np.zeros((model.nv, 4), np.int32)
    joint_data = np.zeros((model.nv, 7))
    vectors = np.zeros((9, model.nv))
    vectors[7, :] = -np.inf
    vectors[8, :] = np.inf
    vectors[1, :6] = model.dof_damping[:6]
    vectors[5, :6] = model.dof_armature[:6]
    for dof in range(6):
        joint_ids[dof] = (int(dof >= 3), dof, dof, root)
    for joint in range(1, model.njnt):
        q, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        joint_ids[dof] = (
            int(model.jnt_type[joint] == mj.mjtJoint.mjJNT_HINGE),
            q,
            dof,
            int(model.jnt_bodyid[joint]),
        )
        joint_data[dof, :3] = model.jnt_axis[joint]
        joint_data[dof, 3:6] = model.jnt_pos[joint]
        joint_data[dof, 6] = model.qpos0[q]
        vectors[1, dof] = model.dof_damping[dof]
        vectors[2, dof] = model.jnt_stiffness[joint]
        vectors[3, dof] = model.qpos_spring[q]
        vectors[5, dof] = model.dof_armature[dof]
        if model.jnt_limited[joint]:
            low, high = model.jnt_range[joint]
            margin = model.jnt_margin[joint]
            vectors[7, dof], vectors[8, dof] = low + margin, high - margin
            if not low + margin < model.qpos0[q] < high - margin:
                issue("joint-limits", "source initial q enters native limit domain")
    ancestors = np.zeros((model.nbody, model.nv), np.int32)
    for body in range(1, model.nbody):
        cursor = body
        while cursor:
            start = int(model.body_jntadr[cursor])
            for joint in range(start, start + int(model.body_jntnum[cursor])):
                dof = int(model.jnt_dofadr[joint])
                if joint == 0:
                    ancestors[body, :6] = 1
                else:
                    ancestors[body, dof] = 1
            cursor = int(model.body_parentid[cursor])
    actuator_ids = np.zeros((max(model.nu, 1), 3), np.int32)
    actuator_values = np.zeros((max(model.nu, 1), 10))
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        actuator_ids[actuator] = (
            model.jnt_qposadr[joint],
            model.jnt_dofadr[joint],
            int(bool(model.opt.disableactuator & (1 << int(model.actuator_group[actuator])))),
        )
        bias = (
            model.actuator_biasprm[actuator, :3]
            if model.actuator_biastype[actuator] == mj.mjtBias.mjBIAS_AFFINE
            else np.zeros(3)
        )
        actuator_values[actuator, :5] = (
            model.actuator_gear[actuator, 0],
            model.actuator_gainprm[actuator, 0],
            *bias,
        )
        for offset, limited, limits in (
            (5, model.actuator_ctrllimited, model.actuator_ctrlrange),
            (7, model.actuator_forcelimited, model.actuator_forcerange),
        ):
            actuator_values[actuator, offset : offset + 2] = (
                limits[actuator] if limited[actuator] else (-np.inf, np.inf)
            )
    if issues:
        raise MuJoCoTyxUnsupported(issues)
    ranges = np.zeros((model.nbody, 2), np.int32)
    for body in range(1, model.nbody):
        if body != root and model.body_jntnum[body]:
            start = int(model.body_jntadr[body])
            ranges[body] = (int(model.jnt_dofadr[start]), int(model.body_jntnum[body]))
    return TyxFloatingModel(
        kind=3,
        nq=int(model.nq),
        nv=int(model.nv),
        nu=int(model.nu),
        body=root,
        root_body=root,
        pair=None,
        matrices=np.empty((0, 0, 0), np.float32),
        vectors=fp32(vectors, "tree static vectors"),
        body_jacobian=np.empty((0, 0, 0), np.float32),
        point_jacobian=np.empty((0, 0), np.float32),
        contact=np.zeros(32, np.float32),
        frame=np.eye(3, dtype=np.float32),
        actuator_ids=actuator_ids,
        actuator_values=fp32(actuator_values, "tree actuators"),
        source_solver=str(model.opt.solver),
        source_iterations=int(model.opt.iterations),
        parents=np.asarray(model.body_parentid, np.int32).copy(),
        body_joint_range=ranges,
        body_data=fp32(body_data, "native body inertias/frames"),
        joint_ids=joint_ids,
        joint_data=fp32(joint_data, "native joint anchors/axes/refs"),
        ancestors=ancestors,
        gravity=fp32(model.opt.gravity, "gravity"),
    )

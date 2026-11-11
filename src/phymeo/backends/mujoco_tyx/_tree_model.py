# Copyright 2025 The Newton Developers
# Copyright 2026 Phymeo contributors
# Licensed under Apache License, Version 2.0. See LICENSE and NOTICE in this package.
# Model fields and scalar passive/transmission semantics follow mujoco-warp 0.0.2.

"""Static native tree fields only: no configuration-specific M/J/bias freezing."""

from dataclasses import dataclass
from dataclasses import field as dataclass_field

import numpy as np

from ...dynamics.rotation import quaternion_matrix
from . import MuJoCoTyxIssue as Issue
from . import MuJoCoTyxUnsupported
from ._model import WIDTH, TyxModel, fp32

MAX_BODIES = 128


@dataclass(frozen=True)
class TyxTreeModel(TyxModel):
    parents: np.ndarray
    body_joint_range: np.ndarray
    body_data: np.ndarray
    joint_ids: np.ndarray
    joint_data: np.ndarray
    ancestors: np.ndarray
    gravity: np.ndarray
    # Keyword-only default preserves the independent floating-tree constructor.
    dof_ids: np.ndarray | None = dataclass_field(default=None, kw_only=True)


def _rotation(wxyz):
    return quaternion_matrix(np.asarray(wxyz)[None, [1, 2, 3, 0]])[0]


def compile_ball_model(mj, model, *, allow_activation=False):
    """Compose admitted BALL mechanics with existing declared contact solvers."""
    from ._constraint_model import compile_constraint_model
    from ._model import _mix_contact
    from ._normal_model import compile_normal_model, possible_pairs

    # This structural pass is deliberate: contact wrappers select a floating
    # base for any FREE joint, so BALL must be checked before that dispatch.
    base = compile_tree_model(
        mj, model, allow_normal_contacts=True, allow_activation=allow_activation
    )
    pairs = possible_pairs(model)
    if not pairs:
        return base
    if any(_mix_contact(model, left, right, mj)[0] == 3 for left, right in pairs):
        return compile_constraint_model(mj, model, pairs, allow_activation=allow_activation)
    return compile_normal_model(mj, model, pairs, allow_activation=allow_activation)


def compile_tree_model(
    mj, model, *, allow_normal_contacts=False, allow_constraints=False, allow_activation=False
):
    issues = []

    def issue(code, detail):
        issues.append(Issue(code, detail))

    has_ball = bool(np.any(model.jnt_type == mj.mjtJoint.mjJNT_BALL))
    allowed = (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE, mj.mjtJoint.mjJNT_BALL)
    if (
        not 1 <= model.nv <= WIDTH
        or not np.all(np.isin(model.jnt_type, allowed))
        or (not has_ball and model.nq != model.nv)
        or (has_ball and model.nq > 10)
    ):
        issue("joint-domain", "fixed-root hinge/slide/ball requires nv<=8 and nq<=10")
    if has_ball:
        if model.nu > 32 or model.na > 32:
            issue(
                "ball-actuator-capacity", "ball trees permit at most 32 scalar actors/activations"
            )
        for actor in range(model.nu):
            joint = int(model.actuator_trnid[actor, 0])
            if (
                model.actuator_trntype[actor] != mj.mjtTrn.mjTRN_JOINT
                or not 0 <= joint < model.njnt
                or model.jnt_type[joint] not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE)
            ):
                issue("ball-actuator-domain", "ball trees admit only scalar JOINT actuator targets")
        if np.any(model.jnt_limited) or np.any(model.dof_frictionloss):
            issue("ball-constraint-domain", "BALL1 excludes every joint limit and dry-friction row")
        for joint in range(model.njnt):
            if model.jnt_type[joint] == mj.mjtJoint.mjJNT_BALL:
                if model.jnt_stiffness[joint]:
                    issue("ball-spring", "quaternion springs are a separate capability")
                if int(model.body_jntnum[int(model.jnt_bodyid[joint])]) != 1:
                    issue("ball-same-body", "BALL1 requires a ball to be its body's only joint")
        if allow_normal_contacts:
            from ._constraint_model import PAIR_SLOTS
            from ._model import _mix_contact
            from ._normal_model import possible_pairs

            pairs = possible_pairs(model)
            slots = sum(
                PAIR_SLOTS.get(tuple(map(int, model.geom_type[list(pair)])), 0) for pair in pairs
            )
            if pairs and (model.ngeom > 128 or len(pairs) > 16 or slots > 32):
                issue("ball-contact-capacity", "ball contact permits 128 geoms/16 pairs/32 slots")
            if pairs and int(model.opt.cone) != int(mj.mjtCone.mjCONE_PYRAMIDAL):
                issue("ball-contact-cone", "ball contact requires a pyramidal source cone")
            if any(_mix_contact(model, left, right, mj)[0] not in (1, 3) for left, right in pairs):
                issue("ball-contact-dimension", "ball contact permits mixed native condim1/3 only")
    if model.nbody > MAX_BODIES:
        issue("body-capacity", "T1 preserves every body and supports at most 128 native bodies")
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
        if int(getattr(model, field, 0)) and not (allow_constraints and field == "nexclude"):
            issue(code, f"native {field} is not migrated in T1")
    if model.na and not allow_activation:
        issue("actuator-state", "activation dynamics are not migrated")
    if int(model.opt.integrator) != int(mj.mjtIntegrator.mjINT_EULER):
        issue("integrator", "T1 ports native Euler only")
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
    if np.any(model.body_gravcomp) or (np.any(model.dof_frictionloss) and not allow_constraints):
        issue("passive", "gravity compensation and dry joint friction are not migrated")
    if np.any(model.dof_armature < 0) or np.any(model.dof_damping < 0):
        issue("passive", "negative reflected inertia/damping is outside T1")
    for actuator in range(model.nu):
        joint = int(model.actuator_trnid[actuator, 0])
        if (
            model.actuator_trntype[actuator] != mj.mjtTrn.mjTRN_JOINT
            or not 0 <= joint < model.njnt
            or (
                model.actuator_dyntype[actuator] != mj.mjtDyn.mjDYN_NONE
                and not (
                    allow_activation
                    and model.actuator_dyntype[actuator] == mj.mjtDyn.mjDYN_FILTEREXACT
                )
            )
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
                    issue("contact-domain", f"T1 has a possible native contact pair {left}/{right}")
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
    joint_ids = np.empty((model.njnt, 4), np.int32)
    joint_data = np.empty((model.njnt, 7))
    dof_ids = np.full((WIDTH, 3), -1, np.int32)
    vectors = np.zeros((9, max(WIDTH, int(model.nq))))
    vectors[4, : model.nq] = model.qpos0
    vectors[7, :] = -np.inf
    vectors[8, :] = np.inf
    for joint in range(model.njnt):
        q, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        ball = model.jnt_type[joint] == mj.mjtJoint.mjJNT_BALL
        kind = 2 if ball else int(model.jnt_type[joint] == mj.mjtJoint.mjJNT_HINGE)
        joint_ids[joint] = (kind, q, dof, int(model.jnt_bodyid[joint]))
        joint_data[joint, :3] = model.jnt_axis[joint]
        joint_data[joint, 3:6] = model.jnt_pos[joint]
        joint_data[joint, 6] = model.qpos0[q]
        width = 3 if ball else 1
        for column in range(dof, dof + width):
            dof_ids[column] = (int(kind != 0), -1 if ball else q, joint)
            vectors[1, column] = model.dof_damping[column]
            vectors[5, column] = model.dof_armature[column]
        if not ball:
            vectors[2, dof] = model.jnt_stiffness[joint]
            vectors[3, dof] = model.qpos_spring[q]
        if model.jnt_limited[joint] and not allow_constraints:
            low, high = model.jnt_range[joint]
            margin = model.jnt_margin[joint]
            vectors[7, dof], vectors[8, dof] = low + margin, high - margin
            if not low + margin < model.qpos0[q] < high - margin:
                issue("joint-limits", "source initial q enters native limit domain")
    ancestors = np.zeros((model.nbody, WIDTH), np.int32)
    for body in range(1, model.nbody):
        cursor = body
        while cursor:
            start = int(model.body_jntadr[cursor])
            for joint in range(start, start + int(model.body_jntnum[cursor])):
                dof = int(model.jnt_dofadr[joint])
                width = 3 if model.jnt_type[joint] == mj.mjtJoint.mjJNT_BALL else 1
                ancestors[body, dof : dof + width] = 1
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
    ranges = np.stack((model.body_jntadr, model.body_jntnum), axis=1).astype(np.int32)
    return TyxTreeModel(
        kind=2,
        nq=int(model.nq),
        nv=int(model.nv),
        nu=int(model.nu),
        body=-1,
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
        dof_ids=dof_ids,
        joint_data=fp32(joint_data, "native joint anchors/axes/refs"),
        ancestors=ancestors,
        gravity=fp32(model.opt.gravity, "gravity"),
    )

# Copyright 2025 The Newton Developers
# Copyright 2026 Phymeo contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by law or agreed in writing, distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND. See LICENSE in this package.
#
# Contact parameter mixing adapted from mujoco-warp v0.0.2; priority/direct
# handling corrected to MuJoCo 3.3.7 engine_collision_driver.c::mj_contactParam.
# _src/collision_primitive.py. Added native model admission/constant-mass lowering.

"""Compile a narrow native model domain to static TYX data; no per-step CPU solver."""

from dataclasses import dataclass

import numpy as np

from . import MuJoCoTyxIssue as Issue
from . import MuJoCoTyxUnsupported

WIDTH = 8


def fp32(value, label):
    original = np.asarray(value, dtype=np.float64)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = original.astype(np.float32)
    bad = np.isfinite(original) & ~np.isfinite(result)
    tiny = (original != 0) & np.isfinite(original) & (np.abs(result) < np.finfo(np.float32).tiny)
    if np.any(bad | tiny):
        raise MuJoCoTyxUnsupported((Issue("fp32-range", f"{label} is outside normal FP32 range"),))
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class TyxModel:
    kind: int
    nq: int
    nv: int
    nu: int
    body: int
    pair: tuple | None
    matrices: np.ndarray
    vectors: np.ndarray
    body_jacobian: np.ndarray
    point_jacobian: np.ndarray
    contact: np.ndarray
    frame: np.ndarray
    actuator_ids: np.ndarray
    actuator_values: np.ndarray
    source_solver: str
    source_iterations: int


def _mix_contact(model, left, right, mj):
    p1, p2 = model.geom_priority[[left, right]]
    s1, s2 = model.geom_solmix[[left, right]]
    if p1 > p2:
        mix, dimension = 1.0, int(model.geom_condim[left])
        friction = model.geom_friction[left]
    elif p2 > p1:
        mix, dimension = 0.0, int(model.geom_condim[right])
        friction = model.geom_friction[right]
    else:
        mix = 0.5 if s1 < mj.mjMINVAL and s2 < mj.mjMINVAL else s1 / (s1 + s2)
        if s1 < mj.mjMINVAL <= s2:
            mix = 0.0
        if s2 < mj.mjMINVAL <= s1:
            mix = 1.0
        dimension = int(max(model.geom_condim[left], model.geom_condim[right]))
        friction = np.maximum(model.geom_friction[left], model.geom_friction[right])
    # CPU 3.3.7 semantics differ from Warp 0.0.2 here: priority wins before
    # deciding whether to mix positive or min direct-format reference values.
    if p1 > p2:
        ref, imp = model.geom_solref[left].copy(), model.geom_solimp[left].copy()
    elif p2 > p1:
        ref, imp = model.geom_solref[right].copy(), model.geom_solimp[right].copy()
    else:
        if model.geom_solref[left, 0] > 0 and model.geom_solref[right, 0] > 0:
            ref = mix * model.geom_solref[left] + (1 - mix) * model.geom_solref[right]
        else:
            ref = np.minimum(model.geom_solref[left], model.geom_solref[right])
        imp = mix * model.geom_solimp[left] + (1 - mix) * model.geom_solimp[right]
    friction = np.maximum(friction[[0, 0, 1, 2, 2]], mj.mjMINMU)
    return dimension, ref, imp, friction


def compile_model(mj, model):
    free_ids = np.flatnonzero(model.jnt_type == mj.mjtJoint.mjJNT_FREE)
    if len(free_ids):
        root = int(model.jnt_bodyid[free_ids[0]])
        # Preserve the established T0 free-body route. Select the new tree
        # domain from source structure, never by retrying a failed lowering.
        legacy_free = (
            model.njnt == 1
            and np.count_nonzero(model.body_rootid == root) == 1
            and not np.any(model.body_ipos[root])
            and np.all(model.body_inertia[root] == model.body_inertia[root, 0])
            and not np.any(model.jnt_stiffness)
            and model.nu == 0
        )
        if not legacy_free:
            from ._floating_model import compile_floating_model

            return compile_floating_model(mj, model)
    if np.any(model.jnt_type == mj.mjtJoint.mjJNT_HINGE):
        from ._tree_model import compile_tree_model

        return compile_tree_model(mj, model)
    issues = []

    def issue(code, detail):
        issues.append(Issue(code, detail))

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
            issue(code, f"native {field} is not migrated in the first TYX domain")
    if int(model.opt.integrator) != int(mj.mjtIntegrator.mjINT_EULER):
        issue("integrator", "first TYX domain ports native Euler only")
    if (
        int(model.opt.iterations) < 1
        or int(model.opt.noslip_iterations) != 0
        or not np.isfinite(model.opt.tolerance)
        or model.opt.tolerance < 0
    ):
        issue("solver-options", "positive iterations, nonnegative tolerance and no noslip required")
    if int(model.opt.solver) != int(mj.mjtSolver.mjSOL_PGS) and int(model.opt.ls_iterations) < 1:
        issue("solver-options", "Newton/CG source must permit a line-search iteration")
    if int(model.opt.disableflags) or int(model.opt.enableflags):
        issue("option-flags", "nondefault native simulation flags are not migrated")
    if np.any(model.opt.wind) or model.opt.viscosity or model.opt.density:
        issue("fluid", "fluid dynamics are not migrated")
    if np.any(model.dof_armature < 0) or np.any(model.dof_damping < 0):
        issue("passive", "negative reflected inertia/damping is outside the first domain")
    if np.any(model.body_gravcomp) or np.any(model.dof_frictionloss):
        issue("passive", "gravity compensation and dry joint friction are not migrated")
    free = model.njnt == 1 and model.jnt_type[0] == mj.mjtJoint.mjJNT_FREE
    prismatic = model.njnt > 0 and np.all(model.jnt_type == mj.mjtJoint.mjJNT_SLIDE)
    if not (free or prismatic) or not 1 <= model.nv <= WIDTH:
        issue("joint-domain", "one isotropic free root or 1..8 all-prismatic native DOFs required")
    if model.na:
        issue("actuator-state", "activation dynamics are not migrated")
    body = int(model.jnt_bodyid[0]) if free else -1
    if free:
        dynamic = np.flatnonzero(model.body_rootid == body)
        if (
            len(dynamic) != 1
            or np.any(model.body_ipos[body])
            or not np.all(model.body_inertia[body] == model.body_inertia[body, 0])
            or np.any(model.jnt_stiffness)
            or model.nu
        ):
            issue(
                "free-domain",
                "free root requires centered isotropic inertia, no child/spring/actuator",
            )
    for actuator in range(model.nu):
        if (
            model.actuator_trntype[actuator] != mj.mjtTrn.mjTRN_JOINT
            or model.actuator_dyntype[actuator] != mj.mjtDyn.mjDYN_NONE
            or model.actuator_gaintype[actuator] != mj.mjtGain.mjGAIN_FIXED
            or model.actuator_biastype[actuator]
            not in (mj.mjtBias.mjBIAS_NONE, mj.mjtBias.mjBIAS_AFFINE)
            or np.any(model.actuator_gear[actuator, 1:])
            or np.any(model.actuator_gainprm[actuator, 1:])
            or np.any(model.actuator_biasprm[actuator, 3:])
        ):
            issue("actuator-domain", f"actuator {actuator} is not a scalar stateless affine motor")
    if issues:
        raise MuJoCoTyxUnsupported(issues)
    if model.nq != (7 if free else model.nv):
        raise MuJoCoTyxUnsupported((Issue("coordinate-domain", "unexpected native qpos layout"),))
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    pairs = []
    for left in range(model.ngeom):
        for right in range(left + 1, model.ngeom):
            b1, b2 = map(int, model.geom_bodyid[[left, right]])
            w1, w2 = map(int, model.body_weldid[[b1, b2]])
            if w1 == w2:
                continue
            if (model.body_parentid[w1] == w2 and w2 != 0) or (
                model.body_parentid[w2] == w1 and w1 != 0
            ):
                continue
            if not (
                (int(model.geom_contype[left]) & int(model.geom_conaffinity[right]))
                or (int(model.geom_contype[right]) & int(model.geom_conaffinity[left]))
            ):
                continue
            if (
                w1 == 0
                and model.geom_type[left] == mj.mjtGeom.mjGEOM_PLANE
                and w2 != 0
                and model.geom_type[right] == mj.mjtGeom.mjGEOM_SPHERE
            ):
                pairs.append((left, right))
            elif (
                w2 == 0
                and model.geom_type[right] == mj.mjtGeom.mjGEOM_PLANE
                and w1 != 0
                and model.geom_type[left] == mj.mjtGeom.mjGEOM_SPHERE
            ):
                pairs.append((right, left))
            else:
                issue(
                    "collision-domain",
                    f"possible native pair {left}/{right} is not static plane/sphere",
                )
    if len(pairs) > 1:
        issue("contact-domain", "multiple possible contact pairs require the next multi-row solver")
    pair = pairs[0] if len(pairs) == 1 else None
    if free:
        for geom in np.flatnonzero(model.geom_bodyid == body):
            if model.geom_type[geom] == mj.mjtGeom.mjGEOM_SPHERE and np.any(model.geom_pos[geom]):
                issue("free-domain", "free collision sphere must be centered at its body COM")
    contact = np.zeros(32)
    frame = np.eye(3)
    if pair:
        plane, sphere = pair
        dimension, ref, imp, friction = _mix_contact(model, plane, sphere, mj)
        if dimension != 1:
            issue(
                "contact-dimension",
                "possible contact must use native condim=1; friction not migrated",
            )
        contact[:3] = data.geom_xpos[plane]
        contact[3:6] = data.geom_xmat[plane].reshape(3, 3)[:, 2]
        contact[6:9] = data.geom_xpos[sphere]
        contact[9] = model.geom_size[sphere, 0]
        contact[10] = max(model.geom_margin[plane], model.geom_margin[sphere])
        contact[11] = max(model.geom_gap[plane], model.geom_gap[sphere])
        contact[12] = sum(model.body_invweight0[int(model.geom_bodyid[g]), 0] for g in pair)
        if not ((ref[0] > 0 and ref[1] > 0) or (ref[0] <= 0 and ref[1] <= 0)):
            issue("contact-reference", "mixed-sign solref format is outside the first domain")
        contact[14:16], contact[16:21], contact[24:29] = ref, imp, friction
        normal = contact[3:6]
        tangent = np.eye(3)[int(np.argmin(abs(normal)))]
        tangent -= normal * (tangent @ normal)
        tangent /= np.linalg.norm(tangent)
        frame = np.stack((normal, tangent, np.cross(normal, tangent)))
    contact[21:24] = mj.mjMINVAL, mj.mjMINIMP, mj.mjMAXIMP
    if issues:
        raise MuJoCoTyxUnsupported(issues)
    mass = np.empty((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    inverse = np.linalg.inv(mass)
    euler = np.linalg.solve(mass + model.opt.timestep * np.diag(model.dof_damping), mass)
    matrices = np.zeros((2, WIDTH, WIDTH))
    matrices[0, : model.nv, : model.nv] = inverse
    matrices[1, : model.nv, : model.nv] = euler
    vectors = np.zeros((9, WIDTH))
    vectors[7, :] = -np.inf
    vectors[8, :] = np.inf
    vectors[0, : model.nv] = data.qfrc_bias
    vectors[1, : model.nv] = model.dof_damping
    vectors[4, : model.nq] = model.qpos0
    for joint in range(model.njnt):
        if not free:
            dof, q = int(model.jnt_dofadr[joint]), int(model.jnt_qposadr[joint])
            vectors[2, dof] = model.jnt_stiffness[joint]
            vectors[3, dof] = model.qpos_spring[q]
            if model.jnt_limited[joint]:
                lo, hi = model.jnt_range[joint]
                margin = model.jnt_margin[joint]
                vectors[7, dof], vectors[8, dof] = lo + margin, hi - margin
                if not lo + margin < model.qpos0[q] < hi - margin:
                    raise MuJoCoTyxUnsupported(
                        (Issue("joint-limits", "source initial q enters native limit domain"),)
                    )
    body_jac = np.zeros((model.nbody, 3, WIDTH))
    point_jac = np.zeros((3, WIDTH))
    if not free:
        for index in range(model.nbody):
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mj.mj_jacBodyCom(model, data, jp, jr, index)
            body_jac[index, :, : model.nv] = jp
    if pair:
        if free:
            point_jac[:, :3] = np.eye(3)
        else:
            point_jac = body_jac[int(model.geom_bodyid[pair[1]])].copy()
        j = contact[3:6] @ point_jac[:, : model.nv]
        vectors[5, : model.nv] = j
        vectors[6, : model.nv] = inverse @ j
        contact[13] = j @ inverse @ j
    actuator_ids = np.zeros((max(model.nu, 1), 3), dtype=np.int32)
    actuator_values = np.zeros((max(model.nu, 1), 10))
    for index in range(model.nu):
        joint = int(model.actuator_trnid[index, 0])
        actuator_ids[index] = (
            model.jnt_qposadr[joint],
            model.jnt_dofadr[joint],
            int(bool(model.opt.disableactuator & (1 << int(model.actuator_group[index])))),
        )
        actuator_values[index, :5] = (
            model.actuator_gear[index, 0],
            model.actuator_gainprm[index, 0],
            *(
                model.actuator_biasprm[index, :3]
                if model.actuator_biastype[index] == mj.mjtBias.mjBIAS_AFFINE
                else np.zeros(3)
            ),
        )
        for start, flag, limits in (
            (5, model.actuator_ctrllimited, model.actuator_ctrlrange),
            (7, model.actuator_forcelimited, model.actuator_forcerange),
        ):
            actuator_values[index, start : start + 2] = (
                limits[index] if flag[index] else (-np.inf, np.inf)
            )
    # Joint-wide force limits are applied after all actuator contributions.
    for joint in range(model.njnt):
        if bool(model.jnt_actfrclimited[joint]):
            raise MuJoCoTyxUnsupported(
                (Issue("joint-actuator-limits", "joint-wide summed force clamp is not migrated"),)
            )
    return TyxModel(
        int(free),
        int(model.nq),
        int(model.nv),
        int(model.nu),
        body,
        pair,
        fp32(matrices, "mass operators"),
        fp32(vectors, "static vectors"),
        fp32(body_jac, "body Jacobians"),
        fp32(point_jac, "contact Jacobian"),
        fp32(contact, "contact law"),
        fp32(frame, "contact frame"),
        actuator_ids,
        fp32(actuator_values, "actuators"),
        str(model.opt.solver),
        int(model.opt.iterations),
    )

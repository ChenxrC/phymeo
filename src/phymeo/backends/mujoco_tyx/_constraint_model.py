# Copyright 2026 Phymeo contributors
# Apache-2.0; native fields and filter semantics follow MuJoCo 3.3.7.
"""Source-structure admission for a shared pyramidal/joint constraint solve."""

from dataclasses import dataclass

import numpy as np

from . import MuJoCoTyxIssue as Issue
from . import MuJoCoTyxUnsupported
from ._model import _mix_contact, fp32
from ._normal_model import possible_pairs
from ._tree_model import _rotation

MAX_GEOMS = 128
MAX_PAIRS = 1024
MAX_SLOTS = 2048
MAX_CONTACTS = 128
MAX_EFC = 512
PAIR_SLOTS = {
    (0, 2): 1,
    (0, 3): 2,
    (0, 6): 8,
    (2, 2): 1,
    (2, 3): 1,
    (2, 6): 1,
    (3, 3): 2,
    (3, 6): 2,
    (6, 6): 24,
}


def constraint_pairs(model):
    """Native body excludes after the shared mask/weld/parent predicate."""
    excluded = set(map(int, model.exclude_signature))
    result = []
    for left, right in possible_pairs(model):
        low, high = sorted(map(int, model.geom_bodyid[[left, right]]))
        if (low * 65536 + high) not in excluded:
            result.append((left, right))
    return tuple(result)


def needs_constraints(mj, model, pairs):
    if model.nexclude or np.any(model.jnt_limited) or np.any(model.dof_frictionloss):
        return True
    count = 0
    for left, right in pairs:
        dimension, _, _, _ = _mix_contact(model, left, right, mj)
        if dimension != 1:
            return True
        count += PAIR_SLOTS.get(tuple(map(int, model.geom_type[[left, right]])), 0)
    return len(pairs) > 16 or count > 32


@dataclass(frozen=True)
class TyxConstraintModel:
    base: object
    geom_ids: np.ndarray
    geom_data: np.ndarray
    pair_ids: np.ndarray
    pair_law: np.ndarray
    pair_condim: np.ndarray
    pair_friction: np.ndarray
    slot_ids: np.ndarray
    friction_ids: np.ndarray
    friction_law: np.ndarray
    limit_ids: np.ndarray
    limit_law: np.ndarray
    ngeom: int
    npair: int
    nslot: int
    nfriction: int
    nlimit: int
    solver_tolerance: float
    solver_iterations: int
    impratio: float
    extended_contacts: bool
    constraint_contact: bool = True

    def __getattr__(self, name):
        return getattr(self.base, name)


def compile_constraint_model(mj, model, pairs):
    from ._floating_model import compile_floating_model
    from ._tree_model import compile_tree_model

    issues = []
    if model.ngeom > MAX_GEOMS or len(pairs) > MAX_PAIRS:
        issues.append(
            Issue("constraint-layout-capacity", "T4 permits 128 geoms/1024 possible pairs")
        )
    if int(model.opt.cone) != int(mj.mjtCone.mjCONE_PYRAMIDAL):
        issues.append(Issue("constraint-cone", "T4 preserves pyramidal source cone only"))
    if not 1 <= model.opt.iterations <= 512:
        issues.append(Issue("solver-options", "T4 face solve budget must be in 1..512"))
    if not np.isfinite(model.opt.impratio) or model.opt.impratio <= 0:
        issues.append(Issue("constraint-parameters", "positive finite native impratio required"))
    if np.any(model.dof_frictionloss < 0):
        issues.append(Issue("constraint-parameters", "negative dry friction is unsupported"))
    pair_ids, laws, dims, frictions, slots = [], [], [], [], []
    for pair, (left, right) in enumerate(pairs):
        shape = tuple(map(int, model.geom_type[[left, right]]))
        if shape not in PAIR_SLOTS:
            issues.append(Issue("contact-domain", f"T4 geometry pair {shape} is not migrated"))
            continue
        dimension, ref, imp, friction = _mix_contact(model, left, right, mj)
        if dimension not in (1, 3, 4, 6):
            issues.append(Issue("contact-domain", "T4 supports native condim1/3/4/6"))
        if (ref[0] > 0) != (ref[1] > 0):
            issues.append(Issue("contact-solref", "mixed positive/direct solref is unsupported"))
        law = np.zeros(16)
        law[:3] = (
            max(model.geom_margin[left], model.geom_margin[right]),
            max(model.geom_gap[left], model.geom_gap[right]),
            np.sum(model.body_invweight0[model.geom_bodyid[[left, right]], 0]),
        )
        law[3:5], law[5:10] = ref, imp
        law[10:13] = mj.mjMINVAL, mj.mjMINIMP, mj.mjMAXIMP
        if law[0] < 0 or not np.all(np.isfinite(law)) or not np.all(np.isfinite(friction)):
            issues.append(
                Issue("contact-parameters", "finite parameters and nonnegative margin required")
            )
        pair_ids.append((left, right, len(slots), PAIR_SLOTS[shape]))
        slots.extend((pair, feature, left, right) for feature in range(PAIR_SLOTS[shape]))
        laws.append(law)
        dims.append(dimension)
        frictions.append(friction)
    if len(slots) > MAX_SLOTS:
        issues.append(Issue("constraint-layout-capacity", "T4 permits 2048 static geometry slots"))
    friction_ids = np.flatnonzero(model.dof_frictionloss > 0)
    friction_law = np.zeros((max(1, len(friction_ids)), 9))
    for i, dof in enumerate(friction_ids):
        friction_law[i] = np.r_[
            model.dof_frictionloss[dof],
            model.dof_invweight0[dof],
            model.dof_solref[dof],
            model.dof_solimp[dof],
        ]
    limit_ids, limit_law = [], []
    for joint in np.flatnonzero(model.jnt_limited):
        if model.jnt_type[joint] not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            issues.append(
                Issue("joint-limit-domain", "only scalar hinge/slide soft limits are supported")
            )
            continue
        dof = int(model.jnt_dofadr[joint])
        limit_ids.append((joint, model.jnt_qposadr[joint], dof))
        limit_law.append(
            np.r_[
                model.jnt_range[joint],
                model.jnt_margin[joint],
                model.dof_invweight0[dof],
                model.jnt_solref[joint],
                model.jnt_solimp[joint],
            ]
        )
    limits = np.asarray(limit_law, dtype=np.float64).reshape((-1, 11))
    for label, table, refindex in (("dry friction", friction_law, 2), ("joint limit", limits, 4)):
        if not np.all(np.isfinite(table)):
            issues.append(Issue("constraint-parameters", f"nonfinite {label} parameters"))
        if len(table) and np.any((table[:, refindex] > 0) != (table[:, refindex + 1] > 0)):
            issues.append(
                Issue("constraint-solref", f"mixed positive/direct {label} solref unsupported")
            )
    if issues:
        raise MuJoCoTyxUnsupported(issues)
    compile_base = (
        compile_floating_model
        if np.any(model.jnt_type == mj.mjtJoint.mjJNT_FREE)
        else compile_tree_model
    )
    base = compile_base(mj, model, allow_normal_contacts=True, allow_constraints=True)
    geom_data = np.zeros((max(1, model.ngeom), 15))
    geom_ids = np.zeros((max(1, model.ngeom), 2), dtype=np.int32)
    for geom in range(model.ngeom):
        geom_ids[geom] = model.geom_bodyid[geom], model.geom_type[geom]
        geom_data[geom, :3] = model.geom_pos[geom]
        geom_data[geom, 3:12] = _rotation(model.geom_quat[geom]).reshape(-1)
        geom_data[geom, 12:15] = model.geom_size[geom]

    def padded(values, width, dtype):
        out = np.zeros((max(1, len(values)), width), dtype=dtype)
        if len(values):
            out[: len(values)] = values
        return out

    return TyxConstraintModel(
        base,
        geom_ids,
        fp32(geom_data, "T4 geometry"),
        padded(pair_ids, 4, np.int32),
        fp32(padded(laws, 16, np.float64), "T4 pair law"),
        np.asarray(dims if dims else [1], np.int32),
        fp32(padded(frictions, 5, np.float64), "T4 mixed friction"),
        padded(slots, 4, np.int32),
        np.asarray(friction_ids if len(friction_ids) else [-1], np.int32),
        fp32(friction_law, "T4 DOF law"),
        padded(limit_ids, 3, np.int32),
        fp32(padded(limits, 11, np.float64), "T4 limit law"),
        int(model.ngeom),
        len(pair_ids),
        len(slots),
        len(friction_ids),
        len(limit_ids),
        fp32(max(float(model.opt.tolerance), 2e-5), "T4 KKT tolerance").item(),
        int(model.opt.iterations),
        fp32(model.opt.impratio, "T4 impratio").item(),
        any(int(model.geom_type[left]) >= 3 for left, _ in pairs),
    )

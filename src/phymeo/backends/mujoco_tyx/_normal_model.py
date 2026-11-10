# Copyright 2025 The Newton Developers
# Copyright 2026 Phymeo contributors
# Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.
# Native pair filtering/material fields follow MuJoCo 3.3.7 and Warp 0.0.2.

"""Static normal-contact layout, preserving the complete admitted tree model."""

from dataclasses import dataclass

import numpy as np

from . import MuJoCoTyxIssue as Issue
from . import MuJoCoTyxUnsupported
from ._model import _mix_contact, fp32
from ._tree_model import _rotation

MAX_GEOMS = 128
MAX_PAIRS = 16
MAX_SLOTS = 32


def possible_pairs(model):
    """Implicit native mask/weld/parent filter; never based on initial overlap."""
    pairs = []
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
                pairs.append(tuple(sorted((left, right), key=lambda i: (model.geom_type[i], i))))
    return tuple(pairs)


@dataclass(frozen=True)
class TyxNormalModel:
    base: object
    geom_ids: np.ndarray
    geom_data: np.ndarray
    pair_ids: np.ndarray
    pair_law: np.ndarray
    pair_friction: np.ndarray
    slot_ids: np.ndarray
    solver_tolerance: float
    solver_iterations: int
    extended_contacts: bool = False
    normal_contact: bool = True

    def __getattr__(self, name):
        return getattr(self.base, name)


def compile_normal_model(mj, model, pairs):
    from ._floating_model import compile_floating_model
    from ._tree_model import compile_tree_model

    issues = []
    if model.ngeom > MAX_GEOMS or len(pairs) > MAX_PAIRS:
        issues.append(Issue("contact-layout-capacity", "normal domain permits 128 geoms/16 pairs"))
    if not 1 <= model.opt.iterations <= 512:
        issues.append(Issue("solver-options", "normal active-set budget must be in 1..512"))
    shapes = {
        (int(mj.mjtGeom.mjGEOM_PLANE), int(mj.mjtGeom.mjGEOM_SPHERE)): 1,
        (int(mj.mjtGeom.mjGEOM_PLANE), int(mj.mjtGeom.mjGEOM_CAPSULE)): 2,
        (int(mj.mjtGeom.mjGEOM_PLANE), int(mj.mjtGeom.mjGEOM_BOX)): 8,
        (int(mj.mjtGeom.mjGEOM_SPHERE), int(mj.mjtGeom.mjGEOM_SPHERE)): 1,
        (int(mj.mjtGeom.mjGEOM_SPHERE), int(mj.mjtGeom.mjGEOM_CAPSULE)): 1,
        (int(mj.mjtGeom.mjGEOM_SPHERE), int(mj.mjtGeom.mjGEOM_BOX)): 1,
        (int(mj.mjtGeom.mjGEOM_CAPSULE), int(mj.mjtGeom.mjGEOM_CAPSULE)): 2,
        (int(mj.mjtGeom.mjGEOM_CAPSULE), int(mj.mjtGeom.mjGEOM_BOX)): 2,
        # CPU raw box manifold enumerates up to 24 candidates before filtering.
        (int(mj.mjtGeom.mjGEOM_BOX), int(mj.mjtGeom.mjGEOM_BOX)): 24,
    }
    pair_ids, laws, frictions, slots = [], [], [], []
    for pair_index, (left, right) in enumerate(pairs):
        shape = tuple(map(int, model.geom_type[[left, right]]))
        if shape not in shapes:
            issues.append(Issue("contact-domain", f"normal geometry pair {shape} is not migrated"))
            continue
        dimension, ref, imp, friction = _mix_contact(model, left, right, mj)
        if dimension != 1:
            issues.append(Issue("contact-domain", "normal domain requires mixed native condim=1"))
        if (ref[0] > 0) != (ref[1] > 0):
            issues.append(Issue("contact-solref", "mixed positive/direct solref is not migrated"))
        bodies = model.geom_bodyid[[left, right]]
        law = np.zeros(16)
        law[:3] = (
            max(model.geom_margin[left], model.geom_margin[right]),
            max(model.geom_gap[left], model.geom_gap[right]),
            np.sum(model.body_invweight0[bodies, 0]),
        )
        law[3:5], law[5:10] = ref, imp
        law[10:13] = (mj.mjMINVAL, mj.mjMINIMP, mj.mjMAXIMP)
        if law[0] < 0:
            issues.append(
                Issue("contact-margin", "normal primitive domain requires nonnegative mixed margin")
            )
        if not np.all(np.isfinite(law)) or not np.all(np.isfinite(friction)):
            issues.append(Issue("contact-parameters", "nonfinite native contact parameter"))
        pair_ids.append((left, right, len(slots), shapes[shape]))
        slots.extend((pair_index, feature, left, right) for feature in range(shapes[shape]))
        laws.append(law)
        frictions.append(friction)
    if len(slots) > MAX_SLOTS:
        issues.append(Issue("contact-layout-capacity", "normal domain permits 32 candidate slots"))
    if issues:
        raise MuJoCoTyxUnsupported(issues)
    base = (
        compile_floating_model(mj, model, allow_normal_contacts=True)
        if np.any(model.jnt_type == mj.mjtJoint.mjJNT_FREE)
        else compile_tree_model(mj, model, allow_normal_contacts=True)
    )
    geom_data = np.zeros((model.ngeom, 15))
    for geom in range(model.ngeom):
        geom_data[geom, :3] = model.geom_pos[geom]
        geom_data[geom, 3:12] = _rotation(model.geom_quat[geom]).reshape(-1)
        geom_data[geom, 12:15] = model.geom_size[geom]
    return TyxNormalModel(
        base=base,
        geom_ids=np.stack((model.geom_bodyid, model.geom_type), axis=1).astype(np.int32),
        geom_data=fp32(geom_data, "normal native geometry"),
        pair_ids=np.asarray(pair_ids, np.int32),
        pair_law=fp32(laws, "normal native parameters"),
        pair_friction=fp32(frictions, "normal native friction metadata"),
        slot_ids=np.asarray(slots, np.int32),
        solver_tolerance=fp32(max(float(model.opt.tolerance), 2e-5), "normal KKT tolerance").item(),
        solver_iterations=int(model.opt.iterations),
        extended_contacts=any(int(model.geom_type[left]) >= 3 for left, _ in pairs),
    )

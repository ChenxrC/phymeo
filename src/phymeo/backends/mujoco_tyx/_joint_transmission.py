# Copyright 2026 Phymeo contributors
# Native JOINT transmission semantics follow MuJoCo 3.3.7 (Apache-2.0).
"""Owned native-address CSR for scalar and BALL JOINT actuator moments."""
from dataclasses import dataclass

import numpy as np

from . import MuJoCoTyxIssue, MuJoCoTyxUnsupported

MAX_ACTUATORS = 32
MAX_MOMENTS = 96


@dataclass(frozen=True)
class TyxJointTransmission:
    ids: np.ndarray
    gear: np.ndarray
    row_offsets: np.ndarray
    dof_indices: np.ndarray
    moment: np.ndarray


def has_ball_transmission(mj, model):
    """Only direct native JOINT targets; invalid other transmissions stay rejected."""
    return any(
        int(model.actuator_trntype[a]) == int(mj.mjtTrn.mjTRN_JOINT)
        and 0 <= int(model.actuator_trnid[a, 0]) < model.njnt
        and int(model.jnt_type[int(model.actuator_trnid[a, 0])]) == int(mj.mjtJoint.mjJNT_BALL)
        for a in range(model.nu)
    )


def _immutable(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def compile_joint_transmission(mj, model):
    """Compile static moments, never configuration-specific M/J or motor length.

    The mechanical compiler separately checks the complete model/law domain.
    All capacity/address/representability checks here precede output allocation.
    """
    from ._model import fp32

    def reject(code, detail):
        raise MuJoCoTyxUnsupported((MuJoCoTyxIssue(code, detail),))

    nu, nv, nq = int(model.nu), int(model.nv), int(model.nq)
    if not 1 <= nu <= MAX_ACTUATORS or not 1 <= nv <= 8 or not 1 <= nq <= 10:
        reject("joint-transmission-capacity", "joint CSR requires nu<=32/nv<=8/nq<=10")
    rows = []
    nnz = 0
    for actor in range(nu):
        joint = int(model.actuator_trnid[actor, 0])
        if int(model.actuator_trntype[actor]) != int(mj.mjtTrn.mjTRN_JOINT) or not 0 <= joint < model.njnt:
            reject("joint-transmission-target", f"actor {actor} requires a native JOINT target")
        kind = int(model.jnt_type[joint])
        if kind == int(mj.mjtJoint.mjJNT_BALL):
            width, qwidth = 3, 4
        elif kind in (int(mj.mjtJoint.mjJNT_HINGE), int(mj.mjtJoint.mjJNT_SLIDE)):
            width, qwidth = 1, 1
        else:
            reject("joint-transmission-target", f"actor {actor} has an unsupported target kind")
        qa, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
        if not 0 <= qa <= nq-qwidth or not 0 <= dof <= nv-width:
            reject("joint-transmission-address", f"actor {actor} exceeds native qpos/DOF storage")
        gear = model.actuator_gear[actor]
        if not np.isfinite(gear).all() or np.any(gear[width:]):
            reject("joint-transmission-gear", f"actor {actor} requires finite native-width gear")
        # Reuse the original normal-or-zero representation gate on ALL columns.
        fp32(gear[:width], f"actor {actor} native joint moments")
        rows.append((actor, joint, kind, qa, dof, width))
        nnz += width
    if nnz > MAX_MOMENTS:
        reject("joint-transmission-capacity", "complete moments exceed 96 entries")
    ids = np.asarray(rows, dtype=np.int32)
    offsets = np.zeros(nu+1, dtype=np.int32)
    columns = np.empty(nnz, dtype=np.int32)
    moments = np.empty(nnz, dtype=np.float32)
    cursor = 0
    for actor, _, _, _, dof, width in rows:
        for axis in range(width):
            columns[cursor] = dof+axis
            moments[cursor] = model.actuator_gear[actor, axis]
            cursor += 1
        offsets[actor+1] = cursor
    return TyxJointTransmission(
        ids=_immutable(ids), gear=_immutable(np.asarray(model.actuator_gear, dtype=np.float64)),
        row_offsets=_immutable(offsets), dof_indices=_immutable(columns), moment=_immutable(moments),
    )

# Copyright 2026 Phymeo contributors
# Native actuator semantics follow MuJoCo 3.3.7 (Apache-2.0).
"""Bounded native JOINT activation over original mechanical/contact routes."""

from dataclasses import dataclass, fields

import numpy as np

from . import MuJoCoTyxIssue, MuJoCoTyxUnsupported
from ._model import fp32
from ._tree_model import TyxTreeModel, compile_ball_model, compile_tree_model

MAX_ACTUATORS = 32
MAX_ACTIVATIONS = 32
# ids: qpos, dof, joint, actadr(-1 for NONE), early, disabled, ctrl/act/force limited
# values: gear, gain, bias[3], ctrl[2], force[2], effective tau, act[2]


def _immutable(value):
    array = np.ascontiguousarray(value)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass(frozen=True)
class TyxActuatorLayout:
    na: int
    ids: np.ndarray
    values: np.ndarray
    schema: str = "native-scalar-filterexact-v1"


@dataclass(frozen=True)
class TyxActivationTreeModel(TyxTreeModel):
    activation: TyxActuatorLayout


@dataclass(frozen=True)
class TyxActivationModel:
    """Orthogonal state layout; all original plan fields remain on its base."""

    base: object
    activation: TyxActuatorLayout

    def __getattr__(self, name):
        return getattr(self.base, name)


def compile_activation_model(mj, model):
    issues = []

    def issue(code, detail):
        issues.append(MuJoCoTyxIssue(code, detail))

    if not 1 <= int(model.nu) <= MAX_ACTUATORS or not 1 <= int(model.na) <= MAX_ACTIVATIONS:
        issue(
            "activation-capacity", "stateful TYX requires 1..32 actors and activation coordinates"
        )
    if issues:
        raise MuJoCoTyxUnsupported(issues)
    ids = np.zeros((int(model.nu), 9), np.int32)
    values = np.zeros((int(model.nu), 12), np.float64)
    addresses = []
    for actor in range(model.nu):
        dtype = int(model.actuator_dyntype[actor])
        adr, count = int(model.actuator_actadr[actor]), int(model.actuator_actnum[actor])
        if dtype == int(mj.mjtDyn.mjDYN_NONE):
            if adr != -1 or count != 0:
                issue("activation-layout", f"NONE actor {actor} owns activation coordinates")
        elif dtype == int(mj.mjtDyn.mjDYN_FILTEREXACT):
            if count != 1 or not 0 <= adr < model.na:
                issue("activation-layout", f"FILTEREXACT actor {actor} requires one valid address")
            else:
                addresses.append(adr)
            tau = float(model.actuator_dynprm[actor, 0])
            if not np.isfinite(tau) or tau <= 0:
                issue("activation-time-constant", f"actor {actor} requires finite positive tau")
            values[actor, 9] = max(float(mj.mjMINVAL), tau)
        else:
            issue("activation-dynamics", f"actor {actor} is neither NONE nor FILTEREXACT")
        joint = int(model.actuator_trnid[actor, 0])
        if (
            model.actuator_trntype[actor] != mj.mjtTrn.mjTRN_JOINT
            or not 0 <= joint < model.njnt
            or model.jnt_type[joint]
            not in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE, mj.mjtJoint.mjJNT_BALL)
        ):
            issue(
                "activation-transmission", f"actor {actor} requires a scalar or BALL JOINT target"
            )
            continue
        if (
            model.jnt_type[joint] == mj.mjtJoint.mjJNT_BALL
            and model.actuator_biastype[actor] != mj.mjtBias.mjBIAS_NONE
        ):
            issue("activation-ball-bias", f"BALL actor {actor} requires BIAS_NONE")
        ids[actor] = (
            int(model.jnt_qposadr[joint]),
            int(model.jnt_dofadr[joint]),
            joint,
            adr,
            int(model.actuator_actearly[actor]),
            int(bool(model.opt.disableactuator & (1 << int(model.actuator_group[actor])))),
            int(model.actuator_ctrllimited[actor]),
            int(model.actuator_actlimited[actor]),
            int(model.actuator_forcelimited[actor]),
        )
        bias = (
            model.actuator_biasprm[actor, :3]
            if model.actuator_biastype[actor] == mj.mjtBias.mjBIAS_AFFINE
            else np.zeros(3)
        )
        values[actor, :5] = (model.actuator_gear[actor, 0], model.actuator_gainprm[actor, 0], *bias)
        for mask_column, offset, ranges in (
            (6, 5, model.actuator_ctrlrange),
            (7, 10, model.actuator_actrange),
            (8, 7, model.actuator_forcerange),
        ):
            if ids[actor, mask_column]:
                interval = np.asarray(ranges[actor], dtype=np.float64)
                if not np.isfinite(interval).all() or interval[0] > interval[1]:
                    issue("activation-range", f"actor {actor} has an invalid enabled interval")
                values[actor, offset : offset + 2] = interval
    if sorted(addresses) != list(range(int(model.na))):
        issue("activation-layout", "activation addresses must cover native na exactly once")
    if not np.isfinite(values).all():
        issue("activation-parameters", "all used actuator parameters must be finite")
    if issues:
        raise MuJoCoTyxUnsupported(issues)
    # Select from the original structural source predicates, never by catching
    # a failed compilation. Each base keeps all its geometry/solver/option gates.
    from ._constraint_model import compile_constraint_model, constraint_pairs, needs_constraints
    from ._floating_model import compile_floating_model
    from ._model import _mix_contact
    from ._normal_model import compile_normal_model, possible_pairs

    candidate_pairs = constraint_pairs(model)
    has_ball = bool(np.any(model.jnt_type == mj.mjtJoint.mjJNT_BALL))
    if has_ball and np.any(model.jnt_type == mj.mjtJoint.mjJNT_FREE):
        # Validate the complete narrow FREE+BALL domain first. The generic
        # constraint predicate must not admit limits, dry friction, excludes,
        # larger contact capacities or higher contact dimensions through it.
        base = compile_floating_model(
            mj, model, allow_normal_contacts=True, allow_activation=True
        )
        pairs = possible_pairs(model)
        if any(_mix_contact(model, left, right, mj)[0] == 3 for left, right in pairs):
            base = compile_constraint_model(mj, model, pairs, allow_activation=True)
        elif pairs:
            base = compile_normal_model(mj, model, pairs, allow_activation=True)
    elif has_ball:
        # Preserve fixed BALL structural gates before contact composition.
        base = compile_ball_model(mj, model, allow_activation=True)
    elif needs_constraints(mj, model, candidate_pairs):
        base = compile_constraint_model(mj, model, candidate_pairs, allow_activation=True)
    else:
        pairs = possible_pairs(model)
        if pairs:
            base = compile_normal_model(mj, model, pairs, allow_activation=True)
        elif np.any(model.jnt_type == mj.mjtJoint.mjJNT_FREE):
            base = compile_floating_model(mj, model, allow_activation=True)
        else:
            base = compile_tree_model(mj, model, allow_activation=True)
    layout = TyxActuatorLayout(
        int(model.na),
        _immutable(ids),
        _immutable(fp32(values, "activation parameters")),
        schema=(
            "native-joint-filterexact-v1"
            if getattr(base, "joint_transmission", None) is not None
            else "native-scalar-filterexact-v1"
        ),
    )
    # Preserve the accepted fixed T1 plan type and checkpoint identity.
    if type(base) is TyxTreeModel:
        return TyxActivationTreeModel(
            **{field.name: getattr(base, field.name) for field in fields(base)}, activation=layout
        )
    return TyxActivationModel(base=base, activation=layout)

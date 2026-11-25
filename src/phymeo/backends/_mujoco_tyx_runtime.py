"""Native-coordinate MuJoCo normal-law subset executed by independent TYX kernels.

The source-bound MjModel is decoded once. No per-step mj_step, mj_forward,
Warp import or inherited Phymeo contact law implements dynamics. Explicit
snapshot() observation uses a detached CPU forward pass and is labelled as
such. Fixed and floating trees recompute full dynamics each step. Admitted
primitive manifolds share one normal QP; the original constant-mass
single-normal domain remains a separate route.
"""

import copy
import hashlib
import json
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..training.resident_stepper import PhysicsStepFailure
from ._mujoco_tyx_mirror import MIRROR_FIELDS, NativeStateLayout
from .mujoco_runtime import (
    MuJoCoControl,
    MuJoCoInitialState,
    MuJoCoRuntime,
    MuJoCoState,
    _contacts,
    _frozen,
)
from .mujoco_tyx import MuJoCoTyxUnsupported as MuJoCoTyxUnsupported
from .mujoco_tyx._model import compile_model, fp32

LAW = "mujoco-soft-normal-constraint-v1"
MIXED_LAW = "mujoco-pyramidal-joint-constraint-v1"


@dataclass(frozen=True, eq=False)
class _PreparedStep:
    owner: object
    device: object
    revision: int
    active: int
    entry_data: object
    entry_controls: object
    data: object
    receipts: object
    ticks: object
    accepted: object
    completed_steps: int
    accepted_world_steps: int
    destination_revision: int
    destination_bank: int


@dataclass(frozen=True, eq=False)
class _PreparedControls:
    owner: object
    revision: int
    entry_controls: object
    controls: object
    retired: object
    destination_revision: int


@dataclass(frozen=True)
class MuJoCoTyxResetReceipt:
    """One committed reset boundary; requested and stored native states differ.

    Cleanup failures describe retired owners, never a rejected publication.
    This receipt is not a general per-step state-write or training checkpoint.
    """

    schema: str
    source_revision: int
    destination_revision: int
    world_keys: tuple
    binding_sha256: str
    requested: object
    stored: object
    clock: np.ndarray
    cleanup_errors: tuple = ()


def _capacity(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a nonnegative integer")
    if value < 0 or value > np.iinfo(np.int32).max:
        raise ValueError(f"{name} must be a nonnegative i32 integer")
    return int(value)


@dataclass(frozen=True)
class MuJoCoTyxContact:
    schema: str
    geometry_ids: tuple
    dimension: int
    distance: float
    position: np.ndarray
    frame: np.ndarray
    include_margin: float
    friction: np.ndarray
    solref: np.ndarray
    solimp: np.ndarray
    active: bool
    wrench_contact_frame: np.ndarray
    force_on_sphere_world: np.ndarray
    static_reaction_world: np.ndarray
    impedance: float
    regularization: float
    reference_acceleration: float
    normal_velocity: float


@dataclass(frozen=True)
class MuJoCoTyxNormalContact:
    """One detected site; world wrenches use the same point and each body COM."""

    schema: str
    geometry_ids: tuple
    body_ids: tuple
    feature_index: int
    dimension: int
    distance: float
    position: np.ndarray
    frame: np.ndarray
    include_margin: float
    friction: np.ndarray
    solref: np.ndarray
    solimp: np.ndarray
    active: bool
    wrench_contact_frame: np.ndarray
    force_on_geometry1_world: np.ndarray
    force_on_geometry2_world: np.ndarray
    wrench_on_body1_world: np.ndarray
    wrench_on_body2_world: np.ndarray
    impedance: float
    regularization: float
    reference_acceleration: float
    normal_velocity: float
    jacobian: np.ndarray
    normal_force: float


@dataclass(frozen=True)
class MuJoCoTyxDynamics:
    """Same input-step device fields; linear/angular Jacobian rows in world axes.

    applied includes actuator effort, qfrc_applied and the COM world wrench
    projection. passive is separate. For native constraints, M @ smooth_acceleration
    = applied + passive - bias. The opt-in area-pressure route uses M+hD on the
    left instead, where D is the actual joint damping matrix; this is the free
    acceleration response used by its backward-Euler velocity solve.
    """

    body_position: np.ndarray
    body_rotation: np.ndarray
    body_com: np.ndarray
    jacobian_com: np.ndarray
    mass_matrix: np.ndarray
    bias: np.ndarray
    passive: np.ndarray
    applied: np.ndarray
    smooth_acceleration: np.ndarray
    input_q: np.ndarray
    input_v: np.ndarray
    time: float
    phase: str = "tyx-euler-input-dynamics"


@dataclass(frozen=True)
class MuJoCoTyxStepReceipt:
    world: int
    tick: int
    input_time: float
    output_time: float
    h: float
    control: MuJoCoControl
    stored_control: MuJoCoControl
    contacts: tuple
    input_q: np.ndarray
    input_v: np.ndarray
    qacc: np.ndarray
    integration_acceleration: np.ndarray
    qfrc_actuator: np.ndarray
    qfrc_constraint: np.ndarray
    raw_contact_fields: object
    source_solver: str
    numerical_solver: str
    contact_time: float
    backend: str
    contact_phase: str = "tyx-euler-input-constraint-evaluation"
    contact_law: str = LAW
    state_precision: str = "fp32-device-integration"
    dynamics: MuJoCoTyxDynamics | None = None
    constraints: tuple = ()


def _require_musa_plan(plan, model, mj):
    from .mujoco_tyx import MuJoCoTyxIssue
    from .mujoco_tyx._constraint_model import TyxConstraintModel
    from .mujoco_tyx._model import TyxModel
    from .mujoco_tyx._tree_model import TyxTreeModel

    if type(plan) is TyxModel and plan.kind == 0 and not np.any(model.jnt_limited):
        # compile_model has already checked the all-prismatic layout, fixed
        # mass matrix, and at most one plane/sphere normal pair.
        return
    if (
        type(plan) is TyxConstraintModel
        and type(plan.base) is TyxTreeModel
        and plan.base.kind == 2
        and np.all(np.isin(model.jnt_type, (mj.mjtJoint.mjJNT_SLIDE, mj.mjtJoint.mjJNT_HINGE)))
        and not (plan.extended_contacts or plan.nlimit or plan.nfriction)
        and plan.npair
        and np.all(np.isin(plan.pair_condim[: plan.npair], (1, 3)))
    ):
        return
    raise MuJoCoTyxUnsupported(
        (
            MuJoCoTyxIssue(
                "musa-domain",
                "MUSA admits the unrestricted prismatic solver or a fixed-base scalar-joint "
                "tree with basic primitive pyramidal contacts (condim 1/3), without joint rows",
            ),
        )
    )


class MuJoCoTyxRuntime(MuJoCoRuntime):
    """Own TYX interpret, CUDA or MUSA execution and atomic publication for sparse worlds.

    MUSA executes the unrestricted prismatic route and the fixed-base scalar
    tree's basic primitive contact route with native pyramidal condim 1/3.

    Floating domain: one native free root and 0..29 hinge/slide joints, full
    root/joint coupling, origin world linear and local angular velocities.
    Rotational domain: fixed-root 1..8 hinge/slide DOFs, complete live M/J/bias,
    welded bodies and off-center world wrenches. Both tree domains admit the
    bounded primitive condim=1 manifold with coupled normal forces.
    Ball extension: fixed-root ball/hinge/slide with at most 8 velocity and 10
    position coordinates, complete SO(3) integration and full M/J/bias. This
    extension permits scalar JOINT actors and direct BALL motors, including
    admitted FILTEREXACT activation and primitive normal/pyramidal3 contact.
    The same fixed/free BALL trees admit scalar soft limits and native DOF dry
    friction through the mixed constraint solver. BALL angular limits remain
    unsupported; capability records distinguish joint-only and contact composition.
    Original domain: 1..8 all-prismatic native DOFs with full coupled static
    mass, stateless direct/affine motors, scalar stiffness/armature/damping; or
    one centered isotropic free body, with native local angular coordinates and
    real world Cartesian body torque. At most one possible static plane/sphere
    pair with condim=1. Its soft-normal QP is solved directly, not replaced with
    a hard-contact projection or force-gap spring. Unsupported semantics fail
    construction. Legacy direct routes retain strict-interior guards; supported
    scalar limits select the mixed soft-constraint route before device allocation.
    """

    def __init__(
        self,
        compilation,
        *,
        world_keys=(0,),
        initial_states=None,
        mode="cuda",
        implementation=None,
        contact_capacity=64,
        constraint_capacity=256,
        cache_root=None,
        resource_budget=None,
        area_pressure_pairs=(),
        area_friction_laws=(),
    ):
        if not isinstance(area_pressure_pairs, tuple):
            raise ValueError("area_pressure_pairs must be a tuple")
        if mode == "musa" and (area_pressure_pairs or area_friction_laws):
            raise ValueError("MUSA does not support area-pressure or area-friction declarations")
        from .area_friction_law import capture_pair_laws

        self._area_friction_laws = capture_pair_laws(area_pressure_pairs, area_friction_laws)
        if self._area_friction_laws and mode != "cuda":
            raise ValueError("live area friction currently requires CUDA")
        self._area_pressure_pairs = area_pressure_pairs
        self._area_packet = None
        from .mujoco_resource_budget import MuJoCoTyxResourceBudget

        if resource_budget is not None and type(resource_budget) is not MuJoCoTyxResourceBudget:
            raise TypeError("resource_budget requires MuJoCoTyxResourceBudget")
        self._resource_budget = resource_budget
        if implementation not in (None, "tyx"):
            raise ValueError("MuJoCoTyxRuntime requires implementation='tyx'")
        if mode not in ("interpret", "cuda", "musa"):
            raise ValueError("MuJoCo TYX mode must be interpret, cuda or musa")
        self._contact_capacity = _capacity(contact_capacity, "contact_capacity")
        self._constraint_capacity = _capacity(constraint_capacity, "constraint_capacity")
        self._mode, self._cache_root = mode, cache_root
        self._preparing_batch = False
        self._prepared_step = self._prepared_controls = None
        self._mirror_layout, self._observer_data = None, None
        super().__init__(
            compilation, world_keys=world_keys, initial_states=initial_states, mode="cpu"
        )
        # A shared payload budget also serializes its related owner operations.
        # This prevents public admit() -> owner vs owner -> admit() lock inversion.
        if resource_budget is not None:
            self._lock = resource_budget._lock
        if area_pressure_pairs:
            from .mujoco_area_pressure import compile_area_plan

            self._plan = compile_area_plan(
                self._mj, self._model, area_pressure_pairs, self._area_friction_laws
            )
        else:
            self._plan = compile_model(self._mj, self._model)
        if mode == "musa":
            _require_musa_plan(self._plan, self._model, self._mj)
        if getattr(self._plan, "constraint_contact", False):
            if self._contact_capacity > 128 or self._constraint_capacity > 512:
                raise ValueError("T4 contact/constraint capacities cannot exceed 128/512")
        from .mujoco_tyx._allocation_manifest import allocation_manifest

        self._allocation_manifest = allocation_manifest(
            self._plan,
            len(self._world_keys),
            self._model.nbody,
            (self._contact_capacity, self._constraint_capacity),
        )
        self._checkpoint_revision = 0
        self._checkpoint_retired = []
        self._control_packet, self._control_retired = None, []
        self._control_epoch, self._control_cleanup_errors = 0, ()
        self._extra_counts = dict(
            area_copy_bytes=0,
            area_copies=0,
            completed_steps=0,
            accepted_world_steps=0,
            failed_batches=0,
            device_control_submissions=0,
            device_control_refusals=0,
            control_copy_d2d_bytes=0,
            control_audit_downloaded_bytes=0,
            control_status_downloaded_bytes=0,
            control_initial_uploaded_bytes=0,
            control_pack_kernels=0,
        )
        quantized = self._quantized_mirrors(self._data)
        self._device = self._build_device(self._data)
        self._data = quantized
        info = dict(self._capabilities)
        info.update(
            backend=f"mujoco-tyx-{mode}",
            mode=mode,
            implementation="tyx",
            gpu=mode in ("cuda", "musa"),
            native_model=False,
            contact_law=LAW,
            domain=(
                "tree-primitive-coupled-normal-v1"
                if getattr(self._plan, "normal_contact", False)
                else "free-root-hinge-slide-no-contact-v1"
                if self._plan.kind == 3
                else "fixed-ball-joint-motor-no-contact-v1"
                if getattr(self._plan, "joint_transmission", None) is not None
                else "fixed-ball-scalar-actuation-no-contact-v1"
                if self._model.nu and np.any(self._model.jnt_type == self._mj.mjtJoint.mjJNT_BALL)
                else "fixed-ball-hinge-slide-no-contact-v1"
                if np.any(self._model.jnt_type == self._mj.mjtJoint.mjJNT_BALL)
                else "fixed-hinge-slide-no-contact-v1"
                if self._plan.kind == 2
                else "prismatic-or-isotropic-free/one-normal-pair-v1"
            ),
            state_precision="fp32-device-integration",
            publication="whole-batch-atomic",
            possible_contact_pairs=(
                len(self._plan.pair_ids)
                if getattr(self._plan, "normal_contact", False)
                else int(self._plan.pair is not None)
            ),
            limits="strict-interior-reject-on-exit",
            numerical_solver=(
                "normal-dual-active-set-v1"
                if getattr(self._plan, "normal_contact", False)
                else "tree-dense-cholesky"
                if self._plan.kind in (2, 3)
                else "single-normal-direct-qp"
            ),
            source_solver=self._plan.source_solver,
            source_iterations=self._plan.source_iterations,
            contact_capacity=self._contact_capacity,
            constraint_capacity=self._constraint_capacity,
            endpoint_observer="detached-cpu-mj-forward",
            capability_scope="admitted-model-domain; trajectory-validation-separate",
            mixed_articulated_constraints_supported=False,
            floating_mixed_constraints_supported=False,
            deprecated_capabilities=("g1_supported",),
            checkpoint="exact-accepted-tyx-batch-v1",
            device_controls="copy-owned-native-batch-cas-v1",
            device_control_host_audit="exact-channels-on-submission",
            g1_supported=False,
        )
        if mode == "musa" and not getattr(self._plan, "constraint_contact", False):
            has_contact_pair = self._plan.pair is not None
            family = "single-slide" if self._plan.nv == 1 else "prismatic"
            info.update(
                domain=(
                    f"{family}-plane-sphere-normal-v1"
                    if has_contact_pair
                    else f"{family}-no-contact-v1"
                ),
                contact_law=LAW if has_contact_pair else "none",
                contact_dimensions=(1,) if has_contact_pair else (),
                numerical_solver=(
                    "single-normal-direct-qp" if has_contact_pair else "constant-mass-damped-euler"
                ),
                friction="none",
                limits="unsupported",
                device_controls=False,
                device_control_host_audit=None,
            )
        if getattr(self._plan, "normal_contact", False):
            info.update(
                normal_candidate_slots=len(self._plan.slot_ids),
                normal_geometry_launches=2 if self._plan.extended_contacts else 1,
                normal_pair_families=(
                    "plane-sphere",
                    "plane-capsule",
                    "plane-box",
                    "sphere-sphere",
                    "sphere-capsule",
                    "sphere-box",
                    "capsule-capsule",
                    "capsule-box",
                    "box-box",
                ),
                extended_feature_identity="current-manifold-local-ordinal",
                solver_tolerance=self._plan.solver_tolerance,
                solver_iteration_semantics="source-iterations-as-active-set-face-solve-budget",
                warmstart="cold-feasible-zero-force",
            )
        self._capabilities = MappingProxyType(info)
        if getattr(self._plan, "constraint_contact", False):
            info.update(
                contact_law=MIXED_LAW,
                domain="tree-pyramidal-mixed-constraints-v1",
                possible_contact_pairs=self._plan.npair,
                candidate_slots=self._plan.nslot,
                limits="native-scalar-soft-constraints",
                friction="native-dof-loss-and-pyramidal-condim1-3-4-6",
                numerical_solver="pyramidal-box-active-set-v1",
                constraint_compaction="stable-device-active-rows",
                solver_tolerance=self._plan.solver_tolerance,
                solver_iteration_semantics="source-iterations-as-active-set-face-solve-budget",
                warmstart="cold-feasible-zero-force",
                sensors="detached-cpu-observation-only",
                mixed_articulated_constraints_supported=self._plan.kind in (2, 3),
                floating_mixed_constraints_supported=self._plan.kind == 3,
                # Compatibility alias: model-domain support, not a trajectory claim.
                g1_supported=self._plan.kind == 3,
            )
            if mode == "musa":
                dimensions = self._plan.pair_condim[: self._plan.npair]
                info.update(
                    contact_dimensions=tuple(sorted(set(map(int, dimensions)))),
                    limits="unsupported",
                    friction="pyramidal-condim1-3-only",
                    device_controls=False,
                    device_control_host_audit=None,
                )
            self._capabilities = MappingProxyType(info)

        if hasattr(self._plan, "activation"):
            domain = (
                "tree-pyramidal-mixed-filterexact-v1"
                if getattr(self._plan, "constraint_contact", False)
                else "tree-normal-contact-filterexact-v1"
                if getattr(self._plan, "normal_contact", False)
                else "free-root-scalar-filterexact-v1"
                if self._plan.kind == 3
                else "fixed-ball-joint-motor-filterexact-no-contact-v1"
                if getattr(self._plan, "joint_transmission", None) is not None
                else "fixed-ball-scalar-filterexact-no-contact-v1"
                if np.any(self._model.jnt_type == self._mj.mjtJoint.mjJNT_BALL)
                else "fixed-hinge-slide-filterexact-v1"
            )
            info.update(
                domain=domain,
                actuator_state="native-filterexact-device-bank-v1",
                checkpoint="exact-accepted-tyx-batch-v2",
            )
            self._capabilities = MappingProxyType(info)

        if self._plan.kind == 3 and np.any(self._model.jnt_type == self._mj.mjtJoint.mjJNT_BALL):
            activation = "-filterexact" if hasattr(self._plan, "activation") else ""
            info.update(
                domain=(
                    f"free-ball-joint-motor{activation}-no-contact-v1"
                    if getattr(self._plan, "joint_transmission", None) is not None
                    else "free-ball-no-contact-v1"
                ),
                limits="unsupported",
                joint_limits_supported=False,
                dry_friction_supported=False,
                mixed_articulated_constraints_supported=False,
                floating_mixed_constraints_supported=False,
                g1_supported=False,
            )
            self._capabilities = MappingProxyType(info)

        if np.any(self._model.jnt_type == self._mj.mjtJoint.mjJNT_BALL) and (
            getattr(self._plan, "normal_contact", False)
            or getattr(self._plan, "constraint_contact", False)
        ):
            pyramidal = bool(getattr(self._plan, "constraint_contact", False))
            family = "pyramidal" if pyramidal else "normal"
            activation = "-filterexact" if hasattr(self._plan, "activation") else ""
            actuator_family = (
                "joint-motor"
                if getattr(self._plan, "joint_transmission", None) is not None
                else "scalar"
            )
            if self._plan.kind == 3 and np.any(
                self._model.jnt_type == self._mj.mjtJoint.mjJNT_BALL
            ):
                domain = (
                    f"free-ball-joint-motor-{family}-contact{activation}-v1"
                    if getattr(self._plan, "joint_transmission", None) is not None
                    else f"free-ball-{family}-contact-v1"
                )
            else:
                domain = f"fixed-ball-{actuator_family}-{family}-contact{activation}-v1"
            info.update(
                domain=domain,
                contact_dimensions=(1, 3) if pyramidal else (1,),
                limits="unsupported",
                joint_limits_supported=False,
                dry_friction_supported=False,
                friction="pyramidal-condim1-3-only" if pyramidal else "none",
                mixed_articulated_constraints_supported=False,
                floating_mixed_constraints_supported=False,
                g1_supported=False,
            )
            self._capabilities = MappingProxyType(info)

        if (
            np.any(self._model.jnt_type == self._mj.mjtJoint.mjJNT_BALL)
            and getattr(self._plan, "constraint_contact", False)
            and (self._plan.nfriction or self._plan.nlimit)
        ):
            # A T4 wrapper also represents joint-only models. Report its actual
            # declared row families without borrowing the contact-only labels.
            family = "free" if self._plan.kind == 3 else "fixed"
            rows = "mixed-contact-joint-constraints" if self._plan.npair else "joint-constraints"
            activation = "-filterexact" if hasattr(self._plan, "activation") else ""
            dimensions = tuple(sorted(set(map(int, self._plan.pair_condim[: self._plan.npair]))))
            contact_rows = sum(
                1 if int(self._plan.pair_condim[int(slot[0])]) == 1 else 4
                for slot in self._plan.slot_ids[: self._plan.nslot]
            )
            info.update(
                domain=f"{family}-ball-{rows}{activation}-v1",
                contact_dimensions=dimensions,
                limits="native-scalar-soft-constraints" if self._plan.nlimit else "none",
                joint_limits_supported=bool(self._plan.nlimit),
                scalar_joint_limits_supported=bool(self._plan.nlimit),
                ball_angular_limits_supported=False,
                dry_friction_supported=bool(self._plan.nfriction),
                friction=(
                    "native-dof-loss-and-pyramidal-condim1-3"
                    if self._plan.nfriction and self._plan.npair
                    else "native-dof-loss"
                    if self._plan.nfriction
                    else "pyramidal-condim1-3-only"
                    if self._plan.npair
                    else "none"
                ),
                native_dry_dof_ids=tuple(map(int, self._plan.friction_ids[: self._plan.nfriction])),
                native_scalar_limit_joint_ids=tuple(
                    map(int, self._plan.limit_ids[: self._plan.nlimit, 0])
                ),
                dry_constraint_rows=self._plan.nfriction,
                scalar_limited_joints=self._plan.nlimit,
                conservative_efc_bound=self._plan.nfriction + 2 * self._plan.nlimit + contact_rows,
                constraint_capacity_policy="actual-active-rows; static-bound-diagnostic-only",
                mixed_articulated_constraints_supported=True,
                floating_mixed_constraints_supported=self._plan.kind == 3,
                g1_supported=False,
            )
            self._capabilities = MappingProxyType(info)

        if self._plan.kind == 3:
            tree_roots = tuple(
                sorted(set(map(int, self._model.body_rootid[self._model.jnt_bodyid])))
            )
            info.update(
                articulation_layout="native-joint-address-forest-v1",
                articulation_root_body_ids=tree_roots,
                free_root_body_ids=self._plan.free_root_bodies,
                max_generalized_velocities=35,
                max_generalized_positions=45,
            )
            if len(tree_roots) > 1:
                info["domain"] = info["domain"].replace("free-root-", "forest-")
                info["g1_supported"] = False
            self._capabilities = MappingProxyType(info)

        if self._area_pressure_pairs:
            info.update(
                area_pressure_pairs=self._plan.area_pair_names,
                area_pressure_law="frozen-surface-elastic-backward-euler-v1",
                area_pressure_input="fresh-host-surface-per-world-and-pair",
                area_pressure_publication="last_step.raw_contact_fields.area_pressure",
                area_pressure_entry_abi="mujoco-tyx-area-device-entry-v1",
                area_pressure_checkpoint=True,
                area_pressure_observation="host-owned-accepted-receipt-v1",
                area_pressure_friction=bool(self._area_friction_laws),
                smooth_acceleration_operator="M"
                if self._plan.damped_contact_dofs
                else "M+h*joint_damping",
            )
            if self._area_friction_laws:
                from .area_friction_law import LAW as FRICTION_LAW

                info.update(
                    contact_law=FRICTION_LAW,
                    domain=(
                        "area-friction-cuda-scalar-hinge-v1"
                        if (self._plan.nfriction or self._plan.nlimit)
                        and np.any(self._plan.joint_ids[:, 0] == 1)
                        else "area-friction-cuda-slide-limit-v1"
                        if self._plan.nlimit
                        else "area-friction-cuda-coupled-slides-v1"
                        if self._plan.nfriction and self._plan.nv == 2
                        else "area-friction-cuda-dry-slide-v1"
                        if self._plan.nfriction
                        else "area-friction-cuda-six-dof-v1"
                        if self._plan.nv <= 6
                        else "area-friction-cuda-eight-dof-v1"
                    ),
                    numerical_solver="area-friction-newton-v1",
                    area_pressure_law=FRICTION_LAW,
                    area_pressure_publication="last_step.raw_contact_fields.area_friction",
                    area_pressure_entry_abi="mujoco-tyx-area-device-entry-v2",
                    area_pressure_integration="common-velocity-native-coordinate-tail-v1",
                    smooth_acceleration_operator="M+h*joint_damping",
                    native_guide_friction=bool(self._plan.nfriction),
                    scalar_joint_limits_supported=bool(self._plan.nlimit),
                    joint_limits_supported=bool(self._plan.nlimit),
                    limits="native-scalar-soft-constraints" if self._plan.nlimit else "none",
                    native_scalar_limit_joint_ids=tuple(
                        map(int, self._plan.limit_ids[: self._plan.nlimit, 0])
                    ),
                    native_dry_dof_ids=tuple(
                        map(int, self._plan.friction_ids[: self._plan.nfriction])
                    ),
                    area_pressure_observation="host-owned-friction-receipt-v1",
                    mixed_articulated_constraints_supported=False,
                    g1_supported=False,
                )
                if self._plan.damped_contact_dofs:
                    info.update(
                        domain=(
                            f"area-friction-cuda-damped-coupled-support-{self._plan.coupled_contact_variant}"
                            if self._plan.coupled_damped_contacts
                            else "area-friction-cuda-damped-scalar-support-v1"
                        ),
                        native_contact_operator="M",
                        native_contact_scope="two-condim3-plane-sphere"
                        if self._plan.npair == 2
                        else "single-condim3-plane-sphere",
                    )
            if getattr(self._plan, "resident_native_contacts", False):
                from .area_friction_law import numerical_solver

                info.update(
                    domain="area-friction-cuda-resident-native-free-scalar-v1",
                    numerical_solver=numerical_solver(self._plan),
                    native_contact_operator="M",
                    native_contact_scope="source-condim3-plane-primitives-with-separated-other-candidates",
                    area_pressure_integration="native-damped-euler-coordinate-tail-v1",
                    mixed_articulated_constraints_supported=True,
                )
            self._capabilities = MappingProxyType(info)

    @property
    def counters(self):
        return MappingProxyType({**self._device.counts, **self._extra_counts})

    def resource_snapshot(self):
        """Solver-bank array payload only; native contexts and HOST peaks remain unknown."""
        with self._lock:
            device = getattr(self, "_device", None)
            return MappingProxyType(
                dict(
                    schema="mujoco-tyx-runtime-resources-v1",
                    mode=self._mode,
                    closed=self._closed,
                    manifest=self._allocation_manifest,
                    allocation=None if device is None else device.allocation_snapshot(),
                    budget=None
                    if self._resource_budget is None
                    else self._resource_budget.snapshot(),
                    retained_owner_count=len(self._checkpoint_retired),
                    completeness=(
                        "Solver-bank MUSA payload only; HOST/native/context/JIT/external controls/workspaces unknown"
                        if self._mode == "musa"
                        else "Solver-bank CUDA payload only; HOST/native/context/JIT/external controls/workspaces unknown"
                    ),
                )
            )

    def musa_device_report(self):
        """Return detached driver and launch counters for the active MUSA bank owner."""
        with self._lock:
            self._check_open()
            if self._mode != "musa":
                raise ValueError("musa_device_report requires mode='musa'")
            report = copy.deepcopy(dict(self._device.runtime.report()))
            report.update(active_bank=self._device.active, revision=self._checkpoint_revision)
            return report

    def _initial_data(self, packet):
        # Preserve the original FP64 declaration checks and native initial
        # observer, then release that one arena before constructing another world.
        native = super()._initial_data(packet)
        if self._mirror_layout is None:
            self._mirror_layout = NativeStateLayout(self._mj, self._model)
        return self._mirror_layout.capture(native)

    def _observe_native(self, mirror):
        # Called only while the runtime lock is held. No native view escapes.
        if self._observer_data is None:
            self._observer_data = self._mj.MjData(self._model)
        observed = self._mirror_layout.materialize(mirror, self._observer_data)
        self._mj.mj_forward(self._model, observed)
        self._finite_state(observed)
        return observed

    def _pack_states(self, data):
        q = np.zeros((len(self._world_keys), max(8, self._plan.nq)), np.float32)
        v = np.zeros((len(self._world_keys), max(8, self._plan.nv)), np.float32)
        clock = np.zeros((len(self._world_keys), 2), np.float32)
        for index, key in enumerate(self._world_keys):
            original = data[key]
            q[index, : self._plan.nq] = fp32(original.qpos, "initial qpos")
            v[index, : self._plan.nv] = fp32(original.qvel, "initial qvel")
            clock[index, 0] = fp32(original.time, "initial time").item()
            clock[index, 1] = np.float32(float(clock[index, 0]) - original.time)
            if self._plan.kind in (2, 3):
                scalar = np.flatnonzero(
                    np.isin(
                        self._model.jnt_type,
                        (self._mj.mjtJoint.mjJNT_HINGE, self._mj.mjtJoint.mjJNT_SLIDE),
                    )
                )
                qa = self._model.jnt_qposadr[scalar]
                dof = self._model.jnt_dofadr[scalar]
                low, high = self._plan.vectors[7:9, dof]
                if np.any(q[index, qa] <= low) or np.any(q[index, qa] >= high):
                    raise ValueError("initial native q is outside the strict joint-limit interior")
            elif self._plan.kind != 1:
                low, high = self._plan.vectors[7:9, : self._plan.nv]
                if np.any(q[index, : self._plan.nq] <= low) or np.any(
                    q[index, : self._plan.nq] >= high
                ):
                    raise ValueError("initial native q is outside the strict joint-limit interior")
        if hasattr(self._plan, "activation"):
            act = np.empty((len(self._world_keys), self._plan.activation.na), np.float32)
            for index, key in enumerate(self._world_keys):
                act[index] = fp32(data[key].act, "initial native activation")
            return q, v, clock, act
        return q, v, clock

    def _build_device(self, data, *, packed=None):
        from .mujoco_tyx._device import TyxDevice

        device_type = TyxDevice
        if getattr(self._plan, "constraint_contact", False):
            from .mujoco_tyx._constraint_device import TyxConstraintDevice

            device_type = TyxConstraintDevice
        elif getattr(self._plan, "normal_contact", False):
            from .mujoco_tyx._normal_device import TyxNormalDevice

            device_type = TyxNormalDevice
        elif self._plan.kind == 2:
            from .mujoco_tyx._tree_device import TyxTreeDevice

            device_type = TyxTreeDevice
        elif self._plan.kind == 3:
            from .mujoco_tyx._floating_device import TyxFloatingDevice

            device_type = TyxFloatingDevice
        packed_state = self._pack_states(data) if packed is None else packed
        activation_args = (
            {"activation": packed_state[3]} if hasattr(self._plan, "activation") else {}
        )
        device = device_type(
            self._plan,
            self._mode,
            self._cache_root,
            *packed_state[:3],
            self._model.nbody,
            self._timestep,
            (self._contact_capacity, self._constraint_capacity),
            resource_budget=self._resource_budget,
            entry_body_ids=(
                tuple(int(v) for v in self._model.geom_bodyid[list(self._plan.pair)])
                if getattr(self._plan, "pair", None) is not None
                and not getattr(self._plan, "normal_contact", False)
                and not getattr(self._plan, "constraint_contact", False)
                else None
            ),
            **activation_args,
        )
        # All device subclasses expose the same transport counters, including
        # fresh replacement owners used by A1's exact checkpoint continuation.
        device.counts["device_control_d2d_bytes"] = 0
        device.counts["control_host_uploaded_bytes"] = 0
        return device

    def _quantized_mirrors(self, data):
        packed = self._pack_states(data)
        q, v, time = packed[:3]
        prepared = {}
        for index, key in enumerate(self._world_keys):
            updates = dict(
                qpos=q[index, : self._plan.nq],
                qvel=v[index, : self._plan.nv],
                time=float(time[index, 0]) - float(time[index, 1]),
            )
            if hasattr(self._plan, "activation"):
                updates["act"] = packed[3][index]
            prepared[key] = data[key].changed(**updates)
        return prepared

    def _check_open(self):
        super()._check_open()
        if (
            getattr(self, "_preparing_batch", False)
            or getattr(self, "_prepared_step", None) is not None
            or getattr(self, "_prepared_controls", None) is not None
        ):
            raise RuntimeError("runtime is reserved by an unpublished batch transaction")

    def _prepare_controls_locked(self, mapping):
        with self._lock:
            self._check_open()
            self._preparing_batch = True
            try:
                self._exact_mapping(mapping, "controls")
                prepared = {key: self._validated_control(mapping[key]) for key in self._world_keys}
                for packet in prepared.values():
                    for value in (packet.ctrl, packet.qfrc_applied, packet.xfrc_applied):
                        fp32(value, "held native control")
                from .mujoco_device_controls import retirement_list

                retired = retirement_list(self)
                draft = _PreparedControls(
                    self,
                    self._checkpoint_revision,
                    self._controls,
                    prepared,
                    retired,
                    self._checkpoint_revision + 1,
                )
            finally:
                self._preparing_batch = False
            self._prepared_controls = draft
            return draft

    def _validate_controls_draft_locked(self, draft):
        MuJoCoRuntime._check_open(self)
        if (
            type(draft) is not _PreparedControls
            or draft is not self._prepared_controls
            or draft.owner is not self
            or draft.revision != self._checkpoint_revision
            or draft.entry_controls is not self._controls
        ):
            raise ValueError("stale or foreign controls draft")

    def _commit_controls_draft_locked(self, draft):
        # Internal coordinator validates every draft before entering this section.
        self._controls, self._control_retired = draft.controls, draft.retired
        self._checkpoint_revision = draft.destination_revision
        self._control_packet = self._prepared_controls = None

    def _discard_controls_draft_locked(self, draft=None):
        if draft is None or draft is self._prepared_controls:
            self._prepared_controls = None

    def _finish_controls_cleanup_locked(self):
        from .mujoco_device_controls import _cleanup_retired

        _cleanup_retired(self)

    def set_controls(self, mapping):
        with self._lock:
            draft = self._prepare_controls_locked(mapping)
            try:
                self._validate_controls_draft_locked(draft)
            except BaseException:
                self._discard_controls_draft_locked(draft)
                raise
            self._commit_controls_draft_locked(draft)
            self._finish_controls_cleanup_locked()

    def set_controls_from_device(
        self, ctrl, *, qfrc_applied=None, xfrc_applied=None, expected_revision, producer_stream=None
    ):
        """Copy one ordered native action batch, then atomically publish held controls.

        Inputs use world_keys order; equal-shaped row permutations cannot be inferred.
        None forces are fresh zeros. After return callers may mutate/free their arrays.
        A later failed physics step retains this successfully submitted held action.
        """
        from .mujoco_device_controls import submit

        with self._lock:
            self._check_open()
            try:
                if self._mode == "musa":
                    raise ValueError("MUSA accepts host MuJoCoControl inputs; device control import is unavailable")
                return submit(
                    self, (ctrl, qfrc_applied, xfrc_applied), expected_revision, producer_stream
                )
            except BaseException:
                self._extra_counts["device_control_refusals"] += 1
                raise

    @property
    def control_cleanup_errors(self):
        """Separate resource errors; never a rejection of an accepted action."""
        with self._lock:
            return self._control_cleanup_errors

    def _pack_controls(self):
        if self._control_packet is not None:
            return self._control_packet.inputs(self)
        ctrl = np.zeros((len(self._world_keys), max(self._plan.nu, 1)), np.float32)
        applied = np.zeros((len(self._world_keys), max(8, self._plan.nv)), np.float32)
        body_force = np.zeros((len(self._world_keys), self._model.nbody, 6), np.float32)
        stored = {}
        for index, key in enumerate(self._world_keys):
            packet = self._controls[key]
            ctrl[index, : self._plan.nu] = fp32(packet.ctrl, "ctrl")
            applied[index, : self._plan.nv] = fp32(packet.qfrc_applied, "qfrc_applied")
            body_force[index] = fp32(packet.xfrc_applied, "xfrc_applied")
            stored[key] = MuJoCoControl(
                ctrl[index, : self._plan.nu],
                qfrc_applied=applied[index, : self._plan.nv],
                xfrc_applied=body_force[index],
            )
        return ctrl, applied, body_force, stored

    def _contacts_from_row(self, row):
        if int(row[0]) == 0:
            return ()
        if int(row[0]) != 1 or self._plan.pair is None:
            raise RuntimeError("TYX returned an invalid contact count")
        law = self._plan.contact
        force = np.r_[row[3], np.zeros(5)]
        world_force = float(row[3]) * law[3:6].astype(np.float64)
        return (
            MuJoCoTyxContact(
                schema="mujoco-tyx-plane-sphere-normal-v1",
                geometry_ids=self._plan.pair,
                dimension=1,
                distance=float(row[2]),
                position=_frozen(row[9:12]),
                frame=_frozen(self._plan.frame),
                include_margin=float(law[10] - law[11]),
                friction=_frozen(law[24:29]),
                solref=_frozen(law[14:16]),
                solimp=_frozen(law[16:21]),
                active=bool(row[1]),
                wrench_contact_frame=_frozen(force),
                force_on_sphere_world=_frozen(world_force),
                static_reaction_world=_frozen(-world_force),
                impedance=float(row[8]),
                regularization=float(row[4]),
                reference_acceleration=float(row[6]),
                normal_velocity=float(row[7]),
            ),
        )

    def _normal_publication(self, result, index):
        plan = self._plan
        rows = result["normal_rows"][index]
        jacobian = result["normal_jacobian"][index, :, : plan.nv]
        matrix = result["dual_matrix"][index]
        diag = result["solver_diagnostics"][index]
        count = len(plan.slot_ids)
        if rows.shape != (count, 32) or jacobian.shape != (count, plan.nv):
            raise RuntimeError("invalid normal row shape")
        if matrix.shape != (count, count) or diag.shape != (8,):
            raise RuntimeError("invalid normal solver shape")
        if not all(np.all(np.isfinite(x)) for x in (rows, jacobian, matrix, diag)):
            raise RuntimeError("nonfinite normal contact result")
        if np.any(~np.isin(rows[:, :2], (0, 1))) or np.any(rows[:, 1] > rows[:, 0]):
            raise RuntimeError("invalid detected/active normal flags")
        present = int(np.sum(rows[:, 0]))
        active = int(np.sum(rows[:, 1]))
        if (
            present > self._contact_capacity
            or active > self._constraint_capacity
            or diag[4] != present
            or diag[5] != active
            or diag[6] != 1
            or diag[1] > plan.solver_tolerance
            or diag[7] != plan.solver_iterations
            or not 0 <= diag[0] <= plan.solver_iterations
            or diag[0] != int(diag[0])
            or np.any(diag[1:4] < 0)
        ):
            raise RuntimeError("invalid normal capacity or convergence receipt")
        contacts = []
        for slot, row in enumerate(rows):
            if not row[0]:
                continue
            pair, feature, geom1, geom2 = map(int, plan.slot_ids[slot])
            body1, body2 = map(int, plan.geom_ids[[geom1, geom2], 0])
            law = plan.pair_law[pair]
            frame = row[12:21].reshape(3, 3)
            point = row[9:12]
            force = float(row[3]) * frame[0].astype(np.float64)
            if row[3] < 0 or (not row[1] and row[3] != 0):
                raise RuntimeError("invalid normal force")
            com1, com2 = result["body_com"][index, [body1, body2]]
            contacts.append(
                MuJoCoTyxNormalContact(
                    schema="mujoco-tyx-primitive-normal-v1",
                    geometry_ids=(geom1, geom2),
                    body_ids=(body1, body2),
                    feature_index=feature,
                    dimension=1,
                    distance=float(row[2]),
                    position=_frozen(point),
                    frame=_frozen(frame),
                    include_margin=float(row[21]),
                    friction=_frozen(plan.pair_friction[pair]),
                    solref=_frozen(law[3:5]),
                    solimp=_frozen(law[5:10]),
                    active=bool(row[1]),
                    wrench_contact_frame=_frozen(np.r_[row[3], np.zeros(5)]),
                    force_on_geometry1_world=_frozen(-force),
                    force_on_geometry2_world=_frozen(force),
                    wrench_on_body1_world=_frozen(np.r_[-force, np.cross(point - com1, -force)]),
                    wrench_on_body2_world=_frozen(np.r_[force, np.cross(point - com2, force)]),
                    impedance=float(row[8]),
                    regularization=float(row[4]),
                    reference_acceleration=float(row[6]),
                    normal_velocity=float(row[7]),
                    jacobian=_frozen(jacobian[slot]),
                    normal_force=float(row[3]),
                )
            )
        diagnostics = MappingProxyType(
            dict(
                zip(
                    (
                        "iterations",
                        "normalized_kkt",
                        "absolute_kkt",
                        "min_force",
                        "present_count",
                        "active_count",
                        "converged",
                        "budget",
                    ),
                    map(float, diag),
                    strict=True,
                )
            )
        )
        return tuple(contacts), MappingProxyType(
            {
                "device_rows": _frozen(rows),
                "normal_jacobian": _frozen(jacobian),
                "dual_matrix": _frozen(matrix),
                "slot_ids": _frozen(plan.slot_ids, dtype=np.int32),
                "solver_diagnostics": diagnostics,
            }
        )

    def area_pressure_configuration_id(self, key):
        """Bind supplied surface geometry to this exact accepted entry."""
        from .mujoco_area_pressure import configuration_id

        with self._lock:
            return configuration_id(self, key)

    def step_area_pressure(self, h, mapping, *, expected_revision):
        """Consume one fresh surface batch and use the existing atomic step."""
        from .mujoco_area_pressure import pack

        with self._lock:
            self._check_open()
            self._area_packet = pack(self, mapping, expected_revision)
            try:
                return self.step(h)
            finally:
                self._area_packet = None

    def step(self, h):
        with self._lock:
            draft = self._prepare_step_locked(h)
            try:
                self._validate_step_draft_locked(draft)
            except BaseException:
                self._discard_step_draft_locked(draft)
                raise
            self._commit_step_draft_locked(draft)
            return draft.accepted

    def _prepare_step_locked(self, h):
        with self._lock:
            self._check_open()
            if not np.isfinite(h) or float(h) != self._timestep:
                raise ValueError("step h must equal the original compiled MuJoCo timestep")
            if self._area_pressure_pairs and self._area_packet is None:
                raise ValueError("area route requires fresh step_area_pressure surfaces")
            accepted = dict.fromkeys(self._world_keys, False)
            self._preparing_batch = True
            try:
                ctrl, applied, body_force, stored = self._pack_controls()
                if self._area_pressure_pairs:
                    self._device.area_packet = self._area_packet
                self._device.advance(ctrl, applied, body_force)
                from .mujoco_tyx._entry_storage import prepare_entry

                prepare_entry(self._device)
                with self._device.runtime_scope():
                    self._device.runtime.sync()
                    self._device.counts["synchronizations"] += 1
                result = self._device.download()
                with self._device.runtime_scope():
                    result["entry_meta"] = self._device._get(
                        self._device.entry["entry_meta"][1 - self._device.active]
                    )
                if self._area_pressure_pairs:
                    from .mujoco_area_pressure_device import AREA_FIELDS

                    with self._device.runtime_scope():
                        for name in AREA_FIELDS:
                            result[name] = self._device._get(
                                self._device.entry[name][1 - self._device.active]
                            )
                if np.any(result["status"]):
                    reasons = {
                        1: "contact_or_constraint_capacity",
                        2: "nonfinite_candidate",
                        3: "joint_limit_domain",
                        4: "nonpositive_mass_matrix",
                        5: "normal_solver_not_converged",
                        6: "nonpositive_normal_matrix",
                        7: "constraint_solver_budget",
                        8: "area_friction_solver_or_publication",
                    }
                    codes = tuple(
                        reasons.get(int(code), "invalid_status")
                        for code in result["status"]
                        if code
                    )
                    raise RuntimeError(f"MuJoCo TYX candidate rejected: {codes}")
                data, receipts, ticks = {}, {}, {}
                for index, key in enumerate(self._world_keys):
                    original = self._data[key]
                    if self._plan.kind in (2, 3) and (
                        not np.array_equal(result["input_q"][index, : self._plan.nq], original.qpos)
                        or not np.array_equal(
                            result["input_v"][index, : self._plan.nv], original.qvel
                        )
                    ):
                        raise RuntimeError("TYX input state does not match its accepted world")
                    if hasattr(self._plan, "activation") and not np.array_equal(
                        result["input_act"][index], original.act
                    ):
                        raise RuntimeError("TYX input activation does not match its accepted world")
                    old_clock = result["input_time"][index]
                    new_clock = result["time"][index]
                    updates = dict(
                        qpos=result["q"][index, : self._plan.nq],
                        qvel=result["v"][index, : self._plan.nv],
                        qacc=result["qacc"][index, : self._plan.nv],
                        ctrl=stored[key].ctrl,
                        qfrc_applied=stored[key].qfrc_applied,
                        xfrc_applied=stored[key].xfrc_applied,
                        qfrc_actuator=result["actuator"][index, : self._plan.nv],
                        qfrc_constraint=result["constraint"][index, : self._plan.nv],
                        time=float(new_clock[0]) - float(new_clock[1]),
                    )
                    if hasattr(self._plan, "activation"):
                        updates["act"] = result["act"][index]
                    candidate = original.changed(**updates)
                    self._finite_state(candidate)
                    increment = np.float32(np.float32(self._timestep) - old_clock[1])
                    total = np.float32(old_clock[0] + increment)
                    compensation = np.float32(np.float32(total - old_clock[0]) - increment)
                    if (
                        float(old_clock[0]) - float(old_clock[1]) != original.time
                        or not np.array_equal(new_clock, [total, compensation])
                        or candidate.time <= original.time
                    ):
                        raise RuntimeError(
                            "TYX clock did not perform one compensated device timestep"
                        )
                    ticks[key] = self._ticks[key] + 1
                    if ticks[key] > np.iinfo(np.int32).max:
                        raise RuntimeError("TYX logical tick exceeds i32 capacity")
                    row = result["rows"][index]
                    dynamics = None
                    if self._plan.kind in (2, 3):
                        n = self._plan.nv
                        dynamics = MuJoCoTyxDynamics(
                            body_position=_frozen(result["body_position"][index]),
                            body_rotation=_frozen(result["body_rotation"][index]),
                            body_com=_frozen(result["body_com"][index]),
                            jacobian_com=_frozen(result["jacobian_com"][index, :, :, :n]),
                            mass_matrix=_frozen(result["mass_matrix"][index, :n, :n]),
                            bias=_frozen(result["bias"][index, :n]),
                            passive=_frozen(result["passive"][index, :n]),
                            applied=_frozen(result["applied"][index, :n]),
                            smooth_acceleration=_frozen(result["smooth_acceleration"][index, :n]),
                            input_q=_frozen(result["input_q"][index, : self._plan.nq]),
                            input_v=_frozen(result["input_v"][index, :n]),
                            time=float(old_clock[0]) - float(old_clock[1]),
                        )
                    constraints = ()
                    if self._area_friction_laws:
                        from .mujoco_area_friction_live import publish

                        contacts, constraints, raw_fields = publish(self, result, index)
                    elif getattr(self._plan, "constraint_contact", False):
                        from .mujoco_tyx._constraint_publication import publish

                        contacts, constraints, raw_fields = publish(self, result, index)
                    elif getattr(self._plan, "normal_contact", False):
                        contacts, raw_fields = self._normal_publication(result, index)
                    else:
                        contacts = self._contacts_from_row(row)
                        raw_fields = MappingProxyType({"device_row": _frozen(row)})
                    receipt_type = MuJoCoTyxStepReceipt
                    receipt_extra = {}
                    if hasattr(self._plan, "activation"):
                        from .mujoco_actuation import (
                            MuJoCoTyxActivationStepReceipt,
                            publish_actuation,
                        )

                        receipt_type = MuJoCoTyxActivationStepReceipt
                        receipt_extra = {"actuation": publish_actuation(self._plan, result, index)}
                    if getattr(self._plan, "joint_transmission", None) is not None:
                        from .mujoco_joint_transmission import (
                            MuJoCoTyxJointActivationStepReceipt,
                            MuJoCoTyxJointMotorStepReceipt,
                            publish_joint_transmission,
                        )

                        receipt_type = (
                            MuJoCoTyxJointActivationStepReceipt
                            if hasattr(self._plan, "activation")
                            else MuJoCoTyxJointMotorStepReceipt
                        )
                        receipt_extra["transmission"] = publish_joint_transmission(
                            self._plan, result, index, self._controls[key].ctrl
                        )
                    receipts[key] = receipt_type(
                        world=key,
                        tick=ticks[key],
                        input_time=float(original.time),
                        output_time=float(candidate.time),
                        h=self._timestep,
                        control=self._controls[key],
                        stored_control=stored[key],
                        contacts=contacts,
                        input_q=_frozen(original.qpos),
                        input_v=_frozen(original.qvel),
                        qacc=_frozen(candidate.qacc),
                        integration_acceleration=_frozen(
                            result["integration_acc"][index, : self._plan.nv]
                        ),
                        qfrc_actuator=_frozen(candidate.qfrc_actuator),
                        qfrc_constraint=_frozen(candidate.qfrc_constraint),
                        raw_contact_fields=raw_fields,
                        source_solver=self._plan.source_solver,
                        numerical_solver=(
                            self._capabilities["numerical_solver"]
                            if self._area_friction_laws or self._mode == "musa"
                            else "pyramidal-box-active-set-v1"
                            if getattr(self._plan, "constraint_contact", False)
                            else "normal-dual-active-set-v1"
                            if getattr(self._plan, "normal_contact", False)
                            else "tree-dense-cholesky"
                            if self._plan.kind in (2, 3)
                            else "single-normal-direct-qp"
                        ),
                        contact_time=float(original.time),
                        dynamics=dynamics,
                        constraints=constraints,
                        contact_law=self._capabilities["contact_law"],
                        backend=f"mujoco-tyx-{self._mode}",
                        **receipt_extra,
                    )
                    if self._area_pressure_pairs:
                        from .mujoco_area_pressure_device import validate_row

                        validate_row(result, index, receipts[key])
                        if self._area_friction_laws:
                            from .mujoco_area_friction_state import validate_friction_receipt

                            validate_friction_receipt(self, key, receipts[key], candidate)
                    expected_entry = (
                        1,
                        self._ticks[key],
                        ticks[key],
                        len(contacts),
                        sum(int(c.active) for c in contacts),
                        0,
                    )
                    if not np.array_equal(result["entry_meta"][index], expected_entry):
                        raise RuntimeError("TYX accepted entry metadata mismatch")
                    data[key] = candidate
            except Exception as error:
                self._extra_counts["failed_batches"] += 1
                raise PhysicsStepFailure(
                    "MuJoCo TYX candidate batch failed before publication", accepted_worlds=accepted
                ) from error
            finally:
                self._preparing_batch = False
            draft = _PreparedStep(
                self,
                self._device,
                self._checkpoint_revision,
                self._device.active,
                self._data,
                self._controls,
                data,
                receipts,
                ticks,
                MappingProxyType(dict.fromkeys(self._world_keys, True)),
                self._extra_counts["completed_steps"] + 1,
                self._extra_counts["accepted_world_steps"] + len(self._world_keys),
                self._checkpoint_revision + 1,
                1 - self._device.active,
            )
            self._prepared_step = draft
            return draft

    def _validate_step_draft_locked(self, draft):
        MuJoCoRuntime._check_open(self)
        if (
            type(draft) is not _PreparedStep
            or draft is not self._prepared_step
            or draft.owner is not self
            or draft.device is not self._device
            or draft.revision != self._checkpoint_revision
            or draft.active != self._device.active
            or draft.entry_data is not self._data
            or draft.entry_controls is not self._controls
        ):
            raise ValueError("stale or foreign step draft")

    def _commit_step_draft_locked(self, draft):
        # Device.publish() has exactly this bank assignment. Do not invoke a
        # callback between swaps of an already validated heterogeneous batch.
        self._device.active = draft.destination_bank
        self._data, self._receipts, self._ticks = draft.data, draft.receipts, draft.ticks
        self._checkpoint_revision = draft.destination_revision
        self._extra_counts["completed_steps"] = draft.completed_steps
        self._extra_counts["accepted_world_steps"] = draft.accepted_world_steps
        self._prepared_step = None

    def _discard_step_draft_locked(self, draft=None):
        if draft is None or draft is self._prepared_step:
            self._prepared_step = None

    def integration_snapshot(self, keys=None):
        """Copy accepted integration state without CPU endpoint reconstruction.

        Physics validation and the complete last-step receipt are unchanged.
        This explicitly narrower observation does not replace snapshot().
        """
        from .mujoco_tyx_state import MuJoCoTyxIntegrationState

        with self._lock:
            self._check_open()
            result = {}
            for key in self._select(keys):
                data = self._data[key]
                result[key] = MuJoCoTyxIntegrationState(
                    world=key,
                    q=_frozen(data.qpos),
                    v=_frozen(data.qvel),
                    act=_frozen(data.act),
                    ctrl=_frozen(data.ctrl),
                    time=float(data.time),
                    tick=self._ticks[key],
                    qfrc_applied=_frozen(data.qfrc_applied),
                    xfrc_applied=_frozen(data.xfrc_applied),
                    held_control=self._controls[key],
                    last_step=self._receipts[key],
                )
            return MappingProxyType(result)

    def snapshot(self, keys=None):
        with self._lock:
            self._check_open()
            states = {}
            for key in self._select(keys):
                committed = self._data[key]
                endpoint = self._observe_native(committed)
                states[key] = MuJoCoState(
                    world=key,
                    q=_frozen(committed.qpos),
                    v=_frozen(committed.qvel),
                    act=_frozen(committed.act),
                    ctrl=_frozen(committed.ctrl),
                    time=float(committed.time),
                    tick=self._ticks[key],
                    body_positions=_frozen(endpoint.xpos),
                    body_quaternions=_frozen(endpoint.xquat),
                    geometry_positions=_frozen(endpoint.geom_xpos),
                    geometry_matrices=_frozen(endpoint.geom_xmat.reshape(-1, 3, 3)),
                    qacc=_frozen(endpoint.qacc),
                    qfrc_actuator=_frozen(endpoint.qfrc_actuator),
                    qfrc_passive=_frozen(endpoint.qfrc_passive),
                    qfrc_constraint=_frozen(endpoint.qfrc_constraint),
                    qfrc_applied=_frozen(committed.qfrc_applied),
                    xfrc_applied=_frozen(committed.xfrc_applied),
                    sensordata=_frozen(endpoint.sensordata),
                    energy=_frozen(endpoint.energy),
                    contacts=_contacts(self._mj, self._model, endpoint),
                    contact_time=float(committed.time),
                    held_control=self._controls[key],
                    last_step=self._receipts[key],
                    contact_law=self._capabilities["contact_law"],
                    contact_phase="endpoint-cpu-forward-recomputed",
                )
            return MappingProxyType(states)

    def reset(self, keys=None):
        """Reset to original declarations, retaining their original times."""
        with self._lock:
            self._check_open()
            selected = self._select(keys)
            if not selected:
                return
            self._reset_states_locked({key: self._initial[key] for key in selected})

    def reset_states(self, states, *, expected_revision):
        """Atomically reset a selected native-state mapping at an episode boundary.

        Explicit states require time zero and pass the original initial-state
        admission. The default reset templates remain unchanged. Unselected
        worlds retain their full mirrors, controls and exact device clock bits.
        Empty mappings are a validated no-op and do not advance the revision.
        """
        with self._lock:
            self._check_open()
            if type(expected_revision) is not int or expected_revision != self._checkpoint_revision:
                raise ValueError("stale reset destination revision")
            if not isinstance(states, Mapping):
                raise TypeError("reset states must be a world-key mapping")
            self._select(states)
            prepared = {}
            for key in self._world_keys:
                if key not in states:
                    continue
                packet = states[key]
                if type(packet) is not MuJoCoInitialState:
                    raise TypeError("reset states require MuJoCoInitialState values")
                if packet.time != 0.0:
                    raise ValueError("explicit episode reset time must be zero")
                prepared[key] = MuJoCoInitialState(packet.q, packet.v, act=packet.act)
            return self._reset_states_locked(prepared)

    def _reset_states_locked(self, states):
        from .mujoco_checkpoint import _accepted_bank, _verify_bank
        from .mujoco_device_controls import invalidate, retirement_list
        from .mujoco_entry_checkpoint import _accepted_entry, _validate_entry, _write_entry

        selected = tuple(key for key in self._world_keys if key in states)
        source = self._compilation.source
        binding = {
            "schema": "mujoco-tyx-reset-binding-v1",
            "model_sha256": hashlib.sha256(source.model_binary).hexdigest(),
            "archive_sha256": hashlib.sha256(source.archive_bytes).hexdigest(),
            "mujoco_version": source.mujoco_version,
            "mode": self._mode,
            "world_keys": self._world_keys,
            "dimensions": tuple(
                int(getattr(self._model, key)) for key in ("nq", "nv", "nu", "na", "nbody")
            ),
            "layout": (max(8, self._plan.nq), max(8, self._plan.nv), 2),
            "domain": self.capabilities["domain"],
        }
        digest = hashlib.sha256(
            json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        revision = self._checkpoint_revision
        if not selected:
            return MuJoCoTyxResetReceipt(
                "mujoco-tyx-reset-v1",
                revision,
                revision,
                (),
                digest,
                MappingProxyType({}),
                MappingProxyType({}),
                _frozen(np.empty((0, 2)), dtype=np.float32),
            )
        proposed, controls = dict(self._data), dict(self._controls)
        ticks, receipts = dict(self._ticks), dict(self._receipts)
        for key in selected:
            proposed[key] = self._initial_data(states[key])
            controls[key] = self._validated_control(MuJoCoControl(np.zeros(self._model.nu)))
            ticks[key], receipts[key] = 0, None
        # Admission/packing occurs before allocating a replacement. Only the
        # selected rows use this packing; all other rows come from the actual
        # accepted bank, including both compensated-clock components.
        proposed_packed = self._pack_states(proposed)
        packed = tuple(value.copy() for value in _accepted_bank(self._device))
        if len(packed) != len(proposed_packed):
            raise RuntimeError("reset accepted bank schema differs")
        for old, new in zip(packed, proposed_packed, strict=True):
            if old.dtype != new.dtype or old.shape != new.shape or not np.isfinite(old).all():
                raise RuntimeError("reset accepted bank layout differs")
        rows = []
        stored = {}
        for index, key in enumerate(self._world_keys):
            if key not in states:
                # Refuse a corrupt mirror before carrying it into another bank.
                original = self._data[key]
                if (
                    packed[0][index, : self._plan.nq].astype(np.float64).tobytes()
                    != original.qpos.tobytes()
                    or packed[1][index, : self._plan.nv].astype(np.float64).tobytes()
                    != original.qvel.tobytes()
                    or float(packed[2][index, 0]) - float(packed[2][index, 1]) != original.time
                    or (
                        len(packed) == 4
                        and packed[3][index].astype(np.float64).tobytes() != original.act.tobytes()
                    )
                ):
                    raise RuntimeError("reset unselected accepted mirror differs")
                continue
            rows.append(index)
            for old, new in zip(packed, proposed_packed, strict=True):
                old[index] = new[index]
            updates = dict(
                qpos=packed[0][index, : self._plan.nq],
                qvel=packed[1][index, : self._plan.nv],
                time=float(packed[2][index, 0]) - float(packed[2][index, 1]),
            )
            if len(packed) == 4:
                updates["act"] = packed[3][index]
            proposed[key] = proposed[key].changed(**updates)
            data = proposed[key]
            stored[key] = MuJoCoInitialState(data.qpos, data.qvel, act=data.act, time=data.time)
        result = MuJoCoTyxResetReceipt(
            "mujoco-tyx-reset-v1",
            revision,
            revision + 1,
            selected,
            digest,
            MappingProxyType(dict(states)),
            MappingProxyType(stored),
            _frozen(packed[2][rows], dtype=np.float32),
        )
        retired_controls = retirement_list(self)
        old = self._device
        entry = _accepted_entry(old)
        for value in entry.values():
            value[rows] = 0
        _validate_entry(self, entry, ticks, receipts, packed[2])
        old_retirement = [*self._checkpoint_retired, old]
        replacement = None
        try:
            replacement = self._build_device(proposed, packed=packed)
            _verify_bank(replacement, packed)
            _write_entry(replacement, entry)
            for key in selected:
                self._observe_native(proposed[key])
        except BaseException as error:
            if replacement is not None:
                try:
                    replacement.close()
                except BaseException as cleanup:
                    self._checkpoint_retired.append(replacement)
                    raise BaseExceptionGroup(
                        "reset candidate failed; cleanup owner retained", [error, cleanup]
                    ) from None
            raise
        for key, value in old.counts.items():
            replacement.counts[key] += value
        self._device = replacement
        self._data, self._controls, self._ticks, self._receipts = (
            proposed,
            controls,
            ticks,
            receipts,
        )
        self._checkpoint_revision = revision + 1
        self._checkpoint_retired, self._control_retired = old_retirement, retired_controls
        invalidate(self)
        errors = list(self._control_cleanup_errors)
        try:
            old.close()
        except BaseException as error:
            errors.append(repr(error))
        else:
            self._checkpoint_retired.remove(old)
        # The frozen record has not escaped yet; finalize resource diagnostics
        # without turning an accepted physics reset into a false refusal.
        object.__setattr__(result, "cleanup_errors", tuple(errors))
        return result

    @property
    def checkpoint_revision(self):
        """Monotonic destination CAS revision, independent of rewindable ticks."""
        with self._lock:
            self._check_open()
            return self._checkpoint_revision

    @property
    def area_pressure_device_layout(self):
        from .mujoco_area_pressure_device import copy_layout

        with self._lock:
            self._check_open()
            return copy_layout(self)

    def copy_area_pressure(self, outputs, *, expected_revision):
        """Copy accepted device area history and endpoints to caller-owned arrays."""
        from .mujoco_area_pressure_device import copy_accepted

        return copy_accepted(self, outputs, expected_revision)

    def area_pressure_snapshot(self, keys=None):
        """Owned HOST history, with input force phase separate from endpoint state."""
        from .mujoco_area_pressure_state import observe

        return observe(self, keys)

    def checkpoint(self):
        """Capture the whole accepted TYX batch, including exact clock bits.

        The immutable capsule is separate from user-authored initial states.
        Policy/Env/application histories remain explicitly caller-owned.
        """
        from .mujoco_checkpoint import capture

        with self._lock:
            self._check_open()
            return capture(self)

    def restore(self, checkpoint, *, expected_revision):
        """Atomically restore a compatible capsule without altering its q/v.

        expected_revision protects the destination against intervening controls,
        steps, resets or restores. Historical capsules remain intentionally
        restorable when the caller supplies the current destination revision.
        """
        from .mujoco_checkpoint import restore
        from .mujoco_device_controls import invalidate, retirement_list

        with self._lock:
            self._check_open()
            retired_controls = retirement_list(self)
            result = restore(self, checkpoint, expected_revision)
            self._control_retired = retired_controls
            invalidate(self)
            return result

    def prepare_task_workspace(self, spec, *, cache_root=None):
        """Prepare owned device task outputs; physics retains full HOST audit."""
        from .mujoco_task_workspace import MuJoCoTyxTaskWorkspace

        with self._lock:
            self._check_open()
            if self._mode == "musa":
                raise ValueError("MUSA does not support resident task workspaces")
            return MuJoCoTyxTaskWorkspace(self, spec, cache_root=cache_root)

    def _refresh_task_workspace(self, workspace, expected_revision):
        from .mujoco_task_workspace import MuJoCoTyxTaskWorkspace

        with self._lock:
            self._check_open()
            if type(workspace) is not MuJoCoTyxTaskWorkspace or workspace._engine is not self:
                raise ValueError("task workspace belongs to another runtime")
            if type(expected_revision) is not int or expected_revision != self._checkpoint_revision:
                raise ValueError("task workspace source revision is stale")
            return workspace._refresh_locked(
                self._device,
                self._checkpoint_revision,
                tuple(self._ticks[key] for key in self._world_keys),
                tuple(float(self._data[key].time) for key in self._world_keys),
            )

    def clone(self, world_keys=None):
        with self._lock:
            self._check_open()
            if self._area_pressure_pairs:
                from .mujoco_area_pressure_state import validate_area_receipt

                for key in self._world_keys:
                    validate_area_receipt(self, key, self._receipts[key], self._data[key])
            selected = self._select(world_keys)
            from .mujoco_tyx._allocation_manifest import allocation_manifest

            size = allocation_manifest(
                self._plan,
                len(selected),
                self._model.nbody,
                (self._contact_capacity, self._constraint_capacity),
            ).payload_bytes
            admission = (
                self._resource_budget.admit(2 * size)
                if self._resource_budget is not None and self._mode in ("cuda", "musa")
                else nullcontext()
            )
            with admission:
                return self._clone_selected(selected)

    def _clone_selected(self, selected):
        from .mujoco_checkpoint import _accepted_bank, _verify_bank
        from .mujoco_entry_checkpoint import _accepted_entry, _validate_entry, _write_entry

        # Caller holds the source lock and the whole two-owner budget admission.
        with self._lock:
            cloned = type(self)(
                self._compilation,
                world_keys=selected,
                initial_states={key: self._initial[key] for key in selected},
                mode=self._mode,
                implementation="tyx",
                contact_capacity=self._contact_capacity,
                constraint_capacity=self._constraint_capacity,
                cache_root=self._cache_root,
                resource_budget=self._resource_budget,
                area_pressure_pairs=self._area_pressure_pairs,
                area_friction_laws=self._area_friction_laws,
            )
            try:
                proposed = {}
                for key in selected:
                    copied = copy.copy(self._data[key])
                    proposed[key] = cloned._mirror_layout.from_parts(
                        copied.cpu_state, {name: getattr(copied, name) for name in MIRROR_FIELDS}
                    )
                indices = [self._world_keys.index(key) for key in selected]
                packed = tuple(value[indices].copy() for value in _accepted_bank(self._device))
                entry = {
                    name: value[indices].copy()
                    for name, value in _accepted_entry(self._device).items()
                }
                _validate_entry(cloned, entry, self._ticks, self._receipts, packed[2])
                replacement = cloned._build_device(proposed, packed=packed)
                # Retain candidate ownership before any copy/verification can fail.
                cloned._checkpoint_retired.append(replacement)
                _verify_bank(replacement, packed)
                _write_entry(replacement, entry)
                previous = cloned._device
                # Own both banks before retirement: a failed close must not
                # abandon the replacement or the original clone allocation.
                cloned._checkpoint_retired.append(previous)
                cloned._device = replacement
                cloned._checkpoint_retired.remove(replacement)
                previous.close()
                cloned._checkpoint_retired.remove(previous)
                cloned._data = proposed
                cloned._controls = {key: self._controls[key] for key in selected}
                cloned._ticks = {key: self._ticks[key] for key in selected}
                cloned._receipts = {key: self._receipts[key] for key in selected}
                return cloned
            except BaseException as error:
                try:
                    cloned.close()
                except BaseException as cleanup:
                    # The clone never escapes to a caller. Retain a retryable
                    # cleanup owner without changing this source's physical bank.
                    self._checkpoint_retired.append(cloned)
                    raise BaseExceptionGroup(
                        "TYX clone failed and cleanup needs retry on source.close()",
                        [error, cleanup],
                    ) from None
                raise

    def close(self):
        with self._lock:
            if (
                getattr(self, "_preparing_batch", False)
                or getattr(self, "_prepared_step", None) is not None
                or getattr(self, "_prepared_controls", None) is not None
            ):
                raise RuntimeError("cannot close during an unpublished batch transaction")
            retired = list(getattr(self, "_checkpoint_retired", ()))
            controls = list(getattr(self, "_control_retired", ()))
            packet = getattr(self, "_control_packet", None)
            if packet is not None:
                controls.append(packet)
            if self._closed and not retired and not controls:
                return
            device = getattr(self, "_device", None)
            physical = retired + ([] if device is None else [device])
            failed_controls, failed_physical, errors = [], [], []
            for owners, failed in ((controls, failed_controls), (physical, failed_physical)):
                for owner in owners:
                    try:
                        owner.close()
                    except BaseException as error:
                        failed.append(owner)
                        errors.append(error)
            self._control_packet = None
            self._control_retired, self._checkpoint_retired = failed_controls, failed_physical
            self._control_cleanup_errors = tuple(repr(error) for error in errors)
            self._device = None
            self._observer_data = None
            super().close()
            if errors:
                raise BaseExceptionGroup(
                    "TYX cleanup failed; close() can retry retained owners", errors
                )

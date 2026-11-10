"""Native-coordinate MuJoCo normal-law subset executed by independent TYX kernels.

The source-bound MjModel is decoded once. No per-step mj_step, mj_forward,
Warp import or inherited Phymeo contact law implements dynamics. Explicit
snapshot() observation uses a detached CPU forward pass and is labelled as
such. The fixed hinge/slide tree recomputes full dynamics each step and admits
no possible contact pairs; the original constant-mass single-normal domain
remains a separate route.
"""

import copy
from dataclasses import dataclass, replace
from numbers import Integral
from types import MappingProxyType

import numpy as np

from ..training.resident_stepper import PhysicsStepFailure
from .mujoco_runtime import MuJoCoControl, MuJoCoRuntime, _frozen
from .mujoco_tyx import MuJoCoTyxUnsupported as MuJoCoTyxUnsupported
from .mujoco_tyx._model import compile_model, fp32

LAW = "mujoco-soft-normal-constraint-v1"


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
    projection. passive is separate; M @ smooth_acceleration = applied + passive - bias.
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


class MuJoCoTyxRuntime(MuJoCoRuntime):
    """Own TYX interpret/CUDA execution and atomic publication for sparse worlds.

    Floating domain: one native free root and 0..29 hinge/slide joints, full
    root/joint coupling, origin world linear and local angular velocities.
    Rotational domain: fixed-root 1..8 hinge/slide DOFs, complete live M/J/bias,
    welded bodies and off-center world wrenches, without possible contact pairs.
    Original domain: 1..8 all-prismatic native DOFs with full coupled static
    mass, stateless direct/affine motors, scalar stiffness/armature/damping; or
    one centered isotropic free body, with native local angular coordinates and
    real world Cartesian body torque. At most one possible static plane/sphere
    pair with condim=1. Its soft-normal QP is solved directly, not replaced with
    a hard-contact projection or force-gap spring. Unsupported semantics fail
    construction. Finite joint limits are strict-interior guards: entering a
    limit margin rejects publication, not silently omits a stop response.
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
    ):
        if implementation not in (None, "tyx"):
            raise ValueError("MuJoCoTyxRuntime requires implementation='tyx'")
        if mode not in ("interpret", "cuda"):
            raise ValueError("MuJoCo TYX mode must be interpret or cuda")
        self._contact_capacity = _capacity(contact_capacity, "contact_capacity")
        self._constraint_capacity = _capacity(constraint_capacity, "constraint_capacity")
        self._mode, self._cache_root = mode, cache_root
        super().__init__(
            compilation, world_keys=world_keys, initial_states=initial_states, mode="cpu"
        )
        self._plan = compile_model(self._mj, self._model)
        self._extra_counts = dict(completed_steps=0, accepted_world_steps=0, failed_batches=0)
        self._device = self._build_device(self._data)
        self._quantize_mirrors()
        info = dict(self._capabilities)
        info.update(
            backend=f"mujoco-tyx-{mode}",
            mode=mode,
            implementation="tyx",
            gpu=mode == "cuda",
            native_model=False,
            contact_law=LAW,
            domain=(
                "tree-primitive-coupled-normal-v1"
                if getattr(self._plan, "normal_contact", False)
                else "free-root-hinge-slide-no-contact-v1"
                if self._plan.kind == 3
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
            g1_supported=False,
        )
        if getattr(self._plan, "normal_contact", False):
            info.update(
                normal_candidate_slots=len(self._plan.slot_ids),
                solver_tolerance=self._plan.solver_tolerance,
                solver_iteration_semantics="source-iterations-as-active-set-face-solve-budget",
                warmstart="cold-feasible-zero-force",
            )
        self._capabilities = MappingProxyType(info)

    @property
    def counters(self):
        return MappingProxyType({**self._device.counts, **self._extra_counts})

    def _pack_states(self, data):
        q = np.zeros((len(self._world_keys), max(8, self._plan.nq)), np.float32)
        v = np.zeros((len(self._world_keys), max(8, self._plan.nv)), np.float32)
        clock = np.zeros(len(self._world_keys), np.float32)
        for index, key in enumerate(self._world_keys):
            original = data[key]
            q[index, : self._plan.nq] = fp32(original.qpos, "initial qpos")
            v[index, : self._plan.nv] = fp32(original.qvel, "initial qvel")
            clock[index] = fp32(original.time, "initial time").item()
            if self._plan.kind == 3:
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
        return q, v, clock

    def _build_device(self, data):
        from .mujoco_tyx._device import TyxDevice

        device_type = TyxDevice
        if getattr(self._plan, "normal_contact", False):
            from .mujoco_tyx._normal_device import TyxNormalDevice

            device_type = TyxNormalDevice
        elif self._plan.kind == 2:
            from .mujoco_tyx._tree_device import TyxTreeDevice

            device_type = TyxTreeDevice
        elif self._plan.kind == 3:
            from .mujoco_tyx._floating_device import TyxFloatingDevice

            device_type = TyxFloatingDevice
        return device_type(
            self._plan,
            self._mode,
            self._cache_root,
            *self._pack_states(data),
            self._model.nbody,
            self._timestep,
            (self._contact_capacity, self._constraint_capacity),
        )

    def _quantize_mirrors(self):
        q, v, time = self._pack_states(self._data)
        for index, key in enumerate(self._world_keys):
            self._data[key].qpos[:] = q[index, : self._plan.nq]
            self._data[key].qvel[:] = v[index, : self._plan.nv]
            self._data[key].time = time[index]

    def set_controls(self, mapping):
        with self._lock:
            self._check_open()
            self._exact_mapping(mapping, "controls")
            prepared = {key: self._validated_control(mapping[key]) for key in self._world_keys}
            for packet in prepared.values():
                for value in (packet.ctrl, packet.qfrc_applied, packet.xfrc_applied):
                    fp32(value, "held native control")
            self._controls = prepared

    def _pack_controls(self):
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

    def step(self, h):
        with self._lock:
            self._check_open()
            if not np.isfinite(h) or float(h) != self._timestep:
                raise ValueError("step h must equal the original compiled MuJoCo timestep")
            accepted = dict.fromkeys(self._world_keys, False)
            try:
                ctrl, applied, body_force, stored = self._pack_controls()
                self._device.advance(ctrl, applied, body_force)
                result = self._device.download()
                if np.any(result["status"]):
                    reasons = {
                        1: "contact_or_constraint_capacity",
                        2: "nonfinite_candidate",
                        3: "joint_limit_domain",
                        4: "nonpositive_mass_matrix",
                        5: "normal_solver_not_converged",
                        6: "nonpositive_normal_matrix",
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
                    candidate = copy.copy(original)
                    candidate.qpos[:] = result["q"][index, : self._plan.nq]
                    candidate.qvel[:] = result["v"][index, : self._plan.nv]
                    candidate.qacc[:] = result["qacc"][index, : self._plan.nv]
                    candidate.ctrl[:] = stored[key].ctrl
                    candidate.qfrc_applied[:] = stored[key].qfrc_applied
                    candidate.xfrc_applied[:] = stored[key].xfrc_applied
                    candidate.qfrc_actuator[:] = result["actuator"][index, : self._plan.nv]
                    candidate.qfrc_constraint[:] = result["constraint"][index, : self._plan.nv]
                    candidate.time = float(result["time"][index])
                    self._finite_state(candidate)
                    expected = float(
                        np.float32(np.float32(original.time) + np.float32(self._timestep))
                    )
                    if candidate.time != expected or candidate.time <= original.time:
                        raise RuntimeError("TYX did not advance exactly one FP32 native timestep")
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
                            time=float(result["input_time"][index]),
                        )
                    if getattr(self._plan, "normal_contact", False):
                        contacts, raw_fields = self._normal_publication(result, index)
                    else:
                        contacts = self._contacts_from_row(row)
                        raw_fields = MappingProxyType({"device_row": _frozen(row)})
                    receipts[key] = MuJoCoTyxStepReceipt(
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
                            "normal-dual-active-set-v1"
                            if getattr(self._plan, "normal_contact", False)
                            else "tree-dense-cholesky"
                            if self._plan.kind in (2, 3)
                            else "single-normal-direct-qp"
                        ),
                        contact_time=float(original.time),
                        dynamics=dynamics,
                        backend=f"mujoco-tyx-{self._mode}",
                    )
                    data[key] = candidate
            except Exception as error:
                self._extra_counts["failed_batches"] += 1
                raise PhysicsStepFailure(
                    "MuJoCo TYX candidate batch failed before publication", accepted_worlds=accepted
                ) from error
            self._device.publish()
            self._data, self._receipts, self._ticks = data, receipts, ticks
            self._extra_counts["completed_steps"] += 1
            self._extra_counts["accepted_world_steps"] += len(self._world_keys)
            return MappingProxyType(dict.fromkeys(self._world_keys, True))

    def snapshot(self, keys=None):
        states = super().snapshot(keys)
        return MappingProxyType(
            {
                key: replace(
                    state, contact_law=LAW, contact_phase="endpoint-cpu-forward-recomputed"
                )
                for key, state in states.items()
            }
        )

    def reset(self, keys=None):
        with self._lock:
            self._check_open()
            selected = self._select(keys)
            if not selected:
                return
            proposed, controls = dict(self._data), dict(self._controls)
            ticks, receipts = dict(self._ticks), dict(self._receipts)
            for key in selected:
                proposed[key] = self._initial_data(self._initial[key])
                controls[key] = self._validated_control(MuJoCoControl(np.zeros(self._model.nu)))
                ticks[key], receipts[key] = 0, None
            replacement = self._build_device(proposed)
            old = self._device
            self._device = replacement
            self._data, self._controls, self._ticks, self._receipts = (
                proposed,
                controls,
                ticks,
                receipts,
            )
            self._quantize_mirrors()
            old.close()

    def clone(self, world_keys=None):
        with self._lock:
            self._check_open()
            selected = self._select(world_keys)
            cloned = type(self)(
                self._compilation,
                world_keys=selected,
                initial_states={key: self._initial[key] for key in selected},
                mode=self._mode,
                implementation="tyx",
                contact_capacity=self._contact_capacity,
                constraint_capacity=self._constraint_capacity,
                cache_root=self._cache_root,
            )
            proposed = {key: copy.copy(self._data[key]) for key in selected}
            replacement = cloned._build_device(proposed)
            cloned._device.close()
            cloned._device = replacement
            cloned._data = proposed
            cloned._controls = {key: self._controls[key] for key in selected}
            cloned._ticks = {key: self._ticks[key] for key in selected}
            cloned._receipts = {key: self._receipts[key] for key in selected}
            return cloned

    def close(self):
        with self._lock:
            if self._closed:
                return
            device = getattr(self, "_device", None)
            if device is not None:
                device.close()
            super().close()

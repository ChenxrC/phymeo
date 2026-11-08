"""Owned floating affine declarations over the existing physical free-root tree.

Only initial/static compilation and explicit restore validation run on HOST.
The actual full mechanics, reduced solve and full expansion remain on device.
"""

from dataclasses import dataclass, field, replace

import numpy as np

from ..dynamics._mimic_coordinates import compile_mimic_coordinates
from ..dynamics.floating_mimic import FloatingMimicRobotAsset
from ..dynamics.floating_robot import FloatingRobotState
from ..solvers.floating_tree import FloatingRobotInputs
from ._resident_affine_tree import _intervals, _near
from ._resident_floating_tree import (
    FLOATING_COORDINATE_LIMIT,
    FloatingRobotTree,
    FloatingTreeCheckpoint,
    FloatingTreeDiagnostic,
    FloatingTreeKinematics,
    FloatingTreeWorldSpec,
    _coordinate,
    _freeze_fields,
    _integer,
    compile_floating_tree_world,
    floating_model_token,
)
from ._resident_tree_world import _nested, _owned
from ._world_step_layout import as_fp32, world_key


def _coordinates(asset):
    if not isinstance(asset, FloatingMimicRobotAsset):
        raise TypeError("FloatingAffineRobotTree needs a FloatingMimicRobotAsset")
    tree = asset.tree
    if tree.model.children is None:
        raise ValueError("floating affine needs a nonempty physical child tree")
    return compile_mimic_coordinates(
        tree.joint_names,
        tree.model.children.joint_types,
        tree.lower,
        tree.upper,
        tree.velocity_limit,
        tree.effort_limit,
        tuple((r.driver, r.follower, r.multiplier, r.offset) for r in asset.relations),
    )


def floating_affine_model_token(asset):
    """Full original physical/graph identity, not just the FP32 composed map."""
    coordinates = _coordinates(asset)
    tree = asset.tree
    return (
        "floating_affine_graph_v1",
        floating_model_token(tree.model),
        tuple(tree.joint_names),
        tuple(tree.lower),
        tuple(tree.upper),
        tuple(tree.velocity_limit),
        tuple(tree.effort_limit),
        tuple(
            (r.driver, r.follower, float(r.multiplier), float(r.offset)) for r in asset.relations
        ),
        coordinates.independent_indices,
        _nested(coordinates.matrix),
        _nested(coordinates.offset_vector),
    )


@dataclass(frozen=True, eq=False)
class FloatingAffineTreeInputs:
    """K motor triplets plus instantaneous COM loads on all n+1 real bodies."""

    graph_token: tuple
    drive: np.ndarray
    external: FloatingRobotInputs

    def __post_init__(self):
        if not isinstance(self.graph_token, tuple):
            raise ValueError("floating affine input must bind an original graph token")
        if not isinstance(self.external, FloatingRobotInputs):
            raise TypeError("floating affine external must be full FloatingRobotInputs")
        raw = np.asarray(self.drive)
        n = len(self.external.effort)
        if raw.ndim != 2 or raw.shape[1] != 3 or not 1 <= len(raw) < n <= 16:
            raise ValueError("floating affine drive needs (K,3), 1<=K<n<=16")
        values = []
        for name, shape in (("effort", (n,)), ("force", (n + 1, 3)), ("torque", (n + 1, 3))):
            value = as_fp32(getattr(self.external, name), "floating affine " + name, shape)
            if name == "effort" and np.any(value != 0):
                raise ValueError("external full joint effort must be zero; use independent drive")
            values.append(_owned(value, np.float64))
        external = FloatingRobotInputs(*values)
        _freeze_fields(external, ("effort", "force", "torque"))
        object.__setattr__(self, "external", external)
        object.__setattr__(self, "drive", _owned(as_fp32(raw, "floating affine drive", raw.shape)))


@dataclass(frozen=True, eq=False)
class FloatingAffineRobotTree:
    """A real free root and full scalar tree under a constant passive affine graph."""

    asset: FloatingMimicRobotAsset
    state: FloatingRobotState
    kp: np.ndarray = field(kw_only=True)
    kd: np.ndarray = field(kw_only=True)
    effort_limit: np.ndarray = field(kw_only=True)
    delay_steps: int = field(default=0, kw_only=True)
    control_max_iterations: int = field(default=512, kw_only=True)
    _graph_token: tuple = field(init=False, repr=False)

    def __post_init__(self):
        coordinates = _coordinates(self.asset)
        n, k = coordinates.matrix.shape
        if not 1 <= k < n <= 16:
            raise ValueError("floating affine must satisfy 1<=K<n<=16")
        if not isinstance(self.state, FloatingRobotState):
            raise TypeError("floating affine state must be independent FloatingRobotState")
        if self.state.q.shape != (7 + k,) or self.state.v.shape != (6 + k,):
            raise ValueError("floating affine initial state must contain root plus K coordinates")
        state = FloatingRobotState(self.state.q, self.state.v)
        self.asset.full_state(state)
        _freeze_fields(state, ("q", "v"))
        object.__setattr__(self, "state", state)
        original_cap = np.asarray(self.effort_limit, dtype=float)
        if original_cap.shape != (k,) or np.any(original_cap > coordinates.effort_limit):
            raise ValueError("floating affine cap exceeds the independent source motor rating")
        for name in ("kp", "kd", "effort_limit"):
            value = as_fp32(getattr(self, name), "floating affine " + name, (k,))
            if np.any(value < 0):
                raise ValueError("floating affine gains and caps must be nonnegative")
            object.__setattr__(self, name, _owned(value))
        object.__setattr__(self, "delay_steps", _integer(self.delay_steps, "affine delay", 0, 1))
        object.__setattr__(
            self,
            "control_max_iterations",
            _integer(self.control_max_iterations, "affine iteration budget", 1, 4096),
        )
        object.__setattr__(self, "_graph_token", floating_affine_model_token(self.asset))

    def inputs(self, drive, external=None):
        n, k = len(self.asset.tree.joint_names), len(self.state.q) - 7
        if external is None:
            external = FloatingRobotInputs.zeros(n)
        packet = FloatingAffineTreeInputs(self._graph_token, drive, external)
        if packet.drive.shape != (k, 3) or packet.external.effort.shape != (n,):
            raise ValueError("floating affine input dimensions differ from this graph")
        return packet


@dataclass(frozen=True)
class FloatingAffineTreeCheckpoint:
    world: int
    topology: tuple
    tick: int
    position: tuple
    velocity: tuple
    applied_effort: tuple
    history: tuple
    control: tuple
    generalized_applied: tuple
    full_position: tuple
    full_velocity: tuple
    full_applied_effort: tuple
    full_generalized_applied: tuple
    local_origin_m: tuple


@dataclass(frozen=True, eq=False)
class FloatingAffineTreeKinematics(FloatingTreeKinematics):
    independent_position: np.ndarray
    independent_velocity: np.ndarray
    joint_names: tuple
    full_joint_names: tuple

    def __post_init__(self):
        super().__post_init__()
        object.__setattr__(
            self, "independent_position", _owned(self.independent_position, np.float64)
        )
        object.__setattr__(self, "independent_velocity", _owned(self.independent_velocity))


@dataclass(frozen=True, eq=False)
class FloatingAffineTreeDiagnostic(FloatingTreeDiagnostic):
    full_entry_position: np.ndarray
    full_entry_velocity: np.ndarray
    reduced_mass: np.ndarray
    reduced_bias: np.ndarray
    reduced_external: np.ndarray
    reduced_linear_jacobian: np.ndarray
    reduced_angular_jacobian: np.ndarray
    full_acceleration: np.ndarray
    constraint_effort_full: np.ndarray
    row_multipliers: np.ndarray
    row_powers_w: np.ndarray
    constraint_power_w: float
    root_constraint_residual: np.ndarray

    def __post_init__(self):
        super().__post_init__()
        for name in (
            "full_entry_position",
            "full_entry_velocity",
            "reduced_mass",
            "reduced_bias",
            "reduced_external",
            "reduced_linear_jacobian",
            "reduced_angular_jacobian",
            "full_acceleration",
            "constraint_effort_full",
            "row_multipliers",
            "row_powers_w",
            "root_constraint_residual",
        ):
            dtype = np.float64 if name == "full_entry_position" else np.float32
            object.__setattr__(self, name, _owned(getattr(self, name), dtype))
        object.__setattr__(self, "constraint_power_w", float(self.constraint_power_w))


@dataclass(frozen=True, eq=False)
class FloatingAffineTreeWorldSpec:
    physical: FloatingTreeWorldSpec
    n: int
    k: int
    r: int
    delay: int
    control_max_iterations: int
    topology: tuple
    graph_token: tuple
    joint_names: tuple
    full_joint_names: tuple
    matrix: np.ndarray
    projection: np.ndarray
    offset: np.ndarray
    constraints: np.ndarray
    independent: np.ndarray
    row_indices: np.ndarray
    row_values: np.ndarray
    reverse_rows: np.ndarray
    full_limits: np.ndarray
    limits: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    cap: np.ndarray
    initial_state: np.ndarray
    initial_drive: np.ndarray
    initial_effort: np.ndarray
    initial_loads: np.ndarray
    initial_generalized: np.ndarray

    @property
    def domain(self):
        return "floating_affine_tree"

    @property
    def kind(self):
        return 11 + self.delay

    @property
    def b(self):
        return self.n + 1

    @property
    def d(self):
        return self.n + 6

    @property
    def reduced_d(self):
        return self.k + 6

    @property
    def q_count(self):
        return self.n + 7

    @property
    def reduced_q_count(self):
        return self.k + 7

    @property
    def model_token(self):
        return self.physical.model_token

    @property
    def anchor(self):
        return self.physical.anchor

    @property
    def initial_root(self):
        return self.physical.initial_root

    @property
    def joint_types(self):
        return self.physical.joint_types

    def packet(self, value):
        if not isinstance(value, FloatingAffineTreeInputs) or value.graph_token != self.graph_token:
            raise ValueError("floating affine input belongs to a different graph or input kind")
        drive = as_fp32(value.drive, "floating affine drive", (self.k, 3))
        effort = as_fp32(value.external.effort, "floating affine effort", (self.n,))
        if np.any(effort != 0):
            raise ValueError("external full joint effort must be zero; use independent drive")
        force = as_fp32(value.external.force, "floating affine force", (self.b, 3))
        torque = as_fp32(value.external.torque, "floating affine torque", (self.b, 3))
        return _owned(drive), _owned(effort), _owned(np.column_stack((force, torque)))

    def check_state(self, reduced, full, reduced_applied, full_applied):
        for row, limits, name in (
            (reduced, self.limits, "independent"),
            (full, self.full_limits, "full"),
        ):
            q, v = row[:, 0], row[:, 1]
            if not np.isfinite(row).all() or np.any(abs(q) > FLOATING_COORDINATE_LIMIT):
                raise ValueError(f"floating affine {name} state exceeds finite coordinate domain")
            if (
                np.any(q < limits[:, 0])
                or np.any(q > limits[:, 1])
                or np.any(abs(v) > limits[:, 2])
            ):
                raise ValueError(
                    f"floating affine {name} state exceeds joint interval or velocity limit"
                )
        _near(full[:, 0], self.matrix.astype(float) @ reduced[:, 0] + self.offset, "full position")
        _near(full[:, 1], self.matrix.astype(float) @ reduced[:, 1], "full velocity")
        motor = np.zeros(self.n)
        motor[self.independent] = reduced[:, 2]
        _near(full[:, 2], motor, "full motor effort")
        _near(reduced_applied, self.projection.astype(float).T @ full_applied, "generalized load")
        _near(reduced[:, 3], reduced_applied[6:], "independent applied load")
        _near(full[:, 3], full_applied[6:], "full applied load")
        for row, (driver, follower) in enumerate(self.row_indices):
            gain, offset = self.row_values[row].astype(float)
            _near(
                full[follower, 0], gain * float(full[driver, 0]) + offset, "original row position"
            )
            _near(full[follower, 1], gain * float(full[driver, 1]), "original row velocity")

    def checkpoint(self, world, root, reduced, full, reduced_applied, full_applied, tick, control):
        drive, effort, loads = self.packet(control)
        a, b = np.asarray(reduced)[: self.k], np.asarray(full)[: self.n]
        prefix = np.r_[self.anchor + np.asarray(root[:3], dtype=float), root[3:7]]
        return FloatingAffineTreeCheckpoint(
            world_key(world),
            self.topology,
            int(tick),
            _nested(np.r_[prefix, a[:, 0]]),
            _nested(np.r_[root[7:13], a[:, 1]]),
            _nested(a[:, 2]),
            _nested(a[:, 4:7]) if self.delay else (),
            (
                self.graph_token,
                _nested(drive),
                _nested(effort),
                _nested(loads[:, :3]),
                _nested(loads[:, 3:]),
            ),
            _nested(np.asarray(reduced_applied)[: self.reduced_d]),
            _nested(np.r_[prefix, b[:, 0]]),
            _nested(np.r_[root[7:13], b[:, 1]]),
            _nested(b[:, 2]),
            _nested(np.asarray(full_applied)[: self.d]),
            _nested(root[:3]),
        )

    def restore_row(self, checkpoint, world):
        if not isinstance(checkpoint, FloatingAffineTreeCheckpoint):
            raise ValueError("floating affine restore requires its complete checkpoint kind")
        if world_key(checkpoint.world) != world or checkpoint.topology != self.topology:
            raise ValueError("floating affine checkpoint world/topology/graph differs")
        tick = _integer(checkpoint.tick, "affine checkpoint tick", 0, 2147483647)
        q = np.asarray(checkpoint.position)
        full_q = np.asarray(checkpoint.full_position)
        for value, shape in ((q, (self.reduced_q_count,)), (full_q, (self.q_count,))):
            if (
                value.dtype.kind not in "iuf"
                or value.shape != shape
                or not np.isfinite(value).all()
            ):
                raise ValueError("floating affine checkpoint configuration has invalid dimensions")
        v = as_fp32(checkpoint.velocity, "restored floating affine velocity", (self.reduced_d,))
        as_fp32(checkpoint.full_velocity, "restored floating affine full velocity", (self.d,))
        if not np.array_equal(q[:7], full_q[:7]) or not np.array_equal(
            np.asarray(checkpoint.velocity)[:6], np.asarray(checkpoint.full_velocity)[:6]
        ):
            raise ValueError("floating affine full and independent root states disagree")
        if not isinstance(checkpoint.control, tuple) or len(checkpoint.control) != 5:
            raise ValueError("floating affine held control must retain all five blocks")
        token, drive, effort, force, torque = checkpoint.control
        packet = FloatingAffineTreeInputs(
            token,
            np.asarray(drive, dtype=float),
            FloatingRobotInputs(
                np.asarray(effort, dtype=float),
                np.asarray(force, dtype=float),
                np.asarray(torque, dtype=float),
            ),
        )
        self.packet(packet)
        # The ordinary validator only sees explicit full dimensions and its
        # neutral full motor/history channel. Public kind and K history stay here.
        full_control = (
            _nested(np.zeros((self.n, 3))),
            _nested(packet.external.effort),
            _nested(packet.external.force),
            _nested(packet.external.torque),
        )
        physical_checkpoint = FloatingTreeCheckpoint(
            world,
            self.physical.topology,
            tick,
            checkpoint.full_position,
            checkpoint.full_velocity,
            checkpoint.full_applied_effort,
            (),
            full_control,
            checkpoint.full_generalized_applied,
            checkpoint.local_origin_m,
        )
        root, full, full_applied, _, _ = self.physical.restore_row(physical_checkpoint, world)
        reduced = np.zeros((self.k, 7), np.float32)
        reduced[:, 0] = _coordinate(q[7:], "restored floating affine joints", (self.k,))
        reduced[:, 1] = v[6:]
        reduced[:, 2] = as_fp32(
            checkpoint.applied_effort, "restored floating affine motor", (self.k,)
        )
        applied = as_fp32(
            checkpoint.generalized_applied, "restored floating affine applied", (self.reduced_d,)
        )
        reduced[:, 3] = applied[6:]
        if self.delay:
            reduced[:, 4:7] = as_fp32(
                checkpoint.history, "restored floating affine history", (self.k, 3)
            )
        elif checkpoint.history != ():
            raise ValueError("undelayed floating affine checkpoint cannot contain history")
        self.check_state(reduced, full, applied, full_applied)
        return (
            _owned(root),
            _owned(reduced),
            _owned(full),
            _owned(applied),
            _owned(full_applied),
            packet,
            tick,
        )


def compile_floating_affine_tree_world(world):
    """Compile one real full model and constant coordinate map, without a K-body model."""
    if not isinstance(world, FloatingAffineRobotTree):
        raise TypeError("expected FloatingAffineRobotTree")
    asset, coordinates = world.asset, _coordinates(world.asset)
    n, k = coordinates.matrix.shape
    token = floating_affine_model_token(asset)
    if token != world._graph_token:
        raise ValueError("floating affine source model/graph changed after declaration")
    full_initial = asset.full_state(world.state)
    zeros = np.zeros(n)
    physical = compile_floating_tree_world(
        FloatingRobotTree(
            asset.tree.model,
            full_initial,
            kp=zeros,
            kd=zeros,
            effort_limit=zeros,
            delay_steps=0,
            control_max_iterations=world.control_max_iterations,
        )
    )
    matrix = _owned(as_fp32(coordinates.matrix, "floating affine P", (n, k)))
    offset = _owned(as_fp32(coordinates.offset_vector, "floating affine c", (n,)))
    projection = np.zeros((n + 6, k + 6), np.float32)
    projection[:6, :6], projection[6:, 6:] = np.eye(6, dtype=np.float32), matrix
    constraints = _owned(
        as_fp32(coordinates.constraint_rows, "floating affine original rows", (n - k, n))
    )
    full_limits = np.column_stack(
        (asset.tree.lower, asset.tree.upper, asset.tree.velocity_limit)
    ).astype(float)
    full_limits[:, 0] = np.maximum(full_limits[:, 0], -FLOATING_COORDINATE_LIMIT)
    full_limits[:, 1] = np.minimum(full_limits[:, 1], FLOATING_COORDINATE_LIMIT)
    packed_limits = as_fp32(full_limits, "floating affine full limits", (n, 3))
    for i in range(n):
        if float(packed_limits[i, 0]) < full_limits[i, 0]:
            packed_limits[i, 0] = np.nextafter(packed_limits[i, 0], np.float32(np.inf))
        for j in (1, 2):
            if float(packed_limits[i, j]) > full_limits[i, j]:
                packed_limits[i, j] = np.nextafter(packed_limits[i, j], np.float32(-np.inf))
    full_limits = _owned(packed_limits)
    limits = _intervals(full_limits, matrix, offset, coordinates.independent_indices)
    initial = np.zeros((k, 7), np.float32)
    initial[:, 0] = _coordinate(world.state.q[7:], "initial floating affine q", (k,))
    initial[:, 1] = as_fp32(world.state.v[6:], "initial floating affine v", (k,))
    drive = np.zeros((k, 3), np.float32)
    drive[:, 0] = initial[:, 0]
    if world.delay_steps:
        initial[:, 4:7] = drive
    full = np.zeros((n, 7), np.float32)
    full[:, 0], full[:, 1] = matrix @ initial[:, 0] + offset, matrix @ initial[:, 1]
    topology = (
        "floating_affine_tree_v1",
        token,
        _nested(physical.anchor),
        _nested(world.kp),
        _nested(world.kd),
        _nested(world.effort_limit),
        world.delay_steps,
        world.control_max_iterations,
    )
    physical = replace(
        physical,
        topology=topology,
        initial_state=_owned(full),
        initial_drive=_owned(np.zeros((n, 3))),
    )
    result = FloatingAffineTreeWorldSpec(
        physical,
        n,
        k,
        n - k,
        world.delay_steps,
        world.control_max_iterations,
        topology,
        token,
        coordinates.joint_names,
        tuple(asset.tree.joint_names),
        matrix,
        _owned(projection),
        offset,
        constraints,
        _owned(coordinates.independent_indices, np.int32),
        _owned([(d, f) for d, f, _, _ in coordinates.indices], np.int32),
        _owned(
            as_fp32(
                [(a, b) for _, _, a, b in coordinates.indices],
                "floating affine row values",
                (n - k, 2),
            )
        ),
        _owned(tuple(reversed(coordinates.forward)), np.int32),
        full_limits,
        limits,
        _owned(world.kp),
        _owned(world.kd),
        _owned(world.effort_limit),
        _owned(initial),
        _owned(drive),
        _owned(np.zeros(n)),
        _owned(np.zeros((n + 1, 6))),
        _owned(np.zeros(k + 6)),
    )
    result.check_state(initial, full, result.initial_generalized, physical.initial_generalized)
    return result

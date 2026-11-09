"""Owned affine world declarations for the existing resident TYX batch.

Only static compilation and explicit checkpoint validation happen on HOST.
The physical N-row tree and independent K coordinates remain separate.
"""

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from ..core.robot import RobotInputs, RobotState
from ..dynamics._mimic_coordinates import compile_mimic_coordinates
from ..dynamics.joint_mimic import MimicRobotAsset
from ._resident_tree_world import (
    TREE_COORDINATE_LIMIT,
    TreeDiagnostic,
    TreeKinematics,
    TreeWorldSpec,
    _nested,
    _owned,
    compile_tree_physical,
    tree_model_token,
)
from ._world_step_layout import as_fp32, world_key


def _coordinates(asset):
    if not isinstance(asset, MimicRobotAsset):
        raise TypeError("AffineRobotTree needs a MimicRobotAsset")
    tree = asset.tree
    return compile_mimic_coordinates(
        tree.joint_names,
        tree.model.joint_types,
        tree.lower,
        tree.upper,
        tree.velocity_limit,
        tree.effort_limit,
        tuple((r.driver, r.follower, r.multiplier, r.offset) for r in asset.relations),
    )


def affine_model_token(asset):
    """Original FP64 physical and mechanical graph identity, never rounded P alone."""
    coordinates = _coordinates(asset)
    tree = asset.tree
    return (
        "affine_graph_v1",
        tree_model_token(tree.model),
        tuple(tree.joint_names),
        tuple(tree.lower),
        tuple(tree.upper),
        tuple(tree.velocity_limit),
        tuple(tree.effort_limit),
        tuple(
            (r.driver, r.follower, float(r.multiplier), float(r.offset)) for r in asset.relations
        ),
        tuple(coordinates.independent_indices),
        _nested(coordinates.matrix),
        _nested(coordinates.offset_vector),
    )


def _budget(value, name, lo, hi):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or not lo <= value <= hi
    ):
        raise ValueError(f"affine {name} must be an integer in [{lo},{hi}]")
    return int(value)


def _nonnegative(value, name, count):
    result = as_fp32(value, "affine " + name, (count,))
    if np.any(result < 0):
        raise ValueError(f"affine {name} must be nonnegative")
    return _owned(result)


def _near(actual, expected, name):
    actual, expected = np.asarray(actual), np.asarray(expected)
    scale = np.maximum(1.0, np.maximum(abs(actual), abs(expected)))
    if (
        not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
        or np.any(abs(actual - expected) > 2e-5 * scale)
    ):
        raise ValueError(f"inconsistent affine {name}")


def _intervals(full_limits, matrix, offset, independent):
    """Conservative representable independent intervals after FP32 compilation."""
    k = len(independent)
    limits = np.empty((k, 3), np.float32)
    for j in range(k):
        lower, upper, speed = -np.inf, np.inf, np.inf
        for i in range(len(matrix)):
            gain = float(matrix[i, j])
            if gain != 0:
                lo, hi = sorted(
                    (
                        (float(full_limits[i, 0]) - float(offset[i])) / gain,
                        (float(full_limits[i, 1]) - float(offset[i])) / gain,
                    )
                )
                lower, upper = max(lower, lo), min(upper, hi)
                speed = min(speed, float(full_limits[i, 2]) / abs(gain))
        if not np.isfinite((lower, upper, speed)).all() or lower >= upper:
            raise ValueError("affine FP32 graph has an empty feasible interval")
        low32, high32, speed32 = np.float32(lower), np.float32(upper), np.float32(speed)
        if float(low32) < lower:
            low32 = np.nextafter(low32, np.float32(np.inf))
        if float(high32) > upper:
            high32 = np.nextafter(high32, np.float32(-np.inf))
        if float(speed32) > speed:
            speed32 = np.nextafter(speed32, np.float32(-np.inf))
        if low32 >= high32 or not np.isfinite((low32, high32, speed32)).all():
            raise ValueError("affine FP32 intersection has no representable interval")
        limits[j] = (low32, high32, speed32)
    return _owned(limits)


@dataclass(frozen=True, eq=False)
class AffineTreeInputs:
    """Owned Kx3 drive and Nx7 current full physical loads, in SI units."""

    graph_token: tuple
    drive: np.ndarray
    loads: np.ndarray

    def __post_init__(self):
        for name, columns in (("drive", 3), ("loads", 7)):
            value = np.asarray(getattr(self, name))
            if value.ndim != 2 or value.shape[1] != columns:
                raise ValueError(f"affine {name} must have {columns} columns")
            object.__setattr__(self, name, _owned(as_fp32(value, "affine " + name, value.shape)))


@dataclass(frozen=True, eq=False)
class AffineRobotTree:
    """A fixed-root physical tree under a constant acyclic passive affine graph.

    Motors and state use K independent coordinates. Loads use all N physical
    groups' generalized efforts and world COM forces/torques. Only the drive
    triplet can be delayed, by zero or one accepted physical substep.
    """

    asset: MimicRobotAsset
    state: RobotState
    kp: np.ndarray
    kd: np.ndarray
    effort_limit: np.ndarray
    delay_steps: int = 0
    control_max_iterations: int = 512
    _graph_token: tuple = field(init=False, repr=False)

    def __post_init__(self):
        coordinates = _coordinates(self.asset)
        n, k = coordinates.matrix.shape
        if not 1 <= k < n <= 32:
            raise ValueError("affine physical joint count must satisfy 1<=K<N<=32")
        if not isinstance(self.state, RobotState):
            raise TypeError("affine state must be an independent RobotState")
        if np.asarray(self.state.q).shape != (k,) or np.asarray(self.state.v).shape != (k,):
            raise ValueError("affine state must use K independent coordinates")
        object.__setattr__(
            self,
            "state",
            RobotState(_owned(self.state.q, np.float64), _owned(self.state.v, np.float64)),
        )
        original_cap = np.asarray(self.effort_limit, dtype=float)
        if original_cap.shape != (k,) or np.any(
            original_cap > np.asarray(coordinates.effort_limit)
        ):
            raise ValueError("affine effort cap exceeds the independent source motor rating")
        for name in ("kp", "kd", "effort_limit"):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name, k))
        object.__setattr__(self, "delay_steps", _budget(self.delay_steps, "delay", 0, 1))
        object.__setattr__(
            self,
            "control_max_iterations",
            _budget(self.control_max_iterations, "iteration budget", 1, 4096),
        )
        object.__setattr__(self, "_graph_token", affine_model_token(self.asset))

    def inputs(self, target, target_velocity=None, *, feedforward=None, loads=None):
        k, n = len(self.state.q), self.asset.tree.model.dof_count
        drive = np.zeros((k, 3), np.float32)
        for column, value in enumerate((target, target_velocity, feedforward)):
            if value is not None:
                drive[:, column] = as_fp32(value, "affine drive", (k,))
            elif column == 0:
                raise ValueError("affine target is required")
        if loads is None:
            loads = RobotInputs.zeros(n)
        if not isinstance(loads, RobotInputs):
            raise TypeError("affine full loads must be RobotInputs")
        full = np.zeros((n, 7), np.float32)
        full[:, 0] = as_fp32(loads.effort, "affine full effort", (n,))
        full[:, 1:4] = as_fp32(loads.force, "affine full COM force", (n, 3))
        full[:, 4:7] = as_fp32(loads.torque, "affine full COM torque", (n, 3))
        return AffineTreeInputs(self._graph_token, drive, full)


@dataclass(frozen=True)
class AffineTreeCheckpoint:
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


@dataclass(frozen=True, eq=False)
class AffineTreeKinematics(TreeKinematics):
    independent_position: np.ndarray
    independent_velocity: np.ndarray
    joint_names: tuple
    full_joint_names: tuple

    def __post_init__(self):
        super().__post_init__()
        for name in ("independent_position", "independent_velocity"):
            object.__setattr__(self, name, _owned(getattr(self, name)))


@dataclass(frozen=True, eq=False)
class AffineTreeDiagnostic(TreeDiagnostic):
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
        ):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        object.__setattr__(self, "constraint_power_w", float(self.constraint_power_w))


@dataclass(frozen=True, eq=False)
class AffineTreeWorldSpec:
    physical: TreeWorldSpec
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
    initial_loads: np.ndarray

    @property
    def kind(self):
        return 7 + self.delay

    @property
    def model_token(self):
        return self.physical.model_token

    def packet(self, value):
        if not isinstance(value, AffineTreeInputs) or value.graph_token != self.graph_token:
            raise ValueError("affine command belongs to a different graph or input kind")
        return (
            _owned(as_fp32(value.drive, "affine drive", (self.k, 3))),
            _owned(as_fp32(value.loads, "affine loads", (self.n, 7))),
        )

    def check_state(self, reduced, full):
        for row, limits, name in (
            (reduced, self.limits, "independent"),
            (full, self.full_limits, "full"),
        ):
            q, v = row[:, 0], row[:, 1]
            if not np.isfinite(row).all() or np.any(abs(q) > TREE_COORDINATE_LIMIT):
                raise ValueError(f"affine {name} state exceeds finite coordinate domain")
            if (
                np.any(q < limits[:, 0])
                or np.any(q > limits[:, 1])
                or np.any(abs(v) > limits[:, 2])
            ):
                raise ValueError(f"affine {name} state exceeds joint interval or velocity limit")
        _near(full[:, 0], self.matrix.astype(float) @ reduced[:, 0] + self.offset, "full position")
        _near(full[:, 1], self.matrix.astype(float) @ reduced[:, 1], "full velocity")
        motor = np.zeros(self.n)
        motor[self.independent] = reduced[:, 2]
        _near(full[:, 2], motor, "full motor effort")
        _near(reduced[:, 3], self.matrix.astype(float).T @ full[:, 3], "generalized applied load")
        for row, (driver, follower) in enumerate(self.row_indices):
            gain, offset = self.row_values[row].astype(float)
            _near(
                full[follower, 0], gain * float(full[driver, 0]) + offset, "original row position"
            )
            _near(full[follower, 1], gain * float(full[driver, 1]), "original row velocity")

    def checkpoint(self, world, reduced, full, tick, drive, loads):
        a, b = np.asarray(reduced)[: self.k], np.asarray(full)[: self.n]
        return AffineTreeCheckpoint(
            world_key(world),
            self.topology,
            int(tick),
            _nested(a[:, 0]),
            _nested(a[:, 1]),
            _nested(a[:, 2]),
            _nested(a[:, 4:7]) if self.delay else (),
            (_nested(np.asarray(drive)[: self.k]), _nested(np.asarray(loads)[: self.n])),
            _nested(a[:, 3]),
            _nested(b[:, 0]),
            _nested(b[:, 1]),
            _nested(b[:, 2]),
            _nested(b[:, 3]),
        )

    def restore_row(self, checkpoint, world):
        if (
            not isinstance(checkpoint, AffineTreeCheckpoint)
            or checkpoint.world != world
            or checkpoint.topology != self.topology
        ):
            raise ValueError("affine checkpoint world/topology/graph does not match")
        tick = _budget(checkpoint.tick, "checkpoint tick", 0, 2147483647)
        reduced, full = np.zeros((self.k, 7), np.float32), np.zeros((self.n, 7), np.float32)
        for j, name in enumerate(("position", "velocity", "applied_effort", "generalized_applied")):
            reduced[:, j] = as_fp32(getattr(checkpoint, name), "restored affine " + name, (self.k,))
            full[:, j] = as_fp32(
                getattr(checkpoint, "full_" + name), "restored affine full " + name, (self.n,)
            )
        if self.delay:
            reduced[:, 4:7] = as_fp32(checkpoint.history, "restored affine history", (self.k, 3))
        elif checkpoint.history != ():
            raise ValueError("undelayed affine checkpoint cannot contain history")
        if not isinstance(checkpoint.control, tuple) or len(checkpoint.control) != 2:
            raise ValueError("affine held control needs independent drive and full loads")
        drive = as_fp32(checkpoint.control[0], "restored affine drive", (self.k, 3))
        loads = as_fp32(checkpoint.control[1], "restored affine loads", (self.n, 7))
        self.check_state(reduced, full)
        return reduced, full, drive, loads, tick


def compile_affine_tree_world(world):
    """Return owned original graph plus C4 physical layout; never create a program."""
    if not isinstance(world, AffineRobotTree):
        raise TypeError("expected AffineRobotTree")
    asset, coordinates = world.asset, _coordinates(world.asset)
    physical = dict(compile_tree_physical(asset.tree.model))
    n, k = physical.pop("n"), len(coordinates.joint_names)
    model_token, joint_types = physical.pop("model_token"), physical.pop("joint_types")
    if not 1 <= k < n <= 32:
        raise ValueError("affine physical joint count must satisfy 1<=K<N<=32")
    graph_token = affine_model_token(asset)
    if graph_token != world._graph_token:
        raise ValueError("affine source model/graph changed after world construction")
    matrix = _owned(as_fp32(coordinates.matrix, "affine composed multiplier matrix", (n, k)))
    offset = _owned(as_fp32(coordinates.offset_vector, "affine composed offset", (n,)))
    constraints = _owned(
        as_fp32(coordinates.constraint_rows, "affine original constraint rows", (n - k, n))
    )
    row_indices = _owned([(d, f) for d, f, _, _ in coordinates.indices], np.int32)
    row_values = _owned(
        as_fp32(
            [(a, b) for _, _, a, b in coordinates.indices],
            "affine original row multiplier/offset",
            (n - k, 2),
        )
    )
    full_limits = np.column_stack((asset.tree.lower, asset.tree.upper, asset.tree.velocity_limit))
    full_limits[:, 0] = np.maximum(full_limits[:, 0], -TREE_COORDINATE_LIMIT)
    full_limits[:, 1] = np.minimum(full_limits[:, 1], TREE_COORDINATE_LIMIT)
    full_limits = _owned(as_fp32(full_limits, "affine full limits", (n, 3)))
    limits = _intervals(full_limits, matrix, offset, coordinates.independent_indices)
    initial = np.zeros((k, 7), np.float32)
    initial[:, 0] = as_fp32(world.state.q, "initial affine position", (k,))
    initial[:, 1] = as_fp32(world.state.v, "initial affine velocity", (k,))
    drive = np.zeros((k, 3), np.float32)
    drive[:, 0] = initial[:, 0]
    if world.delay_steps:
        initial[:, 4:7] = drive
    full = np.zeros((n, 7), np.float32)
    # Initial compilation only. Runtime expansion happens on device every step.
    full[:, 0] = matrix @ initial[:, 0] + offset
    full[:, 1] = matrix @ initial[:, 1]
    zeros = _owned(np.zeros(n, np.float32))
    topology = (
        "affine_tree_v1",
        graph_token,
        _nested(world.kp),
        _nested(world.kd),
        _nested(world.effort_limit),
        world.delay_steps,
        world.control_max_iterations,
    )
    full_spec = TreeWorldSpec(
        "affine_physical",
        n,
        0,
        world.control_max_iterations,
        topology,
        model_token,
        joint_types,
        **physical,
        kp=zeros,
        kd=zeros,
        cap=zeros,
        initial_state=_owned(full),
        initial_command=_owned(np.zeros((n, 10), np.float32)),
    )
    result = AffineTreeWorldSpec(
        full_spec,
        n,
        k,
        n - k,
        world.delay_steps,
        world.control_max_iterations,
        topology,
        graph_token,
        coordinates.joint_names,
        tuple(asset.tree.joint_names),
        matrix,
        offset,
        constraints,
        _owned(coordinates.independent_indices, np.int32),
        row_indices,
        row_values,
        _owned(tuple(reversed(coordinates.forward)), np.int32),
        full_limits,
        limits,
        _owned(world.kp),
        _owned(world.kd),
        _owned(world.effort_limit),
        _owned(initial),
        _owned(drive),
        _owned(np.zeros((n, 7), np.float32)),
    )
    result.check_state(initial, full)
    return result

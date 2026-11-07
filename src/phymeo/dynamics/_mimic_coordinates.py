"""One affine coordinate compiler shared by fixed and floating scalar trees.

This module owns no robot, state or integration method. Relations are raw
(driver, follower, multiplier, offset) tuples so the existing public relation
class can keep its original module without an import cycle.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from ..core.validation import finite_real


def _frozen(values):
    source = np.asarray(values, dtype=np.float64)
    return np.frombuffer(source.tobytes(), dtype=np.float64).reshape(source.shape)


@dataclass(frozen=True, eq=False)
class MimicCoordinates:
    """Owned affine map and original constraint rows in full scalar-joint order."""

    indices: tuple
    forward: tuple
    independent_indices: tuple
    matrix: np.ndarray
    offset_vector: np.ndarray
    constraint_rows: np.ndarray
    joint_names: tuple
    lower: tuple
    upper: tuple
    velocity_limit: tuple
    effort_limit: tuple


def _limits(values, count, name, *, nonnegative=False):
    values = tuple(values)
    if len(values) != count:
        raise ValueError(f"{name} must match the full scalar-joint count")
    result = tuple(finite_real(value, name) for value in values)
    if nonnegative and any(value < 0 for value in result):
        raise ValueError(f"{name} must be nonnegative")
    return result


def compile_mimic_coordinates(names, types, lower, upper, velocity, effort, relations_tuple):
    """Compile a nonempty acyclic same-type affine graph with bounded metadata.

    Every follower has exactly one incoming relation. All graph calculations
    use the original scalar order; only independent roots receive motors.
    Matrices are backed by owned immutable bytes, including empty dimensions.
    """
    if isinstance(names, (str, bytes)) or isinstance(types, (str, bytes)):
        raise ValueError("names and types must be scalar-joint sequences")
    names, types = tuple(names), tuple(types)
    count = len(names)
    if not count or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("joint names must be nonempty strings")
    if len(set(names)) != count:
        raise ValueError("joint names must be unique")
    if len(types) != count or any(kind not in ("revolute", "prismatic") for kind in types):
        raise ValueError("mimic requires revolute or prismatic scalar joint types")
    full_lower = _limits(lower, count, "lower position")
    full_upper = _limits(upper, count, "upper position")
    speeds = _limits(velocity, count, "velocity limit", nonnegative=True)
    efforts = _limits(effort, count, "effort limit", nonnegative=True)
    if any(lo >= hi for lo, hi in zip(full_lower, full_upper, strict=True)):
        raise ValueError("each full joint needs a nonempty position interval")
    if not isinstance(relations_tuple, tuple) or not relations_tuple:
        raise ValueError("mimic relations must be a nonempty tuple of raw rows")
    if any(not isinstance(row, tuple) or len(row) != 4 for row in relations_tuple):
        raise ValueError("each mimic row must be (driver, follower, multiplier, offset)")

    name_index = {name: i for i, name in enumerate(names)}
    followers, edges = set(), []
    outgoing = [[] for _ in names]
    rows = np.zeros((len(relations_tuple), count))
    for i, (driver, follower, multiplier, offset) in enumerate(relations_tuple):
        if not isinstance(driver, str) or not isinstance(follower, str):
            raise ValueError("mimic driver and follower must be named scalar joints")
        d, f = name_index.get(driver), name_index.get(follower)
        if d is None or f is None:
            raise ValueError(f"unknown mimic driver or follower in {driver!r} -> {follower!r}")
        if d == f:
            raise ValueError(f"mimic cycle at joint {follower!r}")
        if f in followers:
            raise ValueError(f"mimic follower {follower!r} has multiple drivers")
        if types[d] != types[f]:
            raise ValueError(f"mimic {follower!r} and driver must use the same scalar joint type")
        a = finite_real(multiplier, "mimic multiplier")
        b = finite_real(offset, "mimic offset")
        if a == 0:
            raise ValueError(f"zero mimic multiplier on {follower!r}")
        followers.add(f)
        outgoing[d].append(i)
        edges.append((d, f, a, b))
        rows[i, d], rows[i, f] = -a, 1.0

    independent = tuple(i for i in range(count) if i not in followers)
    columns = {i: column for column, i in enumerate(independent)}
    matrix = np.zeros((count, len(independent)))
    for i, column in columns.items():
        matrix[i, column] = 1.0
    offsets = np.zeros(count)
    lower = [full_lower[i] for i in independent]
    upper = [full_upper[i] for i in independent]
    speed = [speeds[i] for i in independent]
    queue, forward = deque(independent), []
    while queue:
        parent = queue.popleft()
        for row in outgoing[parent]:
            d, f, a, b = edges[row]
            column = columns[d]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
                gain = a * matrix[d, column]
                intercept = a * offsets[d] + b
                if gain == 0 or not np.isfinite((gain, intercept)).all():
                    raise ValueError(f"mimic chain to {names[f]!r} cannot be represented in FP64")
                lo, hi = sorted(
                    ((full_lower[f] - intercept) / gain, (full_upper[f] - intercept) / gain)
                )
                bound = speeds[f] / abs(gain)
            if not np.isfinite((lo, hi, bound)).all() or (bound == 0 and speeds[f] > 0):
                raise ValueError(f"mimic interval or speed at {names[f]!r} is outside FP64 range")
            matrix[f, column], offsets[f], columns[f] = gain, intercept, column
            lower[column] = max(lower[column], float(lo))
            upper[column] = min(upper[column], float(hi))
            if lower[column] >= upper[column]:
                raise ValueError(
                    f"empty movable intersection for {names[independent[column]]!r} at mimic follower {names[f]!r}"
                )
            speed[column] = min(speed[column], float(bound))
            forward.append(row)
            queue.append(f)
    if len(forward) != len(edges):
        unresolved = [names[f] for _, f, _, _ in edges if f not in columns]
        raise ValueError(f"mimic cycle prevents resolving joints {unresolved}")
    return MimicCoordinates(
        tuple(edges),
        tuple(forward),
        independent,
        _frozen(matrix),
        _frozen(offsets),
        _frozen(rows),
        tuple(names[i] for i in independent),
        tuple(lower),
        tuple(upper),
        tuple(speed),
        tuple(efforts[i] for i in independent),
    )

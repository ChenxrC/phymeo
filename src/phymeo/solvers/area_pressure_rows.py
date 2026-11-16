"""HOST algebra for area-pressure rows in one shared generalized basis.

These are frozen-geometry elastic coefficients, not MuJoCo solref/solimp rows,
physical gaps, impulses, or a dynamics solve. Force mapping consumes explicit
caller-supplied normal forces without advancing or publishing any state.
"""

from dataclasses import dataclass

import numpy as np

from ..geometry.area_pressure import (
    AreaPressureIdentity,
    AreaPressureStamp,
    _array,
    _identifier,
    _identity,
    _normals,
    _quadrature,
    _same,
    _sample_ids,
    _sequence,
    _triangle_samples,
    _stamp,
)

MAX_DOFS = 256
MAX_JACOBIAN_ITEMS = 1_000_000


def _dof_ids(value):
    entries = _sequence(value, MAX_DOFS, "dof_ids")
    if not entries:
        raise ValueError("at least one shared generalized coordinate is required")
    result = tuple(_identifier(entry, "dof_id") for entry in entries)
    if len(set(result)) != len(result):
        raise ValueError("dof_ids must be unique")
    return result


def _jacobian_shape(count, dofs):
    if count * 3 * dofs > MAX_JACOBIAN_ITEMS:
        raise ValueError("each Jacobian is limited to 1,000,000 elements")
    return count, 3, dofs


@dataclass(frozen=True, eq=False)
class AreaPressureKinematics:
    """Two point Jacobians at the stamped samples, in the same complete basis."""

    stamp: AreaPressureStamp
    sample_ids: tuple[tuple[str, int], ...]
    dof_ids: tuple[str, ...]
    jacobian_a: np.ndarray
    jacobian_b: np.ndarray

    def __post_init__(self):
        stamp = _stamp(self.stamp)
        samples = _sample_ids(self.sample_ids)
        dofs = _dof_ids(self.dof_ids)
        shape = _jacobian_shape(len(samples), len(dofs))
        ja = _array(self.jacobian_a, shape, "jacobian_a")
        jb = _array(self.jacobian_b, shape, "jacobian_b")
        object.__setattr__(self, "stamp", stamp)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "dof_ids", dofs)
        object.__setattr__(self, "jacobian_a", ja)
        object.__setattr__(self, "jacobian_b", jb)


def _kinematics(value):
    if not isinstance(value, AreaPressureKinematics):
        raise ValueError("expected AreaPressureKinematics")
    return AreaPressureKinematics(
        value.stamp, value.sample_ids, value.dof_ids, value.jacobian_a, value.jacobian_b
    )


def _normal_jacobian(normals, ja, jb):
    try:
        with np.errstate(over="raise", invalid="raise"):
            difference = ja - jb
            normal = np.einsum("qi,qid->qd", normals, difference)
    except FloatingPointError as error:
        raise ValueError("relative point Jacobian arithmetic overflowed") from error
    if not np.isfinite(difference).all() or not np.isfinite(normal).all():
        raise ValueError("relative point Jacobian must be finite")
    return normal


@dataclass(frozen=True, eq=False)
class AreaPressureRows:
    identity: AreaPressureIdentity
    stamp: AreaPressureStamp
    sample_ids: tuple[tuple[str, int], ...]
    source_ids: tuple[tuple[str, str], ...]
    dof_ids: tuple[str, ...]
    points: np.ndarray
    normals: np.ndarray
    jacobian_a: np.ndarray
    jacobian_b: np.ndarray
    normal_jacobian: np.ndarray
    stiffness: np.ndarray
    surrogate_gap: np.ndarray
    elastic_force: np.ndarray

    def __post_init__(self):
        identity = _identity(self.identity)
        kin = AreaPressureKinematics(
            self.stamp, self.sample_ids, self.dof_ids, self.jacobian_a, self.jacobian_b
        )
        count, dofs = len(kin.sample_ids), len(kin.dof_ids)
        _, sources = _triangle_samples(kin.sample_ids, self.source_ids)
        values = {
            "points": _array(self.points, (count, 3), "points"),
            "normals": _normals(self.normals, count),
            "normal_jacobian": _array(self.normal_jacobian, (count, dofs), "normal_jacobian"),
            "stiffness": _array(self.stiffness, (count,), "stiffness"),
            "surrogate_gap": _array(self.surrogate_gap, (count,), "surrogate_gap"),
            "elastic_force": _array(self.elastic_force, (count,), "elastic_force"),
        }
        if np.any(values["stiffness"] <= 0) or np.any(values["surrogate_gap"] > 0) or np.any(values["elastic_force"] < 0):
            raise ValueError("rows require positive stiffness and compressive elastic coefficients")
        expected_j = _normal_jacobian(values["normals"], kin.jacobian_a, kin.jacobian_b)
        _same(values["normal_jacobian"], expected_j, "normal Jacobian")
        try:
            with np.errstate(over="raise", invalid="raise"):
                expected_force = -values["stiffness"] * values["surrogate_gap"]
        except FloatingPointError as error:
            raise ValueError("elastic row coefficient arithmetic overflowed") from error
        if not np.isfinite(expected_force).all():
            raise ValueError("elastic row coefficient product must be finite")
        _same(values["elastic_force"], expected_force, "elastic force and -k*phi")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "stamp", kin.stamp)
        object.__setattr__(self, "sample_ids", kin.sample_ids)
        object.__setattr__(self, "source_ids", sources)
        object.__setattr__(self, "dof_ids", kin.dof_ids)
        object.__setattr__(self, "jacobian_a", kin.jacobian_a)
        object.__setattr__(self, "jacobian_b", kin.jacobian_b)
        for name, value in values.items():
            object.__setattr__(self, name, value)


def _rows(value):
    if not isinstance(value, AreaPressureRows):
        raise ValueError("expected AreaPressureRows")
    return AreaPressureRows(
        value.identity, value.stamp, value.sample_ids, value.source_ids, value.dof_ids,
        value.points, value.normals, value.jacobian_a, value.jacobian_b,
        value.normal_jacobian, value.stiffness, value.surrogate_gap, value.elastic_force,
    )


def assemble_area_pressure_rows(quadrature, kinematics):
    """Return J=n.T*(JA-JB), k=dA*g, phi=-p/g and f0=dA*p in SI units."""
    quad, kin = _quadrature(quadrature), _kinematics(kinematics)
    if quad.stamp != kin.stamp or quad.sample_ids != kin.sample_ids:
        raise ValueError("quadrature and kinematics must have the same stamp and sample order")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            stiffness = quad.dA * quad.gradient
            gap = -quad.pressure / quad.gradient
            force = quad.dA * quad.pressure
    except FloatingPointError as error:
        raise ValueError("area-pressure row arithmetic overflowed") from error
    return AreaPressureRows(
        quad.identity, quad.stamp, quad.sample_ids, quad.source_ids, kin.dof_ids,
        quad.points, quad.normals, kin.jacobian_a, kin.jacobian_b,
        _normal_jacobian(quad.normals, kin.jacobian_a, kin.jacobian_b),
        stiffness, gap, force,
    )


@dataclass(frozen=True, eq=False)
class AreaPressureLoads:
    """Caller-force pullbacks and action/reaction wrenches about world origin."""

    identity: AreaPressureIdentity
    stamp: AreaPressureStamp
    sample_ids: tuple[tuple[str, int], ...]
    source_ids: tuple[tuple[str, str], ...]
    dof_ids: tuple[str, ...]
    generalized_a: np.ndarray
    generalized_b: np.ndarray
    generalized_total: np.ndarray
    wrench_a: np.ndarray
    wrench_b: np.ndarray

    def __post_init__(self):
        identity, stamp = _identity(self.identity), _stamp(self.stamp)
        samples, dofs = _sample_ids(self.sample_ids), _dof_ids(self.dof_ids)
        _, sources = _triangle_samples(samples, self.source_ids)
        values = {
            name: _array(getattr(self, name), (len(dofs),), name)
            for name in ("generalized_a", "generalized_b", "generalized_total")
        }
        values.update({name: _array(getattr(self, name), (6,), name) for name in ("wrench_a", "wrench_b")})
        try:
            with np.errstate(over="raise", invalid="raise"):
                total = values["generalized_a"] + values["generalized_b"]
        except FloatingPointError as error:
            raise ValueError("generalized force sum overflowed") from error
        if not np.isfinite(total).all():
            raise ValueError("generalized force sum must be finite")
        _same(values["generalized_total"], total, "generalized total")
        _same(values["wrench_b"], -values["wrench_a"], "common-origin action/reaction")
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "stamp", stamp)
        object.__setattr__(self, "sample_ids", samples)
        object.__setattr__(self, "source_ids", sources)
        object.__setattr__(self, "dof_ids", dofs)
        for name, value in values.items():
            object.__setattr__(self, name, value)


def map_area_pressure_normal_forces(rows, normal_forces, *, stamp):
    """Map nonnegative forces in N; preserve both full shared-coordinate terms."""
    rows = _rows(rows)
    supplied_stamp = _stamp(stamp)
    if supplied_stamp != rows.stamp:
        raise ValueError("force stamp must match the area-pressure row stamp")
    count = len(rows.sample_ids)
    forces = _array(normal_forces, (count,), "normal_forces")
    if np.any(forces < 0):
        raise ValueError("normal_forces must be nonnegative")
    try:
        with np.errstate(over="raise", invalid="raise"):
            world = forces[:, None] * rows.normals
            moments = np.cross(rows.points, world)
            if not np.isfinite(world).all() or not np.isfinite(moments).all():
                raise ValueError("each world force and origin moment must be finite")
            generalized_a = np.einsum("qid,qi->d", rows.jacobian_a, world)
            generalized_b = np.einsum("qid,qi->d", rows.jacobian_b, -world)
            generalized_total = generalized_a + generalized_b
            wrench_a = np.concatenate((world.sum(axis=0), moments.sum(axis=0)))
    except FloatingPointError as error:
        raise ValueError("area-pressure force pullback arithmetic overflowed") from error
    return AreaPressureLoads(
        rows.identity, rows.stamp, rows.sample_ids, rows.source_ids, rows.dof_ids,
        generalized_a, generalized_b, generalized_total, wrench_a, -wrench_a,
    )

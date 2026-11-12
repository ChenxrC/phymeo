"""Owned axial material parameters and a read-only device-phase observer.

No TYX import or runtime is created here. This observer validates recorded
velocities and impulses; it never supplies replacement physics velocities.
"""

from dataclasses import dataclass, field

import numpy as np


def _owned(value, dtype):
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _same32(a, b):
    return np.array_equal(np.asarray(a, np.float32).view(np.uint32),
                          np.asarray(b, np.float32).view(np.uint32))


def _native(value, shape, label):
    array = np.asarray(value)
    if array.dtype != np.dtype(np.float32) or array.shape != shape:
        raise ValueError(f"{label} must be stored float32 of shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be finite")
    return array


@dataclass(frozen=True, eq=False)
class DistanceDissipation:
    ordered_point_ids: np.ndarray
    coefficients_ns_per_m: np.ndarray

    def __post_init__(self):
        ids = np.asarray(self.ordered_point_ids)
        if ids.dtype != np.dtype(np.int64) or ids.ndim != 2 or ids.shape[1] != 2:
            raise ValueError("ordered_point_ids must be an int64 (E,2) array")
        if len(ids) > 1536 or np.any(ids < 0) or np.any(ids[:, 0] == ids[:, 1]):
            raise ValueError("ordered dissipation edge IDs are outside the bounded domain")
        source = np.asarray(self.coefficients_ns_per_m)
        if source.dtype.kind not in "fiu" or source.shape != (len(ids),):
            raise ValueError("coefficients_ns_per_m must have one numeric value per edge")
        coefficients = source.astype(np.float64)
        if np.any(~np.isfinite(coefficients)) or np.any(coefficients < 0):
            raise ValueError("coefficients_ns_per_m must be finite and nonnegative")
        with np.errstate(over="ignore", under="ignore"):
            stored = coefficients.astype(np.float32)
        if (
            np.any(~np.isfinite(stored))
            or np.any((coefficients != 0) & (stored == 0))
            or np.any((stored != 0) & (np.abs(stored) < np.finfo(np.float32).tiny))
        ):
            raise ValueError("coefficients must be finite normal-or-zero in float32 without underflow")
        object.__setattr__(self, "ordered_point_ids", _owned(ids, np.int64))
        object.__setattr__(self, "coefficients_ns_per_m", _owned(coefficients, np.float64))


@dataclass(frozen=True, eq=False)
class DistanceDissipationStep:
    ordered_point_ids: np.ndarray
    coefficients_ns_per_m: np.ndarray
    reconstructed_velocity: np.ndarray
    dissipated_velocity: np.ndarray
    trace: np.ndarray
    node_impulse: np.ndarray
    node_velocity_impulse: np.ndarray
    kinetic_change: np.ndarray
    ideal_work: np.ndarray
    rounding_work: np.ndarray
    schema: str = field(default="distance-dissipation-step-v1", init=False)

    def __post_init__(self):
        parameters = DistanceDissipation(self.ordered_point_ids, self.coefficients_ns_per_m)
        e = len(parameters.ordered_point_ids)
        before = np.asarray(self.reconstructed_velocity)
        if before.ndim != 2 or before.shape[1] != 3 or not 1 <= len(before) <= 512:
            raise ValueError("phase velocities require 1..512 three-coordinate nodes")
        n = len(before)
        object.__setattr__(self, "ordered_point_ids", parameters.ordered_point_ids)
        object.__setattr__(self, "coefficients_ns_per_m", parameters.coefficients_ns_per_m)
        for name, shape, dtype in (
            ("reconstructed_velocity", (n,3), np.float32),
            ("dissipated_velocity", (n,3), np.float32),
            ("trace", (e,32), np.float32),
            ("node_impulse", (n,3), np.float64),
            ("node_velocity_impulse", (n,3), np.float64),
            ("kinetic_change", (e,), np.float64),
            ("ideal_work", (e,), np.float64),
            ("rounding_work", (e,), np.float64),
        ):
            array = np.asarray(getattr(self, name))
            if array.dtype != np.dtype(dtype) or array.shape != shape or not np.all(np.isfinite(array)):
                raise ValueError(f"invalid complete phase field {name}")
            object.__setattr__(self, name, _owned(array, dtype))


def _dot32(a, b):
    mul = lambda x, y: np.float32(np.float32(x) * np.float32(y))
    add = lambda x, y: np.float32(np.float32(x) + np.float32(y))
    return add(add(mul(a[0],b[0]), mul(a[1],b[1])), mul(a[2],b[2]))


def make_dissipation_step(parameters, native_edges, weights, reconstructed_velocity,
                          dissipated_velocity, trace):
    """Audit one complete ordered device pass using registered stored inverse masses."""
    if type(parameters) is not DistanceDissipation:
        raise TypeError("parameters must be an owned DistanceDissipation")
    edges = np.asarray(native_edges)
    e = len(parameters.ordered_point_ids)
    if edges.dtype.kind not in "iu" or edges.shape != (e,2):
        raise ValueError("native_edges must preserve the complete declared edge order")
    before = np.asarray(reconstructed_velocity)
    if before.ndim != 2 or before.shape[1] != 3 or not 1 <= len(before) <= 512:
        raise ValueError("invalid phase node layout")
    n = len(before)
    before = _native(before, (n,3), "reconstructed_velocity")
    after = _native(dissipated_velocity, (n,3), "dissipated_velocity")
    w = _native(weights, (n,), "registered inverse masses")
    rows = _native(trace, (e,32), "trace")
    if np.any(w < 0) or np.any((w != 0) & (w < np.finfo(np.float32).tiny)):
        raise ValueError("registered inverse masses must be nonnegative normal-or-zero")
    if np.any(edges < 0) or np.any(edges >= n) or np.any(edges[:,0] == edges[:,1]):
        raise ValueError("native edge lies outside the registered node layout")
    if np.any(before[w == 0] != 0) or np.any(after[w == 0] != 0):
        raise ValueError("moving prescribed fixed nodes are unsupported")
    # Check consistent stable-ID/native-index correspondence; the owner also
    # checks exact IDs against its original point_ids[constraints.indices].
    point_to_native, native_to_point = {}, {}
    for pair, native in zip(parameters.ordered_point_ids, edges, strict=True):
        for point, index in zip(pair, native, strict=True):
            point, index = int(point), int(index)
            if point_to_native.setdefault(point,index) != index or native_to_point.setdefault(index,point) != point:
                raise ValueError("declared edge order/orientation does not bind native IDs")
    current = before.copy()
    node_impulse = np.zeros((n,3), np.float64)
    velocity_impulse = np.zeros((n,3), np.float64)
    kinetic = np.zeros(e, np.float64)
    ideal = np.zeros(e, np.float64)
    rounding = np.zeros(e, np.float64)
    f = np.float32
    mul = lambda a, b: f(f(a) * f(b))
    add = lambda a, b: f(f(a) + f(b))
    for k, (i, j) in enumerate(edges):
        row = rows[k]
        coefficient = f(parameters.coefficients_ns_per_m[k])
        if not _same32(row[19:22], [coefficient,w[i],w[j]]):
            raise ValueError("trace parameters do not equal registered stored parameters")
        if row[18] <= 0 or (k and not _same32(row[18],rows[0,18])):
            raise ValueError("every edge must consume the same positive physical h exactly once")
        if np.any(row[28:] != 0):
            raise ValueError("reserved trace columns must remain zero")
        if not _same32(row[6:9],current[i]) or not _same32(row[9:12],current[j]):
            raise ValueError("trace pre-velocity does not connect in declared order")
        if coefficient == 0:
            if np.any(row[:6] != 0) or np.any(row[22:23] != 0) or row[23] != 1 or np.any(row[24:28] != 0):
                raise ValueError("zero coefficient row invented a direction or impulse")
            if not _same32(row[6:12],row[12:18]):
                raise ValueError("zero coefficient row modified endpoint velocities")
        else:
            if row[22] <= 0 or row[27] <= 0 or row[23] < 1:
                raise ValueError("active dissipation direction/denominator is invalid")
            direction = row[:3]
            g2 = _dot32(direction,direction)
            relative_before = np.array([add(a,-b) for a,b in zip(row[6:9],row[9:12],strict=True)],np.float32)
            relative_after = np.array([add(a,-b) for a,b in zip(row[12:15],row[15:18],strict=True)],np.float32)
            s = _dot32(direction,relative_before)
            s_after = _dot32(direction,relative_after)
            b = mul(row[18],coefficient)
            denominator = add(1,mul(mul(b,add(w[i],w[j])),g2))
            impulse = f(-mul(b,s)/denominator)
            expected = np.array([s,impulse,s_after,denominator,g2],np.float32)
            if not np.all(np.isfinite(expected)) or not _same32(row[[3,4,5,23,27]],expected):
                raise ValueError("trace scalar primitive evidence is inconsistent")
            requested = np.array([mul(impulse,d) for d in direction],np.float32)
            if not _same32(row[24:27],requested):
                raise ValueError("trace requested device impulse is inconsistent")
        # Follow the device evidence, not a HOST-generated update. Reject a
        # fixed endpoint change before placing recorded values into the chain.
        if (w[i] == 0 and not _same32(row[12:15],current[i])) or (w[j] == 0 and not _same32(row[15:18],current[j])):
            raise ValueError("fixed endpoint velocity changed")
        for index, old, new in ((i,row[6:9],row[12:15]),(j,row[9:12],row[15:18])):
            if w[index] > 0:
                old64, new64 = old.astype(np.float64), new.astype(np.float64)
                kinetic[k] += float((new64 @ new64 - old64 @ old64)/(2*float(w[index])))
        ji, rate, g2 = map(float,row[[4,3,27]])
        ideal[k] = ji*rate + .5*(float(w[i])+float(w[j]))*g2*ji*ji
        rounding[k] = kinetic[k]-ideal[k]
        node_impulse[i] += row[24:27].astype(np.float64)
        node_impulse[j] -= row[24:27].astype(np.float64)
        current[i],current[j] = row[12:15],row[15:18]
    if not _same32(current,after):
        raise ValueError("final dissipated velocity is not the complete recorded chain")
    for i in range(n):
        if w[i] > 0:
            velocity_impulse[i] = (after[i].astype(np.float64)-before[i].astype(np.float64))/float(w[i])
    return DistanceDissipationStep(
        parameters.ordered_point_ids, parameters.coefficients_ns_per_m,
        before, after, rows, node_impulse, velocity_impulse, kinetic, ideal, rounding)

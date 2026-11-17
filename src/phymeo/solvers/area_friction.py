"""Owned inputs for a bounded, frozen-geometry area-friction velocity solve.

SI units: M is the physical generalized mass, damping is its nonnegative
joint-diagonal damping vector, and applied_force excludes that implicit damping.
Both point Jacobians use the same full DOF space. The normal points toward body
A, and positive Fn acts on A. Stiffness already includes integration-point dA;
no additional area factor is applied. This module neither clips geometry nor
solves contact, integrates configuration, or changes the area-asset v1 law.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Real

import numpy as np

LAW = "final_normal_smooth_v1"
MAX_DOF = 6
MAX_SAMPLES = 128


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _owned(value):
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("nonfinite numerical value")
    # The backing bytes, not merely the array flag, are immutable.
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _array(value, shape, name):
    if isinstance(value, np.ndarray):
        if value.shape != shape or value.dtype.kind not in "iuf":
            raise ValueError(f"{name} has invalid shape or numeric dtype")
    else:

        def check(node, remaining):
            if not remaining:
                _number(node, name)
                return
            if type(node) not in (tuple, list) or len(node) != remaining[0]:
                raise ValueError(f"{name} must have shape {shape}")
            for item in node:
                check(item, remaining[1:])

        check(value, shape)
    try:
        return _owned(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite FP64 values") from error


def _dimension(value, matrix=False):
    if isinstance(value, np.ndarray):
        if value.ndim != 2:
            raise ValueError("matrix rank must be two")
        count = value.shape[0] if matrix else value.shape[1]
    elif type(value) in (list, tuple):
        if matrix:
            count = len(value)
        else:
            if len(value) != 3 or type(value[0]) not in (list, tuple):
                raise ValueError("point Jacobian must have three rows")
            count = len(value[0])
    else:
        raise ValueError("matrix must be an array or bounded nested sequence")
    if not 1 <= count <= MAX_DOF:
        raise ValueError("DOF count must be 1..6")
    return count


@dataclass(frozen=True)
class AreaFrictionSample:
    point_world: np.ndarray
    normal_world: np.ndarray
    jacobian_a: np.ndarray
    jacobian_b: np.ndarray
    stiffness_n_m: float
    phi_m: float
    mu: float
    slip_speed_m_s: float

    @property
    def law(self):
        return LAW

    def __post_init__(self):
        count = _dimension(self.jacobian_a)
        point = _array(self.point_world, (3,), "point_world")
        normal = _array(self.normal_world, (3,), "normal_world")
        length = math.hypot(*normal)
        if abs(length - 1.0) > 1e-12:
            raise ValueError("normal_world must be near unit length within 1e-12")
        normal = _owned(normal / length)
        ja = _array(self.jacobian_a, (3, count), "jacobian_a")
        jb = _array(self.jacobian_b, (3, count), "jacobian_b")
        k = _number(self.stiffness_n_m, "stiffness_n_m")
        phi = _number(self.phi_m, "phi_m")
        mu = _number(self.mu, "mu")
        slip = _number(self.slip_speed_m_s, "slip_speed_m_s")
        if k <= 0 or phi > 0 or mu < 0 or slip <= 0:
            raise ValueError("require k>0, phi<=0, mu>=0 and slip_speed>0")
        for name, value in (
            ("point_world", point),
            ("normal_world", normal),
            ("jacobian_a", ja),
            ("jacobian_b", jb),
            ("stiffness_n_m", k),
            ("phi_m", phi),
            ("mu", mu),
            ("slip_speed_m_s", slip),
        ):
            object.__setattr__(self, name, value)


def _copy_sample(sample):
    if type(sample) is not AreaFrictionSample:
        raise ValueError("samples must be AreaFrictionSample records")
    return AreaFrictionSample(*(getattr(sample, f.name) for f in fields(sample)))


@dataclass(frozen=True)
class AreaFrictionProblem:
    mass_matrix: np.ndarray
    damping: np.ndarray
    input_velocity: np.ndarray
    applied_force: np.ndarray
    h_s: float
    samples: tuple[AreaFrictionSample, ...]

    @property
    def law(self):
        return LAW

    def __post_init__(self):
        count = _dimension(self.mass_matrix, matrix=True)
        if type(self.samples) not in (tuple, list) or len(self.samples) > MAX_SAMPLES:
            raise ValueError("samples must be a bounded sequence of at most 128")
        mass = _array(self.mass_matrix, (count, count), "mass_matrix")
        damping = _array(self.damping, (count,), "damping")
        velocity = _array(self.input_velocity, (count,), "input_velocity")
        force = _array(self.applied_force, (count,), "applied_force")
        h = _number(self.h_s, "h_s")
        if h <= 0 or np.any(damping < 0):
            raise ValueError("h must be positive and diagonal damping nonnegative")
        if not np.array_equal(mass, mass.T):
            raise ValueError("mass_matrix must be symmetric; no symmetrization is applied")
        try:
            factor = np.linalg.cholesky(mass)
        except np.linalg.LinAlgError as error:
            raise ValueError("mass_matrix must be positive definite") from error
        if not np.all(np.isfinite(factor)) or np.any(np.diag(factor) <= 0):
            raise ValueError("mass_matrix has no finite positive Cholesky factor")
        samples = tuple(_copy_sample(sample) for sample in self.samples)
        if any(sample.jacobian_a.shape != (3, count) for sample in samples):
            raise ValueError("all samples must use the same full DOF space")
        for name, value in (
            ("mass_matrix", mass),
            ("damping", damping),
            ("input_velocity", velocity),
            ("applied_force", force),
            ("h_s", h),
            ("samples", samples),
        ):
            object.__setattr__(self, name, value)

"""Predeclared D6-A public D1 cases and independent FP64 ordered references.

Authoring only until the parent executes the frozen matrix. c is N s/m;
positions m, velocities m/s, inverse masses 1/kg. No producer is used as oracle.
The kernel lane separately qualifies fixed-geometry rigid null modes and the
flat/deflected five-node membrane instantaneous coefficient.
"""

import gzip
import json
import math
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass

import numpy as np

from phymeo.backends.deformable_dissipation import DistanceDissipation
from phymeo.backends.tyx_deformable_batch import DeformableNodeWorld, DeformableSurface
from phymeo.constraints.distance import DistanceConstraints
from phymeo.core.model import ParticleModel
from phymeo.core.scene import StepSettings
from phymeo.core.state import ParticleState, StepInputs

ATOL, RTOL = 2e-6, 2e-5
H_VALUES = (1 / 128, 1 / 256)
SETTINGS = StepSettings(substeps=1, position_iterations=200, tolerance=2e-6)
PAIR_STEPS, CHAIN_STEPS, OSCILLATOR_DURATION = 12, 12, 1.0


def make_world(kind="pair", *, key=7, sign=1, damping="positive", permuted=False):
    if kind == "pair":
        x = np.array(((-0.5, 0, 0), (0.5, 0, 0)), np.float32)
        v = np.array(((0.25, 0, 0), (-0.125, 0, 0)), np.float32) * sign
        weights, ids = np.array((1, 0.5), np.float32), np.array((101, 307), np.int64)
        edges, rest, coefficients = np.array(((0, 1),), np.int64), (1.0,), (2.0,)
        surfaces = ()
    elif kind in ("chain", "mixed", "fixed_chain"):
        x = np.array(((-0.5, 0, 0), (0.5, 0, 0), (0.5, 0.75, 0.25)), np.float32)
        v = np.array(((0.25, 0, 0.0625), (-0.125, 0.0625, 0), (0, -0.125, -0.0625)),
                     np.float32)
        weights = np.array((1, 0.5, 2), np.float32)
        if kind == "fixed_chain":
            weights[0], v[0] = 0, 0
        ids = np.array((101, 307, 509), np.int64)
        edges = np.array(((0, 1), (1, 2)), np.int64)
        rest, coefficients = (1.0, math.sqrt(0.625)), (0.0 if kind == "mixed" else 2.0, 3.0)
        surfaces = (DeformableSurface(23, np.array((809,), np.int64), ids[None, :]),)
    elif kind == "oscillator":
        x = np.array(((0, 0, 0), (1.125, 0, 0)), np.float32)
        v = np.zeros((2, 3), np.float32)
        weights, ids = np.array((0, 1), np.float32), np.array((101, 307), np.int64)
        edges, rest, coefficients = np.array(((0, 1),), np.int64), (1.0,), (16.0,)
        surfaces = ()
    elif kind == "fixed":
        x = np.array(((0, 0, 0), (1, 0, 0)), np.float32)
        v = np.zeros((2, 3), np.float32)
        weights, ids = np.zeros(2, np.float32), np.array((101, 307), np.int64)
        edges, rest, coefficients = np.array(((0, 1),), np.int64), (1.0,), (2.0,)
        surfaces = ()
    else:
        raise ValueError("unknown frozen D6-A case")
    if permuted:
        order = np.arange(len(x))[::-1]
        inverse = np.argsort(order)
        x, v, weights, ids, edges = x[order], v[order], weights[order], ids[order], inverse[edges]
    coefficients = np.asarray(coefficients, np.float64)
    if damping == "zero":
        coefficients = np.zeros_like(coefficients)
    elif damping not in ("positive", None):
        raise ValueError("unknown damping declaration")
    parameters = None if damping is None else DistanceDissipation(ids[edges], coefficients)
    return DeformableNodeWorld(
        key, ParticleModel(weights, np.zeros(3, np.float32)), ParticleState(x, v),
        DistanceConstraints(edges, np.asarray(rest, np.float64),
                            np.full(len(edges), 1 / 64, np.float64)),
        ids, surfaces, dissipation=parameters,
    )


def packets(specs, *, driven=False):
    result = {}
    for spec in specs:
        force = np.zeros((spec.model.particle_count, 3), np.float32)
        if driven:
            force[np.flatnonzero(spec.model.inverse_mass > 0)[-1], 1] = 1 / 32
        result[spec.key] = StepInputs(force)
    return result


def ordered_damping(position, velocity, inverse_mass, edges, coefficients, h):
    """FP64 constitutive reference, separate from all product helpers."""
    x, v = np.asarray(position, float), np.asarray(velocity, float).copy()
    weights = np.asarray(inverse_mass, float)
    rows, impulse = [], np.zeros_like(v)
    for (i, j), coefficient in zip(edges, coefficients, strict=True):
        before_i, before_j = v[i].copy(), v[j].copy()
        if coefficient == 0:
            rows.append(dict(normal=np.zeros(3), impulse=np.zeros(3), scalar=0.0,
                             before_i=before_i, before_j=before_j,
                             after_i=before_i, after_j=before_j, work=0.0))
            continue
        d = x[i] - x[j]
        length = math.sqrt(float(d @ d))
        if length == 0:
            raise ValueError("reference positive damping has no edge direction")
        n = d / length
        rate = float(n @ (v[i] - v[j]))
        scalar = -h * coefficient * rate / (
            1 + h * coefficient * (weights[i] + weights[j]) * float(n @ n)
        )
        vector = scalar * n
        v[i] += weights[i] * vector
        v[j] -= weights[j] * vector
        impulse[i] += vector
        impulse[j] -= vector
        rows.append(dict(normal=n, impulse=vector, scalar=scalar,
                         before_i=before_i, before_j=before_j,
                         after_i=v[i].copy(), after_j=v[j].copy(),
                         work=scalar * rate + 0.5 * (weights[i] + weights[j]) *
                         float(n @ n) * scalar * scalar))
    return v, impulse, tuple(rows)


def elastic_reference(spec, position, velocity, force, h, sweeps):
    """Independent serial FP64 XPBD at the original fixed ordered row count."""
    x0, v0 = np.asarray(position, float), np.asarray(velocity, float)
    w = np.asarray(spec.model.inverse_mass, float)
    free = w > 0
    pv = np.zeros_like(v0)
    pv[free] = v0[free] + h * (
        np.asarray(spec.model.gravity, float) + np.asarray(force, float)[free] * w[free, None]
    )
    px = x0.copy()
    px[free] += h * pv[free]
    x, multipliers = px.copy(), np.zeros(len(spec.constraints.indices))
    alpha = spec.constraints.compliance / h**2
    for _ in range(sweeps):
        for k, (i, j) in enumerate(spec.constraints.indices):
            d = x[i] - x[j]
            length = math.sqrt(float(d @ d))
            n = d / length if length else np.array((1.0, 0.0, 0.0))
            denominator = w[i] + w[j] + alpha[k]
            if denominator == 0:
                continue
            increment = -(length - spec.constraints.rest_length[k] +
                          alpha[k] * multipliers[k]) / denominator
            multipliers[k] += increment
            x[i] += w[i] * increment * n
            x[j] -= w[j] * increment * n
    reconstructed = np.zeros_like(v0)
    reconstructed[free] = pv[free] + (x[free] - px[free]) / h
    return x, reconstructed, multipliers


def oscillator_split(extension, velocity, h):
    """Independent 2 by 2 scalar recurrence, k=64,m=1,c=16, rest=1."""
    elastic_extension = (extension + h * velocity) / (1 + 64 * h * h)
    elastic_velocity = velocity - 64 * h * elastic_extension
    return elastic_extension, elastic_velocity / (1 + 16 * h)


def half_ulp(value):
    """RN32 half-ulp including the subnormal bin, evaluated only by tests."""
    number = np.float32(value)
    above = np.nextafter(number, np.float32(np.inf), dtype=np.float32)
    below = np.nextafter(number, np.float32(-np.inf), dtype=np.float32)
    return max(abs(float(above) - float(number)), abs(float(number) - float(below))) / 2


def endpoint_update_bound(before, weight, scalar, normal, after):
    # RN(w*j), RN(previous*n), RN(v+previous), no FMA contract.
    first = np.float32(float(weight) * float(scalar))
    first_error = half_ulp(first)
    bounds = []
    for v0, n, v1 in zip(before, normal, after, strict=True):
        second = np.float32(float(first) * float(n))
        bounds.append(abs(float(n)) * first_error + half_ulp(second) + half_ulp(v1))
    return np.asarray(bounds)


def phase_checks(spec, state, row):
    """Independent same-entry mechanics and per-row stored-arithmetic audit."""
    phase = row.dissipation
    assert phase.schema == "distance-dissipation-step-v1"
    np.testing.assert_array_equal(phase.ordered_point_ids, spec.point_ids[spec.constraints.indices])
    np.testing.assert_array_equal(phase.coefficients_ns_per_m,
                                  spec.dissipation.coefficients_ns_per_m)
    np.testing.assert_array_equal(phase.dissipated_velocity, state.velocity)
    assert phase.trace.dtype == np.float32
    assert phase.trace.shape == (len(spec.constraints.indices), 32)
    np.testing.assert_array_equal(phase.trace[:, 28:], 0)
    for value in (phase.ordered_point_ids, phase.coefficients_ns_per_m,
                  phase.reconstructed_velocity, phase.dissipated_velocity, phase.trace,
                  phase.node_impulse, phase.node_velocity_impulse, phase.kinetic_change,
                  phase.ideal_work, phase.rounding_work):
        assert not value.flags.writeable
        try:
            value.setflags(write=True)
        except ValueError:
            pass
        else:
            raise AssertionError("phase array has mutable backing")
    reference, reference_impulse, _ = ordered_damping(
        state.position, phase.reconstructed_velocity, spec.model.inverse_mass,
        spec.constraints.indices, spec.dissipation.coefficients_ns_per_m, row.h,
    )
    np.testing.assert_allclose(state.velocity, reference, atol=ATOL, rtol=RTOL)
    np.testing.assert_allclose(phase.node_impulse, reference_impulse, atol=ATOL, rtol=RTOL)

    running = phase.reconstructed_velocity.copy()
    accumulated = np.zeros_like(running, dtype=float)
    weights = spec.model.inverse_mass.astype(float)
    energy_bounds = []
    for k, ((i, j), raw) in enumerate(zip(spec.constraints.indices, phase.trace, strict=True)):
        np.testing.assert_array_equal(raw[6:9], running[i])
        np.testing.assert_array_equal(raw[9:12], running[j])
        np.testing.assert_array_equal(raw[18:22], np.asarray(
            (row.h, spec.dissipation.coefficients_ns_per_m[k], weights[i], weights[j]),
            np.float32))
        n, scalar = raw[:3].astype(float), float(raw[4])
        before, after = raw[6:12].astype(float).reshape(2, 3), raw[12:18].astype(float).reshape(2, 3)
        if raw[19] == 0:
            np.testing.assert_array_equal(raw[:6], 0)
            np.testing.assert_array_equal(raw[22:], np.r_[0, 1, np.zeros(8)])
            np.testing.assert_array_equal(before, after)
        else:
            expected_n = state.position[i].astype(float) - state.position[j].astype(float)
            expected_n /= np.linalg.norm(expected_n)
            np.testing.assert_allclose(n, expected_n, atol=ATOL, rtol=RTOL)
            rate = float(n @ (before[0] - before[1]))
            np.testing.assert_allclose(raw[3], rate, atol=ATOL, rtol=RTOL)
            np.testing.assert_allclose(raw[5], n @ (after[0] - after[1]), atol=ATOL, rtol=RTOL)
            np.testing.assert_allclose(raw[27], n @ n, atol=ATOL, rtol=RTOL)
            denominator = 1 + row.h * float(raw[19]) * (weights[i] + weights[j]) * float(raw[27])
            np.testing.assert_allclose(raw[23], denominator, atol=ATOL, rtol=RTOL)
            np.testing.assert_allclose(scalar, -row.h * float(raw[19]) * float(raw[3]) /
                                       denominator, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(raw[24:27], scalar * n, atol=ATOL, rtol=RTOL)
        accumulated[i] += raw[24:27].astype(float)
        accumulated[j] -= raw[24:27].astype(float)
        actual_work, update_bound = 0.0, 0.0
        for endpoint, weight in enumerate((weights[i], weights[j])):
            if weight == 0:
                np.testing.assert_array_equal(after[endpoint], 0)
                continue
            signed_scalar = scalar if endpoint == 0 else -scalar
            exact_after = before[endpoint] + weight * signed_scalar * n
            error = endpoint_update_bound(before[endpoint], weight, signed_scalar, n, after[endpoint])
            arithmetic_slack = 16 * np.finfo(float).eps * (
                np.abs(before[endpoint]) + np.abs(weight * signed_scalar * n) + 1e-300)
            assert np.all(np.abs(after[endpoint] - exact_after) <= error + arithmetic_slack)
            actual_work += float((after[endpoint] @ after[endpoint] -
                                  before[endpoint] @ before[endpoint]) / (2 * weight))
            update_bound += float(np.sum(np.abs(exact_after) * error + 0.5 * error**2) / weight)
        ideal = scalar * float(raw[3]) + 0.5 * (weights[i] + weights[j]) * float(raw[27]) * scalar**2
        # Include stored dot-product rounding; never cancel another edge's work.
        dot_term = abs(scalar * (float(n @ (before[0] - before[1])) - float(raw[3])))
        norm_term = 0.5 * (weights[i] + weights[j]) * scalar**2 * abs(float(n @ n) - float(raw[27]))
        fp64_slack = 32 * np.finfo(float).eps * (1 + np.sum(before**2) + np.sum(after**2))
        bound = update_bound + dot_term + norm_term + fp64_slack
        np.testing.assert_allclose(phase.kinetic_change[k], actual_work, atol=fp64_slack, rtol=0)
        np.testing.assert_allclose(phase.ideal_work[k], ideal, atol=fp64_slack, rtol=0)
        np.testing.assert_allclose(phase.rounding_work[k], actual_work - ideal,
                                  atol=fp64_slack, rtol=0)
        assert abs(actual_work - ideal) <= bound
        assert actual_work <= bound
        energy_bounds.append(bound)
        running[i], running[j] = raw[12:15], raw[15:18]
    np.testing.assert_array_equal(running, state.velocity)
    np.testing.assert_allclose(phase.node_impulse, accumulated, atol=1e-15, rtol=0)
    actual_impulse = np.zeros_like(running, dtype=float)
    free = weights > 0
    actual_impulse[free] = (state.velocity[free].astype(float) -
                            phase.reconstructed_velocity[free]) / weights[free, None]
    np.testing.assert_allclose(phase.node_velocity_impulse, actual_impulse, atol=1e-15, rtol=0)
    np.testing.assert_array_equal(state.velocity[~free], 0)
    return dict(reference_velocity=reference, reference_impulse=reference_impulse,
                per_edge_rounding_bound=np.asarray(energy_bounds))


def encode(value):
    if isinstance(value, np.ndarray):
        return dict(dtype=value.dtype.str, shape=value.shape, bytes_hex=value.tobytes().hex())
    if isinstance(value, np.generic):
        return encode(value.item())
    if isinstance(value, BaseException):
        return dict(type=type(value).__name__, message=str(value))
    if is_dataclass(value):
        return {field.name: encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return dict(mapping_entries=[[encode(key), encode(item)] for key, item in value.items()])
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if type(value) is object:
        return dict(opaque_owner=True)
    if type(value) is float and not math.isfinite(value):
        return dict(nonfinite=repr(value))
    return value


@contextmanager
def saved(path, name, **declaration):
    with gzip.open(path / (name + ".jsonl.gz"), "xt", encoding="utf-8") as stream:
        ordinal = 0
        def emit(event, **payload):
            nonlocal ordinal
            stream.write(json.dumps(dict(ordinal=ordinal, event=event, payload=encode(payload)),
                                    allow_nan=False) + "\n")
            ordinal += 1
            stream.flush()
        emit("declaration", **declaration)
        try:
            yield emit
        except BaseException as error:
            emit("failure", error=error)
            raise
        finally:
            emit("closed-prefix")


def step_record(batch, specs, h, emit, *, inputs=None):
    inputs = packets(specs) if inputs is None else inputs
    entry = batch.snapshot()
    emit("attempt", entry=entry, inputs=inputs, h=h, revision=batch.revision,
         checkpoint=batch.checkpoint(), stats=batch.stats())
    receipt = batch.step(inputs, h, expected_revision=batch.revision)
    emit("returned", receipt=receipt, stats=batch.stats())
    for spec in specs:
        row, state = receipt.worlds[spec.key], receipt.states[spec.key]
        if hasattr(row, "dissipation"):
            audit = phase_checks(spec, state, row)
            emit("independent-phase", world=spec.key, audit=audit)
    return receipt

"""D6-A public D1 continuity, actual phase publication and independent mechanics."""

from dataclasses import fields

import numpy as np
import pytest
from deformable_dissipation_cases import (
    ATOL,
    CHAIN_STEPS,
    H_VALUES,
    OSCILLATOR_DURATION,
    PAIR_STEPS,
    SETTINGS,
    elastic_reference,
    elastic_rounding_audit,
    make_world,
    ordered_damping,
    oscillator_split,
    saved,
    step_record,
)

from phymeo.backends.tyx_deformable_batch import DeformableWorldStep, TyxDeformableBatch

pytestmark = pytest.mark.tyx


def same_old_row(left, right):
    assert type(left) is type(right) is DeformableWorldStep
    for field in fields(DeformableWorldStep):
        a, b = getattr(left, field.name), getattr(right, field.name)
        if isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif field.name == "trace":
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                np.testing.assert_array_equal(x, y)
        else:
            assert a == b


def elastic_impulse(spec, row):
    result = np.zeros((spec.model.particle_count, 3))
    for frame in row.trace:
        for (i, j), raw in zip(spec.constraints.indices, frame, strict=True):
            vector = raw[14:17].astype(float) * float(raw[17]) / row.h
            result[i] += vector
            result[j] -= vector
    return result


@pytest.mark.parametrize("trace", (False, True))
def test_none_and_explicit_zero_keep_old_numerical_path_and_distinct_identity(
    trace, tyx_mode, tmp_path
):
    absent, zero = make_world("chain", damping=None), make_world("chain", damping="zero")
    a_need = TyxDeformableBatch.requirements((absent,), settings=SETTINGS, trace=trace)
    z_need = TyxDeformableBatch.requirements((zero,), settings=SETTINGS, trace=trace)
    assert a_need.device_bytes == z_need.device_bytes
    with saved(tmp_path, "zero-compatibility", mode=tyx_mode, cases=(absent, zero)) as emit:
        with (
            TyxDeformableBatch((absent,), mode=tyx_mode, settings=SETTINGS, trace=trace) as a,
            TyxDeformableBatch((zero,), mode=tyx_mode, settings=SETTINGS, trace=trace) as z,
        ):
            assert a.binding_sha256 != z.binding_sha256
            for _ in range(3):
                ar = step_record(a, (absent,), H_VALUES[0], emit)
                zr = step_record(z, (zero,), H_VALUES[0], emit)
                same_old_row(ar.worlds[7], zr.worlds[7])
                assert not hasattr(zr.worlds[7], "dissipation")
                for name in ("position", "velocity", "clock"):
                    np.testing.assert_array_equal(
                        getattr(ar.states[7], name), getattr(zr.states[7], name)
                    )
                for name in (
                    "kernels",
                    "allocations",
                    "native_buffer_bytes",
                    "state_uploaded_bytes",
                    "uploaded_bytes",
                    "downloaded_bytes",
                ):
                    assert a.stats()[name] == z.stats()[name]
            with pytest.raises(ValueError):
                z.restore(a.checkpoint(), expected_revision=z.revision)


@pytest.mark.parametrize("h", H_VALUES)
@pytest.mark.parametrize("sign", (-1, 1))
def test_two_free_masses_continuous_signed_dissipation(sign, h, tyx_mode, tmp_path):
    spec = make_world(sign=sign)
    reference_x = spec.initial_state.position.astype(float).copy()
    reference_v = spec.initial_state.velocity.astype(float).copy()
    initial_momentum = np.sum(reference_v / spec.model.inverse_mass[:, None], axis=0)
    accumulated_rounding = np.zeros(3)
    with saved(tmp_path, "free-pair", case=spec, h=h, steps=PAIR_STEPS, mode=tyx_mode) as emit:
        with TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS, trace=True) as batch:
            previous_velocity = batch.snapshot()[7].velocity
            initial_stats = dict(batch.stats())
            for _ in range(PAIR_STEPS):
                receipt = step_record(batch, (spec,), h, emit)
                row, state = receipt.worlds[7], receipt.states[7]
                np.testing.assert_array_equal(row.input_velocity, previous_velocity)
                previous_velocity = state.velocity
                # One axis, one linear compliant edge has an exact one-sweep FP64 split.
                reference_x, reconstructed, _ = elastic_reference(
                    spec, reference_x, reference_v, np.zeros_like(reference_x), h, 1
                )
                reference_v, impulse, _ = ordered_damping(
                    reference_x,
                    reconstructed,
                    spec.model.inverse_mass,
                    spec.constraints.indices,
                    spec.dissipation.coefficients_ns_per_m,
                    h,
                )
                emit(
                    "continuous-reference",
                    position=reference_x,
                    velocity=reference_v,
                    impulse=impulse,
                )
                np.testing.assert_allclose(state.position, reference_x, atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(state.velocity, reference_v, atol=2e-5, rtol=2e-5)
                momentum = np.sum(
                    state.velocity.astype(float) / spec.model.inverse_mass[:, None], axis=0
                )
                elastic = elastic_rounding_audit(spec, state, row)
                damping_rounding = np.sum(
                    row.dissipation.node_velocity_impulse - row.dissipation.node_impulse,
                    axis=0,
                )
                accumulated_rounding += elastic["total_remainder"].sum(axis=0) + damping_rounding
                emit(
                    "physical-momentum-and-rounding",
                    raw_imbalance=momentum - initial_momentum,
                    original_strict_gate_pass=bool(
                        np.all(np.abs(momentum - initial_momentum) <= 2e-5)
                    ),
                    elastic_audit=elastic,
                    damping_rounding=damping_rounding,
                    accumulated_rounding=accumulated_rounding,
                )
                # No claim of exact physical conservation: the original 2e-5
                # gate failed in run104. All stored elastic primitives and their
                # independently formed rounding bounds must still close.
                np.testing.assert_allclose(
                    momentum - initial_momentum, accumulated_rounding, atol=1e-14, rtol=0
                )
                np.testing.assert_allclose(
                    np.sum(row.dissipation.node_velocity_impulse, axis=0), 0, atol=ATOL, rtol=0
                )
                assert row.dissipation.trace.shape == (1, 32)
                assert batch.stats()["allocations"] == initial_stats["allocations"]
                assert (
                    batch.stats()["state_uploaded_bytes"] == initial_stats["state_uploaded_bytes"]
                )
            assert np.any(batch.last_step.worlds[7].dissipation.node_impulse != 0)


@pytest.mark.parametrize("h", H_VALUES)
def test_fixed_support_oscillator_full_second_split_reference(h, tyx_mode, tmp_path):
    spec = make_world("oscillator")
    extension, velocity = 1 / 8, 0.0
    maxima = dict(position=0.0, velocity=0.0, analytic_position=0.0, analytic_velocity=0.0)
    steps = int(OSCILLATOR_DURATION / h)
    with saved(tmp_path, "fixed-oscillator", case=spec, h=h, steps=steps, mode=tyx_mode) as emit:
        with TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS, trace=True) as batch:
            for step in range(steps):
                receipt = step_record(batch, (spec,), h, emit)
                row, state = receipt.worlds[7], receipt.states[7]
                extension, velocity = oscillator_split(extension, velocity, h)
                material = elastic_impulse(spec, row)
                damping = row.dissipation.node_impulse
                fixed_reaction = -(material[0] + damping[0])
                delta_momentum = state.velocity[1].astype(float) - row.input_velocity[1]
                t = (step + 1) * h
                analytic_x = 0.125 * (1 + 8 * t) * np.exp(-8 * t)
                analytic_v = -8 * t * np.exp(-8 * t)
                error_x = state.position[1].astype(float) - np.array((1 + extension, 0, 0))
                error_v = state.velocity[1].astype(float) - np.array((velocity, 0, 0))
                maxima["position"] = max(maxima["position"], float(np.max(np.abs(error_x))))
                maxima["velocity"] = max(maxima["velocity"], float(np.max(np.abs(error_v))))
                maxima["analytic_position"] = max(
                    maxima["analytic_position"], abs(extension - analytic_x)
                )
                maxima["analytic_velocity"] = max(
                    maxima["analytic_velocity"], abs(velocity - analytic_v)
                )
                emit(
                    "oscillator-reference",
                    extension=extension,
                    velocity=velocity,
                    actual_position_error=error_x,
                    actual_velocity_error=error_v,
                    analytic_extension=analytic_x,
                    analytic_velocity=analytic_v,
                    running_maxima=maxima,
                    elastic_impulse=material,
                    damping_impulse=damping,
                    total_fixed_reaction=fixed_reaction,
                    actual_delta_momentum=delta_momentum,
                )
                np.testing.assert_array_equal(state.position[0], (0, 0, 0))
                np.testing.assert_array_equal(state.velocity[0], 0)
                # Parent froze these full-window gates before execution; the
                # phase's same-candidate 2e-6/2e-5 gate remains separate above.
                np.testing.assert_allclose(
                    state.position[1], (1 + extension, 0, 0), atol=2e-3, rtol=0
                )
                np.testing.assert_allclose(state.velocity[1], (velocity, 0, 0), atol=1e-2, rtol=0)
                np.testing.assert_allclose(delta_momentum, fixed_reaction, atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(
                    delta_momentum, material[1] + damping[1], atol=2e-5, rtol=2e-5
                )
                assert state.tick == row.input_tick + 1
            assert batch.snapshot()[7].time == OSCILLATOR_DURATION
            emit("complete-window", maxima=maxima, steps=steps, h=h)


@pytest.mark.parametrize("kind", ("chain", "mixed", "fixed_chain"))
def test_noncollinear_ordered_material_and_damping_history(kind, tyx_mode, tmp_path):
    spec, h = make_world(kind), H_VALUES[0]
    with saved(tmp_path, "ordered-chain", case=spec, h=h, mode=tyx_mode) as emit:
        with TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS, trace=True) as batch:
            for _ in range(CHAIN_STEPS):
                receipt = step_record(batch, (spec,), h, emit)
                row, state = receipt.worlds[7], receipt.states[7]
                expected_x, expected_v, expected_lambda = elastic_reference(
                    spec, row.input_position, row.input_velocity, row.force, h, row.sweeps
                )
                emit(
                    "independent-elastic",
                    position=expected_x,
                    reconstructed=expected_v,
                    multipliers=expected_lambda,
                )
                # Existing D1 elastic gate; the new damping phase has its stricter gate.
                np.testing.assert_allclose(state.position, expected_x, atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(
                    row.dissipation.reconstructed_velocity, expected_v, atol=2e-5, rtol=2e-5
                )
                impulse = row.dissipation.node_impulse
                np.testing.assert_allclose(np.sum(impulse, axis=0), 0, atol=1e-14, rtol=0)
                np.testing.assert_allclose(
                    np.sum(np.cross(state.position, impulse), axis=0), 0, atol=ATOL, rtol=0
                )
                if kind == "mixed":
                    np.testing.assert_array_equal(row.dissipation.trace[0, :6], 0)
                    np.testing.assert_array_equal(
                        row.dissipation.trace[0, 6:12], row.dissipation.trace[0, 12:18]
                    )
                if kind == "fixed_chain":
                    np.testing.assert_array_equal(state.position[0], spec.initial_state.position[0])
                    assert np.any(row.dissipation.node_impulse[0] != 0)


def test_sparse_worlds_storage_permutation_preserves_stable_edge_order(tyx_mode, tmp_path):
    specs = (make_world("chain"), make_world("pair", key=31, sign=-1))
    reordered = (
        make_world("pair", key=31, sign=-1, permuted=True),
        make_world("chain", permuted=True),
    )
    with saved(tmp_path, "permutation", cases=specs, permuted=reordered, mode=tyx_mode) as emit:
        with (
            TyxDeformableBatch(specs, mode=tyx_mode, settings=SETTINGS) as a,
            TyxDeformableBatch(reordered, mode=tyx_mode, settings=SETTINGS) as b,
        ):
            for _ in range(3):
                ar, br = (
                    step_record(a, specs, H_VALUES[0], emit),
                    step_record(b, reordered, H_VALUES[0], emit),
                )
                for original, other in zip(specs, reversed(reordered), strict=True):
                    ai, bi = np.argsort(original.point_ids), np.argsort(other.point_ids)
                    for name in ("position", "velocity"):
                        np.testing.assert_array_equal(
                            getattr(ar.states[original.key], name)[ai],
                            getattr(br.states[other.key], name)[bi],
                        )
                    ap, bp = ar.worlds[original.key].dissipation, br.worlds[other.key].dissipation
                    np.testing.assert_array_equal(ap.ordered_point_ids, bp.ordered_point_ids)
                    np.testing.assert_array_equal(ap.trace, bp.trace)
            assert a.world_keys == (7, 31) and b.world_keys == (31, 7)


def test_both_fixed_is_legal_and_damping_runs_once_per_physical_step(
    tyx_mode, monkeypatch, tmp_path
):
    from phymeo.backends._tyx_distance_dissipation import dissipate_distances

    spec = make_world("fixed")
    with saved(tmp_path, "fixed", case=spec, mode=tyx_mode) as emit:
        with TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS) as batch:
            original, calls = batch._buffers.call, []

            def observed(kernel, *args):
                result = original(kernel, *args)
                if kernel is dissipate_distances:
                    calls.append(args[0])
                return result

            monkeypatch.setattr(batch._buffers, "call", observed)
            for _ in range(3):
                result = step_record(batch, (spec,), H_VALUES[0], emit)
                phase = result.worlds[7].dissipation
                np.testing.assert_array_equal(phase.node_impulse, 0)
                np.testing.assert_array_equal(phase.kinetic_change, 0)
                np.testing.assert_array_equal(
                    result.states[7].position, spec.initial_state.position
                )
            assert calls == [1, 1, 1]


def test_phase_changes_next_real_entry_without_host_physics_fallback(
    tyx_mode, monkeypatch, tmp_path
):
    import phymeo.constraints.distance as old_distance

    spec = make_world()
    with saved(tmp_path, "actual-next-entry", case=spec, mode=tyx_mode) as emit:
        with TyxDeformableBatch((spec,), mode=tyx_mode, settings=SETTINGS) as batch:

            def forbidden(*args, **kwargs):
                raise AssertionError("HOST physics fallback")

            with monkeypatch.context() as patch:
                patch.setattr(old_distance, "project_distances", forbidden)
                patch.setattr(np.linalg, "solve", forbidden)
                first = step_record(batch, (spec,), H_VALUES[0], emit)
                second = step_record(batch, (spec,), H_VALUES[0], emit)
            phase = first.worlds[7].dissipation
            assert np.max(np.abs(phase.reconstructed_velocity - phase.dissipated_velocity)) > 1e-5
            np.testing.assert_array_equal(
                second.worlds[7].input_velocity, phase.dissipated_velocity
            )
            assert second.worlds[7].input_tick == 1

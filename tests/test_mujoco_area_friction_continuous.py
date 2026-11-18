"""B3: a preregistered twenty-step live slip/release window, not a benchmark.

The geometry/material/COM are inherited from I3v; the two velocities below are
new fixed declarations. Physics runs only through Scene.step. All full accepted
records are retained before assertions, including a failed run's accepted prefix.
No claim of exact static Coulomb friction, long-term stability, or training.
"""

import gzip
import json
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest
from test_mujoco_area_friction import H, _close, _declarations, _equations, _native, _plain
from test_mujoco_area_friction_live import _published, _surface

from phymeo.backends.area_friction_law import AreaFrictionPairLaw
from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.backends.tyx_area_friction import TyxAreaFrictionWorkspace

STEPS = 20
KEYS = (7, 73)
SLIDING = (0.12, -0.04, -0.005, 0.1, 0.2, -0.1)
RELEASING = (0.02, -0.01, 1.25, 0.0, 0.0, 0.0)


def _cases():
    old, worlds, source_assets = _declarations()
    asset = source_assets[old[0].key][0]
    model = replace(
        old[0],
        key="continuous-slip-release",
        area_friction_laws=(AreaFrictionPairLaw(asset.identity, 0.5, 0.001),),
    )
    states = tuple(
        MuJoCoTyxWorldSpec(key, model.key, MuJoCoInitialState(worlds[0].initial_state.q, velocity))
        for key, velocity in zip(KEYS, (SLIDING, RELEASING), strict=True)
    )
    return (model,), states, {model.key: (asset,)}


def _audit(spec, before, after, geometries, audit):
    """Independent native entry plus algebraic force substitution; no contact solve."""
    import mujoco as mj

    receipt = after.last_step
    rows, solver, problem = _published(receipt)
    audit["geometry"] = tuple(
        _surface(row, geometry, receipt) for row, geometry in zip(rows, geometries, strict=True)
    )
    assert receipt.world == after.world == before.world
    assert receipt.tick == after.tick == before.tick + 1
    assert receipt.input_time == before.time == receipt.contact_time == receipt.dynamics.time
    assert receipt.output_time == after.time > before.time
    assert receipt.h == H and problem.h_s == float(np.float32(H))
    assert receipt.contact_phase == "tyx-euler-input-constraint-evaluation"
    assert receipt.dynamics.phase == "tyx-euler-input-dynamics"
    np.testing.assert_array_equal(receipt.input_q, before.q)
    np.testing.assert_array_equal(receipt.input_v, before.v)
    np.testing.assert_array_equal(receipt.dynamics.input_q, before.q)
    np.testing.assert_array_equal(receipt.dynamics.input_v, before.v)
    np.testing.assert_array_equal(solver["output_velocity"], after.v)
    quads = tuple(SimpleNamespace(identity=row["identity"], points=row["points"]) for row in rows)
    native = audit["native"] = _native(spec, receipt, quads)
    dyn, mass = receipt.dynamics, np.asarray(problem.mass_matrix, dtype=float)
    for left, right, absolute in (
        (slice(None, 3), slice(None, 3), 2e-6),
        (slice(3, None), slice(3, None), 2e-9),
        (slice(None, 3), slice(3, None), 2e-8),
        (slice(3, None), slice(None, 3), 2e-8),
    ):
        _close(mass[left, right], native["mass"][left, right], absolute, 3e-4)
    assert np.max(np.abs(native["mass"][:3, 3:])) > 1e-4
    for name in ("bias", "passive", "applied"):
        _close(getattr(dyn, name), native[name], 2e-5, 3e-4)
    _close(dyn.body_com, native["body_com"], 2e-6, 2e-5)
    _close(dyn.jacobian_com, native["com_j"], 2e-6, 2e-5)
    np.testing.assert_array_equal(mass, dyn.mass_matrix)
    np.testing.assert_array_equal(problem.damping, native["damping"].astype(np.float32))
    np.testing.assert_array_equal(problem.input_velocity, before.v)
    expected_f = dyn.applied + dyn.passive - dyn.bias + problem.damping * before.v
    bound = (
        8
        * np.finfo(np.float32).eps
        * (
            np.abs(dyn.applied)
            + np.abs(dyn.passive)
            + np.abs(dyn.bias)
            + np.abs(problem.damping * before.v)
        )
        + np.finfo(np.float32).tiny
    )
    assert np.all(np.abs(problem.applied_force - expected_f) <= bound)
    for index, sample in enumerate(problem.samples):
        _close(sample.jacobian_a, native["ja"][index], 2e-6, 2e-5)
        _close(sample.jacobian_b, native["jb"][index], 2e-6, 2e-5)
    velocity = after.v.astype(float)
    e = audit["equations"] = _equations(problem, velocity)
    assert e["residual_norm"] <= 2e-5 * e["scale"]
    _close(velocity, solver["solver_velocity"], 5e-5, 5e-5)
    total = np.zeros(6)
    normal_power = tangent_power = normal_load = 0.0
    force_sum, moment_sum = np.zeros(3), np.zeros(3)
    max_slip, offset = 0.0, 0
    for row in rows:
        count = len(row["sample_ids"])
        section = slice(offset, offset + count)
        _close(row["normal_force"], e["normal"][section], 0.002, 0.002)
        _close(row["tangential_force"], e["tangent"][section], 0.002, 0.002)
        _close(row["force_on_a"], e["force"][section], 0.002, 0.002)
        _close(row["torque_on_a_world_origin"], e["moment"][section], 1e-5, 0.002)
        np.testing.assert_array_equal(row["force_on_b"], -row["force_on_a"])
        np.testing.assert_array_equal(
            row["torque_on_b_world_origin"], -row["torque_on_a_world_origin"]
        )
        for index, sample in enumerate(problem.samples[section]):
            g = np.asarray(sample.jacobian_a, float) - np.asarray(sample.jacobian_b, float)
            relative = g @ velocity
            _close(row["relative_velocity"][index], relative, 5e-5, 5e-5)
            n = np.asarray(sample.normal_world, float)
            tangent = relative - n * (n @ relative) / (n @ n)
            fn = float(row["normal_force"][index])
            ft = row["tangential_force"][index].astype(float)
            force = row["force_on_a"][index].astype(float)
            assert fn >= 0
            total += g.T @ force
            normal_power += fn * float(n @ relative)
            tangent_power += float(ft @ tangent)
            normal_load += fn
            force_sum += force
            moment_sum += row["torque_on_a_world_origin"][index]
            max_slip = max(max_slip, float(np.linalg.norm(tangent)))
        offset += count
    actual_q = np.asarray(solver["generalized_force"], float)
    _close(actual_q, total, 2e-5, 2e-5)
    np.testing.assert_array_equal(receipt.qfrc_constraint, actual_q)
    stored_residual = e["response"] @ velocity - e["rhs"] - problem.h_s * actual_q
    stored_norm = np.sqrt(
        max(0.0, float(stored_residual @ np.linalg.solve(e["response"], stored_residual)))
    )
    assert stored_norm <= 2e-5 * e["scale"]
    _close(solver["residual"], stored_residual, 1e-12, 1e-10)
    assert solver["tolerance"] == 2e-5
    assert abs(solver["normalized_residual"] - stored_norm / e["scale"]) <= 1e-12
    rhs = dyn.applied + dyn.passive - dyn.bias + actual_q
    physical, integration = np.linalg.solve(mass, rhs), np.linalg.solve(e["response"], rhs)
    h = problem.h_s
    _close(h * receipt.qacc, h * physical, 5e-5, 5e-5)
    _close(h * receipt.integration_acceleration, h * integration, 5e-5, 5e-5)
    _close(velocity, before.v + h * integration, 5e-5, 5e-5)
    np.testing.assert_allclose(
        velocity, before.v + h * receipt.integration_acceleration, rtol=2e-6, atol=2e-7
    )
    model = spec.compilation.fresh_native_model()
    q = before.q.copy()
    mj.mj_integratePos(model, q, velocity, h)
    np.testing.assert_allclose(after.q[:3], q[:3], rtol=0, atol=2e-7)
    sign = 1 if after.q[3:] @ q[3:] >= 0 else -1
    np.testing.assert_allclose(sign * after.q[3:], q[3:], rtol=0, atol=3e-7)
    assert abs(float(after.q[3:] @ after.q[3:]) - 1) <= 2e-6
    e.update(
        stored_force_residual=stored_residual,
        stored_force_norm=stored_norm,
        strict_host_gate=e["residual_norm"] <= 1e-10 * e["scale"],
        physical_acceleration=physical,
        integration_acceleration=integration,
        integrated_q=q,
    )
    audit["loads"] = dict(
        samples=offset,
        normal_force_n=normal_load,
        force_on_a_world=force_sum,
        moment_on_a_world_origin=moment_sum,
        max_tangent_speed_m_s=max_slip,
        normal_power_w=normal_power,
        tangent_power_w=tangent_power,
        normal_work_j=h * normal_power,
        tangent_work_j=h * tangent_power,
        impulse_world_ns=h * force_sum,
    )
    return audit


@pytest.mark.cuda
def test_twenty_live_steps_slide_and_unload_share_one_bucket(tmp_path, monkeypatch):
    import mujoco as mj

    def forbidden(*args, **kwargs):
        raise AssertionError("continuous live case forbids CPU stepping or HOST-packed solve_many")

    monkeypatch.setattr(mj, "mj_step", forbidden)
    monkeypatch.setattr(TyxAreaFrictionWorkspace, "solve_many", forbidden)
    evidence = dict(
        schema="mujoco-area-friction-continuous-test-v1",
        status="IN_PROGRESS",
        mode="cuda",
        h_s=H,
        steps_requested=STEPS,
        world_order=KEYS,
        sliding_velocity=SLIDING,
        releasing_velocity=RELEASING,
        records=[],
        accepted_steps=0,
        timing_scope="public Scene.step includes HOST geometry, device solve/integration and publication; no subphase attribution",
        observation_scope="public integration_snapshot plus accepted receipts/geometry; not the old device-area ABI",
        cache_scope="fresh runtime owner; existing compiler cache unchanged; first step not guaranteed cold compilation",
        qualification="new fixed small-slip/release window; not static Coulomb, training or throughput benchmark",
    )
    import phymeo.backends.mujoco_area_friction_live as live

    original_publish = live.publish

    def diagnosed_publish(runtime, result, index):
        try:
            return original_publish(runtime, result, index)
        except Exception:
            evidence["rejected_candidate"] = dict(
                world=runtime.world_keys[index], index=index, result=result
            )
            raise

    monkeypatch.setattr(live, "publish", diagnosed_publish)
    start_total, phase = perf_counter(), "declarations"
    batch = None
    try:
        started = perf_counter()
        specs, worlds, assets = _cases()
        evidence.update(
            declarations_s=perf_counter() - started,
            worlds=worlds,
            assets=assets,
            models={
                spec.key: dict(
                    source=spec.compilation.source,
                    area_friction_laws=spec.area_friction_laws,
                    area_pressure_pairs=spec.area_pressure_pairs,
                    contact_capacity=spec.contact_capacity,
                    constraint_capacity=spec.constraint_capacity,
                )
                for spec in specs
            },
        )
        phase, started = "initialization", perf_counter()
        with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as batch:
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
            evidence["initialization_s"] = perf_counter() - started
            evidence.update(
                initial=batch.integration_snapshot(),
                layout=batch.layout,
                resources_initial=batch.resource_snapshot(),
                counters_initial=batch.counters,
            )
            assert len({batch.layout[key].bucket for key in KEYS}) == 1
            assert batch.layout[KEYS[0]].row != batch.layout[KEYS[1]].row
            before = evidence["initial"]
            for step in range(STEPS):
                revision = batch.checkpoint_revision
                phase, started = "scene-step", perf_counter()
                accepted = scene.step(H, expected_revision=revision)
                step_s = perf_counter() - started
                # Batch publication already owns complete receipt/state evidence.
                # Append it before observation and oracle checks can fail.
                record = dict(
                    step=step,
                    before=before,
                    accepted=accepted,
                    publication=batch.last_step,
                    scene_step_s=step_s,
                    observation_complete=False,
                    audits={},
                )
                evidence["records"].append(record)
                evidence["accepted_steps"] += 1
                phase, started = "observation", perf_counter()
                record["geometry"] = scene.last_geometry
                after = record["after"] = batch.integration_snapshot()
                record["observation_s"] = perf_counter() - started
                record["observation_complete"] = True
                record["pipeline_s"] = step_s + record["observation_s"]
                phase, started = "independent-audit", perf_counter()
                assert all(accepted.values()) and batch.checkpoint_revision == revision + 1
                assert tuple(after) == KEYS
                for key in KEYS:
                    audit = record["audits"][key] = {}
                    _audit(specs[0], before[key], after[key], record["geometry"][key], audit)
                    assert after[key].tick == step + 1
                    assert abs(after[key].time - (step + 1) * float(np.float32(H))) <= 1e-8
                    if key == KEYS[1]:
                        samples = audit["loads"]["samples"]
                        assert samples > 0 if step == 0 else samples == 0
                        assert audit["loads"]["normal_force_n"] == 0
                        np.testing.assert_array_equal(
                            audit["loads"]["force_on_a_world"], np.zeros(3)
                        )
                        assert after[key].q[2] > before[key].q[2]
                assert not np.array_equal(after[KEYS[0]].q, after[KEYS[1]].q)
                record["audit_s"] = perf_counter() - started
                record["iteration_s"] = record["pipeline_s"] + record["audit_s"]
                before = after
            phase = "window-audit"
            slip = [record["audits"][KEYS[0]]["loads"] for record in evidence["records"]]
            support_steps = sum(row["normal_force_n"] > 0 for row in slip)
            slipping_steps = sum(
                row["normal_force_n"] > 0 and row["max_tangent_speed_m_s"] > 0.001 for row in slip
            )
            assert support_steps == STEPS and slipping_steps >= 10
            assert sum(row["tangent_work_j"] for row in slip) < -1e-8
            displacement = before[KEYS[0]].q[:3] - evidence["initial"][KEYS[0]].q[:3]
            assert np.linalg.norm(displacement[:2]) > 1e-4
            assert np.linalg.norm(displacement[:2]) < 0.01
            evidence["window"] = dict(
                support_steps=support_steps,
                slipping_steps=slipping_steps,
                slip_displacement_world_m=displacement,
                normal_work_j=sum(row["normal_work_j"] for row in slip),
                tangent_work_j=sum(row["tangent_work_j"] for row in slip),
                impulse_world_ns=sum((row["impulse_world_ns"] for row in slip), np.zeros(3)),
            )
            evidence.update(
                final=before,
                resources_final=batch.resource_snapshot(),
                counters_final=batch.counters,
            )
            phase = "close"
        evidence["resources_closed"] = batch.resource_snapshot()
        evidence["timing"] = dict(
            first_step_s=evidence["records"][0]["scene_step_s"],
            later_step_s=tuple(row["scene_step_s"] for row in evidence["records"][1:]),
            public_pipeline_total_s=sum(row["pipeline_s"] for row in evidence["records"]),
            observation_total_s=sum(row["observation_s"] for row in evidence["records"]),
            independent_audit_total_s=sum(row["audit_s"] for row in evidence["records"]),
            total_before_serialization_s=perf_counter() - start_total,
        )
        evidence["status"] = "PASS"
    except BaseException as error:
        evidence.update(
            status="FAILED_PREFIX",
            failure_phase=phase,
            error_type=type(error).__name__,
            error=str(error),
        )
        if batch is not None:
            try:
                evidence["resources_at_failure"] = batch.resource_snapshot()
            except BaseException as resource_error:
                evidence["resource_observation_error"] = repr(resource_error)
        raise
    finally:
        evidence["wall_before_serialization_s"] = perf_counter() - start_total
        with gzip.open(
            tmp_path / "live-continuous-slip-release.json.gz", "xt", encoding="utf-8"
        ) as stream:
            json.dump(_plain(evidence), stream, allow_nan=False)

"""B1: current Scene geometry -> resident friction -> accepted native motion.

Only the production scene advances physics. Native MuJoCo supplies independent
entry M/J/bias and a position-integration oracle, never a physical step. The
two fixed windows retain all accepted evidence before their assertions.
"""

import gzip
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from test_mujoco_area_friction import H, _close, _declarations, _equations, _native, _owned, _plain

from phymeo.backends.mujoco_area_friction import AreaFrictionPairLaw
from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.backends.tyx_area_friction import TyxAreaFrictionWorkspace
from phymeo.solvers.area_friction import LAW

KEYS = (7, 73, 305, 41)
UPWARD = (0.02, -0.01, 1.25, 0.0, 0.0, 0.0)


def _cases(*, unloading=False):
    original, worlds, source_assets = _declarations()
    asset = source_assets[original[0].key][0]
    declarations = (
        ("live-mu5", original[0], 0.5),
        ("live-mu0", original[0], 0.0),
        ("normal-only", original[0], None),
        ("live-empty-D40", original[1], 0.5),
    )
    specs = tuple(
        replace(
            spec,
            key=name,
            area_friction_laws=()
            if mu is None
            else (AreaFrictionPairLaw(asset.identity, mu, 0.001),),
        )
        for name, spec, mu in declarations
    )
    states = (worlds[0].initial_state,) * 3 + (worlds[1].initial_state,)
    batch_worlds = tuple(
        MuJoCoTyxWorldSpec(key, spec.key, state)
        for key, spec, state in zip(KEYS, specs, states, strict=True)
    )
    if unloading:
        specs = specs[:1]
        batch_worlds = (
            replace(
                batch_worlds[0], initial_state=MuJoCoInitialState(worlds[0].initial_state.q, UPWARD)
            ),
        )
    return specs, batch_worlds, {spec.key: (asset,) for spec in specs}


def _published(receipt):
    """The sole adapter for the new public, law-aware receipt schema."""
    fields = receipt.raw_contact_fields
    assert "area_pressure" not in fields
    assert receipt.contacts == () and receipt.constraints == ()
    rows, solver = fields["area_friction"], fields["area_friction_solver"]
    assert solver["schema"] == "mujoco-tyx-area-friction-live-v1"
    assert solver["law"] == LAW and solver["precision"] == "actual-packed-point-fp32"
    assert solver["solver_status"] == 0
    samples = []
    for row in rows:
        assert row["law"] == LAW
        for i in range(len(row["sample_ids"])):
            samples.append(
                SimpleNamespace(
                    point_world=row["points"][i],
                    normal_world=row["normals"][i],
                    jacobian_a=row["jacobian_a"][i],
                    jacobian_b=row["jacobian_b"][i],
                    stiffness_n_m=row["stiffness"][i],
                    phi_m=row["surrogate_gap"][i],
                    mu=row["mu"],
                    slip_speed_m_s=row["slip_speed_m_s"],
                )
            )
    problem = SimpleNamespace(
        **{
            name: solver[name]
            for name in ("mass_matrix", "damping", "input_velocity", "applied_force", "h_s")
        },
        samples=tuple(samples),
    )
    _owned(rows)
    _owned(solver)
    return rows, solver, problem


def _surface(row, intersection, receipt):
    surface = intersection.surface
    assert row["identity"] == surface.identity and row["stamp"] == surface.stamp
    assert surface.stamp.step_index == receipt.tick - 1
    assert surface.stamp.sample_time_s == receipt.input_time
    assert surface.stamp.world_id == str(receipt.world)
    ids, sources, points, normals, areas, pressures, gradients = [], [], [], [], [], [], []
    bary = np.array(((2 / 3, 1 / 6, 1 / 6), (1 / 6, 2 / 3, 1 / 6), (1 / 6, 1 / 6, 2 / 3)))
    for triangle in surface.triangles:
        vertices = triangle.vertices
        cross = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        magnitude = float(np.linalg.norm(cross))
        normal = cross / magnitude
        gradient = float(normal @ triangle.pressure_gradient)
        assert magnitude > 0 and gradient > 0
        ids.extend((triangle.triangle_id, slot) for slot in range(3))
        sources.extend(((triangle.compliant_cell_id, triangle.rigid_face_id),) * 3)
        points.extend(bary @ vertices)
        normals.extend((normal,) * 3)
        areas.extend((magnitude / 6,) * 3)
        pressures.extend(bary @ triangle.vertex_pressure)
        gradients.extend((gradient,) * 3)
    n = len(ids)
    assert row["sample_ids"] == tuple(ids) and row["source_ids"] == tuple(sources)
    points, normals = np.asarray(points).reshape(n, 3), np.asarray(normals).reshape(n, 3)
    areas, pressures, gradients = map(np.asarray, (areas, pressures, gradients))
    np.testing.assert_allclose(row["dA"], areas, rtol=1e-12, atol=1e-18)
    _close(row["points"], points.astype(np.float32), 2e-9, 2e-7)
    _close(row["normals"], normals.astype(np.float32), 2e-7, 2e-7)
    _close(row["stiffness"], (areas * gradients).astype(np.float32), 2e-5, 3e-4)
    _close(row["surrogate_gap"], (-pressures / gradients).astype(np.float32), 2e-8, 3e-4)
    return dict(
        area_m2=float(areas.sum()),
        samples=n,
        source_points=points,
        source_pressure=pressures,
        source_gradient=gradients,
    )


def _entry(spec, receipt, before, after, rows, solver, problem):
    np.testing.assert_array_equal(receipt.input_q, before.q)
    np.testing.assert_array_equal(receipt.input_v, before.v)
    assert receipt.input_time == before.time and after.time == receipt.output_time
    assert receipt.contact_time == receipt.input_time == receipt.dynamics.time
    assert receipt.contact_phase == "tyx-euler-input-constraint-evaluation"
    assert receipt.dynamics.phase == "tyx-euler-input-dynamics"
    np.testing.assert_array_equal(receipt.dynamics.input_q, before.q)
    np.testing.assert_array_equal(receipt.dynamics.input_v, before.v)
    assert receipt.tick == after.tick == before.tick + 1
    assert after.last_step is receipt
    quads = tuple(SimpleNamespace(identity=row["identity"], points=row["points"]) for row in rows)
    native = _native(spec, receipt, quads)
    mass, expected = problem.mass_matrix, native["mass"]
    if len(before.v) == 6:
        _close(mass[:3, :3], expected[:3, :3], 2e-6, 3e-4)
        _close(mass[3:, 3:], expected[3:, 3:], 2e-9, 3e-4)
        _close(mass[:3, 3:], expected[:3, 3:], 2e-8, 3e-4)
        _close(mass[3:, :3], expected[3:, :3], 2e-8, 3e-4)
        assert np.max(np.abs(expected[:3, 3:])) > 1e-4
    else:
        _close(mass, expected, 2e-6, 3e-4)
    dyn = receipt.dynamics
    for name in ("bias", "passive", "applied"):
        _close(getattr(dyn, name), native[name], 2e-5, 3e-4)
    _close(dyn.jacobian_com, native["com_j"], 2e-6, 2e-5)
    _close(dyn.body_com, native["body_com"], 2e-6, 2e-5)
    np.testing.assert_array_equal(mass, dyn.mass_matrix.astype(np.float32))
    np.testing.assert_array_equal(problem.damping, native["damping"].astype(np.float32))
    np.testing.assert_array_equal(problem.input_velocity, before.v)
    np.testing.assert_array_equal(solver["output_velocity"], after.v)
    assert problem.h_s == float(np.float32(H))
    f = (
        (dyn.applied.astype(np.float32) + dyn.passive.astype(np.float32))
        - dyn.bias.astype(np.float32)
    ) + problem.damping.astype(np.float32) * before.v.astype(np.float32)
    # CUDA may contract the cancellation into FMA; bind within its FP32
    # arithmetic envelope while all physical residuals use the actual input.
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
    assert np.all(np.abs(problem.applied_force - f) <= bound)
    for i, sample in enumerate(problem.samples):
        _close(sample.jacobian_a, native["ja"][i], 2e-6, 2e-5)
        _close(sample.jacobian_b, native["jb"][i], 2e-6, 2e-5)
    return native


def _law_and_motion(spec, receipt, after, rows, solver, problem):
    import mujoco as mj

    velocity = after.v.astype(float)
    e = _equations(problem, velocity)
    assert e["residual_norm"] <= 2e-5 * e["scale"]
    assert solver["normalized_residual"] <= solver["tolerance"]
    _close(solver["solver_velocity"], velocity, 5e-5, 5e-5)
    total = np.zeros(len(velocity))
    offset = 0
    for row in rows:
        n = len(row["sample_ids"])
        section = slice(offset, offset + n)
        _close(row["normal_force"], e["normal"][section], 0.002, 0.002)
        _close(row["tangential_force"], e["tangent"][section], 0.002, 0.002)
        _close(row["force_on_a"], e["force"][section], 0.002, 0.002)
        _close(row["torque_on_a_world_origin"], e["moment"][section], 1e-5, 0.002)
        np.testing.assert_array_equal(row["force_on_b"], -row["force_on_a"])
        np.testing.assert_array_equal(
            row["torque_on_b_world_origin"], -row["torque_on_a_world_origin"]
        )
        for i, sample in enumerate(problem.samples[section]):
            g = sample.jacobian_a.astype(float) - sample.jacobian_b.astype(float)
            relative = g @ velocity
            _close(row["relative_velocity"][i], relative, 2e-5, 2e-5)
            normal = sample.normal_world.astype(float)
            tangent = relative - normal * (normal @ relative) / (normal @ normal)
            fn, ft, force = row["normal_force"][i], row["tangential_force"][i], row["force_on_a"][i]
            assert (
                fn >= 0 and np.linalg.norm(ft.astype(float)) <= float(sample.mu) * float(fn) + 1e-5
            )
            assert float(ft.astype(float) @ tangent) <= 1e-14
            total += g.T @ force.astype(float)
        offset += n
    actual_q = np.asarray(solver["generalized_force"], dtype=float)
    _close(actual_q, total, 2e-5, 2e-5)
    np.testing.assert_array_equal(receipt.qfrc_constraint, solver["generalized_force"])
    stored_residual = e["response"] @ velocity - e["rhs"] - problem.h_s * total
    stored_norm = np.sqrt(
        max(0.0, float(stored_residual @ np.linalg.solve(e["response"], stored_residual)))
    )
    assert stored_norm <= 2e-5 * e["scale"]
    dyn, h = receipt.dynamics, problem.h_s
    rhs = dyn.applied + dyn.passive - dyn.bias + actual_q
    physical = np.linalg.solve(problem.mass_matrix.astype(float), rhs)
    integration = np.linalg.solve(e["response"], rhs)
    _close(h * receipt.qacc, h * physical, 5e-5, 5e-5)
    _close(h * receipt.integration_acceleration, h * integration, 5e-5, 5e-5)
    _close(velocity, receipt.input_v + h * integration, 5e-5, 5e-5)
    np.testing.assert_allclose(
        velocity, receipt.input_v + h * receipt.integration_acceleration, rtol=2e-6, atol=2e-7
    )
    model = spec.compilation.fresh_native_model()
    for row in rows:
        identity = row["identity"]
        expected_bodies = tuple(
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            for name in (identity.compliant_body, identity.rigid_body)
        )
        assert tuple(row["body_ids"]) == expected_bodies
    q = receipt.input_q.copy()
    mj.mj_integratePos(model, q, velocity, h)
    if model.nv == 6:
        np.testing.assert_allclose(after.q[:3], q[:3], rtol=0, atol=2e-7)
        sign = 1 if after.q[3:] @ q[3:] >= 0 else -1
        np.testing.assert_allclose(sign * after.q[3:], q[3:], rtol=0, atol=3e-7)
        assert abs(float(after.q[3:] @ after.q[3:]) - 1) <= 2e-6
    else:
        np.testing.assert_allclose(after.q, q, rtol=0, atol=2e-7)
        m, d, v0 = (
            float(problem.mass_matrix[0, 0]),
            float(problem.damping[0]),
            float(receipt.input_v[0]),
        )
        correct, double = m * v0 / (m + h * d), (m - h * d) * v0 / (m + h * d)
        _close(after.v, np.array([correct]), 5e-5, 5e-5)
        assert abs(float(after.v[0]) - double) > 0.001
        e["damping_oracle"] = dict(correct=correct, repeated=double)
    e.update(
        stored_force_residual=stored_residual,
        stored_force_norm=stored_norm,
        strict_host_gate=e["residual_norm"] <= 1e-10 * e["scale"],
        physical_acceleration=physical,
        integration_acceleration=integration,
        integrated_q=q,
    )
    return e


def _run(tmp_path, monkeypatch, *, unloading):
    import mujoco as mj

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "live Scene must not CPU step or use the independent HOST-packed workspace"
        )

    monkeypatch.setattr(mj, "mj_step", forbidden)
    monkeypatch.setattr(TyxAreaFrictionWorkspace, "solve_many", forbidden)
    evidence = dict(schema="mujoco-area-friction-live-test-v1", unloading=unloading, records=[])
    try:
        specs, worlds, assets = _cases(unloading=unloading)
        by_key = {world.world: spec for world, spec in zip(worlds, specs, strict=True)}
        evidence.update(
            models={
                spec.key: dict(
                    source=spec.compilation.source,
                    contact_capacity=spec.contact_capacity,
                    constraint_capacity=spec.constraint_capacity,
                    area_pressure_pairs=spec.area_pressure_pairs,
                    area_friction_laws=spec.area_friction_laws,
                )
                for spec in specs
            },
            worlds=worlds,
            assets=assets,
        )
        with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as batch:
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
            evidence.update(initial=batch.integration_snapshot(), layout=batch.layout)
            if not unloading:
                assert len({batch.layout[key].bucket for key in KEYS}) == 4
            for step in range(2 if unloading else 3):
                before, revision = batch.integration_snapshot(), batch.checkpoint_revision
                accepted = scene.step(H, expected_revision=revision)
                after = batch.integration_snapshot()
                record = dict(
                    step=step,
                    before=before,
                    after=after,
                    receipt=batch.last_step,
                    geometry=scene.last_geometry,
                    accepted=accepted,
                    audits={},
                )
                evidence["records"].append(record)
                assert all(accepted.values()) and batch.checkpoint_revision == revision + 1
                for key, state in after.items():
                    if key == 305:
                        assert "area_friction" not in state.last_step.raw_contact_fields
                        assert "area_pressure" in state.last_step.raw_contact_fields
                        continue
                    receipt = state.last_step
                    rows, solver, problem = _published(receipt)
                    audit = record["audits"][key] = dict()
                    audit["geometry"] = tuple(
                        _surface(row, geometry, receipt)
                        for row, geometry in zip(rows, scene.last_geometry[key], strict=True)
                    )
                    audit["native"] = _entry(
                        by_key[key], receipt, before[key], state, rows, solver, problem
                    )
                    audit["equations"] = _law_and_motion(
                        by_key[key], receipt, state, rows, solver, problem
                    )
                    if unloading:
                        assert (
                            (len(problem.samples) > 0) if step == 0 else (len(problem.samples) == 0)
                        )
                        e = audit["equations"]
                        predicted = np.array([s.phi_m for s in problem.samples]) + problem.h_s * (
                            e["normal_rows"] @ e["free"]
                        )
                        assert np.all(predicted > 0)
                        audit["free_separation_gaps"] = predicted
                        for row in rows:
                            assert not np.any(row["normal_force"]) and not np.any(
                                row["tangential_force"]
                            )
                    elif key == 41:
                        assert not problem.samples
                if not unloading:
                    _close(after[73].v, after[305].v, 5e-5, 5e-5)
                    _close(after[73].q, after[305].q, 5e-5, 5e-5)
                    new = after[73].last_step.raw_contact_fields["area_friction"][0]
                    old = after[305].last_step.raw_contact_fields["area_pressure"][0]
                    _close(new["normal_force"], old["normal_force"], 0.002, 0.002)
                    assert not np.any(new["tangential_force"])
                    if step == 0:
                        sliding = after[7].last_step.raw_contact_fields["area_friction"][0]
                        assert np.linalg.norm(sliding["tangential_force"]) > 0.001
                        assert np.max(np.abs(after[7].v - after[73].v)) > 1e-7
                assert any(not np.array_equal(after[key].q, before[key].q) for key in after)
            evidence.update(status="PASS", final=batch.integration_snapshot())
    except BaseException as error:
        evidence.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        with gzip.open(
            tmp_path / ("live-unload.json.gz" if unloading else "live-four-worlds.json.gz"),
            "xt",
            encoding="utf-8",
        ) as stream:
            json.dump(_plain(evidence), stream, allow_nan=False)


@pytest.mark.cuda
def test_current_four_world_scene_publishes_friction_and_native_motion(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch, unloading=False)


@pytest.mark.cuda
def test_current_surface_unloads_then_becomes_valid_empty(tmp_path, monkeypatch):
    einsum = np.einsum

    def checked_einsum(subscripts, *operands, **kwargs):
        if subscripts == "sij,si->j" and operands[0].shape[0] == 0:
            pytest.fail("empty contact contributes exactly zero generalized force")
        return einsum(subscripts, *operands, **kwargs)

    monkeypatch.setattr(np, "einsum", checked_einsum)
    _run(tmp_path, monkeypatch, unloading=True)

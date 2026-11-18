"""Fixed live-friction windows; absent events are reported, never manufactured."""

import gzip
import json
from contextlib import ExitStack
from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import pytest
from test_mujoco_area_friction import _close, _equations, _native, _plain
from test_mujoco_area_friction_live import _published, _surface

from experiments.area_pressure.free_body_support import case_declarations as free_declarations
from experiments.area_pressure.three_body_release import case_declarations as dual_declarations
from phymeo.backends.mujoco_area_friction import AreaFrictionPairLaw
from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.backends.tyx_area_friction import TyxAreaFrictionWorkspace
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

H = 0.001
WINDOWS = ("center-drop", "offset-support", "dual-patch")


def _declarations(name):
    if name in ("center-drop", "offset-support"):
        specs, worlds, assets, manifest = free_declarations()
        world_key = 41 if name == "center-drop" else 305
        world = next(world for world in worlds if world.world == world_key)
        base = specs[0]
        laws = tuple(AreaFrictionPairLaw(asset.identity, 0.5, 0.001) for asset in assets[base.key])
        return (
            replace(base, area_friction_laws=laws),
            world,
            assets,
            150,
            dict(
                original=manifest,
                scope="original I3s full FREE geometry and state; new friction law only",
            ),
        )
    specs, _, assets, manifest = dual_declarations()
    base = specs[0]
    xml = base.compilation.source.primary_xml.decode("utf-8")
    marker = '<joint name="pressure-vertical"'
    assert xml.count(marker) == 1
    xml = xml.replace(
        marker,
        '<joint name="pressure-x" type="slide" axis="1 0 0" '
        'limited="false" damping="0" frictionloss="0" armature="0"/>\n' + marker,
    )
    compilation = compile_mujoco_xml(xml, require_asset=False)
    laws = tuple(AreaFrictionPairLaw(asset.identity, 0.5, 0.001) for asset in assets[base.key])
    spec = replace(base, compilation=compilation, area_friction_laws=laws)
    world = MuJoCoTyxWorldSpec(
        13, spec.key, MuJoCoInitialState((0.0, -0.003, 0.0, 0.0), (0.02, 0.0, 0.0, 0.0))
    )
    return (
        spec,
        world,
        assets,
        40,
        dict(
            original=manifest,
            actual_xml=xml,
            actual_q_order=("pressure-x", "pressure-vertical", "left-vertical", "right-vertical"),
            scope="one additional pressure X-slide; original three Z-slides/masses/areas/pressure/gravity/damping retained; guides carry excluded reactions",
        ),
    )


def _native_binding(spec, receipt, before, after, rows, problem):
    np.testing.assert_array_equal(receipt.input_q, before.q)
    np.testing.assert_array_equal(receipt.input_v, before.v)
    assert receipt.tick == after.tick == before.tick + 1
    assert receipt.input_time == before.time == receipt.dynamics.time == receipt.contact_time
    assert receipt.output_time == after.time
    assert problem.h_s == float(np.float32(H))
    native = _native(
        spec,
        receipt,
        tuple(SimpleNamespace(identity=r["identity"], points=r["points"]) for r in rows),
    )
    m, expected = problem.mass_matrix, native["mass"]
    if len(before.v) == 6:
        _close(m[:3, :3], expected[:3, :3], 2e-6, 3e-4)
        _close(m[3:, 3:], expected[3:, 3:], 2e-9, 3e-4)
        _close(m[:3, 3:], expected[:3, 3:], 2e-8, 3e-4)
        _close(m[3:, :3], expected[3:, :3], 2e-8, 3e-4)
    else:
        _close(m, expected, 2e-6, 3e-4)
    dynamics = receipt.dynamics
    np.testing.assert_array_equal(m, dynamics.mass_matrix.astype(np.float32))
    np.testing.assert_array_equal(problem.damping, native["damping"].astype(np.float32))
    np.testing.assert_array_equal(problem.input_velocity, before.v)
    for name in ("bias", "passive", "applied"):
        _close(getattr(dynamics, name), native[name], 2e-5, 3e-4)
    _close(dynamics.jacobian_com, native["com_j"], 2e-6, 2e-5)
    _close(dynamics.body_com, native["body_com"], 2e-6, 2e-5)
    a, p, b = (
        getattr(dynamics, name).astype(np.float32) for name in ("applied", "passive", "bias")
    )
    dv = problem.damping.astype(np.float32) * before.v.astype(np.float32)
    force = ((a + p) - b) + dv
    bound = (
        8
        * np.finfo(np.float32).eps
        * (
            np.abs(a.astype(float))
            + np.abs(p.astype(float))
            + np.abs(b.astype(float))
            + np.abs(dv.astype(float))
        )
        + np.finfo(np.float32).tiny
    )
    assert np.all(np.abs(problem.applied_force - force) <= bound)
    for i, sample in enumerate(problem.samples):
        _close(sample.jacobian_a, native["ja"][i], 2e-6, 2e-5)
        _close(sample.jacobian_b, native["jb"][i], 2e-6, 2e-5)
    return native


def _audit(spec, before, after, geometry):
    import mujoco as mj

    receipt = after.last_step
    rows, solver, problem = _published(receipt)
    audit = dict(
        surface=tuple(
            _surface(row, piece, receipt) for row, piece in zip(rows, geometry, strict=True)
        )
    )
    native = _native_binding(spec, receipt, before, after, rows, problem)
    audit["native"] = native
    velocity = after.v.astype(float)
    np.testing.assert_array_equal(solver["output_velocity"], after.v)
    e = _equations(problem, velocity)
    audit["equations"] = e
    assert e["residual_norm"] <= 2e-5 * e["scale"]
    assert solver["normalized_residual"] <= solver["tolerance"]
    _close(solver["solver_velocity"], velocity, 5e-5, 5e-5)
    generalized, native_generalized, friction_power, offset = (
        np.zeros(len(velocity)),
        np.zeros(len(velocity)),
        0.0,
        0,
    )
    audit["reference_moments"] = []
    for row in rows:
        ns = len(row["sample_ids"])
        sl = slice(offset, offset + ns)
        _close(row["normal_force"], e["normal"][sl], 0.002, 0.002)
        _close(row["tangential_force"], e["tangent"][sl], 0.002, 0.002)
        _close(row["force_on_a"], e["force"][sl], 0.002, 0.002)
        _close(row["torque_on_a_world_origin"], e["moment"][sl], 1e-5, 0.002)
        np.testing.assert_array_equal(row["force_on_b"], -row["force_on_a"])
        np.testing.assert_array_equal(
            row["torque_on_b_world_origin"], -row["torque_on_a_world_origin"]
        )
        moments = {}
        for side, body in zip(("a", "b"), row["body_ids"], strict=True):
            force = row["force_on_" + side].astype(float)
            for label, origin in (
                ("world", np.zeros(3)),
                ("body", native["body_position"][body]),
                ("com", native["body_com"][body]),
            ):
                independent = np.sum(np.cross(row["points"] - origin, force), axis=0)
                published = np.sum(
                    row["torque_on_" + side + "_world_origin"] - np.cross(origin, force), axis=0
                )
                _close(published, independent, 1e-5, 0.002)
                moments[side + "_" + label] = independent
        audit["reference_moments"].append(moments)
        for local, sample in enumerate(problem.samples[sl]):
            g = sample.jacobian_a.astype(float) - sample.jacobian_b.astype(float)
            normal = sample.normal_world.astype(float)
            relative = g @ velocity
            tangent = relative - normal * (normal @ relative) / (normal @ normal)
            fn, ft, force = (
                row["normal_force"][local],
                row["tangential_force"][local].astype(float),
                row["force_on_a"][local].astype(float),
            )
            _close(row["relative_velocity"][local], relative, 2e-5, 2e-5)
            assert fn >= 0 and np.linalg.norm(ft) <= float(sample.mu) * float(fn) + 1e-5
            power = float(ft @ tangent)
            assert power <= 1e-14
            friction_power += power
            generalized += g.T @ force
            native_generalized += (
                native["ja"][offset + local] - native["jb"][offset + local]
            ).T @ force
        offset += ns
    actual_q = np.asarray(solver["generalized_force"], dtype=float)
    _close(actual_q, generalized, 2e-5, 2e-5)
    np.testing.assert_array_equal(receipt.qfrc_constraint, solver["generalized_force"])
    if len(velocity) == 6:
        _close(actual_q[:3], native_generalized[:3], 0.002, 0.002)
        _close(actual_q[3:], native_generalized[3:], 1e-5, 0.002)
    else:
        _close(actual_q, native_generalized, 0.002, 0.002)
    residual = e["response"] @ velocity - e["rhs"] - problem.h_s * generalized
    norm = np.sqrt(max(0.0, float(residual @ np.linalg.solve(e["response"], residual))))
    assert norm <= 2e-5 * e["scale"]
    rhs = receipt.dynamics.applied + receipt.dynamics.passive - receipt.dynamics.bias + actual_q
    physical = np.linalg.solve(problem.mass_matrix.astype(float), rhs)
    integration = np.linalg.solve(e["response"], rhs)
    h = problem.h_s
    _close(h * receipt.qacc, h * physical, 5e-5, 5e-5)
    _close(h * receipt.integration_acceleration, h * integration, 5e-5, 5e-5)
    np.testing.assert_allclose(
        velocity, before.v + h * receipt.integration_acceleration, rtol=2e-6, atol=2e-7
    )
    model = spec.compilation.fresh_native_model()
    for row in rows:
        assert tuple(row["body_ids"]) == tuple(
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
            for name in (row["identity"].compliant_body, row["identity"].rigid_body)
        )
    q = before.q.copy()
    mj.mj_integratePos(model, q, velocity, h)
    if model.nv == 6:
        np.testing.assert_allclose(after.q[:3], q[:3], rtol=0, atol=2e-7)
        sign = 1 if after.q[3:] @ q[3:] >= 0 else -1
        np.testing.assert_allclose(sign * after.q[3:], q[3:], rtol=0, atol=3e-7)
        assert abs(float(after.q[3:] @ after.q[3:]) - 1) <= 2e-6
    else:
        np.testing.assert_allclose(after.q, q, rtol=0, atol=2e-7)
    audit.update(
        actual_generalized=actual_q,
        native_generalized=native_generalized,
        stored_force_residual=residual,
        stored_force_norm=norm,
        strict_host_gate=e["residual_norm"] <= 1e-10 * e["scale"],
        integrated_q=q,
        physical_acceleration=physical,
        integration_acceleration=integration,
        friction_power_w=friction_power,
        contact_power_w=float(actual_q @ velocity),
        kinetic_input_j=0.5 * float(before.v @ problem.mass_matrix @ before.v),
        kinetic_output_entry_mass_j=0.5 * float(velocity @ problem.mass_matrix @ velocity),
        energy_scope="entry-mass kinetic/power diagnostics only; no global contact potential claimed",
    )
    return audit


def _coverage(records):
    geometry_counts, force_counts, tangential_loads, angular_speeds = [], [], [], []
    for record in records:
        if not record.get("state_accepted"):
            continue
        rows = record["after"].last_step.raw_contact_fields["area_friction"]
        geometry_counts.append(sum(bool(len(row["sample_ids"])) for row in rows))
        force_counts.append(sum(bool(np.any(row["normal_force"] > 0)) for row in rows))
        tangential_loads.append(
            sum(float(np.sum(np.linalg.norm(row["tangential_force"], axis=1))) for row in rows)
        )
        velocity = record["after"].v
        angular_speeds.append(float(np.linalg.norm(velocity[3:])) if len(velocity) == 6 else None)
    contact_seen, separated, recontact, impact, unload = False, False, False, False, False
    for i, count in enumerate(geometry_counts):
        if count:
            if separated:
                recontact = True
            if i and not geometry_counts[i - 1]:
                impact = True
            contact_seen = True
        elif contact_seen:
            separated = True
        if i and force_counts[i - 1] > 0 and force_counts[i] == 0:
            unload = True

    def status(value):
        return "OBSERVED" if value else "NOT_OBSERVED"

    return dict(
        geometry_pair_counts=geometry_counts,
        positive_force_pair_counts=force_counts,
        tangential_load_sum_n=tangential_loads,
        free_body_angular_speed_rad_s=angular_speeds,
        impact=status(impact),
        force_unload=status(unload),
        geometric_separation=status(separated),
        recontact=status(recontact),
        dual_to_single=status(
            any(a == 2 and b == 1 for a, b in zip(force_counts, force_counts[1:]))
        ),
        any_empty=status(any(n == 0 for n in geometry_counts)),
        note="Same-run coverage, independent of numerical qualification; no event is synthesized",
    )


@pytest.mark.cuda
@pytest.mark.parametrize("window", WINDOWS)
def test_registered_live_friction_window(window, tmp_path, monkeypatch):
    import mujoco as mj

    payload = dict(schema="mujoco-area-friction-window-v1", window=window, h=H, records=[])

    def forbidden(*args, **kwargs):
        pytest.fail("window must advance only the public live TYX Scene")

    monkeypatch.setattr(mj, "mj_step", forbidden)
    monkeypatch.setattr(TyxAreaFrictionWorkspace, "solve_many", forbidden)
    try:
        spec, world, assets, steps, declaration = _declarations(window)
        payload.update(
            model=dict(
                key=spec.key,
                source=spec.compilation.source,
                area_pressure_pairs=spec.area_pressure_pairs,
                area_friction_laws=spec.area_friction_laws,
                contact_capacity=spec.contact_capacity,
                constraint_capacity=spec.constraint_capacity,
            ),
            world=world,
            assets=assets,
            declared_steps=steps,
            declaration=declaration,
        )
        with ExitStack() as stack:
            batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda"))
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
            ws = stack.enter_context(
                batch.prepare_device_workspace(include_bodies=True, include_area_pressure=True)
            )
            before = batch.integration_snapshot()[world.world]
            payload.update(initial=before, layout=batch.layout, observation_binding=ws.binding)
            for step in range(steps):
                config = batch.area_pressure_configuration_id(world.world)
                record = dict(
                    step=step, before=before, state_accepted=False, observation_complete=False
                )
                payload["records"].append(record)
                started = perf_counter()
                accepted = scene.step(H, expected_revision=batch.checkpoint_revision)
                after = batch.last_step.states[world.world]
                geometry = scene.last_geometry[world.world]
                record.update(
                    state_accepted=True,
                    accepted=accepted,
                    after=after,
                    geometry=geometry,
                    scene_step_s=perf_counter() - started,
                )
                started = perf_counter()
                frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                arrays = ws.readback(frame)
                record.update(
                    frame=frame,
                    arrays=arrays,
                    observation_complete=True,
                    observation_s=perf_counter() - started,
                )
                assert accepted == {world.world: True} and frame.ticks == (step + 1,)
                assert all(piece.surface.stamp.configuration_id == config for piece in geometry)
                record["audit"] = _audit(spec, before, after, geometry)
                before = after
            payload.update(
                final=batch.integration_snapshot()[world.world],
                counters=batch.counters,
                status="PASS",
                qualification="registered finite window; event coverage reported separately",
            )
    except BaseException as error:
        causes, cause = [], error
        while cause is not None:
            causes.append(dict(type=type(cause).__name__, message=str(cause)))
            cause = cause.__cause__
        payload.update(status="FAIL", errors=causes)
        raise
    finally:
        payload["coverage"] = _coverage(payload["records"])
        payload["accepted_steps"] = sum(
            row.get("state_accepted", False) for row in payload["records"]
        )
        with gzip.open(tmp_path / (window + ".json.gz"), "xt", encoding="utf-8") as stream:
            json.dump(_plain(payload), stream, allow_nan=False)
        (tmp_path / (window + "-summary.json")).write_text(
            json.dumps(
                dict(
                    window=window,
                    status=payload.get("status"),
                    accepted_steps=payload["accepted_steps"],
                    declared_steps=payload.get("declared_steps"),
                    coverage=payload["coverage"],
                    errors=payload.get("errors", []),
                ),
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

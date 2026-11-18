"""Fixed live approach/load/unload windows: two slides, one native limit.

Only Scene.step advances state. Native forward supplies current coefficients;
the independent finite active-set oracle solves the area/limit/dry equation.
No measured result changes the 8 + 6 step load schedule or the initial state.
"""

import gzip
import itertools
import json
import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np
import pytest
from test_mujoco_area_friction import _close, _plain
from test_mujoco_area_friction_coupled_guide import MASS, H
from test_mujoco_area_friction_coupled_guide import _declarations as _coupled
from test_mujoco_area_friction_lifecycle import _same
from test_mujoco_area_friction_live import _surface

from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
from phymeo.backends.mujoco_heterogeneous_area_scene import (
    MuJoCoTyxHeterogeneousAreaPressureScene,
    MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
)
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.backends.tyx_area_friction import TyxAreaFrictionWorkspace
from phymeo.builders.mujoco_compiler import compile_mujoco_xml

CASES = ((41, 1, 0.0), (7, -1, 0.0), (305, 1, 0.125))
STEPS, LOAD_STEPS = 14, 8


def _declarations(key, side, dry):
    base, _, assets = _coupled(key, 0, dry)
    tree = ET.fromstring(base.compilation.source.primary_xml)
    joint = tree.find(".//joint[@name='child-slide']")
    for name, value in dict(
        limited="true",
        range="-.02 .0005" if side == 1 else "-.0005 .02",
        margin="0",
        solreflimit=".02 1",
        solimplimit=".5 .5 .001 .5 2",
    ).items():
        joint.set(name, value)
    spec = replace(
        base,
        key=f"slide-limit-{side}-{dry}",
        compilation=compile_mujoco_xml(ET.tostring(tree, encoding="unicode"), require_asset=False),
    )
    world = MuJoCoTyxWorldSpec(key, spec.key, MuJoCoInitialState((-0.002, 0.0), (0.0, side * 0.01)))
    return spec, world, {spec.key: assets[base.key]}


def _control(side, step):
    drive = side * (0.5 if step < LOAD_STEPS else -0.5)
    return MuJoCoControl((), qfrc_applied=(drive - 1.0, drive))


def _oracle(mass, v0, base, h, jn, k, phi, native):
    """Enumerate two area faces and each native row's exact box faces."""
    options = [(-1, 0, 1) if r["kind"] == 1 else (-1, 0) for r in native]
    answers = []
    for area_on in (False, True) if len(k) else (False,):
        for faces in itertools.product(*options):
            matrix, rhs = mass.copy(), base.copy()
            if area_on:
                matrix += h * h * np.einsum("s,si,sj->ij", k, jn, jn)
                rhs -= jn.T @ (k * (phi + h * (jn @ v0)))
            for row, face in zip(native, faces, strict=True):
                j, r = row["j"], row["r"]
                if face == 0:
                    matrix += np.outer(j, j) / r
                    rhs += j * row["aref"] / r
                else:
                    rhs += j * (face * row["loss"] if row["kind"] == 1 else 0.0)
            acceleration = np.linalg.solve(matrix, rhs)
            velocity = v0 + h * acceleration
            extension = phi + h * (jn @ velocity)
            if (area_on and np.any(extension > 1e-12)) or (
                not area_on and np.any(extension < -1e-12)
            ):
                continue
            forces, gradients, admissible = [], [], True
            for row, face in zip(native, faces, strict=True):
                force = (
                    (row["aref"] - row["j"] @ acceleration) / row["r"]
                    if face == 0
                    else face * row["loss"]
                    if row["kind"] == 1
                    else 0.0
                )
                grad = float(row["j"] @ acceleration + row["r"] * force - row["aref"])
                if face == 0:
                    admissible &= (
                        -row["loss"] - 1e-10 <= force <= row["loss"] + 1e-10
                        if row["kind"] == 1
                        else force >= -1e-10
                    )
                else:
                    admissible &= grad >= -1e-10 if face == -1 else grad <= 1e-10
                forces.append(float(force))
                gradients.append(grad)
            if admissible:
                answers.append(
                    dict(
                        v=velocity,
                        a=acceleration,
                        normal=np.maximum(0.0, -k * extension),
                        forces=forces,
                        gradients=gradients,
                        area_on=area_on,
                        faces=faces,
                    )
                )
    assert answers, "no admissible finite active set"
    for answer in answers[1:]:
        np.testing.assert_allclose(answer["v"], answers[0]["v"], rtol=0, atol=1e-10)
    return answers[0]


def test_registered_free_approach_reaches_limit_on_fourth_input():
    for side in (-1, 1):
        q, v = np.array((-0.002, 0.0)), np.array((0.0, side * 0.01))
        for expected in (0.00015, 0.00035, 0.00060):
            result = _oracle(
                MASS,
                v,
                np.array((side * 0.5 - 1.0, side * 0.5)),
                0.01,
                np.array(((1.0, 0.0),)),
                np.array([1000.0]),
                np.array([-0.001]),
                (),
            )
            np.testing.assert_allclose(result["a"], (0.0, side * 0.5), atol=1e-13, rtol=0)
            v = result["v"]
            q += 0.01 * v
            assert q[1] == pytest.approx(side * expected, abs=1e-14)
        assert side * q[1] > 0.0005


def _native(spec, before, pair):
    import mujoco as mj

    model = spec.compilation.fresh_native_model()
    data = mj.MjData(model)
    assert model.nq == model.nv == 2 and np.count_nonzero(model.jnt_limited) == 1
    assert not np.any(model.dof_damping)
    data.qpos[:], data.qvel[:], data.time = before.q, before.v, before.time
    data.qfrc_applied[:] = before.held_control.qfrc_applied
    mj.mj_forward(model, data)
    assert data.ncon == 0
    ids = tuple(
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        for name in (pair["identity"].compliant_body, pair["identity"].rigid_body)
    )
    assert ids == tuple(pair["body_ids"])
    mass = np.empty((2, 2))
    mj.mj_fullM(model, mass, data.qM)
    np.testing.assert_allclose(mass, MASS, rtol=0, atol=1e-14)
    ja, jb = [], []
    for point in pair["points"]:
        left, right, angular = (np.empty((3, 2)) for _ in range(3))
        mj.mj_jac(model, data, left, angular, point, int(pair["body_ids"][0]))
        mj.mj_jac(model, data, right, angular, point, int(pair["body_ids"][1]))
        ja.append(left.copy())
        jb.append(right.copy())
    jac = data.efc_J.reshape(data.nefc, 2)
    rows = []
    for i in range(data.nefc):
        kind, source = int(data.efc_type[i]), int(data.efc_id[i])
        assert kind in (1, 3)
        dof = source if kind == 1 else int(model.jnt_dofadr[source])
        side = -1 if kind == 1 or jac[i, dof] > 0 else 1
        rows.append(
            dict(
                kind=kind,
                source=source,
                dof=dof,
                side=side,
                j=jac[i].copy(),
                r=float(data.efc_R[i]),
                aref=float(data.efc_aref[i]),
                loss=float(data.efc_frictionloss[i]),
                position=float(data.efc_pos[i]),
                margin=float(data.efc_margin[i]),
                velocity=float(data.efc_vel[i]),
                native_only_force=float(data.efc_force[i]),
            )
        )
    return dict(
        mass=mass,
        ja=np.asarray(ja).reshape(-1, 3, 2),
        jb=np.asarray(jb).reshape(-1, 3, 2),
        bias=data.qfrc_bias.copy(),
        passive=data.qfrc_passive.copy(),
        applied=data.qfrc_applied.copy(),
        rows=rows,
        range=model.jnt_range[1].copy(),
        margin=float(model.jnt_margin[1]),
    )


def _audit(spec, case, before, after, geometry, out):
    import mujoco as mj

    key, side, dry = case
    receipt = after.last_step
    pair = receipt.raw_contact_fields["area_friction"][0]
    solver = receipt.raw_contact_fields["area_friction_solver"]
    assert solver["schema"] == "mujoco-tyx-area-friction-limit-v1"
    assert solver["solver_status"] == 0
    assert receipt.contacts == () and receipt.tick == before.tick + 1 == after.tick
    assert receipt.contact_time == receipt.input_time == receipt.dynamics.time == before.time
    assert receipt.output_time == after.time
    assert receipt.contact_phase == "tyx-euler-input-constraint-evaluation"
    assert receipt.dynamics.phase == "tyx-euler-input-dynamics"
    np.testing.assert_array_equal(receipt.input_q, before.q)
    np.testing.assert_array_equal(receipt.input_v, before.v)
    np.testing.assert_array_equal(receipt.dynamics.input_q, before.q)
    np.testing.assert_array_equal(receipt.dynamics.input_v, before.v)
    out["surface"] = _surface(pair, geometry[0], receipt)
    native = out["native"] = _native(spec, before, pair)
    actual = receipt.constraints
    assert len(actual) == len(native["rows"])
    expected_active = any(
        distance < native["margin"]
        for distance in (before.q[1] - native["range"][0], native["range"][1] - before.q[1])
    )
    assert sum(r.constraint_type == 3 for r in actual) == int(expected_active)
    assert sum(r.constraint_type == 1 for r in actual) == int(dry > 0)
    for row, source in zip(actual, native["rows"], strict=True):
        assert (row.constraint_type, row.source_id, row.dof, row.side_or_edge) == (
            source["kind"],
            source["source"],
            source["dof"],
            source["side"],
        )
        np.testing.assert_array_equal(row.jacobian, source["j"])
        assert row.geometry_slot == -1
        assert source["r"] == pytest.approx(2.0 if source["kind"] == 3 else 1.0, abs=1e-12)
        assert row.regularization == pytest.approx(source["r"], abs=2e-6, rel=3e-4)
        assert row.reference_acceleration == pytest.approx(source["aref"], abs=2e-5, rel=3e-4)
        for name in ("position", "margin", "velocity"):
            assert getattr(row, name) == pytest.approx(source[name], abs=2e-7, rel=3e-4)
        if row.constraint_type == 3:
            assert row.side_or_edge == side and row.dof == row.source_id == 1
            assert row.lower == 0 and row.upper is None and row.force >= 0
        else:
            assert row.dof == 0 and (row.lower, row.upper) == (-dry, dry)
    _close(solver["mass_matrix"], native["mass"], 2e-6, 3e-4)
    _close(pair["jacobian_a"], native["ja"], 2e-6, 2e-5)
    _close(pair["jacobian_b"], native["jb"], 2e-6, 2e-5)
    for name in ("bias", "passive", "applied"):
        _close(getattr(receipt.dynamics, name), native[name], 2e-5, 3e-4)
    g = native["ja"] - native["jb"]
    jn = np.einsum("si,sij->sj", pair["normals"], g)
    np.testing.assert_array_equal(jn, np.tile((1.0, 0.0), (len(jn), 1)))
    h, v0, v = float(solver["h_s"]), before.v, after.v
    assert h == float(np.float32(H))
    np.testing.assert_array_equal(solver["input_velocity"], v0)
    np.testing.assert_array_equal(solver["output_velocity"], v)
    np.testing.assert_array_equal(solver["damping"], 0)
    base = native["applied"] + native["passive"] - native["bias"]
    _close(solver["applied_force"], base, 2e-7, 2e-5)
    solution = out["oracle"] = _oracle(
        native["mass"], v0, base, h, jn, pair["stiffness"], pair["surrogate_gap"], native["rows"]
    )
    _close(v, solution["v"], 5e-5, 5e-5)
    _close(pair["normal_force"], solution["normal"], 0.002, 0.002)
    _close(np.array([r.force for r in actual]), np.asarray(solution["forces"]), 0.002, 0.002)
    final_normal = np.maximum(0.0, -pair["stiffness"] * (pair["surrogate_gap"] + h * (jn @ v)))
    _close(pair["normal_force"], final_normal, 0.002, 0.002)
    np.testing.assert_array_equal(pair["tangential_force"], 0)
    _close(pair["relative_velocity"], np.einsum("sij,j->si", g, v), 2e-5, 2e-5)
    _close(pair["force_on_a"], pair["normal_force"][:, None] * pair["normals"], 0.002, 0.002)
    np.testing.assert_array_equal(pair["force_on_b"], -pair["force_on_a"])
    _close(
        pair["torque_on_a_world_origin"], np.cross(pair["points"], pair["force_on_a"]), 1e-5, 0.002
    )
    np.testing.assert_array_equal(
        pair["torque_on_b_world_origin"], -pair["torque_on_a_world_origin"]
    )
    total = np.einsum("sij,si->j", g, pair["force_on_a"])
    for row in actual:
        total += row.jacobian * row.force
    _close(solver["generalized_force"], total, 2e-5, 2e-5)
    np.testing.assert_array_equal(receipt.qfrc_constraint, solver["generalized_force"])
    physical = np.linalg.solve(native["mass"], base + total)
    _close(h * receipt.qacc, h * physical, 5e-5, 5e-5)
    _close(h * receipt.integration_acceleration, h * physical, 5e-5, 5e-5)
    common = (v - v0) / h
    for row, source in zip(actual, native["rows"], strict=True):
        gradient = float(
            row.jacobian @ physical + row.regularization * row.force - row.reference_acceleration
        )
        assert row.gradient == pytest.approx(gradient, abs=2e-5, rel=2e-5)
        projected = (source["aref"] - source["j"] @ common) / source["r"]
        predicted = max(0.0, projected) if source["kind"] == 3 else np.clip(projected, -dry, dry)
        assert row.force == pytest.approx(predicted, abs=0.002, rel=0.002)
        scale = (
            1
            + abs(row.jacobian @ physical)
            + abs(row.regularization * row.force)
            + abs(row.reference_acceleration)
        )
        violation = (
            max(0.0, -gradient)
            if row.force == row.lower
            else max(0.0, gradient)
            if row.upper is not None and row.force == row.upper
            else abs(gradient)
        )
        assert violation <= 2e-5 * scale
    residual = native["mass"] @ (v - v0) - h * (base + total)
    free = v0 + h * np.linalg.solve(native["mass"], base)
    assert np.sqrt(max(0.0, residual @ np.linalg.solve(native["mass"], residual))) <= 2e-5 * (
        1 + np.sqrt(free @ native["mass"] @ free)
    )
    q = before.q.copy()
    mj.mj_integratePos(spec.compilation.fresh_native_model(), q, v, h)
    np.testing.assert_allclose(after.q, q, rtol=0, atol=2e-7)
    assert solver["normalized_residual"] <= solver["tolerance"] == 2e-5
    limits = [r for r in actual if r.constraint_type == 3]
    out.update(
        limit_present=bool(limits),
        limit_force=sum(r.force for r in limits),
        dry_force=sum(r.force for r in actual if r.constraint_type == 1),
        physical_acceleration=physical,
        momentum_residual=residual,
        integrated_q=q,
    )


def _forged_limit(scene, monkeypatch, evidence):
    import phymeo.backends.mujoco_checkpoint as codec

    batch = scene.batch
    owner = batch._buckets[batch.layout[41].bucket]
    leaf = owner.checkpoint()
    revision, geometry, original = (
        batch.checkpoint_revision,
        scene.last_geometry,
        scene.checkpoint().to_bytes(),
    )
    for label in ("identity", "missing-active-row"):
        state = dict(leaf._read(leaf.binding))
        item = dict(state["worlds"][0])
        receipt = item["receipt"]
        assert any(r.constraint_type == 3 and r.force > 0 for r in receipt.constraints)
        rows = (
            tuple(
                replace(r, source_id=0) if r.constraint_type == 3 else r
                for r in receipt.constraints
            )
            if label == "identity"
            else tuple(r for r in receipt.constraints if r.constraint_type != 3)
        )
        item["receipt"] = replace(receipt, constraints=rows)
        state["worlds"] = (item,)
        forged = codec._make_checkpoint(leaf.binding, leaf.source_revision, state)
        with monkeypatch.context() as patch:
            patch.setattr(
                owner,
                "_build_device",
                lambda *a, **kw: pytest.fail("forged limit reached allocation"),
            )
            with pytest.raises(ValueError) as error:
                owner.restore(forged, expected_revision=owner.checkpoint_revision)
        evidence.append(dict(kind=label, error=str(error.value)))
        assert batch.checkpoint_revision == revision and scene.last_geometry is geometry
        assert scene.checkpoint().to_bytes() == original


@pytest.mark.cuda
def test_live_limit_sides_dry_unload_replay_and_capacity(tmp_path, monkeypatch):
    import mujoco as mj

    def forbidden(*args, **kwargs):
        raise AssertionError("only Scene.step may advance or assemble the physical solve")

    monkeypatch.setattr(mj, "mj_step", forbidden)
    monkeypatch.setattr(TyxAreaFrictionWorkspace, "solve_many", forbidden)
    evidence = dict(
        schema="area-friction-slide-limit-test-v1",
        cases=CASES,
        records=[],
        capacity=[],
        status="IN_PROGRESS",
    )
    try:
        declared = [_declarations(*case) for case in CASES]
        specs, worlds = tuple(d[0] for d in declared), tuple(d[1] for d in declared)
        assets = {k: v for d in declared for k, v in d[2].items()}
        evidence.update(
            sources={s.key: s.compilation.source for s in specs}, worlds=worlds, assets=assets
        )
        with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as batch:
            for key in batch.world_keys:
                assert (
                    batch._buckets[batch.layout[key].bucket].capabilities["domain"]
                    == "area-friction-cuda-slide-limit-v1"
                )
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
            with batch.prepare_device_workspace(
                include_bodies=True, include_area_pressure=True
            ) as ws:

                def step(index, label):
                    batch.set_controls({key: _control(side, index) for key, side, _ in CASES})
                    record = dict(
                        index=index,
                        label=label,
                        before=batch.integration_snapshot(),
                        state_accepted=False,
                    )
                    evidence["records"].append(record)
                    assert scene.step(
                        H, expected_revision=batch.checkpoint_revision
                    ) == dict.fromkeys(batch.world_keys, True)
                    record.update(
                        state_accepted=True,
                        after=batch.last_step.states,
                        geometry=scene.last_geometry,
                        audits={},
                    )
                    frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                    record.update(frame=frame, arrays=ws.readback(frame))
                    for spec, case in zip(specs, CASES, strict=True):
                        audit = record["audits"][case[0]] = {}
                        _audit(
                            spec,
                            case,
                            record["before"][case[0]],
                            record["after"][case[0]],
                            record["geometry"][case[0]],
                            audit,
                        )
                    return record

                for index in range(STEPS):
                    record = step(index, "loading" if index < LOAD_STEPS else "unloading")
                    if index == LOAD_STEPS - 1:
                        capsule = scene.checkpoint()
                        saved = record
                        evidence["checkpoint"] = capsule.to_bytes()
                        evidence["forged"] = []
                        _forged_limit(scene, monkeypatch, evidence["forged"])
                    if index == LOAD_STEPS:
                        expected = record
                events = evidence["events"] = {}
                for key, _, dry in CASES:
                    rows = [r["audits"][key] for r in evidence["records"]]
                    events[key] = dict(
                        no_row=[i for i, r in enumerate(rows) if not r["limit_present"]],
                        positive=[i for i, r in enumerate(rows) if r["limit_force"] > 0.002],
                        unloaded=[
                            i
                            for i, r in enumerate(rows)
                            if i >= LOAD_STEPS and r["limit_force"] == 0.0
                        ],
                        exited=[
                            i
                            for i, r in enumerate(rows)
                            if i >= LOAD_STEPS and not r["limit_present"]
                        ],
                    )
                    assert (
                        0 in events[key]["no_row"]
                        and events[key]["positive"]
                        and events[key]["unloaded"]
                    )
                    assert min(events[key]["positive"]) < max(events[key]["unloaded"])
                    if dry:
                        assert any(
                            r["limit_force"] > 0.002 and abs(r["dry_force"]) > 0.002 for r in rows
                        )
                restored = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(capsule.to_bytes())
                scene.restore(restored, expected_revision=batch.checkpoint_revision)
                _same(batch.integration_snapshot(), saved["after"])
                replay = step(LOAD_STEPS, "replay-unload")
                for key in batch.world_keys:
                    assert (
                        replay["geometry"][key][0].surface.stamp.configuration_id
                        != expected["geometry"][key][0].surface.stamp.configuration_id
                    )
                _same(replay["after"], expected["after"], fresh_stamp=True)
                _same(replay["geometry"], expected["geometry"], fresh_stamp=True)
                _same(replay["arrays"], expected["arrays"])
        spec, world, assets = declared[0]
        spec = replace(spec, constraint_capacity=6)
        with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
            batch.set_controls({41: _control(1, 0)})
            for index in range(3):
                assert scene.step(H, expected_revision=batch.checkpoint_revision) == {41: True}
                state = batch.last_step.states[41]
                evidence["capacity"].append(state)
                assert state.last_step.constraints == ()
                assert (
                    len(state.last_step.raw_contact_fields["area_friction"][0]["sample_ids"]) == 6
                )
            assert state.q[1] > 0.0005
            original, revision, geometry = (
                scene.checkpoint().to_bytes(),
                batch.checkpoint_revision,
                scene.last_geometry,
            )
            with pytest.raises(ValueError, match="capacity") as error:
                scene.step(H, expected_revision=revision)
            evidence["capacity_error"] = str(error.value)
            assert batch.checkpoint_revision == revision and scene.last_geometry is geometry
            assert scene.checkpoint().to_bytes() == original
        evidence["status"] = "PASS"
    except BaseException as error:
        evidence.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        with gzip.open(tmp_path / "slide-limit.json.gz", "xt", encoding="utf-8") as stream:
            json.dump(_plain(evidence), stream, allow_nan=False)


@pytest.mark.cuda
def test_scalar_limit_and_dry_friction_share_one_coordinate():
    from test_mujoco_area_friction_guide import _declarations as slider

    spec, world, assets = slider(0.125, 41)
    tree = ET.fromstring(spec.compilation.source.primary_xml)
    joint = tree.find(".//joint[@name='guide']")
    joint.attrib.update(
        limited="true",
        range="-.0016 .02",
        margin="0",
        solreflimit=".02 1",
        solimplimit=".5 .5 .001 .5 2",
    )
    spec = replace(
        spec,
        compilation=compile_mujoco_xml(ET.tostring(tree, encoding="unicode"), require_asset=False),
    )
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
        scene.step(H, expected_revision=batch.checkpoint_revision)
        state = batch.last_step.states[41]
        receipt = state.last_step
        dry, lower = receipt.constraints
        assert (dry.constraint_type, lower.constraint_type, lower.side_or_edge) == (1, 3, -1)
        assert lower.dof == dry.dof == 0 and lower.upper is None
        # a = (1 - .1*a) - 1/8 + (2 - a). Both native rows act on DOF 0.
        assert dry.force == pytest.approx(-1 / 8, abs=2e-6)
        assert lower.force == pytest.approx(53 / 84, abs=2e-5)
        area = receipt.raw_contact_fields["area_friction"][0]
        assert area["normal_force"].sum() == pytest.approx(145 / 168, abs=2e-5)
        np.testing.assert_allclose(receipt.qacc, [115 / 84], atol=2e-5)
        np.testing.assert_allclose(state.v, [H * 115 / 84], atol=2e-7)
        assert batch._buckets[0].capabilities["native_scalar_limit_joint_ids"] == (0,)


@pytest.mark.cuda
def test_limit_force_releases_before_row_disappears_in_the_same_world():
    # A stronger reversed load drives the child back inside its range in one step.
    spec, world, assets = _declarations(41, 1, 0.0)
    world = replace(world, initial_state=MuJoCoInitialState((-0.002, 0.0006), (0.0, 0.0)))
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
        history = []
        for drive in (0.5, -4.0, -4.0):
            batch.set_controls({41: MuJoCoControl((), qfrc_applied=(drive - 1.0, drive))})
            before = batch.integration_snapshot()[41]
            assert scene.step(H, expected_revision=batch.checkpoint_revision) == {41: True}
            after = batch.last_step.states[41]
            audit = {}
            _audit(spec, (41, 1, 0.0), before, after, scene.last_geometry[41], audit)
            history.append((audit["limit_present"], audit["limit_force"], after.q[1]))
        assert history[0][0] and history[0][1] > 0.002 and history[0][2] > 0.0005
        assert history[1][0] and history[1][1] == 0.0 and history[1][2] < 0.0005
        assert not history[2][0] and history[2][1] == 0.0 and history[2][2] < 0.0005


@pytest.mark.cuda
def test_limit_margin_at_equality_and_inside_with_direct_solref():
    spec, _, assets = _declarations(41, 1, 0.0)
    tree = ET.fromstring(spec.compilation.source.primary_xml)
    joint = tree.find(".//joint[@name='child-slide']")
    upper, margin = 1 / 1024, 1 / 2048
    joint.set("range", f"-0.02 {upper}")
    joint.set("margin", str(margin))
    joint.set("solreflimit", "-50 -5")
    joint.set("solimplimit", ".2 .85 .002 .4 3")
    spec = replace(
        spec,
        compilation=compile_mujoco_xml(ET.tostring(tree, encoding="unicode"), require_asset=False),
    )
    worlds = tuple(
        MuJoCoTyxWorldSpec(
            key, spec.key, MuJoCoInitialState((-0.002, upper - distance), (0.0, 0.01))
        )
        for key, distance in ((10, margin), (20, margin / 2))
    )
    with MuJoCoTyxHeterogeneousBatch((spec,), worlds, mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
        batch.set_controls({k: _control(1, 0) for k in batch.world_keys})
        before = batch.integration_snapshot()
        assert scene.step(H, expected_revision=batch.checkpoint_revision) == {10: True, 20: True}
        for key, state in batch.last_step.states.items():
            receipt = state.last_step
            pair = receipt.raw_contact_fields["area_friction"][0]
            native = _native(spec, before[key], pair)
            assert len(receipt.constraints) == len(native["rows"]) == int(key == 20)
            if key == 20:
                row = receipt.constraints[0]
                ref = native["rows"][0]
                assert row.position == margin / 2 and row.margin == margin
                assert row.upper is None and row.force > 0
                assert 0.2 < row.impedance < 0.85
                assert row.regularization == pytest.approx(ref["r"], rel=3e-4)
                assert row.reference_acceleration == pytest.approx(ref["aref"], rel=3e-4)
            normal_j = np.einsum("si,sij->sj", pair["normals"], native["ja"] - native["jb"])
            answer = _oracle(
                MASS,
                before[key].v,
                native["applied"],
                H,
                normal_j,
                pair["stiffness"],
                pair["surrogate_gap"],
                native["rows"],
            )
            np.testing.assert_allclose(state.v, answer["v"], atol=5e-5, rtol=5e-5)
            np.testing.assert_allclose(
                pair["normal_force"], answer["normal"], atol=0.002, rtol=0.002
            )
            np.testing.assert_allclose(
                [r.force for r in receipt.constraints], answer["forces"], atol=0.002, rtol=0.002
            )

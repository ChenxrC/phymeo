"""Two physical slider bodies couple area force to one native dry-friction row.

Both bodies weigh one kilogram; only the parent has pressure geometry. The
mass matrix is [[2, 1], [1, 1]], not a diagonal pair of independent solvers.
Native forward supplies entry coefficients; an independent three-branch linear
oracle handles the one bounded guide row. Only the production Scene steps.
"""

import gzip
import json
import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np
import pytest
from test_mujoco_area_friction import _close, _plain
from test_mujoco_area_friction_guide import H
from test_mujoco_area_friction_guide import _declarations as _one_slider
from test_mujoco_area_friction_lifecycle import _same
from test_mujoco_area_friction_live import _surface

from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch, MuJoCoTyxWorldSpec
from phymeo.backends.mujoco_heterogeneous_area_scene import (
    MuJoCoTyxHeterogeneousAreaPressureScene,
    MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
)
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.backends.tyx_area_friction import TyxAreaFrictionWorkspace
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.solvers.area_friction import LAW

CASES = ((41, 1, 2.0), (7, 1, 0.125), (305, 0, 2.0))
MASS = np.array(((2.0, 1.0), (1.0, 1.0)))
FIRST = {
    41: (40 / 43, 10 / 43, (30 / 43, -20 / 43)),
    7: (81 / 88, 1 / 8, (70 / 88, -59 / 88)),
    305: (20 / 21, -10 / 21, (10 / 21, -10 / 21)),
}


def _declarations(key, dry_dof, loss):
    base, _, old_assets = _one_slider(loss, key)
    tree = ET.fromstring(base.compilation.source.primary_xml)
    pressure = tree.find("./worldbody/body[@name='pressure']")
    parent_joint = pressure.find("joint")
    parent_joint.set("name", "parent-slide")
    parent_joint.set("frictionloss", str(loss if dry_dof == 0 else 0.0))
    child = ET.SubElement(pressure, "body", name="child", pos="0 0 .04")
    ET.SubElement(
        child,
        "joint",
        name="child-slide",
        type="slide",
        axis="0 0 1",
        limited="false",
        damping="0",
        frictionloss=str(loss if dry_dof == 1 else 0.0),
        armature="0",
        solimpfriction=".5 .5 .001 .5 2",
    )
    child.append(ET.fromstring(ET.tostring(pressure.find("inertial"))))
    spec = replace(
        base,
        key=f"coupled-guide-{dry_dof}-{loss}",
        compilation=compile_mujoco_xml(ET.tostring(tree, encoding="unicode"), require_asset=False),
    )
    world = MuJoCoTyxWorldSpec(key, spec.key, MuJoCoInitialState((-0.002, 0.0), (0.0, 0.0)))
    return spec, world, {spec.key: old_assets[base.key]}


def _native(spec, before, pair, dry_dof):
    import mujoco as mj

    model = spec.compilation.fresh_native_model()
    assert model.nq == model.nv == 2 and model.neq == model.ntendon == 0
    assert not np.any(model.dof_damping) and not np.any(model.jnt_limited)
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:], data.time = before.q, before.v, before.time
    data.ctrl[:] = before.held_control.ctrl
    for name in ("qfrc_applied", "xfrc_applied"):
        value = getattr(before.held_control, name)
        if value is not None:
            np.testing.assert_array_equal(value, 0)
            getattr(data, name)[:] = value
    mj.mj_forward(model, data)
    assert data.ncon == 0 and data.nefc == 1
    assert int(data.efc_type[0]) == int(mj.mjtConstraint.mjCNSTR_FRICTION_DOF)
    assert int(data.efc_id[0]) == dry_dof
    mass = np.empty((2, 2))
    mj.mj_fullM(model, mass, data.qM)
    np.testing.assert_allclose(mass, MASS, rtol=0, atol=1e-14)
    ids = tuple(
        mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, name)
        for name in (pair["identity"].compliant_body, pair["identity"].rigid_body)
    )
    assert ids == tuple(pair["body_ids"])
    child = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "child")
    assert model.body_mass[ids[0]] == model.body_mass[child] == 1.0
    assert not np.any(model.geom_bodyid == child)
    ja, jb = [], []
    for point in pair["points"]:
        left, right, angular = (np.empty((3, 2)) for _ in range(3))
        mj.mj_jac(model, data, left, angular, point, ids[0])
        mj.mj_jac(model, data, right, angular, point, ids[1])
        ja.append(left.copy())
        jb.append(right.copy())
    return dict(
        mass=mass,
        ja=np.asarray(ja),
        jb=np.asarray(jb),
        bias=data.qfrc_bias.copy(),
        passive=data.qfrc_passive.copy(),
        applied=data.qfrc_applied.copy() + data.qfrc_actuator.copy(),
        body_position=data.xpos.copy(),
        body_com=data.xipos.copy(),
        body_rotation=data.xmat.reshape(-1, 3, 3).copy(),
        dry_jacobian=data.efc_J.reshape(1, 2)[0].copy(),
        regularization=float(data.efc_R[0]),
        reference=float(data.efc_aref[0]),
        loss=float(data.efc_frictionloss[0]),
        native_only_force=float(data.efc_force[0]),
    )


def _active_set(mass, v0, applied, h, rows, stiffness, gap, dry_j, regularization, reference, loss):
    """Independent free/lower/upper enumeration while every area row is active."""
    matrix = mass + h * h * np.einsum("s,si,sj->ij", stiffness, rows, rows)
    rhs = applied - rows.T @ (stiffness * (gap + h * (rows @ v0)))
    matches, candidates = [], []
    for branch in ("free", "lower", "upper"):
        if branch == "free":
            acceleration = np.linalg.solve(
                matrix + np.outer(dry_j, dry_j) / regularization,
                rhs + dry_j * reference / regularization,
            )
            force = (reference - dry_j @ acceleration) / regularization
        else:
            force = -loss if branch == "lower" else loss
            acceleration = np.linalg.solve(matrix, rhs + dry_j * force)
        velocity = v0 + h * acceleration
        normal = -stiffness * (gap + h * (rows @ velocity))
        gradient = float(dry_j @ acceleration + regularization * force - reference)
        bounded = (
            -loss < force < loss
            if branch == "free"
            else gradient >= 0
            if branch == "lower"
            else gradient <= 0
        )
        item = dict(
            branch=branch,
            acceleration=acceleration,
            velocity=velocity,
            normal=normal,
            dry=float(force),
            gradient=gradient,
            all_area_active=bool(np.all(normal > 0)),
            bound_admissible=bool(bounded),
        )
        candidates.append(item)
        if np.all(normal > 0) and bounded:
            matches.append(item)
    assert len(matches) == 1, candidates
    return dict(solution=matches[0], candidates=candidates, matrix=matrix, rhs=rhs)


def test_three_hand_derived_coupled_first_steps():
    for key, dof, loss in CASES:
        result = _active_set(
            MASS,
            np.zeros(2),
            np.zeros(2),
            0.01,
            np.array(((1.0, 0.0),)),
            np.array([1000.0]),
            np.array([-0.001]),
            np.eye(2)[dof],
            float((1.0, 2.0)[dof]),
            0.0,
            loss,
        )["solution"]
        force, dry, acceleration = FIRST[key]
        np.testing.assert_allclose(result["acceleration"], acceleration, rtol=0, atol=1e-13)
        assert result["normal"][0] == pytest.approx(force, abs=1e-13, rel=0)
        assert result["dry"] == pytest.approx(dry, abs=1e-13, rel=0)
        assert result["branch"] == ("upper" if key == 7 else "free")


def _audit(spec, dry_dof, loss, before, after, geometry, audit):
    import mujoco as mj

    receipt = after.last_step
    raw = receipt.raw_contact_fields
    assert receipt.contacts == () and "area_pressure" not in raw
    assert len(receipt.constraints) == len(raw["area_friction"]) == len(geometry) == 1
    dry, pair = receipt.constraints[0], raw["area_friction"][0]
    solver = raw["area_friction_solver"]
    assert solver["schema"] == "mujoco-tyx-area-friction-guide-v1"
    assert solver["law"] == pair["law"] == LAW and solver["solver_status"] == 0
    assert receipt.tick == after.tick == before.tick + 1
    assert receipt.world == before.world == after.world
    assert receipt.input_time == before.time == receipt.contact_time == receipt.dynamics.time
    assert receipt.output_time == after.time
    assert receipt.contact_phase == "tyx-euler-input-constraint-evaluation"
    assert receipt.dynamics.phase == "tyx-euler-input-dynamics"
    np.testing.assert_array_equal(receipt.input_q, before.q)
    np.testing.assert_array_equal(receipt.input_v, before.v)
    np.testing.assert_array_equal(receipt.dynamics.input_q, before.q)
    np.testing.assert_array_equal(receipt.dynamics.input_v, before.v)
    audit["surface"] = _surface(pair, geometry[0], receipt)
    assert audit["surface"]["samples"] > 0
    native = audit["native"] = _native(spec, before, pair, dry_dof)
    assert dry.constraint_type == 1 and dry.dof == dry.source_id == dry_dof
    np.testing.assert_array_equal(dry.jacobian, np.eye(2)[dry_dof])
    np.testing.assert_array_equal(dry.jacobian, native["dry_jacobian"])
    assert native["regularization"] == pytest.approx((1.0, 2.0)[dry_dof], abs=1e-12)
    assert dry.regularization == pytest.approx(native["regularization"], abs=2e-6, rel=3e-4)
    assert dry.reference_acceleration == pytest.approx(native["reference"], abs=2e-5, rel=3e-4)
    assert native["loss"] == loss and (dry.lower, dry.upper) == (-loss, loss)
    _close(solver["mass_matrix"], native["mass"], 2e-6, 3e-4)
    _close(pair["jacobian_a"], native["ja"], 2e-6, 2e-5)
    _close(pair["jacobian_b"], native["jb"], 2e-6, 2e-5)
    for name in ("bias", "passive", "applied"):
        _close(getattr(receipt.dynamics, name), native[name], 2e-5, 3e-4)
    np.testing.assert_array_equal(solver["damping"], 0)
    np.testing.assert_array_equal(solver["input_velocity"], before.v)
    np.testing.assert_array_equal(solver["output_velocity"], after.v)
    _close(solver["solver_velocity"], after.v, 5e-5, 5e-5)
    h = float(solver["h_s"])
    assert h == float(np.float32(H))
    base = native["applied"] + native["passive"] - native["bias"]
    _close(solver["applied_force"], base, 2e-7, 2e-5)
    g = pair["jacobian_a"] - pair["jacobian_b"]
    rows = np.einsum("si,sij->sj", pair["normals"], g)
    np.testing.assert_array_equal(rows, np.tile((1.0, 0.0), (len(rows), 1)))
    result = audit["active_set"] = _active_set(
        native["mass"],
        before.v,
        base,
        h,
        rows,
        pair["stiffness"],
        pair["surrogate_gap"],
        native["dry_jacobian"],
        native["regularization"],
        native["reference"],
        loss,
    )
    expected = result["solution"]
    _close(after.v, expected["velocity"], 5e-5, 5e-5)
    _close(pair["normal_force"], expected["normal"], 0.002, 0.002)
    assert dry.force == pytest.approx(expected["dry"], abs=0.002, rel=0.002)
    common_acc = (after.v - before.v) / h
    final_normal = np.maximum(
        0.0, -pair["stiffness"] * (pair["surrogate_gap"] + h * (rows @ after.v))
    )
    final_dry = np.clip(
        (native["reference"] - native["dry_jacobian"] @ common_acc) / native["regularization"],
        -loss,
        loss,
    )
    _close(pair["normal_force"], final_normal, 0.002, 0.002)
    assert dry.force == pytest.approx(final_dry, abs=0.002, rel=0.002)
    np.testing.assert_array_equal(pair["tangential_force"], 0)
    _close(pair["relative_velocity"], np.einsum("sij,j->si", g, after.v), 2e-5, 2e-5)
    _close(pair["force_on_a"], pair["normal_force"][:, None] * pair["normals"], 0.002, 0.002)
    np.testing.assert_array_equal(pair["force_on_b"], -pair["force_on_a"])
    _close(
        pair["torque_on_a_world_origin"], np.cross(pair["points"], pair["force_on_a"]), 1e-5, 0.002
    )
    np.testing.assert_array_equal(
        pair["torque_on_b_world_origin"], -pair["torque_on_a_world_origin"]
    )
    area_q = np.einsum("sij,si->j", g, pair["force_on_a"])
    total = area_q + dry.jacobian * dry.force
    _close(solver["generalized_force"], total, 2e-5, 2e-5)
    np.testing.assert_array_equal(receipt.qfrc_constraint, solver["generalized_force"])
    physical = np.linalg.solve(native["mass"], base + total)
    _close(h * receipt.qacc, h * physical, 5e-5, 5e-5)
    _close(h * receipt.integration_acceleration, h * physical, 5e-5, 5e-5)
    np.testing.assert_allclose(
        after.v, before.v + h * receipt.integration_acceleration, rtol=2e-6, atol=2e-7
    )
    gradient = float(
        dry.jacobian @ receipt.qacc + dry.regularization * dry.force - dry.reference_acceleration
    )
    assert dry.gradient == pytest.approx(gradient, abs=2e-5, rel=2e-5)
    if expected["branch"] == "free":
        assert abs(dry.force) < loss - 0.1
        assert abs(gradient) <= 2e-5 * (
            1 + abs(dry.jacobian @ physical) + abs(dry.reference_acceleration)
        )
    else:
        assert expected["branch"] == "upper" and dry.force == pytest.approx(loss, abs=2e-6)
        assert gradient < -0.1
    residual = native["mass"] @ (after.v - before.v) - h * (base + total)
    mass_norm = np.sqrt(max(0.0, float(residual @ np.linalg.solve(native["mass"], residual))))
    free = before.v + h * np.linalg.solve(native["mass"], base)
    scale = 1 + np.sqrt(max(0.0, float(free @ native["mass"] @ free)))
    assert mass_norm <= 2e-5 * scale
    assert solver["normalized_residual"] <= solver["tolerance"] == 2e-5
    q = before.q.copy()
    mj.mj_integratePos(spec.compilation.fresh_native_model(), q, after.v, h)
    np.testing.assert_allclose(after.q, q, rtol=0, atol=2e-7)
    assert np.all(after.q != before.q)
    if before.tick == 0:
        force, reaction, acceleration = FIRST[before.world]
        _close(after.v, H * np.asarray(acceleration), 5e-5, 5e-5)
        assert np.sum(pair["normal_force"]) == pytest.approx(force, abs=0.002, rel=0.002)
        assert dry.force == pytest.approx(reaction, abs=0.002, rel=0.002)
        assert after.v[0] > 0 and after.v[1] < 0
    audit.update(
        area_generalized=area_q,
        total_generalized=total,
        common_acceleration=common_acc,
        physical_acceleration=physical,
        final_normal=final_normal,
        final_dry=float(final_dry),
        momentum_residual=residual,
        mass_norm=mass_norm,
        physical_gradient=gradient,
        integrated_q=q,
    )


@pytest.mark.cuda
def test_two_body_guide_live_coupling_checkpoint_and_wrong_row(tmp_path, monkeypatch):
    import mujoco as mj

    import phymeo.backends.mujoco_checkpoint as codec

    def forbidden(*args, **kwargs):
        raise AssertionError("live coupled guide forbids CPU stepping and HOST-packed solve")

    monkeypatch.setattr(mj, "mj_step", forbidden)
    monkeypatch.setattr(TyxAreaFrictionWorkspace, "solve_many", forbidden)
    evidence = dict(
        schema="mujoco-area-friction-coupled-guide-test-v1",
        records=[],
        cases=CASES,
        status="IN_PROGRESS",
    )
    try:
        declared = [_declarations(*case) for case in CASES]
        specs, worlds = tuple(d[0] for d in declared), tuple(d[1] for d in declared)
        assets = {key: value for d in declared for key, value in d[2].items()}
        evidence.update(
            models={
                s.key: dict(
                    source=s.compilation.source,
                    laws=s.area_friction_laws,
                    pairs=s.area_pressure_pairs,
                    contact_capacity=s.contact_capacity,
                    constraint_capacity=s.constraint_capacity,
                )
                for s in specs
            },
            worlds=worlds,
            assets=assets,
        )
        with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as batch:
            evidence["capabilities"] = {}
            for key, dof, _ in CASES:
                caps = batch._buckets[batch.layout[key].bucket].capabilities
                assert caps["domain"] == "area-friction-cuda-coupled-slides-v1"
                assert caps["native_dry_dof_ids"] == (dof,)
                evidence["capabilities"][key] = caps
            scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
            with batch.prepare_device_workspace(
                include_bodies=True, include_area_pressure=True
            ) as ws:
                evidence["initial"] = batch.integration_snapshot()

                def step(label):
                    before = batch.integration_snapshot()
                    configs = {
                        key: batch.area_pressure_configuration_id(key) for key in batch.world_keys
                    }
                    record = dict(label=label, before=before, state_accepted=False)
                    evidence["records"].append(record)
                    accepted = scene.step(H, expected_revision=batch.checkpoint_revision)
                    after = batch.last_step.states
                    record.update(
                        state_accepted=True,
                        after=after,
                        accepted=accepted,
                        geometry=scene.last_geometry,
                    )
                    assert accepted == dict.fromkeys(batch.world_keys, True)
                    frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                    record.update(frame=frame, arrays=ws.readback(frame), audits={})
                    for spec, (key, dof, loss) in zip(specs, CASES, strict=True):
                        assert (
                            record["geometry"][key][0].surface.stamp.configuration_id
                            == configs[key]
                        )
                        audit = record["audits"][key] = {}
                        _audit(
                            spec, dof, loss, before[key], after[key], record["geometry"][key], audit
                        )
                    return record

                saved = step("first")
                capsule = scene.checkpoint()
                replay_capsule = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(
                    capsule.to_bytes()
                )
                assert replay_capsule.to_bytes() == capsule.to_bytes()
                evidence["checkpoint"] = capsule.to_bytes()
                owner = batch._buckets[batch.layout[41].bucket]
                leaf = owner.checkpoint()
                revision, original = owner.checkpoint_revision, leaf.to_bytes()
                batch_revision, geometry = batch.checkpoint_revision, scene.last_geometry
                evidence["forged_rejections"] = []
                wrong_j = np.frombuffer(np.array((1.0, 0.0)).tobytes(), dtype=np.float64)
                for label, mutation in (
                    ("wrong-dof", dict(dof=0, source_id=0)),
                    ("wrong-jacobian", dict(jacobian=wrong_j)),
                ):
                    state = dict(leaf._read(leaf.binding))
                    item = dict(state["worlds"][0])
                    receipt = item["receipt"]
                    assert receipt.constraints[0].dof == 1
                    item["receipt"] = replace(
                        receipt, constraints=(replace(receipt.constraints[0], **mutation),)
                    )
                    state["worlds"] = (item,)
                    forged = codec._make_checkpoint(leaf.binding, leaf.source_revision, state)
                    with monkeypatch.context() as patch:

                        def no_replacement(*args, **kwargs):
                            pytest.fail("wrong guide identity/J reached replacement allocation")

                        patch.setattr(owner, "_build_device", no_replacement)
                        with pytest.raises(ValueError) as rejected:
                            owner.restore(forged, expected_revision=revision)
                    evidence["forged_rejections"].append(
                        dict(label=label, message=str(rejected.value))
                    )
                    assert (
                        owner.checkpoint_revision == revision
                        and owner.checkpoint().to_bytes() == original
                    )
                    assert (
                        batch.checkpoint_revision == batch_revision
                        and scene.last_geometry is geometry
                    )
                    _same(batch.integration_snapshot(), saved["after"])
                    _same(ws.readback(saved["frame"]), saved["arrays"])
                    assert scene.checkpoint().to_bytes() == capsule.to_bytes()
                expected = step("second")
                scene.restore(replay_capsule, expected_revision=batch.checkpoint_revision)
                _same(batch.integration_snapshot(), saved["after"])
                assert not scene.last_geometry
                with pytest.raises(ValueError):
                    ws.readback(expected["frame"])
                frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                _same(ws.readback(frame), saved["arrays"])
                replay = step("replay")
                for key in batch.world_keys:
                    assert (
                        replay["geometry"][key][0].surface.stamp.configuration_id
                        != expected["geometry"][key][0].surface.stamp.configuration_id
                    )
                _same(replay["after"], expected["after"], fresh_stamp=True)
                _same(replay["geometry"], expected["geometry"], fresh_stamp=True)
                _same(replay["arrays"], expected["arrays"])
                evidence.update(
                    final=batch.integration_snapshot(), resources=batch.resource_snapshot()
                )
        evidence["status"] = "PASS"
    except BaseException as error:
        evidence.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        with gzip.open(tmp_path / "coupled-guide.json.gz", "xt", encoding="utf-8") as stream:
            json.dump(_plain(evidence), stream, allow_nan=False)

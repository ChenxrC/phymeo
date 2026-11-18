"""Two finite supports on one serial carriage/hinge, through the production CUDA scene."""

import json
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from test_mujoco_area_friction_eccentric_hinge import INERTIA, X0, H, _model
from test_mujoco_area_friction_live import _surface

from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch
from phymeo.backends.mujoco_heterogeneous_area_scene import MuJoCoTyxHeterogeneousAreaPressureScene
from phymeo.backends.mujoco_runtime import MuJoCoControl, MuJoCoInitialState
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.control.body_contact_wrench import BodyContactWrenchReader, BodyContactWrenchRequest

PLATES = ("left", "right")
CENTERS = (-0.009, 0.009)
HALF_X, HALF_Y = 0.003, 0.006


def _declarations(key, pivot, release=False, reverse=False):
    spec, world, (asset,) = _model(key, pivot, False)
    xml = ET.fromstring(spec.compilation.source.primary_xml)
    root = xml.find("worldbody")
    plate = root.find("body[@name='plate']")
    root.remove(plate)
    rigid = asset.rigid_local
    vertices = rigid.world_vertices * (HALF_X / 0.01, HALF_Y / 0.01, 1.0)
    mesh = replace(rigid, world_vertices=vertices)
    assets, laws = [], []
    for index, (name, center) in enumerate(zip(PLATES, CENTERS, strict=True)):
        part = deepcopy(plate)
        z = -0.0025 - (0.0008 if release and index == 0 else 0.0)
        part.attrib.update(name=name, pos=f"{X0 + center} 0 {z}")
        part.find("geom").attrib.update(name=f"{name}-geom", size=f"{HALF_X} {HALF_Y} .0025")
        root.append(part)
        ident = replace(asset.identity, rigid_body=name, rigid_material=f"{name}-surface")
        assets.append(replace(asset, identity=ident, rigid_local=mesh))
        mu = (0.15, 0.35)[index] if release else 0.0
        laws.append(replace(spec.area_friction_laws[0], identity=ident, mu=mu))
    if reverse:
        assets.reverse()
        laws.reverse()
    spec = replace(
        spec,
        compilation=compile_mujoco_xml(ET.tostring(xml, encoding="unicode"), require_asset=False),
        area_pressure_pairs=tuple(("pressure", a.identity.rigid_body) for a in assets),
        area_friction_laws=tuple(laws),
    )
    if release:
        world = replace(world, initial_state=MuJoCoInitialState((-0.002, 0.0), (0.0, 1.0)))
    return spec, world, tuple(assets)


def _flat_reference(pivot):
    # Pressure gradient 2.5 MPa/m, initially 1 mm compression over both rectangles.
    k = 2.5e6 * 4 * HALF_X * HALF_Y
    b = pivot - np.asarray(CENTERS)
    stiffness = k * np.array([[[1.0, lever], [lever, lever**2 + HALF_X**2 / 3]] for lever in b])
    preload = k * 0.001 * np.column_stack((np.ones(2), b))
    mass = np.array(((2.0, pivot), (pivot, INERTIA + pivot**2)))
    acc = np.linalg.solve(mass + H**2 * stiffness.sum(axis=0), preload.sum(axis=0) - (0.01, 0.0))
    assert acc[0] > 0.01 * np.linalg.inv(mass)[0, 0]
    assert H**2 * (abs(acc[0]) + (max(abs(b)) + HALF_X) * abs(acc[1])) < 0.001
    return acc, preload - H**2 * (stiffness @ acc)


def _audit(model, before, after, geometry, frame, body_reads, angular_velocity, release):
    import mujoco as mj

    receipt = after.last_step
    data = mj.MjData(model)
    data.qpos[:], data.qvel[:] = before.q, before.v
    mj.mj_forward(model, data)
    pivot = model.joint("pivot").id
    anchor, axis = data.xanchor[pivot], data.xaxis[pivot]
    raw = receipt.raw_contact_fields["area_friction"]
    assert len(raw) == len(geometry) == len(frame.area_catalogs[before.world]) == 2
    total = np.zeros(2)
    regions = {}
    combined, *supports = body_reads
    for row, surface, catalog in zip(raw, geometry, frame.area_catalogs[before.world], strict=True):
        _surface(row, surface, receipt)
        name = row["identity"].rigid_body
        idx = PLATES.index(name)
        support = supports[idx]
        points = row["points"].astype(float)
        assert catalog["identity"] == row["identity"] and catalog["sample_ids"] == row["sample_ids"]
        assert len(support.contacts) == len(points)
        assert all(c.identity == row["identity"] and c.group_member for c in support.contacts)
        arms = points - anchor
        jp = np.zeros((len(points), 3, 2))
        jp[:, 2, 0], jp[:, :, 1] = 1.0, np.cross(axis, arms)
        np.testing.assert_allclose(row["jacobian_a"], jp, atol=2e-7)
        np.testing.assert_array_equal(row["jacobian_b"], 0.0)
        velocity = jp @ after.v
        normals = row["normals"].astype(float)
        vn = np.einsum("ij,ij->i", velocity, normals)
        tangent = velocity - vn[:, None] * normals
        mu = (0.15, 0.35)[idx] if release else 0.0
        normal = np.maximum(0.0, -row["stiffness"] * (row["surrogate_gap"] + H * vn))
        friction = (
            -mu * normal[:, None] * tangent / np.sqrt(0.001**2 + (tangent**2).sum(axis=1))[:, None]
        )
        force = row["force_on_a"].astype(float)
        np.testing.assert_allclose(
            force, normal[:, None] * normals + friction, atol=2e-5, rtol=0.002
        )
        np.testing.assert_array_equal(row["force_on_b"], -row["force_on_a"])
        resultant = force.sum(axis=0)
        moment = np.cross(arms, force).sum(axis=0)
        np.testing.assert_allclose(support.total_force_world, -resultant, atol=1e-7)
        np.testing.assert_allclose(support.total_moment_world, -moment, atol=1e-7)
        np.testing.assert_allclose(
            combined.group_normal_loads[idx], normal.sum(), atol=1e-5, rtol=0.002
        )
        fq = np.array((resultant[2], axis @ moment))
        total += fq
        regions[name] = dict(
            samples=len(points),
            Fz=float(fq[0]),
            My=float(fq[1]),
            Fx=float(resultant[0]),
            area_m2=float(row["dA"].sum()),
        )
    assert set(regions) == set(PLATES)
    np.testing.assert_array_equal(combined.group_force_world, combined.total_force_world)
    np.testing.assert_allclose(
        [combined.area_force_world[2], axis @ combined.area_moment_world], total, atol=1e-7
    )
    np.testing.assert_allclose(
        np.sum([v.total_force_world for v in body_reads], axis=0), 0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        np.sum([v.total_moment_world for v in body_reads], axis=0), 0.0, atol=1e-7
    )
    (dry,) = receipt.constraints
    assert dry.constraint_type == 1 and dry.dof == 0
    mass = np.empty((2, 2))
    mj.mj_fullM(model, mass, data.qM)
    native_dry = np.clip((data.efc_aref[0] - receipt.qacc[0]) / data.efc_R[0], -0.01, 0.01)
    assert dry.force == pytest.approx(native_dry, abs=2e-6)
    applied = np.asarray(before.held_control.qfrc_applied)
    np.testing.assert_allclose(receipt.qfrc_constraint, total + (dry.force, 0), atol=2e-6)
    acc = np.linalg.solve(mass, total + (dry.force, 0) + applied - data.qfrc_bias)
    np.testing.assert_allclose(receipt.qacc, acc, atol=4e-5, rtol=3e-4)
    np.testing.assert_allclose(after.v, before.v + H * acc, atol=3e-6)
    np.testing.assert_allclose(after.q, before.q + H * after.v, atol=2e-7)
    np.testing.assert_allclose(angular_velocity, axis * after.v[1], atol=2e-6)
    if before.tick == 0 and not release:
        ref_acc, loads = _flat_reference(model.jnt_pos[pivot, 0])
        np.testing.assert_allclose(acc, ref_acc, rtol=0.0003, atol=4e-5)
        np.testing.assert_allclose(
            [[regions[name]["Fz"], regions[name]["My"]] for name in PLATES],
            loads,
            rtol=0.0003,
            atol=2e-6,
        )
    return dict(regions=regions, angle_rad=float(after.q[1]), omega_rad_s=float(after.v[1]))


@pytest.mark.cuda
@pytest.mark.parametrize(
    "release", (False, True), ids=("add-and-cancel", "release-and-reverse-order")
)
def test_two_supports_drive_one_hinge(release, tmp_path, monkeypatch):
    import mujoco as mj

    def no_cpu_step(*args, **kwargs):
        pytest.fail("only the CUDA scene may advance the state")

    monkeypatch.setattr(mj, "mj_step", no_cpu_step)
    settings = (
        ((41, 0.0, False), (7, 0.0, True)) if release else ((41, 0.03, False), (7, 0.0, False))
    )
    declarations = [_declarations(key, b, release, reverse) for key, b, reverse in settings]
    specs, worlds, assets = zip(*declarations, strict=True)
    models = {
        w.world: s.compilation.fresh_native_model() for s, w in zip(specs, worlds, strict=True)
    }
    by_model = {s.key: a for s, a in zip(specs, assets, strict=True)}
    history = []
    with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, by_model)
        batch.set_controls(
            {
                k: MuJoCoControl((), qfrc_applied=(-0.216 if release else 0.0, 0.0))
                for k in batch.world_keys
            }
        )
        with batch.prepare_device_workspace(
            include_bodies=True, include_contacts=True, include_area_pressure=True
        ) as ws:
            reader = BodyContactWrenchReader(
                ws, area_identities={k: tuple(p.identity for p in v) for k, v in by_model.items()}
            )
            for _ in range(8 if release else 3):
                before = batch.integration_snapshot()
                assert scene.step(H, expected_revision=batch.checkpoint_revision) == dict.fromkeys(
                    batch.world_keys, True
                )
                frame = ws.refresh(expected_revision=batch.checkpoint_revision)
                requests = []
                for key in batch.world_keys:
                    model = models[key]
                    b = model.jnt_pos[model.joint("pivot").id, 0]
                    anchor = (X0 + b, 0.0, 0.011 + before[key].q[0])
                    requests.append(BodyContactWrenchRequest(key, "pressure", anchor, PLATES))
                    requests.extend(
                        BodyContactWrenchRequest(key, name, anchor, ("pressure",))
                        for name in PLATES
                    )
                measurements = reader.measure_many(frame, tuple(requests))
                device = ws.readback(frame)
                record = {}
                for i, key in enumerate(batch.world_keys):
                    model = models[key]
                    record[key] = _audit(
                        model,
                        before[key],
                        batch.last_step.states[key],
                        scene.last_geometry[key],
                        frame,
                        measurements[3 * i : 3 * i + 3],
                        device["body_angular_velocity"][i, model.body("pressure").id],
                        release,
                    )
                history.append(record)
    (tmp_path / "two-supports.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if release:
        rows = [r[41]["regions"] for r in history]
        assert all(rows[0][name]["Fz"] > 0.001 for name in PLATES)
        assert all(r["right"]["samples"] > 0 and r["right"]["Fz"] > 0.01 for r in rows)
        assert rows[-1]["left"]["samples"] == 0 and rows[-1]["left"]["Fz"] == 0
        assert rows[-1]["right"]["Fz"] > rows[0]["right"]["Fz"]
        assert abs(rows[0]["right"]["Fx"]) > 0.001
        for record in history:
            for name in PLATES:
                original, rev = (record[k]["regions"][name] for k in (41, 7))
                assert original["samples"] == rev["samples"]
                for field in ("Fz", "My", "Fx", "area_m2"):
                    np.testing.assert_allclose(original[field], rev[field], rtol=3e-5, atol=2e-7)
            for field in ("omega_rad_s", "angle_rad"):
                assert record[41][field] == pytest.approx(record[7][field], abs=2e-6)
    else:
        add, cancel = (history[0][k]["regions"] for k in (41, 7))
        assert min(add[n]["My"] for n in PLATES) > 0.001
        assert cancel["left"]["My"] > 0.001 and cancel["right"]["My"] < -0.001
        assert abs(sum(cancel[n]["My"] for n in PLATES)) < 1e-7
        assert history[0][41]["omega_rad_s"] > 0.005 and abs(history[0][7]["omega_rad_s"]) < 1e-6


@pytest.mark.cuda
def test_second_support_exceeding_shared_capacity_preserves_last_step():
    spec, world, assets = _declarations(41, 0.0)
    # Six triangle samples fit alongside one native dry row. Two finite plates
    # demand at least twelve at the first compressed entry.
    spec = replace(spec, constraint_capacity=7)
    world = replace(world, initial_state=MuJoCoInitialState((-0.0008, 0.0), (-0.08, 0.0)))
    with MuJoCoTyxHeterogeneousBatch((spec,), (world,), mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: assets})
        assert scene.step(H, expected_revision=batch.checkpoint_revision) == {41: True}
        last = batch.last_step
        rows = last.states[41].last_step.raw_contact_fields["area_friction"]
        assert len(rows) == 2 and all(len(row["normal_force"]) == 0 for row in rows)
        assert -0.003 < last.states[41].q[0] < -0.001
        geometry, rev, saved = (
            scene.last_geometry,
            batch.checkpoint_revision,
            scene.checkpoint().to_bytes(),
        )
        with pytest.raises(ValueError, match="area-pressure samples exceed constraint capacity"):
            scene.step(H, expected_revision=rev)
        assert batch.checkpoint_revision == rev and batch.last_step is last
        assert scene.last_geometry is geometry and scene.checkpoint().to_bytes() == saved

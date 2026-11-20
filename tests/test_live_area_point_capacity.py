"""Real automatic Scene geometry beyond 128 points, with exact planar mechanics."""

import numpy as np
import pytest
from test_mujoco_area_friction_lifecycle import _same
from test_mujoco_area_friction_live import _surface as _audit_surface
from test_mujoco_tyx_area_pressure_heterogeneous_scene import H, _compilation
from test_mujoco_tyx_area_pressure_scene import AREA, MASS, PAIR, K, _depth, asset

import phymeo.backends.mujoco_area_pressure_scene as geometry_module
from phymeo.backends.area_friction_readback import AreaFrictionEntryReader
from phymeo.backends.mujoco_area_friction import AreaFrictionPairLaw
from phymeo.backends.mujoco_area_pressure_scene import BodyAreaPressureAsset
from phymeo.backends.mujoco_heterogeneous import (
    MuJoCoTyxHeterogeneousBatch,
    MuJoCoTyxModelSpec,
    MuJoCoTyxWorldSpec,
)
from phymeo.backends.mujoco_heterogeneous_area_scene import (
    MuJoCoTyxHeterogeneousAreaPressureScene,
    MuJoCoTyxHeterogeneousAreaSceneCheckpoint,
)
from phymeo.backends.mujoco_runtime import MuJoCoInitialState
from phymeo.geometry.area_pressure_intersection import RigidPressureMesh, TetraPressureMesh
from phymeo.training.resident_stepper import PhysicsStepFailure

DENSE_POINTS = 216
COARSE_POINTS = 6
GRADIENT = K / AREA


def _subdivided_asset(divisions):
    """Same physical face and affine pressure as the established bilateral asset.

    Split the shared tetrahedral edge 0--1 conformingly in BOTH incident cells.
    The midpoint pressure is its affine value, so the source field is unchanged.
    A genuine x=0 cell seam cuts eight of the 64 dense rigid triangles: 72 output
    triangles produce 216 points. No face or contact triangle is duplicated.
    """
    base = asset()
    pressure = base.pressure_local
    vertices = np.vstack(
        (pressure.world_vertices, (pressure.world_vertices[0] + pressure.world_vertices[1]) / 2)
    )
    cells = ((0, 5, 2, 4), (5, 1, 2, 4), (0, 2, 3, 4), (0, 3, 5, 4), (5, 3, 1, 4), (1, 3, 2, 4))
    refined = TetraPressureMesh(
        (*pressure.node_ids, "edge-midpoint"),
        vertices,
        cells,
        (*pressure.nodal_pressure, 0.0),
        ("bottom-left", "bottom-right", "side-a", "side-b-left", "side-b-right", "slant"),
    )
    a, b, c = base.rigid_local.world_vertices
    positions, indices = [], {}
    for i in range(divisions + 1):
        for j in range(divisions + 1 - i):
            point = a + (i / divisions) * (b - a) + (j / divisions) * (c - a)
            if 2 * i + j == divisions:
                point[0] = 0.0  # Exact semantic seam, avoiding an artificial roundoff sliver.
            indices[i, j] = len(positions)
            positions.append(point)
    faces = []
    for i in range(divisions):
        for j in range(divisions - i):
            faces.append((indices[i, j], indices[i + 1, j], indices[i, j + 1]))
            if i + j < divisions - 1:
                faces.append((indices[i + 1, j], indices[i + 1, j + 1], indices[i, j + 1]))
    mesh = RigidPressureMesh(
        tuple(f"grid-node-{i}" for i in range(len(positions))),
        np.asarray(positions),
        tuple(faces),
        tuple(f"grid-face-{i}" for i in range(len(faces))),
    )
    assert len(faces) == divisions**2 and len(set(map(tuple, faces))) == len(faces)
    return BodyAreaPressureAsset(base.identity, refined, mesh)


def _spec(key, capacity, geometry):
    return MuJoCoTyxModelSpec(
        key,
        _compilation(),
        contact_capacity=1,
        constraint_capacity=capacity,
        area_pressure_pairs=(PAIR,),
        area_friction_laws=(AreaFrictionPairLaw(geometry.identity, 0.5, 0.001),),
    )


def _world(key, spec):
    return MuJoCoTyxWorldSpec(key, spec.key, MuJoCoInitialState((-0.0025, 0.0), (-0.2, 0.0)))


def _audit(before, after, geometry, expected_count):
    receipt = after.last_step
    assert receipt.contacts == receipt.constraints == ()
    (row,) = receipt.raw_contact_fields["area_friction"]
    _audit_surface(row, geometry, receipt)
    assert len(geometry.surface.triangles) * 3 == len(row["sample_ids"]) == expected_count
    assert len(set(row["sample_ids"])) == expected_count
    assert np.all(row["dA"] > 0) and row["dA"].sum() == pytest.approx(AREA, rel=1e-10)
    assert set(cell for cell, _face in row["source_ids"]) == {"bottom-left", "bottom-right"}
    h, depth = float(np.float32(H)), _depth(before.q)
    relative0 = before.v[0] - before.v[1]
    force = K * (depth - h * relative0) / (1 + h * h * K * np.sum(1 / MASS))
    assert force > 0
    expected_v = before.v + h * force * np.array([1 / MASS[0], -1 / MASS[1]])
    np.testing.assert_allclose(after.v, expected_v, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(MASS @ after.v, MASS @ before.v, rtol=0, atol=2e-6)
    # Every point, including the final point beyond 128, carries its true dA.
    fn = GRADIENT * row["dA"] * (depth - h * (after.v[0] - after.v[1]))
    assert fn[-1] > 0 and row["normal_force"][-1] > 0
    np.testing.assert_allclose(row["normal_force"], fn, rtol=2e-4, atol=2e-6)
    expected = np.column_stack((np.zeros(expected_count), np.zeros(expected_count), fn))
    np.testing.assert_allclose(row["force_on_a"], expected, rtol=2e-4, atol=2e-6)
    np.testing.assert_array_equal(row["force_on_b"], -row["force_on_a"])
    np.testing.assert_allclose(row["tangential_force"], 0.0, atol=1e-8)
    total_a, total_b = row["force_on_a"].sum(axis=0), row["force_on_b"].sum(axis=0)
    np.testing.assert_allclose(total_a, [0.0, 0.0, force], rtol=2e-4, atol=1e-5)
    np.testing.assert_array_equal(total_b, -total_a)
    np.testing.assert_allclose(receipt.qfrc_constraint, [force, -force], rtol=2e-4, atol=1e-5)
    torque = np.cross(row["points"], expected)
    np.testing.assert_allclose(row["torque_on_a_world_origin"], torque, rtol=2e-4, atol=2e-7)
    np.testing.assert_allclose(torque.sum(axis=0), [-0.035 * force, 0.0, 0.0], rtol=2e-4, atol=1e-6)
    assert after.tick == before.tick + 1
    return row


def _read(ws, batch, identities, states):
    frame = ws.refresh(expected_revision=batch.checkpoint_revision)
    arrays = ws.readback(frame)
    decoded = AreaFrictionEntryReader(ws, identities).validate(arrays, frame)
    for key, state in states.items():
        (row,) = state.last_step.raw_contact_fields["area_friction"]
        samples = decoded[key].samples
        assert len(samples) == len(row["sample_ids"])
        assert samples[-1].sample_id == row["sample_ids"][-1]
        assert samples[-1].normal_force_n > 0
        np.testing.assert_array_equal([s.area_m2 for s in samples], row["dA"])
        np.testing.assert_array_equal([s.point_world for s in samples], row["points"])
        np.testing.assert_array_equal([s.force_on_a_world for s in samples], row["force_on_a"])
        np.testing.assert_array_equal([s.normal_force_n for s in samples], row["normal_force"])
    return frame, arrays


@pytest.mark.cuda
def test_scene_generates_216_live_points_matches_coarse_physics_and_restores_all_samples():
    coarse, dense = _subdivided_asset(1), _subdivided_asset(8)
    specs = (_spec("coarse", 256, coarse), _spec("dense", 256, dense))
    worlds = tuple(_world(key, spec) for key, spec in zip((101, 102), specs, strict=True))
    assets = {"coarse": (coarse,), "dense": (dense,)}
    identities = {key: (rows[0].identity,) for key, rows in assets.items()}
    with MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="cuda") as batch:
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
        with batch.prepare_device_workspace(include_bodies=True, include_area_pressure=True) as ws:
            before = batch.integration_snapshot()
            assert scene.step(H, expected_revision=batch.checkpoint_revision) == {
                101: True,
                102: True,
            }
            accepted = batch.integration_snapshot()
            rows = {
                key: _audit(before[key], accepted[key], scene.last_geometry[key][0], count)
                for key, count in ((101, COARSE_POINTS), (102, DENSE_POINTS))
            }
            assert 200 <= len(rows[102]["sample_ids"]) <= 256
            np.testing.assert_allclose(accepted[101].v, accepted[102].v, rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(
                rows[101]["force_on_a"].sum(axis=0),
                rows[102]["force_on_a"].sum(axis=0),
                rtol=2e-4,
                atol=1e-5,
            )
            _frame, saved_arrays = _read(ws, batch, identities, accepted)
            saved = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(
                scene.checkpoint().to_bytes()
            )
            assert scene.step(H, expected_revision=batch.checkpoint_revision) == {
                101: True,
                102: True,
            }
            advanced = batch.integration_snapshot()
            for key, count in ((101, COARSE_POINTS), (102, DENSE_POINTS)):
                _audit(accepted[key], advanced[key], scene.last_geometry[key][0], count)
            stale_frame, advanced_arrays = _read(ws, batch, identities, advanced)
            scene.restore(saved, expected_revision=batch.checkpoint_revision)
            _same(batch.integration_snapshot(), accepted)
            with pytest.raises(ValueError):
                ws.readback(stale_frame)
            _restored_frame, restored_arrays = _read(ws, batch, identities, accepted)
            _same(restored_arrays, saved_arrays)
            assert scene.step(H, expected_revision=batch.checkpoint_revision) == {
                101: True,
                102: True,
            }
            replay = batch.integration_snapshot()
            _same(replay, advanced, fresh_stamp=True)
            _replay_frame, replay_arrays = _read(ws, batch, identities, replay)
            _same(replay_arrays, advanced_arrays)


@pytest.mark.cuda
def test_215_capacity_rejects_real_216_point_scene_and_retains_loaded_publication(monkeypatch):
    coarse, dense = _subdivided_asset(1), _subdivided_asset(8)
    spec = _spec("one-short", DENSE_POINTS - 1, coarse)
    with MuJoCoTyxHeterogeneousBatch((spec,), (_world(101, spec),), mode="cuda") as batch:
        coarse_scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (coarse,)})
        dense_scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {spec.key: (dense,)})
        identities = {spec.key: (coarse.identity,)}
        with batch.prepare_device_workspace(include_bodies=True, include_area_pressure=True) as ws:
            before = batch.integration_snapshot()[101]
            assert coarse_scene.step(H, expected_revision=batch.checkpoint_revision) == {101: True}
            accepted = batch.integration_snapshot()
            _audit(before, accepted[101], coarse_scene.last_geometry[101][0], COARSE_POINTS)
            frame, arrays = _read(ws, batch, identities, accepted)
            revision, publication = batch.checkpoint_revision, batch.last_step
            checkpoint = batch.checkpoint().to_bytes()
            geometry, dense_geometry = coarse_scene.last_geometry, dense_scene.last_geometry
            measured = []
            original = geometry_module._intersect_checked

            def observe_actual_geometry(*args, **kwargs):
                result = original(*args, **kwargs)
                measured.append(result)
                return result  # Observe the real clipped surface without altering any output.

            with monkeypatch.context() as patch:
                patch.setattr(geometry_module, "_intersect_checked", observe_actual_geometry)
                with pytest.raises((ValueError, PhysicsStepFailure), match="capacity"):
                    dense_scene.step(H, expected_revision=revision)
            assert len(measured) == 1 and 3 * len(measured[0].surface.triangles) == DENSE_POINTS
            assert all(np.all(t.vertex_pressure > 0) for t in measured[0].surface.triangles)
            assert batch.checkpoint_revision == revision and batch.last_step is publication
            assert batch.checkpoint().to_bytes() == checkpoint
            _same(batch.integration_snapshot(), accepted)
            assert (
                coarse_scene.last_geometry is geometry
                and dense_scene.last_geometry is dense_geometry
            )
            _same(ws.readback(frame), arrays)
            assert coarse_scene.step(H, expected_revision=revision) == {101: True}
            _audit(
                accepted[101],
                batch.integration_snapshot()[101],
                coarse_scene.last_geometry[101][0],
                COARSE_POINTS,
            )

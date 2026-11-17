"""Bounded automatic heterogeneous I1 geometry -> shared TYX dynamics tests."""

from contextlib import ExitStack
from dataclasses import replace

import numpy as np
import pytest
from test_mujoco_tyx_area_pressure_heterogeneous import NATIVE_XML, _exact_arrays
from test_mujoco_tyx_area_pressure_lifecycle import _readonly, _same
from test_mujoco_tyx_area_pressure_scene import AREA, MASS, PAIR, _depth, asset

import phymeo.backends.mujoco_area_pressure_scene as geometry_module
from phymeo.backends._mujoco_tyx_runtime import MuJoCoTyxRuntime
from phymeo.backends.mujoco_area_pressure_scene import (
    BodyAreaPressureAsset,
    MuJoCoTyxAreaPressureScene,
)
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
from phymeo.builders.mujoco_compiler import compile_mujoco_xml
from phymeo.training.resident_stepper import PhysicsStepFailure

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
H = 0.0005
KEYS = (91, 7, 41, 11, 305)
AREA_KEYS = (91, 41, 11, 305)


def _compilation(rotating=False):
    """Reuse runtime_for's exact XML physics without creating temporary devices."""
    kind, axis = ("hinge", "0 1 0") if rotating else ("slide", "0 0 1")
    height = 0.027 if rotating else 0.002
    xml = f'''<mujoco model="current-area-surface">
    <compiler angle="radian" fusestatic="false"/>
    <option timestep="{H}" gravity="0 0 0" integrator="Euler"
      iterations="128" tolerance="1e-10" cone="pyramidal" jacobian="dense"/>
    <worldbody>
      <body name="pressure" pos="0 0 {height}">
        <joint name="a" type="{kind}" axis="{axis}" damping="0" armature="0"/>
        <inertial pos="0 0 0" mass=".5" diaginertia=".02 .02 .02"/>
        <geom type="sphere" size=".01" contype="0" conaffinity="0"/>
      </body>
      <body name="plate">
        <joint name="b" type="slide" axis="0 0 1" damping="0" armature="0"/>
        <inertial pos="0 0 0" mass="1.25" diaginertia=".03 .03 .03"/>
        <geom type="sphere" size=".01" contype="0" conaffinity="0"/>
      </body>
    </worldbody></mujoco>'''
    return compile_mujoco_xml(xml, require_asset=False)


def _declarations():
    slide, hinge = _compilation(), _compilation(rotating=True)
    native = compile_mujoco_xml(
        NATIVE_XML.replace('timestep="0.002"', f'timestep="{H}"'), require_asset=False
    )
    models = tuple(
        MuJoCoTyxModelSpec(
            name,
            source,
            contact_capacity=4,
            constraint_capacity=32,
            area_pressure_pairs=() if name == "native" else (PAIR,),
        )
        for name, source in (
            ("soft", slide),
            ("stiff", slide),
            ("native", native),
            ("hinge", hinge),
        )
    )
    soft = asset()
    stiff = BodyAreaPressureAsset(
        replace(
            soft.identity,
            compliant_material="linear-pressure-stiff",
            rigid_material="shifted-finite-face",
        ),
        replace(soft.pressure_local, nodal_pressure=soft.pressure_local.nodal_pressure * 1.5),
        replace(soft.rigid_local, world_vertices=soft.rigid_local.world_vertices + [0.01, 0, 0]),
    )
    assets = {"soft": (soft,), "stiff": (stiff,), "hinge": (asset(rotating=True),)}
    starts = {
        91: ("hinge", [0.04, 0], [0.25, 0]),
        7: ("native", [0], [-0.01]),
        41: ("soft", [-0.0025, 0], [-0.2, 0]),
        11: ("stiff", [-0.0025, 0], [-0.2, 0]),
        305: ("soft", [0, 0], [-0.2, 0]),
    }
    worlds = tuple(
        MuJoCoTyxWorldSpec(key, starts[key][0], MuJoCoInitialState(starts[key][1], starts[key][2]))
        for key in KEYS
    )
    return models, worlds, assets


def _make(stack, mode):
    models, worlds, assets = _declarations()
    batch = stack.enter_context(MuJoCoTyxHeterogeneousBatch(models, worlds, mode=mode))
    scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets)
    workspace = stack.enter_context(batch.prepare_device_workspace(include_area_pressure=True))
    return batch, scene, workspace, models, worlds, assets


def _step(scene):
    assert scene.step(H, expected_revision=scene.batch.checkpoint_revision) == dict.fromkeys(
        KEYS, True
    )


def _refresh(batch, workspace):
    frame = workspace.refresh(expected_revision=batch.checkpoint_revision)
    return frame, workspace.readback(frame)


def _retained(batch, scene, workspace, state, geometry, frame, arrays, capsule, revision):
    assert batch.checkpoint_revision == revision
    assert batch.checkpoint().to_bytes() == capsule
    _same(batch.integration_snapshot(), state)
    assert scene.last_geometry is geometry
    _exact_arrays(workspace.readback(frame), arrays)
    for key in KEYS:
        assert batch.integration_snapshot()[key].last_step is state[key].last_step


@pytest.mark.parametrize("mode", MODES)
def test_twenty_steps_distinct_assets_in_shared_bucket_match_single_scenes(mode):
    with ExitStack() as stack:
        batch, scene, ws, models, worlds, _ = _make(stack, mode)
        assert batch.layout[41].bucket == batch.layout[11].bucket == batch.layout[305].bucket
        assert len({row.bucket for row in batch.layout.values()}) == 3
        singles, single_scenes = {}, {}
        for world in worlds:
            spec = next(item for item in models if item.key == world.model_key)
            single = stack.enter_context(
                MuJoCoTyxRuntime(
                    spec.compilation,
                    mode=mode,
                    world_keys=(world.world,),
                    initial_states={world.world: world.initial_state},
                    contact_capacity=spec.contact_capacity,
                    constraint_capacity=spec.constraint_capacity,
                    area_pressure_pairs=spec.area_pressure_pairs,
                )
            )
            singles[world.world] = single
            if spec.area_pressure_pairs:
                single_scenes[world.world] = MuJoCoTyxAreaPressureScene(
                    single, scene.assets[spec.key]
                )
        signatures = {key: [] for key in AREA_KEYS}
        seen_empty, seen_loaded = set(), set()
        reduced = 1 / np.sum(1 / MASS)
        for step in range(20):
            before = batch.integration_snapshot()
            stamps = {key: batch.area_pressure_configuration_id(key) for key in AREA_KEYS}
            _step(scene)
            after = batch.integration_snapshot()
            assert tuple(scene.last_geometry) == AREA_KEYS
            for key, single in singles.items():
                if key in single_scenes:
                    single_scenes[key].step(H, expected_revision=single.checkpoint_revision)
                    _same(scene.last_geometry[key], single_scenes[key].last_geometry[key])
                else:
                    single.step(H)
                _same(after[key], single.integration_snapshot()[key])
            frame, arrays = _refresh(batch, ws)
            assert frame.area_catalogs[7] == ()
            for key in AREA_KEYS:
                row = KEYS.index(key)
                intersection = scene.last_geometry[key][0]
                surface = intersection.surface
                raw = after[key].last_step.raw_contact_fields["area_pressure"][0]
                stamp = surface.stamp
                assert (
                    stamp.world_id,
                    stamp.step_index,
                    stamp.sample_time_s,
                    stamp.configuration_id,
                ) == (str(key), before[key].tick, before[key].time, stamps[key])
                model_key = batch.layout[key].model_key
                assert raw["identity"] == scene.assets[model_key][0].identity == surface.identity
                assert raw["stamp"] == stamp
                assert frame.area_catalogs[key][0]["stamp"] == stamp
                np.testing.assert_array_equal(
                    arrays["area_meta"][row, :2], [1, len(raw["sample_ids"])]
                )
                points = raw["points"]
                if len(points):
                    np.testing.assert_allclose(points[:, 2], before[key].q[1], rtol=0, atol=2e-8)
                    if np.sum(raw["normal_force"]) > 0:
                        seen_loaded.add(key)
                else:
                    seen_empty.add(key)
                signatures[key].append((points.tobytes(), raw["surrogate_gap"].tobytes()))
                if key != 91:
                    depth, velocity = _depth(before[key].q), before[key].v
                    stiffness = 13500.0 if model_key == "stiff" else 9000.0
                    expected_force = (
                        max(
                            0.0,
                            stiffness
                            * (depth - H * (velocity[0] - velocity[1]))
                            / (1 + H * H * stiffness / reduced),
                        )
                        if depth >= 0
                        else 0.0
                    )
                    force = float(np.sum(raw["normal_force"]))
                    assert abs(force - expected_force) <= 0.001 + 0.001 * abs(expected_force)
                    expected_v = velocity + H * expected_force * np.array(
                        [1 / MASS[0], -1 / MASS[1]]
                    )
                    np.testing.assert_allclose(after[key].v, expected_v, rtol=0, atol=3e-5)
                    assert np.sum(raw["dA"]) == pytest.approx(
                        AREA if depth >= 0 else 0, rel=1e-10, abs=1e-12
                    )
                elif len(points):
                    angle = before[key].q[0]
                    local_z = np.sin(angle) * points[:, 0] + np.cos(angle) * (
                        points[:, 2] - float(np.float32(0.027))
                    )
                    np.testing.assert_allclose(
                        -raw["surrogate_gap"],
                        (local_z + 0.025) / np.cos(angle),
                        rtol=3e-4,
                        atol=2e-8,
                    )
            if step == 0:
                a = after[41].last_step.raw_contact_fields["area_pressure"][0]["source_quadrature"]
                b = after[11].last_step.raw_contact_fields["area_pressure"][0]["source_quadrature"]
                np.testing.assert_allclose(b.gradient, 1.5 * a.gradient, rtol=1e-12, atol=1e-10)
                assert not np.array_equal(a.points, b.points)
                assert not scene.last_geometry[305][0].surface.triangles
                assert after[7].last_step.contacts
        assert {41, 11, 91}.issubset(seen_loaded) and 305 in seen_empty
        for key in (41, 11, 91):
            assert len(set(signatures[key])) > 1
        _readonly(scene.assets)
        _readonly(scene.last_geometry)


@pytest.mark.parametrize("mode", MODES)
def test_asset_scope_ownership_preflight_and_changed_asset_checkpoint_refusal(mode, monkeypatch):
    with ExitStack() as stack:
        batch, scene, ws, _, _, supplied = _make(stack, mode)
        assert scene.batch is batch and set(scene.assets) == {"soft", "stiff", "hinge"}
        for name in supplied:
            assert scene.assets[name] is not supplied[name]
            assert scene.assets[name][0] is not supplied[name][0]
        owned = dict(scene.assets)
        supplied.clear()
        assert set(scene.assets) == set(owned)
        _readonly(scene.assets)
        bad_assets = [
            {key: value for key, value in owned.items() if key != "stiff"},
            {**owned, "native": owned["soft"]},
            {**owned, "stiff": ()},
            {
                **owned,
                "stiff": (
                    replace(
                        owned["stiff"][0],
                        identity=replace(owned["stiff"][0].identity, compliant_body="missing"),
                    ),
                ),
            },
            {
                **owned,
                "stiff": (
                    replace(
                        owned["stiff"][0],
                        identity=replace(
                            owned["stiff"][0].identity,
                            compliant_body="plate",
                            rigid_body="pressure",
                        ),
                    ),
                ),
            },
        ]
        original = batch.checkpoint().to_bytes()
        for bad in bad_assets:
            with pytest.raises(ValueError):
                MuJoCoTyxHeterogeneousAreaPressureScene(batch, bad)
            assert batch.checkpoint().to_bytes() == original
        for revision in (False, 0.0, batch.checkpoint_revision + 1):
            with pytest.raises(ValueError):
                scene.step(H, expected_revision=revision)
        with pytest.raises(ValueError):
            scene.step(H * 2, expected_revision=batch.checkpoint_revision)
        assert batch.checkpoint().to_bytes() == original
        _step(scene)
        checkpoint = scene.checkpoint()
        frame, arrays = _refresh(batch, ws)
        state, geometry = batch.integration_snapshot(), scene.last_geometry
        capsule, revision = batch.checkpoint().to_bytes(), batch.checkpoint_revision
        stiff = owned["stiff"][0]
        replacements = (
            replace(
                stiff,
                pressure_local=replace(
                    stiff.pressure_local, nodal_pressure=stiff.pressure_local.nodal_pressure * 1.1
                ),
            ),
            replace(
                stiff,
                rigid_local=replace(
                    stiff.rigid_local,
                    world_vertices=stiff.rigid_local.world_vertices + [0.001, 0, 0],
                ),
            ),
        )
        for changed in replacements:
            other = MuJoCoTyxHeterogeneousAreaPressureScene(batch, {**owned, "stiff": (changed,)})
            assert other.checkpoint().asset_sha256 != checkpoint.asset_sha256
            with monkeypatch.context() as patch:
                patch.setattr(
                    batch,
                    "restore",
                    lambda *a, **k: pytest.fail("asset mismatch reached physics restore"),
                )
                with pytest.raises(ValueError, match="asset binding"):
                    other.restore(checkpoint, expected_revision=revision)
            _retained(batch, scene, ws, state, geometry, frame, arrays, capsule, revision)


@pytest.mark.parametrize("mode", MODES)
def test_partial_reset_and_scene_capsule_restore_resolve_new_owners(mode, tmp_path):
    with ExitStack() as stack:
        batch, scene, ws, _, _, _ = _make(stack, mode)
        _step(scene)
        _step(scene)
        saved_frame, saved = _refresh(batch, ws)
        path = tmp_path / "heterogeneous-area-step-2.checkpoint"
        path.write_bytes(scene.checkpoint().to_bytes())
        checkpoint = MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(path.read_bytes())
        assert checkpoint.to_bytes() == path.read_bytes() and len(checkpoint.asset_sha256) == 64
        assert checkpoint.batch_checkpoint.to_bytes() == batch.checkpoint().to_bytes()
        damaged = bytearray(checkpoint.to_bytes())
        damaged[-1] ^= 1
        with pytest.raises(ValueError, match="checksum"):
            MuJoCoTyxHeterogeneousAreaSceneCheckpoint.from_bytes(damaged)
        _step(scene)
        reference_state, reference_geometry = batch.integration_snapshot(), scene.last_geometry
        _, reference_arrays = _refresh(batch, ws)
        owners = tuple(batch._buckets)
        scene.reset((41, 7))
        assert any(old is not new for old, new in zip(owners, batch._buckets, strict=True))
        assert dict(scene.last_geometry) == {}
        with pytest.raises(ValueError):
            ws.readback(saved_frame)
        reset_frame, reset = _refresh(batch, ws)
        np.testing.assert_array_equal(reset["area_meta"][KEYS.index(41)], 0)
        assert reset_frame.area_catalogs[41] == ()
        for key in (91, 11, 305):
            row = KEYS.index(key)
            for name in ("area_meta", "area_ids", "area_data", "area_measure_words"):
                assert reset[name][row].tobytes() == reference_arrays[name][row].tobytes()
        owners = tuple(batch._buckets)
        scene.restore(checkpoint, expected_revision=batch.checkpoint_revision)
        assert all(old is not new for old, new in zip(owners, batch._buckets, strict=True))
        assert dict(scene.last_geometry) == {}
        with pytest.raises(ValueError):
            ws.readback(reset_frame)
        restored_frame, restored = _refresh(batch, ws)
        _exact_arrays(restored, saved)
        assert restored_frame.area_catalogs == saved_frame.area_catalogs
        _step(scene)
        _same(batch.integration_snapshot(), reference_state)
        _same(scene.last_geometry, reference_geometry)
        repeated_frame, repeated_arrays = _refresh(batch, ws)
        _exact_arrays(repeated_arrays, reference_arrays)
        geometry, revision = scene.last_geometry, batch.checkpoint_revision
        scene.reset(())
        assert scene.last_geometry is geometry and batch.checkpoint_revision == revision
        states = batch.integration_snapshot()
        batch.set_controls({key: states[key].held_control for key in KEYS})
        assert dict(scene.last_geometry) == {}
        with pytest.raises(ValueError):
            ws.readback(repeated_frame)
        _readonly(geometry)


@pytest.mark.parametrize("mode", MODES)
def test_final_geometry_capacity_and_physics_failures_keep_all_publications(mode, monkeypatch):
    with ExitStack() as stack:
        batch, scene, ws, _, _, _ = _make(stack, mode)
        _step(scene)
        for fault in ("geometry", "capacity", "physics"):
            frame, arrays = _refresh(batch, ws)
            state, geometry = batch.integration_snapshot(), scene.last_geometry
            capsule, revision = batch.checkpoint().to_bytes(), batch.checkpoint_revision
            seen = []
            with monkeypatch.context() as patch:
                if fault == "physics":
                    download = batch._buckets[-1]._device.download

                    def late_download():
                        download()
                        seen.append(91)
                        raise RuntimeError("deliberate final scene physics failure")

                    patch.setattr(batch._buckets[-1]._device, "download", late_download)
                    with pytest.raises(PhysicsStepFailure) as caught:
                        _step(scene)
                    assert caught.value.accepted_worlds == dict.fromkeys(KEYS, False)
                else:
                    original = geometry_module.intersect_area_pressure

                    def late_geometry(*args, **kwargs):
                        result = original(*args, **kwargs)
                        key = int(kwargs["stamp"].world_id)
                        seen.append(key)
                        if key != 91:
                            return result
                        if fault == "geometry":
                            raise RuntimeError("deliberate final geometry failure")
                        # Synthetic producer fault only: exceed cap32 with 33
                        # samples. These duplicate triangles are not an oracle.
                        assert result.surface.triangles
                        triangle = result.surface.triangles[0]
                        triangles = tuple(
                            replace(triangle, triangle_id=f"overflow-{i}") for i in range(11)
                        )
                        return replace(
                            result,
                            surface=replace(result.surface, triangles=triangles),
                            candidate_pairs=11,
                            polygon_count=11,
                            shared_face_duplicates=0,
                        )

                    patch.setattr(geometry_module, "intersect_area_pressure", late_geometry)
                    with pytest.raises(
                        RuntimeError if fault == "geometry" else ValueError,
                        match="final geometry" if fault == "geometry" else "capacity",
                    ):
                        _step(scene)
                    assert seen == [41, 11, 305, 91]
            assert seen[-1] == 91
            _retained(batch, scene, ws, state, geometry, frame, arrays, capsule, revision)
            _step(scene)
            assert all(
                value.tick == state[key].tick + 1
                for key, value in batch.integration_snapshot().items()
            )
            assert tuple(scene.last_geometry) == AREA_KEYS

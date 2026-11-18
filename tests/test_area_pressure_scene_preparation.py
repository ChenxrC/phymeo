"""One complete validation of each distinct current mesh side before shared execution.

Spies call the original validators and never manufacture geometry or forces.
The large-translation fault tests FP64 world-coordinate collapse, not a changed
physical fixture. Initialization and reset construction are outside the counts.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pytest
from test_area_pressure_intersection import _case, _identity, _stamp
from test_area_pressure_step_mesh_reuse import uncached_inputs
from test_mujoco_tyx_area_pressure_heterogeneous import _exact_arrays
from test_mujoco_tyx_area_pressure_heterogeneous_scene import (
    AREA_KEYS,
    KEYS,
    MODES,
    H,
    _make,
    _refresh,
    _retained,
    _step,
)
from test_mujoco_tyx_area_pressure_lifecycle import _same

import phymeo.backends.mujoco_area_pressure_scene as scene_geometry
import phymeo.geometry.area_pressure_intersection as mesh_geometry
from phymeo.backends.mujoco_heterogeneous import MuJoCoTyxHeterogeneousBatch
from phymeo.backends.mujoco_heterogeneous_area_scene import (
    MuJoCoTyxHeterogeneousAreaPressureScene,
)
from phymeo.backends.tyx_area_pressure_intersection import TyxAreaPressureIntersector
from phymeo.geometry.area_pressure_intersection import AreaPressureIntersectionError


@contextmanager
def _validation_calls(monkeypatch):
    counts = {"pressure": 0, "rigid": 0}
    tet_geometry = mesh_geometry._tet_geometry
    rigid_geometry = mesh_geometry._rigid_geometry

    def pressure(*args, **kwargs):
        counts["pressure"] += 1
        return tet_geometry(*args, **kwargs)

    def rigid(*args, **kwargs):
        counts["rigid"] += 1
        return rigid_geometry(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(mesh_geometry, "_tet_geometry", pressure)
        patch.setattr(mesh_geometry, "_rigid_geometry", rigid)
        yield counts


def _intersector(stack, mode, route):
    if route == "host":
        return None
    return stack.enter_context(TyxAreaPressureIntersector(mode=mode, capacity=8))


def _signature(intersection):
    return tuple(
        (
            triangle.vertices.tobytes(),
            triangle.vertex_pressure.tobytes(),
            triangle.pressure_gradient.tobytes(),
        )
        for triangle in intersection.surface.triangles
    )


def _step_once_and_check_validation(monkeypatch, scene):
    before = scene.batch.integration_snapshot((41, 305))
    # Only the two soft worlds can share a plate. Both use the same shape,
    # material and orientation, and their second slide translates the plate in Z.
    shared_plate = before[41].q[1] == before[305].q[1]
    with _validation_calls(monkeypatch) as counts:
        _step(scene)
    assert counts == {"pressure": 4, "rigid": 3 if shared_plate else 4}


def _compare_current_outputs(
    batch, scene, workspace, reference_batch, reference_scene, reference_ws
):
    _same(batch.integration_snapshot(), reference_batch.integration_snapshot())
    _same(scene.last_geometry, reference_scene.last_geometry)
    frame, arrays = _refresh(batch, workspace)
    reference_frame, reference_arrays = _refresh(reference_batch, reference_ws)
    rows = [reference_batch.world_keys.index(key) for key in batch.world_keys]
    _exact_arrays(arrays, {name: value[rows] for name, value in reference_arrays.items()})
    _same(frame.area_catalogs, reference_frame.area_catalogs)
    state = batch.integration_snapshot()
    assert frame.ticks == tuple(state[key].tick for key in batch.world_keys)
    return frame, arrays


def test_owned_raw_declaration_copies_inputs_and_defers_complete_geometry_checks(monkeypatch):
    def inputs():
        pressure, rigid, domain, *_ = _case("full_body")
        pressure = dict(
            pressure,
            world_vertices=np.array(pressure["world_vertices"], dtype=np.float64),
            cells=np.array(pressure["cells"], dtype=np.int64),
            nodal_pressure=np.array(pressure["nodal_pressure"], dtype=np.float64),
            pressure_domain=domain,
        )
        rigid = dict(
            rigid,
            world_vertices=np.array(rigid["world_vertices"], dtype=np.float64),
            faces=np.array(rigid["faces"], dtype=np.int64),
        )
        return pressure, rigid

    pressure_input, rigid_input = inputs()
    with _validation_calls(monkeypatch) as counts:
        raw = mesh_geometry._AreaPressureGeometryInput(
            pressure_input, rigid_input, _identity(), _stamp()
        )
        assert counts == {"pressure": 0, "rigid": 0}
        captured = {}
        for name, declaration in (("pressure", raw.pressure), ("rigid", raw.rigid)):
            assert isinstance(declaration, MappingProxyType)
            with pytest.raises(TypeError):
                declaration["node_ids"] = ()
            captured[name] = {}
            for field, value in declaration.items():
                if isinstance(value, np.ndarray):
                    captured[name][field] = (value.shape, value.dtype.str, value.tobytes())
                    assert not value.flags.writeable
                    with pytest.raises(ValueError):
                        value.setflags(write=True)
                else:
                    captured[name][field] = value
        with pytest.raises(FrozenInstanceError):
            raw.identity = _identity()
        for source in (pressure_input, rigid_input):
            for value in source.values():
                if isinstance(value, np.ndarray):
                    value.fill(42)
            source.clear()
        for name, declaration in (("pressure", raw.pressure), ("rigid", raw.rigid)):
            assert set(declaration) == set(captured[name])
            for field, value in declaration.items():
                if isinstance(value, np.ndarray):
                    assert (value.shape, value.dtype.str, value.tobytes()) == captured[name][field]
                else:
                    assert value == captured[name][field]
        ((pressure, rigid, identity, stamp),) = mesh_geometry._build_geometry_inputs((raw,))
        assert counts == {"pressure": 1, "rigid": 1}
        assert identity == _identity() and stamp == _stamp()
        for field in ("world_vertices", "cells", "nodal_pressure"):
            np.testing.assert_array_equal(getattr(pressure, field), raw.pressure[field])
        for field in ("world_vertices", "faces"):
            np.testing.assert_array_equal(getattr(rigid, field), raw.rigid[field])
        assert pressure.pressure_domain == "full"

    object.__setattr__(raw, "pressure", dict(raw.pressure, unexpected_field=0))
    with _validation_calls(monkeypatch) as counts:
        with pytest.raises(AreaPressureIntersectionError) as error:
            tuple(mesh_geometry._build_geometry_inputs((raw,)))
        assert error.value.category == "invalid_input"
        assert counts == {"pressure": 0, "rigid": 0}

    for fault in ("world_coordinates", "overlapping_cells", "full_boundary_pressure"):
        pressure_input, rigid_input = inputs()
        if fault == "world_coordinates":
            pressure_input["world_vertices"][-1] = pressure_input["world_vertices"][0]
        elif fault == "overlapping_cells":
            # Legal indices and distinct nodes; geometrically the whole outer
            # tetrahedron overlaps the three remaining center-subdivision cells.
            pressure_input["cells"][0] = (0, 1, 2, 3)
        else:
            pressure_input["nodal_pressure"][0] = 0.25
        with _validation_calls(monkeypatch) as counts:
            invalid = mesh_geometry._AreaPressureGeometryInput(
                pressure_input, rigid_input, _identity(), _stamp()
            )
            assert counts == {"pressure": 0, "rigid": 0}
            with pytest.raises(AreaPressureIntersectionError) as error:
                tuple(mesh_geometry._build_geometry_inputs((invalid,)))
            assert error.value.category == "invalid_mesh"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("route", ("host", "mixed"))
def test_current_geometry_validates_distinct_sides_and_matches_host_after_reset(
    mode, route, monkeypatch
):
    with ExitStack() as stack:
        reference_batch, reference_scene, reference_ws, models, worlds, assets = _make(stack, mode)
        # Reverse the public world rows while retaining the same physical inputs,
        # material declarations, and one shared soft/stiff compilation bucket.
        batch = stack.enter_context(
            MuJoCoTyxHeterogeneousBatch(models, tuple(reversed(worlds)), mode=mode)
        )
        query = _intersector(stack, mode, route)
        scene = MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets, intersector=query)
        workspace = stack.enter_context(batch.prepare_device_workspace(include_area_pressure=True))
        assert batch.world_keys == tuple(reversed(KEYS))
        assert batch.layout[41].bucket == batch.layout[11].bucket == batch.layout[305].bucket

        initial_signatures = None
        for index in range(2):
            before = batch.integration_snapshot()
            configuration = {key: batch.area_pressure_configuration_id(key) for key in AREA_KEYS}
            # Only the tested scene's complete physical step is in the spy scope.
            with monkeypatch.context() as patch:
                patch.setattr(scene_geometry, "_build_geometry_inputs", uncached_inputs)
                _step(reference_scene)
            _step_once_and_check_validation(monkeypatch, scene)
            frame, _ = _compare_current_outputs(
                batch, scene, workspace, reference_batch, reference_scene, reference_ws
            )
            signatures = {}
            for key in AREA_KEYS:
                intersection = scene.last_geometry[key][0]
                stamp = intersection.surface.stamp
                assert (
                    stamp.world_id,
                    stamp.step_index,
                    stamp.sample_time_s,
                    stamp.configuration_id,
                ) == (str(key), before[key].tick, before[key].time, configuration[key])
                assert frame.area_catalogs[key][0]["stamp"] == stamp
                signatures[key] = _signature(intersection)
            if index == 0:
                initial_signatures = signatures
                assert signatures[41] and signatures[305] == ()
            else:
                assert signatures[41] != initial_signatures[41]

        previous_frame = frame
        before_reset = batch.integration_snapshot()
        owners = tuple(batch._buckets)
        scene.reset((41, 7))
        reference_scene.reset((41, 7))
        assert any(old is not new for old, new in zip(owners, batch._buckets, strict=True))
        assert not scene.last_geometry
        with pytest.raises(ValueError):
            workspace.readback(previous_frame)
        after_reset = batch.integration_snapshot()
        for key in AREA_KEYS:
            if key == 41:
                assert after_reset[key].tick == 0
            else:
                _same(after_reset[key], before_reset[key])

        configuration = {key: batch.area_pressure_configuration_id(key) for key in AREA_KEYS}
        with monkeypatch.context() as patch:
            patch.setattr(scene_geometry, "_build_geometry_inputs", uncached_inputs)
            _step(reference_scene)
        _step_once_and_check_validation(monkeypatch, scene)
        _compare_current_outputs(
            batch, scene, workspace, reference_batch, reference_scene, reference_ws
        )
        for key in AREA_KEYS:
            stamp = scene.last_geometry[key][0].surface.stamp
            assert stamp.step_index == after_reset[key].tick
            assert stamp.sample_time_s == after_reset[key].time
            assert stamp.configuration_id == configuration[key]
        if query is not None:
            assert query.stats["attempts"] == 12 and query.stats["batch_calls"] == 3


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("route", ("host", "mixed"))
def test_last_world_degenerate_stored_geometry_rejects_before_launch_and_preserves_state(
    mode, route, monkeypatch
):
    with ExitStack() as stack:
        batch, default_scene, workspace, _, _, assets = _make(stack, mode)
        query = _intersector(stack, mode, route)
        scene = (
            default_scene
            if query is None
            else MuJoCoTyxHeterogeneousAreaPressureScene(batch, assets, intersector=query)
        )
        _step(scene)
        frame, arrays = _refresh(batch, workspace)
        state, geometry = batch.integration_snapshot(), scene.last_geometry
        capsule, revision = batch.checkpoint().to_bytes(), batch.checkpoint_revision
        area_owners = tuple(owner for owner in batch._buckets if owner._area_pressure_pairs)
        last_owner = area_owners[-1]
        last_world = last_owner.world_keys[-1]
        pressure_body = int(last_owner._plan.area_body_ids[0][0])
        expected_pose_order = tuple(key for owner in area_owners for key in owner.world_keys)
        original_poses = scene_geometry._current_body_poses
        seen, launches = [], []
        previous_results = query.last_results if query is not None else None
        previous_routes = query.last_batch_routes if query is not None else None
        previous_launches = query.stats["launches"] if query is not None else None

        def collapse_last_world(owner):
            # Obtain actual accepted-state FK first. Only its returned copy is
            # corrupted; rotations, native state and the local asset stay intact.
            current = original_poses(owner)
            result = {
                key: (positions.copy(), rotations.copy())
                for key, (positions, rotations) in current.items()
            }
            seen.extend(owner.world_keys)
            if owner is last_owner:
                positions, _ = result[last_world]
                positions[pressure_body] = (1e20, 1e20, 1e20)
            return result

        def forbidden_device(*args, **kwargs):
            launches.append(True)
            raise AssertionError("world-mesh validation was deferred beyond a device launch")

        with monkeypatch.context() as patch:
            patch.setattr(scene_geometry, "_current_body_poses", collapse_last_world)
            patch.setattr(TyxAreaPressureIntersector, "_device_clip", forbidden_device)
            with pytest.raises(AreaPressureIntersectionError) as error:
                scene.step(H, expected_revision=batch.checkpoint_revision)
            assert error.value.category == "invalid_mesh"
        assert tuple(seen) == expected_pose_order and seen[-1] == last_world
        assert launches == []
        _retained(batch, scene, workspace, state, geometry, frame, arrays, capsule, revision)
        if query is not None:
            assert query.last_results is previous_results
            assert query.last_batch_routes == previous_routes
            assert query.stats["launches"] == previous_launches
        _step_once_and_check_validation(monkeypatch, scene)
        assert scene.last_geometry is not geometry
        for key, after in batch.integration_snapshot().items():
            assert after.tick == state[key].tick + 1
        for key in AREA_KEYS:
            stamp = scene.last_geometry[key][0].surface.stamp
            assert stamp.step_index == state[key].tick and stamp.sample_time_s == state[key].time

"""HOST-only asset and transaction protocol tests for the geometry scene.

The fake runtime and empty intersection receipts test orchestration only. They
are not collision, forward-kinematics, dynamics, or device correctness oracles.
Real accepted-device FK and continuous physical motion are tested separately.
"""

from dataclasses import FrozenInstanceError
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import phymeo.backends.mujoco_area_pressure_scene as scene_module
from phymeo.backends.mujoco_area_pressure_scene import (
    BodyAreaPressureAsset,
    MuJoCoTyxAreaPressureScene,
)
from phymeo.geometry.area_pressure import AreaPressureIdentity, AreaPressureSurface
from phymeo.geometry.area_pressure_intersection import (
    AreaPressureIntersection,
    AreaPressureIntersectionError,
    RigidPressureMesh,
    TetraPressureMesh,
)

H = 0.002
KEYS = (23, 7)
PAIRS = (("soft", "hard"), ("other-soft", "hard"))
RX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
RZ = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)


def _pressure(*, local=False):
    if local:
        return TetraPressureMesh(
            ("o", "x", "y", "z"),
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ((0, 1, 2, 3),),
            (0, 0, 0, 1),
            ("local-cell",),
            pressure_domain="local",
        )
    return TetraPressureMesh(
        ("o", "x", "y", "z", "inside"),
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.25, 0.25, 0.25)),
        ((0, 1, 2, 4), (0, 2, 3, 4), (0, 3, 1, 4), (1, 3, 2, 4)),
        (0, 0, 0, 0, 1),
        ("base", "x-side", "y-side", "slant"),
    )


def _rigid():
    return RigidPressureMesh(
        ("a", "b", "c"),
        ((0.125, 0.125, 0.125), (0.625, 0.125, 0.125), (0.125, 0.625, 0.125)),
        ((0, 1, 2),),
        ("finite-face",),
    )


def _asset(pair=PAIRS[0]):
    return BodyAreaPressureAsset(
        AreaPressureIdentity(*pair, "pressure-material", "rigid-material"), _pressure(), _rigid()
    )


class _TrackedLock:
    """A single-threaded lock spy; this test does not claim concurrency coverage."""

    def __init__(self):
        self.depth = 0

    def __enter__(self):
        self.depth += 1
        return self

    def __exit__(self, exc_type, value, traceback):
        self.depth -= 1


class _Runtime:
    """Minimal declared runtime protocol, with no device or simulated dynamics."""

    def __init__(self, pairs=PAIRS):
        self._lock = _TrackedLock()
        self._world_keys = KEYS
        self._timestep = H
        self._checkpoint_revision = 0
        self._ticks = dict.fromkeys(KEYS, 0)
        self._data = {key: SimpleNamespace(time=0.0) for key in KEYS}
        self._model = SimpleNamespace(nbody=4)
        self._area_pressure_pairs = pairs
        self._plan = SimpleNamespace(
            area_pair_names=pairs,
            area_body_ids=np.asarray([(2, 1), (3, 1)][: len(pairs)], dtype=np.int32).reshape(-1, 2),
        )
        self.closed = False
        self.physics_calls = []
        self.physics_error = None
        self.poses = {}
        for index, key in enumerate(KEYS):
            positions = np.array([[0, 0, 0], [-2, 1, 0.5], [3, -1, 2], [0.5, 2, -1]], dtype=float)
            positions[2:, 0] += index
            rotations = np.stack((np.eye(3), RZ, RX, RZ.T))
            self.poses[key] = (positions, rotations)

    @property
    def world_keys(self):
        return self._world_keys

    @property
    def checkpoint_revision(self):
        self._check_open()
        return self._checkpoint_revision

    def _check_open(self):
        if self.closed:
            raise RuntimeError("fake runtime is closed")

    def area_pressure_configuration_id(self, key):
        assert self._lock.depth > 0
        return f"fake-state-{key}-{self._checkpoint_revision}-{self._ticks[key]}"

    def integration_snapshot(self):
        return MappingProxyType(
            {key: SimpleNamespace(tick=self._ticks[key], time=self._data[key].time) for key in KEYS}
        )

    def step_area_pressure(self, h, mapping, *, expected_revision):
        assert self._lock.depth > 0
        assert h == H
        assert type(expected_revision) is int and expected_revision == self._checkpoint_revision
        assert set(mapping) == set(KEYS)
        for key, surfaces in mapping.items():
            assert isinstance(surfaces, tuple) and len(surfaces) == len(self._plan.area_pair_names)
            for pair, surface in zip(self._plan.area_pair_names, surfaces, strict=True):
                assert (surface.identity.compliant_body, surface.identity.rigid_body) == pair
                assert surface.stamp.world_id == str(key)
                assert surface.stamp.step_index == self._ticks[key]
                assert surface.stamp.sample_time_s == self._data[key].time
                assert surface.stamp.configuration_id == self.area_pressure_configuration_id(key)
        self.physics_calls.append(mapping)
        if self.physics_error is not None:
            raise self.physics_error
        for key in KEYS:
            self._ticks[key] += 1
            self._data[key].time += h
        self._checkpoint_revision += 1
        return dict.fromkeys(KEYS, True)

    def external_boundary(self, kind):
        """Prescribed metadata/pose change, never a model of physical response."""
        self._checkpoint_revision += 1
        if kind == "reset":
            self._ticks = dict.fromkeys(KEYS, 0)
            for key in KEYS:
                self._data[key].time = 0.0
                # Prescribe a different reset pose; the static body stays put.
                self.poses[key][0][2] += [0.25, -0.5, 1]
                self.poses[key][1][2] = RZ
        elif kind != "control":
            raise AssertionError("unknown fake boundary")


class _GeometrySpy:
    def __init__(self, runtime):
        self.runtime = runtime
        self.pose_calls = []
        self.intersections = []
        self.fail_at = None

    def poses(self, runtime):
        assert runtime is self.runtime and runtime._lock.depth > 0
        self.pose_calls.append(runtime._checkpoint_revision)
        return {
            key: (positions.copy(), rotations.copy())
            for key, (positions, rotations) in runtime.poses.items()
        }

    def intersect(self, pressure, rigid, *, identity, stamp):
        assert self.runtime._lock.depth > 0
        self.intersections.append((pressure, rigid, identity, stamp))
        if self.fail_at == len(self.intersections):
            raise AreaPressureIntersectionError(
                "unsupported_gradient", "deliberate final-world failure"
            )
        return AreaPressureIntersection(
            AreaPressureSurface(identity, stamp, ()),
            "full",
            len(pressure.cells) * len(rigid.faces),
            0,
            0,
        )


def _fixture(monkeypatch):
    runtime = _Runtime()
    spy = _GeometrySpy(runtime)
    monkeypatch.setattr(scene_module, "_current_body_poses", spy.poses)
    monkeypatch.setattr(scene_module, "intersect_area_pressure", spy.intersect)
    assets = tuple(_asset(pair) for pair in PAIRS)
    scene = MuJoCoTyxAreaPressureScene(runtime, assets)
    return runtime, spy, assets, scene


def _state(runtime):
    return (
        runtime._checkpoint_revision,
        tuple((key, runtime._ticks[key], runtime._data[key].time) for key in KEYS),
    )


def _assert_preserved_mesh_fields(pressure, rigid, asset):
    assert pressure.pressure_domain == "full"
    assert pressure.node_ids == asset.pressure_local.node_ids
    assert pressure.cell_ids == asset.pressure_local.cell_ids
    assert rigid.node_ids == asset.rigid_local.node_ids
    assert rigid.face_ids == asset.rigid_local.face_ids
    np.testing.assert_array_equal(pressure.cells, asset.pressure_local.cells)
    np.testing.assert_array_equal(pressure.nodal_pressure, asset.pressure_local.nodal_pressure)
    np.testing.assert_array_equal(rigid.faces, asset.rigid_local.faces)


def test_body_asset_rebuilds_full_local_coordinates_as_owned_immutable_meshes():
    pressure, rigid = _pressure(), _rigid()
    identity = AreaPressureIdentity(*PAIRS[0], "elastic", "rigid")
    asset = BodyAreaPressureAsset(identity, pressure, rigid)
    assert asset.identity == identity
    assert asset.pressure_local is not pressure and asset.rigid_local is not rigid
    np.testing.assert_array_equal(asset.pressure_local.world_vertices, pressure.world_vertices)
    np.testing.assert_array_equal(asset.rigid_local.world_vertices, rigid.world_vertices)
    for array in (
        asset.pressure_local.world_vertices,
        asset.pressure_local.cells,
        asset.pressure_local.nodal_pressure,
        asset.rigid_local.world_vertices,
        asset.rigid_local.faces,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        asset.identity = identity


@pytest.mark.parametrize("bad", ["local-domain", "identity", "pressure", "rigid"])
def test_asset_rejects_manufactured_local_field_and_wrong_metadata_types(bad):
    identity = AreaPressureIdentity(*PAIRS[0], "elastic", "rigid")
    pressure, rigid = _pressure(), _rigid()
    if bad == "local-domain":
        pressure = _pressure(local=True)
    elif bad == "identity":
        identity = "not-an-identity"
    elif bad == "pressure":
        pressure = rigid
    else:
        rigid = pressure
    with pytest.raises((TypeError, ValueError)):
        BodyAreaPressureAsset(identity, pressure, rigid)


@pytest.mark.parametrize("bad", ["missing", "extra", "reversed", "duplicate", "list"])
def test_scene_requires_complete_ordered_declared_pairs(bad):
    runtime = _Runtime()
    assets = tuple(_asset(pair) for pair in PAIRS)
    invalid = {
        "missing": assets[:1],
        "extra": (*assets, assets[0]),
        "reversed": assets[::-1],
        "duplicate": (assets[0], assets[0]),
        "list": list(assets),
    }[bad]
    with pytest.raises((TypeError, ValueError)):
        MuJoCoTyxAreaPressureScene(runtime, invalid)
    assert runtime.physics_calls == [] and runtime._checkpoint_revision == 0


def test_success_binds_each_current_body_transform_world_and_input_phase(monkeypatch):
    runtime, spy, assets, scene = _fixture(monkeypatch)
    assert dict(scene.last_geometry) == {}
    assert isinstance(scene.assets, tuple) and scene.assets is not assets
    for owned, original_asset in zip(scene.assets, assets, strict=True):
        assert owned is not original_asset
        assert owned.pressure_local is not original_asset.pressure_local
        assert owned.rigid_local is not original_asset.rigid_local
    with pytest.raises(AttributeError):
        scene.assets = assets
    original = [
        (asset.pressure_local.world_vertices.tobytes(), asset.rigid_local.world_vertices.tobytes())
        for asset in assets
    ]
    accepted = scene.step(H, expected_revision=0)
    assert accepted == dict.fromkeys(KEYS, True)
    assert spy.pose_calls == [0] and len(spy.intersections) == 4
    assert len(runtime.physics_calls) == 1
    geometry = scene.last_geometry
    assert isinstance(geometry, MappingProxyType)
    assert tuple(geometry) == KEYS
    for world_index, key in enumerate(KEYS):
        assert isinstance(geometry[key], tuple) and len(geometry[key]) == 2
        positions, rotations = runtime.poses[key]
        for pair_index, asset in enumerate(assets):
            pressure, rigid, identity, stamp = spy.intersections[2 * world_index + pair_index]
            soft, hard = runtime._plan.area_body_ids[pair_index]
            np.testing.assert_array_equal(
                pressure.world_vertices,
                asset.pressure_local.world_vertices @ rotations[soft].T + positions[soft],
            )
            # The static body's nonidentity transform is not world identity,
            # nor the compliant body's transform or a COM-origin transform.
            np.testing.assert_array_equal(
                rigid.world_vertices,
                asset.rigid_local.world_vertices @ rotations[hard].T + positions[hard],
            )
            _assert_preserved_mesh_fields(pressure, rigid, asset)
            assert identity == asset.identity
            assert stamp.world_id == str(key) and stamp.step_index == 0 and stamp.sample_time_s == 0
            assert stamp.configuration_id == f"fake-state-{key}-0-0"
            assert geometry[key][pair_index].surface.stamp == stamp
            submitted = runtime.physics_calls[0][key][pair_index]
            assert submitted.identity == identity and submitted.stamp == stamp
            assert submitted.triangles == geometry[key][pair_index].surface.triangles == ()
            assert geometry[key][pair_index].pressure_domain == "full"
    assert runtime.checkpoint_revision == 1
    assert scene.last_geometry is geometry
    with pytest.raises(TypeError):
        geometry[23] = ()
    for asset, (pressure_bytes, rigid_bytes) in zip(assets, original, strict=True):
        assert asset.pressure_local.world_vertices.tobytes() == pressure_bytes
        assert asset.rigid_local.world_vertices.tobytes() == rigid_bytes


def test_stale_revision_wrong_h_and_closed_runtime_refuse_before_fk(monkeypatch):
    runtime, spy, _, scene = _fixture(monkeypatch)
    for revision in (1, False, 0.0):
        with pytest.raises(ValueError):
            scene.step(H, expected_revision=revision)
    for h in (H / 2, float("nan")):
        with pytest.raises(ValueError):
            scene.step(h, expected_revision=0)
    runtime.closed = True
    with pytest.raises(RuntimeError, match="closed"):
        scene.step(H, expected_revision=0)
    assert spy.pose_calls == spy.intersections == runtime.physics_calls == []
    assert _state(runtime) == (0, ((23, 0, 0.0), (7, 0, 0.0)))
    assert runtime._lock.depth == 0


def test_final_world_geometry_failure_preserves_previous_geometry_and_no_physics(monkeypatch):
    runtime, spy, _, scene = _fixture(monkeypatch)
    scene.step(H, expected_revision=0)
    old, state = scene.last_geometry, _state(runtime)
    spy.intersections.clear()
    spy.fail_at = 4
    with pytest.raises(AreaPressureIntersectionError):
        scene.step(H, expected_revision=1)
    assert [item[3].world_id for item in spy.intersections] == ["23", "23", "7", "7"]
    assert _state(runtime) == state and len(runtime.physics_calls) == 1
    assert scene.last_geometry is old and runtime._lock.depth == 0
    spy.fail_at = None
    scene.step(H, expected_revision=1)
    assert runtime.checkpoint_revision == 2 and scene.last_geometry is not old


def test_physics_failure_does_not_publish_fresh_successful_geometry(monkeypatch):
    runtime, spy, _, scene = _fixture(monkeypatch)
    scene.step(H, expected_revision=0)
    old, state = scene.last_geometry, _state(runtime)
    runtime.physics_error = RuntimeError("deliberate unaccepted physics failure")
    with pytest.raises(RuntimeError, match="unaccepted physics"):
        scene.step(H, expected_revision=1)
    assert len(spy.intersections) == 8 and len(runtime.physics_calls) == 2
    assert _state(runtime) == state and scene.last_geometry is old
    assert runtime._lock.depth == 0
    runtime.physics_error = None
    scene.step(H, expected_revision=1)
    assert runtime.checkpoint_revision == 2 and scene.last_geometry is not old


@pytest.mark.parametrize("boundary", ["control", "reset"])
def test_external_revision_invalidates_geometry_and_next_step_rebuilds_pose(monkeypatch, boundary):
    runtime, spy, assets, scene = _fixture(monkeypatch)
    scene.step(H, expected_revision=0)
    old = scene.last_geometry
    old_pressure = spy.intersections[0][0].world_vertices.copy()
    old_rigid = spy.intersections[0][1].world_vertices.copy()
    runtime.external_boundary(boundary)
    assert dict(scene.last_geometry) == {}
    assert old[23][0].surface.stamp.step_index == 0
    with pytest.raises(ValueError):
        scene.step(H, expected_revision=1)
    assert spy.pose_calls == [0]
    expected_tick = 0 if boundary == "reset" else 1
    scene.step(H, expected_revision=2)
    assert spy.pose_calls == [0, 2]
    pressure, rigid, _, stamp = spy.intersections[4]
    pos, rotation = runtime.poses[23]
    np.testing.assert_array_equal(
        pressure.world_vertices, assets[0].pressure_local.world_vertices @ rotation[2].T + pos[2]
    )
    if boundary == "reset":
        assert not np.array_equal(pressure.world_vertices, old_pressure)
    else:
        np.testing.assert_array_equal(pressure.world_vertices, old_pressure)
    np.testing.assert_array_equal(rigid.world_vertices, old_rigid)
    assert stamp.step_index == expected_tick
    assert stamp.sample_time_s == (0.0 if boundary == "reset" else H)
    assert stamp.configuration_id == f"fake-state-23-2-{expected_tick}"
    assert scene.last_geometry[23][0].surface.stamp == stamp
    assert runtime.checkpoint_revision == 3

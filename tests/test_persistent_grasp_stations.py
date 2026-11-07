"""Tests-first: declare original grasp material stations before contact is born.

UNEXECUTED by the author. The parent owns all project execution.
"""

import sys
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from pathlib import Path

import numpy as np
import phymeo as ph
import pytest

from phymeo.runtime.material_stations import MaterialStationRegistry
from phymeo.runtime.robot_payload_scene import PadFace
from phymeo.solvers.tangent_history import StationKey


OFFSETS = tuple(product((-0.005, 0.005), repeat=2))
TOKENS = ((101, 107, 131, 149), (211, 239, 263, 307))


def _acquisition():
    name = "phymeo_original_persistent_grasp_acquisition"
    if name not in sys.modules:
        path = Path(__file__).resolve().parents[1] / "examples" / "robot_force_acquisition.py"
        spec = spec_from_file_location(name, path)
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _declarations(case):
    if case == "panda-sphere":
        names = tuple((f"panda_{side}finger", f"panda_{side}finger_pad") for side in ("left", "right"))
        axis, signs = 1, (-1, 1)
        counter = ((0,), (0,))
    else:
        names = (("left", "left_rubber"), ("right", "right_rubber"))
        axis, signs = 0, (1, -1)
        counter = ((0, 1, 2, 3), (4, 5, 6, 7))
    return dict(
        robot=(0, 10),
        pads=tuple(PadFace.named(*name, axis, sign, OFFSETS) for name, sign in zip(names, signs, strict=True)),
        station_keys={name: tuple(StationKey(0, token) for token in tokens) for name, tokens in zip(names, TOKENS, strict=True)},
        counter_features=dict(zip(names, counter, strict=True)),
        chart_ids={name: (1001 + i, 2001 + i) for i, name in enumerate(names)},
    )


def _named(pad):
    return pad.shape_key.link_name, pad.shape_key.collision_name


def _registry(scene, info, declaration):
    # Deliberately resolve the new public function only after each real premise.
    return ph.material_stations_from_pads(scene, payload_key=info["primary_key"], **declaration)


def _assert_mimic_open_premise(scene, info, declaration):
    reading, geometry = scene.observe_scene()
    asset = scene.assets[0, 10]
    assert asset.joint_names == ("lift", "screw")
    assert asset.tree.joint_names == ("lift", "screw", "idler")
    np.testing.assert_array_equal(asset.matrix, ((1, 0), (0, 1), (0, -1)))
    np.testing.assert_array_equal(reading.robots[0, 10].q, (0, 0))
    np.testing.assert_array_equal(reading.robots[0, 10].v, (0, 0))
    np.testing.assert_allclose(reading.robots[0, 10].full_q, (0, 0, 0.004), atol=1e-15, rtol=0)
    np.testing.assert_array_equal(reading.robots[0, 10].full_v, (0, 0, 0))
    assert info["dt"] == 0.001
    assert info["masses"] == (0.1, 0.03)
    np.testing.assert_allclose(info["center"], (0.25, -0.125, 0.25), atol=1e-15, rtol=0)
    assert reading.step_number == 0
    assert reading.time_s == geometry.time_s == 0.0
    assert reading.load_time_s is None
    assert scene.tangent_history is None
    assert scene.tangent_receipt is None

    # Independent authored URDF positions/box dimensions, not sampled contact rows.
    world_anchors = []
    for pad, origin_x in zip(declaration["pads"], (0.215, 0.285), strict=True):
        name = _named(pad)
        key = scene.shape_key(10, *name)
        col = next(c for c in asset.tree.collisions if (c.link_name, c.name) == name)
        origin, rotation = asset.authored_link_pose(name[0], reading.robots[0, 10].q)
        np.testing.assert_allclose(origin, (origin_x, -0.125, 0.25), atol=1e-14, rtol=0)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-14, rtol=0)
        vertices = np.asarray(geometry.poses[key].start)
        np.testing.assert_allclose(vertices, origin + np.asarray(col.local_vertices) @ rotation.T, atol=1e-14, rtol=0)
        np.testing.assert_allclose(np.ptp(vertices, axis=0), (0.006, 0.024, 0.018), atol=1e-14, rtol=0)
        face_x = origin_x + pad.side * 0.003
        for u, v in pad.offsets_m:
            point = np.array((face_x, -0.125 + u, 0.25 + v))
            assert point[0] == pytest.approx(vertices[:, 0].max() if pad.side > 0 else vertices[:, 0].min(), abs=1e-14)
            assert vertices[:, 1].min() < point[1] < vertices[:, 1].max()
            assert vertices[:, 2].min() < point[2] < vertices[:, 2].max()
            world_anchors.append(point)
    assert len(world_anchors) == 8
    assert len({tuple(p) for p in world_anchors}) == 8
    pairs = {tuple(sorted((key, info["primary_key"]))) for key in info["pad_keys"]}
    assert not any(load.force_n > 0 for load in scene.sample().contacts if (load.first, load.second) in pairs)
    assert scene.observe().step_number == 0
    return reading


def test_named_pad_registry_predeclares_real_open_mimic_sites():
    scene, _controller, info = _acquisition().make_case("mimic-box")
    with scene:
        declaration = _declarations("mimic-box")
        _assert_mimic_open_premise(scene, info, declaration)
        events = scene.contact_events

        registry = _registry(scene, info, declaration)

        assert isinstance(registry, MaterialStationRegistry)
        assert len(registry.registrations) == 8
        assert tuple(row.key for row in registry.registrations) == tuple(sorted(key for keys in declaration["station_keys"].values() for key in keys))
        assert set(registry.pairs) == {tuple(sorted((key, info["primary_key"]))) for key in info["pad_keys"]}
        assert scene.contact_events is events
        _assert_mimic_open_premise(scene, info, declaration)

"""Authored R3 finite planes: metres, known poses, no physics or hidden assets."""

import base64
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.cameras import Camera
from phymeo_rendering.scene import (
    DirectionalLight, FrameKey, Instance, NodePose, Pose, SceneFrame, VisualScene, compose_pose,
)


BOARD_IDS = (0x01020304, 0x00010203, 0x00A1B2C3, 0xF1020304)
MOUNT = Pose((0.012, 0.0, 0.006))
_BOARD_CENTERS = ((-0.075, 0.05, -0.25), (0.15, 0.10, -0.5),
                  (-0.30, -0.20, -1.0), (0.60, -0.40, -2.0))
_BOARD_HALF = ((0.045, 0.03), (0.09, 0.06), (0.18, 0.12), (0.36, 0.24))


@dataclass(frozen=True)
class Surface:
    identity: int
    node: str
    local_center: tuple
    half_size: tuple
    marker_channel: int | None = None


@dataclass(frozen=True)
class CalibrationCase:
    name: str
    frame: SceneFrame
    cameras: tuple


def _f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _axis(axis, angle):
    values = [0.0, 0.0, 0.0, math.cos(angle/2)]
    values[axis] = math.sin(angle/2)
    return tuple(values)


def _write_quad(path, half_size):
    hx, hy = half_size
    positions = ((-hx, -hy, 0.), (hx, -hy, 0.), (hx, hy, 0.), (-hx, hy, 0.))
    raw = (struct.pack("<12f", *(x for p in positions for x in p))
           + struct.pack("<12f", *(x for _ in positions for x in (0., 0., 1.)))
           + struct.pack("<6H", 0, 1, 2, 0, 2, 3))
    data = {
        "asset": {"version": "2.0", "generator": "Phymeo authored R3 finite metric quad"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1},
                                      "indices": 2, "mode": 4}]}],
        "buffers": [{"byteLength": len(raw), "uri": "data:application/octet-stream;base64,"
                     + base64.b64encode(raw).decode("ascii")}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 48},
                        {"buffer": 0, "byteOffset": 48, "byteLength": 48},
                        {"buffer": 0, "byteOffset": 96, "byteLength": 12}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
                       "min": [-hx, -hy, 0.], "max": [hx, hy, 0.]},
                      {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
                      {"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR"}],
    }
    path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")


def build_calibration_fixture(directory):
    """Return VisualScene, five true pose cases, and analytic surface declarations.

    Finite quad extents are rounded once to the FLOAT glTF coordinates and
    retained independently for the plane oracle. No decoded backend mesh is
    consulted by that oracle. Markers sit 0.5 mm in front of each board, avoiding
    coplanar depth ties; their 6 px front-view side is not a feature texture.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    surfaces, meshes, instances = [], [], []
    def add(identity, node, center, half_size, material, marker_channel=None):
        half_size = tuple(_f32(v) for v in half_size)
        name = f"quad-{identity}"
        path = directory / f"{name}.gltf"
        _write_quad(path, half_size)
        meshes.append(Mesh(name, path.resolve().as_uri()))
        instances.append(Instance(identity, name, material, node, Pose(center)))
        surfaces.append(Surface(identity, node, center, half_size, marker_channel))
    for index, (identity, half_size) in enumerate(zip(BOARD_IDS, _BOARD_HALF, strict=True)):
        node = f"board-{index}"
        add(identity, node, (0., 0., 0.), half_size, "board")
        depth = -_BOARD_CENTERS[index][2]
        for marker, (dx, dy) in enumerate(((-0.55, 0.20), (0., -0.35), (0.55, 0.20))):
            add(1000+index*10+marker, node,
                (dx*half_size[0], dy*half_size[1], 0.0005), (depth*0.006,)*2,
                ("red", "green", "blue")[marker], marker)
    add(4097, "occluder", (0., 0., 0.), (0.016, 0.020), "occluder")
    materials = (
        Material("board", (0.18,)*3, 0., 0.9), Material("occluder", (0.08,)*3, 0., 0.9),
        Material("red", (0.8, 0.01, 0.01), 0., 0.9),
        Material("green", (0.01, 0.8, 0.01), 0., 0.9),
        Material("blue", (0.01, 0.01, 0.8), 0., 0.9),
    )
    scene = VisualScene("r3-calibration-v1", tuple(meshes), materials, tuple(instances),
                        (DirectionalLight((0., 0., -1.), (1., 1., 1.)),))
    cases = []
    for index, name in enumerate(("front", "tilted", "occluded", "flipped", "reset")):
        nodes = []
        for board, center in enumerate(_BOARD_CENTERS):
            orientation = (0., 0., 0., 1.)
            if name == "tilted":
                orientation = _axis(1 if board % 2 == 0 else 0, math.radians(12 if board % 2 == 0 else -9))
            elif name == "flipped":
                center = (-center[0], -center[1], center[2])
                orientation = (0., 0., 1., 0.)
            nodes.append(NodePose(f"board-{board}", Pose(center, orientation)))
        blocker = (0.1596, 0.0896, -0.4) if name == "occluded" else (3., 0., -0.4)
        nodes.append(NodePose("occluder", Pose(blocker)))
        wrist = Pose((-0.012, 0., -0.006))
        if name == "tilted":
            wrist = Pose((-0.004, 0.002, 0.004), _axis(1, math.radians(5)))
        elif name == "occluded":
            wrist = Pose((-0.016, 0.002, 0.004), _axis(2, -0.08))
        elif name == "flipped":
            wrist = Pose((-0.008, -0.003, 0.002), _axis(2, 0.22))
        nodes.append(NodePose("wrist", wrist))
        epoch, tick = (1, 0) if name == "reset" else (0, index)
        frame = SceneFrame(FrameKey(scene.key, "calibration", epoch, tick, tick, "authored"),
                           tick*0.01, tuple(nodes))
        cameras = tuple(Camera(label, frame.key, pose, 640, 480, 500., 500., 320., 240., 0.1, 5.)
                        for label, pose in (("fixed", Pose()), ("wrist", compose_pose(wrist, MOUNT))))
        cases.append(CalibrationCase(name, frame, cameras))
    return scene, tuple(cases), tuple(surfaces)

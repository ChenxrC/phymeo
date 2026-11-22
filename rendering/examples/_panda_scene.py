"""Scene construction for the archived Panda capture example (metres throughout)."""

import base64
import json
import math
from pathlib import Path

import numpy as np
from phymeo_rendering.assets import Material, Mesh
from phymeo_rendering.cameras import Camera
from phymeo_rendering.scene import DirectionalLight, Instance, Pose, VisualScene, compose_pose


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def pose_matrix(pose):
    x, y, z, w = unit(pose.orientation_xyzw)
    matrix = np.eye(4)
    matrix[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
    matrix[:3, 3] = pose.position_m
    return matrix


def look_at(eye, target, up):
    back = unit(np.subtract(eye, target))
    right = unit(np.cross(up, back))
    a, b, c = np.column_stack((right, np.cross(back, right), back))
    # Symmetric unit-quaternion extraction also covers exact half turns.
    k = np.array(
        [
            [a[0] - b[1] - c[2], a[1] + b[0], a[2] + c[0], c[1] - b[2]],
            [a[1] + b[0], b[1] - a[0] - c[2], b[2] + c[1], a[2] - c[0]],
            [a[2] + c[0], b[2] + c[1], c[2] - a[0] - b[1], b[0] - a[1]],
            [c[1] - b[2], a[2] - c[0], b[0] - a[1], a[0] + b[1] + c[2]],
        ]
    )
    q = np.linalg.eigh(k)[1][:, -1]
    return Pose(tuple(eye), tuple(q))


def cameras(frame):
    fixed = look_at((1.27, -0.86, 0.79), (0.10, 0, 0.43), (0, 0, 1))
    mount = look_at((0.095, 0, -0.025), (0, 0, 0.138), (1, 0, 0))
    hand = next(n.world_pose for n in frame.nodes if n.node_id == "panda_hand")
    detail = look_at((1.27, -0.86, 0.79), (0.345, 0, 0.49), (0, 0, 1))
    return (
        Camera("fixed", frame.key, fixed, 960, 720, 1150, 1150, 480, 360, 0.005, 5.0),
        Camera(
            "wrist", frame.key, compose_pose(hand, mount), 640, 480, 440, 440, 320, 240, 0.005, 5.0
        ),
        Camera("detail", frame.key, detail, 800, 600, 3900, 3900, 400, 300, 0.005, 5.0),
    )


def write_mesh(root, name, positions, normals, triangles):
    arrays = (np.array(positions, "<f4"), np.array(normals, "<f4"), np.array(triangles, "<u2"))
    views, chunks = [], []
    offset = 0
    for array in arrays:
        data = array.tobytes()
        views.append(dict(buffer=0, byteOffset=offset, byteLength=len(data)))
        chunks.append(data)
        offset += len(data)
    raw = b"".join(chunks)
    doc = dict(
        asset={"version": "2.0"},
        scene=0,
        scenes=[{"nodes": [0]}],
        nodes=[{"mesh": 0}],
        meshes=[
            {"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "mode": 4}]}
        ],
        buffers=[
            {
                "byteLength": len(raw),
                "uri": "data:application/octet-stream;base64," + base64.b64encode(raw).decode(),
            }
        ],
        bufferViews=views,
        accessors=[
            dict(
                bufferView=0,
                componentType=5126,
                type="VEC3",
                count=len(positions),
                min=arrays[0].min(axis=0).tolist(),
                max=arrays[0].max(axis=0).tolist(),
            ),
            dict(bufferView=1, componentType=5126, type="VEC3", count=len(normals)),
            dict(bufferView=2, componentType=5123, type="SCALAR", count=arrays[2].size),
        ],
    )
    path = Path(root) / (name + ".gltf")
    path.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
    return Mesh(name, path.resolve().as_uri())


def box(root, name, size):
    h = np.array(size) / 2
    p, n, t = [], [], []
    for axis in range(3):
        for sign in (-1, 1):
            normal = np.eye(3)[axis] * sign
            u = np.eye(3)[(axis + 1) % 3]
            v = np.cross(normal, u)
            start = len(p)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p.append((normal + a * u + b * v) * h)
                n.append(normal)
            t.extend(((start, start + 1, start + 2), (start, start + 2, start + 3)))
    return write_mesh(root, name, p, n, t)


def marker(root):
    # Small round, viewpoint-independent calibration target with a 3 mm radius.
    normals = [(0, 0, 1)]
    rings, count = 10, 20
    for ring in range(1, rings):
        polar = math.pi * ring / rings
        for i in range(count):
            angle = 2 * math.pi * i / count
            normals.append(
                (
                    math.sin(polar) * math.cos(angle),
                    math.sin(polar) * math.sin(angle),
                    math.cos(polar),
                )
            )
    bottom = len(normals)
    normals.append((0, 0, -1))
    triangles = []
    for i in range(count):
        j = (i + 1) % count
        triangles.append((0, 1 + i, 1 + j))
        for k in range(rings - 2):
            a = 1 + k * count
            triangles.extend(((a + i, a + count + i, a + j), (a + j, a + count + i, a + count + j)))
        a = 1 + (rings - 2) * count
        triangles.append((a + i, bottom, a + j))
    return write_mesh(root, "fiducial", np.array(normals) * 0.003, normals, triangles)


def build_scene(prepared, directory, box_size, table_size):
    from phymeo_rendering.panda_visuals import panda_visual_assets

    meshes, materials, instances = panda_visual_assets(prepared)
    geometry = (
        box(directory, "payload", box_size),
        box(directory, "tabletop", table_size),
        box(directory, "floor", (1.7, 1.3, 0.024)),
        box(directory, "stand", (0.082, 0.065, 0.38)),
    )
    finishes = (
        Material("oxide-red", (0.54, 0.032, 0.017), 0.0, 0.41),
        Material("bench-grey", (0.29, 0.32, 0.32), 0.18, 0.47),
        Material("studio-ground", (0.31, 0.32, 0.32), 0.0, 0.73),
    )
    props = (
        Instance(1, "payload", "oxide-red", "cube"),
        Instance(2, "tabletop", "bench-grey", "table"),
        Instance(3, "floor", "studio-ground", "panda_link0", Pose((0.13, 0, -0.014))),
        Instance(4, "stand", "bench-grey", "table", Pose((0, 0, -table_size[2] / 2 - 0.19))),
    )
    key = DirectionalLight(tuple(unit((0.6, 0, -0.8))), (1, 1, 1))
    return VisualScene(
        "panda-r1-full-v1",
        tuple(meshes) + geometry,
        tuple(materials) + finishes,
        tuple(instances) + props,
        (key,),
    )

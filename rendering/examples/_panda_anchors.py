"""Independent camera-ray references and observed R1 link-marker measurements."""

from dataclasses import dataclass

import numpy as np
from _panda_scene import pose_matrix
from phymeo_rendering.mesh_data import load_mesh


def turn(axis, angle):
    result = np.eye(3)
    i, j = (axis + 1) % 3, (axis + 2) % 3
    c, s = np.cos(angle), np.sin(angle)
    result[i, i], result[i, j], result[j, i], result[j, j] = c, -s, s, c
    return result


def affine(offset, rot=None):
    m = np.eye(4)
    if rot is not None:
        m[:3, :3] = rot
    m[:3, 3] = offset
    return m


def reference_links(q):
    # Nominal classic Panda dimension chain. Independent of the replay reader's
    # XML tree traversal and of the GPU session's rigid transforms.
    origins = (
        (0, 0, 0.333),
        (0, 0, 0),
        (0, -0.316, 0),
        (0.0825, 0, 0),
        (-0.0825, 0.384, 0),
        (0, 0, 0),
        (0.088, 0, 0),
    )
    base = np.eye(4)
    nodes = {"panda_link0": base}
    for i, (offset, r, angle) in enumerate(
        zip(origins, (0, -1, 1, 1, -1, 1, 1), q[:7], strict=True), 1
    ):
        base = base @ affine(offset, turn(0, r * np.pi / 2) @ turn(2, angle))
        nodes[f"panda_link{i}"] = base
    hand = base @ affine((0, 0, 0.107), turn(2, -np.pi / 4))
    nodes["panda_hand"] = hand
    for i, side, sign in ((7, "left", 1), (8, "right", -1)):
        nodes["panda_" + side + "finger"] = hand @ affine((0, sign * q[i], 0.0584))
    return nodes


def move(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def project(points, camera):
    transform = pose_matrix(camera.world_pose)
    xyz = (points - transform[:3, 3]) @ transform[:3, :3]
    depth = -xyz[..., 2]
    uv = np.stack(
        (
            camera.cx_px + camera.fx_px * xyz[..., 0] / depth,
            camera.cy_px - camera.fy_px * xyz[..., 1] / depth,
        ),
        axis=-1,
    )
    return uv, depth


@dataclass
class ProjectedScene:
    uv: np.ndarray
    z: np.ndarray
    owners: np.ndarray
    triangle_ids: np.ndarray
    node_points: np.ndarray
    world_points: np.ndarray
    bounds_min: np.ndarray
    bounds_max: np.ndarray

    def first_hit(self, uv):
        mask = np.all((self.bounds_min <= uv) & (uv <= self.bounds_max), axis=1)
        tri = self.uv[mask]
        a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        p = uv - tri[:, 0]
        det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
        valid = np.abs(det) > 1e-10
        den = np.where(valid, det, 1.0)
        s = (p[:, 0] * b[:, 1] - p[:, 1] * b[:, 0]) / den
        t = (a[:, 0] * p[:, 1] - a[:, 1] * p[:, 0]) / den
        inside = valid & (s >= 0) & (t >= 0) & (s + t <= 1)
        if not inside.any():
            return None, np.inf
        weights = np.column_stack((1 - s - t, s, t))[inside]
        depth = 1 / np.sum(weights / self.z[mask][inside], axis=1)
        k = np.argmin(depth)
        return int(np.flatnonzero(mask)[inside][k]), float(depth[k])


def prepare(scene, camera, links, geometry):
    ids, indices, local, world = [], [], [], []
    for ins in scene.instances:
        points, triangles = geometry[ins.mesh_id]
        visual = pose_matrix(ins.local_pose)
        node = move(points, visual)
        local.append(node[triangles])
        world.append(move(node, links[ins.node_id])[triangles])
        ids.extend([ins.id] * len(triangles))
        indices.extend(range(len(triangles)))
    world = np.concatenate(world)
    uv, z = project(world, camera)
    if np.any(z <= camera.near_m) or np.any(z >= camera.far_m):
        raise ValueError("The fixed R1 anchor reference expects all vertices inside its clip slab")
    return ProjectedScene(
        uv,
        z,
        np.array(ids),
        np.array(indices),
        np.concatenate(local).mean(axis=1),
        world.mean(axis=1),
        uv.min(axis=1),
        uv.max(axis=1),
    )


def freeze_anchors(scene, cameras, refs):
    geometry = {}
    for mesh in scene.meshes:
        decoded = load_mesh(mesh)
        geometry[mesh.id] = np.asarray(decoded.positions), np.asarray(decoded.triangles)
    views = [
        prepare(scene, camera, links, geometry) for camera, links in zip(cameras, refs, strict=True)
    ]
    areas = []
    for view in views:
        edges = view.uv[:, 1:] - view.uv[:, 0, None, :]
        areas.append(np.abs(edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]) / 2)
    minimum = np.minimum(*areas)
    info = {ins.id: ins for ins in scene.instances}
    anchors = []
    for group, nodes in (
        ("base", ("panda_link0",)),
        ("wrist", ("panda_hand",)),
        ("finger", ("panda_leftfinger", "panda_rightfinger")),
    ):
        owners = [i for i, ins in info.items() if i >= 100 and ins.node_id in nodes]
        subset = np.flatnonzero(np.isin(views[0].owners, owners))
        order = subset[np.argsort(-minimum[subset], kind="stable")]
        for k in order:
            if minimum[k] < 1:
                break
            expected = []
            for camera, view in zip(cameras, views, strict=True):
                uv, _ = project(view.world_points[k], camera)
                if np.any(uv < 4) or np.any(uv >= [camera.width - 4, camera.height - 4]):
                    break
                pixel = np.floor(uv).astype(int)
                hit, _ = view.first_hit(uv)
                pixel_hit, depth = view.first_hit(pixel + 0.5)
                if hit != k or pixel_hit != k:
                    break
                expected.append(dict(uv=uv.tolist(), pixel=pixel.tolist(), depth_m=depth))
            if len(expected) == len(views):
                owner = int(views[0].owners[k])
                anchors.append(
                    dict(
                        name=group,
                        owner=owner,
                        mesh=info[owner].mesh_id,
                        node=info[owner].node_id,
                        triangle=int(views[0].triangle_ids[k]),
                        node_point=views[0].node_points[k].tolist(),
                        expected=expected,
                    )
                )
                break
        else:
            raise ValueError("No material anchor candidates for " + group)
        if not anchors or anchors[-1]["name"] != group:
            raise ValueError("No common visible material point for " + group)
    return anchors


def measure(anchors, index, fixed_ids, fixed_z, marker_ids):
    results = []
    for i, a in enumerate(anchors):
        exp = a["expected"][index]
        x, y = exp["pixel"]
        owner = int(fixed_ids[y, x])
        depth = float(fixed_z[y, x])
        yy, xx = np.where(marker_ids == 10 + i)
        if not len(xx):
            raise AssertionError(f"Missing {a['name']} fiducial")
        centroid = np.array([xx.mean() + 0.5, yy.mean() + 0.5])
        err = float(np.linalg.norm(centroid - exp["uv"]))
        result = dict(
            name=a["name"],
            owner_expected=a["owner"],
            owner_observed=owner,
            expected=exp,
            observed_centroid=centroid.tolist(),
            error_px=err,
            marker_pixels=len(xx),
            observed_depth_m=depth,
        )
        if owner != a["owner"] or abs(depth - exp["depth_m"]) > 0.002 or err > 1:
            raise AssertionError(result)
        results.append(result)
    return results

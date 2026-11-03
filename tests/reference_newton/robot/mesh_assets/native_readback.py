"""Newton 1.7 import and device-side FK, indexed triangle surfaces and rigid contacts."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import newton
import numpy as np
import warp as wp


@wp.kernel
def triangle_corners(
    source: wp.array(dtype=wp.uint64),
    shapes: wp.array(dtype=wp.transform),
    bodies: wp.array(dtype=wp.transform),
    shape: int,
    body: int,
    world_vertices: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    u = float(0.0)
    v = float(0.0)
    if i % 3 == 0:
        u = 1.0
    elif i % 3 == 1:
        v = 1.0
    point = wp.mesh_eval_position(source[shape], i // 3, u, v)
    frame = wp.transform_multiply(bodies[body], shapes[shape])
    world_vertices[i] = wp.transform_point(frame, point)


@wp.kernel
def surface_distances(
    source: wp.array(dtype=wp.uint64),
    shapes: wp.array(dtype=wp.transform),
    bodies: wp.array(dtype=wp.transform),
    shape: int,
    body: int,
    probes: wp.array(dtype=wp.vec3),
    world_distances: wp.array(dtype=float),
):
    i = wp.tid()
    frame = wp.transform_multiply(bodies[body], shapes[shape])
    local = wp.transform_point(wp.transform_inverse(frame), probes[i])
    query = wp.mesh_query_point_sign_normal(source[shape], local, 2.0)
    if query.result:
        point = wp.mesh_eval_position(source[shape], query.face, query.u, query.v)
        world_distances[i] = wp.length(point - local) * query.sign
    else:
        world_distances[i] = 1.0e20


XY = ((0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2))
L_VERTS = np.asarray([(x, y, z) for z in (-0.5, 0.5) for x, y in XY])
CAP = ((0, 1, 3), (1, 2, 3), (0, 3, 5), (3, 4, 5))
L_FACES = {frozenset(x) for x in CAP} | {frozenset(i + 6 for i in x) for x in CAP}
for a in range(6):
    b = (a + 1) % 6
    L_FACES.update((frozenset((a, b, b + 6)), frozenset((a, b + 6, a + 6))))
TETRA = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
PROBES = np.asarray([[1.375, 1.375, 0], [0.5, 1.5, 0], [1.0625, 1.5, 0]])
RAD = 1 / 64


def transform(raw, q):
    x, y, z = (np.asarray(raw) * [0.125, 0.25, 0.25]).T
    local = np.stack((0.125 - y, -0.25 + x, 0.0625 + z), axis=-1)
    c, s = np.cos(q), np.sin(q)
    return [0.5, 0.25, -0.125] + local @ np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def named(labels, key):
    return next(i for i, label in enumerate(labels) if label == key or label.endswith("/" + key))


def run_one(file, q, device):
    b = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, 0.0))
    b.rigid_gap = 0.0
    if file.suffix == ".urdf":
        b.add_urdf(str(file), floating=False, up_axis=newton.Axis.Z, joint_damping_mode="passive")
    else:
        b.add_mjcf(str(file), floating=False)
    points = transform(PROBES, q)
    cfg = newton.ModelBuilder.ShapeConfig(margin=0.0, gap=0.0, has_particle_collision=False)
    ids = []
    for i, point in enumerate(points):
        body = b.add_body(
            xform=wp.transform(wp.vec3(*point), wp.quat_identity()), label=f"probe_{i}"
        )
        ids.append(b.add_shape_sphere(body, radius=RAD, cfg=cfg, label=f"probe_shape_{i}"))
    model = b.finalize(device=device)
    state = model.state()
    joint = named(model.joint_label, "tilt")
    qbuf = state.joint_q.numpy()
    qbuf[model.joint_q_start.numpy()[joint]] = q
    state.joint_q.assign(qbuf)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    owners = model.shape_body.numpy()
    flags = model.shape_flags.numpy()
    solid = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    shapes = {}
    for name in ("base", "carrier"):
        body = named(model.body_label, name)
        (shape,) = np.flatnonzero((owners == body) & solid)
        shape = int(shape)
        assert int(model.shape_type.numpy()[shape]) == int(newton.GeoType.MESH)
        np.testing.assert_allclose(model.shape_scale.numpy()[shape], [1, 1, 1], rtol=0, atol=0)
        native = model.shape_source[shape]
        count = np.asarray(native.indices).size // 3
        corners = wp.empty(3 * count, dtype=wp.vec3, device=device)
        wp.launch(
            triangle_corners,
            dim=3 * count,
            inputs=[model.shape_source_ptr, model.shape_transform, state.body_q, shape, body],
            outputs=[corners],
            device=device,
        )
        got = corners.numpy().reshape(-1, 3, 3)
        expected = (
            (TETRA * [0.125, 0.25, 0.5] + [-1, 0, 0]) if name == "base" else transform(L_VERTS, q)
        )
        distances = np.linalg.norm(got[:, :, None, :] - expected[None, None, :, :], axis=-1)
        mapping = np.argmin(distances, axis=-1)
        error = float(distances.min(axis=-1).max())
        faces = {frozenset(tri) for tri in mapping}
        wanted = {frozenset(set(range(4)) - {i}) for i in range(4)} if name == "base" else L_FACES
        assert error < 4e-6 and len(got) == len(wanted) and faces == wanted, (
            file,
            name,
            count,
            error,
            faces,
            wanted,
        )
        assert str(corners.device) == str(device) and str(state.body_q.device) == str(device)
        shapes[name] = dict(id=shape, owner=body, face_count=count, max_vertex_error_m=error)
    shape = shapes["carrier"]["id"]
    body = shapes["carrier"]["owner"]
    distances = wp.empty(len(points), dtype=float, device=device)
    wp.launch(
        surface_distances,
        dim=len(points),
        inputs=[
            model.shape_source_ptr,
            model.shape_transform,
            state.body_q,
            shape,
            body,
            wp.array(points, dtype=wp.vec3, device=device),
        ],
        outputs=[distances],
        device=device,
    )
    actual = distances.numpy()
    np.testing.assert_allclose(actual, [0.046875, -0.0625, 0.0078125], rtol=0, atol=4e-6)
    pipe = newton.CollisionPipeline(
        model, broad_phase="nxn", reduce_contacts=False, rigid_contact_max=256
    )
    contacts = pipe.contacts()
    pipe.collide(state, contacts)
    gaps = wp.empty(contacts.rigid_contact_max, dtype=float, device=device)
    points0 = wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=device)
    points1 = wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=device)
    newton.eval_rigid_contact_kinematics(
        model,
        state,
        contacts,
        out_distance=gaps,
        out_point0_world=points0,
        out_point1_world=points1,
    )
    wp.synchronize_device(device)
    n = int(contacts.rigid_contact_count.numpy()[0])
    assert 0 <= n < contacts.rigid_contact_max
    c0 = contacts.rigid_contact_shape0.numpy()[:n]
    c1 = contacts.rigid_contact_shape1.numpy()[:n]
    gap = gaps.numpy()[:n]
    contacts_by_probe = []
    witnesses = []
    marg0, marg1 = (
        contacts.rigid_contact_margin0.numpy()[:n],
        contacts.rigid_contact_margin1.numpy()[:n],
    )
    normals = contacts.rigid_contact_normal.numpy()[:n]
    a_world, b_world = points0.numpy()[:n], points1.numpy()[:n]
    for probe in ids:
        mask = ((c0 == shape) & (c1 == probe)) | ((c0 == probe) & (c1 == shape))
        contacts_by_probe.append(gap[mask].tolist())
        for k in np.flatnonzero(mask):
            projected = float(normals[k] @ (b_world[k] - a_world[k]))
            np.testing.assert_allclose(gap[k], projected - marg0[k] - marg1[k], atol=3e-7, rtol=0)
            mesh_first = c0[k] == shape
            witnesses.append(
                dict(
                    probe=int(probe),
                    gap_m=float(gap[k]),
                    core_projection_m=projected,
                    mesh_margin_m=float(marg0[k] if mesh_first else marg1[k]),
                    sphere_margin_m=float(marg1[k] if mesh_first else marg0[k]),
                    mesh_core_world_m=(a_world[k] if mesh_first else b_world[k]).tolist(),
                    sphere_core_world_m=(b_world[k] if mesh_first else a_world[k]).tolist(),
                    normal_mesh_to_sphere=(normals[k] if mesh_first else -normals[k]).tolist(),
                )
            )
    assert not contacts_by_probe[0] and contacts_by_probe[2], (file, q, contacts_by_probe)
    deepest = min(contacts_by_probe[2])
    assert deepest < 0
    return dict(
        file=file.name,
        q=q,
        device=str(model.device),
        shapes=shapes,
        signed_center_distance_m=actual.tolist(),
        rigid_contact_distances_m=contacts_by_probe,
        witnesses=witnesses,
        analytic_cut_surface_gap_m=-0.0078125,
        cut_gap_error_m=deepest + 0.0078125,
        cut_gap_within_original_60um=abs(deepest + 0.0078125) <= 6e-5,
        registered_mesh_margin_m=float(model.shape_margin.numpy()[shape]),
        view="notch, solid leg, cut surface",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    args = p.parse_args()
    assert version("newton") == "1.7.0" and version("warp-lang") == "1.17.0"
    root = Path(__file__).parent
    with wp.ScopedDevice(args.device):
        rows = [
            run_one(root / name, q, args.device)
            for name in ("mesh_arm.urdf", "mesh_arm.xml")
            for q in (0.0, float(np.arctan2(0.6, 0.8)))
        ]
    print(
        json.dumps(
            dict(newton=version("newton"), warp=version("warp-lang"), records=rows), allow_nan=False
        )
    )


if __name__ == "__main__":
    main()

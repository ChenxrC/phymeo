"""Independent closed-bar/U and finite-shaft targets, with public discrete or swept observations."""

import argparse
import json
from dataclasses import asdict
from importlib.metadata import version
from math import acos, cos, hypot, sin, sqrt
from pathlib import Path

import numpy as np
from mesh_round_poses import geometry

HERE = Path(__file__).resolve().parent
CAT = json.loads((HERE / "rotating_mesh_capsule_cases.json").read_text("utf-8"))
R = CAT["capsule_radius_m"]
CYCLE = np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0)))


def coordinates(case):
    if case["mesh"] == "bar":
        raw, flat = geometry("box")
        low, high = np.asarray(CAT["bar_bounds_m"])
        vertices = low + (raw.astype(float) + 1) * (high - low) / 2
    else:
        vertices, flat = geometry("u")
    axis = np.array([[0, case["axis_y_m"], z] for z in CAT["axis_ends_z_m"]])
    placed = case["placement"] == "cyclic_translation"
    frame = CYCLE if placed else np.eye(3)
    origin = np.array((2.25, -0.75, 0.375)) if placed else np.zeros(3)
    return np.asarray(vertices, dtype=float), flat.reshape(-1, 3), axis, frame, origin


def rotate(points, angle):
    c, s = cos(angle), sin(angle)
    return np.asarray(points) @ np.array(((c, s, 0), (-s, c, 0), (0, 0, 1)))


def axis_skin_gap(case, t):
    # The vertical shaft straddles the entire extrusion, hence a 2D rectangle-distance
    # formula includes top/bottom boundary crossings even for an interior projection.
    x, y = rotate((0, case["axis_y_m"], 0), -case["angle_rad"] * t)[:2]
    if case["mesh"] == "bar":
        lo, hi = CAT["bar_bounds_m"]
        boxes = ((lo[0], hi[0], lo[1], hi[1]),)
    else:
        boxes = ((-2, -1, -2, 2), (1, 2, -2, 2), (-1, 1, -2, -1))
    return min(hypot(max(a - x, 0, x - b), max(c - y, 0, y - d)) for a, b, c, d in boxes) - R


def face_addresses(vertices, faces, point, side):
    found = []
    for face in faces:
        tri = vertices[face]
        if not np.all(tri[:, 1] == side):
            continue
        b, c = np.linalg.lstsq((tri[1:] - tri[0]).T, point - tri[0], rcond=None)[0]
        weights = np.array([1 - b - c, b, c])
        if weights.min() >= -1e-13:
            np.testing.assert_allclose(weights @ tri, point, atol=1e-13, rtol=0)
            found.append(dict(face=(41 + 7 * face).tolist(), barycentric=weights.tolist()))
    assert found
    return found


def oracle(case):
    vertices, faces, axis, frame, origin = coordinates(case)
    ys = case["axis_y_m"]
    if case["mesh"] == "u":
        # Every projected U point has radius >=1. The center's radius is 1/4.
        # At a quarter turn the inverse center points toward the x=1 wall.
        minimum = 1 - abs(ys) - R
        assert abs(axis_skin_gap(case, 0.25) - minimum) < 1e-14
        return dict(target="clear", full_path_min_skin_gap_m=minimum)
    lo, hi = CAT["bar_bounds_m"]
    reach = hypot(hi[0], hi[1])
    outer = abs(ys) - R - reach
    endpoint = min(lo[2] - axis[0, 2], axis[-1, 2] - hi[2]) - R
    assert endpoint == 5 / 8
    if outer > 0:
        phase = np.arctan2(hi[0], hi[1]) / abs(case["angle_rad"])
        assert abs(axis_skin_gap(case, phase) - outer) < 1e-14
        assert abs(outer - (17 - sqrt(257)) / 8) < 1e-14
        return dict(target="clear", full_path_min_skin_gap_m=outer, end_ball_lower_m=endpoint)
    side = np.sign(ys) * hi[1]
    time = acos((abs(side) + R) / abs(ys)) / abs(case["angle_rad"])
    cx = sqrt(ys**2 - (abs(side) + R) ** 2)
    assert lo[0] < cx < hi[0] and abs(cx - sqrt(35) / 4) < 1e-14
    assert axis_skin_gap(case, 7 / 16) > 0 > axis_skin_gap(case, 15 / 32)
    span = (np.array([lo[2], hi[2]]) - axis[0, 2]) / (axis[1, 2] - axis[0, 2])
    np.testing.assert_array_equal(span, [3 / 8, 5 / 8])
    witnesses = []
    for s in (*span[:1], 0.5, span[1]):
        core = (1 - s) * axis[0] + s * axis[1]
        local = np.array([cx, side, core[2]])
        normal = frame @ rotate((0, np.sign(ys), 0), case["angle_rad"] * time)
        world = frame @ rotate(local, case["angle_rad"] * time) + origin
        np.testing.assert_allclose(world, frame @ core + origin - R * normal, atol=1e-13, rtol=0)
        witnesses.append(
            dict(
                u=s,
                capsule_weights=[[41, 1 - s], [93, s]],
                face_material=face_addresses(vertices, faces, local, side),
                common_point_m=world.tolist(),
                mesh_to_capsule_normal=normal.tolist(),
            )
        )
    return dict(
        target="first_contact",
        t=time,
        entry_local_x_m=cx,
        axis_span=span.tolist(),
        end_ball_lower_m=endpoint,
        witness_examples=witnesses,
    )


def read_host(case, target):
    import phymeo as ph

    vertices, faces, axis, frame, origin = coordinates(case)
    reverse = case["body_order"] == "capsule_first"
    wm, wc = case["worlds"]
    mk, ck = ph.CollisionKey(wm, int(reverse), 0), ph.CollisionKey(wc, int(not reverse), 0)
    ids = tuple(41 + 7 * i for i in range(len(vertices)))
    mesh = ph.CollisionShape(
        mk, "mesh", ids, triangles=tuple(tuple(ids[i] for i in f) for f in faces)
    )
    cap = ph.CollisionShape(ck, "capsule", (41, 93), R)
    model = ph.CollisionModel([mesh, cap])
    turn = ph.RigidTurn(origin, origin, frame @ (0, 0, 1), case["angle_rad"])
    paths = {
        mk: ph.CollisionMotion(vertices @ frame.T + origin, turn=turn),
        ck: ph.CollisionMotion(axis @ frame.T + origin),
    }
    poses = []
    with ph.CollisionWorkspace(model, "host") as work:
        (row,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
        if wm != wc:
            assert (
                not row.candidate and row.status == "filtered" and row.reason == "different_world"
            )
            coverage = "filtered_by_world"
        elif row.status in ("unsupported", "unresolved"):
            assert row.sweep is None
            coverage = "open_ccd"
        elif target["target"] == "clear":
            assert row.status == "separated"
            coverage = "resolved_clear"
        else:
            assert row.status == "hit" and row.sweep is not None
            lo, hi = row.sweep.first_t
            assert lo <= target["t"] <= hi and lo <= row.sweep.witness_time <= hi
            assert axis_skin_gap(case, row.sweep.witness_time) <= 1e-12
            coverage = "resolved_time"
        if wm == wc:
            for t in CAT["pose_samples"]:
                paths[mk] = ph.CollisionMotion(
                    rotate(vertices, t * case["angle_rad"]) @ frame.T + origin
                )
                (static,) = ph.detect(model, paths, ph.CollisionPolicy(margin=4), work).pairs
                expected = axis_skin_gap(case, t)
                assert static.status == ("separated" if expected > 0 else "overlap")
                pose = static.mesh_round.start
                assert abs(pose.axis_gap_m - expected) < 1e-12
                poses.append(
                    dict(
                        t=t,
                        expected_gap_m=expected,
                        relation=pose.relation,
                        status=static.status,
                        reported_gap_m=pose.axis_gap_m,
                        material=asdict(pose.supports[0]),
                    )
                )
    return dict(
        coverage=coverage,
        status=row.status,
        reason=row.reason,
        candidate=row.candidate,
        published_sweep=None if row.sweep is None else asdict(row.sweep),
        discrete_poses=poses,
    )


def read_newton(case, t, device):
    import newton
    import warp as wp

    vertices, faces, axis, frame, origin = coordinates(case)
    assert np.array_equal(frame, np.eye(3)) and np.array_equal(origin, np.zeros(3))
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        cfg = builder.ShapeConfig(margin=0, gap=0, has_particle_collision=False)
        mesh = builder.add_shape_mesh(
            builder.add_body(),
            mesh=newton.Mesh(
                rotate(vertices, t * case["angle_rad"]).astype(np.float32),
                faces.astype(np.int32).ravel(),
            ),
            cfg=cfg,
        )
        cap = builder.add_shape_capsule(
            builder.add_body(xform=wp.transform(axis.mean(0), wp.quat_identity())),
            radius=R,
            half_height=np.linalg.norm(axis[1] - axis[0]) / 2,
            cfg=cfg,
        )
        model = builder.finalize(device=device)
        types = [newton.GeoType(int(x)).name for x in model.shape_type.numpy()[[mesh, cap]]]
        assert types == ["MESH", "CAPSULE"]
        pipe = newton.CollisionPipeline(
            model, broad_phase="nxn", reduce_contacts=False, rigid_contact_max=256
        )
        state, contacts = model.state(), pipe.contacts()
        pipe.collide(state, contacts)
        gap = wp.empty(contacts.rigid_contact_max, dtype=float, device=device)
        a, b = [
            wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=device) for _ in range(2)
        ]
        newton.eval_rigid_contact_kinematics(
            model, state, contacts, out_distance=gap, out_point0_world=a, out_point1_world=b
        )
        wp.synchronize_device(device)
        count = int(contacts.rigid_contact_count.numpy()[0])
        assert 0 <= count < contacts.rigid_contact_max and str(
            contacts.rigid_contact_count.device
        ) == str(device)
        arrays = [
            x.numpy()[:count]
            for x in (
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                gap,
                contacts.rigid_contact_normal,
                a,
                b,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
            )
        ]
        rows = []
        for i, j, d, n, p, q, op, oq in zip(*arrays, strict=True):
            assert {int(i), int(j)} == {mesh, cap}
            if int(i) != mesh:
                n, p, q, op, oq = -n, q, p, oq, op
            mp, cp = p + op, q + oq
            assert np.isfinite([d, *n, *mp, *cp]).all()
            np.testing.assert_allclose((cp - mp) @ n, d, atol=2e-6, rtol=0)
            rows.append(
                dict(
                    gap_m=float(d),
                    normal_mesh_to_capsule=n.tolist(),
                    mesh_point_m=mp.tolist(),
                    capsule_point_m=cp.tolist(),
                    mesh_offset_m=op.tolist(),
                    capsule_offset_m=oq.tolist(),
                )
            )
    return dict(
        t=t,
        expected_axis_gap_m=axis_skin_gap(case, t),
        native_geo_types=types,
        count=count,
        contacts=rows,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, choices=("phymeo", "newton"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    cases = CAT["cases"]
    if args.engine == "newton":
        assert version("newton") == "1.7.0"
        cases = [case for case in cases if case["id"] == "bar-shaft"]
    elif args.device != "cpu":
        parser.error("the public Phymeo lane uses HOST")
    records = []
    for case in cases:
        target = oracle(case)
        assert target["target"] == CAT["analytic_targets"][case["id"]]["status"]
        if args.engine == "phymeo":
            observed = read_host(case, target)
        else:
            observed = [read_newton(case, t, args.device) for t in (7 / 16, 15 / 32)]
        records.append(dict(id=case["id"], target=target, observed=observed))
    output = dict(engine=args.engine, device=args.device, cases=records)
    with args.out.open("x", encoding="utf-8") as file:
        json.dump(output, file, allow_nan=False, indent=2)
    for row in records:
        result = row["observed"]
        print(
            row["id"],
            result["coverage"] if isinstance(result, dict) else [r["count"] for r in result],
        )


if __name__ == "__main__":
    main()

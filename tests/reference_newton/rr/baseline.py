"""Rigid geometry comparison. Each engine is imported only when its lane is run."""

import json
from pathlib import Path

import numpy as np


CATALOG = json.loads(Path(__file__).with_name("cases.json").read_text(encoding="utf-8"))


def _position(path, fraction):
    start, end = np.asarray(path, dtype=float)
    return start + fraction * (end - start)


def _u_geometry(scale, flip):
    prism = CATALOG["u_prism"]
    ring = prism["u_ring_xy_m"]
    n = len(ring)
    vertices = np.array([[x, y, z] for z in prism["u_z_m"] for x, y in ring]) * scale

    def triangles(i, j, k, last):
        if flip:
            return [(i, j, last), (j, k, last)]
        return [(i, j, k), (i, k, last)]

    cap = [tri for quad in prism["u_cap_quads_ccw"] for tri in triangles(*quad)]
    faces = [tri[::-1] for tri in cap] + [tuple(i + n for i in tri) for tri in cap]
    for i in range(n):
        j = (i + 1) % n
        faces.extend(triangles(i, j, j + n, i + n))
    return vertices.astype(np.float32), np.asarray(faces, dtype=np.int32).ravel()


def _add_shape(builder, asset, path, cfg, flip):
    import newton
    import warp as wp

    kind = asset["kind"]
    if kind in ("plane", "heightfield"):
        if np.any(path):
            raise ValueError("The terrain and ground fixtures are fixed at the world origin")
        if kind == "plane":
            return builder.add_ground_plane(cfg=cfg)
        grid = np.asarray(asset["heights_yx_m"], dtype=np.float32)
        hx, hy = asset["half_extent_xy_m"]
        field = newton.Heightfield(data=grid, nrow=grid.shape[0], ncol=grid.shape[1], hx=hx, hy=hy)
        return builder.add_shape_heightfield(heightfield=field, cfg=cfg)

    body = builder.add_body(xform=wp.transform(path[0], wp.quat_identity()))
    if kind == "sphere":
        return builder.add_shape_sphere(body, radius=asset["radius_m"], cfg=cfg)
    if kind == "u_mesh":
        vertices, indices = _u_geometry(asset["scale_xyz"], flip)
        return builder.add_shape_mesh(body, mesh=newton.Mesh(vertices, indices), cfg=cfg)
    rotation = wp.transform((0.0, 0.0, 0.0), asset.get("local_quat_xyzw", (0, 0, 0, 1)))
    if kind == "ellipsoid":
        rx, ry, rz = asset["radii_xyz_m"]
        return builder.add_shape_ellipsoid(body, rx=rx, ry=ry, rz=rz, xform=rotation, cfg=cfg)
    if kind not in ("cylinder", "cone"):
        raise ValueError(f"Unknown RR shape {kind}")
    add = builder.add_shape_cylinder if kind == "cylinder" else builder.add_shape_cone
    return add(body, radius=asset["radius_m"], half_height=asset["half_height_m"], xform=rotation, cfg=cfg)


def _deepest_newton_contact(contacts, count, distance, points, shape_ids):
    first = contacts.rigid_contact_shape0.numpy()[:count]
    second = contacts.rigid_contact_shape1.numpy()[:count]
    assert np.all(((first == shape_ids[0]) & (second == shape_ids[1])) |
                  ((first == shape_ids[1]) & (second == shape_ids[0]))), "Unexpected shape pair"
    gaps = distance.numpy()[:count]
    normals = contacts.rigid_contact_normal.numpy()[:count]
    assert np.isfinite(gaps).all()
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=0, atol=2e-4)
    deepest = int(np.argmin(gaps))
    side_a = int(first[deepest] != shape_ids[0])
    normal = normals[deepest] if side_a == 0 else -normals[deepest]
    a_point = points[side_a].numpy()[deepest]
    b_point = points[1 - side_a].numpy()[deepest]
    assert np.isfinite([a_point, b_point]).all()
    return dict(gap_m=float(gaps[deepest]), normal_a_to_b=normal.tolist(),
                a_support_world_m=a_point.tolist(), b_support_world_m=b_point.tolist(),
                observed_pair_ids=[int(first[deepest]), int(second[deepest])])


def _newton_samples(inputs, assets, device):
    import newton
    import warp as wp

    paths = (inputs["a_path_m"], inputs["b_path_m"])
    selected = wp.get_device(device)
    assert str(selected) == device and (device == "cpu" or selected.is_cuda)
    with wp.ScopedDevice(selected):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        cfg = builder.ShapeConfig(margin=0.0, gap=0.0, has_particle_collision=False)
        shape_ids = [None, None]
        for side in ((1, 0) if inputs.get("reverse_registration") else (0, 1)):
            shape_ids[side] = _add_shape(
                builder, assets[side], paths[side], cfg, inputs.get("flip_quad_diagonal", False)
            )
        model = builder.finalize(device=selected)
        bodies = model.shape_body.numpy()[shape_ids]
        assert (bodies >= 0).tolist() == [a["kind"] not in ("plane", "heightfield") for a in assets]
        expected_types = {"u_mesh": "MESH", "heightfield": "HFIELD"}
        types = [newton.GeoType(int(i)).name for i in model.shape_type.numpy()[shape_ids]]
        assert types == [expected_types.get(a["kind"], a["kind"].upper()) for a in assets]
        dynamic = bodies[bodies >= 0]
        masses = model.body_mass.numpy()[dynamic]
        assert len(set(dynamic)) == len(dynamic) and np.all(np.isfinite(masses)) and np.all(masses > 0)
        state = model.state()
        pipeline = newton.CollisionPipeline(
            model, broad_phase="nxn", reduce_contacts=False, rigid_contact_max=256
        )
        contacts = pipeline.contacts()
        assert str(contacts.rigid_contact_count.device) == device
        distance = wp.empty(contacts.rigid_contact_max, dtype=float, device=selected)
        points = [wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=selected) for _ in shape_ids]
        poses = state.body_q.numpy().copy()
        samples = []
        for fraction in inputs["sample_fractions"]:
            for body, path in zip(bodies, paths, strict=True):
                if body >= 0:
                    poses[body, :3] = _position(path, fraction)
            state.body_q.assign(poses)
            pipeline.collide(state, contacts)
            newton.eval_rigid_contact_kinematics(
                model, state, contacts, out_distance=distance,
                out_point0_world=points[0], out_point1_world=points[1]
            )
            wp.synchronize_device(selected)
            count = int(contacts.rigid_contact_count.numpy()[0])
            assert 0 <= count < contacts.rigid_contact_max, f"Contact count {count} exhausted the buffer"
            readback = state.body_q.numpy()
            translations = [readback[b, :3].tolist() if b >= 0 else [0.0, 0.0, 0.0] for b in bodies]
            for path, published in zip(paths, translations, strict=True):
                np.testing.assert_allclose(published, _position(path, fraction), rtol=0, atol=5e-7)
            sample = dict(fraction=fraction, prescribed_time_s=fraction * inputs["duration_s"],
                          origins_world_m=translations, contact=count > 0, contact_count=count)
            if count:
                sample.update(_deepest_newton_contact(contacts, count, distance, points, shape_ids))
            samples.append(sample)
        return dict(device=str(contacts.rigid_contact_count.device), shape_ids=shape_ids, shape_types=types,
                    body_ids=bodies.tolist(), dynamic_masses_kg=masses.tolist(), samples=samples)


def _phymeo(case, inputs, assets, device):
    import phymeo as ph

    if any(asset["kind"] != "sphere" for asset in assets):
        raise ValueError("The comparison has no Phymeo adapter for these native shapes")
    keys = (ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0))
    model = ph.CollisionModel([ph.CollisionShape(key, "sphere", (0,), asset["radius_m"])
                               for key, asset in zip(keys, assets, strict=True)])
    paths = (inputs["a_path_m"], inputs["b_path_m"])
    mode = {"cpu": "host", "cuda:0": "cuda"}[device]
    # This margin only retains separated pairs for signed-distance readout.
    policy = ph.CollisionPolicy(margin=1.0)
    with ph.CollisionWorkspace(model, mode) as work:
        if inputs["query"] == "sampled_poses":
            samples = []
            for fraction in inputs["sample_fractions"]:
                motion = {key: ph.CollisionMotion([_position(path, fraction)])
                          for key, path in zip(keys, paths, strict=True)}
                (row,) = ph.detect(model, motion, policy, work).pairs
                assert row.first == keys[0] and row.second == keys[1] and row.family == "RR"
                assert row.status in ("separated", "overlap", "touching") and row.witness is not None
                samples.append(dict(fraction=fraction, prescribed_time_s=fraction * inputs["duration_s"],
                                    status=row.status, contact=row.status != "separated",
                                    gap_m=row.witness.gap_m, normal_a_to_b=row.witness.normal))
            return dict(backend=work.stats()["backend"], samples=samples)

        motion = {key: ph.CollisionMotion([start], [end])
                  for key, (start, end) in zip(keys, paths, strict=True)}
        (row,) = ph.detect(model, motion, policy, work).pairs
        assert (row.first, row.second, row.family, row.status) == (*keys, "RR", "hit"), row
        assert row.sweep is not None and row.sweep.first_t is not None and row.witness is not None
        lo, hi = row.sweep.first_t
        observed = dict(first_interval_fraction=[lo, hi],
                        first_interval_s=[v * inputs["duration_s"] for v in (lo, hi)],
                        start_gap_m=row.sweep.start_gap_m,
                        end_gap_m=row.witness.gap_m, first_gap_m=row.sweep.witness.gap_m,
                        first_normal_a_to_b=row.sweep.witness.normal)
        expected, tol = case["expected"], case["tolerances"]
        assert lo <= expected["first_fraction"] <= hi and 0 <= hi - lo <= tol["time_fraction"], observed
        lower_s, upper_s = observed["first_interval_s"]
        assert lower_s <= expected["first_time_s"] <= upper_s, observed
        np.testing.assert_allclose([observed["start_gap_m"], observed["end_gap_m"], observed["first_gap_m"]],
                                   [expected["start_gap_m"], expected["end_gap_m"], 0],
                                   rtol=0, atol=tol["length_m"])
        np.testing.assert_allclose(observed["first_normal_a_to_b"], expected["first_normal_a_to_b"],
                                   rtol=0, atol=tol["normal_component"])
        stats = work.stats()
        observed["backend"] = stats["backend"]
        if device == "cuda:0":
            counters = ("completed_launches", "sweep_completed_launches", "sweep_downloads",
                        "sweep_synchronizations")
            for key in counters:
                assert stats[key] > 0, (key, stats[key])
                observed[key] = stats[key]
        return observed


def _check_samples(case, observed):
    tol = case["tolerances"]
    for sample, target in zip(observed["samples"], case["expected"]["samples"], strict=True):
        context = f"{case['id']} at {sample['fraction']}: {sample}"
        assert sample["contact"] == target["contact"], context
        if not target["contact"]:
            if "gap_m" in sample and "host_witness_gap_m" in target:
                np.testing.assert_allclose(sample["gap_m"], target["host_witness_gap_m"],
                                           rtol=0, atol=tol["length_m"], err_msg=context)
            continue
        gap = sample["gap_m"]
        if "gap_range_m" in target:
            lo, hi = target["gap_range_m"]
            assert lo - tol["length_m"] <= gap <= hi + tol["length_m"], context
        else:
            np.testing.assert_allclose(gap, target["gap_m"], rtol=0, atol=tol["length_m"], err_msg=context)
        np.testing.assert_allclose(sample["normal_a_to_b"], target["normal_a_to_b"],
                                   rtol=0, atol=tol["normal_component"], err_msg=context)
        # The receiver has zero effective support radius in all landmark cases.
        if "a_point_world_m" in target:
            np.testing.assert_allclose(sample["a_support_world_m"], target["a_point_world_m"],
                                       rtol=0, atol=tol["length_m"], err_msg=context)
        if "a_local_x_m" in target:
            local_x = sample["a_support_world_m"][0] - sample["origins_world_m"][0][0]
            np.testing.assert_allclose(local_x, target["a_local_x_m"],
                                       rtol=0, atol=tol["length_m"], err_msg=context)


def run_case(case, *, engine, device):
    inputs = case["inputs"]
    assets = [CATALOG["assets"][name] for name in inputs["assets"]]
    if engine == "phymeo":
        observed = _phymeo(case, inputs, assets, device)
    elif engine == "newton" and inputs["query"] == "sampled_poses":
        observed = _newton_samples(inputs, assets, device)
    else:
        raise ValueError(f"No {engine} adapter for {inputs['query']}")
    if inputs["query"] == "sampled_poses":
        _check_samples(case, observed)
    return observed

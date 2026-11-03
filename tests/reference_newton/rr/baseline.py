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
    return add(
        body, radius=asset["radius_m"], half_height=asset["half_height_m"], xform=rotation, cfg=cfg
    )


def _newton_witnesses(contacts, count, distance, points, shape_ids, keep_all):
    first = contacts.rigid_contact_shape0.numpy()[:count]
    second = contacts.rigid_contact_shape1.numpy()[:count]
    assert np.all(
        ((first == shape_ids[0]) & (second == shape_ids[1]))
        | ((first == shape_ids[1]) & (second == shape_ids[0]))
    ), "Unexpected shape pair"
    gaps = distance.numpy()[:count]
    normals = contacts.rigid_contact_normal.numpy()[:count]
    assert np.isfinite(gaps).all()
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, rtol=0, atol=2e-4)
    support = [array.numpy()[:count] for array in points]
    witnesses = []
    for i in range(count) if keep_all else [int(np.argmin(gaps))]:
        side_a = int(first[i] != shape_ids[0])
        normal = normals[i] if side_a == 0 else -normals[i]
        a_point, b_point = support[side_a][i], support[1 - side_a][i]
        assert np.isfinite([a_point, b_point]).all()
        row = dict(
            gap_m=float(gaps[i]),
            normal_a_to_b=normal.tolist(),
            a_support_world_m=a_point.tolist(),
            b_support_world_m=b_point.tolist(),
            observed_pair_ids=[int(first[i]), int(second[i])],
        )
        if keep_all:
            row["buffer_index"] = i
        witnesses.append(row)
    return {"contacts": witnesses} if keep_all else witnesses[0]


def _newton_samples(inputs, assets, device, *, keep_all=False):
    import newton
    import warp as wp

    paths = (inputs["a_path_m"], inputs["b_path_m"])
    selected = wp.get_device(device)
    assert str(selected) == device and (device == "cpu" or selected.is_cuda)
    with wp.ScopedDevice(selected):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        cfg = builder.ShapeConfig(margin=0.0, gap=0.0, has_particle_collision=False)
        shape_ids = [None, None]
        for side in (1, 0) if inputs.get("reverse_registration") else (0, 1):
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
        assert (
            len(set(dynamic)) == len(dynamic) and np.all(np.isfinite(masses)) and np.all(masses > 0)
        )
        state = model.state()
        pipeline = newton.CollisionPipeline(
            model, broad_phase="nxn", reduce_contacts=False, rigid_contact_max=256
        )
        contacts = pipeline.contacts()
        assert str(contacts.rigid_contact_count.device) == device
        distance = wp.empty(contacts.rigid_contact_max, dtype=float, device=selected)
        points = [
            wp.empty(contacts.rigid_contact_max, dtype=wp.vec3, device=selected) for _ in shape_ids
        ]
        poses = state.body_q.numpy().copy()
        samples = []
        for fraction in inputs["sample_fractions"]:
            for body, path in zip(bodies, paths, strict=True):
                if body >= 0:
                    poses[body, :3] = _position(path, fraction)
            state.body_q.assign(poses)
            pipeline.collide(state, contacts)
            newton.eval_rigid_contact_kinematics(
                model,
                state,
                contacts,
                out_distance=distance,
                out_point0_world=points[0],
                out_point1_world=points[1],
            )
            wp.synchronize_device(selected)
            count = int(contacts.rigid_contact_count.numpy()[0])
            assert 0 <= count < contacts.rigid_contact_max, (
                f"Contact count {count} exhausted the buffer"
            )
            readback = state.body_q.numpy()
            translations = [readback[b, :3].tolist() if b >= 0 else [0.0, 0.0, 0.0] for b in bodies]
            for path, published in zip(paths, translations, strict=True):
                np.testing.assert_allclose(published, _position(path, fraction), rtol=0, atol=5e-7)
            sample = dict(
                fraction=fraction,
                prescribed_time_s=fraction * inputs["duration_s"],
                origins_world_m=translations,
                contact=count > 0,
                contact_count=count,
            )
            if count:
                sample.update(
                    _newton_witnesses(contacts, count, distance, points, shape_ids, keep_all)
                )
            elif keep_all:
                sample["contacts"] = []
            samples.append(sample)
        return dict(
            device=str(contacts.rigid_contact_count.device),
            shape_ids=shape_ids,
            shape_types=types,
            body_ids=bodies.tolist(),
            dynamic_masses_kg=masses.tolist(),
            samples=samples,
        )


def _heightfield_sweep_observation(case, pair, grid_motion, path, radius):
    expected, tol = case["expected"], case["tolerances"]
    assert pair.status == "hit" and pair.sweep.start_gap_m > 0 and pair.witness.gap_m > 0
    observed = dict(start_gap_m=pair.sweep.start_gap_m, end_gap_m=pair.witness.gap_m)
    duration = case["inputs"]["duration_s"]
    for name, target, event in (
        ("entry", expected["first_fraction"], pair.heightfield.first_contact),
        ("last_contact", expected["leave_fraction"], pair.heightfield.last_contact),
    ):
        lo, hi = event.interval
        assert lo <= target <= hi and hi - lo <= tol["time_fraction"]
        assert lo <= event.witness_time <= hi
        center = _position(path, event.witness_time)
        w, face = event.witness, event.face
        np.testing.assert_allclose(
            np.asarray(face.barycentric) @ grid_motion.start[list(face.vertex_ids)],
            w.first_point,
            rtol=0,
            atol=tol["length_m"],
        )
        np.testing.assert_allclose(
            center - radius * np.asarray(w.normal), w.second_point, rtol=0, atol=tol["length_m"]
        )
        assert abs(w.gap_m) <= tol["length_m"]
        observed[name] = dict(
            interval_fraction=event.interval,
            interval_s=(lo * duration, hi * duration),
            witness_fraction=event.witness_time,
            gap_m=w.gap_m,
            center_m=center.tolist(),
            normal_a_to_b=w.normal,
            terrain_point_m=w.first_point,
            sphere_point_m=w.second_point,
            cell=face.cell,
            triangle=face.triangle,
            vertex_ids=face.vertex_ids,
            barycentric=face.barycentric,
        )
    first = observed["entry"]
    assert first["interval_fraction"] == pair.sweep.first_t
    assert first["interval_s"][0] <= expected["first_time_s"] <= first["interval_s"][1]
    np.testing.assert_allclose(
        first["center_m"], expected["first_center_m"], rtol=0, atol=tol["length_m"]
    )
    np.testing.assert_allclose(
        first["terrain_point_m"], expected["first_point_world_m"], rtol=0, atol=tol["length_m"]
    )
    np.testing.assert_allclose(
        first["normal_a_to_b"],
        expected["first_normal_a_to_b"],
        rtol=0,
        atol=tol["normal_component"],
    )
    return observed


def _phymeo_heightfield(case, inputs, assets, device):
    import phymeo as ph

    if (
        device not in ("cpu", "cuda:0")
        or (device == "cuda:0" and inputs["query"] != "first_touch")
        or inputs["query"] not in ("sampled_poses", "first_touch")
        or np.any(inputs["a_path_m"])
    ):
        raise ValueError("The heightfield comparison needs a fixed grid and a supported query")
    field, ball = assets
    heights = np.array(field["heights_yx_m"])
    rows, columns = heights.shape
    hx, hy = field["half_extent_xy_m"]
    grid = ph.Heightfield(
        heights, (2 * hx / (columns - 1), 2 * hy / (rows - 1)), origin=(-hx, -hy, 0)
    )
    field_key, ball_key = ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0)
    terrain = ph.CollisionShape(
        field_key, "heightfield", tuple(range(grid.vertex_count)), heightfield=grid
    )
    model = ph.CollisionModel(
        [terrain, ph.CollisionShape(ball_key, "sphere", (0,), ball["radius_m"])]
    )
    field_motion = ph.CollisionMotion(grid.world_vertices())
    samples = []
    mode = "cuda" if device == "cuda:0" else "host"
    with ph.CollisionWorkspace(model, mode) as work:
        if inputs["query"] == "first_touch":
            start, end = inputs["b_path_m"]
            paths = {field_key: field_motion, ball_key: ph.CollisionMotion([start], [end])}
            (pair,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
            assert pair.first == field_key and pair.second == ball_key and pair.family == "RR"
            values = _heightfield_sweep_observation(
                case, pair, field_motion, inputs["b_path_m"], ball["radius_m"]
            )
            stats = work.stats()
            if mode == "cuda":
                counters = {key: n for key, n in stats.items() if key.startswith("heightfield_")}
                for key in ("completed_launches", "downloads", "synchronizations"):
                    assert counters[f"heightfield_{key}"] > 0
                values["device_face_selection"] = counters
            return dict(backend=stats["backend"], **values)
        for fraction in inputs["sample_fractions"]:
            center = _position(inputs["b_path_m"], fraction)
            paths = {field_key: field_motion, ball_key: ph.CollisionMotion([center])}
            (pair,) = ph.detect(model, paths, ph.CollisionPolicy(), work).pairs
            assert pair.family == "RR" and pair.first == field_key and pair.second == ball_key
            assert pair.status in ("separated", "touching", "overlap") and pair.sweep is None
            sample = dict(
                fraction=fraction,
                prescribed_time_s=fraction * inputs["duration_s"],
                contact=pair.status != "separated",
                status=pair.status,
                reason=pair.reason,
            )
            if pair.witness is not None:
                w, chart = pair.witness, pair.heightfield
                assert chart is not None
                tri = field_motion.start[list(chart.vertex_ids)]
                np.testing.assert_allclose(
                    chart.barycentric @ tri, w.first_point, rtol=0, atol=1e-10
                )
                sample.update(
                    gap_m=w.gap_m,
                    normal_a_to_b=w.normal,
                    a_support_world_m=w.first_point,
                    b_support_world_m=w.second_point,
                    cell=chart.cell,
                    triangle=chart.triangle,
                    vertex_ids=chart.vertex_ids,
                    barycentric=chart.barycentric,
                )
            samples.append(sample)
        return dict(backend=work.stats()["backend"], samples=samples)


def _phymeo(case, inputs, assets, device):
    import phymeo as ph

    if any(asset["kind"] in ("cylinder", "cone", "ellipsoid") for asset in assets):
        from .curved_reference import phymeo_curved

        return phymeo_curved(case, inputs, assets, device)
    if [a["kind"] for a in assets] == ["heightfield", "sphere"]:
        return _phymeo_heightfield(case, inputs, assets, device)
    if all(asset["kind"] == "u_mesh" for asset in assets):
        from .mesh_mesh_reference import run_case

        return run_case(case, inputs, assets, device)
    if any(asset["kind"] != "sphere" for asset in assets):
        raise ValueError("The comparison has no Phymeo adapter for these native shapes")
    keys = (ph.CollisionKey(0, 1, 0), ph.CollisionKey(0, 2, 0))
    model = ph.CollisionModel(
        [
            ph.CollisionShape(key, "sphere", (0,), asset["radius_m"])
            for key, asset in zip(keys, assets, strict=True)
        ]
    )
    paths = (inputs["a_path_m"], inputs["b_path_m"])
    mode = {"cpu": "host", "cuda:0": "cuda"}[device]
    # This margin only retains separated pairs for signed-distance readout.
    policy = ph.CollisionPolicy(margin=1.0)
    with ph.CollisionWorkspace(model, mode) as work:
        if inputs["query"] == "sampled_poses":
            samples = []
            for fraction in inputs["sample_fractions"]:
                motion = {
                    key: ph.CollisionMotion([_position(path, fraction)])
                    for key, path in zip(keys, paths, strict=True)
                }
                (row,) = ph.detect(model, motion, policy, work).pairs
                assert row.first == keys[0] and row.second == keys[1] and row.family == "RR"
                assert (
                    row.status in ("separated", "overlap", "touching") and row.witness is not None
                )
                samples.append(
                    dict(
                        fraction=fraction,
                        prescribed_time_s=fraction * inputs["duration_s"],
                        status=row.status,
                        contact=row.status != "separated",
                        gap_m=row.witness.gap_m,
                        normal_a_to_b=row.witness.normal,
                    )
                )
            return dict(backend=work.stats()["backend"], samples=samples)

        motion = {
            key: ph.CollisionMotion([start], [end])
            for key, (start, end) in zip(keys, paths, strict=True)
        }
        (row,) = ph.detect(model, motion, policy, work).pairs
        assert (row.first, row.second, row.family, row.status) == (*keys, "RR", "hit"), row
        assert row.sweep is not None and row.sweep.first_t is not None and row.witness is not None
        lo, hi = row.sweep.first_t
        observed = dict(
            first_interval_fraction=[lo, hi],
            first_interval_s=[v * inputs["duration_s"] for v in (lo, hi)],
            start_gap_m=row.sweep.start_gap_m,
            end_gap_m=row.witness.gap_m,
            first_gap_m=row.sweep.witness.gap_m,
            first_normal_a_to_b=row.sweep.witness.normal,
        )
        expected, tol = case["expected"], case["tolerances"]
        assert lo <= expected["first_fraction"] <= hi and 0 <= hi - lo <= tol["time_fraction"], (
            observed
        )
        lower_s, upper_s = observed["first_interval_s"]
        assert lower_s <= expected["first_time_s"] <= upper_s, observed
        np.testing.assert_allclose(
            [observed["start_gap_m"], observed["end_gap_m"], observed["first_gap_m"]],
            [expected["start_gap_m"], expected["end_gap_m"], 0],
            rtol=0,
            atol=tol["length_m"],
        )
        np.testing.assert_allclose(
            observed["first_normal_a_to_b"],
            expected["first_normal_a_to_b"],
            rtol=0,
            atol=tol["normal_component"],
        )
        stats = work.stats()
        observed["backend"] = stats["backend"]
        if device == "cuda:0":
            counters = (
                "completed_launches",
                "sweep_completed_launches",
                "sweep_downloads",
                "sweep_synchronizations",
            )
            for key in counters:
                assert stats[key] > 0, (key, stats[key])
                observed[key] = stats[key]
        return observed


def _u_surface_distance(local_point, scale):
    """Exact distance to the boundary of the extruded ring, independent of mesh triangulation."""
    ring = np.asarray(CATALOG["u_prism"]["u_ring_xy_m"]) * scale[:2]
    z_low, z_high = np.asarray(CATALOG["u_prism"]["u_z_m"]) * scale[2]
    xy, z = local_point[:2], local_point[2]
    ends = np.roll(ring, -1, axis=0)
    edges = ends - ring
    delta = xy - ring
    along = np.clip(np.sum(delta * edges, axis=1) / np.sum(edges * edges, axis=1), 0, 1)
    edge_distance = min(np.linalg.norm(delta - along[:, None] * edges, axis=1))
    # Half-open ray crossings exclude horizontal edges and count a shared vertex once.
    cross = (ring[:, 1] > xy[1]) != (ends[:, 1] > xy[1])
    crossing_x = ring[cross, 0] + delta[cross, 1] * edges[cross, 0] / edges[cross, 1]
    inside_xy = np.count_nonzero(xy[0] < crossing_x) % 2 == 1
    outside_z = max(z_low - z, z - z_high, 0.0)
    if inside_xy:
        return float(outside_z if outside_z else min(edge_distance, z - z_low, z_high - z))
    return float(np.hypot(edge_distance, outside_z))


def _check_u_walls(case, observed, assets):
    """Test the original wall criteria against every contact tied at the deepest gap."""
    expected = case["expected"]["samples"]
    wall = next(row for row in expected if "a_local_x_m" in row)
    gap_low, gap_high = wall["gap_range_m"]
    length_tol = case["tolerances"]["length_m"]
    normal_tol = case["tolerances"]["normal_component"]
    a_wall_x = wall["a_local_x_m"]
    insert_x = np.asarray(CATALOG["u_prism"]["u_ring_xy_m"])[:, 0] * assets[1]["scale_xyz"][0]
    b_outer_x = float(max(insert_x) if a_wall_x > 0 else min(insert_x))
    failures = []
    for sample, target in zip(observed["samples"], expected, strict=True):
        contacts = sample["contacts"]
        minimum = min((row["gap_m"] for row in contacts), default=None)
        origin_a, origin_b = np.asarray(sample["origins_world_m"])
        for row in contacts:
            a = np.asarray(row["a_support_world_m"]) - origin_a
            b = np.asarray(row["b_support_world_m"]) - origin_b
            a_error, b_error = float(a[0] - a_wall_x), float(b[0] - b_outer_x)
            normal_error = float(
                np.max(np.abs(np.asarray(row["normal_a_to_b"]) - wall["normal_a_to_b"]))
            )
            row.update(
                is_deepest=row["gap_m"] - minimum <= length_tol,
                normal_max_component_error=normal_error,
                normal_matches=normal_error <= normal_tol,
                a_wall_x_error_m=a_error,
                b_outer_face_x_error_m=b_error,
                a_wall_x_matches=abs(a_error) <= length_tol,
                b_outer_face_x_matches=abs(b_error) <= length_tol,
                a_surface_distance_m=_u_surface_distance(a, assets[0]["scale_xyz"]),
                b_surface_distance_m=_u_surface_distance(b, assets[1]["scale_xyz"]),
                depth_matches=gap_low - length_tol <= row["gap_m"] <= gap_high + length_tol,
            )
        tied = [row for row in contacts if row["is_deepest"]]
        summary = dict(
            minimum_gap_m=minimum,
            band_m=length_tol,
            count=len(tied),
            normal_matches=sum(row["normal_matches"] for row in tied),
            a_wall_x_matches=sum(row["a_wall_x_matches"] for row in tied),
            b_outer_face_x_matches=sum(row["b_outer_face_x_matches"] for row in tied),
            strict_matches=sum(
                row["normal_matches"] and row["a_wall_x_matches"] and row["depth_matches"]
                for row in tied
            ),
        )
        for field in (
            "a_wall_x_error_m",
            "b_outer_face_x_error_m",
            "a_surface_distance_m",
            "b_surface_distance_m",
        ):
            values = [row[field] for row in tied]
            summary[field.removesuffix("_m") + "_range_m"] = (
                [min(values), max(values)] if values else None
            )
        sample["deepest_set"] = summary
        if sample["contact"] != target["contact"]:
            failures.append(
                f"t={sample['fraction']} expected contact={target['contact']}, got {len(contacts)} rows"
            )
        elif target["contact"] and summary["strict_matches"] != len(tied):
            failures.append(
                f"t={sample['fraction']} {summary['strict_matches']}/{len(tied)} deepest rows "
                f"match depth, A-to-B normal and the receiver wall"
            )
    if failures:
        error = AssertionError(f"{case['id']}: " + ", ".join(failures))
        error.observed = observed
        raise error


def _check_samples(case, observed):
    tol = case["tolerances"]
    for sample, target in zip(observed["samples"], case["expected"]["samples"], strict=True):
        context = f"{case['id']} at {sample['fraction']}: {sample}"
        assert sample["contact"] == target["contact"], context
        if not target["contact"]:
            if "gap_m" in sample and "host_witness_gap_m" in target:
                np.testing.assert_allclose(
                    sample["gap_m"],
                    target["host_witness_gap_m"],
                    rtol=0,
                    atol=tol["length_m"],
                    err_msg=context,
                )
            continue
        gap = sample["gap_m"]
        if "gap_range_m" in target:
            lo, hi = target["gap_range_m"]
            assert lo - tol["length_m"] <= gap <= hi + tol["length_m"], context
        else:
            np.testing.assert_allclose(
                gap, target["gap_m"], rtol=0, atol=tol["length_m"], err_msg=context
            )
        np.testing.assert_allclose(
            sample["normal_a_to_b"],
            target["normal_a_to_b"],
            rtol=0,
            atol=tol["normal_component"],
            err_msg=context,
        )
        # The receiver has zero effective support radius in all landmark cases.
        if "a_point_world_m" in target:
            np.testing.assert_allclose(
                sample["a_support_world_m"],
                target["a_point_world_m"],
                rtol=0,
                atol=tol["length_m"],
                err_msg=context,
            )
        if "a_local_x_m" in target:
            local_x = sample["a_support_world_m"][0] - sample["origins_world_m"][0][0]
            np.testing.assert_allclose(
                local_x, target["a_local_x_m"], rtol=0, atol=tol["length_m"], err_msg=context
            )


def run_case(case, *, engine, device):
    inputs = case["inputs"]
    assets = [CATALOG["assets"][name] for name in inputs["assets"]]
    if engine == "phymeo":
        observed = _phymeo(case, inputs, assets, device)
    elif engine == "newton" and inputs["query"] == "sampled_poses":
        inspect_walls = case["id"] in ("RR-U-RIGHT", "RR-U-LEFT-HOLDOUT")
        observed = _newton_samples(inputs, assets, device, keep_all=inspect_walls)
        if inspect_walls:
            _check_u_walls(case, observed, assets)
            return observed
    else:
        raise ValueError(f"No {engine} adapter for {inputs['query']}")
    if inputs["query"] == "sampled_poses":
        _check_samples(case, observed)
    return observed

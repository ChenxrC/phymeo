"""Measure rigid-soft contacts on one selected engine. Static records and steps use separate cases."""

import warnings
from contextlib import nullcontext
from tempfile import TemporaryDirectory

import numpy as np


def _close(actual, expected, tolerance, field):
    np.testing.assert_allclose(
        actual, expected, atol=tolerance, rtol=0, equal_nan=False, err_msg=field
    )


def _newton_model(case, newton, wp, device):
    p = case["inputs"]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for xyz in p["vertices_m"]:
        builder.add_particle(
            wp.vec3(*xyz), wp.vec3(0.0, 0.0, 0.0), 1.0, radius=p["particle_radius_m"]
        )
    builder.add_triangle(0, 1, 2)
    # Newton's edge rows end with the two endpoints. -1 identifies the absent second face.
    for opposite, a, b in ((2, 0, 1), (0, 1, 2), (1, 2, 0)):
        builder.add_edge(opposite, -1, a, b)
    cfg = newton.ModelBuilder.ShapeConfig(
        has_particle_collision=True, gap=0.0, margin=p["shape_margin_m"]
    )
    family = case["family"]
    if family == "sphere_triangle":
        pose = wp.transform(wp.vec3(*p["center_m"]), wp.quat_identity())
        builder.add_shape_sphere(-1, radius=p["radius_m"], xform=pose, cfg=cfg)
    elif family == "curved_triangle":
        pose = wp.transform(wp.vec3(*p["center_m"]), wp.quat(*p["orientation_xyzw"]))
        dims = p["dimensions_m"]
        if p["kind"] == "ellipsoid":
            builder.add_shape_ellipsoid(-1, rx=dims[0], ry=dims[1], rz=dims[2], xform=pose, cfg=cfg)
        else:
            add = {"cylinder": builder.add_shape_cylinder, "cone": builder.add_shape_cone}[p["kind"]]
            add(-1, radius=dims[0], half_height=dims[1], xform=pose, cfg=cfg)
    elif family == "finite_plane":
        builder.add_shape_plane(width=p["width_m"], length=p["length_m"], cfg=cfg)
    elif family == "heightfield":
        heights = np.asarray(p["heights_m"], dtype=np.float32)
        hx, hy = p["half_extents_xy_m"]
        field = newton.Heightfield(heights, *heights.shape, hx=hx, hy=hy)
        builder.add_shape_heightfield(heightfield=field, cfg=cfg)
    elif family == "mesh_sdf":
        mesh = newton.Mesh.create_box(
            *p["cube_half_extents_m"],
            duplicate_vertices=False,
            compute_normals=False,
            compute_uvs=False,
            compute_inertia=False,
        )
        if p["sdf_resolution"] is not None:
            mesh.build_sdf(
                device=device, max_resolution=p["sdf_resolution"], margin=p["sdf_padding_m"]
            )
        builder.add_shape_mesh(-1, mesh=mesh, cfg=cfg)
    else:
        raise ValueError(f"No Newton measurement for {family}")
    return builder.finalize(device=device)


def _read_soft(pipeline, state, q, device):
    model = pipeline.model
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    assert str(contacts.soft_contact_indices.device) == device
    count = int(contacts.soft_contact_count.numpy()[0])
    assert 0 <= count <= contacts.soft_contact_max, f"soft contact overflow: {count}"
    assert int(contacts.rigid_contact_count.numpy()[0]) == 0
    columns = {
        name: getattr(contacts, "soft_contact_" + name).numpy()[:count]
        for name in (
            "indices",
            "barycentric",
            "particle",
            "shape",
            "body_pos",
            "body_vel",
            "normal",
        )
    }
    radii = model.particle_radius.numpy()
    margins = model.shape_margin.numpy()
    bodies = model.shape_body.numpy()
    records = []
    for i in range(count):
        corners = columns["indices"][i].tolist()
        weights = columns["barycentric"][i].astype(float)
        ids = [j for j in corners if j >= 0]
        size = len(ids)
        assert size and len(set(ids)) == size and max(ids) < len(q), corners
        assert corners == ids + [-1] * (3 - size), corners
        assert np.all(weights[:size] >= -1e-6) and np.all(weights[size:] == 0), weights
        _close(weights.sum(), 1.0, 3e-6, "material weight sum")
        particle = int(columns["particle"][i])
        assert particle == (ids[0] if size == 1 else -1)
        shape = int(columns["shape"][i])
        assert shape == 0 and bodies[shape] == -1, (
            "body-local coordinates must be world coordinates"
        )
        rigid_point = columns["body_pos"][i].astype(float)
        soft_point = weights[:size] @ q[ids]
        normal = columns["normal"][i].astype(float)
        velocity = columns["body_vel"][i].astype(float)
        _close(np.linalg.norm(normal), 1.0, 1e-4, "unit world normal")
        _close(velocity, [0, 0, 0], 1e-8, "stationary rigid surface velocity")
        gap = float(normal @ (soft_point - rigid_point) - max(radii[ids]) - margins[shape])
        records.append(
            dict(
                indices=corners,
                barycentric=weights.tolist(),
                particle=particle,
                rigid_shape=shape,
                soft_point_m=soft_point.tolist(),
                rigid_point_m=rigid_point.tolist(),
                rigid_surface_velocity_m_s=velocity.tolist(),
                normal=normal.tolist(),
                gap_m=gap,
            )
        )
    return dict(
        count=count,
        capacity=int(contacts.soft_contact_max),
        device=str(contacts.device),
        edge_candidates=len(pipeline.soft_edge_rigid_pairs),
        face_candidates=len(pipeline.soft_face_rigid_pairs),
        particle_candidates=pipeline.soft_contact_pair_count,
        records=sorted(records, key=lambda row: row["indices"]),
    )


def _check_soft_geometry(case, observation, expected_features):
    e, tol = case["expected"], case["tolerances"]
    features = []
    for row in observation["records"]:
        ids = [i for i in row["indices"] if i >= 0]
        features.append(sorted(ids))
        # On an adjacent face, Frank-Wolfe approaches a boundary optimum. The 1-D
        # edge optimizer has a separate, tighter acceptance window there.
        prefix = "newton_"
        if len(ids) == 3 and "newton_adjacent_face_gap_m" in tol:
            prefix = "newton_adjacent_face_"
        normal = row["normal"]
        _close(normal, e["normal"], tol[prefix + "normal"], "rigid to soft normal")
        _close(row["gap_m"], e["gap_m"], tol[prefix + "gap_m"], "surface gap")
        if case["family"] == "sphere_triangle":
            _close(row["soft_point_m"], e["soft_point_m"], tol[prefix + "point_m"], "sheet point")
            target = e["rigid_point_m"]
        else:
            # Every material point here projects into the flat top; tangential minima need not be unique.
            target = [*row["soft_point_m"][:2], e["top_z_m"]]
        _close(row["rigid_point_m"], target, tol[prefix + "point_m"], "rigid point")
    assert sorted(features) == sorted(expected_features), (features, expected_features)


def _newton(case, device):
    import newton
    import warp as wp

    p, e = case["inputs"], case["expected"]
    model = _newton_model(case, newton, wp, device)
    state = model.state()
    assert str(state.particle_q.device) == device
    q = state.particle_q.numpy().astype(float)
    _close(q, p["vertices_m"], 1e-7, "frozen particle coordinates")
    assert (model.particle_count, model.tri_count, model.edge_count, model.shape_count) == (
        3,
        1,
        3,
        1,
    )
    options = dict(
        broad_phase="nxn",
        soft_contact_gap=p["soft_gap_m"],
        rigid_contact_max=1,
        max_triangle_pairs=32,
    )
    default = _read_soft(newton.CollisionPipeline(model, **options), state, q, device)
    pairs = (default["particle_candidates"], default["edge_candidates"], default["face_candidates"])
    assert pairs == (3, 0, 0), pairs
    out = dict(time_semantics="static configuration", particle_default=default)
    if case["family"] == "sphere_triangle":
        center = model.shape_transform.numpy()[0, :3]
        radius = float(model.shape_scale.numpy()[0, 0])
        closest_node = float(np.min(np.linalg.norm(q - center, axis=1)) - radius)
        assert closest_node > e["node_gap_min_gt_m"], closest_node
        out["min_node_gap_from_state_m"] = closest_node
        assert default["count"] == e["particle_count"]
    else:
        _check_soft_geometry(case, default, e["particle_features"])

    with warnings.catch_warnings(record=True) as notes:
        warnings.simplefilter("always")
        try:
            full = newton.CollisionPipeline(
                model, enable_rigid_soft_full_surface_contact=True, **options
            )
        except ValueError as exc:
            if "error_contains" not in e or e["error_contains"] not in str(exc):
                raise
            out["full_surface_construction_error"] = dict(type=type(exc).__name__, message=str(exc))
            return out
    assert "error_contains" not in e, "full-surface accepted an unprovisioned rigid mesh"
    messages = [str(note.message) for note in notes if "full_surface_contact" in str(note.message)]
    if "warning_contains" in e:
        assert any(e["warning_contains"] in message for message in messages), messages
    else:
        assert not messages, messages
    observed = _read_soft(full, state, q, device)
    candidates = (observed["edge_candidates"], observed["face_candidates"])
    assert candidates == ((0, 0) if "warning_contains" in e else (3, 1)), candidates
    _check_soft_geometry(case, observed, e["full_features"])
    out.update(full_surface=observed, full_surface_warnings=messages)
    return out


def _sphere_scene(ph, inputs, *, swept):
    rigid = ph.CollisionKey(0, 0, 0)
    soft = ph.CollisionKey(0, 1, 0)
    model = ph.CollisionModel(
        [
            ph.CollisionShape(rigid, "sphere", (7,), inputs["radius_m"]),
            ph.CollisionShape(soft, "triangle", (0, 1, 2)),
        ]
    )
    sphere = (
        ph.CollisionMotion([inputs["start_m"]], [inputs["end_m"]])
        if swept
        else ph.CollisionMotion([inputs["center_m"]])
    )
    return model, {rigid: sphere, soft: ph.CollisionMotion(inputs["vertices_m"])}


def _phymeo_sphere(case, device, ph):
    p, e, tol = case["inputs"], case["expected"], case["tolerances"]
    swept = case["family"] == "sphere_sweep"
    model, paths = _sphere_scene(ph, p, swept=swept)
    backend = "cuda" if device == "cuda:0" else "host"
    cache = TemporaryDirectory(prefix="phymeo-rd-sweep-") if backend == "cuda" else nullcontext()
    with cache as root, ph.CollisionWorkspace(model, backend, cache_root=root) as workspace:
        policy = ph.CollisionPolicy(margin=p.get("phymeo_broad_margin_m", 0))
        answer = ph.detect(model, paths, policy, workspace)
        assert answer.status == "ok" and workspace.last_result is answer
        (pair,) = answer.pairs
        assert pair.candidate and pair.family == "RD"
        out = {"status": pair.status}
        if swept:
            assert pair.status == "hit"
            hit = pair.sweep
            lo, hi = hit.first_t
            assert lo <= e["first_t_fraction"] <= hi and hi - lo <= tol["first_t_width"]
            _close(
                [hit.start_gap_m, pair.witness.gap_m],
                [e["endpoint_gap_m"]] * 2,
                tol["phymeo_m"],
                "both endpoints separated",
            )
            witness = hit.witness
            out.update(
                time_semantics="normalized affine sweep",
                first_t=[lo, hi],
                endpoint_gaps_m=[hit.start_gap_m, pair.witness.gap_m],
            )
            if backend == "cuda":
                stats = workspace.stats()
                keys = (
                    "surface_completed_launches",
                    "surface_features",
                    "sweep_downloads",
                    "sweep_synchronizations",
                )
                out["tyx_counters"] = {key: stats[key] for key in keys}
                assert all(count > 0 for count in out["tyx_counters"].values()), out["tyx_counters"]
            rigid_target = soft_target = e["point_m"]
            gap = 0.0
        else:
            status = "overlap" if e["gap_m"] < 0 else "separated"
            assert pair.status == status and pair.sweep is None
            witness = pair.witness
            out["time_semantics"] = "static configuration"
            rigid_target, soft_target, gap = e["rigid_point_m"], e["soft_point_m"], e["gap_m"]
        weights = dict(witness.second_weights)
        assert set(weights) == {0, 1, 2}
        material = [weights[i] for i in range(3)]
        _close(material, e["face_weights"], tol["phymeo_unit"], "face material coordinates")
        _close(witness.first_point, rigid_target, tol["phymeo_m"], "rigid point")
        _close(witness.second_point, soft_target, tol["phymeo_m"], "soft point")
        _close(witness.normal, e["normal"], tol["phymeo_unit"], "rigid to soft normal")
        _close(witness.gap_m, gap, tol["phymeo_m"], "surface gap")
        out.update(
            rigid_point_m=witness.first_point,
            soft_point_m=witness.second_point,
            face_weights=material,
            normal=witness.normal,
            gap_m=witness.gap_m,
        )
        return out


def _phymeo_plate(case, ph):
    p, e, tol = case["inputs"], case["expected"], case["tolerances"]
    plate = ph.CollisionKey(0, 0, 0)
    shapes = [ph.CollisionShape(plate, "plate", (0, 1, 2, 3), p["rounding_m"])]
    paths = {plate: ph.CollisionMotion(p["polygon_m"])}
    for i, point in enumerate(p["probes_m"]):
        key = ph.CollisionKey(0, 1, i)
        shapes.append(ph.CollisionShape(key, "particle", (i,)))
        paths[key] = ph.CollisionMotion([point])
    model = ph.CollisionModel(shapes)
    policy = ph.CollisionPolicy(margin=p["phymeo_broad_margin_m"])
    with ph.CollisionWorkspace(model) as workspace:
        answer = ph.detect(model, paths, policy, workspace)
        assert answer.status == "ok" and workspace.last_result is answer
        rows = []
        for pair in answer.pairs:
            if pair.family != "RD":
                continue
            i = pair.second.shape
            w = pair.witness
            assert pair.candidate and pair.sweep is None
            assert pair.plate.end.feature == e["features"][i]
            assert pair.status == ("overlap" if e["gaps_m"][i] < 0 else "separated")
            _close(w.gap_m, e["gaps_m"][i], tol["phymeo_m"], "rounded plate gap")
            _close(w.normal, e["normals"][i], tol["phymeo_unit"], "rounded plate normal")
            q = np.asarray(p["probes_m"][i])
            _close(w.second_point, q, tol["phymeo_m"], "probe")
            target = q - e["gaps_m"][i] * np.asarray(e["normals"][i])
            _close(w.first_point, target, tol["phymeo_m"], "rounded plate surface")
            rows.append(
                dict(
                    probe=i,
                    feature=pair.plate.end.feature,
                    status=pair.status,
                    gap_m=w.gap_m,
                    normal=w.normal,
                    surface_point_m=w.first_point,
                )
            )
        assert sorted(row["probe"] for row in rows) == list(range(len(p["probes_m"])))
        return dict(time_semantics="static configuration", probes=rows)


def _phymeo_link(case, device, ph):
    from phymeo.constraints.cloth_material import ClothMaterial
    from phymeo.constraints.robot_contact import LinkCapsule
    from phymeo.dynamics.robot_contact import RobotContactSystem

    p, e, tol = case["inputs"], case["expected"], case["tolerances"]
    x = np.asarray(p["vertices_m"], dtype=float)
    masses = np.asarray(p["node_masses_kg"], dtype=float)
    gravity = np.asarray(p["gravity_m_s2"], dtype=float)
    robot = ph.RobotModel(
        parents=np.array([-1]),
        joint_types=("prismatic",),
        joint_position=np.asarray([p["link_origin_m"]], dtype=float),
        joint_orientation=np.array([[0.0, 0.0, 0.0, 1.0]]),
        joint_axis=np.array([[0.0, 0.0, 1.0]]),
        mass=np.array([p["link_mass_kg"]]),
        com=np.zeros((1, 3)),
        inertia_diagonal=np.ones((1, 3)),
        gravity=gravity,
    )
    start, end = np.asarray(p["link_axis_ends_m"], dtype=float)
    link = LinkCapsule("beam", "arm", 0, start, end, p["link_radius_m"])
    links = RobotContactSystem({"arm": robot}, capsules=(link,))
    cloth = ph.ParticleModel(1 / masses, gravity)
    model = ph.SceneModel(
        (
            ph.DomainBinding("cloth", "particles", cloth),
            ph.DomainBinding("arm", "robot", robot),
        )
    )
    constraint = ph.DistanceConstraints(np.array([[0, 1]]), np.array([2.0]), np.array([1.0]))
    material = ClothMaterial(distance=constraint)
    surface = ph.TrianglePrismSurface(
        "sheet", "cloth", (0, 1, 2), half_thickness_m=p["half_thickness_m"]
    )
    layer = ph.ClothLinkMaterial(
        width_m=p["layer_width_m"],
        stiffness_pa_per_m=p["stiffness_pa_per_m"],
        friction=p["friction"],
        damping_s_per_m=p["damping_s_per_m"],
    )
    state = ph.SceneState(
        {
            "cloth": ph.ParticleState(x, np.zeros_like(x)),
            "arm": ph.RobotState(np.zeros(1), np.zeros(1)),
        }
    )
    loads = ph.SceneInputs({"cloth": ph.StepInputs.zeros(len(x)), "arm": ph.RobotInputs.zeros(1)})
    settings = ph.StepSettings(substeps=1, position_iterations=1, velocity_iterations=1)
    cache = TemporaryDirectory(prefix="phymeo-rd-layer-") if device == "cuda:0" else nullcontext()
    with cache as root:
        operator = (
            ph.TyxNormalLayer("cuda", cache_root=root) if device == "cuda:0" else nullcontext()
        )
        with operator as native:
            program = ph.CompliantClothContactProgram(
                model,
                "cloth",
                material,
                (surface,),
                x,
                (),
                robot_contact=links,
                max_penetration_m=0.01,
                cloth_link_material=layer,
                cloth_link_operator=native,
            )
            with program.stepper(settings) as stepper:
                result = stepper.step(state, loads, p["dt_s"])
            assert result.report.accepted, result.report.reason
            native_counts = native.stats() if native is not None else {}
    if device == "cuda:0":
        assert native_counts["backend"] == "cuda"
        for name in ("completed_launches", "sites", "uploads", "downloads"):
            assert native_counts[name] > 0, native_counts
    diagnostics = result.report.substeps[0].diagnostics
    body = diagnostics["robot_contact"]
    assert diagnostics["force_sample"] == "substep-midpoint"
    (contact,) = body["cloth_links"]
    assert len(contact["sites"]) == e["site_count"]
    effort = float(body["cloth_link_efforts"]["arm"][0])
    arm = result.state.domains["arm"]
    cloth = result.state.domains["cloth"]
    momentum = float(p["link_mass_kg"] * arm.v[0] + masses @ cloth.velocity[:, 2])
    _close(contact["gap_m"], e["sample_gap_m"], tol["position_m"], "midpoint layer gap")
    _close(contact["normal_force_n"], e["normal_force_n"], tol["force_n"], "normal force")
    _close(effort, e["joint_effort_n"], tol["force_n"], "prismatic effort")
    _close(arm.q[0], e["end_joint_displacement_m"], tol["position_m"], "joint displacement")
    _close(arm.v[0], e["end_joint_velocity_m_s"], tol["velocity_m_s"], "joint velocity")
    _close(
        cloth.velocity[:, 2], e["end_node_velocity_z_m_s"], tol["velocity_m_s"], "nodal velocities"
    )
    _close(momentum, e["net_momentum_z_kg_m_s"], tol["momentum_kg_m_s"], "net momentum")
    return dict(
        time_semantics="one midpoint substep",
        force_sample=diagnostics["force_sample"],
        dt_s=p["dt_s"],
        site_count=len(contact["sites"]),
        sample_gap_m=contact["gap_m"],
        normal_force_n=contact["normal_force_n"],
        joint_effort_n=effort,
        end_joint_displacement_m=float(arm.q[0]),
        end_joint_velocity_m_s=float(arm.v[0]),
        end_node_velocity_m_s=cloth.velocity.tolist(),
        net_momentum_z_kg_m_s=momentum,
        tyx_counters=native_counts,
    )


def run_case(case, *, engine, device):
    """Return real observations for the catalog's selected backend/device contract."""
    if engine not in ("newton", "phymeo") or device not in case[engine]["devices"]:
        raise ValueError(f"{case['id']} does not run on {engine} {device}")
    if engine == "newton":
        return _newton(case, device)
    import phymeo as ph

    if case["family"] == "rounded_plate":
        return _phymeo_plate(case, ph)
    if case["family"] == "link_layer":
        return _phymeo_link(case, device, ph)
    return _phymeo_sphere(case, device, ph)

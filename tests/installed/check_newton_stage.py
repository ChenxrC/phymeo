"""Run from outside the source checkout against a separately installed stage wheel."""

import argparse
import hashlib
import json
import sys
import sysconfig
from contextlib import ExitStack
from importlib.metadata import distribution
from pathlib import Path
from urllib.parse import urlsplit
from zipfile import ZipFile


def verified_wheel(installed, expected):
    """Tie the independently installed distribution to a concrete local wheel archive."""
    assert expected.is_absolute() and expected.suffix.lower() == ".whl" and expected.is_file()
    wheel = expected.resolve()
    with wheel.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    raw = installed.read_text("direct_url.json")
    assert raw is not None, "install the supplied wheel file directly to record its archive SHA256"
    direct = json.loads(raw)
    origin = urlsplit(direct["url"])
    assert origin.scheme == "file" and origin.path.lower().endswith(".whl")
    archive = direct["archive_info"]
    recorded = archive.get("hashes", {}).get("sha256")
    if recorded is None:
        algorithm, _, recorded = archive.get("hash", "").partition("=")
        assert algorithm == "sha256"
    assert recorded.lower() == actual, "installed archive SHA256 differs from --expected-wheel"
    return dict(file=str(wheel), sha256=actual, installed_from=direct["url"])


def mixed_world_step(ph, np, mode, root):
    """One true scene step with a free sphere, a loaded finite cloth face and a clear face."""
    from phymeo.constraints.cloth_material import ClothMaterial

    xyz = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 2.0, 0.0]])
    cloth = ph.ParticleModel(np.ones(3), np.zeros(3))
    arm = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.array([[0.0, 0.0, 0.25]]),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([0.5]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.zeros(3),
    )
    model = ph.SceneModel(
        (ph.DomainBinding("cloth", "particles", cloth), ph.DomainBinding("arm", "robot", arm))
    )
    rod = ph.LinkCapsule("beam", "arm", 0, (-0.25, 0, 0), (0.25, 0, 0), 0.125)
    face = ph.TrianglePrismSurface("sheet", "cloth", (0, 1, 2), half_thickness_m=1 / 32)
    spring = ph.DistanceConstraints(np.array([[0, 1]]), np.array([2.0]), np.array([1.0]))
    program = ph.CompliantClothContactProgram(
        model,
        "cloth",
        ClothMaterial(distance=spring),
        (face,),
        xyz,
        (),
        robot_contact=ph.RobotContactSystem({"arm": arm}, capsules=(rod,)),
        max_penetration_m=0.01,
        cloth_link_material=ph.ClothLinkMaterial(0.125, 64 / 3, 0.0, 0.0),
    )

    def material(q):
        state = ph.SceneState(
            {
                "cloth": ph.ParticleState(xyz, np.zeros_like(xyz)),
                "arm": ph.RobotState(np.array([q]), np.zeros(1)),
            }
        )
        return ph.ClothLinkWorld(program, state)

    ball = ph.FreeSphere(2.0, 0.125, position=(0, 0, 2), gravity=(0, 0, -8))
    worlds = [(13, material(0.0)), (61, ball), (29, material(0.5))]
    loads = {
        key: ph.SceneInputs({"cloth": ph.StepInputs.zeros(3), "arm": ph.RobotInputs.zeros(1)})
        for key in (13, 29)
    }
    engine = "cuda" if mode == "cuda" else "interpret"
    h = 1 / 256
    with ph.TyxBatchedScene(worlds, engine, cache_root=root) as scene:
        if engine == "cuda":
            import tyx

            before = tyx.runtime_stats().cuda_kernel_launches
        result = scene.step(h, loads)
        points = scene.snapshot()
        counts = scene.stats()
        launches = 0 if engine == "interpret" else tyx.runtime_stats().cuda_kernel_launches - before
    assert tuple(result.accepted) == (13, 61, 29) and all(result.accepted.values())
    assert tuple(result.contact_results) == (13, 29) and all(p.tick == 1 for p in points.values())
    loaded = result.contact_results[13].audit
    clear = result.contact_results[29].audit.sample
    assert points[13].last_audit == loaded and len(loaded.sample.sites) == 2
    np.testing.assert_allclose(
        [loaded.sample.gap_m, loaded.sample.normal_force_n, loaded.sample.joint_effort_n],
        [3 / 32, 2.0, 2.0],
        rtol=0,
        atol=2e-7,
    )
    np.testing.assert_allclose(
        loaded.sample.node_forces_n, np.tile([0, 0, -2 / 3], (3, 1)), atol=2e-7
    )
    np.testing.assert_allclose(np.asarray(points[13].velocity)[:, 2], [-1 / 384] * 3, atol=2e-9)
    np.testing.assert_allclose([points[13].q, points[13].qd], [1 / 32768, 1 / 64], atol=2e-9)
    assert (
        clear.sites == ()
        and clear.normal_force_n == 0
        and points[29].q == 0.5
        and points[29].qd == 0
    )
    assert points[61].position == (0.0, 0.0, 2 - 8 * h**2) and points[61].velocity == (
        0.0,
        0.0,
        -8 * h,
    )
    assert (
        counts["resident"]["submitted_kernels"] == 1
        and counts["material"]["completed_kernels"] == 6
    )
    if engine == "cuda":
        assert launches == 7 and counts["material"]["downloaded_bytes"] > 0
    return dict(
        physical_backend="TYX CUDA" if engine == "cuda" else "TYX HOST interpret",
        geometry_and_audit="HOST",
        keys=list(result.accepted),
        h_s=h,
        ticks=[p.tick for p in points.values()],
        material_normal_n=[loaded.sample.normal_force_n, clear.normal_force_n],
        loaded_node_velocity_m_s=points[13].velocity,
        loaded_link_velocity_m_s=points[13].qd,
        sphere_velocity_m_s=points[61].velocity,
        actual_cuda_launches=launches,
        resident_submissions=counts["resident"]["submitted_kernels"],
        material_submissions=counts["material"]["completed_kernels"],
    )


def robot_sample(ph, np):
    model = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.zeros((1, 3)),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, 1.0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.ones((1, 3)),
        np.array([0.0, 0.0, -4.0]),
    )
    site = ph.RobotSite(0)
    targets = [
        ph.SiteIKTarget(site, position_world_m=[0, 0, 0.375]),
        ph.SiteIKTarget(ph.RobotSite(0, (0.25, 0, 0.125)), position_world_m=[0.25, 0, 0.5]),
    ]
    ik = ph.solve_multi_site_ik(model, np.array([0.0]), targets, tolerance_m=1e-8)
    assert ik.converged and abs(ik.configuration[0] - 0.375) < 1e-8
    state = ph.RobotState(np.array([0.125]), np.zeros(1))
    closure = ph.RobotPointClosure(site, ph.RobotSite(None, (0, 0, 0.125)), [[0.0, 0.0, 1.0]])
    sample = ph.step_robot_closures(model, state, ph.RobotInputs.zeros(1), [closure], 1 / 64)
    assert sample.report.accepted
    imu = ph.measure_robot_imu(model, state, sample.reaction.acceleration, site, time_s=0.0)
    np.testing.assert_allclose(imu.specific_force_m_s2, [0, 0, 4], atol=1e-10)
    np.testing.assert_allclose(sample.reaction.force_first_world, [[0, 0, 8]], atol=1e-10)
    return dict(
        backend="HOST NumPy",
        ik_position_m=ik.configuration.tolist(),
        closure_force_n=sample.reaction.force_first_world.tolist(),
        imu_specific_force_m_s2=imu.specific_force_m_s2.tolist(),
    )


def collision_and_force(ph, np, mode, root):
    dtype = np.float64 if mode == "host" else np.float32
    keys = [ph.CollisionKey(8, k, 0) for k in (1, 2)]
    shapes = [
        ph.CollisionShape(key, "ellipsoid", (5,), curved=ph.CurvedSolid("ellipsoid", dims))
        for key, dims in zip(keys, ((0.25, 0.125, 0.125), (0.125, 0.25, 0.125)), strict=True)
    ]
    model = ph.CollisionModel(shapes)
    pm = ph.ParticleModel(np.empty(0, dtype=dtype), np.zeros(3, dtype=dtype))
    rm = ph.RigidBodyModel(
        np.array([2.0, 1.0], dtype=dtype),
        np.full((2, 3), 0.25, dtype=dtype),
        np.zeros(3, dtype=dtype),
    )
    p = ph.ParticleState(np.empty((0, 3), dtype=dtype), np.empty((0, 3), dtype=dtype))
    r = ph.RigidBodyState(
        np.array([[0, 0, 0], [0.40625, 0, 0]], dtype=dtype),
        np.array([[0, 0, 0, 1.0]] * 2, dtype=dtype),
        np.array([[0, 0.25, 0], [0, 0, 0]], dtype=dtype),
        np.zeros((2, 3), dtype=dtype),
    )
    with ExitStack() as stack:
        box = stack.enter_context(ph.CollisionWorkspace(model, mode, cache_root=root))
        law = (
            None
            if mode == "host"
            else stack.enter_context(ph.TyxNormalLayer(mode, cache_root=root))
        )
        solver = (
            ph.RigidSemiImplicitSolver()
            if mode == "host"
            else stack.enter_context(ph.TyxRigidSemiImplicitSolver(mode=mode, cache_root=root))
        )
        system = ph.CollisionLoadSystem(
            model,
            {k: ph.RigidAttachment(i, ((0, 0, 0),)) for i, k in enumerate(keys)},
            {tuple(keys): ph.NormalLayer(1 / 16, 64.0, friction=0.25)},
            layer_operator=law,
        )
        output = system.step(
            ph.CoupledState(p, r),
            ph.CoupledInputs.zeros(0, 2, dtype=dtype),
            (pm, rm),
            (None, solver),
            1 / 256,
            box,
        )
        reading = ph.measure_contact_wrench(
            system, ph.ContactSelector(8, 2), reference_world_m=(0, 0, 0)
        )
        np.testing.assert_allclose(reading.force_n, [2.0, 0.5, 0], atol=2e-5, rtol=0)
        np.testing.assert_allclose(reading.torque_nm, [0, 0, 0.1328125], atol=2e-5, rtol=0)
        assert output.path.pairs[0].status == "separated" and len(reading.contributing_pairs) == 1
        return dict(
            broad_phase=mode,
            narrow_phase="HOST",
            material_and_rigid_step="HOST NumPy" if mode == "host" else "TYX CUDA",
            force_n=reading.force_n.tolist(),
            torque_nm=reading.torque_nm.tolist(),
            rigid_velocity_m_s=output.state.rigid.velocity.tolist(),
            path=output.path.pairs[0].reason,
            completed_broad=box.stats()["completed_launches"],
            force_sites=None if law is None else law.stats()["sites"],
        )


def floating_impact(ph, np, mode, cache_root):
    child = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.array([[1.0, 0, 0]]),
        np.array([[0.0, 0, 0, 1.0]]),
        np.array([[1.0, 0, 0]]),
        np.ones(1),
        np.zeros((1, 3)),
        np.full((1, 3), 0.5),
        np.zeros(3),
    )
    root = ph.RigidBodyModel(np.array([0.5]), np.full((1, 3), 0.5), np.zeros(3))
    arm = ph.FloatingRobotModel(root, np.zeros(3), child)
    q = np.array([0.0, 0, 0, 0, 0, 0, 1, 0])
    body = ph.RigidBodyModel(np.ones(1), np.full((1, 3), 0.5), np.zeros(3))
    s = ph.RigidBodyState(
        np.array([[1.0, 0.125, 0.0]]),
        np.array([[0.0, 0, 0, 1.0]]),
        np.array([[0.0, -1.0, 0.0]]),
        np.zeros((1, 3)),
    )
    rk, bk = [ph.CollisionKey(1, k, 0) for k in (3, 8)]
    shapes = ph.CollisionModel([ph.CollisionShape(k, "sphere", (0,), 0.125) for k in (rk, bk)])
    with ph.CollisionWorkspace(shapes, mode, cache_root=cache_root) as work:
        (pair,) = ph.detect(
            shapes,
            {rk: ph.CollisionMotion([[1.0, -0.125, 0.0]]), bk: ph.CollisionMotion(s.position)},
            ph.CollisionPolicy(),
            work,
        ).pairs
        launches = work.stats()["completed_launches"]
        if mode == "cuda":
            assert launches > 0
    assert pair.status == "touching"
    ports = {
        "arm": ph.FloatingRobotPointPort(
            arm,
            ph.FloatingRobotState(q, np.zeros(7)),
            np.array([1]),
            np.zeros((1, 3)),
            domain="arm",
        ),
        "ball": ph.RigidPointPort(
            body, s, np.array([0]), np.array([[0.0, -0.125, 0.0]]), domain="ball"
        ),
    }
    impulse = ph.solve_witness_impulses(
        ports,
        [(pair, {rk: ("arm", 0), bk: ("ball", 0)})],
        [0.0],
        [0.0],
        local_vertices={rk: ((0.0, -0.125, 0.0),)},
    )
    np.testing.assert_allclose(impulse.states["arm"].v, [0, -0.125, 0, 0, 0, -0.25, 0], atol=1e-10)
    assert abs(impulse.normal_impulses[0] - 0.625) < 1e-10
    return dict(
        collision_broad_backend=mode,
        collision_completed_launches=launches,
        impulse_backend="HOST NumPy",
        impulse_ns=float(impulse.normal_impulses[0]),
        kinetic_change_j=impulse.energy_change,
    )


def floating_support_window(ph, np):
    """Advance an installed free-root sphere under a load, then lift it off the plane."""
    h = 1 / 64
    root = ph.RigidBodyModel(np.array([2.0]), np.ones((1, 3)), np.array([0.0, 0.0, -8.0]))
    model = ph.FloatingRobotModel(root, np.zeros(3))
    state = ph.FloatingRobotState(np.array([0.0, 0.0, 0.125, 0, 0, 0, 1]), np.zeros(6))
    fixed = ph.RigidBodyModel(np.zeros(1), np.zeros((1, 3)), np.zeros(3))
    base = ph.RigidBodyState(
        np.zeros((1, 3)), np.array([[0.0, 0.0, 0.0, 1.0]]), np.zeros((1, 3)), np.zeros((1, 3))
    )
    floor, ball = (ph.CollisionKey(1, k, 0) for k in (1, 2))
    collision = ph.CollisionModel(
        (
            ph.CollisionShape(floor, "plane", (6,), normal=(0, 0, 1)),
            ph.CollisionShape(ball, "sphere", (9,), radius=0.125),
        )
    )
    attachments = {
        "tree": (np.array([0]), np.array([[0.0, 0.0, -0.125]])),
        "floor": (np.array([0]), np.zeros((1, 3))),
    }
    bindings = {floor: ("floor", 0), ball: ("tree", 0)}
    reactions = []
    with ph.CollisionWorkspace(collision) as workspace:
        for step in range(6):
            paths = {
                floor: ph.CollisionMotion(base.position),
                ball: ph.CollisionMotion([state.q[:3]]),
            }
            (pair,) = ph.detect(collision, paths, ph.CollisionPolicy(), workspace).pairs
            rows = [(pair, bindings)] if pair.status == "touching" else []
            assert pair.status in ("touching", "separated") and bool(rows) == (step < 5)
            force = np.array([[0.0, 0.0, 8.0 if step >= 4 else 0.0]])
            inputs = ph.FloatingRobotInputs(np.zeros(0), force, np.zeros((1, 3)))
            result = ph.step_floating_contacts(
                {"tree": (model, state, inputs)},
                attachments,
                rows,
                np.zeros(len(rows)),
                np.zeros(len(rows)),
                h,
                fixed_bodies={"floor": (fixed, base)},
                local_normals={floor: (0, 0, 1)} if rows else None,
            )
            assert isinstance(result, ph.FloatingContactStep) and result.report.accepted
            reactions.append(float(np.sum(result.impulses.normal_impulses) / h))
            state = result.states["tree"]
    np.testing.assert_allclose(reactions, [4.0, 4.0, 4.0, 4.0, 0.0, 0.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        [state.v[2], state.q[2] - 0.125], [16 * h, 24 * h**2], rtol=0, atol=1e-12
    )
    return dict(
        backend="HOST NumPy",
        reaction_n=reactions,
        liftoff_m=float(state.q[2] - 0.125),
        final_velocity_m_s=float(state.v[2]),
    )


def field_sample(ph, np, mode, root):
    values = np.broadcast_to((np.arange(3) * 0.25 - 0.25)[:, None, None], (3, 3, 3)).copy()
    grid = ph.SignedDistanceGrid(values, origin_m=(0, 0, 0), spacing_m=(0.25, 0.25, 0.25))
    points = np.array([[2.3125, -0.875, 0.625], [5.0, 4.0, 3.0]])
    pose = dict(translation_m=(2, -1, 0.5), orientation_xyzw=(0.5, 0.5, 0.5, 0.5))
    stats = None
    if mode == "cuda":
        with ph.TyxSdfSampler(grid, "cuda", capacity=2, cache_root=root) as sampler:
            first, second = sampler.sample_world(points, **pose)
            stats = sampler.stats()
    else:
        first, second = grid.sample_world(points, **pose)
    assert first.status == "valid" and second.status == "outside_grid"
    assert abs(first.distance_m - 0.0625) < 2e-6
    np.testing.assert_allclose(first.gradient, [1, 0, 0], atol=2e-6)
    return dict(
        distance_m=first.distance_m, gradient=first.gradient, outside=second.status, backend=stats
    )


def many_worlds(ph, mode, root):
    worlds, paths = {}, {}
    for world in range(65):
        key = [ph.CollisionKey(world, k, 0) for k in range(4)]
        shapes = [ph.CollisionShape(k, "sphere", (17,), 0.125) for k in key]
        worlds[world] = ph.CollisionModel(shapes)
        paths[world] = {
            k: ph.CollisionMotion([[x, 0, 0]]) for k, x in zip(key, (0, 0.1875, 2, 4), strict=True)
        }
    with ph.PagedWorldCollisionWorkspace(
        worlds.items(), mode, cache_root=root, result_format="contacts"
    ) as batch:
        obs = batch.detect(paths, {w: ph.CollisionPolicy() for w in worlds})
        assert len(obs) == 65 and len(batch.layout) >= 2
        hits = []
        for w, rows in obs.items():
            assert isinstance(rows, ph.CompactContactBuffer) and rows.required_contacts == 1
            assert rows.required_candidates == 1 and not rows.diagnostics
            (pair,) = rows.contacts
            assert pair.status == "overlap"
            assert (
                pair.first.world == pair.second.world == w
                and abs(pair.witness.gap_m + 1 / 16) < 1e-6
            )
            hits.append(w)
        stats = batch.stats()
        assert stats["backend_counts"]["compact_skipped_slots"] == 65 * 5
        if mode == "cuda":
            assert stats["completed_launches"] > 0
        return dict(
            worlds=len(obs),
            shape_count=260,
            pages=len(batch.layout),
            local_hits=len(hits),
            broad_phase=mode,
            narrow_phase="HOST",
            completed_broad_launches=stats["completed_launches"],
            skipped_pair_rows=65 * 5,
        )


def deformable_force_step(ph, np, mode, cache):
    dtype = np.float64 if mode == "host" else np.float32
    ids = ((11, 23, 37), (53, 67, 79))
    xyz = np.array(
        [
            [-1, -1, 0],
            [1, -1, 0],
            [0, 1, 0],
            [0, 0, 3 / 64],
            [-0.125, 0.125, 0.75],
            [0.125, 0.125, 0.75],
        ],
        dtype=dtype,
    )
    shapes = [
        ph.CollisionShape(ph.CollisionKey(8, body, 0), "triangle", nodes)
        for body, nodes in zip((7, 13), ids, strict=True)
    ]
    model = ph.CollisionModel(shapes)
    bindings = {
        shape.key: ph.ParticleAttachment(tuple(range(3 * i, 3 * i + 3)))
        for i, shape in enumerate(shapes)
    }
    particles = ph.ParticleModel(np.ones(6, dtype=dtype), np.zeros(3, dtype=dtype))
    state = ph.ParticleState(xyz, np.zeros_like(xyz))
    with ExitStack() as scope:
        work = scope.enter_context(ph.CollisionWorkspace(model, mode, cache_root=cache))
        law = (
            None
            if mode == "host"
            else scope.enter_context(ph.TyxNormalLayer(mode, cache_root=cache))
        )
        solver = (
            ph.SemiImplicitSolver()
            if mode == "host"
            else scope.enter_context(ph.TyxSemiImplicitSolver(mode, cache_root=cache))
        )
        system = ph.DeformableFeatureLoadSystem(
            model,
            bindings,
            ph.NormalLayer(1 / 16, 128.0),
            search_distance_m=3 / 32,
            layer_operator=law,
        )
        initial = system.sample_particles(state, work)
        np.testing.assert_allclose(
            initial.nodal_force_n[:, 2], [-0.5, -0.5, -1.0, 2.0, 0.0, 0.0], rtol=0, atol=2e-6
        )
        step = solver.step(particles, state, initial.step_inputs(), 1 / 256)
        assert step.report.accepted
        after = system.sample_particles(step.state, work)
        (first,) = initial.contacts
        (last,) = after.contacts
        assert first.feature.kind == "VT" and last.feature.distance_m > first.feature.distance_m
        assert last.force_n < first.force_n
        return dict(
            first_force_n=first.force_n,
            after_force_n=last.force_n,
            final_gap_m=last.feature.distance_m,
            receiver_velocity_m_s=step.state.velocity[3].tolist(),
            physics_backend=step.report.backend,
        )


def imported_mesh_contacts(ph, np, assets, mode, cache):
    probes = ((1.375, 1.375, 0.0), (0.5, 1.5, 0.0))
    pivot = np.array([0.5, 0.25, -0.125])
    offset = np.array([0.125, -0.25, 0.0625])
    scale = np.array([0.125, 0.25, 0.25])
    rz = np.array([[0.0, -1, 0], [1.0, 0, 0], [0, 0, 1.0]])
    ry = np.array([[0.8, 0, 0.6], [0, 1.0, 0], [-0.6, 0, 0.8]])
    output = {}
    for suffix, loader in (("urdf", ph.load_urdf), ("xml", ph.load_mjcf)):
        asset = loader(assets / ("mesh_arm." + suffix))
        _, bracket = asset.collisions
        assert bracket.shape.kind == "mesh" and len(bracket.shape.triangles) == 20
        keys = [ph.CollisionKey(0, 77, i) for i in (0, 1)]
        spheres = [ph.CollisionShape(k, "sphere", (0,), 1 / 64) for k in keys]
        model = ph.CollisionModel(
            (*asset.collision_model.shapes, *spheres), asset.collision_model.excluded
        )
        paths = asset.collision_motion(np.array([np.arctan2(0.6, 0.8)]))
        for key, local in zip(keys, probes, strict=True):
            center = pivot + (offset + (np.asarray(local) * scale) @ rz.T) @ ry.T
            paths[key] = ph.CollisionMotion([center])
        with ph.CollisionWorkspace(model, mode, cache_root=cache) as work:
            pairs = ph.detect(model, paths, ph.CollisionPolicy(margin=0.5), work).pairs
            poses = [
                next(
                    p for p in pairs if {p.first, p.second} == {bracket.shape.key, k}
                ).mesh_round.end
                for k in keys
            ]
            assert [p.relation for p in poses] == ["separated", "mesh_contains_round"]
            np.testing.assert_allclose(
                [p.boundary_gap_m for p in poses], [0.03125, 0.046875], atol=2e-12
            )
            if mode == "cuda":
                assert work.stats()["completed_launches"] > 0
        output[suffix] = [dict(relation=p.relation, boundary_gap_m=p.boundary_gap_m) for p in poses]
    return output


def reference_handoff(cli, target):
    """Run the external CLI against the installed package and verify its physical frames."""
    import gzip
    import runpy

    trace_path = target / "articulated-reference-handoff.json.gz"
    old = sys.argv
    try:
        sys.argv = [
            str(cli),
            "--case",
            "handoff",
            "--robot",
            "articulated",
            "--fabric",
            "reference",
            "--output",
            str(trace_path),
        ]
        try:
            runpy.run_path(str(cli), run_name="__main__")
        except SystemExit as result:
            assert result.code == 0, result.code
    finally:
        sys.argv = old
    with gzip.open(trace_path, "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    assert record["schema"] == "phymeo-compliant-case-v1"
    assert record["case"] == "articulated-handoff" and record["robot_layout"] == "articulated"
    assert record["backend"] == "numpy" and record["initial_tilt_deg"] == 0
    assert record["dt_s"] == 0.01 and record["substeps"] == 128
    fabric = record["fabric"]
    assert fabric["name"] == "reference" and fabric["grid"] == [2, 3]
    assert fabric["mass_kg"] == 0.048 and fabric["half_thickness_m"] == 0.0015
    assert (
        fabric["stretch_compliance_m_per_n"] == 0.001 and fabric["bend_compliance_per_nm"] == 20.0
    )
    frames, events = record["frames"], record["events"]
    expected = [
        "grasp_A",
        "lift_A",
        "share",
        "release_A",
        "carry_B",
        "lower_B",
        "open_B",
        "settle",
        "placed",
    ]
    assert record["completed"] and [e["phase"] for e in events] == expected
    steps = (len(frames) - 1) * record["substeps"]
    assert steps == len(record["robot_samples"]) and len(record["samples"]) == steps * len(
        record["patches"]
    )
    assert len(frames) >= 10 and abs(frames[-1]["t"] - record["time_s"]) < 1e-9
    at = {}
    for event in events:
        if event["phase"] in {"release_A", "carry_B", "lower_B", "placed"}:
            frame = min(frames, key=lambda src: abs(src["t"] - event["t"]))
            assert abs(frame["t"] - event["t"]) < 1e-9
            at[event["phase"]] = frame["measurements"]
    share, first, last, placed = (at[k] for k in ("release_A", "carry_B", "lower_B", "placed"))
    weight = record["metrics"]["weight_n"]
    assert abs(weight - fabric["mass_kg"] * 9.81) < 1e-12
    assert min(share["A_both_normal_n"], share["B_both_normal_n"]) > 0.20
    assert (
        min(share["A_up_n"], share["B_up_n"]) > 0.08 * weight
        and share["table_up_n"] < 0.02 * weight
    )
    assert (
        first["A_normal_n"] < 0.01
        and first["B_up_n"] > 0.3 * weight
        and first["table_up_n"] < 0.02 * weight
    )
    transport = last["cloth_center_x_m"] - first["cloth_center_x_m"]
    assert transport > 0.035 and last["B_up_n"] > 0.2 * weight and last["A_normal_n"] < 0.01
    assert all(
        f["measurements"]["table_up_n"] > 0.35 * weight
        and f["measurements"]["A_normal_n"] < 0.01
        and f["measurements"]["B_normal_n"] < 0.01
        for f in frames[-10:]
    )
    assert placed["speed_m_s"] < 0.05 and 0.9 * weight < placed["table_up_n"] < 1.1 * weight
    return dict(
        case=record["case"],
        backend=record["backend"],
        robot_layout=record["robot_layout"],
        fabric=fabric,
        dt_s=record["dt_s"],
        substeps_per_frame=record["substeps"],
        physical_frames=len(frames),
        substeps=steps,
        time_s=record["time_s"],
        phases=[e["phase"] for e in events],
        b_independent_transport_m=transport,
        share_both_jaw_normal_n=[share["A_both_normal_n"], share["B_both_normal_n"]],
        share_up_n=[share["A_up_n"], share["B_up_n"]],
        share_table_n=share["table_up_n"],
        b_carry_up_n=last["B_up_n"],
        last_jaw_n=[placed["A_normal_n"], placed["B_normal_n"]],
        final_table_n=placed["table_up_n"],
        cloth_weight_n=weight,
        final_node_speed_m_s=placed["speed_m_s"],
        trace=str(trace_path),
        external_cli=str(cli.resolve()),
        external_cli_sha256=hashlib.sha256(cli.read_bytes()).hexdigest(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wheel-site", type=Path, required=True)
    p.add_argument("--expected-wheel", type=Path, required=True)
    p.add_argument("--checkout", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["host", "cuda", "usd", "handoff"], required=True)
    p.add_argument("--usd-fixture", type=Path)
    p.add_argument("--mesh-assets", type=Path)
    p.add_argument("--case-cli", type=Path)
    args = p.parse_args()
    if not __debug__:
        p.error("run with -I and assertions enabled")
    site, checkout = args.wheel_site.resolve(), args.checkout.resolve()
    assert sys.flags.isolated and all(
        not path.resolve().is_relative_to(checkout) for path in (site, Path(__file__), Path.cwd())
    )
    sys.path.insert(0, str(site))
    import numpy as np

    import phymeo as ph

    installed = distribution("phymeo")
    wheel = verified_wheel(installed, args.expected_wheel)
    origin = Path(ph.__file__).resolve()
    assert origin.is_relative_to(site) and Path(installed.locate_file("")).resolve() == site
    # The optional USD interpreter lives in a worktree-local venv. Its installed
    # dependencies are distinct from a source checkout on the import path.
    dependency_dirs = {
        Path(sysconfig.get_path(name)).resolve()
        for name in ("stdlib", "platstdlib", "purelib", "platlib")
    }
    local_env_paths = [
        Path(path).resolve()
        for path in sys.path
        if path and Path(path).resolve().is_relative_to(checkout)
    ]
    for path in local_env_paths:
        assert (
            Path(sys.prefix).resolve().is_relative_to(checkout)
            and Path(sys.prefix).resolve() != checkout
        )
        assert path.is_relative_to(Path(sys.prefix).resolve()) and (
            path == Path(sys.prefix).resolve()
            or any(path.is_relative_to(folder) for folder in dependency_dirs)
        ), path
        assert path != checkout / "src" and path != checkout, path
    result = dict(
        package_version=installed.version,
        wheel=wheel,
        mode=args.mode,
        package=str(origin),
        cwd=str(Path.cwd()),
        interpreter_dependency_paths_under_checkout=[str(path) for path in local_env_paths],
    )
    assert not any(n == "pxr" or n.startswith("pxr.") for n in sys.modules)
    if args.mode == "handoff":
        assert args.case_cli and not args.case_cli.resolve().is_relative_to(checkout)
        result["reference_handoff"] = reference_handoff(args.case_cli, args.out.parent)
    elif args.mode == "usd":
        assert args.usd_fixture and not args.usd_fixture.resolve().is_relative_to(checkout)
        asset = ph.load_usd(args.usd_fixture)
        sdk = distribution("usd-core")
        assert sdk.version == "26.8"
        loaded_sdk = {}
        for name in ("Usd", "UsdGeom", "UsdPhysics"):
            module = sys.modules["pxr." + name]
            path = Path(module.__file__).resolve()
            assert path == Path(sdk.locate_file(f"pxr/{name}/__init__.py")).resolve()
            loaded_sdk[name] = str(path)
        usd_version = tuple(sys.modules["pxr.Usd"].GetVersion())
        assert usd_version[-2:] == (26, 8)
        assert asset.model.dof_count > 0 and asset.collisions
        pose = asset.collision_motion(np.zeros(asset.model.dof_count))
        assert set(pose) == {g.shape.key for g in asset.collisions}
        (tool,) = asset.collisions
        key = ph.CollisionKey(0, 93, 0)
        cylinder = ph.CollisionShape(
            key, "cylinder", (71,), curved=ph.CurvedSolid("cylinder", (0.25, 0.25))
        )
        model = ph.CollisionModel(
            (*asset.collision_model.shapes, cylinder), asset.collision_model.excluded
        )
        fixed = {key: ph.CollisionMotion([[17 / 64, 1 / 16, 5 / 32]])}
        with ph.CollisionWorkspace(model) as work:
            (clear,) = ph.detect(model, pose | fixed, ph.CollisionPolicy(), work).pairs
            (contact,) = ph.detect(
                model,
                asset.collision_motion(np.array([np.pi / 2, 1 / 8])) | fixed,
                ph.CollisionPolicy(),
                work,
            ).pairs
        assert clear.status == "separated" and contact.status == "overlap"
        assert (contact.first, contact.second) == (tool.shape.key, key) and abs(
            contact.witness.gap_m + 1 / 64
        ) < 2e-6
        # Put a separate probe at the middle of a real authored yaw interval.
        # The endpoint gaps and the circle chord give an independent first time.
        q0, q1 = np.array([0.0, 0.125]), np.array([np.pi / 2, 0.125])
        arc = asset.collision_motion(q0, q1)[tool.shape.key]
        mid = asset.collision_motion(0.5 * (q0 + q1))[tool.shape.key].start
        probe = ph.CollisionShape(ph.CollisionKey(0, 94, 0), "sphere", (9,), radius=1 / 32)
        orbit = ph.CollisionModel((tool.shape, probe))
        with ph.CollisionWorkspace(orbit, "host") as work:
            (swept,) = ph.detect(
                orbit,
                {tool.shape.key: arc, probe.key: ph.CollisionMotion(mid)},
                ph.CollisionPolicy(),
                work,
            ).pairs
        arm = arc.start[0] - np.asarray(arc.turn.pivot_start)
        radius = np.linalg.norm(np.cross(arc.turn.axis, arm))
        angle = 2 * np.arcsin((tool.shape.radius + probe.radius) / (2 * radius))
        first_time = 0.5 - angle / abs(arc.turn.angle)
        assert swept.status == "hit" and swept.sweep.start_gap_m > 0 and swept.sweep.end_gap_m > 0
        lo, hi = swept.sweep.first_t
        assert lo - 1e-12 <= first_time <= hi + 1e-12 and hi - lo < 1e-6
        result["usd"] = dict(
            sdk=sdk.version,
            sdk_runtime_version=usd_version,
            sdk_modules=loaded_sdk,
            collision_backend="HOST",
            moving_links=asset.model.dof_count,
            shapes=len(pose),
            tool_cylinder_gap_m=contact.witness.gap_m,
            joint_arc_first_time=first_time,
            joint_arc_interval=[lo, hi],
        )
    else:
        assert "pxr" not in sys.modules
        result.update(
            robot=robot_sample(ph, np),
            force=collision_and_force(ph, np, args.mode, args.out.parent / "cache"),
            field=field_sample(ph, np, args.mode, args.out.parent / "cache"),
            pages=many_worlds(ph, args.mode, args.out.parent / "cache"),
        )
    if args.mode in ("host", "cuda"):
        assert args.mesh_assets and not args.mesh_assets.resolve().is_relative_to(checkout)
        result["mesh_assets"] = imported_mesh_contacts(
            ph, np, args.mesh_assets, args.mode, args.out.parent / "cache"
        )
        result["floating_impact"] = floating_impact(ph, np, args.mode, args.out.parent / "cache")
        result["floating_support"] = floating_support_window(ph, np)
        result["mixed_world_step"] = mixed_world_step(ph, np, args.mode, args.out.parent / "cache")
        result["material_step"] = deformable_force_step(
            ph, np, args.mode, args.out.parent / "cache"
        )
    if args.mode != "usd":
        assert not any(n == "pxr" or n.startswith("pxr.") for n in sys.modules)
    used = [
        Path(module.__file__).resolve()
        for name, module in tuple(sys.modules.items())
        if (name == "phymeo" or name.startswith("phymeo.")) and getattr(module, "__file__", None)
    ]
    assert used and all(file.is_relative_to(site) for file in used)
    with ZipFile(args.expected_wheel) as archive:
        for file in set(used):
            member = file.relative_to(site).as_posix()
            assert archive.read(member) == file.read_bytes(), member
    result["installed_phymeo_module_count"] = len(used)
    result["verified_wheel_member_count"] = len(set(used))
    with args.out.open("x", encoding="utf8") as out:
        json.dump(result, out, allow_nan=False, indent=2)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

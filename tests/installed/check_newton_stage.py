"""Run from outside the source checkout against a separately installed stage wheel."""

import argparse
import json
import sys
import sysconfig
from contextlib import ExitStack
from importlib.metadata import distribution
from pathlib import Path


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
            force_n=reading.force_n.tolist(),
            torque_nm=reading.torque_nm.tolist(),
            rigid_velocity_m_s=output.state.rigid.velocity.tolist(),
            path=output.path.pairs[0].reason,
            completed_broad=box.stats()["completed_launches"],
            force_sites=None if law is None else law.stats()["sites"],
        )


def floating_impact(ph, np):
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
    with ph.CollisionWorkspace(shapes) as work:
        (pair,) = ph.detect(
            shapes,
            {rk: ph.CollisionMotion([[1.0, -0.125, 0.0]]), bk: ph.CollisionMotion(s.position)},
            ph.CollisionPolicy(),
            work,
        ).pairs
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
        ports, [(pair, {rk: ("arm", 0), bk: ("ball", 0)})], [0.0], [0.0]
    )
    np.testing.assert_allclose(impulse.states["arm"].v, [0, -0.125, 0, 0, 0, -0.25, 0], atol=1e-10)
    assert abs(impulse.normal_impulses[0] - 0.625) < 1e-10
    return dict(
        impulse_ns=float(impulse.normal_impulses[0]), kinetic_change_j=impulse.energy_change
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
        assert batch.stats()["backend_counts"]["compact_skipped_slots"] == 65 * 5
        return dict(
            worlds=len(obs),
            shape_count=260,
            pages=len(batch.layout),
            local_hits=len(hits),
            skipped_pair_rows=65 * 5,
        )


def deformable_force_step(ph, np, mode, cache):
    dtype = np.float64 if mode == "host" else np.float32
    ids = ((11, 23, 37), (53, 67, 79))
    xyz = np.array([[-1,-1,0],[1,-1,0],[0,1,0],
                    [0,0,3/64],[-.125,.125,.75],[.125,.125,.75]],dtype=dtype)
    shapes = [ph.CollisionShape(ph.CollisionKey(8, body, 0), "triangle", nodes)
              for body, nodes in zip((7,13),ids,strict=True)]
    model = ph.CollisionModel(shapes)
    bindings = {shape.key:ph.ParticleAttachment(tuple(range(3*i,3*i+3))) for i,shape in enumerate(shapes)}
    particles = ph.ParticleModel(np.ones(6,dtype=dtype),np.zeros(3,dtype=dtype))
    state=ph.ParticleState(xyz,np.zeros_like(xyz))
    with ExitStack() as scope:
        work=scope.enter_context(ph.CollisionWorkspace(model,mode,cache_root=cache))
        law=None if mode=="host" else scope.enter_context(ph.TyxNormalLayer(mode,cache_root=cache))
        solver=ph.SemiImplicitSolver() if mode=="host" else scope.enter_context(ph.TyxSemiImplicitSolver(mode,cache_root=cache))
        system=ph.DeformableFeatureLoadSystem(model,bindings,ph.NormalLayer(1/16,128.),search_distance_m=3/32,layer_operator=law)
        initial=system.sample_particles(state,work)
        np.testing.assert_allclose(initial.nodal_force_n[:,2],[-.5,-.5,-1.,2.,0.,0.],rtol=0,atol=2e-6)
        step=solver.step(particles,state,initial.step_inputs(),1/256)
        assert step.report.accepted
        after=system.sample_particles(step.state,work)
        first,=initial.contacts
        last,=after.contacts
        assert first.feature.kind=="VT" and last.feature.distance_m>first.feature.distance_m
        assert last.force_n<first.force_n
        return dict(first_force_n=first.force_n,after_force_n=last.force_n,
                    final_gap_m=last.feature.distance_m,receiver_velocity_m_s=step.state.velocity[3].tolist(),
                    physics_backend=step.report.backend)


def imported_mesh_contacts(ph, np, assets, mode, cache):
    probes = ((1.375, 1.375, 0.), (.5, 1.5, 0.))
    pivot = np.array([.5, .25, -.125])
    offset = np.array([.125, -.25, .0625])
    scale = np.array([.125, .25, .25])
    rz = np.array([[0., -1, 0], [1., 0, 0], [0, 0, 1.]])
    ry = np.array([[.8, 0, .6], [0, 1., 0], [-.6, 0, .8]])
    output = {}
    for suffix, loader in (("urdf", ph.load_urdf), ("xml", ph.load_mjcf)):
        asset = loader(assets / ("mesh_arm." + suffix))
        _, bracket = asset.collisions
        assert bracket.shape.kind == "mesh" and len(bracket.shape.triangles) == 20
        keys = [ph.CollisionKey(0, 77, i) for i in (0, 1)]
        spheres = [ph.CollisionShape(k, "sphere", (0,), 1/64) for k in keys]
        model = ph.CollisionModel((*asset.collision_model.shapes, *spheres), asset.collision_model.excluded)
        paths = asset.collision_motion(np.array([np.arctan2(.6, .8)]))
        for key, local in zip(keys, probes, strict=True):
            center = pivot + (offset + (np.asarray(local) * scale) @ rz.T) @ ry.T
            paths[key] = ph.CollisionMotion([center])
        with ph.CollisionWorkspace(model, mode, cache_root=cache) as work:
            pairs = ph.detect(model, paths, ph.CollisionPolicy(margin=.5), work).pairs
            poses = [next(p for p in pairs if {p.first,p.second} == {bracket.shape.key,k}).mesh_round.end
                     for k in keys]
            assert [p.relation for p in poses] == ["separated", "mesh_contains_round"]
            np.testing.assert_allclose([p.boundary_gap_m for p in poses], [.03125, .046875], atol=2e-12)
            if mode == "cuda":
                assert work.stats()["completed_launches"] > 0
        output[suffix] = [dict(relation=p.relation, boundary_gap_m=p.boundary_gap_m) for p in poses]
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wheel-site", type=Path, required=True)
    p.add_argument("--checkout", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["host", "cuda", "usd"], required=True)
    p.add_argument("--usd-fixture", type=Path)
    p.add_argument("--mesh-assets", type=Path)
    args = p.parse_args()
    site, checkout = args.wheel_site.resolve(), args.checkout.resolve()
    assert sys.flags.isolated and all(
        not path.resolve().is_relative_to(checkout) for path in (site, Path(__file__), Path.cwd())
    )
    sys.path.insert(0, str(site))
    import numpy as np

    import phymeo as ph

    origin = Path(ph.__file__).resolve()
    assert (
        origin.is_relative_to(site)
        and Path(distribution("phymeo").locate_file("")).resolve() == site
    )
    # The optional USD interpreter lives in a worktree-local venv. Its installed
    # dependencies are distinct from a source checkout on the import path.
    dependency_dirs = {
        Path(sysconfig.get_path(name)).resolve()
        for name in ("stdlib", "platstdlib", "purelib", "platlib")
    }
    local_env_paths = [Path(path).resolve() for path in sys.path
                       if path and Path(path).resolve().is_relative_to(checkout)]
    for path in local_env_paths:
        assert Path(sys.prefix).resolve().is_relative_to(checkout) and Path(sys.prefix).resolve() != checkout
        assert path.is_relative_to(Path(sys.prefix).resolve()) and (
            path == Path(sys.prefix).resolve() or any(path.is_relative_to(folder) for folder in dependency_dirs)
        ), path
        assert path != checkout / "src" and path != checkout, path
    result = dict(
        package_version=distribution("phymeo").version,
        mode=args.mode,
        package=str(origin),
        cwd=str(Path.cwd()),
        interpreter_dependency_paths_under_checkout=[str(path) for path in local_env_paths],
    )
    if args.mode == "usd":
        assert args.usd_fixture and not args.usd_fixture.resolve().is_relative_to(checkout)
        asset = ph.load_usd(args.usd_fixture)
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
        result["usd"] = dict(
            sdk=distribution("usd-core").version,
            moving_links=asset.model.dof_count,
            shapes=len(pose),
            tool_cylinder_gap_m=contact.witness.gap_m,
        )
    else:
        assert "pxr" not in sys.modules
        result.update(
            robot=robot_sample(ph, np),
            force=collision_and_force(ph, np, args.mode, args.out.parent / "cache"),
            field=field_sample(ph, np, args.mode, args.out.parent / "cache"),
            pages=many_worlds(ph, args.mode, args.out.parent / "cache"),
        )
    if args.mode != "usd":
        assert args.mesh_assets and not args.mesh_assets.resolve().is_relative_to(checkout)
        result["mesh_assets"] = imported_mesh_contacts(ph, np, args.mesh_assets, args.mode, args.out.parent / "cache")
        result["floating_impact"] = floating_impact(ph, np)
        result["material_step"] = deformable_force_step(ph, np, args.mode, args.out.parent / "cache")
    used = [
        Path(module.__file__).resolve()
        for name, module in tuple(sys.modules.items())
        if (name == "phymeo" or name.startswith("phymeo.")) and getattr(module, "__file__", None)
    ]
    assert used and all(file.is_relative_to(site) for file in used)
    result["installed_phymeo_module_count"] = len(used)
    with args.out.open("x", encoding="utf8") as out:
        json.dump(result, out, allow_nan=False, indent=2)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

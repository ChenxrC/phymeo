"""Independent mass, material work and finite-face boundaries for moving cloth stiction."""

import json
from contextlib import ExitStack

import numpy as np
import pytest

import phymeo as ph


def scene(*, joined=False, angled=False, symmetric=False, dtype=np.float64, law=None, x=None):
    from phymeo.solvers.deformable_moving_stiction import MovingMaterialStickSlipStepper

    fixed_xyz = (
        [[-1, 0, 0], [0, -1, 0], [0, 1, 0], [1, 0, 0]]
        if joined
        else [[-2, -1, 0], [2, -1, 0], [0, 1, 0]]
    )
    tip_x = (-0.25 if joined else 0.0) if x is None else x
    top_xyz = [[tip_x, 0, 1 / 32], [tip_x + 0.1875, 0.25, 0.75], [tip_x - 0.375, 0.5, 1.0]]
    x0 = np.array(fixed_xyz + top_xyz, dtype=dtype)
    nface = len(fixed_xyz)
    mass = np.array(
        ([1.0, 2.0, 4.0, 3.0] if joined else [1.0, 2.0, 4.0]) + [1.0, 1.5, 2.0], dtype=dtype
    )
    if symmetric:
        mass[1] = mass[0]
    faces = [[0, 1, 2], [1, 3, 2]] if joined else [[0, 1, 2]]
    faces.append(list(range(nface, nface + 3)))
    rot = np.array([[0.8, 0, -0.6], [0, 1, 0], [0.6, 0, 0.8]] if angled else np.eye(3), dtype=dtype)
    order = np.arange(len(mass))[::-1] if angled else np.arange(len(mass))
    rows = np.argsort(order)
    triangles = [
        ph.CollisionShape(
            ph.CollisionKey(7, 31 if j == len(faces) - 1 else 8, j),
            "triangle",
            tuple(13 * i + 7 for i in ids),
        )
        for j, ids in enumerate(faces)
    ]
    geometry = ph.CollisionModel(triangles[::-1] if angled else triangles)
    bindings = {
        shape.key: ph.ParticleAttachment(tuple(int(rows[i]) for i in nodes))
        for shape, nodes in zip(triangles, faces, strict=True)
    }
    model = ph.ParticleModel(1 / mass[order], np.zeros(3, dtype=dtype))
    op = MovingMaterialStickSlipStepper(
        model,
        geometry,
        bindings,
        ph.NormalLayer(1 / 16, 64, friction=0.25),
        static_friction=0.5,
        tangent_stiffness_n_per_m=256,
        layer_operator=law,
    )
    v0 = np.full_like(x0, 0.0)
    v0[:] = [0.25, -0.125, 0] if not joined else [0.0, 0.0, 0.0]
    initial = op.initial_state(ph.ParticleState((x0 @ rot.T)[order], (v0 @ rot.T)[order]))
    return geometry, op, initial, rows, mass, rot, nface


def weights(x, vertex, face, n):
    """Independent orthogonal projection and barycentric inversion at the input state."""
    a, b, c = x[face]
    q = x[vertex] - (n @ (x[vertex] - a)) * n
    mat = np.stack([b - a, c - a], axis=1)
    uv = np.linalg.solve(mat.T @ mat, mat.T @ (q - a))
    return np.array([1 - uv.sum(), *uv])


def pressure(state, tip, face, n, shear):
    x = state.particles.position
    w = weights(x, tip, face, n)
    normal = 64 * (1 / 16 - float(n @ (x[tip] - x[face[0]])))
    f = ph.StepInputs.zeros(len(x), dtype=x.dtype)
    f.force[tip] = shear - normal * n
    f.force[face] = w[:, None] * (normal * n)
    return f


def one_step(case, workspace, *, shear=4.0, solver=None, atol=1e-12):
    geo, op, start, rows, m, R, j = case
    top, face = rows[j], rows[:3]
    t, n = R[:, 0], R[:, 2]
    h = 1 / 16
    w = np.array([0.25, 0.25, 0.5])
    inv_eff = 1 / m[j] + sum(w * w / m[:3])
    force_trial = (256 * h * h * shear / m[j]) / (1 + 256 * h * h * inv_eff)
    f = -force_trial if force_trial <= 1 else -0.5
    command = pressure(start, top, face, n, shear * t)
    out = op.step(start, command, h, workspace, particle_solver=solver)
    (c,) = out.contacts
    old, end = start.particles, out.state.particles
    assert out.report.accepted and c.normal_force_n == pytest.approx(2, abs=atol)
    assert c.history.regime == ("stick" if abs(force_trial) <= 1 else "slide")
    force_rows = np.zeros_like(old.velocity, dtype=float)
    force_rows[top] = f * t
    force_rows[face] = -w[:, None] * f * t
    np.testing.assert_allclose(out.nodal_tangent_force_n, force_rows, atol=atol)
    external_tangent = np.zeros_like(force_rows)
    external_tangent[top] = shear * t
    node_mass = np.empty_like(m)
    node_mass[rows] = m
    predicted_v = old.velocity + h * (external_tangent + force_rows) / node_mass[:, None]
    np.testing.assert_allclose(end.velocity, predicted_v, atol=atol)
    dx = end.position.astype(float) - old.position
    np.testing.assert_allclose(dx, h * predicted_v, atol=atol)
    np.testing.assert_allclose(
        ((end.velocity - old.velocity) * node_mass[:, None]).sum(axis=0), h * shear * t, atol=atol
    )
    angular0 = np.cross(old.position, node_mass[:, None] * old.velocity).sum(axis=0)
    angular1 = np.cross(end.position, node_mass[:, None] * end.velocity).sum(axis=0)
    couple = np.cross(old.position[top] - w @ old.position[face], f * t)
    assert np.linalg.norm(couple) > 0
    np.testing.assert_allclose(
        angular1 - angular0 - h * np.cross(old.position[top], shear * t), h * couple, atol=atol
    )
    relative = dx[top] - w @ dx[face]
    np.testing.assert_allclose(c.material_displacement_m, relative, atol=atol)
    assert c.effective_inverse_mass_per_kg == pytest.approx(inv_eff, abs=atol)
    assert c.vertex_tangent_work_j == pytest.approx(force_rows[top] @ dx[top], abs=atol)
    expected_work = np.sum(force_rows[face] * dx[face], axis=1)
    # The response uses canonical material-node order, independent of shape registration and array order.
    np.testing.assert_allclose(c.support_node_tangent_work_j, expected_work, atol=atol)
    assert c.tangent_work_j == pytest.approx(c.vertex_tangent_work_j + sum(expected_work), abs=atol)
    assert (
        c.tangent_work_j
        + c.history.stored_energy_j
        + c.plastic_dissipation_j
        + c.implicit_dissipation_j
        == pytest.approx(0, abs=atol)
    )
    delta_k = 0.5 * np.sum(
        node_mass[:, None] * (end.velocity - old.velocity) * (end.velocity + old.velocity)
    )
    impulse_loss = 0.5 * np.sum(node_mass[:, None] * (end.velocity - old.velocity) ** 2)
    assert delta_k + impulse_loss == pytest.approx(shear * t @ dx[top] + c.tangent_work_j, abs=atol)
    return out


@pytest.mark.parametrize("angled,shear", [(False, 0.5), (True, 4.0)])
def test_moving_finite_face_two_sided_response_and_work(angled, shear, tmp_path):
    case = scene(angled=angled)
    with ph.CollisionWorkspace(case[0]) as work:
        out = one_step(case, work, shear=shear)
    c = out.contacts[0]
    (tmp_path / "moving-face-step.json").write_text(
        json.dumps(
            dict(
                mass_kg=case[4].tolist(),
                world_normal=case[5][:, 2].tolist(),
                h_s=1 / 16,
                applied_shear_n=shear,
                inverse_mass_per_kg=c.effective_inverse_mass_per_kg,
                normal_force_n=c.normal_force_n,
                vertex_traction_n=c.tangent_force_on_vertex_n,
                relative_displacement_m=c.material_displacement_m,
                plastic_m=c.history.plastic_slip_m,
                elastic_energy_j=c.history.stored_energy_j,
                vertex_work_j=c.vertex_tangent_work_j,
                face_work_j=c.support_node_tangent_work_j,
                plastic_loss_j=c.plastic_dissipation_j,
                implicit_loss_j=c.implicit_dissipation_j,
                x_m=out.state.particles.position.tolist(),
                velocity_m_s=out.state.particles.velocity.tolist(),
                node_shear_n=out.nodal_tangent_force_n.tolist(),
            ),
            allow_nan=False,
        ),
        encoding="utf8",
    )


def impose_positions(state, masses, target, h, tangent_rows=None, normal_rows=None):
    old = state.particles
    value = masses[:, None] * ((target - old.position) / h - old.velocity) / h
    if tangent_rows is not None:
        value -= tangent_rows
    if normal_rows is not None:
        value -= normal_rows
    return ph.StepInputs(value.astype(old.position.dtype))


def balanced_rows(state, top, face, normal, shear):
    x = state.particles.position
    w = weights(x, top, face, normal)
    pressure = 64 * (1 / 16 - normal @ (x[top] - x[face[0]]))
    row = np.zeros_like(x)
    row[top] = pressure * normal + shear
    row[face] = -w[:, None] * row[top]
    return row


def test_loaded_face_transports_its_elastic_vector_through_actual_in_plane_rotation():
    case = scene()
    geo, op, start, rows, m, R, top = case
    face = rows[:3]
    normal = R[:, 2]
    h = 1 / 16
    spin = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    with ph.CollisionWorkspace(geo) as work:
        first = one_step(case, work, shear=0.5)
        state = first.state
        # Constant initial traction derived from the translating 4-mass one-step solution.
        spring = np.array(
            [h * h * 0.5 / (1 + 256 * h * h * (1 + 0.25**2 + 0.25**2 / 2 + 0.5**2 / 4)), 0.0, 0.0]
        )
        old = state.particles.position
        # The unequal support masses already shear the face during the preload.
        # Independently recover its polar rotation from the measured/analytically checked nodes.
        edge0 = start.particles.position[face[1:], :2] - start.particles.position[face[0], :2]
        edge1 = old[face[1:], :2] - old[face[0], :2]
        transform = np.linalg.solve(edge0, edge1).T
        left, singular, right = np.linalg.svd(transform)
        assert np.min(singular) > 0 and np.linalg.det(left @ right) > 0
        spring[:2] = (left @ right) @ spring[:2]
        np.testing.assert_allclose(state.active[0].elastic_displacement_m, spring, atol=2e-14)
        target = old @ spin.T + [0.03125, -0.0625, 0.125]
        internal = balanced_rows(state, top, face, normal, -256 * spring)
        turned = op.step(state, impose_positions(state, m, target, h, internal), h, work)
        c = turned.contacts[0]
        np.testing.assert_allclose(turned.state.particles.position, target, atol=4e-14)
        np.testing.assert_allclose(c.history.elastic_displacement_m, spin @ spring, atol=2e-14)
        assert c.support_turn_rad == pytest.approx(np.pi / 2)
        assert c.history.stored_energy_j == pytest.approx(first.contacts[0].history.stored_energy_j)
        assert c.history.plastic_slip_m == 0 and abs(c.tangent_work_j) < 1e-13
        assert c.vertex_tangent_work_j == pytest.approx(
            -sum(c.support_node_tangent_work_j), abs=1e-13
        )
        internal = balanced_rows(turned.state, top, face, normal, -256 * (spin @ spring))
        stopped = op.step(
            turned.state, impose_positions(turned.state, m, target, h, internal), h, work
        )
        np.testing.assert_allclose(
            stopped.nodal_tangent_force_n[top], -256 * (spin @ spring), atol=1e-12
        )
        assert stopped.contacts[0].history.active_steps == 3


def test_through_edge_is_rejected_and_unloaded_motion_can_reenter_the_neighbor(tmp_path):
    geo, op, start, rows, m, R, top = scene(joined=True)
    h = 1 / 16
    n, t = R[:, 2], R[:, 0]
    left, right = rows[[0, 1, 2]], rows[[1, 3, 2]]
    with ph.CollisionWorkspace(geo) as work:
        holding = op.step(start, pressure(start, top, left, n, 0.5 * t), h, work)
        (old,) = holding.state.active
        assert old.stored_energy_j > 0 and old.plastic_slip_m == 0
        with pytest.raises(ph.SolverFailure, match="edge"):
            op.step(holding.state, pressure(holding.state, top, left, n, 160 * t), h, work)
        assert holding.state.active == (old,) and holding.state.tick == 1
        # Lift first, then traverse the shared edge with the normal layer clear.
        lift = pressure(holding.state, top, left, n, 0.5 * t)
        lift.force[top] += 32 * n
        leaving = op.step(holding.state, lift, h, work)
        assert leaving.state.particles.position[top, 2] > 1 / 16
        target = leaving.state.particles.position.copy()
        target[top, 0] = 0.25
        clear = op.step(leaving.state, impose_positions(leaving.state, m, target, h), h, work)
        assert not clear.contacts and clear.ended == leaving.state.active
        assert clear.released_spring_energy_j > 0
        np.testing.assert_array_equal(clear.nodal_tangent_force_n, np.zeros_like(target))
        target[top, 2] = 1 / 32
        landing = op.step(clear.state, impose_positions(clear.state, m, target, h), h, work)
        assert not landing.state.active
        internal = balanced_rows(landing.state, top, right, n, np.zeros(3))
        caught = op.step(
            landing.state, impose_positions(landing.state, m, target, h, internal), h, work
        )
        (fresh,) = caught.state.active
        assert fresh.key[1] == old.key[1] and fresh.key[2] != old.key[2]
        assert len(set(fresh.key[2]) & set(old.key[2])) == 2
        assert fresh.start_tick == landing.state.tick and fresh.active_steps == 1
        assert fresh.stored_energy_j < 1e-25 and fresh.plastic_slip_m == 0
        np.testing.assert_allclose(caught.nodal_tangent_force_n, 0, atol=1e-12)
        replay = op.step(holding.state, lift, h, work)
        assert replay.contacts == leaving.contacts
        (tmp_path / "moving-face-edge-release.json").write_text(
            json.dumps(
                dict(
                    old_face=old.key[2],
                    new_face=fresh.key[2],
                    ended_tick=clear.state.tick,
                    released_spring_energy_j=clear.released_spring_energy_j,
                    fresh_start_tick=fresh.start_tick,
                    new_spring_energy_j=fresh.stored_energy_j,
                    vertex_xyz=target[top].tolist(),
                )
            ),
            encoding="utf8",
        )

    geo, op, on_edge, *_ = scene(joined=True, x=0.0)
    with ph.CollisionWorkspace(geo) as work, pytest.raises(ph.SolverFailure, match="interior|edge"):
        op.step(on_edge, ph.StepInputs.zeros(len(on_edge.particles.position)), h, work)


def test_tilt_and_midstep_collapse_do_not_publish():
    geo, op, start, rows, m, R, top = scene()
    h = 1 / 16
    n = R[:, 2]
    with ph.CollisionWorkspace(geo) as work:
        tilt = pressure(start, top, rows[:3], n, np.zeros(3))
        tilt.force[0] += n
        with pytest.raises(ph.SolverFailure, match="tilt"):
            op.step(start, tilt, h, work)
        # Both endpoints are valid, but linearly interpolating a 180-degree spin
        # brings all three material nodes to the same projected point at h/2.
        target = start.particles.position * np.array([-1.0, -1.0, 1.0])
        load = balanced_rows(start, top, rows[:3], n, np.zeros(3))
        with pytest.raises(ph.SolverFailure, match="degenerate"):
            op.step(start, impose_positions(start, m, target, h, load), h, work)
        assert not start.active and start.tick == 0


@pytest.mark.tyx
def test_device_moves_both_material_sides_in_the_same_actual_particle_step(tyx_mode, tmp_path):
    with ExitStack() as stack:
        law = stack.enter_context(ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path / "cache"))
        solver = stack.enter_context(
            ph.TyxSemiImplicitSolver(tyx_mode, cache_root=tmp_path / "cache")
        )
        case = scene(dtype=np.float32, law=law)
        workspace = stack.enter_context(
            ph.CollisionWorkspace(case[0], tyx_mode, cache_root=tmp_path / "cache")
        )
        out = one_step(case, workspace, solver=solver, atol=3e-6)
        assert (
            out.report.backend == "tyx:" + tyx_mode and out.normal_loads.force_backend == tyx_mode
        )
        assert law.stats()["completed_launches"] == 1
        if tyx_mode == "cuda":
            assert workspace.stats()["completed_launches"] > 0 and law.stats()["downloads"] > 0
        (tmp_path / "moving-face-device.json").write_text(
            json.dumps(
                dict(
                    mode=tyx_mode,
                    normal=law.stats(),
                    broad=workspace.stats(),
                    particle=out.report.backend,
                    vertex_work_j=out.contacts[0].vertex_tangent_work_j,
                    face_work_j=out.contacts[0].support_node_tangent_work_j,
                    contact_work_j=out.contacts[0].tangent_work_j,
                    plastic_m=out.contacts[0].history.plastic_slip_m,
                    position=out.state.particles.position.tolist(),
                    velocity=out.state.particles.velocity.tolist(),
                ),
                allow_nan=False,
            ),
            encoding="utf8",
        )


def test_moving_deforming_face_continuously_holds_then_slides_against_a_scalar_reference(tmp_path):
    geo, op, state, rows, m, R, top = scene(symmetric=True)
    h, k = 1 / 16, 256.0
    y = state.particles.position[:, 1].copy()
    v = state.particles.velocity[:, 1].copy()
    e = plastic = work_total = 0.0
    slid = False
    log = []
    with ph.CollisionWorkspace(geo) as space:
        for tick, ext in enumerate([0.5] * 16 + [4.0] * 4, 1):
            # A mirror pair of face nodes has the same y coordinate and velocity.
            # The third node gives an independently computed affine material address.
            apex = (y[top] - y[0]) / (y[2] - y[0])
            w = np.array([(1 - apex) / 2, (1 - apex) / 2, apex])
            mobility = 1 / m[top] + np.sum(w * w / m[:3])
            trial = (e + h * (v[top] - w @ v[:3] + h * ext / m[top])) / (1 + k * h * h * mobility)
            slid = k * abs(trial) > (0.5 if slid and trial * e >= 0 else 1.0)
            f = -np.sign(trial) * 0.5 if slid else -k * trial
            external = ph.StepInputs.zeros(len(m))
            external.force[top] = [0, ext, -2]
            external.force[:3, 2] = 2 * w
            v[top] += h * (ext + f) / m[top]
            v[:3] -= h * w * f / m[:3]
            y += h * v
            du = h * (v[top] - w @ v[:3])
            enew = -f / k if slid else e + du
            plastic += abs(e + du - enew) if slid else 0.0
            work_total += f * du
            out = op.step(state, external, h, space)
            (c,) = out.contacts
            assert c.history.regime == ("slide" if slid else "stick")
            assert c.history.regime == ("stick" if tick <= 16 else "slide")
            np.testing.assert_allclose(out.state.particles.position[:, 1], y, atol=2e-13)
            np.testing.assert_allclose(out.state.particles.velocity[:, 1], v, atol=2e-13)
            assert c.tangent_force_on_vertex_n[1] == pytest.approx(f, abs=2e-12)
            assert c.history.plastic_slip_m == pytest.approx(plastic, abs=2e-13)
            assert c.history.tangent_work_j == pytest.approx(work_total, abs=2e-13)
            assert c.history.stored_energy_j == pytest.approx(0.5 * k * enew**2, abs=2e-13)
            log.append(
                dict(
                    tick=tick,
                    pressure_n=c.normal_force_n,
                    regime=c.history.regime,
                    shear_n=c.tangent_force_on_vertex_n,
                    vertex_y_m=out.state.particles.position[top, 1],
                    face_y_m=out.state.particles.position[:3, 1].tolist(),
                    weights=w.tolist(),
                    stored_j=c.history.stored_energy_j,
                    plastic_m=c.history.plastic_slip_m,
                    cumulative_tangential_work_j=c.history.tangent_work_j,
                )
            )
            state, e = out.state, enew
        assert abs((y[2] - y[0]) - 2) > 1e-4 and plastic > 0
        (tmp_path / "moving-face-hold-window.json").write_text(
            json.dumps(log, allow_nan=False), encoding="utf8"
        )


def test_two_loaded_vertices_sharing_free_support_nodes_are_rejected():
    from phymeo.solvers.deformable_moving_stiction import MovingMaterialStickSlipStepper

    face = np.array([[-2.0, -1.0, 0.0], [2.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
    tip = np.array([[0, 0, 1 / 32], [0.1875, 0.25, 0.75], [-0.375, 0.5, 1.0]])
    x = np.vstack([face, tip - [0.5, 0, 0], tip + [0.5, 0, 0]])
    shapes = [ph.CollisionShape(ph.CollisionKey(1, k, 0), "triangle", (1, 2, 3)) for k in (3, 7, 9)]
    geom = ph.CollisionModel(shapes)
    attach = {
        s.key: ph.ParticleAttachment((3 * i, 3 * i + 1, 3 * i + 2)) for i, s in enumerate(shapes)
    }
    mass = ph.ParticleModel(np.ones(9), np.zeros(3))
    op = MovingMaterialStickSlipStepper(
        mass,
        geom,
        attach,
        ph.NormalLayer(1 / 16, 64, friction=0.25),
        static_friction=0.5,
        tangent_stiffness_n_per_m=256,
    )
    start = op.initial_state(ph.ParticleState(x, np.zeros_like(x)))
    with ph.CollisionWorkspace(geom) as space, pytest.raises(ph.SolverFailure, match="share"):
        op.step(start, ph.StepInputs.zeros(9), 1 / 16, space)
    assert not start.active and start.tick == 0


def test_endpoint_interior_does_not_hide_a_midstep_finite_edge_excursion():
    geo, op, start, rows, m, R, top = scene()
    h = 1 / 16
    n = R[:, 2]
    particles = start.particles
    particles.position[3:, 1] += 0.9
    start = op.initial_state(particles)
    spin = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])
    old = start.particles.position
    target = old @ spin.T
    target[3:, :2] += [0.9, 1.8] - target[top, :2]
    face = rows[:3]
    w = weights(old, top, face, n)
    assert np.min(w) > 0 and np.min(weights(target, top, face, n)) > 0
    middle = (old + target) / 2
    assert middle[top, 1] > middle[face, 1].max() + 0.8
    rel = (target[top] - old[top]) - w @ (target[face] - old[face])
    shear = -0.5 * rel / np.linalg.norm(rel)
    internal = balanced_rows(start, top, face, n, shear)
    external = impose_positions(start, m, target, h, internal)
    with ph.CollisionWorkspace(geo) as space, pytest.raises(ph.SolverFailure, match="edge"):
        op.step(start, external, h, space)
    assert not start.active and start.tick == 0

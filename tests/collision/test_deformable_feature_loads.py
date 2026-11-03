"""Actual two-sided particle force/power, canonical sites, and an accepted dynamics step."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_deformable_features import CAT, key, query, register

import phymeo as ph


def scene(id_, *, dtype=np.float64, reverse=False):
    case = CAT[id_]
    shapes, paths = register(case)
    lookup = {}
    for shape in shapes:
        for v, p in zip(shape.vertex_ids, paths[shape.key].start, strict=True):
            lookup[(shape.key.world, shape.key.body, v)] = p
    nodes = sorted(lookup, reverse=reverse)
    offsets = {n: i for i, n in enumerate(nodes)}
    bindings = {
        s.key: ph.ParticleAttachment(
            tuple(offsets[s.key.world, s.key.body, n] for n in s.vertex_ids)
        )
        for s in shapes
    }
    xyz = np.array([lookup[n] for n in nodes], dtype=dtype)
    return (
        ph.CollisionModel(shapes),
        bindings,
        ph.ParticleState(xyz, np.zeros_like(xyz)),
        nodes,
        paths,
    )


def system(model, bindings, **kw):
    return ph.DeformableFeatureLoadSystem(
        model,
        bindings,
        kw.pop("layer", ph.NormalLayer(1 / 16, 128.0)),
        search_distance_m=3 / 32,
        **kw,
    )


def force_map(nodes, nodal):
    return {n[2]: f for n, f in zip(nodes, nodal, strict=True)}


def conservation(q, state, tol=1e-11):
    np.testing.assert_allclose(q.nodal_force_n.sum(0), 0.0, rtol=0, atol=tol)
    np.testing.assert_allclose(
        np.cross(state.position, q.nodal_force_n).sum(0), 0.0, rtol=0, atol=tol
    )
    np.testing.assert_allclose(
        (state.velocity * q.nodal_force_n).sum(), q.power_w, rtol=0, atol=tol
    )
    assert q.power_w == pytest.approx(sum(x.power_w for x in q.contacts), abs=tol)


@pytest.mark.parametrize(
    "id_", ["DD-VT-TWO-SHEETS", "DD-EE-TWO-SHEETS", "DD-EE-MIRROR-SKEW", "DD-VT-DEEP-LAW"]
)
def test_material_feature_scatter_matches_independent_physical_forces(id_):
    m, b, state, nodes, paths = scene(id_)
    with ph.CollisionWorkspace(m, "host") as work:
        geo = query(m, paths, work)
        s = system(m, b)
        result = s.sample_particles(state, work, features=geo)
    assert s.last_sample is result and result.geometry is geo and len(result.contacts) == 1
    exp = CAT[id_]["expected"]
    magnitude = exp["phymeo_unique_force_n"]
    actual = force_map(nodes, result.nodal_force_n)
    expected = {n[2]: np.zeros(3) for n in nodes}
    for sign, weights in ((-1, exp["focus_weights_a"]), (1, exp["focus_weights_b"])):
        for i, w in weights:
            expected[i][2] = sign * w * magnitude
    np.testing.assert_allclose(
        [actual[i] for i in sorted(actual)], [expected[i] for i in sorted(actual)], atol=1e-12
    )
    assert result.contacts[0].force_n == magnitude and result.power_w == 0.0
    assert result.potential_j == pytest.approx(
        0.5 * 128 * (1 / 16 - exp["focus_distance_m"]) ** 2, abs=1e-15
    )
    conservation(result, state)


def test_fold_publishes_si_nodal_force_and_does_not_change_face_pair_calibration(tmp_path):
    m, b, state, nodes, _ = scene("DD-REMOTE-FOLD")
    records = []
    for reverse in (False, True):
        model, bind, st, n, _ = scene("DD-REMOTE-FOLD", reverse=reverse)
        with ph.CollisionWorkspace(model, "host") as work:
            layers = {
                (a.key, c.key): ph.NormalLayer(1 / 16, 128.0)
                for i, a in enumerate(model.shapes)
                for c in model.shapes[i + 1 :]
            }
            default = ph.CollisionLoadSystem(model, bind, layers).sample_particles(st, work)
            base_force = force_map(n, default.inputs.particles.force)
            baseline_lower = sum(base_force[i][2] for i in range(100, 106))
            assert baseline_lower == pytest.approx(-18, abs=1e-12)
            print("ordinary lower material force N", baseline_lower)
            s = system(model, bind)
            q = s.sample_particles(st, work)
        fm = force_map(n, q.nodal_force_n)
        f0 = force_map(n, default.inputs.particles.force)
        assert len(q.contacts) == 32 and sum(x.feature.kind == "VT" for x in q.contacts) == 8
        assert sum(len(x.sites) == 2 for x in q.contacts) == 6
        assert all(abs(sum(site.fraction for site in x.sites) - 1) < 1e-15 for x in q.contacts)
        assert all(x.force_n == 2 for x in q.contacts)
        assert sum(fm[i][2] for i in range(100, 106)) == pytest.approx(-64, abs=1e-12)
        assert sum(fm[i][2] for i in range(112, 118)) == pytest.approx(64, abs=1e-12)
        assert sum(f0[i][2] for i in range(100, 106)) == pytest.approx(-18, abs=1e-12)
        assert q.potential_j == 0.5
        conservation(q, st)
        records.append({str(i): v.tolist() for i, v in fm.items()})
    assert records[0] == records[1]
    with pytest.raises(ValueError):
        q.nodal_force_n.setflags(write=True)
    payload = dict(
        physical_nodal_n=records[0],
        default_lower_n=-18.0,
        feature_lower_n=-64.0,
        stored_energy_j=q.potential_j,
        sites=sum(len(x.sites) for x in q.contacts),
    )
    (tmp_path / "fold-feature-forces.json").write_text(
        json.dumps(payload, allow_nan=False), encoding="utf-8"
    )


def test_velocity_damping_power_uses_the_two_addressed_material_velocities():
    m, b, state, nodes, paths = scene("DD-VT-TWO-SHEETS")
    vel = state.velocity.copy()
    for i, v in {11: 1 / 8, 23: 3 / 8, 37: -1 / 8, 53: -3 / 8}.items():
        vel[next(k for k, n in enumerate(nodes) if n[2] == i), 2] = v
    vel[:, 0] = [1, 2, -3, 6, 7, 8]
    current = ph.ParticleState(state.position, vel)
    s = system(m, b, layer=ph.NormalLayer(1 / 16, 128.0, damping_s_per_m=0.5))
    with ph.CollisionWorkspace(m, "host") as work:
        geo = query(m, paths, work)
        result = s.sample_particles(current, work, features=geo)
    (load,) = result.contacts
    (site,) = load.sites
    # VT first is the tip, pointing downward to the surface. The stored normal
    # rate is closing; source velocities are still those of the original nodes.
    assert site.first_velocity_m_s[2] == -3 / 8 and site.second_velocity_m_s[2] == 1 / 16
    assert site.rate_m_per_s == -7 / 16
    assert load.force_n == 2 * (1 + 7 / 32)
    assert result.power_w == load.force_n * (-7 / 16)
    assert result.potential_j == 1 / 64
    np.testing.assert_allclose(site.force_on_first_n, -np.asarray(site.force_on_second_n), atol=0)
    conservation(result, current)


def test_external_feature_binding_rejects_stale_shapes_and_preserves_last_sample():
    m, b, state, nodes, paths = scene("DD-VT-TWO-SHEETS")
    s = system(m, b)
    with ph.CollisionWorkspace(m, "host") as work:
        good = query(m, paths, work)
        published = s.sample_particles(state, work, features=good)
        pose = state.position.copy()
        pose[next(i for i, n in enumerate(nodes) if n[2] == 67), 1] += 0.125
        with pytest.raises(ph.SolverFailure, match="stale"):
            s.sample_particles(ph.ParticleState(pose, state.velocity), work, features=good)
        for invalid in (
            query(m, paths, work, force=1 / 32),
            query(m, paths, work, topology_rings=1),
            replace(good, contacts=()),
        ):
            with pytest.raises(ph.SolverFailure, match="stale"):
                s.sample_particles(state, work, features=invalid)
        newmodel = ph.CollisionModel(m.shapes)
        with ph.CollisionWorkspace(newmodel, "host") as other:
            foreign = query(newmodel, paths, other)
            with pytest.raises(ph.SolverFailure, match="stale"):
                s.sample_particles(state, work, features=foreign)
        failed = query(m, paths, work, feature_capacity=0)
        with pytest.raises(ph.SolverFailure, match="capacity"):
            s.sample_particles(state, work, features=failed)
        coincident = state.position.copy()
        coincident[next(i for i, n in enumerate(nodes) if n[2] == 53), 2] = 0.0
        with pytest.raises(ph.SolverFailure, match="normal"):
            s.sample_particles(ph.ParticleState(coincident, state.velocity), work)
        assert s.last_sample is published


def dynamic(s, model, reg, state, nodes, work, solver):
    h = 1 / 256
    pmodel = ph.ParticleModel(
        np.ones(len(nodes), dtype=state.position.dtype), np.zeros(3, dtype=state.position.dtype)
    )
    first = s.sample_particles(state, work)
    external = ph.StepInputs.zeros(len(nodes), dtype=state.position.dtype)
    inputs = first.step_inputs(external)
    assert inputs.force.dtype == state.position.dtype
    step = solver.step(pmodel, state, inputs, h)
    assert step.report.accepted
    v = state.velocity + h * inputs.force
    np.testing.assert_allclose(step.state.velocity, v, rtol=0, atol=2e-7)
    np.testing.assert_allclose(step.state.position, state.position + h * v, rtol=0, atol=2e-7)
    second = s.sample_particles(step.state, work)
    f0, f1 = first.contacts[0], second.contacts[0]
    assert 0.046875 < f1.feature.distance_m < 0.04694 and f1.force_n < f0.force_n
    kinetic = float(np.sum(step.state.velocity.astype(float) ** 2) / 2)
    assert kinetic == pytest.approx(2.75 * h * h, abs=2e-11)
    assert 0 < second.potential_j < first.potential_j and first.power_w == 0
    return dict(
        backend=step.report.backend,
        initial_q=state.position.tolist(),
        final_q=step.state.position.tolist(),
        final_v=step.state.velocity.tolist(),
        initial_stored_j=first.potential_j,
        final_stored_j=second.potential_j,
        kinetic_j=kinetic,
        external_work_j=float((external.force * (step.state.position - state.position)).sum()),
        first_gap_m=f0.feature.distance_m,
        end_gap_m=f1.feature.distance_m,
        initial_force_n=f0.force_n,
        end_force_n=f1.force_n,
    )


def test_original_host_solver_advances_both_sides_then_queries_the_new_pose(tmp_path):
    m, b, state, nodes, _ = scene("DD-VT-TWO-SHEETS")
    s = system(m, b)
    with ph.CollisionWorkspace(m, "host") as work:
        obs = dynamic(s, m, b, state, nodes, work, ph.SemiImplicitSolver())
    (tmp_path / "feature-step.json").write_text(json.dumps(obs, allow_nan=False), encoding="utf-8")


def test_capacity_at_new_pose_cannot_replace_previous_clear_observation():
    m, b, state, nodes, _ = scene("DD-REMOTE-FOLD")
    s = system(m, b, feature_capacity=31)
    # A simple open, planar strip with the same registered material graph is far
    # from its nonlocal triangles. Restore the original fold to require 32 rows.
    unrolled = np.array([[float((n[2] - 100) // 2), float(n[2] % 2), 0.0] for n in nodes])
    with ph.CollisionWorkspace(m, "host") as work:
        previous = s.sample_particles(ph.ParticleState(unrolled, state.velocity), work)
        assert not previous.contacts
        with pytest.raises(ph.SolverFailure, match="capacity"):
            s.sample_particles(state, work)
        assert s.last_sample is previous and np.count_nonzero(previous.nodal_force_n) == 0


def test_parallel_edge_interval_uses_half_measure_and_its_two_different_material_rates():
    from test_deformable_features import fixture

    ids = (7, 13, 19)
    a = np.array([[-1, 0, 0], [1, 0, 0], [0, -1, -0.75]])
    b = np.array([[-0.5, 0, 1 / 32], [1.5, 0, 1 / 32], [0, 1, 0.75]])
    model, paths = fixture([(0, 2, 0, ids, a), (0, 4, 0, ids, b)])
    bindings = {
        model.shapes[0].key: ph.ParticleAttachment((0, 1, 2)),
        model.shapes[1].key: ph.ParticleAttachment((3, 4, 5)),
    }
    xyz = np.vstack((a, b))
    vel = np.zeros_like(xyz)
    vel[:3, 2] = a[:, 0]
    state = ph.ParticleState(xyz, vel)
    s = system(model, bindings, layer=ph.NormalLayer(1 / 16, 128.0, damping_s_per_m=0.5))
    with ph.CollisionWorkspace(model, "host") as work:
        result = s.sample_particles(state, work, features=query(model, paths, work))
    target = next(load for load in result.contacts if key(load.feature) == ("EE", (7, 13), (7, 13)))
    assert target.force_n == 4.5 and target.power_w == -2.25
    assert [site.fraction for site in target.sites] == [0.5, 0.5]
    assert [site.rate_m_per_s for site in target.sites] == [0.5, -1.0]
    assert [site.force_n for site in target.sites] == [1.5, 3.0]
    conservation(result, state)


def test_external_work_and_contact_work_are_observed_separately_on_a_real_step():
    m, b, state, nodes, _ = scene("DD-VT-TWO-SHEETS")
    s = system(m, b)
    h = 1 / 256
    external = ph.StepInputs.zeros(len(nodes))
    external.force[:, 0] = 0.25
    with ph.CollisionWorkspace(m, "host") as work:
        initial = s.sample_particles(state, work)
        total = initial.step_inputs(external)
        advance = ph.SemiImplicitSolver().step(
            ph.ParticleModel(np.ones(len(nodes)), np.zeros(3)), state, total, h
        )
        assert advance.report.accepted
        after = s.sample_particles(advance.state, work)
    dx = advance.state.position - state.position
    external_work = float((external.force * dx).sum())
    material_work = float((initial.nodal_force_n * dx).sum())
    kinetic = float((advance.state.velocity**2).sum() / 2)
    assert external_work == pytest.approx(len(nodes) * 0.25**2 * h * h, abs=1e-14)
    assert material_work == pytest.approx(5.5 * h * h, abs=1e-14)
    assert kinetic == pytest.approx((external_work + material_work) / 2, abs=1e-14)
    assert after.potential_j < initial.potential_j
    total.force.fill(99)
    np.testing.assert_array_equal(initial.nodal_force_n[:, 0], 0.0)
    np.testing.assert_array_equal(external.force[:, 0], 0.25)


def test_reused_local_ids_remain_isolated_across_worlds_and_velocity_is_current():
    from test_deformable_features import fixture

    ids = (7, 13, 19)
    a = np.array([[-1, -1, 0.0], [1, -1, 0.0], [0, 1, 0.0]])
    b = np.array([[0, 0, 3 / 64], [-0.125, 0.125, 0.75], [0.125, 0.125, 0.75]])
    c = b.copy()
    c[0, 2] = 5 / 64
    m, _ = fixture([(0, 1, 0, ids, a), (0, 2, 0, ids, b), (1, 1, 0, ids, a), (1, 2, 0, ids, c)])
    bindings = {
        s.key: ph.ParticleAttachment(tuple(range(3 * i, 3 * i + 3))) for i, s in enumerate(m.shapes)
    }
    xyz = np.vstack((a, b, a, c))
    state = ph.ParticleState(xyz, np.zeros_like(xyz))
    s = system(m, bindings)
    with ph.CollisionWorkspace(m, "host") as work:
        observed = s.sample_particles(state, work)
    assert len(observed.contacts) == 1 and observed.nodal_force_n[3, 2] == 2
    np.testing.assert_array_equal(observed.nodal_force_n[6:], 0)
    conflict = dict(bindings)
    conflict[m.shapes[2].key] = bindings[m.shapes[0].key]
    with pytest.raises(ValueError, match="material identity"):
        system(m, conflict)


def test_graph_neighbors_exclude_unshared_close_materials_only_at_the_chosen_ring():
    from test_deformable_features import fixture

    a = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    b = [a[1], a[0], (0.25, 0.25, 1 / 32)]
    m, _ = fixture([(0, 5, 0, (11, 19, 31), a), (0, 5, 1, (19, 11, 43), b)])
    bindings = {
        m.shapes[0].key: ph.ParticleAttachment((0, 1, 2)),
        m.shapes[1].key: ph.ParticleAttachment((1, 0, 3)),
    }
    xyz = np.array([*a, b[2]])
    state = ph.ParticleState(xyz, np.zeros_like(xyz))
    with ph.CollisionWorkspace(m, "host") as work:
        one = system(m, bindings, topology_rings=1).sample_particles(state, work)
        two = system(m, bindings, topology_rings=2).sample_particles(state, work)
    assert any(key(f.feature) == ("VT", (43,), (11, 19, 31)) for f in one.contacts)
    assert one.nodal_force_n[3, 2] > 0 and not two.contacts and np.all(two.nodal_force_n == 0)
    conservation(one, state)


def advance_fold(state, nodes, loads, system_, work, solver):
    h = 1 / 1024
    dtype = state.position.dtype
    model = ph.ParticleModel(np.ones(len(nodes), dtype=dtype), np.zeros(3, dtype=dtype))
    ext = ph.StepInputs.zeros(len(nodes), dtype=dtype)
    advanced = solver.step(model, state, loads.step_inputs(ext), h)
    assert advanced.report.accepted
    again = system_.sample_particles(advanced.state, work)
    lower = [i for i, n in enumerate(nodes) if 100 <= n[2] <= 105]
    upper = [i for i, n in enumerate(nodes) if 112 <= n[2] <= 117]
    linear = advanced.state.velocity.astype(float) - state.velocity
    impulse_low = linear[lower].sum(0)
    impulse_high = linear[upper].sum(0)
    np.testing.assert_allclose(impulse_low, [0, 0, -64 * h], rtol=0, atol=3e-8)
    np.testing.assert_allclose(impulse_high, [0, 0, 64 * h], rtol=0, atol=3e-8)
    dx = advanced.state.position.astype(float) - state.position
    sep = dx[upper, 2].mean() - dx[lower, 2].mean()
    assert sep == pytest.approx(2 * 64 * h * h / 6, abs=1e-8)
    assert again.potential_j < loads.potential_j and again.contacts
    return dict(
        backend=advanced.report.backend,
        dt_s=h,
        lower_impulse_ns=impulse_low.tolist(),
        upper_impulse_ns=impulse_high.tolist(),
        mean_separation_increase_m=float(sep),
        entry_stored_j=loads.potential_j,
        end_stored_j=again.potential_j,
        kinetic_j=float((advanced.state.velocity.astype(float) ** 2).sum() / 2),
        external_work_j=float((dx * ext.force).sum()),
        end_features=len(again.contacts),
        end_node_forces_n=again.nodal_force_n.tolist(),
        final_position_m=advanced.state.position.tolist(),
        final_velocity_m_s=advanced.state.velocity.tolist(),
    )


def test_sixty_four_newtons_advance_both_sides_of_the_fold(tmp_path):
    m, b, state, nodes, _ = scene("DD-REMOTE-FOLD")
    s = system(m, b)
    with ph.CollisionWorkspace(m, "host") as work:
        initial = s.sample_particles(state, work)
        obs = advance_fold(state, nodes, initial, s, work, ph.SemiImplicitSolver())
    (tmp_path / "actual-fold-step.json").write_text(
        json.dumps(obs, allow_nan=False), encoding="utf-8"
    )

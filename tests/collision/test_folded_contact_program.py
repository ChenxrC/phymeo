"""Repeated full-cloth input, geometric acquisition/release and simultaneous shared support nodes."""

import json
from contextlib import ExitStack

import numpy as np
import pytest
from test_folded_site_selection import setup_mesh

import phymeo as ph


def world(*, two=False, dtype=np.float64, operator=None, shift=False, structure=False):
    from phymeo.solvers.folded_contact_program import FoldedContactProgram

    registration, bind, x, m = setup_mesh(second=True, two_probes=two)
    n = np.array([-1 / 16, 0, 1.0])
    n /= np.linalg.norm(n)
    t = np.array([1.0, 0, 1 / 16])
    t /= np.linalg.norm(t)
    v = np.zeros_like(x)
    top = 8
    origin = np.array([-0.22 + (4 if shift else 0), 0, -0.22 / 16])
    x[top] = origin + (0.03125 if two else 0.084) * n
    v[top] = 0.12 * t if two else -0.32 * n
    if two:
        nr = np.array([1 / 16, 0, 1.0])
        nr /= np.linalg.norm(nr)
        x[9] = np.array([0.19 + (4 if shift else 0), 0, -0.19 / 16]) + 0.042 * nr
        v[9] = [-0.18, 0, 0]
    model = ph.ParticleModel(np.asarray(1 / m, dtype=dtype), np.zeros(3, dtype=dtype))
    law = ph.NormalLayer(1 / 16, 64, friction=0.25)
    from phymeo.constraints.cloth_material import ClothMaterial

    cloth = (
        ClothMaterial(
            ph.DistanceConstraints(np.array([[0, 1]]), np.array([1.75]), np.array([0.25]))
        )
        if structure
        else None
    )
    program = FoldedContactProgram(
        model,
        registration,
        bind,
        layer=law,
        static_friction=0.5,
        tangent_stiffness_n_per_m=16.0,
        layer_operator=operator,
        material=cloth,
    )
    state = program.initial_state(ph.ParticleState(x.astype(dtype), v.astype(dtype)))
    force = ph.StepInputs.zeros(len(m), dtype=dtype)
    if two:
        force.force[8] = 0.25 * t - 1.7 * n
        force.force[9] = [-0.25, 0, -1.1]
    else:
        force.force[top] = -0.4 * n + 0.02 * t
    return program, state, force, m


def independent_gap(x):
    a = x[0]
    tri = x[[2, 0, 1]]
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    normal /= np.linalg.norm(normal)
    return float((x[8] - a) @ normal)


def test_fixed_external_load_discovers_holds_releases_then_rebuilds(tmp_path):
    op, state, load, m = world()
    h = 1 / 32
    start = state
    events = []
    frames = []
    free_steps = 0
    for i in range(96):
        before = state
        out = op.step(state, load, h)
        assert out.report.accepted and out.selection.eligible == 2
        state = out.state
        change = (state.particles.velocity - before.particles.velocity) * m[:, None]
        np.testing.assert_allclose(
            change, h * load.force + out.nodal_contact_impulse_ns, atol=2e-10, rtol=0
        )
        np.testing.assert_allclose(
            change.sum(axis=0), h * load.force.sum(axis=0), atol=2e-10, rtol=0
        )
        x0 = before.particles.position
        x1 = state.particles.position
        for ev in out.events:
            assert ev.key.point == (3, 70, 101) and ev.key.body == (3, 10)
            xx = x0 + ev.fraction * (x1 - x0)
            assert independent_gap(xx) == pytest.approx(1 / 16, abs=2e-9)
            events.append(
                dict(
                    kind=ev.kind,
                    step=i,
                    t=ev.fraction,
                    physical_time=(i + ev.fraction) * h,
                    episode=(ev.history.start_tick, ev.history.start_fraction),
                    stored=ev.history.stored_energy_j,
                )
            )
        if not out.contacts:
            free_steps += 1
            assert np.array_equal(out.nodal_contact_impulse_ns, np.zeros_like(x0))
        for contact in out.contacts:
            assert contact.normal_force_n > 0 and contact.history.key == contact.key
            for leg in contact.legs:
                assert leg.start.spatial and leg.end.spatial
                assert (
                    leg.start.normal_gap_m <= 1 / 16 + 2e-9
                    and leg.end.normal_gap_m <= 1 / 16 + 2e-9
                )
                np.testing.assert_allclose(np.sum(leg.nodal_impulses_ns, axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(
            np.cross(x1, m[:, None] * state.particles.velocity).sum(axis=0)
            - np.cross(x0, m[:, None] * before.particles.velocity).sum(axis=0)
            - h * np.cross(x0, load.force).sum(axis=0),
            out.discrete_angular_impulse_nms,
            atol=2e-11,
        )
        frames.append(
            dict(
                tick=state.tick,
                eligible=out.selection.eligible,
                contact=[
                    dict(
                        phase=c.phase,
                        normal_n=c.normal_force_n,
                        shear_n=c.shear_n,
                        t0=c.t0,
                        t1=c.t1,
                        material=c.legs[0].start.material_weights,
                        spatial=[s.distance_m for s in c.legs[0].start.spatial],
                        work_j=c.tangent_work_j,
                        episode_start=c.history.start_tick,
                    )
                    for c in out.contacts
                ],
                impulse=out.nodal_contact_impulse_ns.tolist(),
                angle_nms=out.discrete_angular_impulse_nms,
                x=x1.tolist(),
                v=state.particles.velocity.tolist(),
            )
        )
    enters = [row for row in events if row["kind"] == "enter"]
    exits = [row for row in events if row["kind"] == "release"]
    assert (
        len(enters) >= 2
        and exits[0]["physical_time"] < enters[1]["physical_time"]
        and free_steps >= 2
    )
    assert enters[0]["episode"] != enters[1]["episode"]
    assert any(c["phase"] == "continued" for f in frames for c in f["contact"])
    (tmp_path / "continuous-folded-sites.json").write_text(
        json.dumps(
            dict(
                dt=h,
                external=load.force.tolist(),
                mass=m.tolist(),
                start_x=start.particles.position.tolist(),
                start_v=start.particles.velocity.tolist(),
                events=events,
                free_steps=free_steps,
                frames=frames,
            ),
            allow_nan=False,
        ),
        encoding="utf8",
    )


def test_two_probes_share_live_material_nodes_and_one_global_particle_update(tmp_path):
    from test_folded_material_seam import reference_average

    op, before, F, m = world(two=True, structure=True)
    initial = before.particles
    h = 1 / 32
    out = op.step(before, F, h)
    assert out.report.accepted and len(out.contacts) == 2 and out.selection.eligible == 4
    assert {a.key.point for a in out.contacts} == {(3, 70, 101), (3, 71, 101)}
    assert {a.key.body for a in out.contacts} == {(3, 10)}
    assert {a.phase for a in out.contacts} == {"entered"}
    np.testing.assert_allclose(out.structural_force_n[0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(out.structural_force_n[1], [0, -1, 0], atol=1e-12)
    expected = np.zeros_like(F.force)
    for load in out.contacts:
        chart = next(s.chart for s in out.selection.sites if s.key == load.key)
        rows = list(load.rows)
        intervals = [(leg.t0, leg.t1, chart.keys.index(leg.face)) for leg in load.legs]
        ref = reference_average(
            initial.position[rows], out.state.particles.position[rows], intervals
        )
        local = sum(
            h * np.tensordot([load.shear_n, load.normal_force_n], j[0], axes=1) for j in ref
        )
        expected[rows] += local
        np.testing.assert_allclose(
            sum(q.nodal_impulses_ns for q in load.legs), local, atol=2e-8, rtol=0
        )
    np.testing.assert_allclose(out.nodal_contact_impulse_ns, expected, atol=3e-8, rtol=0)
    measured = m[:, None] * (out.state.particles.velocity - initial.velocity)
    np.testing.assert_allclose(
        measured, expected + h * (F.force + out.structural_force_n), atol=3e-8, rtol=0
    )
    np.testing.assert_allclose(out.integrated_angular_impulse_nms, 0, atol=2e-12)
    assert len(out.state.active) == 2

    class Reject:
        def step(self, model, state, inputs, dt):
            from phymeo.core.result import StepReport, StepResult

            return StepResult(state.copy(), StepReport(False, reason="deliberately rejected"))

    rejected = op.step(out.state, F, h, particle_solver=Reject())
    assert not rejected.report.accepted and rejected.state is out.state and not rejected.events
    with pytest.raises(ph.SolverFailure, match="capacity"):
        op.step(out.state, F, h, capacity=1)
    resumed = op.step(rejected.state, F, h)
    replay = op.step(out.state, F, h)
    assert {a.phase for a in resumed.contacts} == {"continued"} and not resumed.events
    assert resumed.state.active == replay.state.active and all(
        a.active_steps == 2 for a in resumed.state.active
    )
    np.testing.assert_array_equal(resumed.state.particles.position, replay.state.particles.position)
    # Identical labels, different initial geometry. Both probes select the second material object.
    alternate, st, ef, _ = world(two=True, shift=True)
    assert {a.key.body for a in alternate.step(st, ef, h).contacts} == {(3, 11)}
    (tmp_path / "coupled-fold-sites.json").write_text(
        json.dumps(
            dict(
                ids=[c.key.point for c in out.contacts],
                folds=[c.key.body for c in out.contacts],
                normal=[c.normal_force_n for c in out.contacts],
                impulse=out.nodal_contact_impulse_ns.tolist(),
                shared_structural_n=out.structural_force_n.tolist(),
                actual_velocity=out.state.particles.velocity.tolist(),
                residual_n=out.force_residual_n,
                iterations=out.iterations,
                active_counts=[s.active_steps for s in resumed.state.active],
                angular_entry=out.discrete_angular_impulse_nms,
                angular_path=out.integrated_angular_impulse_nms,
            ),
            allow_nan=False,
        ),
        encoding="utf8",
    )


def test_persistent_one_site_matches_the_original_loaded_fold_solver():
    from phymeo.solvers.folded_material_seam import FoldedMaterialSeamStepper

    op, s, F, m = world(two=False)
    x = s.particles
    x.position[8, 2] = 0.025
    current = op.initial_state(x)
    mesh, _, _, _ = setup_mesh()
    faces = ph.CollisionModel(tuple(f for f in mesh.shapes if f.key.body == 10))
    # Standard isolated stepper uses exactly this point and these material nodes.
    local = [8, 0, 1, 2, 3]
    mass = ph.ParticleModel(1 / m[local], np.zeros(3))
    single = FoldedMaterialSeamStepper(
        mass,
        faces,
        point_row=0,
        point_material=(3, 70, 101),
        material_rows={17: 1, 31: 2, 7: 3, 49: 4},
        layer=ph.NormalLayer(1 / 16, 64, friction=0.25),
        static_friction=0.5,
        tangent_stiffness_n_per_m=16,
    )
    solo = single.initial_state(ph.ParticleState(x.position[local], x.velocity[local]))
    for _ in range(2):
        a = op.step(current, F, 1 / 64)
        b = single.step(solo, ph.StepInputs(F.force[local]), 1 / 64)
        np.testing.assert_allclose(
            a.state.particles.position[local], b.state.particles.position, atol=3e-10, rtol=0
        )
        np.testing.assert_allclose(
            a.state.particles.velocity[local], b.state.particles.velocity, atol=3e-10, rtol=0
        )
        assert a.contacts[0].normal_force_n == pytest.approx(b.normal_force_n, abs=2e-10)
        assert a.contacts[0].history.stored_energy_j == pytest.approx(
            b.state.history.stored_energy_j, abs=2e-10
        )
        current, solo = a.state, b.state


def test_same_externally_driven_window_reports_step_sensitivity_and_angular_work(tmp_path):
    samples = []
    for count in (14, 28):
        op, state, F, m = world()
        h = 0.7 / count
        start = state.particles
        impulse = np.zeros(3)
        entry_moment = np.zeros(3)
        path_moment = np.zeros(3)
        internal = np.zeros(3)
        integrated = np.zeros(3)
        events = []
        for i in range(count):
            old = state.particles
            a = op.step(state, F, h)
            state = a.state
            current = state.particles
            impulse += a.nodal_contact_impulse_ns[8]
            entry_moment += h * np.cross(old.position, F.force).sum(axis=0)
            path_moment += h * np.cross((old.position + current.position) / 2, F.force).sum(axis=0)
            internal += a.discrete_angular_impulse_nms
            integrated += a.integrated_angular_impulse_nms
            events.extend((e.kind, h * (i + e.fraction)) for e in a.events)
        last = state.particles
        dL = (
            np.cross(last.position, m[:, None] * last.velocity)
            - np.cross(start.position, m[:, None] * start.velocity)
        ).sum(axis=0)
        np.testing.assert_allclose(dL - entry_moment, internal, atol=2e-10)
        np.testing.assert_allclose(integrated, 0, atol=2e-10)
        assert [e[0] for e in events][:2] == ["enter", "release"]
        samples.append(
            dict(
                steps=count,
                h_s=h,
                events=events,
                point_impulse_ns=impulse.tolist(),
                end_position=last.position[8].tolist(),
                end_velocity=last.velocity[8].tolist(),
                actual_dL_nms=dL.tolist(),
                entry_external_nms=entry_moment.tolist(),
                drift_external_nms=path_moment.tolist(),
                entry_internal_nms=internal.tolist(),
                drift_internal_nms=integrated.tolist(),
            )
        )
    (tmp_path / "fold-sites-time-resolution.json").write_text(
        json.dumps(samples, allow_nan=False), encoding="utf8"
    )


@pytest.mark.tyx
def test_two_normal_rows_and_one_global_particle_step_use_the_real_device(tyx_mode, tmp_path):
    with ExitStack() as stack:
        layer = stack.enter_context(ph.TyxNormalLayer(tyx_mode, cache_root=tmp_path / "cache"))
        solver = stack.enter_context(
            ph.TyxSemiImplicitSolver(tyx_mode, cache_root=tmp_path / "cache")
        )
        op, old, F, m = world(two=True, dtype=np.float32, operator=layer, structure=True)
        if tyx_mode == "cuda":
            import tyx

            baseline = tyx.runtime_stats().cuda_kernel_launches
        a = op.step(old, F, 1 / 32, particle_solver=solver)
        launches = (
            None if tyx_mode != "cuda" else tyx.runtime_stats().cuda_kernel_launches - baseline
        )
        assert (
            a.report.accepted
            and a.report.backend == "tyx:" + tyx_mode
            and a.normal_backend == tyx_mode
        )
        assert (
            len(a.contacts) == 2
            and layer.stats()["completed_launches"] == 1
            and layer.stats()["sites"] == 2
        )
        if launches is not None:
            assert launches == 2 and layer.stats()["downloads"] == 1
        dv = a.state.particles.velocity - old.particles.velocity
        np.testing.assert_allclose(
            m[:, None] * dv,
            (F.force + a.structural_force_n) / 32 + a.nodal_contact_impulse_ns,
            atol=2e-6,
            rtol=0,
        )
        (tmp_path / "joint-fold-device.json").write_text(
            json.dumps(
                dict(
                    backend=a.report.backend,
                    kernels=launches,
                    layer=layer.stats(),
                    normals=[c.normal_force_n for c in a.contacts],
                    all_node_impulses=a.nodal_contact_impulse_ns.tolist(),
                    actual_state=a.state.particles.position.tolist(),
                    residual_n=a.force_residual_n,
                ),
                allow_nan=False,
            ),
            encoding="utf8",
        )

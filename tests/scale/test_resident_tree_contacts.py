"""C6 frozen entry-contact feedback cases; tests-first authoring, UNEXECUTED.

Current-pose contact is sampled once from accepted device geometry. Independent
oracles below never use product dynamics, FK, or contact output as their truth.
Execution and all backend/static checks belong to the parent serial owner.
"""

import gzip
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from itertools import product

import numpy as np
import pytest

import phymeo as ph
from phymeo.backends import tyx_world_batch as resident
from phymeo.geometry.contact_patch_adapter import contact_patch_sample
from phymeo.solvers.contact_patch_loads import sample_patch_loads

H = 1 / 1024
RTOL = 5e-5
ATOL = 5e-6
WINDOW_ATOL = 2e-4


def _json(value):
    if is_dataclass(value):
        return {field.name: _json(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, (tuple, list)):
        return [_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": repr(value)}
    return value


def _save(path, name, value):
    with gzip.open(path / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(_json(value), stream, indent=2, allow_nan=False)


def _box(lo, hi):
    return np.asarray(list(product(*zip(lo, hi, strict=True))), dtype=float)


def _close(actual, expected, *, atol=ATOL):
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=RTOL)


def _program(model, *, q=None, v=None, kp=None, kd=None, delay=0):
    n = model.dof_count
    program = ph.DrivenRobotProgram(
        model,
        kp=np.zeros(n) if kp is None else np.asarray(kp, dtype=float),
        kd=np.zeros(n) if kd is None else np.asarray(kd, dtype=float),
        effort_limit=np.full(n, 8.0),
        delay_steps=delay,
    )
    initial = program.initial_state(
        ph.RobotState(
            np.zeros(n) if q is None else np.asarray(q, dtype=float),
            np.zeros(n) if v is None else np.asarray(v, dtype=float),
        )
    )
    return program, resident.FixedRobotTree(program, initial, control_max_iterations=512)


def _action(program, *, target=None, rate=None, ff=None, effort=None, force=None, torque=None):
    n = program.model.dof_count

    def vector(value):
        return np.zeros(n) if value is None else np.asarray(value, dtype=float)

    def field(value):
        return np.zeros((n, 3)) if value is None else np.asarray(value, dtype=float)

    return program.inputs(
        vector(target), vector(rate), feedforward=vector(ff),
        loads=ph.RobotInputs(vector(effort), field(force), field(torque)),
    )


def _slider(*, origin=(0, 0, 3 / 32), q=0.0, v=0.0, kp=0.0, kd=0.0, delay=0):
    model = ph.RobotModel(
        np.array([-1]), ("prismatic",), np.array([origin], dtype=float),
        np.array([[0.0, 0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, 1.0]]),
        np.array([2.0]), np.zeros((1, 3)), np.full((1, 3), 1 / 8), np.zeros(3),
    )
    return _program(model, q=[q], v=[v], kp=[kp], kd=[kd], delay=delay)


def _r01_geometry(batch, world=17):
    moving, fixed = ph.CollisionKey(world, 100, 11), ph.CollisionKey(world, 900, 1)
    local = _box([-1 / 8] * 3, [1 / 8] * 3)
    support = _box([-1 / 2, -1 / 2, -1 / 4], [1 / 2, 1 / 2, 0])
    first = ph.CollisionShape(moving, "hull", tuple(range(101, 109)))
    second = ph.CollisionShape(fixed, "hull", tuple(range(201, 209)))
    model = ph.CollisionModel((first, second))
    topology = batch.snapshot([world])[world].topology
    bindings = (
        ph.ResidentTreeBinding(moving, world, 0, local, topology=topology),
        ph.ResidentStaticBinding(fixed, ph.CollisionMotion(support)),
    )
    return model, bindings, local, support


@pytest.mark.tyx
def test_r01_public_contact_step_follows_real_host_patch_two_newtons(tyx_mode, tmp_path):
    """Valid old geometry/law/tree first; RED is the missing public contact entry."""
    program, world = _slider()
    law = ph.RobotContactMaterial(stiffness=64, damping=0, friction=0)
    command = _action(program)
    with resident.TyxWorldBatch([(17, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({17: command})
        model, bindings, local, support = _r01_geometry(batch)
        # This old HOST path is only a precondition, not the dynamics oracle.
        patch = contact_patch_sample(
            model.shapes[0], model.shapes[1], local + [0, 0, 3 / 32], support,
        )
        assert patch is not None and patch.dimension == 2
        assert abs(sum(site.weight for site in patch.sites) - 1) <= 1e-12
        sampled = sample_patch_loads(
            patch, law, np.zeros((len(patch.sites), 3)), np.zeros((len(patch.sites), 3)),
        )
        np.testing.assert_allclose(sampled["first_force"].sum(axis=0), [0, 0, 2], atol=1e-12, rtol=0)
        np.testing.assert_allclose(sampled["second_force"].sum(axis=0), [0, 0, -2], atol=1e-12, rtol=0)
        before = batch.snapshot()[17]
        _save(tmp_path, "r01-public-input", dict(
            h_s=H, model=program.model, command=command, collision_model=model,
            local_material=local, static_material=support, law=law,
            host_patch=patch, host_sampled=sampled, before=before, mode=tyx_mode,
        ))
        with batch.prepare_contacts(model, bindings, {(model.shapes[0].key, model.shapes[1].key): law}) as plan:
            assert batch.step_contacts(plan, H) == {17: True}
            after = batch.snapshot()[17]
            receipt = plan.last_accepted[17]
            assert plan.last_status[17] == "ACCEPTED"
            assert before.tick == 0 and after.tick == 1
            # All input values and this single response are exactly dyadic FP32.
            assert after.velocity == (H,) and after.position == (H * H,)
            assert after.applied_effort == (0.0,) and after.generalized_applied == (2.0,)
            assert after.control == before.control and after.history == before.history == ()
            assert (receipt.world, receipt.entry_tick, receipt.exit_tick, receipt.dt_s) == (17, 0, 1, H)
            assert receipt.entry_state == before and receipt.accepted_state == after
            _close(receipt.law_wrench_fp64, [[0, 0, 2, 0, 0, 0]])
            _close(receipt.uploaded_wrench_fp32, [[0, 0, 2, 0, 0, 0]])
            _close(receipt.contact_generalized, [2])
            _close(receipt.actual_drive, [0])
            _save(tmp_path, "r01-public-result", dict(
                before=before, after=after, receipt=receipt,
                diagnostic=batch.tree_diagnostics(), geometry=batch.tree_kinematics(), stats=batch.stats(),
            ))


def _rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _law(damping=0):
    return ph.RobotContactMaterial(stiffness=64, damping=damping, friction=0)


def _bind(batch, world, local, support, *, group=0, moving_kind="hull", static_kind="hull", radius=0, static_radius=0, reverse=False):
    """Keep original finite material coordinates/IDs; keys are deliberately not row IDs."""
    moving = ph.CollisionKey(world, 900 if reverse else 100, 11)
    fixed = ph.CollisionKey(world, 100 if reverse else 900, 1)
    local, support = np.asarray(local, dtype=float), np.asarray(support, dtype=float)
    a = ph.CollisionShape(moving, moving_kind, tuple(range(101, 101 + len(local))), radius=radius)
    b = ph.CollisionShape(fixed, static_kind, tuple(range(201, 201 + len(support))), radius=static_radius)
    model = ph.CollisionModel((a, b))
    bindings = (
        ph.ResidentTreeBinding(moving, world, group, local, topology=batch.snapshot([world])[world].topology),
        ph.ResidentStaticBinding(fixed, ph.CollisionMotion(support)),
    )
    sources = {moving: (a, group, local.copy()), fixed: (b, None, support.copy())}
    return model, bindings, sources, tuple(sorted((moving, fixed)))


def _r01_bound(batch, world=17, *, reverse=False):
    return _bind(batch, world, _box([-1 / 8] * 3, [1 / 8] * 3), _box([-1 / 2, -1 / 2, -1 / 4], [1 / 2, 1 / 2, 0]), reverse=reverse)


def _audit_receipt(receipt, diagnostic, sources):
    """Audit the public same-attempt record, separately from physical trajectory oracles.

    The 1e-10 m bound concerns construction from the *stored* FP32 frames widened
    to FP64. It is not a tighter device-physics accuracy claim. No FK regeneration
    or orthogonalization is used to transport entry material/application points.
    """
    assert isinstance(receipt.sites, tuple)
    assert receipt.entry_tick == receipt.entry_state.tick
    assert receipt.exit_tick == receipt.accepted_state.tick == receipt.entry_tick + 1
    assert receipt.entry_geometry.tick == receipt.entry_tick
    assert receipt.accepted_geometry.tick == receipt.exit_tick
    np.testing.assert_array_equal(receipt.entry_geometry.position, receipt.entry_state.position)
    np.testing.assert_array_equal(receipt.accepted_geometry.position, receipt.accepted_state.position)
    n = len(receipt.entry_state.position)
    wrench = np.zeros((n, 6))
    static = {}
    work, material_work = 0.0, 0.0
    totals = {}
    for item in receipt.sites:
        assert item.pair == tuple(sorted(item.pair)) and len(item.pair) == 2
        site, witness = item.geometry, item.geometry.witness
        normal = np.asarray(witness.normal, dtype=float)
        load = np.asarray(site.load_point, dtype=float)
        np.testing.assert_allclose(np.linalg.norm(normal), 1, atol=1e-12, rtol=0)
        np.testing.assert_allclose(load, (np.asarray(witness.first_point) + witness.second_point) / 2, atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.dot(np.asarray(witness.second_point) - witness.first_point, normal), witness.gap_m, atol=1e-12, rtol=0)
        force_pair = [np.asarray(item.first_force_n), np.asarray(item.second_force_n)]
        np.testing.assert_allclose(force_pair[0] + force_pair[1], 0, atol=1e-12, rtol=0)
        vn = np.dot(np.asarray(item.first_velocity_m_s) - item.second_velocity_m_s, -normal)
        scalar = max(0.0, -item.law.stiffness * witness.gap_m - item.law.damping * vn) if witness.gap_m < 0 else 0.0
        np.testing.assert_allclose(force_pair[0], -normal * scalar * site.weight, atol=1e-10, rtol=1e-10)
        actual_work, actual_material_work = 0.0, 0.0
        for index, side in enumerate(("first", "second")):
            key = item.pair[index]
            shape, group, original = sources[key]
            point = np.asarray(getattr(witness, side + "_point"), dtype=float)
            weights = getattr(witness, side + "_weights")
            offset = getattr(witness, side + "_offset")
            by_id = dict(zip(shape.vertex_ids, original, strict=True))
            assert weights and set(i for i, _ in weights) <= set(by_id)
            np.testing.assert_allclose(sum(w for _, w in weights), 1, atol=1e-12, rtol=0)
            local = sum((w * by_id[i] for i, w in weights), np.zeros(3))
            if group is None:
                rebuilt, reference = local, original.mean(axis=0)
                velocity, displacement, application = np.zeros(3), np.zeros(3), np.zeros(3)
            else:
                entry, end = receipt.entry_geometry, receipt.accepted_geometry
                p0, p1 = np.asarray(entry.link_position_m[group], float), np.asarray(end.link_position_m[group], float)
                r0, r1 = np.asarray(entry.link_rotation[group], float), np.asarray(end.link_rotation[group], float)
                rebuilt = p0 + r0 @ local
                reference = np.asarray(entry.com_position_m[group], float)
                qd = np.asarray(receipt.entry_state.velocity)
                angular = np.asarray(diagnostic.angular_jacobian[group], float) @ qd
                velocity = np.asarray(diagnostic.linear_jacobian[group], float) @ qd + np.cross(angular, load - reference)
                displacement = p1 + r1 @ np.linalg.solve(r0, point - p0) - point
                application = p1 + r1 @ np.linalg.solve(r0, load - p0) - load
            if offset is not None:
                rebuilt = rebuilt + offset
            np.testing.assert_allclose(rebuilt, point, atol=1e-10, rtol=0)
            np.testing.assert_allclose(getattr(item, side + "_reference_point_m"), reference, atol=1e-10, rtol=0)
            _close(getattr(item, side + "_velocity_m_s"), velocity)
            np.testing.assert_allclose(getattr(item, side + "_material_displacement_m"), displacement, atol=1e-10, rtol=0)
            np.testing.assert_allclose(getattr(item, side + "_application_displacement_m"), application, atol=1e-10, rtol=0)
            force = force_pair[index]
            actual_work += float(force @ application)
            actual_material_work += float(force @ displacement)
            contribution = np.r_[force, np.cross(load - reference, force)]
            if group is None:
                static.setdefault(key, np.zeros(6))
                static[key] += contribution
            else:
                wrench[group] += contribution
            totals.setdefault(key, np.zeros(3))
            totals[key] += force
        np.testing.assert_allclose(item.application_work_j, actual_work, atol=1e-12, rtol=1e-10)
        np.testing.assert_allclose(item.material_work_j, actual_material_work, atol=1e-12, rtol=1e-10)
        work += actual_work
        material_work += actual_material_work
    _close(receipt.law_wrench_fp64, wrench)
    np.testing.assert_array_equal(receipt.uploaded_wrench_fp32, np.asarray(receipt.law_wrench_fp64, np.float32))
    projected = np.einsum("bij,bi->j", np.asarray(diagnostic.linear_jacobian), wrench[:, :3]) + np.einsum("bij,bi->j", np.asarray(diagnostic.angular_jacobian), wrench[:, 3:])
    _close(receipt.contact_generalized, projected)
    _close(diagnostic.external, np.asarray(receipt.input_external) + receipt.contact_generalized)
    _close(receipt.actual_drive, receipt.accepted_state.applied_effort)
    assert set(static) <= set(receipt.static_reactions)
    for key in receipt.static_reactions:
        assert key in sources and sources[key][1] is None
        value = static.get(key, np.zeros(6))
        reaction = receipt.static_reactions[key]
        np.testing.assert_allclose(reaction.reference_point_m, sources[key][2].mean(axis=0), atol=1e-12, rtol=0)
        _close(reaction.force_n, value[:3])
        _close(reaction.torque_nm, value[3:])
    return dict(application_work_j=work, material_work_j=material_work, force_by_key=totals)


@pytest.mark.tyx
@pytest.mark.parametrize("velocity,external,normal,acceleration", ((1 / 8, 0, 3 / 2, 3 / 4), (-1 / 8, 0, 5 / 2, 5 / 4), (1 / 8, 2, 3 / 2, 7 / 4)))
def test_r01_damping_uses_entry_velocity_before_external_load(velocity, external, normal, acceleration, tyx_mode, tmp_path):
    program, world = _slider(v=velocity)
    command, law = _action(program, force=[[0, 0, external]]), _law(4)
    with resident.TyxWorldBatch([(17, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({17: command})
        model, bindings, sources, pair = _r01_bound(batch)
        before = batch.snapshot()
        with batch.prepare_contacts(model, bindings, {pair: law}) as plan:
            assert batch.step_contacts(plan, H) == {17: True}
            after, diagnostic, receipt = batch.snapshot()[17], batch.tree_diagnostics()[17], plan.last_accepted[17]
            _close(after.velocity, [velocity + H * acceleration])
            _close(after.position, [H * (velocity + H * acceleration)])
            _close(receipt.contact_generalized, [normal])
            _close(receipt.input_external, [external])
            _close(diagnostic.acceleration, [acceleration])
            assert after.control == before[17].control
            audit = _audit_receipt(receipt, diagnostic, sources)
            _save(tmp_path, "r01-damping", dict(mode=tyx_mode, h_s=H, model=program.model, command=command, sources=sources, law=law, before=before, after=after, receipt=receipt, diagnostic=diagnostic, audit=audit, stats=batch.stats()))


@pytest.mark.tyx
def test_r01_contact_enters_the_same_bounded_pd_solve(tyx_mode, tmp_path):
    program, world = _slider(kp=4, kd=1)
    command = _action(program)
    expected_v = 2 * H / (2 + H + 4 * H * H)
    expected_drive = -(1 + 4 * H) * expected_v
    with resident.TyxWorldBatch([(17, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({17: command})
        model, bindings, sources, pair = _r01_bound(batch)
        with batch.prepare_contacts(model, bindings, {pair: _law()}) as plan:
            assert batch.step_contacts(plan, H) == {17: True}
            receipt, diagnostic = plan.last_accepted[17], batch.tree_diagnostics()[17]
            _close(receipt.accepted_state.velocity, [expected_v])
            _close(receipt.accepted_state.position, [H * expected_v])
            _close(receipt.actual_drive, [expected_drive])
            _close(diagnostic.acceleration, [expected_v / H])
            _close(receipt.selected_drive, [[0, 0, 0]])
            _audit_receipt(receipt, diagnostic, sources)
            _save(tmp_path, "r01-pd", dict(mode=tyx_mode, h_s=H, model=program.model, command=command, receipt=receipt, diagnostic=diagnostic, stats=batch.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("delay", (0, 1))
def test_r01_only_drive_triplet_is_delayed_contact_and_com_force_are_current(delay, tyx_mode, tmp_path):
    program, world = _slider(delay=delay)
    q, v, old_ff, rows = 0.0, 0.0, 0.0, []
    with resident.TyxWorldBatch([(17, world)], tyx_mode, cache_root=tmp_path) as batch:
        model, bindings, sources, pair = _r01_bound(batch)
        with batch.prepare_contacts(model, bindings, {pair: _law()}) as plan:
            for ff, external in ((1.0, 2.0), (-1.0, -2.0)):
                command = _action(program, ff=[ff], force=[[0, 0, external]])
                batch.set_controls({17: command})
                before = batch.snapshot()[17]
                normal = 64 * max(0.0, 1 / 32 - q)
                selected = old_ff if delay else ff
                v += H * (normal + external + selected) / 2
                q += H * v
                assert batch.step_contacts(plan, H) == {17: True}
                receipt, diagnostic = plan.last_accepted[17], batch.tree_diagnostics()[17]
                _close(receipt.accepted_state.position, [q])
                _close(receipt.accepted_state.velocity, [v])
                _close(receipt.input_external, [external])
                _close(receipt.contact_generalized, [normal])
                _close(receipt.actual_drive, [selected])
                _close(receipt.selected_drive, [[0, 0, selected]])
                assert receipt.accepted_state.control == before.control
                assert receipt.accepted_state.history == (((0.0, 0.0, ff),) if delay else ())
                _audit_receipt(receipt, diagnostic, sources)
                rows.append(dict(command=command, before=before, receipt=receipt, diagnostic=diagnostic))
                old_ff = ff
            _save(tmp_path, "r01-delay", dict(mode=tyx_mode, h_s=H, delay=delay, model=program.model, sources=sources, rows=rows, stats=batch.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("kind", ("sphere-sphere", "capsule-axis", "capsule-end", "capsule-beyond", "rotated-box-frustum", "finite-box-miss"))
def test_r01_actual_finite_shape_variants(kind, tyx_mode, tmp_path):
    origin = (1, 0, 3 / 32) if kind in ("capsule-beyond", "finite-box-miss") else ((1 / 4, 0, 3 / 32) if kind == "capsule-end" else (0, 0, 3 / 32))
    program, world = _slider(origin=origin)
    if kind == "rotated-box-frustum":
        original = program.model
        model = ph.RobotModel(original.parents, original.joint_types, original.joint_position, np.array([[0, 0, np.sin(0.37 / 2), np.cos(0.37 / 2)]]), original.joint_axis, original.mass, original.com, original.inertia_diagonal, original.gravity)
        program, world = _program(model)
    with resident.TyxWorldBatch([(17, world)], tyx_mode, cache_root=tmp_path) as batch:
        command = _action(program)
        batch.set_controls({17: command})
        if kind == "sphere-sphere":
            built = _bind(batch, 17, [[0, 0, 0]], [[0, 0, -1 / 8]], moving_kind="sphere", static_kind="sphere", radius=1 / 8, static_radius=1 / 8)
        elif kind.startswith("capsule"):
            built = _bind(batch, 17, [[0, 0, 0]], [[-1 / 4, 0, -1 / 8], [1 / 4, 0, -1 / 8]], moving_kind="sphere", static_kind="capsule", radius=1 / 8, static_radius=1 / 8)
        elif kind == "rotated-box-frustum":
            frustum = np.array([[x, y, z] for z, half in ((-1 / 4, 3 / 4), (0, 1 / 2)) for x, y in product((-half, half), repeat=2)])
            built = _bind(batch, 17, _box([-1 / 8] * 3, [1 / 8] * 3), frustum)
        else:
            built = _r01_bound(batch)
        model, bindings, sources, pair = built
        expected_n = 0.0 if kind in ("capsule-beyond", "finite-box-miss") else 2.0
        with batch.prepare_contacts(model, bindings, {pair: _law()}) as plan:
            assert batch.step_contacts(plan, H) == {17: True}
            receipt, diagnostic = plan.last_accepted[17], batch.tree_diagnostics()[17]
            _close(receipt.contact_generalized, [expected_n])
            _close(receipt.law_wrench_fp64, [[0, 0, expected_n, 0, 0, 0]])
            _close(receipt.accepted_state.velocity, [H * expected_n / 2])
            _close(receipt.accepted_state.position, [H * H * expected_n / 2])
            assert bool(receipt.sites) == bool(expected_n)
            _audit_receipt(receipt, diagnostic, sources)
            if kind == "rotated-box-frustum":
                _close(receipt.entry_geometry.link_rotation[0], _rz(0.37))
                assert len(receipt.sites) >= 3
            _save(tmp_path, "r01-geometry", dict(mode=tyx_mode, kind=kind, h_s=H, model=program.model, command=command, sources=sources, receipt=receipt, diagnostic=diagnostic, stats=batch.stats()))


def _branch(*, reordered=False, transformed=False):
    order = np.array([0, 2, 1] if reordered else [0, 1, 2])
    rotation = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float) if transformed else np.eye(3)
    translation = np.array([1 / 2, -1 / 4, 3 / 4]) if transformed else np.zeros(3)
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    positions[0] = translation
    orientations = np.tile([0.0, 0.0, 0.0, 1.0], (3, 1))
    if transformed:
        orientations[0] = [-0.5, -0.5, -0.5, 0.5]
    model = ph.RobotModel(np.array([-1, 0, 0]), tuple(np.array(["revolute", "prismatic", "prismatic"])[order]), positions[order], orientations[order], np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=float)[order], np.array([2.0, 1.0, 1.0])[order], np.zeros((3, 3)), np.array([[1 / 8, 1 / 8, 1 / 4], [1 / 16, 1 / 16, 1 / 8], [1 / 16, 1 / 16, 1 / 8]])[order], np.zeros(3))
    program, world = _program(model)
    return program, world, order, rotation, translation


def _branch_bound(batch, world, order, rotation, translation):
    support = _box([-1 / 2, -1 / 4, -1 / 2], [1 / 2, 0, 1 / 2]) @ rotation.T + translation
    return _bind(batch, world, [[-1, 7 / 64, 0]], support, group=int(np.flatnonzero(order == 1)[0]), moving_kind="sphere", radius=1 / 8)


def _branch_oracle(q, v):
    theta, slide, other = q
    omega, slide_rate, other_rate = v
    mass = np.array([[5 / 2 + slide * slide + other * other, 1, -1], [1, 1, 0], [-1, 0, 1]], dtype=float)
    bias = np.array([2 * omega * (slide * slide_rate + other * other_rate), -slide * omega * omega, -other * omega * omega])
    cx, cy = -np.sin(theta) * (slide + 7 / 64), np.cos(theta) * (slide + 7 / 64)
    normal = 64 * max(0.0, 1 / 8 - cy)
    generalized = np.array([cx * normal, np.cos(theta) * normal, 0])
    return mass, bias, generalized, np.linalg.solve(mass, generalized - bias)


@pytest.mark.tyx
@pytest.mark.parametrize("reordered,transformed", tuple(product((False, True), repeat=2)))
def test_r02_branch_com_torque_cancels_root_load_but_couples_sibling(reordered, transformed, tyx_mode, tmp_path):
    program, world, order, rotation, translation = _branch(reordered=reordered, transformed=transformed)
    with resident.TyxWorldBatch([(103, world)], tyx_mode, cache_root=tmp_path) as batch:
        command = _action(program)
        batch.set_controls({103: command})
        model, bindings, sources, pair = _branch_bound(batch, 103, order, rotation, translation)
        with batch.prepare_contacts(model, bindings, {pair: _law()}) as plan:
            assert batch.step_contacts(plan, H) == {103: True}
            receipt, diagnostic = plan.last_accepted[103], batch.tree_diagnostics()[103]
            acceleration = np.array([-2.0, 3.0, -2.0])[order]
            mass = np.array([[5 / 2, 1, -1], [1, 1, 0], [-1, 0, 1]])[np.ix_(order, order)]
            _close(diagnostic.mass, mass)
            _close(diagnostic.acceleration, acceleration)
            _close(receipt.contact_generalized, np.array([0, 1, 0])[order])
            _close(receipt.accepted_state.velocity, H * acceleration)
            _close(receipt.accepted_state.position, H * H * acceleration)
            expected = np.zeros((3, 6))
            expected[int(np.flatnonzero(order == 1)[0])] = np.r_[rotation @ [0, 1, 0], rotation @ [0, 0, -1]]
            _close(receipt.law_wrench_fp64, expected)
            audit = _audit_receipt(receipt, diagnostic, sources)
            _save(tmp_path, "r02-first", dict(mode=tyx_mode, h_s=H, model=program.model, command=command, order=order, rotation=rotation, translation=translation, sources=sources, receipt=receipt, diagnostic=diagnostic, audit=audit, stats=batch.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("reordered,transformed", ((False, False), (True, True)))
def test_r02_forty_real_steps_keep_full_cross_branch_mass_and_bias(reordered, transformed, tyx_mode, tmp_path):
    program, world, order, rotation, translation = _branch(reordered=reordered, transformed=transformed)
    q, v, rows = np.zeros(3), np.zeros(3), []
    with resident.TyxWorldBatch([(103, world)], tyx_mode, cache_root=tmp_path) as batch:
        command = _action(program)
        batch.set_controls({103: command})
        model, bindings, sources, pair = _branch_bound(batch, 103, order, rotation, translation)
        with batch.prepare_contacts(model, bindings, {pair: _law()}) as plan:
            try:
                for index in range(40):
                    mass, bias, load, acceleration = _branch_oracle(q, v)
                    kinetic_before = 0.5 * v @ mass @ v
                    v = v + H * acceleration
                    q = q + H * v
                    assert batch.step_contacts(plan, H) == {103: True}
                    receipt, diagnostic = plan.last_accepted[103], batch.tree_diagnostics()[103]
                    audit = _audit_receipt(receipt, diagnostic, sources)
                    next_mass = _branch_oracle(q, v)[0]
                    rows.append(dict(receipt=receipt, diagnostic=diagnostic, oracle_q=q.copy(), oracle_v=v.copy(), oracle_mass=mass, oracle_bias=bias, oracle_load=load, oracle_delta_kinetic_j=float(0.5 * v @ next_mass @ v - kinetic_before), audit=audit))
                    _close(receipt.accepted_state.position, q[order], atol=WINDOW_ATOL)
                    _close(receipt.accepted_state.velocity, v[order], atol=WINDOW_ATOL)
                    _close(diagnostic.mass, mass[np.ix_(order, order)], atol=WINDOW_ATOL)
                    _close(diagnostic.bias, bias[order], atol=WINDOW_ATOL)
                    _close(receipt.contact_generalized, load[order], atol=WINDOW_ATOL)
                    assert receipt.exit_tick == index + 1
                    assert abs(q[1]) < 1 / 16 and abs(q[0]) < 1 / 16
                assert len(rows) == 40
            finally:
                _save(tmp_path, "r02-window", dict(mode=tyx_mode, h_s=H, fixed_steps=40, model=program.model, command=command, order=order, rotation=rotation, translation=translation, sources=sources, rows=rows, stats=batch.stats()))


def _rotor(*, velocity=0.0):
    model = ph.RobotModel(np.array([-1]), ("revolute",), np.zeros((1, 3)), np.array([[0.0, 0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, 1.0]]), np.array([1.0]), np.zeros((1, 3)), np.array([[1 / 8, 1 / 8, 1 / 4]]), np.zeros(3))
    return _program(model, v=[velocity])


def _rotor_bound(batch, world=7, *, reverse=False):
    return _bind(batch, world, [[1 / 2, 3 / 32, 0]], _box([-1, -1 / 4, -1 / 2], [1, 0, 1 / 2]), moving_kind="sphere", radius=1 / 8, reverse=reverse)


@pytest.mark.tyx
@pytest.mark.parametrize("velocity,damping,normal,acceleration,reverse", ((0.0, 0, 2, 4, False), (0.5, 4, 1, 2, False), (0.5, 4, 1, 2, True)))
def test_r03_eccentric_sphere_uses_load_point_velocity_and_true_arc(velocity, damping, normal, acceleration, reverse, tyx_mode, tmp_path):
    program, world = _rotor(velocity=velocity)
    with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
        command = _action(program)
        batch.set_controls({7: command})
        model, bindings, sources, pair = _rotor_bound(batch, reverse=reverse)
        with batch.prepare_contacts(model, bindings, {pair: _law(damping)}) as plan:
            assert batch.step_contacts(plan, H) == {7: True}
            receipt, diagnostic = plan.last_accepted[7], batch.tree_diagnostics()[7]
            expected_v, expected_q = velocity + H * acceleration, H * (velocity + H * acceleration)
            _close(receipt.accepted_state.velocity, [expected_v])
            _close(receipt.accepted_state.position, [expected_q])
            _close(receipt.law_wrench_fp64, [[0, normal, 0, 0, 0, normal / 2]])
            _close(receipt.contact_generalized, [normal / 2])
            _close(diagnostic.acceleration, [acceleration])
            _close(receipt.accepted_geometry.link_rotation[0] @ np.array([1 / 2, 3 / 32, 0]), _rz(expected_q) @ [1 / 2, 3 / 32, 0])
            audit = _audit_receipt(receipt, diagnostic, sources)
            _save(tmp_path, "r03-first", dict(mode=tyx_mode, h_s=H, model=program.model, command=command, velocity=velocity, damping=damping, reverse_keys=reverse, sources=sources, receipt=receipt, diagnostic=diagnostic, audit=audit, stats=batch.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("velocity,damping", ((0.0, 0), (0.5, 4)))
def test_r03_forty_rotating_steps_record_actual_material_and_application_work(velocity, damping, tyx_mode, tmp_path):
    program, world = _rotor(velocity=velocity)
    q, v, rows = 0.0, velocity, []
    with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
        command = _action(program)
        batch.set_controls({7: command})
        model, bindings, sources, pair = _rotor_bound(batch)
        with batch.prepare_contacts(model, bindings, {pair: _law(damping)}) as plan:
            try:
                for index in range(40):
                    cx, cy, _ = _rz(q) @ [1 / 2, 3 / 32, 0]
                    depth = max(0.0, 1 / 8 - cy)
                    normal = max(0.0, 64 * depth - damping * cx * v) if depth else 0.0
                    torque, old_v, old_depth = cx * normal, v, depth
                    v += H * 4 * torque
                    q += H * v
                    assert batch.step_contacts(plan, H) == {7: True}
                    receipt, diagnostic = plan.last_accepted[7], batch.tree_diagnostics()[7]
                    audit = _audit_receipt(receipt, diagnostic, sources)
                    new_depth = max(0.0, 1 / 8 - (_rz(q) @ [1 / 2, 3 / 32, 0])[1])
                    rows.append(dict(receipt=receipt, diagnostic=diagnostic, oracle_q=q, oracle_v=v, oracle_force_n=normal, oracle_torque_nm=torque, entry_power_times_h_j=torque * old_v * H, oracle_delta_kinetic_j=(v * v - old_v * old_v) / 8, oracle_delta_spring_j=32 * (new_depth * new_depth - old_depth * old_depth), audit=audit))
                    _close(receipt.accepted_state.position, [q], atol=WINDOW_ATOL)
                    _close(receipt.accepted_state.velocity, [v], atol=WINDOW_ATOL)
                    _close(receipt.contact_generalized, [torque], atol=WINDOW_ATOL)
                    _close(receipt.accepted_geometry.link_rotation[0] @ np.array([1 / 2, 3 / 32, 0]), _rz(q) @ [1 / 2, 3 / 32, 0], atol=WINDOW_ATOL)
                    assert receipt.exit_tick == index + 1
                    assert abs(cx) < 1 and cy > -1 / 4
                assert len(rows) == 40
            finally:
                _save(tmp_path, "r03-window", dict(mode=tyx_mode, h_s=H, fixed_steps=40, model=program.model, command=command, damping=damping, initial_velocity=velocity, sources=sources, rows=rows, stats=batch.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("dt", (H, H / 2))
def test_r04_continuous_unload_and_reentry_has_no_stale_contact_force(dt, tyx_mode, tmp_path):
    program, world = _slider(v=1 / 2)
    q, v, rows, separated, reentered = 0.0, 0.5, [], False, False
    switch_step, maximum_steps = int((1 / 8) / dt), int((7 / 8) / dt)
    with resident.TyxWorldBatch([(17, world)], tyx_mode, cache_root=tmp_path) as batch:
        model, bindings, sources, pair = _r01_bound(batch)
        with batch.prepare_contacts(model, bindings, {pair: _law()}) as plan:
            try:
                for index in range(maximum_steps):
                    external = 0.0 if index < switch_step else -4.0
                    command = _action(program, force=[[0, 0, external]])
                    batch.set_controls({17: command})
                    before = batch.snapshot()[17]
                    normal = 64 * max(0.0, 1 / 32 - q)
                    v += dt * (normal + external) / 2
                    q += dt * v
                    assert batch.step_contacts(plan, dt) == {17: True}
                    receipt, diagnostic = plan.last_accepted[17], batch.tree_diagnostics()[17]
                    audit = _audit_receipt(receipt, diagnostic, sources)
                    actual_entry_depth = max(0.0, 1 / 32 - before.position[0])
                    actual_n = float(receipt.contact_generalized[0])
                    rows.append(dict(command=command, before=before, receipt=receipt, diagnostic=diagnostic, oracle_q=q, oracle_v=v, oracle_normal_n=normal, actual_entry_depth_m=actual_entry_depth, audit=audit))
                    _close(receipt.accepted_state.position, [q], atol=WINDOW_ATOL)
                    _close(receipt.accepted_state.velocity, [v], atol=WINDOW_ATOL)
                    _close(receipt.contact_generalized, [normal], atol=WINDOW_ATOL)
                    _close(receipt.input_external, [external])
                    assert receipt.accepted_state.tick == index + 1
                    assert receipt.accepted_state.control == before.control
                    if actual_entry_depth == 0:
                        separated = True
                        assert actual_n == 0.0
                        np.testing.assert_array_equal(receipt.law_wrench_fp64, np.zeros((1, 6)))
                    elif separated and actual_n > 0:
                        reentered = True
                        break
                assert separated and reentered
                assert rows[0]["receipt"].contact_generalized[0] > 0
                assert len(rows) <= maximum_steps
            finally:
                _save(tmp_path, "r04-unload-reentry", dict(mode=tyx_mode, h_s=dt, switch_time_s=1 / 8, maximum_time_s=7 / 8, initial_velocity_m_s=0.5, model=program.model, sources=sources, rows=rows, separated=separated, reentered=reentered, stats=batch.stats()))

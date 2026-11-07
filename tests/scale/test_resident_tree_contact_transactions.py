"""C6 independent world/transaction controls, frozen before execution.

Only new C6 fixtures are shared. No old test or product solver supplies an oracle.
Each failure retains accepted physics, original held packets and prior receipts.
"""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from test_resident_tree_contacts import (
    WINDOW_ATOL,
    H,
    _action,
    _audit_receipt,
    _bind,
    _box,
    _branch,
    _branch_bound,
    _branch_oracle,
    _close,
    _json,
    _law,
    _r01_bound,
    _rotor,
    _rotor_bound,
    _rz,
    _save,
    _slider,
)

import phymeo as ph
from phymeo.backends import tyx_world_batch as resident


def _cases():
    slider, slider_world = _slider()
    branch, branch_world, order, rotation, translation = _branch()
    rotor, rotor_world = _rotor()
    empty, empty_world = _slider()
    return {
        17: (slider, slider_world, "slider", ()),
        103: (branch, branch_world, "branch", (order, rotation, translation)),
        7: (rotor, rotor_world, "rotor", ()),
        409: (empty, empty_world, "empty", ()),
    }


def _combined_geometry(batch, cases):
    shapes, bindings, sources, laws = [], [], {}, {}
    for world, (_, _, kind, extra) in cases.items():
        if kind == "empty":
            continue
        if kind == "slider":
            built = _r01_bound(batch, world)
        elif kind == "branch":
            built = _branch_bound(batch, world, *extra)
        else:
            built = _rotor_bound(batch, world)
        model, attached, source, pair = built
        shapes.extend(model.shapes)
        bindings.extend(attached)
        sources[world] = source
        laws[pair] = _law()
    return ph.CollisionModel(tuple(shapes)), tuple(bindings), sources, laws


def _oracle_next(kind, q, v):
    if kind == "branch":
        acceleration = _branch_oracle(q, v)[3]
    elif kind == "rotor":
        cx, cy, _ = _rz(q[0]) @ [1 / 2, 3 / 32, 0]
        normal = 64 * max(0.0, 1 / 8 - cy)
        acceleration = np.array([4 * cx * normal])
    elif kind == "slider":
        acceleration = np.array([32 * max(0.0, 1 / 32 - q[0])])
    else:
        acceleration = np.array([0.5])
    next_v = v + H * acceleration
    return q + H * next_v, next_v


def _window(cases, mode, directory):
    directory.mkdir(parents=True, exist_ok=True)
    inputs = {
        w: _action(p, ff=[1] if kind == "empty" else None) for w, (p, _, kind, _) in cases.items()
    }
    oracle = {
        w: (np.zeros(p.model.dof_count), np.zeros(p.model.dof_count))
        for w, (p, _, _, _) in cases.items()
    }
    rows = []
    with resident.TyxWorldBatch(
        [(w, item[1]) for w, item in cases.items()], mode, cache_root=directory
    ) as batch:
        batch.set_controls(inputs)
        model, bindings, sources, laws = _combined_geometry(batch, cases)
        with batch.prepare_contacts(model, bindings, laws) as plan:
            try:
                for index in range(40):
                    expected = {w: _oracle_next(item[2], *oracle[w]) for w, item in cases.items()}
                    accepted = batch.step_contacts(plan, H)
                    state, geometry, diagnostic = (
                        batch.snapshot(),
                        batch.tree_kinematics(),
                        batch.tree_diagnostics(),
                    )
                    receipts = dict(plan.last_accepted)
                    rows.append(
                        dict(
                            accepted=accepted,
                            state=state,
                            geometry=geometry,
                            diagnostic=diagnostic,
                            receipts=receipts,
                            oracle=expected,
                            stats=batch.stats(),
                        )
                    )
                    assert accepted == dict.fromkeys(cases, True)
                    for world, (q, v) in expected.items():
                        _close(state[world].position, q, atol=WINDOW_ATOL)
                        _close(state[world].velocity, v, atol=WINDOW_ATOL)
                        assert state[world].tick == index + 1
                        if world in sources:
                            _audit_receipt(receipts[world], diagnostic[world], sources[world])
                    oracle = expected
                assert len(rows) == 40
            finally:
                _save(
                    directory,
                    "r05-world-window",
                    dict(
                        mode=mode,
                        h_s=H,
                        fixed_steps=40,
                        models={w: p.model for w, (p, _, _, _) in cases.items()},
                        inputs=inputs,
                        collision_model=model,
                        sources=sources,
                        rows=rows,
                        stats=batch.stats(),
                    ),
                )
    return rows


@pytest.mark.tyx
def test_r05_w1_w4_duplicate_local_ids_and_reverse_registration_match(tyx_mode, tmp_path):
    cases = _cases()
    independent = {
        w: _window({w: case}, tyx_mode, tmp_path / f"w1-{w}") for w, case in cases.items()
    }
    reversed_cases = dict(reversed(tuple(cases.items())))
    combined = _window(reversed_cases, tyx_mode, tmp_path / "w4-reversed")
    for index, row in enumerate(combined):
        for world in cases:
            reference = independent[world][index]
            assert row["state"][world] == reference["state"][world]
            for name in (
                "position",
                "velocity",
                "link_position_m",
                "link_rotation",
                "com_position_m",
            ):
                np.testing.assert_array_equal(
                    getattr(row["geometry"][world], name),
                    getattr(reference["geometry"][world], name),
                )
            if world in row["receipts"]:
                a, b = row["receipts"][world], reference["receipts"][world]
                assert a.entry_state == b.entry_state and a.accepted_state == b.accepted_state
                assert _json(a.sites) == _json(b.sites)
                for name in (
                    "law_wrench_fp64",
                    "uploaded_wrench_fp32",
                    "contact_generalized",
                    "input_external",
                    "selected_drive",
                    "actual_drive",
                ):
                    np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


@pytest.mark.tyx
@pytest.mark.parametrize("warm", (0, 1))
def test_r05_site_overflow_retains_only_that_world_and_new_plan_retries(warm, tyx_mode, tmp_path):
    cases = _cases()
    with resident.TyxWorldBatch(
        [(w, item[1]) for w, item in cases.items()], tyx_mode, cache_root=tmp_path
    ) as batch:
        commands = {
            w: _action(p, ff=[1] if kind == "empty" else None)
            for w, (p, _, kind, _) in cases.items()
        }
        batch.set_controls(commands)
        model, bindings, sources, laws = _combined_geometry(batch, cases)
        with batch.prepare_contacts(model, bindings, laws) as full:
            for _ in range(warm):
                assert batch.step_contacts(full, H) == dict.fromkeys(cases, True)
            before, fk = batch.snapshot(), batch.tree_kinematics()
            prior = _json(full.last_accepted)
            with batch.prepare_contacts(
                model, bindings, laws, site_capacity={17: 0, 103: 128, 7: 128}
            ) as limited:
                assert batch.step_contacts(limited, H) == {17: False, 103: True, 7: True, 409: True}
                after = batch.snapshot()
                assert limited.last_status[17] == "SITE_CAPACITY_EXCEEDED"
                assert after[17] == before[17]
                assert 17 not in limited.last_accepted
                np.testing.assert_array_equal(
                    batch.tree_kinematics()[17].link_position_m, fk[17].link_position_m
                )
                for world in (103, 7, 409):
                    assert after[world].tick == before[world].tick + 1
                assert _json(full.last_accepted) == prior
                # The contact configuration is not mutated to relax its cap.
                retained = batch.snapshot()[17]
                assert batch.step_contacts(full, H) == dict.fromkeys(cases, True)
                retry = full.last_accepted[17]
                assert retry.entry_state == retained and retry.entry_tick == before[17].tick
                _audit_receipt(retry, batch.tree_diagnostics()[17], sources[17])
                _save(
                    tmp_path,
                    "r05-site-capacity",
                    dict(
                        mode=tyx_mode,
                        warm=warm,
                        inputs=commands,
                        before=before,
                        after_limited=after,
                        limited_status=limited.last_status,
                        retry=retry,
                        final=batch.snapshot(),
                        stats=batch.stats(),
                    ),
                )


@pytest.mark.tyx
def test_r05_site_capacity_counts_geometry_even_when_damping_clips_force_to_zero(
    tyx_mode, tmp_path
):
    program, world = _slider(v=1.0)
    with resident.TyxWorldBatch([(17, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({17: _action(program)})
        model, bindings, _, pair = _r01_bound(batch)
        before = batch.snapshot()
        with batch.prepare_contacts(model, bindings, {pair: _law(4)}, site_capacity=0) as plan:
            assert batch.step_contacts(plan, H) == {17: False}
            assert plan.last_status[17] == "SITE_CAPACITY_EXCEEDED"
            assert batch.snapshot() == before
            assert not plan.last_accepted
        with batch.prepare_contacts(model, bindings, {pair: _law(4)}, site_capacity=128) as plan:
            assert batch.step_contacts(plan, H) == {17: True}
            receipt = plan.last_accepted[17]
            assert len(receipt.sites) >= 3
            np.testing.assert_array_equal(receipt.contact_generalized, [0])
            _close(receipt.accepted_state.velocity, [1])
            _save(
                tmp_path,
                "r05-zero-force-sites",
                dict(mode=tyx_mode, before=before, receipt=receipt, stats=batch.stats()),
            )


@pytest.mark.tyx
def test_r05_candidate_capacity_overflow_is_batch_wide_not_world_retention(tyx_mode, tmp_path):
    cases = _cases()
    with resident.TyxWorldBatch(
        [(w, item[1]) for w, item in cases.items()], tyx_mode, cache_root=tmp_path
    ) as batch:
        commands = {
            w: _action(p, ff=[1] if kind == "empty" else None)
            for w, (p, _, kind, _) in cases.items()
        }
        batch.set_controls(commands)
        model, bindings, _, laws = _combined_geometry(batch, cases)
        before, stats = batch.snapshot(), batch.stats()
        with batch.prepare_contacts(model, bindings, laws, candidate_capacity=0) as plan:
            with pytest.raises(ValueError, match="capacity"):
                batch.step_contacts(plan, H)
            assert batch.snapshot() == before and not plan.last_accepted
            assert batch.stats()["state_epoch"] == stats["state_epoch"]
            assert batch.stats()["current_buffer"] == stats["current_buffer"]
        with batch.prepare_contacts(model, bindings, laws) as retry:
            assert batch.step_contacts(retry, H) == dict.fromkeys(cases, True)
            _save(
                tmp_path,
                "r05-candidate-capacity",
                dict(
                    mode=tyx_mode,
                    inputs=commands,
                    before=before,
                    retry=batch.snapshot(),
                    receipts=retry.last_accepted,
                    stats=batch.stats(),
                ),
            )


def _affine(tmp_path):
    source = tmp_path / "unbound_affine.urdf"
    source.write_text(
        """<robot name="ordinary_passive_pair"><link name="base"/>
<link name="driver_link"><inertial><mass value="1"/><inertia ixx=".125" iyy=".125" izz=".125" ixy="0" ixz="0" iyz="0"/></inertial></link>
<link name="follower_link"><inertial><mass value="1"/><inertia ixx=".125" iyy=".125" izz=".125" ixy="0" ixz="0" iyz="0"/></inertial></link>
<joint name="driver" type="prismatic"><parent link="base"/><child link="driver_link"/><axis xyz="0 0 1"/><limit lower="-16" upper="16" velocity="16" effort="8"/></joint>
<joint name="follower" type="prismatic"><parent link="base"/><child link="follower_link"/><axis xyz="0 0 1"/><limit lower="-16" upper="16" velocity="16" effort="8"/><mimic joint="driver" multiplier="-1" offset=".125"/></joint></robot>""",
        encoding="utf-8",
    )
    asset = ph.load_mimic_urdf(source, gravity=(0, 0, 0))
    world = ph.AffineRobotTree(
        asset,
        ph.RobotState(np.zeros(1), np.zeros(1)),
        kp=np.zeros(1),
        kd=np.zeros(1),
        effort_limit=np.array([8.0]),
    )
    return world


@pytest.mark.tyx
def test_r05_unbound_affine_and_scalar_keep_original_steps_beside_contact_world(tyx_mode, tmp_path):
    program, ordinary = _slider()
    affine = _affine(tmp_path)
    free = resident.FreeSphere(2, 1 / 16, gravity=(0, 0, 0))
    worlds = [(17, ordinary), (51, affine), (23, free)]
    commands = {17: _action(program), 51: affine.inputs([0.0], feedforward=[1.0]), 23: (2, 0, 0)}
    with resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls(commands)
        model, bindings, _, pair = _r01_bound(batch)
        with batch.prepare_contacts(model, bindings, {pair: _law()}) as plan:
            assert batch.step_contacts(plan, H) == {17: True, 51: True, 23: True}
            state = batch.snapshot()
            _close(state[17].velocity, [H])
            _close(state[51].velocity, [H / 2])
            _close(state[51].full_velocity, [H / 2, -H / 2])
            _close(state[23].velocity, [H, 0, 0])
            assert all(value.tick == 1 for value in state.values())
            assert set(plan.last_accepted) == {17}
            _save(
                tmp_path,
                "r05-unbound-kinds",
                dict(
                    mode=tyx_mode,
                    commands=commands,
                    state=state,
                    geometry=batch.tree_kinematics([17, 51]),
                    receipts=plan.last_accepted,
                    stats=batch.stats(),
                ),
            )


@pytest.mark.tyx
def test_r05_affine_contact_binding_is_explicitly_rejected(tyx_mode, tmp_path):
    world = _affine(tmp_path)
    with resident.TyxWorldBatch([(51, world)], tyx_mode, cache_root=tmp_path) as batch:
        moving, fixed = ph.CollisionKey(51, 100, 11), ph.CollisionKey(51, 900, 1)
        shapes = (
            ph.CollisionShape(moving, "sphere", (101,), radius=1 / 8),
            ph.CollisionShape(fixed, "hull", tuple(range(201, 209))),
        )
        bindings = (
            ph.ResidentAffineTreeBinding(
                moving, 51, 0, [[0, 0, 3 / 32]], topology=batch.snapshot()[51].topology
            ),
            ph.ResidentStaticBinding(
                fixed, ph.CollisionMotion(_box([-1 / 2, -1 / 2, -1 / 4], [1 / 2, 1 / 2, 0]))
            ),
        )
        before, allocations = batch.snapshot(), batch.stats()["device_allocations"]
        with pytest.raises((ValueError, TypeError), match="affine|ordinary|contact"):
            batch.prepare_contacts(ph.CollisionModel(shapes), bindings, {(moving, fixed): _law()})
        assert batch.snapshot() == before
        assert batch.stats()["device_allocations"] == allocations


def test_r05_empty_batch_and_plan_are_valid_without_device_allocation():
    with resident.TyxWorldBatch([], "interpret") as batch:
        with batch.prepare_contacts(ph.CollisionModel(()), (), {}) as plan:
            assert batch.step_contacts(plan, H) == {}
            assert not plan.last_accepted and not plan.last_status
            assert plan.stats()["device_allocations"] == 0
            assert batch.stats()["device_allocations"] == 0


@pytest.mark.tyx
@pytest.mark.parametrize("capacity", (-1, 4097, 1.5, True, {17: 1, 99: 1}, {}))
def test_r05_site_capacity_is_bounded_and_exactly_covers_bound_worlds(capacity, tyx_mode, tmp_path):
    program, world = _slider()
    with resident.TyxWorldBatch([(17, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({17: _action(program)})
        model, bindings, _, pair = _r01_bound(batch)
        before, stats = batch.snapshot(), batch.stats()
        with pytest.raises((ValueError, TypeError)):
            batch.prepare_contacts(model, bindings, {pair: _law()}, site_capacity=capacity)
        assert batch.snapshot() == before
        assert batch.stats()["device_allocations"] == stats["device_allocations"]


@pytest.mark.tyx
@pytest.mark.parametrize(
    "problem",
    (
        "friction",
        "wrong-law",
        "missing-law",
        "missing-binding",
        "wrong-world",
        "wrong-topology",
        "wrong-group",
        "sampling",
        "dynamic-pair",
        "unsupported",
    ),
)
def test_r05_invalid_configuration_never_allocates_contact_scratch(problem, tyx_mode, tmp_path):
    _, world = _slider()
    with resident.TyxWorldBatch([(17, world)], tyx_mode, cache_root=tmp_path) as batch:
        model, bindings, _, pair = _r01_bound(batch)
        laws, kwargs = {pair: _law()}, {}
        if problem == "friction":
            laws[pair] = ph.RobotContactMaterial(stiffness=64, damping=0, friction=1 / 2)
        elif problem == "wrong-law":
            laws[pair] = ph.NormalLayer(1 / 32, 64)
        elif problem == "missing-law":
            laws = {}
        elif problem == "missing-binding":
            bindings = bindings[:1]
        elif problem == "wrong-world":
            bindings = (
                replace(bindings[0], key=ph.CollisionKey(999, 100, 11), world=999),
                bindings[1],
            )
        elif problem == "wrong-topology":
            bindings = (replace(bindings[0], topology=("different-source",)), bindings[1])
        elif problem == "wrong-group":
            bindings = (replace(bindings[0], group_index=1), bindings[1])
        elif problem == "sampling":
            kwargs["sampling"] = "swept"
        elif problem == "dynamic-pair":
            bindings = (
                bindings[0],
                ph.ResidentTreeBinding(
                    pair[1],
                    17,
                    0,
                    _box([-1 / 2, -1 / 2, -1 / 4], [1 / 2, 1 / 2, 0]),
                    topology=batch.snapshot()[17].topology,
                ),
            )
        else:
            shape = ph.CollisionShape(pair[1], "plane", (201,), normal=(0, 0, 1))
            model = ph.CollisionModel((model.shapes[0], shape))
            bindings = (
                bindings[0],
                ph.ResidentStaticBinding(pair[1], ph.CollisionMotion([[0, 0, 0]])),
            )
        before, stats = batch.snapshot(), batch.stats()
        with pytest.raises((ValueError, TypeError)):
            batch.prepare_contacts(model, bindings, laws, **kwargs)
        assert batch.snapshot() == before
        assert batch.stats()["device_allocations"] == stats["device_allocations"]


@pytest.mark.tyx
@pytest.mark.parametrize("problem", ("deep-containment", "coincident-spheres"))
def test_r05_undefined_normal_is_an_error_not_false_separation(problem, tyx_mode, tmp_path):
    program, world = _slider(origin=(0, 0, 0))
    with resident.TyxWorldBatch(
        [(17, world), (23, resident.FreeSphere(2, 1 / 16, gravity=(0, 0, 0)))],
        tyx_mode,
        cache_root=tmp_path,
    ) as batch:
        batch.set_controls({17: _action(program), 23: (2, 0, 0)})
        if problem == "deep-containment":
            model, bindings, _, pair = _bind(
                batch,
                17,
                [[0, 0, 0]],
                _box([-1 / 2] * 3, [1 / 2] * 3),
                moving_kind="sphere",
                radius=1 / 8,
            )
        else:
            model, bindings, _, pair = _bind(
                batch,
                17,
                [[0, 0, 0]],
                [[0, 0, 0]],
                moving_kind="sphere",
                static_kind="sphere",
                radius=1 / 8,
                static_radius=1 / 8,
            )
        before, stats = batch.snapshot(), batch.stats()
        with batch.prepare_contacts(model, bindings, {pair: _law()}) as plan:
            with pytest.raises(ValueError, match="deep|ambiguous|normal|coincident"):
                batch.step_contacts(plan, H)
            assert batch.snapshot() == before and not plan.last_accepted
            assert batch.stats()["state_epoch"] == stats["state_epoch"]
            assert batch.stats()["current_buffer"] == stats["current_buffer"]
            _save(
                tmp_path,
                "r05-undefined-normal",
                dict(
                    mode=tyx_mode,
                    problem=problem,
                    before=before,
                    after=batch.snapshot(),
                    statuses=plan.last_status,
                    stats=batch.stats(),
                ),
            )


@pytest.mark.tyx
@pytest.mark.parametrize(
    "warm,phase",
    tuple(product((0, 1), ("upload", "launch", "sync", "status", "candidate", "receipt"))),
)
def test_r05_late_failures_never_publish_a_partial_batch_or_receipt(
    warm, phase, tyx_mode, tmp_path, monkeypatch
):
    program, world = _slider(delay=1)
    worlds = [(17, world), (23, resident.FreeSphere(2, 1 / 16, gravity=(0, 0, 0)))]
    with (
        resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "failed") as batch,
        resident.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "control") as control,
    ):
        initial = {17: _action(program), 23: (0, 1, 0)}
        batch.set_controls(initial)
        control.set_controls(initial)
        model, bindings, _, pair = _r01_bound(batch)
        other_model, other_bindings, _, other_pair = _r01_bound(control)
        with (
            batch.prepare_contacts(model, bindings, {pair: _law()}) as plan,
            control.prepare_contacts(
                other_model, other_bindings, {other_pair: _law()}
            ) as reference,
        ):
            for _ in range(warm):
                assert (
                    batch.step_contacts(plan, H)
                    == control.step_contacts(reference, H)
                    == {17: True, 23: True}
                )
            next_inputs = {17: _action(program, ff=[1 / 4], force=[[0, 0, 1 / 2]]), 23: (1, 0, 0)}
            batch.set_controls(next_inputs)
            control.set_controls(next_inputs)
            before, geometry, stats = batch.snapshot(), batch.tree_kinematics(), batch.stats()
            prior = _json(plan.last_accepted)
            owner = plan if phase in ("candidate", "receipt") else batch
            method = {
                "upload": "_upload",
                "launch": "_launch",
                "sync": "_sync",
                "status": "_read_status",
                "candidate": "_read_candidate",
                "receipt": "_finish_attempt",
            }[phase]
            original = getattr(owner, method)
            calls = []

            def call_then_fail(*args, **kwargs):
                value = original(*args, **kwargs)
                calls.append(True)
                if len(calls) == 1:
                    raise RuntimeError("c6 injected " + phase)
                return value

            with monkeypatch.context() as patch:
                patch.setattr(owner, method, call_then_fail)
                with pytest.raises(RuntimeError, match="c6 injected " + phase):
                    batch.step_contacts(plan, H)
            assert calls
            assert batch.snapshot() == before
            assert batch.stats()["current_buffer"] == stats["current_buffer"]
            assert batch.stats()["state_epoch"] == stats["state_epoch"]
            assert _json(plan.last_accepted) == prior
            for name in (
                "position",
                "velocity",
                "link_position_m",
                "link_rotation",
                "com_position_m",
            ):
                np.testing.assert_array_equal(
                    getattr(batch.tree_kinematics()[17], name), getattr(geometry[17], name)
                )
            # A new plan is allowed after a failed scratch attempt; original physics is not reset.
            plan.close()
            with batch.prepare_contacts(model, bindings, {pair: _law()}) as retry:
                assert (
                    batch.step_contacts(retry, H)
                    == control.step_contacts(reference, H)
                    == {17: True, 23: True}
                )
                assert batch.snapshot() == control.snapshot()
                assert _json(retry.last_accepted[17].sites) == _json(
                    reference.last_accepted[17].sites
                )
                _save(
                    tmp_path,
                    "r05-fault-retry",
                    dict(
                        mode=tyx_mode,
                        phase=phase,
                        warm=warm,
                        input=next_inputs,
                        before=before,
                        before_geometry=geometry,
                        prior_receipts=prior,
                        after=batch.snapshot(),
                        receipt=retry.last_accepted[17],
                        control=control.snapshot(),
                        stats=batch.stats(),
                    ),
                )


@pytest.mark.tyx
@pytest.mark.parametrize("warm", (0, 1))
def test_r05_same_tick_restore_inside_candidate_read_invalidates_attempt(
    warm, tyx_mode, tmp_path, monkeypatch
):
    program, world = _slider(delay=1)
    with resident.TyxWorldBatch([(17, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({17: _action(program, ff=[1 / 4])})
        model, bindings, _, pair = _r01_bound(batch)
        with batch.prepare_contacts(model, bindings, {pair: _law()}) as plan:
            for _ in range(warm):
                assert batch.step_contacts(plan, H) == {17: True}
            saved, prior = batch.snapshot(), _json(plan.last_accepted)
            old_epoch = batch.stats()["state_epoch"]
            original, restored = plan._read_candidate, []

            def restore_after_read(*args, **kwargs):
                value = original(*args, **kwargs)
                batch.restore(saved)
                restored.append(batch.stats())
                return value

            with monkeypatch.context() as patch:
                patch.setattr(plan, "_read_candidate", restore_after_read)
                with pytest.raises((ValueError, RuntimeError), match="epoch|stale|changed|entry"):
                    batch.step_contacts(plan, H)
            assert len(restored) == 1
            assert restored[0]["state_epoch"] == old_epoch + 1
            assert batch.stats()["state_epoch"] == restored[0]["state_epoch"]
            assert batch.stats()["current_buffer"] == restored[0]["current_buffer"]
            assert batch.snapshot() == saved
            assert _json(plan.last_accepted) == prior
            plan.close()
            with batch.prepare_contacts(model, bindings, {pair: _law()}) as fresh:
                assert batch.step_contacts(fresh, H) == {17: True}
                assert fresh.last_accepted[17].entry_state == saved[17]
                _save(
                    tmp_path,
                    "r05-stale-same-tick",
                    dict(
                        mode=tyx_mode,
                        warm=warm,
                        saved=saved,
                        restoration_stats=restored,
                        receipt=fresh.last_accepted[17],
                        stats=batch.stats(),
                    ),
                )


@pytest.mark.tyx
def test_r05_plan_owner_must_match_even_when_world_ids_and_ticks_match(tyx_mode, tmp_path):
    program, world = _slider()
    with (
        resident.TyxWorldBatch([(17, world)], tyx_mode, cache_root=tmp_path / "first") as a,
        resident.TyxWorldBatch([(17, world)], tyx_mode, cache_root=tmp_path / "second") as b,
    ):
        a.set_controls({17: _action(program)})
        b.set_controls({17: _action(program)})
        model, bindings, _, pair = _r01_bound(a)
        before_a, before_b = a.snapshot(), b.snapshot()
        with a.prepare_contacts(model, bindings, {pair: _law()}) as plan:
            with pytest.raises((ValueError, RuntimeError), match="owner|batch|belong"):
                b.step_contacts(plan, H)
            assert a.snapshot() == before_a and b.snapshot() == before_b
            assert not plan.last_accepted
            assert a.step_contacts(plan, H) == {17: True}


@pytest.mark.tyx
def test_r05_receipt_is_owned_immutable_and_observation_does_not_advance(tyx_mode, tmp_path):
    program, world = _rotor(velocity=0.5)
    with resident.TyxWorldBatch([(7, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: _action(program)})
        model, bindings, sources, pair = _rotor_bound(batch)
        with batch.prepare_contacts(model, bindings, {pair: _law(4)}) as plan:
            assert batch.step_contacts(plan, H) == {7: True}
            mapping, original = plan.last_accepted, plan.last_accepted[7]
            retained = _json(original)
            before, stats = batch.snapshot(), batch.stats()
            with pytest.raises(TypeError):
                mapping[7] = original
            vectors = [
                original.law_wrench_fp64,
                original.uploaded_wrench_fp32,
                original.contact_generalized,
                original.actual_drive,
                original.input_external,
                original.selected_drive,
                original.entry_geometry.link_rotation,
                original.accepted_geometry.com_position_m,
            ]
            for site in original.sites:
                vectors.extend(
                    getattr(site, name)
                    for name in (
                        "first_velocity_m_s",
                        "second_velocity_m_s",
                        "first_force_n",
                        "second_force_n",
                        "first_reference_point_m",
                        "second_reference_point_m",
                        "first_material_displacement_m",
                        "second_material_displacement_m",
                        "first_application_displacement_m",
                        "second_application_displacement_m",
                    )
                )
            for reaction in original.static_reactions.values():
                vectors.extend((reaction.reference_point_m, reaction.force_n, reaction.torque_nm))
            for vector in vectors:
                array = np.asarray(vector)
                assert not array.flags.writeable
                with pytest.raises(ValueError):
                    array.setflags(write=True)
            for _ in range(3):
                assert _json(plan.last_accepted[7]) == retained
                assert batch.snapshot() == before
                batch.tree_kinematics()
            assert batch.stats()["state_epoch"] == stats["state_epoch"]
            assert batch.stats()["current_buffer"] == stats["current_buffer"]
            assert batch.step_contacts(plan, H) == {7: True}
            assert _json(original) == retained
            assert original.exit_tick == 1 and plan.last_accepted[7].exit_tick == 2
            _audit_receipt(plan.last_accepted[7], batch.tree_diagnostics()[7], sources)
            _save(
                tmp_path,
                "r05-owned-observation",
                dict(
                    mode=tyx_mode,
                    retained=retained,
                    current=plan.last_accepted[7],
                    stats=batch.stats(),
                ),
            )


@pytest.mark.tyx
def test_r05_second_real_contact_scratch_allocation_failure_releases_both(
    tyx_mode, tmp_path, monkeypatch
):
    from phymeo.backends._resident_contact_plan import ResidentContactPlan

    program, world = _slider()
    with resident.TyxWorldBatch([(17, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({17: _action(program)})
        model, bindings, _, pair = _r01_bound(batch)
        before, prior = batch.snapshot(), batch.stats()
        original, calls, owner = ResidentContactPlan._buffer, [], []

        def allocate_then_fail(self, *args, **kwargs):
            value = original(self, *args, **kwargs)
            calls.append(value)
            owner.append(self)
            if len(calls) == 2:
                raise MemoryError("c6 after second real contact scratch allocation")
            return value

        with monkeypatch.context() as patch:
            patch.setattr(ResidentContactPlan, "_buffer", allocate_then_fail)
            with pytest.raises(MemoryError, match="second real contact"):
                batch.prepare_contacts(model, bindings, {pair: _law()})
        assert len(calls) == 2 and owner[0] is owner[1]
        assert owner[0].stats()["closed"]
        assert owner[0].stats()["device_owned_bytes"] == 0
        assert batch.snapshot() == before
        assert batch.stats()["current_buffer"] == prior["current_buffer"]
        assert batch.stats()["state_epoch"] == prior["state_epoch"]
        with batch.prepare_contacts(model, bindings, {pair: _law()}) as retry:
            assert batch.step_contacts(retry, H) == {17: True}
            _save(
                tmp_path,
                "r05-allocation-recovery",
                dict(
                    mode=tyx_mode,
                    before=before,
                    failed_plan_stats=owner[0].stats(),
                    receipt=retry.last_accepted[17],
                    stats=batch.stats(),
                ),
            )


@pytest.mark.tyx
def test_r05_plan_close_borrows_runtime_and_all_close_callbacks_are_attempted(
    tyx_mode, tmp_path, monkeypatch
):
    program, world = _slider()
    batch = resident.TyxWorldBatch([(17, world)], tyx_mode, cache_root=tmp_path)
    try:
        batch.set_controls({17: _action(program)})
        model, bindings, _, pair = _r01_bound(batch)
        first = batch.prepare_contacts(model, bindings, {pair: _law()})
        second = batch.prepare_contacts(model, bindings, {pair: _law()})
        first.close()
        first.close()
        assert first.stats()["closed"] and first.stats()["device_owned_bytes"] == 0
        with pytest.raises(RuntimeError, match="closed"):
            batch.step_contacts(first, H)
        assert batch.step_contacts(second, H) == {17: True}
        third = batch.prepare_contacts(model, bindings, {pair: _law()})
        seen = []
        originals = [second.close, third.close]

        def failing_close():
            originals[0]()
            seen.append(0)
            raise RuntimeError("c6 close callback failure")

        def other_close():
            originals[1]()
            seen.append(1)

        with monkeypatch.context() as patch:
            patch.setattr(second, "close", failing_close)
            patch.setattr(third, "close", other_close)
            with pytest.raises(RuntimeError, match="close callback"):
                batch.close()
        assert sorted(seen) == [0, 1]
        assert all(
            plan.stats()["closed"] and plan.stats()["device_owned_bytes"] == 0
            for plan in (first, second, third)
        )
        assert batch.stats()["device_owned_bytes"] == 0
    finally:
        batch.close()


def test_r05_floating_model_cannot_masquerade_as_an_ordinary_tree_world():
    root = ph.RigidBodyModel(np.array([0.5]), np.array([[1 / 8, 1 / 8, 1 / 8]]), np.zeros(3))
    floating = ph.FloatingRobotModel(root, np.zeros(3))
    state = ph.FloatingRobotState(np.array([0, 0, 3 / 32, 0, 0, 0, 1], dtype=float), np.zeros(6))
    np.testing.assert_array_equal(
        ph.floating_kinematics(floating, state).position, [[0, 0, 3 / 32]]
    )
    with pytest.raises(TypeError, match="FixedRobotTree|expected"):
        resident.TyxWorldBatch([(17, floating)], "interpret")


@pytest.mark.tyx
def test_r05_rejected_touch_keeps_the_same_plans_previous_accepted_receipt(tyx_mode, tmp_path):
    program, world = _slider(q=1 / 32 + H, v=-1.0)
    with resident.TyxWorldBatch([(17, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({17: _action(program)})
        model, bindings, _, pair = _r01_bound(batch)
        with batch.prepare_contacts(model, bindings, {pair: _law()}, site_capacity=0) as plan:
            # Entry gap H > the established broad-phase engineering padding.
            assert batch.step_contacts(plan, H) == {17: True}
            before, receipt = batch.snapshot()[17], plan.last_accepted[17]
            assert before.tick == 1 and before.position == (1 / 32,) and before.velocity == (-1.0,)
            assert not receipt.sites
            retained = _json(receipt)
            # Exact touch has a finite material patch even though its spring force is zero.
            assert batch.step_contacts(plan, H) == {17: False}
            assert plan.last_status[17] == "SITE_CAPACITY_EXCEEDED"
            assert batch.snapshot()[17] == before
            assert _json(plan.last_accepted[17]) == retained
            assert _json(receipt) == retained
            _save(
                tmp_path,
                "r05-retained-receipt",
                dict(
                    mode=tyx_mode,
                    entry_gap_m=H,
                    before_reject=before,
                    last_accepted=receipt,
                    last_status=plan.last_status,
                    after=batch.snapshot(),
                    stats=batch.stats(),
                ),
            )


@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_r05_nonfinite_material_input_is_rejected_before_any_contact_attempt(value):
    key = ph.CollisionKey(17, 100, 11)
    with pytest.raises(ValueError, match="finite"):
        ph.ResidentTreeBinding(key, 17, 0, [[value, 0, 0]], topology=("explicit-invalid-input",))
    with pytest.raises(ValueError, match="finite"):
        ph.CollisionMotion([[0, value, 0]])

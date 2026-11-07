"""C7 mixed resident contact transactions; authoring only, UNEXECUTED.

Physical oracles use only declared inputs and the accepted independent C5 source.
Faults call the actual operation before throwing; no fabricated acceptance flags.
"""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from test_resident_affine_contacts import (
    H, WORLD, WINDOW_ATOL, _asset, _audit_sites, _audit_step, _box, _close,
    _contact_geometry, _control, _expected, _json, _placed_case, _save, _state, _world,
)

import phymeo as ph

KEYS = (503, 91, 1207, 6)


def _ordinary():
    model = ph.RobotModel(
        np.array([-1]), ("prismatic",), np.array([[0, 0, 3 / 32]]),
        np.array([[0, 0, 0, 1.0]]), np.array([[0, 0, 1.0]]),
        np.array([2.0]), np.zeros((1, 3)), np.full((1, 3), 1 / 8), np.zeros(3),
    )
    program = ph.DrivenRobotProgram(model, kp=np.zeros(1), kd=np.zeros(1), effort_limit=np.array([8.0]))
    world = ph.FixedRobotTree(program, program.initial_state(ph.RobotState(np.zeros(1), np.zeros(1))))
    control = program.inputs(np.zeros(1), np.zeros(1), feedforward=np.zeros(1), loads=ph.RobotInputs.zeros(1))
    return world, control


def _mixed(directory, *, delay=0):
    ordinary, ordinary_control = _ordinary()
    result = {503: dict(kind="ordinary", world=ordinary, control=ordinary_control)}
    for key, name in ((91, "A03"), (1207, "A05")):
        data = _placed_case(name)
        asset = _asset(directory / str(key), data)
        world = _world(asset, data, delay=delay)
        result[key] = dict(kind="affine", data=data, asset=asset, world=world, control=_control(world, asset, data))
    result[6] = dict(kind="free", world=ph.FreeSphere(2, 1 / 8, position=(0, 0, 2), gravity=(0, 0, -8)), control=(0, 0, 0))
    return result


def _ordinary_geometry(batch, world):
    a, b = ph.CollisionKey(world, 100, 11), ph.CollisionKey(world, 900, 1)
    local, static = _box([-1 / 8] * 3, [1 / 8] * 3), _box([-1 / 2, -1 / 2, -1 / 4], [1 / 2, 1 / 2, 0])
    first, second = ph.CollisionShape(a, "hull", tuple(range(101, 109))), ph.CollisionShape(b, "hull", tuple(range(201, 209)))
    bindings = (
        ph.ResidentTreeBinding(a, world, 0, local, topology=batch.snapshot([world])[world].topology),
        ph.ResidentStaticBinding(b, ph.CollisionMotion(static)),
    )
    return ph.CollisionModel((first, second)), bindings, {a: (first, 0, local), b: (second, None, static)}, (a, b)


def _geometry(batch, cases):
    shapes, bindings, sources, laws = [], [], {}, {}
    for key, item in cases.items():
        if item["kind"] == "free":
            continue
        built = _ordinary_geometry(batch, key) if item["kind"] == "ordinary" else _contact_geometry(batch, item["asset"], item["data"], world=key)
        model, bound, source, pair = built
        shapes.extend(model.shapes)
        bindings.extend(bound)
        sources[key] = source
        laws[pair] = ph.RobotContactMaterial(64, 0, 0)
    return ph.CollisionModel(tuple(shapes)), tuple(bindings), sources, laws


def _snapshot_geometry(batch):
    return _json(batch.tree_kinematics())


def _independent_initial(cases):
    return {key: ((item["data"]["s"].copy(), item["data"]["u"].copy()) if item["kind"] == "affine" else (np.zeros(1), np.zeros(1))) for key, item in cases.items() if item["kind"] != "free"}


def _advance_oracle(cases, previous):
    result = {}
    for key, (q, v) in previous.items():
        item = cases[key]
        acceleration = _expected(item["data"], q, v)["acceleration"] if item["kind"] == "affine" else np.array([32 * max(0.0, 1 / 32 - q[0])])
        next_v = v + H * acceleration
        result[key] = (q + H * next_v, next_v)
    return result


def _audit_mixed(batch, cases, entry, receipts, sources, expected):
    state, diag = batch.snapshot(), batch.tree_diagnostics()
    audits = {}
    for key, item in cases.items():
        checkpoint = state[key]
        assert checkpoint.tick == entry[key].tick + 1
        if item["kind"] == "free":
            n = checkpoint.tick
            _close(checkpoint.position, (0, 0, 2 - 4 * H * H * n * (n + 1)), atol=WINDOW_ATOL)
            _close(checkpoint.velocity, (0, 0, -8 * H * n), atol=WINDOW_ATOL)
            assert key not in receipts
            continue
        if item["kind"] == "affine":
            audits[key] = _audit_step(batch, key, item["asset"], item["data"], entry[key], receipts[key], sources[key])
            q, v = _state(item["asset"], item["data"], checkpoint)
        else:
            q, v = np.asarray(checkpoint.position), np.asarray(checkpoint.velocity)
            audits[key] = _audit_sites(receipts[key], diag[key], sources[key])
            normal = 64 * max(0.0, 1 / 32 - entry[key].position[0])
            _close(receipts[key].contact_generalized, [normal])
            _close(receipts[key].law_wrench_fp64, [[0, 0, normal, 0, 0, 0]])
            assert checkpoint.control == entry[key].control
        _close(q, expected[key][0], atol=WINDOW_ATOL)
        _close(v, expected[key][1], atol=WINDOW_ATOL)
    return audits


def _mixed_window(cases, mode, directory, *, replay=False):
    directory.mkdir(parents=True, exist_ok=True)
    rows, repeated, saved = [], [], None
    expected = _independent_initial(cases)
    with ph.TyxWorldBatch([(key, item["world"]) for key, item in cases.items()], mode, cache_root=directory) as batch:
        batch.set_controls({key: item["control"] for key, item in cases.items()})
        model, bindings, sources, laws = _geometry(batch, cases)
        with batch.prepare_contacts(model, bindings, laws) as plan:
            try:
                for index in range(40):
                    entry = batch.snapshot()
                    expected = _advance_oracle(cases, expected)
                    assert batch.step_contacts(plan, H) == dict.fromkeys(cases, True)
                    receipts = dict(plan.last_accepted)
                    audit = _audit_mixed(batch, cases, entry, receipts, sources, expected)
                    rows.append(dict(state=batch.snapshot(), geometry=batch.tree_kinematics(), diagnostic=batch.tree_diagnostics(), receipts=receipts, expected=expected, audit=audit, stats=batch.stats()))
                    if index == 18:
                        saved = batch.snapshot()
                if replay:
                    batch.restore(saved)
                    assert batch.snapshot() == saved
                    for index in range(19, 40):
                        assert batch.step_contacts(plan, H) == dict.fromkeys(cases, True)
                        row = dict(state=batch.snapshot(), geometry=batch.tree_kinematics(), diagnostic=batch.tree_diagnostics(), receipts=dict(plan.last_accepted), stats=batch.stats())
                        repeated.append(row)
                        assert row["state"] == rows[index]["state"]
                        assert _json(row["geometry"]) == _json(rows[index]["geometry"])
                        assert _json(row["receipts"]) == _json(rows[index]["receipts"])
                    assert len(repeated) == 21
                assert batch.stats()["submitted_kernels"] > 0
            finally:
                _save(directory, "c7-mixed-window", dict(mode=mode, h_s=H, declarations={key: item.get("data", item["kind"]) for key, item in cases.items()}, inputs={key: item["control"] for key, item in cases.items()}, collision_model=model, bindings=bindings, rows=rows, checkpoint_19=saved, replay=repeated, last_accepted=plan.last_accepted, diagnostic=batch.tree_diagnostics(), final=batch.snapshot(), stats=batch.stats(), plan_stats=plan.stats()))
    return rows


@pytest.mark.tyx
def test_c7_w1_w4_reversed_worlds_duplicate_local_ids_and_21_step_replay(tyx_mode, tmp_path):
    cases = _mixed(tmp_path / "assets")
    separate = {key: _mixed_window({key: item}, tyx_mode, tmp_path / f"w1-{key}") for key, item in cases.items()}
    combined = _mixed_window(dict(reversed(tuple(cases.items()))), tyx_mode, tmp_path / "w4", replay=True)
    for index, row in enumerate(combined):
        for key in cases:
            assert row["state"][key] == separate[key][index]["state"][key]
            if key in row["receipts"]:
                assert _json(row["receipts"][key]) == _json(separate[key][index]["receipts"][key])


@pytest.mark.tyx
@pytest.mark.parametrize("warm", (0, 1))
def test_c7_affine_local_capacity_keeps_both_coordinate_banks_and_neighbors_advance(warm, tyx_mode, tmp_path):
    cases = _mixed(tmp_path / "assets", delay=1)
    with ph.TyxWorldBatch([(key, item["world"]) for key, item in cases.items()], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({key: item["control"] for key, item in cases.items()})
        model, bindings, sources, laws = _geometry(batch, cases)
        with batch.prepare_contacts(model, bindings, laws) as full:
            for _ in range(warm):
                assert batch.step_contacts(full, H) == dict.fromkeys(cases, True)
            before, fk, previous, energy = batch.snapshot(), _snapshot_geometry(batch), _json(full.last_accepted), batch.summary()
            with batch.prepare_contacts(model, bindings, laws, site_capacity={503: 128, 91: 0, 1207: 128}) as limited:
                assert batch.step_contacts(limited, H) == {503: True, 91: False, 1207: True, 6: True}
                assert limited.last_status[91] == "SITE_CAPACITY_EXCEEDED"
                assert batch.stats()["affine_tree_status_codes"][91] == "SITE_CAPACITY_EXCEEDED"
                assert batch.snapshot()[91] == before[91]
                assert _snapshot_geometry(batch)["91"] == fk["91"]
                assert batch.summary()[91] == energy[91]
                assert 91 not in limited.last_accepted
                assert _json(full.last_accepted) == previous
                for key in (503, 1207, 6):
                    assert batch.snapshot()[key].tick == before[key].tick + 1
                paused = batch.snapshot()[91]
                assert batch.step_contacts(full, H) == dict.fromkeys(cases, True)
                assert full.last_accepted[91].entry_state == paused
                # First delayed selected drive is zero; the new held 7 N remains
                # pending in history until this world's own accepted tick.
                if warm == 0:
                    _close(full.last_accepted[91].actual_drive, [0])
                    _close(batch.snapshot()[91].history, [(0, 0, 7)])
                _save(tmp_path, "c7-capacity-local", dict(warm=warm, before=before, prior_receipts=previous, limited_status=limited.last_status, retry=full.last_accepted, final=batch.snapshot(), stats=batch.stats()))


@pytest.mark.tyx
def test_c7_zero_force_geometric_site_still_counts_and_affine9_is_not_constraint7_or8(tyx_mode, tmp_path):
    from phymeo.backends._resident_affine_storage import AFFINE_STATUS_NAMES
    assert AFFINE_STATUS_NAMES[7] == "AFFINE_CONSTRAINT_RESIDUAL"
    assert AFFINE_STATUS_NAMES[8] == "AFFINE_FULL_EQUATION_RESIDUAL"
    assert AFFINE_STATUS_NAMES[9] == "SITE_CAPACITY_EXCEEDED"
    data = _placed_case("A01F")
    data["u"][:] = 1
    asset = _asset(tmp_path, data)
    with ph.TyxWorldBatch([(WORLD, _world(asset, data))], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: _control(_world(asset, data), asset, data)})
        model, bindings, _, pair = _contact_geometry(batch, asset, data)
        before = batch.snapshot()
        law = ph.RobotContactMaterial(64, 4, 0)
        with batch.prepare_contacts(model, bindings, {pair: law}, site_capacity=0) as cap:
            assert batch.step_contacts(cap, H) == {WORLD: False}
            assert cap.last_status[WORLD] == "SITE_CAPACITY_EXCEEDED"
            assert batch.stats()["affine_tree_status_codes"][WORLD] == "SITE_CAPACITY_EXCEEDED"
            assert batch.snapshot() == before
        with batch.prepare_contacts(model, bindings, {pair: law}, site_capacity=1) as full:
            assert batch.step_contacts(full, H) == {WORLD: True}
            receipt = full.last_accepted[WORLD]
            assert len(receipt.sites) == 1
            np.testing.assert_array_equal(receipt.contact_generalized, [0, 0])
            np.testing.assert_array_equal(receipt.law_wrench_fp64, np.zeros((2, 6)))
            _close(receipt.accepted_state.velocity, [1])
            _close(receipt.accepted_state.full_velocity, [1, 2])
            _save(tmp_path, "c7-zero-force-capacity", dict(input=data, law=law, before=before, receipt=receipt, stats=batch.stats()))


@pytest.mark.tyx
def test_c7_prior_same_plan_receipt_survives_exact_touch_capacity_rejection(tyx_mode, tmp_path):
    data = _placed_case("A01F")
    data["s"][:] = 3 / 128 + H
    data["u"][:] = -1
    asset = _asset(tmp_path, data)
    world = _world(asset, data)
    with ph.TyxWorldBatch([(WORLD, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: _control(world, asset, data)})
        model, bindings, _, pair = _contact_geometry(batch, asset, data)
        with batch.prepare_contacts(model, bindings, {pair: ph.RobotContactMaterial(64, 0, 0)}, site_capacity=0) as plan:
            assert batch.step_contacts(plan, H) == {WORLD: True}
            previous = plan.last_accepted[WORLD]
            assert not previous.sites and batch.snapshot()[WORLD].position == (3 / 128,)
            before = batch.snapshot()
            assert batch.step_contacts(plan, H) == {WORLD: False}
            assert batch.snapshot() == before
            assert _json(plan.last_accepted[WORLD]) == _json(previous)
            _save(tmp_path, "c7-retained-receipt", dict(before=before, receipt=previous, final=batch.snapshot(), status=plan.last_status))


@pytest.mark.tyx
def test_c7_candidate_capacity_is_global_and_empty_affine_binding_is_not_an_error(tyx_mode, tmp_path):
    cases = _mixed(tmp_path / "assets")
    with ph.TyxWorldBatch([(key, item["world"]) for key, item in cases.items()], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({key: item["control"] for key, item in cases.items()})
        model, bindings, _, laws = _geometry(batch, cases)
        before, stats = batch.snapshot(), batch.stats()
        with batch.prepare_contacts(model, bindings, laws, candidate_capacity=0) as plan:
            with pytest.raises(ValueError, match="capacity"):
                batch.step_contacts(plan, H)
            assert batch.snapshot() == before and not plan.last_accepted
            assert batch.stats()["state_epoch"] == stats["state_epoch"]
            assert batch.stats()["current_buffer"] == stats["current_buffer"]
        with batch.prepare_contacts(ph.CollisionModel(()), (), {}) as empty:
            assert batch.step_contacts(empty, H) == dict.fromkeys(cases, True)
            assert not empty.last_accepted
            # No contact is declared; original A03 motor 7 / mass 7 still steps.
            _close(batch.snapshot()[91].velocity, [H])
            _save(tmp_path, "c7-global-cap-empty", dict(before=before, final=batch.snapshot(), stats=batch.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("problem", ("graph", "group", "world", "friction", "law", "dynamic_pair", "missing", "sampling"))
def test_c7_invalid_affine_contact_configuration_allocates_nothing(problem, tyx_mode, tmp_path):
    data = _placed_case("A03")
    asset = _asset(tmp_path / "chain", data)
    flat = _asset(tmp_path / "flat", _placed_case("A03_flat"))
    with ph.TyxWorldBatch([(WORLD, _world(asset, data)), (7, _world(flat, _placed_case("A03_flat")))], tyx_mode, cache_root=tmp_path) as batch:
        model, bindings, _, pair = _contact_geometry(batch, asset, data)
        laws, options = {pair: ph.RobotContactMaterial(64, 0, 0)}, {}
        if problem == "graph":
            bindings = (replace(bindings[0], topology=batch.snapshot()[7].topology), bindings[1])
        elif problem == "group":
            bindings = (replace(bindings[0], group_index=3), bindings[1])
        elif problem == "world":
            bindings = (replace(bindings[0], key=ph.CollisionKey(999, 102, 11), world=999), bindings[1])
        elif problem == "friction":
            laws[pair] = ph.RobotContactMaterial(64, 0, 1 / 2)
        elif problem == "law":
            laws[pair] = ph.NormalLayer(1 / 32, 64)
        elif problem == "dynamic_pair":
            bindings = (bindings[0], ph.ResidentAffineTreeBinding(pair[1], WORLD, 0, bindings[1].motion.start, topology=batch.snapshot()[WORLD].topology))
        elif problem == "missing":
            laws = {}
        else:
            options["sampling"] = "swept"
        before, stats = batch.snapshot(), batch.stats()
        with pytest.raises((ValueError, TypeError)):
            batch.prepare_contacts(model, bindings, laws, **options)
        assert batch.snapshot() == before
        assert batch.stats()["device_allocations"] == stats["device_allocations"]


@pytest.mark.tyx
@pytest.mark.parametrize("warm,phase", tuple(product((0, 1), ("upload", "candidate", "receipt"))))
def test_c7_real_late_failure_retains_full_k_physics_and_retries_identically(warm, phase, tyx_mode, tmp_path, monkeypatch):
    cases = _mixed(tmp_path / "assets", delay=1)
    worlds = [(key, item["world"]) for key, item in cases.items()]
    with ph.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "failed") as batch, ph.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "control") as control:
        commands = {key: item["control"] for key, item in cases.items()}
        batch.set_controls(commands)
        control.set_controls(commands)
        model, bindings, _, laws = _geometry(batch, cases)
        other_model, other_bindings, _, other_laws = _geometry(control, cases)
        with batch.prepare_contacts(model, bindings, laws) as plan, control.prepare_contacts(other_model, other_bindings, other_laws) as reference:
            for _ in range(warm):
                assert batch.step_contacts(plan, H) == control.step_contacts(reference, H) == dict.fromkeys(cases, True)
            before, fk, stats, energy = batch.snapshot(), _snapshot_geometry(batch), batch.stats(), batch.summary()
            prior, statuses = _json(plan.last_accepted), dict(plan.last_status)
            # Exercise actual contact upload, after full entry mechanics/law, not
            # a pre-entry dummy error. Candidate/receipt wrappers also call first.
            method = {"upload": "_upload_contact", "candidate": "_read_candidate", "receipt": "_finish_attempt"}[phase]
            original, calls = getattr(plan, method), []
            def call_then_fail(*args, **kwargs):
                value = original(*args, **kwargs)
                calls.append(value)
                raise RuntimeError("c7 real " + phase + " completed before fault")
            with monkeypatch.context() as patch:
                patch.setattr(plan, method, call_then_fail)
                with pytest.raises(RuntimeError, match="c7 real"):
                    batch.step_contacts(plan, H)
            assert calls and batch.snapshot() == before
            assert _snapshot_geometry(batch) == fk
            assert batch.summary() == energy
            assert _json(plan.last_accepted) == prior and dict(plan.last_status) == statuses
            assert batch.stats()["state_epoch"] == stats["state_epoch"]
            assert batch.stats()["current_buffer"] == stats["current_buffer"]
            plan.close()
            with batch.prepare_contacts(model, bindings, laws) as retry:
                assert batch.step_contacts(retry, H) == control.step_contacts(reference, H) == dict.fromkeys(cases, True)
                assert batch.snapshot() == control.snapshot()
                assert _json(retry.last_accepted) == _json(reference.last_accepted)
                _save(tmp_path, "c7-real-fault", dict(warm=warm, phase=phase, before=before, before_geometry=fk, prior=prior, retry=retry.last_accepted, final=batch.snapshot(), reference=control.snapshot(), stats=batch.stats()))


@pytest.mark.tyx
def test_c7_second_real_scratch_allocation_failure_closes_owned_resources(tyx_mode, tmp_path, monkeypatch):
    from phymeo.backends._resident_contact_plan import ResidentContactPlan
    data = _placed_case("A01F")
    asset = _asset(tmp_path, data)
    with ph.TyxWorldBatch([(WORLD, _world(asset, data))], tyx_mode, cache_root=tmp_path) as batch:
        model, bindings, _, pair = _contact_geometry(batch, asset, data)
        before, owner, calls = batch.snapshot(), [], []
        original = ResidentContactPlan._buffer
        def fail_second(self, *args, **kwargs):
            value = original(self, *args, **kwargs)
            owner.append(self)
            calls.append(value)
            if len(calls) == 2:
                raise MemoryError("c7 after actual second scratch allocation")
            return value
        with monkeypatch.context() as patch:
            patch.setattr(ResidentContactPlan, "_buffer", fail_second)
            with pytest.raises(MemoryError, match="actual second"):
                batch.prepare_contacts(model, bindings, {pair: ph.RobotContactMaterial(64, 0, 0)})
        assert len(calls) == 2 and owner[0] is owner[1]
        assert owner[0].stats()["closed"] and owner[0].stats()["device_owned_bytes"] == 0
        assert batch.snapshot() == before
        with batch.prepare_contacts(model, bindings, {pair: ph.RobotContactMaterial(64, 0, 0)}) as retry:
            assert batch.step_contacts(retry, H) == {WORLD: True}


@pytest.mark.tyx
@pytest.mark.parametrize("warm", (0, 1))
def test_c7_same_tick_restore_inside_candidate_read_invalidates_attempt(warm, tyx_mode, tmp_path, monkeypatch):
    data = _placed_case("A03")
    asset = _asset(tmp_path, data)
    world = _world(asset, data, delay=1)
    with ph.TyxWorldBatch([(WORLD, world)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: _control(world, asset, data)})
        model, bindings, _, pair = _contact_geometry(batch, asset, data)
        with batch.prepare_contacts(model, bindings, {pair: ph.RobotContactMaterial(64, 0, 0)}) as plan:
            for _ in range(warm):
                assert batch.step_contacts(plan, H) == {WORLD: True}
            saved, prior, epoch = batch.snapshot(), _json(plan.last_accepted), batch.stats()["state_epoch"]
            original, restored = plan._read_candidate, []
            def restore_after_actual_read(*args, **kwargs):
                candidate = original(*args, **kwargs)
                batch.restore(saved)
                restored.append(batch.stats())
                return candidate
            with monkeypatch.context() as patch:
                patch.setattr(plan, "_read_candidate", restore_after_actual_read)
                with pytest.raises((ValueError, RuntimeError), match="stale|epoch|changed|entry"):
                    batch.step_contacts(plan, H)
            assert len(restored) == 1 and restored[0]["state_epoch"] == epoch + 1
            assert batch.snapshot() == saved and _json(plan.last_accepted) == prior
            plan.close()
            with batch.prepare_contacts(model, bindings, {pair: ph.RobotContactMaterial(64, 0, 0)}) as fresh:
                assert batch.step_contacts(fresh, H) == {WORLD: True}
                assert fresh.last_accepted[WORLD].entry_state == saved[WORLD]


@pytest.mark.tyx
def test_c7_foreign_plan_graph_checkpoint_and_readonly_receipt(tyx_mode, tmp_path):
    chain, flat = _placed_case("A03"), _placed_case("A03_flat")
    asset, flat_asset = _asset(tmp_path / "chain", chain), _asset(tmp_path / "flat", flat)
    with ph.TyxWorldBatch([(WORLD, _world(asset, chain))], tyx_mode, cache_root=tmp_path / "a") as a, ph.TyxWorldBatch([(WORLD, _world(flat_asset, flat))], tyx_mode, cache_root=tmp_path / "b") as b:
        a.set_controls({WORLD: _control(_world(asset, chain), asset, chain)})
        b.set_controls({WORLD: _control(_world(flat_asset, flat), flat_asset, flat)})
        model, bindings, _, pair = _contact_geometry(a, asset, chain)
        with a.prepare_contacts(model, bindings, {pair: ph.RobotContactMaterial(64, 0, 0)}) as plan:
            before_a, before_b = a.snapshot(), b.snapshot()
            with pytest.raises((ValueError, RuntimeError), match="owner|batch|belong"):
                b.step_contacts(plan, H)
            with pytest.raises((ValueError, TypeError), match="topology|graph|source"):
                b.restore(before_a)
            assert a.snapshot() == before_a and b.snapshot() == before_b
            assert a.step_contacts(plan, H) == {WORLD: True}
            receipt, serialized = plan.last_accepted[WORLD], _json(plan.last_accepted[WORLD])
            meta, diag = receipt.diagnostics, receipt.diagnostics["affine_diagnostic"]
            arrays = [receipt.law_wrench_fp64, receipt.uploaded_wrench_fp32, receipt.input_external, receipt.contact_generalized, receipt.selected_drive, receipt.actual_drive, meta["projection_matrix"], meta["offset_vector"], meta["constraint_rows"], diag.full_acceleration, diag.row_multipliers, receipt.entry_geometry.link_rotation, receipt.accepted_geometry.independent_position]
            for site in receipt.sites:
                arrays.extend(getattr(site, side + suffix) for side in ("first", "second") for suffix in ("_force_n", "_velocity_m_s", "_application_displacement_m", "_material_displacement_m"))
            for array in arrays:
                assert not np.asarray(array).flags.writeable
                with pytest.raises(ValueError):
                    np.asarray(array).setflags(write=True)
            with pytest.raises(TypeError):
                meta["coordinate_kind"] = "ordinary"
            before, stats = a.snapshot(), a.stats()
            for _ in range(2):
                assert _json(plan.last_accepted[WORLD]) == serialized
                a.tree_kinematics()
                a.tree_diagnostics()
            assert a.snapshot() == before and a.stats()["state_epoch"] == stats["state_epoch"]
            assert a.step_contacts(plan, H) == {WORLD: True}
            assert _json(receipt) == serialized


@pytest.mark.tyx
def test_c7_close_and_rebuild_from_public_affine_checkpoint(tyx_mode, tmp_path):
    data = _placed_case("A05")
    asset = _asset(tmp_path, data)
    world = _world(asset, data, delay=1)
    batch = ph.TyxWorldBatch([(WORLD, world)], tyx_mode, cache_root=tmp_path / "original")
    try:
        batch.set_controls({WORLD: _control(world, asset, data)})
        model, bindings, _, pair = _contact_geometry(batch, asset, data)
        plan = batch.prepare_contacts(model, bindings, {pair: ph.RobotContactMaterial(64, 0, 0)})
        assert batch.step_contacts(plan, H) == {WORLD: True}
        saved = batch.snapshot()
        assert batch.step_contacts(plan, H) == {WORLD: True}
        expected, receipt = batch.snapshot(), _json(plan.last_accepted[WORLD])
    finally:
        batch.close()
        batch.close()
    assert plan.stats()["closed"] and plan.stats()["device_owned_bytes"] == 0
    assert batch.stats()["device_owned_bytes"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        batch.step_contacts(plan, H)
    with ph.TyxWorldBatch([(WORLD, world)], tyx_mode, cache_root=tmp_path / "rebuilt") as rebuilt:
        rebuilt.restore(saved)
        assert rebuilt.snapshot() == saved
        new_model, new_bindings, _, new_pair = _contact_geometry(rebuilt, asset, data)
        with rebuilt.prepare_contacts(new_model, new_bindings, {new_pair: ph.RobotContactMaterial(64, 0, 0)}) as new:
            assert rebuilt.step_contacts(new, H) == {WORLD: True}
            assert rebuilt.snapshot() == expected
            assert _json(new.last_accepted[WORLD]) == receipt

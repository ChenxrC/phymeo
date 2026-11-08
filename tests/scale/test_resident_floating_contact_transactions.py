"""C10 all-bank contact transactions and mixed worlds, AUTHORING_UNEXECUTED.

Failure injection calls the actual operation before throwing. No forged success
flags, changed physics budgets, source-private state repair, or product oracle.
"""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from test_resident_floating_contacts import (
    WINDOW_ATOL,
    WORLD,
    H,
    _audit,
    _close,
    _geometry,
    _item,
    _json,
    _reference,
    _save,
    _selected,
    _window,
)

import phymeo as ph

KEYS = (503, 91, 1207, 6)


def _four():
    ref = _reference()
    inputs = (
        ref.case("F01"),
        ref.case("F02", variant="motor"),
        ref.case("F08", n=9),
        ref.case("F08", n=16),
    )
    return {key: _item(data, delay=1) for key, data in zip(KEYS, inputs, strict=True)}


def _combined(batch, items, *, reverse=False):
    shapes, bindings, exclusions, sources, laws = [], [], [], {}, {}
    for world, item in items.items():
        model, bound, source, material = _geometry(batch, item, world=world, reverse=reverse)
        shapes.extend(model.shapes)
        bindings.extend(bound)
        exclusions.extend(model.excluded)
        sources[world] = source
        laws.update(material)
    if reverse:
        shapes.reverse()
        bindings.reverse()
    return ph.CollisionModel(tuple(shapes), tuple(exclusions)), tuple(bindings), sources, laws


def _snapshot(batch, plan):
    return dict(
        state=batch.snapshot(),
        geometry=_json(batch.tree_kinematics()),
        summary=batch.summary(),
        receipts=_json(plan.last_accepted),
        status=dict(plan.last_status),
        epoch=batch.stats()["state_epoch"],
        bank=batch.stats()["current_buffer"],
    )


def _assert_retained(batch, plan, before, *, epoch=True):
    after = _snapshot(batch, plan)
    for field in ("state", "geometry", "summary", "receipts", "status", "bank"):
        assert after[field] == before[field]
    if epoch:
        assert after["epoch"] == before["epoch"]


def _record(batch, plan):
    return dict(
        state=batch.snapshot(),
        geometry=batch.tree_kinematics(),
        receipts=dict(plan.last_accepted),
        summary=batch.summary(),
    )


def _batch_window(items, mode, directory, *, reverse=False):
    directory.mkdir(parents=True, exist_ok=True)
    rows, replay, saved, failure = [], [], None, None
    expected = {w: (i["data"]["q"].copy(), i["data"]["v"].copy()) for w, i in items.items()}
    declarations = [(w, i["tree"]) for w, i in items.items()]
    if reverse:
        declarations.reverse()
    archive = dict(
        inputs={w: i["data"] for w, i in items.items()},
        commands={w: i["command"] for w, i in items.items()},
        mode=mode,
        dt_s=H,
        steps=40,
        restore_tick=19,
        replay_steps=21,
        rows=rows,
        replay=replay,
        reverse=reverse,
    )
    with ph.TyxWorldBatch(declarations, mode, cache_root=directory) as batch:
        batch.set_controls(archive["commands"])
        model, bindings, sources, laws = _combined(batch, items, reverse=reverse)
        archive.update(model=model, bindings=bindings, laws=laws)
        try:
            with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as plan:
                for index in range(40):
                    assert batch.step_contacts(plan, H) == dict.fromkeys(items, True)
                    audits = {}
                    row = _record(batch, plan)
                    row["audit"] = audits
                    rows.append(row)
                    for w, item in items.items():
                        audits[w] = _audit(batch, item, plan.last_accepted[w], sources[w], world=w)
                        q, v, truth = _reference().advance(
                            item["data"],
                            *expected[w],
                            h=H,
                            drive=_selected(item, index),
                            external_packet=item["external"],
                        )
                        expected[w] = q, v
                        after = plan.last_accepted[w].accepted_state
                        _close(after.position, q, atol=WINDOW_ATOL)
                        _close(after.velocity, v, atol=WINDOW_ATOL)
                        audits[w]["window"] = dict(q=q, v=v, truth=truth)
                    if index == 18:
                        saved = batch.snapshot()
                batch.restore(saved)
                assert batch.snapshot() == saved
                for index in range(19, 40):
                    assert batch.step_contacts(plan, H) == dict.fromkeys(items, True)
                    row = _record(batch, plan)
                    replay.append(row)
                    for field in ("state", "geometry", "receipts", "summary"):
                        assert _json(row[field]) == _json(rows[index][field])
                archive.update(saved=saved, plan_stats=plan.stats())
                assert plan.stats()["contact_kernel_launches"] > 0
                assert plan.stats()["floating_bounds_kernel_launches"] > 0
        except BaseException as exc:
            failure = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            archive.update(
                failure=failure,
                retained=batch.snapshot(),
                retained_geometry=batch.tree_kinematics(),
                stats=batch.stats(),
            )
            _save(directory, "floating-contact-batch", archive)
    return rows


@pytest.mark.tyx
def test_f10_single_four_thirtytwo_worlds_and_full_restore21(tyx_mode, tmp_path):
    items = _four()
    singles = {
        w: _batch_window({w: item}, tyx_mode, tmp_path / ("w1-" + str(w)))
        for w, item in items.items()
    }
    four = _batch_window(items, tyx_mode, tmp_path / "w4", reverse=True)
    expanded = dict(items)
    for i in range(28):
        original = items[KEYS[i % 4]]
        expanded[10001 + 97 * i] = _item(original["data"], delay=1)
    thirtytwo = _batch_window(expanded, tyx_mode, tmp_path / "w32", reverse=True)
    for w in KEYS:
        for index in range(40):
            for field in ("state", "geometry", "receipts", "summary"):
                assert _json(singles[w][index][field][w]) == _json(four[index][field][w])
                assert _json(four[index][field][w]) == _json(thirtytwo[index][field][w])
    # Additional W32 keys retain their own world identities and full independent
    # audits; no world/owner field is deleted from the saved public records.
    assert set(thirtytwo[-1]["state"]) == set(expanded)


@pytest.mark.tyx
@pytest.mark.parametrize("warm,limited_world", tuple(product((0, 1), (503, 6))))
def test_f10_capacity_retains_root_and_last_body_all_banks(warm, limited_world, tyx_mode, tmp_path):
    items = _four()
    with ph.TyxWorldBatch(
        [(w, i["tree"]) for w, i in items.items()], tyx_mode, cache_root=tmp_path
    ) as batch:
        commands = {w: i["command"] for w, i in items.items()}
        batch.set_controls(commands)
        model, bindings, sources, laws = _combined(batch, items)
        with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as full:
            for _ in range(warm):
                assert batch.step_contacts(full, H) == dict.fromkeys(items, True)
            capacities = dict.fromkeys(items, 128)
            capacities[limited_world] = 0
            with batch.prepare_contacts(
                model, bindings, laws, tree_dynamic_pairs=True, site_capacity=capacities
            ) as limited:
                before = _snapshot(batch, full)
                assert batch.step_contacts(limited, H) == {w: w != limited_world for w in items}
                assert limited.last_status[limited_world] == "SITE_CAPACITY_EXCEEDED"
                assert batch.snapshot()[limited_world] == before["state"][limited_world]
                assert (
                    _json(batch.tree_kinematics()[limited_world])
                    == before["geometry"][str(limited_world)]
                )
                assert batch.summary()[limited_world] == before["summary"][limited_world]
                assert limited_world not in limited.last_accepted
                assert _json(full.last_accepted) == before["receipts"]
                for w in items:
                    assert batch.snapshot()[w].tick == warm + (w != limited_world)
                retained = batch.snapshot()[limited_world]
                assert batch.step_contacts(full, H) == dict.fromkeys(items, True)
                assert full.last_accepted[limited_world].entry_state == retained
                audits = {
                    w: _audit(batch, i, full.last_accepted[w], sources[w], world=w)
                    for w, i in items.items()
                }
                _save(
                    tmp_path,
                    "floating-contact-capacity",
                    dict(
                        inputs={w: i["data"] for w, i in items.items()},
                        commands=commands,
                        warm=warm,
                        limited_world=limited_world,
                        before=before,
                        limited_status=limited.last_status,
                        retry=full.last_accepted,
                        audit=audits,
                        retained=batch.snapshot(),
                        geometry=batch.tree_kinematics(),
                        stats=batch.stats(),
                    ),
                )


@pytest.mark.tyx
@pytest.mark.parametrize("delay,cap", tuple(product((0, 1), (1 / 8, 8.0))))
def test_f10_drive_delay_cap_and_immediate_external_full_response(delay, cap, tyx_mode, tmp_path):
    data = _reference().case("F08_branch")
    # Same three-axis control declaration as C9 P06; contact replaces no external
    # input. Only this new four-step holdout enables nonzero PD, N <= 4.
    data["kp"], data["kd"] = np.array([16.0, 8.0, 4.0]), np.array([2.0, 1.0, 0.5])
    data["caps"] = np.full(3, cap)
    data["drive"] = np.column_stack((np.full(3, 1 / 8), np.full(3, -1 / 16), [1.0, -1.0, 0.5]))
    data["effort"][:] = [1 / 32, -1 / 32, 1 / 32]
    data["force"][-1] = [1 / 8, -1 / 16, 1 / 32]
    data["torque"][0] = [-1 / 32, 1 / 64, 0]
    rows = _window(_item(data, delay=delay), tyx_mode, tmp_path, steps=4)
    for row in rows:
        assert np.max(abs(row["receipt"].input_external)) > 0
        assert np.all(abs(row["receipt"].actual_drive) <= cap + 5e-5)
    if delay:
        initial_target = np.column_stack(
            (np.asarray(data["q"][7:], dtype=np.float32), np.zeros((3, 2)))
        )
        _close(rows[0]["receipt"].selected_drive, initial_target, atol=0, rtol=0)
        _close(rows[1]["receipt"].selected_drive, data["drive"], atol=0, rtol=0)


@pytest.mark.tyx
@pytest.mark.parametrize("phase,warm", tuple(product(("upload", "candidate", "receipt"), (0, 1))))
def test_f10_actual_late_fault_retains_every_bank_and_retry_matches(
    phase, warm, tyx_mode, tmp_path, monkeypatch
):
    items = {w: i for w, i in _four().items() if w in (503, 91)}
    declarations = [(w, i["tree"]) for w, i in items.items()]
    with (
        ph.TyxWorldBatch(declarations, tyx_mode, cache_root=tmp_path / "failure") as batch,
        ph.TyxWorldBatch(declarations, tyx_mode, cache_root=tmp_path / "control") as control,
    ):
        commands = {w: i["command"] for w, i in items.items()}
        batch.set_controls(commands)
        control.set_controls(commands)
        model, bindings, _, laws = _combined(batch, items)
        other, other_bindings, _, other_laws = _combined(control, items)
        with (
            batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as plan,
            control.prepare_contacts(
                other, other_bindings, other_laws, tree_dynamic_pairs=True
            ) as reference,
        ):
            for _ in range(warm):
                assert (
                    batch.step_contacts(plan, H)
                    == control.step_contacts(reference, H)
                    == dict.fromkeys(items, True)
                )
            before = _snapshot(batch, plan)
            method = dict(
                upload="_upload_contact", candidate="_read_candidate", receipt="_finish_attempt"
            )[phase]
            original, calls = getattr(plan, method), []

            def real_then_fail(*args, **kwargs):
                result = original(*args, **kwargs)
                calls.append(result)
                raise RuntimeError("c10 actual " + phase + " before publication")

            with monkeypatch.context() as patch:
                patch.setattr(plan, method, real_then_fail)
                with pytest.raises(RuntimeError, match="c10 actual"):
                    batch.step_contacts(plan, H)
            assert calls
            _assert_retained(batch, plan, before)
            plan.close()
            with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as retry:
                assert (
                    batch.step_contacts(retry, H)
                    == control.step_contacts(reference, H)
                    == dict.fromkeys(items, True)
                )
                actual, expected = _record(batch, retry), _record(control, reference)
                assert _json(actual) == _json(expected)
                _save(
                    tmp_path,
                    "floating-contact-late-fault",
                    dict(
                        inputs={w: i["data"] for w, i in items.items()},
                        commands=commands,
                        warm=warm,
                        phase=phase,
                        before=before,
                        retry=actual,
                        control=expected,
                        stats=batch.stats(),
                    ),
                )


@pytest.mark.tyx
@pytest.mark.parametrize("warm", (0, 1))
def test_f10_restore_after_real_candidate_rejects_stale_attempt(
    warm, tyx_mode, tmp_path, monkeypatch
):
    item = _item(_reference().case("F04"), delay=1)
    with ph.TyxWorldBatch([(WORLD, item["tree"])], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: item["command"]})
        model, bindings, _, laws = _geometry(batch, item)
        with batch.prepare_contacts(model, bindings, laws) as plan:
            for _ in range(warm):
                assert batch.step_contacts(plan, H) == {WORLD: True}
            before, original, calls = _snapshot(batch, plan), plan._read_candidate, []

            def read_then_restore(*args, **kwargs):
                result = original(*args, **kwargs)
                batch.restore(before["state"])
                calls.append(batch.stats()["state_epoch"])
                return result

            with monkeypatch.context() as patch:
                patch.setattr(plan, "_read_candidate", read_then_restore)
                with pytest.raises((ValueError, RuntimeError), match="stale|epoch|changed|entry"):
                    batch.step_contacts(plan, H)
            assert calls == [before["epoch"] + 1]
            assert batch.snapshot() == before["state"]
            assert _json(batch.tree_kinematics()) == before["geometry"]
            assert _json(plan.last_accepted) == before["receipts"]
            plan.close()
            with batch.prepare_contacts(model, bindings, laws) as retry:
                assert batch.step_contacts(retry, H) == {WORLD: True}
                assert retry.last_accepted[WORLD].entry_state == before["state"][WORLD]
                _save(
                    tmp_path,
                    "floating-contact-stale",
                    dict(
                        input=item["data"],
                        command=item["command"],
                        warm=warm,
                        before=before,
                        retry=_record(batch, retry),
                        stats=batch.stats(),
                    ),
                )


@pytest.mark.tyx
def test_f10_partial_allocation_and_global_candidate_cap_clean_up(tyx_mode, tmp_path, monkeypatch):
    from phymeo.backends._resident_contact_plan import ResidentContactPlan

    item = _item(_reference().case("F01"))
    with ph.TyxWorldBatch([(WORLD, item["tree"])], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: item["command"]})
        model, bindings, _, laws = _geometry(batch, item)
        original, calls, owners = ResidentContactPlan._buffer, [], []
        before = batch.snapshot()

        def allocate_then_fail(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            calls.append(result)
            owners.append(self)
            if len(calls) == 2:
                raise MemoryError("c10 actual second contact scratch allocation")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(ResidentContactPlan, "_buffer", allocate_then_fail)
            with pytest.raises(MemoryError, match="actual second"):
                batch.prepare_contacts(model, bindings, laws)
        assert len(calls) == 2 and owners[0] is owners[1]
        assert owners[0].stats()["closed"] and owners[0].stats()["device_owned_bytes"] == 0
        assert batch.snapshot() == before
        with batch.prepare_contacts(model, bindings, laws, candidate_capacity=0) as capped:
            checkpoint = _snapshot(batch, capped)
            with pytest.raises(ValueError, match="capacity"):
                batch.step_contacts(capped, H)
            _assert_retained(batch, capped, checkpoint)
        with batch.prepare_contacts(model, bindings, laws) as retry:
            assert batch.step_contacts(retry, H) == {WORLD: True}
            _save(
                tmp_path,
                "floating-contact-resource",
                dict(
                    input=item["data"],
                    before=before,
                    scratch_closed=owners[0].stats(),
                    retry=_record(batch, retry),
                    stats=batch.stats(),
                ),
            )


@pytest.mark.tyx
@pytest.mark.parametrize("name", ("F01", "F03"))
def test_f10_owned_receipt_foreign_batch_and_close_rebuild(name, tyx_mode, tmp_path):
    item = _item(_reference().case(name), delay=1)
    batch = ph.TyxWorldBatch([(WORLD, item["tree"])], tyx_mode, cache_root=tmp_path / "first")
    with ph.TyxWorldBatch(
        [(WORLD, item["tree"])], tyx_mode, cache_root=tmp_path / "foreign"
    ) as foreign:
        try:
            batch.set_controls({WORLD: item["command"]})
            model, bindings, _, laws = _geometry(batch, item)
            plan = batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True)
            before_other = foreign.snapshot()
            with pytest.raises(ValueError, match="owner|batch|belong"):
                foreign.step_contacts(plan, H)
            assert foreign.snapshot() == before_other
            assert batch.step_contacts(plan, H) == {WORLD: True}
            saved, receipt = batch.snapshot(), plan.last_accepted[WORLD]
            frozen = _json(receipt)
            for site in receipt.sites:
                for side in ("first", "second"):
                    for suffix in (
                        "_force_n",
                        "_velocity_m_s",
                        "_material_displacement_m",
                        "_application_displacement_m",
                    ):
                        with pytest.raises(ValueError):
                            np.asarray(getattr(site, side + suffix)).setflags(write=True)
            with pytest.raises(TypeError):
                receipt.diagnostics["coordinate_kind"] = "fixed"
            epoch = batch.stats()["state_epoch"]
            batch.snapshot()
            batch.tree_kinematics()
            batch.tree_diagnostics()
            assert batch.stats()["state_epoch"] == epoch and batch.snapshot() == saved
            assert batch.step_contacts(plan, H) == {WORLD: True}
            expected = _record(batch, plan)
            assert _json(receipt) == frozen
        finally:
            batch.close()
            batch.close()
        assert plan.stats()["closed"] and plan.stats()["device_owned_bytes"] == 0
        assert batch.stats()["device_owned_bytes"] == 0
        with pytest.raises(RuntimeError, match="closed"):
            batch.step_contacts(plan, H)
    with ph.TyxWorldBatch(
        [(WORLD, item["tree"])], tyx_mode, cache_root=tmp_path / "rebuilt"
    ) as rebuilt:
        rebuilt.restore(saved)
        model, bindings, _, laws = _geometry(rebuilt, item)
        with rebuilt.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as fresh:
            assert rebuilt.step_contacts(fresh, H) == {WORLD: True}
            actual = _record(rebuilt, fresh)
            assert _json(actual) == _json(expected)
            _save(
                tmp_path,
                "floating-contact-close",
                dict(
                    input=item["data"],
                    saved=saved,
                    original_receipt=receipt,
                    expected=expected,
                    rebuilt=actual,
                    stats=rebuilt.stats(),
                ),
            )


@pytest.mark.tyx
def test_f10_old_fixed_affine_scalar_and_unbound_floating_publish_together(tyx_mode, tmp_path):
    from test_resident_affine_contact_transactions import (
        _advance_oracle,
        _audit_mixed,
        _independent_initial,
    )
    from test_resident_affine_contact_transactions import (
        _geometry as old_geometry,
    )
    from test_resident_affine_contact_transactions import (
        _mixed as old_mixed,
    )
    from test_resident_floating_step import _audit as audit_free_floating

    old = old_mixed(tmp_path / "assets", delay=1)
    del old[1207]
    data = _reference().base().case("P03")
    # This world intentionally has no collision binding. Its original internal
    # motor must still stage through the same root/joint/held banks once.
    from test_resident_floating_step import _packet as free_packet
    from test_resident_floating_step import _tree as free_tree

    floating = free_tree(data, delay=1)
    command, external = free_packet(floating, data)
    declarations = [(w, i["world"]) for w, i in old.items()] + [(1207, floating)]
    commands = {w: i["control"] for w, i in old.items()}
    commands[1207] = command
    rows, replay, saved = [], [], None
    expected = _independent_initial(old)
    fq, fv = data["q"].copy(), data["v"].copy()
    with ph.TyxWorldBatch(declarations, tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls(commands)
        model, bindings, sources, laws = old_geometry(batch, old)
        with batch.prepare_contacts(model, bindings, laws) as plan:
            try:
                for index in range(40):
                    entry = batch.snapshot()
                    # The reused C7 helper assumes immediate motor delivery.
                    # This authored mixed window has delay=1, kp=kd=0: select
                    # the declared held-zero motor on step 1 without changing
                    # the worlds, commands, external loads or old helper.
                    oracle_cases = old
                    if index == 0:
                        oracle_cases = {
                            w: dict(
                                i, data=dict(i["data"], motor=np.zeros_like(i["data"]["motor"]))
                            )
                            if i["kind"] == "affine"
                            else i
                            for w, i in old.items()
                        }
                    expected = _advance_oracle(oracle_cases, expected)
                    selected = np.zeros((1, 3)) if index == 0 else data["drive"]
                    fq, fv, truth = _reference().base().advance(data, fq, fv, selected, external, H)
                    assert batch.step_contacts(plan, H) == dict.fromkeys(commands, True)
                    assert 1207 not in plan.last_accepted and 6 not in plan.last_accepted
                    audits = _audit_mixed(
                        batch, oracle_cases, entry, plan.last_accepted, sources, expected
                    )
                    selected_motor = oracle_cases[91]["data"]["motor"]
                    _close(plan.last_accepted[91].actual_drive, selected_motor)
                    _close(plan.last_accepted[91].selected_drive, [[0, 0, selected_motor[0]]])
                    after = batch.snapshot()
                    diagnostic = batch.tree_diagnostics()[1207]
                    audits[1207] = audit_free_floating(
                        data,
                        entry[1207],
                        after[1207],
                        batch.tree_kinematics()[1207],
                        diagnostic,
                        batch.summary()[1207],
                        selected,
                        external,
                    )
                    _close(after[1207].position, fq, atol=WINDOW_ATOL)
                    _close(after[1207].velocity, fv, atol=WINDOW_ATOL)
                    assert after[1207].tick == index + 1
                    row = _record(batch, plan)
                    row.update(
                        audit=audits,
                        independent_floating=dict(q=fq.copy(), v=fv.copy(), truth=truth),
                        independent_old=expected,
                        selected_affine_motor=selected_motor.copy(),
                        diagnostics=batch.tree_diagnostics(),
                    )
                    rows.append(row)
                    if index == 18:
                        saved = batch.snapshot()
                batch.restore(saved)
                for index in range(19, 40):
                    assert batch.step_contacts(plan, H) == dict.fromkeys(commands, True)
                    row = _record(batch, plan)
                    row["diagnostics"] = batch.tree_diagnostics()
                    replay.append(row)
                    for field in ("state", "geometry", "receipts", "summary", "diagnostics"):
                        assert _json(row[field]) == _json(rows[index][field])
            finally:
                _save(
                    tmp_path,
                    "floating-contact-old-mixed",
                    dict(
                        old_inputs={w: i.get("data", i["kind"]) for w, i in old.items()},
                        floating_input=data,
                        commands=commands,
                        model=model,
                        bindings=bindings,
                        laws=laws,
                        saved=saved,
                        rows=rows,
                        replay=replay,
                        retained=batch.snapshot(),
                        stats=batch.stats(),
                    ),
                )


@pytest.mark.tyx
@pytest.mark.parametrize(
    "filter_kind", ("default", "same_body", "mask", "excluded", "parent_mask", "parent_excluded")
)
def test_f10_explicit_pair_filters_precede_actual_parent_gate(filter_kind, tyx_mode, tmp_path):
    data = _reference().case("F03")
    if filter_kind.startswith("parent_"):
        data["children"]["parents"][1] = 0
    item = _item(data)
    with ph.TyxWorldBatch([(WORLD, item["tree"])], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: item["command"]})
        model, bindings, _, laws = _geometry(batch, item)
        shapes, bound, exclusions = list(model.shapes), list(bindings), ()
        if filter_kind == "default":
            before = batch.snapshot()
            with pytest.raises(ValueError):
                batch.prepare_contacts(model, bindings, laws)
            assert batch.snapshot() == before
            with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as plan:
                assert batch.step_contacts(plan, H) == {WORLD: True}
                assert len(plan.last_accepted[WORLD].sites) == 1
                _save(
                    tmp_path,
                    "floating-contact-domain",
                    dict(
                        kind=filter_kind,
                        input=data,
                        before=before,
                        accepted=_record(batch, plan),
                        stats=batch.stats(),
                    ),
                )
            return
        if filter_kind == "same_body":
            key = ph.CollisionKey(WORLD, shapes[0].key.body, 99)
            shapes[1] = replace(shapes[1], key=key)
            bound[1] = replace(bound[1], key=key, group_index=bound[0].group_index)
        elif filter_kind.endswith("mask"):
            shapes[1] = replace(shapes[1], mask=0)
        else:
            exclusions = (tuple(sorted((shapes[0].key, shapes[1].key))),)
        model = ph.CollisionModel(tuple(shapes), exclusions)
        pair = tuple(sorted((shapes[0].key, shapes[1].key)))
        before = batch.snapshot()
        with pytest.raises(ValueError):
            batch.prepare_contacts(
                model, bound, {pair: ph.RobotContactMaterial(64, 0, 0)}, tree_dynamic_pairs=True
            )
        with batch.prepare_contacts(model, bound, {}, tree_dynamic_pairs=True) as plan:
            assert batch.step_contacts(plan, H) == {WORLD: True}
            receipt = plan.last_accepted[WORLD]
            assert not receipt.sites
            _close(receipt.law_wrench_fp64, np.zeros((3, 6)), atol=0, rtol=0)
            assert receipt.accepted_state.position == before[WORLD].position
            _save(
                tmp_path,
                "floating-contact-domain",
                dict(
                    kind=filter_kind,
                    input=data,
                    before=before,
                    model=model,
                    bindings=bound,
                    accepted=_record(batch, plan),
                    stats=batch.stats(),
                ),
            )


@pytest.mark.tyx
@pytest.mark.parametrize(
    "problem",
    (
        "parent",
        "body_alias",
        "two_bodies_one_group",
        "world",
        "missing",
        "extra",
        "friction",
        "law",
        "sampling",
        "foreign_topology",
        "source_revision",
    ),
)
def test_f10_invalid_graph_material_source_and_domain_reject_before_allocation(
    problem, tyx_mode, tmp_path
):
    data = _reference().case("F06" if problem == "source_revision" else "F03")
    if problem == "parent":
        data["children"]["parents"][1] = 0
    item = _item(data)
    with ph.TyxWorldBatch([(WORLD, item["tree"])], tyx_mode, cache_root=tmp_path) as batch:
        model, bindings, _, laws = _geometry(batch, item)
        shapes, bound, options = list(model.shapes), list(bindings), {}
        pair = next(iter(laws))
        if problem == "body_alias":
            key = ph.CollisionKey(WORLD, shapes[0].key.body, 99)
            shapes[1], bound[1] = replace(shapes[1], key=key), replace(bound[1], key=key)
            laws = {tuple(sorted((shapes[0].key, key))): ph.RobotContactMaterial(64, 0, 0)}
        elif problem == "two_bodies_one_group":
            bound[1] = replace(bound[1], group_index=bound[0].group_index)
        elif problem == "world":
            key = ph.CollisionKey(999, shapes[1].key.body, shapes[1].key.shape)
            shapes[1], bound[1] = replace(shapes[1], key=key), replace(bound[1], key=key, world=999)
        elif problem == "missing":
            laws = {}
        elif problem == "extra":
            laws[tuple(reversed(pair))] = ph.RobotContactMaterial(64, 0, 0)
        elif problem == "friction":
            laws[pair] = ph.RobotContactMaterial(64, 0, 0.5)
        elif problem == "law":
            laws[pair] = ph.NormalLayer(1 / 32, 64)
        elif problem == "sampling":
            options["sampling"] = "swept"
        elif problem == "source_revision":
            shapes[0] = replace(shapes[0], revision=shapes[0].revision + 1)
        elif problem == "foreign_topology":
            other_data = _reference().case("F03")
            other_data["root_mass"] = 3.0
            other = _item(other_data)
            with ph.TyxWorldBatch(
                [(WORLD, other["tree"])], tyx_mode, cache_root=tmp_path / "foreign"
            ) as foreign:
                bound[0] = replace(bound[0], topology=foreign.snapshot()[WORLD].topology)
        before, allocations = batch.snapshot(), batch.stats()["device_allocations"]
        with pytest.raises((ValueError, TypeError)):
            batch.prepare_contacts(
                ph.CollisionModel(tuple(shapes)), bound, laws, tree_dynamic_pairs=True, **options
            )
        assert batch.snapshot() == before and batch.stats()["device_allocations"] == allocations


@pytest.mark.tyx
@pytest.mark.parametrize("flag", (1, np.bool_(True), "true", None))
def test_f10_dynamic_opt_in_requires_builtin_bool(flag, tyx_mode, tmp_path):
    item = _item(_reference().case("F03"))
    with ph.TyxWorldBatch([(WORLD, item["tree"])], tyx_mode, cache_root=tmp_path) as batch:
        model, bindings, _, laws = _geometry(batch, item)
        before, allocations = batch.snapshot(), batch.stats()["device_allocations"]
        with pytest.raises((ValueError, TypeError)):
            batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=flag)
        assert batch.snapshot() == before and batch.stats()["device_allocations"] == allocations


@pytest.mark.tyx
@pytest.mark.parametrize("n,invalid_group", ((16, 17), (15, 16), (0, 1)))
def test_f10_physical_body_count_is_not_motor_or_padded_count(n, invalid_group, tyx_mode, tmp_path):
    data = _reference().case("F01") if n == 0 else _reference().case("F08", n=n)
    item = _item(data)
    with ph.TyxWorldBatch([(WORLD, item["tree"])], tyx_mode, cache_root=tmp_path) as batch:
        model, bindings, _, laws = _geometry(batch, item)
        before, allocations = batch.snapshot(), batch.stats()["device_allocations"]
        with pytest.raises(ValueError):
            bound = list(bindings)
            bound[0] = replace(bound[0], group_index=invalid_group)
            batch.prepare_contacts(model, bound, laws)
        assert batch.snapshot() == before and batch.stats()["device_allocations"] == allocations


@pytest.mark.tyx
@pytest.mark.parametrize("kind", ("ordinary", "affine"))
def test_f10_larger_floating_body_limit_does_not_invent_fixed_bodies(kind, tyx_mode, tmp_path):
    from test_resident_affine_contact_transactions import _geometry as old_geometry
    from test_resident_affine_contact_transactions import _mixed as old_mixed

    old = old_mixed(tmp_path / "assets")
    world = 503 if kind == "ordinary" else 91
    item = old[world]
    with ph.TyxWorldBatch([(world, item["world"])], tyx_mode, cache_root=tmp_path) as batch:
        model, bindings, _, laws = old_geometry(batch, {world: item})
        before, allocations = batch.snapshot(), batch.stats()["device_allocations"]
        with pytest.raises(ValueError):
            bound = list(bindings)
            bound[0] = replace(bound[0], group_index=16)
            batch.prepare_contacts(model, bound, laws)
        assert batch.snapshot() == before and batch.stats()["device_allocations"] == allocations


@pytest.mark.tyx
def test_f10_huge_anchor_losing_nonzero_material_offset_rejects_without_repair(tyx_mode, tmp_path):
    data = _reference().case("F01")
    data["q"][:3] = [float(2**60), 0.0, 0.0]
    item = _item(data)
    local = np.array([[1 / 8, 0.0, 0.0]])
    assert float(2**60) + float(local[0, 0]) == float(2**60)
    first, second = ph.CollisionKey(WORLD, 101, 1), ph.CollisionKey(WORLD, 900, 2)
    shapes = (
        ph.CollisionShape(first, "sphere", (11,), 1 / 8),
        ph.CollisionShape(second, "sphere", (91,), 1 / 8),
    )
    with ph.TyxWorldBatch([(WORLD, item["tree"])], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: item["command"]})
        bindings = (
            ph.ResidentFloatingTreeBinding(
                first, WORLD, 0, local, topology=batch.snapshot()[WORLD].topology
            ),
            ph.ResidentStaticBinding(
                second, ph.CollisionMotion(np.array([[float(2**60), 0.0, 7 / 32]]))
            ),
        )
        before = batch.snapshot()
        geometry = _json(batch.tree_kinematics())
        with batch.prepare_collision(ph.CollisionModel(shapes), bindings) as broad:
            broad_candidates = broad.query().candidates
            assert (first, second) in broad_candidates
        assert batch.snapshot() == before
        plan = None
        try:
            with pytest.raises(ValueError) as caught:
                plan = batch.prepare_contacts(
                    ph.CollisionModel(shapes),
                    bindings,
                    {(first, second): ph.RobotContactMaterial(64, 0, 0)},
                )
                batch.step_contacts(plan, H)
            assert batch.snapshot() == before
            assert _json(batch.tree_kinematics()) == geometry
            if plan is not None:
                assert not plan.last_accepted
            _save(
                tmp_path,
                "floating-contact-source-representability",
                dict(
                    input=data,
                    local_points=local,
                    model=ph.CollisionModel(shapes),
                    bindings=bindings,
                    broad_candidates=broad_candidates,
                    before=before,
                    error=dict(type=type(caught.value).__name__, message=str(caught.value)),
                    retained=batch.snapshot(),
                    geometry=batch.tree_kinematics(),
                    stats=batch.stats(),
                ),
            )
        finally:
            if plan is not None:
                plan.close()
                assert plan.stats()["device_owned_bytes"] == 0


@pytest.mark.tyx
@pytest.mark.parametrize("field", ("shape", "group", "world"))
def test_f10_actual_candidate_material_packet_mismatch_is_not_published(
    field, tyx_mode, tmp_path, monkeypatch
):
    item = _item(_reference().case("F02", variant="motor"), delay=1)
    with ph.TyxWorldBatch([(WORLD, item["tree"])], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: item["command"]})
        model, bindings, _, laws = _geometry(batch, item)
        with batch.prepare_contacts(model, bindings, laws) as plan:
            before, original, calls = _snapshot(batch, plan), plan._read_candidate, []

            def actual_then_wrong_source(*args, **kwargs):
                result = original(*args, **kwargs)
                packet = dict(result["packet"])
                world_packet = dict(packet[WORLD])
                frames = dict(world_packet["frames"])
                key = next(k for k, frame in frames.items() if frame.group_index is not None)
                frame = frames[key]
                if field == "shape":
                    changed = replace(
                        frame, shape=replace(frame.shape, revision=frame.shape.revision + 1)
                    )
                elif field == "group":
                    changed = replace(frame, group_index=0)
                else:
                    changed = replace(
                        frame,
                        shape=replace(frame.shape, key=ph.CollisionKey(999, key.body, key.shape)),
                    )
                frames[key] = changed
                world_packet["frames"] = frames
                packet[WORLD] = world_packet
                calls.append(dict(original=frame, changed=changed))
                return dict(result, packet=packet)

            with monkeypatch.context() as patch:
                patch.setattr(plan, "_read_candidate", actual_then_wrong_source)
                with pytest.raises(ValueError):
                    batch.step_contacts(plan, H)
            assert len(calls) == 1
            _assert_retained(batch, plan, before)
            plan.close()
            with batch.prepare_contacts(model, bindings, laws) as retry:
                assert batch.step_contacts(retry, H) == {WORLD: True}
                assert retry.last_accepted[WORLD].entry_state == before["state"][WORLD]
                _save(
                    tmp_path,
                    "floating-contact-candidate-source",
                    dict(
                        input=item["data"],
                        command=item["command"],
                        field=field,
                        injected=calls,
                        before=before,
                        retry=_record(batch, retry),
                        stats=batch.stats(),
                    ),
                )

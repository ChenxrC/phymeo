"""C8 same-tree dynamic pairs: mixed worlds, filtering and atomic retry.

AUTHORING ONLY / UNEXECUTED. Fault injection calls the real operation first.
The parent owns all imports, collection, backend and static executions.
"""

from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from test_resident_affine_contact_transactions import _ordinary, _ordinary_geometry
from test_resident_affine_contacts import _audit_sites
from test_resident_tree_dynamic_contacts import (
    WINDOW_ATOL,
    WORLD,
    H,
    _audit,
    _build,
    _case,
    _close,
    _geometry,
    _json,
    _save,
    _state,
    _truth,
)

import phymeo as ph

KEYS = (503, 91, 1207, 6)


def _mixed(directory, *, hull=False, delay=0):
    items = {
        key: _build(
            _case(name, shape="hull" if hull else "sphere"), directory / str(key), delay=delay
        )
        for key, name in ((503, "D01"), (91, "D04"))
    }
    ordinary, control = _ordinary()
    items[1207] = dict(kind="static", world=ordinary, control=control)
    items[6] = dict(
        kind="free",
        world=ph.FreeSphere(2, 1 / 8, position=(0, 0, 2), gravity=(0, 0, -8)),
        control=(0, 0, 0),
    )
    return items


def _combined_geometry(batch, items):
    shapes, bindings, excluded, all_sources, laws = [], [], [], {}, {}
    for world, item in items.items():
        if item.get("kind") == "free":
            continue
        if item.get("kind") == "static":
            model, bound, sources, pair = _ordinary_geometry(batch, world)
            material = {pair: ph.RobotContactMaterial(64, 0, 0)}
        else:
            model, bound, sources, material = _geometry(batch, item, world=world)
        shapes.extend(model.shapes)
        bindings.extend(bound)
        excluded.extend(model.excluded)
        all_sources[world] = sources
        laws.update(material)
    return ph.CollisionModel(tuple(shapes), tuple(excluded)), tuple(bindings), all_sources, laws


def _snapshot(batch, plan):
    return dict(
        state=batch.snapshot(),
        geometry=_json(batch.tree_kinematics()),
        energy=batch.summary(),
        receipts=_json(plan.last_accepted),
        status=dict(plan.last_status),
        epoch=batch.stats()["state_epoch"],
        bank=batch.stats()["current_buffer"],
    )


def _assert_retained(batch, plan, before, *, epoch=True):
    now = _snapshot(batch, plan)
    for field in ("state", "geometry", "energy", "receipts", "status", "bank"):
        assert now[field] == before[field]
    if epoch:
        assert now["epoch"] == before["epoch"]


def _next_expected(item, q, v):
    if item.get("kind") == "free":
        a = np.array([0.0, 0.0, -8.0])
    elif item.get("kind") == "static":
        a = np.array([max(0.0, 2 - 64 * q[0]) / 2])
    else:
        a = _truth(item["data"], q, v)["acceleration"]
    next_v = v + H * a
    return q + H * next_v, next_v


def _mixed_window(items, mode, directory, *, replay=False):
    directory.mkdir(parents=True, exist_ok=True)
    expected = {
        key: (np.array([0.0, 0.0, 2.0]), np.zeros(3))
        if item.get("kind") == "free"
        else (np.zeros(1), np.zeros(1))
        if item.get("kind") == "static"
        else (item["data"]["s"].copy(), item["data"]["u"].copy())
        for key, item in items.items()
    }
    rows, repeated, saved, failure = [], [], None, None
    with ph.TyxWorldBatch(
        [(key, item["world"]) for key, item in items.items()], mode, cache_root=directory
    ) as batch:
        batch.set_controls({key: item["control"] for key, item in items.items()})
        model, bindings, sources, laws = _combined_geometry(batch, items)
        with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as plan:
            try:
                for step in range(40):
                    entry = batch.snapshot()
                    expected = {
                        key: _next_expected(item, *expected[key]) for key, item in items.items()
                    }
                    assert batch.step_contacts(plan, H) == dict.fromkeys(items, True)
                    receipts, audits = dict(plan.last_accepted), {}
                    for key, item in items.items():
                        state = batch.snapshot()[key]
                        if item.get("kind") == "free":
                            assert key not in receipts
                            q, v = np.asarray(state.position), np.asarray(state.velocity)
                        elif item.get("kind") == "static":
                            receipt = receipts[key]
                            d = receipt.diagnostics
                            audits[key] = _audit_sites(
                                receipt,
                                SimpleNamespace(
                                    linear_jacobian=d["linear_jacobian"],
                                    angular_jacobian=d["angular_jacobian"],
                                ),
                                sources[key],
                            )
                            normal = max(0.0, 2 - 64 * entry[key].position[0])
                            _close(receipt.contact_generalized, [normal])
                            assert receipt.static_reactions
                            assert d["tree_dynamic_pairs"] is True
                            q, v = np.asarray(state.position), np.asarray(state.velocity)
                        else:
                            audits[key] = _audit(
                                batch, item, entry[key], receipts[key], sources[key], world=key
                            )
                            q, v = _state(item, state)
                        _close(q, expected[key][0], atol=WINDOW_ATOL)
                        _close(v, expected[key][1], atol=WINDOW_ATOL)
                        assert state.tick == step + 1
                    rows.append(
                        dict(
                            state=batch.snapshot(),
                            geometry=batch.tree_kinematics(),
                            diagnostics=batch.tree_diagnostics(),
                            receipts=receipts,
                            expected={
                                key: (q.copy(), v.copy()) for key, (q, v) in expected.items()
                            },
                            audits=audits,
                            stats=batch.stats(),
                        )
                    )
                    if step == 18:
                        saved = batch.snapshot()
                if replay:
                    batch.restore(saved)
                    assert batch.snapshot() == saved
                    for index in range(19, 40):
                        assert batch.step_contacts(plan, H) == dict.fromkeys(items, True)
                        row = dict(
                            state=batch.snapshot(),
                            geometry=batch.tree_kinematics(),
                            receipts=dict(plan.last_accepted),
                            stats=batch.stats(),
                        )
                        repeated.append(row)
                        assert row["state"] == rows[index]["state"]
                        assert _json(row["geometry"]) == _json(rows[index]["geometry"])
                        assert _json(row["receipts"]) == _json(rows[index]["receipts"])
            except Exception as error:
                failure = dict(type=type(error).__name__, message=str(error))
                raise
            finally:
                _save(
                    directory,
                    "dynamic-mixed-window",
                    dict(
                        mode=mode,
                        h_s=H,
                        input={
                            key: item.get("data", item.get("kind")) for key, item in items.items()
                        },
                        commands={key: item["control"] for key, item in items.items()},
                        collision_model=model,
                        bindings=bindings,
                        laws=laws,
                        rows=rows,
                        checkpoint_19=saved,
                        replay=repeated,
                        failure=failure,
                        retained=batch.snapshot(),
                        last_accepted=plan.last_accepted,
                        stats=batch.stats(),
                        plan_stats=plan.stats(),
                    ),
                )
    return rows


@pytest.mark.tyx
def test_d08_w1_w4_reversed_worlds_and_complete_21_step_replay(tyx_mode, tmp_path):
    items = _mixed(tmp_path / "assets")
    separate = {
        key: _mixed_window({key: item}, tyx_mode, tmp_path / f"w1-{key}")
        for key, item in items.items()
    }
    combined = _mixed_window(
        dict(reversed(tuple(items.items()))), tyx_mode, tmp_path / "w4", replay=True
    )
    for index, row in enumerate(combined):
        for key in items:
            assert row["state"][key] == separate[key][index]["state"][key]
            if key in row["geometry"]:
                assert _json(row["geometry"][key]) == _json(separate[key][index]["geometry"][key])
                assert _json(row["receipts"][key]) == _json(separate[key][index]["receipts"][key])
                assert all(
                    site.pair[0].world == site.pair[1].world == key
                    for site in row["receipts"][key].sites
                )


@pytest.mark.tyx
@pytest.mark.parametrize("warm,limited_world", tuple(product((0, 1), (503, 91))))
def test_d08_site_capacity_retains_both_dynamic_groups_and_neighbors_advance(
    warm, limited_world, tyx_mode, tmp_path
):
    items = _mixed(tmp_path / "assets", hull=True, delay=1)
    with ph.TyxWorldBatch(
        [(key, item["world"]) for key, item in items.items()], tyx_mode, cache_root=tmp_path
    ) as batch:
        batch.set_controls({key: item["control"] for key, item in items.items()})
        model, bindings, sources, laws = _combined_geometry(batch, items)
        with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as full:
            for _ in range(warm):
                assert batch.step_contacts(full, H) == dict.fromkeys(items, True)
            caps = {503: 128, 91: 128, 1207: 128}
            caps[limited_world] = 1
            with batch.prepare_contacts(
                model, bindings, laws, tree_dynamic_pairs=True, site_capacity=caps
            ) as limited:
                before = _snapshot(batch, full)
                result = batch.step_contacts(limited, H)
                assert result == {key: key != limited_world for key in items}
                assert limited.last_status[limited_world] == "SITE_CAPACITY_EXCEEDED"
                assert batch.snapshot()[limited_world] == before["state"][limited_world]
                assert (
                    _json(batch.tree_kinematics())[str(limited_world)]
                    == before["geometry"][str(limited_world)]
                )
                assert batch.summary()[limited_world] == before["energy"][limited_world]
                assert _json(full.last_accepted) == before["receipts"]
                assert limited_world not in limited.last_accepted
                for key in items:
                    assert batch.snapshot()[key].tick == warm + (key != limited_world)
                paused = batch.snapshot()[limited_world]
                assert batch.step_contacts(full, H) == dict.fromkeys(items, True)
                assert full.last_accepted[limited_world].entry_state == paused
                _audit(
                    batch,
                    items[limited_world],
                    paused,
                    full.last_accepted[limited_world],
                    sources[limited_world],
                    world=limited_world,
                )
                _save(
                    tmp_path,
                    "dynamic-capacity",
                    dict(
                        warm=warm,
                        limited_world=limited_world,
                        input={key: i.get("data", i.get("kind")) for key, i in items.items()},
                        before=before,
                        limited_status=limited.last_status,
                        retry=full.last_accepted,
                        retained=batch.snapshot(),
                        stats=batch.stats(),
                    ),
                )


@pytest.mark.tyx
@pytest.mark.parametrize("warm", (0, 1))
def test_d08_exact_touch_counts_sites_and_retains_same_plan_owned_receipt(warm, tyx_mode, tmp_path):
    data = _case("D01", shape="hull", velocity=[0, -1])
    data["s"][1] = 1 / 32 + (warm + 1) * H
    item = _build(data, tmp_path / "asset")
    with ph.TyxWorldBatch([(WORLD, item["world"])], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: item["control"]})
        model, bindings, _, laws = _geometry(batch, item)
        with batch.prepare_contacts(
            model, bindings, laws, tree_dynamic_pairs=True, site_capacity=1
        ) as plan:
            for _ in range(warm + 1):
                assert batch.step_contacts(plan, H) == {WORLD: True}
                assert not plan.last_accepted[WORLD].sites
            before = _snapshot(batch, plan)
            old = plan.last_accepted[WORLD]
            assert batch.snapshot()[WORLD].position == (0.0, 1 / 32)
            assert batch.step_contacts(plan, H) == {WORLD: False}
            assert batch.snapshot() == before["state"]
            assert _json(batch.tree_kinematics()) == before["geometry"]
            assert batch.summary() == before["energy"]
            assert plan.last_accepted[WORLD] is old
            assert _json(plan.last_accepted) == before["receipts"]
            assert plan.last_status[WORLD] == "SITE_CAPACITY_EXCEEDED"
            _save(
                tmp_path,
                "dynamic-touch-retain",
                dict(
                    input=data,
                    warm=warm,
                    before=before,
                    retained=batch.snapshot(),
                    last_accepted=plan.last_accepted,
                    status=plan.last_status,
                ),
            )


@pytest.mark.tyx
def test_d08_one_world_combines_dynamic_pair_and_real_static_reaction_once(tyx_mode, tmp_path):
    item = _build(_case("D01"), tmp_path / "asset")
    with ph.TyxWorldBatch([(WORLD, item["world"])], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: item["control"]})
        model, bindings, sources, laws = _geometry(batch, item)
        key = ph.CollisionKey(WORLD, 900, 1)
        shape = ph.CollisionShape(key, "sphere", (901,), radius=1 / 8)
        static = np.array([[-15 / 64, 0, 0]], float)
        model = ph.CollisionModel(model.shapes + (shape,))
        bindings += (ph.ResidentStaticBinding(key, ph.CollisionMotion(static)),)
        for moving in model.shapes[:-1]:
            laws[tuple(sorted((moving.key, key)))] = ph.RobotContactMaterial(64, 0, 0)
        sources[key] = (shape, None, static)
        before = batch.snapshot()
        with batch.prepare_contacts(
            model, bindings, laws, tree_dynamic_pairs=True, site_capacity=1
        ) as capped:
            assert batch.step_contacts(capped, H) == {WORLD: False}
            assert batch.snapshot() == before and not capped.last_accepted
        with batch.prepare_contacts(
            model, bindings, laws, tree_dynamic_pairs=True, site_capacity=2
        ) as plan:
            assert batch.step_contacts(plan, H) == {WORLD: True}
            receipt = plan.last_accepted[WORLD]
            _close(receipt.contact_generalized, [-1, 2])
            _close(receipt.diagnostics["acceleration"], [-0.5, 2])
            assert len(receipt.sites) == 2 and set(receipt.static_reactions) == {key}
            _close(receipt.static_reactions[key].force_n, [-1, 0, 0])
            d = receipt.diagnostics
            _audit_sites(
                receipt,
                SimpleNamespace(
                    linear_jacobian=d["linear_jacobian"], angular_jacobian=d["angular_jacobian"]
                ),
                sources,
            )
            _save(
                tmp_path,
                "dynamic-and-static",
                dict(
                    input=item["data"],
                    collision_model=model,
                    bindings=bindings,
                    before=before,
                    receipt=receipt,
                    retained=batch.snapshot(),
                    stats=batch.stats(),
                ),
            )


@pytest.mark.tyx
@pytest.mark.parametrize("phase,warm", tuple(product(("upload", "candidate", "receipt"), (0, 1))))
def test_d08_real_late_failure_preserves_all_banks_then_retries(
    phase, warm, tyx_mode, tmp_path, monkeypatch
):
    items = _mixed(tmp_path / "assets", delay=1)
    worlds = [(key, item["world"]) for key, item in items.items()]
    with (
        ph.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "failed") as batch,
        ph.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "reference") as reference,
    ):
        commands = {key: item["control"] for key, item in items.items()}
        batch.set_controls(commands)
        reference.set_controls(commands)
        model, bindings, _, laws = _combined_geometry(batch, items)
        other_model, other_bindings, _, other_laws = _combined_geometry(reference, items)
        with (
            batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as plan,
            reference.prepare_contacts(
                other_model, other_bindings, other_laws, tree_dynamic_pairs=True
            ) as control,
        ):
            for _ in range(warm):
                assert (
                    batch.step_contacts(plan, H)
                    == reference.step_contacts(control, H)
                    == dict.fromkeys(items, True)
                )
            before = _snapshot(batch, plan)
            method = {
                "upload": "_upload_contact",
                "candidate": "_read_candidate",
                "receipt": "_finish_attempt",
            }[phase]
            original, calls = getattr(plan, method), []

            def real_then_fail(*args, **kwargs):
                result = original(*args, **kwargs)
                calls.append(result)
                raise RuntimeError("c8 actual " + phase + " before publication")

            with monkeypatch.context() as patch:
                patch.setattr(plan, method, real_then_fail)
                with pytest.raises(RuntimeError, match="c8 actual"):
                    batch.step_contacts(plan, H)
            assert calls
            _assert_retained(batch, plan, before)
            plan.close()
            with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as retry:
                assert (
                    batch.step_contacts(retry, H)
                    == reference.step_contacts(control, H)
                    == dict.fromkeys(items, True)
                )
                assert batch.snapshot() == reference.snapshot()
                assert _json(batch.tree_kinematics()) == _json(reference.tree_kinematics())
                assert _json(retry.last_accepted) == _json(control.last_accepted)
                _save(
                    tmp_path,
                    "dynamic-late-fault",
                    dict(
                        warm=warm,
                        phase=phase,
                        before=before,
                        commands=commands,
                        retry=retry.last_accepted,
                        retained=batch.snapshot(),
                        reference=reference.snapshot(),
                        reference_receipts=control.last_accepted,
                        stats=batch.stats(),
                    ),
                )


@pytest.mark.tyx
@pytest.mark.parametrize("warm", (0, 1))
def test_d08_same_tick_restore_after_real_candidate_invalidates_attempt(
    warm, tyx_mode, tmp_path, monkeypatch
):
    item = _build(_case("D04"), tmp_path / "asset", delay=1)
    with ph.TyxWorldBatch([(WORLD, item["world"])], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({WORLD: item["control"]})
        model, bindings, _, laws = _geometry(batch, item)
        with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as plan:
            for _ in range(warm):
                assert batch.step_contacts(plan, H) == {WORLD: True}
            before = _snapshot(batch, plan)
            original, calls = plan._read_candidate, []

            def read_then_restore(*args, **kwargs):
                candidate = original(*args, **kwargs)
                batch.restore(before["state"])
                calls.append(batch.stats()["state_epoch"])
                return candidate

            with monkeypatch.context() as patch:
                patch.setattr(plan, "_read_candidate", read_then_restore)
                with pytest.raises((ValueError, RuntimeError), match="stale|epoch|changed|entry"):
                    batch.step_contacts(plan, H)
            assert calls == [before["epoch"] + 1]
            assert batch.snapshot() == before["state"]
            assert _json(plan.last_accepted) == before["receipts"]
            assert _json(batch.tree_kinematics()) == before["geometry"]
            plan.close()
            with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as retry:
                assert batch.step_contacts(retry, H) == {WORLD: True}
                assert retry.last_accepted[WORLD].entry_state == before["state"][WORLD]
                _save(
                    tmp_path,
                    "dynamic-stale-restore",
                    dict(
                        warm=warm,
                        before=before,
                        retry=retry.last_accepted,
                        retained=batch.snapshot(),
                        stats=batch.stats(),
                    ),
                )


@pytest.mark.tyx
def test_d08_second_actual_scratch_failure_and_candidate_global_cap(
    tyx_mode, tmp_path, monkeypatch
):
    from phymeo.backends._resident_contact_plan import ResidentContactPlan

    items = _mixed(tmp_path / "assets")
    with ph.TyxWorldBatch(
        [(key, item["world"]) for key, item in items.items()], tyx_mode, cache_root=tmp_path
    ) as batch:
        batch.set_controls({key: item["control"] for key, item in items.items()})
        model, bindings, _, laws = _combined_geometry(batch, items)
        before, calls, owners = batch.snapshot(), [], []
        original = ResidentContactPlan._buffer

        def allocate_then_fail(self, *args, **kwargs):
            value = original(self, *args, **kwargs)
            calls.append(value)
            owners.append(self)
            if len(calls) == 2:
                raise MemoryError("c8 actual second scratch allocation")
            return value

        with monkeypatch.context() as patch:
            patch.setattr(ResidentContactPlan, "_buffer", allocate_then_fail)
            with pytest.raises(MemoryError, match="actual second"):
                batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True)
        assert len(calls) == 2 and owners[0] is owners[1]
        assert owners[0].stats()["closed"] and owners[0].stats()["device_owned_bytes"] == 0
        assert batch.snapshot() == before
        with batch.prepare_contacts(
            model, bindings, laws, tree_dynamic_pairs=True, candidate_capacity=0
        ) as cap:
            snapshot = _snapshot(batch, cap)
            with pytest.raises(ValueError, match="capacity"):
                batch.step_contacts(cap, H)
            _assert_retained(batch, cap, snapshot)
        with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as retry:
            assert batch.step_contacts(retry, H) == dict.fromkeys(items, True)
            _save(
                tmp_path,
                "dynamic-resource-retry",
                dict(
                    before=before,
                    receipts=retry.last_accepted,
                    retained=batch.snapshot(),
                    stats=batch.stats(),
                ),
            )


@pytest.mark.tyx
@pytest.mark.parametrize("name", ("D01", "D04"))
def test_d08_owned_receipt_foreign_owner_and_close_rebuild(name, tyx_mode, tmp_path):
    item = _build(_case(name), tmp_path / "asset", delay=1)
    batch = ph.TyxWorldBatch([(WORLD, item["world"])], tyx_mode, cache_root=tmp_path / "original")
    with ph.TyxWorldBatch(
        [(WORLD, item["world"])], tyx_mode, cache_root=tmp_path / "foreign"
    ) as foreign:
        try:
            batch.set_controls({WORLD: item["control"]})
            model, bindings, _, laws = _geometry(batch, item)
            plan = batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True)
            other_before = foreign.snapshot()
            with pytest.raises(ValueError, match="owner|batch|belong"):
                foreign.step_contacts(plan, H)
            assert foreign.snapshot() == other_before
            assert batch.step_contacts(plan, H) == {WORLD: True}
            saved, receipt = batch.snapshot(), plan.last_accepted[WORLD]
            serialized = _json(receipt)
            arrays = [
                receipt.law_wrench_fp64,
                receipt.uploaded_wrench_fp32,
                receipt.contact_generalized,
                receipt.entry_geometry.link_position_m,
            ]
            arrays += [
                getattr(site, side + suffix)
                for site in receipt.sites
                for side in ("first", "second")
                for suffix in (
                    "_force_n",
                    "_velocity_m_s",
                    "_material_displacement_m",
                    "_application_displacement_m",
                )
            ]
            for array in arrays:
                assert not np.asarray(array).flags.writeable
                with pytest.raises(ValueError):
                    np.asarray(array).setflags(write=True)
            with pytest.raises(TypeError):
                receipt.diagnostics["tree_dynamic_pairs"] = False
            epoch = batch.stats()["state_epoch"]
            batch.tree_kinematics()
            batch.tree_diagnostics()
            assert batch.snapshot() == saved and batch.stats()["state_epoch"] == epoch
            assert batch.step_contacts(plan, H) == {WORLD: True}
            expected, next_receipt = batch.snapshot(), _json(plan.last_accepted)
            assert _json(receipt) == serialized
        finally:
            batch.close()
            batch.close()
        assert plan.stats()["closed"] and plan.stats()["device_owned_bytes"] == 0
        assert batch.stats()["device_owned_bytes"] == 0
        with pytest.raises(RuntimeError, match="closed"):
            batch.step_contacts(plan, H)
    with ph.TyxWorldBatch(
        [(WORLD, item["world"])], tyx_mode, cache_root=tmp_path / "rebuilt"
    ) as rebuilt:
        rebuilt.restore(saved)
        model, bindings, _, laws = _geometry(rebuilt, item)
        with rebuilt.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as fresh:
            assert rebuilt.step_contacts(fresh, H) == {WORLD: True}
            assert rebuilt.snapshot() == expected and _json(fresh.last_accepted) == next_receipt
            _save(
                tmp_path,
                "dynamic-close-rebuild",
                dict(
                    input=item["data"],
                    saved=saved,
                    expected=expected,
                    retained=rebuilt.snapshot(),
                    receipts=fresh.last_accepted,
                    stats=rebuilt.stats(),
                ),
            )


@pytest.mark.tyx
@pytest.mark.parametrize("api", ("collision", "contacts"))
@pytest.mark.parametrize("flag", (1, np.bool_(True), "true", None))
def test_d07_flag_is_builtin_bool_before_any_device_allocation(api, flag, tyx_mode, tmp_path):
    item = _build(_case("D01"), tmp_path / "asset")
    with ph.TyxWorldBatch([(WORLD, item["world"])], tyx_mode, cache_root=tmp_path) as batch:
        model, bindings, _, laws = _geometry(batch, item)
        before, allocations = batch.snapshot(), batch.stats()["device_allocations"]
        with pytest.raises((TypeError, ValueError)):
            if api == "collision":
                batch.prepare_collision(model, bindings, tree_dynamic_pairs=flag)
            else:
                batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=flag)
        assert batch.snapshot() == before and batch.stats()["device_allocations"] == allocations


@pytest.mark.tyx
@pytest.mark.parametrize("name", ("D01", "D04"))
def test_d07_default_keeps_old_domain_and_opt_in_reaches_actual_device_pair(
    name, tyx_mode, tmp_path
):
    item = _build(_case(name), tmp_path / "asset")
    with ph.TyxWorldBatch([(WORLD, item["world"])], tyx_mode, cache_root=tmp_path) as batch:
        model, bindings, _, laws = _geometry(batch, item)
        before = batch.snapshot()
        for api in (batch.prepare_collision, batch.prepare_contacts):
            args = (
                (model, bindings)
                if api.__name__ == "prepare_collision"
                else (model, bindings, laws)
            )
            with pytest.raises(ValueError):
                api(*args)
            with pytest.raises(ValueError):
                api(*args, tree_dynamic_pairs=False)
        with batch.prepare_collision(model, bindings, tree_dynamic_pairs=True) as collision:
            result = batch.collision(collision)
            assert set(result.candidates) == set(laws)
            assert batch.snapshot() == before
        with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as contact:
            assert batch.step_contacts(contact, H) == {WORLD: True}
            assert contact.last_accepted[WORLD].diagnostics["tree_dynamic_pairs"] is True


@pytest.mark.tyx
@pytest.mark.parametrize(
    "filter_kind", ("same_body", "mask", "excluded", "parent_mask", "parent_excluded")
)
def test_d07_explicit_filters_precede_adjacency_and_forbid_filtered_materials(
    filter_kind, tyx_mode, tmp_path
):
    data = _case("D01")
    if filter_kind.startswith("parent_"):
        data["model"]["parents"][1] = 0
    item = _build(data, tmp_path / "asset")
    with ph.TyxWorldBatch([(WORLD, item["world"])], tyx_mode, cache_root=tmp_path) as batch:
        model, bindings, _, _ = _geometry(batch, item)
        shapes, bound = list(model.shapes), list(bindings)
        excluded = ()
        if filter_kind == "same_body":
            key = ph.CollisionKey(WORLD, shapes[0].key.body, 8)
            shapes[1] = replace(shapes[1], key=key)
            bound[1] = replace(bound[1], key=key, group_index=bound[0].group_index)
        elif filter_kind.endswith("mask"):
            shapes[1] = replace(shapes[1], mask=0)
        else:
            excluded = (tuple(sorted((shapes[0].key, shapes[1].key))),)
        model = ph.CollisionModel(tuple(shapes), excluded)
        pair = tuple(sorted((shapes[0].key, shapes[1].key)))
        before = batch.snapshot()
        with batch.prepare_collision(model, bound, tree_dynamic_pairs=True) as plan:
            # registered_pairs is a physical-domain declaration; only actual
            # SpatialPairs-filtered candidates/materials have this empty set.
            assert batch.collision(plan).candidates == ()
            assert batch.snapshot() == before
        with pytest.raises(ValueError):
            batch.prepare_contacts(
                model, bound, {pair: ph.RobotContactMaterial(64, 0, 0)}, tree_dynamic_pairs=True
            )
        with batch.prepare_contacts(model, bound, {}, tree_dynamic_pairs=True) as plan:
            assert batch.step_contacts(plan, H) == {WORLD: True}
            receipt = plan.last_accepted[WORLD]
            assert not receipt.sites
            np.testing.assert_array_equal(receipt.law_wrench_fp64, np.zeros((2, 6)))
            assert receipt.accepted_state.position == before[WORLD].position


@pytest.mark.tyx
@pytest.mark.parametrize(
    "problem",
    (
        "parent",
        "group",
        "body_alias",
        "two_bodies_one_group",
        "world",
        "missing",
        "extra",
        "friction",
        "law",
        "sampling",
        "graph",
    ),
)
def test_d07_invalid_owner_graph_material_and_unfiltered_parent_reject_atomically(
    problem, tyx_mode, tmp_path
):
    data = _case("D04" if problem == "graph" else "D01")
    if problem == "parent":
        data["model"]["parents"][1] = 0
    item = _build(data, tmp_path / "asset")
    with ph.TyxWorldBatch([(WORLD, item["world"])], tyx_mode, cache_root=tmp_path) as batch:
        model, bindings, _, laws = _geometry(batch, item)
        shapes, bindings, options = list(model.shapes), list(bindings), {}
        pair = next(iter(laws))
        if problem == "group":
            bindings[1] = replace(bindings[1], group_index=2)
        elif problem == "body_alias":
            key = ph.CollisionKey(WORLD, shapes[0].key.body, 8)
            shapes[1] = replace(shapes[1], key=key)
            bindings[1] = replace(bindings[1], key=key)
            laws = {tuple(sorted((shapes[0].key, key))): ph.RobotContactMaterial(64, 0, 0)}
        elif problem == "two_bodies_one_group":
            bindings[1] = replace(bindings[1], group_index=0)
        elif problem == "world":
            key = ph.CollisionKey(999, shapes[1].key.body, shapes[1].key.shape)
            shapes[1] = replace(shapes[1], key=key)
            bindings[1] = replace(bindings[1], key=key, world=999)
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
        elif problem == "graph":
            # Same full geometry with a different passive relation is a genuinely
            # different physical source; do not mutate a private owner token.
            other = _build(_case("D04Z"), tmp_path / "other_asset")
            with ph.TyxWorldBatch(
                [(WORLD, other["world"])], tyx_mode, cache_root=tmp_path / "other"
            ) as foreign:
                topology = foreign.snapshot()[WORLD].topology
            bindings[1] = replace(bindings[1], topology=topology)
        before, allocations = batch.snapshot(), batch.stats()["device_allocations"]
        with pytest.raises((ValueError, TypeError)):
            batch.prepare_contacts(
                ph.CollisionModel(tuple(shapes)), bindings, laws, tree_dynamic_pairs=True, **options
            )
        assert batch.snapshot() == before and batch.stats()["device_allocations"] == allocations


@pytest.mark.tyx
def test_d07_scalar_world_cannot_become_two_physical_bodies_by_opt_in(tyx_mode, tmp_path):
    world = ph.FreeSphere(2, 1 / 8, position=(0, 0, 0), gravity=(0, 0, 0))
    keys = (ph.CollisionKey(WORLD, 10, 1), ph.CollisionKey(WORLD, 20, 1))
    shapes = tuple(
        ph.CollisionShape(key, "sphere", (10 + i,), radius=1 / 8) for i, key in enumerate(keys)
    )
    bindings = tuple(ph.ResidentSphereBinding(key, WORLD) for key in keys)
    with ph.TyxWorldBatch([(WORLD, world)], tyx_mode, cache_root=tmp_path) as batch:
        before = batch.snapshot()
        with pytest.raises(ValueError):
            batch.prepare_collision(ph.CollisionModel(shapes), bindings, tree_dynamic_pairs=True)
        with pytest.raises((ValueError, TypeError)):
            batch.prepare_contacts(
                ph.CollisionModel(shapes),
                bindings,
                {keys: ph.RobotContactMaterial(64, 0, 0)},
                tree_dynamic_pairs=True,
            )
        assert batch.snapshot() == before

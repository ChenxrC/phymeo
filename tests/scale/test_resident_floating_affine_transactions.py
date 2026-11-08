"""C11 full/reduced/graph publication and bounded mixed-world tests, unexecuted.

Core W1/W4 is four steps; only contact W1/W4 and A04-p1 carry 40+21.
The optional W32 node consumes an explicitly supplied same-source W4 record.
"""

import gzip
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_resident_floating_affine_contacts import _audit_contact, _geometry
from test_resident_floating_affine_step import (
    JAWS,
    WINDOW_ATOL,
    WORLD,
    H,
    _audit,
    _close,
    _compiled,
    _item,
    _jaws_state,
    _json,
    _record,
    _reference,
    _save,
    _selected,
)

import phymeo as ph
from phymeo.dynamics.floating_robot import FloatingRobotState
from phymeo.solvers.floating_tree import FloatingRobotInputs

KEYS = (503, 91, 1207, 6)


def _sources():
    root = Path(__file__).resolve().parents[2]
    names = (
        "tests/scale/test_resident_floating_affine_step.py",
        "tests/scale/test_resident_floating_affine_contacts.py",
        "tests/scale/test_resident_floating_affine_transactions.py",
        "tests/reference_newton/robot/resident_floating_affine_reference.py",
        "tests/reference_newton/robot/resident_floating_reference.py",
        "tests/reference_newton/robot/resident_affine_reference.py",
        "tests/scale/test_resident_affine_contacts.py",
        "tests/scale/test_resident_affine_contact_transactions.py",
        "tests/fixtures/urdf/floating_mimic_jaws.urdf",
        "tests/fixtures/urdf/floating_mimic_elbow.urdf",
    )
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _mixed(directory, category):
    ref = _reference()
    if category == "core":
        from test_resident_floating_step import _packet as old_packet
        from test_resident_floating_step import _tree as old_tree

        data = ref.base().case("P05", n=0)
        tree = old_tree(data, delay=1)
        command, external = old_packet(tree, data)
        return {
            503: _item(directory / "503", ref.case("A01"), delay=1),
            91: _item(directory / "91", ref.case("A03", variant="chain"), delay=1),
            1207: dict(
                kind="floating", data=data, tree=tree, command=command, external=external, delay=1
            ),
            6: dict(
                kind="free",
                tree=ph.FreeSphere(2, 1 / 8, position=(0, 0, 2), gravity=(0, 0, -8)),
                command=(0, 0, 0),
            ),
        }
    from test_resident_affine_contact_transactions import _ordinary
    from test_resident_affine_contacts import (
        _asset as fixed_asset,
    )
    from test_resident_affine_contacts import (
        _control,
        _placed_case,
    )
    from test_resident_affine_contacts import (
        _world as fixed_world,
    )

    ordinary, command = _ordinary()
    data = _placed_case("A03")
    asset = fixed_asset(directory / "fixed", data)
    fixed = fixed_world(asset, data)
    return {
        503: _item(directory / "503", ref.case("lifecycle")),
        91: _item(directory / "91", ref.case("A05")),
        1207: dict(
            kind="ordinary", tree=ordinary, world=ordinary, command=command, control=command
        ),
        6: dict(
            kind="affine",
            data=data,
            asset=asset,
            tree=fixed,
            world=fixed,
            command=_control(fixed, asset, data),
            control=_control(fixed, asset, data),
        ),
    }


def _combined(batch, items):
    from test_resident_affine_contact_transactions import _ordinary_geometry
    from test_resident_affine_contacts import _contact_geometry

    shapes, bindings, exclusions, sources, laws = [], [], [], {}, {}
    for world, item in items.items():
        if item["kind"] == "floating_affine":
            if not item["data"]["contacts"]:
                continue
            model, bound, source, material = _geometry(batch, item, world=world)
        elif item["kind"] == "ordinary":
            model, bound, source, pair = _ordinary_geometry(batch, world)
            material = {pair: ph.RobotContactMaterial(64, 0, 0)}
        elif item["kind"] == "affine":
            model, bound, source, pair = _contact_geometry(
                batch, item["asset"], item["data"], world=world
            )
            material = {pair: ph.RobotContactMaterial(64, 0, 0)}
        else:
            continue
        shapes.extend(model.shapes)
        bindings.extend(bound)
        exclusions.extend(model.excluded)
        sources[world] = source
        laws.update(material)
    return ph.CollisionModel(tuple(shapes), tuple(exclusions)), tuple(bindings), sources, laws


def _snap(batch, plan):
    return dict(
        state=batch.snapshot(),
        geometry=_json(batch.tree_kinematics()),
        diagnostics=_json(batch.tree_diagnostics()),
        summary=batch.summary(),
        receipts=_json(plan.last_accepted),
        status=dict(plan.last_status),
        epoch=batch.stats()["state_epoch"],
        bank=batch.stats()["current_buffer"],
    )


def _retained(batch, plan, before, *, check_epoch=True):
    after = _snap(batch, plan)
    for field in ("state", "geometry", "summary", "receipts", "status"):
        assert after[field] == before[field], field
    # Existing batch invalidates transient diagnostics after a failed attempt.
    # Accepted receipt diagnostics stay intact; do not redefine this old API.
    assert all(value is None for value in after["diagnostics"].values())
    if check_epoch:
        assert after["bank"] == before["bank"]
        assert after["epoch"] == before["epoch"]


def _initial(items):
    from test_resident_affine_contact_transactions import _independent_initial

    result = {}
    for key, item in items.items():
        if item["kind"] in ("floating_affine", "floating"):
            result[key] = (item["data"]["q"].copy(), item["data"]["v"].copy())
        elif item["kind"] in ("ordinary", "affine"):
            result[key] = _independent_initial({key: item})[key]
    return result


def _advance_expected(items, previous, tick):
    from test_resident_affine_contact_transactions import _advance_oracle

    ref, result = _reference(), {}
    for key, (q, v) in previous.items():
        item = items[key]
        if item["kind"] == "floating_affine":
            result[key] = ref.advance(item["data"], q, v, _selected(item, tick))[:2]
        elif item["kind"] == "floating":
            initial = np.zeros_like(item["data"]["drive"])
            initial[:, 0] = np.asarray(item["data"]["q"][7:], np.float32)
            selected = initial if tick < item["delay"] else item["data"]["drive"]
            result[key] = ref.base().advance(item["data"], q, v, selected, item["external"], H)[:2]
        else:
            result[key] = _advance_oracle({key: item}, {key: (q, v)})[key]
    return result


def _audit_mixed(batch, items, entry, plan, sources, expected):
    from test_resident_affine_contact_transactions import _audit_mixed as old_audit
    from test_resident_floating_step import _audit as old_floating_audit

    rows, states, geometry, diagnostics, summary = (
        {},
        batch.snapshot(),
        batch.tree_kinematics(),
        batch.tree_diagnostics(),
        batch.summary(),
    )
    receipts = {} if plan is None else plan.last_accepted
    for key, item in items.items():
        if item["kind"] == "floating_affine":
            if item["data"]["contacts"]:
                rows[key] = _audit_contact(batch, item, receipts[key], sources[key], world=key)
            else:
                assert key not in receipts
                rows[key] = _audit(
                    item, entry[key], states[key], geometry[key], diagnostics[key], summary[key]
                )
        elif item["kind"] == "floating":
            assert key not in receipts
            initial = np.zeros_like(item["data"]["drive"])
            initial[:, 0] = np.asarray(item["data"]["q"][7:], np.float32)
            selected = initial if entry[key].tick < item["delay"] else item["data"]["drive"]
            rows[key] = old_floating_audit(
                item["data"],
                entry[key],
                states[key],
                geometry[key],
                diagnostics[key],
                summary[key],
                selected,
                item["external"],
            )
        else:
            rows[key] = old_audit(
                batch,
                {key: item},
                entry,
                receipts,
                sources,
                {key: expected[key]} if key in expected else {},
            ).get(key)
        if key in expected:
            _close(states[key].position, expected[key][0], atol=WINDOW_ATOL)
            _close(states[key].velocity, expected[key][1], atol=WINDOW_ATOL)
    return rows


def _mixed_window(items, mode, directory, *, category, steps, replay):
    expected, saved = _initial(items), None
    output = dict(
        category=category,
        mode=mode,
        source_sha256=_sources(),
        inputs={w: i.get("data", {"kind": i["kind"]}) for w, i in items.items()},
        declarations={w: i["tree"] for w, i in items.items()},
        commands={w: i["command"] for w, i in items.items()},
        rows=[],
        replay=[],
        saved=None,
    )
    with ph.TyxWorldBatch(
        [(w, i["tree"]) for w, i in items.items()], mode, cache_root=directory / "cache"
    ) as batch:
        batch.set_controls(output["commands"])
        plan, sources = None, {}
        if category == "contact":
            model, bound, sources, laws = _combined(batch, items)
            output.update(model=model, bindings=bound, sources=sources, laws=laws)
            plan = batch.prepare_contacts(model, bound, laws, tree_dynamic_pairs=True)
        output["initial"] = _record(batch, plan)
        output["compiled"] = _compiled(batch)
        try:
            for index in range(steps):
                entry = batch.snapshot()
                expected = _advance_expected(items, expected, index)
                flags = batch.step(H) if plan is None else batch.step_contacts(plan, H)
                assert flags == dict.fromkeys(items, True)
                row = _record(batch, plan)
                row.update(
                    entry=entry,
                    audit=_audit_mixed(batch, items, entry, plan, sources, expected),
                    independent={w: (q.copy(), v.copy()) for w, (q, v) in expected.items()},
                )
                output["rows"].append(row)
                if index == 18:
                    saved = batch.snapshot()
                    output["saved"] = saved
            if category == "contact" and steps == 40:
                for world, item in items.items():
                    if item["kind"] == "floating_affine" and item["data"]["name"] == "lifecycle":
                        loaded = [
                            bool(np.linalg.norm(r["receipts"][world].law_wrench_fp64) > 0.0)
                            for r in output["rows"]
                        ]
                        first_open = next(
                            (i for i, active in enumerate(loaded) if not active), None
                        )
                        assert (
                            loaded[0] and first_open is not None and any(loaded[first_open + 1 :])
                        ), loaded
                        output.setdefault("lifecycle", {})[world] = loaded
            if replay:
                assert steps == 40 and saved is not None
                batch.restore(saved)
                for index in range(19, 40):
                    assert batch.step_contacts(plan, H) == dict.fromkeys(items, True)
                    row = _record(batch, plan)
                    output["replay"].append(row)
                    for field in ("state", "geometry", "diagnostics", "summary", "receipts"):
                        assert _json(row[field]) == _json(output["rows"][index][field]), (
                            index,
                            field,
                        )
        except BaseException as exc:
            output["failure"] = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            output.update(
                retained=_record(batch, plan),
                stats=batch.stats(),
                plan_stats=None if plan is None else plan.stats(),
            )
            _save(directory, "trace", output)
            if plan is not None:
                plan.close()
    return output


def _compare_world(single, combined, world, steps):
    assert single["source_sha256"] == combined["source_sha256"]
    for i in range(steps):
        for field in ("state", "geometry", "diagnostics", "summary"):
            a, b = single["rows"][i][field], combined["rows"][i][field]
            if world in a:
                assert _json(a[world]) == _json(b[world]), (world, i, field)
        if "receipts" in single["rows"][i]:
            assert _json(single["rows"][i]["receipts"][world]) == _json(
                combined["rows"][i]["receipts"][world]
            )


@pytest.mark.tyx
@pytest.mark.parametrize("variant", ("core",), ids=("core",))
def test_w1_w4_affine_core_same_input(variant, tyx_mode, tmp_path):
    items = _mixed(tmp_path / "assets", "core")
    single = {
        w: _mixed_window(
            {w: item},
            tyx_mode,
            tmp_path / "core" / f"w1-{w}",
            category="core",
            steps=4,
            replay=False,
        )
        for w, item in items.items()
    }
    combined = _mixed_window(
        dict(reversed(tuple(items.items()))),
        tyx_mode,
        tmp_path / "core" / "w4",
        category="core",
        steps=4,
        replay=False,
    )
    for w in items:
        _compare_world(single[w], combined, w, 4)


@pytest.mark.tyx
@pytest.mark.parametrize("variant", ("contact",), ids=("contact",))
def test_w1_w4_affine_contact_replay(variant, tyx_mode, tmp_path):
    items = _mixed(tmp_path / "assets", "contact")
    single = {
        w: _mixed_window(
            {w: item},
            tyx_mode,
            tmp_path / "contact" / f"w1-{w}",
            category="contact",
            steps=40,
            replay=True,
        )
        for w, item in items.items()
    }
    combined = _mixed_window(
        dict(reversed(tuple(items.items()))),
        tyx_mode,
        tmp_path / "contact" / "w4",
        category="contact",
        steps=40,
        replay=True,
    )
    for w in items:
        _compare_world(single[w], combined, w, 40)


@pytest.mark.tyx
@pytest.mark.parametrize("variant", ("contact",), ids=("contact",))
def test_w32_bounded_affine_contact_batch(variant, tyx_mode, tmp_path):
    source = os.environ.get("PHYMEO_C11_W4_TRACE")
    assert source, "optional W32 requires explicit PHYMEO_C11_W4_TRACE from same-source W4"
    path = Path(source)
    assert path.is_absolute() and path.is_file()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        saved = json.load(stream)
    assert saved["source_sha256"] == _sources() and saved["mode"] == tyx_mode
    assert (
        saved["category"] == "contact" and len(saved["rows"]) == 40 and len(saved["replay"]) == 21
    )
    originals = _mixed(tmp_path / "assets", "contact")
    keys = (*KEYS, *(10001 + 97 * i for i in range(28)))
    items = {key: originals[KEYS[i % 4]] for i, key in enumerate(keys)}
    output = _mixed_window(
        items, tyx_mode, tmp_path / "contact" / "w32", category="contact", steps=4, replay=False
    )
    for index in range(4):
        for key in KEYS:
            for field in ("state", "geometry", "diagnostics", "summary", "receipts"):
                assert (
                    _json(output["rows"][index][field][key])
                    == saved["rows"][index][field][str(key)]
                )
    assert set(output["rows"][-1]["state"]) == set(keys)


X01 = (
    "fixed_asset",
    "full_width",
    "joint_position",
    "joint_velocity",
    "nonfinite_gain",
    "negative_cap",
    "model_token",
    "graph_token",
)


@pytest.mark.tyx
@pytest.mark.parametrize("problem", X01, ids=X01)
def test_x01_affine_spec_and_source_preflight(problem, tyx_mode, tmp_path):
    data = (
        _reference().case("A03", variant="fan")
        if problem == "graph_token"
        else _reference().case("A01")
    )
    item = _item(tmp_path / "asset", data)
    with ph.TyxWorldBatch([(WORLD, item["tree"])], tyx_mode, cache_root=tmp_path / "base") as batch:
        batch.set_controls({WORLD: item["command"]})
        before, allocation = batch.snapshot(), batch.stats()["device_allocations"]
        if problem in ("model_token", "graph_token"):
            if problem == "graph_token":
                foreign_item = _item(tmp_path / "chain", _reference().case("A03", variant="chain"))
                foreign_tree = foreign_item["tree"]
            else:
                original = item["asset"].tree
                root = replace(original.model.root, inverse_mass=np.array([1 / 3]))
                physical = replace(original, model=replace(original.model, root=root))
                changed = replace(item["asset"], tree=physical)
                foreign_tree = replace(item["tree"], asset=changed)
            with ph.TyxWorldBatch(
                [(WORLD, foreign_tree)], tyx_mode, cache_root=tmp_path / "foreign"
            ) as foreign:
                checkpoint = foreign.snapshot()
                with pytest.raises(
                    (ValueError, TypeError), match="graph|topology|model|belong|source"
                ):
                    batch.restore(checkpoint)
        else:

            def bad_declaration():
                if problem == "fixed_asset":
                    return replace(item["tree"], asset=ph.load_mimic_urdf(JAWS, gravity=(0, 0, 0)))
                if problem == "full_width":
                    return replace(item["tree"], state=item["asset"].full_state(_jaws_state()))
                if problem in ("joint_position", "joint_velocity"):
                    q, v = data["q"].copy(), data["v"].copy()
                    if problem == "joint_position":
                        q[7] = item["asset"].upper[0] + 1 / 1024
                    else:
                        v[6] = item["asset"].velocity_limit[0] + 1 / 1024
                    return replace(item["tree"], state=FloatingRobotState(q, v))
                if problem == "nonfinite_gain":
                    return replace(item["tree"], kp=np.array([np.nan]))
                return replace(item["tree"], effort_limit=np.array([-0.25]))

            with pytest.raises((ValueError, TypeError)):
                invalid = bad_declaration()
                with ph.TyxWorldBatch([(91, invalid)], tyx_mode, cache_root=tmp_path / "invalid"):
                    pass
        assert batch.snapshot() == before
        assert batch.stats()["device_allocations"] == allocation


@pytest.mark.tyx
@pytest.mark.parametrize(
    "problem",
    ("large_anchor", "full_q_mismatch", "full_v_mismatch"),
    ids=("large_anchor", "full_q_mismatch", "full_v_mismatch"),
)
def test_x02_local_origin_full_reduced_restore(problem, tyx_mode, tmp_path):
    data = _reference().case("A01")
    if problem == "large_anchor":
        data["q"][0] = 2.0**60
    item = _item(tmp_path / "asset", data, delay=1)
    output = dict(input=data, source=item["source"], command=item["command"], problem=problem)
    with ph.TyxWorldBatch(
        [(WORLD, item["tree"])], tyx_mode, cache_root=tmp_path / "cache"
    ) as batch:
        batch.set_controls({WORLD: item["command"]})
        initial = batch.snapshot()[WORLD]
        if problem == "large_anchor":
            world_q, full_q = list(initial.position), list(initial.full_position)
            world_q[0] = full_q[0] = float(2.0**60 + 1 / 1024)
            moved = replace(
                initial,
                position=tuple(world_q),
                full_position=tuple(full_q),
                local_origin_m=(1 / 1024, 0.0, 0.0),
            )
            batch.restore({WORLD: moved})
            assert batch.snapshot()[WORLD] == moved
            assert moved.position[0] == initial.position[0]
            assert moved.local_origin_m != initial.local_origin_m
        entry = batch.snapshot()
        try:
            assert batch.step(H) == {WORLD: True}
            first = _record(batch)
            saved = batch.snapshot()
            if problem != "large_anchor":
                field = "full_position" if problem == "full_q_mismatch" else "full_velocity"
                wrong = list(getattr(saved[WORLD], field))
                wrong[-1] += 1 / 1024
                allocation = batch.stats()["device_allocations"]
                with pytest.raises(ValueError):
                    batch.restore({WORLD: replace(saved[WORLD], **{field: tuple(wrong)})})
                assert batch.snapshot() == saved
                assert batch.stats()["device_allocations"] == allocation
            else:
                wrong_q = list(saved[WORLD].position)
                wrong_q[1] += 1 / 1024
                with pytest.raises(ValueError):
                    batch.restore({WORLD: replace(saved[WORLD], position=tuple(wrong_q))})
                assert batch.snapshot() == saved
            # Replay the same first request from its exact complete pre-step bank.
            batch.restore(entry)
            assert batch.step(H) == {WORLD: True}
            replay = _record(batch)
            assert _json(replay) == _json(first)
            output.update(initial=initial, entry=entry, saved=saved, first=first, replay=replay)
        except BaseException as exc:
            output["failure"] = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            output.update(retained=_record(batch), stats=batch.stats())
            _save(tmp_path, "restore-" + problem, output)


@pytest.mark.tyx
@pytest.mark.parametrize(
    "variant,warm,limited_world", (("bank0", 0, 503), ("bank1", 1, 91)), ids=("bank0", "bank1")
)
def test_x03_contact_capacity_preserves_both_banks(
    variant, warm, limited_world, tyx_mode, tmp_path
):
    items = _mixed(tmp_path / "assets", "contact")
    output = dict(
        inputs={w: i.get("data", i["kind"]) for w, i in items.items()},
        commands={w: i["command"] for w, i in items.items()},
        warm=warm,
        limited_world=limited_world,
        rows=[],
    )
    with ph.TyxWorldBatch(
        [(w, i["tree"]) for w, i in items.items()], tyx_mode, cache_root=tmp_path / "cache"
    ) as batch:
        batch.set_controls(output["commands"])
        model, bindings, sources, laws = _combined(batch, items)
        output.update(
            model=model, bindings=bindings, sources=sources, laws=laws, compiled=_compiled(batch)
        )
        with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as full:
            try:
                for _ in range(warm):
                    assert batch.step_contacts(full, H) == dict.fromkeys(items, True)
                    output["rows"].append(_record(batch, full))
                capacities = dict.fromkeys(items, 128)
                capacities[limited_world] = 0
                with batch.prepare_contacts(
                    model, bindings, laws, tree_dynamic_pairs=True, site_capacity=capacities
                ) as limited:
                    before = _snap(batch, full)
                    output["before"] = before
                    assert batch.step_contacts(limited, H) == {w: w != limited_world for w in items}
                    output["rejected"] = _record(batch, limited)
                    output["limited_status"] = dict(limited.last_status)
                    assert limited.last_status[limited_world] == "SITE_CAPACITY_EXCEEDED"
                    assert batch.snapshot()[limited_world] == before["state"][limited_world]
                    assert (
                        _json(batch.tree_kinematics()[limited_world])
                        == before["geometry"][str(limited_world)]
                    )
                    assert batch.summary()[limited_world] == before["summary"][limited_world]
                    assert limited_world not in limited.last_accepted
                    assert _json(full.last_accepted) == before["receipts"]
                    for world in items:
                        assert batch.snapshot()[world].tick == warm + (world != limited_world)
                    retained = batch.snapshot()[limited_world]
                    assert batch.step_contacts(full, H) == dict.fromkeys(items, True)
                    assert full.last_accepted[limited_world].entry_state == retained
                    output["retry"] = _record(batch, full)
                    for world in (503, 91):
                        _audit_contact(
                            batch,
                            items[world],
                            full.last_accepted[world],
                            sources[world],
                            world=world,
                        )
            except BaseException as exc:
                output["failure"] = dict(type=type(exc).__name__, message=str(exc))
                raise
            finally:
                output.update(retained=_record(batch, full), stats=batch.stats())
                _save(tmp_path, "capacity-" + variant, output)


LATE = tuple(
    (f"{phase}-b{warm}", phase, warm)
    for phase in ("upload", "candidate", "receipt")
    for warm in (0, 1)
)


@pytest.mark.tyx
@pytest.mark.parametrize("variant,phase,warm", LATE, ids=tuple(row[0] for row in LATE))
def test_x04_late_failure_preserves_graph_state_and_history(
    variant, phase, warm, tyx_mode, tmp_path, monkeypatch
):
    items = {
        503: _item(tmp_path / "a01", _reference().case("A01"), delay=1),
        91: _item(tmp_path / "a05", _reference().case("A05"), delay=1),
    }
    declarations, commands = (
        [(w, i["tree"]) for w, i in items.items()],
        {w: i["command"] for w, i in items.items()},
    )
    output = dict(
        inputs={w: i["data"] for w, i in items.items()}, commands=commands, phase=phase, warm=warm
    )
    with (
        ph.TyxWorldBatch(declarations, tyx_mode, cache_root=tmp_path / "fault") as batch,
        ph.TyxWorldBatch(declarations, tyx_mode, cache_root=tmp_path / "control") as control,
    ):
        batch.set_controls(commands)
        control.set_controls(commands)
        model, bindings, sources, laws = _combined(batch, items)
        other, bound2, _, laws2 = _combined(control, items)
        output.update(
            model=model, bindings=bindings, sources=sources, laws=laws, compiled=_compiled(batch)
        )
        with (
            batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as plan,
            control.prepare_contacts(other, bound2, laws2, tree_dynamic_pairs=True) as reference,
        ):
            try:
                for _ in range(warm):
                    assert (
                        batch.step_contacts(plan, H)
                        == control.step_contacts(reference, H)
                        == dict.fromkeys(items, True)
                    )
                before = _snap(batch, plan)
                output["before"] = before
                method = dict(
                    upload="_upload_contact", candidate="_read_candidate", receipt="_finish_attempt"
                )[phase]
                original, calls = getattr(plan, method), []

                def real_then_fail(*args, **kwargs):
                    result = original(*args, **kwargs)
                    calls.append(result)
                    if phase == "upload":
                        output["actual_uploaded_array"] = np.asarray(
                            args[1] if len(args) > 1 else kwargs["data"]
                        ).copy()
                    elif phase == "candidate":
                        output["actual_candidate_packets"] = result["packet"]
                        output["actual_candidate_groups"] = [
                            dict(worlds=tuple(storage.specs), data=arrays)
                            for storage, arrays in result["groups"].items()
                        ]
                    else:
                        output["actual_prepublication_receipts"] = result[0]
                        output["actual_prepublication_status"] = result[1]
                    raise RuntimeError("c11 actual " + phase + " before publication")

                with monkeypatch.context() as patch:
                    patch.setattr(plan, method, real_then_fail)
                    with pytest.raises(RuntimeError, match="c11 actual"):
                        batch.step_contacts(plan, H)
                assert calls
                _retained(batch, plan, before)
                output["failed_retained"] = _snap(batch, plan)
                plan.close()
                with batch.prepare_contacts(
                    model, bindings, laws, tree_dynamic_pairs=True
                ) as retry:
                    assert (
                        batch.step_contacts(retry, H)
                        == control.step_contacts(reference, H)
                        == dict.fromkeys(items, True)
                    )
                    actual, expected = _record(batch, retry), _record(control, reference)
                    assert _json(actual) == _json(expected)
                    output.update(retry=actual, control=expected)
            except BaseException as exc:
                output["failure"] = dict(type=type(exc).__name__, message=str(exc))
                raise
            finally:
                output.update(retained=_record(batch, plan), stats=batch.stats())
                _save(tmp_path, "late-" + variant, output)


@pytest.mark.tyx
@pytest.mark.parametrize("variant", ("stale", "close_rebuild"), ids=("stale", "close_rebuild"))
def test_x05_stale_restore_and_closed_rebuild(variant, tyx_mode, tmp_path, monkeypatch):
    item = _item(tmp_path / "asset", _reference().case("A05"), delay=1)
    output = dict(
        input=item["data"], source=item["source"], command=item["command"], variant=variant
    )
    batch = ph.TyxWorldBatch([(WORLD, item["tree"])], tyx_mode, cache_root=tmp_path / "first")
    plan = None
    try:
        batch.set_controls({WORLD: item["command"]})
        model, bindings, sources, laws = _combined(batch, {WORLD: item})
        output.update(
            model=model, bindings=bindings, sources=sources, laws=laws, compiled=_compiled(batch)
        )
        plan = batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True)
        if variant == "stale":
            before, original, calls = _snap(batch, plan), plan._read_candidate, []

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
            _retained(batch, plan, before, check_epoch=False)
            plan.close()
            with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as retry:
                assert batch.step_contacts(retry, H) == {WORLD: True}
                assert retry.last_accepted[WORLD].entry_state == before["state"][WORLD]
                output.update(before=before, retry=_record(batch, retry))
        else:
            assert batch.step_contacts(plan, H) == {WORLD: True}
            saved = batch.snapshot()
            output["saved"] = saved
            assert batch.step_contacts(plan, H) == {WORLD: True}
            expected = _record(batch, plan)
            output["expected"] = expected
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
                m2, b2, _, l2 = _combined(rebuilt, {WORLD: item})
                with rebuilt.prepare_contacts(m2, b2, l2, tree_dynamic_pairs=True) as retry:
                    assert rebuilt.step_contacts(retry, H) == {WORLD: True}
                    actual = _record(rebuilt, retry)
                    assert _json(actual) == _json(expected)
                    output.update(rebuilt=actual, rebuilt_stats=rebuilt.stats())
    except BaseException as exc:
        output["failure"] = dict(type=type(exc).__name__, message=str(exc))
        raise
    finally:
        output["stats_before_close"] = batch.stats()
        batch.close()
        output["closed_stats"] = batch.stats()
        _save(tmp_path, "owner-" + variant, output)


X06 = (
    "body17-n16",
    "body16-n15",
    "drive_full_n",
    "external_joint_effort",
    "wrong_reduced_width",
    "cross_independent_tree",
)


@pytest.mark.tyx
@pytest.mark.parametrize("problem", X06, ids=X06)
def test_x06_body_motor_and_domain_preflight(problem, tyx_mode, tmp_path):
    ref = _reference()
    data = ref.case("counts", variant="n16-k1") if problem.startswith("body") else ref.case("A05")
    if problem == "body16-n15":
        for field, value in data["children"].items():
            data["children"][field] = value[:15]
        ref._graph(data, data["P"][:15], data["offset"][:15], data["C"][:14, 6:21], [0])
        data["caps"][:] = 8.0
    item = _item(tmp_path / "asset", data)
    worlds = [(WORLD, item["tree"])]
    if problem == "cross_independent_tree":
        worlds.append((91, item["tree"]))
    with ph.TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "cache") as batch:
        batch.set_controls({w: item["command"] for w, _ in worlds})
        before, allocation = batch.snapshot(), batch.stats()["device_allocations"]
        with pytest.raises((ValueError, TypeError)):
            if problem == "drive_full_n":
                batch.set_controls({WORLD: item["tree"].inputs(np.zeros((len(data["effort"]), 3)))})
            elif problem == "external_joint_effort":
                effort = np.zeros(len(data["effort"]))
                effort[-1] = 1 / 8
                loads = FloatingRobotInputs(effort, data["force"], data["torque"])
                batch.set_controls({WORLD: item["tree"].inputs(data["drive"], external=loads)})
            elif problem == "wrong_reduced_width":
                original = before[WORLD]
                batch.restore({WORLD: replace(original, position=original.position + (0.0,))})
            else:
                group = 17 if problem == "body17-n16" else 16 if problem == "body16-n15" else 1
                owner = 91 if problem == "cross_independent_tree" else WORLD
                key = ph.CollisionKey(WORLD, 100 + group, 11)
                shape = ph.CollisionShape(key, "sphere", (31,), radius=1 / 8)
                bound = ph.ResidentFloatingAffineTreeBinding(
                    key,
                    owner,
                    group,
                    np.zeros((1, 3)),
                    topology=before[owner].topology,
                    graph_token=item["command"].graph_token,
                )
                with batch.prepare_collision(ph.CollisionModel((shape,)), (bound,)):
                    pass
        assert batch.snapshot() == before
        assert batch.stats()["device_allocations"] == allocation


@pytest.mark.tyx
@pytest.mark.parametrize(
    "variant",
    ("full_nan", "reduced_nan", "stale_tick"),
    ids=("full_nan", "reduced_nan", "stale_tick"),
)
def test_x07_candidate_diagnostics_reject_corrupted_readback(
    variant, tyx_mode, tmp_path, monkeypatch
):
    """Real readback enters the normal prepublication constructor before rejection."""
    item = _item(tmp_path / "asset", _reference().case("A05"), delay=1)
    output = dict(
        input=item["data"],
        source=item["source"],
        command=item["command"],
        variant=variant,
        mode=tyx_mode,
        source_sha256=_sources(),
        injections=[],
    )
    with ph.TyxWorldBatch(
        [(WORLD, item["tree"])], tyx_mode, cache_root=tmp_path / "cache"
    ) as batch:
        batch.set_controls({WORLD: item["command"]})
        model, bindings, sources, laws = _combined(batch, {WORLD: item})
        output.update(
            model=model, bindings=bindings, sources=sources, laws=laws, compiled=_compiled(batch)
        )
        with batch.prepare_contacts(model, bindings, laws, tree_dynamic_pairs=True) as plan:
            try:
                output["initial"] = _record(batch, plan)
                assert batch.step_contacts(plan, H) == {WORLD: True}
                assert batch.snapshot()[WORLD].tick == 1
                output["warm"] = _record(batch, plan)
                output["warm_audit"] = _audit_contact(
                    batch, item, plan.last_accepted[WORLD], sources[WORLD]
                )
                before = _snap(batch, plan)
                output["before"] = before
                targets = [storage for specs, storage in batch._tree_storages() if WORLD in specs]
                assert len(targets) == 1
                storage = targets[0]
                assert storage.is_floating and storage.is_affine
                assert storage.specs[WORLD].domain == "floating_affine_tree"
                row = storage.rows[WORLD]
                construct = storage.diagnostics_from_arrays

                def corrupt_once(worlds, full_arrays, reduced_arrays):
                    selected = tuple(worlds)
                    if WORLD not in selected or output["injections"]:
                        return construct(selected, full_arrays, reduced_arrays)
                    assert selected == (WORLD,)
                    # These are actual same-attempt device reads, not synthetic
                    # mechanics. Copy only to inject a failed transfer result.
                    original_full = {name: value.copy() for name, value in full_arrays.items()}
                    original_reduced = {
                        name: value.copy() for name, value in reduced_arrays.items()
                    }
                    full = {name: value.copy() for name, value in full_arrays.items()}
                    reduced = {name: value.copy() for name, value in reduced_arrays.items()}
                    assert full["info"][row, 0] == before["state"][WORLD].tick
                    assert full["info"][row, 1] == before["state"][WORLD].tick + 1
                    assert full["info"][row, 2] == 0
                    if variant == "full_nan":
                        owner, field = "full", "mass"
                        assert np.isfinite(full[field][row]).all()
                        full[field][row, 0, 0] = np.nan
                    elif variant == "reduced_nan":
                        owner, field = "reduced", "mass"
                        assert np.isfinite(reduced[field][row]).all()
                        reduced[field][row, 0, 0] = np.nan
                    else:
                        owner, field = "full", "info"
                        full[field][row, 1] = full[field][row, 0]
                    old_array = (original_full if owner == "full" else original_reduced)[field][row]
                    new_array = (full if owner == "full" else reduced)[field][row]
                    changed = dict(
                        owner=owner,
                        field=field,
                        dtype=str(new_array.dtype),
                        shape=tuple(new_array.shape),
                        before=old_array.copy(),
                        after=new_array.copy(),
                    )
                    if variant in ("full_nan", "reduced_nan"):
                        assert new_array.dtype == np.dtype(np.float32)
                        changed.update(
                            before_uint32_bits=old_array.view(np.uint32).copy(),
                            after_uint32_bits=new_array.view(np.uint32).copy(),
                        )
                    else:
                        assert new_array.dtype == np.dtype(np.int32)
                        changed.update(
                            before_int32_values=old_array.copy(),
                            after_int32_values=new_array.copy(),
                        )
                    output["injections"].append(
                        dict(
                            world=WORLD,
                            storage_row=row,
                            selected_worlds=selected,
                            original_full=original_full,
                            original_reduced=original_reduced,
                            modified_full=full,
                            modified_reduced=reduced,
                            modified_array=changed,
                        )
                    )
                    # Failure must come from the production constructor itself.
                    return construct(selected, full, reduced)

                with monkeypatch.context() as patch:
                    patch.setattr(storage, "diagnostics_from_arrays", corrupt_once)
                    with pytest.raises(
                        ValueError, match="nonfinite|ticks.*physical step"
                    ) as rejected:
                        batch.step_contacts(plan, H)
                assert len(output["injections"]) == 1
                output["rejected"] = dict(
                    type=type(rejected.value).__name__, message=str(rejected.value)
                )
                _retained(batch, plan, before)
                output["failed_retained"] = _snap(batch, plan)
                # Same held command and unchanged accepted entry; no restore,
                # extra warm step, control replacement or manual device write.
                assert batch.step_contacts(plan, H) == {WORLD: True}
                receipt = plan.last_accepted[WORLD]
                assert receipt.entry_state == before["state"][WORLD]
                assert batch.snapshot()[WORLD].tick == before["state"][WORLD].tick + 1 == 2
                output["retry_audit"] = _audit_contact(batch, item, receipt, sources[WORLD])
                output["retry"] = _record(batch, plan)
                assert len(output["injections"]) == 1
            except BaseException as exc:
                output["failure"] = dict(type=type(exc).__name__, message=str(exc))
                raise
            finally:
                output.update(retained=_record(batch, plan), stats=batch.stats())
                _save(tmp_path, "diagnostic-" + variant, output)

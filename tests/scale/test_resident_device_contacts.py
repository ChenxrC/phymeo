"""B2 single-page device patches: independent loaded mechanics and atomic publication.

Predeclared: three continuous steps; original physical gates 5e-6 absolute,
5e-5 relative (mass 2e-5). Geometry gates are B1's original local-scale bounds.
No physics state is reset between the positive trajectory steps.
"""

import gzip
import json
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pytest
import test_resident_floating_affine_contacts as floating_affine
import test_resident_floating_contacts as floating
import test_resident_tree_dynamic_contacts as dynamic
from test_resident_contact_bundle import _spies
from test_resident_tree_contacts import H, _action, _law, _r01_bound, _slider

import phymeo as ph
from phymeo.backends import _resident_contact_bridge as bridge
from phymeo.backends import _resident_contact_plan as contact_module


def fixture(layout, path):
    if layout in ("ordinary", "affine"):
        data = dynamic._case(
            "D03" if layout == "ordinary" else "D04",
            velocity=(1 / 128, -1 / 256) if layout == "ordinary" else (1 / 128,),
        )
        data["motor"][0] = 1 / 16
        data["force"][-1, 0] = 1 / 32
        data["torque"][0, 2] = 1 / 64
        item = dynamic._build(data, path)
        return item, item["world"], item["control"]
    if layout == "floating":
        data = floating._reference().case("F04")
        data["force"][0, 1] = 1 / 32
        data["torque"][0, 2] = 1 / 64
        item = floating._item(data)
    else:
        data = floating_affine._reference().case("A02")
        data["v"][0] = 1 / 256
        data["drive"][0, 2] = 1 / 16
        data["force"][0, 1] = 1 / 32
        data["torque"][0, 2] = 1 / 64
        item = floating_affine._item(path, data)
    return item, item["tree"], item["command"]


def geometry(layout, batch, item, world):
    module = (
        dynamic
        if layout in ("ordinary", "affine")
        else floating
        if layout == "floating"
        else floating_affine
    )
    return module._geometry(batch, item, world=world)


def audit(layout, batch, item, entry, receipt, sources, world):
    if layout in ("ordinary", "affine"):
        return dynamic._audit(batch, item, entry, receipt, sources, world=world)
    if layout == "floating":
        return floating._audit(batch, item, receipt, sources, world=world)
    return floating_affine._audit_contact(batch, item, receipt, sources, world=world)


def no_host_search(*args, **kwargs):
    raise AssertionError("device_patch attempted HOST contact patch search")


def compact_record(conversion):
    query = conversion.query
    # Owner/registration are opaque process identities, retained in memory and
    # compared by identity. Every numerical/raw/source field is saved.
    return dict(
        state_epoch=query.state_epoch,
        active=query.active,
        ticks=query.ticks,
        shape_keys=query.shape_keys,
        shape_revisions=query.shape_revisions,
        rows=query.rows,
        raw=query.raw,
        source=query.source,
        patch=query.patch,
        stats=query.stats,
        attempt_tag=query.attempt_tag,
        samples=conversion.samples,
        differences=conversion.differences,
        patch_differences=conversion.patch_differences,
    )


def encode(value):
    if isinstance(value, np.ndarray):
        return dict(
            dtype=value.dtype.str,
            shape=value.shape,
            bytes_hex=value.tobytes(order="C").hex(),
            values=encode(value.tolist()),
        )
    if isinstance(value, np.generic):
        return encode(value.item())
    if type(value) is object:
        return dict(opaque_process_local_type="builtins.object", identity_serialized=False)
    if is_dataclass(value):
        return {f.name: encode(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return dict(mapping_entries=[(encode(k), encode(v)) for k, v in value.items()])
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("unrecorded value " + type(value).__qualname__)


def save_rows(tmp_path, name, rows):
    with gzip.open(tmp_path / (name + ".json.gz"), "wt", encoding="utf-8") as stream:
        json.dump(encode(rows), stream, allow_nan=False)


@pytest.mark.tyx
@pytest.mark.parametrize("layout", ("ordinary", "affine", "floating", "floating_affine"))
def test_four_layouts_true_entry_velocity_drive_and_loads(layout, tyx_mode, tmp_path, monkeypatch):
    item, tree, command = fixture(layout, tmp_path / "asset")
    world, rows = 107, []
    with ph.TyxWorldBatch([(world, tree)], tyx_mode, cache_root=tmp_path / "cache") as batch:
        batch.set_controls({world: command})
        model, bindings, sources, laws = geometry(layout, batch, item, world)
        with batch.prepare_contacts(
            model, bindings, laws, tree_dynamic_pairs=True, narrow_phase="device_patch"
        ) as plan:
            host_search = bridge.contact_patch_sample
            monkeypatch.setattr(bridge, "contact_patch_sample", no_host_search)
            counts = _spies(monkeypatch, batch, plan)
            try:
                assert any(abs(x) > 0 for x in batch.snapshot()[world].velocity)
                for tick in range(3):
                    entry = batch.snapshot()[world]
                    assert batch.step_contacts(plan, H) == {world: True}
                    receipt = plan.last_accepted[world]
                    converted = receipt.diagnostics["device_patch"]
                    truth = audit(layout, batch, item, entry, receipt, sources, world)
                    host_samples = {}
                    entry_frames = receipt.diagnostics["entry_frames"]
                    for pair, effective in converted.samples.items():
                        first, second = (entry_frames[key] for key in pair)
                        host_sample = host_search(
                            first.shape, second.shape, first.world_points_m, second.world_points_m
                        )
                        # These four predeclared sphere-based physical cases have
                        # a unique point. General face bijections are in the other file.
                        assert len(host_sample.sites) == len(effective.sites) == 1
                        wanted = host_sample.sites[0]
                        raw_row = next(row for row in converted.query.rows if row.pair == pair)
                        raw = raw_row.patch.sites[0]
                        raw_origin = np.asarray(raw_row.origin_m)
                        scale = max(
                            1.0,
                            first.shape.radius,
                            second.shape.radius,
                            float(np.ptp(first.world_points_m, axis=0).max()),
                            float(np.ptp(second.world_points_m, axis=0).max()),
                        )
                        for observed_gap in (raw.gap_m, effective.sites[0].witness.gap_m):
                            assert abs(observed_gap - wanted.witness.gap_m) <= 1e-4 * scale
                        for side in ("first", "second"):
                            wanted_point = getattr(wanted.witness, side + "_point")
                            for observed in (
                                raw_origin + getattr(raw, side).surface_relative_m,
                                getattr(effective.sites[0].witness, side + "_point"),
                            ):
                                np.testing.assert_allclose(
                                    observed, wanted_point, atol=2e-4 * scale, rtol=0
                                )
                        for observed in (raw.normal, effective.sites[0].witness.normal):
                            np.testing.assert_allclose(
                                observed, wanted.witness.normal, atol=2e-3, rtol=0
                            )
                        host_samples[pair] = host_sample
                    assert plan.last_device_query is converted.query
                    assert converted.query.state_epoch == plan.last_result.state_epoch
                    assert tuple(converted.query.requested_pairs) == tuple(
                        plan.last_result.candidates
                    )
                    assert converted.query.ticks[0] == entry.tick
                    assert receipt.accepted_state.tick == tick + 1
                    with pytest.raises(ValueError, match="stale"):
                        plan._device_patch_query.require_current(plan.last_device_query)
                    assert counts[(0, "mechanics")] == counts[(0, "solve")] == tick + 1
                    assert plan.stats()["device_patch_queries"] == tick + 1
                    assert receipt.sites and np.any(receipt.law_wrench_fp64 != 0)
                    np.testing.assert_allclose(
                        sum(
                            (s.first_force_n + s.second_force_n for s in receipt.sites), np.zeros(3)
                        ),
                        0,
                        atol=1e-12,
                        rtol=0,
                    )
                    # Each physical owner gets the correct COM couple, including
                    # two dynamic branches; no duplicated quadrature weighting.
                    for key, frame in receipt.diagnostics["entry_frames"].items():
                        force, moment = np.zeros(3), np.zeros(3)
                        for site in receipt.sites:
                            for side, owner in zip(("first", "second"), site.pair):
                                if owner == key:
                                    f = getattr(site, side + "_force_n")
                                    force += f
                                    moment += np.cross(
                                        np.asarray(site.geometry.load_point)
                                        - frame.reference_point_m,
                                        f,
                                    )
                        if frame.group_index is None:
                            actual = receipt.static_reactions[key]
                            np.testing.assert_allclose(actual.force_n, force, atol=1e-12, rtol=0)
                            np.testing.assert_allclose(actual.torque_nm, moment, atol=1e-12, rtol=0)
                    rows.append(
                        dict(
                            entry=entry,
                            accepted=receipt.accepted_state,
                            truth=truth,
                            host_same_entry=host_samples,
                            geometry=compact_record(converted),
                            sites=receipt.sites,
                            wrench=receipt.law_wrench_fp64,
                            diagnostics={
                                k: v for k, v in receipt.diagnostics.items() if k != "device_patch"
                            },
                        )
                    )
            finally:
                save_rows(
                    tmp_path,
                    "B2-four-layout-" + layout,
                    dict(inputs=item["data"], command=command, rows=rows),
                )


def slider_batch(mode, path, worlds=(7, 31)):
    program, tree = _slider(v=1 / 64)
    batch = ph.TyxWorldBatch([(w, tree) for w in worlds], mode, cache_root=path)
    batch.set_controls({w: _action(program, ff=[1 / 8], effort=[1 / 4]) for w in worlds})
    shapes, bindings, laws = [], [], {}
    for world in worlds:
        model, attached, _, pair = _r01_bound(batch, world)
        shapes.extend(model.shapes)
        bindings.extend(attached)
        laws[pair] = _law()
    return batch, ph.CollisionModel(tuple(shapes)), tuple(bindings), laws


@pytest.mark.tyx
def test_default_host_and_device_same_input_and_independent_slider(tyx_mode, tmp_path, monkeypatch):
    paths = {}
    calls = []
    original = bridge.contact_patch_sample

    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(bridge, "contact_patch_sample", observed)
    for mode in ("host_patch", "device_patch"):
        batch, model, bindings, laws = slider_batch(tyx_mode, tmp_path / mode, (7,))
        with batch:
            kwargs = {} if mode == "host_patch" else dict(narrow_phase=mode)
            with batch.prepare_contacts(model, bindings, laws, **kwargs) as plan:
                start_calls = len(calls)
                q, v, rows = 0.0, 1 / 64, []
                for _ in range(3):
                    force = 64 * (1 / 32 - q)
                    v += H * (force / 2 + 3 / 16)
                    q += H * v
                    assert batch.step_contacts(plan, H) == {7: True}
                    state, receipt = batch.snapshot()[7], plan.last_accepted[7]
                    np.testing.assert_allclose(state.position, [q], atol=5e-6, rtol=5e-5)
                    np.testing.assert_allclose(state.velocity, [v], atol=5e-6, rtol=5e-5)
                    np.testing.assert_allclose(
                        receipt.contact_generalized, [force], atol=5e-6, rtol=5e-5
                    )
                    assert len(receipt.sites) == 4
                    rows.append((state.position, state.velocity, receipt.law_wrench_fp64))
                assert (len(calls) > start_calls) == (mode == "host_patch")
                paths[mode] = rows
    for host, device in zip(paths["host_patch"], paths["device_patch"], strict=True):
        for a, b in zip(host, device, strict=True):
            np.testing.assert_allclose(a, b, atol=5e-6, rtol=5e-5)
    save_rows(tmp_path, "B2-host-device-independent", paths)


@pytest.mark.tyx
@pytest.mark.parametrize(
    "fault",
    (
        "patch_readback",
        "last_load",
        "post_solve",
        "unknown_patch",
        "source_epoch",
        "source_owner",
        "source_registration",
        "source_bank",
        "source_layout",
        "source_membership",
    ),
)
def test_late_failure_retains_all_publications_and_retry(fault, tyx_mode, tmp_path, monkeypatch):
    batch, model, bindings, laws = slider_batch(tyx_mode, tmp_path / "cache")
    with batch:
        with batch.prepare_contacts(model, bindings, laws, narrow_phase="device_patch") as plan:
            assert batch.step_contacts(plan, H) == {7: True, 31: True}
            before = batch.snapshot()
            accepted, status, broad, device = (
                plan.last_accepted,
                plan.last_status,
                plan.last_result,
                plan.last_device_query,
            )
            query = plan._device_patch_query
            queries = query.stats["queries"]
            calls = [0]
            with monkeypatch.context() as patch:
                if fault == "patch_readback":
                    target, name = query, "_read_patch"

                    def broken(*args, **kwargs):
                        raise RuntimeError("late patch readback")
                elif fault == "last_load":
                    target, name = contact_module, "sample_tree_contacts"
                    original = getattr(target, name)

                    def broken(*args, **kwargs):
                        calls[0] += 1
                        if calls[0] == 2:
                            raise RuntimeError("last world load")
                        return original(*args, **kwargs)
                elif fault == "post_solve":
                    target, name = plan, "_finish_attempt"
                    original = getattr(target, name)

                    def broken(*args, **kwargs):
                        original(*args, **kwargs)
                        raise RuntimeError("complete receipt before publication")
                else:
                    target, name = query, "_build_result"
                    original = getattr(target, name)

                    def broken(*args, **kwargs):
                        result = original(*args, **kwargs)
                        if fault == "source_epoch":
                            return replace(result, state_epoch=result.state_epoch + 1)
                        if fault == "source_owner":
                            return replace(result, owner=object())
                        if fault == "source_registration":
                            return replace(result, registration=object())
                        if fault == "source_bank":
                            return replace(result, active=1 - result.active)
                        if fault == "source_layout":
                            source = dict(result.source)
                            source["shape_layouts"] = np.ones_like(source["shape_layouts"])
                            return replace(result, source=source)
                        if fault == "source_membership":
                            return replace(result, rows=result.rows[:-1])
                        row = result.rows[-1]
                        row = replace(row, patch=replace(row.patch, status="UNRESOLVED"))
                        return replace(result, rows=result.rows[:-1] + (row,))

                patch.setattr(target, name, broken)
                with pytest.raises((RuntimeError, ValueError)):
                    batch.step_contacts(plan, H)
            assert batch.snapshot() == before
            assert plan.last_accepted is accepted and plan.last_status is status
            assert plan.last_result is broad and plan.last_device_query is device
            assert query.stats["queries"] == queries
            assert batch.step_contacts(plan, H) == {7: True, 31: True}
            assert all(s.tick == 2 for s in batch.snapshot().values())
            assert query.stats["queries"] == queries + 1
            save_rows(
                tmp_path,
                "B2-fault-" + fault,
                dict(
                    before=before,
                    after=batch.snapshot(),
                    conversion=compact_record(plan.last_accepted[31].diagnostics["device_patch"]),
                    stats=plan.stats(),
                ),
            )


@pytest.mark.tyx
def test_global_resource_error_and_exact_per_world_site_capacity(tyx_mode, tmp_path):
    for global_capacity in (7, 8):
        batch, model, bindings, laws = slider_batch(tyx_mode, tmp_path / str(global_capacity))
        with batch:
            before = batch.snapshot()
            with batch.prepare_contacts(
                model,
                bindings,
                laws,
                narrow_phase="device_patch",
                device_patch_site_capacity=global_capacity,
                site_capacity={7: 4, 31: 3},
            ) as plan:
                if global_capacity == 7:
                    with pytest.raises((RuntimeError, ValueError), match="patch|resource|RESOURCE"):
                        batch.step_contacts(plan, H)
                    assert batch.snapshot() == before
                    assert not plan.last_accepted and plan.last_result is None
                    assert plan.last_device_query is None
                else:
                    assert batch.step_contacts(plan, H) == {7: True, 31: False}
                    assert batch.snapshot()[31] == before[31]
                    assert batch.snapshot()[7].tick == 1
                    assert plan.last_status[31] == "SITE_CAPACITY_EXCEEDED"
                    assert set(plan.last_accepted) == {7}
                    old = plan.last_device_query
                    batch.reset([7])
                    with pytest.raises(ValueError, match="stale"):
                        plan._device_patch_query.require_current(old)
                    assert batch.step_contacts(plan, H) == {7: True, 31: False}
                query = plan._device_patch_query
            assert query.closed


@pytest.mark.tyx
def test_independent_query_still_publishes_and_bundle_stays_host(tyx_mode, tmp_path, monkeypatch):
    from contextlib import closing

    from phymeo.backends._resident_contact_bundle import ResidentContactPageSpec

    batch, model, bindings, laws = slider_batch(tyx_mode, tmp_path / "cache", (7,))
    with batch:
        with batch.prepare_collision(model, bindings) as broad:
            with closing(batch.prepare_convex_query(broad, site_capacity=8)) as query:
                result = query.query()
                assert query.last_result is result and broad.last_result is not None
                assert query.stats["queries"] == 1
                assert query.require_current(result) is result
        calls = []
        original = bridge.contact_patch_sample

        def observed(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        monkeypatch.setattr(bridge, "contact_patch_sample", observed)
        with batch.prepare_contact_bundle(
            (ResidentContactPageSpec(model, bindings, laws),)
        ) as bundle:
            assert batch.step_contacts(bundle, H) == {7: True}
            assert calls
            assert "device_patch" not in bundle.last_accepted[7].diagnostics


@pytest.mark.tyx
@pytest.mark.parametrize(
    "option", ("bad_mode", "host_capacity", "zero_capacity", "overflow_capacity")
)
def test_public_device_options_fail_before_registration(option, tyx_mode, tmp_path):
    batch, model, bindings, laws = slider_batch(tyx_mode, tmp_path, (7,))
    kwargs = {
        "bad_mode": dict(narrow_phase="automatic"),
        "host_capacity": dict(device_patch_site_capacity=8),
        "zero_capacity": dict(narrow_phase="device_patch", device_patch_site_capacity=0),
        "overflow_capacity": dict(narrow_phase="device_patch", device_patch_site_capacity=4097),
    }[option]
    with batch:
        before = batch.snapshot()
        with pytest.raises((TypeError, ValueError)):
            batch.prepare_contacts(model, bindings, laws, **kwargs)
        assert batch.snapshot() == before


@pytest.mark.tyx
@pytest.mark.parametrize("sign", (-1, 1))
def test_same_entry_sliding_friction_has_one_weight_and_real_drive(
    sign, tyx_mode, tmp_path, monkeypatch
):
    from test_resident_tree_contacts import _program

    model = ph.RobotModel(
        np.array([-1]),
        ("prismatic",),
        np.array([[0.0, 0.0, 3 / 32]]),
        np.array([[0.0, 0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        np.array([2.0]),
        np.zeros((1, 3)),
        np.full((1, 3), 1 / 8),
        np.zeros(3),
    )
    program, tree = _program(model, v=[sign / 16])
    command = _action(program, ff=[1 / 8], effort=[1 / 4])
    with ph.TyxWorldBatch([(7, tree)], tyx_mode, cache_root=tmp_path) as batch:
        batch.set_controls({7: command})
        collision, bindings, _, pair = _r01_bound(batch, 7)
        law = ph.RobotContactMaterial(64, 1, 1 / 2, 1 / 32)
        with batch.prepare_contacts(
            collision, bindings, {pair: law}, narrow_phase="device_patch"
        ) as plan:
            monkeypatch.setattr(bridge, "contact_patch_sample", no_host_search)
            q, v, rows = 0.0, sign / 16, []
            for _ in range(3):
                # Full patch N=2 N, saturated tangential magnitude=1 N.
                # Two kg actual slider, plus original 1/8 drive + 1/4 load.
                v += H * (-sign + 3 / 8) / 2
                q += H * v
                assert batch.step_contacts(plan, H) == {7: True}
                receipt = plan.last_accepted[7]
                np.testing.assert_allclose(
                    receipt.law_wrench_fp64[0],
                    [-sign, 0, 2, 0, sign * 7 / 64, 0],
                    atol=5e-6,
                    rtol=5e-5,
                )
                np.testing.assert_allclose(
                    receipt.contact_generalized, [-sign], atol=5e-6, rtol=5e-5
                )
                np.testing.assert_allclose(
                    receipt.accepted_state.velocity, [v], atol=5e-6, rtol=5e-5
                )
                np.testing.assert_allclose(
                    receipt.accepted_state.position, [q], atol=5e-6, rtol=5e-5
                )
                assert len(receipt.sites) == 4
                for site in receipt.sites:
                    assert site.first_force_n @ site.first_velocity_m_s < 0
                rows.append(
                    dict(
                        state=receipt.accepted_state,
                        wrench=receipt.law_wrench_fp64,
                        sites=receipt.sites,
                        geometry=compact_record(receipt.diagnostics["device_patch"]),
                    )
                )
            save_rows(
                tmp_path, "B2-sliding-" + str(sign), dict(law=law, command=command, rows=rows)
            )


@pytest.mark.tyx
def test_zero_force_sites_still_consume_the_complete_capacity(tyx_mode, tmp_path):
    batch, model, bindings, laws = slider_batch(tyx_mode, tmp_path)
    laws = {pair: ph.RobotContactMaterial(64, 256, 0) for pair in laws}
    with batch:
        before = batch.snapshot()
        with batch.prepare_contacts(
            model, bindings, laws, narrow_phase="device_patch", site_capacity={7: 4, 31: 3}
        ) as plan:
            assert batch.step_contacts(plan, H) == {7: True, 31: False}
            assert batch.snapshot()[31] == before[31]
            receipt = plan.last_accepted[7]
            assert len(receipt.sites) == 4
            np.testing.assert_array_equal(receipt.law_wrench_fp64, np.zeros((1, 6)))
            assert plan.last_status[31] == "SITE_CAPACITY_EXCEEDED"
            assert len(plan.last_device_query.rows[-1].patch.sites) == 4

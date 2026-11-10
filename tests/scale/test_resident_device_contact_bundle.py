"""B3 paged device contacts: frozen tests authored before implementation.

All modes use real entry velocities, original H and complete per-world forces.
The common geometry is serialized once per step; receipt fields reference that
same full payload explicitly, avoiding W256 quadratic duplicate evidence.
"""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields, is_dataclass, replace

import numpy as np
import pytest
import test_resident_device_contacts as b2
from test_resident_contact_bundle import _batch, _spies
from test_resident_tree_contacts import H, _audit_receipt, _law, _r01_bound

import phymeo as ph
from phymeo.backends import _resident_contact_bridge as bridge
from phymeo.backends import _resident_contact_bundle as bundle_module
from phymeo.backends._resident_contact_bundle import ResidentContactPageSpec


def pages(batch, groups):
    specs, sources = [], {}
    for group in groups:
        shapes, bindings, laws = [], [], {}
        for world in group:
            model, bound, original, pair = _r01_bound(batch, world)
            shapes.extend(model.shapes)
            bindings.extend(bound)
            laws[pair] = _law()
            sources[world] = original
        specs.append(
            ResidentContactPageSpec(
                ph.CollisionModel(tuple(shapes)), tuple(bindings), laws, len(group)
            )
        )
    return tuple(specs), sources


def empty_page():
    return ResidentContactPageSpec(ph.CollisionModel(()), (), {}, 0)


def receipt_record(receipt, *, geometry_ref=None):
    result = {f.name: getattr(receipt, f.name) for f in fields(receipt)}
    result["diagnostics"] = dict(receipt.diagnostics)
    if geometry_ref is not None:
        geometry = result["diagnostics"]["device_patch"]
        query = geometry.query if hasattr(geometry, "query") else geometry
        result["diagnostics"]["device_patch"] = dict(
            shared_payload_ref=geometry_ref, scope="this-step-record",
            state_epoch=query.state_epoch, active=query.active,
        )
    return result


def capture(batch, plan, entry):
    aggregate, retained, receipts = plan.last_device_patch, [], {}
    for world, receipt in plan.last_accepted.items():
        geometry = receipt.diagnostics.get("device_patch")
        reference = None
        if geometry is not None:
            if geometry is aggregate:
                reference = "/geometry"
            else:
                index = next((i for i, value in enumerate(retained) if value is geometry), None)
                if index is None:
                    index = len(retained)
                    retained.append(geometry)
                reference = f"/retained_geometry/{index}"
        receipts[world] = receipt_record(receipt, geometry_ref=reference)
    return dict(
        entry=entry, states=batch.snapshot(), receipts=receipts,
        geometry=aggregate, retained_geometry=tuple(retained),
        broad=plan.last_result, status=plan.last_status,
        stats=plan.stats(), batch_stats=batch.stats(),
    )


def close_values(actual, expected):
    """Compare complete physical records, allowing only original FP32 gates."""
    if isinstance(actual, np.ndarray):
        assert isinstance(expected, np.ndarray) and actual.shape == expected.shape
        if actual.dtype.kind in "fc":
            np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=5e-5)
        else:
            np.testing.assert_array_equal(actual, expected)
    elif is_dataclass(actual):
        assert type(actual) is type(expected)
        for f in fields(actual):
            close_values(getattr(actual, f.name), getattr(expected, f.name))
    elif isinstance(actual, Mapping):
        assert set(actual) == set(expected)
        for key in actual:
            close_values(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            close_values(a, b)
    elif isinstance(actual, (float, np.floating)):
        np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=5e-5)
    else:
        assert actual == expected


def physical_record(receipt):
    result = receipt_record(receipt)
    result["diagnostics"].pop("device_patch", None)
    return result


def assert_geometry(plan, entry):
    aggregate = plan.last_device_patch
    assert aggregate is not None
    assert aggregate.owner is plan.last_result.owner
    assert aggregate.registration is plan.last_result.registration
    assert aggregate.topology_generation == plan.last_result.topology_generation
    assert aggregate.state_epoch == plan.last_result.state_epoch
    assert len(aggregate.page_consumptions) == len(plan.page_specs)
    assert aggregate.page_queries is plan.last_device_queries
    wanted = {}
    for spec, broad, conversion, query in zip(
        plan.page_specs,
        plan.last_page_results,
        aggregate.page_consumptions,
        aggregate.page_queries,
        strict=True,
    ):
        if not spec.model.shapes:
            assert conversion is None and query is None
            continue
        assert conversion.query is query
        assert query.registration is broad.registration and query.owner is broad.owner
        assert query.state_epoch == aggregate.state_epoch and query.active == aggregate.active
        assert query.world_keys == broad.world_keys
        assert query.shape_keys == broad.shape_keys
        assert query.shape_revisions == broad.shape_revisions
        assert tuple(row.pair for row in query.rows) == tuple(sorted(broad.candidates))
        assert tuple(query.requested_pairs) == tuple(sorted(broad.candidates))
        for i, world in enumerate(query.world_keys):
            assert query.ticks[i] == entry[world].tick
        assert not (set(wanted) & set(conversion.samples))
        wanted.update(conversion.samples)
        for delta in conversion.patch_differences.values():
            assert np.linalg.norm(delta["first_moment_delta_m"]) <= delta["first_moment_limit_m"]
    assert set(aggregate.samples) == set(wanted)
    assert all(aggregate.samples[k] is value for k, value in wanted.items())
    for world, receipt in plan.last_accepted.items():
        if plan.last_status[world] == "ACCEPTED":
            assert receipt.diagnostics["device_patch"] is aggregate
    return aggregate


def slider_truth(batch, receipt, sources, expected_q, expected_v, expected_force, diagnostic=None):
    state = receipt.accepted_state
    np.testing.assert_allclose(state.position, [expected_q], atol=5e-6, rtol=5e-5)
    np.testing.assert_allclose(state.velocity, [expected_v], atol=5e-6, rtol=5e-5)
    np.testing.assert_allclose(
        receipt.contact_generalized, [expected_force], atol=5e-6, rtol=5e-5
    )
    np.testing.assert_allclose(
        receipt.law_wrench_fp64, [[0, 0, expected_force, 0, 0, 0]], atol=5e-6, rtol=5e-5
    )
    assert len(receipt.sites) == 4
    np.testing.assert_allclose([s.geometry.weight for s in receipt.sites], .25, atol=1e-12, rtol=0)
    for site in receipt.sites:
        np.testing.assert_allclose(
            np.asarray(site.first_force_n) + site.second_force_n, 0, atol=1e-12, rtol=0
        )
    reaction = next(iter(receipt.static_reactions.values()))
    np.testing.assert_allclose(reaction.force_n, [0, 0, -expected_force], atol=5e-6, rtol=5e-5)
    np.testing.assert_allclose(reaction.torque_nm, 0, atol=5e-6, rtol=5e-5)
    return _audit_receipt(
        receipt, batch.tree_diagnostics()[receipt.world] if diagnostic is None else diagnostic, sources
    )


@pytest.mark.tyx
@pytest.mark.parametrize("count", (129, 256))
def test_all_worlds_loaded_four_site_pages(count, tyx_mode, tmp_path, monkeypatch):
    worlds = tuple(1001 + 3 * i for i in range(count))
    with _batch(tyx_mode, tmp_path / "cache", worlds) as batch:
        specs, sources = pages(batch, (worlds[:128], worlds[128:]))
        with batch.prepare_contact_bundle(
            specs, narrow_phase="device_patch", device_patch_site_capacity=512
        ) as plan:
            monkeypatch.setattr(bridge, "contact_patch_sample", b2.no_host_search)
            calls = _spies(monkeypatch, batch, plan)
            q, v, records = 0.0, 1 / 64, []
            for tick in range(3):
                entry = batch.snapshot()
                force = 64 * max(0.0, 1 / 32 - q)
                v += H * (force / 2 + 3 / 16)
                q += H * v
                assert batch.step_contacts(plan, H) == dict.fromkeys(worlds, True)
                aggregate = assert_geometry(plan, entry)
                assert len(aggregate.samples) == count
                diagnostics = batch.tree_diagnostics()
                for world in worlds:
                    assert plan.last_accepted[world].exit_tick == tick + 1
                    slider_truth(batch, plan.last_accepted[world], sources[world], q, v, force,
                                 diagnostics[world])
                assert calls[(0, "mechanics")] == calls[(0, "solve")] == tick + 1
                assert calls["packets"] == 2 * (tick + 1)
                records.append(capture(batch, plan, entry))
            assert batch.stats()["completed_steps"] == 3
            b2.save_rows(
                tmp_path, f"B3-loaded-{count}", dict(worlds=worlds, h=H, records=records, calls=calls)
            )


def slider_run(mode, path, groups, *, route="bundle", order=None):
    worlds = tuple(sorted(w for group in groups for w in group))
    with _batch(mode, path / "cache", worlds if order is None else order) as batch:
        specs, sources = pages(batch, groups)
        if route == "single":
            spec = specs[0]
            plan = batch.prepare_contacts(
                spec.model, spec.bindings, spec.materials_by_pair, narrow_phase="device_patch"
            )
        else:
            kwargs = {} if route == "host" else dict(narrow_phase="device_patch")
            plan = batch.prepare_contact_bundle(specs, **kwargs)
        with plan:
            records, physical, samples = [], [], []
            q, v = 0.0, 1 / 64
            for _ in range(3):
                entry = batch.snapshot()
                force = 64 * max(0.0, 1 / 32 - q)
                v += H * (force / 2 + 3 / 16)
                q += H * v
                assert batch.step_contacts(plan, H) == dict.fromkeys(worlds, True)
                for world in worlds:
                    slider_truth(batch, plan.last_accepted[world], sources[world], q, v, force)
                physical.append({w: physical_record(r) for w, r in plan.last_accepted.items()})
                if route == "single":
                    conversion = plan.last_accepted[worlds[0]].diagnostics["device_patch"]
                    samples.append(conversion.samples)
                    records.append(dict(entry=entry, states=batch.snapshot(), geometry=conversion,
                                        receipts={w: receipt_record(r, geometry_ref="/geometry")
                                                  for w, r in plan.last_accepted.items()}))
                else:
                    if route != "host":
                        samples.append(assert_geometry(plan, entry).samples)
                    records.append(capture(batch, plan, entry))
            b2.save_rows(path, "complete", dict(h=H, route=route, groups=groups, records=records))
            return physical, samples


@pytest.mark.tyx
def test_single_page_matches_b2_and_default_host(tyx_mode, tmp_path):
    routes = {}
    for route in ("single", "bundle", "host"):
        path = tmp_path / route
        path.mkdir()
        routes[route] = slider_run(tyx_mode, path, ((7, 31),), route=route)
    close_values(routes["single"][0], routes["bundle"][0])
    close_values(routes["single"][1], routes["bundle"][1])
    close_values(routes["host"][0], routes["bundle"][0])


@pytest.mark.tyx
def test_world_page_and_partition_permutation(tyx_mode, tmp_path, monkeypatch):
    monkeypatch.setattr(bridge, "contact_patch_sample", b2.no_host_search)
    runs = []
    layouts = (((7, 31), (103, 211)), ((211, 103), (31, 7)), ((7, 211), (31, 103)))
    for index, layout in enumerate(layouts):
        path = tmp_path / str(index)
        path.mkdir()
        order = (211, 103, 31, 7) if index == 1 else (7, 31, 103, 211)
        runs.append(slider_run(tyx_mode, path, layout, order=order))
    for observed in runs[1:]:
        close_values(observed[0], runs[0][0])
        close_values(observed[1], runs[0][1])


@pytest.mark.tyx
def test_all_four_layouts_are_mixed_across_two_pages(tyx_mode, tmp_path, monkeypatch):
    layouts = ("ordinary", "affine", "floating", "floating_affine")
    entries, worlds, commands = {}, [], {}
    for i, layout in enumerate(layouts):
        item, tree, command = b2.fixture(layout, tmp_path / layout)
        for page in range(2):
            world = 101 + i * 10 + page * 100
            worlds.append((world, tree))
            entries[world] = (layout, item)
            commands[world] = command
    with ph.TyxWorldBatch(tuple(reversed(worlds)), tyx_mode, cache_root=tmp_path / "cache") as batch:
        batch.set_controls(commands)
        specs, originals = [], {}
        for page in range(2):
            shapes, bindings, laws = [], [], {}
            for i, layout in enumerate(layouts):
                world = 101 + i * 10 + page * 100
                item = entries[world][1]
                model, bound, source, pair_laws = b2.geometry(layout, batch, item, world)
                shapes.extend(model.shapes)
                bindings.extend(bound)
                laws.update(pair_laws)
                originals[world] = source
            specs.append(ResidentContactPageSpec(ph.CollisionModel(tuple(shapes)), tuple(bindings), laws))
        with batch.prepare_contact_bundle(
            tuple(specs), tree_dynamic_pairs=True, narrow_phase="device_patch"
        ) as plan:
            monkeypatch.setattr(bridge, "contact_patch_sample", b2.no_host_search)
            calls = _spies(monkeypatch, batch, plan)
            records = []
            for tick in range(3):
                before = batch.snapshot()
                assert all(any(abs(v) > 0 for v in s.velocity) for s in before.values())
                assert batch.step_contacts(plan, H) == dict.fromkeys(entries, True)
                aggregate = assert_geometry(plan, before)
                assert all(len(c.query.world_keys) == 4 for c in aggregate.page_consumptions)
                audits = {
                    w: b2.audit(layout, batch, item, before[w], plan.last_accepted[w], originals[w], w)
                    for w, (layout, item) in entries.items()
                }
                for index in range(4):
                    assert calls[(index, "mechanics")] == calls[(index, "solve")] == tick + 1
                assert calls["packets"] == 8 * (tick + 1)
                record = capture(batch, plan, before)
                record["independent_audits"] = audits
                records.append(record)
            b2.save_rows(
                tmp_path, "B3-four-layout-two-pages",
                dict(commands=commands, inputs={w: item["data"] for w, (_, item) in entries.items()},
                     records=records, calls=calls),
            )


@pytest.mark.tyx
@pytest.mark.parametrize("fault", (
    "patch_readback", "unknown_patch", "source_epoch", "source_owner",
    "source_registration", "source_bank", "source_layout", "source_membership",
    "moment_conversion", "post_solve", "precommit",
))
def test_last_page_fault_never_publishes_partial_result(fault, tyx_mode, tmp_path, monkeypatch):
    with _batch(tyx_mode, tmp_path / "cache", (7, 31)) as batch:
        specs, _ = pages(batch, ((7,), (31,)))
        with batch.prepare_contact_bundle(specs, narrow_phase="device_patch") as plan:
            assert batch.step_contacts(plan, H) == {7: True, 31: True}
            before = batch.snapshot()
            old = (plan.last_accepted, plan.last_status, plan.last_result,
                   plan.last_device_patch, plan.last_device_queries)
            old_layouts = tuple(p._trees.layouts for p in plan._pages)
            old_broads = tuple(p.last_result for p in plan._pages)
            query = plan._device_patch_queries[-1]
            success_counts = tuple(q.stats["queries"] for q in plan._device_patch_queries)
            attempts = tuple(q.stats["attempts"] for q in plan._device_patch_queries)
            with monkeypatch.context() as patch:
                if fault == "patch_readback":
                    target, name = query, "_read_patch"

                    def broken(*args, **kwargs):
                        raise RuntimeError("last-page patch readback")
                elif fault == "moment_conversion":
                    target, name = bundle_module, "patches_from_device"
                    original = getattr(target, name)

                    def broken(result, *args, **kwargs):
                        if result.world_keys == (31,):
                            raise ValueError("last-page first moment conversion gate")
                        return original(result, *args, **kwargs)
                elif fault == "post_solve":
                    target, name = plan, "_finish_attempt"
                    original = getattr(target, name)

                    def broken(*args, **kwargs):
                        original(*args, **kwargs)
                        raise RuntimeError("last-page receipt before publication")
                elif fault == "precommit":
                    target, name = plan, "_check_attempt"
                    original = getattr(target, name)
                    checks = [0]

                    def broken(*args, **kwargs):
                        original(*args, **kwargs)
                        checks[0] += 1
                        if checks[0] == 2:
                            raise RuntimeError("last precommit source check")
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
                        return replace(
                            result, rows=result.rows[:-1] +
                            (replace(row, patch=replace(row.patch, status="UNRESOLVED")),)
                        )
                patch.setattr(target, name, broken)
                with pytest.raises((RuntimeError, ValueError)):
                    batch.step_contacts(plan, H)
            assert batch.snapshot() == before
            current = (plan.last_accepted, plan.last_status, plan.last_result,
                       plan.last_device_patch, plan.last_device_queries)
            assert all(a is b for a, b in zip(old, current, strict=True))
            assert all(p.last_result is r for p, r in zip(plan._pages, old_broads, strict=True))
            assert all(p._trees.layouts is v for p, v in zip(plan._pages, old_layouts, strict=True))
            assert tuple(q.stats["queries"] for q in plan._device_patch_queries) == success_counts
            assert all(q.stats["attempts"] > a for q, a in
                       zip(plan._device_patch_queries, attempts, strict=True))
            assert batch.step_contacts(plan, H) == {7: True, 31: True}
            assert all(s.tick == 2 for s in batch.snapshot().values())
            b2.save_rows(tmp_path, "B3-last-page-" + fault, capture(batch, plan, before))


@pytest.mark.tyx
def test_last_page_resource_cap_is_per_page_and_rolls_back(tyx_mode, tmp_path):
    with _batch(tyx_mode, tmp_path / "cache", (7, 31, 103)) as batch:
        specs, _ = pages(batch, ((7,), (31, 103)))
        before = batch.snapshot()
        with batch.prepare_contact_bundle(
            specs, narrow_phase="device_patch", device_patch_site_capacity=7
        ) as bad:
            with pytest.raises((RuntimeError, ValueError), match="patch|resource|RESOURCE"):
                batch.step_contacts(bad, H)
            assert batch.snapshot() == before
            assert bad.last_result is None and bad.last_device_patch is None
            assert bad.last_device_queries == (None, None)
            assert all(p.last_result is None for p in bad._pages)
            assert all(q.last_result is None and q.stats["queries"] == 0
                       for q in bad._device_patch_queries)
            failed_stats = bad.stats()
        with batch.prepare_contact_bundle(
            specs, narrow_phase="device_patch", device_patch_site_capacity=8
        ) as plan:
            # Twelve global sites exceed eight, but each page fits its own cap.
            assert batch.step_contacts(plan, H) == {7: True, 31: True, 103: True}
            assert_geometry(plan, before)
            assert sum(len(r.sites) for r in plan.last_accepted.values()) == 12
            b2.save_rows(tmp_path, "B3-page-cap", dict(failed=failed_stats, accepted=capture(batch, plan, before)))


@pytest.mark.tyx
def test_partial_reset_and_world_site_rejection_preserve_independent_worlds(tyx_mode, tmp_path):
    with _batch(tyx_mode, tmp_path / "cache", (7, 31)) as batch:
        specs, _ = pages(batch, ((7,), (31,)))
        with batch.prepare_contact_bundle(specs, narrow_phase="device_patch") as plan:
            assert batch.step_contacts(plan, H) == {7: True, 31: True}
            old_queries = plan.last_device_queries
            before = batch.snapshot()
            batch.reset([31])
            assert batch.snapshot()[7] == before[7]
            for query, result in zip(plan._device_patch_queries, old_queries, strict=True):
                with pytest.raises(ValueError, match="stale"):
                    query.require_current(result)
            reset = batch.snapshot()
            assert reset[31].tick == 0
            assert batch.step_contacts(plan, H) == {7: True, 31: True}
            assert_geometry(plan, reset)
            assert batch.snapshot()[7].tick == 2 and batch.snapshot()[31].tick == 1
            record = capture(batch, plan, reset)
            # Change only this plan's test-injected resource ceiling, not state.
            previous = plan.last_accepted[31]
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(plan, "_site_caps", {7: 4, 31: 3})
                entry = batch.snapshot()
                assert batch.step_contacts(plan, H) == {7: True, 31: False}
                assert batch.snapshot()[31] == entry[31]
                assert plan.last_accepted[31] is previous
                assert plan.last_status[31] == "SITE_CAPACITY_EXCEEDED"
                assert batch.snapshot()[7].tick == entry[7].tick + 1
                assert_geometry(plan, entry)
                rejected = capture(batch, plan, entry)
            b2.save_rows(tmp_path, "B3-reset-local-cap", dict(reset=record, rejected=rejected))


@pytest.mark.tyx
@pytest.mark.parametrize("empty_count", (0, 1, 2))
def test_all_empty_device_pages_take_one_free_step(empty_count, tyx_mode, tmp_path):
    with _batch(tyx_mode, tmp_path / "cache", (7,)) as batch:
        with batch.prepare_contact_bundle(
            (empty_page(),) * empty_count, narrow_phase="device_patch"
        ) as plan:
            assert batch.step_contacts(plan, H) == {7: True}
            assert batch.snapshot()[7].tick == 1
            assert plan.last_device_patch is None
            assert plan.last_device_queries == (None,) * empty_count
            assert plan._device_patch_queries == (None,) * empty_count
            assert not plan.last_accepted and len(plan.last_page_results) == empty_count
            b2.save_rows(tmp_path, "B3-all-empty", capture(batch, plan, {}))
        assert batch.step(H) == {7: True} and batch.snapshot()[7].tick == 2


@pytest.mark.tyx
def test_empty_interleaved_and_nonempty_zero_pair_pages(tyx_mode, tmp_path, monkeypatch):
    with _batch(tyx_mode, tmp_path / "cache", (7, 31)) as batch:
        specs, _ = pages(batch, ((7,), (31,)))
        # One original dynamic shape, hence no possible collision pair.
        zero = ResidentContactPageSpec(
            ph.CollisionModel((specs[1].model.shapes[0],)),
            (specs[1].bindings[0],), {}, 0,
        )
        with batch.prepare_contact_bundle(
            (empty_page(), specs[0], zero, empty_page()), narrow_phase="device_patch"
        ) as plan:
            monkeypatch.setattr(bridge, "contact_patch_sample", b2.no_host_search)
            entry = batch.snapshot()
            assert batch.step_contacts(plan, H) == {7: True, 31: True}
            aggregate = assert_geometry(plan, entry)
            assert aggregate.page_consumptions[0] is aggregate.page_consumptions[3] is None
            assert aggregate.page_consumptions[2] is not None
            assert not aggregate.page_consumptions[2].samples
            assert not aggregate.page_queries[2].rows
            assert len(plan.last_accepted[7].sites) == 4
            assert len(plan.last_accepted[31].sites) == 0
            with pytest.raises(FrozenInstanceError):
                aggregate.active = 9
            with pytest.raises(TypeError):
                aggregate.samples[object()] = object()
            b2.save_rows(tmp_path, "B3-empty-interleaved", capture(batch, plan, entry))


@pytest.mark.tyx
@pytest.mark.parametrize("option", ("bad_mode", "host_capacity", "zero", "oversize", "bool"))
def test_invalid_device_options_precede_any_allocation(option, tyx_mode, tmp_path, monkeypatch):
    with _batch(tyx_mode, tmp_path / "cache", (7,)) as batch:
        specs, _ = pages(batch, ((7,),))
        options = {
            "bad_mode": dict(narrow_phase="automatic"),
            "host_capacity": dict(device_patch_site_capacity=4),
            "zero": dict(narrow_phase="device_patch", device_patch_site_capacity=0),
            "oversize": dict(narrow_phase="device_patch", device_patch_site_capacity=4097),
            "bool": dict(narrow_phase="device_patch", device_patch_site_capacity=True),
        }[option]

        def forbidden(*args, **kwargs):
            raise AssertionError("allocation preceded option validation")

        before = batch.snapshot()
        existing = set(batch._collision_plans)
        monkeypatch.setattr(bundle_module._ContactCollisionPage, "_allocate_device", forbidden)
        with pytest.raises((TypeError, ValueError)):
            batch.prepare_contact_bundle(specs, **options)
        assert batch.snapshot() == before and batch._collision_plans == existing


@pytest.mark.tyx
def test_last_query_constructor_failure_closes_every_prepared_owner(tyx_mode, tmp_path, monkeypatch):
    with _batch(tyx_mode, tmp_path / "cache", (7, 31)) as batch:
        specs, _ = pages(batch, ((7,), (31,)))
        original = bundle_module.ConvexQueryPlan
        built, page_owners = [], []
        before = batch.snapshot()

        def fail_last(owner, page, **kwargs):
            page_owners.append(page)
            if len(page_owners) == 2:
                raise RuntimeError("last-page query allocation")
            query = original(owner, page, **kwargs)
            built.append(query)
            return query

        with monkeypatch.context() as patch:
            patch.setattr(bundle_module, "ConvexQueryPlan", fail_last)
            with pytest.raises(RuntimeError, match="last-page"):
                batch.prepare_contact_bundle(specs, narrow_phase="device_patch")
        assert len(built) == 1 and built[0].closed
        assert len(page_owners) == 2 and all(p._closed for p in page_owners)
        assert not batch._collision_plans and not batch._contact_bundles
        assert batch.snapshot() == before
        assert batch.step(H) == {7: True, 31: True}
        b2.save_rows(tmp_path, "B3-constructor-cleanup", dict(before=before, after=batch.snapshot()))


@pytest.mark.tyx
def test_close_attempts_all_dependents_and_preserves_batch(tyx_mode, tmp_path, monkeypatch):
    with _batch(tyx_mode, tmp_path / "cache", (7, 31)) as batch:
        specs, _ = pages(batch, ((7,), (31,)))
        plan = batch.prepare_contact_bundle(specs, narrow_phase="device_patch")
        queries = plan._device_patch_queries
        closed = []
        for index, query in enumerate(queries):
            original = query.close

            def observed(_original=original, _index=index):
                closed.append(_index)
                _original()
                if _index == 1:
                    raise RuntimeError("reported last query close error")

            monkeypatch.setattr(query, "close", observed)
        with pytest.raises(RuntimeError, match="last query"):
            plan.close()
        assert sorted(closed) == [0, 1]
        assert all(q.closed for q in queries)
        assert all(p._closed for p in plan._pages)
        assert not batch._collision_plans and not batch._contact_bundles
        plan.close()
        assert batch.step(H) == {7: True, 31: True}
        b2.save_rows(tmp_path, "B3-close-all", dict(closed=closed, state=batch.snapshot()))

"""World-complete loaded pages: new code/test authoring, parent-only execution."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
import test_resident_affine_contacts as affine
import test_resident_floating_affine_contacts as floating_affine
import test_resident_floating_contacts as floating
from test_resident_tree_contacts import (
    H, _action, _audit_receipt, _json, _law, _r01_bound, _save, _slider,
)

import phymeo as ph
from phymeo.backends import _resident_contact_bundle as bundle_module
from phymeo.backends import _resident_contact_plan as contact_module
from phymeo.backends._resident_contact_bundle import ResidentContactPageSpec
from phymeo.training.resident_stepper import ResidentContactStepper


def _spec(batch, worlds, *, candidate_capacity=None):
    shapes, bindings, laws = [], [], {}
    for world in worlds:
        model, bound, _, pair = _r01_bound(batch, world)
        shapes.extend(model.shapes)
        bindings.extend(bound)
        laws[pair] = _law()
    return ResidentContactPageSpec(
        ph.CollisionModel(tuple(shapes)), tuple(bindings), laws,
        len(worlds) if candidate_capacity is None else candidate_capacity,
    )


def _page_record(result):
    # Opaque process owner/registration identities are not JSON values. Preserve
    # every published geometric value and stamp, with explicit page membership.
    return dict(
        topology_generation=result.topology_generation, state_epoch=result.state_epoch,
        world_keys=result.world_keys, candidate_capacity=result.candidate_capacity,
        candidates=result.candidates,
        pages=tuple(dict(
            topology_generation=r.topology_generation, state_epoch=r.state_epoch,
            world_keys=r.world_keys, ticks=r.ticks, shape_keys=r.shape_keys,
            shape_revisions=r.shape_revisions, bounds=r.bounds,
            registered_pairs=r.registered_pairs, candidates=r.candidates,
        ) for r in result.page_results),
    )


def _batch(mode, path, worlds):
    program, tree = _slider(v=1 / 64)
    batch = ph.TyxWorldBatch([(w, tree) for w in worlds], mode, cache_root=path)
    batch.set_controls({w: _action(program, ff=[1 / 8], effort=[1 / 4]) for w in worlds})
    return batch


def _spies(monkeypatch, batch, plan):
    counts = {}
    for index, (_, storage) in enumerate(batch._tree_storages()):
        physical = getattr(storage, "physical", storage)
        for obj, method, label in (
            (physical, "stage_mechanics", "mechanics"),
            (storage, "stage_solve_from_mechanics" if hasattr(storage, "physical") else "stage_solve", "solve"),
        ):
            original = getattr(obj, method)
            key = (index, label)
            counts[key] = 0

            def wrapped(*args, _original=original, _key=key, **kwargs):
                counts[_key] += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(obj, method, wrapped)
    original = plan._read_storage_packet
    counts["packets"] = 0

    def packet(*args, **kwargs):
        counts["packets"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(plan, "_read_storage_packet", packet)
    return counts


@pytest.mark.tyx
@pytest.mark.parametrize("count", (127, 128, 129, 256))
def test_loaded_world_boundaries_one_solve_per_storage(count, tyx_mode, tmp_path, monkeypatch):
    worlds = tuple(1001 + 3 * i for i in range(count))
    record = dict(worlds=worlds, mode=tyx_mode, frames=[])
    with _batch(tyx_mode, tmp_path / "cache", worlds) as batch:
        specs = tuple(_spec(batch, worlds[i:i + 128]) for i in range(0, count, 128))
        assert all(len(s.model.shapes) <= 256 for s in specs)
        with batch.prepare_contact_bundle(specs) as plan:
            counts = _spies(monkeypatch, batch, plan)
            stepper = ResidentContactStepper(batch, plan)
            q, v = 0.0, 1 / 64
            for tick in range(3):
                # 2 kg slider: 64 * depth N plus the original 1/8 + 1/4 Nm input.
                force = 64 * max(0.0, 1 / 32 - q)
                v += H * (force / 2 + 3 / 16)
                q += H * v
                assert stepper(H) == dict.fromkeys(worlds, True)
                states = batch.snapshot()
                assert set(plan.last_accepted) == set(worlds)
                assert set(plan.last_status.values()) == {"ACCEPTED"}
                for world, state in states.items():
                    assert state.tick == tick + 1
                    np.testing.assert_allclose(state.position, [q], atol=5e-6, rtol=5e-5)
                    np.testing.assert_allclose(state.velocity, [v], atol=5e-6, rtol=5e-5)
                    receipt = plan.last_accepted[world]
                    assert receipt.world == world and receipt.exit_tick == tick + 1
                    assert receipt.law_wrench_fp64[0, 2] > 0
                    assert all(site.pair[0].world == world and site.pair[1].world == world for site in receipt.sites)
                result = plan.last_query
                assert len(result.page_results) == len(specs)
                assert len(result.candidates) == count
                assert all(a.world == b.world for a, b in result.candidates)
                assert plan.stats()["candidate_capacity"] == count
                assert counts[(0, "mechanics")] == counts[(0, "solve")] == tick + 1
                assert counts["packets"] == 2 * (tick + 1)
                record["frames"].append(dict(states=states, receipts=plan.last_accepted, pages=_page_record(result), stats=plan.stats()))
            assert batch.stats()["completed_steps"] == 3
            record["calls"] = counts
            _save(tmp_path, "loaded-world-pages", record)


def _equivalent_run(mode, path, layout, *, legacy=False, reversed_worlds=False):
    worlds = tuple(sorted(w for group in layout for w in group))
    with _batch(mode, path, tuple(reversed(worlds)) if reversed_worlds else worlds) as batch:
        specs = tuple(_spec(batch, group) for group in layout)
        if legacy:
            spec = specs[0]
            plan = batch.prepare_contacts(spec.model, spec.bindings, spec.materials_by_pair)
        else:
            plan = batch.prepare_contact_bundle(specs)
        with plan:
            result = []
            for _ in range(3):
                assert batch.step_contacts(plan, H) == dict.fromkeys(worlds, True)
                result.append(_json(dict(states=batch.snapshot(), receipts=plan.last_accepted)))
            return result


@pytest.mark.tyx
def test_single_page_equivalence_and_page_world_permutation(tyx_mode, tmp_path):
    legacy = _equivalent_run(tyx_mode, tmp_path / "legacy", ((7, 19, 41, 103),), legacy=True)
    single = _equivalent_run(tyx_mode, tmp_path / "single", ((7, 19, 41, 103),))
    split = _equivalent_run(tyx_mode, tmp_path / "split", ((103, 19), (41, 7)), reversed_worlds=True)
    assert legacy == single == split
    eight = _equivalent_run(tyx_mode, tmp_path / "eight", tuple((w,) for w in range(8)))
    eight_legacy = _equivalent_run(tyx_mode, tmp_path / "eight-legacy", (tuple(range(8)),), legacy=True)
    assert eight == eight_legacy
    _save(tmp_path, "single-versus-world-pages", dict(
        legacy=legacy, single=single, split=split, eight=eight, eight_legacy=eight_legacy
    ))


@pytest.mark.tyx
def test_four_coordinate_storages_share_pages_and_keep_full_oracles(tyx_mode, tmp_path, monkeypatch):
    program, ordinary = _slider(v=1 / 64)
    data = affine._case("A04")
    asset = affine._asset(tmp_path / "affine", data)
    reduced = affine._world(asset, data)
    fd = floating._reference().case("F02", variant="motor")
    fd["v"][0] = 1 / 128
    fi = floating._item(fd)
    fad = floating_affine._reference().case("A02")
    fad["v"][0] = 1 / 256
    fad["force"][0, 1] = 1 / 32
    fad["torque"][0, 2] = 1 / 64
    fai = floating_affine._item(tmp_path / "floating-affine", fad)
    worlds = {11: ordinary, 23: reduced, 37: fi["tree"], 53: fai["tree"]}
    commands = {
        11: _action(program, ff=[1 / 8], effort=[1 / 4]),
        23: affine._control(reduced, asset, data), 37: fi["command"], 53: fai["command"],
    }
    with ph.TyxWorldBatch(tuple(worlds.items()), tyx_mode, cache_root=tmp_path / "cache") as batch:
        batch.set_controls(commands)
        m, b, sources, pair = _r01_bound(batch, 11)
        ordinary_spec = ResidentContactPageSpec(m, b, {pair: _law()})
        m, b, asources, pair = affine._contact_geometry(batch, asset, data, world=23)
        affine_spec = ResidentContactPageSpec(m, b, {pair: _law()})
        m, b, fsources, laws = floating._geometry(batch, fi, world=37)
        floating_spec = ResidentContactPageSpec(m, b, laws)
        m, b, fasources, laws = floating_affine._geometry(batch, fai, world=53)
        floating_affine_spec = ResidentContactPageSpec(m, b, laws)
        specs = (floating_affine_spec, ordinary_spec, floating_spec, affine_spec)
        with batch.prepare_contact_bundle(specs, tree_dynamic_pairs=True) as plan:
            counts = _spies(monkeypatch, batch, plan)
            records = []
            for tick in range(3):
                before = batch.snapshot()
                assert batch.step_contacts(plan, H) == dict.fromkeys(worlds, True)
                receipts = plan.last_accepted
                audits = {
                    11: _audit_receipt(receipts[11], batch.tree_diagnostics()[11], sources),
                    23: affine._audit_step(batch, 23, asset, data, before[23], receipts[23], asources),
                    37: floating._audit(batch, fi, receipts[37], fsources, world=37),
                    53: floating_affine._audit_contact(batch, fai, receipts[53], fasources, world=53),
                }
                assert all(receipt.sites for receipt in receipts.values())
                for index in range(4):
                    assert counts[(index, "mechanics")] == counts[(index, "solve")] == tick + 1
                assert counts["packets"] == 8 * (tick + 1)
                records.append(dict(entry=before, receipts=receipts, independent_audits=audits, states=batch.snapshot()))
            _save(tmp_path, "four-coordinate-page-mechanics", dict(inputs=commands, records=records, counts=counts))


@pytest.mark.tyx
def test_partial_reset_and_site_capacity_isolate_worlds(tyx_mode, tmp_path):
    worlds = (7, 19, 41)
    with _batch(tyx_mode, tmp_path / "cache", worlds) as batch:
        specs = (_spec(batch, (7,)), _spec(batch, (19, 41)))
        with batch.prepare_contact_bundle(specs, site_capacity={7: 128, 19: 128, 41: 0}) as plan:
            initial = batch.snapshot()
            assert batch.step_contacts(plan, H) == {7: True, 19: True, 41: False}
            assert batch.snapshot()[41] == initial[41]
            assert plan.last_status[41] == "SITE_CAPACITY_EXCEEDED"
            assert set(plan.last_accepted) == {7, 19}
            query = plan.query()
            plan.require_current(query)
            forged = (
                replace(query, page_results=query.page_results[:-1]),
                replace(query, candidate_capacity=query.candidate_capacity + 1),
                replace(query, state_epoch=query.state_epoch + 1),
                replace(
                    query, page_results=tuple(reversed(query.page_results)),
                    candidates=tuple(pair for r in reversed(query.page_results) for pair in r.candidates),
                ),
            )
            for result in forged:
                with pytest.raises(ValueError):
                    plan.require_current(result)
            changed = replace(query.page_results[-1], shape_revisions=(99, 99, 99, 99))
            with pytest.raises(ValueError, match="topology"):
                plan.require_current(replace(query, page_results=(*query.page_results[:-1], changed)))
            before = batch.snapshot()
            batch.reset([19])
            assert batch.snapshot()[7] == before[7] and batch.snapshot()[41] == before[41]
            with pytest.raises(ValueError, match="stale"):
                plan.require_current(query)
            assert batch.snapshot()[19].tick == 0
            assert batch.step_contacts(plan, H) == {7: True, 19: True, 41: False}
            assert batch.snapshot()[7].tick == 2 and batch.snapshot()[19].tick == 1
            _save(tmp_path, "partial-reset-site-failure", dict(states=batch.snapshot(), receipts=plan.last_accepted, status=plan.last_status))


@pytest.mark.tyx
def test_last_page_overflow_and_late_finish_leave_no_partial_publication(tyx_mode, tmp_path, monkeypatch):
    worlds = (7, 19)
    with _batch(tyx_mode, tmp_path / "cache", worlds) as batch:
        first, second = _spec(batch, (7,)), _spec(batch, (19,))
        before = batch.snapshot()
        with batch.prepare_contact_bundle((first, replace(second, candidate_capacity=0))) as bad:
            calls = _spies(monkeypatch, batch, bad)
            with pytest.raises(ValueError, match="candidate capacity exceeded"):
                batch.step_contacts(bad, H)
            assert batch.snapshot() == before
            assert bad.last_result is None and bad.last_page_results == ()
            assert bad.last_accepted == {}
            assert all(page._last is None for page in bad._pages)
            assert calls[(0, "mechanics")] == calls[(0, "solve")] == 0
        # Restore all temporary storage spies before this independent attempt.
        monkeypatch.undo()
        with batch.prepare_contact_bundle((first, second)) as plan:
            assert batch.step_contacts(plan, H) == dict.fromkeys(worlds, True)
            before, last, accepted = batch.snapshot(), plan.last_result, plan.last_accepted
            layouts = tuple(page._trees.layouts for page in plan._pages)
            original = contact_module.finish_contact_loads
            calls = []

            def fail_final(loads, frames):
                calls.append(tuple(frames))
                if any(key.world == 19 for key in frames):
                    raise RuntimeError("injected final-page finish failure")
                return original(loads, frames)

            with monkeypatch.context() as patch:
                patch.setattr(contact_module, "finish_contact_loads", fail_final)
                with pytest.raises(RuntimeError, match="final-page"):
                    batch.step_contacts(plan, H)
            assert len(calls) == 2
            assert batch.snapshot() == before
            assert plan.last_result is last and plan.last_accepted is accepted
            assert all(page._trees.layouts is old for page, old in zip(plan._pages, layouts, strict=True))
            assert batch.step_contacts(plan, H) == dict.fromkeys(worlds, True)
            assert all(state.tick == 2 for state in batch.snapshot().values())
            _save(tmp_path, "last-page-rollback-retry", dict(states=batch.snapshot(), receipts=plan.last_accepted, stats=plan.stats()))


@pytest.mark.tyx
@pytest.mark.parametrize("fault", ("duplicate_world", "duplicate_shape", "later_material", "later_owner", "too_many", "candidate_cap"))
def test_all_registration_errors_precede_first_page_allocation(fault, tyx_mode, tmp_path, monkeypatch):
    with _batch(tyx_mode, tmp_path / "cache", (7, 19)) as batch:
        first, second = _spec(batch, (7,)), _spec(batch, (19,))
        original, calls, state = set(batch._collision_plans), [], batch.snapshot()

        def forbidden(*args):
            calls.append(True)
            raise AssertionError("device allocation preceded full page validation")

        monkeypatch.setattr(bundle_module._ContactCollisionPage, "_allocate_device", forbidden)
        with pytest.raises((ValueError, TypeError)):
            if fault == "candidate_cap":
                replace(second, candidate_capacity=32641)
            elif fault == "too_many":
                batch.prepare_contact_bundle((ResidentContactPageSpec(ph.CollisionModel(()), (), {}),) * 9)
            else:
                if fault == "duplicate_shape":
                    second = first
                elif fault == "duplicate_world":
                    key_map = {
                        s.key: ph.CollisionKey(s.key.world, s.key.body + 2000, s.key.shape + 2000)
                        for s in first.model.shapes
                    }
                    second = ResidentContactPageSpec(
                        ph.CollisionModel(tuple(replace(s, key=key_map[s.key]) for s in first.model.shapes)),
                        tuple(replace(b, key=key_map[b.key]) for b in first.bindings),
                        {tuple(key_map[k] for k in pair): law for pair, law in first.materials_by_pair.items()},
                    )
                elif fault == "later_material":
                    second = replace(second, materials_by_pair={})
                elif fault == "later_owner":
                    moving, fixed = second.bindings
                    second = replace(second, bindings=(replace(moving, topology=('foreign-topology',)), fixed))
                batch.prepare_contact_bundle((first, second))
        assert calls == []
        assert set(batch._collision_plans) == original and batch._contact_bundles == set()
        assert batch.snapshot() == state


@pytest.mark.tyx
def test_allocation_failure_closes_earlier_page_and_bundle_close_preserves_batch(tyx_mode, tmp_path, monkeypatch):
    with _batch(tyx_mode, tmp_path / "cache", (7, 19)) as batch:
        specs = (_spec(batch, (7,)), _spec(batch, (19,)))
        allocate = bundle_module._ContactCollisionPage._allocate_device
        pages, before = [], batch.snapshot()

        def fail_second(page, view):
            pages.append(page)
            if len(pages) == 2:
                raise RuntimeError("injected second-page allocation failure")
            return allocate(page, view)

        with monkeypatch.context() as patch:
            patch.setattr(bundle_module._ContactCollisionPage, "_allocate_device", fail_second)
            with pytest.raises(RuntimeError, match="second-page"):
                batch.prepare_contact_bundle(specs)
        assert len(pages) == 2 and all(page._closed for page in pages)
        assert all(page.stats()["device_owned_bytes"] == 0 for page in pages)
        assert batch._collision_plans == set() and batch._contact_bundles == set()
        assert batch.snapshot() == before
        plan = batch.prepare_contact_bundle(specs)
        with pytest.raises(FrozenInstanceError):
            plan.page_specs[0].candidate_capacity = 0
        stepper = ResidentContactStepper(batch, plan)
        plan.close()
        assert plan.stats()["device_owned_bytes"] == 0
        with pytest.raises(RuntimeError, match="closed"):
            stepper(H)
        assert batch.step(H) == {7: True, 19: True}


@pytest.mark.tyx
@pytest.mark.parametrize("empty_count", (0, 1))
def test_empty_pages_are_one_free_step_and_never_own_batch(empty_count, tyx_mode, tmp_path):
    with _batch(tyx_mode, tmp_path / "cache", (7,)) as batch:
        pages = (ResidentContactPageSpec(ph.CollisionModel(()), (), {}, 0),) * empty_count
        with batch.prepare_contact_bundle(pages) as plan:
            assert plan.stats()["page_count"] == empty_count
            assert plan.stats()["candidate_capacity"] == 0
            assert batch.step_contacts(plan, H) == {7: True}
            assert batch.snapshot()[7].tick == 1
            assert plan.last_accepted == {} and len(plan.last_page_results) == empty_count
        assert batch.step(H) == {7: True}
        assert batch.snapshot()[7].tick == 2

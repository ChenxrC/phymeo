"""M-B1 fixed A/B/E matrix; tests-first, no new solver or precision gates.

A: two masses and off-COM sphere contact, with a distinct second collider variant.
B: 16 uninterrupted steps; staggered sphere/capsule load and unload across worlds.
E: full publication, layout/identity, source, capacity and actual retry contracts.
All fixtures and independent mechanics live in the standalone installed consumer.
"""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "deformable_multibody_consumer",
    Path(__file__).parents[1] / "installed" / "check_deformable_multibody.py",
)
case = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(case)
pytestmark = pytest.mark.tyx


@pytest.fixture(autouse=True)
def public_api():
    case.initialize()


@pytest.mark.parametrize("extra", (False, True), ids=("two-bodies", "two-colliders-one-body"))
def test_two_spheres_full_body_wrench_and_motion(extra, tyx_mode, tmp_path):
    items = (case.fixture("A", world=7, extra=extra),)
    with case.saved(tmp_path / "two-spheres.jsonl.gz") as journal:
        case.run_window(items, tyx_mode, journal, steps=4)


def test_staggered_sphere_capsule_real_unload_and_zero_body(tyx_mode, tmp_path):
    items = (case.fixture("B", world=7), case.fixture("B", world=31))
    with case.saved(tmp_path / "staggered.jsonl.gz") as journal:
        case.run_window(items, tyx_mode, journal, steps=16)


def test_permuted_participants_shapes_worlds_keep_keyed_results(tyx_mode, tmp_path):
    items = (case.fixture("A", world=7, extra=True), case.fixture("A", world=31))
    with case.saved(tmp_path / "permutation.jsonl.gz") as journal:
        records = []
        for reverse in (False, True):
            with case.compile_scene(items, tyx_mode, reverse=reverse) as scene:
                references = case.reference_initial(items)
                states = []
                for step in range(3):
                    state, references, _ = case.advance(scene, items, references, journal, step)
                    states.append(state)
                records.append(states)
        for left, right in zip(*records, strict=True):
            case.state_equal(left, right, items)


def test_checkpoint_query_control_and_whole_world_reset_successors(tyx_mode, tmp_path):
    items = (case.fixture("A", world=7), case.fixture("A", world=31, extra=True))
    with case.saved(tmp_path / "lifecycle.jsonl.gz") as journal:
        case.lifecycle(items, tyx_mode, journal)


def retained(scene, items, before, checkpoint, revision, old, banks, journal):
    after = scene.snapshot()
    journal.emit("retained", state=after, checkpoint=scene.checkpoint(),
                 stats=scene.stats(), last_attempt=scene.last_attempt)
    case.state_equal(before, after, items)
    assert case.encode(scene.checkpoint()) == case.encode(checkpoint)
    assert scene.revision == revision and scene.last_step is old
    response = scene._response
    assert (response._nodes._active, response._scene._service._batch._active) == banks


@pytest.mark.parametrize("seam", ("query", "last-publication", "wrong-body-slot"))
def test_late_complete_attempt_keeps_all_bodies_and_retries(seam, tyx_mode, tmp_path, monkeypatch):
    items = (case.fixture("A", world=7), case.fixture("A", world=31))
    with case.saved(tmp_path / (seam + ".jsonl.gz")) as journal:
        with case.compile_scene(items, tyx_mode) as scene:
            references = case.reference_initial(items)
            _, references, _ = case.advance(scene, items, references, journal, "before")
            before, checkpoint = scene.snapshot(), scene.checkpoint()
            revision, old = scene.revision, scene.last_step
            response = scene._response
            banks = (response._nodes._active, response._scene._service._batch._active)
            completed = []
            attribute = {"query": "_query_entry", "last-publication": "_prepare_response_publication",
                         "wrong-body-slot": "_law_rows"}[seam]
            original = getattr(response, attribute)

            def reject(*args, **kwargs):
                value = original(*args, **kwargs)
                completed.append(value)
                journal.emit("actual-complete-fault-prefix", seam=seam, value=value)
                if seam == "wrong-body-slot":
                    sites, rows, identities, bary, levers = value
                    identities = identities.copy()
                    assert len(identities) and identities[-1, 0] == 1
                    identities[-1, 4] = 0  # Valid slot in WRONG world: never an out-of-range read.
                    return sites, rows, identities, bary, levers
                if seam == "last-publication":
                    assert set(value.rigid_publication["candidate"]) == set(scene.participant_keys)
                raise RuntimeError("M-B1 actual final candidate refused")

            with monkeypatch.context() as patch:
                patch.setattr(response, attribute, reject)
                journal.emit("fault-attempt", input=case.forces(items), h=case.H,
                             expected_revision=revision, before=before, checkpoint=checkpoint)
                with pytest.raises((ValueError, RuntimeError, case.ph.SolverFailure)) as caught:
                    scene.step(case.forces(items), case.H, expected_revision=revision)
                journal.emit("fault-return", error=caught.value, last_attempt=scene.last_attempt)
                assert len(completed) == 1
                retained(scene, items, before, checkpoint, revision, old, banks, journal)
            actual, _, _ = case.advance(scene, items, references, journal, "retry")
            with case.compile_scene(items, tyx_mode) as independent:
                refs = case.reference_initial(items)
                _, refs, _ = case.advance(independent, items, refs, journal, "independent-before")
                expected, _, _ = case.advance(independent, items, refs, journal, "independent-next")
            case.state_equal(actual, expected, items)


def test_active_site_exact_capacity_and_one_less(tyx_mode, tmp_path):
    items = (case.fixture("A", world=7), case.fixture("A", world=31))
    with case.saved(tmp_path / "site-capacity.jsonl.gz") as journal:
        with case.compile_scene(items, tyx_mode, site_capacity=4) as exact:
            _, _, receipt = case.advance(exact, items, case.reference_initial(items), journal, "exact")
            assert sum(len(r["sites"]) for r in receipt.worlds.values()) == 4
        with case.compile_scene(items, tyx_mode, site_capacity=3) as short:
            before, capsule, revision = short.snapshot(), short.checkpoint(), short.revision
            journal.emit("capacity-attempt", before=before, capacity=3, h=case.H)
            with pytest.raises(ValueError, match="capacity"):
                short.step(case.forces(items), case.H, expected_revision=revision)
            journal.emit("capacity-rejected", last_attempt=short.last_attempt, state=short.snapshot())
            case.state_equal(before, short.snapshot(), items)
            assert case.encode(short.checkpoint()) == case.encode(capsule)
            assert short.revision == revision and short.last_step is None


def test_complete_query_response_native_budget_exact_and_one_less(tyx_mode, tmp_path):
    items = (case.fixture("A", world=7), case.fixture("A", world=31))
    with case.saved(tmp_path / "native-budget.jsonl.gz") as journal:
        with case.compile_scene(items, tyx_mode, site_capacity=4) as first:
            expected, _, _ = case.advance(first, items, case.reference_initial(items), journal, "reference")
            stats = first.stats()
            native = 116 * 4 + 12 * 8 + 24 * 4 + 4 * 2 + 8 * 3
            assert stats["response_native_bytes"] == native
            budget = native + stats["query"]["peak_required_bytes"]
            journal.emit("registered-budget", stats=stats, exact_bytes=budget)
        with case.compile_scene(items, tyx_mode, site_capacity=4, memory_limit_bytes=budget) as exact:
            actual, _, _ = case.advance(exact, items, case.reference_initial(items), journal, "exact")
            case.state_equal(actual, expected, items)
        with case.compile_scene(items, tyx_mode, site_capacity=4, memory_limit_bytes=budget - 1) as short:
            before, capsule, revision = short.snapshot(), short.checkpoint(), short.revision
            with pytest.raises((MemoryError, ValueError), match="memory|budget|reservation|limit"):
                short.step(case.forces(items), case.H, expected_revision=revision)
            case.state_equal(before, short.snapshot(), items)
            assert case.encode(short.checkpoint()) == case.encode(capsule)
            journal.emit("one-less-rejected", state=short.snapshot(), last_attempt=short.last_attempt)


def test_static_layout_order_offsets_alias_capacity_and_immutability():
    from phymeo.backends._deformable_response_layout import compile_deformable_response_layout

    participants = case.participants((case.fixture("A", world=31), case.fixture("A", world=7)))
    left = compile_deformable_response_layout(participants)
    right = compile_deformable_response_layout(tuple(reversed(participants)))
    assert left.worlds == (7, 31) and left.multibody
    assert left.participant_keys == tuple(sorted(p.key for p in participants)) == right.participant_keys
    np.testing.assert_array_equal(left.body_offsets, (0, 2, 4))
    assert left.body_offsets.dtype == np.int32 and not left.body_offsets.flags.writeable
    with pytest.raises(ValueError):
        left.body_offsets.setflags(write=True)
    with pytest.raises(TypeError):
        left.participant_slots[left.participant_keys[0]] = 5
    with pytest.raises(ValueError):
        compile_deformable_response_layout((participants[0], participants[0]))
    same_world = tuple(p for p in participants if p.key.world == 7)
    alias = replace(same_world[1], body_ids=same_world[0].body_ids)
    with pytest.raises(ValueError):
        compile_deformable_response_layout((same_world[0], alias))
    eight = case.capacity_participants(8)
    assert len(compile_deformable_response_layout(eight).participant_keys) == 8
    with pytest.raises(ValueError):
        compile_deformable_response_layout(case.capacity_participants(9))


@pytest.mark.parametrize("bad", ("wrong-world", "wrong-body", "missing-source"))
def test_original_asset_installation_rejected_before_response_allocation(bad, monkeypatch):
    import phymeo.backends.tyx_deformable_contact as product

    item = case.fixture("A", world=7)
    rows = [b["row"] for b in item["bodies"]]
    first = rows[0]
    mapping = dict(first.collider_keys)
    source = next(iter(mapping))
    installed = mapping[source]
    if bad == "wrong-world":
        mapping[source] = case.ph.CollisionKey(31, installed.body, installed.shape)
    elif bad == "wrong-body":
        mapping[source] = case.ph.CollisionKey(7, 999, installed.shape)
    else:
        mapping = {}
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid registration allocated response buffers")

    monkeypatch.setattr(product._ResponseEngine, "__init__", forbidden)
    with pytest.raises((TypeError, ValueError)):
        rows[0] = replace(first, collider_keys=mapping)
        declared = replace(case.declaration((item,)), articulations=tuple(rows))
        case.ph.compile_participant_scene(declared, mode="interpret")
    assert calls == []

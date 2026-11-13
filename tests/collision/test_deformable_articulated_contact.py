"""M-B2 tests-first matrix; authoring only, fixed C/D inputs and original gates."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "deformable_articulated_consumer",
    Path(__file__).parents[1] / "installed" / "check_deformable_articulated.py",
)
case = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(case)
pytestmark = pytest.mark.tyx


@pytest.fixture(autouse=True)
def public_api():
    case.initialize()


@pytest.mark.parametrize("kind", ("C", "D"))
def test_full_tree_loaded_links_eight_continuous_steps(kind, tyx_mode, tmp_path):
    items = tuple(case.fixture(kind, world=w) for w in (31, 7))
    with case.saved(tmp_path / (kind + ".jsonl.gz")) as journal:
        case.run_window(items, tyx_mode, journal, steps=8)


def test_registration_permutation_preserves_full_tree_and_material_identity(tyx_mode, tmp_path):
    items = (case.fixture("D", world=31), case.fixture("C", world=7))
    with case.saved(tmp_path / "permutation.jsonl.gz") as journal:
        records = []
        for reverse in (False, True):
            with case.compile_scene(items, tyx_mode, reverse=reverse) as scene:
                refs = case.reference_initial(items)
                states = []
                for step in range(2):
                    state, refs, _ = case.advance(scene, items, refs, journal, step)
                    states.append(state)
                records.append(states)
        for left, right in zip(*records, strict=True):
            case.state_equal(left, right, items)


def test_mixed_accepted_independent_bodies_and_articulated_world(tyx_mode, tmp_path):
    items = (case.legacy_fixture(world=7), case.fixture("D", world=31))
    with case.saved(tmp_path / "mixed.jsonl.gz") as journal:
        case.run_window(items, tyx_mode, journal, steps=2)


def test_checkpoint_partial_reset_restore_and_true_successors(tyx_mode, tmp_path):
    items = (case.fixture("C", world=7), case.fixture("D", world=31))
    with case.saved(tmp_path / "lifecycle.jsonl.gz") as journal:
        case.lifecycle(items, tyx_mode, journal)


def retained(scene, items, before, capsule, revision, old, banks, journal):
    state = scene.snapshot()
    journal.emit(
        "retained",
        state=state,
        checkpoint=scene.checkpoint(),
        stats=scene.stats(),
        last_attempt=scene.last_attempt,
    )
    case.state_equal(before, state, items)
    assert case.encode(scene.checkpoint()) == case.encode(capsule)
    assert scene.revision == revision and scene.last_step is old
    response = scene._response
    assert (response._nodes._active, response._scene._service._batch._active) == banks


@pytest.mark.parametrize("seam", ("query", "publication", "cross-world-body"))
def test_complete_late_attempt_preserves_all_owners_and_real_retry(
    seam, tyx_mode, tmp_path, monkeypatch
):
    items = (case.fixture("C", world=7), case.fixture("D", world=31))
    with case.saved(tmp_path / (seam + ".jsonl.gz")) as journal:
        with case.compile_scene(items, tyx_mode) as scene:
            refs = case.reference_initial(items)
            _, refs, _ = case.advance(scene, items, refs, journal, "warm")
            before, capsule, revision, old = (
                scene.snapshot(),
                scene.checkpoint(),
                scene.revision,
                scene.last_step,
            )
            response = scene._response
            banks = (response._nodes._active, response._scene._service._batch._active)
            attribute = {
                "query": "_query_entry",
                "publication": "_prepare_response_publication",
                "cross-world-body": "_law_rows",
            }[seam]
            original = getattr(response, attribute)
            completed = []

            def reject(*args, **kwargs):
                value = original(*args, **kwargs)
                completed.append(value)
                journal.emit("actual-fault-prefix", seam=seam, value=value)
                if seam == "cross-world-body":
                    sites, rows, identities, bary, levers = value
                    identities = identities.copy()
                    assert len(identities) and identities[-1, 0] == 1
                    identities[-1, 4] = 0
                    return sites, rows, identities, bary, levers
                if seam == "publication":
                    assert set(value.rigid_publication["candidate"]) == set(scene.participant_keys)
                raise RuntimeError("M-B2 complete attempt deliberately refused")

            with monkeypatch.context() as patch:
                patch.setattr(response, attribute, reject)
                journal.emit(
                    "fault-attempt",
                    before=before,
                    capsule=capsule,
                    inputs=case.forces(items),
                    h=case.H,
                )
                with pytest.raises(
                    (ValueError, RuntimeError, FloatingPointError, case.ph.SolverFailure)
                ) as caught:
                    scene.step(case.forces(items), case.H, expected_revision=revision)
                journal.emit("fault-return", error=caught.value, last_attempt=scene.last_attempt)
                assert len(completed) == 1
                retained(scene, items, before, capsule, revision, old, banks, journal)
            actual, _, _ = case.advance(scene, items, refs, journal, "retry")
            with case.compile_scene(items, tyx_mode) as clone:
                fresh = case.reference_initial(items)
                _, fresh, _ = case.advance(clone, items, fresh, journal, "clone-warm")
                expected, _, _ = case.advance(clone, items, fresh, journal, "clone-next")
            case.state_equal(actual, expected, items)


def test_complete_body_native_budget_exact_and_one_less(tyx_mode, tmp_path):
    items = (case.fixture("C", world=7), case.fixture("D", world=31))
    capacity = 6
    with case.saved(tmp_path / "body-budget.jsonl.gz") as journal:
        with case.compile_scene(items, tyx_mode, site_capacity=capacity) as first:
            expected, _, _ = case.advance(
                first, items, case.reference_initial(items), journal, "reference"
            )
            stats = first.stats()
            native = 116 * capacity + 12 * 8 + 24 * 7 + 4 * 2 + 8 * 3
            assert stats["response_native_bytes"] == native
            budget = native + stats["query"]["peak_required_bytes"]
            journal.emit("exact-native-budget", bytes=budget, stats=stats)
        with case.compile_scene(
            items, tyx_mode, site_capacity=capacity, memory_limit_bytes=budget
        ) as exact:
            actual, _, _ = case.advance(
                exact, items, case.reference_initial(items), journal, "exact"
            )
            case.state_equal(actual, expected, items)
        with case.compile_scene(
            items, tyx_mode, site_capacity=capacity, memory_limit_bytes=budget - 1
        ) as short:
            before, capsule, revision = short.snapshot(), short.checkpoint(), short.revision
            journal.emit("one-less-attempt", before=before, capsule=capsule, h=case.H)
            with pytest.raises((MemoryError, ValueError), match="memory|budget|reservation|limit"):
                short.step(case.forces(items), case.H, expected_revision=revision)
            case.state_equal(before, short.snapshot(), items)
            assert case.encode(capsule) == case.encode(short.checkpoint())
            journal.emit("one-less-rejected", last_attempt=short.last_attempt)


def test_no_response_sites_still_advances_every_body(tyx_mode, tmp_path):
    items = tuple(case.fixture("D", world=w) for w in (7, 31))
    for item in items:
        item["materials"] = dict.fromkeys(item["materials"], None)
    with case.saved(tmp_path / "zero-response.jsonl.gz") as journal:
        case.run_window(items, tyx_mode, journal, steps=2, require_loaded=False)


def test_host_layout_full_addresses_and_same_world_tree_mixture_rejection():
    from phymeo.backends._deformable_response_layout import compile_deformable_response_layout

    items = (case.fixture("D", world=31), case.fixture("C", world=7))
    participants = case.participants(items)
    layout = compile_deformable_response_layout(tuple(reversed(participants)))
    assert layout.articulated and layout.multibody and layout.worlds == (7, 31)
    expected = tuple(
        (b["key"], group, body_id)
        for item in sorted(items, key=lambda x: x["world"])
        for b in item["bodies"]
        for group, body_id in enumerate(b["body_ids"])
    )
    assert layout.body_addresses == expected
    np.testing.assert_array_equal(layout.body_offsets, (0, 3, 7))
    for slot, (key, group, _) in enumerate(expected):
        assert layout.body_slots[key, group] == slot
    assert tuple(layout.participant_body_counts.values()) == (3, 4)
    with pytest.raises(ValueError):
        layout.body_offsets.setflags(write=True)
    with pytest.raises(TypeError):
        layout.body_slots[expected[0][:2]] = 99
    extra = case.participants((case.legacy_fixture(world=7),))[0]
    extra = replace(extra, key=case.ph.ResidentParticipantKey(7, 91), body_ids=(901,))
    with pytest.raises(ValueError, match="articulated|single|participant|world"):
        compile_deformable_response_layout((participants[1], extra))


@pytest.mark.parametrize("bad", ("wrong-world", "wrong-body", "wrong-link", "missing-source"))
def test_host_bad_original_body_source_rejected_before_response_allocation(bad, monkeypatch):
    import phymeo.backends.tyx_deformable_contact as product

    item = case.fixture("C", world=7)
    body = item["bodies"][0]
    row = body["row"]
    mapping = dict(row.collider_keys)
    source = next(iter(mapping))
    installed = mapping[source]
    if bad == "wrong-world":
        mapping[source] = case.ph.CollisionKey(31, installed.body, installed.shape)
    elif bad == "wrong-body":
        mapping[source] = case.ph.CollisionKey(7, 999, installed.shape)
    elif bad == "wrong-link":
        mapping[source] = case.ph.CollisionKey(7, body["body_ids"][1], installed.shape)
    else:
        mapping = {}
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("invalid original body source allocated response")

    monkeypatch.setattr(product._ResponseEngine, "__init__", forbidden)
    with pytest.raises((TypeError, ValueError)):
        wrong = replace(row, collider_keys=mapping)
        declared = replace(case.declaration((item,)), articulations=(wrong,))
        case.ph.compile_participant_scene(declared, mode="interpret")
    assert calls == []

"""Whole heterogeneous publication, boundary restore and late-failure tests."""

import numpy as np
import pytest
from test_mujoco_tyx_heterogeneous import (
    MODES,
    ORDER,
    H,
    MuJoCoControl,
    MuJoCoTyxHeterogeneousBatch,
    ball,
    declarations,
    packets,
    scalar,
)

from phymeo.backends.mujoco_heterogeneous_checkpoint import MuJoCoTyxHeterogeneousCheckpoint
from phymeo.training.resident_stepper import PhysicsStepFailure


def make(mode):
    mj, xml, models, specs, worlds = declarations()
    return models, MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode)


def checkpoint_rows(batch):
    capsule = batch.checkpoint()
    leaves, _ = capsule._read(batch._binding)
    result = {}
    for leaf in leaves:
        raw = leaf._read(leaf.binding)
        for index, world in enumerate(raw["worlds"]):
            result[world["key"]] = dict(
                world=world,
                banks={
                    name: raw[name][index] for name in ("q", "v", "clock", "act") if name in raw
                },
            )
    return result


def step(batch, models, tick=0):
    batch.set_controls(packets(models, tick))
    accepted = batch.step(H)
    assert dict(accepted) == dict.fromkeys(ORDER, True)
    return batch.integration_snapshot()


@pytest.mark.parametrize("mode", MODES)
def test_checkpoint_bytes_restore_and_partial_original_reset(mode, tmp_path):
    models, batch = make(mode)
    history = {}
    try:
        start = batch.integration_snapshot()
        step(batch, models)
        capsule = batch.checkpoint()
        (tmp_path / "heterogeneous.checkpoint").write_bytes(capsule.to_bytes())
        restored = MuJoCoTyxHeterogeneousCheckpoint.from_bytes(capsule.to_bytes())
        successor = step(batch, models, 1)
        history["successor"] = successor
        revision = batch.checkpoint_revision
        batch.restore(restored, expected_revision=revision)
        assert batch.checkpoint_revision == revision + 1
        repeated = step(batch, models, 1)
        history["repeated"] = repeated
        ball.exact(successor, repeated, "checkpoint exact actual successor")
        before = batch.integration_snapshot()
        before_raw = checkpoint_rows(batch)
        old_owners = tuple(batch._buckets)
        result = batch.reset((41, 7))
        after = batch.integration_snapshot()
        after_raw = checkpoint_rows(batch)
        history.update(
            before_reset=before,
            after_reset=after,
            before_raw=before_raw,
            after_raw=after_raw,
            reset=result,
        )
        assert batch._buckets[2] is old_owners[2]
        for key in ORDER:
            if key in (41, 7):
                assert after[key].tick == 0 and after[key].last_step is None
                for name in ("q", "v", "act", "time"):
                    ball.exact(getattr(after[key], name), getattr(start[key], name), name)
            else:
                ball.exact(after[key], before[key], "unselected full accepted record")
                ball.exact(
                    after_raw[key],
                    before_raw[key],
                    "unselected exact raw clock/CPU mirror/receipt/initial",
                )
        history["after_reset_successor"] = step(batch, models, 2)
        assert history["after_reset_successor"][41].tick == 1
        assert history["after_reset_successor"][11].tick == before[11].tick + 1
        assert batch.counters["boundary_clones"] == 5
    finally:
        batch.close()
        scalar.save(tmp_path / "heterogeneous-lifecycle.json.gz", history)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fault", ("last-status", "last-download", "late-aggregate"))
def test_last_bucket_failure_retains_every_world_and_retry(mode, fault, monkeypatch, tmp_path):
    models, batch = make(mode)
    _, reference = make(mode)
    history = {}
    try:
        command = packets(models, 0)
        batch.set_controls(command)
        reference.set_controls(command)
        before = batch.checkpoint().to_bytes()
        old_state, old_receipt = batch.integration_snapshot(), batch.last_step
        target = batch._buckets[-1]
        if fault == "late-aggregate":

            def fail(operation, pending):
                if operation == "step":
                    raise RuntimeError("last aggregate preparation fault")

            monkeypatch.setattr(batch, "_before_commit", fail)
        else:
            original = target._device.download

            def broken():
                actual = original()
                if fault == "last-download":
                    raise RuntimeError("last real download complete then fail")
                actual["status"][-1] = 2
                return actual

            monkeypatch.setattr(target._device, "download", broken)
        with pytest.raises(PhysicsStepFailure) as caught:
            batch.step(H)
        assert dict(caught.value.accepted_worlds) == dict.fromkeys(ORDER, False)
        assert batch.checkpoint().to_bytes() == before
        assert batch.last_step is old_receipt
        ball.exact(batch.integration_snapshot(), old_state, "all buckets retain")
        history.update(
            fault=fault, before=old_state, failure=str(caught.value), counters=batch.counters
        )
        assert batch.counters["completed_steps"] == 0
        assert batch.counters["accepted_world_steps"] == 0
        assert batch.counters["bucket_physics_prepares"] == 3
        monkeypatch.undo()
        assert batch.step(H) == reference.step(H)
        history["retry"] = batch.integration_snapshot()
        ball.exact(history["retry"], reference.integration_snapshot(), "real retry")
    finally:
        batch.close()
        reference.close()
        scalar.save(tmp_path / "heterogeneous-late-step.json.gz", history)


@pytest.mark.parametrize("mode", MODES)
def test_invalid_last_control_has_no_partial_publication(mode, tmp_path):
    models, batch = make(mode)
    try:
        good = packets(models, 0)
        batch.set_controls(good)
        before = batch.checkpoint().to_bytes()
        launches = batch.counters["completed_kernels"]
        bad = dict(good)
        bad[59] = MuJoCoControl(np.zeros(models["free"].nu + 1))
        with pytest.raises((ValueError, TypeError)):
            batch.set_controls(bad)
        assert batch.checkpoint().to_bytes() == before
        assert batch.counters["completed_kernels"] == launches
        batch.step(H)
        scalar.save(
            tmp_path / "heterogeneous-control.json.gz",
            dict(output=batch.integration_snapshot(), receipt=batch.last_step),
        )
    finally:
        batch.close()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("operation", ("reset", "restore"))
def test_boundary_last_prepublication_failure_retains_and_retries(
    mode, operation, monkeypatch, tmp_path
):
    models, batch = make(mode)
    history = {}
    try:
        step(batch, models)
        saved = batch.checkpoint()
        step(batch, models, 1)
        before = batch.checkpoint().to_bytes()
        old = tuple(batch._buckets)

        def fail(name, pending):
            if name == operation:
                raise RuntimeError("all replacement buckets prepared before refusal")

        monkeypatch.setattr(batch, "_before_commit", fail)
        with pytest.raises(RuntimeError, match="replacement"):
            if operation == "reset":
                batch.reset((41, 7, 305))
            else:
                batch.restore(saved, expected_revision=batch.checkpoint_revision)
        assert tuple(batch._buckets) == old
        assert batch.checkpoint().to_bytes() == before
        history["retained"] = batch.integration_snapshot()
        monkeypatch.undo()
        if operation == "reset":
            batch.reset((41, 7, 305))
        else:
            batch.restore(saved, expected_revision=batch.checkpoint_revision)
        history["retry"] = step(batch, models, 2)
    finally:
        batch.close()
        scalar.save(tmp_path / "heterogeneous-boundary-failure.json.gz", history)


@pytest.mark.parametrize("mode", MODES)
def test_binding_and_stale_cas_rejected_before_replacement(mode, monkeypatch, tmp_path):
    models, batch = make(mode)
    mj, xml, native, specs, worlds = declarations(tuple(reversed(ORDER)))
    other = MuJoCoTyxHeterogeneousBatch(specs, worlds, mode=mode)
    try:
        saved = batch.checkpoint()
        step(batch, models)
        before = batch.checkpoint().to_bytes()
        calls = []

        def forbidden(*args, **kwargs):
            calls.append(True)
            raise AssertionError("clone called before binding validation")

        monkeypatch.setattr(batch, "_clone_bucket", forbidden)
        with pytest.raises(ValueError, match="revision"):
            batch.restore(saved, expected_revision=batch.checkpoint_revision - 1)
        with pytest.raises(ValueError, match="binding"):
            batch.restore(other.checkpoint(), expected_revision=batch.checkpoint_revision)
        assert not calls and batch.checkpoint().to_bytes() == before
        scalar.save(
            tmp_path / "heterogeneous-binding.json.gz",
            dict(binding=saved.binding, retained=batch.integration_snapshot()),
        )
    finally:
        batch.close()
        other.close()


@pytest.mark.parametrize("mode", MODES)
def test_reentrant_writer_rejected_without_dropping_original_draft(mode, monkeypatch):
    models, batch = make(mode)
    try:
        batch.set_controls(packets(models, 0))
        calls = []

        def nested(operation, pending):
            if operation == "step":
                for function in (
                    lambda: batch.step(H),
                    lambda: batch.reset(),
                    lambda: batch.set_controls(packets(models, 1)),
                    batch.close,
                ):
                    with pytest.raises(RuntimeError, match="transaction"):
                        function()
                    calls.append(True)

        monkeypatch.setattr(batch, "_before_commit", nested)
        batch.step(H)
        assert len(calls) == 4 and batch.counters["completed_steps"] == 1
    finally:
        monkeypatch.undo()
        batch.close()


@pytest.mark.parametrize("mode", MODES)
def test_postpublication_cleanup_failure_keeps_owner_and_accepted_reset(mode, monkeypatch):
    models, batch = make(mode)
    try:
        step(batch, models)
        target = batch._buckets[0]
        original = target.close
        calls = []

        def fail_once():
            calls.append(True)
            if len(calls) == 1:
                raise RuntimeError("owner close refused before actual free")
            return original()

        monkeypatch.setattr(target, "close", fail_once)
        receipt = batch.reset((41,))
        assert batch.integration_snapshot()[41].tick == 0
        assert receipt.cleanup_errors and batch.cleanup_errors
        assert target in batch._retired
        batch.close()
        assert len(calls) == 2
    finally:
        batch.close()


@pytest.mark.parametrize("close_fails", (False, True))
def test_counter_observation_failure_never_skips_owner_close(close_fails, monkeypatch):
    batch = object.__new__(MuJoCoTyxHeterogeneousBatch)
    batch._retired, batch._cleanup_errors, batch._counter_errors = [], (), ()
    calls = []

    class Owner:
        def close(self):
            calls.append("close")
            if close_fails:
                raise RuntimeError("original close fault")

    owner = Owner()

    def bad_counter(value):
        assert value is owner
        raise RuntimeError("counter fault")

    monkeypatch.setattr(batch, "_observe_owner", bad_counter)
    failures = batch._close_owners((owner,))
    assert calls == ["close"]
    assert len(failures) == (2 if close_fails else 1)
    assert str(failures[0]) == "counter fault"
    assert (owner in batch._retired) == close_fails
    if close_fails:
        assert str(failures[1]) == "original close fault"


@pytest.mark.parametrize("mode", MODES)
def test_postcommit_counter_failure_does_not_refuse_accepted_control(mode, monkeypatch):
    models, batch = make(mode)
    try:
        original = batch._observe_owner
        revision = batch.checkpoint_revision
        commands = packets(models, 1)

        def bad_counter(owner):
            raise RuntimeError("postcommit counter fault")

        monkeypatch.setattr(batch, "_observe_owner", bad_counter)
        batch.set_controls(commands)
        assert batch.checkpoint_revision == revision + 1
        assert len(batch._counter_errors) == len(batch._buckets)
        monkeypatch.setattr(batch, "_observe_owner", original)
        current = batch.integration_snapshot()
        for key in ORDER:
            ball.exact(current[key].held_control, commands[key], "accepted held control")
        batch.step(H)
        assert batch.counters["control_submissions"] == 1
        assert batch.counters["completed_steps"] == 1
    finally:
        monkeypatch.undo()
        batch.close()


@pytest.mark.parametrize("mode", MODES)
def test_cross_owner_restore_reserializes_historical_publication_at_lower_revision(mode):
    """A restored publication keeps source history; CAS belongs to its new owner."""
    from contextlib import ExitStack

    from phymeo.backends import mujoco_heterogeneous_checkpoint as codec

    with ExitStack() as stack:
        models, source = make(mode)
        stack.enter_context(source)
        _, target = make(mode)
        stack.enter_context(target)
        _, relay = make(mode)
        stack.enter_context(relay)

        step(source, models, 0)
        historical = source.last_step
        # Also distinguish current held input from the input of the historical
        # accepted publication. Neither may be rewritten during a restore.
        source.set_controls(packets(models, 1))
        saved_state = source.integration_snapshot()
        capsule = MuJoCoTyxHeterogeneousCheckpoint.from_bytes(source.checkpoint().to_bytes())
        target_revision = target.checkpoint_revision
        assert historical.revision_after > target_revision + 1
        assert capsule.source_revision == source.checkpoint_revision

        boundary = target.restore(capsule, expected_revision=target_revision)
        assert (boundary.revision_before, boundary.revision_after) == (
            target_revision,
            target_revision + 1,
        )
        assert target.checkpoint_revision == target_revision + 1
        ball.exact(
            target.integration_snapshot(), saved_state, "cross-owner complete physical state"
        )
        ball.exact(target.last_step, historical, "original aggregate versions, controls and states")

        # This immediate public operation used to reject the historical
        # revision_after (2) because the receiving owner's revision is only 1.
        forwarded = MuJoCoTyxHeterogeneousCheckpoint.from_bytes(target.checkpoint().to_bytes())
        assert forwarded.source_revision == target.checkpoint_revision
        _, forwarded_publication = forwarded._read(forwarded.binding)
        origin = dict(sha256=capsule.sha256, revision=capsule.source_revision)
        assert forwarded.to_bytes().startswith(codec.HISTORICAL_MAGIC)
        assert dict(forwarded_publication["source_checkpoint"]) == origin
        assert forwarded_publication["bucket_revisions"] == historical.bucket_revisions
        with pytest.raises(ValueError, match="revision"):
            target.restore(forwarded, expected_revision=capsule.source_revision)
        assert target.checkpoint_revision == target_revision + 1
        ball.exact(target.last_step, historical, "stale source CAS cannot alter history")

        relay_revision = relay.checkpoint_revision
        boundary = relay.restore(forwarded, expected_revision=relay_revision)
        assert (boundary.revision_before, boundary.revision_after) == (
            relay_revision,
            relay_revision + 1,
        )
        assert relay.checkpoint_revision == relay_revision + 1
        ball.exact(relay.integration_snapshot(), saved_state, "second-owner physical state")
        ball.exact(relay.last_step, historical, "forwarded historical aggregate is unchanged")
        relayed = MuJoCoTyxHeterogeneousCheckpoint.from_bytes(relay.checkpoint().to_bytes())
        assert relayed.source_revision == relay.checkpoint_revision
        _, relayed_publication = relayed._read(relayed.binding)
        assert dict(relayed_publication["source_checkpoint"]) == origin
        assert relayed_publication == forwarded_publication
        assert relayed_publication["source_checkpoint"]["sha256"] != forwarded.sha256

        # A held-input replacement advances local CAS but does not create a
        # physical publication or re-anchor its original checkpoint provenance.
        target_before_control = target.checkpoint_revision
        target.set_controls(packets(models, 1))
        assert target.checkpoint_revision == target_before_control + 1
        held = MuJoCoTyxHeterogeneousCheckpoint.from_bytes(target.checkpoint().to_bytes())
        assert held.source_revision == target.checkpoint_revision
        _, held_publication = held._read(held.binding)
        assert held.to_bytes().startswith(codec.HISTORICAL_MAGIC)
        assert held_publication == forwarded_publication
        ball.exact(target.last_step, historical, "held input preserves historical publication")

        # All three owners consume the same already-held command, without
        # another reset, state injection or control submission.
        for owner in (source, target, relay):
            revision = owner.checkpoint_revision
            child_revisions = tuple(child.checkpoint_revision for child in owner._buckets)
            assert dict(owner.step(H)) == dict.fromkeys(ORDER, True)
            assert owner.checkpoint_revision == revision + 1
            assert (owner.last_step.revision_before, owner.last_step.revision_after) == (
                revision,
                revision + 1,
            )
            assert owner.last_step.bucket_revisions == tuple((r, r + 1) for r in child_revisions)
            fresh = owner.checkpoint()
            _, publication = fresh._read(fresh.binding)
            assert fresh.to_bytes().startswith(codec.MAGIC)
            assert "source_checkpoint" not in publication
            assert fresh.source_revision == owner.checkpoint_revision
        successor = source.integration_snapshot()
        ball.exact(target.integration_snapshot(), successor, "restored next physical step")
        ball.exact(relay.integration_snapshot(), successor, "forwarded next physical step")


def _publication_codec_fixture():
    """Typed leaf bytes and envelope data only; not a physics-state fixture."""
    import json

    from phymeo.backends import mujoco_heterogeneous_checkpoint as codec
    from phymeo.backends.mujoco_checkpoint import _make_checkpoint

    binding = {"world_keys": [7]}
    leaf = _make_checkpoint({"fixture": "codec-only"}, 629, {"marker": "opaque leaf data"})
    publication = dict(
        schema="mujoco-tyx-heterogeneous-step-v1",
        revision_before=627,
        revision_after=628,
        bucket_revisions=[[627, 628]],
        world_keys=[7],
    )
    source = codec._capture(binding, 629, (leaf,), publication)
    body = source.to_bytes().split(b"\n", 2)[2]
    return codec, _make_checkpoint, source, json.loads(body)


def _publication_codec_bytes(magic, payload):
    """Recompute checksum so each negative reaches schema/origin validation."""
    import hashlib
    import json

    body = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return magic + hashlib.sha256(body).hexdigest().encode() + b"\n" + body


@pytest.mark.parametrize("changed", ("future-publication", "v2-origin", "unknown-field"))
def test_v1_publication_codec_keeps_original_revision_and_field_constraints(changed):
    codec, _, source, payload = _publication_codec_fixture()
    if changed == "future-publication":
        payload["revision"] = 1
    elif changed == "v2-origin":
        payload["publication"]["source_checkpoint"] = dict(
            sha256=source.sha256, revision=source.source_revision
        )
    else:
        payload["publication"]["historical_revision"] = 629
    with pytest.raises(ValueError):
        codec.MuJoCoTyxHeterogeneousCheckpoint.from_bytes(
            _publication_codec_bytes(codec.MAGIC, payload)
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("revision", True),
        ("revision", 629.0),
        ("revision", "629"),
        ("revision", None),
        ("revision", -1),
        ("revision", 627),
        ("sha256", 123),
        ("sha256", True),
        ("sha256", None),
        ("sha256", ""),
        ("sha256", "a" * 63),
        ("sha256", "a" * 65),
        ("sha256", "g" * 64),
        ("sha256", "A" * 64),
    ),
)
def test_v2_publication_codec_rejects_bad_source_revision_or_hash(field, value):
    codec, _, source, payload = _publication_codec_fixture()
    payload["revision"] = 1
    origin = dict(sha256=source.sha256, revision=source.source_revision)
    origin[field] = value
    payload["publication"]["source_checkpoint"] = origin
    with pytest.raises(ValueError):
        codec.MuJoCoTyxHeterogeneousCheckpoint.from_bytes(
            _publication_codec_bytes(codec.HISTORICAL_MAGIC, payload)
        )


@pytest.mark.parametrize(
    "changed",
    (
        "missing-origin",
        "missing-revision",
        "missing-sha",
        "extra-origin-field",
        "origin-not-object",
        "missing-history-field",
        "missing-publication",
    ),
)
def test_v2_publication_codec_requires_complete_exact_origin_and_history_schema(changed):
    codec, _, source, payload = _publication_codec_fixture()
    payload["revision"] = 1
    origin = dict(sha256=source.sha256, revision=source.source_revision)
    payload["publication"]["source_checkpoint"] = origin
    if changed == "missing-origin":
        payload["publication"].pop("source_checkpoint")
    elif changed == "missing-revision":
        origin.pop("revision")
    elif changed == "missing-sha":
        origin.pop("sha256")
    elif changed == "extra-origin-field":
        origin["owner"] = "unvalidated-owner"
    elif changed == "origin-not-object":
        payload["publication"]["source_checkpoint"] = [source.sha256, source.source_revision]
    elif changed == "missing-history-field":
        payload["publication"].pop("bucket_revisions")
    else:
        payload["publication"] = None
    with pytest.raises(ValueError):
        codec.MuJoCoTyxHeterogeneousCheckpoint.from_bytes(
            _publication_codec_bytes(codec.HISTORICAL_MAGIC, payload)
        )


def test_v2_codec_round_trip_keeps_original_history_and_origin_with_current_leaf_revisions():
    codec, make_leaf, source, payload = _publication_codec_fixture()
    assert source.to_bytes().startswith(codec.MAGIC)
    publication = payload["publication"]
    publication["source_checkpoint"] = dict(sha256=source.sha256, revision=source.source_revision)
    current_leaf = make_leaf({"fixture": "codec-only"}, 2, {"marker": "restored leaf data"})
    first = codec._capture(source.binding, 1, (current_leaf,), publication)
    parsed = codec.MuJoCoTyxHeterogeneousCheckpoint.from_bytes(first.to_bytes())
    leaves, history = parsed._read(parsed.binding)
    assert parsed.to_bytes() == first.to_bytes()
    assert parsed.to_bytes().startswith(codec.HISTORICAL_MAGIC)
    assert parsed.source_revision == 1
    assert leaves[0].to_bytes() == current_leaf.to_bytes() and leaves[0].source_revision == 2
    assert history["revision_before"] == 627 and history["revision_after"] == 628
    assert history["bucket_revisions"] == ((627, 628),)
    assert dict(history["source_checkpoint"]) == dict(sha256=source.sha256, revision=629)
    with pytest.raises(TypeError):
        history["source_checkpoint"]["revision"] = 1

    relay_leaf = make_leaf({"fixture": "codec-only"}, 4, {"marker": "relayed leaf data"})
    second = codec._capture(parsed.binding, 3, (relay_leaf,), history)
    later_leaves, later_history = second._read(second.binding)
    assert second.source_revision == 3 and later_leaves[0].source_revision == 4
    assert later_history == history
    assert later_history["source_checkpoint"]["sha256"] == source.sha256 != first.sha256
    assert later_history["bucket_revisions"] == ((627, 628),)

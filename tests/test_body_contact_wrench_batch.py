"""Batch readback contracts, hand wrench algebra, and one short CUDA lifecycle.

HandWorkspace is a HOST protocol fixture, never a physics/GPU oracle. The
marked integration case reuses the registered I3j scene and takes three steps.
"""

from contextlib import ExitStack
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import numpy as np
import pytest
from test_body_contact_wrench import (
    ORIGIN,
    REFERENCE,
    _assert_vector,
    _changed,
    _packet,
    _reader,
)

from phymeo.backends.area_pressure_readback import AreaPressureEntryReader
from phymeo.control.body_contact_wrench import BodyContactWrenchRequest


def _request(world=13, body="object", reference=ORIGIN, groups=()):
    return BodyContactWrenchRequest(world, body, reference, groups)


def _bytes(ws):
    return {name: value.tobytes() for name, value in ws.arrays.items()}


def _validation_spy(monkeypatch):
    calls = []
    original = AreaPressureEntryReader.validate

    def validate(self, arrays, frame):
        calls.append(frame)
        return original(self, arrays, frame)

    monkeypatch.setattr(AreaPressureEntryReader, "validate", validate)
    return calls


def test_many_hand_vectors_order_duplicates_groups_and_both_sides(monkeypatch):
    ws = _packet()
    reader = _reader(ws)
    original = _bytes(ws)
    calls = _validation_spy(monkeypatch)
    queries = (
        _request(41, "support", REFERENCE, ("object",)),
        _request(groups=("finger",)),
        _request(reference=REFERENCE, groups=("support", "finger", "bystander")),
        _request(41, "finger", groups=("object",)),
        _request(groups=("finger",)),
        _request(41, "bystander", REFERENCE),
    )
    values = reader.measure_many(ws.frame, queries)
    assert type(values) is tuple and len(values) == len(queries)
    assert ws.reads == 1 and calls == [ws.frame]
    assert values[1] == values[4]
    expected = (
        ((0, -3, 0), (4.8, 0, 3.6)),
        ((1, 5, 4), (-3.5, -1.25, -2.25)),
        ((1, 5, 4), (-0.3, -0.85, -3.55)),
        ((-1, -2, -4), (-2.5, 1.25, -0.75)),
        ((1, 5, 4), (-3.5, -1.25, -2.25)),
        (ORIGIN, ORIGIN),
    )
    for result, query, (force, moment) in zip(values, queries, expected, strict=True):
        assert (result.time_s, result.input_tick, result.output_tick) == (0.375, 4, 5)
        assert result.reference_point_world == query.reference_point_world
        _assert_vector(result.total_force_world, force)
        _assert_vector(result.total_moment_world, moment)
    assert values[0].group_normal_loads == (3.0,)
    assert values[1].group_normal_loads == (4.0,)
    assert values[2].group_normal_loads == (3.0, 4.0, 0.0)
    _assert_vector(values[1].group_force_world, (1, 2, 4))
    _assert_vector(values[1].other_force_world, (0, 3, 0))
    _assert_vector(values[2].group_force_world, values[2].total_force_world)
    _assert_vector(values[2].other_force_world, ORIGIN)
    native = next(c for c in values[1].contacts if c.source == "native" and c.active)
    _assert_vector(native.couple_world, (0.5, -0.25, 0.75))
    assert all(c.area_m2 == 0.02 / 3 for c in values[1].contacts if c.source == "area")

    # All interacting bodies, about one common nonzero point, close in each
    # world. No omitted counterpart is silently absorbed into a group total.
    closure = tuple(
        _request(key, name, REFERENCE)
        for key in (41, 13)
        for name in ("object", "finger", "support", "bystander")
    )
    closed = reader.measure_many(ws.frame, closure)
    assert ws.reads == 2 and len(calls) == 2
    for start in (0, 4):
        _assert_vector(
            np.sum([r.total_force_world for r in closed[start : start + 4]], axis=0), ORIGIN
        )
        _assert_vector(
            np.sum([r.total_moment_world for r in closed[start : start + 4]], axis=0), ORIGIN
        )
    refined = _packet(split=True)
    fine = _reader(refined).measure_many(refined.frame, queries)
    assert refined.reads == 1
    for coarse, split in zip(values, fine, strict=True):
        _assert_vector(split.total_force_world, coarse.total_force_world)
        _assert_vector(split.total_moment_world, coarse.total_moment_world)
    assert len([c for c in fine[1].contacts if c.source == "area"]) == 12
    assert _bytes(ws) == original


def test_request_owns_reference_and_frozen_results_are_independent():
    supplied = np.array(REFERENCE)
    query = _request(reference=supplied, groups=("finger",))
    supplied[:] = 99
    assert type(query.reference_point_world) is tuple
    assert query.reference_point_world == REFERENCE
    with pytest.raises(FrozenInstanceError):
        query.world_key = 41
    with pytest.raises(TypeError):
        query.reference_point_world[0] = 99
    for invalid in ((0, 1), (np.inf, 0, 0), (np.nan, 0, 0), "not-a-point"):
        with pytest.raises(ValueError):
            _request(reference=invalid)
    ws = _packet()
    (result,) = _reader(ws).measure_many(ws.frame, (query,))
    assert result.reference_point_world == REFERENCE
    _assert_vector(result.total_moment_world, (-0.3, -0.85, -3.55))
    with pytest.raises(FrozenInstanceError):
        result.time_s = 10
    with pytest.raises(FrozenInstanceError):
        result.contacts[0].normal_load = 10


def test_entire_request_metadata_preflight_and_tuple_caps_before_readback(monkeypatch):
    ws = _packet()
    reader = _reader(ws)
    calls = _validation_spy(monkeypatch)
    valid = _request()

    class TupleSubclass(tuple):
        pass

    class RequestSubclass(BodyContactWrenchRequest):
        pass

    for bad in (
        [],
        [valid],
        (),
        (valid,) * 1025,
        (valid, object()),
        TupleSubclass((valid,)),
        (valid, RequestSubclass(13, "object", ORIGIN)),
    ):
        with pytest.raises(ValueError):
            reader.measure_many(ws.frame, bad)
    # Recheck every public field even if a caller bypasses frozen assignment.
    # The valid first request must not start a read before the last is checked.
    for field, value in (
        ("world_key", 99),
        ("world_key", True),
        ("body_name", "missing"),
        ("group_body_names", ("finger", "finger")),
        ("group_body_names", ("object",)),
        ("group_body_names", ("missing",)),
        ("group_body_names", ["finger"]),
        ("reference_point_world", (0, 1)),
        ("reference_point_world", (0, np.inf, 0)),
    ):
        last = _request(41)
        object.__setattr__(last, field, value)
        with pytest.raises(ValueError):
            reader.measure_many(ws.frame, (valid, last))
    assert ws.reads == 0 and calls == []
    # A normal retry remains usable and still performs one whole-frame read.
    assert reader.measure_many(ws.frame, (valid, valid))[0].output_tick == 5
    assert ws.reads == 1 and calls == [ws.frame]


def test_last_numeric_failure_is_all_or_nothing_and_preserves_inputs():
    clean = _packet()
    arrays = {name: value.copy() for name, value in clean.arrays.items()}
    # Genuine device protocol sentinel: finite FP32 operands, but the point x
    # force product exceeds 2**125. Unrelated selected bodies remain usable.
    arrays["area_ids"][1, :3, 5] = 0
    arrays["area_data"][1, :3, :3] = (2.0**64, 0, 0)
    arrays["area_data"][1, :3, 8] = 2.0**64
    arrays["area_data"][1, :3, 9:12] = (0, 2.0**64, 0)
    arrays["area_data"][1, :3, 12:15] = 0
    unavailable = _changed(clean, arrays=arrays)
    reader = _reader(unavailable)
    safe_queries = (_request(), _request(41, "finger"))
    earlier = reader.measure_many(unavailable.frame, safe_queries)
    _assert_vector(earlier[0].total_force_world, (1, 5, 4))
    _assert_vector(earlier[1].total_force_world, (-1, -2, -4))
    saved = _bytes(unavailable)
    sentinel = object()
    result = sentinel
    with pytest.raises(ValueError, match="moment"):
        result = reader.measure_many(unavailable.frame, (*safe_queries, _request(41)))
    assert result is sentinel and unavailable.reads == 2
    assert _bytes(unavailable) == saved
    assert reader.measure_many(unavailable.frame, safe_queries) == earlier

    # Valid finite request data can overflow only when actual forces are
    # translated/aggregated. A valid prefix must never escape as a partial tuple.
    reader = _reader(clean)
    stable = reader.measure_many(clean.frame, (_request(),))
    saved = _bytes(clean)
    result = sentinel
    with pytest.raises(ValueError, match="overflow"):
        result = reader.measure_many(
            clean.frame, (_request(), _request(41, "support", (1e308, 0, 0)))
        )
    assert result is sentinel and _bytes(clean) == saved
    assert reader.measure_many(clean.frame, (_request(),)) == stable

    arrays = {name: value.copy() for name, value in clean.arrays.items()}
    arrays["area_meta"][1, 3] = 1
    overflow = _changed(clean, arrays=arrays)
    saved = _bytes(overflow)
    result = sentinel
    with pytest.raises(ValueError):
        result = _reader(overflow).measure_many(overflow.frame, (_request(), _request(41)))
    assert result is sentinel and overflow.reads == 1 and _bytes(overflow) == saved


def test_invalid_history_empty_history_and_native_only_many():
    queries = (_request(41, "support"), _request(), _request(41, "finger"))
    absent = _packet(invalid=True)
    assert _reader(absent).measure_many(absent.frame, queries) == (None, None, None)
    assert absent.reads == 1
    empty = _packet(empty=True)
    values = _reader(empty).measure_many(empty.frame, queries)
    assert empty.reads == 1 and all(v is not None and v.contacts == () for v in values)
    for value in values:
        assert (value.input_tick, value.output_tick) == (4, 5)
        assert value.total_force_world == value.total_moment_world == ORIGIN
    native = _packet(area=False)
    reader = _reader(native)
    values = reader.measure_many(native.frame, queries)
    assert native.reads == 1
    singles = tuple(
        reader.measure(
            native.frame,
            world_key=q.world_key,
            body_name=q.body_name,
            reference_point_world=q.reference_point_world,
            group_body_names=q.group_body_names,
        )
        for q in queries
    )
    assert values == singles and native.reads == 1 + len(queries)
    assert all(not any(v.area_force_world) and not any(v.area_moment_world) for v in values)


def test_stale_and_foreign_frame_rejection_is_delegated_to_workspace(monkeypatch):
    ws = _packet()
    reader = _reader(ws)
    calls = _validation_spy(monkeypatch)
    original = ws.readback
    attempted = []
    marker = ValueError("workspace rejected stale or foreign frame")

    def reject(frame):
        attempted.append(frame)
        raise marker

    with monkeypatch.context() as patch:
        patch.setattr(ws, "readback", reject)
        for frame in (object(), ws.frame):
            with pytest.raises(ValueError) as caught:
                reader.measure_many(frame, (_request(), _request(41)))
            assert caught.value is marker
    assert len(attempted) == 2 and calls == [] and ws.reads == 0
    assert ws.readback == original
    assert reader.measure_many(ws.frame, (_request(),))[0].output_tick == 5
    assert ws.reads == 1 and calls == [ws.frame]


def test_no_cross_frame_cache_and_single_entry_uses_the_batch_path(monkeypatch):
    ws = _packet()
    reader = _reader(ws)
    calls = _validation_spy(monkeypatch)
    query = _request()
    (first,) = reader.measure_many(ws.frame, (query,))
    first_frame = ws.frame
    arrays = {name: value.copy() for name, value in ws.arrays.items()}
    arrays["entry_meta"][:, 1:3] = (5, 6)
    arrays["entry_clock"][:] = (0.625, 0.125, 0.75, 0.125)
    arrays["area_data"][:, :3, 8:15] *= 2
    catalogs = {}
    for key, rows in ws.frame.area_catalogs.items():
        catalog = dict(rows[0])
        catalog["stamp"] = replace(
            catalog["stamp"],
            step_index=5,
            sample_time_s=0.5,
            configuration_id="next-hand-configuration",
        )
        catalogs[key] = (MappingProxyType(catalog),)
    changed = _changed(ws, arrays=arrays, catalogs=catalogs)
    ws.arrays, ws.frame = changed.arrays, changed.frame
    (later,) = reader.measure_many(ws.frame, (query,))
    assert ws.reads == 2 and calls == [first_frame, ws.frame]
    assert (later.time_s, later.input_tick, later.output_tick) == (0.5, 5, 6)
    _assert_vector(later.total_force_world, (1, 8, 4))
    _assert_vector(later.total_moment_world, (-9.5, -1.25, -5.25))
    _assert_vector(first.total_force_world, (1, 5, 4))
    original_many = reader.measure_many
    batches = []

    def tracked(frame, requests):
        batches.append(requests)
        return original_many(frame, requests)

    monkeypatch.setattr(reader, "measure_many", tracked)
    single = reader.measure(
        ws.frame, world_key=13, body_name="object", reference_point_world=ORIGIN
    )
    assert single == later and batches == [(query,)]
    assert ws.reads == 3 and len(calls) == 3


@pytest.mark.cuda
def test_actual_cuda_many_one_readback_singles_exact_reset_and_paired_restore(
    tmp_path, monkeypatch
):
    from test_manipulation_area_env import _declarations, _make
    from test_manipulation_env import evidence, record, step
    from test_mujoco_tyx_area_pressure_lifecycle import _same

    from phymeo.control.body_contact_wrench import BodyContactWrenchReader
    from phymeo.training.manipulation_checkpoint import ManipulationEnvCheckpoint

    declarations = _declarations()
    identities = {
        model: tuple(a.identity for a in assets) for model, assets in declarations["assets"].items()
    }
    queries = (
        _request(305, "object", groups=("support-body",)),
        _request(7, "support-body", REFERENCE, ("object",)),
        _request(41, "object", groups=("left-finger-body", "right-finger-body", "support-body")),
        _request(41, "support-body", groups=("object",)),
        _request(305, "right-finger-body", REFERENCE, ("object",)),
        _request(7, "object"),
        _request(305, "object", groups=("support-body",)),
    )
    with (
        ExitStack() as stack,
        evidence(tmp_path / "body-wrench-batch-current-entry.json.gz") as records,
    ):
        env, batch, _ = _make(stack, declarations)
        reader = BodyContactWrenchReader(env.workspace, area_identities=identities)
        assert reader.measure_many(env.current_frame, queries) == (None,) * len(queries)
        first_step = step(env)
        first = reader.measure_many(env.current_frame, queries)
        checkpoint = ManipulationEnvCheckpoint.from_bytes(env.checkpoint().to_bytes())
        record(records, env, batch, first_step, measured=first)
        second_step = step(env)
        frame = env.current_frame
        original_read = env.workspace.readback
        reads = []

        def counted(current):
            reads.append(current)
            return original_read(current)

        with monkeypatch.context() as patch:
            patch.setattr(env.workspace, "readback", counted)
            validations = _validation_spy(patch)
            many = reader.measure_many(frame, queries)
            assert reads == [frame] and validations == [frame]
            reads.clear()
            validations.clear()
            singles = tuple(
                reader.measure(
                    frame,
                    world_key=q.world_key,
                    body_name=q.body_name,
                    reference_point_world=q.reference_point_world,
                    group_body_names=q.group_body_names,
                )
                for q in queries
            )
            assert many == singles
            assert len(reads) == len(validations) == len(queries)
        assert all(value is not None and value.output_tick == 2 for value in many)
        assert many[0] == many[-1]
        record(records, env, batch, second_step, measured=many)
        env.reset((41,), expected_frame=frame)
        revision, snapshot = batch.checkpoint_revision, batch.integration_snapshot()
        with pytest.raises(ValueError):
            reader.measure_many(frame, queries)
        assert batch.checkpoint_revision == revision
        _same(batch.integration_snapshot(), snapshot)
        reset = reader.measure_many(env.current_frame, queries)
        for query, current, before in zip(queries, reset, many, strict=True):
            assert current is None if query.world_key == 41 else current == before
        env.restore(checkpoint, expected_revision=batch.checkpoint_revision)
        assert reader.measure_many(env.current_frame, queries) == first
        replay = step(env)  # Third and final real step; paired history replay.
        replayed = reader.measure_many(env.current_frame, queries)
        assert replayed == many
        record(records, env, batch, replay, measured=replayed)

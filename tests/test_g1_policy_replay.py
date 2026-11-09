"""Small journal selection contracts, not G1 dynamics or rendering validation.

AUTHORING_ONLY / UNEXECUTED. The hand-authored records below are parser schema
fixtures. Their q/v distinguish published, attempted and reset data; they are
not measurements or a successful physical trajectory. No large asset is used.
"""

import copy
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

H = 1 / 1024


@pytest.fixture(scope="module")
def replay_module():
    path = Path(__file__).resolve().parents[1] / "examples/export_g1_policy_replay.py"
    spec = importlib.util.spec_from_file_location("g1_replay_parser_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def parser(replay_module, monkeypatch):
    def no_physics_or_model(*args, **kwargs):
        raise AssertionError("journal selection must not load a model, compute FK or render")

    for name in ("_bind_scene", "load_floating_mjcf", "floating_kinematics", "_frame"):
        monkeypatch.setattr(replay_module, name, no_physics_or_model)
    return replay_module


def _state(x):
    # Valid saved storage shape/XY ZW, with a non-round decimal and signed zero.
    # The values are labels for schema tests, not a claimed solution of motion.
    return dict(q=[x, 0.0, .8, 0.0, 0.0, 0.0, 1.0] + [.10000000000000002] * 29,
                v=[2**-48, -0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * 29)


def _headers(*, worlds=1, standalone=True):
    return [
        dict(event="request", arguments=dict(worlds=worlds)),
        dict(event="initialization", schema="parser-schema-fixture-only",
             sources={("scene" if standalone else "g1_primitive_actuated.xml"):
                      dict(sha256="1" * 64, bytes=1)},
             initial_q=_state(0.0)["q"], initial_v=_state(0.0)["v"], dt_s=H),
    ]


def _accepted(tick, x, *, world=None):
    result = dict(event="accepted", tick=tick, time_s=(tick + 1) * H,
                  load_time_s=tick * H, state=_state(x),
                  support=dict(sample_time_s=tick * H, commit_time_s=(tick + 1) * H,
                               total_force_n=[0.0, 0.0, 123.0]),
                  fixture_note="not a physical result")
    if world is not None:
        result["world"] = world
    return result


def _publication(step, x, *, time=None):
    return dict(step_number=step, time_s=step * H if time is None else time,
                robots={"(0, 10)": _state(x)})


def _snapshot(step, x, *, world=0, time=None):
    return dict(event="failure_publication_snapshot",
                publications={str(world): _publication(step, x, time=time)})


def _context(step, x, *, world=0, time=None):
    return dict(last_accepted={str(world): dict(
        step_number=step, time_s=step * H if time is None else time, state=_state(x))})


def _failure(*, accepted=True, known=True, context=None):
    return dict(event="first_failure", success=False,
                error="ParserFixtureFailure", message="fixture: no successful env transition",
                accepted_worlds={"0": accepted}, acceptance_known=known,
                physics_accepted=accepted or not known, context={} if context is None else context)


def _summary(steps, *, success=False, context=None, world=None):
    result = dict(event="summary", success=success,
                  first_failure=None if success else dict(message="fixture failure retained"))
    stats = dict(accepted_steps=steps, accepted_duration_s=steps * H)
    if world is None:
        result["statistics"] = stats
    else:
        result["statistics_by_world"] = {str(world): stats}
    if success:
        result["requested_duration_s"] = steps * H
    if context is not None:
        result["context"] = context
    return result


def _scan(parser, tmp_path, records, *, compressed=False, stride=40, world=0, tail=b""):
    logical = b"".join(json.dumps(record, allow_nan=False).encode("utf-8") + b"\n" for record in records) + tail
    stored = gzip.compress(logical, mtime=0) if compressed else logical
    path = tmp_path / ("schema-trace.jsonl.gz" if compressed else "schema-trace.jsonl")
    path.write_bytes(stored)
    original_records = copy.deepcopy(records)
    selected = []
    result = parser.scan_journal(path, stride=stride, world=world, on_frame=selected.append)
    assert records == original_records
    assert path.read_bytes() == stored
    source = result["metadata"]["source"]
    assert source["original_bytes"] == len(stored)
    assert source["original_sha256"] == hashlib.sha256(stored).hexdigest()
    assert source["decoded_jsonl_bytes"] == len(logical)
    assert source["decoded_jsonl_sha256"] == hashlib.sha256(logical).hexdigest()
    assert source["gzip"] is compressed and source["decoded_hash_complete"]
    return result, selected


def _ticks(selected):
    return [item["physical_tick"] for item in selected]


def _positions(selected):
    return [item["record"]["state"]["q"][0] for item in selected]


@pytest.mark.parametrize("compressed", [False, True], ids=("plain", "gzip"))
def test_normal_publication_selection_is_full_precision_and_source_is_unchanged(parser, tmp_path, compressed):
    publications = [_accepted(tick, .10000000000000002 + tick) for tick in range(5)]
    records = _headers() + publications + [_summary(5, success=True)]
    result, selected = _scan(parser, tmp_path, records, compressed=compressed, stride=2)
    assert _ticks(selected) == [0, 2, 4, 5]
    assert selected[0]["initial"] and not any(item["initial"] for item in selected[1:])
    assert selected[1]["record"] == publications[1]
    assert selected[-1]["record"] == publications[-1]
    assert selected[-1]["record"]["state"]["q"][7] == .10000000000000002
    assert selected[-1]["record"]["state"]["v"][0] == 2**-48
    assert math.copysign(1.0, selected[-1]["record"]["state"]["v"][1]) == -1
    assert result["metadata"]["journal_complete"] and result["metadata"]["reported_success"] is True
    assert result["metadata"]["accepted_records"] == 5
    assert result["summary"] == records[-1]


def test_atomic_refusal_does_not_turn_attempted_state_into_a_terminal_frame(parser, tmp_path):
    context = dict(last_attempt=dict(step_number=2, time_s=2 * H, entry_q=_state(999)["q"],
                                    q=_state(999)["q"], v=_state(999)["v"]))
    failure = _failure(accepted=False, context=context)
    records = _headers() + [_accepted(0, 1.0), _snapshot(1, 1.0), failure, _summary(1)]
    result, selected = _scan(parser, tmp_path, records)
    assert _ticks(selected) == [0, 1] and _positions(selected) == [0.0, 1.0]
    assert result["metadata"]["extra_failure_publication_frames"] == 0
    assert result["metadata"]["first_failure"] == failure
    assert result["metadata"]["reported_success"] is False
    assert result["metadata"]["journal_complete"]


@pytest.mark.parametrize("ordinary_prefix", [False, True], ids=("first-step-failure", "later-step-failure"))
def test_actual_post_error_publication_is_preserved_without_invented_force(parser, tmp_path, ordinary_prefix):
    step = 2 if ordinary_prefix else 1
    records = _headers()
    if ordinary_prefix:
        records.append(_accepted(0, 1.0))
    failure = _failure(accepted=True, known=False)
    records += [_snapshot(step, 2.0), failure, _summary(step - 1)]
    result, selected = _scan(parser, tmp_path, records)
    assert _ticks(selected) == ([0, 1, 2] if ordinary_prefix else [0, 1])
    terminal = selected[-1]["record"]
    assert terminal["state"] == _state(2.0)
    assert terminal["publication_without_env_transition"] is True
    assert terminal["support"] is None and terminal["load_time_s"] is None
    assert terminal["publication_evidence"]["source_event"] == "failure_publication_snapshot"
    assert result["metadata"]["ordinary_accepted_physical_tick"] == step - 1
    assert result["metadata"]["last_accepted_physical_tick"] == step
    assert result["metadata"]["actual_published_duration_s"] == step * H
    assert result["metadata"]["extra_failure_publication_frames"] == 1
    assert result["metadata"]["first_failure"] == failure
    assert result["metadata"]["first_failure"]["acceptance_known"] is False
    assert result["metadata"]["journal_complete"] and result["summary"]["success"] is False


@pytest.mark.parametrize("count_in_summary", [1, 2], ids=("before-postprocessing", "after-postprocessing"))
def test_actual_last_accepted_context_is_distinct_from_the_attempt_and_summary_count(parser, tmp_path, count_in_summary):
    context = _context(2, 2.0)
    context["last_attempt"] = dict(step_number=3, time_s=3 * H, state=_state(999))
    failure = _failure(context=context)
    report = _summary(count_in_summary, context=context)
    records = _headers() + [_accepted(0, 1.0), failure, report]
    result, selected = _scan(parser, tmp_path, records)
    assert _ticks(selected) == [0, 1, 2] and _positions(selected) == [0.0, 1.0, 2.0]
    assert selected[-1]["record"]["publication_evidence"]["source_path"] == "context.last_accepted"
    assert selected[-1]["record"]["support"] is None
    assert result["metadata"]["accepted_records"] == 1
    assert result["metadata"]["journal_complete"]
    assert result["summary"] == report and result["metadata"]["first_failure"] == failure


@pytest.mark.parametrize("pre_reset_state_saved", [False, True], ids=("reset-only", "pre-reset-endpoint-saved"))
def test_reset_publication_is_never_promoted_as_later_accepted_motion(parser, tmp_path, pre_reset_state_saved):
    context = _context(2, 2.0) if pre_reset_state_saved else {}
    records = _headers() + [_accepted(0, 1.0), _snapshot(0, 999.0),
                            _failure(context=context), _summary(1, context=context)]
    result, selected = _scan(parser, tmp_path, records)
    assert _ticks(selected) == ([0, 1, 2] if pre_reset_state_saved else [0, 1])
    assert 999.0 not in _positions(selected)
    assert result["metadata"]["extra_failure_publication_frames"] == int(pre_reset_state_saved)


def test_duplicate_consistent_failure_snapshots_create_one_terminal_frame(parser, tmp_path):
    context = _context(2, 2.0)
    records = _headers() + [_accepted(0, 1.0), _snapshot(2, 2.0),
                            _snapshot(2, 2.0), _failure(context=context), _summary(1, context=context)]
    result, selected = _scan(parser, tmp_path, records)
    assert _ticks(selected) == [0, 1, 2]
    assert _positions(selected).count(2.0) == 1
    assert result["metadata"]["selected_frames"] == 3
    assert result["metadata"]["extra_failure_publication_frames"] == 1
    assert result["metadata"]["trace_error"] is None


@pytest.mark.parametrize("conflict", ["state", "time"], ids=("conflicting-q", "conflicting-time"))
def test_contradictory_publication_evidence_is_reported_without_displaying_disputed_state(parser, tmp_path, conflict):
    context = _context(2, 2.0)
    if conflict == "state":
        context["last_accepted"]["0"]["state"] = _state(999.0)
    else:
        context["last_accepted"]["0"]["time_s"] = 3 * H
    records = _headers() + [_accepted(0, 1.0), _snapshot(2, 2.0),
                            _failure(context=context), _summary(1)]
    result, selected = _scan(parser, tmp_path, records)
    assert _ticks(selected) == [0, 1] and _positions(selected) == [0.0, 1.0]
    assert result["metadata"]["trace_error"] is not None
    assert "conflicting" in result["metadata"]["trace_error"]["message"]
    assert not result["metadata"]["journal_complete"]
    assert result["metadata"]["extra_failure_publication_frames"] == 0


def test_attempt_entry_and_legacy_tick_without_publication_counter_never_enter_frames(parser, tmp_path):
    decoy = dict(tick=1, time_s=2 * H, q=_state(999.0)["q"], v=_state(999.0)["v"])
    context = dict(last_accepted=decoy,
                   last_entry=dict(step_number=2, time_s=2 * H, state=_state(998.0)),
                   last_attempt=dict(step_number=2, time_s=2 * H, state=_state(997.0)))
    records = _headers() + [_accepted(0, 1.0), _failure(accepted=False, context=context),
                            _summary(1, context=context)]
    result, selected = _scan(parser, tmp_path, records)
    assert _ticks(selected) == [0, 1] and _positions(selected) == [0.0, 1.0]
    assert result["metadata"]["extra_failure_publication_frames"] == 0
    assert result["metadata"]["failure_publication"] is None


@pytest.mark.parametrize("snapshot_time", [0.5 * H, H], ids=("older-time", "same-time"))
def test_later_counter_without_later_publication_time_is_not_motion(parser, tmp_path, snapshot_time):
    records = _headers() + [_accepted(0, 1.0), _snapshot(2, 999.0, time=snapshot_time),
                            _failure(), _summary(1)]
    result, selected = _scan(parser, tmp_path, records)
    assert _ticks(selected) == [0, 1] and 999.0 not in _positions(selected)
    assert result["metadata"]["trace_error"]["type"] == "PublicationOrdering"
    assert not result["metadata"]["journal_complete"]


def test_selected_world_uses_its_actual_records_and_ignores_other_world_failure_pose(parser, tmp_path):
    records = _headers(worlds=2, standalone=False)
    failure = _failure()
    failure["accepted_worlds"] = {"0": True, "1": True}
    records += [_accepted(0, 999.0, world=0), _accepted(0, 1.0, world=1),
                _snapshot(2, 998.0, world=0), _snapshot(2, 2.0, world=1),
                failure, _summary(1, world=1)]
    result, selected = _scan(parser, tmp_path, records, world=1)
    assert _ticks(selected) == [0, 1, 2] and _positions(selected) == [0.0, 1.0, 2.0]
    assert result["metadata"]["world"] == 1
    assert result["metadata"]["ordinary_accepted_physical_tick"] == 1


def test_truncated_record_keeps_valid_accepted_prefix_and_full_original_hash(parser, tmp_path):
    records = _headers() + [_accepted(0, 1.0)]
    result, selected = _scan(parser, tmp_path, records, tail=b'{"event":"accepted","state":')
    assert _ticks(selected) == [0, 1] and _positions(selected) == [0.0, 1.0]
    assert result["metadata"]["trace_error"] is not None
    assert not result["metadata"]["journal_complete"]
    assert not result["metadata"]["journal_summary_present"]
    assert result["summary"] is None

@pytest.mark.parametrize("invalid", ["policy-vector", "nonunit-root"], ids=("policy-q-is-not-physical-q", "invalid-root-storage"))
def test_nonphysical_storage_cannot_be_displayed_as_an_accepted_body_state(parser, tmp_path, invalid):
    bad = _accepted(1, 999.0)
    if invalid == "policy-vector":
        bad["state"]["q"] = [0.0] * 29
    else:
        bad["state"]["q"][6] = 2.0
    records = _headers() + [_accepted(0, 1.0), bad, _summary(2)]
    result, selected = _scan(parser, tmp_path, records)
    assert _ticks(selected) == [0, 1] and _positions(selected) == [0.0, 1.0]
    assert result["metadata"]["trace_error"] is not None
    assert not result["metadata"]["journal_complete"]
    assert result["metadata"]["accepted_records"] == 1
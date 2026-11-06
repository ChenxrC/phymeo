"""Protocol invariants and focused original-law/physical qualification for the C0 profiler."""

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from material_phase_transfer_cases import frozen_layout, law_protocol_rows

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "profile_material_phase_transfer.py"


def profiler():
    name = "material_phase_transfer_profile_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def clock_window(*, extra=1490, all_pages=3000, sites=129):
    # Recorded protocol input, not a pretend physical run or a device measurement.
    return dict(
        whole_ns=10000,
        extra_page_api_ns=extra,
        all_page_api_ns=all_pages,
        exclusive_ns={"page_api": all_pages, "host_and_handoff": 10000 - all_pages},
        phase_sites=[sites] + [0] * 11,
        counter_audit="passed",
        accepted_world_steps=128,
        complete_output_sha256="a" * 64,
    )


def summarize(p, *, extra=1490, mode="cuda", real_cuda=True, match=True, sites=129):
    config = p.ProfileConfig(worlds=32, faces=8, mode=mode)
    return p.summarize_windows(
        config,
        cold=clock_window(extra=9000, all_pages=9500),
        warmups=[clock_window(extra=9000, all_pages=9500) for _ in range(2)],
        formal=[
            clock_window(extra=extra, all_pages=9500 if extra > 3000 else 3000, sites=sites)
            for _ in range(3)
        ],
        complete_output_match=match,
        real_cuda=real_cuda,
    )


def test_cost_gate_uses_full_task_and_excludes_cold_compile():
    p = profiler()
    result = summarize(p)
    assert result["formal_whole_ns"] == 30000
    assert result["formal_extra_page_api_ns"] == 4470
    assert result["extra_page_fraction_of_whole"] == pytest.approx(0.149)
    assert result["all_page_fraction_of_whole"] == pytest.approx(0.3)
    assert result["eligible_formal_cuda_dense"] is True
    assert result["measured_cost_gate"] is False
    # Including first-page calls, cold compilation, or an observation-only denominator
    # would incorrectly authorize an optimization in this independent example.


def test_fifteen_percent_only_authorizes_a_comparison_not_a_speed_claim():
    result = summarize(profiler(), extra=1500)
    assert result["measured_cost_gate"] is True
    assert result["candidate_cost_scope"] == "extra-page API wall upper bound"
    assert result["default_changed"] is False
    assert "speedup" not in result


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "interpret", "real_cuda": False},
        {"real_cuda": False},
        {"match": False},
        {"sites": 128},
    ],
)
def test_non_cuda_single_page_or_unequal_output_cannot_open_cost_gate(changes):
    result = summarize(profiler(), extra=9000, **changes)
    assert result["eligible_formal_cuda_dense"] is False
    assert result["measured_cost_gate"] is False


def test_overlapping_or_inconsistent_cost_totals_are_not_accepted():
    p = profiler()
    window = clock_window(extra=4000, all_pages=3000)
    with pytest.raises(ValueError, match="extra|cost|wall"):
        p.summarize_windows(
            p.ProfileConfig(32, 8, "cuda"),
            cold=clock_window(),
            warmups=[clock_window(), clock_window()],
            formal=[window, clock_window(), clock_window()],
            complete_output_match=True,
            real_cuda=True,
        )


@pytest.mark.parametrize(
    "sites,pages,uploaded,downloaded",
    [
        (0, 0, 0, 0),
        (127, 1, 6656, 2560),
        (128, 1, 6656, 2560),
        (129, 2, 13312, 5120),
    ],
)
def test_page_counter_audit_preserves_tail_bytes_and_empty_noop(sites, pages, uploaded, downloaded):
    p = profiler()
    counts = dict(
        completed_launches=pages,
        sites=sites,
        uploads=pages,
        downloads=pages,
        uploaded_bytes=uploaded,
        downloaded_bytes=downloaded,
        explicit_synchronizations=pages,
    )
    before = deepcopy(counts)
    result = p.audit_page_counters([sites], counts, mode="cuda")
    assert result["pages"] == pages
    assert result["extra_pages"] == (1 if sites == 129 else 0)
    assert result["status"] == "passed" and counts == before
    if pages:
        wrong = {**counts, "downloaded_bytes": downloaded - 4}
        with pytest.raises(ValueError, match="counter|byte|download"):
            p.audit_page_counters([sites], wrong, mode="cuda")


def test_interpret_counter_audit_does_not_invent_device_transfers():
    p = profiler()
    result = p.audit_page_counters(
        [129, 0, 1],
        dict(
            completed_launches=3,
            sites=130,
            uploads=0,
            downloads=0,
            uploaded_bytes=0,
            downloaded_bytes=0,
            explicit_synchronizations=0,
        ),
        mode="interpret",
    )
    assert result["pages"] == 3 and result["status"] == "passed"


def test_wall_partition_excludes_parked_stacks_and_nested_device_time():
    result = profiler().partition_wall(
        0,
        100,
        [
            (0, 100, 11, "world_stack"),
            (10, 90, 11, "parked"),
            (15, 85, 22, "world_stack"),
            (20, 80, 22, "parked"),
            (0, 20, 0, "await_world"),
            (30, 70, 0, "law_wrapper"),
            (40, 60, 0, "kernel_submit"),
        ],
        caller_thread=0,
    )
    # Caller waits only on [0,20): parked-only [10,15) is idle (5).
    # [20,30), [70,80), [85,90) have no caller wait: other work totals 25.
    assert result["exclusive_ns"] == {
        "world_stack": 30,
        "handoff_idle": 5,
        "scheduler_or_other_caller": 25,
        "law_wrapper": 20,
        "kernel_submit": 20,
    }
    assert result["inclusive_wait_ns"]["parked_world_stack_sum"] == 140
    assert sum(result["exclusive_ns"].values()) == 100


@pytest.mark.parametrize(
    "changes",
    [
        {"worlds": True},
        {"worlds": 2},
        {"faces": 4},
        {"mode": "cpu_cpp"},
        {"warm_windows": 3},
        {"formal_windows": 4},
        {"steps": 5},
        {"max_seconds": 181},
        {"max_raw_bytes": 32 * 1024 * 1024 + 1},
    ],
)
def test_selection_and_resource_limits_fail_before_runtime(changes):
    p = profiler()
    with pytest.raises(ValueError):
        p.ProfileConfig(**(dict(worlds=1, faces=2, mode="interpret") | changes))


def test_frozen_layout_repeats_local_ids_without_merging_world_ownership():
    layout = frozen_layout(32, 8)
    assert tuple(item.key for item in layout[:3]) == (503, 91, 1009)
    assert layout[0].physical_index == layout[4].physical_index == 0
    assert layout[0].key != layout[4].key
    mixed = frozen_layout(3, 8, "heterogeneous")
    assert tuple((w.faces, w.cloth_shift_z_m) for w in mixed) == ((2, 0), (8, 0), (8, -0.5))


@pytest.mark.tyx
@pytest.mark.parametrize("count,pages", [(0, 0), (127, 1), (128, 1), (129, 2)])
def test_original_law_protocol_keeps_all_fields_and_independent_forces(
    count, pages, tyx_mode, tmp_path
):
    import numpy as np

    result = profiler().run_row_protocol(count, mode=tyx_mode, cache_root=tmp_path / "rows")
    rows, expected = law_protocol_rows(count)
    assert result["scope"] == "arithmetic-row protocol, not a physical trajectory"
    assert result["counter_audit"]["pages"] == pages
    np.testing.assert_array_equal(np.asarray(result["inputs"]).reshape(-1, 13), rows)
    np.testing.assert_allclose(
        np.asarray(result["outputs"]).reshape(-1, 5), expected, rtol=0, atol=1e-7
    )


@pytest.mark.tyx
def test_short_real_profile_preserves_full_outputs_and_actual_phase_rows(tyx_mode, tmp_path):
    p = profiler()
    result = p.run_profile(
        p.ProfileConfig(1, 2, tyx_mode, warm_windows=0, formal_windows=1, steps=1),
        cache_root=tmp_path / "physical",
    )
    assert result["status"] == "complete"
    assert result["complete_output_match"] is True
    assert result["gate"]["measured_cost_gate"] is False
    assert result["input_manifest"]["world_keys"] == [503]
    assert result["measurements"][0]["counter_audit"] == "passed"
    assert (
        result["measurements"][0]["complete_output_sha256"]
        == result["reference"]["complete_output_sha256"]
    )
    phases = result["measurements"][0]["phase_receipts"]
    assert [part["phase"] for part in phases] == ["entry", "midpoint", "endpoint"]
    for phase in phases:
        assert phase["sites"] > 0
        assert len(phase["inputs"]) == len(phase["outputs"]) == phase["sites"]
        assert all(len(row) == 13 for row in phase["inputs"])
        assert all(len(row) == 5 for row in phase["outputs"])
        assert {face["world"] for face in phase["faces"]} == {503}
    assert (
        result["measurements"][0]["frames"][0]["worlds"][0]["result"]["report"]["accepted"] is True
    )


def test_canonical_output_only_normalizes_declared_substep_clock_and_owns_data():
    p = profiler()
    source = [
        {
            "step": 0,
            "worlds": [
                {
                    "world": 503,
                    "result": {
                        "state": {"position": [1.0], "timings": {"physical_history": 9.0}},
                        "report": {
                            "accepted": True,
                            "reason": None,
                            "substeps": [
                                {
                                    "timings": {"predict": 12.0},
                                    "diagnostics": {"force": [2.0, -2.0]},
                                }
                            ],
                        },
                        "substep_states": [{"position": [1.0]}],
                    },
                }
            ],
        }
    ]
    result = p.canonical_frames(source)
    row = result[0]["worlds"][0]["result"]
    assert row["report"]["substeps"][0]["timings"] == {"predict": 0.0}
    assert row["state"]["timings"] == {"physical_history": 9.0}
    assert row["report"]["substeps"][0]["diagnostics"] == {"force": [2.0, -2.0]}
    row["state"]["position"][0] = 99
    assert source[0]["worlds"][0]["result"]["state"]["position"] == [1.0]
    assert source[0]["worlds"][0]["result"]["report"]["substeps"][0]["timings"] == {"predict": 12.0}


def test_gzip_budget_keeps_complete_prior_records_and_never_overwrites(tmp_path):
    import gzip
    import json

    p = profiler()
    target = tmp_path / "complete.frames.jsonl.gz"
    with p.RawArchive(target, max_bytes=48) as archive:
        archive.write({"actual": [1, 2, 3]})
        with pytest.raises(p.ResourceLimit, match="budget"):
            archive.write({"actual": "x" * 100})
        assert archive.records == 1
    with gzip.open(target, "rt", encoding="utf8") as stream:
        assert [json.loads(line) for line in stream] == [{"actual": [1, 2, 3]}]
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        with p.RawArchive(target, max_bytes=48):
            pass
    assert target.read_bytes() == before


def test_cli_rejects_existing_or_relative_output_before_creating_runtime(tmp_path):
    p = profiler()
    base = ["--mode", "cuda", "--worlds", "32", "--faces", "8", "--expected-head", "a" * 40]
    with pytest.raises(ValueError, match="absolute"):
        p.parse_args([*base, "--out", "relative.json"])
    target = tmp_path / "existing.json"
    target.write_text("existing evidence", encoding="utf8")
    with pytest.raises(FileExistsError):
        p.parse_args([*base, "--out", str(target)])
    assert target.read_text(encoding="utf8") == "existing evidence"
    parsed = p.parse_args([*base, "--out", str(tmp_path / "fresh.json")])
    assert parsed.config.formal_windows == 3 and parsed.config.steps == 4
    assert not parsed.cache_root.exists() and not parsed.raw_path.exists()


@pytest.mark.tyx
@pytest.mark.parametrize("fault", ("late_capacity", "device_call"))
def test_failed_profile_keeps_partial_work_and_retry_uses_original_state(
    fault, tyx_mode, tmp_path, monkeypatch
):
    from unittest.mock import patch

    import phymeo as ph

    p = profiler()
    config = p.ProfileConfig(1, 2, tyx_mode, warm_windows=0, formal_windows=1, steps=1)
    captured = []
    original = ph.TyxMultifaceContactBatch._evaluate

    def fail_once(batch, requests, index, capacity):
        if index != 1:
            return original(batch, requests, index, capacity)
        if fault == "late_capacity":
            return original(batch, requests, index, 0)
        with patch.object(
            batch._layer, "_kernel", side_effect=RuntimeError("injected device call")
        ):
            return original(batch, requests, index, capacity)

    with monkeypatch.context() as context:
        context.setattr(ph.TyxMultifaceContactBatch, "_evaluate", fail_once)
        with pytest.raises((ValueError, RuntimeError), match="capacity|injected device"):
            p.run_profile(
                config,
                cache_root=tmp_path / "failed",
                on_window=lambda label, index, value: captured.append((label, value)),
            )
    assert captured[0][0] == "input-manifest" and captured[-1][0] == "failed-attempt"
    failed = captured[-1][1]
    assert failed["status"] == "failed" and failed["counter_delta"]["completed_launches"] == 1
    transaction = failed["failure_transaction"]
    assert transaction["published_states_sha256"] == transaction["last_completed_states_sha256"]
    assert transaction["current_phases_sha256"] == transaction["phases_before_failed_step_sha256"]
    assert transaction["automatic_retry"] is False
    assert failed["phase_receipts"][0]["outputs"]
    retry = p.run_profile(config, cache_root=tmp_path / "retry")
    assert retry["status"] == "complete" and retry["complete_output_match"] is True
    assert retry["input_manifest_sha256"] == captured[0][1]["input_manifest_sha256"]


def test_source_binding_rejects_untracked_dirty_or_changed_dependency(tmp_path, monkeypatch):
    p = profiler()
    head = "a" * 40
    paths = ("src/phymeo", "script.py", "case.py")
    package = tmp_path / "src" / "phymeo"
    package.mkdir(parents=True)
    names = ("src/phymeo/__init__.py", "script.py", "case.py")
    for name in names:
        (tmp_path / name).write_text("frozen input", encoding="utf8")
    state = {"dirty": "", "names": names}

    def git(_root, *args):
        if args[:2] == ("rev-parse", "HEAD"):
            return head
        if args[0] == "status":
            return state["dirty"]
        assert args[0] == "ls-files"
        return "\n".join(state["names"])

    monkeypatch.setattr(p, "SOURCE_TARGETS", paths)
    monkeypatch.setattr(p, "_git", git)
    binding = p.bind_source(tmp_path, head)
    assert set(binding["files"]) == set(names)
    p.verify_source(binding)
    (tmp_path / "case.py").write_text("changed physical input", encoding="utf8")
    with pytest.raises(RuntimeError, match="source changed"):
        p.verify_source(binding)
    state["dirty"] = " M case.py"
    with pytest.raises(RuntimeError, match="committed and clean"):
        p.bind_source(tmp_path, head)
    state["dirty"] = ""
    state["names"] = names[:-1]
    with pytest.raises(RuntimeError, match="not tracked"):
        p.bind_source(tmp_path, head)

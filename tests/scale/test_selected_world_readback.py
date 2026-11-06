"""The selected-world profiler preserves outputs and reports measured, exclusive costs."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "profile_selected_world_readback.py"


def profiler():
    assert SCRIPT.is_file(), "missing C0 selected-world readback measurement tool"
    name = "selected_world_readback_profile_under_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.tyx
@pytest.mark.parametrize("kind", ("snapshot", "kinematics"))
def test_selected_readback_preserves_real_mixed_states_and_counts(kind, tyx_mode, tmp_path):
    p = profiler()
    config = p.ProfileConfig(
        worlds=32,
        selection=8,
        kind=kind,
        mode=tyx_mode,
        warm_windows=0,
        formal_windows=1,
        steps=2,
    )
    result = p.run_profile(config, cache_root=tmp_path / "runtime")
    assert result["status"] == "complete"
    assert result["selected_keys"] == [628, 611, 594, 577, 560, 543, 526, 509]
    assert result["final_tick"] == 3
    assert result["audit"]["max_position_error_m"] == 0.0
    assert result["audit"]["max_velocity_error_m_s"] == 0.0
    assert result["gate"]["eligible_formal_cuda_subset"] is False
    sample = result["measurements"][0]
    counts = sample["batch_counter_delta"]
    assert counts["accepted_world_steps"] == 64
    assert counts["rejected_world_steps"] == 0
    assert counts["submitted_kernels"] == (18 if kind == "snapshot" else 2)
    assert counts["synchronizations"] == (18 if kind == "snapshot" else 2)
    assert counts["device_allocations"] == counts["topology_uploaded_bytes"] == 0
    assert counts["control_uploaded_bytes"] == 0
    expected_read = 832 if kind == "snapshot" else 3328
    assert sample["phases"]["observation"]["read_bytes"] == expected_read
    assert sample["phases"]["observation"]["d2h_bytes"] == (
        expected_read if tyx_mode == "cuda" else 0
    )
    assert counts[kind + "_downloaded_bytes"] == (expected_read if tyx_mode == "cuda" else 0)
    frame = sample["frames"][-1]["output"]
    if kind == "snapshot":
        slider = frame["checkpoints"][0]
        assert slider["world"] == 628 and slider["tick"] == 3
        assert slider["position"] == [3 / 16384]
        assert slider["velocity"] == [1 / 64]
        assert slider["applied_effort"] == [2.0]
        assert slider["history"] == [0.0, 0.0, 2.0]
        assert slider["control"] == [0.0, 0.0, 2.0, 0.0]
        assert slider["topology"]
        wire = sample["frames"][-1]["wire_snapshot"]
        assert len(wire) == 16 and wire[0]["shape"] == [12]
        assert wire[0]["values"] == [3 / 16384, 0, 0, 1 / 64, 0, 0, 2, 0, 0, 0, 0, 2]
        assert wire[1]["dtype"] == "int32" and wire[1]["values"] == [3]
    else:
        assert frame["widths"] == [1, 3, 1, 3, 1, 3, 1, 3]
        assert frame["values"][0] == [3 / 16384, 0.0, 0.0, 1 / 64, 0.0, 0.0]
        assert frame["ticks"] == [3] * 8
    for window in [result["cold"], *result["measurements"]]:
        assert sum(window["exclusive_ns"].values()) == window["whole_ns"]
        assert min(window["exclusive_ns"].values()) >= 0
        assert window["output_sha256"] == p.frames_sha256(window["frames"])
        assert window["counter_audit"] == "passed"


def test_matrix_deduplicates_one_world_selection_without_comparing_kinds():
    p = profiler()
    matrix = p.matrix_cases()
    assert len(matrix) == 14
    for kind in ("snapshot", "kinematics"):
        assert [(w, k) for w, k, lane in matrix if lane == kind] == [
            (1, "all"),
            (32, 1),
            (32, 8),
            (32, "all"),
            (128, 1),
            (128, 8),
            (128, "all"),
        ]


@pytest.mark.parametrize(
    "updates",
    [
        {"worlds": 2},
        {"worlds": True},
        {"selection": 0},
        {"selection": 8, "worlds": 1},
        {"selection": True},
        {"kind": "summary"},
        {"mode": "cpu_cpp"},
        {"steps": 0},
        {"warm_windows": -1},
        {"formal_windows": 0},
    ],
)
def test_invalid_profile_configuration_rejected_before_device(updates):
    p = profiler()
    arguments = dict(worlds=32, selection=8, kind="snapshot", mode="interpret")
    arguments.update(updates)
    with pytest.raises(ValueError):
        p.ProfileConfig(**arguments)


def test_frozen_file_binding_rejects_changed_dependency_and_missing_file(tmp_path):
    p = profiler()
    source = tmp_path / "worlds.py"
    source.write_text("original\n", encoding="utf8")
    frozen = p.freeze_files([source])
    p.verify_files(frozen)
    source.write_text("changed\n", encoding="utf8")
    with pytest.raises(RuntimeError, match="source"):
        p.verify_files(frozen)
    source.unlink()
    with pytest.raises(RuntimeError, match="source"):
        p.verify_files(frozen)


def test_source_head_mismatch_rejected_without_loading_backend():
    p = profiler()
    with pytest.raises(RuntimeError, match="HEAD"):
        p.bind_source(ROOT, "0" * 40)


def test_cli_requires_head_and_fresh_absolute_output(tmp_path):
    p = profiler()
    base = [
        "--worlds",
        "32",
        "--selection",
        "8",
        "--kind",
        "snapshot",
        "--mode",
        "interpret",
        "--cache-root",
        str(tmp_path / "cache"),
    ]
    with pytest.raises(SystemExit):
        p.parse_args([*base, "--out", str(tmp_path / "result.json")])
    with pytest.raises(ValueError, match="absolute"):
        p.reserve_output(Path("relative.json"))
    target = tmp_path / "result.json"
    with p.reserve_output(target) as stream:
        stream.write("reserved")
    with pytest.raises(FileExistsError):
        p.reserve_output(target)
    assert target.read_text(encoding="utf8") == "reserved"


@pytest.mark.tyx
def test_profile_instrumentation_unwinds_after_real_read_failure(tyx_mode, tmp_path):
    p = profiler()
    from phymeo.backends.tyx_world_batch import FreeSphere, TyxWorldBatch

    with TyxWorldBatch(
        [(101, FreeSphere(2, 0.125))], tyx_mode, cache_root=tmp_path / "runtime"
    ) as batch:
        old_read = batch._read
        old_launch = batch._launch
        old_sync = batch._sync
        old_snapshot = batch.snapshot()

        def broken_read(buffer, category):
            if category == "kinematics":
                raise RuntimeError("injected selected read failure")
            return old_read(buffer, category)

        batch._read = broken_read
        with pytest.raises(RuntimeError, match="injected"):
            with p.PhaseProbe(batch) as probe:
                probe.call("observation", batch.kinematics, [101])
        assert batch._read is broken_read
        assert batch._launch == old_launch and batch._sync == old_sync
        batch._read = old_read
        assert batch.snapshot() == old_snapshot
        assert batch.step(1 / 128) == {101: True}


def test_frozen_cli_names_resolve_to_the_same_configuration(tmp_path):
    p = profiler()
    args = p.parse_args(
        [
            "--worlds",
            "128",
            "--select",
            "8",
            "--kind",
            "kinematics",
            "--backend",
            "cuda",
            "--expected-head",
            "1" * 40,
            "--cache-root",
            str(tmp_path / "cache"),
            "--out",
            str(tmp_path / "run.json"),
        ]
    )
    assert args.config.worlds == 128 and args.config.selection == 8
    assert args.config.mode == "cuda" and args.config.kind == "kinematics"
    assert args.config.steps == 32
    assert args.config.warm_windows == 5 and args.config.formal_windows == 20


@pytest.mark.parametrize("download,submit,expected", [(750, 0, True), (749, 1, False)])
def test_gate_uses_whole_task_denominator_and_requires_actual_formal_subset(
    download, submit, expected
):
    p = profiler()
    config = p.ProfileConfig(32, 1, "kinematics", "cuda")
    window = dict(
        steps=32,
        whole_ns=5000,
        phases={
            "observation": dict(wall_ns=1000, download_ns=download, submit_ns=submit, sync_ns=0)
        },
        exclusive_ns={"all": 5000},
        batch_counter_delta=dict.fromkeys(p.COUNTERS, 0),
        native_counter_delta={},
    )
    summary, gate = p._summary(config, 1, [window] * 20)
    assert gate["measured_cost_gate"] is expected
    assert summary["observation_download_fraction"] == download / 1000
    assert summary["observation_download_fraction_of_whole"] == download / 5000
    assert p._summary(config, 1, [window])[1]["measured_cost_gate"] is False
    assert p._summary(config, 32, [window] * 20)[1]["measured_cost_gate"] is False

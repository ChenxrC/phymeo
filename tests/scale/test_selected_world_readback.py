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


@pytest.mark.tyx
def test_public_readback_plan_reads_current_accepted_buffer(tyx_mode, tmp_path):
    from phymeo.backends.tyx_world_batch import FreeSphere, TyxWorldBatch

    sphere = FreeSphere(2, 0.125, position=(0, 0, 2), gravity=(0, 0, -8))
    with TyxWorldBatch([(91, sphere)], tyx_mode, cache_root=tmp_path / "runtime") as batch:
        assert batch.step(1 / 128) == {91: True}
        original = batch.kinematics([91])
        assert original.values.tolist() == [[0, 0, 2 - 1 / 2048, 0, 0, -1 / 16]]
        plan = batch.prepare_readback([91], kind="kinematics")
        first = batch.readback(plan)
        assert first.keys == (91,) and first.widths == (3,)
        assert first.ticks.tolist() == [1]
        assert first.values.tolist() == original.values.tolist()
        for tick in (2, 3, 4):
            assert batch.step(1 / 128) == {91: True}
            frame = plan.read()
            assert frame.ticks.tolist() == [tick]
            assert frame.values.tolist() == [
                [
                    0,
                    0,
                    2 - tick * (tick + 1) / 4096,
                    0,
                    0,
                    -tick / 16,
                ]
            ]
        assert first.ticks.tolist() == [1] and first.values.tolist() == original.values.tolist()
        plan.close()
        plan.close()
        with pytest.raises(RuntimeError, match="closed"):
            plan.read()


@pytest.mark.tyx
@pytest.mark.parametrize("kind,width", [("snapshot", 12), ("kinematics", 6)])
def test_readback_plan_reorders_mixed_worlds_with_one_gather_and_owned_frames(
    kind,
    width,
    tyx_mode,
    tmp_path,
):
    p = profiler()
    from phymeo.backends.tyx_world_batch import TyxWorldBatch

    worlds, controls, _ = p.make_worlds(32)
    keys = [628, 101, 611]
    with TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "runtime") as batch:
        batch.set_controls(controls)
        initial = batch.stats()
        with batch.prepare_readback(keys, kind=kind) as plan:
            keys.clear()
            assert plan.keys == (628, 101, 611) and plan.kind == kind
            made = batch.stats()
            device_bytes = 3 * (4 * width + 8) if tyx_mode == "cuda" else 0
            assert made["device_owned_bytes"] - initial["device_owned_bytes"] == device_bytes
            assert made["readback_plan_owned_bytes"] == device_bytes
            assert made["readback_plan_live_count"] == 1
            assert made["device_allocations"] - initial["device_allocations"] == (
                3 if tyx_mode == "cuda" else 0
            )
            assert made["readback_index_uploaded_bytes"] == (12 if tyx_mode == "cuda" else 0)
            old_frame = None
            old_value = None
            for _ in range(32):
                assert all(batch.step(1 / 128).values())
                before = batch.stats()
                frame = batch.readback(plan)
                after = batch.stats()
                assert after["submitted_kernels"] - before["submitted_kernels"] == 1
                assert after["synchronizations"] - before["synchronizations"] == 1
                expected_bytes = 3 * (4 * width + 4) if tyx_mode == "cuda" else 0
                assert (
                    after[kind + "_downloaded_bytes"] - before[kind + "_downloaded_bytes"]
                    == expected_bytes
                )
                assert after["device_allocations"] == made["device_allocations"]
                assert (
                    after["readback_index_uploaded_bytes"] == made["readback_index_uploaded_bytes"]
                )
                assert after["topology_uploaded_bytes"] == made["topology_uploaded_bytes"]
                assert p._canonical_output(frame, kind) == p._canonical_output(
                    getattr(batch, kind)(plan.keys),
                    kind,
                )
                if old_frame is None:
                    old_frame, old_value = frame, p._canonical_output(frame, kind)
            assert p._canonical_output(old_frame, kind) == old_value
            if kind == "kinematics":
                assert not old_frame.values.flags.writeable and not old_frame.ticks.flags.writeable
                old_frame.values.setflags(write=True)
                old_frame.values[:] = 123
                assert p._canonical_output(plan.read(), kind) == p._canonical_output(
                    batch.kinematics(plan.keys),
                    kind,
                )
            else:
                old_frame.clear()
                assert tuple(plan.read()) == plan.keys
            stats = plan.stats()
            stats["reads"] = -1
            assert plan.stats()["reads"] > 0
        closed = batch.stats()
        assert closed["readback_plan_owned_bytes"] == 0
        assert closed["readback_plan_live_count"] == 0
        assert closed["device_owned_bytes"] == initial["device_owned_bytes"]
        assert closed["device_allocations"] == made["device_allocations"]
        assert plan.stats()["device_owned_bytes"] == 0
        assert plan.stats()["host_workspace_bytes"] == 0


@pytest.mark.tyx
def test_readback_plan_public_owner_rejection_and_batch_close(tyx_mode, tmp_path):
    from phymeo.backends.tyx_world_batch import FreeSphere, TyxWorldBatch

    sphere = FreeSphere(2, 0.125)
    left = TyxWorldBatch([(91, sphere)], tyx_mode, cache_root=tmp_path / "left")
    right = TyxWorldBatch([(91, sphere)], tyx_mode, cache_root=tmp_path / "right")
    try:
        plan = left.prepare_readback([91], kind="snapshot")
        import tyx

        before = right.stats()
        native_before = tyx.runtime_stats()
        with pytest.raises(ValueError, match="owner"):
            right.readback(plan)
        with pytest.raises(TypeError, match="WorldReadbackPlan"):
            right.readback(object())
        assert right.stats() == before
        assert tyx.runtime_stats() == native_before
        saved = plan.read()
        left.close()
        left.close()
        plan.close()
        assert left.stats()["readback_plan_owned_bytes"] == 0
        assert left.stats()["device_owned_bytes"] == 0
        assert left.stats()["readback_plan_live_count"] == 0
        with pytest.raises(RuntimeError, match="closed"):
            plan.read()
        assert saved[91].tick == 0
        assert right.step(1 / 128) == {91: True}
    finally:
        left.close()
        right.close()


@pytest.mark.tyx
@pytest.mark.parametrize("kind", ["snapshot", "kinematics"])
def test_readback_plan_empty_and_invalid_selections_have_no_device_work(kind, tyx_mode, tmp_path):
    from phymeo.backends.tyx_world_batch import FreeSphere, TyxWorldBatch

    device_fields = (
        "device_allocations",
        "device_owned_bytes",
        "submitted_kernels",
        "synchronizations",
        "readback_index_uploaded_bytes",
        "readback_plan_owned_bytes",
        "snapshot_downloaded_bytes",
        "kinematics_downloaded_bytes",
    )
    for worlds in ([], [(91, FreeSphere(2, 0.125))]):
        with TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / f"w{len(worlds)}") as batch:
            before = batch.stats()
            with batch.prepare_readback([], kind=kind) as plan:
                result = batch.readback(plan)
                if kind == "snapshot":
                    assert result == {}
                else:
                    assert result.keys == result.widths == () and result.values.shape == (0, 6)
                    assert len(result.ticks) == 0
                with pytest.raises(ValueError):
                    batch.prepare_readback([], kind="summary")
            for name in device_fields:
                assert batch.stats()[name] == before[name]
            if worlds:
                for keys, error in (([91, 91], ValueError), ([True], ValueError), ([8], KeyError)):
                    before = batch.stats()
                    with pytest.raises(error):
                        batch.prepare_readback(keys, kind=kind)
                    assert batch.stats() == before


@pytest.mark.tyx
def test_readback_plan_snapshot_keeps_held_history_and_local_restore(tyx_mode, tmp_path):
    p = profiler()
    from phymeo.backends.tyx_world_batch import TyxWorldBatch

    worlds, controls, _ = p.make_worlds(32)
    selected = (118, 101)
    slider_program = worlds[1][1].program
    import numpy as np

    with TyxWorldBatch(worlds, tyx_mode, cache_root=tmp_path / "runtime") as batch:
        batch.set_controls(controls)
        with batch.prepare_readback(selected, kind="snapshot") as plan:
            assert all(batch.step(1 / 128).values())
            changed = slider_program.inputs(np.zeros(1), feedforward=np.array([-2.0]))
            batch.set_controls({118: changed})
            first = plan.read()[118]
            assert first.control == (0.0, 0.0, -2.0, 0.0)
            assert first.history == (0.0, 0.0, 2.0) and first.applied_effort == (0.0,)
            assert all(batch.step(1 / 128).values())
            saved = plan.read()[118]
            assert saved.history == (0.0, 0.0, -2.0) and saved.applied_effort == (2.0,)
            assert all(batch.step(1 / 128).values())
            goal = plan.read()[118]
            batch.restore({118: saved})
            restored = plan.read()
            assert restored[118].tick == 2 and restored[101].tick == 3
            assert all(batch.step(1 / 128).values())
            assert plan.read()[118] == goal
            batch.reset([118])
            reset = plan.read()[118]
            assert reset.tick == 0 and reset.history == (0.0, 0.0, 0.0)
            assert reset.position == (0.0,) and reset.velocity == (0.0,)


@pytest.mark.tyx
@pytest.mark.parametrize("kind", ["snapshot", "kinematics"])
def test_readback_plan_failed_second_download_preserves_state_and_retries(
    kind,
    tyx_mode,
    tmp_path,
    monkeypatch,
):
    p = profiler()
    from phymeo.backends.tyx_world_batch import FreeSphere, TyxWorldBatch

    with TyxWorldBatch(
        [(91, FreeSphere(2, 0.125))], tyx_mode, cache_root=tmp_path / "runtime"
    ) as batch:
        plan = batch.prepare_readback([91], kind=kind)
        assert batch.step(1 / 128) == {91: True}
        prior = plan.read()
        expected = p._canonical_output(prior, kind)
        full = batch.snapshot()
        stats = batch.stats()
        original_read = batch._read
        count = 0

        def fail_second(buffer, category):
            nonlocal count
            count += 1
            if count == 2:
                raise RuntimeError("injected second plan download failure")
            return original_read(buffer, category)

        with monkeypatch.context() as patcher:
            patcher.setattr(batch, "_read", fail_second)
            with pytest.raises(RuntimeError, match="injected"):
                plan.read()
        after = batch.stats()
        assert after["current_buffer"] == stats["current_buffer"]
        assert after["completed_steps"] == stats["completed_steps"]
        assert after["readback_plan_failed_reads"] == stats["readback_plan_failed_reads"] + 1
        assert after["recovery_sync_attempts"] == stats["recovery_sync_attempts"] + 1
        assert batch.snapshot() == full
        assert p._canonical_output(prior, kind) == expected
        assert p._canonical_output(plan.read(), kind) == expected


@pytest.mark.tyx
def test_readback_plan_observes_per_world_numeric_rejection(tyx_mode, tmp_path):
    from phymeo.backends.tyx_world_batch import FreeSphere, TyxWorldBatch

    ball = FreeSphere(2, 0.125, gravity=(0, 0, -8))
    with TyxWorldBatch([(91, ball), (7, ball)], tyx_mode, cache_root=tmp_path / "runtime") as batch:
        with batch.prepare_readback([7, 91], kind="kinematics") as plan:
            batch.set_controls({7: (3e38, 0, 0)})
            assert batch.step(2) == {91: True, 7: False}
            rejected = plan.read()
            assert rejected.ticks.tolist() == [0, 1]
            assert rejected.values[0].tolist() == [0, 0, 0, 0, 0, 0]
            batch.set_controls({7: (0, 0, 0)})
            assert batch.step(1 / 128) == {91: True, 7: True}
            assert plan.read().ticks.tolist() == [1, 2]
            assert rejected.ticks.tolist() == [0, 1]


@pytest.mark.tyx
def test_readback_plan_partial_allocation_failure_reclaims_buffers_and_live_bytes(
    tyx_mode,
    tmp_path,
    monkeypatch,
):
    from phymeo.backends._world_readback_plan import WorldReadbackPlan
    from phymeo.backends.tyx_world_batch import FreeSphere, TyxWorldBatch

    with TyxWorldBatch(
        [(91, FreeSphere(2, 0.125))], tyx_mode, cache_root=tmp_path / "runtime"
    ) as batch:
        before = batch.stats()
        allocate = WorldReadbackPlan._allocate
        instances = []
        calls = 0

        def fail_after_index(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            instances.append(self)
            if calls == 2:
                raise MemoryError("injected plan scratch allocation failure")
            return allocate(self, *args, **kwargs)

        with monkeypatch.context() as patcher:
            patcher.setattr(WorldReadbackPlan, "_allocate", fail_after_index)
            with pytest.raises(MemoryError, match="injected"):
                batch.prepare_readback([91], kind="snapshot")
        after = batch.stats()
        assert after["readback_plan_live_count"] == 0
        assert after["readback_plan_owned_bytes"] == 0
        assert after["device_owned_bytes"] == before["device_owned_bytes"]
        assert after["device_allocations"] - before["device_allocations"] == (
            1 if tyx_mode == "cuda" else 0
        )
        assert after["readback_index_uploaded_bytes"] == (4 if tyx_mode == "cuda" else 0)
        assert instances[0].stats()["closed"] is True
        assert instances[0].stats()["host_workspace_bytes"] == 0
        with batch.prepare_readback([91], kind="snapshot") as plan:
            with pytest.raises(AttributeError):
                plan.keys = ()
            with pytest.raises(AttributeError):
                plan.kind = "kinematics"
            assert plan.read()[91].tick == 0


@pytest.mark.tyx
@pytest.mark.parametrize("kind", ["snapshot", "kinematics"])
def test_readback_plan_comparison_preserves_complete_public_outputs(kind, tyx_mode, tmp_path):
    p = profiler()
    config = p.ProfileConfig(32, 8, kind, tyx_mode, warm_windows=0, formal_windows=1, steps=2)
    result = p.run_comparison(config, cache_root=tmp_path / "paired", first="plan")
    assert result["status"] == "complete" and result["final_tick"] == 3
    assert result["same_public_outputs"] is True
    assert tuple(result["cold"]["order"]) == ("plan", "original")
    pair = result["measurements"][0]
    assert tuple(pair["order"]) == ("original", "plan")
    original, planned = pair["lanes"]["original"], pair["lanes"]["plan"]
    assert original["frames"] == planned["frames"]
    assert original["public_output_sha256"] == planned["public_output_sha256"]
    assert p.frames_sha256(planned["frames"]) == planned["public_output_sha256"]
    assert planned["batch_counter_delta"]["submitted_kernels"] == 4
    assert original["batch_counter_delta"]["submitted_kernels"] == (18 if kind == "snapshot" else 2)
    expected_bytes = (832 if kind == "snapshot" else 448) if tyx_mode == "cuda" else 0
    assert planned["batch_counter_delta"][kind + "_downloaded_bytes"] == expected_bytes
    assert planned["plan_counter_delta"] == dict(
        readback_index_uploaded_bytes=0,
        readback_plan_owned_bytes=0,
        readback_plan_reads=2,
        readback_plan_failed_reads=0,
    )
    assert planned["counter_audit"] == original["counter_audit"] == "passed"
    assert result["closed_plan_stats"]["closed"] is True
    assert result["closed_plan_stats"]["device_owned_bytes"] == 0
    assert all(
        row["readback_plan_live_count"] == 0 for row in result["closed_batch_stats"].values()
    )
    assert result["summary"]["default_changed"] is False


def test_readback_plan_cli_selects_unwrapped_comparison_and_new_counter_namespace(tmp_path):
    p = profiler()
    args = p.parse_args(
        [
            "--worlds",
            "128",
            "--select",
            "8",
            "--kind",
            "snapshot",
            "--backend",
            "cuda",
            "--compare-plan",
            "--first",
            "plan",
            "--expected-head",
            "1" * 40,
            "--cache-root",
            str(tmp_path / "cache"),
            "--out",
            str(tmp_path / "paired.json"),
        ]
    )
    assert args.compare_plan is True and args.first == "plan"
    assert not set(p.PLAN_COUNTERS).intersection(p.COUNTERS)
    public = dict(tick=1, accepted=[[1, True]], output={"checkpoints": []})
    with_wire = {**public, "wire_snapshot": [{"shape": [12]}]}
    assert p._public_frames([with_wire]) == [public]
    assert p.frames_sha256(p._public_frames([with_wire])) == p.frames_sha256([public])

@pytest.mark.tyx
def test_readback_plan_batch_close_failure_still_closes_all_plans(tyx_mode, tmp_path):
    from phymeo.backends.tyx_world_batch import FreeSphere, TyxWorldBatch

    sphere = FreeSphere(2, 0.125, position=(0, 0, 2), gravity=(0, 0, -8))
    batch = TyxWorldBatch(
        [(91, sphere), (7, sphere)], tyx_mode, cache_root=tmp_path / "runtime"
    )
    plans = []
    try:
        assert batch.step(1 / 128) == {91: True, 7: True}
        plans.append(batch.prepare_readback([91], kind="kinematics"))
        plans.append(batch.prepare_readback([7, 91], kind="snapshot"))
        first_read = plans[0].read()
        second_read = plans[1].read()
        assert first_read.ticks.tolist() == [1]
        assert tuple(second_read) == (7, 91)
        assert all(point.tick == 1 for point in second_read.values())
        assert batch.stats()["readback_plan_live_count"] == 2
        callbacks = []

        def fail_close():
            callbacks.append("called")
            raise RuntimeError("injected first-plan close callback failure")

        first_to_close = tuple(batch._readback_plans)[0]
        first_to_close._resources.callback(fail_close)
        with pytest.raises(RuntimeError, match="injected first-plan close"):
            batch.close()
        assert callbacks == ["called"]
        assert all(plan.stats()["closed"] for plan in plans)
        assert all(plan.stats()["device_owned_bytes"] == 0 for plan in plans)
        assert all(plan.stats()["host_workspace_bytes"] == 0 for plan in plans)
        assert batch.stats()["device_owned_bytes"] == 0
        assert batch.stats()["readback_plan_owned_bytes"] == 0
        assert batch.stats()["readback_plan_live_count"] == 0
        after_close = batch.stats()
        batch.close()
        for plan in plans:
            plan.close()
            with pytest.raises(RuntimeError, match="closed"):
                plan.read()
            with pytest.raises(RuntimeError, match="closed"):
                batch.readback(plan)
        assert batch.stats() == after_close
        assert callbacks == ["called"]
    finally:
        for plan in plans:
            plan.close()
        batch.close()

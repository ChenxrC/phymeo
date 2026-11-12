"""New bounded G2a cells; fixed old oracle, no product-capacity changes."""

import importlib.util
from pathlib import Path

import pytest

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))
CELLS = ("w1-scalar", "w1-ball", "w1-free", "w8-b3", "w16-b3", "w16-b8")


def consumer():
    path = Path(__file__).parent / "installed" / "check_mujoco_tyx_heterogeneous_scale.py"
    spec = importlib.util.spec_from_file_location("heterogeneous_scale_consumer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cell", CELLS)
def test_complete_heterogeneous_scale_cell(mode, cell, tmp_path):
    runner = consumer()
    result = runner.run_cell(mode, cell, tmp_path / "complete")
    worlds = len(runner.cell_layout(cell))
    assert result["progress"]["checked_calls"] == 27
    assert result["progress"]["accepted_world_steps"] == 27 * worlds
    assert result["bucket_count"] == (8 if cell == "w16-b8" else min(worlds, 3))
    assert result["memory"]["live_device_bytes"] == "UNKNOWN"
    assert result["memory"]["peak_device_bytes"] == "UNKNOWN"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cell", ("w16-b3", "w16-b8"))
@pytest.mark.parametrize("fault", ("last-status", "last-download", "late-aggregate"))
def test_scaled_last_bucket_failure_retains_and_retries(mode, cell, fault, monkeypatch, tmp_path):
    from phymeo.training.resident_stepper import PhysicsStepFailure

    runner = consumer()
    with runner.Session(mode, cell, tmp_path / "fault") as run:
        run.advance(0, "accepted-prefix")
        packets = run.commands(1)
        run.batch.set_controls(packets)
        before = run.batch.checkpoint().to_bytes()
        old_state = run.batch.integration_snapshot()
        old_receipt = run.batch.last_step
        counters = dict(run.batch.counters)
        old_banks = tuple(bucket._device.active for bucket in run.batch._buckets)
        run.emit("fault-attempt", fault=fault, entry=old_state, controls=packets,
                 checkpoint=before, counters=counters, active_banks=old_banks)
        with monkeypatch.context() as scoped:
            if fault == "late-aggregate":
                def broken(operation, pending):
                    if operation == "step":
                        run.emit("fault-actual-aggregate", pending=pending)
                        raise RuntimeError("G2a late complete aggregate")
                scoped.setattr(run.batch, "_before_commit", broken)
            else:
                target = run.batch._buckets[-1]._device
                original = target.download

                def broken():
                    actual = original()
                    run.emit("fault-actual-download", actual=actual)
                    if fault == "last-download":
                        raise RuntimeError("G2a final real download")
                    actual["status"][-1] = 2
                    run.emit("fault-status-mutation", actual=actual)
                    return actual

                scoped.setattr(target, "download", broken)
            with pytest.raises(PhysicsStepFailure) as caught:
                with run.forbid_native_step():
                    run.batch.step(runner.H)
        run.emit("fault-return", error=repr(caught.value),
                 accepted=caught.value.accepted_worlds, counters=run.batch.counters,
                 state=run.batch.integration_snapshot())
        assert dict(caught.value.accepted_worlds) == dict.fromkeys(run.keys, False)
        assert run.batch.checkpoint().to_bytes() == before
        assert run.batch.last_step is old_receipt
        run.helper.exact(run.batch.integration_snapshot(), old_state, "all scaled worlds retained")
        assert tuple(bucket._device.active for bucket in run.batch._buckets) == old_banks
        assert run.batch.counters["completed_steps"] == counters["completed_steps"]
        assert run.batch.counters["accepted_world_steps"] == counters["accepted_world_steps"]
        assert (run.batch.counters["bucket_physics_prepares"] -
                counters["bucket_physics_prepares"]) == run.bucket_count
        run.advance(1, "same-input-real-retry")
        assert run.progress["checked_calls"] == 2
        assert run.progress["accepted_world_steps"] == 2 * len(run.keys)
        run.mark_pass(fault=fault)


@pytest.mark.parametrize("limit", ("world17", "model9"))
def test_scaled_limits_reject_before_preflight(limit, monkeypatch, tmp_path):
    import phymeo.backends.mujoco_heterogeneous as product

    runner = consumer()
    helper = runner.load_reference()
    declaration = runner.declarations("w1-scalar", helper)
    model = declaration["specs"][0]
    first = declaration["worlds"][0]
    if limit == "world17":
        models = (model,)
        worlds = tuple(product.MuJoCoTyxWorldSpec(100 + i, model.key, first.initial_state)
                       for i in range(17))
    else:
        models = tuple(product.MuJoCoTyxModelSpec(f"too-many-{i}", model.compilation)
                       for i in range(9))
        worlds = tuple(product.MuJoCoTyxWorldSpec(100 + i, row.key, first.initial_state)
                       for i, row in enumerate(models))

    def forbidden(*args, **kwargs):
        raise AssertionError("capacity must reject before preflight or allocation")

    monkeypatch.setattr(product, "_model_preflight", forbidden)
    monkeypatch.setattr(product, "MuJoCoRuntime", forbidden)
    with pytest.raises(ValueError, match="1..8 models and 1..16 worlds") as caught:
        product.MuJoCoTyxHeterogeneousBatch(models, worlds, mode="interpret")
    trace = helper.Journal(tmp_path / "preflight.jsonl.gz")
    try:
        trace.emit("preflight-rejection", limit=limit, models=len(models),
                   worlds=len(worlds), error=repr(caught.value))
    finally:
        trace.close()

"""G2b-A facade qualification; all expected cases/gates precede first execution."""
import copy
import importlib.util
import sys
from pathlib import Path

import pytest

MODES = ("interpret", pytest.param("cuda", marks=pytest.mark.cuda))


def runner():
    path = Path(__file__).parents[1] / "scripts" / "run_heterogeneous_gallery.py"
    spec = importlib.util.spec_from_file_location("g2ba_gallery_runner", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", MODES)
def test_thirty_two_distinct_models_share_complete_twelve_step_publications(mode, tmp_path):
    tool = runner()
    result = tool.run(mode, 12, tmp_path / "gallery-short")
    assert result["status"] == "PASS"
    assert result["qualification"] == "STRICT_PASS"
    assert result["checked_steps"] == 12
    assert result["accepted_world_steps"] == 384
    assert result["model_count"] == result["bucket_count"] == 32
    assert result["state_records"] == 13
    assert result["closed"] is True
    assert result["memory"]["host_peak_bytes"] == "UNKNOWN"
    assert result["memory"]["cuda_context_peak_bytes"] == "UNKNOWN"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fault", ("last-status", "last-download", "late-aggregate"))
def test_final_bucket_failure_retains_all_thirty_two_and_retries(mode, fault, monkeypatch, tmp_path):
    from phymeo.training.resident_stepper import PhysicsStepFailure

    tool = runner()
    with tool.Session(mode, 12, tmp_path / fault) as run:
        run.advance(0)
        packets = run.commands(1)
        run.batch.set_controls(packets)
        saved = run.batch.checkpoint().to_bytes()
        states = run.batch.integration_snapshot()
        receipt = run.batch.last_step
        banks = tuple(owner._device.active for owner in run.batch._buckets)
        counters = dict(run.batch.counters)
        resources = run.batch.resource_snapshot()
        run.emit("injected-failure-entry", fault=fault, checkpoint=saved, states=states,
                 banks=banks, resources=resources, controls=packets)
        with monkeypatch.context() as patch:
            if fault == "late-aggregate":
                original = run.batch._before_commit

                def broken(operation, pending):
                    original(operation, pending)
                    if operation == "step":
                        run.emit("actual-unpublished-aggregate", pending=pending)
                        raise RuntimeError("G2b complete aggregate failure")

                patch.setattr(run.batch, "_before_commit", broken)
            else:
                device = run.batch._buckets[-1]._device
                original = device.download

                def broken():
                    actual = original()
                    run.emit("actual-last-bucket-readback", actual=actual)
                    if fault == "last-download":
                        raise RuntimeError("G2b last real download failure")
                    actual["status"][-1] = 2
                    return actual

                patch.setattr(device, "download", broken)
            with pytest.raises(PhysicsStepFailure) as caught:
                with run.forbid_native():
                    run.batch.step(tool.H)
        run.emit("injected-failure-return", error=repr(caught.value),
                 accepted=caught.value.accepted_worlds, states=run.batch.integration_snapshot())
        assert dict(caught.value.accepted_worlds) == dict.fromkeys(run.keys, False)
        assert run.batch.checkpoint().to_bytes() == saved
        assert run.batch.last_step is receipt
        tool.exact(run.batch.integration_snapshot(), states, "all world state and history retained")
        assert tuple(owner._device.active for owner in run.batch._buckets) == banks
        assert run.batch.counters["completed_steps"] == counters["completed_steps"]
        assert run.batch.counters["accepted_world_steps"] == counters["accepted_world_steps"]
        assert run.batch.counters["bucket_physics_prepares"] - counters["bucket_physics_prepares"] == 32
        assert run.batch.resource_snapshot()["budget"]["current_bytes"] == resources["budget"]["current_bytes"]
        run.advance(1)
        assert run.checked_steps == 2
        run.mark_pass()


@pytest.mark.parametrize("mode", MODES)
def test_thirty_two_bucket_checkpoint_restore_partial_reset_have_real_successors(mode, tmp_path):
    tool = runner()
    with tool.Session(mode, 12, tmp_path / "lifecycle") as run:
        run.advance(0)
        checkpoint = run.batch.checkpoint()
        saved = run.batch.integration_snapshot()
        saved_refs = {key: copy.copy(data) for key, data in run.refs.items()}
        run.emit("checkpoint", capsule=checkpoint.to_bytes(), states=saved)
        run.advance(1)
        branch = run.batch.integration_snapshot()
        run.batch.restore(checkpoint, expected_revision=run.batch.checkpoint_revision)
        tool.exact(run.batch.integration_snapshot(), saved, "all32 checkpoint restore")
        run.refs = {key: copy.copy(data) for key, data in saved_refs.items()}
        run.advance(1)
        tool.exact(run.batch.integration_snapshot(), branch, "real same-input restored successor")
        selected = (run.keys[0], run.keys[8], run.keys[16], run.keys[-1])
        before = run.batch.integration_snapshot()
        run.emit("partial-reset-attempt", selected=selected, states=before)
        run.batch.reset(selected)
        actual = run.batch.integration_snapshot()
        run.emit("partial-reset-return", states=actual, resources=run.batch.resource_snapshot())
        for key in run.keys:
            if key in selected:
                tool.exact(actual[key], run.initial[key], "selected initial state")
                run.refs[key] = copy.copy(run.initial_refs[key])
            else:
                tool.exact(actual[key], before[key], "unselected state/history")
        run.advance(2)
        run.mark_pass()


@pytest.mark.parametrize("limit", ("world33", "model33", "no-budget-world17", "no-budget-model9"))
def test_expanded_facade_limits_are_checked_before_model_preflight(limit, monkeypatch, tmp_path):
    import phymeo.backends.mujoco_heterogeneous as product
    from phymeo.backends.mujoco_resource_budget import MuJoCoTyxResourceBudget

    tool = runner()
    declaration = tool.declarations((0,))
    spec, world = declaration["specs"][0], declaration["worlds"][0]
    if "world" in limit:
        count = 33 if limit == "world33" else 17
        specs = (spec,)
        worlds = tuple(product.MuJoCoTyxWorldSpec(1000+i, spec.key, world.initial_state) for i in range(count))
    else:
        count = 33 if limit == "model33" else 9
        specs = tuple(product.MuJoCoTyxModelSpec(f"overflow-{i}", spec.compilation) for i in range(count))
        worlds = tuple(product.MuJoCoTyxWorldSpec(1000+i, item.key, world.initial_state) for i, item in enumerate(specs))

    def forbidden(*args, **kwargs):
        raise AssertionError("capacity/budget requirement must reject before native preflight")

    monkeypatch.setattr(product, "_model_preflight", forbidden)
    monkeypatch.setattr(product, "MuJoCoRuntime", forbidden)
    budget = None if limit.startswith("no-budget") else MuJoCoTyxResourceBudget(tool.DEVICE_PAYLOAD_LIMIT)
    with pytest.raises(ValueError) as caught:
        product.MuJoCoTyxHeterogeneousBatch(specs, worlds, mode="interpret", resource_budget=budget)
    (tmp_path / "rejection.txt").write_text(f"{limit}\n{type(caught.value).__name__}: {caught.value}\n", encoding="utf-8")
